cimport cython
from libc.math cimport round, exp, atan2, M_PI

import numpy as np
cimport numpy as cnp

cdef extern from "complex.h":
	double complex exp(double complex z)

cdef double complex I = 1j

@cython.cdivision(True)
cdef double interp_crossing_time(
	double time1,
	double time2,
	double wvf1,
	double wvf2,
	double threshold,
):
	"""
	Compute a single interpolated sampling point, and return the time.
	This does *not* check that wvf1 and wvf2 straddle the threshold value
	(wvf1 < threshold <= wvf2)
	"""
	if wvf1 == wvf2:
		return time1
	return time1 + (threshold - wvf1) * (time2 - time1) / (wvf2 - wvf1)

# @cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def lock_ideal_cdr(
	double[:] time,
	double[:] wvf,
	double threshold,
	double one_threshold,
	double zero_threshold,
	long samples_per_symbol,
	bint debug_mode = False,
):
	"""
	Behavioral model of a CDR block.

	Adjusts the clock offsets to coincide with the rising/falling edges of the
	data trace.  Will also return `sampling_offset`, which is the offset between
	the clock edge and the sampling point that gives the largest eye height opening.

	Args:
		- time (np.ndarray): Time series corresponding to the signal data
		- wvf (np.ndarray): Array of data to apply CDR algorithm to
		- threshold (float): Threshold that the edge should be aligned to
		- one_threshold (float): Threshold above which a datapoint is a 1
		- zero_threshold (float): Threshold below which a datapoint is a 0
		- samples_per_symbol (long): Samples per symbol (sampling rate)
		- debug_mode (bint): If True, generates some diagnostics plots.

	Returns:
		- edge_offset (int):
		- sampling_offset (int):

	"""
	cdef long n = len(time)
	cdef double pdata1, pdata2  # prev data value
	cdef double edge1, edge2
	cdef double ndata1, ndata2  # next data value
	cdef long i, j
	cdef long half_ui = <long>round(0.5 * samples_per_symbol)
	cdef cnp.ndarray[cnp.float64_t, ndim=1] edge_times
	cdef double complex n_edges = 0
	cdef long n_edges_int
	cdef double dt = time[1] - time[0]
	cdef double period = <double>samples_per_symbol * dt
	cdef double ts
	cdef double complex ts_c
	cdef double complex ts_c_avg = 0
	cdef double edge_offset, sampling_offset
	cdef double edge_offset_index, sampling_offset_index

	for i in range(n-(samples_per_symbol+1)):
		pdata1 = wvf[i]
		pdata2 = wvf[i + 1]
		ndata1 = wvf[i + samples_per_symbol]
		ndata2 = wvf[i + samples_per_symbol + 1]
		edge1 = wvf[i + half_ui]
		edge2 = wvf[i + half_ui + 1]

		# Rising edges
		if (pdata1 < zero_threshold) and (pdata2 < zero_threshold) and (ndata1 > one_threshold) and (ndata2 > one_threshold) and (edge1 < threshold) and (edge2 >= threshold):
			ts = interp_crossing_time(time[i + half_ui], time[i + half_ui + 1], edge1, edge2, threshold)
			ts_c = exp(I * 2.0 * M_PI * ts / period)
			ts_c_avg = (ts_c + ts_c_avg * n_edges) / (n_edges + 1)  # running average
			n_edges += 1

		# Falling edges
		if (pdata1 > one_threshold) and (pdata2 > one_threshold) and (ndata1 < zero_threshold) and (ndata2 < zero_threshold) and (edge1 > threshold) and (edge2 <= threshold):
			ts = interp_crossing_time(time[i + half_ui], time[i + half_ui + 1], edge1, edge2, threshold)
			ts_c = exp(I * 2.0 * M_PI * ts / period)
			ts_c_avg = (ts_c + ts_c_avg * n_edges) / (n_edges + 1)  # running average
			n_edges += 1

	if <int>n_edges.real == 0:
		return 0.0, 0.0

	# Calculate offset (from zero) in order to get the first edge position
	edge_offset_index = samples_per_symbol * atan2(ts_c_avg.imag, ts_c_avg.real) / (2 * M_PI)
	edge_offset_index %= samples_per_symbol  # location of the edge in the first UI

	# Below, calculate offset between edges and data
	# Iterate through all sampling offsets, and take the mean of the bottom 1%
	# of the 1 levels, and the top 1% of the 0 levels.  Choose the sampling
	# offset that produces the largest eye opening
	cdef long edge_index = <long>round(edge_offset_index)  # first index to start at
	cdef double d_edge = <double>edge_index - edge_offset_index
	cdef long n_data = (n / samples_per_symbol) - 1
	cdef cnp.ndarray[cnp.float64_t, ndim=2] one_min = np.ones((n_data, samples_per_symbol), dtype=np.float64) * np.max(wvf)
	cdef cnp.ndarray[cnp.float64_t, ndim=2] zero_max = np.ones((n_data, samples_per_symbol), dtype=np.float64) * np.min(wvf)
	cdef cnp.ndarray[cnp.float64_t, ndim=1] eye_opening = np.zeros(samples_per_symbol, dtype=np.float64)
	cdef double avg_data
	cdef long num_ones_to_avg, num_zeros_to_avg

	for i in range(n_data):
		# Determine sign of data section
		avg_data = 0.0
		for j in range(samples_per_symbol):
			avg_data += wvf[edge_index + j]
		avg_data /= samples_per_symbol
		# Add waveform to the corresponding zero_max/one_min matrix
		if avg_data > threshold:
			for j in range(samples_per_symbol):
				one_min[i, j] = wvf[edge_index + j]
		else:
			for j in range(samples_per_symbol):
				zero_max[i, j] = wvf[edge_index + j]
		edge_index += samples_per_symbol

	# Sort the matrices
	one_min.sort(axis=0, kind="quicksort")
	zero_max.sort(axis=0, kind="quicksort")

	# Average only ~1% of the values (1 added to prevent zeros)
	num_ones_to_avg = (one_min.shape[0] / 100) + 1
	num_zeros_to_avg = (zero_max.shape[0] / 100) + 1

	eye_opening = np.mean(one_min[:num_ones_to_avg, :], axis=0) - np.mean(zero_max[zero_max.shape[0]-num_zeros_to_avg:, :], axis=0)
	max_indices = np.argwhere(eye_opening == np.max(eye_opening))
	sampling_offset_index = <double>(max_indices[len(max_indices)//2][0])
	sampling_offset_index += d_edge

	return edge_offset_index, sampling_offset_index
