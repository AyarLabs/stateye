"""
Utilities for waveform analysis

Derek M. Kita
Ayar Labs

"""

cimport cython
from libc.math cimport round, abs, floor, ceil

import numpy as np
cimport numpy as cnp

cpdef double get_wvf_pt(
	double s_idx,
	double[:] wvf,
):
	"""
	Compute a single interpolated sampling point
	"""
	cdef long lower_idx = <long>floor(s_idx)
	cdef long upper_idx = lower_idx + 1
	if upper_idx == <long>len(wvf):  # wrap around
		upper_idx = 0
	return wvf[lower_idx] + (wvf[upper_idx] - wvf[lower_idx]) * (s_idx - <double>lower_idx)


cpdef double[:] get_wvf_pts(
	double[:] s_idx,
	double[:] wvf,
):
	"""
	Compute the list of interpolated sampling points
	"""
	cdef long i
	cdef long n = len(s_idx)
	cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)

	for i in range(n):
		result[i] = get_wvf_pt(s_idx[i], wvf)

	return result


cpdef double get_time_crossing(
	double w_threshold,
	double t0,
	double w0,
	double t1,
	double w1,
):
	"""
	Returns the time at which the waveform intercepts w_threshold
	"""
	cdef double fdist = (w_threshold - w0) / (w1 - w0)
	return t0 + fdist * (t1 - t0)


cpdef long[:] filter(
	long value,
	long[:] fbefore,
	long[:] fafter,
	long[:] data,
):
	"""
	Returns a list of indices, where the pattern fbefore + value + fafter is
	found within the data array.
	"""
	cdef long nb = len(fbefore)
	cdef long na = len(fafter)
	cdef long n = len(data)
	cdef long i
	cdef long j
	cdef cnp.ndarray[cnp.int_t, ndim=1] filter = np.concatenate([fbefore, [value], fafter], dtype=int)
	cdef cnp.ndarray[cnp.int_t, ndim=1] matches = np.zeros(n, dtype=int)
	cdef bint match

	for i in range(n - (nb + na + 1)):
		match = True
		for j in range(nb + na + 1):
			if data[i+j] != filter[j]:
				match = False
				break

		matches[i + nb] = 1 if match else 0

	return matches


cpdef double find_crossing(
	double[:] times,
	double[:] y,
	double ythreshold,
):
	"""
	Given a set of t, y points, return the interpolated time at which the
	line intersects y = ythreshold.
	"""
	cdef long i
	cdef long found = 0
	cdef long n = len(y)
	for i in range(n-1):
		if y[i] <= ythreshold and y[i+1] > ythreshold:
			# upwards intersection
			found = 1
			break
		if y[i] >= ythreshold and y[i+1] < ythreshold:
			# downwards intersection
			found = 1
			break

	if found == 0:
		return np.nan

	return get_time_crossing(ythreshold, times[i], y[i], times[i+1], y[i+1])


cpdef tuple compute_abs_edge_time(
	long[:] pattern_matches,
	double threshold,
	double[:] s_idx,
	double[:] t,
	double[:] y,
	double period,
):
	cdef int num_edges = np.sum(pattern_matches)
	cdef long i, start_i, end_i
	cdef long n = len(pattern_matches)
	cdef double edge_time = 0
	cdef double t_thr
	cdef long skip = 0

	for i in range(n-2):
		if pattern_matches[i]:
			start_i = <long>ceil(s_idx[i])
			end_i = <long>floor(s_idx[i+2])
			t_thr = find_crossing(t[start_i:end_i], y[start_i:end_i], threshold)
			if np.isnan(t_thr):
				skip += 1
				continue
			t_thr -= (<double>i) * period
			edge_time += t_thr

	if num_edges == 0 or skip == num_edges:
		return 0.0, 0.0
	else:
		return edge_time / <double>(num_edges - skip), num_edges - skip

cpdef tuple compute_overshoot(
	long[:] pattern_matches,
	double[:] s_idx,
	double[:] t,
	double[:] y,
	double period,
):
	cdef int num_edges = np.sum(pattern_matches)
	cdef long i, start_i, end_i
	cdef long n = len(pattern_matches)
	cdef double peak_time = 0.0
	cdef double peak_value = 0.0
	cdef double t_p
	cdef long skip = 0
	cdef long filter_width = 4
	cdef long idx_max

	for i in range(n-filter_width):
		if pattern_matches[i]:
			start_i = <long>ceil(s_idx[i])
			end_i = <long>floor(s_idx[i+filter_width])
			idx_max = <long>(np.argmax(y[start_i : end_i]))
			t_p = t[start_i : end_i][idx_max]
			t_p -= (<double>i) * period
			peak_time += t_p
			peak_value += y[start_i : end_i][idx_max]

	if num_edges == 0:
		return np.nan, np.nan, 0.0
	else:
		peak_time /= <double>num_edges
		peak_value /= <double>num_edges
		return peak_time, peak_value, num_edges


cpdef tuple compute_undershoot(
	long[:] pattern_matches,
	double[:] s_idx,
	double[:] t,
	double[:] y,
	double period,
):
	cdef int num_edges = np.sum(pattern_matches)
	cdef long i, start_i, end_i
	cdef long n = len(pattern_matches)
	cdef double peak_time = 0.0
	cdef double peak_value = 0.0
	cdef double t_p
	cdef long skip = 0
	cdef long filter_width = 4
	cdef long idx_min

	for i in range(n-filter_width):
		if pattern_matches[i]:
			start_i = <long>ceil(s_idx[i])
			end_i = <long>floor(s_idx[i+filter_width])
			idx_min = <long>(np.argmin(y[start_i : end_i]))
			t_p = t[start_i : end_i][idx_min]
			t_p -= (<double>i) * period
			peak_time += t_p
			peak_value += y[start_i : end_i][idx_min]

	if num_edges == 0:
		return np.nan, np.nan, 0.0
	else:
		peak_time /= <double>num_edges
		peak_value /= <double>num_edges
		return peak_time, peak_value, num_edges