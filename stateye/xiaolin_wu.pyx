cimport cython
from libc.math cimport round, abs, floor

"""
Derek M. Kita
Ayar Labs

An adaption of Xiaolin Wu's anti-aliasing line-drawing algorithm:

References:
	"An efficient antialiasing technique"
	Xiaolin Wu
	ACM SIGGRAPH Computer Graphics
	Volume 25 Issue 4 pp 143–152
	https://doi.org/10.1145/127719.122734

"""

# @cython.wraparound(False)
# @cython.boundscheck(False)
def quick_draw(
	double[:] time,  # time x-values
	double[:] wvf,  # waveform y-values
	double[:] st,  # sampling times
	double[:] f,  # fracture point indices
	long[:] patterns, # 3rd index of hist_data - which hist to draw waveform
	double[:, :, ::1] hist_data,
):
	cdef long nx = hist_data.shape[0]
	cdef long ny = hist_data.shape[1]
	cdef long i
	cdef long n = len(time)-1
	cdef int fn = len(f)
	cdef int bit_idx = 0
	cdef double fracture_time = f[0]
	cdef double sampling_time = st[0]
	cdef double sampling_index = (<double>nx)/2

	for i in range(n):
		if time[i+1] > fracture_time:
			bit_idx += 1  # increment bit index
			if bit_idx <= fn-1:
				fracture_time = f[bit_idx]
				sampling_time = st[bit_idx]
			else:
				fracture_time = time[n]
				bit_idx -= 1

		if patterns[bit_idx] == -1:  # skip edge bits
			continue
		else:
			draw_line(
				x0=time[i]-sampling_time+sampling_index,
				y0=wvf[i],
				x1=time[i+1]-sampling_time+sampling_index,
				y1=wvf[i+1],
				hist_data=hist_data,
				nx=nx,
				ny=ny,
				zi=patterns[bit_idx],
			)

# @cython.cdivision(True)
# @cython.boundscheck(False)
cpdef void draw_line(
	double x0,
	double y0,
	double x1,
	double y1,
	double[:, :, ::1] hist_data,
	long nx,
	long ny,
	long zi,
):
	"""
	Assumes that all input values are in "grid" units.
	"""
	cdef bint steep = abs(y1 - y0) > abs(x1 - x0)
	cdef long n
	cdef double dx
	cdef double dy
	cdef double gradient
	cdef double xend
	cdef double yend
	cdef double xgap
	cdef long xpxl1, xpxl2
	cdef long ypxl1, ypxl2
	cdef double intery
	cdef double c_fp, c_rfp
	cdef long k
	cdef long xi, yi

	if steep:
		x0, y0 = y0, x0
		x1, y1 = y1, x1

	n = 2 * (abs(<long>round(x1) - <long>round(x0)) + 1)

	if (<long>x1 - <long>x0) == 0:
		# If line does not traverse more than 1 pixel (horizontally), don't draw it.
		pass

	else:

		if x0 > x1:
			x0, x1 = x1, x0
			y0, y1 = y1, y0

		dx = x1 - x0
		dy = y1 - y0
		gradient = 1 if dx == 0 else dy / dx

		# Handle the first endpoint
		xend = round(x0)
		yend = y0 + gradient * (xend - x0)
		xgap = 0.5 - x0 + floor(x0 + 0.5)  # 1 - fpart(x) = 1 - (x - floor(x)) = 1 - x + floor(x), x = x0+0.5
		xpxl1 = <long>xend
		ypxl1 = <long>floor(yend)
		c_fp = yend - floor(yend)
		c_rfp = 1 - c_fp
		if steep:
			xi = ypxl1 % nx
			yi = xpxl1 % ny
			hist_data[xi, yi, zi] += c_rfp * xgap
			xi = (ypxl1 + 1) % nx
			hist_data[xi, yi, zi] += c_fp * xgap
		else:
			xi = xpxl1 % nx
			yi = ypxl1 % ny
			hist_data[xi, yi, zi] += c_rfp * xgap
			yi = (ypxl1 + 1) % ny
			hist_data[xi, yi, zi] += c_fp * xgap

		intery = yend + gradient # first y-intersection for the main loop

		# Handle the second endpoint
		xend = round(x1)
		yend = y1 + gradient * (xend - x1)
		xgap = (x1 + 0.5) - floor(x1 + 0.5)  # formerly fpart()
		xpxl2 = <long>xend  # This will be used in the main loop
		ypxl2 = <long>floor(yend)
		c_fp = yend - floor(yend)
		c_rfp = 1 - c_fp
		if steep:
			xi = ypxl2 % nx
			yi = xpxl2 % ny
			hist_data[xi, yi, zi] += c_rfp * xgap
			xi = (ypxl2 + 1) % nx
			hist_data[xi, yi, zi] += c_fp * xgap
		else:
			xi = xpxl2 % nx
			yi = ypxl2 % ny
			hist_data[xi, yi, zi] += c_rfp * xgap
			yi = (ypxl2 + 1) % ny
			hist_data[xi, yi, zi] += c_fp * xgap

		# Main loop
		if steep:
			for k in range(xpxl1 + 1, xpxl2):
				c_fp = intery - floor(intery)				# Fractional part of intery (e.g. 0.3 if 2.3)
				c_rfp = 1 - c_fp							# Upper fractional part of intery (e.g. 0.7 if 2.3)
				xi = <long>floor(intery) % nx
				yi = k % ny
				hist_data[xi, yi, zi] += c_rfp
				xi = (<long>floor(intery) + 1) % nx
				hist_data[xi, yi, zi] += c_fp
				intery = intery + gradient
		else:
			for k in range(xpxl1 + 1, xpxl2):
				c_fp = intery - floor(intery)				# Fractional part of intery (e.g. 0.3 if 2.3)
				c_rfp = 1 - c_fp							# Upper fractional part of intery (e.g. 0.7 if 2.3)
				xi = k % nx
				yi = <long>floor(intery) % ny
				hist_data[xi, yi, zi] += c_rfp
				yi = (<long>floor(intery) + 1) % ny
				hist_data[xi, yi, zi] += c_fp
				intery = intery + gradient