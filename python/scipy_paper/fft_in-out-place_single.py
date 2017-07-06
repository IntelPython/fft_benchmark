import numpy as np
try:
    import scipy.fftpack
    run_scipy = True
except:
    run_scipy = False

from perf import time_func, print_summary, arg_signature, get_random_state
from platform import system

if system() == 'Windows':
   n1d = 1200000
   size2d = (1300, 1100)
   size3d = (113, 114, 115)
else:
   n1d = 3*(10**7)
   size2d = (1860, 1420)
   size3d = (273, 274, 275)

rs = get_random_state()

vec_z = rs.randn(n1d) + \
        rs.randn(n1d) * 1j

mat_z =  rs.randn(*size2d) + \
         rs.randn(*size2d) * 1j

arr_z =  rs.randn(*size3d) + \
         rs.randn(*size3d) * 1j


tmp = vec_z.astype(np.csingle)
perf_times = time_func(np.fft.fft, tmp, dict(), refresh_buffer=False)
print_summary(perf_times, header='np.fft.fft' + arg_signature(tmp))

tmp = vec_z.astype(np.csingle)
perf_times = time_func(scipy.fftpack.fft, tmp, dict(overwrite_x=True), refresh_buffer=True)
print_summary(perf_times, header='scipy.fftpack.fft' + arg_signature(tmp))


tmp = mat_z.astype(np.csingle)
perf_times = time_func(np.fft.fft2, tmp, dict(), refresh_buffer=False)
print_summary(perf_times, header='np.fft.fft2' + arg_signature(tmp))

tmp = mat_z.astype(np.csingle)
perf_times = time_func(scipy.fftpack.fft2, tmp, dict(overwrite_x=True), refresh_buffer=True)
print_summary(perf_times, header='scipy.fftpack.fft2' + arg_signature(tmp))


tmp = arr_z.astype(np.csingle)
perf_times = time_func(np.fft.fftn, tmp, dict(), refresh_buffer=False)
print_summary(perf_times, header='np.fft.fftn' + arg_signature(tmp))

tmp = arr_z.astype(np.csingle)
perf_times = time_func(scipy.fftpack.fftn, tmp, dict(overwrite_x=True), refresh_buffer=True)
print_summary(perf_times, header='scipy.fftpack.fftn' + arg_signature(tmp))
