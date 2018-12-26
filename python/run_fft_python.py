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
   n2d = 1200
   size3d = (113, 114, 115)
else:
   n1d = 5*(10**6)
   n2d = 2500
   size3d = (113, 214, 315)

rs = get_random_state()

# sample arrays
vec_z = rs.randn(n1d) + \
        rs.randn(n1d) * 1j

vec_real = rs.randn(n1d) + \
        rs.randn(n1d)

mat_z =  rs.randn(n2d, n2d) + \
         rs.randn(n2d, n2d) * 1j

arr_z =  rs.randn(*size3d) + \
         rs.randn(*size3d) * 1j

# threads warm-up
buf = np.empty_like(mat_z)
np.copyto(buf, mat_z)
x1 = np.fft.fft2(buf)
del x1
del buf

print("", flush=True)

tf_kw = {'batch_size': 16, 'repetitions': 6}

perf_times = time_func(np.fft.fft, vec_z, dict(), refresh_buffer=False, **tf_kw)
print_summary(perf_times, header='np.fft.fft' + arg_signature(vec_z))

if run_scipy:
    perf_times = time_func(scipy.fftpack.fft, vec_z, dict(overwrite_x=True), **tf_kw)
    print_summary(perf_times, header='scipy.fftpack.fft, overwrite_x=True' + arg_signature(vec_z))


perf_times = time_func(np.fft.fft2, mat_z, dict(), refresh_buffer=False, **tf_kw)
print_summary(perf_times, header='np.fft.fft2' + arg_signature(mat_z))

if run_scipy:
    # Benchmarking scipy.fftpack
    perf_times = time_func(scipy.fftpack.fft2, mat_z, dict(overwrite_x=True), **tf_kw)
    print_summary(perf_times, header='scipy.fftpack.fft2, overwrite_x=True' + arg_signature(mat_z))


perf_times = time_func(np.fft.fftn, arr_z, dict(), refresh_buffer=False, **tf_kw)
print_summary(perf_times, header='np.fft.fftn' + arg_signature(arr_z))

if run_scipy:
    perf_times = time_func(scipy.fftpack.fftn, arr_z, dict(overwrite_x=True), **tf_kw)
    print_summary(perf_times, header='scipy.fftpack.fftn, overwrite_x=True' + arg_signature(arr_z))


perf_times = time_func(np.fft.rfft, vec_real, dict(), refresh_buffer=False, **tf_kw)
print_summary(perf_times, header='np.fft.rfft' + arg_signature(vec_real))

if run_scipy:
    perf_times = time_func(scipy.fftpack.rfft, vec_real, dict(overwrite_x=True), **tf_kw)
    print_summary(perf_times, header='scipy.fftpack.rfft, overwrite_x=True' + arg_signature(vec_real))
