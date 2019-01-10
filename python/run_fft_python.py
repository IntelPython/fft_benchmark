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

# Parse args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', choices=['fft', 'fft2', 'fft3', 'rfft'],
                    required=True, help='FFT types to run')
parser.add_argument('-p', '--overwrite-x', default=False, action='store_true',
                    help='Allow overwriting input array')
parser.add_argument('-s', '--shape', default=None,
                    help='FFT shape, dimensions separated by comma')

args = parser.parse_args()
shapes = {1: (n1d,), 2: (n2d, n2d), 3: size3d}
if args.shape is not None:
    shape = [int(x) for x in args.shape.split(',')]
    shapes[len(shape)] = tuple(shape)

func_name = args.type if args.type in ['fft', 'rfft', 'fft2'] else 'fftn'
func_module = np.fft if not args.overwrite_x else scipy.fftpack
func = getattr(func_module, func_name)

rs = get_random_state()

# sample arrays
if args.type == 'fft':
    arr = rs.randn(*shapes[1]) + \
          rs.randn(*shapes[1]) * 1j
elif args.type == 'rfft':
    arr = rs.randn(*shapes[1])
elif args.type == 'fft2':
    arr = rs.randn(*shapes[2]) + \
          rs.randn(*shapes[2]) * 1j
elif args.type == 'fft3':
    arr = rs.randn(*shapes[3]) + \
          rs.randn(*shapes[3]) * 1j

# threads warm-up
buf = np.empty_like(arr)
np.copyto(buf, arr)
x1 = func(buf)
del x1
del buf

print("", flush=True)

tf_kw = {'batch_size': 16, 'repetitions': 6}

if not args.overwrite_x:
    perf_times = time_func(func, arr, dict(), refresh_buffer=False, **tf_kw)
    print_summary(perf_times, header='np.fft.' + func_name + arg_signature(arr))
else:
    perf_times = time_func(func, arr, dict(overwrite_x=True), **tf_kw)
    print_summary(perf_times, header='scipy.fftpack.' + func_name +
                  ', overwrite_x=True' + arg_signature(arr))

