# Copyright (c) 2017-2025 Intel Corporation.
#
# SPDX-License-Identifier: MIT

import argparse
import contextlib
import importlib
import inspect
import numpy as np
import scipy.fft
import os
import perf
import re


# Mark which FFT submodules are available...
fft_modules = {'numpy.fft': np.fft, 'scipy.fft': scipy.fft}

def valid_shape(shape_str):
    shape = re.sub(r'[^\d]+', 'x', shape_str).strip('x').split('x')
    shape = tuple(int(i) for i in shape)
    if len(shape) < 1 or any(i < 1 for i in shape):
        raise argparse.ArgumentTypeError(f'parsed shape {shape} has '
                                         'non-positive entries or less than '
                                         'one dimension.')
    return shape


def valid_dtype(dtype_str):
    dtype = np.dtype(dtype_str)
    if dtype.kind not in ('f', 'c'):
        raise argparse.ArgumentTypeError('only complex or real floating-point '
                                         'data-types are allowed')
    return dtype


# Parse args
parser = argparse.ArgumentParser(description='Benchmark FFT using NumPy and '
                                 'SciPy.')

fft_group = parser.add_argument_group(title='FFT problem arguments')
fft_group.add_argument('-t', '--threads', '--num-threads', '--core-number',
                       type=int, default=perf.set_threads(no_guessing=True)[0],
                       help='Number of threads to use for FFT computation. '
                       '%(prog)s will attempt to use mkl-service to get/set '
                       'number of threads globally, and will also try to '
                       'set number of workers in scipy.fft. (default in this '
                       'environment: %(default)d)')
fft_group.add_argument('-m', '--modules', '--submodules', nargs='*',
                       default=tuple(fft_modules.keys()),
                       choices=tuple(fft_modules.keys()),
                       help='Use FFT functions from MODULES. (default: '
                       '%(default)s)')
fft_group.add_argument('-d', '--dtype', default=np.dtype('complex128'),
                       type=valid_dtype,
                       help='use DTYPE as the FFT domain. DTYPE must be '
                       'specified such that it is parsable by numpy.dtype. '
                       '(default: %(default)s)')
fft_group.add_argument('-r', '--rfft', default=False, action='store_true',
                       help='do not copy superfluous harmonics when FFT '
                       'output is conjugate-even, i.e. for real inputs.')
fft_group.add_argument('-P', '--overwrite-x', '--in-place', default=False,
                       action='store_true', help='Allow overwriting the input '
                       'buffer with the FFT outputs')
fft_group.add_argument('-s', '--seed', default=7777, type=int,
                       help='Seed for random number generator')
fft_group.add_argument('--scipy-backend', default='stock',
                       choices=('stock', 'mkl'),
                       help='Which uarray backend to use for scipy.fft. '
                       '"stock" = default pocketfft. "mkl" = register '
                       'mkl_fft.interfaces.scipy_fft via scipy.fft.set_backend '
                       'around the timed region. Ignored for numpy.fft. '
                       '(default: %(default)s)')

timing_group = parser.add_argument_group(title='Timing arguments')
timing_group.add_argument('-i', '--inner-loops', '--batch-size',
                          type=int, default=16, metavar='IL',
                          help='time the benchmark IL times for each printed '
                          'measurement. Copying is not timed. (default: '
                          '%(default)s)')
timing_group.add_argument('-o', '--outer-loops', '--samples', '--repetitions',
                          type=int, default=24, metavar='OL',
                          help='print OL measurements. (default: %(default)s)')

output_group = parser.add_argument_group(title='Output arguments')
output_group.add_argument('-p', '--prefix', default='python',
                          help='Output PREFIX as the first value in outputs '
                          '(default: %(default)s)')
output_group.add_argument('-H', '--no-header', default=True,
                          action='store_false', dest='header',
                          help='do not output CSV header. This can be useful '
                          'if running multiple benchmarks back to back.')
output_group.add_argument('-v', '--verbose', default=False,
                          action='store_true',
                          help='Print excessive debug messages')

parser.add_argument('shape', type=valid_shape,
                    help='FFT shape to run, specified as a tuple of positive '
                    'decimal integers, delimited by any non-digit characters. '
                    'For example, both (101, 203, 305) and 101x203x305 denote '
                    'the same 3D FFT.')

args = parser.parse_args()

# Resolve optional mkl_fft scipy uarray backend. Doing this once up front
# lets us fail fast with a clear message if the user asked for "mkl" but
# the package is missing, rather than silently falling through to pocketfft
# the way a plain scipy.fft.set_backend call would.
_mkl_scipy_backend = None
if args.scipy_backend == 'mkl':
    try:
        import mkl_fft.interfaces.scipy_fft as _mkl_scipy_backend
    except ImportError as e:
        parser.error(f'--scipy-backend mkl requested but mkl_fft is not '
                     f'importable in this environment: {e}')

# Print environment info (conda env, MKL version)
perf.print_environment_info()

# Get timer
timer = perf.get_timer()
if args.verbose:
    print(f'TAG: timer = {timer.name}')

# Set threads
threads, threading_info_source = perf.set_threads(num_threads=args.threads,
                                                  verbose=args.verbose)
if args.verbose:
    print(f'TAG: threading_info_source = {threading_info_source}')

# Get function from shape
assert len(args.shape) >= 1
func_name = {1: 'fft', 2: 'fft2'}.get(len(args.shape), 'fftn')
if args.rfft:
    func_name = 'r' + func_name

if args.rfft and args.dtype.kind == 'c':
    parser.error('--rfft makes no sense for an FFT of complex inputs. The '
                 'FFT output will not be conjugate even, so the whole output '
                 'matrix must be computed!')

# Generate input data
rs, rs_name = perf.get_generator_and_name(seed=args.seed)
if args.verbose:
    print(f'TAG: random = {rs_name}')
arr = rs.standard_normal(args.shape)
if args.dtype.kind == 'c':
    arr = arr + rs.standard_normal(args.shape) * 1j
arr = np.asarray(arr, dtype=args.dtype)
if args.verbose:
    print(f'TAG:{perf.arg_signature(arr)}')

# Print header
print("", flush=True)
if args.header:
    print('prefix,module,function,threads,dtype,size,place,time', flush=True)

# Run benchmarks. One for each selected module
for mod_name in args.modules:
    # Determine arguments to benchmark and get function
    mod = fft_modules[mod_name]
    func = getattr(mod, func_name)
    kwargs = {}
    time_kwargs = dict(timer=timer, batch_size=args.inner_loops,
                       repetitions=args.outer_loops,
                       refresh_buffer=False, verbose=args.verbose)
    in_place = False
    actual_threads = threads

    # Inspect function to see if it allows overwrite_x.
    # For example, numpy.fft functions do not accept overwrite_x.
    sig = inspect.signature(func)
    if 'overwrite_x' in sig.parameters:
        in_place = kwargs['overwrite_x'] = args.overwrite_x
        time_kwargs['refresh_buffer'] = in_place
    else:
        # Skip this if we needed overwrite_x but didn't get it
        if args.overwrite_x:
            continue
    if 'workers' in sig.parameters:
        actual_threads = kwargs['workers'] = args.threads

    # Route scipy.fft through mkl_fft's uarray backend when requested.
    # numpy.fft has no uarray dispatch, so the context is a no-op there.
    if mod_name == 'scipy.fft' and _mkl_scipy_backend is not None:
        backend_ctx = scipy.fft.set_backend(_mkl_scipy_backend, only=True)
        effective_backend = 'mkl'
    else:
        backend_ctx = contextlib.nullcontext()
        effective_backend = 'stock' if mod_name == 'scipy.fft' else 'n/a'

    if args.verbose:
        print(f'TAG: scipy_backend = {effective_backend}')

    with backend_ctx:
        # threads warm-up — inside the backend context so the warmup path
        # matches the timed path exactly (same dispatcher, same planner).
        buf = np.empty_like(arr)
        np.copyto(buf, arr)
        x1 = func(buf)
        del x1
        del buf

        perf_times = perf.time_func(func, arr, kwargs, **time_kwargs)

    # Tag the prefix with the effective backend so CSV rows from the
    # two scipy passes (stock vs mkl in the same env) stay distinguishable.
    if mod_name == 'scipy.fft':
        row_prefix = f'{args.prefix}-scipy-{effective_backend}'
    else:
        row_prefix = args.prefix
    for t in perf_times:
        print(f'{row_prefix},{mod_name},{func_name},{actual_threads},'
              f'{arr.dtype.name},{"x".join(str(i) for i in args.shape)},'
              f'{"in-place" if in_place else "out-of-place"},{t:.5g}')
