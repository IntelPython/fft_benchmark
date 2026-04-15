# Copyright (c) 2017-2025 Intel Corporation.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import gc
import importlib
import os
from typing import Any, Callable, NamedTuple

import numpy as np


class Timer(NamedTuple):
    name: str
    module: Any
    now: Callable[[], float]
    time_delta: Callable[[float, float], float]


def get_timer(time_modules: tuple[str, ...] = ('itimer', 'timeit', 'time')) -> Timer:
    '''
    Get some timer which we can use for benchmarking.

    Parameters
    ----------
    time_modules : iterable, default ('itimer', 'timeit', 'time')
        Timer modules to try, in order of preference

    Returns
    -------
    Timer
        Timer object (namedtuple) with attributes:
        name : str
            name of timer
        module : Python module
            actual Python module containing timer
        now : function
            function to get some description of the current time
        time_delta : function(t0, t1)
            function to find the delta in seconds between two executions of
            now()
    '''
    for mod_name in time_modules:
        try:
            timer_module = importlib.import_module(mod_name)
        except ImportError:
            pass
        else:
            timer_name = mod_name
            break

    now = {
        'itimer': lambda: timer_module.itime(),
        'timeit': lambda: timer_module.default_timer(),  # == time.perf_counter since Python 3.3
        'time': lambda: timer_module.perf_counter()
    }[timer_name]

    time_delta = {
        'itimer': lambda t0, t1: timer_module.itime_delta_in_seconds(t0, t1),
        'timeit': lambda t0, t1: t1 - t0,
        'time': lambda t0, t1: t1 - t0
    }[timer_name]

    return Timer(timer_name, timer_module, now, time_delta)


def set_threads(num_threads: int | None = None, verbose: bool = False,
                no_guessing: bool = False) -> tuple[int | None, str]:
    '''
    Get and set the number of threads used by FFT libraries.

    Parameters
    ----------
    num_threads : int, default None
        Number of threads requested. If None, do not set threads.
    verbose : bool, default False
        If True, output debug messages to STDOUT.
    no_guessing : bool, default false
        If False and MKL is not found at all, return a guess of 1 thread
        since numpy.fft and scipy.fftpack are single-threaded without MKL.
        If True, return len(os.sched_getaffinity(0)) or os.cpu_count().

    Returns
    -------
    int or None
        The number of threads successfully set, or None on failure.
    '''

    try:
        import mkl
    except ImportError:
        if hasattr(np, '__mkl_version__') or no_guessing:
            # MKL present but no mkl-service, so guess number of CPUs
            if verbose:
                print(f'TAG: WARNING: mkl-service module was not '
                      f'found. Number of threads is likely inaccurate!')
            if hasattr(os, 'sched_getaffinity'):
                return len(os.sched_getaffinity(0)), 'os.sched_getaffinity'
            else:
                return os.cpu_count(), 'os.cpu_count'
        else:
            # no MKL, so assume not threaded
            return 1, 'guessing'
    else:
        if num_threads:
            mkl.set_num_threads(num_threads)
        return mkl.get_max_threads(), 'mkl.get_max_threads'


def get_random_state_and_name(seed: int = 7777) -> tuple[np.random.RandomState, str]:
    """Return (RandomState, name) for legacy callers (e.g. scipy_paper/)."""
    rs = np.random.RandomState(seed)
    return rs, 'numpy.random.RandomState'


def get_random_state(seed: int = 7777) -> np.random.RandomState:
    """Return a legacy RandomState. Kept for scipy_paper/ backward compat."""
    return get_random_state_and_name(seed)[0]


def get_generator_and_name(seed: int = 7777) -> tuple[np.random.Generator, str]:
    """Return (Generator, name) using modern NumPy random API."""
    rng = np.random.default_rng(seed)
    return rng, 'numpy.random.Generator'


def get_generator(seed: int = 7777) -> np.random.Generator:
    """Return a modern numpy.random.Generator."""
    return get_generator_and_name(seed)[0]


def print_environment_info() -> None:
    """Print TAG lines with conda env and MKL version info to stdout."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV',
                               'None, -- conda not activated --')
    print(f"TAG: CONDA_DEFAULT_ENV = {conda_env}")
    try:
        print(f'TAG: numpy.__mkl_version__ = {np.__mkl_version__}')
    except AttributeError:
        print('TAG: numpy.__mkl_version__ = None')


def time_func(func: Callable, x: np.ndarray, kwargs: dict,
              timer: Timer | None = None, batch_size: int = 16,
              repetitions: int = 24, refresh_buffer: bool = True,
              verbose: bool = False) -> np.ndarray:
    """
    Time evaluation of func(x, **kwargs) and report the total time of
    `batch_size` evaluations, and produces `repetitions` measurements.

    If `refresh_buffer` is set to True, the input array is copied into the
    buffer before every call to func. This is useful for timing of functions
    working in-place.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError('The argument x must be a Numpy array')
    if not isinstance(kwargs, dict):
        raise ValueError('The keywords must be a dictionary, corresponding to '
                         'keyword argument to func')
    if not timer:
        timer = get_timer()
        if verbose:
            print(f'TAG: timer = {timer.name}')
    #
    times_list = np.empty((repetitions,), dtype=np.float64)

    # allocate the buffer
    buf = np.empty_like(x)
    np.copyto(buf, x)

    # warm-up
    gc.collect()
    gc.disable()
    t0 = timer.now()
    res = func(buf, **kwargs)
    t1 = timer.now()
    time_tot = timer.time_delta(t0, t1)

    # Determine optimal batch_size
    actual_batch_size = batch_size
    if time_tot * batch_size > 5:
        actual_batch_size = 1 + int(5/time_tot)

    if verbose:
        print(f'TAG: batch_size={batch_size}, repetitions={repetitions}, '
              f'refresh_buffer={refresh_buffer}, '
              f'actual_batch_size={actual_batch_size}')

    # start measurements
    for i in range(repetitions):
        time_tot = 0
        if refresh_buffer:
            for _ in range(actual_batch_size):
                np.copyto(buf, x)
                t0 = timer.now()
                res = func(buf, **kwargs)
                t1 = timer.now()
                time_tot += timer.time_delta(t0, t1)
        else:
            t0 = timer.now()
            for _ in range(actual_batch_size):
                res = func(buf, **kwargs)
            t1 = timer.now()
            time_tot += timer.time_delta(t0, t1)
        #
        times_list[i] = time_tot / actual_batch_size
    gc.enable()
    return times_list


def print_summary(data: np.ndarray | list, header: str = '') -> None:
    a = np.asarray(data)
    print(f"TAG: {header}")
    print(f'{np.min(a):0.3f}, {np.median(a):0.3f}, {np.max(a):0.3f}')
    print("", flush=True)


def arg_signature(ar: np.ndarray) -> str:
    if ar.flags['C_CONTIGUOUS']:
        qual = 'C-contig.'
    elif ar.flags['F_CONTIGUOUS']:
        qual = 'F-contig.'
    else:
        if np.all(np.array(ar.strides) % ar.itemsize == 0):
            # strides multiple of element size
            qual = f'strides: {tuple(x // ar.itemsize for x in ar.strides)} ' \
                   f'elems'
        else:
            # strides not divisible by element size
            qual = f'strides: {ar.strides} bytes'
    return f' arg: shape: {ar.shape}, dtype: {ar.dtype}, {qual}'


def measure_and_print(fn: Callable, ar: np.ndarray, kw: dict, **opts) -> np.ndarray:
    perf_times = time_func(fn, ar, kw, **opts)
    print_summary(perf_times,
                  header=f'{fn.__name__}({arg_signature(ar)}, {kw})')
    return perf_times
