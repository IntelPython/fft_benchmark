import numpy as np

try:
    import itimer as it
    use_itimer = True
    def now():
        """Returns current time moment as an integer"""
        return it.itime()
    def time_delta(t_start, t_finish):
        """Computes seconds elapsed between two time moments"""
        return it.itime_delta_in_seconds(t_start, t_finish)
except ImportError:
    import time as t
    use_itimer = False
    def now():
        """Returns current time moment, as number of seconds since an epoch"""
        return t.time()
    def time_delta(t_start, t_finish):
        """Computes seconds elapsed between two time moments"""
        return t_finish - t_start
finally:
    if use_itimer:
        print("TAG: Using itimer")
    else:
        print("TAG: Using time")


_random_seed_benchmark_default_ = 777777
def get_random_state(seed=_random_seed_benchmark_default_):
    try:
        import numpy.random_intel as rnd
        rs = rnd.RandomState(seed, brng='MT19937')
        print("TAG: Using np.random_intel")
    except ImportError:
        import numpy.random as rnd
        rs = rnd.RandomState(seed)
        print("TAG: Using np.random")
    except:
        rs = None
        raise ValueError("Failed to initialize RandomState")
    #
    return rs


import os

conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None, -- conda not activated --')
print("TAG: CONDA_DEFAULT_ENV = " + conda_env)
try:
    print('TAG: numpy.__mkl_version__ = %s' % np.__mkl_version__)
except AttributeError:
    print('TAG: numpy.__mkl_version__ = None')


import gc
def time_func(func, arg_array, keywords, batch_size=16, repetitions=24, refresh_buffer=True):
    """
    Timing function time_func(func, arg_array, keywords) times evaluation of
    func(arg_array, **keywords). It reports the total time of `batch_size` evaluations, and
    produces `repetitions` measurements.

    If `refresh_buffer` is set to True, the input array is copied into the buffer
    before every call to func. This is useful for timing of functions working in-place.
    """
    if not isinstance(arg_array, np.ndarray):
        raise ValueError("The argument must be a Numpy array")
    if not isinstance(keywords, dict):
        raise ValueError("The keywords must be a dictionary, corresponding to keyword argument to func")
    #
    print("TAG: batch_size="  + str(batch_size)  + ", " +
               "repetitions=" + str(repetitions) + ", " +
               "refresh_buffer = " + str(refresh_buffer) )
    times_list = np.empty((repetitions,), dtype=np.float64)
    # allocate the buffer
    buf = np.empty_like(arg_array)
    np.copyto(buf, arg_array)
    # warm-up
    gc.collect()
    gc.disable()
    f = func(buf, **keywords)
    # start measurements
    for i in range(repetitions):
        time_tot = 0
        if refresh_buffer:
            for _ in range(batch_size):
                np.copyto(buf, arg_array)
                t0 = now()
                f = func(buf, **keywords)
                t1 = now()
                time_tot += time_delta(t0, t1)
        else:
            t0 = now()
            for _ in range(batch_size):
                f = func(buf, **keywords)
            t1 = now()
            time_tot += time_delta(t0, t1)
        #
        times_list[i] = time_tot
    gc.enable()
    return times_list


def print_summary(data, header=''):
    a = np.array(data)
    print("TAG: " + header)
    print('{min:0.3f}, {med:0.3f}, {max:0.3f}'.format(
        min=np.min(a), med=np.median(a), max=np.max(a)
    ))
    print("", flush=True)


def arg_signature(ar):
    if ar.flags['C_CONTIGUOUS']:
        qual = 'C-contig.'
    elif ar.flags['F_CONTIGUOUS']:
        qual = 'F-contig.'
    else:
        if np.all( np.array(ar.strides) % ar.itemsize == 0):
            qual = 'srides:' + repr(tuple( x // ar.itemsize for x in ar.strides)) + ' elems'
        else:
            qual = 'strides:' + repr(ar.strides) + ' bytes'
    return ' arg: shape:' + repr(ar.shape) + ' dtype:' + repr(ar.dtype) + ' ' + qual


def measure_and_print(fn, ar, kw, **opts):
    perf_times = time_func(fn, ar, kw, **opts)
    print_summary(perf_times, header=fn.__name__ + '( ' +  arg_signature(ar) + ', ' + repr(kw) + ' )')
    return perf_times
