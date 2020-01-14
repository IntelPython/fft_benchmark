# FFT benchmark

This set of benchmarks measures performance of FFT computations, serving to
highlight performance improvements to FFT computations in NumPy and SciPy in
the Intel(R) Distribution for Python\*.

To reproduce, install Intel(R) Distribution for Python\* as follows:

```bash
conda create -n 'idp3_fft' -c intel numpy scipy
conda activate idp3_fft
```

To benchmark FFT in Python, execute

```bash
python fft_bench.py [-h] [args] size
```

To benchmark FFT using native code, see the `native/` directory.

The methodology is to perform one unmeasured computation, and then repeat 24
total timings for 16 repetitions of FFT computations in the loop.  The 24
measurements are aggregated to report minimum, median and maximum timings,
which are printed to STDOUT.

Other printed lines which start with 'TAG: ' are printed for information only,
and can be filtered out if need be.

## See also
"[Accelerating Scientific Python with Intel
Optimizations](http://conference.scipy.org/proceedings/scipy2017/pdfs/oleksandr_pavlyk.pdf)"
by Oleksandr Pavlyk, Denis Nagorny, Andres Guzman-Ballen, Anton Malakhov, Hai
Liu, Ehsan Totoni, Todd A. Anderson, Sergey Maidanov. Proceedings of the 16th
Python in Science Conference (SciPy 2017), July 10 - July 16, Austin, Texas
