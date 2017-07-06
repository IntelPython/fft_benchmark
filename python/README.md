# FFT benchmark

This set of benchmarks measures performance of FFT computations, serving to highlight performance improvements to FFT computations in NumPy and SciPy in the Intel Distribution for Python

To reproduce, install Intel Distribution for Python using ``conda install`` as follows:

```
conda install -n 'idp3' -c intel intelpython3_full
source activate idp3
```

To benchmark FFT in Python, execute

```
python run_fft_python.py
```

To benchmark FFT using native code:

```
python run_fft_native.py
```

The methodology is to perform one unmeasured computation, and then repeat 24 total timings for 16 repetitions of FFT computations in the loop. 
The 24 measurements are aggregated to report minimum, median and maximum timings, which are printed to STDOUT.

Other printed lines which start with 'TAG: ' are printed for information only, and can be filtered out if need be.