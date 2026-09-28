# FFT benchmarks for NumPy\* and SciPy\*

This FFF benchmarking framework is useful to measure FFT performance of different NumPy and SciPy versions and vendors. 
In addition to Python implementation, it is also possible to benchmark native code (MKL DFTI) implementations of these benchmarks with similar command-line interfaces.

## Python benchmarks

The following example creates a benchmarking environment with NumPy, SciPy and
mkl_fft from the Intel channel in conda:

```bash
conda create -n fft_benchmark --override-channels -c https://software.repos.intel.com/python/conda/ -c conda-forge numpy scipy mkl_fft mkl-service
conda activate fft_benchmark
```

NumPy in this environment is the stock conda-forge build, so `numpy.fft` and
`scipy.fft` use their default pocketfft implementation unless routed to
mkl_fft. Use `--numpy-backend mkl` and `--scipy-backend mkl` to benchmark
oneMKL through these APIs. Both options fail with an error if mkl_fft cannot
be used.

To run the FFT benchmark framework in Python, type:

```bash
python fft_bench.py [-h] [args] size
```

The framework performs an initial warmup call to the respective FFT API, and
then prints 24 (default) measurements, each the average time of 16 (default)
FFT computations in a loop. Measurements are printed to STDOUT as CSV rows.

Other printed lines which start with 'TAG: ' are printed for information
purposes, including the NumPy, mkl_fft and oneMKL versions in use.

### Examples

Benchmark a 2D out-of-place FFT of a `complex128` array of size `(10000, 10000)`:

```bash
python fft_bench.py 10000x10000
```

Benchmark a 1D in-place FFT of a `float32` array of size `100000000`, print
only 5 measurements, only compute the first half of the conjugate-even
DFT coefficients, and allow the FFT backend to only use one thread:

```bash
python fft_bench.py -P -r -t 1 -d float32 -o 5 100000000
```

Benchmark a 3D in-place FFT of a `complex64` array of size `1001x203x3005`,
printing only 5 measurements, each of which average over 24 inner loop
computations:

```bash
python fft_bench.py -P -d complex64 -o 5 -i 24 1001x203x3005
```

Benchmark the same 2D FFT with `numpy.fft` and `scipy.fft` routed through
mkl_fft, using 4 threads:

```bash
python fft_bench.py -m numpy.fft --numpy-backend mkl -t 4 10000x10000
python fft_bench.py -m scipy.fft --scipy-backend mkl -t 4 10000x10000
```

## Native benchmarks

### Compiling on Linux
- Source compiler and MKL, then run `make`.

```bash
source /path_to_oneapi/compiler/latest/env/vars.sh
source /path_to_oneapi/mkl/latest/env/vars.sh
make
```

- Run with `./fft_bench [args] size`.

### Compiling on Windows
- Source compiler and MKL, then run `win_compile_all.bat`.

  ```
  > "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\env\vars.bat"
  > "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\env\vars.bat"
  > win_compile_all.bat
  ```

- Run with `fft_bench.exe [args] size`. Note that long options are not supported on
  Windows. Use short options instead.

### Examples

Benchmark a 2D out-of-place FFT of a `complex128` array of size `(10000, 10000)`:

```bash
./fft_bench 10000x10000
```

Benchmark a 1D in-place FFT of a `float32` array of size `100000000`, print
only 5 measurements, only compute the first half of the conjugate-even
DFT coefficients, allow the FFT backend to only use one thread, and cache
the DFTI descriptor between inner loop runs (similar behavior to `mkl_fft` for
single dimensional FFTs).

```bash
./fft_bench -P -c -r -t 1 -d float32 -o 5 100000000
```

Benchmark a 3D in-place FFT of a `complex64` array of size `1001x203x3005`,
printing only 5 measurements, each of which average over 24 inner loop
computations:

```bash
./fft_bench -P -d complex64 -o 5 -i 24 1001x203x3005
```

Without `-c`, every timed FFT call also creates, commits and frees the DFTI
descriptor, so descriptor setup is part of each measurement. With `-c`, setup
runs once per outer loop and its time is spread over the inner loops. Use `-c`
to measure FFT compute, especially for small sizes where setup can dominate.

## Comparing platforms

To compare FFT performance across machines, keep everything except the
hardware the same:

- Use the same oneMKL, mkl_fft, NumPy and SciPy versions on every machine, and
  keep the `TAG:` lines with the results.
- Run the same shapes, dtypes and thread counts on every machine. Include the
  thread counts that matter for the workload, such as 1 thread and all physical
  cores.
- For the native benchmark, use `-c`.
- Record CPU frequency scaling settings (governor, turbo) and memory
  configuration, since both affect the results.

### Usage

```
usage: ./fft_bench [args] size
Benchmark FFT using Intel(R) MKL DFTI.

FFT problem arguments:
  -t, --threads=THREADS    use THREADS threads for FFT execution
                           (default: use MKL's default)
  -d, --dtype=DTYPE        use DTYPE as the FFT domain. For a list of
                           understood dtypes, use '-d help'.
                           (default: complex128)
  -r, --rfft               do not copy superfluous harmonics when FFT
                           output is even-conjugate, i.e. for real inputs
  -P, --in-place           allow overwriting the input buffer with the
                           FFT outputs
  -c, --cached             use the same DFTI descriptor for the same
                           outer loop, i.e. "cache" the descriptor

Timing arguments:
  -i, --inner-loops=IL     time the benchmark IL times for each printed
                           measurement. Copies are not included in the
                           measurements. (default: 16)
  -o, --outer-loops=OL     print OL measurements. (default: 5)

Output arguments:
  -p, --prefix=PREFIX      output PREFIX as the first value in outputs
                           (default: 'Native-C')
  -H, --no-header          do not output CSV header. This can be useful
                           if running multiple benchmarks back-to-back.
  -h, --help               print this message and exit.

The size argument specifies the input matrix size as a tuple of positive
decimal integers, delimited by any non-digit. For example, both
(101, 203, 305) and 101x203x305 denote the same 3D FFT.
```

## See also
"[Accelerating Scientific Python with Intel
Optimizations](https://proceedings.scipy.org/articles/shinma-7f4c6e7-00f)"
by Oleksandr Pavlyk, Denis Nagorny, Andres Guzman-Ballen, Anton Malakhov, Hai
Liu, Ehsan Totoni, Todd A. Anderson, Sergey Maidanov. Proceedings of the 16th
Python in Science Conference (SciPy 2017), July 10 - July 16, Austin, Texas
