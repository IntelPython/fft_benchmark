# Native benchmark code for MKL DFTI

## Compiling on Linux
- To compile, source compiler and run `make`.
- Run with `./fft_bench`.

## Compiling on Windows
- Source compiler and MKL, then run `win_compile_all.bat`.
  ```
  > "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\bin\compilervars.bat intel64"
  > "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl\bin\mklvars.bat intel64"
  > win_compile_all.bat
  ```
- To run, run `fft_bench.exe`. Note that long options are not supported on
  Windows. Use short options instead.

## Usage

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
  -h, --help               print this message and exit

The size argument specifies the input matrix size as a tuple of positive
decimal integers, delimited by any non-digit. For example, both
(101, 203, 305) and 101x203x305 denote the same 3D FFT.
```
