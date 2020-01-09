# Native benchmark code for MKL DFTI

- To compile, source compiler and run `make`.
- Run with `./fft_bench`.

#### Compilation
*  source compiler, run `./compiler_all.sh`, which will create `*.exe` files in `build/` directory.

#### Execution

* Executables respond to the following environment variable settings:
  * `REPS` - number of repetitions of the call for a single sample (default 16)
  * `S` - number of timing samples to report (default 1)
  * `N` - size of 1D FFT vector
  * `N1`, `N2` - size of 2D FFT array
  * `N1`, `N2`, `N3` - size of 3D FFT array

Use of `REPS` allows to resolve issue timer's granularity. Use of `S` allows for efficient warm-up. Typically, one would use the smallest value in the sample.
