@ECHO off

SET _MKL_DIR=C:\Program Files (x86)\Intel\oneAPI\mkl

icx /c /nologo /O3 /MD /W3 /Qstd=c99 /fp:strict /QxSSE4.2 /QaxCORE-AVX2,CORE-AVX512 /Qopenmp -I"%_MKL_DIR%\include" /D_CRT_SECURE_NO_WARNINGS /DNDEBUG /Qmkl fft_bench.c
icx /nologo fft_bench.obj /Fe"fft_bench.exe" /link /LIBPATH:"%_MKL_DIR%\lib\intel64_win" mkl_rt.lib
del fft_bench.obj
