@ECHO off

SET _MKL_DIR=C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.2.187\windows\mkl

FOR %%f IN (
     fft_cdp-in-c
     fft_cdp-in-nc
     fft_cdp-out-c
     fft_cdp-out-nc
     fft_rdp-out-c
     fft_rdp-out-nc
     fft_rdp2-out-c
     fft2_cdp-in-c
     fft2_cdp-in-nc
     fft3_cdp-in-c
     fft3_cdp-in-nc
     fft2_cdp-out-c
     fft2_cdp-out-nc
     fft3_cdp-out-c
     fft3_cdp-out-nc
) DO (
    icl /c /nologo /O3 /MD /W3 /Qstd=c99 /fp:strict /QxSSE4.2 /QaxCORE-AVX2,COMMON-AVX512 /Qopenmp -I "%_MKL_DIR%\include" /D_CRT_SECURE_NO_WARNINGS /DNDEBUG %%f.c
    icl /nologo %%f.obj /Febuild/%%f.exe /link /LIBPATH:"%_MKL_DIR%\lib\intel64_win" mkl_rt.lib
    del %%f.obj
)
