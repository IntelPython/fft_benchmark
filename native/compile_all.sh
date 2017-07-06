#!/bin/bash -x

for f in fft_cdp-in-c fft_cdp-in-nc \
    fft_cdp-out-c fft_cdp-out-nc \
    fft_rdp-out-c fft_rdp-out-nc \
    fft_rdp2-out-c \
    fft2_cdp-in-c fft2_cdp-in-nc \
    fft3_cdp-in-c fft3_cdp-in-nc \
    fft2_cdp-out-c fft2_cdp-out-nc \
    fft3_cdp-out-c fft3_cdp-out-nc
do
    ./compile.sh $f.c build/$f.exe
done
