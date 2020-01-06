BENCHMARKS = fft_cdp-in-c fft_cdp-in-nc fft_cdp-out-c \
	     fft_cdp-out-nc fft_rdp-out-c fft_rdp-out-nc \
	     rfft_rdp-out-c rfft_rdp-out-nc \
	     fft_rdp2-out-c fft2_cdp-in-c fft2_cdp-in-nc \
	     fft3_cdp-in-c fft3_cdp-in-nc fft2_cdp-out-c \
	     fft2_cdp-out-nc fft3_cdp-out-c fft3_cdp-out-nc

SOURCES = fft_bench.c fft2_cdp-in-c.c fft2_cdp-in-nc.c fft2_cdp-out-c.c \
	  fft2_cdp-out-nc.c

CC = icc
CFLAGS = -m64 -fPIC -fp-model strict -O3 -g -fomit-frame-pointer \
	 -DNDEBUG -qopenmp -xSSE4.2 -axCORE-AVX2,COMMON-AVX512 \
	 -lmkl_rt -DDEBUG -Wall -pedantic

all: fft_bench

clean:
	rm -f *.o fft_bench

fft_bench: $(SOURCES:.c=.o)
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

