BENCHMARKS = fft_cdp-in-c fft_cdp-in-nc fft_cdp-out-c \
	     fft_cdp-out-nc fft_rdp-out-c fft_rdp-out-nc \
	     rfft_rdp-out-c rfft_rdp-out-nc \
	     fft_rdp2-out-c fft2_cdp-in-c fft2_cdp-in-nc \
	     fft3_cdp-in-c fft3_cdp-in-nc fft2_cdp-out-c \
	     fft2_cdp-out-nc fft3_cdp-out-c fft3_cdp-out-nc

CC = icc
CFLAGS = -m64 -fPIC -fp-model strict -O3 -g -fomit-frame-pointer \
	 -DNDEBUG -qopenmp -xSSE4.2 -axCORE-AVX2,COMMON-AVX512 \
	 -lmkl_rt

CINCLUDE := $(addprefix -I,$(CINCLUDE))

all: $(addsuffix .exe,$(addprefix bin/,$(BENCHMARKS)))

clean:
	rm -rf bin

bin:
	mkdir -p bin

bin/%.exe: %.c | bin
	$(CC) $< $(CFLAGS) $(CINCLUDE) $(LDFLAGS) -o $@
