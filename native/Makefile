SOURCES = fft_bench.c

CC = icc
CFLAGS = -m64 -fPIC -fp-model strict -O3 -g -fomit-frame-pointer \
	 -DNDEBUG -qopenmp -xSSE4.2 -axCORE-AVX2,COMMON-AVX512 \
	 -lmkl_rt -DDEBUG -Wall -pedantic

all: fft_bench

clean:
	rm -f *.o fft_bench

fft_bench: $(SOURCES:.c=.o)
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

