# Copyright (c) 2017-2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

SOURCES = fft_bench.c

CC = icx
CFLAGS = -m64 -fPIC -fp-model strict -O3 -g -fomit-frame-pointer \
         -DNDEBUG -qopenmp -xSSE4.2 -axCORE-AVX2,CORE-AVX512 \
         -Wall -pedantic
LDFLAGS = -lmkl_rt

ifneq ($(CONDA_PREFIX),)
    LDFLAGS += -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib
    CFLAGS += -I$(CONDA_PREFIX)/include
endif

all: fft_bench

clean:
	rm -f *.o fft_bench

fft_bench: $(SOURCES:.c=.o)
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
