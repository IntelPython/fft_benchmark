#!/bin/bash -x

if test "$#" -ne 2; then
    echo "Expecting two arguments: source file to compile and binary name to produce"
    exit 1
fi

icc -m64 -fPIC -fp-model strict -O3 -g -fomit-frame-pointer -DNDEBUG -qopenmp -xSSE4.2 -axCORE-AVX2,COMMON-AVX512 -lmkl_rt "$1" -o "$2"

