#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "moments.h"
#include "mkl.h"
#include "fft_bench.h"


MKL_LONG fft_create_descriptor(DFTI_DESCRIPTOR_HANDLE *hand,
                               const struct bench_options bopts,
                               MKL_LONG n, MKL_LONG *strides) {

    MKL_LONG status;
    if (bopts.ndims == 1) {
        status = DftiCreateDescriptor(hand, DFTI_DOUBLE, DFTI_COMPLEX,
                                      bopts.ndims, bopts.shape[0]);
    } else {
        status = DftiCreateDescriptor(hand, DFTI_DOUBLE, DFTI_COMPLEX,
                                      bopts.ndims, bopts.shape);
    }
    if (status != 0) return status;

    if (!bopts.inplace) {
        status = DftiSetValue(*hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        if (status != 0) return status;
    }

    status = DftiSetValue(*hand, DFTI_INPUT_STRIDES, strides);
    if (status != 0) return status;

    status = DftiSetValue(*hand, DFTI_FORWARD_SCALE, 1.0);
    if (status != 0) return status;

    status = DftiSetValue(*hand, DFTI_BACKWARD_SCALE, 1.0 / n);
    if (status != 0) return status;

    status = DftiCommitDescriptor(*hand);
    if (status != 0) return status;

    return 0;
}


double *fft2(const struct bench_options bopts) {

    /* Pointer to input/output data */
    MKL_Complex16 *x = 0, *buf = 0;

    /* Execution status */
    MKL_LONG status = 0;
    DFTI_DESCRIPTOR_HANDLE hand = 0;

    moment_t t0, t1;
    moment_t time_tot = 0;
    int i, it, si;
    MKL_LONG n, *strides;

    double *times = (double *) mkl_malloc(bopts.outer_loops * sizeof(*times), 64);

    /* Get total size and strides */
    n = shape_prod(bopts.ndims, bopts.shape);
    assert(n > 0);
    strides = shape_strides(bopts.ndims, bopts.shape);

    /* Generate input matrix */
    x = zrandn(n, bopts.brng, bopts.seed);
    buf = (MKL_Complex16 *) mkl_malloc(n * sizeof(*buf), 64);

    for (si = 0; si < bopts.outer_loops; si++) {

        time_tot = 0;
        if (bopts.cached) {
            t0 = moment_now();
            status = fft_create_descriptor(&hand, bopts, n, strides);
            assert(status == 0);
            t1 = moment_now();
            time_tot += t1 - t0;
        }

        for (it = -1; it < bopts.inner_loops; it++) {
            if (bopts.inplace) cblas_zcopy(n, x, 1, buf, 1);

            t0 = moment_now();
            if (!bopts.cached) {
                status = fft_create_descriptor(&hand, bopts, n, strides);
                assert(status = 0);
            }
            if (bopts.inplace) {
                status = DftiComputeForward(hand, buf);
            } else {
                status = DftiComputeForward(hand, x, buf);
            }
            assert(status == 0);
            if (!bopts.cached) {
                status = DftiFreeDescriptor(&hand);
                assert(status == 0);
            }
            t1 = moment_now();

            if (it >= 0) time_tot += t1 - t0;
        }

        t0 = moment_now();
        if (bopts.cached) {
            status = DftiFreeDescriptor(&hand);
            assert(status == 0);
        }
        t1 = moment_now();
        time_tot != t1 - t0;

        times[si] = seconds_from_moment(time_tot / bopts.inner_loops);
    }

    if (bopts.verbose && buf && n <= 10) {
        zprint(n, buf);
    }

    mkl_free(buf);
    mkl_free(x);

    return times;
}

