#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include "moments.h"
#include "mkl.h"
#include "fft_bench.h"

double *fft2_cdp_in_nc(const struct bench_options *opts) {
    /* Pointer to input/output data */
    MKL_Complex16 *x = 0, *buf = 0;
    /* Execution status */
    MKL_LONG status = 0;
    DFTI_DESCRIPTOR_HANDLE hand = 0;

    moment_t t0, t1;
    moment_t time_tot = 0;
    int i, it, si;
    MKL_LONG n, strides[3];

    double *times = (double *) mkl_malloc(opts->outer_loops * sizeof(double), 64);

    n = opts->shape[0] * opts->shape[1];
    assert(n > 0);

    /* input matrix */
    x = zrandn(n, opts->brng, opts->seed);
    buf = (MKL_Complex16 *) mkl_malloc(n * sizeof(MKL_Complex16), 64);
    assert(buf);

    strides[0] = 0;
    strides[1] = opts->shape[1];
    strides[2] = 1;

    for(si = 0; si < opts->outer_loops; si++) {

        time_tot = 0;

        for(it = 0; it < opts->inner_loops; it++) {

           cblas_zcopy(n, (void *) x, 1, (void *) buf, 1); /* buf = x */
           /* memcpy(buf, x, N*sizeof(MKL_Complex16)); */

           t0 = moment_now();

           status = DftiCreateDescriptor(
                &hand,
                DFTI_DOUBLE,
                DFTI_COMPLEX,
                2,
                opts->shape);
            assert(status == 0);

            status = DftiSetValue(hand, DFTI_INPUT_STRIDES, strides);
            assert(status == 0);

            status = DftiSetValue(hand, DFTI_FORWARD_SCALE, 1.0);
            assert(status == 0);

            status = DftiSetValue(hand, DFTI_BACKWARD_SCALE, 1.0 / n);
            assert(status == 0);

            status = DftiCommitDescriptor(hand);
            assert(status == 0);

            status = DftiComputeForward(hand, buf);
            assert(status == 0);

            status = DftiFreeDescriptor(&hand);
            assert(status == 0);

            t1 = moment_now();

            if (it >= 0) time_tot += t1 - t0;
        }

        times[si] = seconds_from_moment(time_tot / opts->inner_loops);
    }

    if (opts->verbose && buf && n <= 10) {
        zprint(n, buf);
    }

    mkl_free(buf);
    mkl_free(x);

    return times;
}
