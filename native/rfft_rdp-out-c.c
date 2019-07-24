#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include "moments.h"
#include "mkl.h"
#include "omp.h"

#define SEED 7777

#include "utils.inc"

int main() {
    /* Pointer to input/output data */
    MKL_Complex16 *buf = 0;
    /* Execution status */
    MKL_LONG status = 0, err = 0;
    DFTI_DESCRIPTOR_HANDLE hand = 0;
    VSLStreamStatePtr stream;
    double d_zero = 0.0, d_one = 1.0;

    double *x = 0;
    moment_t t0, t1;
    moment_t time_tot = 0;
    int i, it, reps = 0, N = 0, samps = 0, si;

#define DESCRIPTION_STR "Real double, not in-place, cached"
#include "read_n_echo_env.inc"

    err = vslNewStream(&stream, VSL_BRNG_MT19937, SEED);
    assert(err == VSL_STATUS_OK);

    x = (double *) mkl_malloc(N * sizeof(double), 64);
    assert(x);

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, N, x, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    warm_up_threads();

    for(si = 0; si < samps; time_tot=0, si++) {
        t0 = moment_now();

        buf = (MKL_Complex16 *) mkl_malloc(N * sizeof(MKL_Complex16), 64);
        assert(buf);

        status = DftiCreateDescriptor(
            &hand,
            DFTI_DOUBLE,
            DFTI_REAL,
            1,
            N);
        assert(status == 0);

        status = DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        assert(status == 0);

        status = DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        assert(status == 0);

        status = DftiCommitDescriptor(hand);
        assert(status == 0);

        t1 = moment_now();
        time_tot += t1 - t0;

        for(it = -1; it < reps;  it++) {
            long k_dest;

            t0 = moment_now();

            status = DftiComputeForward(hand, x, buf);
            assert(status == 0);

            t1 = moment_now();

            if (it >= 0) time_tot += t1 - t0;
        }

        t0 = moment_now();

        status = DftiFreeDescriptor(&hand);
        assert(status == 0);

        mkl_free(buf);

        t1 = moment_now();
        time_tot += t1 - t0;

        printf("%.5g\n", seconds_from_moment(time_tot));
    }

    err = vslDeleteStream(&stream);
    assert(err == VSL_STATUS_OK);

    mkl_free(x);

    return 0;
}
