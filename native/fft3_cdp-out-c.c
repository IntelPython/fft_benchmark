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
    MKL_Complex16 *x = 0, *buf = 0;
    /* Execution status */
    MKL_LONG status = 0, err = 0;
    DFTI_DESCRIPTOR_HANDLE hand = 0;
    VSLStreamStatePtr stream;
    double d_zero = 0.0, d_one = 1.0;

    double *re_vec = 0, *im_vec = 0;
    moment_t t0, t1;
    moment_t time_tot = 0;
    int i, it, reps = 0, samps = 0, si;
    size_t N1 = 0, N2 = 0, N3 = 0, N = 0;
    MKL_LONG dims[3];
    MKL_LONG strides[4];

#define DESCRIPTION_STR "Complex double, out-of-place, cached"
#include "read3_n_echo_env.inc"

    N = N1 * N2 * N3;

    err = vslNewStream(&stream, VSL_BRNG_MT19937, SEED);
    assert(err == VSL_STATUS_OK);

    dims[0] = N1;  dims[1] = N2; dims[2] = N3;
    N = N1 * N2 * N3;
    assert( N > 0);
    /* input matrix */
    re_vec = (double *) mkl_malloc(N * sizeof(double), 64);
    assert(re_vec);
    im_vec = (double *) mkl_malloc(N * sizeof(double), 64);
    assert(im_vec);

    x = (MKL_Complex16 *) mkl_malloc(N * sizeof(MKL_Complex16), 64);
    assert(x);

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, N, re_vec, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, N, im_vec, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    for(i=0; i < N; i++) {
        x[i].real = re_vec[i];
        x[i].imag = im_vec[i];
    }
    mkl_free(re_vec);
    mkl_free(im_vec);

    warm_up_threads();

    for(si=0; si<samps; time_tot=0, si++) {
        t0 = moment_now();

        status = DftiCreateDescriptor(
            &hand,
            DFTI_DOUBLE,
            DFTI_COMPLEX,
            3,
            dims);
        assert(status == 0);
        status = DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        assert(status == 0);

        strides[0] = 0;
        strides[1] = N3 * N2;
        strides[2] = N3;
        strides[3] = 1;
        status = DftiSetValue(hand, DFTI_INPUT_STRIDES, strides);
        assert(status == 0);
        status = DftiSetValue(hand, DFTI_OUTPUT_STRIDES, strides);
        assert(status == 0);

        status = DftiSetValue(hand, DFTI_FORWARD_SCALE, 1.0);
        assert(status == 0);

        status = DftiSetValue(hand, DFTI_BACKWARD_SCALE, 1.0/(N1 * N2 * N3));
        assert(status == 0);

        status = DftiCommitDescriptor(hand);
        assert(status == 0);

        buf = (MKL_Complex16 *) mkl_malloc(N * sizeof(MKL_Complex16), 64);
        assert(buf);

        t1 = moment_now();
        time_tot += t1 - t0;


        for(it = -1; it <reps;  it++) {
            t0 = moment_now();

            status = DftiComputeForward(hand, x, buf);
            assert(status == 0);

            t1 = moment_now();

            if (it >= 0) time_tot += t1 - t0;
        }

        t0 = moment_now();

        status = DftiFreeDescriptor(&hand);
        assert(status == 0);

        t1 = moment_now();
        time_tot += t1 - t0;

        printf("%.5g\n", seconds_from_moment(time_tot));
    }

#include "print_buf.inc"

    mkl_free(buf);

    err = vslDeleteStream(&stream);
    assert(err == VSL_STATUS_OK);

    mkl_free(x);

    return 0;
}
