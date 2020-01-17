/*
 * Copyright (C) 2017-2020 Intel Corporation.
 *
 * SPDX-License-Identifier: MIT
 */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mkl.h"
#include "moments.h"
#include "omp.h"

/* GNU specific extensions */
#ifdef __GNUC__
#include <error.h>
#include <getopt.h>
#else
/* error(...) definition specifically for windows */
#define error(status, errnum, format, ...)                                     \
    do {                                                                       \
        fflush(stderr);                                                        \
        if (errnum != 0) {                                                     \
            fprintf(stderr, "fft_bench: " format ": %s\n", ##__VA_ARGS__,      \
                    strerror(errnum));                                         \
        } else {                                                               \
            fprintf(stderr, "fft_bench: " format "\n", ##__VA_ARGS__);         \
        }                                                                      \
        if (status != 0)                                                       \
            exit(status);                                                      \
    } while (0)

/* getopt definition specifically for windows */
/* WARNING: this is a very basic implementation and does not conform
 * to standard! */
static int opterr = 1;
static int optopt;
static int optind = 1;
static char *optarg;

static int getopt(int argc, char *const *argv, const char *options) {

    if (optind == argc) {
        /* We have reached the end of arguments. Stop. */
        return -1;
    }

    if (argv[optind][0] == '-') {
        int optlen = strlen(argv[optind]);
        if (optlen == 1) {
            /* a single - as an argument: treat this as a non-option */
            return -1;
        }

        /* Search options for this character */
        optopt = argv[optind][1];
        const char *optres = strchr(options, optopt);
        if (optres == NULL) {
            /* Option not found. */
            if (opterr)
                error(0, 0, "invalid option -- '%c'", optopt);
            if (options[0] == ':')
                return ':';
            else
                return '?';
        }

        if (optres[1] == ':') {
            /* This option requires an argument */
            if (optlen > 2) {
                /* The user provided the optarg in the same argument */
                optarg = argv[optind] + 2;
                optind++;
                return optopt;
            } else {
                if (optind + 1 == argc) {
                    /* No more arguments! */
                    if (opterr)
                        error(0, 0, "option requires an argument -- '%c'",
                              optopt);
                    return '?';
                } else {
                    /* The optarg is in the next argument */
                    optarg = argv[optind + 1];
                    optind += 2;
                    return optopt;
                }
            }
        } else {
            /* This option does not require an optarg. */
            if (optlen > 2) {
                /* More options are given in the same argument. */
                memmove(argv[optind] + 1, argv[optind] + 2, optlen - 1);
            } else {
                optind++;
            }
            return optopt;
        }
    } else {
        /* We have an argument which does not begin with a '-', and it is not
         * an optarg, so it must be the beginning of non-option arguments. */
        return -1;
    }
}

/* getopt_long calling getopt */
#define getopt_long(argc, argv, options, longopts, indexptr)                   \
    getopt(argc, argv, options)
#endif

#define SEED 7777

#define CHECK_DFTI_STATUS(status, msg, ...)                                    \
    do {                                                                       \
        if (status && !DftiErrorClass((status), DFTI_NO_ERROR)) {              \
            error(1, 0, msg ": %s", ##__VA_ARGS__,                             \
                  DftiErrorMessage((status)));                                 \
            return status;                                                     \
        }                                                                      \
    } while (0)

#define _HELP_STR                                                              \
    "Benchmark FFT using Intel(R) MKL DFTI.\n\n"                               \
    "FFT problem arguments:\n"                                                 \
    "  -t, --threads=THREADS    use THREADS threads for FFT execution.\n"      \
    "                           Specifying zero threads will instruct the \n"  \
    "                           benchmark to use MKL's default number of \n"   \
    "                           threads. (default: use MKL's default)\n"       \
    "  -d, --dtype=DTYPE        use DTYPE as the FFT domain. For a list of\n"  \
    "                           understood dtypes, use '-d help'.\n"           \
    "                           (default: %s)\n"                               \
    "  -r, --rfft               do not copy superfluous harmonics when FFT\n"  \
    "                           output is conjugate-even, i.e. for real "      \
    "inputs\n"                                                                 \
    "  -P, --in-place           allow overwriting the input buffer with the\n" \
    "                           FFT outputs\n"                                 \
    "  -c, --cached             use the same DFTI descriptor for the same\n"   \
    "                           outer loop, i.e. \"cache\" the descriptor\n"   \
    "\n"                                                                       \
    "Timing arguments:\n"                                                      \
    "  -i, --inner-loops=IL     time the benchmark IL times for each "         \
    "printed\n"                                                                \
    "                           measurement. Copies are not included in the\n" \
    "                           measurements. (default: %d)\n"                 \
    "  -o, --outer-loops=OL     print OL measurements. (default: %d)\n"        \
    "\n"                                                                       \
    "Output arguments:\n"                                                      \
    "  -p, --prefix=PREFIX      output PREFIX as the first value in outputs\n" \
    "                           (default: '%s')\n"                             \
    "  -H, --no-header          do not output CSV header. This can be "        \
    "useful\n"                                                                 \
    "                           if running multiple benchmarks "               \
    "back-to-back.\n"                                                          \
    "  -h, --help               print this message and exit\n"                 \
    "\n"                                                                       \
    "The size argument specifies the input matrix size as a tuple of "         \
    "positive\n"                                                               \
    "decimal integers, delimited by any non-digit. For example, both\n"        \
    "(101, 203, 305) and 101x203x305 denote the same 3D FFT.\n"

#ifdef __GNUC__
#define HELP_STR _HELP_STR
#else
#define HELP_STR _HELP_STR "\n**Long options are not supported on Windows!**\n"
#endif

struct dtype {
    /* float or double? */
    enum DFTI_CONFIG_VALUE precision;

    /* real or complex? */
    enum DFTI_CONFIG_VALUE domain;

    /* size in bytes */
    size_t size;

    /* names */
    const char *const *names;
};

static const char *NAMES_FLOAT32[] = {"float32", "float", "f4", 0};
static const char *NAMES_FLOAT64[] = {"float64", "double", "f8", 0};
static const char *NAMES_COMPLEX64[] = {"complex64", "complex float", "c8", 0};
static const char *NAMES_COMPLEX128[] = {"complex128", "complex double", "c16",
                                         0};

static const struct dtype VALID_DTYPES[] = {
    {DFTI_SINGLE, DFTI_REAL, sizeof(float), NAMES_FLOAT32},
    {DFTI_DOUBLE, DFTI_REAL, sizeof(double), NAMES_FLOAT64},
    {DFTI_SINGLE, DFTI_COMPLEX, sizeof(MKL_Complex8), NAMES_COMPLEX64},
    {DFTI_DOUBLE, DFTI_COMPLEX, sizeof(MKL_Complex16), NAMES_COMPLEX128}};

static const struct dtype *parse_dtype(const char *name) {
    size_t i, j;
    const struct dtype *dtype;
    for (i = 0; i < sizeof(VALID_DTYPES) / sizeof(*VALID_DTYPES); i++) {
        dtype = &VALID_DTYPES[i];
        for (j = 0; dtype->names[j] != NULL; j++) {
            if (strcmp(dtype->names[j], name) == 0) {
                return dtype;
            }
        }
    }
    return NULL;
}

static inline void warm_up_threads() {
    int i;
    unsigned int *x = malloc(mkl_get_max_threads() * sizeof(int));

#pragma omp parallel for
    for (i = 0; i < mkl_get_max_threads(); i++) {
        x[i] = rand();
    }

    free(x);
}

/*
 * Parse size string, in form of e.g. 1001x2003x1005 -> [1001 2003 1005].
 * All non-numeric characters are used as separators.
 *
 * strsize. String to parse as null-terminated char array.
 * buf. Pointer to a pointer which will contain the size array on exit.
 *
 * Returns the number of dimensions actually parsed.
 */
static size_t parse_shape(const char *strsize, MKL_LONG **buf) {

    char *endptr;
    static const size_t initial_size = 8;
    size_t i, size;

    if (strsize == NULL)
        return 0;

    /* Seek to the first digit */
    for (; !isdigit(*strsize) && *strsize != '\0'; strsize++)
        ;

    /* No digits found? */
    if (*strsize == '\0')
        return 0;

    *buf = (MKL_LONG *) mkl_malloc(initial_size * sizeof(**buf), 64);
    size = initial_size;

    for (i = 0; *strsize != '\0'; i++) {
        if (i >= size) {
            size *= 2;
            *buf = realloc(*buf, size * sizeof(**buf));
        }
        (*buf)[i] = strtoul(strsize, &endptr, 10);
        if (strsize == endptr || *endptr == '\0')
            break;
        strsize = endptr;
        for (; !isdigit(*strsize) && *strsize != '\0'; strsize++)
            ;
        if (*strsize == '\0')
            break;
    }

    return i + 1;
}

static char *shape_to_str(size_t ndims, const MKL_LONG *shape) {
    char *buf;
    size_t nbytes = 0, pos = 0, i = 0;

    if (shape == NULL)
        return NULL;

    for (i = 0; i < ndims; i++) {
        nbytes += snprintf(NULL, 0, "%ldx", shape[i]);
    }

    buf = (char *) mkl_malloc(nbytes, 64);
    for (i = 0; i < ndims; i++) {
        pos += snprintf(buf + pos, nbytes - pos, "%ld", shape[i]);
        if (i < ndims - 1)
            buf[pos++] = 'x';
    }

    return buf;
}

static MKL_LONG shape_prod(size_t ndims, const MKL_LONG *shape) {
    size_t i;
    MKL_LONG prod = 1;
    for (i = 0; i < ndims; i++)
        prod *= shape[i];
    return prod;
}

static MKL_LONG *shape_strides(size_t ndims, const MKL_LONG *shape) {
    size_t i, j;
    MKL_LONG *strides = mkl_malloc((ndims + 1) * sizeof(*strides), 64);
    strides[0] = 0;
    for (i = 1; i <= ndims; i++) {
        strides[i] = 1;
        for (j = i; j < ndims; j++) {
            strides[i] *= shape[j];
        }
    }
    return strides;
}

static void print_array(const struct dtype *dtype, MKL_LONG n, void *x) {
    int i;
    if (dtype->precision == DFTI_SINGLE) {
        float *f = (float *) x;
        if (dtype->domain == DFTI_REAL) {
            for (i = 0; i < n; i++) {
                printf("x[%d] = %.8g\n", i, f[i]);
            }
        } else {
            for (i = 0; i < n * 2; i += 2) {
                printf("x[%d] = %.8g + %.8gj\n", i, f[i], f[i + 1]);
            }
        }
    } else {
        double *d = (double *) x;
        if (dtype->domain == DFTI_REAL) {
            for (i = 0; i < n; i++) {
                printf("x[%d] = %.8g\n", i, d[i]);
            }
        } else {
            for (i = 0; i < n * 2; i += 2) {
                printf("x[%d] = %.8g + %.8gj\n", i, d[i], d[i + 1]);
            }
        }
    }
}

static void *randn(const struct dtype *dtype, MKL_LONG n, MKL_INT brng,
                   MKL_UINT seed) {
    MKL_LONG err = 0;
    VSLStreamStatePtr stream;

    errno = 0;
    void *x = (void *) mkl_malloc(n * dtype->size, 64);
    if (x == NULL)
        error(1, errno, "failed to allocate %lu bytes for x", n * dtype->size);
    assert(x);

    err = vslNewStream(&stream, brng, seed);
    if (err != VSL_STATUS_OK)
        error(1, 0, "vslNewStream failed: %ld", err);
    assert(err == VSL_STATUS_OK);

    /* Generate twice as many values for complex arrays */
    if (dtype->domain == DFTI_COMPLEX)
        n *= 2;

    if (dtype->precision == DFTI_SINGLE) {
        err = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, x, 0, 1);
    } else {
        err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, x, 0, 1);
    }
    if (err != VSL_STATUS_OK)
        error(1, 0, "v*RngGaussian failed: %ld", err);
    assert(err == VSL_STATUS_OK);

    err = vslDeleteStream(&stream);
    if (err != VSL_STATUS_OK)
        error(1, 0, "vslDeleteStream failed: %ld", err);
    assert(err == VSL_STATUS_OK);

    return x;
}

static MKL_LONG fft_create_descriptor(DFTI_DESCRIPTOR_HANDLE *hand,
                                      MKL_LONG ndims, MKL_LONG *shape,
                                      MKL_LONG *strides,
                                      const struct dtype *dtype,
                                      double forward_scale,
                                      double backward_scale, bool inplace) {

    MKL_LONG status;
    if (ndims == 1) {
        status = DftiCreateDescriptor(hand, dtype->precision, dtype->domain,
                                      ndims, shape[0]);
    } else {
        status = DftiCreateDescriptor(hand, dtype->precision, dtype->domain,
                                      ndims, shape);
    }
    CHECK_DFTI_STATUS(status, "could not create DFTI descriptor");

    if (dtype->domain == DFTI_REAL) {
        status = DftiSetValue(*hand, DFTI_CONJUGATE_EVEN_STORAGE,
                              DFTI_COMPLEX_COMPLEX);
        CHECK_DFTI_STATUS(status, "could not set DFTI_CONJUGATE_EVEN_STORAGE");
    }

    if (!inplace) {
        status = DftiSetValue(*hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        CHECK_DFTI_STATUS(status, "could not set DFTI_PLACEMENT");
    }

    status = DftiSetValue(*hand, DFTI_INPUT_STRIDES, strides);
    CHECK_DFTI_STATUS(status, "could not set DFTI_INPUT_STRIDES to %s",
                      shape_to_str(ndims + 1, strides));

    status = DftiSetValue(*hand, DFTI_FORWARD_SCALE, forward_scale);
    CHECK_DFTI_STATUS(status, "could not set DFTI_FORWARD_SCALE to %f",
                      forward_scale);

    status = DftiSetValue(*hand, DFTI_BACKWARD_SCALE, backward_scale);
    CHECK_DFTI_STATUS(status, "could not set DFTI_BACKWARD_SCALE to %f",
                      backward_scale);

    status = DftiCommitDescriptor(*hand);
    CHECK_DFTI_STATUS(status, "could not commit DFTI descriptor");

    return 0;
}

static void copy_superfluous_harmonics(MKL_LONG ndims, MKL_LONG *shape,
                                       MKL_LONG n, const struct dtype *dtype,
                                       void *buf) {

    MKL_LONG i, j;

    /* TODO: remove this error message once copy_superfluous_harmonics
     * supports multiple dimensions */
    if (ndims != 1) {
        error(1, 0,
              "copy_superfluous_harmonics is unimplemented for "
              "ndims > 1. Try using --rfft option?");
    }

    if (dtype->precision == DFTI_SINGLE) {
        MKL_Complex8 *sbuf = (MKL_Complex8 *) buf;

#pragma omp parallel for simd
        for (i = n / 2 + 1; i < n; i++) {
            j = (n - i) % n;
            sbuf[i].real = sbuf[j].real;
            sbuf[i].imag = -sbuf[j].imag;
        }
    } else {
        MKL_Complex16 *dbuf = (MKL_Complex16 *) buf;

#pragma omp parallel for simd
        for (i = n / 2 + 1; i < n; i++) {
            j = (n - i) % n;
            dbuf[i].real = dbuf[j].real;
            dbuf[i].imag = -dbuf[j].imag;
        }
    }
}

int main(int argc, char *argv[]) {

    size_t i;
    bool header = true, verbose = false, inplace = false, cached = false;
    bool rfft = false;
    MKL_LONG inner_loops = 16, outer_loops = 5;
    size_t goal_outer_loops = 10;
    double time_limit = 10.;
    size_t threads = 0;
    MKL_LONG n = 0, *strides = NULL;

    const char *prefix = "Native-C", *strdtype = "complex128";
    const char *problem = NULL;
    char *strsize = NULL;
    static const char *problems[] = {0, "fft", "fft2", "fftn"};

#ifdef __GNUC__
    static struct option longopts[] = {
        {"inner-loops", required_argument, NULL, 'i'},
        {"outer-loops", required_argument, NULL, 'o'},
        {"goal-outer-loops", required_argument, NULL, 'g'},
        {"time-limit", required_argument, NULL, 'l'},
        {"threads", required_argument, NULL, 't'},
        {"prefix", required_argument, NULL, 'p'},
        {"dtype", required_argument, NULL, 'd'},
        {"verbose", no_argument, NULL, 'v'},
        {"no-header", no_argument, NULL, 'H'},
        {"in-place", no_argument, NULL, 'P'},
        {"cached", no_argument, NULL, 'c'},
        {"rfft", no_argument, NULL, 'r'},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0}};
#endif

    int intarg, opt, optindex = 0;
    char *endptr;
    double darg;
    while ((opt = getopt_long(argc, argv, "p:d:t:i:o:vHPchr", longopts,
                              &optindex)) != -1) {

        /* first pass: parse numeric values and assign other values */
        switch (opt) {
        case 'i':
        case 'o':
        case 'g':
        case 't':
            intarg = strtoul(optarg, &endptr, 0);
            if (*endptr != '\0' || intarg < 0) {
                error(1, 0, "must be positive integer: %s\n", optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'l':
            errno = 0;
            darg = strtod(optarg, &endptr);
            if (errno) {
                perror("fft_bench");
                return EXIT_FAILURE;
            }
            if (*endptr != '\0' || darg < 0. || darg >= INFINITY) {
                error(1, 0, "must be finite, non-negative double: %s\n",
                      optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'p':
            prefix = optarg;
            break;
        case 'd':
            strdtype = optarg;
            break;
        case 'v':
            verbose = true;
            break;
        case 'H':
            header = false;
            break;
        case 'h':
            printf("usage: %s [args] size\n", argv[0]);
            printf(HELP_STR, "complex128", 16, 5, "Native-C");
            return EXIT_SUCCESS;
        case 'P':
            inplace = true;
            break;
        case 'c':
            cached = true;
            break;
        case 'r':
            rfft = true;
            break;
        case '?':
        default:
            return EXIT_FAILURE;
        }

        /* second pass: assign parsed numeric values */
        switch (opt) {
        case 'i':
            inner_loops = intarg;
            break;
        case 'o':
            outer_loops = intarg;
            break;
        case 'g':
            goal_outer_loops = intarg;
            break;
        case 'l':
            time_limit = darg;
            break;
        case 't':
            threads = intarg;
        default:
            break;
        }
    }

    /* Parse and validate dtype */
    const struct dtype *dtype = parse_dtype(strdtype);
    if (dtype == NULL) {
        fprintf(stderr, "%s: dtype '%s' is unknown. Try one of", argv[0],
                strdtype);
        for (i = 0; i < sizeof(VALID_DTYPES) / sizeof(*VALID_DTYPES); i++) {
            fprintf(stderr, " '%s'", VALID_DTYPES[i].names[0]);
        }
        fprintf(stderr, ".\n");
        return EXIT_FAILURE;
    }
    strdtype = dtype->names[0];

    /* Check if a size was passed at all */
    if (optind >= argc) {
        error(1, 0, "no FFT size specified");
        return EXIT_FAILURE;
    }

    if (optind + 1 < argc) {
        error(1, 0, "multiple FFT sizes specified");
        return EXIT_FAILURE;
    }

    strsize = argv[optind];

    /* Set and warm up threads */
    if (threads > 0) {
        mkl_set_num_threads(threads);
        omp_set_num_threads(threads);
    }
    /* TODO */
#ifdef __GNUC__
    if (threads == 1) {
        mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    }
#endif

    threads = mkl_get_max_threads();
    warm_up_threads();

    /* Parse size */
    size_t ndims;
    MKL_LONG *shape;
    ndims = parse_shape(strsize, &shape);

    /* Validate size */
    if (ndims < 1)
        error(1, 0, "number of FFT dimensions must be positive");
    strsize = shape_to_str(ndims, shape);
    for (i = 0; i < ndims; i++) {
        if (shape[i] < 1) {
            error(1, 0, "given shape %s is invalid: shape[%lu] = %ld < 1\n",
                  strsize, i, shape[i]);
        }
    }

    if (rfft && dtype->domain != DFTI_REAL) {
        error(1, 0,
              "--rfft makes no sense for an FFT of complex inputs. "
              "The FFT output will not be conjugate even, so the "
              "whole output matrix must be computed!");
    }

    if (!rfft && dtype->domain == DFTI_REAL && ndims > 1) {
        error(1, 0,
              "Copying extra harmonics in the conjugate-even output of "
              "FFT of real inputs of dimension greater than 1 is "
              "currently unsupported. Try using --rfft option?");
    }

    /* Get total size and strides */
    n = shape_prod(ndims, shape);
    assert(n > 0);
    strides = shape_strides(ndims, shape);

    /* Printable "in-place" and "cached" */
    const char *strplace, *strcache;
    strplace = (inplace) ? "in-place" : "out-of-place";
    strcache = (cached) ? "cached" : "not cached";
    if (rfft) {
        if (ndims == 1)
            problem = "rfft";
        else if (ndims == 2)
            problem = "rfft2";
        else
            problem = "rfftn";
    } else {
        if (ndims == 1)
            problem = "fft";
        else if (ndims == 2)
            problem = "fft2";
        else
            problem = "fftn";
    }

    /* Input/output matrices */
    void *x = 0, *buf = 0;

    /* Execution status */
    MKL_LONG status = 0;
    DFTI_DESCRIPTOR_HANDLE hand = 0;

    moment_t t0, t1;
    moment_t time_tot = 0;
    int it, si;

    double *times = (double *) mkl_malloc(outer_loops * sizeof(*times), 64);

    /* Generate input matrix */
    x = randn(dtype, n, VSL_BRNG_MT19937, SEED);

    /* Real input still has complex output */
    if (dtype->domain == DFTI_COMPLEX) {
        buf = mkl_malloc(n * dtype->size, 64);
    } else {
        buf = mkl_malloc(n * 2 * dtype->size, 64);
    }

    const char *strheader = "prefix,function,threads,dtype,size,"
                            "place,cached,time";
    if (header)
        puts(strheader);

    /* Execute benchmark */
    for (si = 0; si < outer_loops; si++) {

        time_tot = 0;
        if (cached) {
            t0 = moment_now();
            status = fft_create_descriptor(&hand, ndims, shape, strides, dtype,
                                           1., 1. / n, inplace);
            assert(status == 0);
            t1 = moment_now();
            time_tot += t1 - t0;
        }

        for (it = -1; it < inner_loops; it++) {
            if (inplace) {
                /* TODO: is memcpy better than MKL BLAS *copy? */
                memcpy(buf, x, n * dtype->size);
            }

            t0 = moment_now();
            if (!cached) {
                status = fft_create_descriptor(&hand, ndims, shape, strides,
                                               dtype, 1., 1. / n, inplace);
                assert(status == 0);
            }
            /* TODO: might have to cast to (float *) or (double *) here? */
            if (inplace) {
                status = DftiComputeForward(hand, buf);
            } else {
                status = DftiComputeForward(hand, x, buf);
            }
            CHECK_DFTI_STATUS(status, "could not compute FFT");

            /* for real FFTs, without --rfft option, copy
             * superfluous harmonics */
            if (dtype->domain == DFTI_REAL && !rfft) {
                /* TODO: remove assertion once copy_superfluous_harmonics
                 * supports more than one dimension */
                assert(ndims != 1);
                copy_superfluous_harmonics(ndims, shape, n, dtype, buf);
            }

            if (!cached) {
                status = DftiFreeDescriptor(&hand);
                CHECK_DFTI_STATUS(status, "could not free DFTI descriptor");
            }
            t1 = moment_now();

            if (it >= 0)
                time_tot += t1 - t0;
        }

        t0 = moment_now();
        if (cached) {
            status = DftiFreeDescriptor(&hand);
            CHECK_DFTI_STATUS(status, "could not free DFTI descriptor");
        }
        t1 = moment_now();
        time_tot += t1 - t0;

        times[si] = seconds_from_moment(time_tot / inner_loops);
        printf("%s,%s,%lu,%s,%s,%s,%s,%.5g\n", prefix, problem, threads,
               dtype->names[0], strsize, strplace, strcache, times[si]);
    }

    if (verbose && buf && n <= 10) {
        print_array(dtype, n, buf);
    }

    mkl_free(buf);
    mkl_free(x);

    mkl_free(times);
    mkl_free(shape);
    mkl_free(strsize);
}
