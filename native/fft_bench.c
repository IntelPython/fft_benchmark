#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fft_bench.h"
#include "moments.h"
#include "omp.h"

#include "fft2_cdp-in-c.h"
#include "fft2_cdp-in-nc.h"
#include "fft2_cdp-out-c.h"
#include "fft2_cdp-out-nc.h"

#define SEED 7777

static const struct bench benchmarks[] = {
    {fft2_cdp_in_c, 2, "fft2", "complex128", true, true},
    {fft2_cdp_in_nc, 2, "fft2", "complex128", true, false},
    {fft2_cdp_out_c, 2, "fft2", "complex128", false, true},
    {fft2_cdp_out_nc, 2, "fft2", "complex128", false, false},
    {0, 0, 0, 0, 0, 0}
};

struct bench_options *alloc_bench_options(size_t ndims) {
    struct bench_options *opts = (struct bench_options *)
        malloc(sizeof(struct bench_options) + ndims * sizeof(size_t));
    return opts;
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
bench_result_t time_mean_min(struct bench bench, const struct bench_options *opts,
                             bool verbose) {
    
    bench_result_t result;
    moment_t t0, t1;
    moment_t min_time = LLONG_MAX;
    void *buffers = bench.make_args(opts);
    warm_up_threads();

    for (int i = 0; i < opts->outer_loops; i++) {

        moment_t total_time = 0;
        if (bench.pre_inner_loop != NULL) {
            t0 = moment_now();
            bench.pre_inner_loop(opts, buffers);
            t1 = moment_now();
            if (bench.time_pre_post_inner_loop) {
                total_time += t1 - t0;
            }
        }

        for (int j = 0; j < opts->inner_loops; i++) {

            if (bench.copy_args != NULL) {
                bench.copy_args(opts, buffers);
            }

            t0 = moment_now();
            bench.compute(opts, buffers);
            t1 = moment_now();
            total_time += t1 - t0;

        }
        
        if (bench.post_inner_loop != NULL) {
            t0 = moment_now();
            bench.post_inner_loop(opts, buffers);
            t1 = moment_now();
            if (bench.time_pre_post_inner_loop) {
                total_time += t1 - t0;
            }
        }

        if (verbose) {
            printf("@ times[%d] = %.5g\n", i, seconds_from_moment(total_time));
        }

        if (total_time < min_time) {
            min_time = total_time;
        }

    }

    bench.free_args(opts, buffers);

    result.min_time = min_time;
    return result;
}
*/

/*
 * Parse size string, in form of e.g. 1001x2003x1005 -> [1001 2003 1005].
 * All non-numeric characters are used as separators.
 *
 * strsize. String to parse
 * ndims. Number of dimensions expected and size of buf
 * buf. Where the parsed size should go
 * 
 * Returns the number of dimensions actually parsed.
 */
size_t parse_size(const char *strsize, size_t ndims, MKL_LONG *buf) {

    int i;
    char *endptr;

    if (strsize == NULL) {
        return 0;
    }

    for (i = 0; i < ndims; i++) {
        buf[i] = strtoul(strsize, &endptr, 10);
        if (strsize == endptr || *endptr == '\0') break;
        strsize = endptr;
        while (*(++strsize) != '\0') {
            if (*strsize >= '0' && *strsize <= '9') break;
        }
    }

    /* At this point, if the pointer is on a digit, the user specified too
     * many dimensions in the size string. */
    if (*strsize >= '0' && *strsize <= '9') return i + 1;
    return i;

}

void zprint(MKL_LONG n, MKL_Complex16 *x) {
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %.8g + %.8gj\n", i, x[i].real, x[i].imag);
    }
}

void dprint(MKL_LONG n, double *x) {
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %.8g\n", i, x[i]);
    }
}

MKL_Complex16 *zrandn(MKL_LONG n, MKL_INT brng, MKL_UINT seed) {
    MKL_Complex16 *x = (MKL_Complex16 *) drandn(n * 2, brng, seed);
    assert(x);
    return x;
}

double *drandn(MKL_LONG n, MKL_INT brng, MKL_UINT seed) {
    MKL_LONG err = 0;
    VSLStreamStatePtr stream;

    double *x = (double *) mkl_malloc(n * sizeof(*x), 64);
    assert(x);

    err = vslNewStream(&stream, brng, seed);
    assert(err == VSL_STATUS_OK);

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, x, 0., 1.);
    assert(err == VSL_STATUS_OK);

    err = vslDeleteStream(&stream);
    assert(err == VSL_STATUS_OK);

    return x;
}


int main(int argc, char *argv[]) {

    bool header = true;
    bool verbose = false;
    bool in_place = false;
    bool cached = false;
    size_t inner_loops = 16;
    size_t outer_loops = 5;
    size_t goal_outer_loops = 10;
    double time_limit = 10.;
    size_t threads = -1;
    
    const char *prefix = "Native-C";
    const char *dtype = NULL;
    const char *strsize = NULL;

    static struct option longopts[] = {
        {"size", required_argument, NULL, 'n'},
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
        {0, 0, 0, 0}
    };

    int intarg, opt, optindex = 0;
    char *endptr;
    double darg;
    while ((opt = getopt_long(argc, argv, "n:p:d:t:vPc",
                              longopts, &optindex)) != -1) {

        /* first pass: parse numeric values and assign other values */
        switch (opt) {
            case 'n':
                strsize = optarg;
                break;
            case 'i':
            case 'o':
            case 'g':
            case 't':
                intarg = strtoul(optarg, &endptr, 0);
                if (*endptr != '\0' || intarg < 0) {
                    fprintf(stderr, "error: must be positive integer: %s\n",
                            optarg);
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
                    fprintf(stderr, "error: must be finite, non-negative "
                                    "double: %s\n", optarg);
                    return EXIT_FAILURE;
                }
                break;
            case 'p':
                prefix = optarg;
                break;
            case 'd':
                dtype = optarg;
                break;
            case 'v':
                verbose = true;
                break;
            case 'H':
                header = false;
                break;
            case 'P':
                in_place = true;
                break;
            case 'c':
                cached = true;
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
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "error: no benchmark specified!\n");
        return EXIT_FAILURE;
    } 

    if (optind + 1 < argc) {
        fprintf(stderr, "error: more than one benchmark specified!\n");
        return EXIT_FAILURE;
    }

    const char *strheader = "prefix,function,threads,dtype,size,"
                         "place,cached,time";
    if (header) puts(strheader);


    const char *problem = argv[optind];

#ifdef DEBUG
    if (verbose) {
        printf("# requesting %s, cached=%d, in_place=%d, dtype=%s\n",
               problem, cached, in_place, dtype);
    }
#endif

    /* Set and warm up threads */
    if (threads > 0) {
        mkl_set_num_threads(threads);
    }
    if (threads == 1) {
        mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    }

    threads = mkl_get_max_threads();
    warm_up_threads();

    const struct bench *curr;
    const char *strplace, *strcache;
    struct bench_options *opts;
    for (curr = benchmarks; curr->name != NULL; curr++) {

#ifdef DEBUG
        if (verbose) {
            printf("# trying %s, cached=%d, in_place=%d, dtype=%s\n",
                   curr->name, curr->cached, curr->in_place, curr->dtype);
        }
#endif

        if (strcmp(curr->name, problem) != 0) continue;
        if (dtype != NULL && strcmp(curr->dtype, dtype) != 0) continue;
        if (curr->in_place != in_place) continue;
        if (curr->cached != cached) continue;

        opts = alloc_bench_options(curr->ndims);
        size_t ndims = parse_size(strsize, curr->ndims, opts->shape);
        if (ndims != curr->ndims) {
            if (ndims < curr->ndims) {
                fprintf(stderr, "error: expected %lu dimensions for problem "
                                "size, but got %lu\n", curr->ndims, ndims);
            } else {
                fprintf(stderr, "error: expected only %lu dimensions for "
                                "problem size\n", curr->ndims);
            }
            free(opts);
            return EXIT_FAILURE;
        }

        if (verbose) {
            strplace = (in_place) ? "in-place" : "out-of-place";
            strcache = (cached) ? "cached" : "not cached";
            fprintf(stderr, "# executing: %s, %s, %s, %s, with "
                    "inner_loops=%lu, outer_loops=%lu\n", curr->name,
                    curr->dtype, strplace, strcache, inner_loops, outer_loops);
        }

        /* Set options */
        opts->inner_loops = inner_loops;
        opts->outer_loops = outer_loops;
        opts->brng = VSL_BRNG_MT19937;
        opts->seed = SEED;
        opts->verbose = verbose;

        /* Execute benchmark */
        double *times = curr->func(opts);

        /* Print results */
        /* TODO take min. */
        for (int i = 0; i < outer_loops; i++)
            printf("%s,%s,%lu,%s,%s,%s,%s,%.5g\n", prefix, curr->name, threads,
                   curr->dtype, strsize, strplace, strcache, times[i]);

        free(times);
        free(opts);
    }
}
