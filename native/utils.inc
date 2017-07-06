#include <stdio.h>
#include <stdlib.h>

static inline void warm_up_threads(void) {
    int i;
    #pragma omp parallel for
    for (i=0; i < omp_get_num_threads(); i++) {
       printf(".");
    }
    printf("\n");

    return;
}
