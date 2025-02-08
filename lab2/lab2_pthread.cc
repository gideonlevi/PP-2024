#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

typedef struct {
    unsigned long long r_squared;
    unsigned long long start;
    unsigned long long end;
} compute_args;

void *partial_compute(void *args) {
    compute_args *arguments = (compute_args *) args;
    unsigned long long r_squared = arguments->r_squared;
    unsigned long long start = arguments->start;
    unsigned long long end = arguments->end;

    unsigned long long *result = (unsigned long long*)malloc(sizeof(unsigned long long));
    *result = 0;

    for (unsigned long long x = start; x < end; x++) {
        *result += ceil(sqrtl(r_squared - x*x));
    }

    pthread_exit((void *) result);
}

int main(int argc, char** argv) {
    if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

    // int ncpus = atoi(getenv("SLURM_CPUS_PER_TASK"));
    cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);
    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long pixels = 0;
    unsigned long long r_squared = r*r;


    pthread_t threads[ncpus];
    compute_args args[ncpus];
    int rc;

    unsigned long long chunk_size = r / ncpus;
    unsigned long long remainder = r % ncpus;

    // distribute work among threads
    for (unsigned long long t = 0; t < ncpus; t++) {
        args[t].r_squared = r_squared;
        args[t].start = t * chunk_size + std::min(t, remainder);
        args[t].end = (t < remainder) ? args[t].start + chunk_size + 1 : args[t].start + chunk_size;

        pthread_create(&threads[t], NULL, partial_compute, (void *)&args[t]);
    }

    unsigned long long *return_value;
    for (int t = 0; t < ncpus; t++) {
        pthread_join(threads[t], (void **)&return_value);
        pixels += *return_value;
        pixels %= k;
        free(return_value); // free the allocated memory
    }

    printf("%llu\n", (4 * pixels) % k);
}
