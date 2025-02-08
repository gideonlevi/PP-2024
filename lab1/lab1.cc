#include <assert.h>
#include "mpi.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long r_squared = r * r;
    unsigned long long local_sum = 0, global_sum = 0;
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    unsigned long long start = rank;
    unsigned long long block_size = 1000;

    for (unsigned long long i = start; i <= r; i += size) {
        unsigned long long y = ceil(sqrtl(r_squared - i * i));
        local_sum += y;

        if (i % block_size == 0) {
            local_sum %= k;
        }
    }
    local_sum %= k;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        unsigned long long result = (4 * global_sum) % k;
        printf("%llu\n", result);
    }

    MPI_Finalize();
    return 0;
}
