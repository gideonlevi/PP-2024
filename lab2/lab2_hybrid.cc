#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_ranks;
    unsigned long long r, k;
    
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        MPI_Finalize();
        return 1;
    }

    r = atoll(argv[1]);
    k = atoll(argv[2]);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);

	// divide work among processes
    unsigned long long local_pixels = 0;
    unsigned long long chunk_size = r / mpi_ranks;
	unsigned long long remainder = r % mpi_ranks;
    unsigned long long start = mpi_rank * chunk_size + std::min(mpi_rank, (int)remainder);
    unsigned long long end = (mpi_rank < remainder) ? start + chunk_size + 1 : start + chunk_size;
	unsigned long long r_squared = r*r;

    // parallelize using OpenMP
#pragma omp parallel
    {
        unsigned long long omp_pixels = 0;

        #pragma omp for
        for (unsigned long long x = start; x < end; x++) {
            unsigned long long y = ceil(sqrtl(r_squared - x * x));
            omp_pixels += y;
			omp_pixels %= k;
        }

		// atomic to ensure thread safety
        #pragma omp atomic
        local_pixels += omp_pixels;
    }

    // reduce results from all processes
    unsigned long long total_pixels = 0;
    MPI_Reduce(&local_pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // root process prints the result
    if (mpi_rank == 0) {
        printf("%llu\n", (4 * (total_pixels % k)) % k);
    }

    MPI_Finalize();
    return 0;
}
