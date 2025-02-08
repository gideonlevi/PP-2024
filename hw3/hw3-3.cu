#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>
#include <iostream>
#include <chrono>

//======================
#define DEV_NO 0
cudaDeviceProp prop;

void input(char* infile);
void output(char *outFileName);
void block_FW();
int ceil(int a, int b);
__global__ void phase1(int *dDist, int Round, int n);
__global__ void phase2_col(int *dDist, int Round, int n);
__global__ void phase2_row(int *dDist, int Round, int n);
__global__ void phase3(int *dDist, int Round, int n, int yoffset);

const int INF = ((1 << 30) - 1);
#define B 64
#define half_B B / 2

int n, m, ceil_n, gpu_count;
int* hDist = NULL;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    printf("n: %d, m: %d\n", n, m);

    ceil_n = ceil(n, B) * B;
    hDist = (int*) malloc(sizeof(int) * ceil_n * ceil_n);

    for (int i = 0; i < ceil_n; ++ i) {
        int IN = i * ceil_n;
        #pragma GCC ivdep
        for (int j = 0; j < i; ++j) {
            hDist[IN + j] = INF;
        }
        #pragma GCC ivdep
        for (int j = i + 1; j < ceil_n; ++j) {
            hDist[IN + j] = INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) { 
        fread(pair, sizeof(int), 3, file); 
        hDist[pair[0] * ceil_n + pair[1]] = pair[2]; 
    } 
    fclose(file);
}

void output(char *outFileName) {
    FILE *outfile = fopen(outFileName, "w");

    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         if (hDist[i][j] >= INF) hDist[i][j] = INF;
    //     }
    //     fwrite(hDist[i], sizeof(int), n, outfile);
    // }
    for (int block_i = 0; block_i < n; ++block_i) {
        fwrite(&hDist[block_i * ceil_n], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

int main(int /*argc*/, char* argv[]) {
	input(argv[1]);
    block_FW();
	output(argv[2]);
    cudaFreeHost(hDist);
    return 0;
}

void block_FW() {
    const int gpu_count = 2;
    int* dDist[gpu_count];
    int rounds = ceil(ceil_n, B);

    cudaHostRegister(hDist, ceil_n * ceil_n * sizeof(int), cudaHostRegisterDefault);

    // Enable P2P communication between the two GPUs
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);

    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

    #pragma omp parallel num_threads(gpu_count)
    {
        const int tid = omp_get_thread_num();
        cudaSetDevice(tid);

        int rounds_per_thread = rounds / 2;
        const int start_round = tid * rounds_per_thread;
        if (tid == 1 && (rounds % 2) == 1) rounds_per_thread += 1;
        const int end_round = start_round + rounds_per_thread;
        const int pivot_row_size = B * ceil_n;
        const size_t pivot_row_size_int = pivot_row_size * sizeof(int);
        const size_t half_matrix_size_int = pivot_row_size_int * rounds_per_thread;
        const size_t start_pivot_row_offset = start_round * pivot_row_size;

        dim3 block_dim(32, 32);
        dim3 grid_dim(rounds, rounds_per_thread);

        cudaMalloc(&dDist[tid], ceil_n * ceil_n * sizeof(int));
        cudaMemcpy(dDist[tid] + start_pivot_row_offset, hDist + start_pivot_row_offset, half_matrix_size_int, cudaMemcpyHostToDevice);

        for (int r = 0; r < rounds; r++) {
            const size_t round_start_block_offset = r * pivot_row_size;

            // Phase 1 block on GPU responsible for current round
            if ((r >= start_round) && (r < end_round)) {
                // Share phase 1 result directly between GPUs
                cudaMemcpyPeer(dDist[1 - tid] + round_start_block_offset, 
                               1 - tid, 
                               dDist[tid] + round_start_block_offset, 
                               tid, 
                               pivot_row_size_int);
            }
            #pragma omp barrier

            phase1<<<1, block_dim>>>(dDist[tid], r, ceil_n);
            phase2_col<<<rounds, block_dim>>>(dDist[tid], r, ceil_n);
            phase2_row<<<rounds, block_dim>>>(dDist[tid], r, ceil_n);
            phase3<<<grid_dim, block_dim>>>(dDist[tid], r, ceil_n, start_round);
            #pragma omp barrier
        }

        cudaMemcpy(hDist + start_pivot_row_offset,
                   dDist[tid] + start_pivot_row_offset,
                   half_matrix_size_int,
                   cudaMemcpyDeviceToHost);
        cudaFree(dDist[tid]);
    }

    // Disable P2P after computation
    cudaSetDevice(0);
    cudaDeviceDisablePeerAccess(1);

    cudaSetDevice(1);
    cudaDeviceDisablePeerAccess(0);
}


__global__ void phase1(int *dDist, int round, int n) {
    __shared__ int shared_mem[B][B];
    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int offset = round * B;

    shared_mem[i][j] = dDist[(n + 1) * offset + i * n  + j];
    shared_mem[i][j + half_B] = dDist[(n + 1) * offset + i * n + j + half_B];
    shared_mem[i + half_B][j] = dDist[(n + 1) * offset + (i + half_B) * n + j];
    shared_mem[i + half_B][j + half_B] = dDist[(n + 1) * offset + (i + half_B) * n + j + half_B];
    __syncthreads();

    #pragma unroll 4
    for (int k = 0; k < B; ++k) {
        shared_mem[i][j] = min(shared_mem[i][j], shared_mem[i][k] + shared_mem[k][j]);
        shared_mem[i][j + half_B] = min(shared_mem[i][j + half_B], shared_mem[i][k] + shared_mem[k][j + half_B]);
        shared_mem[i + half_B][j] = min(shared_mem[i + half_B][j], shared_mem[i + half_B][k] + shared_mem[k][j]);
        shared_mem[i + half_B][j + half_B] = min(shared_mem[i + half_B][j + half_B], shared_mem[i + half_B][k] + shared_mem[k][j + half_B]);
        __syncthreads();
    }

    dDist[(n + 1) * offset + i * n  + j] = shared_mem[i][j];
    dDist[(n + 1) * offset + i * n + j + half_B] = shared_mem[i][j + half_B];
    dDist[(n + 1) * offset + (i + half_B) * n + j] = shared_mem[i + half_B][j];
    dDist[(n + 1) * offset + (i + half_B) * n + j + half_B] = shared_mem[i + half_B][j + half_B];
}

__global__ void phase2_col(int *dDist, int round, int n) {
    __shared__ int shr_pivot[B][B];
    __shared__ int pivot_col[B][B];

    const int blk_idx = blockIdx.x;
    if (blk_idx == round) return;

    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int offset = round * B;

    // Load pivot block
    shr_pivot[i][j] = dDist[(n + 1) * offset + i * n + j];
    shr_pivot[i][j + half_B] = dDist[(n + 1) * offset + i * n + j + half_B];
    shr_pivot[i + half_B][j] = dDist[(n + 1) * offset + (i + half_B) * n + j];
    shr_pivot[i + half_B][j + half_B] = dDist[(n + 1) * offset + (i + half_B) * n + j + half_B];

    // Load pivot_col
    pivot_col[i][j] = dDist[blk_idx * B * n + offset + i * n + j];
    pivot_col[i][j + half_B] = dDist[blk_idx * B * n + offset + i * n + j + half_B];
    pivot_col[i + half_B][j] = dDist[blk_idx * B * n + offset + (i + half_B) * n + j];
    pivot_col[i + half_B][j + half_B] = dDist[blk_idx * B * n + offset + (i + half_B) * n + j + half_B];
    __syncthreads();

    #pragma unroll 4
    for (int k = 0; k < B; ++k) {
        pivot_col[i][j] = min(pivot_col[i][j], pivot_col[i][k] + shr_pivot[k][j]);
        pivot_col[i][j + half_B] = min(pivot_col[i][j + half_B], pivot_col[i][k] + shr_pivot[k][j + half_B]);
        pivot_col[i + half_B][j] = min(pivot_col[i + half_B][j], pivot_col[i + half_B][k] + shr_pivot[k][j]);
        pivot_col[i + half_B][j + half_B] = min(pivot_col[i + half_B][j + half_B], pivot_col[i + half_B][k] + shr_pivot[k][j + half_B]);
    }

    // Write back
    dDist[blk_idx * B * n + offset + i * n + j] = pivot_col[i][j];
    dDist[blk_idx * B * n + offset + i * n + j + half_B] = pivot_col[i][j + half_B];
    dDist[blk_idx * B * n + offset + (i + half_B) * n + j] = pivot_col[i + half_B][j];
    dDist[blk_idx * B * n + offset + (i + half_B) * n + j + half_B] = pivot_col[i + half_B][j + half_B];
}

__global__ void phase2_row(int *dDist, int round, int n) {
    __shared__ int shr_pivot[B][B];
    __shared__ int pivot_row[B][B];

    const int blk_idx = blockIdx.x;
    if (blk_idx == round) return;

    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int offset = round * B;

    // Load pivot block
    shr_pivot[i][j] = dDist[(n + 1) * offset + i * n + j];
    shr_pivot[i][j + half_B] = dDist[(n + 1) * offset + i * n + j + half_B];
    shr_pivot[i + half_B][j] = dDist[(n + 1) * offset + (i + half_B) * n + j];
    shr_pivot[i + half_B][j + half_B] = dDist[(n + 1) * offset + (i + half_B) * n + j + half_B];

    // Load pivot_row
    pivot_row[i][j] = dDist[blk_idx * B + offset * n + i * n + j];
    pivot_row[i][j + half_B] = dDist[blk_idx * B + offset * n + i * n + j + half_B];
    pivot_row[i + half_B][j] = dDist[blk_idx * B + offset * n + (i + half_B) * n + j];
    pivot_row[i + half_B][j + half_B] = dDist[blk_idx * B + offset * n + (i + half_B) * n + j + half_B];
    __syncthreads();

    #pragma unroll 4
    for (int k = 0; k < B; ++k) {
        pivot_row[i][j] = min(pivot_row[i][j], shr_pivot[i][k] + pivot_row[k][j]);
        pivot_row[i][j + half_B] = min(pivot_row[i][j + half_B], shr_pivot[i][k] + pivot_row[k][j + half_B]);
        pivot_row[i + half_B][j] = min(pivot_row[i + half_B][j], shr_pivot[i + half_B][k] + pivot_row[k][j]);
        pivot_row[i + half_B][j + half_B] = min(pivot_row[i + half_B][j + half_B], shr_pivot[i + half_B][k] + pivot_row[k][j + half_B]);
    }

    // Write back
    dDist[blk_idx * B + offset * n + i * n + j] = pivot_row[i][j];
    dDist[blk_idx * B + offset * n + i * n + j + half_B] = pivot_row[i][j + half_B];
    dDist[blk_idx * B + offset * n + (i + half_B) * n + j] = pivot_row[i + half_B][j];
    dDist[blk_idx * B + offset * n + (i + half_B) * n + j + half_B] = pivot_row[i + half_B][j + half_B];
}

__global__ void phase3(int *dDist, int round, int n, int yoffset) {
    __shared__ int shared_mem[B][B];
    __shared__ int pivot_row[B][B];
    __shared__ int pivot_col[B][B];

    const int blk_i = blockIdx.y + yoffset;
    const int blk_j = blockIdx.x;
    if (blk_i == round || blk_j == round) return;

    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int offset = round * B;

    // Load pivot_col, pivot_row, and processed block
    pivot_col[i][j] = dDist[blk_i * B * n + offset + i * n + j];
    pivot_col[i][j + half_B] = dDist[blk_i * B * n + offset + i * n + j + half_B];
    pivot_col[i + half_B][j] = dDist[blk_i * B * n + offset + (i + half_B) * n + j];
    pivot_col[i + half_B][j + half_B] = dDist[blk_i * B * n + offset + (i + half_B) * n + j + half_B];

    pivot_row[i][j] = dDist[blk_j * B + offset * n + i * n + j];
    pivot_row[i][j + half_B] = dDist[blk_j * B + offset * n + i * n + j + half_B];
    pivot_row[i + half_B][j] = dDist[blk_j * B + offset * n + (i + half_B) * n + j];
    pivot_row[i + half_B][j + half_B] = dDist[blk_j * B + offset * n + (i + half_B) * n + j + half_B];

    shared_mem[i][j] = dDist[blk_i * B * n + blk_j * B + i * n + j];
    shared_mem[i][j + half_B] = dDist[blk_i * B * n + blk_j * B + i * n + j + half_B];
    shared_mem[i + half_B][j] = dDist[blk_i * B * n + blk_j * B + (i + half_B) * n + j];
    shared_mem[i + half_B][j + half_B] = dDist[blk_i * B * n + blk_j * B + (i + half_B) * n + j + half_B];
    __syncthreads();

    #pragma unroll 4
    for (int k = 0; k < B; ++k) {
        shared_mem[i][j] = min(shared_mem[i][j], pivot_col[i][k] + pivot_row[k][j]);
        shared_mem[i][j + half_B] = min(shared_mem[i][j + half_B], pivot_col[i][k] + pivot_row[k][j + half_B]);
        shared_mem[i + half_B][j] = min(shared_mem[i + half_B][j], pivot_col[i + half_B][k] + pivot_row[k][j]);
        shared_mem[i + half_B][j + half_B] = min(shared_mem[i + half_B][j + half_B], pivot_col[i + half_B][k] + pivot_row[k][j + half_B]);
    }

    // Write back
    dDist[blk_i * B * n + blk_j * B + i * n + j] = shared_mem[i][j];
    dDist[blk_i * B * n + blk_j * B + i * n + j + half_B] = shared_mem[i][j + half_B];
    dDist[blk_i * B * n + blk_j * B + (i + half_B) * n + j] = shared_mem[i + half_B][j];
    dDist[blk_i * B * n + blk_j * B + (i + half_B) * n + j + half_B] = shared_mem[i + half_B][j + half_B];
}