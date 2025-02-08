#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <chrono>
#define CHUNK_SIZE 10
using namespace std;

const int INF = ((1 << 30) - 1);
const int V = 6000;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m, NUM_THREADS;
static int Dist[V][V];

int main(int /*argc*/, char* argv[]) {
    std::chrono::steady_clock::time_point total_start = std::chrono::steady_clock::now();
    input(argv[1]);
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    NUM_THREADS = CPU_COUNT(&cpu_set);
    omp_set_num_threads(NUM_THREADS);
    int B = 64;
    block_FW(B);
    output(argv[2]);
    std::chrono::steady_clock::time_point total_end = std::chrono::steady_clock::now();
    std::cout << "[TOTAL_TIME] " << std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count() << "\n";
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    // printf("n: %d, m: %d\n", n, m);

    for (int i = 0; i < n; ++i) {
        #pragma omp simd
        for (int j = 0; j < i; ++j) {
            Dist[i][j] = INF;
        }
        Dist[i][i] = 0;
        #pragma  omp simd
        for (int j = i + 1; j < n; ++j) {
            Dist[i][j] = INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        // for (int j = 0; j < n; ++j) {
        //     if (Dist[i][j] >= INF) Dist[i][j] = INF;
        // }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        /* Phase 1*/
        cal(B, r, r, r, 1, 1);

        /* Phase 2*/
        cal(B, r, r, 0, r, 1);
        cal(B, r, r, r + 1, round - r - 1, 1);
        cal(B, r, 0, r, 1, r);
        cal(B, r, r + 1, r, 1, round - r - 1);

        /* Phase 3*/
        cal(B, r, 0, 0, r, r);
        cal(B, r, 0, r + 1, round - r - 1, r);
        cal(B, r, r + 1, 0, r, round - r - 1);
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    int RoundB = Round * B;
    int RoundB_B = RoundB + B;
    if (RoundB_B > n) RoundB_B = n;

    #pragma omp parallel for schedule(dynamic)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        int block_internal_start_x = b_i * B;
        int block_internal_end_x = (b_i + 1) * B;
        if (block_internal_end_x > n) block_internal_end_x = n;
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = (b_j + 1) * B;
            if (block_internal_end_y > n) block_internal_end_y = n;

            for (int k = RoundB; k < RoundB_B; ++k) {
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    int IK = Dist[i][k];
                    // #pragma clang loop vectorize(enable) interleave(enable)
                    #pragma omp simd
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        int l = IK + Dist[k][j];
                        int r = Dist[i][j];
                        Dist[i][j] = l * (l < r) + r * (l >= r);
                    }
                }
            }
        }
    }
}
