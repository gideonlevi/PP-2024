#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

void input(char *input_filename);
void output(char *output_filename);
void flash_attention(float *q, float *k, float *v, float *o);
__global__ void forward_kernel(float *out, float *q, float *k, float *v, const int N, const int d, int tr, int tc, float scalar);

float _max(float a, float b) { return a > b ? a : b; }
float _min(float a, float b) { return a < b ? a : b; }
double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;
#define bc 32
#define br 32

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    double start, end;
    start = getTimeStamp();

    flash_attention(Q, K, V, O);

    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}

void flash_attention(float *q, float *k, float *v, float *o) {
    const int tr = ceil((float) N / br);
    const int tc = ceil((float) N / bc);

    float *d_q, *d_k, *d_v, *d_o;
    cudaMalloc((void **)&d_q, B * N * d * sizeof(float));
    cudaMalloc((void **)&d_k, B * N * d * sizeof(float));
    cudaMalloc((void **)&d_v, B * N * d * sizeof(float));
    cudaMalloc((void **)&d_o, B * N * d * sizeof(float));
    // copy input data to GPU
    cudaMemcpy(d_q, q, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, B * N * d * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_dim(B, tr);
    dim3 block_dim(br, bc);

    forward_kernel<<<grid_dim, block_dim, ((2 * br * d) + (2 * br * bc) + (2 * br)) * sizeof(float)>>>(
        d_o, d_q, d_k, d_v, N, d, tr, tc, 1.0 / sqrt(d)
    );
    cudaDeviceSynchronize();

    // copy result back to host
    cudaMemcpy(o, d_o, B * N * d * sizeof(float), cudaMemcpyDeviceToHost);

    // free(l);
    // free(m);
}

__global__ void forward_kernel(float *out, float *q, float *k, float *v, const int N, const int d, int tr, int tc, float scalar) {
    extern __shared__ float shared_mem[];

    const int blk_x = blockIdx.x;
    const int blk_y = blockIdx.y;
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    int qkv_batch_offset = blk_x * N * d;
    int tile_size = bc * d;
    int d_per_bc = d / bc;

    float *shared_q = shared_mem;                         // Size: br * d
    float *shared_k = shared_q + br * d;                  // Size: bc * bc
    float *shared_v = shared_k + br * bc;                 // Size: bc * d
    float *shared_s = shared_v + bc * d;                  // Size: br * bc
    float *shared_l = shared_s + br * bc;                 // Size: br
    float *shared_m = shared_l + br;                      // Size: br

    if (tid_x == 0) {
        shared_l[tid_y] = 0.f;
        shared_m[tid_y] = FLT_MIN;
    }
    
    // Load qi into shared memory
    for (int x = 0; x < d_per_bc; x++) {
        shared_q[(tid_y * d) + (x * bc) + tid_x] = q[qkv_batch_offset + (tile_size * blk_y) + (tid_y * d) + (x * bc) + tid_x]; // qi
    }
    for (int j = 0; j < tc; j++) {
        shared_s[(tid_y * bc) + tid_x] = 0.0F;
        float S_ij = 0.f;
        for (int x = 0; x < d_per_bc; x++) {
            // Load kj and vj into shared memory
            // partition kj into d_per_bc partitions
            shared_k[(tid_y * bc) + tid_x] = k[qkv_batch_offset + (tile_size * j) + (tid_y * d) + (x * bc) + tid_x]; // kj
            shared_v[(tid_y * d) + (x * bc) + tid_x] = v[qkv_batch_offset + (tile_size * j) + (tid_y * d) + (x * bc) + tid_x]; // vj
            __syncthreads();

            
            // q dot k and scalar
            #pragma unroll 32
            for (int y = 0; y < bc; y++) {
                S_ij += shared_q[(tid_y * d) + (x * bc) + y] * shared_k[(tid_x * bc) + y];
            }
            __syncthreads();
        }
        shared_s[(tid_y * bc) + tid_x] += S_ij * scalar;
        __syncthreads();
        

        // find max
        float row_m = -INFINITY;
        // #pragma unroll 32
        for (int y = 0; y < bc; y++) {
            if (shared_s[(tid_y * bc) + y] > row_m) {
                row_m = shared_s[(tid_y * bc) + y];
            }
        }


        // minus max, exponent
        shared_s[(tid_y * bc) + tid_x] = __expf(shared_s[(tid_y * bc) + tid_x] - row_m);
        // __syncthreads();

        // find row sum
        float row_l = 0;
        #pragma unroll 32
        for (int y = 0; y < bc; y++) {
            row_l += shared_s[(tid_y * bc) + y];
        }


        // update mi, li, and oi.
        float prev_row_m = shared_m[tid_y];
        float prev_row_l = shared_l[tid_y];
        float new_row_m = max(prev_row_m, row_m);
        float new_row_l = (__expf(prev_row_m - new_row_m) * prev_row_l) + (__expf(row_m - new_row_m) * row_l);
        
        
        // p dot v and write o back to HBM
        for (int x = 0; x < d_per_bc; x++) {
            float pv = 0;
            #pragma unroll 32
            for (int y = 0; y < bc; y++) {
                pv += shared_s[(tid_y * bc) + y] * shared_v[(y * d) + (x * bc) + tid_x];
            }
            out[qkv_batch_offset + (tile_size * blk_y) + (tid_y * d) + (x * bc) + tid_x] = (1 / new_row_l) \
                * ((prev_row_l * __expf(prev_row_m - new_row_m) * out[qkv_batch_offset + (tile_size * blk_y) + (tid_y * d) + (x * bc) + tid_x]) \
                + (__expf(row_m - new_row_m) * pv));
        }

        if (tid_x == 0) {
            // write l, m back to HBM
            shared_m[tid_y] = new_row_m;
            shared_l[tid_y] = new_row_l;
        }
        
        // __syncthreads();
    }
}