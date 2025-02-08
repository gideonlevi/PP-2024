#include <iostream>
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <atomic>
#include <cmath>
#include <mutex>
#include <condition_variable>

std::atomic<int> current_row(-1);

pthread_mutex_t png_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
int* png_ready;

typedef struct {
    double left, right, upper, lower;
    unsigned int width, height;
    int thread_id;
    int num_threads;
    int iters;
    int* image;
    double scale_x, scale_y;
} ThreadData;

void* mandelbrot_slave(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    __m512d four_v = _mm512_set1_pd(4.0);
    __m512d two_v = _mm512_set1_pd(2.0);
    __m512i one_v = _mm512_set1_epi64(1);

    while (true) {
        // int chunk_size = compute_chunk_size(remaining_rows);
        int start_row = current_row.fetch_sub(1);
        if (start_row < 0) break;
        int end_row = std::max(start_row - 1, -1);

        // Write the row to the PNG
        size_t row_size = 3 * data->width * sizeof(png_byte);
        png_bytep row_data = (png_bytep)malloc(row_size);

        for (int row = start_row; row > end_row; --row) {
            double y0 = row * data->scale_y + data->lower;
            __m512d y0_v = _mm512_set1_pd(y0);

            for (int col = 0; col < data->width - 7; col += 8) {
                double x0[8];
                for (int i = 0; i < 8; ++i) {
                    x0[i] = (col + i) * data->scale_x + data->left;
                }
                __m512d x0_v = _mm512_loadu_pd(x0);
                __m512d x_v = _mm512_setzero_pd();
                __m512d y_v = _mm512_setzero_pd();
                __m512d x2_v = _mm512_setzero_pd();
                __m512d y2_v = _mm512_setzero_pd();
                __m512i repeat_v = _mm512_setzero_si512();
                int repeats = 0;

                while (repeats < data->iters) {
                    __mmask8 mask = _mm512_cmp_pd_mask(_mm512_add_pd(x2_v, y2_v), four_v, _MM_CMPINT_LT);
                    if (mask == 0) break;

                    __m512d temp = _mm512_add_pd(_mm512_sub_pd(x2_v, y2_v), x0_v);
                    y_v = _mm512_add_pd(_mm512_mul_pd(_mm512_mul_pd(x_v, y_v), two_v), y0_v);
                    x_v = temp;

                    x2_v = _mm512_mul_pd(x_v, x_v);
                    y2_v = _mm512_mul_pd(y_v, y_v);
                    repeat_v = _mm512_mask_add_epi64(repeat_v, mask, repeat_v, one_v);
                    repeats++;
                }
                __m256i result_v = _mm512_cvtepi64_epi32(repeat_v);
                _mm256_storeu_si256((__m256i*)(data->image + row * data->width + col), result_v);
            }

            for (int col = data->width - (data->width % 8); col < data->width; ++col) {
                double x0 = col * data->scale_x + data->left;
                double x = 0, y = 0, x2 = 0, y2 = 0;
                int repeats = 0;
                while (repeats < data->iters && (x2 + y2) < 4) {
                    y = 2 * x * y + y0;
                    double temp = x2 - y2 + x0;
                    x = temp;
                    x2 = x * x;
                    y2 = y * y;
                    ++repeats;
                }
                data->image[row * data->width + col] = repeats;
            }

            // Ensure rows are written in order
            pthread_mutex_lock(&png_mutex);
            png_ready[row] = 1;
            pthread_mutex_unlock(&png_mutex);
            pthread_cond_signal(&cv);
        }
    }

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    int ncpus = CPU_COUNT(&cpuset);

    const char* filename = argv[1];
    int iters = strtol(argv[2], NULL, 10);
    double left = strtod(argv[3], NULL);
    double right = strtod(argv[4], NULL);
    double lower = strtod(argv[5], NULL);
    double upper = strtod(argv[6], NULL);
    unsigned int width = strtol(argv[7], NULL, 10);
    unsigned int height = strtol(argv[8], NULL, 10);

    double scale_x = (right - left) / width;
    double scale_y = (upper - lower) / height;

    int* image = (int*)malloc(width * height * sizeof(int));
    current_row = height - 1;
    png_ready = (int*)malloc(height * sizeof(int));
    memset(png_ready, 0, height);

    pthread_t threads[ncpus];
    ThreadData thread_data[ncpus];

    for (int i = 0; i < ncpus; ++i) {
        thread_data[i] = {
            .left = left, .right = right, .upper = upper, .lower = lower,
            .width = width, .height = height, .thread_id = i, .num_threads = ncpus,
            .iters = iters, .image = image,
            .scale_x = scale_x, .scale_y = scale_y
        };
        pthread_create(&threads[i], NULL, mandelbrot_slave, &thread_data);
    }

    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0); // set to 0

    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = height - 1; y >= 0; --y) {
        pthread_mutex_lock(&png_mutex);
        while (!png_ready[y]) {
            pthread_cond_wait(&cv, &png_mutex);
        }
        pthread_mutex_unlock(&png_mutex);

        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = image[y * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = (p & 15) << 4;
                } else {
                    color[0] = (p & 15) << 4;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    png_write_end(png_ptr, NULL);


    // for (int i = 0; i < ncpus; ++i) {
    //     pthread_join(threads[i], NULL);
    // }

    // png_write_end(png_ptr, NULL);
    // png_destroy_write_struct(&png_ptr, &info_ptr);
    // fclose(fp);

    // return 0;
}
