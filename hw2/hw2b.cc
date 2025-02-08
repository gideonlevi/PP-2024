#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <iostream>
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include <immintrin.h>

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0); // set to 0
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
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
    // free(row);
    png_write_end(png_ptr, NULL);
    // png_destroy_write_struct(&png_ptr, &info_ptr);
    // fclose(fp);
}

int main(int argc, char** argv) {
    double COM_TOTAL_TIME = 0;

    int rank, size;
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    int i, root, row, col;
    double x0,y0;
    double scale_x = (right - left) / (double)width;
    double scale_y = (upper - lower) / (double)height; 

    int base_rows = (double)height / size;
    int remainder_rows = height % size;
    int local_rows = (rank < remainder_rows) ? base_rows + 1 : base_rows;
    int start_row = rank * base_rows + std::min(rank, remainder_rows);
    int end_row = start_row + local_rows;
    int* gather_image = (rank == 0) ? (int*)malloc(width * height * sizeof(int)) : NULL;
    int* local_image = (int*)malloc(local_rows * width * sizeof(int));

    // compute displacement and receive count of each process
    int* displs = (int*)malloc(size * sizeof(int));
    int* rcounts = (int*)malloc(size * sizeof(int));
    displs[0] = 0;
    for(i = 0 ;i < size; i++){
        if(i < remainder_rows)
            rcounts[i] = (base_rows + 1) * width;
        else
            rcounts[i] = base_rows * width;

        if(i + 1 < size)
            displs[i + 1] = displs[i] + rcounts[i];
    }

    __m512d four_v = _mm512_set1_pd(4.0);
    __m512d two_v = _mm512_set1_pd(2.0);
    __m512i one_v = _mm512_set1_epi64(1);

    #pragma omp parallel for schedule(dynamic) default(shared)
    for (int row = 0; row < local_rows; ++row) {
        double y0 = (rank + size * row) * scale_y + lower;
        __m512d y0_v = _mm512_set1_pd(y0);
        #pragma omp parallel for schedule(dynamic) default(shared)
        for (int col = 0; col < width - 7; col += 8) {
            double x0[8];
            for (int i = 0; i < 8; i++) {
                x0[i] = (col + i) * scale_x + left;
            }
            __m512d x0_v = _mm512_loadu_pd(x0);
            __m512d x_v = _mm512_setzero_pd();
            __m512d y_v = _mm512_setzero_pd();
            __m512d x2_v = _mm512_setzero_pd();
            __m512d y2_v = _mm512_setzero_pd();
            __m512i repeat_v = _mm512_setzero_si512();
            __m512d length_squared_v = _mm512_setzero_pd();
            int repeats = 0;
            
            // Main loop for Mandelbrot calculations
            while(repeats < iters) {
                __mmask8 mask = _mm512_cmp_pd_mask(length_squared_v, four_v, _MM_CMPINT_LT);

                if (!mask) // Break if all values are outside the threshold
                    break;

                __m512d temp = _mm512_add_pd(_mm512_sub_pd(x2_v, y2_v), x0_v);
                y_v = _mm512_add_pd(_mm512_mul_pd(_mm512_mul_pd(x_v, y_v), two_v), y0_v);
                // y_v = _mm512_fmadd_pd(_mm512_add_pd(x_v, x_v), y_v, y0_v); // Fused-multiply-add: y = 2*x*y + y0 (INNACURATE RESULTS)
                x_v = temp;

                // Compute squares
                x2_v = _mm512_mul_pd(x_v, x_v);
                y2_v = _mm512_mul_pd(y_v, y_v);
                length_squared_v = _mm512_add_pd(x2_v, y2_v);

                repeats++;
                repeat_v = _mm512_mask_add_epi64(repeat_v, mask, repeat_v, one_v);
            }

            // Store the results back to local_image
            __m256i result_v = _mm512_cvtepi64_epi32(repeat_v);  // Convert 64-bit doubles to 32-bit integers
            _mm256_storeu_si256((__m256i*)(local_image + row * width + col), result_v);  // Store 256 bits
        }

        // Handle remaining columns if width is not a multiple of 8
        for (int col = width - (width % 8); col < width; col++) {
            double x0 = col * scale_x + left;
            int repeats = 0;
            double x = 0, y = 0, length_squared = 0;

            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            local_image[row * width + col] = repeats;
        }
    }

    MPI_Gatherv(local_image, local_rows * width, MPI_INT, gather_image, rcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank==0){
        int* result_image = (int*)malloc(width * height * sizeof(int));
        int index = 0;
        for(i = 0; i < local_rows; i++){
            int stride = local_rows;
            int check_out_of_remainder = 0;
            for(row = i; row < height && index < height * width; row += stride){
                for(col = 0; col < width; col++){
                    result_image[index] = gather_image[row * width + col];
                    index++;
                }
                if(check_out_of_remainder >= remainder_rows){
                    stride = base_rows;
                }
                check_out_of_remainder++;
            }
        }
        write_png(filename, iters, width, height, result_image);
        // free(result_image);
    }

    // free(displs);
    // free(rcounts);
    // free(local_image);
    // free(gather_image);
    // MPI_Finalize();
}