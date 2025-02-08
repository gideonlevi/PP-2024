#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <cmath>
#include <algorithm>
#include <boost/sort/sort.hpp>

void merge_sort_large_half(float* &arr1, int size1, float* &arr2, int size2, float* &result) {
    int i = size1 - 1, j = size2 - 1, k = size1 - 1;
    // merge the two arrays into the result array
    while (k >= 0) {
        if (arr1[i] >= arr2[j]) {
            result[k--] = arr1[i--];
        } else {
            result[k--] = arr2[j--];
        }
    }
}

void merge_sort_small_half(float* &arr1, int size1, float* &arr2, int size2, float* &result) {
    int i = 0, j = 0, k = 0;
    // merge the two arrays into the result array
    while (k < size1) {
        if (arr1[i] <= arr2[j]) {
            result[k++] = arr1[i++];
        } else {
            result[k++] = arr2[j++];
            // size2 might be < size1 because of load balancing of remainder
            if (j == size2 && k < size1) {
                result[k++] = arr1[i++];
            }
        }
    }
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];


    // calculate the number of elements on each process
    int base_size = floor(n / size);
    int remainder = n % size;
    int local_size = base_size + (rank < remainder ? 1 : 0);

    // for n < size cases
    size = std::min(size, n);

    // cache some calculations
    int rank_plus_1 = rank + 1, rank_minus_1 = rank - 1;
    int size_minus_1 = size - 1, local_size_minus_1 = local_size - 1;

    // calculate own displacement and recvcounts for adjacent processes
    int displacement = rank * base_size + std::min(rank, remainder);
    int prev_size = (rank_minus_1 >= 0) ? base_size + (rank_minus_1 < remainder ? 1 : 0) : 0;
    int next_size = (rank_plus_1 < size) ? base_size + (rank_plus_1 < remainder ? 1 : 0) : 0;

    // allocate buffer for local data on each process
    float *data = new float[local_size];
    float recv_value;
    float *recv_buff = new float[local_size + 1];
    float *res_buff = new float[local_size];


    // each process reads its own chunk of data from the file
    MPI_File input_file, output_file;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * displacement, data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    // MPI_File_close(&input_file);

    // sort the data in a process locally
    if (local_size <= 3000000) std::sort(data, data + local_size);
    else boost::sort::spreadsort::spreadsort(data, data + local_size);
    // else boost::sort::pdqsort(data, data + local_size);

    int threshold = size / 2 + 1;
    // sort using the odd-even sorting phases
    for (int i = 0; i < threshold; i++) {
        // skip comparison and jump to allreduce (for n < size cases)
        if (local_size == 0) {
            continue;
        }

        // even phase
        if (rank & 1) { // compare with previous process
            if (rank != 0) {
                MPI_Sendrecv(&data[0], 1, MPI_FLOAT, rank_minus_1, 0, &recv_value, 1, MPI_FLOAT, rank_minus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                if (recv_value > data[0]) {
                    MPI_Sendrecv(data, local_size, MPI_FLOAT, rank_minus_1, 0, recv_buff, prev_size, MPI_FLOAT, rank_minus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                    
                    merge_sort_large_half(data, local_size, recv_buff, prev_size, res_buff);
                    // std::copy(res_buff, res_buff + local_size, data);
                    std::swap(data, res_buff);
                }
            }
        }
        else { // compare with next process
            if (rank != size_minus_1) {
                MPI_Sendrecv(&data[local_size_minus_1], 1, MPI_FLOAT, rank_plus_1, 0, &recv_value, 1, MPI_FLOAT, rank_plus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                if (recv_value < data[local_size_minus_1]) { 
                    MPI_Sendrecv(data, local_size, MPI_FLOAT, rank_plus_1, 0, recv_buff, next_size, MPI_FLOAT, rank_plus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                    
                    merge_sort_small_half(data, local_size, recv_buff, next_size, res_buff);
                    // std::copy(res_buff, res_buff + local_size, data);
                    std::swap(data, res_buff);
                }
            }   
        }


        // odd phase
        if (rank & 1) { // compare with next process
            if (rank != size_minus_1) {
                MPI_Sendrecv(&data[local_size_minus_1], 1, MPI_FLOAT, rank_plus_1, 0, &recv_value, 1, MPI_FLOAT, rank_plus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                if (recv_value < data[local_size_minus_1]) {
                    MPI_Sendrecv(data, local_size, MPI_FLOAT, rank_plus_1, 0, recv_buff, next_size, MPI_FLOAT, rank_plus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 

                    merge_sort_small_half(data, local_size, recv_buff, next_size, res_buff);
                    // std::copy(res_buff, res_buff + local_size, data);
                    std::swap(data, res_buff);
                }
            }
        }
        else { // compare with previous process
            if (rank != 0) {
                MPI_Sendrecv(&data[0], 1, MPI_FLOAT, rank_minus_1, 0, &recv_value, 1, MPI_FLOAT, rank_minus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                if (recv_value > data[0]) {
                    MPI_Sendrecv(data, local_size, MPI_FLOAT, rank_minus_1, 0, recv_buff, prev_size, MPI_FLOAT, rank_minus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                    
                    merge_sort_large_half(data, local_size, recv_buff, prev_size, res_buff);
                    // std::copy(res_buff, res_buff + local_size, data);
                    std::swap(data, res_buff);
                }
            }
        }
    }

    // // synchronization blocking
    // MPI_Barrier(MPI_COMM_WORLD);

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * displacement, data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    // MPI_File_close(&output_file);

    // if (local_size) {
    //     delete[] data;
    // }
    // delete[] recv_buff;
    // delete[] res_buff;


    // MPI_Finalize();
    // return 0;
}