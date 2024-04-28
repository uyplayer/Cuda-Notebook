/*
 * @Author: uyplayer
 * @Date: 2024/4/28 22:10
 * @Email: uyplayer@qq.com
 * @File: main.cu
 * @Software: CLion
 * @Dir: Cuda-Notebook / src/CUDA-Programming/shared-memory
 * @Project_Name: Cuda-Notebook
 * @Description:
 */


#include <iostream>
#include "error.h"
#include <algorithm>
#include <iomanip>


#ifdef USE_DP
typedef double real;
#else
typedef float real;
#endif


constexpr int NUM_REPEATS = 10;
constexpr int TILE_DIM = 32;


int main(int argc, char **argv) {
    std::cout << "Hello shared-memory" << std::endl;

    constexpr int N = 1 << 10;
    constexpr int N2 = N * N;
    constexpr int M = sizeof(real) * N2;


    real *h_A = new real[M];
    real *h_B = new real[M];

    for (int n = 0; n < N2; ++n) {
        h_A[n] = n;
    }

    real *d_A, *d_B;
    CHECK(cudaMalloc(&d_A, M));
    CHECK(cudaMalloc(&d_B, M));
    CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));


    return 0;
}

void timing(const real *d_A, real *d_B, const int N, const int task) {
    const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_size_y = grid_size_x;

    const dim3 block_size(TILE_DIM, TILE_DIM);
    const dim3 grid_size(grid_size_x, grid_size_y);

    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        switch (task) {

            case 1:
                std::cout << "case 1" << std::endl;
            case 2:
                std::cout << "case 2" << std::endl;
            case 3:
                std::cout << "case 3" << std::endl;
            case 4:
                std::cout << "case 4" << std::endl;

        }
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);\
        std::cout << "Time = " << elapsed_time << " %g ms." << std::endl;


        if (repeat > 0) {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));


    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    std::cout << "Time = " << std::fixed << std::setprecision(2) << t_ave << " +- " << t_err << " ms." << std::endl;

}



