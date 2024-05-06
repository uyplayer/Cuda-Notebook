

//
// Created by uyplayer on 2024/5/6.
//

#include "cuda_kernel.cuh"


__global__ void imageBlurAverageKernel(const uchar3 *input, uchar3 *output, int rows, int cols, int kernel) {

    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < rows && y < cols) {
        int red = 0;
        int green = 0;
        int blue = 0;
        int count = 0;

        // kernel equal to kernel size * kernel size
        for (int i = -kernel / 2; i <= kernel / 2; i++) {
            for (int j = -kernel / 2; j <= kernel / 2; j++) {
                int current_x = x + i;
                int current_y = y + j;

                if (current_x >= 0 && current_x < rows && current_y >= 0 && current_y < cols) {
                    red += input[current_x * cols + current_y].x;
                    green += input[current_x * cols + current_y].y;
                    blue += input[current_x * cols + current_y].z;
                    count++;
                }
            }
        }

        output[x * cols + y].x = red / count;
        output[x * cols + y].y = green / count;
        output[x * cols + y].z = blue / count;
    }


}
