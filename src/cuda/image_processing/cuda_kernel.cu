

//
// Created by uyplayer on 2024/5/6.
//

#include "cuda_kernel.cuh"

__global__ void imageBlurAverageKernel(const uchar3 *input, uchar3 *output, int rows, int cols, int kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= rows && y <= cols) {
        int red = 0;
        int green = 0;
        int blue = 0;
        int count = 0;

        int center_x = x - kernel / 2;
        int center_y = y - kernel / 2;

        for (int i = 0; i < kernel; i++) {
            for (int j = 0; j < kernel; j++) {
                int current_x = center_x + i;
                int current_y = center_y + j;

                // 检查边界
                if (current_x >= 0 && current_x < rows && current_y >= 0 && current_y < cols) {
                    uchar3 pixel = input[current_x * cols + current_y];
                    red += pixel.x;
                    green += pixel.y;
                    blue += pixel.z;
                    count++;
                }
            }
        }

        if (count > 0) {
            output[x * cols + y].x = red / count;
            output[x * cols + y].y = green / count;
            output[x * cols + y].z = blue / count;
        }
    }
}
