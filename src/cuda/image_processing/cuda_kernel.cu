

//
// Created by uyplayer on 2024/5/6.
//

#include "cuda_kernel.cuh"


__global__ void imageBlurGaussianKernel(const uchar3 *input, uchar3 *output, int rows, int cols, int kernel) {

    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
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
                if (current_x >= 0 && current_x < cols && current_y >= 0 && current_y < rows) {
                    uchar3 pixel = input[current_y * cols + current_x];
                    red += pixel.x;
                    green += pixel.y;
                    blue += pixel.z;
                    count++;
                }
            }
        }

        if (count > 0) {
            output[x + y * cols].x = red / count;
            output[x + y * cols].y = green / count;
            output[x + y * cols].z = blue / count;
        }
    }
}

// buble sort
__device__ void buble_sort(int *array_values, int kernel) {
    for (int i = 0; i < kernel; i++) {
        for (int j = i + 1; j < kernel; j++) {
            if (array_values[i] > array_values[j]) {
                int temp = array_values[i];
                array_values[i] = array_values[j];
                array_values[j] = temp;
            }
        }
    }
}


__global__ void imageBlurMedianKernel(const uchar3 *input, uchar3 *output, int rows, int cols) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    constexpr  int kernelSize = 50;
    if (x < cols && y < rows) {
        int red[kernelSize * kernelSize];
        int green[kernelSize * kernelSize];
        int blue[kernelSize * kernelSize];

        int pixel_index = 0;
        for (int i = -kernelSize / 2; i <= kernelSize / 2; i++) {
            for (int j = -kernelSize / 2; j <= kernelSize / 2; j++) {
                int new_x = min(max(x + i, 0), cols - 1);
                int new_y = min(max(y + j, 0), rows - 1);
                uchar3 pixels = input[new_y * cols + new_x];
                red[pixel_index] = pixels.x;
                green[pixel_index] = pixels.y;
                blue[pixel_index] = pixels.z;
                pixel_index++;
            }
        }

        // Sort the pixels
        buble_sort(red, kernelSize);
        buble_sort(green, kernelSize);
        buble_sort(blue, kernelSize);

        int median_index = kernelSize * kernelSize / 2;
        uchar3 median_pixel;
        if (kernelSize % 2 == 0) {
            median_pixel.x = (red[median_index] + red[median_index + 1]) / 2;
            median_pixel.y = (green[median_index] + green[median_index + 1]) / 2;
            median_pixel.z = (blue[median_index] + blue[median_index + 1]) / 2;
        } else {
            median_pixel.x = red[median_index];
            median_pixel.y = green[median_index];
            median_pixel.z = blue[median_index];
        }

        output[y * cols + x] = median_pixel;
    }
}
