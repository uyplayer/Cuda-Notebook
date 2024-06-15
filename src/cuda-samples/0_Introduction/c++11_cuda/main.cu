//
// Created by uyplayer on 2024-06-15.
//


#include <fstream>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include "error.h"


const char *sdkFindFilePath(const char *filename, const char *executable_path) {
    static char filepath[512];
    strcpy(filepath, executable_path);

    char *last_slash = strrchr(filepath, '/');
    if (!last_slash) {
        last_slash = strrchr(filepath, '\\');
    }

    if (last_slash) {
        last_slash[1] = '\0';
    } else {
        filepath[0] = '\0';
    }

    strcat(filepath, filename);
    std::cout << "filepath: " << filepath << std::endl;
    // 检查文件是否存在
    std::ifstream file(filepath);
    if (file.good()) {
        return filepath;
    } else {
        std::cerr << "File not found: " << filepath << std::endl;
        return nullptr;
    }
}

__device__ void count_if(int *count, char *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const char letters[]{'x', 'y', 'z', 'w'};
    for (int i = idx; i < n; i += stride) {
        for (const auto x: letters) {
            if (data[i] == x) {
                atomicAdd(count, 1);
                break;
            }
        }
    }
}

__global__ void xyzw_frequency(int *count, char *text, int n) {
    count_if(count, text, n);
}

int main(int argc, char **argv) {
    std::cout << "Hello, c++11_cuda!" << std::endl;
    const char *filename = sdkFindFilePath("warandpeace.txt",
                                           "D:/Cuda/Cuda-Notebook/src/cuda-samples/0_Introduction/c++11_cuda/");
    if (!filename) {
        std::cerr << "Cannot find the input text file. Exiting.." << std::endl;
        return EXIT_FAILURE;
    }

    int numBytes = 16 * 1048576;
    char *h_text = (char *) malloc(numBytes);
    int devID = 0;
    char *d_text;
    HANDLE_ERROR(cudaMalloc((void **)&d_text, numBytes));

    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Cannot find the input text file\n. Exiting..\n");
        return EXIT_FAILURE;
    }
    int len = (int) fread(h_text, sizeof(char), numBytes, fp);
    fclose(fp);
    std::cout << "Read " << len << " byte corpus from " << filename << std::endl;

    HANDLE_ERROR(cudaMemcpy(d_text, h_text, len, cudaMemcpyHostToDevice));

    int count = 0;
    int *d_count;
    HANDLE_ERROR(cudaMalloc(&d_count, sizeof(int)));
    HANDLE_ERROR(cudaMemset(d_count, 0, sizeof(int)));

    xyzw_frequency<<<8, 256>>>(d_count, d_text, len);
    HANDLE_ERROR(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "counted " << count
            << " instances of 'x', 'y', 'z', or 'w' in \"" << filename << "\""
            << std::endl;

    HANDLE_ERROR(cudaFree(d_count));
    HANDLE_ERROR(cudaFree(d_text));
    free(h_text);

    return EXIT_SUCCESS;
}
