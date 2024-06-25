//
// Created by uyplayer on 2024-06-23.
//


#include <iostream>


#define N 1024
#define THREADS_PER_BLOCK 512

int main() {
    std::cout << "Hello, mergeSort !" << std::endl;
}



__global__ void merge(int *arr, int *temp, int size, int width) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x; // 计算全局线程索引
    int start1 = 2 * tid * width; // 第一个子数组的起始位置
    int start2 = start1 + width; // 第二个子数组的起始位置
    int end1 = min(start1 + width, size); // 第一个子数组的结束位置
    int end2 = min(start2 + width, size); // 第二个子数组的结束位置

     int i = start1, j = start2, k = start1; // 初始化合并的指针
    while (i < end1 && j < end2) {
        if (arr[i] < arr[j]) {
            temp[k++] = arr[i++]; // 将较小的元素复制到临时数组
        } else {
            temp[k++] = arr[j++];
        }
    }
    while (i < end1) {
        temp[k++] = arr[i++]; // 复制剩余的元素
    }
    while (j < end2) {
        temp[k++] = arr[j++];
    }

}


__global__ void copyArray(int *src, int *dest, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        dest[tid] = src[tid];
    }
}

void mergeSort(int *arr, int size) {
    int *d_arr, *d_temp;
    cudaMalloc((void**)&d_arr, size * sizeof(int)); // 分配GPU内存
    cudaMalloc((void**)&d_temp, size * sizeof(int)); // 分配临时数组

    cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice); // 将数据从CPU复制到GPU

    dim3 blockSize(THREADS_PER_BLOCK);
    dim3 gridSize((size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    for (int width = 1; width < size; width *= 2) {
        merge<<<gridSize, blockSize>>>(d_arr, d_temp, width, size); // 调用合并内核
        cudaDeviceSynchronize(); // 同步设备
        copyArray<<<gridSize, blockSize>>>(d_temp, d_arr, size); // 调用拷贝内核
        cudaDeviceSynchronize(); // 同步设备
    }

    cudaMemcpy(arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost); // 将排序结果从GPU复制回CPU

    cudaFree(d_arr); // 释放GPU内存
    cudaFree(d_temp); // 释放临时数组
}
