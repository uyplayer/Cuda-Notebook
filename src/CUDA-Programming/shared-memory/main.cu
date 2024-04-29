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



/*
 为了确保并行线程协作时获得正确的结果，我们必须同步线程。CUDA 提供了一个简单的屏障同步原语，
 __syncthreads()。线程的执行只能__syncthreads()在其块中的所有线程都执行完之后才能继续进行__syncthreads()。
 因此，我们可以通过在将存储到共享内存之后和任何线程从共享内存加载之前调用来避免上述竞争条件__syncthreads()。
 重要的是要注意，__syncthreads()在不同的代码中调用是未定义的，并且可能导致死锁——线程块中的所有线程必须__syncthreads()在同一点调用。
*/

__global__ void staticReverse(int *d,int n){
    __shared__ int s[64];
    int t = threadIdx.x;
    int tr = n-t-1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
};


__global__ void dynamicReverse(int *d, int n)
{
    extern __shared__ int s[];
    int t = threadIdx.x;
    int tr = n-t-1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}



int main(int argc, char **argv) {
    std::cout << "Hello shared-memory" << std::endl;

    constexpr int n =  64;

    const int size = n * sizeof(real);


    real *a = new real[n];
    real *b = new real[n];
    real *c = new real[n];


    // Initialize memory
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = n-i-1;
        c[i] = 0;
    }


    int *d_d;
    CHECK(cudaMalloc(&d_d, size));
    CHECK(cudaMemcpy(d_d, a, size, cudaMemcpyHostToDevice));

    staticReverse<<<1, 64>>>(d_d, n);
    cudaMemcpy(c, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        if (c[i] != b[i])
            std::cout << "Error: d[" << i << "]!=r[" << i << "] (" << c[i] << ", " << b[i] << ")\n";
    }



    CHECK(cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice););
    dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);
    CHECK(cudaMemcpy(c, d_d, n * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++)
        if (c[i] != b[i])
            std::cout << "Error: d[" << i << "]!=r[" << i << "] (" << c[i] << ", " << b[i] << ")\n";

    return 0;
}



__global__ void reverse_dynamic(int nI, int nF, int nC)
{
    extern __shared__ int s[];
    int *integerData = (int*)s;
    float *floatData = (float*)(s + nI * sizeof(int)); // 浮点数数组的起始地址
    char *charData = (char*)(s + nI * sizeof(int) + nF * sizeof(float)); // 字符数组的起始地址


//    myKernel<<<gridSize, blockSize, nI * sizeof(int) + nF * sizeof(float) + nC * sizeof(char)>>>(nI, nF, nC);

}

