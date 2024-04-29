/*
 * @Author: uyplayer
 * @Date: 2024/4/29 21:17
 * @Email: uyplayer@qq.com
 * @File: sumArraysOnHost.cu
 * @Software: CLion
 * @Dir: Cuda-Notebook / src/professional-cuda-programming/examples
 * @Project_Name: Cuda-Notebook
 * @Description:
 */



#include <iostream>
#include <Initializer.h>

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }

}

void sumArraysOnHost(){
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    sumArraysOnHost(h_A, h_B, h_C, nElem);

    free(h_A);
    free(h_B);
    free(h_C);

}