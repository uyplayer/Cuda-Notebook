/*
 * @Author: uyplayer
 * @Date: 2024/4/29 20:39
 * @Email: uyplayer@qq.com
 * @File: utilits.h
 * @Software: CLion
 * @Dir: Cuda-Notebook / common
 * @Project_Name: Cuda-Notebook
 * @Description:
 */



#ifndef CUDA_NOTEBOOK_UTILITS_H
#define CUDA_NOTEBOOK_UTILITS_H


#include <algorithm>


void initialData(float *ip, int size) {


    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xFF) / 10.0f;
    }

}


#include <chrono>

inline double seconds() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    return microseconds.count() * 1.0e-6;
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            std::cout << "Arrays do not match!\n";
            std::cout << "host " << std::fixed << std::setprecision(2) << hostRef[i] << " gpu " << gpuRef[i]
                      << " at current " << i << "\n";
            break;
        }
    }

    if (match) {
        std::cout << "Arrays match.\n\n";
    }

}

#endif
