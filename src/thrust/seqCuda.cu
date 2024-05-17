//
// Created by uyplayer on 2024/5/17.
//


#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>


void seqCuda()
{
    const int size = 10000;
    thrust::device_vector<int> d_vec(size);
    thrust::sequence(d_vec.begin(), d_vec.end(), 0);
    int sumA = thrust::reduce(d_vec.begin(), d_vec.end(), 0);
    int sumCheck = 0;
    for (int i = 0; i < size; i++) sumCheck += i;

    if (sumA == sumCheck) std::cout << "Test Succeeded!" << std::endl;
    else { std::cerr << "Test FAILED!" << std::endl; }
}
