//
// Created by uyplayer on 2024/5/18.
//


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>


void copy_example()
{
    int N = 5;
    int h_data[N] = {0, 1, 2, 3, 4};
    thrust::device_vector<int> D(N);
    thrust::copy(h_data, h_data + N, D.begin());


    int* d_ptr = thrust::raw_pointer_cast(D.data());
    // d_ptr send to kernel function

    thrust::host_vector<int> H(D.begin(), D.end());
    for (int i = 0; i < N; i++)
    {
        std::cout << "H[" << i << "] = " << H[i] << std::endl;
    }
}

struct is_even
{
    __host__ __device__
    bool operator()(const int x) const
    {
        return (x % 2) == 0;
    }
};

void copy_if_example()
{
    // 初始化host_vector并填充数据
    thrust::host_vector<int> h_vec(5);
    h_vec[0] = 0;
    h_vec[1] = 1;
    h_vec[2] = 2;
    h_vec[3] = 3;
    h_vec[4] = 4;

    // 创建device_vector
    thrust::device_vector<int> d_vec(5);

    // 只复制偶数到device_vector
    auto end = thrust::copy_if(h_vec.begin(), h_vec.end(), d_vec.begin(), is_even());

    // 输出设备向量中的数据
    thrust::host_vector<int> h_vec_result(d_vec.begin(), end);
    for (int i = 0; i < h_vec_result.size(); i++)
    {
        std::cout << h_vec_result[i] << " ";
    }
    std::cout << std::endl;
}

void copy_n_example()
{
    // 初始化host_vector并填充数据
    thrust::host_vector<int> h_vec(5);
    h_vec[0] = 0;
    h_vec[1] = 1;
    h_vec[2] = 2;
    h_vec[3] = 3;
    h_vec[4] = 4;

    // 创建device_vector
    thrust::device_vector<int> d_vec(5);

    // 复制前3个元素到device_vector
    thrust::copy_n(h_vec.begin(), 3, d_vec.begin());

    // 输出设备向量中的数据
    thrust::host_vector<int> h_vec_result = d_vec;
    for (int i = 0; i < h_vec_result.size(); i++)
    {
        std::cout << h_vec_result[i] << " ";
    }
    std::cout << std::endl;
}
