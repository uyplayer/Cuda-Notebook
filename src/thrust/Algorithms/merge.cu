//
// Created by uyplayer on 2024/5/18.
//


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <iostream>

void merge_example() {
    int h_data1[] = {1, 3, 5};
    int h_data2[] = {2, 3, 4};

    // 创建host_vector并复制数据
    thrust::host_vector<int> h_vec1(h_data1, h_data1 + 3);
    thrust::host_vector<int> h_vec2(h_data2, h_data2 + 3);

    thrust::device_vector<int> d_vec1 = h_vec1;
    thrust::device_vector<int> d_vec2 = h_vec2;
    thrust::device_vector<int> d_result(6);

    thrust::merge(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_vec2.end(), d_result.begin());

    thrust::host_vector<int> h_result = d_result;
    for (int i = 0; i < h_result.size(); i++) {
        std::cout << "h_result[" << i << "] = " << h_result[i] << std::endl;
    }
}

void set_union_example() {
    int h_data1[] = {1, 3, 5};
    int h_data2[] = {2, 3, 4};

    // 创建host_vector并复制数据
    thrust::host_vector<int> h_vec1(h_data1, h_data1 + 3);
    thrust::host_vector<int> h_vec2(h_data2, h_data2 + 3);
    thrust::device_vector<int> d_vec1 = h_vec1;
    thrust::device_vector<int> d_vec2 = h_vec2;
    thrust::device_vector<int> d_result(5);

    auto end = thrust::set_union(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_vec2.end(), d_result.begin());

    thrust::host_vector<int> h_result(d_result.begin(), end);
    for (int i = 0; i < h_result.size(); i++) {
        std::cout << "h_result[" << i << "] = " << h_result[i] << std::endl;
    }
}

void set_intersection_example() {
    int h_data1[] = {1, 3, 5};
    int h_data2[] = {2, 3, 4};

    // 创建host_vector并复制数据
    thrust::host_vector<int> h_vec1(h_data1, h_data1 + 3);
    thrust::host_vector<int> h_vec2(h_data2, h_data2 + 3);
    thrust::device_vector<int> d_vec1 = h_vec1;
    thrust::device_vector<int> d_vec2 = h_vec2;
    thrust::device_vector<int> d_result(1);

    auto end = thrust::set_intersection(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_vec2.end(), d_result.begin());

    thrust::host_vector<int> h_result(d_result.begin(), end);
    for (int i = 0; i < h_result.size(); i++) {
        std::cout << "h_result[" << i << "] = " << h_result[i] << std::endl;
    }
}


