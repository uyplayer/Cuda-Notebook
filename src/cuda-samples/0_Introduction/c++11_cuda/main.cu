#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include "error.h"
#include "cxxopts.hpp"

__global__ void countChar(char *text, int numBytes, char target, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numBytes; i += stride) {
        if (text[i] == target) {
            atomicAdd(result, 1);
        }
    }
}


int main(int argc, char **argv) {
    std::cout << "Hello, c++11_cuda!" << std::endl;
    std::cout << "Project root directory: " << PRO_ROOT_DIR << std::endl;

    // 解析命令行参数
    cxxopts::Options options("asyncAPI", "A brief description");
    options.add_options()
            ("d,device", "Device ID", cxxopts::value<int>()->default_value("0"))
            ("t,thrust", "Using thrust", cxxopts::value<int>()->default_value("0"));
    const auto command_result = options.parse(argc, argv);
    int device_id = command_result["device"].as<int>();
    int using_thrust = command_result["thrust"].as<int>();
    std::cout << "Device ID: " << device_id << std::endl;
    std::cout << "Using thrust: " << using_thrust << std::endl;

    // 构建文件路径
    const std::string file_name = "warandpeace.txt";
    std::string project_root_dir(PRO_ROOT_DIR);
    std::string file_path = project_root_dir + "/src/cuda-samples/0_Introduction/c++11_cuda/" + file_name;
    std::cout << "File path: " << file_path << std::endl;

    // 检查文件是否存在
    if (!std::filesystem::exists(file_path)) {
        std::cerr << "Cannot find the input text file. Exiting.." << std::endl;
        return EXIT_FAILURE;
    }

    // 打开文件并读取数据
    std::ifstream file(file_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open file." << std::endl;
        return EXIT_FAILURE;
    }
    file.seekg(0, std::ios::end);
    int numBytes = file.tellg();
    file.seekg(0, std::ios::beg);

    // 分配主机和设备上的内存
    char *h_text;
    HANDLE_ERROR(cudaMallocHost((void **)&h_text, numBytes));
    char *d_text;
    HANDLE_ERROR(cudaMalloc((void **)&d_text, numBytes));

    // 从文件读取数据到主机内存
    file.read(h_text, numBytes);
    file.close();

    // 将数据从主机复制到设备
    HANDLE_ERROR(cudaMemcpy(d_text, h_text, numBytes, cudaMemcpyHostToDevice));

    if (using_thrust) {
        // 使用Thrust进行字符统计
        std::cout << "Using thrust" << std::endl;
        thrust::device_ptr<char> dev_ptr(d_text);
        int numAs = thrust::count(thrust::device, dev_ptr, dev_ptr + numBytes, 'a');
        std::cout << "Number of 'a' in the text: " << numAs << std::endl;
    } else {
        // 使用纯CUDA进行字符统计
        std::cout << "Not using thrust" << std::endl;
        int *d_result;
        HANDLE_ERROR(cudaMalloc((void **)&d_result, sizeof(int)));
        HANDLE_ERROR(cudaMemset(d_result, 0, sizeof(int)));
        int numThreads = 32;
        int numBlocks = 256;
        char targetChar = 'a'; // 统计字符 'a' 的数量
        countChar<<<numBlocks, numThreads>>>(d_text, numBytes, targetChar, d_result);
        int result;
        HANDLE_ERROR(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
        std::cout << "Number of '" << targetChar << "' in the text: " << result << std::endl;
        HANDLE_ERROR(cudaFree(d_result));
    }

    // 释放内存
    HANDLE_ERROR(cudaFreeHost(h_text));
    HANDLE_ERROR(cudaFree(d_text));

    return EXIT_SUCCESS;
}
