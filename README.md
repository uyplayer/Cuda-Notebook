# CUDA 进阶学习计划

## 项目简介

本项目提供一个详细的 CUDA 进阶学习计划。通过分阶段的学习任务和实践项目，旨在帮助学习者深入理解并提升 CUDA 编程技能。

## 学习路径

### CUDA基础知识：

学习这些CUDA相关的库时，建议按照以下顺序逐步学习，以确保你建立了良好的基础并逐渐深入：

1. **CUDA**：
   首先，学习CUDA编程基础知识。理解CUDA的并行计算模型、内存管理、线程组织等概念是非常重要的，这将为你后续学习其他CUDA库打下良好的基础。

2. **Thrust**：
   Thrust是一个高性能的并行编程库，可以帮助你在C++中实现并行算法。学习Thrust可以让你更轻松地利用CUDA的并行计算能力，因此建议在学习其他库之前先掌握Thrust。

3. **cuBLAS**：
   cuBLAS是NVIDIA提供的用于基本线性代数运算的库，包括矩阵乘法、向量运算等。学习cuBLAS可以帮助你加速矩阵运算，是深入学习CUDA编程的第一步。

4. **cuSPARSE**：
   cuSPARSE是NVIDIA提供的稀疏矩阵运算库，用于处理稀疏矩阵的乘法、加法等操作。学习cuSPARSE可以帮助你处理大规模的稀疏矩阵计算问题。

5. **cuFFT**：
   cuFFT是NVIDIA提供的用于快速傅里叶变换（FFT）的库。学习cuFFT可以帮助你加速信号处理、图像处理等任务，是很多科学计算和工程应用中的重要组成部分。

6. **cuDNN**：
   cuDNN是NVIDIA提供的用于深度学习的加速库，包括卷积、池化、激活函数等操作。学习cuDNN可以帮助你加速深度学习模型的训练和推理过程。

7. **CUTLASS**：
   CUTLASS是NVIDIA提供的用于深度学习的基础运算库，包括矩阵乘法、卷积等基本操作。学习CUTLASS可以帮助你了解深度学习模型中的基础运算原理，并自定义优化。

8. **cutlass**：
   cutlass是一个面向CUDA的矩阵乘法运算库，其设计旨在提供灵活性和可扩展性。可以作为cuBLAS和CUTLASS之间的桥梁，深入学习cutlass可以帮助你理解矩阵乘法的优化技术。




## 资源

- [CUDA-Programming](https://github.com/brucefan1983/CUDA-Programming/tree/master)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA_by_practice](https://github.com/eegkno/CUDA_by_practice)
- Bruce Fan 的 CUDA 编程资源：[CUDA Programming on GitHub](https://github.com/brucefan1983/CUDA-Programming)
- NVIDIA CUDA
  编程指南中文版（1.1）：[NVIDIA CUDA Programming Guide 1.1 (Chinese)](https://www.nvidia.cn/docs/IO/51635/NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf)
- 北京大学高性能计算中心 CUDA
  教程：[Peking University HPC CUDA Tutorial (Chinese)](https://hpc.pku.edu.cn/docs/20170829223652566150.pdf)
- He Kun 的 CUDA
  编程指南中文翻译：[CUDA Programming Guide in Chinese on GitHub](https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese)
- [CUDA by Example Source Code](https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html)

## 高级资源

- [professional-cuda-c-programming](https://github.com/deeperlearning/professional-cuda-c-programming/tree/master)
- [CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes)


## 配置
```angular2html
- 安装Clion编辑器
- 安装vpkg插件
- 本地安装vcpkg
- 插件配置本地安装vcpkg
```