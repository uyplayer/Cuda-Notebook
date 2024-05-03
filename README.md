# CUDA 进阶学习计划

## 项目简介

本项目提供一个详细的 CUDA 进阶学习计划。通过分阶段的学习任务和实践项目，旨在帮助学习者深入理解并提升 CUDA 编程技能。

## 学习路径

### CUDA基础知识：
- 首先，确保你对CUDA编程模型有基本的了解，包括线程、块、网格、CUDA内存模型等。可以通过阅读NVIDIA的官方文档和教程来学习这些基础知识。
- cuBLAS和cuSPARSE：
 由于cuBLAS和cuSPARSE是用于执行基本的线性代数计算和稀疏矩阵操作的库，因此它们是学习CUDA编程的一个很好的起点。通过学习这两个库，你可以了解如何在GPU上执行常见的线性代数操作。
- cuFFT：
接下来，学习cuFFT库，了解如何在GPU上执行快速傅里叶变换操作。这对于许多信号处理和图像处理应用非常重要。
- cuDNN：
- 一旦你熟悉了基本的CUDA编程和线性代数操作，可以开始学习cuDNN库，这是用于深度学习任务的重要库。cuDNN提供了针对深度神经网络模型的高性能实现，可以加速训练和推理过程。
- CUTLASS：
CUTLASS是一个高级别的矩阵乘法库，可以提供更灵活和可配置的矩阵乘法实现。学习CUTLASS可以帮助你更好地理解如何优化和定制矩阵乘法操作。
- Thrust：
最后，学习Thrust库，这是一个用于并行算法的高级模板库，可以帮助你更轻松地编写并行化的代码。Thrust提供了许多常见的算法和数据结构，可以在GPU上执行高性能的并行计算。

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

### Clion Windows

#### 第一步

```bash 
安装 Visual Studio ，保持 Clion 的其他配置默认
```

配置 Build：

1. 点击 Clion 中的 Edit Configurations。
2. 添加 target/executable。
3. 确保配置包含 CUDA 相关的选项，如指定正确的 CUDA 工具链、设置合适的编译器和链接器选项。
4. 安装vcpkg
5. 通过vcpkg安装你想要的第三方库
6. 顶级CMakelists里面配置vcpkg
7. 顶级CMakelists里面配置vcpkg里面的include头文件
8. 环境变量配置vcpkg安装的bin目录



