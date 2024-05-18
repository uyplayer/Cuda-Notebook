# CUDA 进阶学习计划

## CUDA相关库学习路径

### 1. CUDA
首先，学习CUDA编程基础知识。理解CUDA的并行计算模型、内存管理、线程组织等概念是非常重要的，这将为你后续学习其他CUDA库打下良好的基础。

### 2. Thrust
在掌握CUDA基础后，学习Thrust库。Thrust提供了高层次的并行编程接口，使得你可以更方便地编写并行算法。Thrust类似于C++标准模板库（STL），但专门优化用于GPU计算。

### 3. cuBLAS
学习cuBLAS库，用于加速基本线性代数运算，如矩阵乘法、向量运算等。cuBLAS是很多高性能计算任务的基础。

### 4. cuSPARSE
了解cuSPARSE库，用于稀疏矩阵运算。对于需要处理大规模稀疏矩阵的应用，cuSPARSE是一个非常重要的工具。

### 5. cuFFT
学习cuFFT库，用于快速傅里叶变换（FFT）。FFT在信号处理和图像处理等领域有广泛的应用，cuFFT可以极大地加速这些计算。

### 6. cuDNN
掌握cuDNN库，用于深度学习的加速。cuDNN包括卷积、池化、激活函数等操作，是深度学习框架（如TensorFlow和PyTorch）高效运行的基础。

### 7. CUTLASS
学习CUTLASS库，它提供了高效的矩阵乘法和卷积操作的实现。CUTLASS可以帮助你了解深度学习基础运算的原理，并进行自定义优化。

### 8. TensorRT
学习TensorRT，用于深度学习模型的优化和部署。TensorRT可以极大地加速推理过程，是实际应用中非常重要的工具。

### 9. Libtorch
最后，学习Libtorch（PyTorch的C++库）。Libtorch允许你在C++中使用PyTorch的功能，对于需要高性能和灵活性的应用非常有用。


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