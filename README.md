# CUDA 学习计划

## 项目简介
此项目旨在提供一个详细的 CUDA 进阶学习计划。通过一系列分阶段的学习任务和实践项目，您将能够深入理解和提高 CUDA 编程技能。

## 学习路径

### 第1周：核心概念复习
#### 目标
- 深入理解 CUDA 的基础概念。

#### 任务
- [ ] **线程模型深入**：
  - 学习线程块、网格的详细概念。
  - 研究线程索引计算和多维网格配置。
- [ ] **内存层次结构探索**：
  - 理解全局内存、共享内存、常量内存和纹理内存的差异和使用场景。
  - 学习 CUDA 中的内存管理和优化技巧。

### 第2周：高级内存管理
#### 目标
- 掌握 CUDA 中的高级内存管理技术。

#### 任务
- [ ] **内存拷贝和映射**：
  - 学习异步内存拷贝的原理和使用方法。
  - 探索显存和主存之间的映射技术。
- [ ] **实践项目**：
  - 实现一个使用不同内存类型（全局、共享、常量内存）的 CUDA 程序。

### 第3周：性能优化
#### 目标
- 理解并实践 CUDA 程序的性能优化技巧。

#### 任务
- [ ] **线程配置和优化**：
  - 学习如何合理配置线程块和网格大小。
  - 掌握线程执行效率和内存访问模式的优化。
- [ ] **实践项目**：
  - 选择一个现有的 CUDA 程序进行性能分析和优化，记录性能改进。

### 第4周：并行算法设计
#### 目标
- 学习并实现常见的并行算法。

#### 任务
- [ ] **并行算法学习**：
  - 研究并行算法的基本原理，如归约、扫描、排序。
  - 学习如何在 CUDA 中实现这些算法。
- [ ] **实践项目**：
  - 实现至少一种并行算法，如并行快速排序或归约算法。

### 第5周及以后：项目实践
#### 目标
- 在实际项目中应用所学 CUDA 技术。

#### 任务
- [ ] **项目选择和规划**：
  - 选择一个感兴趣的领域（如图像处理、数据分析）。
  - 规划并开始一个小型 CUDA 项目。
- [ ] **持续学习**：
  - 跟进 CUDA 的最新开发动态和技术论文。

## 资源
- Bruce Fan 的 CUDA 编程资源：[CUDA Programming on GitHub](https://github.com/brucefan1983/CUDA-Programming)
- NVIDIA CUDA 编程指南中文版（1.1）：[NVIDIA CUDA Programming Guide 1.1 (Chinese)](https://www.nvidia.cn/docs/IO/51635/NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf)
- 北京大学高性能计算中心 CUDA 教程：[Peking University HPC CUDA Tutorial (Chinese)](https://hpc.pku.edu.cn/docs/20170829223652566150.pdf)
- He Kun 的 CUDA 编程指南中文翻译：[CUDA Programming Guide in Chinese on GitHub](https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese)
- [CUDA-by-Example-source-code-for-the-book-s-examples](https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-)


## 配置
### Clion Windows
#### 第一步
```bash 
安装 Visual Studio ， Clion其他配置默认的
```
#### 第二步
```bash
配置Build：点击Edit Configurations 添加target/executable
```