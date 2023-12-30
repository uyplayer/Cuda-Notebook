# CUDA 进阶学习计划

## 项目简介
本项目提供一个详细的 CUDA 进阶学习计划。通过分阶段的学习任务和实践项目，旨在帮助学习者深入理解并提升 CUDA 编程技能。

## 学习路径

### 第1周：核心概念复习
#### 目标
- 深入理解 CUDA 基础概念。

#### 任务
- [ ] **线程模型深入**：
  - 学习线程块、网格详细概念。
  - 探究线程索引计算与多维网格配置。
- [ ] **内存层次结构探索**：
  - 了解不同内存类型（全局、共享、常量、纹理）及其使用场景。
  - 学习 CUDA 内存管理与优化技巧。
- [ ] **同步和并行执行**：
  - 了解 CUDA 中的同步机制。
  - 研究如何管理并行执行的线程。

### 第2周：CUDA工具链与平台
#### 目标
- 熟悉 CUDA 工具链和支持的平台。

#### 任务
- [ ] **CUDA 工具链探索**：
  - 了解 CUDA 编译器、调试和性能分析工具。
- [ ] **平台和版本研究**：
  - 研究 CUDA 在不同操作系统和版本的兼容性。

### 第3周：高级内存管理与性能优化
#### 目标
- 掌握 CUDA 高级内存管理技术和性能优化方法。

#### 任务
- [ ] **高级内存管理**：
  - 学习内存拷贝、映射及优化方法。
- [ ] **性能优化实践**：
  - 学习线程配置与优化。
  - 实践提升 CUDA 程序性能的技巧。

### 第4周：并行算法设计与核心线程
#### 目标
- 学习并行算法设计和深入理解核心和线程的关系。

#### 任务
- [ ] **并行算法学习**：
  - 研究并行算法基本原理（如归约、扫描、排序）。
  - 学习在 CUDA 中实现这些算法的方法。
- [ ] **核心和线程深入**：
  - 理解 GPU 核心架构。
  - 学习优化线程分配和执行的方法。

### 第5周及以后：项目实践
#### 目标
- 应用所学 CUDA 技术于实际项目。

#### 任务
- [ ] **项目选择与规划**：
  - 选择感兴趣的领域（如图像处理、数据分析）。
  - 规划并开始小型 CUDA 项目。
- [ ] **持续学习**：
  - 关注 CUDA 最新开发动态与技术论文。

## 资源
- Bruce Fan 的 CUDA 编程资源：[CUDA Programming on GitHub](https://github.com/brucefan1983/CUDA-Programming)
- NVIDIA CUDA 编程指南中文版（1.1）：[NVIDIA CUDA Programming Guide 1.1 (Chinese)](https://www.nvidia.cn/docs/IO/51635/NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf)
- 北京大学高性能计算中心 CUDA 教程：[Peking University HPC CUDA Tutorial (Chinese)](https://hpc.pku.edu.cn/docs/20170829223652566150.pdf)
- He Kun 的 CUDA 编程指南中文翻译：[CUDA Programming Guide in Chinese on GitHub](https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese)
- [CUDA by Example Source Code](https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-)

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

