# 线程索引计算
- threadIdx.x、threadIdx.y、threadIdx.z：当前线程在它的线程块内的索引。
- blockIdx.x、blockIdx.y、blockIdx.z：当前线程块在网格中的索引。
- blockDim.x、blockDim.y、blockDim.z：每个线程块中的线程数量。
- gridDim.x、gridDim.y、gridDim.z：网格的尺寸，以线程块为单位。

## 线程的全局索引计算
### 一维
```cpp
# blockDim.x 线程块x方向的线程数量
# blockIdx.x 当前线程块在网络中的索引
# threadIdx.x 前线程在它的线程块内的索引
int index = threadIdx.x + blockIdx.x * blockDim.x;
```

### 二维
```cpp

const int dataWidth = 10;  // Data array width
const int dataHeight = 4;  // Data array height
float data[dataHeight][dataWidth] = {};

dim3 threadsPerBlock(16, 16); // 每个块 16x16 个线程
dim3 numBlocks((dataWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
(dataHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);
myKernel<<<numBlocks, threadsPerBlock>>>(data);

# global
__global__ void myKernel(float data[][dataWidth]) {
int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < dataWidth && yIndex < dataHeight) {
    // 安全地使用 data[yIndex][xIndex]
    }
}

```

### 三维
```cpp
const int depth = 4;
const int height = 10;
const int width = 10;

dim3 threadsPerBlock(8, 8, 8); 
dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
               (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
               (depth + threadsPerBlock.z - 1) / threadsPerBlock.z);

__global__ void myKernel(float* data, int width, int height, int depth) {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int zIndex = blockIdx.z * blockDim.z + threadIdx.z;

    if (xIndex < width && yIndex < height && zIndex < depth) {
        int index = zIndex * (width * height) + yIndex * width + xIndex;
        // 使用 data[index] 进行操作
    }
}

```