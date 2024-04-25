# Thread Hierarchy

## Block memory limitation

> There is a limit to the number of threads per block, since all threads of a block are expected to reside on the same streaming multiprocessor core and must share the limited memory resources of that core. On current GPUs, a thread block may contain up to 1024 threads. 




## Thread Block Clusters
> how to understand block clusters ? simple understand that these threads are scheduled on a GPU Processing Cluster (GPC) in the GPU. this mean these threads in cluster are executed on same gpu hardware.


## Memory Hierarchy

> In CUDA programming, threads can access various memory spaces during execution. Each thread has private local memory. Thread blocks share a visible shared memory with the same lifetime as the block. Thread blocks in a cluster can perform operations on each other's shared memory. All threads access the same global memory. Additionally, there are read-only memory spaces: constant and texture memory. These spaces are optimized for different usages. Texture memory provides various addressing modes and data filtering for specific formats. Global, constant, and texture memory persist across kernel launches in the same application.



## Unified Memory for CUDA
[Unified Memory](（https://developer.nvidia.com/blog/unified-memory-cuda-beginners/）)


