# Thread Hierarchy

## Block memory limitation

> There is a limit to the number of threads per block, since all threads of a block are expected to reside on the same streaming multiprocessor core and must share the limited memory resources of that core. On current GPUs, a thread block may contain up to 1024 threads. 




## Thread Block Clusters
> how to understand block clusters ? simple understand that these threads are scheduled on same gpu