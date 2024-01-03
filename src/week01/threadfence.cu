
#include "common.h"

__global__ void updateData(int *data, int value) {
    if (threadIdx.x == 0) {
        // 更新数据
        *data = value;

        // 确保更新对所有线程可见
        __threadfence();
    }
}

__global__ void checkData(int *data) {
    // 等待数据更新
    while (*data == 0);

    // 执行后续操作
    // ...
}
