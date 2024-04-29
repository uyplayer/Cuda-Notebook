#include "error.h"

/*
 * @Author: uyplayer
 * @Date: 2024/4/29 18:32
 * @Email: uyplayer@qq.com
 * @File: defineGridBlock.cu
 * @Software: CLion
 * @Dir: Cuda-Notebook / src/professional-cuda-programming/examples
 * @Project_Name: Cuda-Notebook
 * @Description:
 */





void defineGridBlock(){

    constexpr int nElem = 1024;

    dim3 block(512);
    dim3 grid((nElem + block.x - 1) / block.x);

    // reset block
    block.x = 512;
    grid.x  = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 256;
    grid.x  = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 128;
    grid.x  = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset device before you leave
    CHECK(cudaDeviceReset());


}