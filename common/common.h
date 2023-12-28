//
// Created by uyplayer on 2023/12/28.
//

#ifndef CUDA_NOTEBOOK_COMMON_H
#define CUDA_NOTEBOOK_COMMON_H




#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>


static void HandleError( cudaError_t err,const char *file,int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}



#endif
