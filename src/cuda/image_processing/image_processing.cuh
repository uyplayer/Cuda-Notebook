



#ifndef CUDA_NOTEBOOK_IMAGE_PROCESSING_CUH
#define CUDA_NOTEBOOK_IMAGE_PROCESSING_CUH

// 滤波器
enum ImageFilter {
    AVERAGE,      // 平均滤波器
    GAUSSIAN,     // 高斯滤波器
    MEDIAN,       // 中值滤波器
    SOBEL,        // Sobel 滤波器
    LAPLACIAN,    // Laplacian 滤波器
    UNSHARP_MASK  // Unsharp Masking 滤波器
};


void image_blur_average();



#endif //CUDA_NOTEBOOK_IMAGE_PROCESSING_CUH
