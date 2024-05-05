


#ifndef CUDA_NOTEBOOK_BASE_H
#define CUDA_NOTEBOOK_BASE_H

#include <iostream>

class Base {
public:
    explicit Base(std::string image_path);

    ~Base() = default;

    Base(const Base &base) = delete;

    static void check_opencv_package();

private:
    std::string image_path;


};


#endif //CUDA_NOTEBOOK_BASE_H
