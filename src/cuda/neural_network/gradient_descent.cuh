


#ifndef CUDA_NOTEBOOK_GRADIENT_DESCENT_CUH
#define CUDA_NOTEBOOK_GRADIENT_DESCENT_CUH


#include "rapidcsv.h"
#include "data_loader.cpp"

class GradientDescent {


public:
    GradientDescent(std::string file_name, float learning_rate, int epochs);


    ~GradientDescent();

    void update_cpu();
    float predict_cpu(std::vector<float> features) const;

    void update_gpu();


private:
    rapidcsv::Document data;
    float b;
    float learning_rate;
    int epochs;


};

#endif //CUDA_NOTEBOOK_GRADIENT_DESCENT_CUH
