#ifndef CUDA_NOTEBOOK_LINEAR_REGRESSION_CUH
#define CUDA_NOTEBOOK_LINEAR_REGRESSION_CUH

#include <string>
#include <random>
#include "rapidcsv.h"

class LinearRegression {
public:
    explicit LinearRegression(std::string file_name);

    void fit(float learning_rate = 1e-8, int epochs = 1000);

    float *predict(float *X_test, size_t num_samples);

    ~LinearRegression();

    [[nodiscard]] float *get_weights() const;

    [[nodiscard]] float get_bias() const;

    int get_num_features() const;


private:
    rapidcsv::Document data;
    float *weights{};
    float bias{};
    float *X{};
    float *y{};
    float *d_X{};
    float *d_y{};
    float *d_weights{};
    float *d_bias{};
    int num_features;
    float *predictions{};
};

#endif //CUDA_NOTEBOOK_LINEAR_REGRESSION_CUH
