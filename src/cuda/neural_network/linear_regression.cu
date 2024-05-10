


#include <iostream>
#include <utility>
#include "rapidcsv.h"
#include "data_loader.cpp"
#include <random>

class LinearRegression {


public:
    float *weights;
    float bias;
    float *d_weights{};
    float *X{};
    float *y{};
    float *d_X{};
    float *d_y{};
    size_t num_features;

    explicit LinearRegression(std::string file_name) {
        data = CSVDataLoader(std::move(file_name)).load_csv();
        num_features = data.GetColumnCount() - 1;
        weights = new float[num_features];
        bias = 0.0;
        num_features = data.GetColumnCount();
    }


    void train(float learning_rate = 1e-8) {
        size_t num_samples = data.GetRowCount();
        X = new float[num_samples * num_features];
        y = new float[num_samples];

        for (size_t i = 0; i < num_samples; i++) {
            for (size_t j = 0; j < num_features; j++) {
                if (j == num_features - 1) {
                    y[i] = data.GetCell<float>(j, i);
                } else {
                    X[i * num_features + j] = data.GetCell<float>(j, i);
                }
            }
        }

        // device memory
        cudaMalloc(&d_X, num_samples * num_features * sizeof(float));
        cudaMalloc(&d_y, num_samples * sizeof(float));

        cudaMemcpy(d_X, X, num_samples * num_features * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, num_samples * sizeof(float), cudaMemcpyHostToDevice);

        // kernel


        cudaFree(d_X);
        cudaFree(d_y);
    }

    ~LinearRegression() {
        delete[] weights;
        delete[] X;
        delete[] y;
        delete[] d_X;
        delete[] d_y;
        cudaFree(d_weights);
    }


private:
    rapidcsv::Document data;

};