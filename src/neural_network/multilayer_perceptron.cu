//
// Created by uyplayer on 2024/5/18.
//

#include "multilayer_perceptron.cuh"
#include <iostream>
#include "rapidcsv.h"
#include "data_loader.cpp"
#include <cmath>
#include <thrust/device_vector.h>


struct sigmoid
{
    __device__ __host__ float operator()(const float& x) const
    {
        return 1.0f / (1.0f + exp(-x));
    }
};



multilayer_perceptron::multilayer_perceptron(std::string file_name, const int num_hidden_layers): num_hidden_layers(num_hidden_layers)
{
    std::cout << "multilayer perceptron constructor" << std::endl;
    data = CSVDataLoader(std::move(file_name)).load_csv();

    num_features = data.GetColumnCount() - 1;

    // hidden layers
    std::vector<int> layers (num_hidden_layers);
    layers[0] = static_cast<int>(num_features);

    for (int i = 1; i <= num_hidden_layers; ++i)
    {
        layers[i] = 128;
    }
    layers[num_hidden_layers + 1] = 1;

    constexpr int hidden_size = 128;
    // intilize weigths and bias
    for (size_t i = 0; i < layers.size(); ++i)
    {
        thrust::device_vector<float> weight(hidden_size);
        float bias = 0.0f;
        weights.push_back(weight);
        biases.push_back(bias);
    }
}