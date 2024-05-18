//
// Created by uyplayer on 2024/5/18.
//

#ifndef MULTILAYER_PERCEPTRON_CUH
#define MULTILAYER_PERCEPTRON_CUH

#include <string>
#include "rapidcsv.h"
#include <thrust/device_vector.h>


class multilayer_perceptron
{
public:
    explicit multilayer_perceptron(std::string file_name, int num_layers);
    ~multilayer_perceptron();

    void train();
    void predict();
    void test();
    void save();
    void load();
    void print();
    void summary();
    void compile();
    void fit();
    void evaluate();

private:
    rapidcsv::Document data{};
    size_t num_hidden_layers;
    size_t num_features;
    std::vector<thrust::device_vector<float>> weights;
    std::vector<float> biases;
};


#endif //MULTILAYER_PERCEPTRON_CUH
