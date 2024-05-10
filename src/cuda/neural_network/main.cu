

#include <iostream>
#include "linear_regression.cuh"

int main(){
    std::cout << "Hello neural network !" << std::endl;
    LinearRegression model{"src/cuda/neural_network/datasets/Boston Housing Dataset.csv"};
    model.fit(0.00000001, 1000);
    std::cout << "Weights: " << model.get_weights()[0] << std::endl;
    std::cout << "Bias: " << model.get_bias() << std::endl;
    return 0;
}
