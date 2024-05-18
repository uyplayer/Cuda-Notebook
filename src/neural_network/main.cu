

#include <iostream>
#include "linear_regression.cuh"

void run_LinearRegression();

int main() {
    std::cout << "Hello neural network !" << std::endl;
    return 0;
}


void run_LinearRegression() {
    LinearRegression model{"data/Boston Housing Dataset.csv"};
    model.fit(0.00000001, 10000);
    auto num_features = model.get_num_features();
    auto weights = model.get_weights();
    for (int i = 0; i <= num_features; ++i) {
        std::cout << "Weight " << i << ": " << weights[i] << std::endl;
    }
    std::cout << "Bias: " << model.get_bias() << std::endl;
}