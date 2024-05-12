


#include "gradient_descent.cuh"

GradientDescent::GradientDescent(std::string file_name, float learning_rate, int epochs): learning_rate(learning_rate), epochs(epochs), b(0.0) {
    data = CSVDataLoader(std::move(file_name)).load_csv();
}

GradientDescent::~GradientDescent() = default;

void GradientDescent::update_cpu() {
    auto num_features = data.GetColumnCount() - 1;
    auto num_samples = data.GetRowCount();
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < num_samples; j++) {
            std::vector<float> features;
            for (int k = 0; k < num_features; k++) {
                features.push_back(data.GetCell<float>(k, j));
            }
            auto y = data.GetCell<float>(num_features, j);
            float prediction = predict_cpu(features);
            float error = y - prediction;

            for (int k = 0; k < num_features; k++) {
                // 更新权重
                b += learning_rate * error * features[k] / num_samples;
            }
        }
    }
}

float GradientDescent::predict_cpu(std::vector<float> features) {
    float prediction = 0;
    for (int i = 0; i < features.size(); i++) {
        prediction += features[i] * b;
    }
    return prediction;
}