


#include "gradient_descent.cuh"


__global__ void gradientDescentKernel(float *d_X, float *d_y, float *d_b, float learning_rate, int num_features, int num_samples);


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
                b += learning_rate * error * features[k] / num_samples;
            }
        }
    }
}

float GradientDescent::predict_cpu(std::vector<float> features) const {
    float prediction = 0;
    for (int i = 0; i < features.size(); i++) {
        prediction += features[i] * b;
    }
    return prediction;
}

void GradientDescent::update_gpu() {

    auto num_features = data.GetColumnCount() - 1;
    auto num_samples = data.GetRowCount();

    auto *X = new float[num_samples * num_features];
    auto *y = new float[num_samples];
    // 初始化特征和标签
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j <= num_features; j++) {
            if (j == num_features) {
                y[i] = data.GetCell<float>(j, i);
                if (y[i] < 0) {
                    std::cerr << "Error: y["<<i<<"] must be greater than 0" << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                X[i * num_features + j] = data.GetCell<float>(j, i);
                if (X[i * num_features + j] < 0) {
                    std::cerr << "Error: [" <<i * num_features + j << "] must be greater than 0" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

        }
    }

    float *d_data;
    float *d_y;
    float *d_b;
    cudaMalloc(&d_data, num_features * num_samples * sizeof(float));
    cudaMalloc(&d_y, num_samples * sizeof(float));
    cudaMalloc(&d_b, sizeof(float));
    cudaMemcpy(d_data, X, num_features * num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(float), cudaMemcpyHostToDevice);

    auto block_size = 256;
    auto num_blocks = (num_samples + block_size - 1) / block_size;



    for (int epoch = 0; epoch < epochs; ++epoch) {
        gradientDescentKernel<<<num_blocks, block_size>>>(d_data, d_y, d_b, learning_rate, (int)num_features, (int)num_samples);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(&b, d_b, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_y);
    cudaFree(d_b);

    delete[] X;
    delete[] y;


}


__global__ void gradientDescentKernel(float *d_X, float *d_y, float *d_b, float learning_rate, int num_features, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        float *features = d_X + idx * num_features;
        float prediction = 0;
        for (int k = 0; k < num_features; k++) {
            prediction += features[k] * (*d_b);
        }
        float error = d_y[idx] - prediction;
        for (int k = 0; k < num_features; k++) {
            atomicAdd(d_b, learning_rate * error * features[k] / num_samples);
        }
    }
}
