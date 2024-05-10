#include <iostream>
#include <utility>

#include "data_loader.cpp"
#include <random>
#include "linear_regression.cuh"



__global__ void update_weights(float *d_X, float *d_y, float *d_weights, float *d_bias, float learning_rate, int num_samples,
               int num_features);


LinearRegression::LinearRegression(std::string file_name) {
    data = CSVDataLoader(std::move(file_name)).load_csv();
    num_features = data.GetColumnCount() - 1;
    std::cout << data.GetRowCount() << " rows and " << data.GetColumnCount() << " columns" << std::endl;

    auto vec = data.GetColumnNames();
    for (const auto& str : vec) {
        std::cout << "ColumnName >> " << str << std::endl;
    }
    std::cout << "Number of features: " << num_features << std::endl;
}

void LinearRegression::fit(float learning_rate, int epochs) {
    int num_samples = data.GetRowCount();
    std::cout << "Number of samples: " << num_samples << std::endl;
    X = new float[num_samples * num_features];
    y = new float[num_samples];
    weights = new float[num_features];
    bias = 0.0;

    // 随机初始化权重
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (size_t i = 0; i < num_features; ++i) {
        weights[i] = distribution(generator);
        std::cout << "Weight " << i << ": " << weights[i] << std::endl;
    }

    // 初始化特征和标签
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < num_features; j++) {
            if (j == num_features - 1) {
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

    // 设备上分配内存
    cudaMalloc(&d_X, num_samples * num_features * sizeof(float));
    cudaMalloc(&d_y, num_samples * sizeof(float));
    cudaMalloc(&d_weights, num_features * sizeof(float));
    cudaMalloc(&d_bias, sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_X, X, num_samples * num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, &bias, sizeof(float), cudaMemcpyHostToDevice);

    // 训练模型
    int block_size = 256;
    int num_blocks = (num_samples + block_size - 1) / block_size;
    std::cout << " epochs : " << epochs << block_size << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        update_weights<<<num_blocks, block_size>>>(d_X, d_y, d_weights, d_bias, learning_rate, num_samples,
                                                           num_features);
        // 同步核函数
        cudaDeviceSynchronize();
    }
    // 将数据从设备复制到主机
    cudaMemcpy(weights, d_weights, num_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost);

    //  释放设备内存
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_weights);
    cudaFree(d_bias);

    delete[] X;
    delete[] y;


    std::cout << "Training completed" << std::endl;

}

float *LinearRegression::predict(float *X_test, size_t num_samples) {
    float *predictions = new float[num_samples];

    for (size_t i = 0; i < num_samples; ++i) {
        predictions[i] = 0.0;
        for (size_t j = 0; j < num_features; ++j) {
            predictions[i] += X_test[i * num_features + j] * weights[j];
        }
        predictions[i] += bias;
    }

    return predictions;
}

LinearRegression::~LinearRegression() {
    delete[] weights;
}

float *LinearRegression::get_weights() const {
    return weights;
}

float LinearRegression::get_bias() const {
    return bias;
}



__global__ void update_weights(float *d_X, float *d_y, float *d_weights, float *d_bias, float learning_rate, int num_samples,
               int num_features) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
//        printf("idx = %d\n", idx);
        float y_pred = 0.0;
        for (int i = 0; i < num_features; ++i) {
            printf("d_X[%d] = %f\n", idx * num_features + i, d_X[idx * num_features + i]);
            y_pred += d_X[idx * num_features + i] * d_weights[i];
            printf("y_pred = %f\n", y_pred);

        }
        y_pred += *d_bias;
        // 计算误差
        float error = y_pred - d_y[idx];
        for (int i = 0; i < num_features; ++i) {
            atomicAdd(&d_weights[i], -learning_rate * error * d_X[idx * num_features + i]);
        }
        atomicAdd(d_bias, -learning_rate * error);
    }

}