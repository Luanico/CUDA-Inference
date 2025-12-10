#include <cuda_runtime.h>
#include "load_onnx.h"
#include "mlp.h"
#include <stdio.h>
#include <chrono>
#include <map>
#include <tuple>
#include <string>

std::vector<float> transpose(const std::vector<float>& mat, int rows, int cols) {
    std::vector<float> result(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = mat[i * cols + j];
        }
    }
    return result;
}

float benchmark_architecture(int INPUT_DIM, int HIDDEN_SIZE, int OUTPUT_SIZE, int batch_size, const char* FILENAME) {
    std::cout << "\n=== Testing architecture: (" << INPUT_DIM << ", " << HIDDEN_SIZE << ", " << OUTPUT_SIZE << ", " << batch_size << ") ===" << std::endl;
    
    std::vector<std::vector<float>> weights = getWeightsFromFile((char*)FILENAME);

    if (weights.size() < 4) {
        std::cerr << "Error: Expected at least 4 tensors, got " << weights.size() << std::endl;
        return -1.0f;
    }

    std::vector<float> W1_transposed = transpose(weights[0], HIDDEN_SIZE, INPUT_DIM);
    std::vector<float> W2_transposed = transpose(weights[2], OUTPUT_SIZE, HIDDEN_SIZE);

    MLP mlp(INPUT_DIM, HIDDEN_SIZE, OUTPUT_SIZE, batch_size);
    mlp.load_weights(W1_transposed.data(), W2_transposed.data(), weights[1].data(), weights[3].data(), INPUT_DIM, HIDDEN_SIZE, OUTPUT_SIZE, batch_size);

    float *X = new float[batch_size * INPUT_DIM];
    for (size_t i = 0; i < batch_size * INPUT_DIM; i++)
    {
        X[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *dest;
    size_t pitch_size;
    cudaMallocPitch(&dest, &pitch_size, (size_t)INPUT_DIM * sizeof(float), (size_t)batch_size);
    
    cudaMemcpy2D(dest, pitch_size, X, INPUT_DIM * sizeof(float), INPUT_DIM * sizeof(float), batch_size, cudaMemcpyHostToDevice);
    
    float *dev_result;
    size_t pitch_result;
    cudaMallocPitch(&dev_result, &pitch_result, OUTPUT_SIZE * sizeof(float), batch_size);

    int NUMBER_INFERENCE = 100;

    mlp.forward(dest, dev_result, INPUT_DIM, batch_size, pitch_size, pitch_result);
    cudaDeviceSynchronize();

    auto time_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < NUMBER_INFERENCE; i++)
    {
        mlp.forward(dest, dev_result, INPUT_DIM, batch_size, pitch_size, pitch_result);
    }
    
    cudaDeviceSynchronize();
    auto time_end = std::chrono::high_resolution_clock::now();

    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    float avg_time = static_cast<float>(total_time) / NUMBER_INFERENCE;

    std::cout << "CUDA Average inference time: " << avg_time << " ms" << std::endl;

    cudaFree(dest);
    cudaFree(dev_result);
    delete[] X;

    return avg_time;
}

int main(int argc, char* argv[])
{
    std::map<std::tuple<int, int, int, int>, const char*> architectures = {
        {{512, 1024, 10, 256}, "mlp_small.onnx"},
        {{1024, 2048, 10, 512}, "mlp_medium.onnx"},
        {{2048, 4096, 10, 1024}, "mlp_large.onnx"},
        {{4096, 8192, 10, 2048}, "mlp_huge.onnx"}
    };

    std::cout << "========================================" << std::endl;
    std::cout << "CUDA MLP Inference Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;

    for (const auto& [config, filename] : architectures) {
        auto [input_dim, hidden_size, output_size, batch_size] = config;
        float avg_time = benchmark_architecture(input_dim, hidden_size, output_size, batch_size, filename);
        
        if (avg_time > 0) {
            std::cout << "Architecture (" << input_dim << ", " << hidden_size << ", " 
                      << output_size << ", " << batch_size << "): " << avg_time << " ms" << std::endl;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark completed" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}