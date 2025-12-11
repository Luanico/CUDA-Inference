#include <cuda_runtime.h>
#include "load_onnx.h"
#include "CNN.h"
#include "convolutions.h"
#include "matrix_operations.h"
#include <stdio.h>
#include <chrono>
#include <map>
#include <tuple>
#include <string>

float benchmark_cnn_architecture(int input_size, int in_channels, int out_channels, int batch_size, const char* FILENAME) {
    std::cout << "\n=== Testing CNN architecture: (" << input_size << "x" << input_size 
              << ", in_channels=" << in_channels << ", out_channels=" << out_channels 
              << ", batch=" << batch_size << ") ===" << std::endl;
    
    std::vector<std::vector<float>> weights = getWeightsFromFile((char*)FILENAME);

    if (weights.size() < 8) {
        std::cerr << "Error: Expected at least 8 tensors (conv1.w, conv1.b, conv2.w, conv2.b, fc1.w, fc1.b, fc2.w, fc2.b, fc3.w, fc3.b), got " 
                  << weights.size() << std::endl;
        return -1.0f;
    }

    int pooled_size = input_size / 2;
    int conv2_out_channels = out_channels * 2;
    
    convolution_layer conv1(in_channels, out_channels, 3, 1);  // 3x3 kernel, padding 1
    conv1.load_weights(weights[0].data(), weights[1].data());
    
    convolution_layer conv2(out_channels, conv2_out_channels, 3, 1);
    conv2.load_weights(weights[2].data(), weights[3].data());
    
    linear_layer fc1(conv2_out_channels * pooled_size * pooled_size, 512, batch_size);
    fc1.load_weights(weights[4].data(), weights[5].data());
    
    linear_layer fc2(512, 256, batch_size);
    fc2.load_weights(weights[6].data(), weights[7].data());
    
    linear_layer fc3(256, 10, batch_size);
    fc3.load_weights(weights[8].data(), weights[9].data());
    
    CNN cnn(&conv1, &conv2, &fc1, &fc2, &fc3);

    float *X = new float[batch_size * in_channels * input_size * input_size];
    for (size_t i = 0; i < batch_size * in_channels * input_size * input_size; i++)
    {
        X[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *dev_X;
    size_t pitch_X;
    cudaMallocPitch(&dev_X, &pitch_X, input_size * input_size * in_channels * sizeof(float), batch_size);
    cudaMemcpy2D(dev_X, pitch_X, X, input_size * input_size * in_channels * sizeof(float), 
                 input_size * input_size * in_channels * sizeof(float), batch_size, cudaMemcpyHostToDevice);
    
    float *dev_result;
    cudaMalloc(&dev_result, batch_size * 10 * sizeof(float));

    int NUMBER_INFERENCE = 100;

    cnn.forward(dev_X, dev_result, input_size, input_size, in_channels, 10, pitch_X, batch_size);
    cudaDeviceSynchronize();

    auto time_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < NUMBER_INFERENCE; i++)
    {
        cnn.forward(dev_X, dev_result, input_size, input_size, in_channels, 10, pitch_X, batch_size);
    }
    
    cudaDeviceSynchronize();
    auto time_end = std::chrono::high_resolution_clock::now();

    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    float avg_time = static_cast<float>(total_time) / NUMBER_INFERENCE;

    std::cout << "CUDA Average inference time: " << avg_time << " ms" << std::endl;

    cudaFree(dev_X);
    cudaFree(dev_result);
    delete[] X;

    return avg_time;
}

int main(int argc, char* argv[])
{
    std::map<std::tuple<int, int, int, int>, const char*> architectures = {
        {{32, 3, 16, 64}, "cnn_small.onnx"},
        {{64, 3, 32, 64}, "cnn_medium.onnx"},
        {{128, 3, 64, 64}, "cnn_large.onnx"},
        {{128, 3, 128, 64}, "cnn_huge.onnx"}
    };

    std::cout << "CUDA CNN Inference Benchmark :" << std::endl;

    for (const auto& [config, filename] : architectures) {
        auto [input_size, in_channels, out_channels, batch_size] = config;
        float avg_time = benchmark_cnn_architecture(input_size, in_channels, out_channels, batch_size, filename);
        
        if (avg_time > 0) {
            std::cout << "Architecture (" << input_size << ", " << in_channels << ", " 
                      << out_channels << ", " << batch_size << "): " << avg_time << " ms" << std::endl;
        }
    }

    return 0;
}
