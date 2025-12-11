#include <cuda_runtime.h>
#include "load_onnx.h"
#include "CNN.h"
#include "convolutions.h"
#include "matrix_operations.h"
#include <stdio.h>
#include <chrono>

const int INPUT_SIZE = 32;
const int INPUT_CHANNELS = 3;
const int CONV1_OUT_CHANNELS = 6;
const int CONV2_OUT_CHANNELS = 16;
const int FC1_OUT = 120;
const int FC2_OUT = 84;
const int OUTPUT_DIM = 10;
const int BATCH_SIZE = 64;
const char* FILENAME = "cnn_fixed.onnx";

int main(int argc, char* argv[])
{
    std::cout << "Input: " << INPUT_SIZE << "x" << INPUT_SIZE << "x" << INPUT_CHANNELS << std::endl;
    std::cout << "Batch size: " << BATCH_SIZE << std::endl;
    
    std::vector<std::vector<float>> weights = getWeightsFromFile((char*)FILENAME);



    int pooled_size = INPUT_SIZE / 2;
    
    convolution_layer conv1(INPUT_CHANNELS, CONV1_OUT_CHANNELS, 1, 1, 3);
    conv1.load_weights(weights[0].data(), weights[1].data(), INPUT_CHANNELS, CONV1_OUT_CHANNELS, 3);
    
    convolution_layer conv2(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, 1, 1, 3);
    conv2.load_weights(weights[2].data(), weights[3].data(), CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, 3);
    
    linear_layer fc1(CONV2_OUT_CHANNELS * pooled_size * pooled_size, FC1_OUT);
    fc1.load_weights(weights[4].data(), weights[5].data(), CONV2_OUT_CHANNELS * pooled_size * pooled_size, FC1_OUT);
    
    linear_layer fc2(FC1_OUT, FC2_OUT);
    fc2.load_weights(weights[6].data(), weights[7].data(), FC1_OUT, FC2_OUT);
    
    linear_layer fc3(FC2_OUT, OUTPUT_DIM);
    fc3.load_weights(weights[8].data(), weights[9].data(), FC2_OUT, OUTPUT_DIM);
    
    CNN cnn(&conv1, &conv2, &fc1, &fc2, &fc3);

    float *X = new float[BATCH_SIZE * INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE];
    for (size_t i = 0; i < BATCH_SIZE * INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE; i++)
    {
        X[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *dev_X;
    size_t pitch_X;
    cudaMallocPitch(&dev_X, &pitch_X, INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS * sizeof(float), BATCH_SIZE);
    cudaMemcpy2D(dev_X, pitch_X, X, INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS * sizeof(float), 
                 INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS * sizeof(float), BATCH_SIZE, cudaMemcpyHostToDevice);
    
    float *dev_result;
    cudaMalloc(&dev_result, BATCH_SIZE * OUTPUT_DIM * sizeof(float));

    int NUMBER_INFERENCE = 100;

    cnn.forward(dev_X, dev_result, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS, OUTPUT_DIM, pitch_X, BATCH_SIZE);
    cudaDeviceSynchronize();

    auto time_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < NUMBER_INFERENCE; i++)
    {
        cnn.forward(dev_X, dev_result, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS, OUTPUT_DIM, pitch_X, BATCH_SIZE);
    }
    
    cudaDeviceSynchronize();
    auto time_end = std::chrono::high_resolution_clock::now();

    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    float avg_time = static_cast<float>(total_time) / NUMBER_INFERENCE;

    std::cout << "CUDA Average inference time: " << avg_time << " ms" << std::endl;

    cudaFree(dev_X);
    cudaFree(dev_result);
    delete[] X;

    return 0;
}
