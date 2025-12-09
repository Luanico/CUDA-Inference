#include <cuda_runtime.h>
#include "load_onnx.h"
#include "mlp.h"
#include <stdio.h>
#include <chrono>

int INPUT_DIM = 2048;
int HIDDEN_SIZE = 4096;
int OUTPUT_SIZE = 10;
int batch_size = 1024;
char *FILENAME = "mlp_2048_4096_10.onnx";

int main(int argc, char* argv[])
{
    //std::vector<std::vector<float>> weights = getWeightsFromFile(argv[1]);
    std::vector<std::vector<float>> weights = getWeightsFromFile(FILENAME);

    MLP mlp(INPUT_DIM, HIDDEN_SIZE, OUTPUT_SIZE, batch_size);
    mlp.load_weights(weights[0].data(), weights[2].data(), weights[1].data(), weights[3].data(), INPUT_DIM, HIDDEN_SIZE, OUTPUT_SIZE, batch_size);

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

}