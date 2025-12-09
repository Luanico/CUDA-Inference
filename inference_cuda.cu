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

std::vector<float> transpose(const std::vector<float>& mat, int rows, int cols) {
    std::vector<float> result(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = mat[i * cols + j];
        }
    }
    return result;
}

int main(int argc, char* argv[])
{
    std::cout << "Loading ONNX file: " << FILENAME << std::endl;
    
    //std::vector<std::vector<float>> weights = getWeightsFromFile(argv[1]);
    std::vector<std::vector<float>> weights = getWeightsFromFile(FILENAME);

    std::cout << "Loaded " << weights.size() << " tensors from ONNX file" << std::endl;
    for (size_t i = 0; i < weights.size(); i++) {
        std::cout << "  Tensor " << i << ": " << weights[i].size() << " elements" << std::endl;
    }

    if (weights.size() < 4) {
        std::cerr << "Error: Expected at least 4 tensors (W1, B1, W2, B2), got " << weights.size() << std::endl;
        return 1;
    }

    // Expected sizes
    size_t expected_W1 = (size_t)HIDDEN_SIZE * INPUT_DIM;   // 4096 * 2048
    size_t expected_B1 = (size_t)HIDDEN_SIZE;               // 4096
    size_t expected_W2 = (size_t)OUTPUT_SIZE * HIDDEN_SIZE; // 10 * 4096  
    size_t expected_B2 = (size_t)OUTPUT_SIZE;               // 10

    std::cout << "Expected sizes: W1=" << expected_W1 << ", B1=" << expected_B1 
              << ", W2=" << expected_W2 << ", B2=" << expected_B2 << std::endl;

    std::vector<float> W1_transposed = transpose(weights[0], HIDDEN_SIZE, INPUT_DIM);
    std::vector<float> W2_transposed = transpose(weights[2], OUTPUT_SIZE, HIDDEN_SIZE);

    std::cout << "Creating MLP..." << std::endl;
    MLP mlp(INPUT_DIM, HIDDEN_SIZE, OUTPUT_SIZE, batch_size);
    
    std::cout << "Loading weights into MLP..." << std::endl;
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

}