#include <spdlog/spdlog.h>
#include <gtest/gtest.h>
#include <cstdlib>

#include "models/mlp.h"
#include "cpu/cpu_operations.h"
#include "operations/matrix_operations.h"

TEST(MLPTest, BasicForwardPass) {
    int batch_size = 4;
    int input_dim = 5;
    int hidden_dim = 8;
    int output_dim = 3;
    
    // Host allocation for inputs and outputs
    float *X = new float[batch_size * input_dim];
    float *Result = new float[batch_size * output_dim];
    float *Expected = new float[batch_size * output_dim];
    
    // Host allocation for weights and biases
    float *W1 = new float[input_dim * hidden_dim];
    float *W2 = new float[hidden_dim * output_dim];
    float *B1 = new float[hidden_dim];
    float *B2 = new float[output_dim];
    
    // Temporary buffers for CPU computation
    float *H_cpu = new float[batch_size * hidden_dim];
    float *B1_broadcasted = new float[batch_size * hidden_dim];
    float *B2_broadcasted = new float[batch_size * output_dim];
    
    // Initialize input with random values
    for (int i = 0; i < batch_size * input_dim; i++) {
        X[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    
    // Initialize weights and biases with random values
    for (int i = 0; i < input_dim * hidden_dim; i++) {
        W1[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    for (int i = 0; i < hidden_dim * output_dim; i++) {
        W2[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    for (int i = 0; i < hidden_dim; i++) {
        B1[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    for (int i = 0; i < output_dim; i++) {
        B2[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    
    // Broadcast biases for CPU computation
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            B1_broadcasted[i * hidden_dim + j] = B1[j];
        }
    }
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < output_dim; j++) {
            B2_broadcasted[i * output_dim + j] = B2[j];
        }
    }
    
    // Create MLP instance
    MLP mlp(input_dim, hidden_dim, output_dim, batch_size, true);
    
    // Load weights into MLP
    mlp.load_weights(W1, W2, B1, B2, input_dim, hidden_dim, output_dim, batch_size);
    
    // Device allocation for input and output
    float *dev_X, *dev_Result;
    size_t pitch_X, pitch_Result;
    
    ASSERT_EQ(cudaMallocPitch(&dev_X, &pitch_X, input_dim * sizeof(float), batch_size), cudaSuccess);
    ASSERT_EQ(cudaMallocPitch(&dev_Result, &pitch_Result, output_dim * sizeof(float), batch_size), cudaSuccess);
    
    // Copy input to device
    ASSERT_EQ(cudaMemcpy2D(dev_X, pitch_X, X, input_dim * sizeof(float),
                           input_dim * sizeof(float), batch_size, cudaMemcpyHostToDevice), cudaSuccess);
    
    // Run forward pass
    mlp.forward(dev_X, dev_Result, input_dim, batch_size, pitch_X, pitch_Result);
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // Copy result back
    ASSERT_EQ(cudaMemcpy2D(Result, output_dim * sizeof(float), dev_Result, pitch_Result,
                           output_dim * sizeof(float), batch_size, cudaMemcpyDeviceToHost), cudaSuccess);
    
    // CPU computation: Layer 1: H = ReLU(X * W1 + B1)
    cpu_matrix_mul(X, W1, H_cpu, batch_size, input_dim, hidden_dim);
    cpu_matrix_add(H_cpu, B1_broadcasted, H_cpu, hidden_dim, batch_size);
    cpu_matrix_relu(H_cpu, H_cpu, hidden_dim, batch_size);
    
    // CPU computation: Layer 2: Y = H * W2 + B2
    cpu_matrix_mul(H_cpu, W2, Expected, batch_size, hidden_dim, output_dim);
    cpu_matrix_add(Expected, B2_broadcasted, Expected, output_dim, batch_size);
    
    // Compare
    bool eq = cpu_matrix_equals(Result, Expected, output_dim, batch_size, true);
    EXPECT_TRUE(eq);

    if (!eq) {
        print_matrix(X, batch_size, input_dim, "Input X");
        print_matrix(Result, batch_size, output_dim, "Result (GPU)");
        print_matrix(Expected, batch_size, output_dim, "Expected (CPU)");
    }
    
    // Cleanup
    cudaFree(dev_X);
    cudaFree(dev_Result);
    delete[] X;
    delete[] Result;
    delete[] Expected;
    delete[] W1;
    delete[] W2;
    delete[] B1;
    delete[] B2;
    delete[] H_cpu;
    delete[] B1_broadcasted;
    delete[] B2_broadcasted;
}

int main(int argc, char **argv) {
    srand(42);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}