#include <spdlog/spdlog.h>

#include <gtest/gtest.h>
#include <cstdlib>

#include "cpu_operations.h"
#include "matrix_operations.h"

TEST(MatrixAdditionTest, BasicTest) {
    int SIZE_X = 16;
    int SIZE_Y = 16;
    
    // Host allocation
    float *A = new float[SIZE_X * SIZE_Y];
    float *B = new float[SIZE_X * SIZE_Y];
    float *Result = new float[SIZE_X * SIZE_Y];
    float *Expected = new float[SIZE_X * SIZE_Y];
    
    // Initialize with random values
    for (int i = 0; i < SIZE_X * SIZE_Y; i++) {
        A[i] = static_cast<float>(rand() % 10);
        B[i] = static_cast<float>(rand() % 10);
    }
    
    // Device allocation
    float *dev_A, *dev_B, *dev_Result;
    size_t pitch_A, pitch_B, pitch_Result;
    
    ASSERT_EQ(cudaMallocPitch(&dev_A, &pitch_A, SIZE_X * sizeof(float), SIZE_Y), cudaSuccess);
    ASSERT_EQ(cudaMallocPitch(&dev_B, &pitch_B, SIZE_X * sizeof(float), SIZE_Y), cudaSuccess);
    ASSERT_EQ(cudaMallocPitch(&dev_Result, &pitch_Result, SIZE_X * sizeof(float), SIZE_Y), cudaSuccess);
    
    // Copy to device
    ASSERT_EQ(cudaMemcpy2D(dev_A, pitch_A, A, SIZE_X * sizeof(float),
                           SIZE_X * sizeof(float), SIZE_Y, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy2D(dev_B, pitch_B, B, SIZE_X * sizeof(float),
                           SIZE_X * sizeof(float), SIZE_Y, cudaMemcpyHostToDevice), cudaSuccess);
    
    // Launch kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid(1, 1);
    matrix_add<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_Result, SIZE_X, SIZE_Y,
                                      pitch_A, pitch_B, pitch_Result);
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // Copy result back
    ASSERT_EQ(cudaMemcpy2D(Result, SIZE_X * sizeof(float), dev_Result, pitch_Result,
                           SIZE_X * sizeof(float), SIZE_Y, cudaMemcpyDeviceToHost), cudaSuccess);
    
    // CPU computation
    cpu_matrix_add(A, B, Expected, SIZE_X, SIZE_Y);
    
    // Compare
    EXPECT_TRUE(cpu_matrix_equals(Result, Expected, SIZE_X, SIZE_Y));
    
    // Cleanup
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_Result);
    delete[] A;
    delete[] B;
    delete[] Result;
    delete[] Expected;
}

TEST(MatrixMultiplicationTest, BasicTest) {
    int SIZE_M = 5;
    int SIZE_N = 3;
    int SIZE_P = 2;
    
    // Host allocation
    float *A = new float[SIZE_M * SIZE_N];
    float *B = new float[SIZE_N * SIZE_P];
    float *Result = new float[SIZE_M * SIZE_P];
    float *Expected = new float[SIZE_M * SIZE_P];
    
    // Initialize with random values
    for (int i = 0; i < SIZE_M * SIZE_N; i++) {
        A[i] = static_cast<float>(rand() % 10);
    }
    for (int i = 0; i < SIZE_N * SIZE_P; i++) {
        B[i] = static_cast<float>(rand() % 10);
    }

    
    // Device allocation
    float *dev_A, *dev_B, *dev_Result;
    size_t pitch_A, pitch_B, pitch_Result;
    
    ASSERT_EQ(cudaMallocPitch(&dev_A, &pitch_A, SIZE_N * sizeof(float), SIZE_M), cudaSuccess); // width: n, height: m
    ASSERT_EQ(cudaMallocPitch(&dev_B, &pitch_B, SIZE_P * sizeof(float), SIZE_N), cudaSuccess); // width: p, height: n
    ASSERT_EQ(cudaMallocPitch(&dev_Result, &pitch_Result, SIZE_P * sizeof(float), SIZE_M), cudaSuccess); //width: p, height: m
    
    // Copy to device
    ASSERT_EQ(cudaMemcpy2D(dev_A, pitch_A, A, SIZE_N * sizeof(float),
                           SIZE_N * sizeof(float), SIZE_M, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy2D(dev_B, pitch_B, B, SIZE_P * sizeof(float),
                           SIZE_P * sizeof(float), SIZE_N, cudaMemcpyHostToDevice), cudaSuccess);
    
    // Launch kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid((SIZE_P + dimBlock.x - 1) / dimBlock.x,
                 (SIZE_M + dimBlock.y - 1) / dimBlock.y);
    matrix_mul<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_Result, SIZE_M, SIZE_N, SIZE_P,
                                      pitch_A, pitch_B, pitch_Result);
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // Copy result back
    ASSERT_EQ(cudaMemcpy2D(Result, SIZE_P * sizeof(float), dev_Result, pitch_Result,
                           SIZE_P * sizeof(float), SIZE_M, cudaMemcpyDeviceToHost), cudaSuccess);
    
    // CPU computation
    cpu_matrix_mul(A, B, Expected, SIZE_M, SIZE_N, SIZE_P);
    
    // Compare
    bool eq = cpu_matrix_equals(Result, Expected, SIZE_P, SIZE_M, true);
    EXPECT_TRUE(eq);

    if (!eq){
        print_matrix(A, SIZE_M, SIZE_N, "A");
        print_matrix(B, SIZE_N, SIZE_P, "B");
    }
    
    // Cleanup
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_Result);
    delete[] A;
    delete[] B;
    delete[] Result;
    delete[] Expected;
}

TEST(FeedforwardTest, BasicTest) {
    int batch_size = 4;
    int H_in = 5;   // Input features
    int H_out = 8;  // Output features
    
    // Host allocation
    float *X = new float[batch_size * H_in];      // Input: batch_size × H_in
    float *W = new float[H_in * H_out];           // Weights: H_in × H_out
    float *B = new float[H_out];                  // Bias: 1 × H_out
    float *Result = new float[batch_size * H_out]; // Output: batch_size × H_out
    float *Expected = new float[batch_size * H_out];
    float *B_broadcasted = new float[batch_size * H_out]; // For CPU comparison
    
    // Initialize with random values
    for (int i = 0; i < batch_size * H_in; i++) {
        X[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    for (int i = 0; i < H_in * H_out; i++) {
        W[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    for (int i = 0; i < H_out; i++) {
        B[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    
    // Broadcast bias for CPU computation
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < H_out; j++) {
            B_broadcasted[i * H_out + j] = B[j];
        }
    }
    
    // Device allocation
    float *dev_X, *dev_W, *dev_B, *dev_Result;
    size_t pitch_X, pitch_W, pitch_Result;
    
    ASSERT_EQ(cudaMallocPitch(&dev_X, &pitch_X, H_in * sizeof(float), batch_size), cudaSuccess);
    ASSERT_EQ(cudaMallocPitch(&dev_W, &pitch_W, H_out * sizeof(float), H_in), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dev_B, H_out * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMallocPitch(&dev_Result, &pitch_Result, H_out * sizeof(float), batch_size), cudaSuccess);
    
    // Copy to device
    ASSERT_EQ(cudaMemcpy2D(dev_X, pitch_X, X, H_in * sizeof(float),
                           H_in * sizeof(float), batch_size, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy2D(dev_W, pitch_W, W, H_out * sizeof(float),
                           H_out * sizeof(float), H_in, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dev_B, B, H_out * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    
    // Launch kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid((H_out + dimBlock.x - 1) / dimBlock.x,
                 (batch_size + dimBlock.y - 1) / dimBlock.y);
    matrix_feedforward<<<dimGrid, dimBlock>>>(dev_X, dev_W, dev_B, dev_Result, 
                                              H_in, H_out, batch_size,
                                              pitch_X, pitch_W, pitch_Result);
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // Copy result back
    ASSERT_EQ(cudaMemcpy2D(Result, H_out * sizeof(float), dev_Result, pitch_Result,
                           H_out * sizeof(float), batch_size, cudaMemcpyDeviceToHost), cudaSuccess);
    
    // CPU computation: Expected = X * W + B (broadcasted)
    cpu_matrix_mul(X, W, Expected, batch_size, H_in, H_out);
    cpu_matrix_add(Expected, B_broadcasted, Expected, H_out, batch_size);
    
    // Compare
    bool eq = cpu_matrix_equals(Result, Expected, H_out, batch_size, true);
    EXPECT_TRUE(eq);

    if (!eq) {
        print_matrix(X, batch_size, H_in, "Input X");
        print_matrix(W, H_in, H_out, "Weights W");
        print_matrix(B, 1, H_out, "Bias B");
        print_matrix(Result, batch_size, H_out, "Result (GPU)");
        print_matrix(Expected, batch_size, H_out, "Expected (CPU)");
    }
    
    // Cleanup
    cudaFree(dev_X);
    cudaFree(dev_W);
    cudaFree(dev_B);
    cudaFree(dev_Result);
    delete[] X;
    delete[] W;
    delete[] B;
    delete[] Result;
    delete[] Expected;
    delete[] B_broadcasted;
}


TEST(ReLUTest, BasicTest) {
    int batch_size = 4;
    int H_in = 8;
    
    // Host allocation
    float *X = new float[batch_size * H_in];
    float *Result = new float[batch_size * H_in];
    float *Expected = new float[batch_size * H_in];
    
    // Initialize with random values including negative numbers
    for (int i = 0; i < batch_size * H_in; i++) {
        X[i] = static_cast<float>(rand() % 20 - 10) / 10.0f; // Range: -1.0 to 1.0
    }
    
    // Device allocation
    float *dev_X, *dev_Result;
    size_t pitch_X, pitch_Result;
    
    ASSERT_EQ(cudaMallocPitch(&dev_X, &pitch_X, H_in * sizeof(float), batch_size), cudaSuccess);
    ASSERT_EQ(cudaMallocPitch(&dev_Result, &pitch_Result, H_in * sizeof(float), batch_size), cudaSuccess);
    
    // Copy to device
    ASSERT_EQ(cudaMemcpy2D(dev_X, pitch_X, X, H_in * sizeof(float),
                           H_in * sizeof(float), batch_size, cudaMemcpyHostToDevice), cudaSuccess);
    
    // Launch kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid((H_in + dimBlock.x - 1) / dimBlock.x,
                 (batch_size + dimBlock.y - 1) / dimBlock.y);
    matrix_RELU<<<dimGrid, dimBlock>>>(dev_X, dev_Result, H_in, batch_size, pitch_X, pitch_Result);
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // Copy result back
    ASSERT_EQ(cudaMemcpy2D(Result, H_in * sizeof(float), dev_Result, pitch_Result,
                           H_in * sizeof(float), batch_size, cudaMemcpyDeviceToHost), cudaSuccess);
    
    // CPU computation
    cpu_matrix_relu(X, Expected, H_in, batch_size);
    
    // Compare
    bool eq = cpu_matrix_equals(Result, Expected, H_in, batch_size, true);
    EXPECT_TRUE(eq);

    if (!eq) {
        print_matrix(X, batch_size, H_in, "Input X");
        print_matrix(Result, batch_size, H_in, "Result (GPU)");
        print_matrix(Expected, batch_size, H_in, "Expected (CPU)");
    }
    
    // Cleanup
    cudaFree(dev_X);
    cudaFree(dev_Result);
    delete[] X;
    delete[] Result;
    delete[] Expected;
}

int main(int argc, char **argv) {
    srand(42);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}