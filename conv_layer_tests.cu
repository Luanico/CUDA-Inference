#include <gtest/gtest.h>

#include "error_utils.h"
#include "convolutions.h"
#include "cpu_operations.h"

TEST(ConvolutionLayerTest, BasicForwardPass) {
    // Layer parameters
    int batch_size = 2;
    int in_channels = 3;
    int out_channels = 4;
    int kernel_size = 3;
    int padding = 1;  // To maintain size
    int stride = 1;
    
    int input_height = 8;
    int input_width = 8;
    
    // Calculate output dimensions
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    spdlog::info("Input: {}x{}x{}, Output: {}x{}x{}", 
                 in_channels, input_height, input_width,
                 out_channels, output_height, output_width);
    
    // Host allocation for input and output (NCHW format)
    float *X = new float[batch_size * in_channels * input_height * input_width];
    float *Result = new float[batch_size * out_channels * output_height * output_width];
    float *Expected = new float[batch_size * out_channels * output_height * output_width];
    
    // Host allocation for weights and biases
    float *host_kernels = new float[out_channels * in_channels * kernel_size * kernel_size];
    float *host_biases = new float[out_channels];
    
    // Temporary buffers for CPU computation
    float *temp_channel = new float[output_height * output_width];
    
    // Initialize input with random values (NCHW: batch_size * in_channels * height * width)
    for (int i = 0; i < batch_size * in_channels * input_height * input_width; i++) {
        X[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    
    // Initialize kernels with random values
    for (int i = 0; i < out_channels * in_channels * kernel_size * kernel_size; i++) {
        host_kernels[i] = static_cast<float>(rand() % 10) / 10.0f;
        //host_kernels[i] = 1;
    }
    
    // Initialize biases with random values
    for (int i = 0; i < out_channels; i++) {
        host_biases[i] = static_cast<float>(rand() % 10) / 10.0f;
        //host_biases[i] = 0;
    }
    
    // Create convolution layer
    convolution_layer conv_layer(in_channels, out_channels, padding, stride, kernel_size, true);
    
    // Load weights into layer
    conv_layer.load_weights(host_kernels, host_biases, in_channels, out_channels, kernel_size);
    
    // Device allocation for input and output
    float *dev_X, *dev_Result;
    size_t pitch_X, pitch_Result;
    
    // For NCHW, we don't use pitch (flat allocation)
    ASSERT_EQ(cudaMalloc(&dev_X, batch_size * in_channels * input_height * input_width * sizeof(float)), 
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dev_Result, batch_size * out_channels * output_height * output_width * sizeof(float)), 
              cudaSuccess);
    
    // Copy input to device
    ASSERT_EQ(cudaMemcpy(dev_X, X, batch_size * in_channels * input_height * input_width * sizeof(float),
                         cudaMemcpyHostToDevice), cudaSuccess);
    
    // Run forward pass
    conv_layer.forward(dev_X, dev_Result, input_width, input_height, in_channels,
                      output_width, output_height, out_channels,
                      0, 0,  // pitch parameters (unused with flat allocation)
                      batch_size);
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // Copy result back
    ASSERT_EQ(cudaMemcpy(Result, dev_Result, 
                         batch_size * out_channels * output_height * output_width * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    
    // CPU computation: Convolve each input channel with each output channel's kernel
    // NCHW format: [batch][out_channel][height][width]
    for (int n = 0; n < batch_size; n++) {
        for (int out_c = 0; out_c < out_channels; out_c++) {
            // Start with bias
            int out_base = n * (out_channels * output_height * output_width) +
                          out_c * (output_height * output_width);
            
            for (int i = 0; i < output_height * output_width; i++) {
                Expected[out_base + i] = host_biases[out_c];
            }
            
            // Sum contributions from all input channels
            for (int in_c = 0; in_c < in_channels; in_c++) {
                // Get input channel
                int in_base = n * (in_channels * input_height * input_width) +
                             in_c * (input_height * input_width);
                float *input_channel = X + in_base;
                
                // Get kernel for this (out_c, in_c) pair
                int kernel_base = out_c * (in_channels * kernel_size * kernel_size) +
                                 in_c * (kernel_size * kernel_size);
                float *kernel = host_kernels + kernel_base;
                
                // Convolve this input channel with this kernel
                cpu_conv2d(input_channel, kernel, temp_channel,
                          input_height, input_width,
                          kernel_size, kernel_size,
                          stride, padding);
                
                // Add to output
                for (int i = 0; i < output_height * output_width; i++) {
                    Expected[out_base + i] += temp_channel[i];
                }
            }
        }
    }
    
    // Compare with tolerance (convolution has more accumulated error)
    bool eq = cpu_matrix_equals(Result, Expected, 
                               batch_size * out_channels * output_height, output_width, 
                               true, 1e-4);
    EXPECT_TRUE(eq);

    if (!eq) {
        spdlog::error("Convolution test failed!");
        // Print a single channel for debugging
        print_matrix(X + input_height * input_width, input_height, input_width, "Input (second channel)");
        print_matrix(host_kernels + kernel_size * kernel_size, kernel_size, kernel_size, "Kernel (second)");
        print_matrix(Result + output_height * output_width, output_height, output_width, "Result (GPU, second output channel)");
        print_matrix(Expected + output_height * output_width, output_height, output_width, "Expected (CPU, second output channel)");
    }
    
    // Cleanup
    cudaFree(dev_X);
    cudaFree(dev_Result);
    delete[] X;
    delete[] Result;
    delete[] Expected;
    delete[] host_kernels;
    delete[] host_biases;
    delete[] temp_channel;
}