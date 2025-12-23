#include <spdlog/spdlog.h>

#include <gtest/gtest.h>
#include <cstdlib>

#include "cpu/cpu_operations.h"
#include "operations/convolutions.h"


TEST(ConvolutionTest, BasicTest3x3) {
    int im_height = 8;
    int im_width = 8;
    int kernel_width = 3;  // Odd size kernel
    
    // Calculate padding to maintain same output size
    int padding = kernel_width / 2;  // For 3x3 kernel, padding = 1
    
    // Host allocation
    float *input = new float[im_height * im_width];
    float *kernel = new float[kernel_width * kernel_width];
    float *Result = new float[im_height * im_width];
    float *Expected = new float[im_height * im_width];
    
    // Initialize input with random values
    for (int i = 0; i < im_height * im_width; i++) {
        input[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    
    // Initialize kernel (e.g., simple edge detection or blur)
    // Example: 3x3 box blur kernel
    for (int i = 0; i < kernel_width * kernel_width; i++) {
        kernel[i] = 1.0f / (kernel_width * kernel_width);
    }
    
    call_GPU_naive_convolution(input, Result, kernel, im_width, im_height, kernel_width);
    
    // CPU computation (stride=1, padding to maintain same size)
    cpu_conv2d(input, kernel, Expected, im_height, im_width, 
               kernel_width, kernel_width, 1, padding);
    
    // Compare
    bool eq = cpu_matrix_equals(Result, Expected, im_width, im_height, true);
    EXPECT_TRUE(eq);

    if (!eq) {
        print_matrix(input, im_height, im_width, "Input");
        print_matrix(kernel, kernel_width, kernel_width, "Kernel");
        print_matrix(Result, im_height, im_width, "Result (GPU)");
        print_matrix(Expected, im_height, im_width, "Expected (CPU)");
    }
    
    delete[] input;
    delete[] kernel;
    delete[] Result;
    delete[] Expected;
}

/*
TEST(ConvolutionTest, EvenTest2x2) {
    int im_height = 8;
    int im_width = 8;
    int kernel_width = 2;  // Even size kernel
    
    // Calculate padding to maintain same output size
    int padding = kernel_width / 2;  // For 3x3 kernel, padding = 1
    
    // Host allocation
    float *input = new float[im_height * im_width];
    float *kernel = new float[kernel_width * kernel_width];
    float *Result = new float[im_height * im_width];
    float *Expected = new float[im_height * im_width];
    
    // Initialize input with random values
    for (int i = 0; i < im_height * im_width; i++) {
        input[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    
    // Initialize kernel (e.g., simple edge detection or blur)
    // Example: 3x3 box blur kernel
    for (int i = 0; i < kernel_width * kernel_width; i++) {
        kernel[i] = 1.0f / (kernel_width * kernel_width);
    }
    
    call_GPU_naive_convolution(input, Result, kernel, im_width, im_height, kernel_width);
    
    // CPU computation (stride=1, padding to maintain same size)
    cpu_conv2d(input, kernel, Expected, im_height, im_width, 
               kernel_width, kernel_width, 1, padding);
    
    // Compare
    bool eq = cpu_matrix_equals(Result, Expected, im_width, im_height, true);
    EXPECT_TRUE(eq);

    if (!eq) {
        print_matrix(input, im_height, im_width, "Input");
        print_matrix(kernel, kernel_width, kernel_width, "Kernel");
        print_matrix(Result, im_height, im_width, "Result (GPU)");
        print_matrix(Expected, im_height, im_width, "Expected (CPU)");
    }
    
    spdlog::info("uuuh");
    delete[] input;
    spdlog::info("uuuh2");
    delete[] kernel;
    spdlog::info("uuuh3");
    delete[] Expected;
    spdlog::info("uuuh4");
    delete[] Result;
    spdlog::info("uuuh5");
}
    */



TEST(PoolingTest, EvenTest2x2) {
    int in_height = 8;
    int in_width = 8;
    int kernel_width = 2;  // Even size kernel
    
    int stride = 1;
    int padding = 0; 
    
    int out_height = (in_height + 2 * padding - kernel_width) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
    
    // Host allocation
    float *input = new float[in_height * in_width];
    float *Result = new float[out_height * out_width];
    float *Expected = new float[out_height * out_width];
    
    // Initialize input with random values
    for (int i = 0; i < in_height * in_width; i++) {
        input[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    
    
    call_GPU_max_pool2D(input, Result, in_width, in_height, out_width, out_height, kernel_width, padding, stride);
    
    
    // CPU computation (stride=1, padding to maintain same size)
    cpu_max_pool2D(input, Expected, in_height, in_width, out_height, out_width,
               kernel_width, 1, padding);
    
    // Compare
    bool eq = cpu_matrix_equals(Result, Expected, out_width, out_height, true);
    EXPECT_TRUE(eq);

    if (eq) {
        print_matrix(input, in_height, in_width, "Input");
        print_matrix(Result, out_height, out_width, "Result (GPU)");
        print_matrix(Expected, out_height, out_width, "Expected (CPU)");
    }
    
    // Cleanup
    delete[] input;
    delete[] Result;
    delete[] Expected;
}