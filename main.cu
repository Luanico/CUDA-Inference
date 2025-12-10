#include "matrix_operations.h"
#include "cpu_operations.h"
#include "error_utils.h"
#include "convolutions.h"

#include <spdlog/spdlog.h>
#include <cassert>


int main(){

    int im_height = 1920;
    int im_width = 1080;
    int kernel_width = 3;  // Odd size kernel
    
    // Host allocation
    float *input = new float[im_height * im_width];
    float *kernel = new float[kernel_width * kernel_width];
    float *Result = new float[im_height * im_width];

    // Initialize input with random values
    for (int i = 0; i < im_height * im_width; i++) {
        input[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    
    // Initialize kernel (e.g., simple edge detection or blur)
    // Example: 3x3 box blur kernel
    for (int i = 0; i < kernel_width * kernel_width; i++) {
        kernel[i] = 1.0f / (kernel_width * kernel_width);
    }
    
    // Device allocation
    float *dev_input, *dev_kernel, *dev_Result;
    size_t pitch_input, pitch_kernel, pitch_Result;
    
    cudaError_t rc = cudaMallocPitch(&dev_input, &pitch_input, im_width * sizeof(float), im_height), cudaSuccess;
    if(rc)
        abortError("Fail Buffer Allocation");
    rc = cudaMallocPitch(&dev_kernel, &pitch_kernel, kernel_width * sizeof(float), kernel_width), cudaSuccess;
    if(rc)
        abortError("Fail Buffer Allocation");
    rc = cudaMallocPitch(&dev_Result, &pitch_Result, im_width * sizeof(float), im_height), cudaSuccess;
    if(rc)
        abortError("Fail Buffer Allocation");
    
    // Copy to device
    rc = cudaMemcpy2D(dev_input, pitch_input, input, im_width * sizeof(float),
                           im_width * sizeof(float), im_height, cudaMemcpyHostToDevice), cudaSuccess;
    if(rc)
        abortError("Fail Buffer Copy");
    rc = cudaMemcpy2D(dev_kernel, pitch_kernel, kernel, kernel_width * sizeof(float),
                           kernel_width * sizeof(float), kernel_width, cudaMemcpyHostToDevice), cudaSuccess;
    if(rc)
        abortError("Fail Buffer Copy");
    
    // Launch kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid((im_width + dimBlock.x - 1) / dimBlock.x,
                 (im_height + dimBlock.y - 1) / dimBlock.y);

    // Run multiple iterations so kernel time dominates in nsys output
    for (int i = 0; i < 1; i++) {
        call_GPU_naive_convolution(dev_input, dev_Result, dev_kernel, 
                                                 im_width, im_height, kernel_width); //TODO: this behavior changed, cuda parts like allocation in it
    }

    rc = cudaDeviceSynchronize();
    if (rc)
        abortError("Computation Error"); 
    
    // Copy result back
    rc = cudaMemcpy2D(Result, im_width * sizeof(float), dev_Result, pitch_Result,
                           im_width * sizeof(float), im_height, cudaMemcpyDeviceToHost), cudaSuccess;
    if(rc)
        abortError("Fail Buffer Copy");
    
    
    // Cleanup
    cudaFree(dev_input);
    cudaFree(dev_kernel);
    cudaFree(dev_Result);
    delete[] input;
    delete[] kernel;
    delete[] Result;
}