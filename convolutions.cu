#include "convolutions.h"


/**
 * @brief CUDA kernel that performs naive 2D convolution on the GPU (internal)
 * @details Performs convolution with stride=1 and zero-padding to maintain input size.
 *          Uses global memory only (no shared memory optimization).
 *          needs to work with both odd and even kernel sizes using floor division for offset.
 * @param in Input image matrix (im_height × im_width, device memory with pitch)
 * @param out Output convolved matrix (im_height × im_width, device memory with pitch)
 * @param kernel Convolution kernel (kernel_width × kernel_width, device memory with pitch)
 * @param im_width Width of input/output image (number of columns)
 * @param im_height Height of input/output image (number of rows)
 * @param kernel_width Width/height of square convolution kernel
 * @param pitch_in Row pitch in bytes for input matrix
 * @param pitch_out Row pitch in bytes for output matrix
 * @param pitch_kernel Row pitch in bytes for kernel matrix
 */

__global__ void naive_convolution(float *in, float *out, float *kernel, size_t im_width, size_t im_height, size_t kernel_width,
                        size_t pitch_in, size_t pitch_out, size_t pitch_kernel){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= im_width || y >= im_height)
        return;
    
    size_t offset = (kernel_width - 1) / 2 ;
        
    float* row_out = (float*)((char*)out + y * pitch_out);
    row_out[x] = 0;
    for (size_t i = 0; i < kernel_width; i++){
        if (y + i < offset || y + i >= offset + im_height)
            continue;
        float* row_in = (float*)((char*)in + (y - offset + i) * pitch_in);
        float* row_kernel = (float*)((char*)kernel + i * pitch_kernel);
        for (size_t j = 0; j < kernel_width; j++){
            float val_in = 0;
            // for padding
            if (x + j >= offset && x + j < offset + im_width)
                val_in = row_in[(x - offset + j)];

            row_out[x] += val_in * row_kernel[j];
        }
    }
}

/**
 * @brief CUDA kernel that performs 2D max pooling on the GPU (internal)
 * @details Reduces spatial dimensions by taking maximum value in each pooling window.
 *          Uses zero-padding for out-of-bounds regions.
 * @param in Input feature map (in_height × in_width, device memory with pitch)
 * @param out Output pooled feature map (out_height × out_width, device memory with pitch)
 * @param in_width Width of input feature map (number of columns)
 * @param in_height Height of input feature map (number of rows)
 * @param out_width Width of output feature map (number of columns)
 * @param out_height Height of output feature map (number of rows)
 * @param kernel_width Width/height of square pooling window
 * @param pitch_in Row pitch in bytes for input matrix
 * @param pitch_out Row pitch in bytes for output matrix
 * @param stride Stride for pooling operation (step size between windows)
 */
__global__ void max_pool2D(float *in, float *out, size_t in_width, size_t in_height, size_t out_width, size_t out_height,size_t kernel_width, 
                            size_t pitch_in, size_t pitch_out, size_t stride){

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= out_width || y >= out_height)
        return;

    float *row_out = (float*)((char*)out + y * pitch_out);
    float max = -INFINITY;
    for (size_t i = 0; i < kernel_width; i++){
        int input_y = y * stride + i;
        
        if (input_y < 0 || input_y >= in_height)
            continue;
        float* row_in = (float*)((char*)in + input_y * pitch_in);
        for (size_t j = 0; j < kernel_width; j++){
            int input_x = x * stride + j;
            float val_in = 0; 
            // for padding
            if (input_x >= 0 && input_x < in_width)
                val_in = row_in[input_x];

            if (val_in > max)
                max = val_in;
        }
    }
    row_out[x] = max;
}


void call_GPU_naive_convolution(const float* input, float* output, const float *kernel, size_t im_width, size_t im_height, size_t kernel_width){
    float *dev_input, *dev_kernel, *dev_Result;
    size_t pitch_input, pitch_kernel, pitch_Result;

    cudaError_t rc = cudaMallocPitch(&dev_input, &pitch_input, im_width * sizeof(float), im_height);
    if(rc)
        abortError("Fail Buffer Allocation");
    rc = cudaMallocPitch(&dev_kernel, &pitch_kernel, kernel_width * sizeof(float), kernel_width);
    if(rc)
        abortError("Fail Buffer Allocation");
    rc = cudaMallocPitch(&dev_Result, &pitch_Result, im_width * sizeof(float), im_height);
    if(rc)
        abortError("Fail Buffer Allocation");
    
    // Copy to device
    rc = cudaMemcpy2D(dev_input, pitch_input, input, im_width * sizeof(float),
                           im_width * sizeof(float), im_height, cudaMemcpyHostToDevice);
    if(rc)
        abortError("Fail Buffer Copy");
    rc = cudaMemcpy2D(dev_kernel, pitch_kernel, kernel, kernel_width * sizeof(float),
                           kernel_width * sizeof(float), kernel_width, cudaMemcpyHostToDevice);
    if(rc)
        abortError("Fail Buffer Copy");
    
    // Launch kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid((im_width + dimBlock.x - 1) / dimBlock.x,
                 (im_height + dimBlock.y - 1) / dimBlock.y);
    naive_convolution<<<dimGrid, dimBlock>>>(dev_input, dev_Result, dev_kernel, 
                                             im_width, im_height, kernel_width,
                                             pitch_input, pitch_Result, pitch_kernel);
    if (rc)
        abortError("Computation Error"); 
    
    rc = cudaDeviceSynchronize();
    
    // Copy result back
    rc = cudaMemcpy2D(output, im_width * sizeof(float), dev_Result, pitch_Result,
                           im_width * sizeof(float), im_height, cudaMemcpyDeviceToHost);
    if(rc)
        abortError("Fail Buffer Copy");

    // Free device memory
    cudaFree(dev_input);
    cudaFree(dev_kernel);
    cudaFree(dev_Result);
}



void call_GPU_max_pool2D(const float* input, float* output, size_t in_width, size_t in_height, size_t out_width, size_t out_height, size_t kernel_width,
                         size_t padding, size_t stride){
    float *dev_input, *dev_Result;
    size_t pitch_input, pitch_Result;
    
    cudaError_t rc = cudaMallocPitch(&dev_input, &pitch_input, in_width * sizeof(float), in_height);
    if(rc)
        abortError("Fail Buffer Allocation");
    rc = cudaMallocPitch(&dev_Result, &pitch_Result, out_width * sizeof(float), out_height);
    if(rc)
        abortError("Fail Buffer Allocation");
    
    // Copy to device
    rc = cudaMemcpy2D(dev_input, pitch_input, input, in_width * sizeof(float),
                           in_width * sizeof(float), in_height, cudaMemcpyHostToDevice);
    if(rc)
        abortError("Fail Buffer Copy");
    
    // Launch kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid((out_width + dimBlock.x - 1) / dimBlock.x,
                 (out_height + dimBlock.y - 1) / dimBlock.y);
    max_pool2D<<<dimGrid, dimBlock>>>(dev_input, dev_Result, 
                                             in_width, in_height, out_width, out_height, kernel_width,
                                             pitch_input, pitch_Result, stride);
    if (rc)
        abortError("Computation Error"); 
    
    rc = cudaDeviceSynchronize();
    
    // Copy result back
    rc = cudaMemcpy2D(output, out_width * sizeof(float), dev_Result, pitch_Result,
                           out_width * sizeof(float), out_height, cudaMemcpyDeviceToHost);
    if(rc)
        abortError("Fail Buffer Copy");

    // Free device memory
    cudaFree(dev_input);
    cudaFree(dev_Result);
}


convolution_layer::convolution_layer(size_t in_channels_, size_t out_channels_, size_t padding_, size_t stride_, size_t kernel_size_, bool init_zeros)
    : in_channels(in_channels_), out_channels(out_channels_), padding(padding_), stride(stride_), kernel_size(kernel_size_) {

    cudaError_t rc = cudaMalloc(&kernels, in_channels * out_channels * kernel_size * kernel_size * sizeof(float));
    if (rc)
        abortError("Fail Buffer Allocation");
    biases = (float*)malloc(out_channels * sizeof(float));
    if (biases == nullptr)
        abortError("Fail Buffer Allocation On CPU");
    
    if (init_zeros){
        rc = cudaMemset(kernels, 0, in_channels * out_channels * kernel_size * kernel_size * sizeof(float));
        if (rc)
            abortError("Fail Buffer Allocation");
        memset(biases, 0, out_channels * sizeof(float));
    }
}

convolution_layer::~convolution_layer(){
    cudaFree(kernels);
    free(biases);
}

void convolution_layer::load_weights(float *kernels_, float *biases_, size_t in_channels_, size_t out_channels_, size_t kernel_size_){
    if (in_channels_ != in_channels)
        abortError("Input channels do not match Layer parameters!");
    if (out_channels_ != out_channels)
        abortError("Output channels do not match Layer parameters!");
    if (kernel_size_ != kernel_size)
        abortError("Kernel size does not match Layer parameters!");
    cudaError_t rc = cudaMemcpy(kernels, kernels_, in_channels * out_channels * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    if (rc)
        abortError("Fail Buffer Copy");
    memcpy(biases, biases_, out_channels * sizeof(float));
    
}

/** Keep in mind: X and Result are 4D, in the NCHW format unlike NVIDIA recommends: 
 * X: batch_size * in_channels * X_height * X_width 
 * Result: batch_size * output channels  * Result_height * Result_width
 * Both are allocated as single array.
 * */
void convolution_layer::forward(float *X, float *Result, size_t X_width, size_t X_height, size_t X_channels, size_t Result_width, size_t Result_height,
                                size_t Result_channels_, size_t batch_size){
    // forward
    //check stuff

    dim3 dimBlock(32, 32);
    dim3 dimGrid((Result_width + dimBlock.x - 1) / dimBlock.x,
                 (Result_height + dimBlock.y - 1) / dimBlock.y);

    
    cudaError_t rc = cudaMemset(Result, 0, batch_size * out_channels * Result_height * Result_width * sizeof(float));
    if (rc)
        abortError("Fail Buffer Memset");

    //allocate a temporary image of shape of result, to store each convolution result
    float *conv_result;
    size_t conv_result_pitch;
    rc = cudaMallocPitch(&conv_result, &conv_result_pitch, Result_width, Result_height);
    if (rc)
        abortError("Fail Buffer Allocation");

    for (size_t N = 0; N < batch_size; N++){
        float *N_Result_start = Result + N * out_channels * Result_height * Result_width;
        float *N_X_start = X + N * in_channels * X_height * X_width;
        for (size_t C = 0; C < out_channels; C++){
            float *C_Result_start = N_Result_start + C * Result_height * Result_width;
            float *C_kernel_start = kernels + C * in_channels * kernel_size * kernel_size;
            for (size_t i = 0; i < in_channels; i++){
                float *C_X_start = N_X_start + i * X_height * X_width;
                float *cur_kernel = C_kernel_start + i * kernel_size * kernel_size;
                naive_convolution<<<dimGrid, dimBlock>>>(C_X_start, conv_result, cur_kernel, X_width, X_height, kernel_size,
                                  X_width * sizeof(float), Result_width * sizeof(float), kernel_size * sizeof(float));
                rc = cudaDeviceSynchronize();
                if (rc)
                    abortError("Computation Error");
                matrix_add<<<dimGrid, dimBlock>>>(C_Result_start, conv_result, C_Result_start, Result_width, Result_height,
                           Result_width * sizeof(float), Result_width * sizeof(float), Result_width * sizeof(float));
                rc = cudaDeviceSynchronize(); // TODO: test if necessary
                if (rc)
                    abortError("Computation Error");
            }
            
            matrix_add_const<<<dimGrid, dimBlock>>>(C_Result_start, biases[C], C_Result_start, Result_width, Result_height,
                                                    Result_width * sizeof(float), Result_width * sizeof(float));
            if (rc)
                abortError("Computation Error");
        }
    }
    
   
   cudaFree(conv_result); 
}

maxPool2D_layer::maxPool2D_layer(size_t stride_, size_t kernel_size_) 
    : stride(stride_), kernel_size(kernel_size_){

}

void maxPool2D_layer::forward(float *X, float *Result, size_t X_width, size_t X_height, size_t X_channels, size_t Result_width, size_t Result_height,
                              size_t batch_size){
    
    //forward

    dim3 dimBlock(32, 32);
    dim3 dimGrid((Result_width + dimBlock.x - 1) / dimBlock.x,
                 (Result_height + dimBlock.y - 1) / dimBlock.y);

    
    cudaError_t rc = cudaMemset(Result, 0, batch_size * X_channels * Result_height * Result_width * sizeof(float));
    if (rc)
        abortError("Fail Buffer Memset");

    for (size_t N = 0; N < batch_size; N++){
        float *N_Result_start = Result + N * X_channels * Result_height * Result_width;
        float *N_X_start = X + N * X_channels * X_height * X_width;
        for (size_t C = 0; C < X_channels; C++){
            float *C_Result_start = N_Result_start + C * Result_height * Result_width;
            float *C_X_start = N_X_start + C * X_height * X_width;
            max_pool2D<<<dimGrid, dimBlock>>>(C_X_start, C_Result_start, X_width, X_height, Result_width, Result_height, kernel_size,
                 X_width * sizeof(float), Result_width * sizeof(float), stride);
            rc = cudaDeviceSynchronize();
            if (rc)
                abortError("Computation Error");
        }
    }
    
}