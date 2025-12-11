#include "matrix_operations.h"

#include "error_utils.h"
#include <sstream>

/**
 * @brief CUDA kernel that performs element-wise matrix addition on the GPU
 * @param A First input matrix (device memory with pitch)
 * @param B Second input matrix (device memory with pitch)
 * @param Result Output matrix (A + B, device memory with pitch)
 * @param width Matrix width (number of columns)
 * @param height Matrix height (number of rows)
 * @param pitch_A Row pitch in bytes for matrix A
 * @param pitch_B Row pitch in bytes for matrix B
 * @param pitch_Result Row pitch in bytes for result matrix
 */
__global__ void matrix_add(float *A, float *B, float *Result, int width, int height, size_t pitch_A, size_t pitch_B, size_t pitch_Result){
    // Indices in the matrices
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    // can be called with values higher than matrices dimensions because of gpu stuff so we check
    if (x >= width || y >= height)
        return;

    // Cast to char* for byte-level arithmetic, then back to float*
    float* row_A = (float*)((char*)A + y * pitch_A);
    float* row_B = (float*)((char*)B + y * pitch_B);
    float* row_Result = (float*)((char*)Result + y * pitch_Result);
    
    row_Result[x] = row_A[x] + row_B[x];
}

/**
 * @brief CUDA kernel that performs single value matrix addition on the GPU
 * @param A First input matrix (device memory with pitch)
 * @param val value to add
 * @param Result Output matrix (A + val, device memory with pitch)
 * @param width Matrix width (number of columns)
 * @param height Matrix height (number of rows)
 * @param pitch_A Row pitch in bytes for matrix A
 * @param pitch_Result Row pitch in bytes for result matrix
 */
__global__ void matrix_add_const(float *A, float val, float *Result, int width, int height, size_t pitch_A, size_t pitch_Result){
    // Indices in the matrices
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    // can be called with values higher than matrices dimensions because of gpu stuff so we check
    if (x >= width || y >= height)
        return;

    // Cast to char* for byte-level arithmetic, then back to float*
    float* row_A = (float*)((char*)A + y * pitch_A);
    float* row_Result = (float*)((char*)Result + y * pitch_Result);
    
    row_Result[x] = row_A[x] + val;
}

/**
 * @brief CUDA kernel that performs matrix multiplication on the GPU
 * @param A First input matrix : m * n
 * @param B Second input matrix : n * p
 * @param Result Output matrix : m * p
 * @param m 
 * @param n
 * @param p
 * @param pitch_A Row pitch in bytes for matrix A
 * @param pitch_B Row pitch in bytes for matrix B
 * @param pitch_Result Row pitch in bytes for result matrix
 */
__global__ void matrix_mul(float *A, float *B, float *Result, int m, int n, int p, size_t pitch_A, size_t pitch_B, size_t pitch_Result){
     int x = blockDim.x * blockIdx.x + threadIdx.x;
     int y = blockDim.y * blockIdx.y + threadIdx.y;
 
     if (x >= p || y >= m)
         return;
    
    float* row_Result = (float*)((char*)Result + y * pitch_Result);
    float *row_A = (float*)((char*)A + y * pitch_A);

    row_Result[x] = 0;
    for (size_t i = 0; i < n; i++)
    {
        float* row_B = (float*)((char*)B + i * pitch_B);
        row_Result[x] += row_A[i] * row_B[x];
    }
}

/**
 * @brief CUDA kernel that performs feedforward layer on the GPU
 * @param X input matrix : batch_size * H_in
 * @param W weights matrix: H_in, H_out
 * @param B biaises matrix: 1 *  H_out
 * @param Result Output matrix : batch_size * H_out
 * @param H_in
 * @param H_out
 * @param batch_size
 * @param pitch_X Row pitch in bytes for matrix X
 * @param pitch_W Row pitch in bytes for matrix W
 * @param pitch_Result Row pitch in bytes for result matrix
 */
__global__ void matrix_feedforward(float *X, float *W,float *B, float *Result, int H_in, int H_out, int batch_size, size_t pitch_X, size_t pitch_W, size_t pitch_Result){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= H_out || y >= batch_size)
        return;
   
   float* row_Result = (float*)((char*)Result + y * pitch_Result);
   float *row_X = (float*)((char*)X + y * pitch_X);

   row_Result[x] = B[x];
   for (size_t i = 0; i < H_in; i++)
   {
       float* row_W = (float*)((char*)W + i * pitch_W);
       row_Result[x] += row_X[i] * row_W[x];
   }
}


/**
 * @brief CUDA kernel that performs RELU layer on the GPU
 * @param X input matrix : batch_size * H_in
 * @param Result Output matrix : batch_size * H_in
 * @param H_in
 * @param batch_size
 * @param pitch_X Row pitch in bytes for matrix X
 * @param pitch_Result Row pitch in bytes for result matrix
 */
__global__ void matrix_RELU(float *X, float *Result, int H_in, int batch_size, size_t pitch_X, size_t pitch_Result){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= H_in || y >= batch_size)
        return;
   
    float* row_Result = (float*)((char*)Result + y * pitch_Result);
    float *row_X = (float*)((char*)X + y * pitch_X);

    if (row_X[x] > 0.){
        row_Result[x] = row_X[x];
    } else{
        row_Result[x] = 0;
    }
}


linear_layer::linear_layer(size_t input_dim_, size_t output_dim_, bool init_zeros) : input_dim(input_dim_), output_dim(output_dim_) {
    cudaError_t rc = cudaMallocPitch(&weights, &pitch_weights, output_dim * sizeof(float), input_dim);
    if(rc)
        abortError("Fail Buffer Allocation");
    rc = cudaMalloc(&biases, output_dim * sizeof(float));
    if(rc)
        abortError("Fail Buffer Allocation");
    if (init_zeros){
        cudaMemset2D(weights, pitch_weights, 0, output_dim * sizeof(float), input_dim);
        cudaMemset(biases, 0, output_dim);
    }
}
    
linear_layer::~linear_layer(){
    cudaFree(weights);
    cudaFree(biases);
}

void linear_layer::load_weights(float *weights_, float *biases_, size_t input_dim_, size_t output_dim_){
    if (input_dim_ != input_dim){
        std::stringstream ss;
        ss << "Input dim does not match linear layer parameters! Got " << input_dim_ << "; Expected " << input_dim;
        abortError(ss.str().c_str());
    }
    if (output_dim_ != output_dim)
        abortError("Output dim does not match lineaer layer parameters!");
    

    cudaError_t rc = cudaMemcpy2D(weights, pitch_weights, weights_, output_dim * sizeof(float), output_dim * sizeof(float), input_dim, cudaMemcpyHostToDevice);
    if(rc)
        abortError("Fail Buffer Copy");

    rc = cudaMemcpy(biases, biases_, output_dim * sizeof(float), cudaMemcpyHostToDevice);
    if(rc)
        abortError("Fail Buffer Copy");
}

void linear_layer::forward(float *X, float *Result, size_t X_dim, size_t Result_dim, size_t batch_size, size_t pitch_X, size_t pitch_Result){
    if (X_dim != input_dim)
        abortError("Input size does not match linear layer parameters!");

    int bsize = 32;
    int w     = std::ceil((float)output_dim / bsize);
    int h     = std::ceil((float)batch_size / bsize);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    //linear
    matrix_feedforward<<<dimGrid, dimBlock>>>(X, weights, biases, Result, input_dim, output_dim, batch_size, pitch_X, pitch_weights, pitch_Result);

    if (cudaPeekAtLastError())
        abortError("Computation Error"); 
}

void ReLU_layer::forward(float *X, float *Result, size_t X_dim, size_t batch_size, size_t pitch_X, size_t pitch_Result){
    int bsize = 32;
    int w     = std::ceil((float)X_dim / bsize);
    int h     = std::ceil((float)batch_size / bsize);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    matrix_RELU<<<dimGrid, dimBlock>>>(X, Result, X_dim, batch_size, pitch_X, pitch_Result);
}