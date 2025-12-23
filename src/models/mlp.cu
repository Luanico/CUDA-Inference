#include "mlp.h"

#include "utils/error_utils.h"
#include "operations/matrix_operations.h"


MLP::MLP(size_t input_dim_, size_t hidden_dim_, size_t output_dim_, size_t batch_size_, bool init_zeros) 
    : input_dim(input_dim_), hidden_dim(hidden_dim_), output_dim(output_dim_), batch_size(batch_size_)
{
    cudaError_t rc = cudaMallocPitch(&W1, &pitch_W1, hidden_dim * sizeof(float), input_dim);
    if(rc)
        abortError("Fail Buffer Allocation");
    rc = cudaMalloc(&B1, hidden_dim * sizeof(float));
    if(rc)
        abortError("Fail Buffer Allocation");


    rc = cudaMallocPitch(&W2, &pitch_W2, output_dim * sizeof(float), hidden_dim);
    if(rc)
        abortError("Fail Buffer Allocation");
    rc = cudaMalloc(&B2, output_dim * sizeof(float));
    if(rc)
        abortError("Fail Buffer Allocation");
    if (init_zeros){
        cudaMemset2D(W1, pitch_W1, 0, hidden_dim * sizeof(float), input_dim);
        cudaMemset(B1, 0, hidden_dim);
        cudaMemset2D(W2, pitch_W2, 0, output_dim * sizeof(float), hidden_dim);
        cudaMemset(B2, 0, output_dim);
    }
}

MLP::~MLP()
{
    cudaFree(W1);
    cudaFree(B1);
    cudaFree(W2);
    cudaFree(B2);
}


void MLP::load_weights(float *W1_, float *W2_, float *B1_, float *B2_, size_t input_dim_, size_t hidden_dim_, size_t output_dim_, size_t batch_size_){
    if (input_dim_ != input_dim)
        abortError("Input dim does not match MLP parameters!");
    if (hidden_dim_ != hidden_dim)
        abortError("Hidden dim does not match MLP parameters!");
    if (output_dim_ != output_dim)
        abortError("Output dim does not match MLP parameters!");
    

    cudaError_t rc = cudaMemcpy2D(W1, pitch_W1, W1_, hidden_dim * sizeof(float), hidden_dim * sizeof(float), input_dim, cudaMemcpyHostToDevice);
    if(rc)
        abortError("Fail Buffer Copy");
    
    rc = cudaMemcpy2D(W2, pitch_W2, W2_, output_dim * sizeof(float), output_dim * sizeof(float), hidden_dim, cudaMemcpyHostToDevice);
    if(rc)
        abortError("Fail Buffer Copy");

    rc = cudaMemcpy(B1, B1_, hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    if(rc)
        abortError("Fail Buffer Copy");
    
    rc = cudaMemcpy(B2, B2_, output_dim * sizeof(float), cudaMemcpyHostToDevice);
    if(rc)
        abortError("Fail Buffer Copy");
} 


void MLP::forward(float *X, float *Result, size_t X_input_size, size_t X_batch_size ,size_t pitch_X, size_t pitch_Result){
    if (X_input_size != input_dim)
        abortError("Input size does not match MLP parameters!");
    if (X_batch_size != batch_size)
        abortError("Batch size does not match MLP parameters!");


    spdlog::info("Forward pass: batch_size={}, input_dim={}, hidden_dim={}, output_dim={}", 
                 batch_size, input_dim, hidden_dim, output_dim);

    //allocate the uncactivated hidden layer
    float *H;
    size_t pitch_H;
    cudaError_t rc = cudaMallocPitch(&H, &pitch_H, hidden_dim, batch_size * sizeof(float));
    if(rc)
        abortError("Fail Buffer Allocation");

    int bsize = 32;
    int w     = std::ceil((float)output_dim / bsize);
    int h     = std::ceil((float)batch_size / bsize);

    spdlog::info("Grid for layer 1: ({},{}) block: ({},{})", w, h, bsize, bsize);
    spdlog::info("Feedforward 1: X->H, H_in={}, H_out={}, batch={}, pitch_X={}, pitch_W1={}, pitch_H={}", 
                 input_dim, hidden_dim, batch_size, pitch_X, pitch_W1, pitch_H);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    //linear1
    matrix_feedforward<<<dimGrid, dimBlock>>>(X, W1, B1, H, input_dim, hidden_dim, batch_size, pitch_X, pitch_W1, pitch_H);

    if (cudaPeekAtLastError())
        abortError("Computation Error"); 
    
    //allocate the activated hidden layer
    float *Z;
    size_t pitch_Z;
    rc = cudaMallocPitch(&Z, &pitch_Z, hidden_dim, batch_size * sizeof(float));
    if(rc)
        abortError("Fail Buffer Allocation");

    //ReLU1
    matrix_RELU<<<dimGrid, dimBlock>>>(H, Z, hidden_dim, batch_size, pitch_H, pitch_Z);

    if (cudaPeekAtLastError())
        abortError("Computation Error"); 
    
    
    //linear2
    matrix_feedforward<<<dimGrid, dimBlock>>>(Z, W2, B2, Result, hidden_dim, output_dim, batch_size, pitch_Z, pitch_W2, pitch_Result);
    
    if (cudaPeekAtLastError())
        abortError("Computation Error"); 
    
    cudaFree(H);
    cudaFree(Z);
}