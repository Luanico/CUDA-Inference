#include "matrix_operations.h"
#include "cpu_operations.h"
#include "error_utils.h"

#include <spdlog/spdlog.h>
#include <cassert>

const int SIZE_X = 16;
const int SIZE_Y = 16;


/**
 * @brief Main function that tests GPU matrix addition against CPU implementation
 * @return Exit code (0 on success, 1 on error)
 */
int main(){
    size_t total_size = SIZE_X * SIZE_Y * sizeof(float);
    

    // Host allocation
    float *A = (float*) malloc(total_size);
    float *B = (float*) malloc(total_size);
    float *Result = (float*) malloc(total_size);
    float *Expected = (float*) malloc(total_size);

    // Init A and B
    for (size_t i = 0; i < SIZE_X * SIZE_Y; i++)
    {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    // device value definitions
    int bsize = 32;
    int w     = std::ceil((float)SIZE_X / bsize);
    int h     = std::ceil((float)SIZE_Y / bsize);

    spdlog::debug("running kernel of size ({},{})", w, h);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    // Device allocation
    float *dev_A, *dev_B, *dev_Result;
    size_t pitch_A, pitch_B, pitch_Result;
    cudaError_t rc = cudaMallocPitch(&dev_A, &pitch_A, SIZE_X * sizeof(float), SIZE_Y);
    if(rc)
        abortError("Fail Buffer Allocation");
    rc = cudaMallocPitch(&dev_B, &pitch_B, SIZE_X * sizeof(float), SIZE_Y);
    if(rc)
        abortError("Fail Buffer Allocation");
    rc = cudaMallocPitch(&dev_Result, &pitch_Result, SIZE_X * sizeof(float), SIZE_Y);
    if(rc)
        abortError("Fail Buffer Allocation");

   
    // Copy values from host matrices to device matrices
    rc = cudaMemcpy2D(dev_A, pitch_A, A, SIZE_X * sizeof(float), SIZE_X * sizeof(float), SIZE_Y, cudaMemcpyHostToDevice);
    if(rc)
        abortError("Fail Buffer Copy");
    rc = cudaMemcpy2D(dev_B, pitch_B, B, SIZE_X * sizeof(float), SIZE_X * sizeof(float), SIZE_Y, cudaMemcpyHostToDevice);
    if(rc)
        abortError("Fail Buffer Copy");
    

    // call the kernel that does the addition
    matrix_add<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_Result, SIZE_X, SIZE_Y, pitch_A, pitch_B, pitch_Result);
    
    // Check for an error (it also performs a cudaDeviceSynchronize, to wait for GPU operations to finish)
    if (cudaPeekAtLastError())
        abortError("Computation Error"); 

    // Copy result from device to host
    rc = cudaMemcpy2D(Result, SIZE_X * sizeof(float), dev_Result, pitch_Result, SIZE_X * sizeof(float), SIZE_Y, cudaMemcpyDeviceToHost);
    if(rc)
        abortError("Fail Buffer Copy");

    // Check if result is valid
    cpu_matrix_add(A, B, Expected, SIZE_X, SIZE_Y);
    if (cpu_matrix_equals(Result, Expected, SIZE_X, SIZE_Y)){
        spdlog::info("Matrices addition success!");
    }

    // Free everything nono memory leaks
    rc = cudaFree(dev_A);
    if(rc)
        abortError("Fail Buffer Free");
    rc = cudaFree(dev_B);
    if(rc)
        abortError("Fail Buffer Free");
    rc = cudaFree(dev_Result);
    if(rc)
        abortError("Fail Buffer Free");

    free(A);
    free(B);
    free(Result);
}