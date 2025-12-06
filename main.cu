#include "matrix_operations.h"
#include <spdlog/spdlog.h>
#include <cassert>

const int SIZE_X = 32;
const int SIZE_Y = 32;

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

void print_matrix(const float* matrix, int rows, int cols, const char* name = nullptr) {
    if (name) {
        printf("%s (%d x %d):\n", name, rows, cols);
    }
    
    // Find maximum width needed for formatting
    int max_width = 7; // minimum width
    for (int i = 0; i < rows * cols; i++) {
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%.4f", matrix[i]);
        int len = strlen(buffer);
        if (len > max_width) max_width = len;
    }
    
    // Print matrix
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) {
            printf("%*.4f", max_width, matrix[i * cols + j]);
            if (j < cols - 1) printf(" ");
        }
        printf("]\n");
    }
    printf("\n");
}


void cpu_matrix_add(float *A, float *B, float *Result, ssize_t w, ssize_t h){
    for (size_t i = 0; i < w; i++)
    {
        for (size_t j = 0; j < h; j++)
        {
            Result[i,j] = A[i,j] = B[i,j];
        }
    }
}

bool cpu_matrix_equals(float *A, float *B, ssize_t w, ssize_t h, float epsilon = 1e-5f, bool printIfFalse = false){
    bool res = true;
    for (int i = 0; i < w * h; i++) {
        if (std::fabs(A[i] - B[i]) > epsilon) {
            printf("Mismatch at index %d: %.8f vs %.8f (diff: %.8e)\n", 
                   i, A[i], A[i], std::fabs(A[i] - B[i]));
            res =  false;
        }
    }

    if (res && printIfFalse){
        print_matrix(A, h, w, "A");
        print_matrix(B, h, w, "B");
    }
    return res;
}


int main(){
    ssize_t total_size = SIZE_X * SIZE_Y * sizeof(float);
    
    float *A = (float*) malloc(total_size);
    float *B = (float*) malloc(total_size);
    float *Result = (float*) malloc(total_size);
    float *Expected = (float*) malloc(total_size);

    
    for (size_t i = 0; i < SIZE_X; i++)
    {
        for (size_t j = 0; j < SIZE_Y; j++)
        {
            A[i,j] = rand();
            B[i,j] = rand();
        }
    }

    int bsize = 32;
    int w     = std::ceil((float)SIZE_X / bsize);
    int h     = std::ceil((float)SIZE_Y / bsize);

    spdlog::debug("running kernel of size ({},{})", w, h);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);


    // TODO: check if cudamallocpitch better
    float *dev_A, *dev_B, *dev_Result;
    cudaError_t rc = cudaMalloc(&dev_A, total_size);
    if(rc)
        abortError("Fail Buffer Allocation");
    rc = cudaMalloc(&dev_B, total_size);
    if(rc)
        abortError("Fail Buffer Allocation");
    rc = cudaMalloc(&dev_Result, total_size);
    if(rc)
        abortError("Fail Buffer Allocation");

    rc = cudaMemcpy(dev_A, A, total_size, cudaMemcpyDeviceToHost);
    if(rc)
        abortError("Fail Buffer Copy");
    rc = cudaMemcpy(dev_B, B, total_size, cudaMemcpyDeviceToHost);
    if(rc)
        abortError("Fail Buffer Copy");
     
    matrix_add<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_Result);

    if (cudaPeekAtLastError())
        abortError("Computation Error"); 

    rc = cudaMemcpy(Result, dev_Result, total_size, cudaMemcpyHostToDevice);
    if(rc)
        abortError("Fail Buffer Copy");

    // Check if result is valid
    cpu_matrix_add(A, B, Expected, SIZE_X, SIZE_Y);
    cpu_matrix_equals(A, B, SIZE_X, SIZE_Y);

    
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