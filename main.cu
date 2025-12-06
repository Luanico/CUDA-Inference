#include "matrix_operations.h"
#include <spdlog/spdlog.h>
#include <cassert>

const int SIZE_X = 16;
const int SIZE_Y = 16;

/**
 * @brief Aborts execution with a CUDA error message
 * @param msg Error message to display
 * @param fname Function name where error occurred
 * @param line Line number where error occurred
 */
[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

/**
 * @brief Prints a matrix to stdout with aligned formatting
 * @param matrix Pointer to matrix data in row-major order
 * @param rows Number of rows
 * @param cols Number of columns
 * @param name Optional name to display above the matrix
 */
void print_matrix(const float* matrix, int rows, int cols, const char* name = nullptr) {
    if (name) {
        printf("%s (%d x %d):\n", name, rows, cols);
    }
    
    // Find maximum width needed for formatting
    int max_width = 1; // minimum width
    for (int i = 0; i < rows * cols; i++) {
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%.1f", matrix[i]);
        int len = strlen(buffer);
        if (len > max_width) max_width = len;
    }
    
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) {
            printf("%*.1f", max_width, matrix[i * cols + j]);
            if (j < cols - 1) printf(" ");
        }
        printf("]\n");
    }
    printf("\n");
}

/**
 * @brief Performs element-wise matrix addition on the CPU
 * @param A First input matrix
 * @param B Second input matrix
 * @param Result Output matrix (A + B)
 * @param w Matrix width (number of columns)
 * @param h Matrix height (number of rows)
 */
void cpu_matrix_add(float *A, float *B, float *Result, ssize_t w, ssize_t h){
    for (size_t i = 0; i < w * h; i++)
    {
        Result[i] = A[i] + B[i];
    }
}

/**
 * @brief Compares two matrices for equality within epsilon tolerance
 * @param Result First matrix to compare
 * @param Expected Second matrix to compare
 * @param w Matrix width (number of columns)
 * @param h Matrix height (number of rows)
 * @param printIfFalse If true, prints both matrices when they differ
 * @param epsilon Maximum allowed difference between elements
 * @return true if matrices are equal within epsilon, false otherwise
 */
bool cpu_matrix_equals(float *Result, float *Expected, ssize_t w, ssize_t h, bool printIfFalse = false, float epsilon = 1e-5f){
    bool res = true;
    for (int i = 0; i < w * h; i++) {
        if (std::fabs(Result[i] - Expected[i]) > epsilon) {
            printf("Mismatch at index %d: %.8f vs %.8f (diff: %.8e)\n", 
                   i, Result[i], Expected[i], std::fabs(Result[i] - Expected[i]));
            res =  false;
        }
    }

    if (!res && printIfFalse){
        print_matrix(Result, h, w, "Result");
        print_matrix(Expected, h, w, "Expected");
    }
    return res;
}

/**
 * @brief Main function that tests GPU matrix addition against CPU implementation
 * @return Exit code (0 on success, 1 on error)
 */
int main(){
    ssize_t total_size = SIZE_X * SIZE_Y * sizeof(float);
    

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