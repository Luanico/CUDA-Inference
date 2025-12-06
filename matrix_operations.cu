
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
__global__ void matrix_add(float *A, float *B, float *Result, int width, int height, ssize_t pitch_A, ssize_t pitch_B, ssize_t pitch_Result){
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