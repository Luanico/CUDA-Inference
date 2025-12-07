#include "cpu_operations.h"

/**
 * @brief Prints a matrix to stdout with aligned formatting
 * @param matrix Pointer to matrix data in row-major order
 * @param rows Number of rows
 * @param cols Number of columns
 * @param name Optional name to display above the matrix
 */
void print_matrix(const float* matrix, int rows, int cols, const char* name) {
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
 * @brief Compares two matrices for equality within epsilon tolerance
 * @param Result First matrix to compare
 * @param Expected Second matrix to compare
 * @param rows height
 * @param cols width
 * @param printIfFalse If true, prints both matrices when they differ
 * @param epsilon Maximum allowed difference between elements
 * @return true if matrices are equal within epsilon, false otherwise
 */
bool cpu_matrix_equals(float *Result, float *Expected, ssize_t rows, ssize_t cols, bool printIfFalse, float epsilon){
    bool res = true;
    for (int i = 0; i < rows * cols; i++) {
        if (std::fabs(Result[i] - Expected[i]) > epsilon) {
            printf("Mismatch at index %d: %.8f vs %.8f (diff: %.8e)\n", 
                   i, Result[i], Expected[i], std::fabs(Result[i] - Expected[i]));
            res =  false;
        }
    }

    if (!res && printIfFalse){
        print_matrix(Result, rows, cols, "Result");
        print_matrix(Expected, rows, cols, "Expected");
    }
    return res;
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
 * @brief Performs matrix multiplication on the CPU (Result = A * B)
 * @param A First input matrix (M x N)
 * @param B Second input matrix (N x P)
 * @param Result Output matrix (M x P)
 * @param M Number of rows in A (and Result)
 * @param N Number of columns in A / rows in B (inner dimension)
 * @param P Number of columns in B (and Result)
 * 
 * @note Matrices stored in row-major order
 *       Result[i][j] = sum(A[i][k] * B[k][j]) for k = 0 to N-1
 */
void cpu_matrix_mul(float *A, float *B, float *Result, ssize_t M, ssize_t N, ssize_t P){
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * P + j];
            }
            Result[i * P + j] = sum;
        }
    }
}

/**
 * @brief CPU implementation of ReLU activation
 * @param X Input matrix
 * @param Result Output matrix
 * @param w Width (number of columns)
 * @param h Height (number of rows)
 */
void cpu_matrix_relu(float *X, float *Result, ssize_t w, ssize_t h) {
    for (ssize_t i = 0; i < h * w; i++) {
        Result[i] = X[i] > 0.0f ? X[i] : 0.0f;
    }
}