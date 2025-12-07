#include <spdlog/spdlog.h>
#include <cassert>


void print_matrix(const float* matrix, int rows, int cols, const char* name = nullptr);

bool cpu_matrix_equals(float *Result, float *Expected, size_t w, size_t h, bool printIfFalse = false, float epsilon = 1e-5f);

void cpu_matrix_add(float *A, float *B, float *Result, size_t w, size_t h);

void cpu_matrix_mul(float *A, float *B, float *Result, size_t M, size_t N, size_t P);

void cpu_matrix_relu(float *X, float *Result, size_t w, size_t h);