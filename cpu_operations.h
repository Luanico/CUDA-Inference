#include <spdlog/spdlog.h>
#include <cassert>


void print_matrix(const float* matrix, int rows, int cols, const char* name = nullptr);

bool cpu_matrix_equals(float *Result, float *Expected, ssize_t w, ssize_t h, bool printIfFalse = false, float epsilon = 1e-5f);

void cpu_matrix_add(float *A, float *B, float *Result, ssize_t w, ssize_t h);

void cpu_matrix_mul(float *A, float *B, float *Result, ssize_t M, ssize_t N, ssize_t P);

void cpu_matrix_relu(float *X, float *Result, ssize_t w, ssize_t h);