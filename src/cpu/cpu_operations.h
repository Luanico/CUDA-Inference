#include <spdlog/spdlog.h>
#include <cassert>


void print_matrix(const float* matrix, int rows, int cols, const char* name = nullptr);

bool cpu_matrix_equals(float *Result, float *Expected, size_t w, size_t h, bool printIfFalse = false, float epsilon = 1e-5f);

void cpu_matrix_add(float *A, float *B, float *Result, size_t w, size_t h);

void cpu_matrix_mul(float *A, float *B, float *Result, size_t M, size_t N, size_t P);

void cpu_matrix_relu(float *X, float *Result, size_t w, size_t h);


void cpu_conv2d(float *input, float *kernel, float *output, 
    int input_height, int input_width,
    int kernel_height, int kernel_width,
    int stride = 1, int padding = 0);


void cpu_max_pool2D(float *input, float *output, 
    int input_height, int input_width, int output_height, int output_width, int kernel_width,
    int stride = 1, int padding = 0);