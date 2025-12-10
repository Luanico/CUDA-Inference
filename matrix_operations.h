__global__ void matrix_add(float *A, float *B, float *Result, int width, int height, size_t pitch_A, size_t pitch_B, size_t pitch_Result);

__global__ void matrix_add_const(float *A, float val, float *Result, int width, int height, size_t pitch_A, size_t pitch_Result);

__global__ void matrix_mul(float *A, float *B, float *Result, int m, int n, int p, size_t pitch_A, size_t pitch_B, size_t pitch_Result);

__global__ void matrix_feedforward(float *X, float *W,float *B, float *Result, int H_in, int H_out, int batch_size, size_t pitch_X, size_t pitch_W, size_t pitch_Result);

__global__ void matrix_RELU(float *X, float *Result, int H_in, int batch_size, size_t pitch_X, size_t pitch_Result);