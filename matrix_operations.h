__global__ void matrix_add(float *A, float *B, float *Result, int width, int height, ssize_t pitch_A, ssize_t pitch_B, ssize_t pitch_Result);

__global__ void matrix_mul(float *A, float *B, float *Result, int m, int n, int p, ssize_t pitch_A, ssize_t pitch_B, ssize_t pitch_Result);

__global__ void matrix_feedforward(float *X, float *W,float *B, float *Result, int H_in, int H_out, int batch_size, ssize_t pitch_A, ssize_t pitch_W, ssize_t pitch_Result);

__global__ void matrix_RELU(float *X, float *Result, int H_in, int batch_size, ssize_t pitch_X, ssize_t pitch_Result);