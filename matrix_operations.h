#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

__global__ void matrix_add(float *A, float *B, float *Result, int width, int height, size_t pitch_A, size_t pitch_B, size_t pitch_Result);

__global__ void matrix_add_const(float *A, float val, float *Result, int width, int height, size_t pitch_A, size_t pitch_Result);

__global__ void matrix_mul(float *A, float *B, float *Result, int m, int n, int p, size_t pitch_A, size_t pitch_B, size_t pitch_Result);

__global__ void matrix_feedforward(float *X, float *W,float *B, float *Result, int H_in, int H_out, int batch_size, size_t pitch_X, size_t pitch_W, size_t pitch_Result);

__global__ void matrix_RELU(float *X, float *Result, int H_in, int batch_size, size_t pitch_X, size_t pitch_Result);

class linear_layer{
    private:
        size_t input_dim, output_dim;
        float *weights;
        float *biases;
        size_t pitch_weights;
    
    public:
        linear_layer(size_t input_dim_, size_t output_dim_, bool init_zeros = true);
    
        ~linear_layer();
    
        void load_weights(float *weights_, float *biases_, size_t input_dim_, size_t output_dim_);
    
        void forward(float *X, float *Result, size_t X_dim, size_t Result_dim, size_t batch_size, size_t pitch_X, size_t pitch_Result);
    };

class ReLU_layer{
    private:
    
    public:
        void forward(float *X, float *Result, size_t X_dim, size_t batch_size, size_t pitch_X, size_t pitch_Result);
    };

#endif