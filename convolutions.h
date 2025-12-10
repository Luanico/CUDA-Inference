#include "matrix_operations.h"
#include "error_utils.h"


/**
 * @brief Host function to perform naive 2D convolution using GPU
 * @details Allocates device memory, copies data to GPU, launches convolution kernel,
 *          and copies results back to host. Handles all CUDA memory management internally.
 * @param input Input image on host (im_height × im_width, row-major layout)
 * @param output Output convolved image on host (im_height × im_width, row-major layout)
 * @param kernel Convolution kernel on host (kernel_width × kernel_width, row-major layout)
 * @param im_width Width of input/output image
 * @param im_height Height of input/output image
 * @param kernel_width Width/height of square convolution kernel
 */
void call_GPU_naive_convolution(const float* input, float* output, const float *kernel, size_t im_width, size_t im_height, size_t kernel_width);


/**
 * @brief Host function to perform 2D max pooling using GPU
 * @details Allocates device memory, copies data to GPU, launches max pooling kernel,
 *          and copies results back to host. Handles all CUDA memory management internally.
 * @param input Input feature map on host (in_height × in_width, row-major layout)
 * @param output Output pooled feature map on host (out_height × out_width, row-major layout)
 * @param in_width Width of input feature map
 * @param in_height Height of input feature map
 * @param out_width Width of output feature map
 * @param out_height Height of output feature map
 * @param kernel_width Width/height of square pooling window
 * @param padding Padding size (currently unused in implementation)
 * @param stride Stride for pooling operation
 */
void call_GPU_max_pool2D(const float* input, float* output, size_t in_width, size_t in_height, size_t out_width, size_t out_height, size_t kernel_width, 
    size_t padding = 0, size_t stride = 1);


// See https://cs231n.github.io/convolutional-networks/ for illustration
class convolution_layer{
private:
    size_t in_channels, out_channels;
    size_t padding, stride, kernel_size;
    // Images size do not need to be defined, since it can work for any size
    float *kernels; // 1D: we represent the 4D array as a flat array: in_channels kernel per layer, out_channels layers -> in_channels * out_channels * kernel_size **2
    float *biases; // 1D: out_channels biaises (!!! ALLOCATED ON THE CPU)

public:
    convolution_layer(size_t in_channels_, size_t out_channels_, size_t padding_, size_t stride_, size_t kernel_size_, bool init_zeros = true);

    ~convolution_layer();

    void load_weights(float *kernels, float *biases, size_t in_channels_, size_t out_channels_, size_t kernel_size_);

    void forward(float *X, float *Result, size_t X_width, size_t X_height, size_t X_channels, size_t Result_width, size_t Result_height,
                 size_t Result_channels, size_t pitch_X, size_t pitch_Result, size_t batch_size);
};