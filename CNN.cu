#include "CNN.h"
#include "cpu_operations.h"

CNN::CNN(convolution_layer *conv1_, convolution_layer *conv2_, linear_layer *fc1_, linear_layer *fc2_, linear_layer *fc3_)
    : conv1(conv1_), conv2(conv2_), fc1(fc1_), fc2(fc2_), fc3(fc3_), pool(2, 2), relu() {
}


void CNN::forward(float *X, float *Result, size_t X_width, size_t X_height, size_t X_channels, size_t Result_dim,
                  size_t pitch_X, size_t batch_size){
    //Check for error inputs
    if (X_channels != input_channels)
        abortError("Input channels does not match CNN parameters!");
    if (Result_dim != output_dim)
        abortError("Output dim does not match CNN parameters!");
    
    // Allocation of intermediate results
    float *res_conv1, *res_pool1, *res_conv2, *res_pool2, *res_fc1, *res_fc2;
    cudaError_t rc = cudaMalloc(&res_conv1, batch_size * layer2_channels * X_width * X_height * sizeof(float));
    if (rc)
        abortError("Fail Buffer Allocation");
    size_t pool1_out_height = (X_height + 2 * 2 - 5) / 2 + 1;
    size_t pool1_out_width = (X_width + 2 * 2 - 5) / 2 + 1;
    rc = cudaMalloc(&res_pool1, batch_size * layer2_channels * pool1_out_height * pool1_out_width * sizeof(float));
    if (rc)
        abortError("Fail Buffer Allocation");

    rc = cudaMalloc(&res_conv2, batch_size * layer3_channels * pool1_out_height * pool1_out_width * sizeof(float));
    if (rc)
        abortError("Fail Buffer Allocation");
    size_t pool2_out_height = (pool1_out_height + 2 * 2 - 5) / 2 + 1;
    size_t pool2_out_width = (pool1_out_width + 2 * 2 - 5) / 2 + 1;
    rc = cudaMalloc(&res_pool2, batch_size * layer3_channels * pool2_out_height * pool2_out_width * sizeof(float));
    if (rc)
        abortError("Fail Buffer Allocation");
    
    rc = cudaMalloc(&res_fc1, batch_size * linear2_in_dim * sizeof(float));
    if (rc)
        abortError("Fail Buffer Allocation");
    rc = cudaMalloc(&res_fc2, batch_size * linear3_in_dim * sizeof(float));
    if (rc)
        abortError("Fail Buffer Allocation");

    size_t flat_dim = layer2_channels * pool1_out_width * pool1_out_height;
    // Calculate forward
    conv1->forward(X, res_conv1, X_width, X_height, X_channels, X_width, X_height, layer2_channels, batch_size);
    pool.forward(res_conv1, res_pool1, X_width, X_height, layer2_channels, pool1_out_width, pool1_out_height, batch_size);
    relu.forward(res_pool1, res_pool1, layer2_channels * pool1_out_width * pool1_out_height, batch_size, flat_dim * sizeof(float), flat_dim * sizeof(float));


    flat_dim = layer3_channels * pool2_out_width * pool2_out_height;

    conv2->forward(res_pool1, res_conv2, pool1_out_width, pool1_out_height, layer2_channels, pool1_out_width, pool1_out_height, layer3_channels, batch_size);
    pool.forward(res_conv2, res_pool2, pool1_out_width, pool1_out_height, layer3_channels, pool2_out_width, pool2_out_height, batch_size);
    relu.forward(res_pool2, res_pool2, layer3_channels * pool2_out_width * pool2_out_height, batch_size, flat_dim * sizeof(float), flat_dim * sizeof(float));

    
    // flatten except batch but it's already all flat?
    size_t pool2_flat_size = layer3_channels * pool2_out_width * pool2_out_height; // 16 * 5 * 5 = 400
    size_t fc1_out_size = linear2_in_dim; // 120
    size_t fc2_out_size = linear3_in_dim; // 84
    size_t fc3_out_size = output_dim;     // 10

    fc1->forward(res_pool2, res_fc1, layer3_channels * pool2_out_width * pool2_out_height, linear2_in_dim, batch_size,
         pool2_flat_size * sizeof(float),  fc1_out_size * sizeof(float));
    relu.forward(res_fc1, res_fc1, linear2_in_dim, batch_size, fc1_out_size * sizeof(float), fc1_out_size * sizeof(float));

    fc2->forward(res_fc1, res_fc2, linear2_in_dim, linear3_in_dim, batch_size,
        fc1_out_size * sizeof(float), fc2_out_size * sizeof(float));
    relu.forward(res_fc2, res_fc2, linear3_in_dim, batch_size, fc2_out_size * sizeof(float), fc2_out_size * sizeof(float));

    fc3->forward(res_fc2, Result, linear3_in_dim, output_dim, batch_size,
        fc2_out_size * sizeof(float), fc3_out_size * sizeof(float));
}