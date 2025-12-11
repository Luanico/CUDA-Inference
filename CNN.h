#ifndef CNN_H
#define CNN_H

#include <stdlib.h>

#include "error_utils.h"
#include "convolutions.h"
#include "matrix_operations.h"

class CNN{
private:
    convolution_layer *conv1, *conv2;
    maxPool2D_layer pool;
    ReLU_layer relu;
    linear_layer *fc1, *fc2, *fc3;

    size_t input_channels = 3;
    size_t layer2_channels = 6;
    size_t layer3_channels = 16;
    size_t linear1_in_dim = 16 * 8 * 8;
    size_t linear2_in_dim = 120;
    size_t linear3_in_dim = 84;
    size_t output_dim = 10;

public:
    CNN(convolution_layer *conv1_, convolution_layer *conv2_, linear_layer *fc1_, linear_layer *fc2_, linear_layer *fc3_);

    void forward(float *X, float *Result, size_t X_width, size_t X_height, size_t X_channels, size_t Result_dim,
        size_t pitch_X, size_t batch_size);

};

#endif