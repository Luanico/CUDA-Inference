#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <vector>
#include <cmath>
#include <random>

#include "CNN.h"
#include "cpu_operations.h"

// =================================================================================
// HELPER: CPU Implementation of the full CNN for verification
// =================================================================================

void cpu_linear_forward(float* input, float* weights, float* biases, float* output,
                        int batch_size, int input_dim, int output_dim) {
    // Zero out output
    memset(output, 0, batch_size * output_dim * sizeof(float));
    
    // Y = X * W + B
    // Weights in code are stored as (Input_Dim x Output_Dim)
    cpu_matrix_mul(input, weights, output, batch_size, input_dim, output_dim);
    
    // Add biases
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < output_dim; o++) {
            output[b * output_dim + o] += biases[o];
        }
    }
}


void cpu_cnn_forward(
    // Inputs
    float* X, 
    // Weights & Biases
    float* k1, float* b1,
    float* k2, float* b2,
    float* w_fc1, float* b_fc1,
    float* w_fc2, float* b_fc2,
    float* w_fc3, float* b_fc3,
    // Output
    float* Result,
    // Dims
    int batch_size, int input_h, int input_w
) {
    // Hardcoded architecture parameters from CNN.h
    int c_in = 3;
    int c_l2 = 6;
    int c_l3 = 16;
    
    // --- Layer 1: Conv 5x5, Pad 2, Stride 1 ---
    // Output size: (20 + 4 - 5) / 1 + 1 = 20
    int h1 = input_h; 
    int w1 = input_w;
    std::vector<float> res_conv1(batch_size * c_l2 * h1 * w1, 0.0f);
    std::vector<float> temp_conv(h1 * w1);

    for (int b = 0; b < batch_size; b++) {
        for (int out_c = 0; out_c < c_l2; out_c++) {
            float* out_ptr = res_conv1.data() + b * c_l2 * h1 * w1 + out_c * h1 * w1;
            
            // Initialize with bias
            for(int i=0; i<h1*w1; i++) out_ptr[i] = b1[out_c];

            for (int in_c = 0; in_c < c_in; in_c++) {
                float* in_ptr = X + b * c_in * input_h * input_w + in_c * input_h * input_w;
                float* k_ptr = k1 + out_c * c_in * 25 + in_c * 25; // 5x5 kernels
                
                cpu_conv2d(in_ptr, k_ptr, temp_conv.data(), input_h, input_w, 5, 5, 1, 2);
                cpu_matrix_add(out_ptr, temp_conv.data(), out_ptr, w1, h1);
            }
        }
    }


    // --- Layer 2: MaxPool 2x2, Stride 2 ---
    // Output size: (20 - 2)/2 + 1 = 10
    int h2 = 10;
    int w2 = 10;
    std::vector<float> res_pool1(batch_size * c_l2 * h2 * w2);
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < c_l2; c++) {
            float* in_ptr = res_conv1.data() + b * c_l2 * h1 * w1 + c * h1 * w1;
            float* out_ptr = res_pool1.data() + b * c_l2 * h2 * w2 + c * h2 * w2;
            cpu_max_pool2D(in_ptr, out_ptr, h1, w1, h2, w2, 2, 2, 0); // Padding 0 for pool
        }
    }

    // --- Layer 3: ReLU ---
    cpu_matrix_relu(res_pool1.data(), res_pool1.data(), res_pool1.size(), 1);


    // --- Layer 4: Conv 5x5, Pad 2, Stride 1 ---
    // Output size: 10
    int h3 = 10;
    int w3 = 10;
    std::vector<float> res_conv2(batch_size * c_l3 * h3 * w3, 0.0f);
    
    for (int b = 0; b < batch_size; b++) {
        for (int out_c = 0; out_c < c_l3; out_c++) {
            float* out_ptr = res_conv2.data() + b * c_l3 * h3 * w3 + out_c * h3 * w3;
             // Initialize with bias
            for(int i=0; i<h3*w3; i++) out_ptr[i] = b2[out_c];

            for (int in_c = 0; in_c < c_l2; in_c++) {
                float* in_ptr = res_pool1.data() + b * c_l2 * h2 * w2 + in_c * h2 * w2;
                float* k_ptr = k2 + out_c * c_l2 * 25 + in_c * 25;
                
                cpu_conv2d(in_ptr, k_ptr, temp_conv.data(), h2, w2, 5, 5, 1, 2);
                cpu_matrix_add(out_ptr, temp_conv.data(), out_ptr, w3, h3);
            }
        }
    }

    // --- Layer 5: MaxPool 2x2, Stride 2 ---
    // Output size: (10 - 2)/2 + 1 = 5
    int h4 = 5;
    int w4 = 5;
    std::vector<float> res_pool2(batch_size * c_l3 * h4 * w4);
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < c_l3; c++) {
            float* in_ptr = res_conv2.data() + b * c_l3 * h3 * w3 + c * h3 * w3;
            float* out_ptr = res_pool2.data() + b * c_l3 * h4 * w4 + c * h4 * w4;
            cpu_max_pool2D(in_ptr, out_ptr, h3, w3, h4, w4, 2, 2, 0);
        }
    }

    // --- Layer 6: ReLU ---
    cpu_matrix_relu(res_pool2.data(), res_pool2.data(), res_pool2.size(), 1);



    // --- Flatten & FC Layers ---
    // res_pool2 is already (Batch, 16*5*5) in memory layout
    int dim_fc1 = 16 * 5 * 5; // 400
    int dim_fc2 = 120;
    int dim_fc3 = 84;
    int dim_out = 10;

    std::vector<float> res_fc1(batch_size * dim_fc2);
    std::vector<float> res_fc2(batch_size * dim_fc3);

    // FC 1
    cpu_linear_forward(res_pool2.data(), w_fc1, b_fc1, res_fc1.data(), batch_size, dim_fc1, dim_fc2);
    cpu_matrix_relu(res_fc1.data(), res_fc1.data(), res_fc1.size(), 1);

    // FC 2
    cpu_linear_forward(res_fc1.data(), w_fc2, b_fc2, res_fc2.data(), batch_size, dim_fc2, dim_fc3);
    cpu_matrix_relu(res_fc2.data(), res_fc2.data(), res_fc2.size(), 1);

    // FC 3
    cpu_linear_forward(res_fc2.data(), w_fc3, b_fc3, Result, batch_size, dim_fc3, dim_out);
    // No ReLU on final output
}


// =================================================================================
// TESTS
// =================================================================================

class CNNTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use 20x20 input so the hardcoded architecture (5x5 final feature map) works
        // (20->20->10->10->5)
        input_width = 20;
        input_height = 20;
        batch_size = 2;
        
        // Architecture Sizes
        in_c = 3; l2_c = 6; l3_c = 16;
        k_size = 5; padding = 2; stride = 1;
        
        lin1_in = 400; lin1_out = 120;
        lin2_out = 84;
        out_dim = 10;
    }

    void InitializeRandom(float* data, size_t size) {
        for(size_t i=0; i<size; i++) data[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    size_t input_width, input_height;
    size_t batch_size;
    size_t in_c, l2_c, l3_c;
    size_t k_size, padding, stride;
    size_t lin1_in, lin1_out, lin2_out, out_dim;
};

TEST_F(CNNTest, ForwardPassIntegrationTest) {
    spdlog::info("Starting CNN Integration Test with Input {}x{}", input_width, input_height);

    // 1. Prepare Host Data
    size_t input_size = batch_size * in_c * input_height * input_width;
    size_t output_size = batch_size * out_dim;

    std::vector<float> h_X(input_size);
    std::vector<float> h_Result(output_size);
    std::vector<float> h_Expected(output_size);
    InitializeRandom(h_X.data(), input_size);

    // Prepare Weights (Host)
    // ... (Your weight initialization code remains exactly the same) ...
    std::vector<float> k1(l2_c * in_c * k_size * k_size);
    std::vector<float> b1(l2_c);
    InitializeRandom(k1.data(), k1.size());
    InitializeRandom(b1.data(), b1.size());

    std::vector<float> k2(l3_c * l2_c * k_size * k_size);
    std::vector<float> b2(l3_c);
    InitializeRandom(k2.data(), k2.size());
    InitializeRandom(b2.data(), b2.size());

    std::vector<float> w_fc1(lin1_in * lin1_out);
    std::vector<float> b_fc1(lin1_out);
    InitializeRandom(w_fc1.data(), w_fc1.size());
    InitializeRandom(b_fc1.data(), b_fc1.size());

    std::vector<float> w_fc2(lin1_out * lin2_out);
    std::vector<float> b_fc2(lin2_out);
    InitializeRandom(w_fc2.data(), w_fc2.size());
    InitializeRandom(b_fc2.data(), b_fc2.size());

    std::vector<float> w_fc3(lin2_out * out_dim);
    std::vector<float> b_fc3(out_dim);
    InitializeRandom(w_fc3.data(), w_fc3.size());
    InitializeRandom(b_fc3.data(), b_fc3.size());


    // 2. Initialize Layers (Stack Allocation)
    // These objects "own" the GPU memory. They will free it when this function exits.
    convolution_layer conv1(in_c, l2_c, padding, stride, k_size, false);
    conv1.load_weights(k1.data(), b1.data(), in_c, l2_c, k_size);

    convolution_layer conv2(l2_c, l3_c, padding, stride, k_size, false);
    conv2.load_weights(k2.data(), b2.data(), l2_c, l3_c, k_size);

    linear_layer fc1(lin1_in, lin1_out, false);
    fc1.load_weights(w_fc1.data(), b_fc1.data(), lin1_in, lin1_out);

    linear_layer fc2(lin1_out, lin2_out, false);
    fc2.load_weights(w_fc2.data(), b_fc2.data(), lin1_out, lin2_out);

    linear_layer fc3(lin2_out, out_dim, false);
    fc3.load_weights(w_fc3.data(), b_fc3.data(), lin2_out, out_dim);

    // 3. Initialize CNN with POINTERS
    // We pass the address (&) of the layers. The CNN class now just holds a reference.
    // This FIXES the double-free bug.
    CNN my_cnn(&conv1, &conv2, &fc1, &fc2, &fc3);

    // 4. Prepare Device Memory
    float *d_X, *d_Result;
    ASSERT_EQ(cudaMalloc(&d_X, input_size * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_Result, output_size * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_X, h_X.data(), input_size * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    // 5. Run CNN Forward (GPU)
    my_cnn.forward(d_X, d_Result, input_width, input_height, in_c, out_dim, input_width*sizeof(float), batch_size);
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // 6. Copy back
    ASSERT_EQ(cudaMemcpy(h_Result.data(), d_Result, output_size * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    // 7. Calculate Expected Result (CPU)
    cpu_cnn_forward(h_X.data(), 
                    k1.data(), b1.data(), 
                    k2.data(), b2.data(),
                    w_fc1.data(), b_fc1.data(),
                    w_fc2.data(), b_fc2.data(),
                    w_fc3.data(), b_fc3.data(),
                    h_Expected.data(),
                    batch_size, input_height, input_width);
    
    // 8. Compare
    bool match = cpu_matrix_equals(h_Result.data(), h_Expected.data(), batch_size, out_dim, true, 1e-3);
    EXPECT_TRUE(match);

    if (!match) {
        spdlog::error("CNN Output Verification Failed");
        print_matrix(h_Result.data(), batch_size, out_dim, "GPU Result");
        print_matrix(h_Expected.data(), batch_size, out_dim, "CPU Expected");
    }

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_Result);
    
    // Destructors for conv1, fc1, etc., run here automatically and free GPU memory.
    // The CNN destructor runs but does not touch the layer memory (since it only holds pointers).
}