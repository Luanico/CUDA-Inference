#ifndef MLP_H
#define MLP_H

#include <stdlib.h>

/**
 * @brief A class implementing a one layer MLP, with matrices on the GPU: Input -> hidden -> output
 */
class MLP {
private:
    size_t input_dim, hidden_dim, output_dim, batch_size;
    size_t pitch_W1, pitch_W2;
    float *W1, *B1, *W2, *B2;

public:
    /**
     * @brief Constructor for the MLP class. Allocates buffers
     * @param input_dim_ Input dimension
     * @param hidden_dim_ Hidden layer dimension
     * @param output_dim_ Output dimension
     * @param batch_size_ Batch size
     * @param init_zeros Whether to init matrices with zeros. Slight cost added but easier debug. Default: true
     */
    MLP(size_t input_dim_, size_t hidden_dim_, size_t output_dim_, size_t batch_size_, bool init_zeros = true);
    
    /**
     * @brief Destructor
     */
    ~MLP();

    /**
     * @brief Loads the weights and biases from CPU matrices to the MLP matrices
     * @param W1_ Weight matrix for first layer (input_dim × hidden_dim)
     * @param W2_ Weight matrix for second layer (hidden_dim × output_dim)
     * @param B1_ Bias vector for first layer (hidden_dim)
     * @param B2_ Bias vector for second layer (output_dim)
     * @param input_dim_ Input dimension (for error checking)
     * @param hidden_dim_ Hidden dimension (for error checking)
     * @param output_dim_ Output dimension (for error checking)
     * @param batch_size_ Batch size (for error checking)
     */
    void load_weights(float *W1_, float *W2_, float *B1_, float *B2_, 
                     size_t input_dim_, size_t hidden_dim_, size_t output_dim_, size_t batch_size_);
    
    /**
     * @brief Calculates the forward pass of the MLP and stores in Result
     * @param X Input matrix (batch_size × input_dim)
     * @param Result Output matrix (batch_size × output_dim)
     * @param X_input_size Input size of X (cols) - for error checking
     * @param X_batch_size Batch size of X (rows) - for error checking
     * @param pitch_X Row pitch in bytes for matrix X
     * @param pitch_Result Row pitch in bytes for result matrix
     */
    void forward(float *X, float *Result, size_t X_input_size, size_t X_batch_size, 
                size_t pitch_X, size_t pitch_Result);
};

#endif // MLP_H