## TODO List

- Test pitch in CNN for linear layers
- convolutions work only for odd sized kernels.
- test convolutions with different paddings and strides
- Test if using inplace operations (for example for relu) is faster
- Test for convolutions is NHWC really is faster than NCHW
- test for convolution optimisation using constant/texture memory for kernels
- test for template based convolutions for unrolling loops ([cf details](./todos_details/unrolling_convolutions.md) )
- experiment with templated for different block sizes instead of just 16x16