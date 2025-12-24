# Cuda Inference Tool

## Overview

This CUDA Inference Tool is a scolar and personal project designed for neural network inference using CUDA. The goal is to try to make a faster tool from scratch in CUDA directly, to outperform PyTorch for implemented models. It supports operations for Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs), optimized for GPU execution. 

## Features

- **Core Operations**: Matrix operations, convolutions, and support for CNN and MLP models.
- **ONNX Support**: Load and process ONNX models for inference.
- **Optimized Execution**: CUDA-based optimizations for high performance.
- **Testing**: Comprehensive test suite using GoogleTest.

## Dependencies

- [spdlog](https://github.com/gabime/spdlog)
- [GoogleTest](https://github.com/google/googletest)
- [Protobuf](https://github.com/protocolbuffers/protobuf)
- [Abseil](https://abseil.io/)

## Usage

### Compile and Run

```bash
mkdir ./build && cd build
cmake ..
make
```

### Run the Application

To execute the main application:
```bash
./build/main_cuda
```

For inference tasks:
```bash
./build/inference_cuda
./build/inference_cnn_cuda
```

### Compile and Test

```bash
mkdir ./build && cd build
cmake ..
ctest
```

### Results

From our tests, MLP inference is indeed faster than on PyTorch. For now, problems in the implementation of the CNN makes it much slower than PyTorch, but still functional.