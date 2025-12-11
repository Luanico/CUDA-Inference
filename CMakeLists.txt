cmake_minimum_required(VERSION 3.18)
project(my_cuda_project CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Fetch spdlog
FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(spdlog)

# Fetch Google Test
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Shared library for matrix operations (used by both main and tests)
add_library(matrix_ops STATIC
  matrix_operations.cu
  convolutions.cu
  CNN.cu
  cpu_operations.cpp
  mlp.cu
)

target_link_libraries(matrix_ops PUBLIC spdlog::spdlog)
set_target_properties(matrix_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(utils STATIC
  error_utils.cu
)

target_link_libraries(utils PUBLIC spdlog::spdlog)
set_target_properties(utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# Main executable
add_executable(out main.cu)
target_link_libraries(out PRIVATE matrix_ops utils spdlog::spdlog)
set_target_properties(out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# Test 1: Basic matrix operations
add_executable(matrix_tests basic_operations_test.cu)
target_link_libraries(matrix_tests PRIVATE 
  matrix_ops
  utils
  GTest::gtest_main
)
set_target_properties(matrix_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Test 2: MLP
add_executable(mlp_tests MLP_test.cu)
target_link_libraries(mlp_tests PRIVATE 
    matrix_ops
    utils
    GTest::gtest_main
)
set_target_properties(mlp_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Test 3: convolutions
add_executable(convolutions_tests convolutions_test.cu)
target_link_libraries(convolutions_tests PRIVATE 
    matrix_ops
    utils
    GTest::gtest_main
)
set_target_properties(convolutions_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# Test 4: conv layer
add_executable(conv_layer_tests conv_layer_tests.cu)
target_link_libraries(conv_layer_tests PRIVATE
    matrix_ops
    utils
    GTest::gtest_main
)
set_target_properties(conv_layer_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# Test 5: cnn
add_executable(cnn_tests CNN_tests.cu)
target_link_libraries(cnn_tests PRIVATE 
    matrix_ops
    utils
    GTest::gtest_main
)
set_target_properties(cnn_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)



if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Adding Conda prefix to search paths: $ENV{CONDA_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    include_directories("$ENV{CONDA_PREFIX}/include")
    link_directories("$ENV{CONDA_PREFIX}/lib")
endif()

find_package(Protobuf REQUIRED)
find_package(absl REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

add_executable(inference_cuda inference_cuda.cu load_onnx.cpp onnx.pb.cc)
target_link_libraries(inference_cuda PRIVATE 
    matrix_ops
    utils
    ${Protobuf_LIBRARIES}
    absl::log_internal_check_op
    absl::log_internal_message
    absl::base
    absl::strings
)

add_executable(inference_cnn_cuda inference_cnn_cuda.cu load_onnx.cpp onnx.pb.cc)
target_link_libraries(inference_cnn_cuda PRIVATE 
    matrix_ops
    utils
    ${Protobuf_LIBRARIES}
    absl::log_internal_check_op
    absl::log_internal_message
    absl::base
    absl::strings
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(matrix_tests)
gtest_discover_tests(mlp_tests)
gtest_discover_tests(convolutions_tests)
gtest_discover_tests(conv_layer_tests)
gtest_discover_tests(cnn_tests)