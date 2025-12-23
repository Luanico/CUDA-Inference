cmake_minimum_required(VERSION 3.18)
project(my_cuda_project LANGUAGES CUDA CXX)

# ------------------------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()
include(FetchContent)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

# spdlog
FetchContent_Declare(
    spdlog
    URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(spdlog)

# GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Conda (optional)
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Using Conda prefix: $ENV{CONDA_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

# Protobuf / Abseil
find_package(Protobuf REQUIRED)
find_package(absl REQUIRED)

# ------------------------------------------------------------------------------
# Core libraries
# ------------------------------------------------------------------------------

add_library(core_ops STATIC
    src/operations/matrix_operations.cu
    src/operations/convolutions.cu

    src/models/CNN.cu
    src/models/mlp.cu

    src/cpu/cpu_operations.cpp
)

target_include_directories(core_ops PUBLIC src)
target_link_libraries(core_ops PUBLIC spdlog::spdlog)
set_target_properties(core_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ------------------------------------------------------------------------------
# Utils library
# ------------------------------------------------------------------------------

add_library(utils STATIC
    src/utils/error_utils.cu
    src/utils/load_onnx.cpp
)

target_include_directories(utils PUBLIC
    src
    ${Protobuf_INCLUDE_DIRS}
)
target_link_libraries(utils PUBLIC
    spdlog::spdlog
    ${Protobuf_LIBRARIES}
)
set_target_properties(utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ------------------------------------------------------------------------------
# Main executable
# ------------------------------------------------------------------------------

add_executable(main_cuda
    src/main.cu
)

target_link_libraries(main_cuda PRIVATE core_ops utils)
set_target_properties(main_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ------------------------------------------------------------------------------
# Inference executables
# ------------------------------------------------------------------------------

add_executable(inference_cuda
    src/inference_cuda.cu
    src/utils/load_onnx.cpp
    src/utils/proto/onnx.pb.cc
)

target_link_libraries(inference_cuda PRIVATE
    core_ops
    utils
    ${Protobuf_LIBRARIES}
    absl::log_internal_check_op
    absl::log_internal_message
    absl::base
    absl::strings
)

add_executable(inference_cnn_cuda
    src/inference_cnn_cuda.cu
    src/utils/load_onnx.cpp
    src/utils/proto/onnx.pb.cc
)

target_link_libraries(inference_cnn_cuda PRIVATE
    core_ops
    utils
    ${Protobuf_LIBRARIES}
    absl::log_internal_check_op
    absl::log_internal_message
    absl::base
    absl::strings
)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

include(GoogleTest)

function(add_cuda_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE
        core_ops
        utils
        GTest::gtest_main
    )
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    gtest_discover_tests(${target})
endfunction()

add_cuda_test(matrix_tests        tests/basic_operations_test.cu)
add_cuda_test(mlp_tests           tests/MLP_test.cu)
add_cuda_test(convolutions_tests  tests/convolutions_test.cu)
add_cuda_test(conv_layer_tests    tests/conv_layer_tests.cu)
add_cuda_test(cnn_tests           tests/CNN_tests.cu)
