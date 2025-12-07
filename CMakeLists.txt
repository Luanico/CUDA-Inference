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
  cpu_operations.cpp
)

target_link_libraries(matrix_ops PUBLIC spdlog::spdlog)
set_target_properties(matrix_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Main executable
add_executable(out main.cu)
target_link_libraries(out PRIVATE matrix_ops spdlog::spdlog)
set_target_properties(out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Test executable
add_executable(matrix_tests basic_operations_test.cu)
target_link_libraries(matrix_tests PRIVATE 
  matrix_ops
  GTest::gtest_main
)
set_target_properties(matrix_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(matrix_tests)