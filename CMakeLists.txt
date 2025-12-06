cmake_minimum_required(VERSION 3.18)
project(my_cuda_project CUDA CXX)

include(FetchContent)

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

FetchContent_MakeAvailable(spdlog)

# Your CUDA executable
add_executable(out main.cu matrix_operations.cu)
target_link_libraries(out PRIVATE spdlog::spdlog)
set_target_properties(out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)