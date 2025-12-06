cmake_minimum_required(VERSION 3.18)
project(my_cuda_project CUDA CXX)

include(FetchContent)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0  # or latest version
)
FetchContent_MakeAvailable(spdlog)

# Your CUDA executable
add_executable(out main.cu matrix_operations.cu)
target_link_libraries(my_program PRIVATE spdlog::spdlog)
set_target_properties(my_program PROPERTIES CUDA_SEPARABLE_COMPILATION ON)