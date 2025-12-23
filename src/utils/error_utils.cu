#include "error_utils.h"

/**
 * @brief Aborts execution with a CUDA error message
 * @param msg Error message to display
 * @param fname Function name where error occurred
 * @param line Line number where error occurred
 */
[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}