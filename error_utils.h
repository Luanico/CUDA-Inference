#include <spdlog/spdlog.h>

void _abortError(const char* msg, const char* fname, int line);

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)