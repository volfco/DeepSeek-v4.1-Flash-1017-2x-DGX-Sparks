// Minimal stub for a syntax-only check with clang++ (no CUDA frontend on this host). Not for codegen.
#pragma once
#include "cuda_runtime.h"
struct __half { unsigned short v; };
inline float __half2float(__half h) { (void)h; return 0.f; }
inline __half __float2half_rn(float f) { (void)f; return __half{0}; }
