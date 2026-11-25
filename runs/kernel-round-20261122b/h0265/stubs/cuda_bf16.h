// Minimal stub for a syntax-only check with clang++ (no CUDA frontend on this host). Not for codegen.
// Round 0260-0263 copy of cuda-syntax-stubs: adds what the 8d838cc tree already uses (bf16 from int, __nv_bfloat162,
// float4, __clz, max) so the check passes on the base and flags only new mistakes.
#pragma once
#include "cuda_runtime.h"
struct __nv_bfloat16 { unsigned short v; __nv_bfloat16() = default; __nv_bfloat16(int) : v(0) {} operator unsigned short() const { return v; } };
struct __nv_bfloat162 { __nv_bfloat16 x, y; };
inline float __bfloat162float(__nv_bfloat16 h) { (void)h; return 0.f; }
inline __nv_bfloat16 __float2bfloat16_rn(float f) { (void)f; return {}; }
