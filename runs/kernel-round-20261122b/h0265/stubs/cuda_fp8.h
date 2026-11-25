// Minimal stub for a syntax-only check with clang++ (no CUDA frontend on this host). Not for codegen.
#pragma once
#include "cuda_runtime.h"
typedef enum { __NV_E4M3 = 0, __NV_E5M2 = 1 } __nv_fp8_interpretation_t;
typedef enum { __NV_NOSAT = 0, __NV_SATFINITE = 1 } __nv_saturation_t;
struct __nv_fp8_storage_t_stub { unsigned char v; };
inline unsigned char __nv_cvt_float_to_fp8(float f, int sat, int interp) { (void)f; (void)sat; (void)interp; return 0; }
