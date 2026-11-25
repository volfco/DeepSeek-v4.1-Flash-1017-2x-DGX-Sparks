// extra.h: declarations cuda-syntax-stubs leaves out, force-included (-include) by check_all.sh when EXTRA is set, so
// the code around them is really parsed. The stock stubs' gaps make clang give up on whole statements (every error
// check.sh prints for dsv41_torch.cu at the base sits in tf_ds_qd_pair_chunk, which this sweep edits), so an edit there
// cannot be judged by "the same error set" alone. Syntax only, as the stubs are: nothing here models the device.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
struct __nv_bfloat162 { __nv_bfloat16 x, y; };
struct float4 { float x, y, z, w; };
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
inline int __clz(int v) { return v == 0 ? 32 : __builtin_clz((unsigned)v); }
