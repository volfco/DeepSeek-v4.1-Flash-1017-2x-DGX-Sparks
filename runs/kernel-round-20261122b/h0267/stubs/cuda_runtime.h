// Minimal stub for a syntax-only check with clang++ (no CUDA frontend on this host). Not for codegen.
#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstddef>
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

#define __global__
#define __device__
#define __forceinline__ inline
#define __constant__
#define __shared__
#define __restrict__
#define __launch_bounds__(...)

struct dim3 { unsigned x, y, z; };
extern dim3 blockIdx;
extern dim3 gridDim;
extern dim3 threadIdx;
extern dim3 blockDim;

inline void __syncthreads() {}
inline void __syncwarp(unsigned mask) {}
inline void __nanosleep(unsigned ns) {}

inline int __popc(unsigned v) { int n = 0; while (v) { v &= v - 1; ++n; } return n; }
template <typename T> inline T __shfl_down_sync(unsigned mask, T var, int delta) { (void)mask; (void)delta; return var; }

struct uint4 { unsigned x, y, z, w; };

inline float atomicAdd(float* a, float b) { return *a += b; }
inline unsigned atomicAdd(unsigned* a, unsigned b) { return __sync_fetch_and_add(a, b); }
inline int atomicAdd(int* a, int b) { return __sync_fetch_and_add(a, b); }
inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long b) { return *a += b; }
inline unsigned atomicOr(unsigned* a, unsigned b) { return __sync_fetch_and_or(a, b); }
inline unsigned atomicExch(unsigned* a, unsigned b) { unsigned old = *a; *a = b; return old; }

inline unsigned __float_as_uint(float f) { unsigned u; __builtin_memcpy(&u, &f, 4); return u; }
inline float __uint_as_float(unsigned u) { float f; __builtin_memcpy(&f, &u, 4); return f; }
inline float __int_as_float(int i) { float f; __builtin_memcpy(&f, &i, 4); return f; }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fsub_rn(float a, float b) { return a - b; }
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fdiv_rn(float a, float b) { return a / b; }
inline float __fmaf_rn(float a, float b, float c) { return a * b + c; }
inline float rsqrtf(float x) { return 1.0f / __builtin_sqrtf(x); }
inline double __double_as_longlong(double d) { long long v; __builtin_memcpy(&v, &d, 8); return v; }
inline double __longlong_as_double(long long v) { double d; __builtin_memcpy(&d, &v, 8); return d; }

#undef isinf
inline int isinf(float x) { return __builtin_isinf(x); }
inline int isinf(double x) { return __builtin_isinf(x); }
inline int min(int a, int b) { return a < b ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }

// host stub types for the development launchers (pointwise/movement) — never executed here
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef struct cudaDeviceProp* cudaDeviceProp_t;
typedef int cudaDevice_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1
inline cudaError_t cudaGetLastError() { return cudaSuccess; }

// --- added for the 0195-0199 fused kernels' syntax check ---
template <typename T> inline T __shfl_xor_sync(unsigned mask, T var, int laneMask) { (void)mask; (void)laneMask; return var; }
template <typename T> inline T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width) { (void)mask; (void)laneMask; (void)width; return var; }
template <typename T> inline T __shfl_up_sync(unsigned mask, T var, int delta) { (void)mask; (void)delta; return var; }
inline unsigned __ballot_sync(unsigned mask, int pred) { (void)mask; return pred ? 1u : 0u; }
inline int __ffs(int x) { return x ? __builtin_ffs(x) : 0; }
inline int __ffs(unsigned x) { return x ? __builtin_ffs((int)x) : 0; }
struct int2 { int x, y; };
struct float2 { float x, y; };
inline int2 make_int2(int x, int y) { return int2{x, y}; }
inline float2 make_float2(float x, float y) { return float2{x, y}; }
#undef isnan
inline int isnan(float x) { return __builtin_isnan(x); }
inline int isnan(double x) { return __builtin_isnan(x); }
inline float copysignf(float m, float s) { return __builtin_copysignf(m, s); }
inline unsigned __reduce_add_sync(unsigned mask, unsigned v) { (void)mask; return v; }
inline unsigned __reduce_max_sync(unsigned mask, unsigned v) { (void)mask; return v; }
inline int __reduce_max_sync(unsigned mask, int v) { (void)mask; return v; }
template <typename T> inline T __ldcv(const T* p) { return *p; }
template <typename T> inline T __ldcs(const T* p) { return *p; }
template <typename T> inline T __ldg(const T* p) { return *p; }
inline void __threadfence_system() {}
inline void __threadfence() {}
inline int isfinite(float x) { return __builtin_isfinite(x); }
inline int isfinite(double x) { return __builtin_isfinite(x); }
// --- added for the 0260-0263 round's check of the 8d838cc base ---
struct float4 { float x, y, z, w; };
inline int __clz(unsigned x) { return x ? __builtin_clz(x) : 32; }
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
