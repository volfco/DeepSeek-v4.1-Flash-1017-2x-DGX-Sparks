// Syntax-only check of the hand-written CUDA on a host without a CUDA frontend: the CUDA qualifiers, shuffles and
// inline PTX are stubbed, so this catches Zig-side/argument-shape typos and C++ errors, not codegen or PTX validity.
#define asm(...) ((void)0)
#include KERNEL_FILE
