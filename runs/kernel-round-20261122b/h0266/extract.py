#!/usr/bin/env python3
# h0266: pull the 0197 decode compressor and the TF_DS_COMPRESS2_PAIR_ONCE variant's exact source spans out of the
# worktree's dsv41_torch.cu into one .inc, then adapt them for the CPU emulation (shared arrays into SimBlock, the
# kit's approximate inline PTX into the harness primitives). Usage: extract.py <cu> <out.inc>
import re, sys

src = open(sys.argv[1]).read()
out = []


def span(start, end):
    a = src.index(start)
    b = src.index(end, a + len(start))
    out.append(src[a:b])
    print(f'extracted {start[:60]!r}: {src[a:b].count(chr(10))} lines')


# the pair-sum arithmetic (tf_ds_compress2_pair)
span('// The pair-sum arithmetic of tf_ds_compress2_kernel', '// -- TF_DS_FUSED_COMPRESS2_NORM (0197)')
# the kit norm's three steps (squares4, warp_sum, inverse_rms)
span('// These three steps mirror the instructions and reduction order of the pinned kit.',
     '// Decode RMSNorm, specialized')
# the partner resolution (tf_ds_compress2_partners)
span('// Each warp resolves both partners before reading any ring data.',
     'template<int D>\n__device__ __forceinline__ void tf_ds_compress2_decode')
# 0197's per-row body + wrapper macro + instantiations (the off path)
span('// Decode ratio-2 scatter, pair compression and kit RMSNorm.', '// -- TF_DS_COMPRESS2_PAIR_ONCE')
# the pair-once body + wrapper macro + instantiations (the on path)
span('// -- TF_DS_COMPRESS2_PAIR_ONCE (compress2_pair.zig, off by default)',
     '// TF_DS_FUSED_ROPE_STORE (0198, off by default).')

k = '\n'.join(out)
# shared arrays become SimBlock members (one block emulated at a time)
k = re.sub(r'[ \t]*__shared__ float row\[D == 512 \? 512 : 1\];\n', '', k)
k = re.sub(r'[ \t]*__shared__ float warp_sum\[4\];\n', '', k)
k = re.sub(r'\brow\[', 'blk->row[', k)
k = re.sub(r'\bwarp_sum\[', 'blk->warp_sum[', k)
# the kit's approximate PTX: the harness runs both kernels' identical text through the same primitives
k = k.replace('asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(root) : "f"(mean_eps));', 'root = kit_sqrt_approx(mean_eps);')
k = k.replace('asm("div.approx.f32 %0, %1, %2;" : "=f"(inv) : "f"(1.0f), "f"(root));', 'inv = kit_div_approx(1.0f, root);')
assert 'blk->row[' in k and 'blk->warp_sum[' in k and 'kit_sqrt_approx' in k and 'kit_div_approx' in k
assert '__shared__' not in k and 'asm(' not in k
open(sys.argv[2], 'w').write(k + '\n')
print('wrote', sys.argv[2])
