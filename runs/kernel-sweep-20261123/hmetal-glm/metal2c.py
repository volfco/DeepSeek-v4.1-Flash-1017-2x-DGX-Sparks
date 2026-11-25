#!/usr/bin/env python3
"""Cut glm_index_select and its helpers (glm_scan, glm_order_key, GlmSelectArgs) out of glm_glue.metal and rewrite the
Metal-only syntax into what emu/emu.h runs: argument attributes dropped, address-space qualifiers dropped, a
`threadgroup` local as the block's shared memory (a function static), the barrier as __syncthreads, the relaxed
atomics as plain operations (one OS thread; the lane order is the scheduler's). The statements themselves are not
touched: the digit loop, the bin walk, the scans and the stores are the file's text.
Usage: metal2c.py <glm_glue.metal> <out.inc>"""
import re
import sys

src = open(sys.argv[1]).read()
a = src.index("// Row r's key list: its TOP best blocks")
b = src.index("\n}\n", src.index("kernel void glm_index_select")) + 3
t = src[a:b]
t = re.sub(r"\s*\[\[[a-z_]+(\(\d+\))?\]\]", "", t)                      # [[buffer(0)]], [[thread_position_in_threadgroup]]
t = t.replace("kernel void", "void")
t = re.sub(r"^(\s+)threadgroup ", r"\1static ", t, flags=re.M)          # a local: the threadgroup's shared memory
t = t.replace("threadgroup uint* part", "uint* part")                   # a parameter: just a pointer
t = t.replace("const device ", "const ").replace("device ", "").replace("constant ", "const ")
t = re.sub(r"threadgroup_barrier\(mem_flags::mem_[a-z]+\)", "__syncthreads()", t)
t = t.replace("as_type<uint>(v)", "__float_as_uint(v)")
t = t.replace("atomic_uint", "uint")
t = re.sub(r"atomic_store_explicit\(&([^,]+), ([^,]+), memory_order_relaxed\)", r"(\1 = \2)", t)
t = re.sub(r"atomic_fetch_add_explicit\(&([^;]+), 1u, memory_order_relaxed\)", r"(\1 += 1u)", t)
t = re.sub(r"atomic_load_explicit\(&([^,]+), memory_order_relaxed\)", r"(\1)", t)
code = re.sub(r"//[^\n]*", "", t)   # (comments may name what the code no longer does)
for left in ("threadgroup", "device", "constant", "atomic_", "as_type", "[["):
    assert left not in code, "untranslated Metal syntax: " + left
open(sys.argv[2], "w").write("// from %s by metal2c.py - do not edit\n%s" % (sys.argv[1], t))
print("translated %d lines" % t.count("\n"))
