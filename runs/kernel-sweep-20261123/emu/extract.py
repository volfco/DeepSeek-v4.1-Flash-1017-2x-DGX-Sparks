#!/usr/bin/env python3
"""Pull exact source spans out of a .cu into one .inc for the CPU harnesses: each span runs from a start marker up to
(not including) an end marker; a marker that drifted raises. `extern "C" ` is stripped so two revisions of the same
kernel can live in two namespaces of one program, and `#include` lines are dropped.
Usage: extract.py <cu> <out.inc> <start1> <end1> [<start2> <end2> ...]"""
import sys

src = open(sys.argv[1]).read()
out = ["// extracted from %s by emu/extract.py - do not edit\n" % sys.argv[1]]
args = sys.argv[3:]
for start, end in zip(args[0::2], args[1::2]):
    a = src.index(start)
    b = src.index(end, a + len(start))
    span = src[a:b]
    out.append(span)
    print("extracted %r: %d lines" % (start[:56], span.count("\n")))
text = "\n".join(out).replace('extern "C" ', "")
text = "".join(l for l in text.splitlines(keepends=True) if not l.startswith("#include"))
open(sys.argv[2], "w").write(text)
