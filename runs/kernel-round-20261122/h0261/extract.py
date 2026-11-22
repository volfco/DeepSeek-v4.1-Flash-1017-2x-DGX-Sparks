# Pull exact source spans out of the worktree's .cu into one .inc: each span runs from a start marker up to (not
# including) an end marker. Usage: extract.py <cu> <out.inc> <start1> <end1> [<start2> <end2> ...]
import sys
src = open(sys.argv[1]).read()
out = []
args = sys.argv[3:]
for start, end in zip(args[0::2], args[1::2]):
    a = src.index(start)
    b = src.index(end, a + len(start))
    out.append(src[a:b])
    print(f'extracted {start[:60]!r}: {src[a:b].count(chr(10))} lines')
open(sys.argv[2], 'w').write('\n'.join(out))
