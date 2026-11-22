# Pull a kernel's exact source out of the worktree's .cu: from its leading comment marker through the closing brace of
# the named function (balanced braces), written to <out>.inc for the C++ driver to #include.
import sys
src_path, marker, name, out = sys.argv[1:5]
src = open(src_path).read()
start = src.index(marker)
fn = src.index(name, start)
i = src.index('{', fn)
depth = 0
while True:
    c = src[i]
    if c == '{': depth += 1
    elif c == '}':
        depth -= 1
        if depth == 0: break
    i += 1
open(out, 'w').write(src[start:i + 1] + '\n')
print(f'extracted {name}: {src[start:i+1].count(chr(10)) + 1} lines')
