#!/usr/bin/env python3
"""Per-kernel SASS histogram + resources for named kernels in a cubin (item evidence)."""
import json, re, subprocess, sys
from collections import Counter
CUOBJ = "/usr/local/cuda/bin/cuobjdump"
PAT = ["LDG.E.128", "LDG.E.64", "LDG.E.U16", "LDG.E", "STG.E.128", "STG.E.64", "STG.E", "LDS", "STS", "SHFL", "BAR.SYNC",
       "MUFU", "VOTE", "ATOM", "BRA", "CALL", "FCHK", "FFMA", "FADD", "FMUL", "FSETP", "ISETP"]
def main():
    cubin, out = sys.argv[1], sys.argv[2]
    names = sys.argv[3:]
    sass = subprocess.run([CUOBJ, "-sass", cubin], capture_output=True, text=True).stdout
    res = subprocess.run([CUOBJ, "-res-usage", cubin], capture_output=True, text=True).stdout
    secs, cur, body = {}, None, []
    for line in sass.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            if cur: secs[cur] = body
            cur, body = m.group(1), []
        elif cur: body.append(line)
    if cur: secs[cur] = body
    usage = {}
    for line in res.splitlines():
        m = re.search(r"Function (\S+):", line)
        if m: cur = m.group(1); continue
        m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", line)
        if m and cur: usage[cur] = {"reg": int(m.group(1)), "stack": int(m.group(2)), "shared": int(m.group(3)), "local": int(m.group(4))}
    outj = {}
    for n in names:
        b = secs[n]
        ops = []
        for line in b:
            m = re.search(r"/\*[0-9a-f]{4}\*/\s+@?!?P?T?\d?\s*([A-Z0-9_.]+)", line)
            if m: ops.append(m.group(1))
        c = Counter()
        for op in ops:
            for p in PAT:
                if op == p or op.startswith(p + "."):
                    c[p] += 1
                    break
        outj[n] = {"insns": len(ops), "ops": dict(sorted(c.items())), **usage.get(n, {})}
        print(n, json.dumps(outj[n]))
    open(out, "w").write(json.dumps(outj, indent=1))
main()
