#!/usr/bin/env bash
# h0267 build + run: extract the kernel's exact source, compile the CPU emulation with no FMA contraction (the kernel's
# --fmad=false MUL then ADD), run the case matrix, then three mutation checks (each must FAIL cases, or the matrix does
# not bite). Output is run.log. No GPU, no network.
set -uo pipefail
cd "$(dirname "$0")"

echo "== splice (the kernel's exact source) =="
python3 splice.py || exit 1

echo "== build (-O2 -ffp-contract=off -pthread) =="
g++ -std=c++17 -O2 -ffp-contract=off -Wno-unknown-pragmas -pthread fused_run.cpp -o fused_run || exit 1

echo "== run =="
./fused_run || exit 1

echo "== mutation checks (each must fail cases) =="
for mut in better_ties no_round miss_mask; do
    echo "-- mutation: $mut"
    python3 splice.py "$mut" || exit 1
    g++ -std=c++17 -O2 -ffp-contract=off -Wno-unknown-pragmas -pthread fused_run.cpp -o fused_run_mut || exit 1
    ./fused_run_mut > mut_run.log 2>&1 # (the mutant exits 1 when it fails cases: that is the point)
    if grep -q "FAIL" mut_run.log; then
        echo "mutation $mut: caught ($(grep -c FAIL mut_run.log) failing checks)"
    else
        echo "mutation $mut: NOT CAUGHT (the matrix does not bite)"; exit 1
    fi
done

echo "== restore the unmutated kernel =="
python3 splice.py || exit 1
g++ -std=c++17 -O2 -ffp-contract=off -Wno-unknown-pragmas -pthread fused_run.cpp -o fused_run || exit 1
echo "h0267: ok"
