"""Print one CUDA benchmark TU from the patched engine and the earlier prototype."""
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
old = (here.parent / 'prune-stats-20261120/micro/prune4.cu').read_text()
engine = Path(sys.argv[1])
ops = (engine / 'zig/kernels/cuda/dsv41_ops.cu').read_text()
print(ops[:ops.index('// `.float()`')])
print(ops[ops.index('struct TfDsPrune {'):ops.index('// rounds.py _candidates_fast')])
print(old[old.index('#include <algorithm>'):old.index('__global__ void pruneTile')])
print(old[old.index('#define CHECK'):old.index('int main(')])
print((here / 'driver.cuh').read_text())
