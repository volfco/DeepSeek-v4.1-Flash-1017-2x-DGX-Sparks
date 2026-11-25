"""Size a dynamic Markov bias write-back region against real draft chains.

Reads the `TF_DS_MARKOV_TRACE` dump (markov_trace.zig): u32 magic, u32 n, u32 steps, u32 0, then i64 toks[n][steps+1].
Row i is stream i's pending token and then each step's pick, so **step j reads toks[i][j]** — the ids its `_bias`
looks up.

The static cache (markov_tokens.cached, 256 ids served) always hits. A dynamic region of R rows is filled only on a
miss: the missed row is written back and a victim (FIFO or LRU) is evicted. A step is slow when **any** stream misses
(one `_bias` over the 33.1 MB head serves the whole step), so the unit that matters is misses a *step*, not a row.

  python3 runs/markov-wb-20261125/sim.py <trace.bin> [--miss-us 143]

Reports, per region size: row hit rate, miss steps a pass, and the time saved a pass against the static-256 baseline.
"""
import struct
import sys

import numpy as np

MAGIC = 0x4D543331
# markov_tokens.cached (256 ids, the served list)
STATIC = [14,201,270,223,16,361,528,304,294,65,305,260,438,295,344,18,5230,775,1823,28,603,362,2280,1137,855,
          29617,271,262,339,19,343,71315,1313,905,90,582,4,682,11,66,396,366,554,21,1507,20,412,455,418,1281,10,
          469,3395,1354,290,9,513,85,377,15,30,2366,989,3730,566,32,944,31,6248,477,436,13,12,6666,515,1351,877,
          538,200,22,9952,426,411,30781,818,1445,13969,17,3682,26,93352,23,25,29,28029,848,7854,982,69,106130,24,
          6013,759,80,61311,1341,67,1018,868,734,5832,61,5,4245,4181,509,3,2672,1760,27,440,46,1915,565,1922,33,
          2272,4216,63,835,1990,26702,588,13587,72,28381,2329,1162,116230,7515,70,15136,3964,671,329,1492,267,71,
          778,12006,769,1505,611,5395,1718,1698,2852,3006,37,68,4099,15156,7302,87,6407,710,2704,3390,1393,1142,
          342,1082,1780,369,345,5224,979,510,11853,84,504,17239,47825,1347,3104,50928,5538,2052,13755,6107,334,
          2606,3816,3608,834,764,105366,790,75,754,915,56,1117,2613,1017,16617,643,47058,4941,2292,35,990,2310,
          4757,1178,26495,2161,14235,8201,940,1353,16153,10993,9000,82,58,18691,91,36,442,20038,1647,79,875,1257,
          86,12124,3316,4731,3767,792,14239,1009,579,6723,53,52,11493,793,94,29611,8567,2296,782,9405,881]


def read_trace(path):
    with open(path, "rb") as f:
        while True:
            head = f.read(16)
            if not head:
                return
            if len(head) != 16:
                raise SystemExit("truncated header")
            magic, n, steps, _ = struct.unpack("<IIII", head)
            if magic != MAGIC:
                raise SystemExit(f"bad magic {magic:#x}")
            body = f.read(8 * n * (steps + 1))
            if len(body) != 8 * n * (steps + 1):
                raise SystemExit("truncated record")
            toks = np.frombuffer(body, dtype="<i8").reshape(n, steps + 1)
            yield n, steps, toks


def simulate(passes, static_set, rows, policy):
    """Returns (row_hits, row_misses, miss_steps) for a dynamic region of `rows` (0: static only)."""
    region = {}          # token -> slot
    order = []           # LRU order (slot indices), unused for FIFO
    free = list(range(rows)) if rows else []
    victim = 0
    hits = misses = miss_steps = 0
    for n, steps, toks in passes:
        for j in range(steps):
            step_miss = False
            for i in range(n):
                t = int(toks[i, j])
                if t in static_set:
                    hits += 1
                    continue
                if rows == 0:
                    misses += 1
                    step_miss = True
                    continue
                if t in region:
                    hits += 1
                    if policy == "lru":
                        order.remove(region[t]); order.append(region[t])
                    continue
                misses += 1
                step_miss = True
                if free:
                    slot = free.pop()
                else:
                    slot = victim
                    victim = (victim + 1) % rows
                    old = next((k for k, v in region.items() if v == slot), None)
                    if old is not None:
                        del region[old]
                region[t] = slot
                if policy == "lru":
                    order.append(slot)
                    if free:
                        pass
            if step_miss:
                miss_steps += 1
    return hits, misses, miss_steps


def main():
    path = sys.argv[1]
    miss_us = float(sys.argv[sys.argv.index("--miss-us") + 1]) if "--miss-us" in sys.argv else 143.0
    passes = list(read_trace(path))
    if not passes:
        raise SystemExit("empty trace")
    npas = len(passes)
    steps_total = sum(s for _, s, _ in passes)
    rows_total = sum(n * s for n, s, _ in passes)
    print(f"{npas} passes, {steps_total} steps, {rows_total} step rows (mean {rows_total/steps_total:.1f} streams a step)")

    static_set = set(STATIC)
    b_hits, b_miss, b_msteps = simulate(passes, static_set, 0, "fifo")
    print(f"static 256 only: rows hit {100*b_hits/rows_total:.1f}%, miss steps a pass {b_msteps/npas:.2f}, "
          f"~{b_msteps*miss_us/1000/npas:.3f} ms a pass of _bias")
    print()
    print(f"{'rows':>5} {'MiB':>6} {'policy':>6} {'row hit%':>9} {'missSteps/p':>11} {'saved ms/p':>10}")
    for rows in (64, 128, 256, 512, 1024, 2048, 4096):
        for policy in ("fifo", "lru"):
            h, m, ms = simulate(passes, static_set, rows, policy)
            saved = (b_msteps - ms) * miss_us / 1000 / npas
            print(f"{rows:5d} {rows*126/1024:6.1f} {policy:>6} {100*h/rows_total:9.1f} {ms/npas:11.2f} {saved:10.3f}")


if __name__ == "__main__":
    main()
