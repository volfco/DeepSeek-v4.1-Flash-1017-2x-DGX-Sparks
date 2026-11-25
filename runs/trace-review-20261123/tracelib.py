"""Shared loader for the 0272 ring traces (read-only analysis, trace-review-20261123)."""
import json, os

RUNS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {
    # name: (path stem, sampling N)
    'fx1L': ('sched-20261122/b2/mech/fx1L', 4),
    'fx1Lb': ('sched-20261122/b2/mech/fx1Lb', 4),
    'np1L': ('sched-20261122/b2/mech/np1L', 4),
    'ns1L': ('sched-20261122/b2/mech/ns1L', 4),
    'k0x1': ('sched-20261122/b2/rowsplit/k0x1', 16),
    'k0x4': ('sched-20261122/b2/rowsplit/k0x4', 16),
    'k6x1': ('sched-20261122/b2/rowsplit/k6x1', 16),
    'k6x4': ('sched-20261122/b2/rowsplit/k6x4', 16),
    'fx1': ('sched-20261122/b2/rowsplit/fx1', 16),
    'fx4': ('sched-20261122/b2/rowsplit/fx4', 16),
    'tr16a': ('overnight-20261122/k272/tr16a', 16),
    'tr16b': ('overnight-20261122/k272/tr16b', 16),
}

def path(name, rank):
    return os.path.join(RUNS, FILES[name][0] + '.trace.r%d.jsonl' % rank)

def load(name, rank, kinds=('gpu',)):
    out = {k: [] for k in kinds}
    with open(path(name, rank)) as f:
        for line in f:
            d = json.loads(line)
            k = d.get('kind')
            if k in out:
                out[k].append(d)
    # the drain writes the 4,096-record ring in slot order, so a wrap puts newer records first: order by sequence
    for k in out:
        if k in ('gpu', 'proxy'): out[k].sort(key=lambda d: d['seq'])
    return out if len(kinds) > 1 else out[kinds[0]]

def pct(xs, q):
    """Nearest-rank percentile of a sorted list (q in 0..100)."""
    if not xs:
        return float('nan')
    i = min(len(xs) - 1, max(0, int(round(q / 100.0 * (len(xs) - 1)))))
    return xs[i]

def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return dict(n=0)
    return dict(n=n, mean=sum(xs) / n, p10=pct(xs, 10), p50=pct(xs, 50), p90=pct(xs, 90), p99=pct(xs, 99), p999=pct(xs, 99.9), min=xs[0], max=xs[-1])
