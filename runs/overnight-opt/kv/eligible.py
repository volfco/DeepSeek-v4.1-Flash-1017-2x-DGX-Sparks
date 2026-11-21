"""Compare 500 ms with and without the isolated victim eligibility guard on one binary."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm as cfg
import math_reference

guard = 'TF_DS_BENCH_PREFER_ELIGIBLE'


def restart(arm, enabled, scratch, original, images):
    cfg.restart(arm, '500', '0', scratch, original, images, extra={guard: str(int(enabled))})
    logs = subprocess.check_output(['docker', 'logs', cfg.policy.name], text=True, stderr=subprocess.STDOUT)
    if ('kv trial eligibility guard enabled' in logs) != enabled:
        raise RuntimeError('Eligibility guard startup state differs')


def balanced(prompts, cases, scratch):
    p, c = cfg.p, cfg.c
    if sum(p.rows()) > 32768:
        raise RuntimeError('Expected a fresh occupancy')
    for key in ('shared', 'f0', 'f1', 'f2'):
        p.probe(key, 'fill', 0, cases[key])
        if key != 'shared':
            p.probe(key, 'pre-pressure', 0, cases[key], p.grid)
    rows = p.rows()
    if rows != [2*p.grid, 2*p.grid]:
        raise RuntimeError(f'Expected a balanced peak: {rows}')
    deadline = time.monotonic() + 120
    while True:
        inventory = cfg.groups(cases, scratch)
        if all(p.grid in rank[key] for rank in inventory for key in ('shared','f0','f1','f2')):
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f'Expected committed snapshots for the entire balanced peak: {inventory}')
        time.sleep(3)
    c.emit('stage', stage='peak', rows=rows, groups=inventory)
    since = time.time()
    q = c.cell(300000, 'admission', prompts, list(range(4200,4204)), budget=1024, sharing='peak', rep=0, expected_cached=p.grid)
    c.emit('patient', events=cfg.run.held_events(since))
    logs = subprocess.check_output(['docker','logs','--since',str(since),cfg.policy.name], text=True, stderr=subprocess.STDOUT)
    c.emit('guard', decisions=[l.split('kv eligibility trial ',1)[1] for l in logs.splitlines() if 'kv eligibility trial ' in l])
    for key in ('f1', 'f0', 'f2'):
        p.probe(key, 'after-pressure', 0, cases[key])
    return rows, [(r['input_sha256'],r['cached'],r['tokens'],r['done']['sha']) for r in q]


def main(args):
    cfg.run.image = 'tensorfold-dsv41:bench-eligible-dc8fe410c9b0'
    cfg.run.controls.add(guard)
    out, original, ids, scratch = cfg.setup(args)
    prompts, cases = cfg.fixture()
    comparisons = {}
    try:
        inventory = cfg.groups(cases, scratch)
        if not all(all(cfg.p.grid in rank[k] for k in ('shared','f0','f1')) for rank in inventory):
            restart('seed', False, scratch, original, ids)
            cfg.seed(cases, scratch)
        else:
            cfg.c.emit('seed-inventory', groups=inventory, reused=True)
        # Bookend the guarded / plain comparisons. The shorter output showed
        # an incremental cold revisit on the original combined binary.
        trials = [('plain-384', False, 384), ('guard-384', True, 384),
                  ('guard-1024', True, 1024), ('plain-1024', False, 1024)]
        if args.balanced:
            trials += [('plain-balanced', False, 1024), ('guard-balanced', True, 1024)]
        for arm, enabled, budget in trials:
            if args.only and arm not in args.only:
                continue
            restart(arm, enabled, scratch, original, ids)
            if arm.endswith('balanced'):
                result = balanced(prompts, cases, scratch)
                key = ('balanced', budget)
            else:
                result = cfg.measure(cases, scratch, budget)
                key = ('selective', budget)
            if key in comparisons:
                cfg.c.emit('matched', measured=result == comparisons[key], shape=key[0], budget=budget)
            comparisons[key] = result
            if arm == 'guard-384':
                cfg.run.exact(arm, out)
            cfg.t.validate(arm)
    except BaseException as exc:
        cfg.c.emit('failure', error=str(exc))
        raise
    finally:
        cfg.restore(original)
        try:
            math_reference.run_if_requested(cfg)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            cfg.c.emit('math-reference-failure',error=str(exc))
    cfg.c.emit('complete')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--store', default='eligible01')
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--only', nargs='+', choices=('plain-384','guard-384','guard-1024','plain-1024','plain-balanced','guard-balanced'))
    parser.add_argument('--traffic-since', type=float)
    args = parser.parse_args()
    if Path(args.store).name != args.store:
        parser.error('Expected private store name')
    with (cfg.run.prior.parent/'.pair.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(args)
