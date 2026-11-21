"""Review fixed-work CPU contention windows with the read loop and two clock/temperature sources."""
import argparse
import collections
import importlib
import json
from pathlib import Path
import re
import statistics
import sys

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent / 'engine-followup'))
import analyze
import duty


def summary(values):
    return dict(count=len(values), minimum=min(values), median=statistics.median(values), maximum=max(values)) if values else dict(count=0)


def analyze_cells(directory, passive):
    offsets = json.loads((directory / 'clock.json').read_text())
    if abs(offsets[0] - offsets[1]) > .05:
        raise ValueError('Head wall-clock correction changed during the test')
    offset = statistics.mean(offsets)
    nvml = [duty.telemetry(passive / f'{rank}-nvml.csv') for rank in ('head', 'worker')]
    hosts = [json.loads(line) for line in (directory/'hosts.jsonl').read_text().splitlines()]
    physical = {(r['depth'], r['width'], r['rep'], r['arm']): r for r in importlib.import_module('sensitivity-results').rows(directory)}
    read_proofs = []
    for path in sorted(directory.glob('*-*.err')):
        match = re.fullmatch(r'(\d+)-(cache-a|stream|cache-b)-(head|worker)\.err', path.name)
        if match is None:
            continue
        depth, mode, rank = match.groups()
        values = [tuple(map(int,row)) for row in re.findall(r'^cpu=(\d+) size=(\d+) checksum=(\d+)$', path.read_text(), re.M)]
        if len(values) != 3 or {r[0] for r in values} != {0,1,2} or any(r[2]==0 for r in values):
            raise ValueError(f'Invalid read-loop completion proof {path}')
        expected = 64*2**20 if mode=='stream' else 16*2**10
        if any(v[1] != expected for v in values):raise ValueError(f'Incorrect read footprint {path}')
        read_proofs.append(dict(depth=int(depth),arm=mode,rank=rank,cores=[0,1,2],bytes_per_core=expected))
    if len(read_proofs)!=12:raise ValueError('Expected all 12 rank/arm/depth reader completion proofs')
    groups = collections.defaultdict(dict)
    for c in analyze.cells(directory/'workload.jsonl'):
        if c['cell'] == 'chat-prime':
            continue
        req = c['requests']
        depth, width, rep, arm = c['depth'], len(req), c['rep'], c['sharing']
        identity = [(r['input_sha256'],r['cached'],r['tokens'],r['done']['sha']) for r in req]
        a, b = [analyze.active_segment(c)[i]['t'] + offset for i in (0,-1)]
        rate = physical[(depth,width,rep,arm)]
        draft = [list(map(int,r['done']['accepted'].split('/'))) for r in req]
        values = dict(identity=identity, active_s=rate['active_s'], physical_rounds=rate['rounds'], ms_per_round=rate['ms_round'],
                      generated_tok_s=rate['tok_s'], request_draft_ids_accepted=sum(x[0] for x in draft),
                      request_draft_ids_proposed=sum(x[1] for x in draft), request_rounds=[r['done']['rounds'] for r in req], ranks={})
        for j, rank in enumerate(('head','worker')):
            inside = [r for r in nvml[j] if a+.05<=r[0]<=b-.05]
            # A 500 ms NVML utilization sample describes an earlier interval. Exclude its first 600 ms.
            duty_samples = [r[1] for r in inside if r[0]>=a+.6]
            thermal = [r['ranks'][j] for r in hosts if 'ranks' in r and a+.1 <= r['at'] <= b-.1]
            gpu = [r['gpu'].split(', ') for r in thermal]
            values['ranks'][rank] = dict(reader_logical_GB_s=rate[rank+'_reader_GB_s'], reader_window_s=rate[rank+'_sample_s'],
                nvml_sm_mhz=summary([r[3] for r in inside]), nvml_busy_percent=summary(duty_samples),
                host_graphics_mhz=summary([int(g[0].split()[0]) for g in gpu]), temperature_c=summary([int(g[1]) for g in gpu]),
                clock_event_masks=dict(collections.Counter(g[2] for g in gpu)), available_gib=summary([r['memory_gib'] for r in thermal]))
        groups[(depth,width,rep)][arm] = values
    result = []
    for (depth,width,rep), arms in sorted(groups.items()):
        a,s,b=(arms[k] for k in ('cache-a','stream','cache-b'))
        if not a['identity'] == s['identity'] == b['identity']:
            raise ValueError(f'Request, generation or cache mismatch at {depth}/{width}/{rep}')
        for v in arms.values(): del v['identity']
        ms=lambda r:r['ms_per_round']
        result.append(dict(depth=depth,width=width,rep=rep,arms=arms,matched_requests=width,
            stream_vs_control_mean_pct=100*(ms(s)/statistics.mean([ms(a),ms(b)])-1),
            stream_vs_slower_control_pct=100*(ms(s)/max(ms(a),ms(b))-1),
            return_vs_open_pct=100*(ms(b)/ms(a)-1)))
    if len(result)!=6:raise ValueError('Expected two shallow and four deep fixed-work comparisons')
    return dict(groups=result,reader_completion=read_proofs,clock_correction_span_ms=1000*(offsets[1]-offsets[0]),host_probe_errors=sum('error' in r for r in hosts))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('passive',type=Path)
    a=p.parse_args();print(json.dumps(analyze_cells(a.directory,a.passive),indent=2))
