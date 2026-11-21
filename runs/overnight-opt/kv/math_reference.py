"""Run the two rank-local cuBLAS reference jobs while the caller holds the serving pair lock."""
from pathlib import Path
import json
import subprocess
import time

binary = '/tmp/dsv41-opt-math-reference'


def run_if_requested(cfg):
    out = Path('runs/overnight-opt/hardware01')
    trigger = out/'math-requested'
    if not trigger.is_file() or (out/'math-complete.json').exists():
        return
    cfg.t.idle()
    before = cfg.t.server.metrics()
    done = cfg.c.bench.DONE_KEY
    running = 'tensorfold:requests_running'
    waiting = 'tensorfold:requests_waiting'
    if before.get(running)!=0 or before.get(waiting)!=0 or cfg.p.health()['status']!='ok':
        raise RuntimeError('Expected an idle serving pair before cuBLAS calibration')
    ranks = cfg.policy.ranks
    processes = []
    outputs = []
    jobs = []
    for name,rank in zip(('head','worker'),ranks):
        live = subprocess.check_output(rank+['nvidia-smi','--query-compute-apps=process_name','--format=csv,noheader'],text=True)
        names = [Path(s.strip()).name for s in live.splitlines() if s.strip()]
        if any(v not in ('tensorfold-native','tf-dsv41-lanes','[Not Found]') for v in names):
            raise RuntimeError(f'Unexpected competing GPU process on {name}: {names}')
        processes.append(names)
        subprocess.run(rank+['test','-x',binary],check=True)
    if any((out/(name+'-math.csv')).exists() for name in ('head','worker')):
        raise FileExistsError('A reference CSV already exists, inspect it before retrying')
    start = time.time()
    try:
        for name,rank in zip(('head','worker'),ranks):
            target = out/(name+'-math.csv')
            outputs.append(target.with_suffix('.tmp').open('w'))
            jobs.append(subprocess.Popen(rank+[binary],stdout=outputs[-1],stderr=subprocess.PIPE,text=True))
        for name,job in zip(('head','worker'),jobs):
            _,err = job.communicate(timeout=90)
            if job.returncode:
                raise RuntimeError(f'{name} cuBLAS driver returned {job.returncode}: {err[:400]}')
    finally:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
                job.wait(timeout=10)
        for output in outputs:
            output.close()
    after = cfg.t.server.metrics()
    if after[done]!=before[done] or after[running]!=0 or after[waiting]!=0:
        raise RuntimeError('Service activity overlapped the cuBLAS reference')
    for name in ('head','worker'):
        target = out/(name+'-math.csv')
        temp = target.with_suffix('.tmp')
        if temp.stat().st_size==0:
            raise RuntimeError('Empty cuBLAS reference '+name)
        temp.replace(target)
    record=dict(start=start,end=time.time(),gpu_processes=processes,completed_requests=after[done]-before[done],
                running_before=before[running],waiting_before=before[waiting],running_after=after[running],waiting_after=after[waiting])
    (out/'math-complete.json').write_text(json.dumps(record,indent=2)+'\n')
    cfg.c.emit('math-reference',**record)
