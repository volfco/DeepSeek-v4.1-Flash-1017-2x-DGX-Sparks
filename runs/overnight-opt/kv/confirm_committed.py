"""Independently price the longer burst and the healthy-tier cost of all-rank presence checks."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here))
import committed as trial

cfg=trial.cfg
order=(3,2,1,0)
offsets={4200+task:position*.075 for position,task in enumerate(order)}


def shifted_arrivals():
    request=cfg.c.request
    def staggered(content,seed,tokens,barrier,temperature,**options):
        delay=offsets.get(seed)
        if delay is None:
            return request(content,seed,tokens,barrier,temperature,**options)
        class StartGate:
            def wait(self,timeout):
                barrier.wait(timeout)
                if delay:time.sleep(delay)
        return request(content,seed,tokens,StartGate(),temperature,**options)
    cfg.c.request=staggered

def settle_completed_burst(c=None):
    if c is None:c=cfg.c
    original=c.cell
    def measured(depth,kind,messages,seeds,**opts):
        if kind!='admission':return original(depth,kind,messages,seeds,**opts)
        metric=c.srv.metrics
        begin=time.time()
        done=c.bench.DONE_KEY
        initial=metric()[done]
        def settled():
            sample=metric();start=time.monotonic()
            while (sample[done]-initial==len(messages) and
                   (sample.get('tensorfold:requests_running') or sample.get('tensorfold:requests_waiting')) and
                   time.monotonic()-start<3):
                time.sleep(.05);sample=metric()
            if time.monotonic()-start>.035:
                c.emit('closing-settle',elapsed_s=time.monotonic()-start,completed_delta=sample[done]-initial,
                       running=sample.get('tensorfold:requests_running'),waiting=sample.get('tensorfold:requests_waiting'))
            return sample
        c.srv.metrics=settled
        try:result=original(depth,kind,messages,seeds,**opts)
        finally:c.srv.metrics=metric
        output=subprocess.check_output(['docker','logs','--since',str(begin),cfg.policy.name],text=True,stderr=subprocess.STDOUT)
        observed=re.findall(r'\bdone (\S+) prompt=',output)
        if sorted(observed)!=sorted(q['id'] for q in result):raise RuntimeError('An extra or missing HTTP done record in the pressure interval')
        c.emit('matched-http-completions',count=len(observed))
        return result
    c.cell=measured

def calibrate_nvml():
    out=here.parent/'hardware01'
    target=out/'nvml-scan.json'
    if target.is_file():return
    cfg.t.idle();before=cfg.t.server.metrics();start=time.time()
    def scan(which,binary,prefix):
        result=[]
        for _ in range(6):
            line=subprocess.check_output(prefix+[binary],text=True,timeout=40).strip()
            if not re.fullmatch(r'read,\d+,\d+,\d+',line):raise ValueError('Invalid scan output '+which)
            result.append(line)
        (out/(which+'-nvml-scan.csv')).write_text('\n'.join(result)+'\n')
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(scan,'head','/tmp/dsv41-pmc-1118/calibrate',[]),
              pool.submit(scan,'worker','/tmp/calibrate',cfg.policy.ranks[1])]
        for job in jobs:job.result()
    after=cfg.t.server.metrics();done=cfg.c.bench.DONE_KEY
    if before[done]!=after[done] or any(after['tensorfold:requests_'+k] for k in ('running','waiting')):
        raise RuntimeError('An inference overlapped the idle scans')
    result=dict(start=start,end=time.time(),completed_before=before[done],completed_after=after[done],
                runs_per_rank=6,running_after=0,waiting_after=0)
    target.write_text(json.dumps(result,indent=2)+'\n');cfg.c.emit('nvml-calibration',**result)


def hooks():
    spec=importlib.util.spec_from_file_location('prune_boundary',here.parent/'prune/campaign.py')
    package=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    for f in (package.pm_catalog,package.hot_kernel,calibrate_nvml):
        try:f()
        except Exception as exc:cfg.c.emit('boundary-reference-failure',function=f.__name__,error=str(exc))


def main(args):
    primary=here/'committed02/guard-384.exact-compare.log'
    if 'IDENTICAL (0 problem(s), 39 cases)' not in primary.read_text():
        raise RuntimeError('Primary exact gate is required')
    digest=hashlib.sha256((here/'committed.patch').read_bytes()).hexdigest()[:12]
    cfg.run.image='tensorfold-dsv41:bench-committed-'+digest
    cfg.run.controls.update((trial.guard,trial.fault))
    cfg.minimum='2048'
    out,original,ids,scratch=cfg.setup(args)
    hooks()
    _,cases=cfg.fixture()
    shifted_arrivals();settle_completed_burst()
    cfg.c.emit('arrival-plan',order=order,offsets_s=offsets)
    matches={}
    try:
        arms=(('guard-long','1','head-only',1024),('plain-long','0','head-only',1024),
              ('plain-healthy','0','both',384),('guard-healthy','1','both',384))
        if args.all_warm:
            arms=(('guard-warm','1','both',384),('plain-warm','0','both',384))
        for arm,enabled,coverage,budget in arms:
            cfg.restart(arm,'500','0',scratch,original,ids,extra={trial.guard:enabled,trial.fault:'1' if coverage=='head-only' else '0'})
            trial.inventory(scratch,coverage if args.all_warm else None)
            cfg.peak=lambda cases,scratch:trial.peak(cases,scratch,coverage,require_warm=args.all_warm)
            result=cfg.measure(cases,scratch,budget,order)
            if coverage in matches:cfg.c.emit('matched',measured=matches[coverage]==result)
            matches[coverage]=result
            cfg.t.validate(arm)
    except BaseException as exc:
        cfg.c.emit('failure',error=str(exc));raise
    finally:cfg.restore(original)
    cfg.c.emit('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);p.add_argument('--store',default='committed01')
    p.add_argument('--traffic-since',type=float);p.add_argument('--all-warm',action='store_true');a=p.parse_args()
    if Path(a.store).name!=a.store:p.error('Expected private store name')
    with (cfg.run.prior.parent/'.pair.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(a)
