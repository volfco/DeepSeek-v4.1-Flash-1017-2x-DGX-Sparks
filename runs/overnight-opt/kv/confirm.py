"""Run the best-runtime 3:1 held-state confirmation with one deliberately ineligible session."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parent))
import campaign as run

p,c,t,policy=run.p,run.c,run.t,run.policy
minimum=str(p.grid)
question='Summarize the preceding context in detail.'
short_grid=p.grid-2048


def fixture():
    prompts,cases=p.history()
    src=cases['f0'].split('\n',1)[1].removesuffix(question).rstrip()
    target=short_grid+528;cut=int(len(src)*target/300576)
    for _ in range(9):
        prompt='[overnight-private-short-f3]\n'+src[:cut]+'\n\n'+question
        n=c.token_count([{'role':'user','content':prompt}])
        if abs(target-n)<=4:break
        cut+=round((target-n)*cut/n)
    else:raise RuntimeError('Could not calibrate the ineligible victim')
    cases['f3']=prompt
    return prompts,cases


def groups(cases,scratch):
    tokens={}
    for key,prompt in cases.items():
        req=urllib.request.Request(t.server.url+'/tokenize',json.dumps({'messages':[{'role':'user','content':prompt[:3072]}],'chat_template_kwargs':{'enable_thinking':True}}).encode(),{'Content-Type':'application/json'})
        with c.bench.OPENER.open(req,timeout=20) as f:tokens[key]=json.load(f)['tokens']
    for count in (32,64,128,256,512,1024):
        if any(len(ids)<count for ids in tokens.values()):break
        prefixes={hashlib.sha256(struct.pack(f'<{count}I',*ids[:count])).hexdigest():key for key,ids in tokens.items()}
        if len(prefixes)==len(tokens):break
    if len(prefixes)!=len(tokens):raise RuntimeError('Ambiguous fixture token prefix')
    # Read only committed synthetic ID prefixes. Unreadable namespaces are an error.
    program="""import collections,hashlib,json,os,pathlib,sys
def fail(e):raise e
p=pathlib.Path(sys.argv[1]);data=collections.defaultdict(list)
if p.exists():
 for base,dirs,files in os.walk(p,onerror=fail):
  for name in files:
   if not name.endswith('.ids'):continue
   f=pathlib.Path(base)/name
   if not f.with_suffix('.man').is_file():continue
   with f.open('rb') as stream:d=hashlib.sha256(stream.read(int(sys.argv[2]))).hexdigest()
   data[d].append(f.stat().st_size//4)
print(json.dumps(data))"""
    inventories=run.read_private(scratch,program,str(4*count))
    return [{key:sorted(rank.get(digest,[])) for digest,key in prefixes.items()} for rank in inventories]


def restart(arm,wait,adaptive,scratch,original,ids,grace='5000',extra=None):
    t.idle();c.args.arm=arm
    flags={**t.candidate,'TF_DS_KV_HELD_WAIT_MS':wait,'TF_DS_FINISH_ADAPTIVE':adaptive,'TF_DS_FINISH_GRACE_MS':grace,**(extra or {})}
    t.restart(arm,dict(flags,IMAGE=run.image,PREPARE='0',SPILL_DIR=scratch,WORKER_SPILL_DIR=scratch,SPILL_MIN_TOKENS=minimum))
    actual=run.identity()
    for i,rank in enumerate(actual):
        if rank['id']!=ids[i] or rank['common']!=original[i]['common'] or rank['spill']!=scratch:raise RuntimeError(f'Unexpected rank {i} recipe')
        for k,v in {**flags,'TF_DS_SPILL_MIN_TOKENS':minimum}.items():
            if rank['env'][k]!=v:raise RuntimeError(f'Unexpected rank {i} {k}')
    if actual[0]['minimum_arg']!=minimum:raise RuntimeError('Rank-0 spill-min argument mismatch')
    logs=subprocess.check_output(['docker','logs',policy.name],text=True,stderr=subprocess.STDOUT)
    if 'kv held trial configured ms='+wait not in logs or 'lane host finish grace: '+grace+' ms' not in logs or ('lane host adaptive finish: on' in logs)!=(adaptive=='1'):raise RuntimeError('Scheduler startup banners differ')
    c.emit('runtime',arm_name=arm,ranks=actual)
    # External compiler processes can transiently consume unified memory during startup.
    # Wait for the established 300K headroom before issuing any request.
    deadline=time.monotonic()+240
    while True:
        free=c.memory()
        if min(free)>=5.2:break
        c.emit('memory-wait',free_gib=free)
        if time.monotonic()>deadline:raise RuntimeError(f'Preflight memory remained below 300K gate: {free} GiB')
        time.sleep(5)


def kept_gate(lo,hi):
    rows=p.rows()
    if not (lo<=rows[0]<=lo+32768 and rows[1]==hi):raise RuntimeError(f'Unexpected kept rows {rows}')
    return rows


def seed(cases,scratch):
    if sum(p.rows())>32768:raise RuntimeError('Expected initial state')
    for key in ('shared','f0','f1'):
        p.probe(key,'seed',0,cases[key])
    expected=['shared','f0','f1'];deadline=time.monotonic()+120
    while True:
        inv=groups(cases,scratch)
        if all(all(p.grid in rank[k] for k in expected) for rank in inv):break
        if time.monotonic()>deadline:raise RuntimeError(f'Full snapshots not persisted {inv}')
        time.sleep(3)
    c.emit('seed-inventory',groups=inv)


def peak(cases,scratch):
    c.args.finish_grace=False
    if sum(p.rows())>32768 or min(c.memory())<5.2:raise RuntimeError('Unexpected initial occupancy/memory')
    pre={}
    for key,cut in (('f0',p.grid),('f3',short_grid),('shared',p.grid),('f1',p.grid)):
        initial=p.probe(key,'fill',0,cases[key],0 if key=='f3' else p.grid)
        if key=='f0':kept_gate(0,p.grid)
        else:pre[key]=p.probe(key,'pre-pressure',0,cases[key],cut)
    rows=kept_gate(2*p.grid+short_grid,p.grid);inv=groups(cases,scratch)
    if any(rank['f3'] or any(p.grid not in rank[k] for k in ('shared','f0','f1')) for rank in inv):raise RuntimeError(f'Unexpected per-family disk coverage {inv}')
    c.emit('stage',stage='peak',rows=rows,groups=inv)
    return rows,pre


def measure(cases,scratch,budget,order=(0,1,2,3)):
    rows,pre=peak(cases,scratch)
    prompts=[cases['f0'].removesuffix(question)+c.tasks[i] for i in order];since=time.time()
    reqs=c.cell(300000,'admission',prompts,[4200+i for i in order],budget=budget,rep=0,sharing='peak',expected_cached=p.grid)
    if any(r['tokens']!=budget for r in reqs):raise RuntimeError('Unequal generated work')
    events=run.held_events(since);c.emit('patient',events=events)
    if not events:raise RuntimeError('Measured held path absent')
    if 'TF_DS_BENCH_PREFER_ELIGIBLE' in run.controls:
        log=subprocess.check_output(['docker','logs','--since',str(since),policy.name],text=True,stderr=subprocess.STDOUT)
        c.emit('guard',decisions=[l.split('kv eligibility trial ',1)[1] for l in log.splitlines() if 'kv eligibility trial ' in l])
    if 'TF_DS_BENCH_PREFER_COMMITTED' in run.controls:
        log=subprocess.check_output(['docker','logs','--since',str(since),policy.name],text=True,stderr=subprocess.STDOUT)
        c.emit('guard',decisions=[l.split('kv committed trial ',1)[1] for l in log.splitlines() if 'kv committed trial ' in l],
               probes=[l.split('kv committed probe ',1)[1] for l in log.splitlines() if 'kv committed probe ' in l])
    for key in ('f3','shared','f1'):
        p.probe(key,'after-pressure',0,cases[key])
    return rows,[(r['input_sha256'],r['cached'],r['tokens'],r['done']['sha']) for r in reqs]


def setup(args):
    out=args.out.resolve();out.mkdir(exist_ok=False);t.out=out;c.health=p.health
    c.args=argparse.Namespace(arm='initial',salt='5',occupancy=True,replay_depth=False,finish_grace=False)
    original=run.identity();t.idle()
    if any(a for _,a in t.identity()):raise RuntimeError('Expected pinned serving pair')
    if args.traffic_since:run.traffic(args.traffic_since)
    ids=[subprocess.check_output(rank+['docker','image','inspect','-f','{{.Id}}',run.image],text=True).strip() for rank in policy.ranks]
    scratch=str(Path.home()/'.cache'/('dsv41-overnight-'+args.store))
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    c.emit('initial',ranks=original)
    return out,original,ids,scratch


def restore(original):
    if '_error' not in t.server.metrics():t.idle()
    t.restart('production',{'PREPARE':'0'})
    if run.identity()!=original:raise RuntimeError('Production restoration differs')
    t.validate('production');t.idle();c.emit('restored',ranks=run.identity(),health=p.health())


def main(args):
    out,original,ids,scratch=setup(args);_,cases=fixture();matches={}
    try:
        saved=groups(cases,scratch)
        if all(all(p.grid in rank[k] for k in ('shared','f0','f1')) for rank in saved):
            c.emit('seed-inventory',groups=saved,reused=True)
        else:
            restart('seed','10000','0',scratch,original,ids);seed(cases,scratch)
        for arm,wait,budget in (('short','500',1024),('control-384','10000',384),('short-384','500',384),('control','10000',1024)):
            restart(arm,wait,'0',scratch,original,ids)
            result=measure(cases,scratch,budget)
            if budget in matches:
                c.emit('matched',budget=budget,measured=result==matches[budget])
                if result!=matches[budget]:raise RuntimeError('Input, output or occupancy drift')
            matches[budget]=result
            if arm=='short':run.exact(arm,out)
            t.validate(arm)
    except BaseException as exc:
        c.emit('failure',error=str(exc));raise
    finally:restore(original)
    c.emit('complete')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--traffic-since',type=float);ap.add_argument('--store',default='asym01');args=ap.parse_args()
    if Path(args.store).name!=args.store:ap.error('Expected an isolated store name')
    with (run.prior.parent/'.pair.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(args)
