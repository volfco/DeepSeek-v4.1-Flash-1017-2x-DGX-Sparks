"""Time all-rank committed-presence protection at the production 2048-token persistence floor."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here))
import confirm as cfg

guard='TF_DS_BENCH_PREFER_COMMITTED'
fault='TF_DS_BENCH_SKIP_FOLLOWER_296960'
# Actual full token-ID files from independent tagged seed requests (shared, f0, f1).
# Each 299008-token digest was reproduced on the preceding study's separate store.
healthy={'shared':'60b1082e7a1bcd33dd8414992754e189080cc787ec10bb60d5eead6885362b6a',
         'f0':'4adb5eebbd51eb45512b66baff2b6ab1f5f621752a80b3aa9ef79891c1563896',
         'f1':'a4bfeac1f516fab18a5d7cc3079e7b3dc8359b79a4c47a7a5c129c2a06dcc6f7'}
prefix={'shared':'c5b05300dc36867c9c7dd95f88ca84117b925932d8b1ae1d6d6e2f5c22b2c65c',
        'f0':'060f9c4cf04a4128fcf0cd04cdaf43e0b5ea2fdb0dc23a8b06fbd256b45c6e6f',
        'f1':'bcf881ac39e10e6f95b9d1a9629ad21cc9fb0a6b2e86efeaf6b151f8bc1bf492'}
f3_digest=None


def disk(scratch):
    if not scratch.startswith(str(Path.home()/'.cache/dsv41-overnight-')):raise ValueError('Expected private benchmark tier')
    script='''import pathlib,json,hashlib,sys,os
out=[]
def fail(e):raise e
for base,dirs,files in os.walk('/spill',onerror=fail):
 for name in files:
  if not name.endswith('.ids'):continue
  f=pathlib.Path(base)/name
  if not f.with_suffix('.man').is_file(): continue
  n=f.stat().st_size//4
  if n not in (296960,299008): continue
  b=f.read_bytes();out.append([n,hashlib.sha256(b[:256]).hexdigest(),hashlib.sha256(b).hexdigest()])
print(json.dumps(out))'''
    inventories=[]
    for pre in cfg.policy.ranks:
        actual=json.loads(subprocess.check_output(pre+['docker','inspect',cfg.policy.name],text=True))[0]
        mounts=[m['Source'] for m in actual['Mounts'] if m['Destination']=='/spill']
        if mounts!=[scratch]:raise RuntimeError('Refusing to inspect a different spill tier')
        cmd=[x for x in pre if x!='-n']+['docker','exec','-i','--user','0',cfg.policy.name,'python3','-I','-']
        q=subprocess.run(cmd,input=script,text=True,check=True,stdout=subprocess.PIPE,timeout=30)
        inventories.append(json.loads(q.stdout))
    return inventories


def inventory(scratch,coverage=None):
    global f3_digest
    deadline=time.monotonic()+120
    while True:
        raw=disk(scratch)
        ok=all(all(any(n==cfg.p.grid and key==healthy[k] and h==prefix[k] for n,h,key in rank) for k in healthy) for rank in raw)
        candidates=[[key for n,h,key in rank if n==cfg.short_grid and h not in prefix.values()] for rank in raw]
        matched=(len(candidates[0])==1 and (not candidates[1] if coverage=='head-only' else candidates[1]==candidates[0]))
        if ok and (coverage is None or matched):
            if coverage is not None:
                digest=candidates[0][0]
                if f3_digest not in (None,digest):raise RuntimeError('Victim IDs changed across policy arms')
                f3_digest=digest
            return raw
        if time.monotonic()>deadline:raise RuntimeError(f'Unmatched private-tier digest inventory: {raw}')
        time.sleep(2)


def peak(cases,scratch,coverage='head-only',require_warm=False):
    p,c=cfg.p,cfg.c
    if sum(p.rows())>32768 or min(c.memory())<5.2:raise RuntimeError('Unexpected starting occupancy/memory')
    for key,cut in (('f0',p.grid),('f3',cfg.short_grid),('shared',p.grid),('f1',p.grid)):
        first=p.probe(key,'fill',0,cases[key],(cut if require_warm else None) if key=='f3' else p.grid)
        if require_warm and first['cached']!=cut:raise RuntimeError('All-warm confirmation expected a full restored point: '+key)
        if key=='f0':cfg.kept_gate(0,p.grid)
        else:p.probe(key,'pre-pressure',0,cases[key],cut)
    rows=cfg.kept_gate(2*p.grid+cfg.short_grid,p.grid)
    raw=inventory(scratch,coverage)
    c.emit('stage',stage='peak',rows=rows,groups=raw,coverage=coverage,victim_digest=f3_digest)
    return rows,{}


def main(args):
    digest=hashlib.sha256((here/'committed.patch').read_bytes()).hexdigest()[:12]
    cfg.run.image='tensorfold-dsv41:bench-committed-'+digest
    cfg.run.controls.update((guard,fault))
    cfg.minimum='2048';cfg.peak=peak
    out,original,images,scratch=cfg.setup(args);_,cases=cfg.fixture()
    seen=None
    try:
        cfg.restart('seed','10000','0',scratch,original,images,extra={guard:'0',fault:'1'})
        if not all(all(any(n==cfg.p.grid and key==healthy[k] and h==prefix[k] for n,h,key in rank) for k in healthy) for rank in disk(scratch)):
            for key in ('shared','f0','f1'):cfg.p.probe(key,'seed',0,cases[key])
        cfg.c.emit('seed-inventory',groups=inventory(scratch))
        for arm,wait,enabled in (('plain-384','500','0'),('guard-384','500','1'),('ten-384','10000','0')):
            if args.only and arm not in args.only:continue
            cfg.restart(arm,wait,'0',scratch,original,images,extra={guard:enabled,fault:'1'})
            heads=subprocess.check_output(['docker','logs',cfg.policy.name],text=True,stderr=subprocess.STDOUT)
            if ('kv trial committed guard enabled' in heads)!=(enabled=='1'):raise RuntimeError('Guard startup banner differs')
            result=cfg.measure(cases,scratch,384)
            if seen is not None:cfg.c.emit('matched',measured=result==seen)
            seen=result
            if enabled=='1':cfg.run.exact(arm,out)
            cfg.t.validate(arm)
    except BaseException as e:
        cfg.c.emit('failure',error=str(e));raise
    finally:cfg.restore(original)
    cfg.c.emit('complete')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('out',type=Path);ap.add_argument('--store',default='committed01')
    ap.add_argument('--only',nargs='+',choices=('plain-384','guard-384','ten-384'));ap.add_argument('--traffic-since',type=float);args=ap.parse_args()
    if Path(args.store).name!=args.store:ap.error('Expected private store name')
    with (cfg.run.prior.parent/'.pair.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(args)
