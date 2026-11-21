"""Isolated best-runtime experiments on retained-state patience and natural completions."""
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

here=Path(__file__).resolve().parent
repo=Path.cwd()
prior=repo/'runs/engine-followup/kv'
sys.path.insert(0,str(prior))
import policy
p=policy.p
c=p.client
t=p.trial
image='tensorfold-dsv41:bench-held-adapt-9063970bbd02'
controls=set(t.candidate)|{'TF_DS_KV_HELD_WAIT_MS','TF_DS_FINISH_ADAPTIVE','TF_DS_FINISH_GRACE_MS','TF_DS_SPILL_MIN_TOKENS'}

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def identity():
    result=[]
    for rank in policy.ranks:
        raw=json.loads(subprocess.check_output(rank+['docker','inspect',policy.name],text=True,timeout=30))[0]
        cfg=raw['Config'];cmd=cfg['Cmd'][:];env=dict(s.split('=',1) for s in cfg['Env']); mount=sorted((m['Destination'],m['Source'],m['RW']) for m in raw['Mounts'])
        if not raw['State']['Running']:raise RuntimeError('Rank stopped')
        spill,=[s for d,s,r in mount if d=='/spill'];normalized=cmd[:];minimum=None
        if '--spill-min-tokens' in cmd:
            pos=cmd.index('--spill-min-tokens')+1;minimum=cmd[pos];normalized[pos]='MINIMUM'
        rest={k:v for k,v in env.items() if k not in controls|{'TF_DS_ENGINE_ID','TF_DS_RECIPE_PATCHES'}}
        result.append(dict(image=cfg['Image'],id=raw['Image'],spill=spill,minimum_arg=minimum,
                           env={k:env.get(k) for k in sorted(controls)},
                           common=digest([rest,normalized,[(d,s if d!='/spill' else 'SPILL',r) for d,s,r in mount],raw['HostConfig']['CpusetCpus']]),
                           full=digest([raw['Image'],cmd,env,mount,raw['HostConfig']['CpusetCpus']])))
    return result

def traffic(start):
    stop=time.time();stamp=datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat()
    cover=json.loads(subprocess.check_output(['docker','inspect',policy.name],text=True))[0]['State']['StartedAt']
    if datetime.datetime.fromisoformat(cover.replace('Z','+00:00')).timestamp()>start:
        raise RuntimeError('The current container does not cover the requested quiet interval')
    lines=subprocess.check_output(['docker','logs','--since',stamp,'--until',str(stop),policy.name],text=True,stderr=subprocess.STDOUT).splitlines()
    counts=collections.Counter()
    for l in lines:
        if 'done ' not in l or 'prefill_s=' not in l:continue
        f=dict(x.split('=',1) for x in l.split() if '=' in x)
        if not {'prompt','cached','queue_s','tokens','prefill_s'}<=f.keys():continue
        counts['completed']+=1;counts['deep100k']+=int(f['prompt'])>=100000;counts['queued1s']+=float(f['queue_s'])>=1;counts['restore']+=float(f.get('restore_s',0))>0
    c.emit('quiet',start=start,end=stop,completed=counts['completed'],counts=dict(counts))

def read_private(path,program,*options):
    """Read synthetic private metadata on the serving mount or an already-restored host store."""
    path=Path(path).resolve()
    if path.parent!=Path.home()/'.cache' or not path.name.startswith('dsv41-overnight-'):
        raise RuntimeError('Expected an isolated benchmark tier')
    result=[]
    for rank in policy.ranks:
        image=json.loads(subprocess.check_output(rank+['docker','inspect',policy.name],text=True))[0]
        mounts=[m['Source'] for m in image['Mounts'] if m['Destination']=='/spill']
        command=[x for x in rank if x!='-n']
        if mounts==[str(path)]:
            command+=['docker','exec','-i','--user','0',policy.name,'python3','-I','-','/spill',*options]
        else:
            command+=['python3','-I','-',str(path),*options]
        response=subprocess.run(command,input=program,text=True,stdout=subprocess.PIPE,check=True,timeout=60)
        result.append(json.loads(response.stdout))
    return result


def inventory(path):
    program="""import json,os,pathlib,sys
def fail(e):raise e
counts=[0,0];root=pathlib.Path(sys.argv[1])
if root.exists():
 for _,_,files in os.walk(root,onerror=fail):
  counts[0]+=sum(f.endswith('.man') for f in files);counts[1]+=sum(f.endswith('.ids') for f in files)
print(json.dumps(counts))"""
    return read_private(path,program)

def held_events(since):
    logs=subprocess.check_output(['docker','logs','--since',str(since),policy.name],text=True,stderr=subprocess.STDOUT)
    return [dict(ms=int(ms),elapsed_ms=int(e),part=int(part),at=int(at),patient=b=='true',outcome=o) for ms,e,part,at,b,o in policy.pattern.findall(logs)]

def balanced(prompts,cases,arm,scratch):
    if sum(p.rows())>32768 or min(c.memory())<5.2:raise RuntimeError('Bad initial occupancy or memory')
    for key in ('shared','f0','f1','f2'):
        p.probe(key,'fill',0,cases[key]);
        if key!='shared':p.probe(key,'pre-pressure',0,cases[key],p.grid)
    kept=p.rows();inv=inventory(scratch)
    if kept!=[2*p.grid,2*p.grid] or inv!=[[0,0],[0,0]]:raise RuntimeError(f'Unmatched retained rows or existing checkpoints {kept}, {inv}')
    c.emit('stage',stage='peak',rows=kept,inventory=inv);since=time.time()
    reqs=c.cell(300000,'admission',prompts,list(range(4200,4204)),budget=1024,rep=0,sharing='peak',expected_cached=p.grid)
    if any(r['tokens']!=1024 for r in reqs):raise RuntimeError('Expected equal work')
    events=held_events(since);c.emit('patient',events=events)
    if not events:raise RuntimeError('Measured path not armed')
    # f1 was the sole displaced session in each of the previous balanced arms. Probe it first to avoid cascade attribution.
    # Stop at the first lost hit; refilling it can itself evict later sessions.
    for key in ('f1','f0','f2'):
        returned=p.probe(key,'after-pressure',0,cases[key])
        if returned['cached'] < p.grid:
            break
    inv=inventory(scratch);c.emit('inventory',counts=inv)
    if inv!=[[0,0],[0,0]]:raise RuntimeError('Snapshot unexpectedly persisted')
    return [(r['input_sha256'],r['cached'],r['tokens'],r['done']['sha']) for r in reqs]

def exact(arm,out):
    rec=out/(arm+'.exact.json')
    for cmd,log in ((['record','--url',t.server.url,'--like',t.reference,'--label',arm,'--out',str(rec)],'record'),(['compare',t.reference,str(rec)],'compare')):
        if t.run([sys.executable,'-I','tools/exact.py',*cmd],arm+'.exact-'+log+'.log'):raise RuntimeError('Exact gate failed')
    c.emit('exact',passed=True)

def run(args):
    out=args.out.resolve();out.mkdir(exist_ok=False);t.out=out;c.health=p.health
    c.args=argparse.Namespace(arm='initial',salt='5',occupancy=True,replay_depth=False,finish_grace=False)
    original=identity();t.idle()
    if any(a for _,a in t.identity()):raise RuntimeError('Expected pinned production')
    if args.traffic_since:traffic(args.traffic_since)
    ids=[subprocess.check_output(rank+['docker','image','inspect','-f','{{.Id}}',image],text=True).strip() for rank in policy.ranks]
    prompts,cases=p.history();match=None;c.emit('initial',ranks=original)
    scratch=str(Path.home()/'.cache'/('dsv41-overnight-'+out.name));minimum='327680'
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for arm,wait,adaptive in (('control','10000','0'),('short','500','0')):
            c.args.arm=arm;t.idle()
            flags={**t.candidate,'TF_DS_KV_HELD_WAIT_MS':wait,'TF_DS_FINISH_ADAPTIVE':adaptive,'TF_DS_FINISH_GRACE_MS':'5000'}
            t.restart(arm,dict(flags,IMAGE=image,PREPARE='0',SPILL_DIR=scratch,WORKER_SPILL_DIR=scratch,SPILL_MIN_TOKENS=minimum));actual=identity()
            for i,rank in enumerate(actual):
                if rank['id']!=ids[i] or rank['common']!=original[i]['common'] or rank['spill']!=scratch:raise RuntimeError(f'Unexpected rank {i} recipe')
                for k,v in {**flags,'TF_DS_SPILL_MIN_TOKENS':minimum}.items():
                    if rank['env'][k]!=v:raise RuntimeError(f'Unexpected rank {i} {k}')
            if actual[0]['minimum_arg']!=minimum:raise RuntimeError('Rank-0 spill-min argument mismatch')
            c.emit('runtime',arm_name=arm,ranks=actual)
            seen=balanced(prompts,cases,arm,scratch)
            if match is not None:
                c.emit('matched',measured=seen==match)
                if seen!=match:raise RuntimeError('Input/length/cut/output drift')
            match=seen
            if args.exact and arm=='short':exact(arm,out)
            t.validate(arm)
    finally:
        if '_error' not in t.server.metrics():t.idle()
        t.restart('production',{'PREPARE':'0'})
        if identity()!=original:raise RuntimeError('Production restoration differs')
        t.validate('production');t.idle();c.emit('restored',ranks=identity(),health=p.health())
    c.emit('complete')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);parser.add_argument('--traffic-since',type=float);parser.add_argument('--exact',action='store_true');args=parser.parse_args()
    with (prior.parent/'.pair.lock').open('a') as file:
        fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)
