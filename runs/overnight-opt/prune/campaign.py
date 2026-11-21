"""Run the harder pruning probe on the pair with separate private stores and deterministic restoration."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

here=Path(__file__).resolve().parent;shared=Path.cwd()/'runs/engine-followup';sys.path.insert(0,str(shared))
import driver as d

t=d.trial
flag='TF_DS_BENCH_PRUNE_REPORT';image='tensorfold-dsv41:bench-prune-report-ab9624920b70'

def pm_catalog():
    root=here.parent/'hardware01';target=root/'pm-metrics.txt'
    if not (root/'pm-metrics-requested').is_file() or target.exists():return
    t.idle();a=t.server.metrics()
    running,waiting='tensorfold:requests_running','tensorfold:requests_waiting'
    if a.get(running)!=0 or a.get(waiting)!=0:raise RuntimeError('Busy at PM catalog boundary')
    cmd=['sudo','-n','/tmp/dsv41-pmc-1118/pm_sampling','-d','0','-q']
    response=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True,timeout=45)
    b=t.server.metrics();counts=t.bench_div.DONE_KEY
    if a[counts]!=b[counts] or b.get(running)!=0 or b.get(waiting)!=0:raise RuntimeError('Serving activity during PM query')
    if not response.stdout.strip():raise RuntimeError('Empty PM catalog')
    target.write_text(response.stdout);d.event('pm-catalog',rows=len(response.stdout.splitlines()),running=0,waiting=0,completed_delta=b[counts]-a[counts])

def hot_kernel():
    out=here.parent/'hardware01';record=out/'hot-plan-build.json'
    if not (out/'hot-plan-build-requested').is_file() or record.exists():return
    t.idle();before=t.server.metrics()
    deadline=time.monotonic()+120
    while any(subprocess.run(['pgrep','-x',tool],stdout=subprocess.DEVNULL).returncode==0 for tool in ('nvcc','cicc','ptxas')):
        if time.monotonic()>=deadline:raise RuntimeError('Another CUDA compile is still active')
        time.sleep(2)
    output=Path('/tmp/dsv41-hot-plan-ops.fatbin')
    started=time.time()
    with Path('/tmp/dsv41-hot-plan-ops.build.log').open('w') as log:
        proc=subprocess.run(['bash','/tmp/build-hot-kernel.sh','/tmp/dsv41_ops.cu',str(output)],stdout=log,stderr=subprocess.STDOUT,timeout=240)
    after=t.server.metrics();delta=after[t.bench_div.DONE_KEY]-before[t.bench_div.DONE_KEY]
    evidence=dict(rc=proc.returncode,elapsed_s=time.time()-started,bytes=output.stat().st_size if output.exists() else 0,
                  completed_delta=delta,running=after.get('tensorfold:requests_running'),waiting=after.get('tensorfold:requests_waiting'))
    record.write_text(json.dumps(evidence,indent=2)+'\n');d.event('hot-plan-build',**evidence)

def hot_package():
    hardware=here.parent/'hardware01'
    record=hardware/'hot-plan-package.json'
    if record.exists() or not (hardware/'hot-plan-package-requested').is_file():return
    digest='0ec939025482'
    package=Path('/tmp')/f'gb10-opt-{digest}.tar.gz'
    if not package.is_file():raise RuntimeError('Requested archive has not landed')
    t.idle();before=t.server.metrics();start=time.time()
    with (hardware/'hot-plan-package.log').open('w') as log:
        subprocess.run(['bash',str(here.parent/'pack-image.sh'),str(package),digest],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    after=t.server.metrics();delta=after[t.bench_div.DONE_KEY]-before[t.bench_div.DONE_KEY]
    proof=here.parent/('image-'+digest)
    left=(proof/'head.sha256').read_text();right=(proof/'worker.sha256').read_text()
    if left!=right or len(left.splitlines())!=2:raise RuntimeError('Executable readback evidence differs')
    if delta or after.get('tensorfold:requests_running') or after.get('tensorfold:requests_waiting'):raise RuntimeError('Serving activity during packaging')
    result=dict(rc=0,elapsed_s=time.time()-start,completed_delta=delta,executables=left.splitlines())
    record.write_text(json.dumps(result,indent=2)+'\n');d.event('hot-plan-package',**result)

def run(a):
    if any(a.item_count%w for w in a.widths):raise ValueError('Selected task count must be divisible by each width')
    if min(a.sql_budget,a.tool_budget)<1:raise ValueError('Output limits must be positive')
    out=a.out.resolve();out.mkdir(exist_ok=False);t.out=out;d.switches.add(flag);original=d.inspect();t.idle()
    if any(r['tag']!=d.base or any(r['flags'][k] for k in t.candidate) for r in original):raise RuntimeError('Expected pinned production')
    ids=[subprocess.check_output(pre+['docker','image','inspect','-f','{{.Id}}',image],text=True).strip() for pre in d.prefixes]
    d.event('initial',ranks=original)
    try:pm_catalog()
    except (OSError,RuntimeError,subprocess.SubprocessError) as e:d.event('pm-catalog-error',error=str(e))
    try:hot_kernel()
    except (OSError,RuntimeError,subprocess.SubprocessError) as e:d.event('hot-plan-build-error',error=str(e))
    try:hot_package()
    except (OSError,RuntimeError,subprocess.SubprocessError) as e:d.event('hot-plan-package-error',error=str(e))
    try:
        spec=importlib.util.spec_from_file_location('nvml_reference',here.parent/'kv/nvml_reference.py')
        nvml=importlib.util.module_from_spec(spec);spec.loader.exec_module(nvml)
        scan=nvml.capture(t,d.prefixes)
        if scan:d.event('nvml-scan',**scan)
    except (OSError,RuntimeError,subprocess.SubprocessError) as e:d.event('nvml-scan-error',error=str(e))
    def stop(*_):raise KeyboardInterrupt
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    extra=here/'confirmation-text.jsonl';items=out/'confirmation-items.jsonl';reference=None
    if hashlib.sha256(extra.read_bytes()).hexdigest()!='e82d67a1e6c06bf4701aa6787bf304062ae9802ba5a42eac862db0a31990b562':
        raise ValueError('The independently frozen 20-source confirmation set changed')
    try:
        for arm,prune,cap in (('current-a','1','0'),('off','0','0'),('cap12','1','12,1'),('current-b','1','0')):
            if a.only and arm not in a.only:continue
            scratch=str(Path.home()/'.cache'/('dsv41-hard-prune-'+out.name+'-'+arm));flags=dict(t.candidate,TF_DS_T13_EXPERT_PRUNE=prune,TF_DS_T13_EXPERT_CAP=cap,TF_DS_BENCH_PRUNE_REPORT='1',TF_DS_SCORE='1')
            t.idle();t.restart(arm,dict(flags,IMAGE=image,PREPARE='0',SPILL_DIR=scratch,WORKER_SPILL_DIR=scratch));actual=d.inspect()
            for i,r in enumerate(actual):
                if r['id']!=ids[i] or r['common']!=original[i]['common'] or r['spill']!=scratch or any(r['flags'][k]!=v for k,v in flags.items()):raise RuntimeError(f'Rank {i} configuration mismatch')
            d.event('runtime',arm=arm,ranks=actual)
            if t.run([sys.executable,'-I',str(here/'hard.py'),'--widths',*map(str,a.widths),'--seeds',*map(str,a.seeds),
                      '--item-count',str(a.item_count),'--sql-budget',str(a.sql_budget),'--tool-budget',str(a.tool_budget)],arm+'.jsonl'):raise RuntimeError('Hard probe failed '+arm)
            rows=[json.loads(l) for l in (out/(arm+'.jsonl')).read_text().splitlines() if l.startswith('{')]
            rt=[r for r in rows if r['kind']=='routing']
            expected=2*len(a.seeds)*len(a.widths)
            if rows[-1]['kind']!='complete' or len(rt)!=expected or any((r['delta'] is None)!=(prune=='0') for r in rt):raise RuntimeError('Incomplete routing evidence '+arm)
            if not a.behavior_only:
                refs=shared/'runtime01/items.jsonl';prior=shared/'runtime-score01/current-a.score.json'
                if t.run([sys.executable,'-I','tools/score.py','record','--url',t.server.url,'--in',str(refs),'--like',str(prior),'--out',str(out/(arm+'.score.json')),'--label',arm],arm+'.score.log'):raise RuntimeError('Frozen held-out NLL failed '+arm)
                if not items.exists() and t.run([sys.executable,'-I',str(shared/'references.py'),str(items),'--mint',str(extra)],'confirmation-mint.log'):
                    raise RuntimeError('Could not mint the independently frozen set')
                scored=out/(arm+'.confirmation.json')
                cmd=[sys.executable,'-I','tools/score.py','record','--url',t.server.url,'--in',str(items),'--out',str(scored),'--label',arm]
                if reference:cmd.extend(['--like',str(reference)])
                if t.run(cmd,arm+'.confirmation.log'):raise RuntimeError('Disjoint-source NLL failed '+arm)
                if reference is None:reference=scored
            cmd=[sys.executable,'-I','tools/validate.py',t.server.url,'--model',t.server.model,'--save',str(out/(arm+'.validate.json')),'--compare','runs/matrix/base.validate.json']
            val=t.run(cmd,arm+'.validate.log');d.event('end',arm=arm,validation_rc=val)
            if val and arm.startswith('current'):raise RuntimeError('Incumbent validation failed')
    finally:
        if '_error' not in t.server.metrics():t.idle()
        t.restart('production',{'PREPARE':'0'});final=d.inspect()
        if final!=original:raise RuntimeError('Production restoration mismatch')
        t.validate('production');t.idle();d.event('restored',ranks=final)
    d.event('complete')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--only',nargs='+');ap.add_argument('--seeds',nargs='+',type=int,default=[6100,6200]);ap.add_argument('--widths',nargs='+',type=int,choices=[1,4,8],default=[1,4,8])
    ap.add_argument('--item-count',type=int,choices=range(1,9),default=8)
    ap.add_argument('--sql-budget',type=int,default=1536);ap.add_argument('--tool-budget',type=int,default=900)
    ap.add_argument('--behavior-only',action='store_true',help='Skip the two separate single-stream teacher-forcing sets in a width-specific behavioral follow-up.')
    a=ap.parse_args()
    with (shared/'.pair.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a)
