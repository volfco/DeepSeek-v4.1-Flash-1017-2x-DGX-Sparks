"""Run the whole five-arm natural-stop screen in one handed-off pair slot."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here))
import natural_minimal as trial

def publish(path, data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n')
    temp.replace(path)

def run(a):
    a.state.mkdir(exist_ok=False)
    if a.out.exists() or a.out.with_suffix('.log').exists():
        raise FileExistsError(a.out)
    # Capture the original both-rank fingerprint with the same flag vocabulary used by both campaigns.
    trial.cfg.run.controls.update((trial.committed.guard,trial.committed.fault))
    with (trial.cfg.run.prior.parent/'.pair.lock').open('a') as lock:
        while True:
            if a.gate.is_file():
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:pass
            time.sleep(2)
        initial=trial.cfg.run.identity();started=time.monotonic()
        trial.cfg.t.idle()
        publish(a.state/'acquired.json',dict(at=time.time(),gate=str(a.gate),pid=os.getpid(),out=str(a.out)))
        failure=None
        with a.out.with_suffix('.log').open('x',buffering=1) as log,contextlib.redirect_stdout(log):
            try:trial.main(a)
            except BaseException as e:
                failure=repr(e)
                traceback.print_exc(file=log)
        rows=[json.loads(row) for row in a.out.with_suffix('.log').read_text().splitlines() if row.startswith('{')]
        if not any(row['kind']=='restored' for row in rows) or trial.cfg.run.identity()!=initial:
            raise RuntimeError('Original production restoration needs inspection. '+str(failure))
        trial.cfg.t.idle()
        result=dict(at=time.time(),complete=rows[-1]['kind']=='complete',error=failure,restored=True,
                    path=str(a.out),arms=[r['arm'] for r in rows if r['kind']=='natural-outcome'])
        if result['complete'] and a.warm_repeat:
            remaining=a.slot_minutes*60-(time.monotonic()-started)
            # Two fresh matched arms previously took ~3m30s each and pinned restoration ~2m20s.
            if remaining>=610:
                warm=a.warm_repeat
                if warm.exists() or warm.with_suffix('.log').exists():raise FileExistsError(warm)
                publish(a.state/'warm-start.json',dict(path=str(warm),at=time.time(),available_s=remaining))
                err=None
                opts=argparse.Namespace(out=warm,store=a.store,traffic_since=None,all_warm=True)
                with warm.with_suffix('.log').open('x',buffering=1) as log,contextlib.redirect_stdout(log):
                    try:trial.confirm_committed.main(opts)
                    except BaseException as e:
                        err=repr(e)
                        traceback.print_exc(file=log)
                ledger=[json.loads(l) for l in warm.with_suffix('.log').read_text().splitlines() if l.startswith('{')]
                if not any(r['kind']=='restored' for r in ledger) or trial.cfg.run.identity()!=initial:
                    raise RuntimeError('All-warm crossover restoration requires inspection')
                trial.cfg.t.idle()
                result['warm']=dict(path=str(warm),complete=ledger[-1]['kind']=='complete',restored=True,error=err)
            else:
                result['warm']=dict(deferred=True,remaining_s=remaining)
            result['at']=time.time()
    publish(a.state/'released.json',result)
    if failure:raise RuntimeError(failure)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path);ap.add_argument('--gate',type=Path,required=True)
    ap.add_argument('--state',type=Path,required=True);ap.add_argument('--store',default='committed01')
    ap.add_argument('--warm-repeat',type=Path);ap.add_argument('--slot-minutes',type=float,default=35)
    ap.add_argument('--traffic-since',type=float);a=ap.parse_args()
    if Path(a.store).name!=a.store:ap.error('Expected private benchmark tier')
    run(a)
