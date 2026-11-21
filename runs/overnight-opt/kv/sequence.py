"""Start the isolated overnight policy and quality pilots after the preceding owner's restore."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import analyze

repo = Path.cwd()
lock_path = repo/'runs/engine-followup/.pair.lock'

class Interleave(Exception):
    pass


def event(kind, **values):
    print(json.dumps(dict(kind=kind, at=time.time(), **values)), flush=True)


def idle_lock():
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def main(args):
    directory = args.out.resolve()
    directory.mkdir(exist_ok=False)
    if Path(args.suffix).name != args.suffix or not args.suffix.isalnum():
        raise ValueError('Use a plain campaign suffix')
    expires = datetime.datetime.fromisoformat(args.stop_by.replace('Z','+00:00')).timestamp()
    release_by = datetime.datetime.fromisoformat(args.return_by.replace('Z','+00:00')).timestamp()
    outcomes = []
    event('waiting', gate=str(args.gate))
    while not (args.gate.is_file() and idle_lock()):
        if time.time() >= expires:
            raise TimeoutError('The preceding campaign did not release in the allotted window')
        time.sleep(4)
    event('acquired')

    def campaign(script, dest, options, minutes):
        if time.time() >= expires:
            raise TimeoutError('No time for a new campaign')
        if time.time()+minutes*60>release_by or (directory/'handoff-after-current').exists():
            raise Interleave(dest.name)
        log = dest.with_suffix('.log')
        if dest.exists() or log.exists():
            raise RuntimeError('Refusing to overwrite '+str(dest))
        event('launch', campaign=dest.name, options=options)
        with log.open('x') as output:
            status = subprocess.run([sys.executable,'-I','-u',str(script),str(dest),*options],
                                    stdout=output,stderr=subprocess.STDOUT).returncode
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.startswith('{')]
        restored = any(r.get('kind')=='restored' for r in rows)
        finished = bool(rows) and rows[-1].get('kind')=='complete' and status==0
        event('outcome',campaign=dest.name,rc=status,restored=restored,complete=finished)
        if not restored:
            raise RuntimeError('Original restoration requires inspection for '+str(dest))
        outcomes.append(dict(campaign=dest.name,complete=finished,restored=restored))
        return log if finished else None

    def run_work():
        suf = args.suffix
        shared = ['--store','eligible'+suf]
        pilot = None
        if args.committed:
            campaign(here/'committed.py',here/('committed'+suf),['--store','committed'+suf],40)
        elif not args.skip_policy:
            pilot=campaign(here/'eligible.py',here/('eligible'+suf),shared+['--only','plain-384','guard-384'],30)
        promising = False
        if pilot:
            try:
                result = analyze.measure(pilot,384,['plain-384','guard-384'])
                (directory/'guard-pilot.json').write_text(json.dumps(result,indent=2)+'\n')
                plain,guard = [next(x for x in result['arms'] if x['arm']==n) for n in ('plain-384','guard-384')]
                promising = (plain['first_victim']['cached']==0 and guard['first_victim']['cached']==296960
                             and guard['first_victim']['output_match'] and bool(guard['guard'])
                             and result['burst_plus_revisits_saved_s']>0)
                event('guard-screen',promising=promising,serving_saved_s=result['burst_plus_revisits_saved_s'])
            except (AssertionError,KeyError,ValueError) as exc:
                event('guard-screen',promising=False,error=repr(exc))
        if promising:
            campaign(here/'eligible.py',here/('eligible-confirm'+suf),
                     shared+['--balanced','--only','guard-1024','plain-1024','plain-balanced','guard-balanced'],35)
        if not args.skip_quality:
            widths='-'.join(str(w) for w in args.quality_widths)
            minutes={8:32,4:45,1:90}
            campaign(here.parent/'prune/campaign.py',here.parent/'prune'/('width'+widths+'-'+suf),
                     ['--widths',*map(str,args.quality_widths),'--seeds',*map(str,args.quality_seeds)],
                     sum(minutes[w] for w in args.quality_widths)*len(args.quality_seeds))
        for factor in args.factors:
            finished=campaign(here/'interaction.py',here/('interact'+suf+'-'+factor),['--store','asym01','--only',factor],30)
            if not finished:
                raise Interleave('inspect natural calibration or missing measurements in '+factor)
    pending=None
    try:
        run_work()
    except Interleave as exc:
        pending=str(exc)
    if not idle_lock():
        raise RuntimeError('Unexpected competing pair owner at release')
    record=dict(at=time.time(),outcomes=outcomes,next=pending,lock_free=True)
    temp=directory/'released.tmp'
    temp.write_text(json.dumps(record,indent=2)+'\n')
    temp.replace(directory/'released.json')
    event('released',next=pending)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path)
    ap.add_argument('--gate',type=Path,required=True)
    ap.add_argument('--suffix',default='01')
    ap.add_argument('--skip-policy',action='store_true')
    ap.add_argument('--committed',action='store_true')
    ap.add_argument('--skip-quality',action='store_true')
    ap.add_argument('--quality-widths',type=int,nargs='+',choices=(1,4,8),default=[8])
    ap.add_argument('--quality-seeds',type=int,nargs='+',default=[6100])
    factors=('ten-fixed-a','short-adaptive','short-fixed','ten-adaptive','ten-fixed-b')
    ap.add_argument('--factors',choices=factors,nargs='*',default=factors)
    ap.add_argument('--stop-by',default='2026-11-18T15:52:00Z')
    ap.add_argument('--return-by',default='2026-11-18T12:00:00Z')
    main(ap.parse_args())
