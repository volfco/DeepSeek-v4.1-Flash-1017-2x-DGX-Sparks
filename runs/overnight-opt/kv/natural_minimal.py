"""Five-arm 300K held-patience by finish-grace screen with a frozen naturally stopping lead."""
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
import committed
import confirm_committed
import interaction as workload

cfg=workload.cfg
c,p,n=cfg.c,cfg.p,workload.n
factors=(('ten-fixed-a','10000','0'),('short-adaptive','500','1'),('short-fixed','500','0'),
         ('ten-adaptive','10000','1'),('ten-fixed-b','10000','0'))
seeds=(6511,4201,4202,4203)
lead_questions=(*workload.short_leads,
    'Which number is larger, 17 times 8 or 19 times 7? Give both products in one sentence.',
    'What is 42 minus 17? In exactly one sentence, give the answer and check it by adding 17.',
    'In two short sentences, explain why the average of 2, 3 and 7 is 4.')


def qualify_lead(ctx,depth,grid,golden):
    indices=[golden[depth][0]] if depth in golden else range(len(lead_questions))
    for index in indices:
        prompt=lead_questions[index]
        q=n.cell(depth,'chat-prime',[ctx+prompt],[seeds[0]],budget=768,sharing='natural-calibration',
                 rep=index,expected_cached=grid,allow_early_stop=True)[0]
        span=q['last']-q['first']
        signature=(index,q['tokens'],q['done']['sha'])
        if depth in golden:
            if q['finish']!='stop' or golden[depth]!=signature:
                raise RuntimeError('Frozen solo lead changed across policies')
        elif not (q['finish']=='stop' and 4<=q['tokens']<=225 and 1.05<=span<=5):
            continue
        golden[depth]=signature
        c.emit('calibration',depth=depth,index=index,tokens=q['tokens'],output_sha=q['done']['sha'],solo_stream_span_s=span)
        return q['tokens']+24,prompt
    raise RuntimeError('No frozen natural lead covers the three scheduled follower arrivals')


def backed(scratch):
    deadline=time.monotonic()+150
    while True:
        raw=committed.disk(scratch)
        full=[{key for count,_,key in r if count==p.grid} for r in raw]
        seed=set(committed.healthy.values())
        if full[0]==full[1] and seed<=full[0] and len(full[0]-seed)==1:
            return sorted(full[0])
        if time.monotonic()>deadline:raise RuntimeError(f'Expected four distinct, jointly committed full histories: {full}')
        time.sleep(3)


def main(args):
    cfg.run.image='tensorfold-dsv41:bench-committed-'+hashlib.sha256((here/'committed.patch').read_bytes()).hexdigest()[:12]
    cfg.run.controls.update((committed.guard,committed.fault))
    cfg.minimum='2048'
    out,original,ids,scratch=cfg.setup(args)
    confirm_committed.settle_completed_burst(n)
    _,cases=cfg.fixture();ctx=cases['shared'].removesuffix(cfg.question);golden={};primary=None
    try:
        for arm,wait,adaptive in factors:
            cfg.restart(arm,wait,adaptive,scratch,original,ids,extra={committed.guard:'0',committed.fault:'0'})
            workload.natural_args(arm)
            tight,prompt=workload.stage_balanced(cases,ctx,golden,True,qualify_lead)
            full=backed(scratch)
            c.emit('qualified',rows=p.rows(),full_points=full,lead_budget=tight,followers_budget=512,seeds=seeds,
                   follower_delays_after_first_s=[.20,.55,.90])
            since=time.time()
            replies=n.cell(300000,'admission',[ctx+q for q in (prompt,*c.tasks[1:4])],list(seeds),
                           budget=512,lead_budget=tight,mixed=True,sharing='natural-tight-pressure',rep=0,
                           expected_cached=p.grid,allow_early_stop=True)
            lead=replies[0];last=lead['start']+lead['total']
            c.emit('natural-outcome',depth=300000,shape='natural-tight-pressure',rep=0,lead_limit=tight,
                   requested=[tight,512,512,512],emitted=[q['tokens'] for q in replies],remaining_at_stop=tight-lead['tokens'],
                   follower_arrival_after_first_s=[q['start']-lead['first'] for q in replies[1:]],
                   follower_arrival_to_lead_end_s=[last-q['start'] for q in replies[1:]])
            if lead['finish']!='stop' or lead['tokens']>=tight:raise RuntimeError('Lead did not stop naturally below its fixed budget')
            signature=[(r['input_sha256'],r['cached'],r['finish'],r['tokens'],r['done']['sha']) for r in replies]
            if primary is None:primary=signature
            c.emit('matched',inputs=all(a[:2]==b[:2] for a,b in zip(signature,primary)),measured=signature==primary)
            c.emit('patient',events=cfg.run.held_events(since));workload.extensions(since)
            for key in ('f1','f0','f2'):p.probe(key,'after-pressure',0,cases[key])
            if arm=='short-adaptive':cfg.run.exact(arm,out)
            cfg.t.validate(arm)
        # All five pressure windows have ended. Inspect only the isolated synthetic stores before unmounting.
        try:
            program=(here/'lease-build/footprint.py').read_text()
            ranks=cfg.run.read_private(scratch,program)
            (out/'lease-footprint.json').write_text(json.dumps(ranks,indent=2)+'\n')
            c.emit('lease-footprint',ranks=[len(r['points']) for r in ranks])
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
            c.emit('lease-footprint-error',error=str(exc))
    except BaseException as exc:
        c.emit('failure',error=str(exc));raise
    finally:cfg.restore(original)
    c.emit('complete')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('out',type=Path);ap.add_argument('--store',default='committed01')
    ap.add_argument('--traffic-since',type=float);a=ap.parse_args()
    if Path(a.store).name!=a.store:ap.error('Expected private trial tier')
    with (cfg.run.prior.parent/'.pair.lock').open('a') as pair:
        fcntl.flock(pair,fcntl.LOCK_EX|fcntl.LOCK_NB);main(a)
