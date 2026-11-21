"""Cross held-prefix patience and finish admission with naturally stopping sampled leads."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here))
import confirm as cfg

run,p,c,t=cfg.run,cfg.p,cfg.c,cfg.t
src=here/'sample-natural.py'
if not src.exists():src=Path.cwd()/'runs/review-round/sample.py'
spec=importlib.util.spec_from_file_location('natural_client',src);n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
n.health=p.health
lead='Return the numeral 4 and no other text. The question is 2+2.'
short_leads=(lead,'Repeat the word Blue once. Answer with just that one word.',
             'Which single lowercase letter follows y in the English alphabet? Reply with that letter alone.')
moderate='List seven tests for a bounded Python queue. For each give the setup, action and expected result in one sentence. Finish after seven tests.'
followers=('Which integer is three plus two? Reply with just the integer.',c.tasks[1],c.tasks[2])
seeds=[6511,6512,6513,6514]

def small_context():
    src=c.bench.Corpus('runs/combo4/corpus').text('div',108000,'overnight-natural-100k',fixed=True).replace('<','(')
    target=98304+1830;cut=4*target
    for _ in range(9):
        ctx='[overnight-natural-low]\n'+src[:cut]+'\n\n';count=c.token_count([{'role':'user','content':ctx+lead}])
        if abs(count-target)<=4:return ctx
        cut+=round((target-count)*cut/count)
    raise RuntimeError('Could not calibrate the deep long-prefill fixture')

def calibrate(ctx,depth,grid,golden):
    for index,prompt in enumerate(short_leads):
        q=n.cell(depth,'chat-prime',[ctx+prompt],seeds[:1],budget=768,sharing='natural-calibration',rep=index,
                 expected_cached=grid,allow_early_stop=True)[0]
        if q['finish']=='stop' and 4<=q['tokens']<=225:break
    else:raise RuntimeError('None of the frozen lead probes stopped in the prespecified 4…225 token window')
    signature=(index,q['tokens'],q['done']['sha']);golden.setdefault(depth,signature)
    if golden[depth]!=signature:raise RuntimeError('Solo natural calibration changed across settings')
    c.emit('calibration',depth=depth,index=index,tokens=q['tokens'],output_sha=q['done']['sha'])
    return q['tokens']+24,prompt

def mixed(ctx,depth,grid,calibration,shape,rep):
    tight,prompt=calibration
    cap=tight if shape=='tight' else (1024 if shape=='moderate' else 768)
    first=moderate if shape=='moderate' else prompt
    result=n.cell(depth,'admission',[ctx+x for x in (first,*followers)],seeds,budget=256,lead_budget=cap,mixed=True,
                  sharing='natural-'+shape,rep=rep,expected_cached=grid,allow_early_stop=True)
    lead_end=result[0]['start']+result[0]['total']
    c.emit('natural-outcome',depth=depth,shape='natural-'+shape,rep=rep,
           lead_limit=cap,requested=[cap,256,256,256],emitted=[q['tokens'] for q in result],
           remaining_at_stop=cap-result[0]['tokens'],
           follower_arrival_after_first_s=[q['start']-result[0]['first'] for q in result[1:]],
           follower_arrival_to_lead_end_s=[lead_end-q['start'] for q in result[1:]])
    if result[0]['finish']!='stop' or result[0]['tokens']>=cap:
        if shape!='moderate':raise RuntimeError('Calibrated lead failed to stop below its budget')
        c.emit('censored-moderate',depth=depth,rep=rep,limit=cap,tokens=result[0]['tokens'])
    return result

def start_request_log():return time.time()

def extensions(start):
    logs=subprocess.check_output(['docker','logs','--since',str(start),cfg.policy.name],text=True,stderr=subprocess.STDOUT)
    matches=[l.rsplit('adaptive finish extension:',1)[1] for l in logs.splitlines() if 'adaptive finish extension:' in l]
    c.emit('extensions',count=len(matches),values=matches)

def stage_balanced(cases,ctx,golden,calibration,calibrate_lead=calibrate):
    if sum(p.rows())>32768 or min(c.memory())<5.2:raise RuntimeError('Nonempty initial occupancy or low memory')
    tight=None
    for key in ('shared','f0','f1','f2'):
        p.probe(key,'fill',0,cases[key]);
        if key=='shared' and calibration:tight=calibrate_lead(ctx,300000,p.grid,golden)
        if key!='shared':p.probe(key,'pre-pressure',0,cases[key],p.grid)
    rows=p.rows()
    if rows!=[2*p.grid,2*p.grid]:raise RuntimeError(f'Balanced peak diverged: {rows}')
    c.emit('stage',stage='peak',rows=rows)
    return tight

def natural_args(arm):
    n.args=argparse.Namespace(arm=arm,salt='5',occupancy=True,replay_depth=False,finish_grace=True)

def main(args):
    out,original,ids,scratch=cfg.setup(args);prompts,cases=cfg.fixture();ctx=cases['shared'].removesuffix(cfg.question);low=small_context();golden={};baseline=None
    # Fixed/held and adaptive/held are crossed, with the original combination at both ends.
    schedule=[('ten-fixed-a','10000','0'),('short-adaptive','500','1'),('short-fixed','500','0'),('ten-adaptive','10000','1'),('ten-fixed-b','10000','0')]
    try:
        for arm,wait,adaptive in schedule:
            if args.only and arm not in args.only:continue
            low_arm=arm+'-low';cfg.restart(low_arm,wait,adaptive,scratch,original,ids);natural_args(low_arm)
            n.cell(100000,'chat-prime',[low+cfg.question],[6499],budget=16,sharing='prime',rep=0,allow_early_stop=True)
            tight=calibrate(low,100000,98304,golden);since=start_request_log()
            for rep,order in enumerate((('tight','slack','moderate'),('moderate','slack','tight'))):
                for shape in order:mixed(low,100000,98304,tight,shape,rep)
            extensions(since)
            long_arm=arm+'-long';cfg.restart(long_arm,wait,adaptive,scratch,original,ids);natural_args(long_arm)
            stage_balanced(cases,ctx,golden,False);since=start_request_log()
            burst=c.cell(300000,'admission',prompts,list(range(4200,4204)),budget=1024,sharing='peak',rep=0,expected_cached=p.grid)
            if any(q['tokens']!=1024 for q in burst):raise RuntimeError('Long burst truncated')
            key=[(q['input_sha256'],q['cached'],q['tokens'],q['done']['sha']) for q in burst]
            if baseline is None:baseline=key
            c.emit('matched',measured=baseline==key);c.emit('patient',events=run.held_events(since));extensions(since)
            for key in ('f1','f0','f2'):p.probe(key,'after-pressure',0,cases[key])
            for shape in ('tight','slack'):
                high_arm=arm+'-'+shape;cfg.restart(high_arm,wait,adaptive,scratch,original,ids);natural_args(high_arm)
                tight=stage_balanced(cases,ctx,golden,True);since=start_request_log()
                mixed(ctx,300000,p.grid,tight,shape,0);c.emit('patient',events=run.held_events(since));extensions(since)
                for key in ('f1','f0','f2'):p.probe(key,'after-pressure',0,cases[key])
            if arm=='short-adaptive':run.exact(arm,out)
            t.validate(arm)
    except BaseException as exc:
        c.emit('failure',error=str(exc));raise
    finally:cfg.restore(original)
    c.emit('complete')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--store',default='asym01');ap.add_argument('--traffic-since',type=float)
    ap.add_argument('--only',nargs='+',choices=('ten-fixed-a','short-adaptive','short-fixed','ten-adaptive','ten-fixed-b'));a=ap.parse_args()
    if Path(a.store).name!=a.store:ap.error('Expected private store name')
    with (run.prior.parent/'.pair.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(a)
