"""Count completions in sampled production windows while the benchmark lock is unheld."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


lock=Path('runs/engine-followup/.pair.lock').stat()
lock_id=(os.major(lock.st_dev),os.minor(lock.st_dev),lock.st_ino)
container='dsv41-1017-exl3-tf'
base='tensorfold-dsv41:exl3-tp2-234f9080b326'


def owned():
    for line in Path('/proc/locks').read_text().splitlines():
        parts=line.split()
        if len(parts)<6 or parts[1]!='FLOCK':continue
        a,b,i=parts[5].split(':')
        if (int(a,16),int(b,16),int(i))==lock_id:return True
    return False


def sample():
    if owned():return None
    cfg=subprocess.check_output(['docker','inspect','--format','{{.Id}} {{.Config.Image}}',container],text=True,timeout=5).split()
    if len(cfg)!=2 or cfg[1]!=base:return None
    with urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=3) as response:metrics=response.read().decode()
    values={}
    for line in metrics.splitlines():
        fields=line.split()
        if len(fields)==2 and fields[0] in ('tensorfold:request_latency_seconds_count','tensorfold:requests_running','tensorfold:requests_waiting'):
            values[fields[0]]=float(fields[1])
    if len(values)!=3 or owned():return None
    return (cfg[0],time.time(),int(values['tensorfold:request_latency_seconds_count']),int(values['tensorfold:requests_running']),int(values['tensorfold:requests_waiting']))


def run(until):
    previous=opening=None
    def close_run():
        if opening and previous and previous[1]>opening[1]:
            print(json.dumps(dict(start=opening[1],end=previous[1],seconds=previous[1]-opening[1],completed=previous[2]-opening[2],
                                  running_max=maximum[0],waiting_max=maximum[1])),flush=True)
    maximum=[0,0]
    while time.time()<until:
        try:current=sample()
        except (OSError,ValueError,subprocess.SubprocessError):current=None
        if not current or not previous or previous[0]!=current[0] or current[1]-previous[1]>3 or current[2]<previous[2]:
            close_run();opening=current;maximum=[0,0]
        if current:
            maximum=[max(maximum[0],current[3]),max(maximum[1],current[4])]
        previous=current
        time.sleep(1)
    close_run()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--until',required=True,type=datetime.datetime.fromisoformat)
    args=parser.parse_args()
    if args.until.tzinfo is None:parser.error('Supply an offset-aware cutoff')
    run(args.until.timestamp())
