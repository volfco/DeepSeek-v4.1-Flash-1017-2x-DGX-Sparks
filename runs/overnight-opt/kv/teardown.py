"""Observe closing lane metrics without sending inference requests or retaining other request identifiers."""
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

root='http://127.0.0.1:8000'
metrics=('requests_running','requests_waiting','request_latency_seconds_count','generation_tokens_running')
log=Path(sys.argv[1])
out=Path(sys.argv[2])

def sample():
    now=time.monotonic()
    with urllib.request.urlopen(root+'/metrics',timeout=1) as f:
        body=f.read().decode()
    result={k:float(v) for k,v in re.findall(r'^tensorfold:([^ {]+) ([0-9.eE+-]+)$',body,re.M) if k in metrics}
    with urllib.request.urlopen(root+'/health',timeout=1) as f:
        h=json.load(f)
    return dict(t=now,at=time.time(),**result,kept=[x['kept_rows'] for x in h['kv_shard']['per_part']],
                live_rows=[x['live_rows'] for x in h['kv_shard']['per_part']],connections=h['live']['connections'],waiting=h['live']['waiting'])

def write(value):
    with out.open('a') as f:print(json.dumps(value),file=f)

def capture(arm,since,row):
    with_log=subprocess.check_output(['docker','logs','--since',str(since),'dsv41-1017-exl3-tf'],text=True,stderr=subprocess.STDOUT,timeout=4)
    ids=re.findall(r'\bdone (\S+) prompt=',with_log);own={q['id'] for q in row['requests']}
    write(dict(kind='observed-completions',arm=arm,matched=sorted(own.intersection(ids)),unknown_count=sum(i not in own for i in ids)))

armed=None
with log.open() as f:
    while True:
        line=f.readline()
        if line.startswith('{'):
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r['kind']=='stage' and r.get('stage')=='peak':
                armed=r['arm'];since=r['at'];next_poll=0;write(dict(kind='stage',arm=armed,since=since))
            if r['kind']=='cell' and r.get('sharing')=='peak' and armed:
                write(dict(kind='closing-cell',arm=armed,clean=r['clean'],at=r['at'],client_ends=[(q['id'],q['start']+q['total']) for q in r['requests']]))
                capture(armed,since,r)
                if not r['clean']:
                    for _ in range(20):
                        try:write(dict(kind='after-rejected',arm=armed,**sample()))
                        except OSError:break
                        time.sleep(.10)
                armed=None
            if r['kind'] in ('restored','complete'):break
        if not line:
            now=time.monotonic()
            if armed and now>=next_poll:
                try:write(dict(kind='live',arm=armed,**sample()))
                except OSError:pass
                next_poll=now+.45
            time.sleep(.05)
