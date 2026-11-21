"""Run six known-byte rank-local scans between inference campaigns while holding the pair lock."""
import concurrent.futures
import json
from pathlib import Path
import re
import subprocess
import time

def capture(trial,prefixes):
    directory=Path('runs/overnight-opt/hardware01')
    target=directory/'nvml-scan.json'
    request=directory/'nvml-scan-retry-requested'
    if target.exists() or not request.exists():return
    trial.idle()
    before=trial.server.metrics()
    done=trial.bench_div.DONE_KEY
    if any(before['tensorfold:requests_'+k] for k in ('running','waiting')):raise RuntimeError('Serving pair busy before scans')
    start=time.time()
    def rank(name,prefix,binary):
        path=directory/(name+'-nvml-scan.csv')
        if path.exists() and not target.exists():
            first=directory/(name+'-nvml-scan-first.csv')
            if not first.exists():first.write_bytes(path.read_bytes())
        scans=[]
        for i in range(6):
            q=subprocess.run(prefix+[binary],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
            data=q.stdout.strip()
            if q.returncode or not re.fullmatch(r'read,\d+,\d+,\d+',data):
                return dict(rank=name,rc=q.returncode,completed=len(scans),error=q.stderr[:160].strip())
            scans.append(data)
        path.write_text('\n'.join(scans)+'\n')
        return dict(rank=name,rc=0,completed=6)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(rank,'head',prefixes[0],'/tmp/dsv41-pmc-1118/calibrate'),
              pool.submit(rank,'worker',prefixes[1],'/tmp/calibrate')]
        results=[j.result() for j in jobs]
    after=trial.server.metrics()
    summary=dict(start=start,end=time.time(),ranks=results,completed_before=before[done],completed_after=after[done],
                 running_before=before['tensorfold:requests_running'],waiting_before=before['tensorfold:requests_waiting'],
                 running_after=after['tensorfold:requests_running'],waiting_after=after['tensorfold:requests_waiting'])
    if after[done]!=before[done] or summary['running_after'] or summary['waiting_after']:raise RuntimeError('Serving request overlapped scans')
    completed=all(r['completed']==6 for r in results)
    dest=target if completed else directory/'nvml-scan-retry-incomplete.json'
    dest.write_text(json.dumps(summary,indent=2)+'\n')
    return summary
