"""Sampled tool selection and SQL generation, with deterministic behavioral grading."""
import argparse
import concurrent.futures
import datetime
import json
from pathlib import Path
import random
import re
import sqlite3
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path.cwd()/"tools"))
import bench_div
import score

srv=bench_div.Server("http://127.0.0.1:8000",score.DEFAULT_MODEL)
sql_tasks=[
 ("List active player IDs in increasing order.", "select id from players where active=1 order by id"),
 ("Return team and highest points per team, in alphabetical team order.", "select team,max(points) from players group by team order by team"),
 ("Return the two highest scoring IDs, breaking ties by smaller ID.", "select id from players order by points desc,id limit 2"),
 ("Return teams with at least two active players in alphabetical order.", "select team from players where active=1 group by team having count(*)>=2 order by team"),
 ("Give the total points among inactive players, returning zero when there are none.", "select coalesce(sum(points),0) from players where active=0"),
 ("Return each ID whose points exceed its own team's average, in ID order.", "select id from players p where points>(select avg(points) from players q where q.team=p.team) order by id"),
 ("Return team and count of distinct point values per team in alphabetical order.", "select team,count(distinct points) from players group by team order by team"),
 ("Return teams in alphabetical order where the sum of points from active players is greater than 10.", "select team from players where active=1 group by team having sum(points)>10 order by team"),
]
tool_specs={
 "book_room":{"room":"string","minutes":"integer","seats":"integer"},
 "send_parcel":{"city":"string","weight_grams":"integer","express":"boolean"},
 "set_timer":{"label":"string","seconds":"integer"},
 "move_event":{"name":"string","offset_minutes":"integer"},
}
tool_tasks=[
 ("Reserve Juniper for half an hour for seven people.", "book_room", {"room":"Juniper","minutes":30,"seats":7}),
 ("Reserve Alder for one hour and a quarter for two people.", "book_room", {"room":"Alder","minutes":75,"seats":2}),
 ("Ship a 2.5 kg parcel to Oslo with express delivery.", "send_parcel", {"city":"Oslo","weight_grams":2500,"express":True}),
 ("Use standard delivery for a 750 gram parcel to Lima.", "send_parcel", {"city":"Lima","weight_grams":750,"express":False}),
 ("Start a timer labeled tea for four and a half minutes.", "set_timer", {"label":"tea","seconds":270}),
 ("Start a timer labeled oven for one hour and ten seconds.", "set_timer", {"label":"oven","seconds":3610}),
 ("Move the event named demo two hours earlier; use a signed offset.", "move_event", {"name":"demo","offset_minutes":-120}),
 ("Move the event named review an hour and a quarter later; use a signed offset.", "move_event", {"name":"review","offset_minutes":75}),
]
tools=[{"type":"function","function":{"name":name,"description":name.replace("_"," "),"parameters":{"type":"object","properties":{k:{"type":v} for k,v in fields.items()},"required":list(fields)}}} for name,fields in tool_specs.items()]

def check_sql(answer,reference):
    match=re.search(r"```(?:sqlite|sql)?\s*(.*?)```",answer,re.S|re.I)
    query=(match.group(1) if match else answer).strip()
    if not re.match(r"^(SELECT|WITH)\b",query,re.I):
        return False
    for seed in (7301,7302,7303,7304):
        rng=random.Random(seed)
        rows=[(i,rng.choice(("elm","ash","fir")),rng.randrange(12),rng.randrange(2)) for i in range(0 if seed==7304 else 14)]
        db=sqlite3.connect(":memory:")
        try:
            db.execute("create table players(id integer primary key, team text, points integer, active integer)")
            db.executemany("insert into players values(?,?,?,?)",rows)
            expected=db.execute(reference).fetchall()
            db.execute("pragma query_only=on")
            stop=time.monotonic()+0.15
            db.set_progress_handler(lambda: int(time.monotonic()>stop),500)
            safe_functions={"sum","avg","min","max","count","coalesce","ifnull","round","rank","dense_rank","row_number","iif"}
            def authorize(action,arg1,arg2,unused_db,unused_trigger):
                if action in (sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ) or (action==sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in safe_functions):
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY
            db.set_authorizer(authorize)
            if db.execute(query).fetchmany(len(expected)+1)!=expected:return False
        except sqlite3.Error:
            return False
        finally:db.close()
    return True

def call(kind,i,seed,barrier):
    if kind=="sql":
        text="SQLite schema: players(id INTEGER PRIMARY KEY, team TEXT, points INTEGER, active INTEGER). active is 0 or 1. Return a single SQL query only. "+sql_tasks[i][0]
    else:text="Use the appropriate tool exactly once with the requested arguments. "+tool_tasks[i][0]
    body={"model":srv.model,"messages":[{"role":"user","content":text}],"temperature":1.0,"top_p":0.95,"top_k":20,
          "seed":seed,"max_tokens":256,"chat_template_kwargs":{"enable_thinking":False}}
    if kind=="tool":body["tools"]=tools
    barrier.wait(timeout=15)
    began=time.monotonic()
    data=score.post(srv.url,"/v1/chat/completions",body,180)
    wall=time.monotonic()-began
    if data.get("error"):raise RuntimeError(data["error"])
    c=data["choices"][0];m=c["message"]
    if kind=="sql":
        passed=check_sql(m.get("content") or "",sql_tasks[i][1])
    else:
        cs=m.get("tool_calls") or []
        passed=False
        if len(cs)==1:
            fn=cs[0]["function"]
            try:
                params=json.loads(fn["arguments"])
                passed=fn["name"]==tool_tasks[i][1] and params==tool_tasks[i][2] and all(type(params[k]) is type(v) for k,v in tool_tasks[i][2].items())
            except (TypeError,ValueError,KeyError):pass
    return dict(kind=kind,item=i,seed=seed,id=data["id"],success=passed,wall=wall,tokens=data["usage"]["completion_tokens"],
                finish=c["finish_reason"],answer=m)

def main():
    total=[]
    for rep in range(2):
        jobs=[(k,i,9100+rep*100+i) for i in range(8) for k in ("sql","tool")]
        random.Random(1118+rep).shuffle(jobs)
        for n in ((1,4,8) if rep==0 else (8,4,1)):
            for j in range(0,len(jobs),n):
                assert bench_div.idle_wait(srv,30)[0]
                m0=srv.metrics(); since=str(int(time.time())-1);bar=threading.Barrier(n)
                trace=[]
                start=time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(n) as pool:
                    fs=[pool.submit(call,*job,bar) for job in jobs[j:j+n]]
                    while not all(f.done() for f in fs):
                        met=srv.metrics()
                        trace.append([round(time.monotonic()-start,4),met.get("tensorfold:requests_running"),
                                      met.get("tensorfold:generation_tokens_running"),met.get("tensorfold:kv_shard_fetch_rounds"),
                                      met.get("tensorfold:kv_shard_owner_rounds"),met.get("tensorfold:requests_waiting")])
                        concurrent.futures.wait(fs,timeout=.04)
                    results=[f.result() for f in fs]
                wall=time.monotonic()-start
                assert bench_div.idle_wait(srv,30)[0]
                m1=srv.metrics()
                assert m1[bench_div.DONE_KEY]-m0[bench_div.DONE_KEY]==n
                logs=subprocess.run(["docker","logs","--timestamps","--since",since,"dsv41-1017-exl3-tf"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True,text=True).stdout
                for r in results:
                    match=re.search(r"^(\S+) [^\n]*?done "+re.escape(r["id"])+r" (.*)",logs,re.M)
                    if match is None:raise RuntimeError("Missing engine record")
                    r["done"]=dict(v.split("=",1) for v in match[2].split() if "=" in v)
                    r["engine_end"]=datetime.datetime.fromisoformat(match[1].replace("Z","+00:00")).timestamp()
                print(json.dumps(dict(width=n,rep=rep,wall_s=wall,trace=trace,
                                      start_epoch_s=time.time()-(time.monotonic()-start),results=results)),flush=True)
                total.extend(results)
    assert len(total)==96

if __name__=="__main__":main()
