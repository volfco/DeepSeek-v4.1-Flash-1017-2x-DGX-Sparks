"""Hard, execution-graded SQLite tasks with a separate Python oracle and hidden fixture seeds."""
import collections
import itertools
import random
import re
import sqlite3
import time

schema='accounts(a INTEGER PRIMARY KEY, grp TEXT, lim INTEGER), moves(k INTEGER PRIMARY KEY, a INTEGER, d INTEGER, v INTEGER), spans(k INTEGER PRIMARY KEY, a INTEGER, lo INTEGER, hi INTEGER), offers(k INTEGER PRIMARY KEY, a INTEGER, cost INTEGER, gain INTEGER)'
prompts=(
    'Merge overlapping or touching half-open spans separately for each account. Return account, start, stop and length of its longest merged interval. Break ties by earliest start. With no spans return (account,NULL,NULL,0). Order by account.',
    'Starting at balance zero, apply moves in (d,k) order per account. Return account, the first move key giving a strictly negative running balance, and the first later key giving a nonnegative balance. Use NULL for each missing key. Order by account.',
    'Group moves by account and integer calendar day; keep days whose net v is strictly positive. Return account, start day and length of the longest streak of consecutive qualifying calendar days. Break ties by earliest start; with no streak return NULL and 0. Order by account.',
    'Starting from 0, calculate the maximum peak-to-subsequent-trough drawdown in running move balance for each account in (d,k) order, also considering the opening balance as a peak. Per grp return the account with the greatest drawdown, and that drawdown (zero is allowed); ties choose smallest account. Return grp, account, drawdown ordered by grp.',
    'Per account choose exactly three distinct offers of that account with total cost <= its lim. Maximize total gain; then minimize total cost; then choose the lexicographically smallest increasing triple of offer keys. Return account, three ordered keys, total gain, total cost. For an infeasible account return five NULLs after the account. Order by account.',
    'spans are half-open integer intervals [lo,hi). For each account return its largest number of simultaneously active spans and the earliest integer time attaining that positive number. A touching end and start are not simultaneous. For no spans return 0 and NULL. Return account, peak, first_time ordered by account.',
    'Group moves into daily net v per account. Find the earliest run of three consecutive calendar days whose daily nets strictly increase, without filling missing dates. Return account, last day of that run, sum of its three daily nets. Use NULL for both values if absent. Order by account.',
    'Within each account order moves by (d,k). Zero breaks runs. Among contiguous runs of the same nonzero sign return the longest. Break ties by its earliest first move in that order. Return account, first move key, last move key, run length. If absent use NULL,NULL,0. Order by account.',
)
refs=(
 '''with p as(select *,max(hi) over(partition by a order by lo,hi,k rows between unbounded preceding and 1 preceding) m from spans), i as(select *,sum(case when m is null or lo>m then 1 else 0 end)over(partition by a order by lo,hi,k) z from p), g as(select a,min(lo) l,max(hi) h from i group by a,z), b as(select *,row_number()over(partition by a order by h-l desc,l) r from g) select accounts.a,l,h,coalesce(h-l,0) from accounts left join b on b.a=accounts.a and r=1 order by accounts.a''',
 '''with b as(select *,sum(v)over(partition by a order by d,k) z,row_number()over(partition by a order by d,k) n from moves),g as(select a,min(n) n from b where z<0 group by a) select x.a,y.k,(select k from b where a=x.a and n>g.n and z>=0 order by n limit 1) from accounts x left join g on g.a=x.a left join b y on y.a=x.a and y.n=g.n order by x.a''',
 '''with d as(select a,d from moves group by a,d having sum(v)>0), w as(select *,d-row_number()over(partition by a order by d) g from d), agg as(select a,min(d) l,count(*) n from w group by a,g), b as(select *,row_number()over(partition by a order by n desc,l) r from agg) select x.a,l,coalesce(n,0) from accounts x left join b on b.a=x.a and r=1 order by x.a''',
 '''with b as(select *,sum(v)over(partition by a order by d,k) z from moves),p as(select *,max(z)over(partition by a order by d,k) q from b),m as(select a,max(max(0,q)-z) dd from p group by a),g as(select x.*,coalesce(dd,0) val,row_number()over(partition by grp order by coalesce(dd,0) desc,x.a) r from accounts x left join m on m.a=x.a) select grp,a,val from g where r=1 order by grp''',
 '''with c as(select x.a,i.k i,j.k j,h.k h,i.gain+j.gain+h.gain g,i.cost+j.cost+h.cost c from accounts x join offers i on i.a=x.a join offers j on j.a=x.a and i.k<j.k join offers h on h.a=x.a and j.k<h.k where i.cost+j.cost+h.cost<=x.lim),b as(select *,row_number()over(partition by a order by g desc,c,i,j,h) r from c)select x.a,i,j,h,g,c from accounts x left join b on x.a=b.a and r=1 order by x.a''',
 '''with e as(select a,lo t,1 n from spans union all select a,hi,-1 from spans),d as(select a,t,sum(n)n from e group by a,t),v as(select a,t,sum(n)over(partition by a order by t) k from d),r as(select *,row_number()over(partition by a order by k desc,t) j from v) select x.a,coalesce(k,0),t from accounts x left join r on r.a=x.a and j=1 order by x.a''',
 '''with d as(select a,d,sum(v)v from moves group by a,d),v as(select j.a,j.d+2 last,j.v+k.v+l.v s from d j join d k on j.a=k.a and k.d=j.d+1 join d l on l.a=j.a and l.d=j.d+2 where j.v<k.v and k.v<l.v),r as(select *,row_number()over(partition by a order by last) n from v)select x.a,last,s from accounts x left join r on r.a=x.a and n=1 order by x.a''',
 '''with w as(select *,case when v>0 then 1 when v<0 then -1 else 0 end s,row_number()over(partition by a order by d,k)n from moves),q as(select *,lag(s)over(partition by a order by n) prev from w),z as(select *,sum(case when s=prev then 0 else 1 end)over(partition by a order by n) g from q),agg as(select a,g,min(n) l,max(n) h,count(*) v from z where s<>0 group by a,g),b as(select *,row_number()over(partition by a order by v desc,l)r from agg)select x.a,u.k,w.k,coalesce(b.v,0)from accounts x left join b on b.a=x.a and b.r=1 left join w u on u.a=x.a and u.n=b.l left join w on w.a=x.a and w.n=b.h order by x.a''',
)

def data(seed):
    rng=random.Random(seed);ac=[(i,('ash','fir','elm')[i%3],rng.choice([5,10,12,14,16])) for i in range(1,8)] if seed else []
    moves=[];spans=[];offers=[]
    for a,_,_ in ac:
        if a==7:continue
        for t,dest in enumerate((moves,spans,offers)):
            for j in range(rng.randint(1,18)):
                k=1000*a+17*j;d=rng.randrange(-3,9)
                row=(k,a,d,rng.randrange(-4,5)) if t==0 else ((k,a,d,d+rng.randrange(1,6)) if t==1 else (k,a,rng.randrange(1,7),rng.randrange(1,8)))
                dest.append(row)
    if seed and ac:
        spans.extend([(90003,7,1,2),(90004,7,2,4),(90005,7,9,12),(90006,7,1,2)])
    return ac,moves,spans,offers

def expected(item,tables):
    ac,moves,spans,offers=tables;rows=[];draw={}
    for a,grp,lim in ac:
        m=sorted((d,k,v) for k,aa,d,v in moves if a==aa);sp=sorted((l,h) for k,aa,l,h in spans if aa==a);off=sorted((k,c,g) for k,aa,c,g in offers if aa==a)
        days=collections.defaultdict(int)
        for d,k,v in m:days[d]+=v
        if item==0:
            merged=[]
            for l,h in sp:
                if merged and l<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],h)
                else:merged.append([l,h])
            l,h=min(merged,key=lambda p:(p[0]-p[1],p[0])) if merged else (None,None);rows.append((a,l,h,0 if l is None else h-l))
        if item in (1,3):
            balance=high=dd=0;negative=back=None
            for d,k,v in m:
                balance+=v;high=max(high,balance);dd=max(dd,high-balance)
                if negative is None and balance<0:negative=k
                elif negative is not None and back is None and balance>=0:back=k
            if item==1:rows.append((a,negative,back))
            else:draw[grp]=min(draw.get(grp,(0,a)),(-dd,a))
        if item==2:
            pos=sorted(d for d,v in days.items() if v>0);g=[list(elt for _,elt in group) for _,group in itertools.groupby(enumerate(pos),lambda z:z[1]-z[0])]
            run=min(g,key=lambda x:(-len(x),x[0])) if g else [];rows.append((a,run[0] if run else None,len(run)))
        if item==4:
            opts=[(-sum(q[2] for q in combo),sum(q[1] for q in combo),tuple(q[0] for q in combo)) for combo in itertools.combinations(off,3) if sum(q[1] for q in combo)<=lim]
            if opts:g,c,ks=min(opts);rows.append((a,*ks,-g,c))
            else:rows.append((a,None,None,None,None,None))
        if item==5:
            if sp:
                n,t=min((-sum(lo<=t<hi for lo,hi in sp),t) for t in range(min(v[0] for v in sp),max(v[1] for v in sp)));rows.append((a,-n,t))
            else:rows.append((a,0,None))
        if item==6:
            ends=[d+2 for d in days if d+1 in days and d+2 in days and days[d]<days[d+1]<days[d+2]];last=min(ends) if ends else None
            rows.append((a,last,None if last is None else sum(days[i] for i in (last-2,last-1,last))))
        if item==7:
            runs=[list(g) for s,g in itertools.groupby(m,lambda q:(q[2]>0)-(q[2]<0)) if s];best=max(runs,key=len) if runs else []
            rows.append((a,best[0][1] if best else None,best[-1][1] if best else None,len(best)))
    return [(g,a,-dd) for g,(dd,a) in sorted(draw.items())] if item==3 else rows

def check(item,answer):
    code=re.search(r'```(?:sql|sqlite)?\s*(.*?)```',answer,re.I|re.S);q=(code.group(1) if code else answer).strip()
    if not re.match(r'^(SELECT|WITH)\b',q,re.I):return False,'no-query'
    tables=('accounts','moves','spans','offers');columns=('a integer primary key,grp text,lim integer','k integer primary key,a integer,d integer,v integer','k integer primary key,a integer,lo integer,hi integer','k integer primary key,a integer,cost integer,gain integer')
    for seed in (0,131,777,1827,2111,9843):
        ds=data(seed);db=sqlite3.connect(':memory:')
        try:
            for t,col,rs in zip(tables,columns,ds):
                db.execute(f'create table {t}({col})');db.executemany(f'insert into {t} values ({",".join("?" for _ in col.split(","))})',rs)
            db.execute('pragma query_only=on');deadline=time.monotonic()+.3;db.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
            # Accept the interpreter's built-in read-only SQL functions, including alternative valid implementations.
            allowed={r[0].lower() for r in db.execute('pragma function_list') if r[1]}-{'load_extension','readfile','writefile'}
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH,1<<20);db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH,1<<18)
            def auth(act,x,y,*unused):
                return sqlite3.SQLITE_OK if act in (sqlite3.SQLITE_READ,sqlite3.SQLITE_SELECT,sqlite3.SQLITE_RECURSIVE) or (act==sqlite3.SQLITE_FUNCTION and (y or '').lower() in allowed) else sqlite3.SQLITE_DENY
            db.set_authorizer(auth);want=expected(item,ds)
            if db.execute(q).fetchmany(len(want)+1)!=want:return False,'wrong-rows'
        except sqlite3.Error:return False,'invalid-or-timeout'
        finally:db.close()
    return True,'pass'

if __name__=='__main__':
    for i,r in enumerate(refs):assert check(i,r)==(True,'pass'),(i,check(i,r))
    assert not any(check(i,'select 1')[0] for i in range(8));print('All 8 SQL references passed 48 independent oracle checks')
