"""Measure read-only direct I/O from allocated extents in this rank's existing Engram files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import mmap
import os
from pathlib import Path
import random
import threading
import time

def extents(fd,length,size):
    cursor=0
    good=[]
    while cursor<length and len(good)<2048:
        try:
            data=os.lseek(fd,cursor,os.SEEK_DATA)
        except OSError as e:
            if e.errno==errno.ENXIO:break
            raise
        hole=os.lseek(fd,data,os.SEEK_HOLE)
        start=(data+size-1)//size*size
        end=hole//size*size
        if end>start:good.append((start,end))
        cursor=hole
    if not good:raise ValueError("No sufficiently large allocated extent")
    return good

def measure(fd,spans,size,threads,duration,random_io):
    barrier=threading.Barrier(threads+1)
    def read(i):
        rng=random.Random(762+i);count=0
        with mmap.mmap(-1,size) as buf:
            n=i%len(spans);pos=spans[n][0]+(i*size)%(spans[n][1]-spans[n][0])
            barrier.wait()
            until=time.monotonic()+duration
            while time.monotonic()<until:
                if random_io:
                    lo,hi=rng.choice(spans)
                    pos=rng.randrange(lo,hi,size)
                elif pos+size>spans[n][1]:
                    n=(n+1)%len(spans);pos=spans[n][0]
                if os.preadv(fd,[buf],pos)!=size:raise IOError("Short read")
                pos+=size;count+=1
        return count
    with ThreadPoolExecutor(threads) as pool:
        tasks=[pool.submit(read,i) for i in range(threads)]
        barrier.wait();start=time.monotonic();ops=sum(t.result() for t in tasks);elapsed=time.monotonic()-start
    return dict(size=size,threads=threads,random=random_io,ops=ops,seconds=elapsed,GB_s=ops*size/elapsed/1e9,IOPS=ops/elapsed)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path);p.add_argument("--seconds",type=float,default=4)
    args=p.parse_args()
    file=max(args.directory.glob("*.safetensors"),key=lambda f:f.stat().st_blocks)
    fd=os.open(file,os.O_RDONLY|os.O_DIRECT)
    try:
        for size,n,random_io in [(4096,1,True),(4096,64,True),(1048576,1,False)]:
            spans=extents(fd,file.stat().st_size,size)
            row=measure(fd,spans,size,n,args.seconds,random_io)
            print(json.dumps(dict(file=file.name,sampled_allocated_bytes=sum(b-a for a,b in spans),**row)),flush=True)
    finally:os.close(fd)
