"""Recover Zig's embedded, NUL-terminated CUDA fatbins for an off-pair AArch64 benchmark build."""
import argparse
from pathlib import Path
import re
import struct
import subprocess

def extract(elf, out):
    sections=subprocess.check_output(["readelf","-WS",elf],text=True)
    sec=re.search(r'\.rodata\s+PROGBITS\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)',sections)
    address,offset,length=(int(x,16) for x in sec.groups())
    data=elf.read_bytes()
    table=subprocess.check_output(["nm","-S",elf],text=True)
    out.mkdir(exist_ok=True)
    for va,size,name in re.findall(r'^(\w+) (\w+) r kernels\.Blob\("fatbin_(\w+)"[^\n]+?\.bytes$',table,re.M):
        pos=int(va,16)-address
        size=int(size,16)
        if not 0<=pos<pos+size<=length:raise ValueError(name)
        raw=data[offset+pos:offset+pos+size]
        magic,version,header,payload=struct.unpack_from("<IHHQ",raw)
        if magic!=0xba55ed50 or payload+header!=size-1 or raw[-1]!=0:raise ValueError((name,magic,payload,header,size))
        (out/(name+".fatbin")).write_bytes(raw[:-1])
    print(len(list(out.glob("*.fatbin"))),"fatbins")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("elf",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args();extract(a.elf,a.out)
