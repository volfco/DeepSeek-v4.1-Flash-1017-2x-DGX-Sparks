"""Price a single hypothetical retention grant from synthetic private manifests and inode metadata."""
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import struct
import sys
import zlib


def inventory(root):
    def fail(e):raise e
    files={}
    for base,_,names in os.walk(root,onerror=fail):
        for name in names:
            if name.endswith('.man'):
                path=Path(base)/name;stat=path.stat();files[path]=(stat.st_size,stat.st_mtime_ns,stat.st_ino)
    return files


def decode(path):
    raw=path.read_bytes()
    if len(raw)<264 or raw[:8]!=b'TFSPILL\x02' or zlib.crc32(raw[:-8])!=struct.unpack_from('<I',raw,len(raw)-8)[0]:
        raise ValueError('Invalid committed manifest or checksum')
    ver,header=struct.unpack_from('<2I',raw,8)
    at,grid,_,media,blocks,sections,forks,layout=struct.unpack_from('<8I',raw,80)
    if ver not in (2,3) or header!=256 or (ver==2 and media) or grid==0:raise ValueError('Unsupported manifest layout')
    offset=header+forks*20+media*48
    if offset+blocks*80+sections*56+8!=len(raw):raise ValueError('Unexpected manifest extent')
    refs={};tokens=0
    for i in range(blocks):
        k=offset+80*i;name=raw[k:k+32].hex();size=struct.unpack_from('<Q',raw,k+40)[0]
        if name in refs:raise ValueError('Repeated block name within a point')
        extent=struct.unpack_from('<I',raw,k+32)[0]
        if extent==0 or extent>grid:raise ValueError('Invalid block extent')
        tokens+=extent
        refs[name]=size
    if tokens!=at:raise ValueError('Block extents and point extent differ')
    if path.stem!=raw[48:80].hex():raise ValueError('Filename and manifest ID changed')
    has_part,part=struct.unpack_from('<2I',raw,148)
    return at,(part if has_part else None),layout,refs,struct.unpack_from('<Q',raw,196)[0],struct.unpack_from('<Q',raw,212)[0]


def audit(root):
    files=inventory(root)
    if not files:raise ValueError('No committed synthetic points')
    ranks=defaultdict(list)
    for path in files:ranks[path.parent.parent].append((path,*decode(path)))
    selected=[]
    for dirname,points in ranks.items():
        refs=Counter(name for _,_,_,_,blocks,_,_ in points for name in blocks)
        block_stats={}
        for path,at,part,layout,blocks,ids_bytes,small_bytes in points:
            if at not in (296960,299008):continue
            files_stats=[path.with_suffix(ext).stat() for ext in ('.man','.ids','.st')]
            if any(x.st_nlink!=1 for x in files_stats):raise ValueError('Point metadata has unexpected hardlinks')
            if [x.st_size for x in files_stats[1:]]!=[ids_bytes,small_bytes]:raise ValueError('Point metadata extent differs from manifest')
            metadata=sum(x.st_size for x in files_stats);allocated=sum(x.st_blocks*512 for x in files_stats)
            exclusive=defaultdict(list)
            for name,size in blocks.items():
                if name not in block_stats:block_stats[name]=(dirname/'blocks'/(name+'.blk')).stat()
                st=block_stats[name]
                if st.st_size!=size:raise ValueError('Block size differed from manifest')
                if refs[name]>1:continue
                exclusive[st.st_dev,st.st_ino].append(st)
            removable=[v[0] for v in exclusive.values() if len(v)==v[0].st_nlink]
            selected.append(dict(rank=dirname.name,namespace=dirname.parent.name,point=path.stem,at=at,part=part,layout=layout,blocks=len(blocks),
                charge_bytes=metadata+sum(blocks.values()),metadata_bytes=metadata,
                exclusive_block_names=sum(1 for name in blocks if refs[name]==1),exclusive_inodes=len(removable),
                marginal_logical_bytes=metadata+sum(st.st_size for st in removable),
                marginal_allocated_bytes=allocated+sum(st.st_blocks*512 for st in removable)))
    if not selected or files!=inventory(root):raise ValueError('Inventory changed during measurement or lacked 300K points')
    return dict(manifests=len(files),points=sorted(selected,key=lambda r:(r['rank'],r['at'],r['point'])),
                scope='Deletion of one point with all other committed manifests and hardlinks held fixed. Metadata and stat only.')


if __name__=='__main__':
    path=Path(sys.argv[1]).resolve()
    if path!=Path('/spill') and not path.name.startswith('dsv41-overnight-'):raise ValueError('Expected isolated private spill mount')
    print(json.dumps(audit(path)))
