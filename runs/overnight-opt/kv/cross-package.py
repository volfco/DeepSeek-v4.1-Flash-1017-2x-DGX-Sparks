"""Build and check the two executables that a benchmark image replaces on the Spark pair."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

here = Path(__file__).resolve().parent
variants = {'eligible':here/'eligible.patch', 'committed':here/'committed.patch', 'prune':here.parent/'prune/marker.patch'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    work = args.work.resolve()
    work.mkdir(exist_ok=False)
    src = work/'source'
    shutil.copytree(args.pinned_source, src)
    subprocess.run(['git','apply',str(variants[args.variant])], cwd=src, check=True)
    build = work/'build'
    subprocess.run([sys.executable,'-I',str(here.parent/'cross-build.py'),str(src),str(args.fatbins.resolve()),
                    str(build),'--zig',args.zig], check=True)
    native = build/'native/bin/tensorfold-native'
    checked = work/'checked-fatbins'
    subprocess.run([sys.executable,'-I',str(here.parent/'fatbins.py'),str(native),str(checked)],check=True)
    reference = {p.name:sha(p) for p in args.fatbins.glob('*.fatbin')}
    actual = {p.name:sha(p) for p in checked.glob('*.fatbin')}
    if len(reference)!=20 or actual!=reference:
        raise RuntimeError('Embedded CUDA payloads differ')
    pkg = work/'package'
    pkg.mkdir()
    exe = [native, build/'bin/tf-dsv41-lanes']
    for p in exe:
        shutil.copy2(p,pkg/p.name)
    sums = ''.join(f'{sha(p)}  {p.name}\n' for p in sorted(exe,key=lambda p:p.name))
    (pkg/'SHA256SUMS').write_text(sums)
    target = work/(args.variant+'.tar.gz')
    with tarfile.open(target,'w:gz',compresslevel=2) as f:
        for p in sorted(pkg.iterdir()):
            f.add(p,arcname=p.name)
    print(target)
    print(sums,end='')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('variant',choices=variants)
    ap.add_argument('pinned_source',type=Path)
    ap.add_argument('fatbins',type=Path)
    ap.add_argument('work',type=Path)
    ap.add_argument('--zig',default='zig')
    main(ap.parse_args())
