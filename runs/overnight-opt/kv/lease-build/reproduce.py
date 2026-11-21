"""Build and exercise the metadata-only retention prototype in a fresh, off-pair checkout copy."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


here=Path(__file__).resolve().parent
store='zig/src/core/spill/store.zig'
store_test='zig/src/core/spill/store_test.zig'
pair_test='zig/src/families/dsv41/spill_tier_test.zig'


def execute(args):
    root=args.source.resolve()
    for path,sha in json.loads((here/'source-identities.json').read_text()).items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=sha:raise RuntimeError('Expected the original pinned source at '+path)
    with tempfile.TemporaryDirectory(prefix='dsv41-lease-test-') as tmp:
        work=Path(tmp);src=work/'src';shutil.copytree(root/'zig/src',src/'zig/src')
        cfg=['-lc','-O','ReleaseSafe','--cache-dir',str(work/'cache'),'--global-cache-dir',str(work/'global')]
        cr=src/'zig/src/core';entry=cr/'metadata_lease_root.zig'
        size=b'const std = @import("std");\ntest "benchmark store size" { std.debug.print("STORE_BYTES={d}\\n", .{@sizeOf(@import("spill/store.zig").Store)}); }\n'
        def run(label,cmd,cwd):
            with (args.out/(label+'.log')).open('x') if args.out else contextlib.nullcontext(None) as log:
                subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
        entry.write_bytes(size);run('pinned-size',[args.zig,'test',*cfg,'-Mroot='+str(entry),'--test-filter','benchmark store size'],src)
        for patch in ('retention.patch','loopback-fixture.patch'):
            subprocess.run(['git','apply','--check',str(here/patch)],cwd=src,check=True)
            subprocess.run(['git','apply',str(here/patch)],cwd=src,check=True)
        for origin,target in [('store-test.zig',store_test),('loopback-test.zig',pair_test)]:
            with (src/target).open('ab') as f:f.write((here/origin).read_bytes())
        run('candidate-size',[args.zig,'test',*cfg,'-Mroot='+str(entry),'--test-filter','benchmark store size'],src)
        entry.write_text('test { _ = @import("spill/store_test.zig"); }\n')
        run('store-test',[args.zig,'test',*cfg,'-Mroot='+str(entry),'--test-filter','benchmark metadata lease'],src)
        run('existing-store-suite',[args.zig,'test',*cfg,'-Mroot='+str(entry),'--test-filter','the GLM 0110 checks'],src)
        exe=work/'pair-test'
        run('loopback-build',[args.zig,'test',*cfg,'--dep','core','--dep','engine_api','-Mroot='+str(src/pair_test),
                              '-Mcore='+str(cr/'root.zig'),'-Mengine_api='+str(cr/'engine_api.zig'),'--test-filter','benchmark metadata lease',
                              '--test-no-exec','-femit-bin='+str(exe)],src)
        run('loopback',[str(exe)],src)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--zig',default='zig');p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.out:a.out=a.out.resolve();a.out.mkdir(exist_ok=False)
    execute(a)
