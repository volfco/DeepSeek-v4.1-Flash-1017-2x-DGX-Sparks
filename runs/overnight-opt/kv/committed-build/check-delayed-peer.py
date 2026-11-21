"""Exercise committed snapshot proof behavior on two CPU loopback ranks in a patched checkout."""
import argparse
from pathlib import Path
import subprocess
import tempfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path)
p.add_argument('--zig',default='zig')
p.add_argument('--case',choices=['delayed-peer','async-invalidation','quiescence'],default='delayed-peer')
a=p.parse_args();root=a.source.resolve()/'zig/src';family=root/'families/dsv41'
fixture=(family/'spill_tier_test.zig').read_text()
test_filter='benchmark aggregate quiescence' if a.case=='quiescence' else 'benchmark cached affirmative'
if a.case=='delayed-peer':
    needle='                T.snap => {'
    assert fixture.count(needle)==1
    fixture=fixture.replace(needle,'                32 => std.Io.sleep(io, .fromMilliseconds(try r.int(u32)), .awake) catch {},\n'+needle)
    test_filter='benchmark proof under delayed'
fixture+=(Path(__file__).resolve().parent/(a.case+'-test.zig')).read_text()
with tempfile.TemporaryDirectory(prefix='dsv41-peer-timeout-') as scratch, tempfile.NamedTemporaryFile(mode='w',suffix='.zig',prefix='benchmark_timeout_',dir=family) as f:
    f.write(fixture);f.flush()
    out=Path(scratch)/'fixture'
    subprocess.run([a.zig,'test','-lc','--dep','core','--dep','engine_api','-Mroot='+f.name,
                    '-Mcore='+str(root/'core/root.zig'),'-Mengine_api='+str(root/'core/engine_api.zig'),
                    '--test-filter',test_filter,'--test-no-exec','--cache-dir',scratch+'/cache',
                    '--global-cache-dir',scratch+'/global','-femit-bin='+str(out)],check=True)
    result=subprocess.run([str(out)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(result.stdout,end='')
    if a.case!='delayed-peer':
        signature='FRESH_BOTH_IDLE=true FIRST=present SECOND=peer_missing' if a.case=='quiescence' else 'CACHED=present CURRENT=peer_missing HEAD=present'
        result.check_returncode()
        if 'All 2 tests passed.' not in result.stdout or signature not in result.stdout:
            raise RuntimeError('Different result from the requested point-presence regression')
        raise SystemExit(0)
    # Zig counts the intentionally logged pair break as an error even when every assertion passed.
    if result.returncode!=1 or 'All 2 tests passed.' not in result.stdout or '1 errors were logged.' not in result.stdout:
        raise RuntimeError('Expected assertions plus the single deliberate error log')
    if result.stdout.count('(err):')!=1 or 'rank 1: ReplyTimeout' not in result.stdout or 'PAIR_CALLBACK=true' not in result.stdout:
        raise RuntimeError('Different failure from the injected slow peer')
