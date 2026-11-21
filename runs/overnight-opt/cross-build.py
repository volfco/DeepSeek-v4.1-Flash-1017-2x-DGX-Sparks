"""Build the serving head and standalone follower against the same source, ABI and embedded CUDA kernels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

def build(a):
    f=a.source/"zig/build/cuda.zig"
    s=f.read_text()
    original='b.installArtifact(b.addExecutable(.{ .name = "tf-dsv41-lanes", .root_module = ds_lanes }));'
    replacement='''const bench_follower = b.addExecutable(.{ .name = "tf-dsv41-lanes", .root_module = ds_lanes });
    b.installArtifact(bench_follower);
    b.step("bench-follower", "Install only the benchmark follower").dependOn(&b.addInstallArtifact(bench_follower,.{}).step);'''
    if original in s:f.write_text(s.replace(original,replacement))
    elif replacement not in s:raise ValueError("Unknown follower build rule")
    cmd=[a.zig,"build","-Dtarget=aarch64-linux-gnu.2.39","-Dcpu=cortex_a725","-Doptimize=ReleaseFast",
         "-Dfatbins="+str(a.fatbins.resolve()),
         "--prefix",str(a.out.resolve()),"-j4","native","bench-follower"]
    subprocess.run(cmd,cwd=a.source,check=True,env=dict(os.environ,ZIG_GLOBAL_CACHE_DIR="/tmp/gb10-cross-global",ZIG_LOCAL_CACHE_DIR="/tmp/gb10-cross-cache",XDG_CACHE_HOME="/tmp/gb10-xdg-cache"))
    proofs={str(p.relative_to(a.out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.rglob("*") if p.is_file()}
    (a.out/"proof.json").write_text(json.dumps(proofs,indent=2)+"\n")
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path);p.add_argument("fatbins",type=Path);p.add_argument("out",type=Path)
    p.add_argument("--zig",default="zig");build(p.parse_args())
