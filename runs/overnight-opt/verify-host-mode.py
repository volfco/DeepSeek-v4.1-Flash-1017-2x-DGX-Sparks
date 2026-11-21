"""Run the Linux host suite in an explicit mode and verify the mode of every compiled test."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def run(source, zig, mode):
    with tempfile.TemporaryDirectory(prefix="gb10-host-mode-") as scratch:
        work = Path(scratch)
        tree = work / "source"
        shutil.copytree(source, tree, ignore=shutil.ignore_patterns(".git", ".zig-cache", "zig-out"))
        graph = tree / "zig/build/cuda.zig"
        code = graph.read_text()
        start = code.index("pub fn hostTests(")
        end = code.index("\n}", start)
        original = code[start:end]
        if original.count(".debug") != 8:
            raise ValueError("Review the updated host test build declarations")
        graph.write_text(code[:start] + original.replace(".debug", "." + mode) + code[end:])
        result = subprocess.run([zig, "build", "test", "--summary", "all", "-j4"], cwd=tree,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env=dict(os.environ, ZIG_LOCAL_CACHE_DIR=str(work / "cache")))
        print(result.stdout, end="")
        modes = re.findall(r"compile test (\w+) native", result.stdout)
        passed = re.search(r"([1-9]\d*)/\1 tests passed", result.stdout)
        if result.returncode or set(modes) != {mode} or not passed:
            raise SystemExit("Failed host test or unverifiable compilation mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--zig", default="zig")
    parser.add_argument("--mode", choices=["debug", "safe"], required=True)
    args = parser.parse_args()
    run(args.source, args.zig, args.mode)
