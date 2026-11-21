"""Finish the supported-mode trial and short configuration attribution after the primary reservation releases."""
import json
from pathlib import Path
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
from engineering import stage_image
import buckets


def launch(path, script, *arguments):
    with path.with_suffix(".log").open("x") as log:
        status = subprocess.run([sys.executable, "-I", "-u", str(here / script), str(path), *arguments],
                                stdout=log, stderr=subprocess.STDOUT).returncode
    if not (path / "restored.json").exists():
        raise RuntimeError(f"No validated restoration for {path}")
    return status


def run(root):
    while not (root / "released.json").is_file():
        time.sleep(5)
    if any(r["tag"] != buckets.driver.base or any(r["flags"][k] for k in buckets.driver.trial.candidate)
           for r in buckets.driver.inspect()):
        raise RuntimeError("Expected the serving pair at its original settings")
    outcomes = {}
    if not (root / "isolated-resume").exists():
        outcomes["isolated-resume"] = launch(root / "isolated-resume", "isolated-resume.py", "--image", stage_image)
    outcomes["config-crossover"] = launch(root / "config-crossover", "config-crossover.py")
    buckets.write_json(root / "finish-outcomes.json", outcomes)


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
