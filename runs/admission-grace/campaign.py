"""Run from the recipe root on the head with a fresh output directory."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "review-round"))
import round as trial

base = trial.image
image = "tensorfold-dsv41:bench-grace-234f9080-31758421"
switch = "TF_DS_FINISH_GRACE_MS"
name = "dsv41-1017-exl3-tf"
worker = ["ssh", "-o", "BatchMode=yes", "10.50.0.11"]
prefixes = ([], worker)
scratch = "/home/colum/.cache/dsv41-admission-grace-1118"

def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def read(prefix, argv):
    return subprocess.check_output(prefix + argv, text=True, timeout=25).strip()


def runtime(expected_image, grace=None):
    ranks = []
    for prefix in prefixes:
        c = json.loads(read(prefix, ["docker", "inspect", name]))[0]
        cfg = c["Config"]
        if not c["State"]["Running"] or cfg["Image"] != expected_image:
            raise RuntimeError("Unexpected serving container")
        env = dict(s.split("=", 1) for s in cfg["Env"])
        spill, = [m["Source"] for m in c["Mounts"] if m["Destination"] == "/spill"]
        flags = {k: env[k] for k in (*trial.candidate, switch) if k in env}
        expected = {} if grace is None else {**trial.candidate, switch: str(grace)}
        if flags != expected or (spill == scratch) != (grace is not None):
            raise RuntimeError(f"Runtime settings or spill path differ on rank {len(ranks)}")
        # The benchmark image has its own cache namespace and mount. Everything else
        # that can change the model or serving runtime must match the original.
        omit = {*trial.candidate, switch, "TF_DS_ENGINE_ID", "TF_DS_RECIPE_PATCHES"}
        mounts = sorted((m["Destination"], m["Source"] if m["Destination"] != "/spill" else "/spill", m["RW"]) for m in c["Mounts"])
        ranks.append({
            "image_id": c["Image"], "flags": flags, "spill": spill,
            "common": digest([cfg["Cmd"], sorted((k,v) for k,v in env.items() if k not in omit), mounts, c["HostConfig"]["CpusetCpus"]]),
            "settings": digest([c["Image"], cfg["Cmd"], sorted(env.items()), sorted((m["Source"], m["Destination"], m["RW"]) for m in c["Mounts"])]),
        })
    if grace is not None:
        log = subprocess.run(["docker", "logs", name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=25, check=True).stdout
        if f"lane host finish grace: {grace} ms" not in log:
            raise RuntimeError("Active host did not report the requested grace")
    return ranks


def gate(out, arm, exact):
    cmd = [sys.executable, "-I", "tools/validate.py", trial.server.url, "--model", trial.server.model,
           "--save", str(out / f"{arm}.validate.json"), "--compare", "runs/matrix/base.validate.json"]
    validate = trial.run(cmd, f"{arm}.validate.log")
    result = {"validate": validate}
    if exact:
        rec = str(out / f"{arm}.exact.json")
        result["record"] = trial.run([sys.executable, "-I", "tools/exact.py", "record", "--url", trial.server.url,
                                      "--like", trial.reference, "--label", arm, "--out", rec], f"{arm}.record.log")
        if result["record"] == 0:
            result["compare"] = trial.run([sys.executable, "-I", "tools/exact.py", "compare", trial.reference, rec], f"{arm}.exact.log")
    trial.event(f"{arm} gates {json.dumps(result)}")
    return result


def sample(out, arm, deep):
    done = threading.Event()
    def clocks():
        query = ["nvidia-smi", "--query-gpu=clocks.gr,temperature.gpu,clocks_event_reasons.active", "--format=csv,noheader"]
        with (out / f"{arm}.clocks.jsonl").open("x") as f:
            while not done.is_set():
                try:
                    row = {"at": time.time(), "ranks": [read(p, query) for p in prefixes]}
                except (OSError, subprocess.SubprocessError) as exc:
                    row = {"at": time.time(), "error": str(exc)}
                print(json.dumps(row), file=f, flush=True)
                done.wait(5)
    sampler = threading.Thread(target=clocks)
    sampler.start()
    try:
        workload = ["--finish-grace"]
        if deep:
            workload.append("--grace-depth")
        trial.sample(arm, "9", *workload)
    finally:
        done.set()
        sampler.join()


def run(out, deep_grace_ms):
    trial.out = out
    initial = runtime(base)
    trial.event("production " + json.dumps(initial))
    shim_ids = [read(p, ["docker", "image", "inspect", "-f", "{{.Id}}", image]) for p in prefixes]
    if not Path(trial.reference).is_file() or not Path("runs/matrix/base.validate.json").is_file():
        raise RuntimeError("Missing correctness reference")
    settings = dict(IMAGE=image, PREPARE="0", SPILL_DIR=scratch, WORKER_SPILL_DIR=scratch, **trial.candidate)
    outcomes = {}
    schedule = [("control-a", 5000), ("zero", 0), ("half-second", 500), ("control-b", 5000)]
    deep = deep_grace_ms is not None
    if deep:
        schedule = [(arm, ms) for arm, ms in schedule if ms in (5000, deep_grace_ms)]
    try:
        for arm, ms in schedule:
            trial.idle()
            trial.restart(arm, {**settings, switch: str(ms)})
            ranks = runtime(image, ms)
            if [r["common"] for r in ranks] != [r["common"] for r in initial] or [r["image_id"] for r in ranks] != shim_ids:
                raise RuntimeError("Model, rank or host settings changed beyond the benchmark switches")
            trial.event(f"{arm} runtime {json.dumps(ranks)}")
            sample(out, arm, deep)
            outcomes[arm] = gate(out, arm, exact=(not deep and arm != "control-b"))
    finally:
        if "_error" not in trial.server.metrics():
            trial.idle()
        trial.restart("production", {"PREPARE":"0"})
        restored = runtime(base)
        if restored != initial:
            raise RuntimeError("Original production settings differ after restoration")
        trial.validate("production")
        trial.event("restored " + json.dumps(restored))
    trial.event("complete " + json.dumps(outcomes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--client-dir", type=Path, default=trial.clients,
                        help="Directory containing the version of sample.py used for this run")
    parser.add_argument("--depth-grace-ms", type=int, choices=(0, 500),
                        help="Compare the chosen screened challenger to two 5,000 ms controls at 100K")
    args = parser.parse_args()
    trial.clients = args.client_dir.resolve()
    dest = args.out.resolve()
    with (here / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dest.mkdir(exist_ok=False)
        def stop(sig, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        run(dest, args.depth_grace_ms)
