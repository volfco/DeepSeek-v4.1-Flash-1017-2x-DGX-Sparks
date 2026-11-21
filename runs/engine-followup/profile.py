"""Collect cumulative eager decode phase timings on the pinned serving image, then restore it."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import driver


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def settings():
    result = []
    for prefix in driver.prefixes:
        raw = subprocess.check_output(prefix + ["docker", "inspect", driver.container])
        config = json.loads(raw)[0]["Config"]
        env = dict(s.split("=", 1) for s in config["Env"])
        cmd = config["Cmd"]
        result.append(dict(graphs=env.get("TF_DS_GRAPHS"),
                           worker_graphs=cmd[cmd.index("--graphs") + 1] if "--graphs" in cmd else None,
                           level=env.get("TF_DS_ROUND_PROFILE")))
    return result


def capture(since, dest):
    proc = subprocess.Popen(["docker", "logs", "--timestamps", "--since", str(since), driver.container],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with proc, dest.open("x") as target:
        frame = {}
        for line in proc.stdout:
            start = line.find('{"rank": 0,')
            if start < 0:
                continue
            event = json.loads(line[start:])
            if "profile" in event:
                stamp = line.split()[0].replace("Z", "+00:00")
                frame = {"at": datetime.datetime.fromisoformat(stamp).timestamp(), "profile": event["profile"]}
            elif "round_clock_us" in event and frame:
                frame["clock_rounds"] = event["round_clock_us"]["rounds"]
            elif "phases_ms" in event and frame:
                frame["phases_ms"] = event["phases_ms"]
            elif "profile_note" in event and "phases_ms" in frame and "clock_rounds" in frame:
                print(json.dumps(frame), file=target)
                frame = {}
        if proc.wait():
            raise RuntimeError("Could not read the profile")


def run(out):
    trial = driver.trial
    trial.out = out
    initial = driver.inspect()
    if any(r["tag"] != driver.base or any(r["flags"][k] for k in trial.candidate) for r in initial):
        raise RuntimeError("Expected original serving config")
    write_json(out / "initial.json", initial)
    trial.idle()
    # The tagged image and same model settings reuse the runtime control's private spill.
    spill = "/home/colum/.cache/dsv41-followup-runtime-p1-cap0"
    try:
        flags = dict(trial.candidate, GRAPHS="0", TF_DS_ROUND_PROFILE="2", TF_DS_SCORE="1",
                     IMAGE=driver.base, PREPARE="0", SPILL_DIR=spill, WORKER_SPILL_DIR=spill)
        trial.restart("profile", flags)
        config = settings()
        ranks = driver.inspect()
        write_json(out / "profile.json", {"settings": config, "ranks": ranks})
        if config[0]["graphs"] != "0" or config[1]["worker_graphs"] != "0" or any(r["level"] != "2" for r in config):
            raise RuntimeError("Eager profile not configured on both ranks")
        for current, original in zip(ranks, initial):
            if current["id"] != original["id"] or current["spill"] != spill or any(
                    current["flags"][k] != v for k, v in trial.candidate.items()):
                raise RuntimeError("Image, spill or warm configuration mismatch")
        since = time.time()
        clock_offset = time.time() - time.monotonic()
        rc = trial.run([sys.executable, "-I", str(driver.sample_client), "eager", "9", "--decode-window"],
                       "workload.jsonl")
        clock_end = time.time() - time.monotonic()
        write_json(out / "clock.json", [clock_offset, clock_end])
        time.sleep(.25)
        capture(since, out / "frames.jsonl")
        if rc or abs(clock_end - clock_offset) > .05 or not (out / "frames.jsonl").stat().st_size:
            raise RuntimeError("Incomplete workload, unstable wall clock or missing phase data")
    finally:
        if "_error" not in trial.server.metrics():
            trial.idle()
        trial.restart("production", {"PREPARE": "0"})
        final = driver.inspect()
        if final != initial:
            raise RuntimeError("Production restoration mismatch")
        trial.validate("production")
        write_json(out / "restored.json", final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    with (here / ".pair.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.out.mkdir(exist_ok=False)
        def stop(sig, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        run(args.out.resolve())
