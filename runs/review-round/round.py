"""Run on the head from the recipe root. Restore the original recipe after the candidate."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
clients = Path(__file__).resolve().parent
out = Path(os.environ.get("REVIEW_RUN_DIR", clients)).resolve()
sys.path.insert(0, str(root / "tools"))
import bench_div

server = bench_div.Server("http://127.0.0.1:8000", "DeepSeek-V4.1-Flash-1017")
image = "tensorfold-dsv41:exl3-tp2-234f9080b326"
candidate = dict(TF_DS_ENGRAM_LEAD="1", TF_DS_ENGRAM_SCALES="map", TF_DS_ENGRAM_STRICT="1",
                 TENSORFOLD_TOKEN_CACHE_MB="32", TF_DS_DEVICE_ACCEPT="1", TF_DS_COMPRESS_SIDE="1",
                 TF_DS_PDL="1", TF_DS_FUSED_COMPRESS2="1")
base_env = dict(os.environ, HF_HUB_OFFLINE="1", IMAGE=image)
base_env["PATH"] = str(Path.home() / ".local/bin") + ":" + base_env["PATH"]
reference = "runs/combo3/blocks/a01.exact.json"


def event(line):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), line, flush=True)


def run(args, log, env=None):
    with (out / log).open("x") as f:
        return subprocess.run(args, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode


def identity():
    result = []
    for prefix in ([], ["ssh", "-o", "BatchMode=yes", "10.50.0.11"]):
        d = json.loads(subprocess.check_output(prefix + ["docker", "inspect", "dsv41-1017-exl3-tf"]))[0]
        if not d["State"]["Running"] or d["Config"]["Image"] != image:
            raise RuntimeError("Expected pinned, running image on both ranks")
        env = dict(x.split("=", 1) for x in d["Config"]["Env"])
        # Compare the full environment and command in memory. Only persist a digest and public switches.
        signature = [d["Image"], d["Config"]["Cmd"], sorted((k,v) for k,v in env.items() if k not in candidate)]
        result.append((hashlib.sha256(json.dumps(signature).encode()).hexdigest(),
                       {k: env[k] for k in candidate if k in env}))
    return result


def idle():
    if not bench_div.idle_wait(server, 120)[0]:
        raise RuntimeError("Pair remained busy")


def restart(arm, flags):
    event("restart " + arm)
    log = arm + ".restart.log"
    rc = run(["./start.sh", "restart"], log, dict(base_env, **flags))
    if rc and "the smoke test request failed" in (out / log).read_text():
        reply = server.chat([{"role": "user", "content": "Reply with OK."}], 16)
        if "error" not in reply and reply.get("completion_tokens"):
            event(f"{arm} start rc={rc}, smoke retry passed")
            return
    if rc:
        raise RuntimeError(f"{arm} restart rc={rc}")


def validate(arm):
    cmd = ["python3", "-I", "tools/validate.py", server.url, "--model", server.model,
           "--save", str(out / (arm + ".validate.json")), "--compare", "runs/matrix/base.validate.json"]
    if run(cmd, arm + ".validate.log"):
        raise RuntimeError("validate failed " + arm)


def sample(arm, salt, *workload):
    event("sample " + arm)
    if run(["python3", "-I", str(clients / "sample.py"), arm, salt, *workload], arm + ".jsonl"):
        raise RuntimeError("sample failed " + arm)


def campaign():
    initial = identity()
    if any(flags for _, flags in initial):
        raise RuntimeError("Expected the production flags")
    if not Path(reference).is_file() or not Path("runs/matrix/base.validate.json").is_file():
        raise RuntimeError("Reference gate record missing")
    event("initial " + json.dumps(initial))
    idle()
    try:
        restart("best-b", candidate)
        if identity() != [(digest, candidate) for digest, _ in initial]:
            raise RuntimeError("Candidate identity mismatch")
        # Time from a fresh restart before validation or exact fills the prefix cache.
        sample("best-b", "2")
        validate("best-b")
        rec = str(out / "best-b.exact.json")
        event("exact record and compare")
        if run(["python3", "-I", "tools/exact.py", "record", "--url", server.url, "--like", reference,
                "--label", "best-b", "--out", rec], "best-b.exact-record.log"):
            raise RuntimeError("exact record failed")
        if run(["python3", "-I", "tools/exact.py", "compare", reference, rec], "best-b.exact-compare.log"):
            raise RuntimeError("exact compare failed")
    finally:
        # Pair may be unhealthy on a failed arm. A healthy, occupied pair gets an idle window.
        if "_error" not in server.metrics():
            idle()
        restart("production-c", {})
        if identity() != initial:
            raise RuntimeError("Restored identity differs")
        event("restored both ranks " + json.dumps(initial))
        try:
            sample("production-c", "3")
        finally:
            validate("production-c")
            event("production validation passed")
    event("complete")


def token_cache_trial():
    initial = identity()
    if any(flags for _, flags in initial):
        raise RuntimeError("Expected the production settings before the token-cache trial")
    flags = {"TENSORFOLD_TOKEN_CACHE_MB": "32"}
    sample("cache-off-a", "6", "--warm-chat")
    idle()
    try:
        restart("cache-on-b", flags)
        if identity() != [(digest, flags) for digest, _ in initial]:
            raise RuntimeError("Token-cache arm settings differ from requested")
        sample("cache-on-b", "6", "--warm-chat")
        validate("cache-on-b")
    finally:
        if "_error" not in server.metrics():
            idle()
        restart("cache-off-c", {})
        if identity() != initial:
            raise RuntimeError("Production restoration differs from baseline")
        event("restored both ranks " + json.dumps(initial))
        try:
            sample("cache-off-c", "6", "--warm-chat")
        finally:
            validate("cache-off-c")
            event("production validation passed")
    event("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-cache", action="store_true")
    cache_trial = parser.parse_args().token_cache
    prefixes = ("cache-off-a", "cache-on-b", "cache-off-c") if cache_trial else ("best-b", "production-c")
    out.mkdir(parents=True, exist_ok=True)
    if any(path.name.startswith(prefixes) for path in out.iterdir()):
        parser.exit(2, "Choose a fresh REVIEW_RUN_DIR for this trial. Existing results are preserved.\n")
    if cache_trial:
        token_cache_trial()
    else:
        campaign()
