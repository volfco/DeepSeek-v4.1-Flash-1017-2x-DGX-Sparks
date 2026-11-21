"""Run on the head from the repository root. Default is graphs-on, eight fixed rows.

python3 -I runs/review-round/t15.py --plan
python3 -I runs/review-round/t15.py --parallel 8 --out runs/review-round/t15-r8

The live serving containers are stopped briefly, retained and started again by ID.
Only the offline lanes containers receive the experimental flags. W4's builder
clones the running ranks' model, environment, mounts and RDMA settings without spill.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import uuid

root = pathlib.Path.cwd()
sys.path.insert(0, str(root / "tools"))
import bench_div

server = bench_div.Server("http://127.0.0.1:8000", "DeepSeek-V4.1-Flash-1017")
worker = "10.50.0.11"
serving = "dsv41-1017-exl3-tf"
image = "tensorfold-dsv41:exl3-tp2-234f9080b326"
wave_budgets = (32, 128, 128, 128)
flags = {"TF_DS_T13_EXPERT_PRUNE": "1", "TF_DS_T13_EXPERT_CAP": "12,1", "TF_DS_FIX_K": "0"}
bound = "TF_DS_T15_PLACE_BOUND"
unset = ("TF_DS_EXPERT_BYTES", "CUDA_LAUNCH_BLOCKING", "TF_DS_CLOCK_MARKS", "TF_DS_ROUND_PROFILE")
name_prefix = "tf-t15-" + uuid.uuid4().hex[:10]


def require(cond, message):
    if not cond:
        raise RuntimeError(message)


def output(argv, *, rank=0, timeout=20, must=True):
    args = argv if rank == 0 else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", worker, shlex.join(argv)]
    # The command builder and docker run can contain private inspect values. Never log their argv or failure output.
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, text=True)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{argv[0]} timed out on rank {rank}") from None
    require(not must or p.returncode == 0, f"{argv[0]} failed on rank {rank}, rc={p.returncode}")
    return p.stdout


def inspect(rank, name):
    return json.loads(output(["docker", "inspect", name], rank=rank))[0]


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def signature(c):
    # Docker may reorder Mounts on restart. Preserve every field on every mount.
    mounts = sorted(c["Mounts"], key=lambda mount: mount["Destination"])
    return digest([c["Id"], c["Image"], c["Config"], c["HostConfig"], mounts])


def fixture(n):
    source = root / f"runs/w4/req/r{n}.json"
    original = json.loads(source.read_text())["requests"]
    require(len(original) == n and len({r["name"] for r in original}) == n, "Bad W4 fixture")
    reqs = []
    for i, length in enumerate(wave_budgets):
        for r in original:
            require(all(isinstance(t, int) and t >= 0 for t in r["prompt"]), "Invalid pretokenized input")
            reqs.append({"name": f"w{i}-{r['name']}", "prompt": r["prompt"], "max_tokens": length})
    return {"requests": reqs}


def event(**fields):
    print(json.dumps({"time": time.time(), **fields}), flush=True)


def bench_command(n, files, reqdir, rank, mode):
    name = name_prefix + "-" + str(rank)
    builder = [sys.executable, "-I", "runs/w4/w4_cmd.py", str(rank), *files, name, "--graphs", "1", "--profile", "0"]
    for key, value in {**flags, bound: str(mode)}.items():
        builder.extend(["--env", key + "=" + value])
    for key in unset:
        builder.extend(["--unset", key])
    if rank == 0:
        builder.extend(["--requests", "requests.json", "--parallel", str(n), "--reqdir", str(reqdir.resolve())])
    argv = shlex.split(output(builder))
    require(argv[:3] == ["docker", "run", "-d"] and argv.count("tf-dsv41-lanes") == 1, "Unexpected batch command")
    require(all("/spill" not in a and not a.startswith(("TF_DS_SPILL_", "TF_SPILL_", "--spill")) for a in argv),
            "The batch would have spill access")
    # FIX_K=0 pins model draft depth, but suffix-copy proposals can still be accepted.
    # The batch driver's --serial disables drafting on each stream. --parallel still admits N streams.
    argv.extend(["--serial", "1"])
    return argv


def prove(rank, container, original, mode):
    env = dict(x.split("=", 1) for x in container["Config"]["Env"])
    cmd = container["Config"]["Cmd"]
    require(container["Image"] == original["Image"] and cmd[0] == "tf-dsv41-lanes", "Offline image/entry differs")
    require(cmd[3:5] == [str(rank), "2"] and cmd[cmd.index("--graphs")+1] == "1" and cmd[cmd.index("--profile")+1] == "0"
            and cmd[cmd.index("--serial")+1] == "1",
            "Offline graph, profile, world or rank mismatch")
    require(all(env.get(k) == v for k,v in {**flags,bound:str(mode)}.items()) and all(k not in env for k in unset),
            "Offline flags differ")
    require(all(m["Destination"] != "/spill" for m in container["Mounts"]), "Offline spill mount")
    public = ("TF_DS_ENGINE_ID","TF_DS_MODEL_REVISION","TF_DS_RANK_CACHE_SHA")
    require(all(env.get(k) for k in public), "Missing model or rank cache identity")
    identity = {k:env[k] for k in public}
    signature = digest([container["Image"], cmd, sorted((k,v) for k,v in env.items() if k != bound),
                        sorted((m["Source"],m["Destination"],m["RW"]) for m in container["Mounts"])])
    return {"image_id":container["Image"], "runtime_signature_except_bound":signature, "model":identity,
            "flags":{k:env[k] for k in (*flags,bound)}, "graphs":1, "profile":0}


def stats(log, requests, n):
    rows = [json.loads(line) for line in log.splitlines() if line.startswith("{")]
    inputs = {r["name"]:r for r in rows if "prompt_tokens" in r}
    replies = {r["name"]:r for r in rows if "token_sha" in r}
    summary, = [r for r in rows if "requests" in r]
    steps = sum(k - 1 for k in wave_budgets)
    require(summary["requests"] == len(requests) == len(inputs) == len(replies) and
            summary["steps"] == summary["shared_rounds"] == steps and summary["drafted"] == summary["accepted"] == 0,
            "Requests did not run at the fixed concurrent shape")
    waves = []
    for wave, budget in enumerate(wave_budgets):
        batch = requests[wave*n:(wave+1)*n]
        for req in batch:
            inp, r = inputs[req["name"]], replies[req["name"]]
            require(inp["cached"] == 0 and inp["prompt_tokens"] == len(req["prompt"]) and
                    r["rounds"] == budget-1 and r["tokens"] == budget and r["reason"] == "length" and r["drafted"] == 0,
                    f"Changed work for {req['name']}")
        first, last = batch[0]["name"], batch[-1]["name"]
        # With K=0 every wave admits all N, then the same N perform exactly budget-1 decode rounds.
        waves.append({"wave":wave, "budget":budget, "whole_s":replies[first]["seconds"],
                      "decode_s":round(replies[last]["seconds"] - inputs[last]["prefill_s"],2)})
    hashes = {k:r["token_sha"] for k,r in replies.items()}
    return {"waves":waves, "summary":summary, "hashes":hashes}


def sample_hosts():
    result = []
    for rank in range(2):
        memory_gib = int(output(["grep","MemAvailable","/proc/meminfo"], rank=rank).split()[1]) / 1048576
        gpu = output(["nvidia-smi", "--query-gpu=clocks.gr,temperature.gpu,clocks_event_reasons.active",
                      "--format=csv,noheader"], rank=rank).strip()
        result.append({"memory_gib":round(memory_gib,3), "gpu":gpu})
    return result


def remove_batch(rank, name):
    for _ in range(3):
        output(["docker","rm","-f",name],rank=rank,must=False)
        remaining = output(["docker","ps","-aq","--filter","name=^/" + name + "$"],rank=rank)
        if not remaining.strip():
            return
        time.sleep(2)
    raise RuntimeError(f"Offline container still present on rank {rank}")


def run_cell(n, mode, label, files, dest, pre, reqs):
    cmds = [bench_command(n, files, dest, rank, mode) for rank in range(2)]
    names = [name_prefix + "-" + str(rank) for rank in range(2)]
    try:
        for rank in (1,0):
            output(cmds[rank], rank=rank, timeout=30)
        proof = [prove(rank, inspect(rank,names[rank]), pre[rank], mode) for rank in range(2)]
        event(cell=label, runtime=proof, logical_grid=min(n*7,12+n+1) if mode else n*7)
        start = time.monotonic()
        while time.monotonic() - start < 300:
            states = [output(["docker","inspect","-f","{{.State.Status}}",names[rank]],rank=rank).strip() for rank in range(2)]
            with (dest / f"{label}.hosts.jsonl").open("a") as f:
                sample = sample_hosts()
                f.write(json.dumps({"elapsed":time.monotonic()-start,"ranks":sample})+"\n")
            require(min(x["memory_gib"] for x in sample) >= 3.5, "Offline memory below floor")
            if "exited" in states:
                break
            time.sleep(4)
        else:
            raise RuntimeError("Batch timed out")
        logs = []
        for rank in range(2):
            rc = output(["docker","wait",names[rank]],rank=rank,timeout=30).strip()
            log = output(["docker","logs",names[rank]],rank=rank,timeout=30)
            (dest / f"{label}.rank{rank}.log").write_text(log)
            require(rc=="0" and "BadExpertPrune" not in log and "the cap is off" not in log, f"Batch failed on rank {rank}")
            logs.append(log)
        require(any('"followed": true' in l for l in logs[1].splitlines()), "Follower did not finish")
        metrics = stats(logs[0],reqs,n)
        event(cell=label, **metrics)
        return proof, metrics
    finally:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            removals = [pool.submit(remove_batch,rank,names[rank]) for rank in (0,1)]
            for task in removals:
                task.result()


def stop_original(pre):
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        stops = [pool.submit(output, ["docker","stop","-t","90",c["Id"]],rank=i,timeout=110) for i,c in enumerate(pre)]
        for f in stops:
            f.result()


def restore(pre, dest):
    # A disconnect after a launch could leave a batch running; both GPUs must
    # be released before the original containers begin loading again.
    for rank in range(2):
        remove_batch(rank,name_prefix + "-" + str(rank))
    for attempt in range(2):
        for rank in (1, 0):
            if not inspect(rank, serving)["State"]["Running"]:
                output(["docker","start",pre[rank]["Id"]],rank=rank,timeout=60)
        for _ in range(120):
            try:
                with bench_div.OPENER.open(server.url + "/health",timeout=3) as response:
                    h = json.load(response)
                if h["status"] == "ok" and not h["pair"]["broken"]:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(2)
        else:
            if attempt == 1:
                raise RuntimeError("Production readiness failed twice")
            stop_original(pre)
            continue
        for _ in range(2):
            reply = server.chat([{"role":"user","content":"Reply with OK."}],16)
            if reply.get("completion_tokens"):
                break
        else:
            if attempt == 0:
                stop_original(pre)
                continue
            raise RuntimeError("Restored chat failed")
        post = [inspect(i,serving) for i in range(2)]
        require([signature(c) for c in pre] == [signature(c) for c in post], "Restored running identities changed")
        event(restored=[signature(c) for c in post])
        with (dest / "restore.validate.log").open("w") as f:
            val = subprocess.run([sys.executable,"-I","tools/validate.py",server.url,"--model",server.model,"--save",
                                  str(dest / "restore.validate.json"),"--compare","runs/matrix/base.validate.json"],
                                 stdout=f,stderr=subprocess.STDOUT,timeout=300)
        require(val.returncode == 0, "Restored validation differs")
        return


def run(n, dest, plan):
    reqfile = fixture(n)
    event(parallel=n, waves=wave_budgets, request_sha256=digest(reqfile), schedule=[0,1,1,0],
          grids=[n*7,min(n*7,12+n+1)], rounds_per_load=sum(k-1 for k in wave_budgets))
    if plan:
        return
    pre = [inspect(i,serving) for i in range(2)]
    require(all(c["State"]["Running"] and c["Config"]["Image"] == image and not c["HostConfig"]["AutoRemove"]
                and c["HostConfig"]["RestartPolicy"]["Name"] in ("", "no") for c in pre),
            "Expected two persistent, running production containers")
    for rank,c in enumerate(pre):
        tag_id = output(["docker","image","inspect","-f","{{.Id}}",image],rank=rank).strip()
        require(tag_id == c["Image"],f"Pinned image tag changed on rank {rank}")
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "requests.json").write_text(json.dumps(reqfile))
    original = [signature(c) for c in pre]
    event(original=original)
    with tempfile.TemporaryDirectory() as private:
        files = []
        for rank,c in enumerate(pre):
            p = pathlib.Path(private) / f"r{rank}.json"
            p.write_text(json.dumps([c]))
            files.append(str(p))
        for i in range(2):
            bench_command(n,files,dest,i,0)
        if not bench_div.idle_wait(server,120)[0]:
            raise RuntimeError("Production remained busy")
        m = server.metrics()
        require(m.get("tensorfold:requests_running") == m.get("tensorfold:requests_waiting") == 0, "Production busy")
        def cancel(signum, frame):
            raise KeyboardInterrupt
        for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig,cancel)
        try:
            stop_original(pre)
            proofs, runs = [], []
            for label, mode in zip(("a1","b1","b2","a2"),(0,1,1,0)):
                proof, record = run_cell(n,mode,label,files,dest,pre,reqfile["requests"])
                if proofs:
                    require([p["runtime_signature_except_bound"] for p in proof] == proofs[0],
                            "Offline rank settings changed beyond T15")
                    require(record["hashes"] == runs[0]["hashes"], "Token output differs under the bounded grid")
                proofs.append([p["runtime_signature_except_bound"] for p in proof])
                runs.append(record)
            for field in ("whole_s","decode_s"):
                times = [statistics.median(w[field] for w in r["waves"][1:]) for r in runs]
                event(metric=field, arm_medians=times, paired_changes_pct=[100*(times[1]/times[0]-1),100*(times[2]/times[3]-1)])
        finally:
            for sig in (signal.SIGHUP,signal.SIGTERM,signal.SIGINT):
                signal.signal(sig,signal.SIG_IGN)
            restore(pre,dest)
    event(complete=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parallel",type=int,choices=(4,8),default=8)
    p.add_argument("--out",type=pathlib.Path,default=pathlib.Path("runs/review-round/t15-r8"))
    p.add_argument("--plan",action="store_true",help="Validate the fixture and print the schedule locally")
    a = p.parse_args()
    run(a.parallel,a.out,a.plan)
