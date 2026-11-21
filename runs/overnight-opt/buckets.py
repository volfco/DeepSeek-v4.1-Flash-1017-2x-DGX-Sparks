"""Run fixed-row context-boundary controls in an isolated served process on the benchmark head."""
import argparse
import contextlib
import fcntl
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

here = Path(__file__).resolve().parent
followup = here.parent / "engine-followup"
sys.path[:0] = [str(here), str(followup)]
import driver
from profile import capture, write_json, settings
import t15


def cpu_numbers(spec):
    result = set()
    for group in spec.split(","):
        parts = list(map(int, group.split("-")))
        result.update(range(parts[0], parts[-1] + 1))
    return result


@contextlib.contextmanager
def sample_hosts(out):
    stop=threading.Event()
    def collect():
        with (out/"hosts.jsonl").open("x",buffering=1) as stream:
            while not stop.is_set():
                try:
                    sample={"ranks":t15.sample_hosts()}
                except (OSError,subprocess.SubprocessError,ValueError) as exc:
                    sample={"error":str(exc)}
                print(json.dumps({"at":time.time(),**sample}),file=stream)
                stop.wait(5)
    thread=threading.Thread(target=collect,daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def load_client():
    path = followup / "client/sample.py"
    if not path.exists():
        path = driver.trial.clients / "sample.py"
    spec = importlib.util.spec_from_file_location("fixed_client", path)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    client.args = client.parser.parse_args(["fixed", "9", "--decode-window"])
    return client


def fixture(client, source, target):
    cut = 4 * target
    for _ in range(10):
        context = "[fixed-boundary-1118]\n" + source[:cut] + "\n\n"
        prompts = [context + task for task in client.tasks]
        measured = client.token_count([{"role": "user", "content": prompts[0]}])
        if abs(measured - target) <= 4:
            return prompts, context
        cut += round((target - measured) * cut / measured)
    raise ValueError(f"Could not calibrate {target} tokens")


def restore(out, initial):
    trial = driver.trial
    trial.out = out
    if "_error" not in trial.server.metrics():
        trial.idle()
    trial.restart("production", {"PREPARE": "0"})
    if driver.inspect() != initial:
        raise RuntimeError("Restored pair differs from the recorded production configuration")
    trial.validate("production")
    write_json(out / "restored.json", initial)


def prepare_spill_copy(label, source, destination):
    # These paths belong to this campaign. Neither production tier participates.
    prefix = "/home/colum/.cache/dsv41-baseline-"
    if not (source.startswith(prefix) and destination.startswith(prefix)):
        raise ValueError("Unknown synthetic spill path")
    rows = []
    for pre in driver.prefixes:
        if subprocess.run(pre + ["test", "-e", destination]).returncode == 0:
            raise FileExistsError(destination)
        helper = pre + ["docker","run","--rm","--network","none","--mount",
                        "type=bind,source="+source+",target=/seed,readonly"]
        used = int(subprocess.check_output(helper + ["--entrypoint","du",driver.base,"-sb","/seed"],text=True).split()[0])
        available = int(subprocess.check_output(pre + ["df", "-B1", "--output=avail", source], text=True).splitlines()[-1])
        if used > 8*2**30 or available-used < 50*2**30:
            raise RuntimeError("Synthetic spill copy would consume the reserved disk headroom")
        subprocess.run(pre + ["mkdir",destination],check=True)
        subprocess.run(helper + ["--mount","type=bind,source="+destination+",target=/target",
                       "--entrypoint","cp",driver.base,"-a","--reflink=auto","/seed/.","/target/"],check=True)
        rows.append(dict(bytes=used, available_before=available, source=source, destination=destination))
    write_json(label, rows)


def baseline_series(args):
    out = args.out.resolve()
    out.mkdir()
    initial = driver.inspect()
    if any(r["tag"] != driver.base or any(r["flags"][k] for k in driver.trial.candidate) for r in initial):
        raise ValueError("Expected the restored serving recipe for the baseline crossover")
    write_json(out / "initial.json", initial)
    try:
        if args.baseline_series and (here/"bf16-fix/hot-package.json").is_file():
            import bf16_probe
            try:
                bf16_probe.install(driver)
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
                write_json(out/"bf16-stage-error.json",{"error":str(exc)})
        sequence = (("deployed-a", True, None), ("incumbent", False, 1), ("deployed-return", True, None))
        if args.keyed_series:
            sequence = (("keyed-off", False, 0), ("keyed-return", False, 1))
        for label, deployed, keyed in sequence:
            arm = argparse.Namespace(**vars(args))
            arm.out = out / label
            arm.deployed = deployed
            arm.keyed = keyed
            arm.read_sensitivity = label == "incumbent"
            if args.keyed_series and max(args.depths) > 3072:
                base_dir = "/home/colum/.cache/dsv41-baseline-"
                prepare_spill_copy(out / (label+"-spill.json"), base_dir+"incumbent", base_dir+label)
            run(arm, baseline=initial)
    finally:
        restore(out, initial)


def run(args, baseline=None):
    out = args.out.resolve()
    out.mkdir()
    trial = driver.trial
    client = load_client()
    trial.out = out
    observed = driver.inspect()
    initial = observed if baseline is None else baseline
    if any(r["tag"] != driver.base or any(r["flags"][k] for k in trial.candidate) for r in initial):
        raise ValueError("Expected the restored serving recipe")
    write_json(out / "initial.json", observed)
    trial.idle()
    spill = "/home/colum/.cache/dsv41-fixed-buckets-1118"
    if args.private_spill:
        spill="/home/colum/.cache/dsv41-baseline-"+out.name
    switches = {} if args.deployed else trial.candidate
    if args.switch_profile == "token":
        switches = {"TENSORFOLD_TOKEN_CACHE_MB": trial.candidate["TENSORFOLD_TOKEN_CACHE_MB"]}
    elif args.switch_profile == "host":
        switches = {k:v for k,v in trial.candidate.items() if k.startswith("TF_DS_ENGRAM_") or k=="TENSORFOLD_TOKEN_CACHE_MB"}
    flags = dict(switches, IMAGE=args.image, PREPARE="0", SPILL_DIR=spill, WORKER_SPILL_DIR=spill,
                 GRAPHS=str(int(args.graphs)), TF_DS_ROUND_PROFILE="0" if args.graphs else "2")
    if not args.serving_drafts:
        flags.update(DRAFTS="0", TF_DS_FIX_K="0", TF_DS_NO_COPIES="1")
    if args.stage_trace:
        if args.graphs:raise ValueError("Stage timing requires eager events")
        flags.update(TF_DS_PHASE_WINDOWS="1",TF_DS_PREFILL_PROFILE="1",RANK1_PROFILE="2")
        spill = "/home/colum/.cache/dsv41-stage-"+out.name
        flags.update(SPILL_DIR=spill,WORKER_SPILL_DIR=spill)
    if args.finer:
        flags["TF_DS_IDX_BUCKET_FINER"] = "1"
    if args.keyed is not None:
        flags["TF_DS_T26_KEYED_DRAFT"] = str(args.keyed)
    expected_ids = [subprocess.check_output(pre+["docker","image","inspect","-f","{{.Id}}",args.image], text=True).strip() for pre in driver.prefixes]
    meter = None
    try:
        recipe_check=here/"argmax-fix/recipe-check.py"
        if args.stage_trace and out.name=="stages" and recipe_check.exists():
            spec=importlib.util.spec_from_file_location("recipe_check",recipe_check)
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            try:
                module.check(recipe_check.parent,driver)
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
                write_json(out/"recipe-check-error.json",{"error":str(exc)})
        trial.restart("buckets", flags)
        ranks = driver.inspect()
        cfg = settings()
        write_json(out / "ranks.json", {"ranks": ranks, "settings": cfg})
        for i, rank in enumerate(ranks):
            assert rank["id"] == expected_ids[i] and rank["spill"] == spill
            expected = dict(initial[i]["flags"], **switches)
            assert all(rank["flags"][k] == expected[k] for k in trial.candidate)
        assert cfg[0]["graphs"] == cfg[1]["worker_graphs"] == flags["GRAPHS"]
        armed = []
        for pre in driver.prefixes:
            cfg = json.loads(subprocess.check_output(pre+["docker","inspect",driver.container]))[0]["Config"]
            env = dict(v.split("=",1) for v in cfg["Env"])
            keys = ("TF_DS_IDX_BUCKET_FINER","TF_DS_FIX_K","TF_DS_NO_COPIES","TF_DS_T26_KEYED_DRAFT")
            armed.append({k:env.get(k) for k in keys})
            for k in keys:
                if k in flags and env.get(k) != flags[k]:
                    raise ValueError(f"Missing {k} on a benchmark rank")
        write_json(out/"armed.json",armed)
        if args.meter_seconds:
            meter = subprocess.Popen([sys.executable,"-I",str(here/"pair-meter.py"),str(out/"meter"),"--seconds",str(args.meter_seconds),"--interval-cycles",str(args.interval_cycles)])
            for _ in range(200):
                if all((out/"meter"/f"{rank}-pm.csv").exists() and "anchor," in (out/"meter"/f"{rank}-pm.csv").read_text()[:300] for rank in ("head","worker")):
                    break
                if meter.poll() is not None:
                    raise RuntimeError("Counter sampler exited during startup")
                time.sleep(.1)
            else:
                raise RuntimeError("Counter sampler failed to start")
        since = time.time()
        offset = time.time() - time.monotonic()
        corpus = client.bench.Corpus("runs/combo4/corpus")
        source = corpus.text("div", 360000, "fixed-boundary-1118", fixed=True)
        with sample_hosts(out), (out / "workload.jsonl").open("x", buffering=1) as f, contextlib.redirect_stdout(f):
            client.emit("start",ranks=ranks)
            points = [(boundary + side*1024, boundary) for boundary in args.boundaries for side in (-1, 1)]
            points.extend((depth, None) for depth in args.depths)
            for depth, boundary in points:
                prompts, context = fixture(client, source, depth)
                grid = depth//2048*2048
                client.cell(depth,"chat-prime",[context+"Prime this history."],[4199],budget=16)
                for rep in range(args.reps):
                    for width in args.widths[::1 if rep%2==0 else -1]:
                        budget = args.solo_budget if width==1 else args.burst_budget
                        q = client.cell(depth,"solo" if width==1 else "admission",prompts[:width],
                                        list(range(4200+rep*10,4200+rep*10+width)),
                                        budget=budget, sharing=f"fixed-{width}",rep=rep,expected_cached=grid)
                        if any(((not args.serving_drafts and int(r["done"]["accepted"].split("/")[1])) or
                                (boundary is not None and int(r["prompt_tokens"]) < boundary <= int(r["prompt_tokens"])+r["tokens"]))
                               for r in q):
                            raise RuntimeError("Drafted rows or boundary crossing")
            client.emit("complete",ranks=driver.inspect())
        write_json(out / "clock.json", [offset, time.time()-time.monotonic()])
        if args.prefill_panel:
            import bf16_probe
            bf16_probe.panel(client,corpus,out/"prefill",args.prefill_panel,2)
        if args.read_sensitivity:
            import sensitivity
            dest = out / "sensitivity"
            dest.mkdir()
            offset = time.time() - time.monotonic()
            try:
                with sample_hosts(dest), (dest / "workload.jsonl").open("x", buffering=1) as stream, contextlib.redirect_stdout(stream):
                    client.emit("start", ranks=ranks)
                    sensitivity.measure(client, corpus, dest)
                    client.emit("complete", ranks=driver.inspect())
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                write_json(dest / "error.json", {"error": str(exc)})
            finally:
                write_json(dest / "clock.json", [offset, time.time() - time.monotonic()])
        if not args.graphs:
            capture(since, out/"frames.jsonl")
        if args.stage_trace:
            for i,pre in enumerate(driver.prefixes):
                with (out/f"rank{i}.stages.bin").open("wb") as f:
                    rc=subprocess.run(pre+["docker","exec",driver.container,"cat","/tmp/dsv41-phase-windows.bin"],stdout=f).returncode
                if rc and i==0:raise RuntimeError("Head stage windows missing")
            write_json(out/"timebase.json", {"end_utc":time.time(),"end_monotonic":time.monotonic()})
    finally:
        if baseline is None:
            restore(out, initial)
        if meter is not None and meter.wait(timeout=args.meter_seconds+60):
            raise RuntimeError("Counter collection failed")


def options():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("--image",default=driver.base)
    parser.add_argument("--finer",action="store_true")
    parser.add_argument("--keyed",type=int,choices=[0,1])
    parser.add_argument("--deployed",action="store_true")
    parser.add_argument("--switch-profile",choices=["warm","host","token"],default="warm")
    parser.add_argument("--private-spill",action="store_true")
    parser.add_argument("--serving-drafts",action="store_true")
    parser.add_argument("--boundaries",nargs="*",type=int,default=[131072,262144])
    parser.add_argument("--depths",nargs="*",type=int,default=[])
    parser.add_argument("--widths",nargs="+",type=int,choices=[1,4],default=[1,4])
    parser.add_argument("--reps",type=int,default=2)
    parser.add_argument("--keyed-series",action="store_true")
    parser.add_argument("--baseline-series",action="store_true")
    parser.add_argument("--read-sensitivity",action="store_true")
    parser.add_argument("--prefill-panel",nargs="+",type=int,default=[])
    parser.add_argument("--stage-trace",action="store_true")
    parser.add_argument("--interval-cycles",type=int,default=2400000)
    parser.add_argument("--meter-seconds",type=int,default=0)
    parser.add_argument("--solo-budget",type=int,default=192)
    parser.add_argument("--burst-budget",type=int,default=384)
    parser.add_argument("--graphs",action="store_true")
    return parser


if __name__ == "__main__":
    args=options().parse_args()
    with (followup/".pair.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def stop(sig, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop)
        (baseline_series if args.baseline_series or args.keyed_series else run)(args)
