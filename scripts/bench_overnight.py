#!/usr/bin/env python3
"""Overnight A/B runner (head, detached): runs runs/overnight/schedule.json block by block, ABAB against base.

Per block: wait for an idle server (<= 10 min, then proceed and note it), ./start.sh restart with the arm's env (every
switch under test set explicitly), check the image tag, save both ranks' TF_* env (docker inspect) and the armed
banners, start the clock/temperature/power/MemAvailable samplers on both Sparks (after the restart), run the block's
cells through tools/bench_div.py's idle gate + contamination check, save the engine logs (both ranks), done lines and
the Engram histogram delta. Results: runs/overnight/blocks/<id>.json (written after every cell). Status:
runs/overnight/status.log. The schedule is re-read after every block (append to it while this runs). A file
runs/overnight/STOP, or an empty queue, ends the campaign with the production restore (restore.json).
Never kills start.sh.
"""
import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bench_div as bd  # noqa: E402
from overnight_report import arm_status, evidence_status  # noqa: E402

os.chdir(ROOT)
OUT = os.environ.get("CAMPAIGN_OUT", "runs/overnight")
BLK = f"{OUT}/blocks"
CONTAINER = "dsv41-1017-exl3-tf"
IMAGE_TAG = os.environ.get("CAMPAIGN_IMAGE", "exl3-tp2-cdd4715dcbe1")
URL = "http://127.0.0.1:8000"
MODEL = "DeepSeek-V4.1-Flash-1017"
WORKER = next(ln.split("=", 1)[1].strip() for ln in open(".env") if ln.startswith("WORKER="))
CLOCK_MAX_OK = int(os.environ.get("CAMPAIGN_CLOCK_MAX", "2405"))  # 2400 MHz lock + readback jitter
# CAMPAIGN_CLOCK_ABORT=0: record the clock as a covariate instead of ending the campaign over it
CLOCK_ABORT = os.environ.get("CAMPAIGN_CLOCK_ABORT", "1") == "1"
# the validate cell compares against this save (tools/validate.py --compare)
VALIDATE_BASE = os.environ.get("CAMPAIGN_VALIDATE_BASE", "runs/matrix/base.validate.json")

BASE = {"TF_DS_ENGRAM_LEAD": "0", "TF_DS_ENGRAM_ROWCACHE_MB": "0", "TF_DS_ENGRAM_STRICT": "0",
        "TF_DS_ENGRAM_SCALES": "io", "TF_DS_ENGRAM_DEDUPE": "0", "TF_DS_ENGRAM_FAST": "0",
        "TF_DS_T25_DRAFT_CAL": "0", "GRAPH_CAP": "128", "ENGRAM_AIO": "1"}
PROD = {"TF_DS_ENGRAM_LEAD": "1", "TF_DS_ENGRAM_ROWCACHE_MB": "256", "TF_DS_ENGRAM_STRICT": "1", "TF_DS_CACHE_METRICS": "1"}
# CAMPAIGN_BASE / CAMPAIGN_PROD (JSON objects) replace the two dicts: the base every block starts from, and the restore
if os.environ.get("CAMPAIGN_BASE"):
    BASE = json.loads(os.environ["CAMPAIGN_BASE"])
if os.environ.get("CAMPAIGN_PROD") is not None and os.environ.get("CAMPAIGN_PROD") != "":
    PROD = json.loads(os.environ["CAMPAIGN_PROD"])
STD_CELLS = ["warm", "prefill_div", "prefill_code", "prefill_p125", "dec1", "dec4", "dec8", "longctx", "think"]

SAMPLED_PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
                   "Explain how a refrigerator works to a curious ten-year-old.",
                   "Write a Python function that merges overlapping intervals, with a docstring and two tests.",
                   "Describe a walk through a busy market in a city you invent, in vivid detail."]

os.makedirs(BLK, exist_ok=True)


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(f"{OUT}/status.log", "a") as fh:
        fh.write(line + "\n")


def sh(cmd, timeout=120):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ""


def wsh(cmd, timeout=120):
    return sh(f"ssh -o BatchMode=yes -o ConnectTimeout=10 {WORKER} {shlex.quote(cmd)}", timeout)


def inspect(rank):
    fmt = "{{.Config.Image}}|{{.Created}}|{{json .Config.Cmd}}"
    envfmt = "{{range .Config.Env}}{{println .}}{{end}}"
    run = sh if rank == 0 else wsh
    head = run(f"docker inspect {CONTAINER} --format '{fmt}'").strip()
    env = sorted(x for x in run(f"docker inspect {CONTAINER} --format '{envfmt}'").splitlines()
                 if re.match(r"^(TF_|TENSORFOLD_|HF_HUB)", x))
    return {"head": head, "env": env}


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


srv = bd.Server(URL, MODEL)


def restart(flags, logpath, label):
    """Wait for idle (<= 10 min), then ./start.sh restart with flags (retry once). Returns (ok, info)."""
    m = srv.metrics()
    if "_error" in m:
        idle, waited = True, 0.0   # server down: nothing to interrupt
    else:
        idle, waited = bd.idle_wait(srv, 600)
        m = srv.metrics()
    info = {"idle_before": idle, "idle_wait_s": round(waited), "running_before": m.get("tensorfold:requests_running"),
            "server_up_before": "_error" not in m}
    env = {k: v for k, v in os.environ.items() if not re.match(r"^(TF_|TENSORFOLD_|GRAPH_CAP$|ENGRAM_AIO$|SPILL_|RANK1_)", k)}
    env.update(PATH=os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", ""), HF_HUB_OFFLINE="1")
    env.update(flags)
    for attempt in (1, 2):
        t0 = time.time()
        with open(logpath + (f".try{attempt}" if attempt > 1 else ""), "w") as fh:
            rc = subprocess.run(["./start.sh", "restart"], env=env, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, close_fds=True).returncode
        info[f"rc{attempt}"] = rc
        info[f"restart_s{attempt}"] = round(time.time() - t0)
        log(f"{label} restart attempt {attempt} rc={rc} in {info[f'restart_s{attempt}']}s "
            f"(idle_before={idle} waited={info['idle_wait_s']}s running_before={info['running_before']})")
        if rc == 0:
            return True, info
        txt = open(logpath + (f".try{attempt}" if attempt > 1 else "")).read()
        if "the smoke test request failed" in txt and known_first_request_failure():
            # decisions.md 2026-11-15: a fresh process's first request fails (CudaFailed, prompt=8 tokens=1), the
            # retry passes. Accept the restart only when that is the one error and a retried smoke answers.
            r = srv.chat([{"role": "user", "content": "Reply with OK."}], 16)
            info[f"smoke_retry{attempt}"] = {k: r.get(k) for k in ("error", "completion_tokens", "finish")}
            if "error" not in r and r.get("completion_tokens"):
                info["known_first_request_failure"] = True
                log(f"{label} known first-request failure; retried smoke ok")
                return True, info
        if "another ./start.sh is already running" in txt:
            log(f"{label} start.sh lock busy: waiting 5 min")
            time.sleep(300)
        else:
            time.sleep(60)
    return False, info


def known_first_request_failure():
    """The rank-0 log since this start holds exactly one error request, the known CudaFailed one, and nothing else."""
    txt = sh(f"docker logs {CONTAINER} 2>&1", 120)
    ended = re.findall(r"ended \S+ reason=error.*", txt)
    errs = [ln for ln in txt.splitlines() if re.match(r"^error:", ln)]
    return (len(ended) == 1 and "CudaFailed" in ended[0] and "prompt=8" in ended[0]
            and len(errs) == 1 and "CUDA_ERROR_INVALID_VALUE" in errs[0])


def want_tag(flags):
    """the image a block must run: its IMAGE flag (tensorfold-dsv41:<tag>) or the campaign's CAMPAIGN_IMAGE"""
    return flags.get("IMAGE", "").rsplit(":", 1)[-1] or IMAGE_TAG


def image_ok(logpath, tag=None):
    tags = set(re.findall(r"exl3-tp2-[0-9a-f]{12}", open(logpath).read()))
    return tags <= {tag or IMAGE_TAG}, sorted(tags)


class Sampler:
    LOOP = ("while :; do echo \"$(date +%s) $(nvidia-smi --query-gpu=clocks.gr,temperature.gpu,power.draw "
            "--format=csv,noheader,nounits | tr -d ' ') $(awk '/MemAvailable/{print $2}' /proc/meminfo)\"; sleep 3; done")

    def __init__(self, prefix):
        self.prefix = prefix
        self.fh = [open(prefix + ".head.samples", "w"), open(prefix + ".worker.samples", "w")]
        self.p = [subprocess.Popen(["timeout", "14400", "bash", "-c", self.LOOP], stdout=self.fh[0],
                                   stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL),
                  subprocess.Popen(["ssh", "-tt", "-o", "BatchMode=yes", WORKER, f"timeout 14400 bash -c {shlex.quote(self.LOOP)}"],
                                   stdout=self.fh[1], stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)]

    def stop(self):
        for p in self.p:
            p.terminate()
        for p in self.p:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        [f.close() for f in self.fh]
        return {"head": self.summ(self.prefix + ".head.samples"), "worker": self.summ(self.prefix + ".worker.samples")}

    @staticmethod
    def summ(path):
        clk, tmp, pw, mem = [], [], [], []
        for ln in open(path, errors="replace"):
            try:
                ts, csv, mk = ln.split()
                c, t, p = csv.split(",")
                clk.append(int(float(c)))
                tmp.append(int(float(t)))
                pw.append(float(p))
                mem.append(int(mk))
            except ValueError:
                continue
        if not clk:
            return {"n": 0}
        return {"n": len(clk), "clock_max": max(clk), "clock_over_2400": sum(c > 2400 for c in clk),
                "clock_loaded_median": sorted(c for c in clk if c > 1000)[len([c for c in clk if c > 1000]) // 2] if any(c > 1000 for c in clk) else None,
                "temp_min": min(tmp), "temp_max": max(tmp), "power_max": max(pw),
                "mem_low_kb": min(mem), "mem_first_kb": mem[0], "mem_last_kb": mem[-1]}


def engram(m):
    return {k.split(":", 1)[1]: v for k, v in m.items() if "engram_lookup_seconds_count" in k or "engram_lookup_seconds_sum" in k}


# ------------------------------------------------------------------ cells
class Block:
    def __init__(self, bid, corpus):
        self.bid, self.corpus = bid, corpus
        self.salt = secrets.token_hex(4)
        self.res = {}
        self.hay100 = None
        self.n = 0
        self.t0 = time.time()
        self.unclean_run = 0
        self.sampler = None

    BLOCK_CAP_S = int(os.environ.get("CAMPAIGN_BLOCK_CAP_S", str(55 * 60)))
    MAX_UNCLEAN_RUN = 4

    class Bail(Exception):
        pass

    def g(self, fn, n_req, retries=3, check=None):
        if time.time() - self.t0 > self.BLOCK_CAP_S:
            raise Block.Bail("block wall-clock cap")
        if self.unclean_run >= self.MAX_UNCLEAN_RUN:
            raise Block.Bail("traffic: %d consecutive reps without a clean attempt" % self.unclean_run)
        atts = bd.guarded(srv, fn, n_req, retries=retries, check=check)
        if any(a.get("server_down") for a in atts):
            raise Block.Bail("server down")
        self.unclean_run = 0 if any(a["clean"] for a in atts) else self.unclean_run + 1
        return atts

    def tag(self, s):
        self.n += 1
        return f"{s}-{self.n}"

    def prefill(self, kind, size, label, reps, retries=3):
        out = []
        for rep in range(reps):
            def fn():
                tg = self.tag(f"{label}-{size}-{rep}")
                text = self.corpus.text(kind, size, f"{self.salt}-{tg}")
                content = f"[{self.salt}:{tg}] " + text + "\n\nReply with the single word: ok"
                r = srv.chat([{"role": "user", "content": content}], 1)
                self.corpus.learn(kind, len(content), r.get("prompt_tokens"))
                r["kind"], r["size"] = kind, size
                if size >= 90000:
                    r["_hay"] = content[:-len("\n\nReply with the single word: ok")]
                return [r]
            atts = self.g(fn, 1, retries=retries, check=lambda res: all(r.get("cached_tokens") == 0 for r in res))
            for a in atts:
                for r in a["results"]:
                    hay = r.pop("_hay", None)
                    if hay and a["clean"]:
                        self.hay100 = hay
            out.append(atts)
        return out

    def decode(self, label, streams, reps, max_tokens, thinking=False, code=False, retries=3):
        out = []
        for rep in range(reps):
            prompts = []
            for s in range(streams):
                seed = f"dec-{label}-{rep}-{s}"
                if code:
                    prompts.append("Read this Python code, then write a substantially extended, complete version of the "
                                   "module with new functions, docstrings and tests.\n\n```python\n"
                                   + self.corpus.text("code", 400, seed, fixed=True) + "\n```")
                else:
                    prompts.append("Continue the following text. Write at least 800 words, keep going in the same style "
                                   "and do not summarise or stop early.\n\n" + self.corpus.text("div", 250, seed, fixed=True))
            atts = self.g(lambda: bd.multi(srv, prompts, max_tokens, thinking), streams, retries=retries)
            for a in atts:
                a["aggregate_tok_s"] = bd.aggregate(a["results"])
            out.append(atts)
        return out

    def longctx(self):
        if not self.hay100:
            return {"skipped": "no clean 100K prompt to resume"}
        out = {}
        for label, streams, reps in (("lc1", 1, 2), ("lc4", 4, 1)):
            cell = []
            for rep in range(reps):
                prompts = [self.hay100 + f"\n\nTask {label}-{rep}-{s}: write a long, detailed commentary on the text above, "
                           "section by section, quoting it. Do not stop early." for s in range(streams)]
                atts = self.g(lambda: bd.multi(srv, prompts, 1024), streams)
                for a in atts:
                    a["aggregate_tok_s"] = bd.aggregate(a["results"])
                cell.append(atts)
            out[label] = cell
        return out

    def hostov(self):
        sysp = "[hostov:" + self.salt + "] " + self.corpus.text("div", 30000, f"{self.salt}-hostov")
        out = []
        for i in range(11):
            msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": f"Question {i}: in one sentence, what is the text about?"}]
            atts = self.g(lambda: [srv.chat(msgs, 16)], 1)
            out.append(atts)
        return out

    def k_seed(self):
        msgs = [{"role": "user", "content": "[ksess:" + self.salt + "] " + self.corpus.text("div", 6000, f"ksess-{self.salt}")
                 + "\n\nSummarise the text above in three sentences."}]
        turns = []
        for t in range(2):
            r = srv.chat(msgs, 96)
            turns.append(r)
            if "error" in r:
                break
            msgs = msgs + [{"role": "assistant", "content": f"(turn {t} reply)"},
                           {"role": "user", "content": f"Now list {t + 3} keywords from it."}]
        save(f"{OUT}/k_session.json", {"block": self.bid, "messages": msgs, "t": time.time()})
        return {"turns": turns, "prompt_tokens": [r.get("prompt_tokens") for r in turns]}

    def k_resume(self):
        try:
            sess = json.load(open(f"{OUT}/k_session.json"))
        except OSError:
            return {"skipped": "no k_session.json"}
        m0 = srv.metrics()
        r = srv.chat(sess["messages"], 32)
        m1 = srv.metrics()
        spill = {k: (m0.get(k), m1.get(k)) for k in m1 if "spill" in k}
        return {"seeded_in": sess["block"], "result": r, "spill_metrics": spill}

    def run(self, cells, doc, path):
        for c in cells:
            t0 = time.time()
            try:
                r = self.cell(c)
            except Block.Bail as exc:
                doc["cells"][c] = {"bailed": str(exc)}
                doc["bailed"] = f"{c}: {exc}"
                save(path, doc)
                log(f"{self.bid} cell {c} BAILED: {exc}")
                return "server down" not in str(exc)
            except Exception as exc:  # noqa: BLE001
                doc["cells"][c] = {"exception": repr(exc)}
                save(path, doc)
                log(f"{self.bid} cell {c} EXCEPTION {exc!r}")
                if not server_alive():
                    return False
                continue
            doc["cells"][c] = r
            doc["cell_s"][c] = round(time.time() - t0)
            save(path, doc)
            nclean, natt = count_clean(r)
            log(f"{self.bid} cell {c}: {round(time.time() - t0)}s attempts={natt} clean={nclean}")
            if self.sampler is not None:
                h = Sampler.summ(self.sampler.prefix + ".head.samples")
                w = Sampler.summ(self.sampler.prefix + ".worker.samples")
                if CLOCK_ABORT and (h.get("clock_max", 0) > CLOCK_MAX_OK + 20 or w.get("clock_max", 0) > CLOCK_MAX_OK + 20):
                    log(f"{self.bid} CLOCK OVER CAP after {c}: head {h.get('clock_max')} worker {w.get('clock_max')}")
                    doc["clock_abort"] = True
                    return True
            if not server_alive():
                log(f"{self.bid} SERVER DOWN after cell {c}")
                doc["broken_after"] = c
                return False
        return True

    def cell(self, c):
        if True:
            if c == "warm":
                # one discarded warm-up per size class; also the cold first-request figure
                r = {str(s): self.prefill("div", s, "warm", 1, retries=0) for s in (2000, 8000, 32000, 100000)}
            elif c == "prefill_div":
                r = {str(s): self.prefill("div", s, "pd", n) for s, n in ((2000, 3), (8000, 3), (32000, 3), (100000, 2))}
            elif c == "prefill_quick":   # the screen's prefill: 8K and 32K, 2 reps each
                r = {str(s): self.prefill("div", s, "pd", 2) for s in (8000, 32000)}
            elif c == "prefill_code":
                r = {"8000": self.prefill("code", 8000, "pc", 3)}
            elif c == "prefill_p125":
                r = {"8000": self.prefill("p125", 8000, "pp", 3)}
            elif c in ("dec1", "dec1b"):   # dec1b: the same prompts again later in the block (soak before/after)
                r = self.decode("d1", 1, 3, 512)
            elif c == "dec8b":
                r = self.decode("d8", 8, 3, 512)
            elif c == "soak":
                # uncached 100K prefills back to back (fresh salt each): drives the process past the ~400K-token
                # onset the row cache showed; per-rep tok/s shows any drift inside the soak itself
                r = {"100000": self.prefill("div", 100000, "soak", int(os.environ.get("CAMPAIGN_SOAK_REPS", "15")))}
            elif c == "dec4":
                r = self.decode("d4", 4, 2, 512)
            elif c == "dec8":
                r = self.decode("d8", 8, 3, 512)
            elif c == "longctx":
                r = self.longctx()
            elif c == "think":
                r = {"t1": self.decode("t1", 1, 2, 768, thinking=True, code=True),
                     "t4": self.decode("t4", 4, 1, 768, thinking=True, code=True)}
            elif c == "resend":
                # 0170/0171: a cold prompt, then the byte-identical prompt again (the second's cached_tokens and TTFT)
                r = {}
                for size in (8000, 32000):
                    cell = []
                    for rep in range(3):
                        def fn(size=size, rep=rep):
                            tg = self.tag(f"rs-{size}-{rep}")
                            content = (f"[{self.salt}:{tg}] " + self.corpus.text("div", size, f"{self.salt}-{tg}")
                                       + "\n\nReply with the single word: ok")
                            a = srv.chat([{"role": "user", "content": content}], 1)
                            b = srv.chat([{"role": "user", "content": content}], 1)
                            a["leg"], b["leg"] = "cold", "resend"
                            return [a, b]
                        cell.append(self.g(fn, 2, check=lambda res: res[0].get("cached_tokens") == 0))
                    r[str(size)] = cell
            elif c == "longreply":
                # 0163: long streamed replies (the per-round whole-text scans grow with the reply)
                r = {"l1": self.decode("l1", 1, 1, 6000, thinking=True, code=True),
                     "l4": self.decode("l4", 4, 1, 3000, thinking=True, code=True)}
            elif c == "t4x":
                # thinking at 4 streams, more reps (the think cell has one t4 rep; the stack showed a low tail)
                r = {"t4": self.decode("t4", 4, int(os.environ.get("CAMPAIGN_T4_REPS", "6")), 768, thinking=True, code=True)}
            elif c == "validate":
                out = f"{BLK}/{self.bid}.validate.json"
                p = subprocess.run(["python3", "-I", "tools/validate.py", URL, "--model", MODEL, "--save", out,
                                    "--compare", VALIDATE_BASE], capture_output=True, text=True, timeout=1800)
                r = {"rc": p.returncode, "compare": VALIDATE_BASE, "tail": p.stdout[-1500:] + p.stderr[-500:]}
                log(f"{self.bid} validate rc={p.returncode} {p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ''}")
            elif c.startswith("exact"):
                # exact (record only, the campaign's base record) or exact:<base record>: record --like it, then compare
                out = f"{BLK}/{self.bid}.exact.json"
                like = c.split(":", 1)[1] if ":" in c else None
                cmd = ["python3", "-I", "tools/exact.py", "record", "--out", out, "--label", self.bid, "--url", URL]
                if like:
                    cmd += ["--like", like]
                t0 = time.time()
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
                r = {"record_rc": p.returncode, "record_s": round(time.time() - t0), "tail": p.stdout[-800:] + p.stderr[-400:]}
                if like and p.returncode == 0:
                    q = subprocess.run(["python3", "-I", "tools/exact.py", "compare", like, out], capture_output=True,
                                       text=True, timeout=600)
                    r.update(compare_rc=q.returncode, compare_tail=q.stdout[-1500:] + q.stderr[-300:])
                log(f"{self.bid} exact record_rc={p.returncode} in {r['record_s']}s compare_rc={r.get('compare_rc')}")
            elif c == "sampled_ident":
                # seeded temperature-1 single-stream replies (production's sampler settings), twice each: does a sampled
                # reply repeat within a process (and, offline, across arms)? token_sha per seed, both passes.
                r = {"passes": []}
                for _ in range(2):
                    shas = {}
                    for i in range(4):
                        body = {"model": MODEL, "max_tokens": 200, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                                "seed": 9100 + i, "chat_template_kwargs": {"enable_thinking": False},
                                "messages": [{"role": "user", "content": SAMPLED_PROMPTS[i]}]}
                        req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                                     {"Content-Type": "application/json"})
                        try:
                            with bd.OPENER.open(req, timeout=600) as resp:
                                j = json.load(resp)
                            txt = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                            sha = (j.get("tensorfold") or {}).get("token_sha")
                            if not sha and txt:
                                sha = "text:" + hashlib.sha256(txt.encode()).hexdigest()[:16]
                            shas[str(9100 + i)] = sha or "error empty reply"
                        except Exception as exc:  # noqa: BLE001
                            shas[str(9100 + i)] = f"error {exc}"
                    r["passes"].append(shas)
                r["repeat_within"] = r["passes"][0] == r["passes"][1]
                log(f"{self.bid} sampled_ident repeat_within={r['repeat_within']} {r['passes'][0]}")
            elif c in ("reuse", "fanout"):
                # hit-rate probes (tools/prompt_reuse.py, tools/prefix_fanout.py): an agent resuming between other
                # same-system conversations, and common-system fan-out at 1/2/4/8; logged for offline comparison
                if c == "reuse":
                    cmd = ["python3", "-I", "tools/prompt_reuse.py", "12000", "--require-counter"]
                else:
                    cmd = ["python3", "-I", "tools/prefix_fanout.py", "--url", URL, "--model", MODEL,
                           "--out", f"{BLK}/{self.bid}.fanout.json", "--label", self.bid]
                env = dict(os.environ, API_URL=URL)
                t0 = time.time()
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
                r = {"rc": p.returncode, "s": round(time.time() - t0), "out": p.stdout[-3000:], "err": p.stderr[-800:]}
                log(f"{self.bid} {c} rc={p.returncode} in {r['s']}s {p.stdout.strip().splitlines()[-1][:160] if p.stdout.strip() else ''}")
            elif c == "hostov":
                r = self.hostov()
            elif c == "k_seed":
                r = self.k_seed()
            elif c == "k_resume":
                r = self.k_resume()
            else:
                r = {"skipped": "unknown cell"}
        return r


def count_clean(r):
    clean = att = 0
    def walk(x):
        nonlocal clean, att
        if isinstance(x, dict):
            if "clean" in x and "attempt" in x:
                att += 1
                clean += bool(x["clean"])
                return
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(r)
    return clean, att


def server_alive():
    for _ in range(3):
        if "_error" not in srv.metrics():
            return True
        time.sleep(5)
    return False


def engine_logs(prefix):
    sh(f"docker logs -t {CONTAINER} > {prefix}.rank0.log 2>&1", 300)
    with open(f"{prefix}.rank1.log", "w") as fh:
        fh.write(wsh(f"docker logs -t {CONTAINER} 2>&1", 300))
    banners = {}
    for rank in (0, 1):
        txt = open(f"{prefix}.rank{rank}.log", errors="replace").read()
        banners[rank] = {
            "engram_lead_on": "engram-lead: on" in txt,
            "row_cache": re.findall(r"Engram row cache: (\d+) MiB", txt),
            "graphs_destroyed_max": max([int(x) for x in re.findall(r"(\d+) destroyed for newer ones", txt)] or [0]),
            "graphs_kept_lines": len(re.findall(r"CUDA graphs: \d+ kept", txt)),
            "errors": [ln[:300] for ln in txt.splitlines() if re.search(r"\berror\b", ln, re.I)][:20],
        }
    done = {}
    for ln in open(f"{prefix}.rank0.log", errors="replace"):
        m = re.search(r"done (chatcmpl-\w+) (.*)", ln)
        if m:
            done[m.group(1)] = dict(kv.split("=", 1) for kv in m.group(2).split() if "=" in kv)
    return banners, done


def arm_proof(flags, env0, env1, cmd1):
    bad = []
    for k, v in flags.items():
        if k == "GRAPH_CAP":
            if f"TF_DS_GRAPH_CAP={v}" not in env0:
                bad.append(f"rank0 TF_DS_GRAPH_CAP!={v}")
            if f'"--graph-cap","{v}"' not in cmd1:
                bad.append(f"rank1 --graph-cap!={v}")
        elif k == "ENGRAM_AIO":
            if f"TF_DS_ENGRAM_AIO={v}" not in env0:
                bad.append(f"rank0 TF_DS_ENGRAM_AIO!={v}")
            if f'"--aio","{v}"' not in cmd1:
                bad.append(f"rank1 --aio!={v}")
        elif k == "SPILL_MIN_TOKENS":
            for r, e in ((0, env0), (1, env1)):
                if f"TF_DS_SPILL_MIN_TOKENS={v}" not in e:
                    bad.append(f"rank{r} SPILL_MIN_TOKENS!={v}")
        elif k.startswith(("TF_", "TENSORFOLD_")) and k == "TENSORFOLD_TOKEN_CACHE_MB":
            if f"{k}={v}" not in env0:   # rank 0's server reads it; rank 1 gets it too but need not
                bad.append(f"rank0 {k}!={v}")
        elif k.startswith(("TF_", "TENSORFOLD_")):
            for r, e in ((0, env0), (1, env1)):
                if f"{k}={v}" not in e:
                    bad.append(f"rank{r} {k}!={v}")
    return bad


def run_block(entry, corpus):
    bid, arm = entry["id"], entry["arm"]
    flags = dict(BASE)
    flags.update(entry.get("flags", {}))
    cells = entry.get("cells", STD_CELLS)
    path = f"{BLK}/{bid}.json"
    prefix = f"{BLK}/{bid}"
    doc = {"id": bid, "arm": arm, "flags": flags, "cells_planned": cells, "t_start": time.time(), "cells": {}, "cell_s": {}}
    log(f"{bid} {arm} START flags={' '.join(f'{k}={v}' for k, v in flags.items() if BASE.get(k) != v) or '(base)'}")
    if os.environ.pop("ADOPT_FIRST", "") == "1" and not arm_proof(flags, inspect(0)["env"], inspect(1)["env"], inspect(1)["head"]) \
            and os.path.exists(prefix + ".restart.log"):
        ok, info = True, {"adopted": True, "note": "the runner was relaunched; this block adopts the restart it made"}
        log(f"{bid} ADOPTED the running pair (arm proof matches; restart log {prefix}.restart.log)")
    else:
        ok, info = restart(flags, prefix + ".restart.log", bid)
    doc["restart"] = info
    tag = want_tag(flags)
    doc["image"] = tag
    good, tags = image_ok(prefix + ".restart.log", tag)
    doc["image_tags_in_restart_log"] = tags
    if not good:
        log(f"{bid} IMAGE TAG MISMATCH {tags}: aborting campaign")
        doc["status"] = "image-mismatch"
        save(path, doc)
        return "abort"
    if not ok:
        doc["status"] = "restart-failed"
        save(path, doc)
        log(f"{bid} RESTART FAILED twice: skipping arm")
        return "restart-failed"
    i0, i1 = inspect(0), inspect(1)
    doc["inspect"] = {"rank0": i0, "rank1": i1}
    if tag not in i0["head"] or tag not in i1["head"]:
        log(f"{bid} running image not {tag}: aborting")
        doc["status"] = "image-mismatch"
        save(path, doc)
        return "abort"
    doc["arm_proof_problems"] = arm_proof(flags, i0["env"], i1["env"], i1["head"])
    proof = arm_status(doc)
    if proof != "ok":
        doc["status"] = proof
        save(path, doc)
        log(f"{bid} skipping {arm}: {proof}; see saved rank settings and prerequisites")
        return proof
    samp = Sampler(prefix)
    blk = Block(bid, corpus)
    blk.sampler = samp
    doc["salt"] = blk.salt
    m0 = srv.metrics()
    alive = True
    try:
        alive = blk.run(cells, doc, path)
    finally:
        m1 = srv.metrics()
        doc["samples"] = samp.stop()
        doc["engram_before"], doc["engram_after"] = engram(m0), engram(m1)
        doc["banners"], done = engine_logs(prefix)
        doc["done_lines"] = done
        doc["t_end"] = time.time()
        s = doc["samples"]
        cmax = max(s["head"].get("clock_max", 0), s["worker"].get("clock_max", 0))
        doc["clock_valid"] = cmax <= CLOCK_MAX_OK and s["head"].get("n", 0) > 0 and s["worker"].get("n", 0) > 0
        doc["status"] = "ok" if alive else "broken"
        doc["status"] = evidence_status(doc)
        save(path, doc)
    nclean, natt = count_clean(doc["cells"])
    log(f"{bid} {arm} DONE status={doc['status']} clean={nclean}/{natt} clock_max H/W="
        f"{s['head'].get('clock_max')}/{s['worker'].get('clock_max')} valid={doc['clock_valid']} "
        f"mem_low H/W GiB={s['head'].get('mem_low_kb', 0) / 1048576:.1f}/{s['worker'].get('mem_low_kb', 0) / 1048576:.1f} "
        f"temp H/W max={s['head'].get('temp_max')}/{s['worker'].get('temp_max')} "
        f"lead={doc['banners'][0]['engram_lead_on']}/{doc['banners'][1]['engram_lead_on']} rc={doc['banners'][0]['row_cache']} "
        f"graphs_destroyed={doc['banners'][0]['graphs_destroyed_max']}/{doc['banners'][1]['graphs_destroyed_max']}")
    if doc.get("clock_abort") or (CLOCK_ABORT and (s["head"].get("clock_max", 0) > CLOCK_MAX_OK + 20 or s["worker"].get("clock_max", 0) > CLOCK_MAX_OK + 20)):
        return "clock-not-holding"
    return doc["status"]


def restore():
    log(f"RESTORE production: {PROD} (+ head .env)")
    doc = {"t_start": time.time()}
    ok, info = restart(PROD, f"{OUT}/restore.restart.log", "restore")
    doc["restart"] = info
    doc["image_tags"] = image_ok(f"{OUT}/restore.restart.log", want_tag(PROD))[1]
    i0, i1 = inspect(0), inspect(1)
    doc["inspect"] = {"rank0": i0, "rank1": i1}
    ref0 = sorted(x.strip() for x in open(f"{OUT}/production.rank0.env") if x.strip())
    ref1 = sorted(x.strip() for x in open(f"{OUT}/production.rank1.env") if x.strip())
    doc["env_diff_rank0"] = {"missing": sorted(set(ref0) - set(i0["env"])), "extra": sorted(set(i0["env"]) - set(ref0))}
    doc["env_diff_rank1"] = {"missing": sorted(set(ref1) - set(i1["env"])), "extra": sorted(set(i1["env"]) - set(ref1))}
    doc["env_match"] = not any(doc["env_diff_rank0"].values()) and not any(doc["env_diff_rank1"].values())
    doc["banners"], _ = engine_logs(f"{OUT}/restore")
    doc["health"] = sh(f"curl -s -m 10 {URL}/health")
    r = srv.chat([{"role": "user", "content": "Reply with OK."}], 16)
    doc["chat"] = {k: r.get(k) for k in ("error", "prompt_tokens", "completion_tokens", "finish")}
    doc["ok"] = bool(ok and doc["env_match"] and "error" not in r)
    save(f"{OUT}/restore.json", doc)
    log(f"RESTORE done ok={doc['ok']} env_match={doc['env_match']} diff0={doc['env_diff_rank0']} diff1={doc['env_diff_rank1']} "
        f"lead={doc['banners'][0]['engram_lead_on']}/{doc['banners'][1]['engram_lead_on']} rc={doc['banners'][0]['row_cache']}/{doc['banners'][1]['row_cache']} chat={doc['chat']}")
    return doc["ok"]


def main():
    if not os.path.exists(f"{OUT}/corpus/prose.json"):
        log(f"corpus built: {bd.build_corpus(f'{OUT}/corpus', ROOT)}")
    corpus = bd.Corpus(f"{OUT}/corpus")
    log(f"campaign runner start pid={os.getpid()}")
    try:
        campaign(corpus)
    except BaseException as exc:  # noqa: BLE001
        log(f"RUNNER EXCEPTION {exc!r}: going to restore")
    finally:
        restore()
        log("CAMPAIGN DONE")


def campaign(corpus):
    fails = 0
    while True:
        if os.path.exists(f"{OUT}/STOP"):
            log("STOP file: ending campaign")
            break
        sched = json.load(open(f"{OUT}/schedule.json"))
        pending = [e for e in sched if not os.path.exists(f"{BLK}/{e['id']}.json")]
        if not pending:
            log("schedule exhausted")
            break
        res = run_block(pending[0], corpus)
        if res == "abort":
            break
        if res == "clock-not-holding":
            log("CLOCK CAP NOT HOLDING on both Sparks: ending campaign")
            break
        fails = fails + 1 if res == "restart-failed" else 0
        if fails >= 2:
            log("two restart failures in a row: ending campaign")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore()
    elif len(sys.argv) > 1 and sys.argv[1] == "corpus":
        print(bd.build_corpus(f"{OUT}/corpus", ROOT))
    else:
        main()
