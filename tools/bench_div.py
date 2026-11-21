#!/usr/bin/env python3
"""High-diversity benchmark client for the overnight A/B campaign (scripts/bench_overnight.py drives it).

Differences from tools/bench.py (whose prose() draws from ~125 words, so ~16K distinct bigrams exist whatever the salt,
which flatters the Engram row cache and the mapped-scale warm-up):
  - corpus "div": natural-ish English lines from the host's /usr/share/doc, licenses and the repo's markdown, shuffled at
    line level with the salt (a large vocabulary); "code": 30-line blocks of the Python stdlib, shuffled with the salt;
    "p125": bench.py's old 125-word prose, kept for comparison and labelled as such.
  - every prefill prompt starts with a fresh salt (no prefix/spill cache), and cached_tokens is recorded per request
    so the caller can assert it is 0.
  - decode prompts are distinct per rep and per stream (fixed seeds across blocks, so arms decode the same text).
  - every timed rep is gated on an idle server (requests_running == 0) and checked for foreign requests: the
    completed-request counter delta must equal the rep's own requests and nothing may be running at its end.
Stdlib only.
"""
import glob
import gzip
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point "
         "home water room mother area money story fact month lot right study book eye job word business issue "
         "side kind head house service friend father power hour game line end member law car city community name "
         "president team minute idea kid body information back parent face others level office door health person "
         "art war history party result change morning reason research girl guy moment air teacher force education "
         "river mountain signal engine garden theory market winter method bridge letter window voice paper field").split()


# ------------------------------------------------------------------ corpus
def _text_ok(line):
    s = line.strip()
    if not (30 <= len(s) <= 400):
        return None
    letters = sum(ch.isalpha() for ch in s)
    if letters < 0.6 * len(s) or s.count(" ") < 4:
        return None
    return s


def build_corpus(out_dir, repo_root):
    os.makedirs(out_dir, exist_ok=True)
    prose, seen = [], set()
    files = []
    for pat in ("/usr/share/doc/**/README*", "/usr/share/doc/**/*.md", "/usr/share/doc/**/*.txt",
                "/usr/share/doc/**/copyright", "/usr/share/doc/**/changelog*.gz", "/usr/share/doc/**/NEWS*",
                "/usr/share/common-licenses/*"):
        files += glob.glob(pat, recursive=True)
    files += glob.glob(os.path.join(repo_root, "**/*.md"), recursive=True)
    for f in sorted(set(files)):
        if "/runs/" in f:
            continue
        try:
            if f.endswith(".gz"):
                with gzip.open(f, "rt", errors="replace") as fh:
                    lines = fh.read(400000).splitlines()
            else:
                with open(f, errors="replace") as fh:
                    lines = fh.read(400000).splitlines()
        except (OSError, EOFError, gzip.BadGzipFile):
            continue
        for ln in lines:
            s = _text_ok(ln)
            if s and s not in seen:
                seen.add(s)
                prose.append(s)
    code = []
    for f in sorted(glob.glob("/usr/lib/python3.12/**/*.py", recursive=True)):
        if "/test" in f or "/idlelib/" in f:
            continue
        try:
            with open(f, errors="replace") as fh:
                lines = [ln.rstrip() for ln in fh.read().splitlines()]
        except OSError:
            continue
        for i in range(0, len(lines) - 30, 30):
            blk = "\n".join(lines[i:i + 30])
            if len(blk.strip()) > 200:
                code.append(blk)
    with open(os.path.join(out_dir, "prose.json"), "w") as fh:
        json.dump(prose, fh)
    with open(os.path.join(out_dir, "code.json"), "w") as fh:
        json.dump(code, fh)
    words = set()
    for s in prose:
        words.update(re.findall(r"[A-Za-z]+", s.lower()))
    return {"prose_lines": len(prose), "prose_chars": sum(map(len, prose)), "prose_vocab": len(words),
            "code_blocks": len(code), "code_chars": sum(map(len, code))}


class Corpus:
    # chars per token, adapted from the server's prompt_tokens as requests come back
    cpt = {"div": 4.0, "code": 3.2, "p125": None}

    def __init__(self, d):
        with open(os.path.join(d, "prose.json")) as fh:
            self.prose = json.load(fh)
        with open(os.path.join(d, "code.json")) as fh:
            self.code = json.load(fh)
        self.lock = threading.Lock()

    FIXED_CPT = {"div": 4.0, "code": 3.2}

    def text(self, kind, tokens, seed, fixed=False):
        rng = random.Random(seed)
        if kind == "p125":
            size = int(tokens / 0.82)
            out = []
            while len(out) < size * 0.72:
                sent = [rng.choice(WORDS) for _ in range(rng.randint(6, 16))]
                out += sent[:-1] + [sent[-1] + "."]
            return " ".join(out)
        target = tokens * (self.FIXED_CPT[kind] if fixed else self.cpt[kind])
        parts, n = [], 0
        if kind == "div":
            while n < target:
                para = [rng.choice(self.prose) for _ in range(rng.randint(3, 8))]
                p = " ".join(para)
                parts.append(p)
                n += len(p) + 2
            return "\n\n".join(parts)
        while n < target:
            b = rng.choice(self.code)
            parts.append(b)
            n += len(b) + 2
        return "\n\n".join(parts)

    def learn(self, kind, chars, prompt_tokens, overhead=30):
        if kind not in ("div", "code") or not prompt_tokens or prompt_tokens < 1000:
            return
        with self.lock:
            obs = chars / max(1, prompt_tokens - overhead)
            self.cpt[kind] = 0.7 * self.cpt[kind] + 0.3 * obs


# ------------------------------------------------------------------ server
class Server:
    def __init__(self, url, model):
        self.url, self.model = url.rstrip("/"), model

    def metrics(self):
        try:
            with OPENER.open(self.url + "/metrics", timeout=10) as r:
                raw = r.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            return {"_error": str(exc)}
        m = {}
        for ln in raw.splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            k, _, v = ln.rpartition(" ")
            try:
                m[k] = float(v)
            except ValueError:
                pass
        return m

    def chat(self, messages, max_tokens, thinking=False, start_gate=None, timeout=3600):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 1234,
                "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": thinking}}
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        if start_gate is not None:
            start_gate.wait()
        t0 = time.monotonic()
        first = last = None
        usage = server = rid = None
        events, finish, ntext = 0, None, 0
        try:
            with OPENER.open(req, timeout=timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ev = json.loads(data)
                    rid = rid or ev.get("id")
                    if ev.get("usage"):
                        usage = ev["usage"]
                    if ev.get("tensorfold"):
                        server = ev["tensorfold"]
                    for ch in ev.get("choices") or []:
                        delta = ch.get("delta") or {}
                        piece = delta.get("content") or ""
                        if piece or delta.get("reasoning_content") or delta.get("reasoning"):
                            now = time.monotonic()
                            first = first or now
                            last = now
                            events += 1
                            ntext += len(piece)
                        finish = ch.get("finish_reason") or finish
        except urllib.error.HTTPError as exc:
            return {"error": f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}
        end = time.monotonic()
        if first is None or usage is None:
            return {"error": f"no tokens or no usage (events={events})", "id": rid}
        pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        ttft = first - t0
        r = {"id": rid, "prompt_tokens": pt, "completion_tokens": ct, "cached_tokens": cached, "ttft_s": ttft,
             "prefill_tok_s": (pt - (cached or 0)) / ttft if ttft > 0 else None,
             "decode_tok_s": (ct - 1) / (last - first) if ct > 1 and last > first else None,
             "total_s": end - t0, "finish": finish, "t_first": first, "t_last": last}
        if server:
            r["server"] = server
        return r


def idle_wait(srv, max_s):
    """Wait until nothing runs or waits; returns (idle, waited_s)."""
    t0 = time.monotonic()
    errs = 0
    while True:
        m = srv.metrics()
        errs = errs + 1 if "_error" in m else 0
        if errs >= 6:
            return False, time.monotonic() - t0
        if "_error" not in m and m.get("tensorfold:requests_running", 1) == 0 and m.get("tensorfold:requests_waiting", 0) == 0:
            return True, time.monotonic() - t0
        if time.monotonic() - t0 > max_s:
            return False, time.monotonic() - t0
        time.sleep(0.5)


DONE_KEY = "tensorfold:request_latency_seconds_count"


def guarded(srv, fn, n_req, retries=3, idle_max=120, check=None):
    """Run fn() (which sends n_req requests and returns (results, meta)) until a clean attempt, at most 1+retries
    times. Clean: idle at the start, completed-request delta == n_req, nothing running/waiting at the end, no errors."""
    attempts = []
    for a in range(retries + 1):
        idle, waited = idle_wait(srv, idle_max)
        m0 = srv.metrics()
        if "_error" in m0:
            attempts.append({"attempt": a, "t": time.time(), "clean": False, "server_down": True, "results": []})
            break
        t0 = time.time()
        res = fn()
        deadline = time.monotonic() + 5
        while True:
            m1 = srv.metrics()
            delta = m1.get(DONE_KEY, 0) - m0.get(DONE_KEY, 0)
            if delta >= n_req or time.monotonic() > deadline:
                break
            time.sleep(0.3)
        errs = [r.get("error") for r in res if "error" in r]
        # the running gauge lags the completed counter by a moment: give it up to 10 s to drain
        deadline = time.monotonic() + 10
        while True:
            running_end = m1.get("tensorfold:requests_running", -1) + m1.get("tensorfold:requests_waiting", 0)
            if running_end == 0 or time.monotonic() > deadline:
                break
            time.sleep(0.5)
            m1 = srv.metrics()
        delta = m1.get(DONE_KEY, 0) - m0.get(DONE_KEY, 0)
        check_ok = True if check is None or errs else bool(check(res))
        clean = bool(idle and delta == n_req and running_end == 0 and not errs and check_ok)
        att = {"attempt": a, "t": t0, "idle_start": idle, "idle_wait_s": round(waited, 1), "done_delta": delta,
               "own": n_req, "running_end": running_end, "check_ok": check_ok, "errors": errs, "clean": clean, "results": res}
        attempts.append(att)
        if clean:
            break
        if errs and any("Connection refused" in (e or "") or "rank pair" in (e or "") for e in errs):
            break
    return attempts


def multi(srv, prompts, max_tokens, thinking=False):
    n = len(prompts)
    gate = threading.Barrier(n)
    res = [None] * n

    def work(i):
        res[i] = srv.chat([{"role": "user", "content": prompts[i]}], max_tokens, thinking, gate)
    ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return res


def aggregate(res):
    ok = [r for r in res if "error" not in r]
    if not ok:
        return None
    window = max(r["t_last"] for r in ok) - min(r["t_first"] for r in ok)
    total = sum(r["completion_tokens"] for r in ok)
    return (total - len(ok)) / window if window > 0 else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None
