#!/usr/bin/env python3
"""Exact-position resume (TF_DS_EXACT_RESUME): a conversation whose turns append 1, 50, 300, 2,000 and 5,000 tokens,
run with the switch off and on, and every turn's full prompt run cold (no prompt cache at all) for its reply's hash.

  # 1. cold rows: the server with the prompt cache off, so every turn's pass runs from position 0
  python3 -I tools/resume_exact.py cold --state S.json --label cold
  # 2. restart the server with the cache on (--prompt-cache-gib 512) and the switch unset
  python3 -I tools/resume_exact.py warm --state S.json --label off
  # 3. restart with TF_DS_EXACT_RESUME=1 (both ranks) and the same cache setting
  python3 -I tools/resume_exact.py warm --state S.json --label on
  # 4. the table and the gates
  python3 -I tools/resume_exact.py report --state S.json [--md report.md]

cold    the conversation ascending, its assistant turns the replies just recorded (so every later prompt is the same
        bytes the warm arms will send); each row must show cached_tokens 0, which needs the server started with
        --prompt-cache-gib 0: the cache keeps and resumes nothing then, so a pass is a cold pass. The switch's state
        does not matter here (nothing is kept).
warm    the same conversation on a server with the prompt cache: per turn the row records prompt/cached tokens, the
        reply's tensorfold.token_sha, prefill_seconds and ttft, and the prompt_cache counters' delta. Turn 1 is cold;
        every later turn appends the previous reply and its own text, so it resumes from what the previous turn kept.
report  per arm and turn: prompt, cached, prefill, TTFT, the reply hash against the cold row's for the same prompt.
        Gates: every warm reply equals its cold one (strong server token_sha on both); with the switch on every turn's
        cached tokens are the previous turn's prompt (the state kept at that pass's end, one token before it when the
        prompt ended on an odd position), and with the switch off they are its chunk grid (a multiple of 2048, at most
        the previous prompt less the window); a cold row is cached 0; a turn after the first resumed something.

The switch is read at start on both ranks (TF_DS_EXACT_RESUME, named in the rank fingerprint): start the whole pair
with it or without it, and restart between arms so the cache does not carry one arm's states into the other's. A row
whose request overlapped another client's is marked (its timings are not the arm's) but its reply still counts.
Stdlib only; checklib for the client.
"""
import argparse
import json
import os
import secrets
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python3 -I does not add the script's folder
import checklib as C  # noqa: E402

DELTAS = [1, 50, 300, 2000, 5000, 1, 300, 2000]   # the tokens each turn appends (the plan's 1/50/300/2,000/5,000)
Q = "Reply with one short sentence about the text above."
PC = "prompt_cache."
KEYS = [PC + k for k in ("memory_hits", "memory_misses", "kept", "evicted", "refused", "failed", "cached_tokens",
                         "computed_tokens", "entries", "held_bytes")]
GRID = 2048      # the engine's prompt chunk (dsv41: its kept states' grid)
WINDOW = 128     # the replay the engine runs past a kept state


def doc(salt, turn, words, seed):
    return f"[ref {salt}-t{turn}]\n" + C.words(words, seed + turn * 7919)


def turn_messages(salt, deltas, seed, replies):
    """The conversation through turn `len(replies)`: each turn's text and question, the recorded replies between."""
    out = []
    for i, n in enumerate(deltas[:len(replies) + 1]):
        out.append({"role": "user", "content": doc(salt, i + 1, n, seed) + "\n\n" + Q})
        if i < len(replies):
            out.append({"role": "assistant", "content": replies[i]})
    return out


class Run:
    def __init__(self, a):
        self.a = a
        self.st = json.load(open(a.state)) if os.path.exists(a.state) else {}

    def save(self):
        tmp = self.a.state + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.st, f, indent=1)
        os.replace(tmp, self.a.state)

    def meta(self, label):
        m = self.st.setdefault("meta", {})
        m.setdefault("salt", self.a.salt or secrets.token_hex(4))
        m.setdefault("seed", self.a.seed)
        m.setdefault("model", self.a.model)
        m.setdefault("max_tokens", self.a.max_tokens)
        m.setdefault("deltas", [int(x) for x in self.a.deltas.split(",") if x])
        m.setdefault("url", self.a.url)
        m.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        self.st.setdefault("arms", {}).setdefault(label, {"turns": {}, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        return m, self.st["arms"][label]

    def server(self, m):
        return C.Server(self.a.url, m["model"], self.a.timeout, m["seed"])

    def row(self, srv, messages, label):
        """One streamed turn: the reply's row with the prompt_cache counters' delta."""
        live = (srv.health() or {}).get("live") or {}
        busy = int(live.get("connections") or 0) > 0
        before = srv.counters()
        r = srv.stream(messages, self.st["meta"]["max_tokens"], keep_text=True)
        time.sleep(1.3)          # rank 1's counters refresh about once a second
        after = srv.counters()
        r["deltas"] = C.deltas(before, after, KEYS)
        r["prefill_s"] = r.get("prefill_seconds")
        r["cached"] = r.get("cached_tokens")
        r["ttft_s"] = r.get("ttft")
        r["arm"] = label
        r["at"] = time.strftime("%H:%M:%S")
        r["cold"] = (r.get("cached_tokens") or 0) == 0
        r["overlap"] = busy
        r.pop("gaps", None)
        C.log(f"  {label:>4} turn {len(self.st['arms'][label]['turns']) + 1}: prompt {r.get('prompt_tokens')} "
              f"cached {r.get('cached_tokens')} prefill {r.get('prefill_seconds')}s ttft {r.get('ttft')}s "
              f"sha {(r.get('token_sha') or '')[:12]} ({r.get('token_sha_source')})"
              + (f" ERROR {r['error']}" if "error" in r else ""))
        return r

    def conversation(self, label):
        """The turns ascending; each one's prompt carries the replies recorded before it (the same bytes every arm)."""
        m, arm = self.meta(label)
        srv = self.server(m)
        if not srv.wait_healthy(self.a.health_timeout):
            C.log("server not healthy")
            return 1
        replies = arm.setdefault("replies", [])
        for i in range(len(m["deltas"])):
            messages = turn_messages(m["salt"], m["deltas"], m["seed"], replies)
            r = self.row(srv, messages, label)
            arm["turns"][str(i + 1)] = r
            self.save()
            if "error" in r:
                C.log(f"  {label} turn {i + 1} failed: {r['error']}: stopping the conversation")
                return 1
            replies.append(r.get("text") or "")
            self.save()
        return 0

    # ------------------------------------------------------------------ phases
    def cold(self):
        return self.conversation(self.a.label)

    def warm(self):
        return self.conversation(self.a.label)

    def report(self):
        st, out, bad = self.st, [], []
        arms = st.get("arms", {})
        meta = st.get("meta", {})
        deltas = meta.get("deltas") or []
        cold = arms.get(self.a.cold, {})
        if not cold:
            bad.append(f"no cold arm ({self.a.cold}): run cold on a server with --prompt-cache-gib 0")
        for label in (self.a.off, self.a.on):
            if not arms.get(label):
                bad.append(f"no {label} arm: run warm --label {label}")
        lines = ["| arm | turn | +tokens | prompt | cached | prefill s | TTFT s | reply == cold |", "|:--|--:|--:|--:|--:|--:|--:|:--|"]
        for label in (self.a.off, self.a.on):
            arm = arms.get(label) or {}
            prev_prompt = None
            for i, n in enumerate(deltas):
                t = str(i + 1)
                r = arm.get("turns", {}).get(t)
                c = cold.get("turns", {}).get(t)
                if not r:
                    bad.append(f"{label} turn {t}: no row")
                    continue
                if "error" in r:
                    bad.append(f"{label} turn {t}: request error: {r['error']}")
                ok, detail = C.same_reply(c, r) if c else (False, "no cold row")
                if not ok and c and c.get("prompt_sha") != r.get("prompt_sha"):
                    detail += " (an earlier turn's reply differed, so this turn's prompt did)"
                if not ok:
                    bad.append(f"{label} turn {t}: reply differs from the cold one: {detail}")
                if (r.get("cached") or 0) > 0 and prev_prompt is not None:
                    if label == self.a.on:
                        # the exact-position resume: the previous pass's end, one token earlier when it ended odd
                        if r["cached"] not in (prev_prompt, prev_prompt - 1):
                            bad.append(f"{label} turn {t}: cached {r['cached']} is not the previous prompt ({prev_prompt}): "
                                       "the previous prompt's end is not a prefix of this turn's (step 0's first check)")
                    elif r["cached"] % GRID != 0 or r["cached"] > prev_prompt - WINDOW:
                        bad.append(f"{label} turn {t}: cached {r['cached']} is not a chunk end before the replay")
                elif prev_prompt is not None and label == self.a.on:
                    bad.append(f"{label} turn {t}: nothing resumed (cached {r.get('cached')})")
                if r.get("overlap"):
                    bad.append(f"{label} turn {t}: overlapped another client (timings not the arm's)")
                lines.append(f"| {label} | {t} | {n:,} | {r.get('prompt_tokens')} | {r.get('cached')} | "
                             f"{r.get('prefill_s') if r.get('prefill_s') is not None else '-'} | "
                             f"{r.get('ttft_s') if r.get('ttft_s') is not None else '-'} | {'yes' if ok else 'NO'} |")
                out.append({"arm": label, "turn": int(t), "delta": n, "prompt_tokens": r.get("prompt_tokens"),
                            "cached": r.get("cached"), "prefill_s": r.get("prefill_s"), "ttft_s": r.get("ttft_s"),
                            "reply_equal": ok, "token_sha": r.get("token_sha"), "deltas": r.get("deltas")})
                prev_prompt = r.get("prompt_tokens") if r.get("prompt_tokens") else prev_prompt
        for t, c in (cold.get("turns") or {}).items():
            if "error" in c:
                bad.append(f"cold turn {t}: request error: {c['error']}")
            elif (c.get("cached") or 0) != 0:
                bad.append(f"cold turn {t}: cached {c['cached']} (the server had the prompt cache on: --prompt-cache-gib 0)")
        md = "\n".join(lines)
        print(md)
        for b in bad:
            print("FAIL " + b)
        print(("PASS" if not bad else "FAIL") + f" resume_exact: {len(deltas)} turns, {len(bad)} problems")
        st["report"] = {"turns": out, "problems": bad, "table": md}
        self.save()
        if self.a.md:
            with open(self.a.md, "w") as fh:
                fh.write(md + "\n")
        return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("cold", "warm", "report"))
    ap.add_argument("--state", required=True, help="the JSON the phases pass on (and the results)")
    ap.add_argument("--url", default=C.default_url())
    ap.add_argument("--model", default=os.environ.get("MODEL", C.DEFAULT_MODEL))
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--health-timeout", type=float, default=900)
    ap.add_argument("--label", default=None, help="cold/warm: this run's arm (cold, off, on)")
    ap.add_argument("--cold", default="cold", help="report: the arm with the prompt cache off (its cold rows)")
    ap.add_argument("--off", default="off", help="report: the arm with TF_DS_EXACT_RESUME unset")
    ap.add_argument("--on", default="on", help="report: the arm with TF_DS_EXACT_RESUME=1")
    ap.add_argument("--deltas", default=",".join(str(d) for d in DELTAS))
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--salt")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--md", help="report: write the table here")
    a = C.use_api_key_args(C.add_api_key_arg(ap).parse_args())
    if a.mode != "report" and not a.label:
        a.label = "cold" if a.mode == "cold" else "on"
    r = Run(a)
    if a.mode != "cold" and not r.st:
        C.log(f"no state in {a.state}: run cold first")
        return 2
    return {"cold": r.cold, "warm": r.warm, "report": r.report}[a.mode]()


if __name__ == "__main__":
    sys.exit(main())
