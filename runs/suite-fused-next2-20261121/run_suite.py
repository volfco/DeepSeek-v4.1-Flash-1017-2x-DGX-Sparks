#!/usr/bin/env python3
"""The repo's campaign runner (scripts/bench_overnight.py, unchanged) with this suite's extra cells.

  probe           this folder's suite_probe.py: tokens a round on fixed prompts, joined with the done lines
  deep300         one cold 300K natural-text prefill, then the runner's long-context cell on that prompt
  fixed_replay    todo "Measure patches 0254-0258 on both Sparks": unsalted fixed-input greedy replay at ~49,967,
                  ~99,904 and ~200,000 tokens (8,192- and 16,384-block decode buckets, then the >16,384 fallback
                  decode and a deep prefill). The text is seeded with fixed=True, so every block sends identical
                  bytes and therefore identical input token IDs; the gate joins done_lines[id] for rounds/decode_s
                  and compares prompt/text/token hashes across blocks.
  dec8_staggered  runs/pre-bench-20261121/dec8-audit/dec8_replay_audit.py record --admission-mode staggered: the dec8
                  replay with locked admission order (fixes the Wave-0 admission-order artifact the dec8 audit
                  found), cross-checked against this suite's own s01 dec8 cell prompts and hashes.

Run from the recipe root on the head with CAMPAIGN_OUT, CAMPAIGN_IMAGE, CAMPAIGN_PROD and CAMPAIGN_CLOCK_ABORT set.
The runner restores production from .env when the schedule is done, when OUT/STOP exists, or on any exception.
"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bench_overnight as bo  # noqa: E402

_cell = bo.Block.cell


def cell(self, c):
    if c == "probe":
        out = f"{bo.BLK}/{self.bid}.probe.json"
        p = subprocess.run(["python3", f"{bo.OUT}/suite_probe.py", self.bid, out, "--narrow"], capture_output=True, text=True, timeout=2400)
        rows = json.load(open(out)) if os.path.exists(out) else []
        bo.log(f"{self.bid} probe rc={p.returncode} rows={len(rows)} failed={sum(1 for r in rows if r.get('failed'))}")
        return {"rc": p.returncode, "rows": [{k: v for k, v in r.items() if k != "per_round"} for r in rows],
                "tail": p.stdout[-600:] + p.stderr[-400:]}
    if c == "deep300":
        r = {"300000": self.prefill("div", 300000, "deep", 1, retries=1)}
        r["longctx"] = self.longctx()
        return r
    if c == "fixed_replay":
        out = {}
        for name, size in (("len_50000", 49967), ("len_100000", 99904), ("len_200000", 200000)):
            text = self.corpus.text("div", size, f"fixed-replay-{name}", fixed=True)
            content = text + "\n\nReply with the single word: ok"
            sha = hashlib.sha256(content.encode()).hexdigest()[:24]

            def fn(content=content, sha=sha):
                r = bo.srv.chat([{"role": "user", "content": content}], 512)
                r["text_sha"] = sha
                return [r]

            # no cached_tokens == 0 check here: a retry after a failed attempt would be warm by construction; the
            # gate flags any compared row with cached_tokens != 0 instead.
            atts = self.g(fn, 1, retries=2, check=lambda res: all("error" not in r for r in res))
            out[name] = atts
            rows = [r for a in atts if a.get("clean") for r in a["results"]]
            bo.log(f"{self.bid} fixed_replay {name} clean={bool(rows)} prompt_tokens={[r.get('prompt_tokens') for r in rows]} "
                   f"text_sha={sha} cached={[r.get('cached_tokens') for r in rows]}")
        return out
    if c == "dec8_staggered":
        out = f"{bo.BLK}/{self.bid}.dec8st.json"
        # s01's own block JSON is written incrementally but its done_lines only at block end: only cross-check
        # against s01 from s02 on.
        exp = bo.OUT if self.bid != "s01" and os.path.exists(f"{bo.BLK}/s01.json") else ""
        cmd = ["python3", "runs/pre-bench-20261121/dec8-audit/dec8_replay_audit.py", "record",
               "--label", self.bid, "--out", out, "--admission-mode", "staggered", "--stagger-ms", "2.5",
               "--reps", str(int(os.environ.get("CAMPAIGN_DEC8_REPS", "3"))), "--max-tokens", "512",
               "--corpus-dir", f"{bo.OUT}/corpus"] + (["--expected-suite", exp] if exp else [])
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
        r = {"rc": p.returncode, "tail": p.stdout[-800:] + p.stderr[-400:]}
        if os.path.exists(out):
            r["rows"] = json.load(open(out))
        bo.log(f"{self.bid} dec8_staggered rc={p.returncode} waves={len((r.get('rows') or {}).get('waves') or [])}")
        return r
    return _cell(self, c)


bo.Block.cell = cell
CAMPAIGN_T0 = __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime())

_run_block = bo.run_block


def run_block(entry, corpus):
    # suite-fused-next: only requests since the campaign started count (production's container carries a real
    # client's history from before the launch window; launch.sh waits for 30 idle minutes before starting)
    seen = bo.sh(f"docker logs --since {CAMPAIGN_T0} {bo.CONTAINER} 2>&1 | grep 'POST /v1/' | grep -v '\\[tensorfold\\] 127\\.0\\.0\\.1 ' | head -3")
    if seen.strip():
        bo.log(f"{entry['id']} NON-LOOPBACK REQUESTS in the running container's log: stopping the campaign: {seen.strip()[:300]}")
        with open(f"{bo.OUT}/STOP", "w") as fh:
            fh.write("a non-loopback client was seen\n")
        return "abort"
    return _run_block(entry, corpus)


bo.run_block = run_block


# neither arm runs an exl3-tp2-<hash> tag: read both ranks' running image instead of grepping restart logs
def image_ok(logpath, tag=None):
    want = tag or bo.IMAGE_TAG
    fmt = "docker inspect -f '{{.Config.Image}}' " + bo.CONTAINER
    tags = sorted({t.strip().rsplit(":", 1)[-1] for t in (bo.sh(fmt), bo.wsh(fmt)) if t.strip()})
    return tags == [want], tags


bo.image_ok = image_ok

if __name__ == "__main__":
    bo.main()
