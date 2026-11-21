#!/usr/bin/env python3
"""The repo's campaign runner (scripts/bench_overnight.py, unchanged) with two more cells for this suite.

  probe    runs/suite-fused-20261121/suite_probe.py: tokens a round on fixed prompts (the draft-block probe's 15 rows, then
           four more code-like prompts and one with thinking on), joined with the done lines
  deep300  one cold 300K natural-text prefill, then the runner's long-context cell on that prompt: warm turns with
           the short uncached tail at one and four streams, 1,024 tokens each

Run from the recipe root on the head with CAMPAIGN_OUT, CAMPAIGN_IMAGE, CAMPAIGN_PROD and CAMPAIGN_CLOCK_ABORT set. The
runner restores production from .env when the schedule is done, when OUT/STOP exists, or on any exception.
"""
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
    return _cell(self, c)


bo.Block.cell = cell

_run_block = bo.run_block


def run_block(entry, corpus):
    # The runner waits ten minutes for an idle server and then restarts anyway. Before each block, look at the
    # running container's log: a request that did not come from this machine means a real client is here, so stop
    # (the runner then goes straight to its production restore).
    seen = bo.sh(f"docker logs {bo.CONTAINER} 2>&1 | grep 'POST /v1/' | grep -v '\\[tensorfold\\] 127\\.0\\.0\\.1 ' | head -3")
    if seen.strip():
        bo.log(f"{entry['id']} NON-LOOPBACK REQUESTS in the running container's log: stopping the campaign: {seen.strip()[:300]}")
        with open(f"{bo.OUT}/STOP", "w") as fh:
            fh.write("a non-loopback client was seen\n")
        return "abort"
    return _run_block(entry, corpus)


bo.run_block = run_block


# suite-fused-next-20261121 (copied from suite-fused-20261121): neither arm runs an exl3-tp2-<hash> tag.
# The stock check greps the restart log for exl3-tp2 tags (the stopped container's tag can appear there), so read both
# ranks' running image instead. run_block's own docker-inspect check still follows it.
def image_ok(logpath, tag=None):
    want = tag or bo.IMAGE_TAG
    fmt = "docker inspect -f '{{.Config.Image}}' " + bo.CONTAINER
    tags = sorted({t.strip().rsplit(":", 1)[-1] for t in (bo.sh(fmt), bo.wsh(fmt)) if t.strip()})
    return tags == [want], tags


bo.image_ok = image_ok

if __name__ == "__main__":
    bo.main()
