#!/usr/bin/env python3
"""Speculative sampling's distribution: does TF_DS_SPEC_SAMPLE=1 give a sampled stream the same token distribution as
the undrafted lane (TF_DS_SPEC_SAMPLE off, DRAFTS=0), and is that distribution the one the sampler serves?

  python3 -I samp_test.py record --out off.json --label off --drafts off     # server: TF_DS_SPEC_SAMPLE unset, DRAFTS=0
  # restart the server with TF_DS_SPEC_SAMPLE=1 (and its drafter on)
  python3 -I samp_test.py record --out on.json  --label on  --drafts on
  python3 -I samp_test.py compare off.json on.json                            # exit 1 on any chi-square under 1%

One prompt (PROMPT, thinking off, non-streaming), --seeds seeds, two requests a seed: temperature 1.0, top_k 20, top_p
0.95, min_p 0, the first with max_tokens 1 and the second with max_tokens 3. A record stores, per request shape, the
token histogram (token id -> count) of the reply's first token, and for the 3-token replies the same histogram of the
second and third tokens conditioned on the first token's id. compare runs a chi-square between the two records' counts
for every histogram both hold: the first token's, and each first-token id's second and third token (only ids with at
least MIN_SAMPLES replies in both, bins expected under 5 dropped, df = bins - 1, the p from the regularised upper
incomplete gamma). A sampled stream's drafts change its tokens for a seed but not its distribution, so the histograms
must agree; a real mismatch means the accept rule, the residual or the proposal's normaliser is wrong.

Stdlib only. API key: API_KEY, else the first line of API_KEY_FILE (as tools/exact.py). Proxies are never used.
"""
import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request

SCHEMA = 1
DEFAULT_MODEL = "DeepSeek-V4.1-Flash-1017"
DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
PROMPT = "Write a short paragraph about the sea."
MIN_SAMPLES = 30      # a conditioned histogram needs this many replies in both records
MIN_EXPECTED = 5.0    # chi-square bins under this expected count are dropped
ALPHA = 0.01          # a p under this fails


def api_key() -> str:
    k = os.environ.get("API_KEY")
    if k:
        return k.strip()
    path = os.environ.get("API_KEY_FILE")
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.readline().strip()
    return ""


def headers() -> dict:
    return {"Content-Type": "application/json", "Authorization": "Bearer " + api_key()}


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers(), method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit("HTTP %d: %s" % (e.code, e.read().decode()[:400]))


def log(*m):
    print(*m, flush=True)


def ask(url, model, seed, max_tokens, timeout):
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": max_tokens,
            "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "seed": seed, "stream": False,
            "chat_template_kwargs": {"thinking": False}}
    r = post(url, body, timeout)
    ids = (r.get("tensorfold") or {}).get("token_ids")
    if not ids:
        raise SystemExit("reply has no tensorfold.token_ids: " + json.dumps(r)[:400])
    return [int(t) for t in ids]


def bump(hist, tok):
    hist[str(tok)] = hist.get(str(tok), 0) + 1


def record(a) -> int:
    meta = {"schema": SCHEMA, "label": a.label, "model": a.model, "prompt": PROMPT, "seeds": a.seeds,
            "drafts": a.drafts, "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0},
            "url": a.url}
    one = {"first": {}}
    three = {"first": {}, "second": {}, "third": {}}
    for i in range(a.seeds):
        seed = a.seed0 + i
        bump(one["first"], ask(a.url, a.model, seed, 1, a.timeout)[0])
        ids = ask(a.url, a.model, seed, 3, a.timeout)
        bump(three["first"], ids[0])
        if len(ids) > 1:
            bump(three["second"].setdefault(str(ids[0]), {}), ids[1])
        if len(ids) > 2:
            bump(three["third"].setdefault(str(ids[0]), {}), ids[2])
        if (i + 1) % 500 == 0:
            log("  %d/%d seeds" % (i + 1, a.seeds))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "max_tokens_1": one, "max_tokens_3": three}, f, indent=1)
    log("recorded %s (%s, drafts %s): %d first-token ids" % (a.out, a.label, a.drafts, len(one["first"])))
    return 0


def gammq(a, x):
    """The regularised upper incomplete gamma Q(a, x) (Numerical Recipes' series and continued fraction)."""
    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        term, total, ap = 1.0 / a, 1.0 / a, a
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c, d = x + 1.0 - a, 1e300, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2(ca, cb):
    """The chi-square p of two count histograms (bins both hold, expected at least MIN_EXPECTED)."""
    ids = sorted(set(ca) | set(cb))
    rows = []
    for t in ids:
        x, y = ca.get(t, 0), cb.get(t, 0)
        if (x + y) / 2.0 >= MIN_EXPECTED:
            rows.append((x, y))
    if len(rows) < 2:
        return 1.0, 0.0, len(rows)
    na = sum(r[0] for r in rows)
    nb = sum(r[1] for r in rows)
    x2 = 0.0
    for x, y in rows:
        ea = na * (x + y) / (na + nb)
        eb = nb * (x + y) / (na + nb)
        x2 += (x - ea) ** 2 / ea + (y - eb) ** 2 / eb
    df = len(rows) - 1
    return gammq(df / 2.0, x2 / 2.0), x2, df


def compare(pa, pb) -> int:
    a = json.load(open(pa, encoding="utf-8"))
    b = json.load(open(pb, encoding="utf-8"))
    log("comparing %s (%s, drafts %s) with %s (%s, drafts %s)" % (pa, a["meta"]["label"], a["meta"]["drafts"],
                                                                 pb, b["meta"]["label"], b["meta"]["drafts"]))
    if a["meta"]["drafts"] == b["meta"]["drafts"]:
        log("  (both records say drafts %s: one of them should be DRAFTS=0)" % a["meta"]["drafts"])
    bad = 0
    for shape, key in (("max_tokens_1", "first"), ("max_tokens_3", "first")):
        p, x2, df = chi2(a[shape][key], b[shape][key])
        ok = p >= ALPHA
        bad += 0 if ok else 1
        log("  %s %s: chi2=%.1f df=%d p=%.4f %s" % (shape, key, x2, df, p, "ok" if ok else "MISMATCH"))
    for key in ("second", "third"):
        for t1 in sorted(set(a["max_tokens_3"][key]) | set(b["max_tokens_3"][key])):
            ca = a["max_tokens_3"][key].get(t1, {})
            cb = b["max_tokens_3"][key].get(t1, {})
            if sum(ca.values()) < MIN_SAMPLES or sum(cb.values()) < MIN_SAMPLES:
                continue
            p, x2, df = chi2(ca, cb)
            ok = p >= ALPHA
            bad += 0 if ok else 1
            log("  max_tokens_3 %s given %s: chi2=%.1f df=%d p=%.4f %s" % (key, t1, x2, df, p, "ok" if ok else "MISMATCH"))
    log("FAIL: %d histogram(s) under p=%.2f" % (bad, ALPHA) if bad else "PASS: every histogram agrees")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--label", required=True, help="what this run is (e.g. off, on)")
    r.add_argument("--drafts", choices=["on", "off"], required=True, help="the server's DRAFTS setting")
    r.add_argument("--seeds", type=int, default=5000)
    r.add_argument("--seed0", type=int, default=0)
    r.add_argument("--url", default=DEFAULT_URL)
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--timeout", type=float, default=120.0)
    c = sub.add_parser("compare")
    c.add_argument("off")
    c.add_argument("on")
    a = ap.parse_args()
    return record(a) if a.cmd == "record" else compare(a.off, a.on)


if __name__ == "__main__":
    sys.exit(main())
