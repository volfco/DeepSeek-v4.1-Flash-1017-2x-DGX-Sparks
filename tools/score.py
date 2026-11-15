#!/usr/bin/env python3
"""Teacher-forced scoring harness: a recorded continuation's negative log-likelihood under a running server.

  python3 -I score.py record --in set.jsonl --out base.json --label base-arm [--url URL]
  python3 -I score.py record --in set.jsonl --out arm.json --label lossy-arm --like base.json
  python3 -I score.py compare base.json arm.json
  python3 -I score.py --self-test

record scores every item of a JSONL set ({"id": ..., "prompt_ids": [...], "continuation_ids": [...]}) through
POST /v1/completions with `tensorfold_score` = the continuation and the prompt as token ids, temperature 0,
stream false, and stores the reply's `tensorfold.score` block per item: the tokens scored, the negative
log-likelihoods' sum and mean, the mean's standard error, the perplexity and the share of rows whose argmax was the
recorded token. The server needs TF_DS_SCORE=1 (rank 0) and the DeepSeek-V4.1 lanes engine; a score request runs
alone in the lane, so a set is scored one item at a time. The whole set's figures are the mean of the items' means
with its standard error over items (an item is the unit, as compare's paired test is), the perplexity of that mean
and the token-weighted share of rows the model's argmax matched.

compare takes two records of the same set (meta.set_sha) and gives the paired per-item difference of the mean
negative log-likelihood with its standard error over items: the same items scored under two arms, so a lossless arm
must give a difference of exactly 0 (and the same per-item NLLs bit for bit), while a lossy arm's difference is its
cost. It exits 1 when an item errored or is missing from one side, 0 otherwise (a difference is the point).

--self-test checks the statistics on synthetic rows (no server). Stdlib only. API key: API_KEY, else the first line
of API_KEY_FILE (as tools/client.py). Proxies are never used.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

SCHEMA = 1
DEFAULT_MODEL = "DeepSeek-V4.1-Flash-1017"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def api_key() -> str:
    key = os.environ.get("API_KEY", "").strip()
    if key:
        return key
    path = os.environ.get("API_KEY_FILE", os.path.expanduser("~/.local/state/dsv41-tensorfold/api-key"))
    try:
        with open(path) as f:
            line = next((x.strip() for x in f if x.strip() and not x.startswith("#")), "")
    except OSError:
        return ""
    return line.split(":", 1)[1].strip() if ":" in line else line


def headers() -> dict:
    h = {"Content-Type": "application/json"}
    k = api_key()
    if k:
        h["Authorization"] = "Bearer " + k
    return h


def sha(s) -> str:
    return hashlib.sha256(s if isinstance(s, bytes) else s.encode("utf-8")).hexdigest()


def post(url, path, body, timeout):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), headers())
    t0 = time.monotonic()
    try:
        with OPENER.open(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"}
    except Exception as e:  # noqa: BLE001 - report, do not crash
        return {"error": f"{type(e).__name__}: {e}"}
    data["_seconds"] = time.monotonic() - t0
    return data


def log(*m):
    print(*m, file=sys.stderr, flush=True)


def mean_se(values):
    """The mean and the standard error of its estimate over the values (the sample's own spread)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = math.fsum(values) / n
    if n < 2:
        return mean, 0.0
    var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def aggregate(items):
    """A set's figures from its items (each a reply's score block plus the item's token count): the mean of the
    items' mean negative log-likelihoods with its standard error over items, the perplexity of that mean, and the
    token-weighted share of rows whose argmax was the recorded token."""
    means = [it["nll_mean"] for it in items]
    mean, se = mean_se(means)
    tokens = sum(it["tokens"] for it in items)
    matched = math.fsum(it["argmax_match"] * it["tokens"] for it in items)
    return {
        "items": len(items),
        "tokens": tokens,
        "nll_mean": mean,
        "nll_se": se,
        "ppl": math.exp(mean) if mean < 709.0 else float("inf"),
        "argmax_match": matched / tokens if tokens else float("nan"),
        "nll_sum": math.fsum(it["nll_sum"] for it in items),
    }


def paired(a_items, b_items):
    """The paired per-item difference A - B of the mean negative log-likelihoods: its mean and standard error."""
    diffs = [a_items[k]["nll_mean"] - b_items[k]["nll_mean"] for k in a_items]
    mean, se = mean_se(diffs)
    return {"items": len(diffs), "diff_mean": mean, "diff_se": se, "diffs": diffs}


def read_set(path):
    """The JSONL set: its items by id and the digest the records are matched on."""
    items, raw = {}, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw.append(line)
            item = json.loads(line)
            for key in ("id", "prompt_ids", "continuation_ids"):
                if key not in item:
                    raise SystemExit(f"{path}: an item has no {key}: {line[:200]}")
            if not item["prompt_ids"] or not item["continuation_ids"]:
                raise SystemExit(f"{path}: item {item['id']!r} needs a prompt and a continuation")
            if item["id"] in items:
                raise SystemExit(f"{path}: duplicate item id {item['id']!r}")
            items[item["id"]] = item
    if not items:
        raise SystemExit(f"{path}: no items")
    return items, sha("\n".join(raw))


def item_from_reply(r, continuation_len):
    """One item's figures from a reply's tensorfold.score block, or {"error": ...} when the reply cannot be scored:
    a missing block, a missing field, or a score that stopped short of the recorded continuation."""
    if "error" in r:
        return {"error": r["error"]}
    try:
        tf = r.get("tensorfold") or {}
        score = tf.get("score")
        u = r.get("usage") or {}
    except (AttributeError, TypeError) as e:
        return {"error": f"unexpected reply shape: {e}: {str(r)[:200]}"}
    if not isinstance(score, dict):
        return {"error": "the reply has no tensorfold.score block (is TF_DS_SCORE=1 set on rank 0, on an engine that scores?)"}
    out = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
           "seconds": round(r.get("_seconds") or 0.0, 3)}
    for key in ("tokens", "nll_sum", "nll_mean", "nll_se", "ppl", "argmax_match"):
        if key not in score:
            return {**out, "error": f"the score block has no {key}: {score}"}
        out[key] = score[key]
    if out["tokens"] != continuation_len:
        return {**out, "error": f"scored {out['tokens']} of {continuation_len} tokens: the reply stopped early"}
    return out


def score_item(a, item):
    """One item's figures, asking the server for its score."""
    body = {"model": a.model, "prompt": item["prompt_ids"], "tensorfold_score": item["continuation_ids"],
            "temperature": 0, "stream": False, "max_tokens": len(item["continuation_ids"])}
    return item_from_reply(post(a.url, "/v1/completions", body, a.timeout), len(item["continuation_ids"]))


def record(a) -> int:
    like = json.load(open(a.like)) if a.like else None
    items, set_sha = read_set(a.inp)
    if like:
        if like["meta"].get("set_sha") != set_sha:
            log(f"--like {a.like} scored another set (meta.set_sha {like['meta'].get('set_sha')}, this {set_sha}): the arms must score the same items")
            return 2
        if like["meta"].get("model") != a.model:
            log(f"warning: --like {a.like} was recorded on model {like['meta'].get('model')!r}; this is {a.model!r}")
    run = a
    log(f"record: label={a.label!r} url={a.url} model={a.model} set={set_sha[:12]} items={len(items)}")
    scored, errors = {}, []
    for name, item in items.items():
        r = score_item(run, item)
        scored[name] = r
        if "error" in r:
            errors.append(name)
            log(f"  {name}: {r['error']}")
        else:
            log(f"  {name}: tokens={r['tokens']} nll_mean={r['nll_mean']:.6f} nll_se={r['nll_se']:.6f} "
                f"ppl={r['ppl']:.4f} argmax_match={r['argmax_match']:.3f} {r['seconds']}s")
    ok = {k: v for k, v in scored.items() if "error" not in v}
    overall = aggregate(list(ok.values())) if ok else {}
    doc = {"meta": {"schema": SCHEMA, "label": a.label, "url": a.url, "model": a.model, "set": a.inp,
                    "set_sha": set_sha, "items": len(items), "scored": len(ok), "errors": errors,
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "like": a.like},
           "overall": overall, "items": scored}
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    if ok:
        log(f"overall: items={overall['items']} tokens={overall['tokens']} nll_mean={overall['nll_mean']:.6f} "
            f"± {overall['nll_se']:.6f} ppl={overall['ppl']:.4f} argmax_match={overall['argmax_match']:.3f}")
    log(f"wrote {a.out}: {len(ok)} scored, {len(errors)} errors {errors}")
    return 1 if errors else 0


def compare(pa, pb) -> int:
    A, B = json.load(open(pa)), json.load(open(pb))
    ma, mb = A["meta"], B["meta"]
    bad = 0
    print(f"A: {pa} label={ma.get('label')!r} {ma.get('time')}")
    print(f"B: {pb} label={mb.get('label')!r} {mb.get('time')}")
    for k in ("set_sha", "model"):
        if ma.get(k) != mb.get(k):
            print(f"MISMATCH meta.{k}: {ma.get(k)!r} vs {mb.get(k)!r} (the arms scored different items or models)")
            bad += 1
    if bad:
        return 1
    a_items, b_items = A["items"], B["items"]
    for name in sorted(set(a_items) | set(b_items)):
        a, b = a_items.get(name), b_items.get(name)
        if a is None or b is None:
            print(f"MISMATCH {name}: present only in {'B' if a is None else 'A'}")
            bad += 1
        elif "error" in a or "error" in b:
            print(f"MISMATCH {name}: request error A={a.get('error')} B={b.get('error')}")
            bad += 1
    common = {k: a_items[k] for k in a_items if k in b_items and "error" not in a_items[k] and "error" not in b_items[k]}
    if not common:
        print("RESULT: no item scored on both sides")
        return 1
    p = paired(common, {k: b_items[k] for k in common})
    oa, ob = A.get("overall") or {}, B.get("overall") or {}
    for name in common:
        print(f"item {name}: A nll_mean={a_items[name]['nll_mean']:.6f} B nll_mean={b_items[name]['nll_mean']:.6f} "
              f"diff={a_items[name]['nll_mean'] - b_items[name]['nll_mean']:+.6f}")
    print(f"overall A: nll_mean={oa.get('nll_mean')} ± {oa.get('nll_se')} ppl={oa.get('ppl')} argmax_match={oa.get('argmax_match')}")
    print(f"overall B: nll_mean={ob.get('nll_mean')} ± {ob.get('nll_se')} ppl={ob.get('ppl')} argmax_match={ob.get('argmax_match')}")
    print(f"RESULT: paired difference A-B = {p['diff_mean']:+.6f} ± {p['diff_se']:.6f} over {p['items']} items "
          f"({'bit-identical NLLs' if all(d == 0 for d in p['diffs']) else 'the arms differ'})")
    return 1 if bad else 0


def self_test() -> int:
    """The statistics on synthetic rows: hand-computed means, standard errors, perplexities and paired differences."""
    def close(got, want, what):
        if not (abs(got - want) <= 1e-9 * max(1.0, abs(want))):
            raise SystemExit(f"self-test FAILED: {what}: {got!r} != {want!r}")

    items = [
        {"tokens": 2, "nll_sum": 2.0, "nll_mean": 1.0, "nll_se": 0.0, "ppl": math.exp(1.0), "argmax_match": 0.5},
        {"tokens": 2, "nll_sum": 4.0, "nll_mean": 2.0, "nll_se": 0.0, "ppl": math.exp(2.0), "argmax_match": 1.0},
        {"tokens": 4, "nll_sum": 12.0, "nll_mean": 3.0, "nll_se": 0.0, "ppl": math.exp(3.0), "argmax_match": 0.5},
    ]
    out = aggregate(items)
    # means 1, 2, 3: mean 2, sample variance 1, standard error sqrt(1/3)
    close(out["nll_mean"], 2.0, "aggregate mean")
    close(out["nll_se"], math.sqrt(1.0 / 3.0), "aggregate standard error")
    close(out["ppl"], math.exp(2.0), "aggregate perplexity")
    # hits: 1 + 2 + 2 of 8 tokens
    close(out["argmax_match"], 5.0 / 8.0, "aggregate argmax share")
    close(out["nll_sum"], 18.0, "aggregate nll sum")
    if out["tokens"] != 8 or out["items"] != 3:
        raise SystemExit("self-test FAILED: aggregate counts")
    one = aggregate(items[:1])
    close(one["nll_se"], 0.0, "one item's standard error")
    close(one["ppl"], math.exp(1.0), "one item's perplexity")

    # the paired difference A - B: a constant +0.5 has no spread, a spread of -1, 0, 1 the mean's error above
    b_items = {k: {"nll_mean": v["nll_mean"] - 0.5} for k, v in zip("abc", items)}
    p = paired(dict(zip("abc", items)), b_items)
    close(p["diff_mean"], 0.5, "paired mean")
    close(p["diff_se"], 0.0, "paired standard error")
    spread = {k: {"nll_mean": v["nll_mean"] + d} for k, v, d in zip("abc", items, (-1.0, 0.0, 1.0))}
    p = paired(dict(zip("abc", items)), spread)
    close(p["diff_mean"], 0.0, "paired mean of a spread")
    close(p["diff_se"], math.sqrt(1.0 / 3.0), "paired standard error of a spread")

    # a reply's block: taken when it scored the whole continuation, an error when it stopped short or is missing
    block = {"tokens": 4, "nll_sum": 2.0, "nll_mean": 0.5, "nll_se": 0.1, "ppl": 1.6487212707001282, "argmax_match": 0.75}
    good = item_from_reply({"tensorfold": {"score": block}, "usage": {"prompt_tokens": 2, "completion_tokens": 4}, "_seconds": 1.0}, 4)
    if good.get("error") or good["nll_mean"] != 0.5 or good["prompt_tokens"] != 2:
        raise SystemExit(f"self-test FAILED: a whole score's block: {good}")
    short = item_from_reply({"tensorfold": {"score": {**block, "tokens": 2}}}, 4)
    if "error" not in short or "stopped early" not in short["error"]:
        raise SystemExit(f"self-test FAILED: a score that stopped short: {short}")
    missing = item_from_reply({"tensorfold": {}}, 4)
    if "error" not in missing or "tensorfold.score" not in missing["error"]:
        raise SystemExit(f"self-test FAILED: a reply without the block: {missing}")

    # a set reads as items by id, and the digest follows its bytes
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "a", "prompt_ids": [1, 2], "continuation_ids": [3, 4]}\n')
        path = f.name
    try:
        got, digest = read_set(path)
        if list(got) != ["a"] or got["a"]["continuation_ids"] != [3, 4] or len(digest) != 64:
            raise SystemExit("self-test FAILED: read_set")
    finally:
        os.unlink(path)
    print("self-test: ok (means, standard errors, perplexities, argmax shares, paired differences, reply checks, set reading)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="check the statistics on synthetic rows and exit")
    sub = ap.add_subparsers(dest="mode")
    r = sub.add_parser("record", help="score a JSONL set")
    r.add_argument("--in", dest="inp", required=True, help="the JSONL set: {id, prompt_ids, continuation_ids}")
    r.add_argument("--out", required=True)
    r.add_argument("--label", default="")
    r.add_argument("--like", help="an earlier record of the same set (its digest is checked)")
    r.add_argument("--url", default=os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8888")),
                   help="server root or .../v1 (default API_URL or http://127.0.0.1:$PORT)")
    r.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    r.add_argument("--timeout", type=float, default=900)
    c = sub.add_parser("compare", help="the paired per-item difference between two records")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.mode == "compare":
        return compare(a.a, a.b)
    if a.mode == "record":
        a.url = a.url.rstrip("/")
        if a.url.endswith("/v1"):
            a.url = a.url[:-3]
        return record(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
