#!/usr/bin/env python3
"""Benchmarked figures against the live records (data/records.json). Prints tables; writes data/compare.json.

    python3 -I scripts/compare.py

Live quantities, each from counter differences between two scrapes (see analyze.py):
  ms a round      = decode seconds / (rounds - requests): /metrics counts decode seconds from the first generated
                    token, which arrives after the request's first round, so one round a request is left out;
  tokens a round  = generated tokens / rounds;  drafted a round = drafted / rounds;  accepted share = accepted / drafted;
  prefill tok/s   = (prompt - cached) tokens / prompt-pass seconds;
  TTFT            = arrival to first generated token (queue, restore, encode, prompt pass, first round).
"""
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(ROOT))


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def desc(xs, fmt="{:.1f}"):
    if not xs:
        return "n=0"
    if len(xs) <= 5:  # too few for percentiles: the observations themselves
        return f"n={len(xs)} values " + ", ".join(fmt.format(x) for x in sorted(xs))
    return f"n={len(xs)} median {fmt.format(q(xs, .5))} (p10 {fmt.format(q(xs, .1))}, p90 {fmt.format(q(xs, .9))})"


def fit(pts):
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    a = my - b * mx
    res = [p[1] - (a + b * p[0]) for p in pts]
    return a, b, (sum(r * r for r in res) / max(1, n - 2)) ** 0.5


def main():
    R = json.load(open(os.path.join(ROOT, "data/records.json")))
    out = {}
    # ---- benchmark reference: the sampled probe's one-stream rows (k269 ctl1 and ctl2, client-timed probe)
    ref = []
    for arm in ("ctl1", "ctl2"):
        for row in json.load(open(os.path.join(REPO, f"runs/overnight-20261122/k269/{arm}.probe.json"))):
            if "ms_per_round" in row:
                ref.append((row["drafted_per_round"], row["ms_per_round"], row["label"], row["exact_tokens_per_round"],
                            eval(row["accepted"])))
    a, b, s = fit([(x[0], x[1]) for x in ref])
    out["bench_line"] = {"intercept_ms": a, "slope_ms_per_drafted": b, "resid_sd_ms": s, "rows": len(ref)}
    print(f"benchmark one-stream line (k269 ctl1+ctl2, {len(ref)} sampled probe rows): ms/round = {a:.2f} + {b:.3f} x drafted/round; residual sd {s:.2f} ms")
    print(f"  benchmark rows span drafted/round {min(x[0] for x in ref):.2f}..{max(x[0] for x in ref):.2f}, tokens/round {min(x[3] for x in ref):.2f}..{max(x[3] for x in ref):.2f}, accepted {100*min(x[4] for x in ref):.0f}..{100*max(x[4] for x in ref):.0f}%")

    for arm in ("ctl1", "ctl2"):
        for row in json.load(open(os.path.join(REPO, f"runs/overnight-20261122/k269/{arm}.probe.json"))):
            if row.get("width") in (2, 4) and "tokens_per_stream_round" in row:
                w = row["width"]
                ms = 1000 * w * row["tokens_per_stream_round"] / row["agg_tok_s_wall"]  # a round, from tokens and wall
                pred = a + b * w * row["drafted_per_stream_round"]
                print(f"  benchmark burst {arm} x{w}: {row['agg_tok_s_wall']} tok/s total, {row['tokens_per_stream_round']:.2f} tokens and "
                      f"{row['drafted_per_stream_round']:.2f} drafted a stream a round -> {ms:.1f} ms a round by wall; the one-stream line at "
                      f"{w * row['drafted_per_stream_round']:.1f} drafted rows gives {pred:.1f} ms ({100 * (ms - pred) / pred:+.1f}%)")

    def live(epochs):
        return [r for r in R if r["epoch"] in epochs and r["cls"] == "unattributed" and not short_life.get(r["life"], False)]

    # short lifetimes (the engine was restarted within 15 minutes): operator or harness work, not clients
    lt = json.load(open(os.path.join(ROOT, "data/lifetimes.json")))
    last = max(l["id"] for l in lt)
    short_life = {l["id"]: ((l["end"] - l["start"]) < 900e6 and l["id"] != last) for l in lt}
    out["short_life_unattributed_requests"] = sum(r["n"] for r in R if r["cls"] == "unattributed" and short_life.get(r["life"]) and r["epoch"] >= 1)
    print("unattributed requests in lifetimes shorter than 15 min (set aside as harness work, epochs E1+):", out["short_life_unattributed_requests"])

    def ms_round(r):
        """Decode seconds a round. A single request with a TPOT observation uses first-to-last-token time (no tail
        after the last token); otherwise decode seconds, which run from the first token to the request's end."""
        d = r["rounds"] - r["dec_n"]
        if d <= 0:
            return None
        if r["n"] == 1 and r["tpot_n"] == 1 and r["gen"] > 1:
            return 1000 * r["tpot_s"] * (r["gen"] - 1) / d
        return 1000 * r["dec_s"] / d

    def tail_s(r):
        return r["dec_s"] - r["tpot_s"] * (r["gen"] - 1)

    for name, eps in (("E7 current production (2bb052f)", (7,)), ("E2 fused image, Nov 21-22", (2,)), ("E1 Nov 18-21", (1,))):
        L = live(eps)
        print(f"\n=== {name}: {sum(r['n'] for r in L)} unattributed requests in {len(L)} records ({sum(1 for r in L if r['n']==1)} single-request)")
        res = {}
        # 1. one stream, decode cycle against the benchmark line
        one = [r for r in L if r["n"] == 1 and r["run_max"] <= 1 and r["wait_max"] == 0 and r["gen"] >= 200 and r["rounds"] > 50]
        for lab, lo, hi in (("prompt < 8K", 0, 8192), ("8K-64K", 8192, 65536), ("64K-200K", 65536, 200000), (">= 200K", 200000, 10**9)):
            x = [r for r in one if lo <= r["prompt"] < hi]
            if not x:
                print(f"  1 stream, {lab}: n=0")
                continue
            ms = [ms_round(r) for r in x]
            pred = [a + b * r["drafted"] / r["rounds"] for r in x]
            dev = [100 * (m - p) / p for m, p in zip(ms, pred)]
            tpr = [r["gen"] / r["rounds"] for r in x]
            acc = [100 * r["accepted"] / r["drafted"] for r in x if r["drafted"] > 0]
            tps = [1000 * (r["gen"] / r["rounds"]) / m for r, m in zip(x, ms)]
            tails = [tail_s(r) for r in x if r["tpot_n"] == 1]
            res_tail = q(tails, .5)
            print(f"  1 stream, {lab}: seconds from last token to request end {desc(tails, '{:.2f}')}")
            print(f"  1 stream, {lab}: ms/round {desc(ms)}; vs benchmark line at the same drafted/round {desc(dev, '{:+.1f}%')}; tokens/round {desc(tpr, '{:.2f}')}; accepted {desc(acc, '{:.0f}%')}; tok/s {desc(tps)}")
            res["one_" + lab] = {"n": len(x), "ms_med": q(ms, .5), "dev_med": q(dev, .5), "dev_p10": q(dev, .1), "dev_p90": q(dev, .9),
                                 "tail_s_med": res_tail, "ms_p10": q(ms, .1), "ms_p90": q(ms, .9),
                                 "tpr_med": q(tpr, .5), "tpr_p10": q(tpr, .1), "tpr_p90": q(tpr, .9), "acc_med": q(acc, .5),
                                 "acc_p10": q(acc, .1), "acc_p90": q(acc, .9), "tps_med": q(tps, .5), "tps_p10": q(tps, .1), "tps_p90": q(tps, .9)}
        # 2. steady concurrency c (all gauge samples during the request(s) read c)
        for c, deep in [(c, d) for c in (2, 3, 4, 5, 6, 7, 8) for d in (False, True)]:
            x = [r for r in L if r["run_min"] == c and r["run_max"] == c and r["rounds"] > 100 * r["n"] and r["n_samp"] >= 2
                 and (r["pool_max"] >= 16384) == deep]
            if not x:
                continue
            ms = [ms_round(r) for r in x]
            tpr = [r["gen"] / r["rounds"] for r in x]
            dpr = [r["drafted"] / r["rounds"] for r in x]
            acc = [100 * r["accepted"] / r["drafted"] for r in x if r["drafted"] > 0]
            agg = [c * 1000 * t / m for t, m in zip(tpr, ms)]
            nq = sum(1 for r in x if r["wait_max"] > 0)
            print(f"  {c} streams steady, deepest live stream {'>= 16K' if deep else '< 16K'} tokens: records {len(x)} ({sum(r['n'] for r in x)} requests, {nq} with a queue behind them): ms/round {desc(ms)}; tokens/stream/round {desc(tpr, '{:.2f}')}; drafted/stream/round {desc(dpr, '{:.2f}')}; accepted {desc(acc, '{:.0f}%')}; implied aggregate tok/s {desc(agg)}")
            res[f"c{c}_{'deep' if deep else 'shallow'}"] = {"records": len(x), "requests": sum(r["n"] for r in x), "ms_med": q(ms, .5), "ms_p10": q(ms, .1), "ms_p90": q(ms, .9),
                            "tpr_med": q(tpr, .5), "dpr_med": q(dpr, .5), "acc_med": q(acc, .5), "agg_med": q(agg, .5), "agg_p10": q(agg, .1), "agg_p90": q(agg, .9)}
        # 3. prefill rate on uncached tokens (single request, its own prompt pass)
        pf = [r for r in L if r["n"] == 1 and r["pre_n"] == 1 and (r["prompt"] - r["cached"]) >= 4096 and r["pre_s"] > 0]
        for lab, sel in (("alone (no other stream sampled)", lambda r: r["run_max"] <= 1), ("with other streams live", lambda r: r["run_max"] > 1)):
            x = [r for r in pf if sel(r)]
            rate = [(r["prompt"] - r["cached"]) / r["pre_s"] for r in x]
            unc = [r["prompt"] - r["cached"] for r in x]
            print(f"  prefill, uncached >= 4,096, {lab}: {desc(rate, '{:.0f}')} tok/s; uncached tokens {desc(unc, '{:.0f}')}")
            res["prefill_" + lab.split()[0]] = {"n": len(x), "med": q(rate, .5), "p10": q(rate, .1), "p90": q(rate, .9), "unc_med": q(unc, .5)}
        for lab, lo, hi in (("4K-8K", 4096, 8192), ("8K-32K", 8192, 32768), ("32K-128K", 32768, 131072), (">= 128K", 131072, 10**9)):
            x = [r for r in pf if r["run_max"] <= 1 and lo <= r["prompt"] - r["cached"] < hi]
            rate = [(r["prompt"] - r["cached"]) / r["pre_s"] for r in x]
            cs = [100 * r["cached"] / r["prompt"] for r in x]
            print(f"    alone, uncached {lab}: {desc(rate, '{:.0f}')} tok/s; cached share {desc(cs, '{:.0f}%')}; prompt {desc([r['prompt'] for r in x], '{:.0f}')}")
            res["prefill_size_" + lab] = {"n": len(x), "med": q(rate, .5), "p10": q(rate, .1), "p90": q(rate, .9), "cached_share_med": q(cs, .5)}
        cold = [r for r in pf if r["run_max"] <= 1 and r["cached"] == 0]
        print(f"    alone, cold (nothing cached), uncached >= 4,096: {desc([(r['prompt']) / r['pre_s'] for r in cold], '{:.0f}')} tok/s; prompt {desc([r['prompt'] for r in cold], '{:.0f}')}")
        res["prefill_cold"] = {"n": len(cold), "med": q([r["prompt"] / r["pre_s"] for r in cold], .5), "p10": q([r["prompt"] / r["pre_s"] for r in cold], .1), "p90": q([r["prompt"] / r["pre_s"] for r in cold], .9)}
        # 4. TTFT of single requests that did not queue: short uncached tails and the resume-grid remainder
        tt = [r for r in L if r["n"] == 1 and r["ttft_n"] == 1 and r["wait_max"] == 0 and r["run_max"] <= 1]
        for lab, lo, hi in (("uncached < 256", 0, 256), ("uncached 256-2,047", 256, 2048), ("uncached 2,048-8,191", 2048, 8192)):
            x = [r for r in tt if lo <= r["prompt"] - r["cached"] < hi]
            print(f"  TTFT alone, {lab}: {desc([r['ttft_s'] for r in x], '{:.2f}')} s; of which prompt pass {desc([r['pre_s'] for r in x if r['pre_n']], '{:.2f}')} s; cached share of prompt {desc([100*r['cached']/max(1,r['prompt']) for r in x], '{:.0f}%')}")
            res["ttft_" + lab] = {"n": len(x), "med": q([r["ttft_s"] for r in x], .5), "p10": q([r["ttft_s"] for r in x], .1), "p90": q([r["ttft_s"] for r in x], .9)}
        # 5. whole-epoch mix
        P = sum(r["prompt"] for r in L); C = sum(r["cached"] for r in L); G = sum(r["gen"] for r in L)
        Rn = sum(r["rounds"] for r in L); D = sum(r["drafted"] for r in L); A = sum(r["accepted"] for r in L)
        DS = sum(r["dec_s"] for r in L); N = sum(r["n"] for r in L)
        print(f"  mix: prompt tokens {int(P):,} ({100*C/max(1,P):.1f}% cached), generated {int(G):,}; tokens/round {G/max(1,Rn):.2f}; accepted/drafted {100*A/max(1,D):.1f}%; per-stream decode {G/max(1e-9,DS):.1f} tok/s; mean prompt {P/max(1,N):,.0f}, mean reply {G/max(1,N):,.0f} tokens")
        print(f"  health: disconnects {int(sum(r['disc'] for r in L))}, preemptions {int(sum(r['preempt'] for r in L))}, pair_failed {int(sum(r['pair_failed'] for r in L))}, service_refused {int(sum(r['service_refused'] for r in L))}; disk restores committed {int(sum(r['pc_disk_committed'] for r in L))} ({int(sum(r['pc_restored_tokens'] for r in L)):,} tokens), restore waits {int(sum(r['pc_restore_waits'] for r in L))} totalling {sum(r['pc_restore_wait_ns'] for r in L)/1e9:.1f} s; cache: memory hits {int(sum(r['pc_mem_hits'] for r in L))}, misses {int(sum(r['pc_mem_miss'] for r in L))} (no prefix {int(sum(r['pc_miss_no_prefix'] for r in L))}, stale {int(sum(r['pc_miss_stale'] for r in L))}, tail/window {int(sum(r['pc_miss_tail'] for r in L))})")
        res["mix"] = {"requests": N, "prompt": P, "cached": C, "gen": G, "tpr": G / max(1, Rn), "acc": 100 * A / max(1, D), "tok_s": G / max(1e-9, DS),
                      "disc": sum(r["disc"] for r in L), "preempt": sum(r["preempt"] for r in L), "pair_failed": sum(r["pair_failed"] for r in L),
                      "disk_committed": sum(r["pc_disk_committed"] for r in L), "restored_tokens": sum(r["pc_restored_tokens"] for r in L),
                      "restore_waits": sum(r["pc_restore_waits"] for r in L), "restore_wait_s": sum(r["pc_restore_wait_ns"] for r in L) / 1e9}
        out[name] = res
    json.dump(out, open(os.path.join(ROOT, "data/compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
