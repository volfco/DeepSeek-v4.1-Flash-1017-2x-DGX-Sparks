#!/usr/bin/env python3
"""Post-restart paired dec8 replay & forensic interval auditor (todo.md item 4).

Modes:
  1. Offline verification (workstation-safe, zero network/SSH/GPU calls):
       python3 dec8_replay_audit.py --offline-verify [SUITE_DIR]

  2. Live recording on the head Spark during a future idle/restart window:
       python3 dec8_replay_audit.py record \
         --url http://127.0.0.1:8000 \
         --corpus-dir runs/suite-fused-next-20261121/corpus \
         --expected-suite runs/suite-fused-next-20261121 \
         --admission-mode staggered --stagger-ms 2.5 \
         --reps 5 --label base --out replay_base.json

  3. Offline comparison of two recorded arm JSONs:
       python3 dec8_replay_audit.py compare replay_base.json replay_next.json
"""
import argparse
import hashlib
import json
import os
import random
import re
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_SUITE = "/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/suite-fused-next-20261121"
DEFAULT_MODEL = "DeepSeek-V4.1-Flash-1017"
DONE_COUNTER_KEY = "tensorfold:request_latency_seconds_count"

EXPECTED_TOK_S = {"s01": 138.97, "s02": 138.14, "s03": 137.77, "s04": 138.73, "s05": 138.59}
EXPECTED_MS_RND = {"s01": 86.62, "s02": 86.66, "s03": 86.54, "s04": 86.92, "s05": 86.48}


# ---------------------------------------------------------------------------
# Prompt generation (matches tools/bench_div.py + scripts/bench_overnight.py)
# ---------------------------------------------------------------------------
def load_prose_corpus(corpus_dir):
    prose_path = os.path.join(corpus_dir, "prose.json")
    raw = open(prose_path, "rb").read()
    md5 = hashlib.md5(raw).hexdigest()
    return json.loads(raw.decode("utf-8")), md5


def build_dec8_prompts(prose_lines, rep_idx, streams=8):
    prompts = []
    target_chars = 250 * 4.0  # Corpus.FIXED_CPT["div"] == 4.0, tokens == 250
    for s in range(streams):
        rng = random.Random(f"dec-d8-{rep_idx}-{s}")
        parts, n = [], 0
        while n < target_chars:
            para = [rng.choice(prose_lines) for _ in range(rng.randint(3, 8))]
            p = " ".join(para)
            parts.append(p)
            n += len(p) + 2
        prompts.append(
            "Continue the following text. Write at least 800 words, keep going in the same style "
            "and do not summarise or stop early.\n\n" + "\n\n".join(parts)
        )
    return prompts


# ---------------------------------------------------------------------------
# Canonical wave & per-round interval decomposition (shared live + offline)
# ---------------------------------------------------------------------------
def analyze_round_events(streams_records):
    """When per-SSE-event monotonic timestamps `events_mono` are captured (1 event = 1 round),
    reconstruct physical round intervals and active-stream occupancy directly from round events."""
    if not all(r.get("events_mono") and len(r["events_mono"]) >= 2 for r in streams_records):
        return None
    tf_max = max(r["t_first"] for r in streams_records)
    # Event 0 is the prefill token; events 1..K are decode rounds 1..K for that stream.
    max_decode_events = max(len(r["events_mono"]) - 1 for r in streams_records)
    per_round_dt_ms = []
    per_round_active = []
    prev_t = tf_max
    for rnd_k in range(1, max_decode_events + 1):
        # A physical lockstep round finishes when all participating streams emit their rnd_k-th decode event
        ts_k = [r["events_mono"][rnd_k] for r in streams_records if len(r["events_mono"]) > rnd_k]
        if not ts_k:
            break
        t_end = max(ts_k)
        per_round_dt_ms.append((t_end - prev_t) * 1000.0)
        per_round_active.append(len(ts_k))
        prev_t = t_end

    by_width = {}
    for act, dt in zip(per_round_active, per_round_dt_ms):
        by_width.setdefault(act, []).append(dt)
    summary_by_width = {
        str(k): {"rounds": len(v), "sum_ms": sum(v), "mean_ms": st.mean(v), "median_ms": st.median(v)}
        for k, v in sorted(by_width.items(), reverse=True)
    }
    return {
        "reconstructed_rounds": len(per_round_dt_ms),
        "total_decode_ms": sum(per_round_dt_ms),
        "by_active_streams": summary_by_width,
    }


def decompose_wave(rep_idx, results, done_lines, t_unix=0.0):
    """Compute both legacy bench_div metrics and unbiased forensic intervals for one 8-stream wave."""
    n = len(results)
    tf = [r["t_first"] for r in results]
    tl = [r["t_last"] for r in results]
    t_req = [r.get("t_req", r["t_first"] - r["ttft_s"]) for r in results]

    tf_min, tf_max = min(tf), max(tf)
    tl_max = max(tl)
    t_req_min = min(t_req)

    window_s = tl_max - tf_min
    phase1_s = tf_max - tf_min
    phase2_s = tl_max - tf_max
    full_batch_s = tl_max - t_req_min
    total_tokens = sum(int(r["completion_tokens"]) for r in results)

    adm_order = sorted(range(n), key=lambda i: tf[i])
    ord0_s = adm_order[0]
    fin_last_s = max(range(n), key=lambda i: tl[i])

    streams = []
    for i, r in enumerate(results):
        dl = done_lines.get(r["id"], {})
        srv = r.get("server") or {}
        srv_pref = float(srv.get("prefill_seconds") or dl.get("prefill_s") or 0.0)
        srv_ttft = float(srv.get("time_to_first_token") or dl.get("ttft_s") or r["ttft_s"])
        srv_sec = float(srv.get("seconds") or (srv_ttft + float(dl.get("decode_s", 0.0))))
        srv_dec = srv_sec - srv_ttft
        log_dec = float(dl["decode_s"]) if "decode_s" in dl else srv_dec
        rounds = int(dl.get("rounds") or (len(r.get("events_mono", [])) - 1))
        acc_str = dl.get("accepted", "0/0")
        acc_n, drf_n = (int(x) for x in acc_str.split("/")) if "/" in acc_str else (0, 0)
        sha = dl.get("sha") or srv.get("token_sha") or ""
        streams.append({
            "stream_idx": i,
            "adm_ord": adm_order.index(i),
            "id": r["id"],
            "prompt_tokens": int(r["prompt_tokens"]),
            "completion_tokens": int(r["completion_tokens"]),
            "sha": sha,
            "rounds": rounds,
            "accepted": acc_n,
            "drafted": drf_n,
            "req_rel_ms": (t_req[i] - t_req_min) * 1000.0,
            "queue_ms": float(dl.get("queue_s", 0.0)) * 1000.0,
            "srv_prefill_ms": srv_pref * 1000.0,
            "srv_ttft_ms": srv_ttft * 1000.0,
            "cli_ttft_ms": r["ttft_s"] * 1000.0,
            "t_first_rel_ms": (r["t_first"] - tf_min) * 1000.0,
            "t_last_rel_ms": (r["t_last"] - tf_min) * 1000.0,
            "cli_decode_ms": (r["t_last"] - r["t_first"]) * 1000.0,
            "srv_decode_ms": srv_dec * 1000.0,
            "log_decode_ms": log_dec * 1000.0,
            "prefill_stall_in_decode_ms": (tf_max - r["t_first"]) * 1000.0,
            "true_active_decode_ms": (r["t_last"] - tf_max) * 1000.0,
        })

    sum_pref_ms = sum(s["srv_prefill_ms"] for s in streams)
    ord0_pref_ms = streams[ord0_s]["srv_prefill_ms"]
    phase1_ms = phase1_s * 1000.0
    phase2_ms = phase2_s * 1000.0
    window_ms = window_s * 1000.0
    handoff_sse_ms = phase1_ms - (sum_pref_ms - ord0_pref_ms)

    fin_sorted = sorted(streams, key=lambda s: s["t_last_rel_ms"])
    stages = []
    prev_ms = phase1_ms
    prev_rnd = 0
    for k_idx, s_row in enumerate(fin_sorted):
        dt_ms = s_row["t_last_rel_ms"] - prev_ms
        dr = s_row["rounds"] - prev_rnd
        stages.append({
            "active_streams": n - k_idx,
            "dt_ms": dt_ms,
            "d_rounds": dr,
            "ms_per_round": dt_ms / dr if dr > 0 else None,
            "ending_stream": s_row["stream_idx"],
            "cum_rounds": s_row["rounds"],
        })
        prev_ms = s_row["t_last_rel_ms"]
        prev_rnd = s_row["rounds"]

    return {
        "rep": rep_idx,
        "t_unix": t_unix,
        "legacy_aggregate_tok_s": (total_tokens - n) / window_s,
        "pure_decode_tok_s": (total_tokens - n) / phase2_s,
        "full_batch_tok_s": total_tokens / full_batch_s,
        "window_ms": window_ms,
        "full_batch_ms": full_batch_s * 1000.0,
        "phase1_prefill_window_ms": phase1_ms,
        "sum_8_srv_prefill_ms": sum_pref_ms,
        "ord0_stream": ord0_s,
        "ord0_excluded_srv_prefill_ms": ord0_pref_ms,
        "phase1_handoff_sse_ms": handoff_sse_ms,
        "phase2_decode_window_ms": phase2_ms,
        "full_8stream_ms": stages[0]["dt_ms"],
        "full_8stream_rounds": stages[0]["d_rounds"],
        "tail_lt8_ms": sum(s["dt_ms"] for s in stages[1:]),
        "tail_lt8_rounds": sum(s["d_rounds"] for s in stages[1:]),
        "physical_rounds": prev_rnd,
        "sum_stream_rounds": sum(s["rounds"] for s in streams),
        "finishing_stream": fin_last_s,
        "finishing_stream_adm_ord": streams[fin_last_s]["adm_ord"],
        "adm_order": adm_order,
        "stages": stages,
        "streams": streams,
        "round_events_summary": analyze_round_events(results),
    }


def summarize_arm(label, waves):
    legacy_toks = [w["legacy_aggregate_tok_s"] for w in waves]
    pure_toks = [w["pure_decode_tok_s"] for w in waves]
    full_toks = [w["full_batch_tok_s"] for w in waves]
    log_dec_s = sum(s["log_decode_ms"] for w in waves for s in w["streams"]) / 1000.0
    srv_dec_s = sum(s["srv_decode_ms"] for w in waves for s in w["streams"]) / 1000.0
    true_dec_s = sum(s["true_active_decode_ms"] for w in waves for s in w["streams"]) / 1000.0
    rnd_tot = sum(w["sum_stream_rounds"] for w in waves)
    return {
        "label": label,
        "n_waves": len(waves),
        "legacy_tok_s_median": st.median(legacy_toks),
        "pure_decode_tok_s_median": st.median(pure_toks),
        "full_batch_tok_s_median": st.median(full_toks),
        "log_ms_per_stream_round": 1000.0 * log_dec_s / rnd_tot,
        "exact_srv_ms_per_stream_round": 1000.0 * srv_dec_s / rnd_tot,
        "true_active_ms_per_stream_round": 1000.0 * true_dec_s / rnd_tot,
        "total_stream_rounds": rnd_tot,
        "waves": waves,
    }


def compare_wave_pair(wb, wc):
    d_win = wc["window_ms"] - wb["window_ms"]
    d_p1 = wc["phase1_prefill_window_ms"] - wb["phase1_prefill_window_ms"]
    d_sum_pref = wc["sum_8_srv_prefill_ms"] - wb["sum_8_srv_prefill_ms"]
    d_ord0_excl = -(wc["ord0_excluded_srv_prefill_ms"] - wb["ord0_excluded_srv_prefill_ms"])
    d_p1_ovh = wc["phase1_handoff_sse_ms"] - wb["phase1_handoff_sse_ms"]
    d_p2 = wc["phase2_decode_window_ms"] - wb["phase2_decode_window_ms"]

    assert abs((d_p1 + d_p2) - d_win) < 1e-6, "window decomposition mismatch"
    assert abs((d_sum_pref + d_ord0_excl + d_p1_ovh) - d_p1) < 1e-6, "phase1 decomposition mismatch"

    rnd_mix_ms = 0.0
    step_speed_ms = 0.0
    for sb, sc in zip(wb["stages"], wc["stages"]):
        ddt = sc["dt_ms"] - sb["dt_ms"]
        ddr = sc["d_rounds"] - sb["d_rounds"]
        dr_b, dr_c = sb["d_rounds"], sc["d_rounds"]
        if dr_b > 0 and dr_c > 0:
            rb, rc = sb["dt_ms"] / dr_b, sc["dt_ms"] / dr_c
            c_eff = ddr * 0.5 * (rb + rc)
            s_eff = 0.5 * (dr_b + dr_c) * (rc - rb)
        elif dr_b == 0 and dr_c == 0:
            c_eff = s_eff = 0.0
        else:
            r_ref = (sc["dt_ms"] / dr_c) if dr_c > 0 else (sb["dt_ms"] / dr_b)
            c_eff = ddr * r_ref
            s_eff = ddt - c_eff
        rnd_mix_ms += c_eff
        step_speed_ms += s_eff

    assert abs((rnd_mix_ms + step_speed_ms) - d_p2) < 1e-6, "phase2 Shapley decomposition mismatch"
    pct = lambda a, b: 100.0 * (a / b - 1.0)
    return {
        "rep": wb["rep"],
        "sha_match_8_of_8": [s["sha"] for s in wb["streams"]] == [s["sha"] for s in wc["streams"]],
        "ord_identical": wb["adm_order"] == wc["adm_order"],
        "d_legacy_tok_s_pct": pct(wc["legacy_aggregate_tok_s"], wb["legacy_aggregate_tok_s"]),
        "d_pure_decode_tok_s_pct": pct(wc["pure_decode_tok_s"], wb["pure_decode_tok_s"]),
        "d_full_batch_tok_s_pct": pct(wc["full_batch_tok_s"], wb["full_batch_tok_s"]),
        "d_window_ms": d_win,
        "d_phase1_ms": d_p1,
        "d_sum_8_prefill_ms": d_sum_pref,
        "d_ord0_exclusion_ms": d_ord0_excl,
        "d_phase1_handoff_sse_ms": d_p1_ovh,
        "d_phase2_decode_ms": d_p2,
        "d_physical_rounds": wc["physical_rounds"] - wb["physical_rounds"],
        "d_round_mix_ms": rnd_mix_ms,
        "d_matched_step_speed_ms": step_speed_ms,
    }


def compare_arms(arm_base, arm_cand):
    pct = lambda a, b: 100.0 * (a / b - 1.0)
    n_common = min(len(arm_base["waves"]), len(arm_cand["waves"]))
    wave_CMP = [compare_wave_pair(arm_base["waves"][i], arm_cand["waves"][i]) for i in range(n_common)]
    return {
        "base": arm_base["label"],
        "cand": arm_cand["label"],
        "d_legacy_tok_s_median_pct": pct(arm_cand["legacy_tok_s_median"], arm_base["legacy_tok_s_median"]),
        "d_pure_decode_tok_s_median_pct": pct(arm_cand["pure_decode_tok_s_median"], arm_base["pure_decode_tok_s_median"]),
        "d_full_batch_tok_s_median_pct": pct(arm_cand["full_batch_tok_s_median"], arm_base["full_batch_tok_s_median"]),
        "d_log_ms_per_round_pct": pct(arm_cand["log_ms_per_stream_round"], arm_base["log_ms_per_stream_round"]),
        "d_exact_srv_ms_per_round_pct": pct(arm_cand["exact_srv_ms_per_stream_round"], arm_base["exact_srv_ms_per_stream_round"]),
        "d_true_active_ms_per_round_pct": pct(arm_cand["true_active_ms_per_stream_round"], arm_base["true_active_ms_per_stream_round"]),
        "all_shas_equal": all(w["sha_match_8_of_8"] for w in wave_CMP),
        "waves": wave_CMP,
    }


# ---------------------------------------------------------------------------
# Live HTTP client (only used when subcommand `record` is explicitly invoked)
# ---------------------------------------------------------------------------
class LiveClient:
    def __init__(self, url, model):
        self.url = url.rstrip("/")
        self.model = model
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def metrics(self):
        with self.opener.open(self.url + "/metrics", timeout=10) as r:
            raw = r.read().decode("utf-8", "replace")
        out = {}
        for ln in raw.splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            k, _, v = ln.rpartition(" ")
            try:
                out[k] = float(v)
            except ValueError:
                pass
        return out

    def wait_idle(self, timeout_s=120.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            m = self.metrics()
            if m.get("tensorfold:requests_running", 1) == 0 and m.get("tensorfold:requests_waiting", 0) == 0:
                return m
            time.sleep(0.4)
        raise RuntimeError("Server not idle within timeout; aborting replay to protect active traffic.")

    def chat_stream_detailed(self, prompt, max_tokens=512, gate=None, delay_s=0.0, timeout=1800):
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 1234,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            self.url + "/v1/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        if gate is not None:
            gate.wait()
        if delay_s > 0:
            time.sleep(delay_s)
        t_req = time.monotonic()
        events_mono = []
        usage = server = rid = finish = None
        with self.opener.open(req, timeout=timeout) as resp:
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
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if piece:
                        events_mono.append(time.monotonic())
                    finish = ch.get("finish_reason") or finish
        t_end = time.monotonic()
        if not events_mono or usage is None:
            raise RuntimeError(f"Incomplete stream for {rid}: events={len(events_mono)}")
        t_first, t_last = events_mono[0], events_mono[-1]
        return {
            "id": rid,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "t_req": t_req,
            "t_first": t_first,
            "t_last": t_last,
            "ttft_s": t_first - t_req,
            "total_s": t_end - t_req,
            "finish": finish,
            "server": server,
            "events_mono": events_mono,
        }


def fetch_docker_done_lines(container, rids):
    p = subprocess.run(["docker", "logs", "--since", "15m", container], capture_output=True, text=True, timeout=30)
    done = {}
    for ln in (p.stdout + p.stderr).splitlines():
        m = re.search(r"done (chatcmpl-\w+) (.*)", ln)
        if m and m.group(1) in rids:
            done[m.group(1)] = dict(kv.split("=", 1) for kv in m.group(2).split() if "=" in kv)
    return done


def run_live_record(args):
    prose, prose_md5 = load_prose_corpus(args.corpus_dir)
    expected_meta = None
    if args.expected_suite:
        s01 = json.load(open(os.path.join(args.expected_suite, "blocks", "s01.json")))
        expected_meta = []
        for rep in s01["cells"]["dec8"]:
            att = [a for a in rep if a.get("clean")][-1]
            expected_meta.append([
                (int(r["prompt_tokens"]), s01["done_lines"][r["id"]]["sha"])
                for r in att["results"]
            ])

    cli = LiveClient(args.url, args.model)
    waves = []
    for rep_idx in range(args.reps):
        prompt_set_idx = rep_idx % 3
        prompts = build_dec8_prompts(prose, prompt_set_idx, 8)
        m0 = cli.wait_idle()
        t_unix = time.time()
        gate = threading.Barrier(8)
        res = [None] * 8
        errs = [None] * 8

        def worker(i):
            delay = (i * args.stagger_ms / 1000.0) if args.admission_mode == "staggered" else 0.0
            try:
                res[i] = cli.chat_stream_detailed(prompts[i], args.max_tokens, gate, delay)
            except Exception as exc:  # noqa: BLE001
                errs[i] = str(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if any(errs):
            raise RuntimeError(f"Wave {rep_idx} errors: {errs}")

        time.sleep(0.8)
        m1 = cli.wait_idle(timeout_s=15.0)
        done_delta = m1.get(DONE_COUNTER_KEY, 0) - m0.get(DONE_COUNTER_KEY, 0)
        if done_delta != 8:
            raise RuntimeError(f"Contaminated wave {rep_idx}: done_delta={done_delta} != 8")

        rids = {r["id"] for r in res}
        done_lines = fetch_docker_done_lines(args.container, rids) if args.container else {}
        wave = decompose_wave(rep_idx, res, done_lines, t_unix=t_unix)
        if expected_meta and prompt_set_idx < len(expected_meta):
            got = [(s["prompt_tokens"], s["sha"]) for s in wave["streams"]]
            want = expected_meta[prompt_set_idx]
            if got != want:
                raise RuntimeError(f"Wave {rep_idx} prompt/SHA mismatch vs s01: got={got} want={want}")
        waves.append(wave)
        print(
            f"Recorded wave {rep_idx}: legacy_tok/s={wave['legacy_aggregate_tok_s']:.2f} "
            f"pure_dec_tok/s={wave['pure_decode_tok_s']:.2f} phys_rounds={wave['physical_rounds']} "
            f"adm_order={wave['adm_order']}",
            flush=True,
        )

    arm = summarize_arm(args.label, waves)
    arm["corpus_prose_md5"] = prose_md5
    arm["admission_mode"] = args.admission_mode
    arm["stagger_ms"] = args.stagger_ms if args.admission_mode == "staggered" else 0.0
    with open(args.out, "w") as fh:
        json.dump(arm, fh, indent=2)
    print(f"Saved {args.out}")


# ---------------------------------------------------------------------------
# Offline verification on runs/suite-fused-next-20261121/blocks/s01..s05
# ---------------------------------------------------------------------------
def selftest_round_events():
    """Offline self-test of per-SSE-event round reconstruction (live mode's round-level path)."""
    base_t = 1000.0
    # 8 streams prefill sequentially 0.5s each; then 4 rounds at 8 streams (0.084s), then 2 tail rounds on s=0 (0.030s)
    fake_res = []
    fake_dl = {}
    t_bar = base_t + 4.0
    for i in range(8):
        tf = base_t + (i + 1) * 0.5
        n_rnd = 6 if i == 0 else 4
        evs = [tf] + [t_bar + k * 0.084 if k <= 4 else t_bar + 4 * 0.084 + (k - 4) * 0.030 for k in range(1, n_rnd + 1)]
        rid = f"chatcmpl-test{i}"
        fake_res.append({
            "id": rid, "prompt_tokens": 300, "completion_tokens": 12,
            "t_req": base_t, "t_first": evs[0], "t_last": evs[-1], "ttft_s": evs[0] - base_t,
            "server": {"prefill_seconds": 0.5, "time_to_first_token": (i + 1) * 0.5, "seconds": evs[-1] - base_t, "token_sha": f"sha{i}"},
            "events_mono": evs,
        })
        fake_dl[rid] = {"rounds": str(n_rnd), "accepted": "5/6", "decode_s": f"{evs[-1]-evs[0]:.1f}", "queue_s": f"{i*0.5:.3f}", "sha": f"sha{i}"}
    w = decompose_wave(0, fake_res, fake_dl)
    re_sum = w["round_events_summary"]
    assert re_sum is not None and re_sum["reconstructed_rounds"] == 6
    assert re_sum["by_active_streams"]["8"]["rounds"] == 4
    assert re_sum["by_active_streams"]["1"]["rounds"] == 2
    assert abs(re_sum["by_active_streams"]["8"]["mean_ms"] - 84.0) < 1e-6
    assert abs(re_sum["by_active_streams"]["1"]["mean_ms"] - 30.0) < 1e-6


def run_offline_verify(suite_dir):
    selftest_round_events()
    arms = {}
    for b in ("s01", "s02", "s03", "s04", "s05"):
        doc = json.load(open(os.path.join(suite_dir, "blocks", f"{b}.json")))
        waves = []
        for ri, rep in enumerate(doc["cells"]["dec8"]):
            att = [a for a in rep if a.get("clean")][-1]
            waves.append(decompose_wave(ri, att["results"], doc["done_lines"], t_unix=att["t"]))
        arm = summarize_arm(f"{b}_{doc['arm']}", waves)
        assert round(arm["legacy_tok_s_median"], 2) == EXPECTED_TOK_S[b], f"{b} tok/s mismatch"
        assert round(arm["log_ms_per_stream_round"], 2) == EXPECTED_MS_RND[b], f"{b} ms/round mismatch"
        arms[b] = arm

    cmp_a = compare_arms(arms["s01"], arms["s02"])
    cmp_b = compare_arms(arms["s04"], arms["s03"])
    cmp_base = compare_arms(arms["s01"], arms["s04"])
    cmp_pdl1 = compare_arms(arms["s01"], arms["s05"])
    cmp_pdl4 = compare_arms(arms["s04"], arms["s05"])

    assert cmp_a["all_shas_equal"] and cmp_b["all_shas_equal"] and cmp_pdl1["all_shas_equal"]
    assert round(cmp_a["d_legacy_tok_s_median_pct"], 2) == -0.59
    assert round(cmp_b["d_legacy_tok_s_median_pct"], 2) == -0.69
    assert round(cmp_base["d_legacy_tok_s_median_pct"], 2) == -0.17
    assert round(cmp_a["d_log_ms_per_round_pct"], 2) == 0.06
    assert round(cmp_b["d_log_ms_per_round_pct"], 2) == -0.43

    print("OFFLINE VERIFY PASSED on s01..s05 + synthetic per-round SSE event trace:")
    for tag, c in (("Pair A (s01->s02)", cmp_a), ("Pair B (s04->s03)", cmp_b), ("Base (s01->s04)", cmp_base), ("PDL (s01->s05)", cmp_pdl1), ("PDL (s04->s05)", cmp_pdl4)):
        w0 = c["waves"][0]
        print(
            f"  {tag:18s}: legacy_med={c['d_legacy_tok_s_median_pct']:+6.2f}% | "
            f"pure_dec_med={c['d_pure_decode_tok_s_median_pct']:+6.2f}% | "
            f"full_batch_med={c['d_full_batch_tok_s_median_pct']:+6.2f}% | "
            f"log_ms/r={c['d_log_ms_per_round_pct']:+6.2f}% true_active_ms/r={c['d_true_active_ms_per_round_pct']:+6.2f}% | "
            f"W0 d_win={w0['d_window_ms']:+7.2f}ms (P1={w0['d_phase1_ms']:+6.2f}ms, P2={w0['d_phase2_decode_ms']:+7.2f}ms, d_phys_r={w0['d_physical_rounds']:+d})"
        )


def print_comparison(cmp_res):
    print(
        f"Comparison {cmp_res['base']} -> {cmp_res['cand']} (all_shas_equal={cmp_res['all_shas_equal']}):\n"
        f"  legacy_tok_s_median     : {cmp_res['d_legacy_tok_s_median_pct']:+.2f}%\n"
        f"  pure_decode_tok_s_median: {cmp_res['d_pure_decode_tok_s_median_pct']:+.2f}%\n"
        f"  full_batch_tok_s_median : {cmp_res['d_full_batch_tok_s_median_pct']:+.2f}%\n"
        f"  log_ms_per_round        : {cmp_res['d_log_ms_per_round_pct']:+.2f}%\n"
        f"  true_active_ms_per_round: {cmp_res['d_true_active_ms_per_round_pct']:+.2f}%"
    )
    for w in cmp_res["waves"]:
        print(
            f"  Wave {w['rep']}: d_win={w['d_window_ms']:+7.2f}ms | "
            f"P1={w['d_phase1_ms']:+6.2f}ms (8pref={w['d_sum_8_prefill_ms']:+6.2f}ms, ord0_excl={w['d_ord0_exclusion_ms']:+6.2f}ms) | "
            f"P2={w['d_phase2_decode_ms']:+7.2f}ms (d_phys_r={w['d_physical_rounds']:+d}, rnd_mix={w['d_round_mix_ms']:+7.2f}ms, step_speed={w['d_matched_step_speed_ms']:+6.2f}ms) | "
            f"ord_same={w['ord_identical']}"
        )


def main():
    if "--offline-verify" in sys.argv:
        idx = sys.argv.index("--offline-verify")
        sdir = sys.argv[idx + 1] if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-") else DEFAULT_SUITE
        run_offline_verify(sdir)
        return

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Record dec8 waves on an idle live server after a restart window")
    rec.add_argument("--url", default="http://127.0.0.1:8000")
    rec.add_argument("--model", default=DEFAULT_MODEL)
    rec.add_argument("--corpus-dir", default="runs/suite-fused-next-20261121/corpus")
    rec.add_argument("--expected-suite", default="")
    rec.add_argument("--container", default="dsv41-1017-exl3-tf")
    rec.add_argument("--reps", type=int, default=5)
    rec.add_argument("--max-tokens", type=int, default=512)
    rec.add_argument("--admission-mode", choices=("staggered", "barrier"), default="staggered")
    rec.add_argument("--stagger-ms", type=float, default=2.5)
    rec.add_argument("--label", required=True)
    rec.add_argument("--out", required=True)

    cmp_p = sub.add_parser("compare", help="Compare two recorded dec8 audit JSONs offline")
    cmp_p.add_argument("base_json")
    cmp_p.add_argument("cand_json")

    args = ap.parse_args()
    if args.cmd == "record":
        run_live_record(args)
    elif args.cmd == "compare":
        b = json.load(open(args.base_json))
        c = json.load(open(args.cand_json))
        print_comparison(compare_arms(b, c))


if __name__ == "__main__":
    main()
