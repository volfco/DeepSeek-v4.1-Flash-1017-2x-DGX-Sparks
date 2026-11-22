#!/usr/bin/env python3
"""Workstation-only forensic attribution of the dec8 8-stream throughput dip
(-0.59% in Pair A s01->s02, -0.69% in Pair B s04->s03, -0.19% in s05 vs base mean)
in runs/suite-fused-next-20261121.

Reads ONLY local artifacts under SUITE_DIR/blocks/s01..s05.{json,rank0.log,rank1.log,head.samples,worker.samples,probe.json}.
Zero network, zero SSH, zero GPU calls.
"""
import argparse
import datetime
import json
import os
import re
import statistics as st

DEFAULT_SUITE = "/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/suite-fused-next-20261121"
BLOCKS = ("s01", "s02", "s03", "s04", "s05")

EXPECTED_TOK_S = {"s01": 138.97, "s02": 138.14, "s03": 137.77, "s04": 138.73, "s05": 138.59}
EXPECTED_MS_RND = {"s01": 86.62, "s02": 86.66, "s03": 86.54, "s04": 86.92, "s05": 86.48}


def parse_iso_utc(ts_str):
    s = ts_str[:-1] if ts_str.endswith("Z") else ts_str
    dt_part, frac = s.split(".")
    dt = datetime.datetime.strptime(dt_part, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp() + float("0." + frac)


def parse_samples(path):
    rows = []
    with open(path, errors="replace") as fh:
        for ln in fh:
            parts = ln.split()
            if len(parts) != 3:
                continue
            try:
                ts = int(parts[0])
                clk, tmp, pwr = parts[1].split(",")
                rows.append({
                    "ts": ts,
                    "clock_mhz": int(float(clk)),
                    "temp_c": int(float(tmp)),
                    "power_w": float(pwr),
                    "mem_avail_kb": int(parts[2]),
                })
            except ValueError:
                continue
    return rows


def parse_rank0_dec8(path, dec8_rids):
    with open(path, errors="replace") as fh:
        lines = fh.readlines()
    done_by_rid = {}
    match_indices = []
    graph_lines = []
    for idx, ln in enumerate(lines):
        if "CUDA graphs:" in ln:
            m_ts = re.match(r"^(\S+)\s+(.*)$", ln.strip())
            if m_ts:
                graph_lines.append({"line": idx, "ts": parse_iso_utc(m_ts.group(1)), "msg": m_ts.group(2)})
        m = re.match(r"^(\S+)\s+\[tensorfold\] done (chatcmpl-\w+) (.*)$", ln.strip())
        if m and m.group(2) in dec8_rids:
            rid = m.group(2)
            fields = dict(kv.split("=", 1) for kv in m.group(3).split() if "=" in kv)
            done_by_rid[rid] = {"line": idx, "ts": parse_iso_utc(m.group(1)), "fields": fields}
            match_indices.append(idx)

    # Extract POST and prompt-cache log lines immediately preceding each wave
    i0 = max(0, min(match_indices) - 30)
    i1 = min(len(lines), max(match_indices) + 5)
    posts, pcache, metrics_gets, anomalies = [], [], [], []
    for idx in range(i0, i1):
        ln = lines[idx].strip()
        m_ts = re.match(r"^(\S+)\s+(.*)$", ln)
        if not m_ts:
            continue
        ts, msg = parse_iso_utc(m_ts.group(1)), m_ts.group(2)
        if '"POST /v1/chat/completions' in msg:
            posts.append({"line": idx, "ts": ts})
        elif "info: prompt cache:" in msg:
            mp = re.search(r"prompt cache: (\d+) tokens, resumed at (\d+), kept (\d+)", msg)
            pcache.append({
                "line": idx,
                "ts": ts,
                "tokens": int(mp.group(1)) if mp else None,
                "resumed": int(mp.group(2)) if mp else None,
                "kept": int(mp.group(3)) if mp else None,
            })
        elif '"GET /metrics' in msg:
            metrics_gets.append({"line": idx, "ts": ts})
        elif "done chatcmpl-" not in msg:
            anomalies.append({"line": idx, "ts": ts, "msg": msg})

    dec8_t0 = posts[0]["ts"] if posts else min(v["ts"] for v in done_by_rid.values()) - 35.0
    dec8_t1 = max(v["ts"] for v in done_by_rid.values())
    graphs_during_dec8 = [g for g in graph_lines if dec8_t0 <= g["ts"] <= dec8_t1]
    return {
        "done_by_rid": done_by_rid,
        "posts": posts,
        "prompt_cache_lines": pcache,
        "metrics_gets": metrics_gets,
        "anomalies": anomalies,
        "graphs_during_dec8": graphs_during_dec8,
        "first_graph_log_ts": graph_lines[0]["ts"] if graph_lines else None,
        "dec8_log_span_utc": [dec8_t0, dec8_t1],
    }


def clean_reps(doc, cell="dec8"):
    out = []
    for rep in doc["cells"][cell]:
        c = [a for a in rep if a.get("clean")]
        if c:
            out.append(c[-1])
    return out


def reproduce_suite_metrics(doc):
    reps = clean_reps(doc, "dec8")
    agg_list = []
    sec = rnd = tok = 0.0
    for a in reps:
        res = a["results"]
        window = max(r["t_last"] for r in res) - min(r["t_first"] for r in res)
        total = sum(r["completion_tokens"] for r in res)
        calc_agg = (total - len(res)) / window
        assert abs(calc_agg - a["aggregate_tok_s"]) < 1e-9, f"{doc['id']} agg mismatch"
        agg_list.append(a["aggregate_tok_s"])
        for r in res:
            dl = doc["done_lines"][r["id"]]
            sec += float(dl["decode_s"])
            rnd += float(dl["rounds"])
            tok += float(dl["tokens"])
    return {
        "tok_s_median": st.median(agg_list),
        "tok_s_reps": agg_list,
        "ms_per_round": 1000.0 * sec / rnd,
        "tok_per_round": tok / rnd,
        "sum_log_decode_s": sec,
        "sum_stream_rounds": int(rnd),
        "sum_tokens": int(tok),
    }


def analyze_wave(doc, rep_idx, att, log_info, head_samples, worker_samples):
    res = att["results"]
    n = len(res)
    assert n == 8
    tf = [r["t_first"] for r in res]
    tl = [r["t_last"] for r in res]
    t_req = [r["t_first"] - r["ttft_s"] for r in res]

    tf_min, tf_max = min(tf), max(tf)
    tl_max = max(tl)
    t_req_min = min(t_req)
    window_s = tl_max - tf_min
    phase1_s = tf_max - tf_min       # Sequential prefills of ord=1..7 after ord=0's first token
    phase2_s = tl_max - tf_max       # Post-prefill-barrier multi-stream decode span
    assert abs((phase1_s + phase2_s) - window_s) < 1e-12

    adm_order = sorted(range(n), key=lambda i: tf[i])
    ord0_s = adm_order[0]
    ord7_s = adm_order[-1]
    fin_last_s = max(range(n), key=lambda i: tl[i])

    streams = []
    for s_idx in range(n):
        r = res[s_idx]
        dl = doc["done_lines"][r["id"]]
        srv = r["server"]
        acc_n, drf_n = (int(x) for x in dl["accepted"].split("/"))
        srv_pref = float(srv["prefill_seconds"])
        srv_ttft = float(srv["time_to_first_token"])
        srv_sec = float(srv["seconds"])
        srv_dec = srv_sec - srv_ttft
        cli_dec = r["t_last"] - r["t_first"]
        log_dec = float(dl["decode_s"])
        post_barrier_dec = r["t_last"] - tf_max
        prefill_stall_in_dec = tf_max - r["t_first"]
        streams.append({
            "stream_idx": s_idx,
            "adm_ord": adm_order.index(s_idx),
            "id": r["id"],
            "prompt_tokens": int(r["prompt_tokens"]),
            "completion_tokens": int(r["completion_tokens"]),
            "sha": dl["sha"],
            "rounds": int(dl["rounds"]),
            "accepted": acc_n,
            "drafted": drf_n,
            "req_rel_ms": (t_req[s_idx] - t_req_min) * 1000.0,
            "queue_ms": float(dl["queue_s"]) * 1000.0,
            "srv_prefill_ms": srv_pref * 1000.0,
            "srv_ttft_ms": srv_ttft * 1000.0,
            "cli_ttft_ms": r["ttft_s"] * 1000.0,
            "cli_minus_srv_ttft_ms": (r["ttft_s"] - srv_ttft) * 1000.0,
            "t_first_rel_ms": (r["t_first"] - tf_min) * 1000.0,
            "t_last_rel_ms": (r["t_last"] - tf_min) * 1000.0,
            "cli_decode_ms": cli_dec * 1000.0,
            "srv_decode_ms": srv_dec * 1000.0,
            "log_decode_ms": log_dec * 1000.0,
            "log_minus_srv_decode_ms": (log_dec - srv_dec) * 1000.0,
            "cli_minus_srv_decode_ms": (cli_dec - srv_dec) * 1000.0,
            "prefill_stall_in_decode_ms": prefill_stall_in_dec * 1000.0,
            "true_active_decode_ms": post_barrier_dec * 1000.0,
            "engram_host_ms": float(dl["engram_host_ms"]),
            "render_ms": float(dl["render_s"]) * 1000.0,
            "encode_ms": float(dl["encode_s"]) * 1000.0,
            "restore_ms": float(dl["restore_s"]) * 1000.0,
        })

    sum_pref_ms = sum(x["srv_prefill_ms"] for x in streams)
    ord0_pref_ms = streams[ord0_s]["srv_prefill_ms"]
    pref_1_to_7_ms = sum_pref_ms - ord0_pref_ms
    phase1_ms = phase1_s * 1000.0
    phase2_ms = phase2_s * 1000.0
    window_ms = window_s * 1000.0
    handoff_and_sse_ms = phase1_ms - pref_1_to_7_ms
    assert abs((sum_pref_ms - ord0_pref_ms + handoff_and_sse_ms) - phase1_ms) < 1e-9

    # Decompose Phase 2 across active-stream stages k = 8, 7, ..., 1
    fin_sorted = sorted(streams, key=lambda x: x["t_last_rel_ms"])
    stages = []
    prev_ms = phase1_ms
    prev_rnd = 0
    for k_idx, st_row in enumerate(fin_sorted):
        n_active = n - k_idx
        dt_ms = st_row["t_last_rel_ms"] - prev_ms
        dr = st_row["rounds"] - prev_rnd
        stages.append({
            "active_streams": n_active,
            "dt_ms": dt_ms,
            "d_rounds": dr,
            "ms_per_round": dt_ms / dr if dr > 0 else None,
            "ending_stream": st_row["stream_idx"],
            "cum_rounds": st_row["rounds"],
        })
        prev_ms = st_row["t_last_rel_ms"]
        prev_rnd = st_row["rounds"]

    assert abs(sum(s["dt_ms"] for s in stages) - phase2_ms) < 1e-9
    full8_ms = stages[0]["dt_ms"]
    full8_rnd = stages[0]["d_rounds"]
    tail_ms = sum(s["dt_ms"] for s in stages[1:])
    tail_rnd = sum(s["d_rounds"] for s in stages[1:])

    # Slice GPU samples covering this wave [t_start - 1, t_start + 31]
    t0_unix = att["t"]
    t1_unix = t0_unix + (tl_max - t_req_min) + 1.0
    hs = [s for s in head_samples if t0_unix - 1.0 <= s["ts"] <= t1_unix + 1.0]
    ws = [s for s in worker_samples if t0_unix - 1.0 <= s["ts"] <= t1_unix + 1.0]

    # Wave POST spread from rank0.log (8 POSTs per wave)
    wave_posts = log_info["posts"][rep_idx * 8:(rep_idx + 1) * 8]
    post_span_ms = (wave_posts[-1]["ts"] - wave_posts[0]["ts"]) * 1000.0 if len(wave_posts) == 8 else None

    total_tokens = sum(x["completion_tokens"] for x in streams)
    full_batch_s = tl_max - t_req_min
    return {
        "rep": rep_idx,
        "t_unix": t0_unix,
        "aggregate_tok_s": att["aggregate_tok_s"],
        "full_batch_tok_s": total_tokens / full_batch_s,
        "pure_decode_tok_s": (total_tokens - n) / phase2_s,
        "window_ms": window_ms,
        "full_batch_ms": full_batch_s * 1000.0,
        "phase1_prefill_window_ms": phase1_ms,
        "sum_8_srv_prefill_ms": sum_pref_ms,
        "ord0_stream": ord0_s,
        "ord0_prompt_tokens": streams[ord0_s]["prompt_tokens"],
        "ord0_excluded_srv_prefill_ms": ord0_pref_ms,
        "prefill_1_to_7_srv_ms": pref_1_to_7_ms,
        "phase1_handoff_and_sse_ms": handoff_and_sse_ms,
        "phase2_decode_window_ms": phase2_ms,
        "full_8stream_ms": full8_ms,
        "full_8stream_rounds": full8_rnd,
        "full_8stream_ms_per_round": full8_ms / full8_rnd,
        "tail_lt8_ms": tail_ms,
        "tail_lt8_rounds": tail_rnd,
        "tail_lt8_ms_per_round": tail_ms / tail_rnd if tail_rnd else 0.0,
        "physical_rounds": prev_rnd,
        "physical_ms_per_round": phase2_ms / prev_rnd,
        "sum_stream_rounds": sum(x["rounds"] for x in streams),
        "sum_accepted": sum(x["accepted"] for x in streams),
        "sum_drafted": sum(x["drafted"] for x in streams),
        "finishing_stream": fin_last_s,
        "finishing_stream_adm_ord": streams[fin_last_s]["adm_ord"],
        "finishing_stream_accepted": streams[fin_last_s]["accepted"],
        "finishing_stream_drafted": streams[fin_last_s]["drafted"],
        "adm_order": adm_order,
        "client_req_spread_ms": max(x["req_rel_ms"] for x in streams),
        "server_post_span_ms": post_span_ms,
        "cli_minus_srv_ttft_mean_ms": st.mean(x["cli_minus_srv_ttft_ms"] for x in streams),
        "cli_minus_srv_decode_fin_ms": streams[fin_last_s]["cli_minus_srv_decode_ms"],
        "head_clocks_mhz": [s["clock_mhz"] for s in hs],
        "worker_clocks_mhz": [s["clock_mhz"] for s in ws],
        "head_temps_c": [s["temp_c"] for s in hs],
        "worker_temps_c": [s["temp_c"] for s in ws],
        "stages": stages,
        "streams": streams,
    }


def compare_waves(w_base, w_cand):
    """Exact millisecond attribution of (w_cand - w_base) batch window delta."""
    d_win = w_cand["window_ms"] - w_base["window_ms"]
    d_p1 = w_cand["phase1_prefill_window_ms"] - w_base["phase1_prefill_window_ms"]
    d_sum_pref = w_cand["sum_8_srv_prefill_ms"] - w_base["sum_8_srv_prefill_ms"]
    d_ord0_excl = -(w_cand["ord0_excluded_srv_prefill_ms"] - w_base["ord0_excluded_srv_prefill_ms"])
    d_p1_ovh = w_cand["phase1_handoff_and_sse_ms"] - w_base["phase1_handoff_and_sse_ms"]
    assert abs((d_sum_pref + d_ord0_excl + d_p1_ovh) - d_p1) < 1e-9

    d_p2 = w_cand["phase2_decode_window_ms"] - w_base["phase2_decode_window_ms"]
    assert abs((d_p1 + d_p2) - d_win) < 1e-9

    # Stage-wise symmetric Shapley decomposition of Phase 2 into:
    #   (a) round_count_and_width_mix_ms: change in rounds at each active-stream width k=8..1
    #   (b) matched_width_step_speed_ms: change in ms/round at matched active-stream width k=8..1
    round_mix_ms = 0.0
    step_speed_ms = 0.0
    stage_deltas = []
    for sb, sc in zip(w_base["stages"], w_cand["stages"]):
        assert sb["active_streams"] == sc["active_streams"]
        k = sb["active_streams"]
        ddt = sc["dt_ms"] - sb["dt_ms"]
        ddr = sc["d_rounds"] - sb["d_rounds"]
        dr_b, dr_c = sb["d_rounds"], sc["d_rounds"]
        if dr_b > 0 and dr_c > 0:
            rb, rc = sb["dt_ms"] / dr_b, sc["dt_ms"] / dr_c
            r_mid = 0.5 * (rb + rc)
            dr_mid = 0.5 * (dr_b + dr_c)
            c_eff = ddr * r_mid
            s_eff = dr_mid * (rc - rb)
        elif dr_b == 0 and dr_c == 0:
            c_eff = s_eff = 0.0
        else:
            r_ref = (sc["dt_ms"] / dr_c) if dr_c > 0 else (sb["dt_ms"] / dr_b)
            c_eff = ddr * r_ref
            s_eff = ddt - c_eff
        round_mix_ms += c_eff
        step_speed_ms += s_eff
        stage_deltas.append({
            "active_streams": k,
            "d_dt_ms": ddt,
            "d_rounds": ddr,
            "round_effect_ms": c_eff,
            "speed_effect_ms": s_eff,
        })
    assert abs((round_mix_ms + step_speed_ms) - d_p2) < 1e-9

    return {
        "d_tok_s": w_cand["aggregate_tok_s"] - w_base["aggregate_tok_s"],
        "d_tok_s_pct": 100.0 * (w_cand["aggregate_tok_s"] / w_base["aggregate_tok_s"] - 1.0),
        "d_pure_decode_tok_s_pct": 100.0 * (w_cand["pure_decode_tok_s"] / w_base["pure_decode_tok_s"] - 1.0),
        "d_full_batch_tok_s_pct": 100.0 * (w_cand["full_batch_tok_s"] / w_base["full_batch_tok_s"] - 1.0),
        "d_window_ms": d_win,
        "d_phase1_ms": d_p1,
        "d_sum_8_prefill_ms": d_sum_pref,
        "d_ord0_exclusion_penalty_ms": d_ord0_excl,
        "d_phase1_handoff_sse_ms": d_p1_ovh,
        "d_phase2_decode_ms": d_p2,
        "d_physical_rounds": w_cand["physical_rounds"] - w_base["physical_rounds"],
        "d_sum_stream_rounds": w_cand["sum_stream_rounds"] - w_base["sum_stream_rounds"],
        "d_full_8stream_ms": w_cand["full_8stream_ms"] - w_base["full_8stream_ms"],
        "d_full_8stream_rounds": w_cand["full_8stream_rounds"] - w_base["full_8stream_rounds"],
        "d_tail_lt8_ms": w_cand["tail_lt8_ms"] - w_base["tail_lt8_ms"],
        "d_tail_lt8_rounds": w_cand["tail_lt8_rounds"] - w_base["tail_lt8_rounds"],
        "d_round_count_and_width_mix_ms": round_mix_ms,
        "d_matched_width_step_speed_ms": step_speed_ms,
        "stage_deltas": stage_deltas,
    }


def check_solo_determinism(suite_dir, docs):
    """Verify solo greedy decode (dec1 + probe solo rows) has 100% identical rounds & accepted across all 5 blocks."""
    dec1_ok = True
    for ri in range(3):
        tuples = set()
        for b in BLOCKS:
            r = clean_reps(docs[b], "dec1")[ri]["results"][0]
            dl = docs[b]["done_lines"][r["id"]]
            tuples.add((dl["sha"], int(dl["rounds"]), dl["accepted"]))
        if len(tuples) != 1:
            dec1_ok = False
    probes = {b: json.load(open(f"{suite_dir}/blocks/{b}.probe.json")) for b in BLOCKS}
    probe_solo_count = 0
    probe_ok = True
    for idx, r0 in enumerate(probes["s01"]):
        if "rounds" not in r0:
            continue
        probe_solo_count += 1
        tuples = {(probes[b][idx]["sha"], probes[b][idx]["rounds"], probes[b][idx]["accepted"]) for b in BLOCKS}
        if len(tuples) != 1:
            probe_ok = False
    return {"dec1_3_of_3_exact_rounds": dec1_ok, "probe_9_of_9_exact_rounds": probe_ok, "solo_cases_checked": 3 + probe_solo_count}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite-dir", default=DEFAULT_SUITE)
    ap.add_argument("--json-out", default="dec8_attribution.json")
    args = ap.parse_args()

    docs = {}
    repro = {}
    waves_by_block = {}
    decode_s_audit = {}
    gpu_summary = {}
    log_audit = {}

    for b in BLOCKS:
        bdir = os.path.join(args.suite_dir, "blocks")
        doc = json.load(open(os.path.join(bdir, f"{b}.json")))
        docs[b] = doc
        rep_m = reproduce_suite_metrics(doc)
        repro[b] = rep_m

        # Assert exact reproduction of RESULTS.md figures
        assert round(rep_m["tok_s_median"], 2) == EXPECTED_TOK_S[b], (
            f"{b} tok/s {rep_m['tok_s_median']:.2f} != {EXPECTED_TOK_S[b]}"
        )
        assert round(rep_m["ms_per_round"], 2) == EXPECTED_MS_RND[b], (
            f"{b} ms/round {rep_m['ms_per_round']:.2f} != {EXPECTED_MS_RND[b]}"
        )

        reps = clean_reps(doc, "dec8")
        dec8_rids = {r["id"] for a in reps for r in a["results"]}
        linfo = parse_rank0_dec8(os.path.join(bdir, f"{b}.rank0.log"), dec8_rids)
        hs = parse_samples(os.path.join(bdir, f"{b}.head.samples"))
        ws = parse_samples(os.path.join(bdir, f"{b}.worker.samples"))

        bwaves = [analyze_wave(doc, ri, a, linfo, hs, ws) for ri, a in enumerate(reps)]
        # Verify Wave 0 is the median repetition in every single block
        assert bwaves[1]["aggregate_tok_s"] < bwaves[0]["aggregate_tok_s"] < bwaves[2]["aggregate_tok_s"], (
            f"{b} unexpected wave ordering"
        )
        waves_by_block[b] = bwaves

        # Aggregate decode_s rounding vs exact server vs prefill-stall inflation across all 24 requests
        rnd_tot = sum(w["sum_stream_rounds"] for w in bwaves)
        log_dec_s = sum(st_r["log_decode_ms"] for w in bwaves for st_r in w["streams"]) / 1000.0
        srv_dec_s = sum(st_r["srv_decode_ms"] for w in bwaves for st_r in w["streams"]) / 1000.0
        cli_dec_s = sum(st_r["cli_decode_ms"] for w in bwaves for st_r in w["streams"]) / 1000.0
        stall_s = sum(st_r["prefill_stall_in_decode_ms"] for w in bwaves for st_r in w["streams"]) / 1000.0
        true_dec_s = sum(st_r["true_active_decode_ms"] for w in bwaves for st_r in w["streams"]) / 1000.0
        engram_ms_mean = st.mean(st_r["engram_host_ms"] for w in bwaves for st_r in w["streams"])

        decode_s_audit[b] = {
            "stream_rounds": rnd_tot,
            "log_decode_sum_s": round(log_dec_s, 3),
            "srv_decode_sum_s": round(srv_dec_s, 6),
            "cli_decode_sum_s": round(cli_dec_s, 6),
            "prefill_stall_inside_decode_sum_s": round(stall_s, 6),
            "true_active_decode_sum_s": round(true_dec_s, 6),
            "log_ms_per_round": 1000.0 * log_dec_s / rnd_tot,
            "exact_srv_ms_per_round": 1000.0 * srv_dec_s / rnd_tot,
            "log_0p1s_rounding_bias_ms_per_round": 1000.0 * (log_dec_s - srv_dec_s) / rnd_tot,
            "cli_ms_per_round": 1000.0 * cli_dec_s / rnd_tot,
            "prefill_stall_inflation_ms_per_round": 1000.0 * stall_s / rnd_tot,
            "true_active_ms_per_stream_round": 1000.0 * true_dec_s / rnd_tot,
            "engram_host_mean_ms": engram_ms_mean,
        }

        t_min_cell = bwaves[0]["t_unix"] - 1.0
        t_max_cell = bwaves[-1]["t_unix"] + bwaves[-1]["full_batch_ms"] / 1000.0 + 1.0
        hs_cell = [s for s in hs if t_min_cell <= s["ts"] <= t_max_cell]
        ws_cell = [s for s in ws if t_min_cell <= s["ts"] <= t_max_cell]
        gpu_summary[b] = {
            "head_n": len(hs_cell),
            "head_clock_min_max_med": [min(x["clock_mhz"] for x in hs_cell), max(x["clock_mhz"] for x in hs_cell), int(st.median(x["clock_mhz"] for x in hs_cell))],
            "head_temp_min_max_c": [min(x["temp_c"] for x in hs_cell), max(x["temp_c"] for x in hs_cell)],
            "head_power_mean_w": round(st.mean(x["power_w"] for x in hs_cell), 2),
            "head_mem_low_gib": round(min(x["mem_avail_kb"] for x in hs_cell) / 1048576.0, 2),
            "worker_n": len(ws_cell),
            "worker_clock_min_max_med": [min(x["clock_mhz"] for x in ws_cell), max(x["clock_mhz"] for x in ws_cell), int(st.median(x["clock_mhz"] for x in ws_cell))],
            "worker_temp_min_max_c": [min(x["temp_c"] for x in ws_cell), max(x["temp_c"] for x in ws_cell)],
            "worker_power_mean_w": round(st.mean(x["power_w"] for x in ws_cell), 2),
            "worker_mem_low_gib": round(min(x["mem_avail_kb"] for x in ws_cell) / 1048576.0, 2),
        }
        log_audit[b] = {
            "anomalies_count": len(linfo["anomalies"]),
            "graphs_destroyed_during_dec8": len(linfo["graphs_during_dec8"]),
            "post_count": len(linfo["posts"]),
            "prompt_cache_hits": sum(1 for x in linfo["prompt_cache_lines"] if x["resumed"] != 0),
        }

    # Verify all 24 reply SHAs match s01 across all blocks
    ref_shas = [(w["rep"], s["stream_idx"], s["sha"]) for w in waves_by_block["s01"] for s in w["streams"]]
    for b in BLOCKS[1:]:
        b_shas = [(w["rep"], s["stream_idx"], s["sha"]) for w in waves_by_block[b] for s in w["streams"]]
        assert b_shas == ref_shas, f"SHA mismatch in {b}"

    # Verify RESULTS.md percentage deltas
    pct = lambda a, b: 100.0 * (a / b - 1.0)
    assert round(pct(repro["s02"]["tok_s_median"], repro["s01"]["tok_s_median"]), 2) == -0.59
    assert round(pct(repro["s03"]["tok_s_median"], repro["s04"]["tok_s_median"]), 2) == -0.69
    assert round(pct(repro["s04"]["tok_s_median"], repro["s01"]["tok_s_median"]), 2) == -0.17
    base_mean_tok = 0.5 * (repro["s01"]["tok_s_median"] + repro["s04"]["tok_s_median"])
    assert round(pct(repro["s05"]["tok_s_median"], base_mean_tok), 2) == -0.19
    assert round(pct(repro["s02"]["ms_per_round"], repro["s01"]["ms_per_round"]), 2) == 0.06
    assert round(pct(repro["s03"]["ms_per_round"], repro["s04"]["ms_per_round"]), 2) == -0.43

    solo_det = check_solo_determinism(args.suite_dir, docs)
    assert solo_det["dec1_3_of_3_exact_rounds"] and solo_det["probe_9_of_9_exact_rounds"]

    # Admission-order vs round-count empirical gradient across all 120 stream-runs
    ord_round_delta = {k: [] for k in range(8)}
    for ri in range(3):
        for s_idx in range(8):
            r_vals = [waves_by_block[b][ri]["streams"][s_idx]["rounds"] for b in BLOCKS]
            r_mean = st.mean(r_vals)
            for b in BLOCKS:
                st_row = waves_by_block[b][ri]["streams"][s_idx]
                ord_round_delta[st_row["adm_ord"]].append(st_row["rounds"] - r_mean)
    ord_gradient = {k: round(st.mean(v), 3) for k, v in ord_round_delta.items()}

    pair_comparisons = {}
    pair_specs = [
        ("Pair_A_s02_vs_s01", "s01", "s02"),
        ("Pair_B_s03_vs_s04", "s04", "s03"),
        ("Base_s04_vs_s01", "s01", "s04"),
        ("PDL_s05_vs_s01", "s01", "s05"),
        ("PDL_s05_vs_s04", "s04", "s05"),
    ]
    for label, b0, b1 in pair_specs:
        pair_comparisons[label] = [compare_waves(waves_by_block[b0][ri], waves_by_block[b1][ri]) for ri in range(3)]

    # Print concise forensic report
    print("== 1. EXACT REPRODUCTION OF RESULTS.md dec8 METRICS (ASSERTIONS PASSED) ==")
    for b in ("s01", "s02", "s04", "s03", "s05"):
        m = repro[b]
        da = decode_s_audit[b]
        gs = gpu_summary[b]
        print(
            f"  {b} ({docs[b]['arm']:8s}): med_tok/s={m['tok_s_median']:.2f} "
            f"[W0={m['tok_s_reps'][0]:.2f}, W1={m['tok_s_reps'][1]:.2f}, W2={m['tok_s_reps'][2]:.2f}] | "
            f"log_ms/rnd={m['ms_per_round']:.2f} exact_srv_ms/rnd={da['exact_srv_ms_per_round']:.2f} "
            f"(0.1s_round_err={da['log_0p1s_rounding_bias_ms_per_round']:+.3f}ms, "
            f"prefill_stall_in_decode_s={da['prefill_stall_inside_decode_sum_s']:.2f}s/+{da['prefill_stall_inflation_ms_per_round']:.2f}ms, "
            f"true_active_ms/rnd={da['true_active_ms_per_stream_round']:.2f}) | "
            f"clk_med H/W={gs['head_clock_min_max_med'][2]}/{gs['worker_clock_min_max_med'][2]}MHz"
        )

    print("\n== 2. MILLISECOND ATTRIBUTION OF THE MEDIAN WAVE (WAVE 0 -> DEFINES ALL REPORTED dec8 tok/s) ==")
    for label, b0, b1 in pair_specs:
        c0 = pair_comparisons[label][0]
        w0, w1 = waves_by_block[b0][0], waves_by_block[b1][0]
        print(
            f"  {label:20s}: tok/s {w0['aggregate_tok_s']:.2f}->{w1['aggregate_tok_s']:.2f} ({c0['d_tok_s_pct']:+.2f}%) | "
            f"d_win={c0['d_window_ms']:+7.2f}ms = "
            f"d_P1_prefill:{c0['d_phase1_ms']:+6.2f}ms "
            f"[8-pref_sum:{c0['d_sum_8_prefill_ms']:+6.2f}ms, ord0_excl:{c0['d_ord0_exclusion_penalty_ms']:+6.2f}ms, ovh:{c0['d_phase1_handoff_sse_ms']:+5.2f}ms] + "
            f"d_P2_decode:{c0['d_phase2_decode_ms']:+7.2f}ms "
            f"[phys_rnd:{w0['physical_rounds']}->{w1['physical_rounds']}({c0['d_physical_rounds']:+d}r), "
            f"rnd_mix:{c0['d_round_count_and_width_mix_ms']:+7.2f}ms, "
            f"matched_step_speed:{c0['d_matched_width_step_speed_ms']:+6.2f}ms]"
        )

    print("\n== 3. ALL 3 WAVES SUMMARY (PAIR A & PAIR B) ==")
    for label in ("Pair_A_s02_vs_s01", "Pair_B_s03_vs_s04"):
        print(f"  -- {label} --")
        for ri, c in enumerate(pair_comparisons[label]):
            print(
                f"    Wave {ri}: d_agg={c['d_tok_s_pct']:+6.2f}% (pure_dec={c['d_pure_decode_tok_s_pct']:+6.2f}%, full_batch={c['d_full_batch_tok_s_pct']:+6.2f}%) | "
                f"d_win={c['d_window_ms']:+7.2f}ms = d_P1:{c['d_phase1_ms']:+6.2f}ms + d_P2:{c['d_phase2_decode_ms']:+7.2f}ms "
                f"[d_phys_r={c['d_physical_rounds']:+d}, rnd_mix={c['d_round_count_and_width_mix_ms']:+7.2f}ms, step_speed={c['d_matched_width_step_speed_ms']:+7.2f}ms]"
            )

    print("\n== 4. ADMISSION-SLOT (ord=0..7) -> STREAM ROUNDS EMPIRICAL GRADIENT (120 STREAM-RUNS) ==")
    print("  " + ", ".join(f"ord={k}:{v:+.2f}r" for k, v in ord_gradient.items()))
    print(
        f"  Solo determinism check: dec1={solo_det['dec1_3_of_3_exact_rounds']} "
        f"probe={solo_det['probe_9_of_9_exact_rounds']} "
        f"cases={solo_det['solo_cases_checked']} | 24/24 dec8 token SHAs identical across s01..s05."
    )

    out_payload = {
        "suite_dir": args.suite_dir,
        "reproduced_metrics": repro,
        "decode_s_forensics": decode_s_audit,
        "gpu_samples_during_dec8": gpu_summary,
        "rank0_log_audit": log_audit,
        "solo_determinism": solo_det,
        "admission_ord_round_delta_mean": ord_gradient,
        "pair_comparisons": pair_comparisons,
        "waves_by_block": waves_by_block,
    }
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out_payload, fh, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
