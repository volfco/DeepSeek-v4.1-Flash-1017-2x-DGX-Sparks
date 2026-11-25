#!/usr/bin/env python3
"""Build the improved dashboard definitions from the exported ones. Writes dashboards-after/<id>.sent.json.

    python3 -I scripts/build_dashboards.py

Nothing is sent anywhere: this only writes files. Every existing tab and panel is kept as exported; new tabs are
appended. Inputs: dashboards-before/<id>.json (the undo), data/streams-metrics.json and data/inventory-metrics.json
(which series exist, their types and labels), data/source-metrics.json (help text).
"""
import copy
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TF_ID, DGX_ID = "7522360476127199234", "7522324382660689920"
T = "tensorfold:"
GRID = 192
CTX = 524288
SPARK = 'host_name=~"$spark"'   # the Spark selector of the DGX Sparks dashboard (both, head or worker)


def load(p):
    return json.load(open(os.path.join(ROOT, p)))


class Builder:
    def __init__(self, base_line, base_metric, prefix, first_i):
        self.base_line, self.base_metric, self.prefix = base_line, base_metric, prefix
        self.n, self.i = 0, first_i

    def panel(self, title, queries, kind="line", unit=None, unit_custom=None, desc="", legends=True, decimals=None, w=64, h=18):
        base = self.base_metric if kind == "metric" else self.base_line
        p = copy.deepcopy(base)
        self.n += 1
        self.i += 1
        p["id"] = f"Panel_{self.prefix}{self.n:04d}"
        p["type"] = kind
        p["title"] = title
        p["description"] = desc
        p["queryType"] = "promql"
        p["queries"] = []
        for q in queries:
            expr, legend = (q, "") if isinstance(q, str) else q
            qq = copy.deepcopy(base["queries"][0])
            qq["query"] = expr
            qq["fields"]["stream"] = expr
            qq["config"]["promql_legend"] = legend
            p["queries"].append(qq)
        c = p["config"]
        c["show_legends"] = legends and kind != "metric"
        c["unit"] = unit or ""
        if unit == "custom":
            c["unit_custom"] = unit_custom or ""
        else:
            c.pop("unit_custom", None)
        if decimals is not None:
            c["decimals"] = decimals
        c["mark_line"] = []
        p["layout"] = {"x": 0, "y": 0, "w": w, "h": h, "i": self.i}
        p["markdownContent"] = ""
        return p

    def note(self, title, text, w=GRID, h=12):
        p = self.panel(title, [], kind="line", w=w, h=h)
        p["type"] = "markdown"
        p["queryType"] = ""
        p["markdownContent"] = text
        p["config"]["show_legends"] = False
        return p


def flow(panels):
    """Left to right, wrapping at the grid width; a row is as tall as its tallest panel."""
    x = y = rowh = 0
    for p in panels:
        w, h = p["layout"]["w"], p["layout"]["h"]
        if x + w > GRID:
            x, y, rowh = 0, y + rowh, 0
        p["layout"]["x"], p["layout"]["y"] = x, y
        x += w
        rowh = max(rowh, h)
    return panels


def tab(tab_id, name, panels):
    return {"tabId": tab_id, "name": name, "panels": flow(panels)}


def rate(m, w="5m", by=None, sel=""):
    inner = f"rate({T}{m}{{{sel}}}[{w}])" if sel else f"rate({T}{m}[{w}])"
    return f"sum by ({by}) ({inner})" if by else f"sum({inner})"


def inc(m, w="1h"):
    return f"sum(increase({T}{m}[{w}]))"


def ref(value, text):
    return (f'label_replace(vector({value}), "ref", "{text}", "", "")', "{ref}")


def custom_var(name, label, options, value):
    return {"type": "custom", "name": name, "label": label, "query_data": None, "value": value,
            "options": [{"label": l, "value": v, "selected": v == value} for l, v in options],
            "multiSelect": False, "hideOnDashboard": False, "selectAllValueForMultiSelect": "first",
            "customMultiSelectValue": [], "escapeSingleQuotes": False, "scope": "global", "tabs": [], "panels": []}


def textbox_var(name, label, value):
    return {"type": "textbox", "name": name, "label": label, "query_data": None, "value": value, "options": [],
            "multiSelect": False, "hideOnDashboard": False, "selectAllValueForMultiSelect": "first",
            "customMultiSelectValue": [], "escapeSingleQuotes": False, "scope": "global", "tabs": [], "panels": []}


# ---------------------------------------------------------------- Tensorfold
def tensorfold():
    saved = load(f"dashboards-before/{TF_ID}.json")
    d = copy.deepcopy(saved["v8"])
    line = next(p for p in d["tabs"][0]["panels"] if p["type"] == "line")
    metric = next(p for p in d["tabs"][0]["panels"] if p["type"] == "metric")
    b = Builder(line, metric, "R", 1000)
    P, N = b.panel, b.note
    gen, dsum, dcnt, rounds = "generation_tokens_total", "request_decode_seconds_sum", "request_decode_seconds_count", "decode_rounds_total"
    drafted, accepted = "mtp_drafted_total", "mtp_accepted_total"
    prompt, cached, psum, pcnt = "prompt_tokens_total", "prompt_tokens_cached_total", "request_prefill_seconds_sum", "request_prefill_seconds_count"
    tsum, tcnt, lsum, lcnt = "time_to_first_token_seconds_sum", "time_to_first_token_seconds_count", "request_latency_seconds_sum", "request_latency_seconds_count"
    osum, ocnt = "request_time_per_output_token_seconds_sum", "request_time_per_output_token_seconds_count"

    def ratio(a, bb, k=1, w="5m"):
        return f"{k} * {rate(a, w)} / ({rate(bb, w)} > 0)" if k != 1 else f"{rate(a, w)} / ({rate(bb, w)} > 0)"

    ms_round = f"1000 * {rate(dsum)} / (({rate(rounds)} - {rate(dcnt)}) > 0)"
    pf_rate = f"({rate(prompt)} - {rate(cached)}) / ({rate(psum)} > 0)"
    bench = [
        N("How to read this tab", (
            "**Live engine rates beside the benchmarked figures.** Every live line is a ratio of counter rates over 5 minutes, "
            "so it is reset-aware and does not depend on how many requests there were. The engine updates these counters "
            "**when a request finishes**, so a long reply shows up at its end, and a line has gaps while nothing finishes.\n\n"
            "- *Decode tok/s a stream* = generated tokens / decode seconds. It is a **per-stream** rate: with N streams decoding "
            "together the pair's total is about N times this (the *all streams* panel shows the total by wall clock).\n"
            "- *ms a round* = decode seconds / (rounds − requests); *tokens a round* = generated / rounds; "
            "*accepted / drafted* = draft tokens kept / draft tokens verified.\n"
            "- *Prefill tok/s* = (prompt − cached) tokens / prompt-pass seconds. TTFT also holds queueing, restore and the first round.\n\n"
            "**Reference lines** (legend entries starting `bench:`) come from the recipe's records: the sampled probe "
            "`runs/overnight-20261122/k265/ctl1.probe.json` and `k269` (thinking off, short prompts, 1 / 2 / 4 streams), "
            "`runs/k278` (cold prefill 32K / 96K, greedy decode at 300K) and `runs/cache-20261122` (8K turns). "
            "The 4- and 8-stream totals marked *dec4* / *dec8* are the decode cells of `runs/suite-fused-20261121` and `suite-fused-next2-20261121`. "
            "The sampled probe stops at 4 streams and uses short prompts; live traffic runs up to 8 streams on contexts of tens to hundreds of "
            "thousands of tokens, so a gap to a reference line is a difference in workload first. Compare *ms a round* and "
            "*prefill tok/s* before reading a tok/s gap as a regression. Details: `runs/observe-review-20261123/RESULTS.md`."), h=16),
        P("Decode tok/s a stream (1 h)", [f"{inc(gen)} / ({inc(dsum)} > 0)"], "metric", "custom", "tok/s", decimals=1, w=32, h=10),
        P("Tokens a round (1 h)", [f"{inc(gen)} / ({inc(rounds)} > 0)"], "metric", decimals=2, w=32, h=10),
        P("Accepted / drafted (1 h)", [f"100 * {inc(accepted)} / ({inc(drafted)} > 0)"], "metric", "percent", decimals=1, w=32, h=10),
        P("Decode ms a round (1 h)", [f"1000 * {inc(dsum)} / (({inc(rounds)} - {inc(dcnt)}) > 0)"], "metric", "custom", "ms", decimals=1, w=32, h=10),
        P("Prefill tok/s, uncached (1 h)", [f"({inc(prompt)} - {inc(cached)}) / ({inc(psum)} > 0)"], "metric", "custom", "tok/s", decimals=0, w=32, h=10),
        P("Cached share of prompt tokens (1 h)", [f"100 * {inc(cached)} / ({inc(prompt)} > 0)"], "metric", "percent", decimals=1, w=32, h=10),
        P("Decode tok/s a stream, live against benchmark", [
            (ratio(gen, dsum), "live, a stream"), ref(128, "bench: sampled code-like x1 (106-135)"), ref(60, "bench: sampled prose x1 (59-60)"),
            ref(55, "bench: sampled x4, each stream (220 total)"), ref(74, "bench: greedy at 300K depth")],
          unit="custom", unit_custom="tok/s", desc="generation_tokens_total / request_decode_seconds_sum. Per-stream rate of the requests that finished."),
        P("Generated tok/s, all streams (wall clock)", [
            (rate(gen), "live, all streams"), ref(219.8, "bench: sampled x4 total"), ref(168.4, "bench: sampled x2 total"), ref(129.7, "bench: streamed x1"),
            ref(138, "bench: 8 streams, dec8 cell (136-138.5)"), ref(117, "bench: 4 streams, dec4 cell (116-119)")],
          unit="custom", unit_custom="tok/s", desc="rate(generation_tokens_total). Tokens are counted when their request ends, so bursts of long replies arrive late and in steps."),
        P("Decode ms a round, live against benchmark", [
            (ms_round, "live"), ref(42.6, "bench: x1 code-like (5.5 drafted a round)"), ref(30.2, "bench: x1 prose (1.5 drafted a round)"), ref(66.2, "bench: x4 (chunk gap p50)"), ref(87, "bench: 8 streams (dec8 round)")],
          unit="custom", unit_custom="ms", desc="request_decode_seconds_sum / (decode_rounds_total - request_decode_seconds_count). A round serves every live stream, so it grows with streams and with drafted rows."),
        P("Tokens a round, live against benchmark", [
            (ratio(gen, rounds), "live"), ref(5.4, "bench: x1 code-like (4.2-6.1)"), ref(1.8, "bench: x1 prose"), ref(3.7, "bench: x4, each stream")],
          desc="generation_tokens_total / decode_rounds_total."),
        P("Drafted rows a round", [
            (ratio(drafted, rounds), "live"), ref(5.5, "bench: x1 code-like"), ref(1.5, "bench: x1 prose"), ref(3.0, "bench: x4, each stream")],
          desc="mtp_drafted_total / decode_rounds_total: the draft tokens a stream had verified a round."),
        P("Accepted / drafted, live against benchmark", [
            (ratio(accepted, drafted, 100), "live"), ref(81, "bench: x1 code-like (73-87)"), ref(53, "bench: x1 prose (51-54)"), ref(90, "bench: x4")],
          unit="percent", desc="mtp_accepted_total / mtp_drafted_total."),
        P("Prefill tok/s on uncached tokens, live against benchmark", [
            (pf_rate, "live"), ref(2105, "bench: cold 32K"), ref(2041, "bench: cold 96K"), ref(1680, "bench: 8K turn, 1.8K uncached tail")],
          unit="custom", unit_custom="tok/s", desc="(prompt_tokens_total - prompt_tokens_cached_total) / request_prefill_seconds_sum."),
        P("Time to first token: mean, and its prompt-pass part", [
            (ratio(tsum, tcnt), "TTFT mean"), (ratio(psum, pcnt), "prompt pass mean"),
            (f"({rate(tsum)} - {rate(psum)}) / ({rate(tcnt)} > 0)", "TTFT less prompt pass (queue, restore, encode, first round)")],
          unit="s", desc="time_to_first_token_seconds and request_prefill_seconds, sum / count."),
        P("TTFT per 1,000 uncached prompt tokens", [
            (f"1000 * {rate(tsum)} / (({rate(prompt)} - {rate(cached)}) > 0)", "live"), ref(0.475, "bench: cold prefill at 2,105 tok/s")],
          unit="s", desc="TTFT against prompt depth: seconds of TTFT for each 1,000 prompt tokens that had to be computed. Queueing lifts it."),
        P("Prompt depth: tokens a request", [
            (ratio(prompt, lcnt), "prompt tokens, mean"), (f"({rate(prompt)} - {rate(cached)}) / ({rate(lcnt)} > 0)", "uncached prompt tokens, mean"),
            (ratio(gen, lcnt), "generated tokens, mean")], desc="Means over the requests that finished."),
        P("Streams running and waiting", [
            (f"sum({T}requests_running)", "running"), (f"sum({T}requests_waiting)", "waiting"), ref(4, "bench: the sampled probe's widest burst"), ref(8, "bench: the widest suite cell (dec8)")],
          desc="Gauges read at scrape time (every 30 s)."),
        P("Tokens each live stream holds", [(f"{T}kv_cache_usage_ratio * {CTX}", "stream {pool}")],
          desc=f"kv_cache_usage_ratio x the {CTX:,}-token context window: prompt plus reply of each live stream."),
    ]
    q = lambda m, x: f"histogram_quantile({x}, sum by (le) (rate({T}{m}_bucket[5m])))"
    thr = [
        P("Requests finished a minute", [(f"60 * {rate(lcnt)}", "requests / min")], desc="request_latency_seconds_count."),
        P("Mean seconds a request: latency, TTFT, prompt pass, decode", [
            (ratio(lsum, lcnt), "latency"), (ratio(tsum, tcnt), "time to first token"), (ratio(psum, pcnt), "prompt pass"), (ratio(dsum, dcnt), "decode")], unit="s",
          desc="Each histogram's _sum / _count."),
        P("Observations a second, by histogram", [
            (rate(lcnt), "latency"), (rate(tcnt), "TTFT"), (rate(pcnt), "prompt pass"), (rate(dcnt), "decode"), (rate(ocnt), "time per output token")],
          desc="The _count series: a request without a first token has no TTFT, decode or per-token observation."),
        P("Busy seconds a second, by histogram sum", [
            (rate(lsum), "latency"), (rate(tsum), "TTFT"), (rate(psum), "prompt pass"), (rate(dsum), "decode")], unit="s",
          desc="The _sum series as rates: request-seconds finished a second (above 1 means overlapping requests)."),
        P("Time per output token: mean, p50, p95", [
            (f"1000 * {rate(osum)} / ({rate(ocnt)} > 0)", "mean"), (f"1000 * {q('request_time_per_output_token_seconds', 0.5)}", "p50"),
            (f"1000 * {q('request_time_per_output_token_seconds', 0.95)}", "p95")], unit="custom", unit_custom="ms",
          desc="request_time_per_output_token_seconds: first token to last, divided by tokens less one, one observation a request."),
        P("Time to first token quantiles", [(q("time_to_first_token_seconds", x), f"p{int(x * 100)}") for x in (0.5, 0.9, 0.99)], unit="s",
          desc="Bucket edges are coarse (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300 s): read the mean panel for small changes."),
        P("Decode seconds quantiles", [(q("request_decode_seconds", x), f"p{int(x * 100)}") for x in (0.5, 0.9, 0.99)], unit="s"),
        P("Prompt pass seconds quantiles", [(q("request_prefill_seconds", x), f"p{int(x * 100)}") for x in (0.5, 0.9, 0.99)], unit="s"),
        P("Request latency quantiles", [(q("request_latency_seconds", x), f"p{int(x * 100)}") for x in (0.5, 0.9, 0.99)], unit="s"),
        P("Tokens a second: prompt, cached, generated", [(rate(prompt), "prompt"), (rate(cached), "of it, cached"), (rate(gen), "generated")],
          unit="custom", unit_custom="tok/s"),
        P("Generated tokens held by live streams", [(f"sum({T}generation_tokens_running)", "tokens")]),
        P("Energy (TF_DS_ENERGY)", [(rate("joules_total"), "watts over finished requests")], unit="custom", unit_custom="W",
          desc="joules_total exists only while TF_DS_ENERGY is on; production runs with it off, so this is empty (last samples 2026-11-17)."),
    ]
    cause = [
        P("Memory misses by cause", [(rate("prompt_cache_misses_no_prefix"), "no prefix matched"), (rate("prompt_cache_misses_tail_window"), "tail / window / grid rule"),
                                     (rate("prompt_cache_misses_stale"), "state was stale")], desc="TF_DS_CACHE_METRICS causes of prompt_cache_memory_misses."),
        P("Keeps refused by cause", [(rate("prompt_cache_refused_budget"), "past the memory budget"), (rate("prompt_cache_refused_restores"), "restores held the budget")]),
        P("Evictions by cause", [(rate("prompt_cache_evicted_superseded"), "superseded turn"), (rate("prompt_cache_evicted_lru"), "least recently used")]),
        P("Failures by cause", [(rate("prompt_cache_failed_save"), "save failed"), (rate("prompt_cache_failed_restore"), "restore failed")]),
        P("Resume depth of prompt passes", [(q("prompt_cache_resume_depth_percent", x), f"p{int(x * 100)}") for x in (0.1, 0.5, 0.9)]
          + [(f"{rate('prompt_cache_resume_depth_percent_sum')} / ({rate('prompt_cache_resume_depth_percent_count')} > 0)", "mean")], unit="percent",
          desc="How far into its prompt a pass resumed from a kept state (buckets 0/25/50/75/90/99/100 percent)."),
        P("Prompt passes with a resume depth", [(f"60 * {rate('prompt_cache_resume_depth_percent_count')}", "passes / min")]),
        P("Chats continued without their own turn state", [(rate("prompt_cache_chats_continued"), "streams / s"),
                                                           (rate("prompt_cache_chats_continued_prefill_tokens"), "prompt tokens they prefilled / s")],
          desc="prompt_cache_chats_continued and prompt_cache_chats_continued_prefill_tokens (the same counts as tensorfold:chats_continued*)."),
    ]
    engram = [
        P("Row-cache has() calls", [(rate("engram_rowcache_has_total"), "calls / s"), (rate("engram_rowcache_fast_total"), "answered by the filter, no lock"),
                                    (rate("engram_rowcache_lock_total"), "took the lock")]),
        P("Share of has() calls answered without the lock", [(f"100 * {rate('engram_rowcache_fast_total')} / ({rate('engram_rowcache_has_total')} > 0)", "fast share")], unit="percent"),
        P("Engram read: mean seconds", [(f"{rate('engram_lookup_seconds_sum')} / ({rate('engram_lookup_seconds_count')} > 0)", "mean")], unit="s",
          desc="engram_lookup_seconds_sum / _count."),
        P("Engram read seconds a second", [(rate("engram_lookup_seconds_sum"), "seconds / s")], unit="s"),
    ]
    rk = 'rank=~"$rank"'
    kv = [P(title, [(rate(m, by="rank", sel=rk), "rank {rank}")], unit=unit, desc=f"{T}{m}")
          for title, m, unit in (
              ("Cross-part copies by rank", "kv_shard_rank_copies", None), ("Cross-part copy bytes by rank", "kv_shard_rank_copy_bytes", "bps"),
              ("Decode rounds that fetched owners' rows, by rank", "kv_shard_rank_fetch_rounds", None),
              ("Decode rounds attended as an owner, by rank", "kv_shard_rank_owner_rounds", None),
              ("Prompt fetch: rows unfolded, by rank", "kv_shard_rank_prompt_picks", None), ("Prompt fetch: compressed rows, by rank", "kv_shard_rank_prompt_rows", None),
              ("Prompt fetch: host reads waited on, by rank", "kv_shard_rank_prompt_syncs", None))]
    kv += [P("Prompt fetch: rows unfolded, all ranks", [(rate("kv_shard_prompt_picks"), "picks / s")], desc=f"{T}kv_shard_prompt_picks"),
           P("Kept states staged before their rows changed", [(rate("kv_shard_staged"), "states / s")], desc=f"{T}kv_shard_staged")]
    vllm = [
        N("About this tab", "The engine publishes some families twice: once under its own name and once under vLLM's, so vLLM dashboards work unchanged. "
          "These panels show the vLLM-named copies; the values equal the native series shown on the other tabs.", h=6),
        P("num_requests_running / num_requests_waiting", [(f"sum({T}num_requests_running)", "running"), (f"sum({T}num_requests_waiting)", "waiting")]),
        P("kv_cache_usage_perc by stream", [(f"100 * {T}kv_cache_usage_perc", "stream {stream}")], unit="percent"),
        P("spec_decode draft and accepted tokens", [(rate("spec_decode_num_draft_tokens_total"), "draft / s"), (rate("spec_decode_num_accepted_tokens_total"), "accepted / s"),
                                                    (f"100 * {rate('spec_decode_num_accepted_tokens_total')} / ({rate('spec_decode_num_draft_tokens_total')} > 0)", "accepted share, %")]),
        P("e2e_request_latency_seconds: mean, p50, p95", [(f"{rate('e2e_request_latency_seconds_sum')} / ({rate('e2e_request_latency_seconds_count')} > 0)", "mean"),
                                                          (q("e2e_request_latency_seconds", 0.5), "p50"), (q("e2e_request_latency_seconds", 0.95), "p95")], unit="s"),
        P("request_decode_time_seconds: mean, p50, p95", [(f"{rate('request_decode_time_seconds_sum')} / ({rate('request_decode_time_seconds_count')} > 0)", "mean"),
                                                          (q("request_decode_time_seconds", 0.5), "p50"), (q("request_decode_time_seconds", 0.95), "p95")], unit="s"),
        P("request_prefill_time_seconds: mean, p50, p95", [(f"{rate('request_prefill_time_seconds_sum')} / ({rate('request_prefill_time_seconds_count')} > 0)", "mean"),
                                                           (q("request_prefill_time_seconds", 0.5), "p50"), (q("request_prefill_time_seconds", 0.95), "p95")], unit="s"),
    ]
    tfjob = 'service_name="tensorfold"'
    health = [
        N("About this tab", "The collector on each Spark scrapes `localhost:8000/metrics` every 30 s. Only rank 0 (the head, `gx10-eb0e`) serves that endpoint, "
          "so the worker's target reads **up = 0** all the time and every `tensorfold:*` series comes from the head; the head reports both ranks where a family "
          "has a `rank` label (spill tier, KV shard). A drop of the head's `up` to 0 is the engine restarting or stopped.", h=7),
        P("Scrape target up, by host", [(f"max by (host_name) (up{{{tfjob}}})", "{host_name}")], desc="1: the last scrape of localhost:8000/metrics worked."),
        P("Samples a scrape, by host", [(f"max by (host_name) (scrape_samples_scraped{{{tfjob}}})", "scraped {host_name}"),
                                        (f"max by (host_name) (scrape_samples_post_metric_relabeling{{{tfjob}}})", "kept {host_name}")],
          desc="The engine emits about 373 samples; a difference between scraped and kept would mean the collector drops some."),
        P("Scrape duration and new series", [(f"max by (host_name) (scrape_duration_seconds{{{tfjob}}})", "seconds {host_name}"),
                                             (f"max by (host_name) (scrape_series_added{{{tfjob}}})", "series added {host_name}")]),
        P("Engine restarts in the last hour", [(f"sum(resets({T}engram_lookup_seconds_count[1h]))", "counter resets"),
                                               (f"sum(changes({T}device_memory_peak_bytes[1h]))", "changes of peak device memory")],
          desc="A restart zeroes the engine's counters; benchmark suites restart it for every arm."),
        P("Device memory: now and peak", [(f"sum({T}device_memory_bytes)", "held now"), (f"max({T}device_memory_peak_bytes)", "peak")], unit="bytes"),
        P("Pair state", [(f"max({T}pair_broken)", "broken"), (f"max({T}pair_stalled)", "stalled"), (f"sum(increase({T}pair_failed[5m]))", "requests failed (5 m)"),
                         (f"sum(increase({T}service_refused[5m]))", "services refused (5 m)")]),
        P("Engine thread busy without progress", [(f"max({T}pair_busy_ms)", "busy ms"), (f"max({T}pair_stall_ms)", "stall bound ms")], unit="custom", unit_custom="ms"),
        P("Disk tier by rank: stored bytes", [(f'sum by (rank) ({T}spill_stored_bytes{{{rk}}})', "rank {rank}")], unit="bytes"),
        P("Disk tier by rank: points and blocks", [(f'sum by (rank) ({T}spill_points{{{rk}}})', "points, rank {rank}"), (f'sum by (rank) ({T}spill_blocks{{{rk}}})', "blocks, rank {rank}")]),
        P("Disk tier by rank: reads and writes", [(rate("spill_reads", by="rank", sel=rk), "reads / s, rank {rank}"), (rate("spill_writes_committed", by="rank", sel=rk), "writes / s, rank {rank}")]),
    ]
    streams = load("data/streams-metrics.json")["list"]
    live = sorted(s["name"] for s in streams if s["name"].startswith(T) and s["stats"]["doc_num"] > 0)
    retired = sorted(s["name"] for s in streams if s["name"].startswith(("tensorfold_", "tensorfold_health:")) and s["stats"]["doc_num"] > 0)
    explorer = [
        N("About this tab", f"Pick any engine series in **Series** at the top: all {len(live)} `tensorfold:*` series that hold samples are listed, then the {len(retired)} retired "
          "`tensorfold_*` / `tensorfold_health:*` names (they stopped on 2026-11-10 and 2026-10-31: widen the time range to see them). "
          "The left panel shows the samples as stored, one line a label set; the right one the per-second rate, which is the reading that makes sense for a counter.", h=6),
        P("$series as stored", [('{__name__="$series"}', "{__name__} {rank} {part} {pool} {stream} {le}")], w=96, h=24),
        P("$series as a rate (counters)", [('rate({__name__="$series"}[5m])', "{rank} {part} {pool} {stream} {le}")], w=96, h=24),
    ]
    d["tabs"] += [tab("bench_live", "Benchmark vs live", bench), tab("thr_lat", "Throughput & latency (means)", thr),
                  tab("cache_causes", "Cache causes & resume depth", cause), tab("engram_rc", "Engram row cache", engram),
                  tab("kv_rank", "KV shard by rank", kv), tab("vllm_names", "vLLM-name copies", vllm),
                  tab("ranks_scrape", "Ranks, scrape & restarts", health), tab("explorer", "Any series", explorer)]
    opts = [(n, n) for n in live] + [("retired: " + n, n) for n in retired]
    d["variables"] = {"list": [custom_var("rank", "Rank", [("all", ".*"), ("0 (head)", "0"), ("1 (worker)", "1")], ".*"),
                               custom_var("series", "Series (Any series tab)", opts, T + "requests_running")],
                      "showDynamicFilters": False}
    d["description"] = ("Live tensorfold-native engine metrics (tensorfold:* families, scraped from rank 0 every 30 s). Original tabs: overview, Engram & disk reads, "
                        "prompt cache, spill tier, KV shard, latency/drafting/pair. Added 2026-11-23: benchmark vs live (derived rates beside the recipe's reference "
                        "figures), means from the histogram sums, cache causes and resume depth, Engram row cache, KV shard by rank, vLLM-name copies, ranks / scrape / "
                        "restarts, and a selector that shows any series. Undo: runs/observe-review-20261123 in the recipe repository.")
    return d


# ---------------------------------------------------------------- DGX Sparks
def dgx():
    saved = load(f"dashboards-before/{DGX_ID}.json")
    d = copy.deepcopy(saved["v8"])
    line = d["tabs"][0]["panels"][0]
    b = Builder(line, line, "S", 2000)
    P, N = b.panel, b.note
    inv = {s["name"]: s for s in load("data/inventory-metrics.json")["streams"]}

    def on(svc):
        return sorted(n for n, s in inv.items() if any(h["h"] == "gx10-eb0e" and h.get("svc") == svc for h in s["hosts"])
                      and not n.startswith(("scrape_", "up")) and not n.startswith(T))

    def expr(n, extra_by=()):
        s = inv[n]
        labs = [l for l in ("device_name", "hca", "port", "device", "direction", "state", "type", "mountpoint", "cpu", "protocol", "status", "mode", "gpu", "err_code")
                if l in s["labels"]]
        if n == "system_cpu_time":
            labs = ["state"]
        if n.startswith("system_filesystem"):
            labs = [l for l in labs if l in ("mountpoint", "state")]
        by = ", ".join(["host_name"] + labs)
        leg = " ".join(["{host_name}"] + ["{%s}" % l for l in labs])
        if s["type"] == "Counter":
            return f"sum by ({by}) (rate({n}{{{SPARK}}}[5m]))", leg
        return f"max by ({by}) ({n}{{{SPARK}}})", leg

    def unit_of(n):
        """(unit, custom text) by series name; counters are shown as per-second rates."""
        rules = [(r"_utilization$", "percent-1", None), (r"^system_(disk|network)_io$", "bps", None), (r"(filesystem|memory|paging)_usage$", "bytes", None),
                 (r"^system_(cpu_time|disk_io_time|disk_operation_time|disk_weighted_io_time)$", "custom", "s/s"),
                 (r"^system_(disk_merged|disk_operations|paging_operations)$", "custom", "ops/s"), (r"^system_network_(dropped|errors|packets)$", "custom", "pkt/s"),
                 (r"^system_paging_faults$", "custom", "faults/s"), (r"^system_processes_created$", "custom", "/s"),
                 (r"_temp$", "custom", "°C"), (r"_util$", "percent", None), (r"power_usage$", "custom", "W"), (r"_clock$", "custom", "MHz"),
                 (r"total_energy_consumption$", "custom", "mJ/s"), (r"pcie_replay_counter$", "custom", "/s")]
        for pat, u, c in rules:
            if re.search(pat, n):
                return u, c
        return None, None

    def one(n):
        u, c = unit_of(n)
        return P(n, [expr(n)], unit=u, unit_custom=c, desc=("per-second rate" if inv[n]["type"] == "Counter" else "gauge"), w=64, h=16)

    host = [one(n) for n in on(None)]
    gpu = [one(n) for n in on("dcgm-gpu-metrics")]
    gpu.insert(0, P("GPU power from the energy counter", [(f"sum by (host_name) (rate(dcgm_fi_dev_total_energy_consumption{{{SPARK}}}[5m])) / 1000", "{host_name}")],
                    unit="custom", unit_custom="W", desc="dcgm_fi_dev_total_energy_consumption is in millijoules; its rate / 1000 is watts.", w=64, h=16))
    doca = on("doca-telemetry")
    dev = [n for n in doca if not re.search(r"\d", n)]            # device-level names (no queue or interface index)
    idx = [n for n in doca if re.search(r"\d", n)]
    groups = {}
    order = [("Link state, speed and width", r"^(ib_port_state|hw_port_state|current_link_|max_link_|link_|local_link|symbol_error|total_success_recovery|module_|excessive_|timestamp$|rdu$|tdu$)"),
             ("InfiniBand-style port counters", r"^port_"), ("RoCE transport (hw_*)", r"^hw_"), ("PCI and device queue", r"^(outbound_pci|dev_internal|rx_pci|tx_pci)"),
             ("Physical layer receive", r"^rx_.*_phy$"), ("Physical layer transmit", r"^tx_.*_phy$"), ("Pause and congestion", r"(pause|ecn|congst|out_of_buffer|buffer_passed)"),
             ("RDMA and vport traffic", r"vport"), ("Receive errors and drops", r"^rx_.*(err|drop|missed|discard|oversize|runt|too_long|unknown|unsupported|if_down|wqe)"),
             ("Transmit errors and queue", r"^tx_.*(err|drop|aborted|collision|deferred|underrun|queue|recover|timeout)"),
             ("Receive offloads: checksum, GRO, LRO, header split", r"^rx_(csum|gro|lro|hds|cqe|mpwqe|removed|arfs|recover|steer|tcam|fragments|jabbers|mac_|octets|bytes|packets|multicast|broadcast)"),
             ("Receive page pool", r"^rx_pp_"), ("Receive XDP and XSK", r"^rx_(xdp|xsk)"), ("Receive kTLS", r"^rx_tls"),
             ("Transmit offloads and traffic", r"^tx_(csum|tso|mpwqe|cqe|nop|added|octets|bytes|packets|multicast|broadcast|unicast|xdp)"), ("Transmit kTLS", r"^tx_tls"),
             ("IPsec", r"^ipsec_"), ("MACsec", r"^macsec_"), ("Interrupt channels", r"^ch_"), ("Other device counters", r".")]
    for n in dev:
        for title, pat in order:
            if re.search(pat, n):
                groups.setdefault(title, []).append(n)
                break
    cx = [N("About these panels", f"DOCA telemetry holds {len(doca):,} series a Spark. The {len(dev)} device-level counters are all on the panels below, grouped by what they describe "
            "(counters as per-second rates, gauges as stored; one line a host and port). The other "
            f"{len(idx):,} are the same counters split by receive or transmit queue (`rx0_*` … `rx19_*`, `tx0_*` … `tx19_*`, `ch0_*`) or by priority; "
            "show any of them on the *Any series* tab by typing its name.", h=7)]
    for title, _ in order:
        names = groups.get(title, [])
        for k in range(0, len(names), 8):
            part = names[k:k + 8]
            suffix = f" ({k // 8 + 1} of {(len(names) + 7) // 8})" if len(names) > 8 else ""
            qs = []
            for n in part:
                e, leg = expr(n)
                qs.append((e, n + " " + leg))
            kinds = set(inv[n]["type"] for n in part)
            if kinds == {"Counter"} and all(re.search(r"bytes|octets|_data$", n) for n in part):
                u, c = "bps", None
            elif kinds == {"Counter"}:
                u, c = "custom", "/s"
            else:
                u, c = None, None   # gauges, or gauges and counters together: plain numbers
            cx.append(P(title + suffix, qs, unit=u, unit_custom=c, desc="Series: " + ", ".join(part), w=96, h=18))
    col = [P("Scrape targets up, by host and job", [(f"max by (host_name, service_name) (up{{{SPARK}}})", "{host_name} {service_name}")], w=96, h=16,
             desc="tensorfold on gx10-235f (the worker) is 0 by design: rank 1 serves no /metrics endpoint."),
           P("Samples a scrape", [(f"max by (host_name, service_name) (scrape_samples_scraped{{{SPARK}}})", "{host_name} {service_name}")], w=96, h=16),
           P("Samples kept after relabeling", [(f"max by (host_name, service_name) (scrape_samples_post_metric_relabeling{{{SPARK}}})", "{host_name} {service_name}")], w=64, h=16),
           P("Scrape duration", [(f"max by (host_name, service_name) (scrape_duration_seconds{{{SPARK}}})", "{host_name} {service_name}")], unit="s", w=64, h=16),
           P("Series added a scrape", [(f"max by (host_name, service_name) (scrape_series_added{{{SPARK}}})", "{host_name} {service_name}")], w=64, h=16)]
    anyp = [N("About this tab", "Type any metric name held for the Sparks into **Any series name** at the top (for example `rx0_packets`, `tx7_bytes`, `ch3_poll`, "
              "`system_cpu_time`, `dcgm_fi_dev_gpu_temp`). The full list of names is in `runs/observe-review-20261123/COVERAGE.md` and `data/` beside it.", h=5),
            P("$any_series as stored", [(f'{{__name__="$any_series", {SPARK}}}', "{host_name} {device_name} {hca} {port} {state} {device} {direction}")], w=96, h=24),
            P("$any_series as a rate (counters)", [(f'sum by (host_name, device_name) (rate({{__name__="$any_series", {SPARK}}}[5m]))', "{host_name} {device_name}")], w=96, h=24)]
    d["tabs"] += [tab("host_all", "Host: every series", host), tab("gpu_all", "GPU: every DCGM series", gpu), tab("cx_all", "ConnectX: every device counter", cx),
                  tab("collector", "Collector & scrape", col), tab("any_series", "Any series", anyp)]
    d["variables"]["list"] = [v for v in d["variables"]["list"] if v["name"] not in ("any_series", "spark")] + [
        custom_var("spark", "Spark (new tabs)", [("both", "gx10-.*"), ("head gx10-eb0e", "gx10-eb0e"), ("worker gx10-235f", "gx10-235f")], "gx10-.*"),
        textbox_var("any_series", "Any series name", "rx0_packets")]
    d["description"] = ("Host, GPU (dcgm-exporter) and ConnectX-7 (DOCA telemetry) metrics for the DGX Spark ring. Added 2026-11-23: every host series, every DCGM series, "
                        "every device-level ConnectX counter, the collector's own scrape health, and a box that shows any series by name; the Spark selector picks the head, the worker or both on those tabs. "
                        "Undo: runs/observe-review-20261123 in the recipe repository.")
    meta = {"doca_total": len(doca), "doca_device_level": dev, "doca_indexed": idx, "doca_groups": groups,
            "host": on(None), "dcgm": on("dcgm-gpu-metrics")}
    json.dump(meta, open(os.path.join(ROOT, "data/dgx-coverage.json"), "w"))
    return d


def main():
    os.makedirs(os.path.join(ROOT, "dashboards-after"), exist_ok=True)
    for dash_id, fn in ((TF_ID, tensorfold), (DGX_ID, dgx)):
        d = fn()
        ids = [p["id"] for t in d["tabs"] for p in t["panels"]]
        assert len(ids) == len(set(ids)), "duplicate panel ids"
        li = [p["layout"]["i"] for t in d["tabs"] for p in t["panels"]]
        for t in d["tabs"]:
            assert all(p["layout"]["x"] + p["layout"]["w"] <= GRID for p in t["panels"]), t["name"]
        json.dump(d, open(os.path.join(ROOT, f"dashboards-after/{dash_id}.sent.json"), "w"), indent=1)
        print(d["title"], "tabs", len(d["tabs"]), "panels", len(ids), "queries", sum(len(p["queries"]) for t in d["tabs"] for p in t["panels"]),
              "| per tab:", [(t["name"], len(t["panels"])) for t in d["tabs"]], "| duplicate layout i:", len(li) - len(set(li)))


if __name__ == "__main__":
    main()
