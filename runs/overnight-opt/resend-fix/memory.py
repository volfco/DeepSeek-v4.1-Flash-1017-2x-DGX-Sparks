"""Compute retained payloads for the actual two-rank model and check the head's budget ledger."""
import json
import re
from pathlib import Path

here = Path(__file__).resolve().parent
geometry = json.loads((here / "model-geometry.json").read_text())
head, peer = geometry["head"], geometry["worker"]
assert head["sha256"] == peer["sha256"] and head["fields"] == peer["fields"]
conf = head["fields"]
source_ratio = {i: conf["compress_ratios"][i] for i in conf["kv_source_layer_ids"]}
index_ratio = {i: ratio for i, ratio in source_ratio.items() if i in conf["index_source_layer_ids"]}
hd = conf["head_dim"]
index_hd = conf["index_head_dim"]
ring = (conf["sliding_window"] + 16) * hd * 2
stage = conf["num_nextn_predict_layers"] * ring
logits = conf["vocab_size"] * 4
raw = sum(2 * 64 * hd * 4 for ratio in source_ratio.values() if ratio > 1)
base = conf["num_hidden_layers"] * ring + raw
full = base + stage + logits
budget = 512 * 2**20
ledger_path = here.parent / "engineering02" / "endstate" / "rank0.log"
ledger = re.findall(r"(\d+) states, (\d+) MiB and \d+ MiB spare of 512 MiB", ledger_path.read_text())
assert ledger and all(int(mib) == int(number) * full // 2**20 for number, mib in ledger)


def retained(depth):
    index_bytes = sum(depth // ratio for ratio in index_ratio.values()) * (index_hd // 2 + index_hd // 32)
    owned_bytes = sum(depth // ratio for ratio in source_ratio.values()) * (hd // 2 + hd // 16)
    return {"prompt_tokens": depth, "head_token_copies_at_keep_bytes": 8 * depth,
            "index_bytes_per_rank": index_bytes, "compressed_bytes_on_owner": owned_bytes,
            "owner_combined_bytes": index_bytes + owned_bytes, "other_rank_bytes": index_bytes}


result = {"model_sha": head["sha256"], "layers": conf["num_hidden_layers"], "kv_ratios": source_ratio,
          "per_rank": {"one_layer_ring_bytes": ring, "base_model_rings_bytes": conf["num_hidden_layers"] * ring,
                       "raw_kv_and_score_bytes": raw, "ordinary_payload_bytes": base,
                       "dspark_rings_bytes": stage, "final_logits_bytes": logits, "full_extra_bytes": stage + logits,
                       "one_full_payload_bytes": full, "two_full_payload_bytes": 2 * full,
                       "two_full_extra_bytes": 2 * (stage + logits), "three_transient_full_bytes": 3 * full},
          "ordinary_trim_only": {"payload_bytes": conf["num_hidden_layers"] // 2 * ring + raw,
                                 "slot_ceiling": budget // (conf["num_hidden_layers"] // 2 * ring + raw)},
          "head_logical_budget_bytes": budget, "steady_full_limit": 2,
          "uniform_slot_ceiling_with_end": budget // full, "ordinary_ceiling_without_end": budget // base,
          "contextual_charging_candidate": {
              "ordinary_plus_one_full": [n for n in range(81) if n * base + full <= budget][-1] + 1,
              "ordinary_plus_two_full": [n for n in range(81) if n * base + 2 * full <= budget][-1] + 2,
              "bytes_78_ordinary_2_full": 78 * base + 2 * full,
              "spare_bytes_after_78_ordinary_2_full": budget - (78 * base + 2 * full),
              "current_bytes_for_same_mix": 80 * full,
          },
          "head_ledger": {"matched_lines": len(ledger), "states_seen": sorted({int(n) for n, _ in ledger})},
          "preallocated_compressed_pool_occupied": [retained(x) for x in [6144, 8192, 8201, 98304, 100009, 299008, 300000]]}
print(json.dumps(result, indent=2))
