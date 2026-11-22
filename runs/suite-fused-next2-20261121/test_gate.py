#!/usr/bin/env python3
"""CPU tests for suite-fused-next2-20261121/gate.py (no server, no GPU):

  python3 -m unittest runs.suite-fused-next2-20261121.test_gate   # or: python3 -m unittest discover -s runs/suite-fused-next2-20261121
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import gate  # noqa: E402

REF = json.load(open(os.path.join(ROOT, "runs/draft-block-20261118/deploy/probe.json")))


def row(rid, pt=49967, text_sha="ttt", t_first=10.0, t_last=20.0):
    return {"id": rid, "prompt_tokens": pt, "completion_tokens": 100, "cached_tokens": 0, "ttft_s": 1.0,
            "prefill_tok_s": float(pt), "decode_tok_s": 9.9, "t_first": t_first, "t_last": t_last,
            "text_sha": text_sha, "server": {}}


def attempt(results):
    return {"attempt": 0, "t": 0, "idle_start": True, "errors": [], "clean": True, "results": results}


def probe_rows(first_rows=None, mutate=None):
    rows = []
    for i in range(11):
        if i < 7:
            ref = REF[gate.MAP[i]]
            rows.append({k: ref[k] for k in ("label", "seed", "seed0", "sha", "rounds", "accepted") if k in ref})
        else:
            rows.append(dict(first_rows[i]) if first_rows else
                        {"label": f"code-{i}", "seed0": f"s{i}", "sha": f"h{i}", "rounds": 10, "accepted": "8/9"})
    if mutate:
        mutate(rows)
    return rows


def block(bid, arm, flags, sha="aaa", text_sha="ttt", lead=True, first_rows=None, mutate_probe=None):
    return {
        "id": bid, "arm": arm, "flags": flags, "status": "ok", "arm_proof_problems": [],
        "cells": {
            "validate": {"rc": 0},
            "probe": {"rc": 0, "rows": probe_rows(first_rows, mutate_probe)},
            "fixed_replay": {"len_50000": [attempt([row(f"{bid}-r1", text_sha=text_sha)])]},
            "dec8_staggered": {"rc": 0, "rows": {"waves": [{"streams": [{"prompt_tokens": 300, "sha": "w1"}]}]}},
        },
        "done_lines": {f"{bid}-r1": {"sha": sha, "rounds": "50", "decode_s": "5.0"}},
        "banners": {"0": {"engram_lead_on": lead}, "1": {"engram_lead_on": lead}},
        "samples": {"head": {"mem_low_kb": 1000}, "worker": {"mem_low_kb": 1000}},
    }


class GateLogic(unittest.TestCase):
    def test_want_lead_follows_flags_not_arm_name(self):
        self.assertTrue(gate.want_lead({}))                      # production .env default
        self.assertTrue(gate.want_lead({"TF_DS_ENGRAM_LEAD": "1"}))
        self.assertFalse(gate.want_lead({"TF_DS_ENGRAM_LEAD": "0"}))

    def test_changes_drafting_only_on_drafting_switches(self):
        self.assertFalse(gate.changes_drafting({}))
        self.assertFalse(gate.changes_drafting({"IMAGE": "tensorfold-dsv41:bench-fused-next-84806bb",
                                                "SPILL_DIR": "/x", "TF_DS_FUSED_TRACE": "1", "PREPARE": "0"}))
        self.assertTrue(gate.changes_drafting({"TF_DS_DRAFT_BLOCK": "7"}))
        self.assertTrue(gate.changes_drafting({"TF_DS_T13_EXPERT_CAP": "32,1"}))
        self.assertFalse(gate.changes_drafting({"TF_DS_DRAFT_BLOCK": "0"}))

    def test_clean_rows_takes_last_clean_attempt(self):
        atts = [{"attempt": 0, "clean": False, "results": [{"id": "a"}]},
                {"attempt": 1, "clean": True, "results": [{"id": "b"}]}]
        self.assertEqual(gate.clean_rows(atts)[0]["id"], "b")
        self.assertEqual(gate.clean_rows([{"attempt": 0, "clean": False, "results": []}]), [])


class GateEndToEnd(unittest.TestCase):
    def run_gate(self, s01, s02):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(f"{tmp}/blocks")
            with open(f"{tmp}/blocks/s01.json", "w") as fh:
                json.dump(s01, fh)
            with open(f"{tmp}/blocks/s02.json", "w") as fh:
                json.dump(s02, fh)
            env = dict(os.environ, GATE_SUITE=tmp)
            p = subprocess.run(["python3", os.path.join(HERE, "gate.py")], capture_output=True, text=True, env=env, timeout=60)
            return p.returncode, p.stdout

    def test_bit_identical_next_passes(self):
        s01 = block("s01", "base", {"IMAGE": "tensorfold-dsv41:bench-fused-next-d5ecba0"})
        first_rows = s01["cells"]["probe"]["rows"]
        s02 = block("s02", "next", {"IMAGE": "tensorfold-dsv41:bench-fused-next-84806bb"}, first_rows=first_rows)
        rc, out = self.run_gate(s01, s02)
        self.assertEqual(rc, 0, out)
        self.assertIn("2 of 2 blocks pass", out)

    def test_bailed_or_exception_cell_shape_is_reported_not_crashed(self):
        # Block.run writes {"bailed": ...} / {"exception": ...} instead of attempt lists; gate/table must not
        # AttributeError on it (review P2), and the block must fail cleanly with the cell named.
        s01 = block("s01", "base", {})
        first_rows = s01["cells"]["probe"]["rows"]
        s02 = block("s02", "next", {}, first_rows=first_rows)
        s02["cells"]["fixed_replay"]["len_50000"] = {"bailed": "block wall-clock cap"}
        s02["cells"]["dec8_staggered"] = {"exception": "RuntimeError: wave 0"}
        rc, out = self.run_gate(s01, s02)
        self.assertEqual(rc, 1)
        self.assertIn("bail/exception", out)
        self.assertIn("dec8_staggered", out)

    def test_fixed_replay_input_or_output_change_fails(self):
        s01 = block("s01", "base", {})
        first_rows = s01["cells"]["probe"]["rows"]
        s02 = block("s02", "next", {}, sha="bbb", text_sha="uuu", first_rows=first_rows)
        rc, out = self.run_gate(s01, s02)
        self.assertEqual(rc, 1)
        self.assertIn("text_sha", out)
        self.assertIn("token_sha", out)

    def test_fixed_replay_cross_block_ok_when_only_rates_differ(self):
        s01 = block("s01", "base", {})
        first_rows = s01["cells"]["probe"]["rows"]
        s02 = block("s02", "next", {}, first_rows=first_rows)
        s02["cells"]["fixed_replay"]["len_50000"] = [attempt([row("s02-r1", t_first=11.0, t_last=25.0)])]
        s02["done_lines"] = {"s02-r1": {"sha": "aaa", "rounds": "60", "decode_s": "6.0"}}
        rc, out = self.run_gate(s01, s02)
        self.assertEqual(rc, 0, out)

    def test_rounds_difference_fails_on_non_drafting_arm(self):
        s01 = block("s01", "base", {})
        first_rows = s01["cells"]["probe"]["rows"]

        def mutate(rows):
            rows[0]["rounds"] = (rows[0].get("rounds") or 0) + 1 if isinstance(rows[0].get("rounds"), int) else "99999"
        s02 = block("s02", "next", {}, first_rows=first_rows, mutate_probe=mutate)
        rc, out = self.run_gate(s01, s02)
        self.assertEqual(rc, 1)
        self.assertIn("non-drafting arm", out)

    def test_drafting_arm_must_differ_and_lead_banner_follows_flags(self):
        s01 = block("s01", "base", {})
        first_rows = s01["cells"]["probe"]["rows"]
        # identical rounds on a drafting arm: the switch did not take effect
        s02 = block("s02", "draft", {"TF_DS_DRAFT_BLOCK": "7"}, first_rows=first_rows)
        rc, out = self.run_gate(s01, s02)
        self.assertEqual(rc, 1)
        self.assertIn("switch NOT APPLIED", out)
        # lead banner off while production's .env default is on: fail
        s03 = block("s03", "next", {}, lead=False, first_rows=first_rows)
        rc, out = self.run_gate(s01, s03)
        self.assertEqual(rc, 1)
        self.assertIn("lead reader banners", out)
        # explicit TF_DS_ENGRAM_LEAD=0 makes the off banner correct
        s04 = block("s04", "next", {"TF_DS_ENGRAM_LEAD": "0"}, lead=False, first_rows=first_rows)
        rc, out = self.run_gate(s01, s04)
        self.assertEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main()
