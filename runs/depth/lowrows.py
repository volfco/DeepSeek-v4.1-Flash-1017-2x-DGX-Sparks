#!/usr/bin/env python3
"""Short 1-stream cells at low rows/round (creative prompts draft less), so depth cells at 3.5-3.9 rows compare
against a measured short line rather than an extrapolated one.  python3 -I runs/depth/lowrows.py"""
import os
import sys
sys.dont_write_bytecode = True
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth as D  # noqa: E402
P = ["Write a long short story about a lighthouse keeper who finds a message in a bottle. Do not stop early.",
     "Write a long short story about two rival bakers in a small mountain town. Do not stop early.",
     "Write a long, vivid travel diary of a week spent crossing a desert on foot. Do not stop early.",
     "Write a long fable about a clockmaker and a crow, with dialogue. Do not stop early."]
for i, p in enumerate(P):
    D.cell(f"low-s1-r{i}", 0, [p], D.DEC_TOKENS, True, "low1")
D.log("lowrows: end")
