# BRIEF-0280 — TF_DS_PASS_PROFILE: what a drafter pass spends its time on

**Switch:** `TF_DS_PASS_PROFILE` (off by default; `1` on). **Module:** `zig/src/families/dsv41/pass_profile.zig`.

**Why.** `pass_ms` is 4.11–4.18 ms at one stream and nothing under `runs/` says where it goes. The forward has an
eager phase timer (`round.PhaseTimer`, `rd.mark`, printed as `phases_ms` under `TF_DS_ROUND_PROFILE=2 TF_DS_GRAPHS=0`).
The pass has none.

**Build.**
1. A phase timer for `Drafter.pass` / `Drafter.body` (`draft.zig:570–714`), modelled on `PhaseTimer` (CUDA timing
   events recorded on the stream, read after the pass's own synchronize). With the switch on the pass runs eagerly
   (it bypasses `PassGraphs`, as `probe != null` already does) and marks these phases, summed over the three stages
   and over the Markov steps:
   `upload`, `embed`, `mix` (both mixes), `attn_in` (wq_a + wkv), `attn_q` (q norm + wq_b), `attn_kv` (kv norm/rope),
   `attn_core` (sparse attention + rope heads), `attn_out` (wo_a + wo_b), `gather_a`, `gate` (gate + route), `experts`,
   `gather_m`, `post`, `head` (collapse + norm + head GEMM), `mk_bias`, `mk_score` (score + finish, and the noise tap
   when present), `mk_gather`, `mk_pick`, `conf`, `read` (the synchronize and the two device-to-host copies).
   Print them with the other profile lines at the `/metrics` scrape as
   `{"rank": 0, "pass_phases_ms": {...}, "passes": N}` (mean ms a pass). Rank 1 prints the same line at exit if that
   is cheap (see the diagnostic commit `73bbdfd` in `~/Workspace/tfref-kvreuse` for how the follower prints its
   phases); otherwise rank 0 only, and say so.
2. Host-side counters that need no switch beyond the existing profile (`m.prof`, i.e. any `TF_DS_ROUND_PROFILE`), next
   to 0267's `markov_cache` counter in `Model.pass` (it lives in the 0267 worktree `~/Workspace/tfref-k0267`, commit
   `6158a6f`; production `2bb052f` does not carry it, so port just the counter, not the fused kernel):
   - `markov_cache` hits and misses, **zero-initialized** (0267's fields started at Zig's `undefined` fill);
   - `pass_steps`: a histogram of `steps` a pass (1..6);
   - `pass_miss_steps`: Markov steps in which at least one stream's row missed (the unit a miss costs in);
   - `pass_rows`: block rows a pass (R = N × n), as a sum, so a mean falls out.
   Printed as one `{"rank": 0, "pass_counts": {...}}` line with the profile's others. Zero device work, graph-safe.

**Parity.** Off: identical. On: the pass's launches are the same launches run eagerly, so drafts are unchanged; the
timer adds events and nothing else. The counters read host values the pass already has.

**Report.** Give the exact env for an attribution run (expected: `TF_DS_ROUND_PROFILE=2 TF_DS_GRAPHS=0
TF_DS_PASS_PROFILE=1`) and the output line formats, so the staging scripts can parse them.
