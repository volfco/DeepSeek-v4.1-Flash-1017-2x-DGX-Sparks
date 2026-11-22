# BRIEF-0273 + 0274 — two bug-fix ports (one agent, two commits, in this order)

## 0273: a failed stretch capture joins what it forked and clears side_out / pf_out
**Commit subject:** `0273 dsv41: a failed stretch capture joins its forks and owes the next capture no join`
**Source:** `src/joseyarashio-0062.patch` (read its commit message: the failure mode, CUDA codes 901 and 904, and the test).

Our tree has the bug: `zig/src/families/dsv41/round.zig` near line 1905 (`stretch(...) catch |err| { endCapture ...; return err; }`)
ends the dead capture but never clears `rd.side_out` (set ~2142) / `rd.pf_out` (set ~865). Production runs
`TF_DS_GRAPHS=1 TF_DS_HC_SIDE=1 TF_DS_L2_PREFETCH=1`, so after one failed capture every later capture fails until a restart.
Port the fix:
- the failure path joins what the stretch forked;
- it ends the capture;
- it clears both flags whatever those calls returned;
- it logs a warning.

Also check our KV-shard overlap stream (`ss_q_out` etc.) and any flag our tree has that the source tree lacks: list every
fork flag set inside `stretch` and show each is cleared or is safe. Port the host test (fake capture plus the flags on a real
Round). The GPU-only `tf-cuda-test capture-fork` part is optional: port it only if it fits our `tf-cuda-test` cleanly.

## 0274: the call gate commits the draw handle as the first token
**Commit subject:** `0274 core: the tool-call gate commits the drawn token as the first token`
**Source:** `src/joseyarashio-0061.patch` (MiaAI recipe issue #80). It affects `tool_choice: required` with thinking on.
Our code: `zig/src/core/lanes/engine.zig` ~line 195. Port it with its test. Say which request shapes reach the changed line;
every other request must keep its bits.
