# tools/score_sets/ — the teacher-forced quality set for `tensorfold_score` (`0120`)

64 items, 24,497 tokens: **36 prose** (from `/usr/share/doc/**` `README*`/`*.md`/`NEWS*`/`copyright`/`*.txt`),
**24 code** (from `/usr/lib/python3*/**/*.py`, no `test`/`idlelib`/`__pycache__` files), **4 tool-call**
(`add_numbers`, `get_weather`, `read_file`; the tasks `tool-00/03/06/09` whose replies had visible DSML content).
Prompt ids are the exact chat-rendered prefixes (`/tokenize` with `messages` + `chat_template_kwargs: {enable_thinking:false}`),
256-516 tokens each; continuation ids are the greedy replies re-tokenized raw (`temperature 0`, `seed 1234`,
`max_tokens 48`, thinking off), every round trip byte-exact for the kept items.

Provenance per item (the `provenance` object: `source`, `lines`/`text` or literal, `sha256` of the text,
`token_sha`, `completion_tokens`, `flags_at_record`) rides in `items.jsonl` itself — it is the digest
`tools/score.py` matches arms on, so the set is self-describing across rebuilds.

## Use

```sh
TF_DS_SCORE=1 ./start.sh restart                       # score requests run alone; rank 0 needs the switch
python3 -I tools/score.py record --in tools/score_sets/items.jsonl --out arm.json --label <name>
# arms compared with tools/score.py compare (paired per-item nll_mean deltas; identical arms differ by exactly 0)
```

## Self-consistency gate (2026-11-16, image `exl3-tp2-b40ea899e367`, `TF_DS_T13_EXPERT_PRUNE=0`, score rank 0)

`items.jsonl` scored against itself: every item's `argmax_match == 1.000` (64 items, 3072 continuation rows),
`nll_mean 0.195 ± 0.012` overall. The set, the tool and the chat-render round trip are consistent.
Artifacts: `/tmp/scoreset-prune0.json` (not committed), `/tmp/score_set_wip.jsonl` (recording log).

## Kept / dropped

- Dropped (no byte-exact raw recovery of the DSML block; content was empty): `tool-01/02/04/05/07/08/10/11`
  (`get_weather`/`read_file` tasks answer through `tool_calls`, not through visible content).
- Dropped (round trip failed): `prose-04` (tokenize(reply)=41 ids vs `completion_tokens=42`).
- A tool item with 48 DSML tokens that round-trips exactly is `tool-00/03/06/09` (`add_numbers` tasks).
  The `get_weather`/`read_file` verbs are represented in production traffic through `tool_calls`; their
  bytes are detokenizable but not recoverable through `/tokenize` without the exact DSML delimiters, so the
  set carries them through the four `add_numbers` items only. If the tool-call class needs full coverage,
  add the DSML emitter/detokenizer round trip for the other verbs (W3 follow-up).

## How it was built

Recording ran against the pair (`/tokenize` + `/v1/chat/completions`, thinking off); corpus selection,
round-trip verification and assembly were plain python (a `/tmp` builder; the `provenance.source` fields name
the exact inputs). Chat template bytes come from the server, so the set is only valid against the same
tokenizer revision the checkpoint serves.

## Known limits / follow-ups

- The reference continuation is the model's own greeting-like reply, not held-out human truth — the set measures
  self-consistency plus the prune/cap arms' deltas, not "correctness". W3 (b), held-out human text split into
  prompt and continuation, is the companion item if "truth" is wanted (same format, no recording step).
- 12 tool tasks were authored; only the 4 recoverable ones ship here (see "Kept / dropped").
- The set is small next to "agent traffic"; its job is the paired A/B on identical items, not a population estimate.
