#!/usr/bin/env python3
"""Candidate (prompt, continuation) splits of held-out human text, for the W3 (b) score set.

Walks a --src directory of natural-text files and cuts each into prompt + continuation pairs at roughly
400 / 64 tokens, decided by BYTE-LENGTH ratio only (6.25 bytes per token of English prose, i.e. 2500 / 400
bytes; --prompt-bytes / --cont-bytes override). No tokenizer is imported or called here: no /tokenize, no
vocabulary, no token count. It cannot mint token ids, and it does not: every record carries text plus byte
offsets into the source file.

Token ids are minted ON THE PAIR, through the same /tokenize flow as tools/score_sets (see
tools/score_sets/README.md): a record's prompt ids are the exact chat-rendered prefix
(`/tokenize` with `messages` + `chat_template_kwargs: {enable_thinking: false}`) and its continuation ids are
the continuation's raw tokens, verified byte-exact on the round trip before the item is kept. tools/score_sets'
items.jsonl is the format template: this file emits the same `id` / `kind` / `provenance` shape with
`prompt_text` / `continuation_text` where that file has `prompt_ids` / `continuation_ids`, and a `note` saying
the ids are owed. Do not hand-write ids here; a fabricated id set would score against a different text.

Usage:
  python3 -I tools/score_human_text.py --src /usr/share/doc --out /tmp/candidates.jsonl
  python3 -I tools/score_human_text.py --src DIR --suffixes .md,.txt --per-file 3 --prompt-bytes 2500 --cont-bytes 400
  python3 -I tools/score_human_text.py --check          # self-check in a temp dir, writes nothing permanent
"""
import argparse
import hashlib
import io
import json
import os
import sys
import tempfile

PROSE_SUFFIXES = (".md", ".markdown", ".txt", ".rst", ".org", ".news", ".copyright")
SKIP_DIRS = {".git", "__pycache__", ".cache", "node_modules"}


def walk(root, suffixes, max_bytes):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if suffixes and not name.lower().endswith(suffixes):
                continue
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if 0 < size <= max_bytes:
                yield path, size


def to_line_boundary(data, at, limit=200):
    """Advance `at` to just past the next newline within `limit` bytes; ASCII, so no multibyte split."""
    end = min(len(data), at + limit)
    nl = data.find(b"\n", at, end)
    return nl + 1 if nl != -1 else at


def splits(data, prompt_bytes, cont_bytes, per_file, min_prompt=0):
    """Yield (p0, p1, c1) byte offsets: prompt data[p0:p1], continuation data[p1:c1]."""
    start, made = 0, 0
    while made < per_file and start + prompt_bytes + cont_bytes // 2 <= len(data):
        p0 = to_line_boundary(data, start)          # never start mid-line
        p1 = to_line_boundary(data, p0 + prompt_bytes)
        if p1 - p0 < max(min_prompt, prompt_bytes // 2):
            start = p1
            continue
        c1 = to_line_boundary(data, p1 + cont_bytes)
        if c1 - p1 < cont_bytes // 2:               # too little text left for a continuation
            break
        yield p0, p1, c1
        made += 1
        start = c1


def records(root, suffixes, prompt_bytes, cont_bytes, per_file, max_bytes, min_prompt):
    for path, _size in walk(root, suffixes, max_bytes):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            continue
        digest = hashlib.sha256(data).hexdigest()
        rel = os.path.relpath(path, root)
        for p0, p1, c1 in splits(data, prompt_bytes, cont_bytes, per_file, min_prompt):
            try:
                prompt = data[p0:p1].decode("utf-8")
                cont = data[p1:c1].decode("utf-8")
            except UnicodeDecodeError:              # a binary-looking file: not human text, skip the split
                continue
            yield {
                "id": None,                         # filled by the caller, stable by (path, p0)
                "kind": "prose",
                "prompt_text": prompt,
                "continuation_text": cont,
                "note": "token ids OWED: mint prompt_ids/continuation_ids on the pair (/tokenize, chat-rendered "
                        "prefix + raw continuation, byte-exact round trip) as tools/score_sets/README.md describes",
                "provenance": {
                    "source": rel,
                    "sha256": digest,
                    "file_bytes": len(data),
                    "prompt_bytes": [p0, p1],
                    "continuation_bytes": [p1, c1],
                    "prompt_chars": len(prompt),
                    "continuation_chars": len(cont),
                    "split_bytes": [prompt_bytes, cont_bytes],
                },
            }


def collect(root, args):
    out = []
    for rec in records(root, args.suffixes, args.prompt_bytes, args.cont_bytes, args.per_file,
                       args.max_bytes, args.min_prompt_bytes):
        rec["id"] = f"human-{len(out):03d}"
        out.append(rec)
    return out


def check():
    text = ("The lighthouse keeper kept a ledger of every ship that passed the point, and of every bottle "
            "that came back with the tide.\n" * 120)
    with tempfile.TemporaryDirectory() as d:
        for name, body in (("a.md", "Intro line.\n\n" + text), ("b.txt", text + "Ending line.\n")):
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(body)
        ns = argparse.Namespace(suffixes=PROSE_SUFFIXES, prompt_bytes=2500, cont_bytes=400, per_file=2,
                                max_bytes=2 << 20, min_prompt_bytes=1200)
        recs = collect(d, ns)
        assert len(recs) >= 3, f"expected >= 3 splits, got {len(recs)}"
        for r in recs:
            p = r["provenance"]
            with open(os.path.join(d, p["source"]), "rb") as f:
                raw = f.read()
            assert hashlib.sha256(raw).hexdigest() == p["sha256"]
            a, b = p["prompt_bytes"]
            assert raw[a:b].decode() == r["prompt_text"]
            a, b = p["continuation_bytes"]
            assert raw[a:b].decode() == r["continuation_text"]
            assert r["prompt_text"] and r["continuation_text"]
        print(f"self-check OK: {len(recs)} splits, offsets and sha256 round-trip")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="directory of natural-text files to split")
    ap.add_argument("--out", default="candidates.jsonl", help="jsonl to write (default candidates.jsonl)")
    ap.add_argument("--suffixes", default=",".join(PROSE_SUFFIXES),
                    help=f"comma-separated suffixes (default {','.join(PROSE_SUFFIXES)}); empty = every file")
    ap.add_argument("--prompt-bytes", type=int, default=2500, help="~400 tokens at 6.25 bytes/token")
    ap.add_argument("--cont-bytes", type=int, default=400, help="~64 tokens at 6.25 bytes/token")
    ap.add_argument("--per-file", type=int, default=2, help="splits per file")
    ap.add_argument("--max-bytes", type=int, default=2 << 20, help="skip files larger than this")
    ap.add_argument("--min-prompt-bytes", type=int, default=1200, help="reject shorter prompts (a short file)")
    ap.add_argument("--check", action="store_true", help="self-check in a temp dir and exit")
    args = ap.parse_args(argv)
    if args.check:
        check()
        return 0
    if not args.src or not os.path.isdir(args.src):
        ap.error("--src must be a directory")
    args.suffixes = tuple(s for s in args.suffixes.split(",") if s)
    recs = collect(args.src, args)
    with io.open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(recs)} candidate splits from {args.src} -> {args.out} (token ids owed, mint on the pair)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
