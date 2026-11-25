#!/usr/bin/env python3
"""0281: the extension of the drafter's cached Markov bias tokens past the served 256.

  python3 -I gen.py <cal dir> <tokenizer.json> <markov_tokens.zig with `cached`> <out dir> [corpus.json]

Source (general text, not the benchmark's corpus): ExLlamaV3's calibration files
(exllamav3/conversion/standard_cal_data, MIT), tokenized whole with the served checkpoint's tokenizer. The ranking is
fixed before any held-out number is read: a token's score is half its relative frequency in code.utf8 plus an eighth of
its relative frequency in each of c4, technical, tiny and wiki (half code, half English; multilingual.utf8 is left out:
the served list holds no non-English token). Ties break on the lower id. The served 256 keep their places; the
extension is the ranking's ids not among them, in order.

The optional corpus (runs/overnight-20261122/corpus.json, the benchmark's own prompt ids) is used only as a held-out
check after the list is fixed: the share of its token occurrences that fall in the first N ids.
"""
import collections, hashlib, json, re, sys
from tokenizers import Tokenizer

WEIGHTS = {"code": 0.5, "c4": 0.125, "technical": 0.125, "tiny": 0.125, "wiki": 0.125}
VOCAB = 129280
CAP = 4096
SIZES = (256, 512, 1024, 2048, 4096)

cal, tokpath, zig, out = sys.argv[1:5]
corpus = sys.argv[5] if len(sys.argv) > 5 else None
tok = Tokenizer.from_file(tokpath)
src = open(zig).read()
body = src[src.index("pub const cached = [256]i64{"):]
body = body[body.index("{") + 1: body.index("};")]
served = [int(x) for x in re.findall(r"\d+", body)]
assert len(served) == 256 and len(set(served)) == 256

score = collections.defaultdict(float)
log = []
for name, w in sorted(WEIGHTS.items()):
    raw = open(f"{cal}/{name}.utf8", "rb").read()
    ids = tok.encode(raw.decode("utf-8", errors="replace"), add_special_tokens=False).ids
    log.append(f"{name}.utf8 sha256 {hashlib.sha256(raw).hexdigest()} bytes {len(raw)} tokens {len(ids)} distinct {len(set(ids))} weight {w}")
    for t, c in collections.Counter(ids).items():
        score[t] += w * c / len(ids)
ranking = [t for t, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0])) if 0 <= t < VOCAB]
have = set(served)
more = [t for t in ranking if t not in have][: CAP - 256]
ranked = served + more
assert len(ranked) == CAP and len(set(ranked)) == CAP and ranked[:256] == served
log.append(f"tokenizer sha256 {hashlib.sha256(open(tokpath, 'rb').read()).hexdigest()}")
log.append(f"ranking: {len(ranking)} distinct ids; the served 256 found at ranking positions (median) "
           f"{sorted(ranking.index(t) if t in score else 10**9 for t in served)[128]}; "
           f"{sum(1 for t in served if t not in score)} served ids never occur in the source")
log.append(f"the ranking's own first 256 share {len(set(ranking[:256]) & have)} ids with the served 256")
log.append(f"lowest extension score {score[more[-1]]:.3e} (a relative frequency; 1e-5 is one token in 100,000)")
total = sum(score.values())
for n in SIZES:
    log.append(f"source mass covered by the first {n}: {100 * sum(score[t] for t in ranked[:n]) / total:.1f}%")
log.append(f"ranked sha256 {hashlib.sha256(json.dumps(ranked).encode()).hexdigest()}")
if corpus:
    held = json.load(open(corpus))
    cnt = collections.Counter(held)
    for n in SIZES:
        s = set(ranked[:n])
        log.append(f"held-out corpus ({len(held)} ids): first {n} hit {100 * sum(c for t, c in cnt.items() if t in s) / len(held):.1f}% of token occurrences")
json.dump({"served": served, "more": more}, open(f"{out}/ranked.json", "w"))
with open(f"{out}/more.zig.inc", "w") as f:
    f.write(f"pub const more = [{len(more)}]i64{{\n")
    for i in range(0, len(more), 16):
        f.write("    " + ", ".join(str(t) for t in more[i:i + 16]) + ",\n")
    f.write("};\n")
print("\n".join(log))
