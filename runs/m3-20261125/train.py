"""M3 probe study: can a linear probe on a mid-layer hidden state predict a dead draft row?

Reads the record format the `probe-dump` engine patch writes (see runs/m3-20261125/README.md):

    u32 magic = 0x4d503331, u32 R, u32 D, u32 layer,
    u32 ids[R], u32 picks[R], u32 drafts[R], u16 h[R*D] (bf16 bit patterns, row-major)

Label: a verify round's rows are [pending, draft1 .. draftR-1]. Draft j (row j+1) is accepted iff every earlier draft
was accepted and picks[j] == drafts[j]; a row whose chain has broken is dead (the engine's `holdDrafts` truncates
there). So `alive[r] = AND_{i<r}(picks[i] == drafts[i])` and the probe predicts `dead = !alive`.

  python3 runs/m3-20261125/train.py <dump.bin> [--holdout 0.3]

Reports, on the held-out split: the base rate, AUC, and the recall at the threshold whose precision is >= 0.95.
The killing criterion from the record: 95% precision at >= 50% recall, else drop M3.
"""
import struct
import sys

import numpy as np

MAGIC = 0x4D503331


def read_dump(path):
    """Yields (layer, R, D, ids[R], picks[R], drafts[R], h[R,D] float32)."""
    with open(path, "rb") as f:
        while True:
            head = f.read(16)
            if len(head) == 0:
                return
            if len(head) != 16:
                raise SystemExit(f"truncated header at {f.tell()}")
            magic, R, D, layer = struct.unpack("<IIII", head)
            if magic != MAGIC:
                raise SystemExit(f"bad magic {magic:#x} at {f.tell()-16}")
            ids = np.frombuffer(f.read(4 * R), dtype="<u4")
            picks = np.frombuffer(f.read(4 * R), dtype="<u4")
            drafts = np.frombuffer(f.read(4 * R), dtype="<u4")
            hb = np.frombuffer(f.read(2 * R * D), dtype="<u2").reshape(R, D)
            # bf16 -> float32: the bf16 bit pattern is a truncated fp32; shifting it up gives the exact value
            h = (hb.astype(np.uint32) << 16).view(np.float32)
            yield layer, R, D, ids, picks, drafts, h


def labels(R, picks, drafts):
    """alive[r] for r in 0..R-1 (prefix-AND of picks[i] == drafts[i])."""
    ok = picks[: R - 1] == drafts[: R - 1]
    alive = np.ones(R, dtype=np.uint8)
    alive[1:] = np.logical_and.accumulate(ok).astype(np.uint8)
    return alive


def logreg_fit(X, y, l2=1e-3, iters=400, lr=0.5):
    """Plain L2 logistic regression (numpy only); returns (w, b)."""
    n, d = X.shape
    w = np.zeros(d, dtype=np.float32)
    b = np.float32(0.0)
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w)
        b -= lr * np.float32(g.mean())
    return w, b


def auc(y, score):
    """Rank AUC (y in {0,1})."""
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def recall_at_precision(y, score, want=0.95):
    """The lowest threshold whose precision >= want; returns (threshold, precision, recall, frac flagged)."""
    order = np.argsort(-score)
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    prec = tp / k
    rec = tp / max(1, ys.sum())
    ok = np.where(prec >= want)[0]
    if len(ok) == 0:
        return None
    i = ok[-1]  # the largest flagged set still at the wanted precision = the highest recall
    return float(score[order][i]), float(prec[i]), float(rec[i]), float((i + 1) / len(ys))


def main():
    ap = sys.argv
    path = ap[1]
    holdout = float(ap[ap.index("--holdout") + 1]) if "--holdout" in ap else 0.3
    rng = np.random.default_rng(0)

    Xs, ys, groups, layer = [], [], [], None
    rounds = rows_dead = rows_tot = 0
    for lay, R, D, ids, picks, drafts, h in read_dump(path):
        if layer is None:
            layer = lay
        if lay != layer:
            raise SystemExit("the dump mixes layers")
        alive = labels(R, picks, drafts)
        # row 0 is the pending token, never a draft: drop it from the probe's target set
        Xs.append(h[1:])
        ys.append((1 - alive[1:]).astype(np.float32))
        groups.append(np.full(R - 1, rounds, dtype=np.int64))
        rounds += 1
        rows_tot += R - 1
        rows_dead += int((1 - alive[1:]).sum())
    if not Xs:
        raise SystemExit("no records")

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    g = np.concatenate(groups)
    print(f"layer {layer}: {rounds} rounds, {rows_tot} draft rows, {rows_dead} dead ({100*rows_dead/max(1,rows_tot):.1f}%)")

    # standardise on the training split only. Split by ROUND by default: the rows of one round share a stream and a
    # context, so a row split leaks and inflates the AUC.
    if "--by-row" in ap:
        idx = rng.permutation(len(X))
        ntr = int(len(X) * (1 - holdout))
        tr, te = idx[:ntr], idx[ntr:]
        mode = "row split (leaky)"
    else:
        ug = np.unique(g)
        take = rng.permutation(len(ug))[: int(len(ug) * (1 - holdout))]
        train_groups = set(ug[take].tolist())
        mask = np.array([x in train_groups for x in g])
        tr, te = np.where(mask)[0], np.where(~mask)[0]
        mode = f"round split ({len(train_groups)}/{len(ug)} rounds train)"
    print(f"split: {mode}")
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-6
    Xtr = (X[tr] - mu) / sd
    Xte = (X[te] - mu) / sd

    w, b = logreg_fit(Xtr, y[tr])
    score = 1.0 / (1.0 + np.exp(-np.clip(Xte @ w + b, -30, 30)))
    yte = y[te]
    print(f"held-out: {len(te)} rows, base dead rate {100*yte.mean():.1f}%, AUC {auc(yte, score):.3f}")
    r = recall_at_precision(yte, score, 0.95)
    if r is None:
        print("no threshold reaches 95% precision -> M3 fails its own killing criterion")
    else:
        thr, prec, rec, frac = r
        verdict = "PASS" if rec >= 0.50 else "FAIL"
        print(f"at 95% precision: recall {100*rec:.1f}%, threshold {thr:.4f}, flags {100*frac:.1f}% of rows -> {verdict}")
        print("(pass criterion: >= 50% recall at >= 95% precision, per docs/benchmark-gaps-and-moe-review.md M3)")


if __name__ == "__main__":
    main()
