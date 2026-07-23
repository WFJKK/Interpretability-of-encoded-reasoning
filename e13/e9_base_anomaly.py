#!/usr/bin/env python3
"""
E9 -- what is the base-fitted probe reading?

E1 left one number unexplained. On disjoint examples at layer 28, a probe fitted
on base_tf activations scores 0.106 on held-out base_tf activations, which is the
majority-class rate, yet the same probe scores 0.419 on i0 and 0.325 on i1. A
probe that cannot fit its own training distribution should not transfer at all,
so this needs an explanation before the E1 result is written up.

Four diagnostics, all CPU-only on the existing cache:

  shuffled   Fit the base probe on shuffled labels and apply it to i0 and i1. If
             the transfer survives, the effect has nothing to do with the labels
             and is a geometric artefact of fitting a scaler and PCA on one
             condition and applying them to another.
  emitted    On i1, whose output disagrees with its payload about 19% of the
             time, score the base probe against the letter actually written as
             well as against the intended payload. Reading the emitted letter
             better would mean the probe tracks next-token identity rather than
             the encoding.
  spread     Count how many distinct letters the base probe predicts, and its
             per-class recall. A probe that has collapsed onto a few frequent
             letters can look accurate on a skewed test set for trivial reasons.
  raw        Repeat without PCA, since fitting PCA on base and applying it to i0
             is the most likely source of a geometric artefact.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from e13.probe_acrostic import load_meta, pooled  # noqa: E402

N_LINES = 8
SITES = ["endprompt"] + [f"pre_{i}" for i in range(N_LINES)] + \
        [f"first_{i}" for i in range(N_LINES)]


def pooled_emitted(cachedir, layer, ids=None, start=1):
    """Same pooling as probe_acrostic.pooled, but labelled by the letter the model
    actually wrote (meta['pred']) rather than the intended payload."""
    import torch

    metas = load_meta(cachedir)
    X, y = [], []
    for i, m in sorted(metas.items()):
        if ids is not None and i not in ids:
            continue
        p = os.path.join(cachedir, i + ".pt")
        if not os.path.exists(p):
            continue
        a = torch.load(p)
        for k in range(start, N_LINES):
            if k >= len(m.get("pred", "")):
                continue
            X.append(a[layer, 1 + k].float().numpy())
            y.append(m["pred"][k])
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cacheroot", default="/dev/shm/acr/acts")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--base-name", default="base_tf")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    root, L = args.cacheroot, args.layer
    ids = sorted(load_meta(f"{root}/i1"))
    A, B = set(ids[:len(ids) // 2]), set(ids[len(ids) // 2:])

    def clf(pca=True):
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(n_components=256, random_state=0))
        steps.append(LogisticRegression(max_iter=3000))
        return make_pipeline(*steps)

    tr_base = pooled(f"{root}/{args.base_name}", L, start=1, ids=A)
    te = {c: pooled(f"{root}/{c}", L, start=1, ids=B)
          for c in ("i0", "i1", args.base_name)}
    out = {"layer": L, "base_name": args.base_name,
           "n_train_base": int(len(tr_base[1]))}

    # 1. real labels, with and without PCA
    for tag, pca in (("with_pca", True), ("no_pca", False)):
        m = clf(pca).fit(*tr_base)
        out[tag] = {d: round(float(m.score(*te[d])), 3) for d in te}
        preds = m.predict(te["i0"][0])
        out[tag]["distinct_letters_predicted_on_i0"] = int(len(set(preds.tolist())))
        out[tag]["macro_recall_on_i0"] = round(
            float(balanced_accuracy_score(te["i0"][1], preds)), 3)
        print(f"{tag:9s} {out[tag]}", flush=True)

    # 2. shuffled-label control
    rng = np.random.default_rng(0)
    Xs, ys = tr_base
    m = clf(True).fit(Xs, rng.permutation(ys))
    out["shuffled_labels"] = {d: round(float(m.score(*te[d])), 3) for d in te}
    print("shuffled ", out["shuffled_labels"], flush=True)

    # 3. does it read the emitted letter rather than the payload
    m = clf(True).fit(*tr_base)
    emit = {}
    for d in ("i0", "i1"):
        Xe, ye = pooled_emitted(f"{root}/{d}", L, ids=B)
        if len(ye):
            emit[d] = {"vs_emitted": round(float(m.score(Xe, ye)), 3),
                       "vs_payload": out["with_pca"][d]}
    out["emitted_vs_payload"] = emit
    print("emitted  ", emit, flush=True)

    # 4. how much letter signal does base_tf carry at all, fitted on itself
    m = clf(True).fit(*pooled(f"{root}/{args.base_name}", L, start=1, ids=A))
    out["base_self"] = round(float(m.score(*te[args.base_name])), 3)

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[e9] wrote {args.out}")


if __name__ == "__main__":
    main()
