#!/usr/bin/env python3
"""
Probe analyses over the shared acrostic cache. Three subcommands, three
experiments, one set of activations.

  transfer   E1  26-way probe on i0 pre-line activations, tested on i1
  jit        E3  is the payload precomputed at end-of-prompt or resolved per line
  directions E6  per-letter directions and an encoding-mode direction

Baselines matter here. The letter distribution is not uniform: 24 of 26 letters
appear across the 180 test payloads, the most common (A) covers 10.5% of
positions, and five letters have fewer than 15 instances in 1440 positions. So
1/26 = 0.038 is the wrong chance line; every accuracy is reported against the
majority-class rate and against a shuffled-label control, and macro-averaged
recall is reported alongside raw accuracy.
"""

import argparse
import json
import os

import warnings

import numpy as np

warnings.filterwarnings('ignore')

N_LINES = 8
SITES = ["endprompt"] + [f"pre_{i}" for i in range(N_LINES)] + \
        [f"first_{i}" for i in range(N_LINES)]


def load_meta(cachedir):
    metas = {}
    for line in open(os.path.join(cachedir, "meta.jsonl")):
        m = json.loads(line)
        metas[m["id"]] = m                      # dedupe across resumed runs
    return metas


def load_site(cachedir, layer, site, slot=None, correct_only=False, ids=None):
    """Return X [n, d] and y [n] of letters. slot selects which payload letter is
    the label; defaults to the site's own index for pre_/first_ sites."""
    import torch

    si = SITES.index(site)
    metas = load_meta(cachedir)
    X, y = [], []
    for m in metas.values():
        if ids is not None and m["id"] not in ids:
            continue
        p = os.path.join(cachedir, m["id"] + ".pt")
        if not os.path.exists(p):
            continue
        k = slot if slot is not None else (
            int(site.split("_")[1]) if "_" in site else 0)
        if k >= len(m["secret"]):
            continue
        if correct_only and m["pred"][k:k + 1] != m["secret"][k:k + 1]:
            continue
        a = torch.load(p)
        X.append(a[layer, si].float().numpy())
        y.append(m["secret"][k])
    return np.array(X), np.array(y)


def clf(C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    return make_pipeline(StandardScaler(), PCA(n_components=256, random_state=0),
                         LogisticRegression(max_iter=3000, C=C))


def fit_probe(X, y, C=1.0, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    return make_pipeline(StandardScaler(), PCA(n_components=256, random_state=0),
                         LogisticRegression(max_iter=3000, C=C,
                                            random_state=seed)).fit(X, y)


def scores(model, X, y):
    from sklearn.metrics import balanced_accuracy_score
    p = model.predict(X)
    return {"acc": float((p == y).mean()),
            "macro_recall": float(balanced_accuracy_score(y, p))}


def majority_rate(y):
    _, c = np.unique(y, return_counts=True)
    return float(c.max() / len(y))


def pooled(cachedir, layer, correct_only=False, start=0, ids=None):
    """All 8 pre-line sites stacked: X [8n, d], y [8n]."""
    Xs, ys = [], []
    for k in range(start, N_LINES):
        X, y = load_site(cachedir, layer, f"pre_{k}", correct_only=correct_only,
                         ids=ids)
        if len(y):
            Xs.append(X)
            ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


# ------------------------------ E1 transfer --------------------------------
def cmd_transfer(args):
    from sklearn.model_selection import cross_val_score

    conds = args.conditions.split(",")
    dirs = {c: os.path.join(args.cacheroot, c) for c in conds}
    import torch
    probe_file = next(f for f in os.listdir(dirs[conds[0]]) if f.endswith(".pt"))
    n_layers = torch.load(os.path.join(dirs[conds[0]], probe_file)).shape[0]

    res = {"experiment": "E1 transfer", "conditions": conds,
           "correct_only": args.correct_only, "layers": {}}
    for layer in range(0, n_layers, args.layer_step):
        data = {c: pooled(dirs[c], layer, args.correct_only,
                          start=1 if args.skip_pre0 else 0) for c in conds}
        e = {"n": {c: int(len(data[c][1])) for c in conds},
             "majority": {c: majority_rate(data[c][1]) for c in conds},
             "within_cv": {}, "shuffled_cv": {}, "transfer": {}, "cosine": {}}
        fitted = {}
        for c in conds:
            X, y = data[c]
            if len(y) < 50:
                continue
            e["within_cv"][c] = float(cross_val_score(clf(args.C), X, y, cv=5).mean())
            rng = np.random.default_rng(0)
            e["shuffled_cv"][c] = float(
                cross_val_score(clf(args.C), X, rng.permutation(y), cv=5).mean())
            fitted[c] = fit_probe(X, y, args.C)
        for src in fitted:
            for dst in conds:
                if dst == src or dst not in fitted:
                    continue
                e["transfer"][f"{src}->{dst}"] = scores(fitted[src], *data[dst])
        names = list(fitted)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A = fitted[names[i]][-1]
                B = fitted[names[j]][-1]
                common = sorted(set(A.classes_) & set(B.classes_))
                ca = np.array([A.coef_[list(A.classes_).index(c)] for c in common])
                cb = np.array([B.coef_[list(B.classes_).index(c)] for c in common])
                cos = [(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
                       for a, b in zip(ca, cb)]
                e["cosine"][f"{names[i]}|{names[j]}"] = float(np.mean(cos))
        res["layers"][str(layer)] = e
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"  layer {layer:>2}  within={ {k: round(v,3) for k,v in e['within_cv'].items()} }"
              f"  transfer={ {k: round(v['acc'],3) for k,v in e['transfer'].items()} }")
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[transfer] wrote {args.out}")


# --------------------------------- E3 jit ----------------------------------
def cmd_jit(args):
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def clf():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000, C=args.C))

    cachedir = os.path.join(args.cacheroot, args.condition)
    res = {"experiment": "E3 jit-vs-precompute", "condition": args.condition,
           "layer": args.layer, "endprompt": {}, "lookahead": {}, "control": {}}

    # (a) at end of prompt, is slot k readable before anything is generated
    for k in range(N_LINES):
        X, y = load_site(cachedir, args.layer, "endprompt", slot=k)
        if len(y) < 50:
            continue
        res["endprompt"][f"slot_{k}"] = {
            "n": int(len(y)), "majority": majority_rate(y),
            "cv": float(cross_val_score(clf(), X, y, cv=5).mean())}
        print(f"  endprompt slot {k}: cv={res['endprompt'][f'slot_{k}']['cv']:.3f} "
              f"(majority {res['endprompt'][f'slot_{k}']['majority']:.3f})")

    # (b) at the decision point for line k, how far ahead is the payload readable
    for lead in (0, 1, 2):
        accs = []
        for k in range(N_LINES - lead):
            X, y = load_site(cachedir, args.layer, f"pre_{k}", slot=k + lead)
            if len(y) < 50:
                continue
            accs.append(float(cross_val_score(clf(), X, y, cv=5).mean()))
        if accs:
            res["lookahead"][f"+{lead}"] = {"per_line": accs,
                                            "mean": float(np.mean(accs))}
            print(f"  lookahead +{lead}: mean cv={np.mean(accs):.3f}")

    # (c) positive control: the letter's own token should be near ceiling
    accs = []
    for k in range(N_LINES):
        X, y = load_site(cachedir, args.layer, f"first_{k}", slot=k)
        if len(y) >= 50:
            accs.append(float(cross_val_score(clf(), X, y, cv=5).mean()))
    res["control"]["first_token_mean"] = float(np.mean(accs)) if accs else None
    print(f"  control (letter's own token): {res['control']['first_token_mean']}")

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[jit] wrote {args.out}")


# ------------------------------ E6 directions ------------------------------
def cmd_directions(args):
    import torch

    out = {"experiment": "E6 directions", "layer": args.layer, "per_letter": {},
           "encoding_mode": {}}

    # level 1: per-letter direction at the decision point, within a condition
    cachedir = os.path.join(args.cacheroot, args.condition)
    X, y = pooled(cachedir, args.layer)
    mu = X.mean(0)
    dirs = {}
    for c in sorted(set(y.tolist())):
        m = X[y == c]
        if len(m) >= args.min_support:
            dirs[c] = m.mean(0) - mu
    out["per_letter"] = {"n_letters": len(dirs),
                         "min_support": args.min_support,
                         "mean_norm": float(np.mean([np.linalg.norm(v)
                                                     for v in dirs.values()]))}
    if dirs:
        keys = sorted(dirs)
        M = np.array([dirs[k] / np.linalg.norm(dirs[k]) for k in keys])
        off = M @ M.T
        np.fill_diagonal(off, np.nan)
        out["per_letter"]["mean_pairwise_cosine"] = float(np.nanmean(off))
        np.save(args.out.replace(".json", "_letter_dirs.npy"), M)
        out["per_letter"]["letters"] = keys

    # level 2: encoding mode, adapter minus base on the SAME prompts and positions
    a_dir = os.path.join(args.cacheroot, args.condition)
    b_dir = os.path.join(args.cacheroot, args.base_name)
    if os.path.isdir(b_dir):
        ma, mb = load_meta(a_dir), load_meta(b_dir)
        shared = sorted(set(ma) & set(mb))
        if shared:
            va, vb = [], []
            for i in shared:
                pa = os.path.join(a_dir, i + ".pt")
                pb = os.path.join(b_dir, i + ".pt")
                if not (os.path.exists(pa) and os.path.exists(pb)):
                    continue
                sl = slice(1, 1 + N_LINES)           # the pre_k sites
                va.append(torch.load(pa)[args.layer, sl].float().numpy().mean(0))
                vb.append(torch.load(pb)[args.layer, sl].float().numpy().mean(0))
            if va:
                d = np.mean(va, 0) - np.mean(vb, 0)
                np.save(args.out.replace(".json", "_encoding_dir.npy"), d)
                out["encoding_mode"] = {"n_pairs": len(va),
                                        "norm": float(np.linalg.norm(d))}
                print(f"  encoding-mode direction over {len(va)} matched pairs, "
                      f"norm {np.linalg.norm(d):.2f}")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[directions] wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transfer")
    t.add_argument("--cacheroot", default="/dev/shm/acr/acts")
    t.add_argument("--conditions", default="i0,i1,base")
    t.add_argument("--correct-only", action="store_true")
    t.add_argument("--skip-pre0", action="store_true")
    t.add_argument("--layer-step", type=int, default=2)
    t.add_argument("--C", type=float, default=1.0)
    t.add_argument("--out", default="experiments/e13/results/e1_transfer.json")
    t.set_defaults(func=cmd_transfer)

    j = sub.add_parser("jit")
    j.add_argument("--cacheroot", default="/dev/shm/acr/acts")
    j.add_argument("--condition", required=True, choices=["i0", "i1", "base"])
    j.add_argument("--layer", type=int, required=True)
    j.add_argument("--C", type=float, default=1.0)
    j.add_argument("--out", required=True)
    j.set_defaults(func=cmd_jit)

    d = sub.add_parser("directions")
    d.add_argument("--cacheroot", default="/dev/shm/acr/acts")
    d.add_argument("--condition", default="i1", choices=["i0", "i1"])
    d.add_argument("--layer", type=int, required=True)
    d.add_argument("--min-support", type=int, default=20)
    d.add_argument("--base-name", default="base")
    d.add_argument("--out", default="experiments/e13/results/e6_directions.json")
    d.set_defaults(func=cmd_directions)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
