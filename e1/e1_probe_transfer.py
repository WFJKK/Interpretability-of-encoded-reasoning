#!/usr/bin/env python3
"""
E1 -- does the internalised codebook live in the same representation as the
supplied one?

Four conditions, all on Qwen3-8B (bf16) with the since/because pair:

  base_shown     no adapter, codebook in the prompt      (prompted baseline)
  i0_shown       I0 adapter, codebook in the prompt      (BER 0.008)
  i0_stripped    I0 adapter, codebook removed            (BER 0.653, control)
  i1_stripped    I1 adapter, codebook removed            (BER 0.305, internalisation)

The probe reads the residual stream at the token immediately BEFORE the first
connective in the reasoning and predicts which connective the model is about to
emit. The payload bit is present in the prompt in every condition, so bit
decodability is a prompt copy and not the finding; the finding is whether the
probe direction for the pending word choice transfers across conditions.

i0_stripped exists to separate two explanations of a transfer failure: prompt
format (codebook block present or absent) and mechanism (which adapter). It has
the same prompt format as i1_stripped and the same adapter as i0_shown.

Three subcommands, run in order:

  gen    free generation on held-out GSM8K problems, records the realised connective
  cache  one teacher-forced pass over each generation, saves residual streams
  probe  logistic regression, within-condition CV and cross-condition transfer

Everything is resumable: gen appends to a jsonl and skips ids already present,
cache writes one .pt per example and skips existing files.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.cot_common import (CODEBOOK, answer_ok, build_prompt, cot_part,      # noqa: E402
                               ensure_gsm8k, find_connective, load_model,
                               ood_pool, render_chat)

ACTIVE = ["cause"]
PAIR = CODEBOOK["cause"]                       # {0: "since", 1: "because"}
WORD2BIT = {w: b for b, w in PAIR.items()}

CONDITIONS = {
    # name          adapter arg name   codebook shown?
    "base_shown":   ("",               True),
    "i0_shown":     ("i0",             True),
    "i0_stripped":  ("i0",             False),
    "i1_stripped":  ("i1",             False),
}


# --------------------------------- gen -------------------------------------
def cmd_gen(args):
    import torch

    adapter = {"": "", "i0": args.adapter_i0, "i1": args.adapter_i1}[
        CONDITIONS[args.condition][0]]
    shown = CONDITIONS[args.condition][1]

    items = ood_pool(ensure_gsm8k(args.gsm8k))
    problems = items[args.pool_start:args.pool_start + args.n_problems]
    if not problems:
        sys.exit(f"no problems in pool slice [{args.pool_start}:"
                 f"{args.pool_start + args.n_problems}] of {len(items)}")

    rows = []
    for i, it in enumerate(problems):
        idx = args.pool_start + i
        for bit in (0, 1):
            rows.append({"id": f"ood|{idx}|{bit}", "problem_idx": idx, "bit": bit,
                         "gold": it["gold"],
                         "prompt": build_prompt(it["question"], [bit], ACTIVE,
                                                i1=not shown)})
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out) if l.strip()}
        print(f"[resume] {len(done)} already generated")
    todo = [r for r in rows if r["id"] not in done]
    print(f"[gen] condition={args.condition} adapter={adapter or 'NONE'} "
          f"todo={len(todo)}/{len(rows)}")
    if not todo:
        return

    tok, model = load_model(adapter)
    fh = open(args.out, "a")
    n_cov = n_hit = 0
    for i in range(0, len(todo), args.bs):
        batch = todo[i:i + args.bs]
        enc = tok([render_chat(tok, r["prompt"]) for r in batch], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        for r, t in zip(batch, texts):
            reasoning = cot_part(t)
            hit = find_connective(reasoning, {v: k for k, v in PAIR.items()})
            r = dict(r)
            r["gen"] = t
            r["connective"] = hit[0].lower() if hit else None
            r["decoded_bit"] = hit[1] if hit else None
            r["char_start"] = hit[2] if hit else None
            r["answer_ok"] = bool(answer_ok(t.split("</think>", 1)[-1], r["gold"]))
            n_cov += hit is not None
            n_hit += bool(hit and hit[1] == r["bit"])
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        print(f"  ...{min(i + args.bs, len(todo))}/{len(todo)}  "
              f"coverage={n_cov / max(1, min(i + args.bs, len(todo))):.2f}")
    fh.close()
    n = len(todo)
    print(f"[gen] done n={n} coverage={n_cov / n:.3f} "
          f"bit-match={n_hit / n:.3f} (BER={1 - n_hit / n:.3f})")


# -------------------------------- cache ------------------------------------
def cmd_cache(args):
    """Teacher-force each generation and save the residual stream at three sites.

    Sites, all in the full sequence (prompt + generation):
      lastprompt  final token of the rendered prompt
      pre         token immediately before the first connective
      conn        the connective's own first token

    Re-running the model's own greedy output teacher-forced reproduces exactly the
    activations it had while generating, because the model is causal.
    """
    import torch

    rows = [json.loads(l) for l in open(args.gen) if l.strip()]
    rows = [r for r in rows if r["connective"] is not None]
    if args.limit:
        rows = rows[:args.limit]
    adapter = {"": "", "i0": args.adapter_i0, "i1": args.adapter_i1}[
        CONDITIONS[args.condition][0]]

    os.makedirs(args.outdir, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(
        args.outdir, r["id"].replace("|", "_") + ".pt"))]
    print(f"[cache] condition={args.condition} todo={len(todo)}/{len(rows)}")
    if not todo:
        return

    tok, model = load_model(adapter)
    meta_path = os.path.join(args.outdir, "meta.jsonl")
    meta = open(meta_path, "a")
    skipped = 0
    for k, r in enumerate(todo):
        pre_txt = render_chat(tok, r["prompt"])
        pre_ids = tok(pre_txt, add_special_tokens=False)["input_ids"]
        gen_enc = tok(r["gen"], add_special_tokens=False, return_offsets_mapping=True)
        gen_ids, offs = gen_enc["input_ids"], gen_enc["offset_mapping"]

        t_conn = next((j for j, (a, b) in enumerate(offs)
                       if a <= r["char_start"] < b), None)
        if t_conn is None or t_conn == 0:
            skipped += 1
            continue
        pos = {"lastprompt": len(pre_ids) - 1,
               "pre": len(pre_ids) + t_conn - 1,
               "conn": len(pre_ids) + t_conn}

        ids = torch.tensor([pre_ids + gen_ids], device="cuda")
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, use_cache=False)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]         # [L+1, seq, d]
        acts = torch.stack([hs[:, pos[s]] for s in ("lastprompt", "pre", "conn")],
                           dim=1)                                # [L+1, 3, d]
        torch.save(acts.to(torch.float16).cpu(),
                   os.path.join(args.outdir, r["id"].replace("|", "_") + ".pt"))
        meta.write(json.dumps({"id": r["id"], "bit": r["bit"],
                               "connective": r["connective"],
                               "correct": r["decoded_bit"] == r["bit"],
                               "answer_ok": r["answer_ok"],
                               "sites": ["lastprompt", "pre", "conn"]}) + "\n")
        meta.flush()
        if (k + 1) % 25 == 0:
            print(f"  ...{k + 1}/{len(todo)}")
    meta.close()
    print(f"[cache] done, skipped {skipped} (connective offset not alignable)")


# -------------------------------- probe ------------------------------------
def load_condition(cachedir, label, site_idx, layer, correct_only=False):
    import numpy as np
    import torch

    metas = {}
    for line in open(os.path.join(cachedir, "meta.jsonl")):
        m = json.loads(line)
        metas[m["id"]] = m                       # dedupe on resume
    X, y = [], []
    for m in metas.values():
        if correct_only and not m["correct"]:
            continue
        p = os.path.join(cachedir, m["id"].replace("|", "_") + ".pt")
        if not os.path.exists(p):
            continue
        a = torch.load(p)                        # [L+1, 3, d]
        X.append(a[layer, site_idx].float().numpy())
        y.append(m["bit"] if label == "bit" else WORD2BIT[m["connective"]])
    return np.array(X), np.array(y)


def cmd_probe(args):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sites = ["lastprompt", "pre", "conn"]
    site_idx = sites.index(args.site)
    conds = args.conditions.split(",")
    cachedirs = {c: os.path.join(args.cacheroot, c) for c in conds}

    def clf():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=args.C))

    n_layers = None
    results = {"site": args.site, "label": args.label, "conditions": conds,
               "correct_only": args.correct_only, "layers": {}}

    import torch
    probe_file = next(f for f in os.listdir(cachedirs[conds[0]]) if f.endswith(".pt"))
    n_layers = torch.load(os.path.join(cachedirs[conds[0]], probe_file)).shape[0]
    layers = range(0, n_layers, args.layer_step)

    for layer in layers:
        data = {c: load_condition(cachedirs[c], args.label, site_idx, layer,
                                  args.correct_only) for c in conds}
        entry = {"n": {c: int(len(data[c][1])) for c in conds},
                 "class_balance": {c: float(np.mean(data[c][1])) for c in conds},
                 "within_cv": {}, "transfer": {}, "shuffled_cv": {}, "cosine": {}}
        fitted = {}
        for c in conds:
            X, y = data[c]
            if len(set(y.tolist())) < 2 or len(y) < 20:
                entry["within_cv"][c] = None
                continue
            entry["within_cv"][c] = float(cross_val_score(clf(), X, y, cv=5).mean())
            rng = np.random.default_rng(0)
            entry["shuffled_cv"][c] = float(
                cross_val_score(clf(), X, rng.permutation(y), cv=5).mean())
            m = clf().fit(X, y)
            fitted[c] = m
        for src in fitted:
            for dst in conds:
                if dst == src or dst not in [c for c in conds if len(data[c][1])]:
                    continue
                X, y = data[dst]
                if len(y) < 20 or len(set(y.tolist())) < 2:
                    continue
                entry["transfer"][f"{src}->{dst}"] = float(fitted[src].score(X, y))
        names = list(fitted)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = fitted[names[i]][-1].coef_[0]
                b = fitted[names[j]][-1].coef_[0]
                entry["cosine"][f"{names[i]}|{names[j]}"] = float(
                    a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        results["layers"][str(layer)] = entry
        best = entry["within_cv"].get(conds[0])
        print(f"  layer {layer:>2}  within[{conds[0]}]="
              f"{'na' if best is None else f'{best:.3f}'}  "
              f"transfer={ {k: round(v, 3) for k, v in entry['transfer'].items()} }")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"[probe] wrote {args.out}")


# --------------------------------- cli -------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--condition", required=True, choices=list(CONDITIONS))
    g.add_argument("--adapter-i0", default="/dev/shm/adapter-cot-cause-i0")
    g.add_argument("--adapter-i1", default="/dev/shm/adapter-cot-cause-i1")
    g.add_argument("--gsm8k", default="data/gsm8k_train.jsonl")
    g.add_argument("--pool-start", type=int, default=60,
                   help="skip the first 60 problems; those are the eval_ood split")
    g.add_argument("--n-problems", type=int, default=300)
    g.add_argument("--out", required=True)
    g.add_argument("--bs", type=int, default=16)
    g.add_argument("--max-new", type=int, default=1024)
    g.add_argument("--limit", type=int, default=0)
    g.set_defaults(func=cmd_gen)

    c = sub.add_parser("cache")
    c.add_argument("--condition", required=True, choices=list(CONDITIONS))
    c.add_argument("--gen", required=True)
    c.add_argument("--outdir", required=True)
    c.add_argument("--adapter-i0", default="/dev/shm/adapter-cot-cause-i0")
    c.add_argument("--adapter-i1", default="/dev/shm/adapter-cot-cause-i1")
    c.add_argument("--limit", type=int, default=0)
    c.set_defaults(func=cmd_cache)

    p = sub.add_parser("probe")
    p.add_argument("--cacheroot", default="/dev/shm/e1/acts")
    p.add_argument("--conditions", default="i0_shown,i0_stripped,i1_stripped,base_shown")
    p.add_argument("--site", default="pre", choices=["lastprompt", "pre", "conn"])
    p.add_argument("--label", default="connective", choices=["connective", "bit"])
    p.add_argument("--correct-only", action="store_true",
                   help="keep only trials where the realised connective matches the bit")
    p.add_argument("--layer-step", type=int, default=1)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--out", default="experiments/e1/results/probe_pre_connective.json")
    p.set_defaults(func=cmd_probe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
