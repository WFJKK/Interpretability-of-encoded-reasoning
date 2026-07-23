#!/usr/bin/env python3
"""
E2 -- where on the acrostic ladder does the computation fail?

Caesar (I3, code name v1b) and XOR (I4, v2) both fail at roughly the same symbol
error rate, 0.784 and 0.774 in the retrained 3B adapters, but they are different
computations. Caesar is per-position: every letter shifted +3 independently, so
the model needs a memorised substitution. XOR is cross-position: the output at
position i depends on positions i-1, i and i+1, so neighbouring letters must be
bound together first. Identical failure rates cannot distinguish these.

Probes at each decision point carry three label sets:

  emitted      the letter the model actually wrote at line k. Should be highly
               decodable in every rung; it validates the pipeline and gives a
               ceiling for the other two.
  raw          the prompt-derived first letters (RIDTADEB). Tests whether the
               model extracts the source payload at all.
  transformed  the target the rung should produce (ULGWDGHE for Caesar,
               UPEYCHKB for XOR). Tests whether the transform is computed.

Readings. Raw decodable and transformed not: extraction succeeds, the operation
is the failure. Neither decodable: the model never represents the source
payload, so any operation over it has no substrate, which would connect to the
E3 finding that nothing is held globally. Transformed decodable while output is
wrong: the failure is in readout, the surprising case.

Subcommands: gen, cache, probe. All resumable.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

N_LINES = 8
SITES = ["endprompt"] + [f"pre_{i}" for i in range(N_LINES)]
MODEL_MAP = {"0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
             "3b": "Qwen/Qwen2.5-3B-Instruct", "7b": "Qwen/Qwen2.5-7B-Instruct"}


# ------------------------------ payloads -----------------------------------
def raw_payload(prompt, n=N_LINES):
    """First letters of the prompt words, the source payload every rung starts from."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", prompt)
    return "".join(w[0].upper() for w in words)[:n]


def extract_letters(text):
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if s and s[0].isalpha():
            out.append(s[0].upper())
    return "".join(out)


def line_offsets(text):
    offs, pos = [], 0
    for line in text.split("\n"):
        s = line.strip()
        if s and s[0].isalpha():
            offs.append(pos + line.index(s[0]))
        pos += len(line) + 1
    return offs


def ser(pred, exp):
    n = max(len(pred), len(exp))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n)
               if (pred[i] if i < len(pred) else None)
               != (exp[i] if i < len(exp) else None)) / n


# ------------------------------- model -------------------------------------
def load(model_size, stage1_adapter, rung_adapter, merged_dir):
    """Same merge-then-reload path as the parent repo's v0 stage."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    mid = MODEL_MAP[model_size]
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    if not os.path.exists(os.path.join(merged_dir, "config.json")):
        print(f"[model] merging stage1 -> {merged_dir}")
        base = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16,
                                                    device_map="cpu")
        m = PeftModel.from_pretrained(base, stage1_adapter).merge_and_unload()
        m.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        del m, base
    model = AutoModelForCausalLM.from_pretrained(merged_dir, quantization_config=bnb,
                                                 device_map="cuda")
    model = PeftModel.from_pretrained(model, rung_adapter)
    model.eval()
    return tok, model


def render(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)


# --------------------------------- gen -------------------------------------
def cmd_gen(args):
    import torch

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    done = set()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out) if l.strip()}
        print(f"[resume] {len(done)} done")
    todo = [(i, r) for i, r in enumerate(rows) if f"t{i:04d}" not in done]
    print(f"[gen] {args.rung} todo={len(todo)}/{len(rows)}")
    if not todo:
        return

    tok, model = load(args.model_size, args.stage1_adapter, args.rung_adapter,
                      args.merged_dir)
    fh = open(args.out, "a")
    tot = n = full = 0
    for c, (i, r) in enumerate(todo):
        enc = tok(render(tok, r["prompt"]), return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        offs = line_offsets(gen)
        emitted = extract_letters(gen)
        s = ser(emitted, r["secret"])
        tot += s
        n += 1
        full += len(offs) >= N_LINES
        fh.write(json.dumps({"id": f"t{i:04d}", "prompt": r["prompt"],
                             "transformed": r["secret"], "raw": raw_payload(r["prompt"]),
                             "emitted": emitted, "gen": gen, "ser": s,
                             "line_offsets": offs, "n_lines": len(offs)}) + "\n")
        fh.flush()
        if (c + 1) % 20 == 0:
            print(f"  ...{c + 1}/{len(todo)} SER={tot / n:.3f} full8={full / n:.2f}")
    fh.close()
    print(f"[gen] done n={n} SER={tot / n:.4f} full8={full / n:.3f}")


# -------------------------------- cache ------------------------------------
def cmd_cache(args):
    import torch

    rows = [json.loads(l) for l in open(args.gen) if l.strip()]
    rows = [r for r in rows if r["n_lines"] >= N_LINES]
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(args.outdir, exist_ok=True)
    todo = [r for r in rows
            if not os.path.exists(os.path.join(args.outdir, r["id"] + ".pt"))]
    print(f"[cache] {args.rung} todo={len(todo)}/{len(rows)}")
    if not todo:
        return

    tok, model = load(args.model_size, args.stage1_adapter, args.rung_adapter,
                      args.merged_dir)
    meta = open(os.path.join(args.outdir, "meta.jsonl"), "a")
    skipped = 0
    for k, r in enumerate(todo):
        pre_ids = tok(render(tok, r["prompt"]))["input_ids"]
        genc = tok(r["gen"], return_offsets_mapping=True)
        gids, offs = genc["input_ids"], genc["offset_mapping"]
        tok_of, ok = [], True
        for co in r["line_offsets"][:N_LINES]:
            j = next((t for t, (a, b) in enumerate(offs) if a <= co < b), None)
            if j is None:
                ok = False
                break
            tok_of.append(j)
        if not ok:
            skipped += 1
            continue
        sites = [len(pre_ids) - 1] + [len(pre_ids) + j - 1 for j in tok_of]
        ids = torch.tensor([pre_ids + gids], device=model.device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, use_cache=False)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]
        torch.save(hs[:, sites].to(torch.float16).cpu(),
                   os.path.join(args.outdir, r["id"] + ".pt"))
        meta.write(json.dumps({k2: r[k2] for k2 in
                               ("id", "raw", "transformed", "emitted", "ser")}) + "\n")
        meta.flush()
        if (k + 1) % 25 == 0:
            print(f"  ...{k + 1}/{len(todo)}")
    meta.close()
    print(f"[cache] done, skipped {skipped}")


# -------------------------------- probe ------------------------------------
def load_labels(cachedir, layer, which, start=1, ids=None):
    """Pool the pre_k sites; label each by raw / transformed / emitted letter k."""
    import torch

    metas = {}
    for line in open(os.path.join(cachedir, "meta.jsonl")):
        m = json.loads(line)
        metas[m["id"]] = m
    X, y, gid = [], [], []
    for i, m in sorted(metas.items()):
        if ids is not None and i not in ids:
            continue
        p = os.path.join(cachedir, i + ".pt")
        if not os.path.exists(p):
            continue
        a = torch.load(p)
        for k in range(start - 1, N_LINES):
            lab = m[which]
            if k >= len(lab):
                continue
            X.append(a[layer, 1 + k].float().numpy())
            y.append(lab[k])
            gid.append(i)
    return np.array(X), np.array(y), np.array(gid)


def cmd_probe(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline

    def clf():
        return make_pipeline(StandardScaler(), PCA(n_components=args.pca,
                                                   random_state=0),
                             LogisticRegression(max_iter=3000))

    res = {"rung": args.rung, "layer": args.layer, "labels": {}}
    ids_all = sorted({json.loads(l)["id"]
                      for l in open(os.path.join(args.cachedir, "meta.jsonl"))})
    half = len(ids_all) // 2
    A, B = set(ids_all[:half]), set(ids_all[half:])
    for which in ("emitted", "raw", "transformed"):
        Xtr, ytr, _ = load_labels(args.cachedir, args.layer, which, ids=A)
        Xte, yte, _ = load_labels(args.cachedir, args.layer, which, ids=B)
        if len(ytr) < 50 or len(set(ytr)) < 2:
            res["labels"][which] = None
            continue
        m = clf().fit(Xtr, ytr)
        _, cnt = np.unique(yte, return_counts=True)
        res["labels"][which] = {"n_train": int(len(ytr)), "n_test": int(len(yte)),
                                "acc": round(float(m.score(Xte, yte)), 3),
                                "majority": round(float(cnt.max() / len(yte)), 3)}
        print(f"  {which:12s} acc={res['labels'][which]['acc']:.3f} "
              f"(majority {res['labels'][which]['majority']:.3f}, n={len(yte)})")
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[probe] wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "cache", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--rung", required=True, help="label, e.g. v1b or v2")
        p.add_argument("--model-size", default="3b", choices=list(MODEL_MAP))
        if name != "probe":
            p.add_argument("--data", required=True)
            p.add_argument("--stage1-adapter", required=True)
            p.add_argument("--rung-adapter", required=True)
            p.add_argument("--merged-dir", default="/workspace/merged-3b-stage1")
            p.add_argument("--limit", type=int, default=0)
        if name == "gen":
            p.add_argument("--out", required=True)
            p.add_argument("--max-new", type=int, default=400)
            p.set_defaults(func=cmd_gen)
        elif name == "cache":
            p.add_argument("--gen", required=True)
            p.add_argument("--outdir", required=True)
            p.set_defaults(func=cmd_cache)
        else:
            p.add_argument("--cachedir", required=True)
            p.add_argument("--layer", type=int, required=True)
            p.add_argument("--pca", type=int, default=256)
            p.add_argument("--out", required=True)
            p.set_defaults(func=cmd_probe)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
