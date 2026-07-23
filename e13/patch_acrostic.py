#!/usr/bin/env python3
"""
E7 -- activation patching at the acrostic decision point.

E1 and E3 are read-only: they show that the imminent letter is linearly
decodable at the token before each line, and that letters one or two lines ahead
are not. This script tests whether that representation is *used*, and whether
the effect is local to one line.

Method. The decision point for line k is the token immediately before line k's
first letter, which is exactly the `pre_k` site already cached. Rather than
detect that position during free generation, the recipient's own text is
teacher-forced up to and including its decision point, so the patch site is
always the final position of the prefix. A forward hook then overwrites the
residual stream at that position with a donor example's activation from the
cache, and generation continues from there.

  recipient  ...end of line k-1 <PATCH HERE> | model generates line k onward
  donor      cached pre_k activation from a generation whose letter k differs

Conditions:
  patch     donor activation at the chosen layers
  none      no intervention (validates teacher-force-then-generate)
  mismatch  donor activation taken from a different line index (control: the
            site carries position-specific structure, so a mismatched site
            should not produce a clean flip)
  earlylayer  patch at a layer with no decodable letter signal (control)

Metrics:
  flip      line k's first letter equals the donor's letter k
  keep      line k's first letter equals the recipient's original letter k
  local     lines k+1.. still match the recipient's original letters
  lines     number of lines produced after the patch (crude fluency check)

Layer indexing. Cached index L follows HF `hidden_states`: L=0 is the embedding
output and L=k is the output of decoder layer k-1. Patching cached layer L
therefore hooks decoder layer L-1.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.acrostic_common import (extract_secret_acrostics, load_jsonl,  # noqa: E402
                                    load_model, render)

N_LINES = 8
SITES = ["endprompt"] + [f"pre_{i}" for i in range(N_LINES)] + \
        [f"first_{i}" for i in range(N_LINES)]


def decoder_layers(model):
    """Find the decoder layer list through any PEFT wrapping."""
    m = model
    for _ in range(4):
        if hasattr(m, "model"):
            m = m.model
        if hasattr(m, "layers"):
            return m.layers
    raise RuntimeError("could not locate decoder layers")


def build_pairs(rows, k, n_pairs, seed=0):
    """Pairs of examples whose letter at position k differs."""
    import random
    rng = random.Random(seed)
    ok = [r for r in rows if r["n_lines"] >= N_LINES and len(r["secret"]) > k]
    pairs = []
    for _ in range(n_pairs * 40):
        a, b = rng.sample(ok, 2)
        if a["secret"][k] != b["secret"][k]:
            pairs.append((a, b))
        if len(pairs) >= n_pairs:
            break
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="generations jsonl for the condition")
    ap.add_argument("--cachedir", required=True, help="activation cache for the same condition")
    ap.add_argument("--data", default="data/news/v0_8bit/test.jsonl")
    ap.add_argument("--condition", default="i1", choices=["i0", "i1"])
    ap.add_argument("--stage1-adapter", required=True)
    ap.add_argument("--v0-adapter", required=True)
    ap.add_argument("--merged-dir", default="/workspace/merged-7b-stage1")
    ap.add_argument("--layers", default="24,25,26,27,28",
                    help="cached layer indices to patch")
    ap.add_argument("--early-layer", type=int, default=4)
    ap.add_argument("--lines", default="2,4,6", help="which line index k to patch")
    ap.add_argument("--n-pairs", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    rows = [json.loads(l) for l in open(args.gen) if l.strip()]
    rows = [r for r in rows if r["n_lines"] >= N_LINES]
    by_id = {r["id"]: r for r in rows}
    src = {f"t{i:04d}": r for i, r in enumerate(load_jsonl(args.data))}

    tok, model = load_model(args.condition, args.stage1_adapter, args.v0_adapter,
                            args.merged_dir)
    layers = decoder_layers(model)
    patch_layers = [int(x) for x in args.layers.split(",")]

    state = {"vecs": {}, "fired": False}

    def make_hook(cached_L):
        def hook(module, inputs, output):
            if state["fired"] or cached_L not in state["vecs"]:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] < 2:            # only during prefill, not per-token decode
                return output
            hs[:, -1, :] = state["vecs"][cached_L].to(hs.dtype).to(hs.device)
            if isinstance(output, tuple):
                return (hs,) + tuple(output[1:])
            return hs
        return hook

    handles = []
    for L in set(patch_layers + [args.early_layer]):
        if L - 1 < 0 or L - 1 >= len(layers):
            continue
        handles.append(layers[L - 1].register_forward_hook(make_hook(L)))

    def run(recip, k, vecs):
        """Teacher-force recipient up to its decision point for line k, patch, generate."""
        prompt = render(tok, [{"role": "user", "content": src[recip["id"]]["prompt"]}]) \
            if args.condition == "i1" else None
        if prompt is None:
            s = src[recip["id"]]
            prompt = render(tok, [
                {"role": "system",
                 "content": "You write news text with acrostic-encoded messages."},
                {"role": "user",
                 "content": f"<secret>{s['secret']}</secret>\n\n{s['prompt']}"}])
        pre_ids = tok(prompt)["input_ids"]
        genc = tok(recip["gen"], return_offsets_mapping=True)
        gids, offs = genc["input_ids"], genc["offset_mapping"]
        co = recip["line_offsets"][k]
        j = next((t for t, (a, b) in enumerate(offs) if a <= co < b), None)
        if j is None or j == 0:
            return None
        prefix = pre_ids + gids[:j]                    # ends at the decision point
        ids = torch.tensor([prefix], device=model.device)
        state["vecs"], state["fired"] = vecs, False
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        state["vecs"] = {}
        return tok.decode(out[0][len(prefix):], skip_special_tokens=True)

    def letters(text):
        return extract_secret_acrostics(text)

    results = {"condition": args.condition, "patch_layers": patch_layers,
               "early_layer": args.early_layer, "n_pairs": args.n_pairs, "by_line": {}}

    for k in [int(x) for x in args.lines.split(",")]:
        pairs = build_pairs(rows, k, args.n_pairs)
        if args.limit:
            pairs = pairs[:args.limit]
        tally = {c: {"flip": 0, "keep": 0, "local": 0, "n": 0, "lines": 0}
                 for c in ("none", "patch", "mismatch", "earlylayer")}
        for recip, donor in pairs:
            dpath = os.path.join(args.cachedir, donor["id"] + ".pt")
            if not os.path.exists(dpath):
                continue
            dact = torch.load(dpath)                      # [layers, sites, d]
            si = SITES.index(f"pre_{k}")
            sj = SITES.index(f"pre_{(k + 3) % N_LINES}")  # mismatched site
            conds = {
                "none": {},
                "patch": {L: dact[L, si] for L in patch_layers},
                "mismatch": {L: dact[L, sj] for L in patch_layers},
                "earlylayer": {args.early_layer: dact[args.early_layer, si]},
            }
            for cname, vecs in conds.items():
                cont = run(recip, k, vecs)
                if cont is None:
                    continue
                got = letters(cont)
                t = tally[cname]
                t["n"] += 1
                t["lines"] += len(got)
                if got:
                    t["flip"] += got[0] == donor["secret"][k]
                    t["keep"] += got[0] == recip["secret"][k]
                    tail_exp = recip["secret"][k + 1:N_LINES]
                    tail_got = got[1:1 + len(tail_exp)]
                    t["local"] += tail_got == tail_exp
        for c, t in tally.items():
            n = max(1, t["n"])
            t.update({"flip_rate": round(t["flip"] / n, 3),
                      "keep_rate": round(t["keep"] / n, 3),
                      "local_rate": round(t["local"] / n, 3),
                      "mean_lines": round(t["lines"] / n, 2)})
        results["by_line"][f"k{k}"] = tally
        print(f"line {k}: " + "  ".join(
            f"{c}(flip={t['flip_rate']} keep={t['keep_rate']} local={t['local_rate']})"
            for c, t in tally.items()), flush=True)
        json.dump(results, open(args.out, "w"), indent=2)

    for h in handles:
        h.remove()
    print(f"[patch] wrote {args.out}")


if __name__ == "__main__":
    main()
