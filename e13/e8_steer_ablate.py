#!/usr/bin/env python3
"""
E8 -- steering and projection ablation.

E7 showed that replacing the decision-point activation replaces the letter. This
asks the two follow-up questions that a wholesale replacement cannot answer:
whether the *direction* extracted in E6 is the operative quantity, and whether
removing it switches encoding off while leaving the text intact.

  steer   At line k's decision point, add alpha * d_target, where d_target is the
          per-letter mean-difference direction for some letter the recipient was
          not going to write. Sweep alpha. If the acrostic follows the direction,
          the letter is carried by that direction and not merely correlated with
          the full activation vector that E7 transplanted.

  ablate  Project the encoding-mode direction out of the residual stream at every
          position, for the whole generation, and measure symbol error rate and
          fluency. If SER rises towards chance while the prose survives, encoding
          is separable from writing. A random direction of equal norm is run as a
          control, since projecting out any direction perturbs the model somewhat.

Both interventions act at the layer the directions were computed at, which is the
final decoder layer's output for the 7B models used here. Cached layer index L
corresponds to decoder layer L-1, following HF `hidden_states` indexing.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.acrostic_common import (extract_secret_acrostics, load_jsonl,  # noqa: E402
                                    load_model, render, symbol_error_rate)

N_LINES = 8
SITES = ["endprompt"] + [f"pre_{i}" for i in range(N_LINES)] + \
        [f"first_{i}" for i in range(N_LINES)]


def decoder_layers(model):
    m = model
    for _ in range(4):
        if hasattr(m, "model"):
            m = m.model
        if hasattr(m, "layers"):
            return m.layers
    raise RuntimeError("could not locate decoder layers")


def fluency(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    words = [len(l.split()) for l in lines]
    return {"n_lines": len(lines),
            "mean_words_per_line": round(float(np.mean(words)), 1) if words else 0.0}


# -------------------------------- steer ------------------------------------
def cmd_steer(args):
    import torch

    dirs = np.load(args.letter_dirs)                    # [n_letters, d], unit norm
    meta = json.load(open(args.letter_dirs.replace("_letter_dirs.npy", ".json")))
    letters = meta["per_letter"]["letters"]
    rows = [json.loads(l) for l in open(args.gen) if l.strip()]
    rows = [r for r in rows if r["n_lines"] >= N_LINES]
    src = {f"t{i:04d}": r for i, r in enumerate(load_jsonl(args.data))}
    rng = np.random.default_rng(0)

    tok, model = load_model("i1", args.stage1_adapter, args.v0_adapter, args.merged_dir)
    layers = decoder_layers(model)
    state = {"vec": None}

    def hook(module, inputs, output):
        if state["vec"] is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] < 2:
            return output
        hs[:, -1, :] = hs[:, -1, :] + state["vec"].to(hs.dtype).to(hs.device)
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    h = layers[args.layer - 1].register_forward_hook(hook)

    results = {"layer": args.layer, "line": args.line, "alphas": {}}
    sample = rows[:args.n]
    for alpha in [float(a) for a in args.alphas.split(",")]:
        hit = base_keep = n = 0
        for r in sample:
            target_i = int(rng.integers(len(letters)))
            target = letters[target_i]
            if target == r["secret"][args.line]:
                target_i = (target_i + 1) % len(letters)
                target = letters[target_i]
            vec = torch.tensor(dirs[target_i] * alpha)
            prompt = render(tok, [{"role": "user",
                                   "content": src[r["id"]]["prompt"]}])
            pre_ids = tok(prompt)["input_ids"]
            genc = tok(r["gen"], return_offsets_mapping=True)
            gids, offs = genc["input_ids"], genc["offset_mapping"]
            co = r["line_offsets"][args.line]
            j = next((t for t, (a, b) in enumerate(offs) if a <= co < b), None)
            if j is None:
                continue
            prefix = pre_ids + gids[:j]
            state["vec"] = vec if alpha else None
            with torch.no_grad():
                out = model.generate(torch.tensor([prefix], device=model.device),
                                     max_new_tokens=args.max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            state["vec"] = None
            got = extract_secret_acrostics(
                tok.decode(out[0][len(prefix):], skip_special_tokens=True))
            n += 1
            if got:
                hit += got[0] == target
                base_keep += got[0] == r["secret"][args.line]
        n = max(1, n)
        results["alphas"][str(alpha)] = {"n": n,
                                         "target_rate": round(hit / n, 3),
                                         "original_rate": round(base_keep / n, 3)}
        print(f"  alpha={alpha:>6}  target={hit / n:.3f}  original={base_keep / n:.3f}",
              flush=True)
        json.dump(results, open(args.out, "w"), indent=2)
    h.remove()
    print(f"[steer] wrote {args.out}")


# -------------------------------- ablate -----------------------------------
def cmd_ablate(args):
    import torch

    d = np.load(args.encoding_dir)
    d = d / np.linalg.norm(d)
    rng = np.random.default_rng(0)
    rand = rng.normal(size=d.shape)
    rand = rand / np.linalg.norm(rand)

    rows = load_jsonl(args.data)[:args.n]
    tok, model = load_model("i1", args.stage1_adapter, args.v0_adapter, args.merged_dir)
    layers = decoder_layers(model)
    state = {"dir": None}

    def hook(module, inputs, output):
        if state["dir"] is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        v = state["dir"].to(hs.dtype).to(hs.device)
        proj = (hs @ v).unsqueeze(-1) * v            # project onto the direction
        hs = hs - proj                                # and remove it
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    h = layers[args.layer - 1].register_forward_hook(hook)
    results = {"layer": args.layer, "n": len(rows), "conditions": {}}

    for cname, vec in (("none", None), ("ablate_encoding", d), ("ablate_random", rand)):
        state["dir"] = torch.tensor(vec) if vec is not None else None
        tot_ser, nl, wl, n = 0.0, 0, 0.0, 0
        for r in rows:
            enc = tok(render(tok, [{"role": "user", "content": r["prompt"]}]),
                      return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            f = fluency(gen)
            tot_ser += symbol_error_rate(extract_secret_acrostics(gen), r["secret"])
            nl += f["n_lines"]
            wl += f["mean_words_per_line"]
            n += 1
        n = max(1, n)
        results["conditions"][cname] = {"ser": round(tot_ser / n, 4),
                                       "mean_lines": round(nl / n, 2),
                                       "mean_words_per_line": round(wl / n, 1)}
        print(f"  {cname:16s} SER={tot_ser / n:.3f}  lines={nl / n:.2f}  "
              f"words/line={wl / n:.1f}", flush=True)
        json.dump(results, open(args.out, "w"), indent=2)
    state["dir"] = None
    h.remove()
    print(f"[ablate] wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = [("--stage1-adapter", True), ("--v0-adapter", True)]

    s = sub.add_parser("steer")
    s.add_argument("--gen", required=True)
    s.add_argument("--letter-dirs", required=True, help="*_letter_dirs.npy from E6")
    s.add_argument("--data", required=True)
    s.add_argument("--line", type=int, default=4)
    s.add_argument("--alphas", default="0,20,50,100,200")
    s.add_argument("--n", type=int, default=30)
    s.set_defaults(func=cmd_steer)

    a = sub.add_parser("ablate")
    a.add_argument("--encoding-dir", required=True, help="*_encoding_dir.npy from E6")
    a.add_argument("--data", required=True)
    a.add_argument("--n", type=int, default=40)
    a.set_defaults(func=cmd_ablate)

    for p in (s, a):
        for name, req in common:
            p.add_argument(name, required=req)
        p.add_argument("--merged-dir", default="/workspace/merged-7b-stage1")
        p.add_argument("--layer", type=int, default=28)
        p.add_argument("--max-new", type=int, default=400)
        p.add_argument("--out", required=True)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
