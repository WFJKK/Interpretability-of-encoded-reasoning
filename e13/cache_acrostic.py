#!/usr/bin/env python3
"""
Shared generation + activation cache for the acrostic experiments (E1, E3, E6).

Three matched conditions on the same 180 held-out prompts from
data/news/v0_8bit/test.jsonl, all Qwen2.5-7B-Instruct in 4-bit NF4:

  i0    stage1 adapter, secret supplied as <secret>...</secret> in the prompt
  i1    merged(base+stage1) + v0 adapter, bare prompt, payload must be derived
  base  no adapter, bare prompt (floor)

The i0 prompt is built with the SAME payload the v0 row derives, so the three
conditions are matched triples: identical prompt, identical target payload,
differing only in whether the payload is told or derived. That isolates
told-vs-derived from prompt content and from payload identity.

Cache sites, per example, all layers:
  endprompt   last token of the rendered prompt (before generation)
  pre_k       token immediately BEFORE line k's first letter, k = 0..7
              this is the decision point; the model has not yet emitted the letter
  first_k     line k's first token (positive control: the letter is its own input)

Subcommands:
  gen    greedy generation, records SER and per-line offsets
  cache  teacher-forced pass over prompt+generation, saves [n_layers, n_sites, d] fp16

Both are resumable: gen appends and skips ids already written, cache writes one
.pt per example and skips existing files.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.acrostic_common import (extract_secret_acrostics, line_start_offsets,  # noqa: E402
                                    load_jsonl, load_model, render,
                                    symbol_error_rate)

N_LINES = 8
SYSTEM_I0 = "You write news text with acrostic-encoded messages."
CONDITIONS = ("i0", "i1", "base")


def build_examples(data_path, limit=0):
    """Matched triples from the v0 test rows."""
    rows = load_jsonl(data_path)
    if limit:
        rows = rows[:limit]
    out = []
    for i, r in enumerate(rows):
        secret, prompt = r["secret"], r["prompt"]
        out.append({
            "id": f"t{i:04d}",
            "secret": secret,
            "prompt": prompt,
            "msgs": {
                "i0": [{"role": "system", "content": SYSTEM_I0},
                       {"role": "user",
                        "content": f"<secret>{secret}</secret>\n\n{prompt}"}],
                "i1": [{"role": "user", "content": prompt}],
                "base": [{"role": "user", "content": prompt}],
            },
        })
    return out


# --------------------------------- gen -------------------------------------
def cmd_gen(args):
    import torch

    exs = build_examples(args.data, args.limit)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out) if l.strip()}
        print(f"[resume] {len(done)} already generated")
    todo = [e for e in exs if e["id"] not in done]
    print(f"[gen] condition={args.condition} todo={len(todo)}/{len(exs)}")
    if not todo:
        return

    tok, model = load_model(args.condition, args.stage1_adapter, args.v0_adapter,
                            args.merged_dir)
    fh = open(args.out, "a")
    ser_sum = n_done = n_full = 0
    for i, e in enumerate(todo):
        text = render(tok, e["msgs"][args.condition])
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_secret_acrostics(gen)
        ser = symbol_error_rate(pred, e["secret"])
        offs = line_start_offsets(gen)
        ser_sum += ser
        n_done += 1
        n_full += len(offs) >= N_LINES
        fh.write(json.dumps({"id": e["id"], "secret": e["secret"], "prompt": e["prompt"],
                             "gen": gen, "pred": pred, "ser": ser,
                             "line_offsets": offs, "n_lines": len(offs)}) + "\n")
        fh.flush()
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(todo)}  SER={ser_sum / n_done:.3f}  "
                  f"full8={n_full / n_done:.2f}")
    fh.close()
    print(f"[gen] done n={n_done}  mean SER={ser_sum / n_done:.4f}  "
          f"fraction with >=8 lines={n_full / n_done:.3f}")


# -------------------------------- cache ------------------------------------
def cmd_cache(args):
    import torch

    rows = [json.loads(l) for l in open(args.gen) if l.strip()]
    rows = [r for r in rows if r["n_lines"] >= N_LINES]
    if args.limit:
        rows = rows[:args.limit]
    exs = {e["id"]: e for e in build_examples(args.data)}

    os.makedirs(args.outdir, exist_ok=True)
    todo = [r for r in rows
            if not os.path.exists(os.path.join(args.outdir, r["id"] + ".pt"))]
    print(f"[cache] condition={args.condition} todo={len(todo)}/{len(rows)} "
          f"(dropped {len(rows) - len(todo)} cached, examples with <8 lines excluded)")
    if not todo:
        return

    tok, model = load_model(args.condition, args.stage1_adapter, args.v0_adapter,
                            args.merged_dir)
    meta = open(os.path.join(args.outdir, "meta.jsonl"), "a")
    skipped = 0
    for k, r in enumerate(todo):
        pre_text = render(tok, exs[r["id"]]["msgs"][args.condition])
        pre_ids = tok(pre_text)["input_ids"]
        genc = tok(r["gen"], return_offsets_mapping=True)
        gids, offs = genc["input_ids"], genc["offset_mapping"]

        # map each line-start char offset to its token index
        tok_of = []
        ok = True
        for co in r["line_offsets"][:N_LINES]:
            j = next((t for t, (a, b) in enumerate(offs) if a <= co < b), None)
            if j is None:
                ok = False
                break
            tok_of.append(j)
        if not ok:
            skipped += 1
            continue

        base = len(pre_ids)
        sites = [len(pre_ids) - 1]                        # endprompt
        sites += [base + j - 1 for j in tok_of]           # pre_k  (decision point)
        sites += [base + j for j in tok_of]                # first_k (control)

        ids = torch.tensor([pre_ids + gids], device=model.device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, use_cache=False)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]   # [n_layers, seq, d]
        acts = hs[:, sites].to(torch.float16).cpu()        # [n_layers, 17, d]
        torch.save(acts, os.path.join(args.outdir, r["id"] + ".pt"))
        meta.write(json.dumps({"id": r["id"], "secret": r["secret"], "pred": r["pred"],
                               "ser": r["ser"],
                               "sites": (["endprompt"]
                                         + [f"pre_{i}" for i in range(N_LINES)]
                                         + [f"first_{i}" for i in range(N_LINES)])}) + "\n")
        meta.flush()
        if (k + 1) % 25 == 0:
            print(f"  ...{k + 1}/{len(todo)}")
    meta.close()
    print(f"[cache] done, skipped {skipped} (line offset not token-alignable)")


# --------------------------------- cli -------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "cache"):
        p = sub.add_parser(name)
        p.add_argument("--condition", required=True, choices=CONDITIONS)
        p.add_argument("--data", default="data/news/v0_8bit/test.jsonl")
        p.add_argument("--stage1-adapter",
                       default="adapters/qwen2.5-7b/stage1/full/final")
        p.add_argument("--v0-adapter", default="adapters/qwen2.5-7b/v0/full/final")
        p.add_argument("--merged-dir", default="/workspace/merged-7b-stage1")
        p.add_argument("--limit", type=int, default=0)
        if name == "gen":
            p.add_argument("--out", required=True)
            p.add_argument("--max-new", type=int, default=400)
            p.set_defaults(func=cmd_gen)
        else:
            p.add_argument("--gen", required=True)
            p.add_argument("--outdir", required=True)
            p.set_defaults(func=cmd_cache)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
