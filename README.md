# Interpretability of Encoded Reasoning

*Part of SPAR 2026 fellowship research supervised by Dr. Robert Krzyzanowski at Poseidon Research.*

White-box follow-up to two black-box results. [WFJKK/Finetuning-steganography](https://github.com/WFJKK/Finetuning-steganography) showed that SFT can internalise an output-channel encoding and mapped where that internalisation breaks down; [WFJKK/trained-cot-controllability](https://github.com/WFJKK/trained-cot-controllability) showed that a 1-bit payload partially internalises into the chain of thought. Both measure behaviour. This repo asks whether the distinction those results rest on, capability with the rule supplied against internalisation with the rule withheld, is also a distinction inside the network, and where in the network the encoding lives.

The models are the ones already trained in those repos. Every experiment here is inference plus hooks plus logistic regression, so the compute is a small fraction of what the training sweeps cost.

## Naming

Two separate I-ladders appear in this work and they do not correspond rung for rung, so every mention is qualified as acrostic or CoT. The acrostic ladder is the one from Finetuning-steganography. Prose uses I₀ to I₆; scripts, data directories and Hub paths use the code names.

| rung | code | rule |
| ---- | ---- | ---- |
| I₀ | `stage1` | explicit secret tag |
| I₁ | `v0` | first letters of prompt words |
| I₂ | `v1a` | reversed |
| I₃ | `v1b` | Caesar shift |
| I₄ | `v2` | XOR with neighbours |
| I₅ | `v3` | majority vote |
| I₆ | `v4` | per-word MD5 hash |

The data and results directories additionally contain `v5` and `v6`, with results in `vladder_3b`, whose transforms the source README does not document. The CoT ladder is the one from trained-cot-controllability. There the payload bit and the encode instruction sit in the prompt at every rung and only the codebook mapping is withheld, at I₁. It is a weaker notion of internalisation than the acrostic ladder's, not an equivalent rung.

## Experiments

| id | question | models | status |
| -- | -------- | ------ | ------ |
| E1 | Does a probe trained on the supplied-codebook model read the internalised model? | CoT I₀, I₁, base | scripts written |
| E2 | Where on the acrostic ladder does the computation fail: extraction, transformation, or readout? | acrostic 3B, I₁ to I₅ | blocked on retrains |
| E3 | Is the payload precomputed at end of prompt or computed just in time at each sentence start? | acrostic 7B, I₀ and I₁ | adapters ready |
| E4 | Does an activation probe recover payloads that a text monitor and paraphrase defence miss? | local SFT models | planned |
| E6 | Is there a direction for encoding, and is it shared across schemes? | acrostic and permutation 7B | cross-scheme arm blocked |

E5 (per-layer LoRA ablation localisation) is deferred.

## Facts verified before writing any code

1. The CoT model is Qwen3-8B in bf16 with no quantisation and LoRA on q/k/v/o/gate/up/down. The acrostic and permutation models are Qwen2.5-Instruct in 4-bit NF4. Activations are not comparable across the two families, and every probe must run in the precision the model was trained and evaluated in. Acrostics and permutation share a base (Qwen2.5-7B-Instruct, NF4), so cross-scheme comparison in E6 is meaningful for that pair only.
2. In the CoT setup the payload bit is stated in the prompt at both I₀ and I₁; only the codebook block differs. Bit decodability is therefore a prompt copy and not evidence of encoding. E1 probes the pending connective choice, with the bit kept as a control that should decode everywhere including in the base model.
3. `strip_codebook` applied to a CoT I₀ prompt reproduces the CoT I₁ prompt byte for byte (verified). The i0_stripped condition therefore isolates the adapter from the prompt format.
4. The acrostic ladder data stores the transformed payload in the `secret` field, and the raw payload is the first letters of the prompt words. Both label sets for E2 are available without reimplementing any transform.
5. The I₃ transform is a Caesar shift of +3, not +1 as the ladder table states. Verified on 200 training rows: 200/200 match +3 and 0/200 match +1. The source README needs a correction.
6. The eval pool in the CoT repo is the leading 15% of GSM8K train, of which only the first 60 problems were used for `eval_ood`. The remaining problems in that pool are unused by training and evaluation, so they are free probe data. E1 draws from `--pool-start 60` onward.
7. The blackbox repo is entirely API-based, so there is no activation access to any model in it. E4's white-box arm runs on the local SFT models only; the frontier numbers stay as motivation.

## Adapter audit (Hub, checked before E1)

`WFJKK/poseidon-sft-adapters` holds 427 files across 49 directories. Present: acrostic I₀ and I₁ for Qwen2.5 1.5B to 72B under `acrostics_news_8bit_n500`, I₀ alone for the n100 and n250 sweeps, a payload sweep at 0.5B and 7B, and several `ccs_*` adapters.

Absent, and consequential:

1. No acrostic rung above I₁. The I₂ to I₆ adapters behind `results/vladder_3b` were never uploaded, so E2 begins by retraining I₂, I₃, I₄ and I₅ at 3B from the data in the source repo, at 1000 examples and 6 epochs to match the published SERs.
2. No permutation adapters. E4's permutation arm and E6's cross-scheme direction test are deferred until the permutation encoder is retrained.
3. The CoT I₀ and I₁ adapters exist neither locally nor on the Hub. E1 opens with `run_cot.sh` on the instance, roughly 25 minutes, which also reproduces the published BER numbers as a bf16 sanity check.

E3 is fully provisioned: acrostic I₀ and I₁ at 7B are both on the Hub.
