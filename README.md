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

## E1/E3/E6 results: acrostic 7B (23 July 2026)

All three experiments read from one activation cache. Qwen2.5-7B-Instruct in 4-bit NF4,
180 held-out prompts from `data/news/v0_8bit/test.jsonl`, three matched conditions built
from the same prompts and the same target payload so that only told-versus-derived varies:
i0 (stage1 adapter, secret supplied as `<secret>`), i1 (merged stage1 + v0 adapter, bare
prompt), and base (no adapter). Activations were cached at 17 sites per example across all
29 layers: end of prompt, the token immediately before each of the 8 acrostic letters (the
decision point, where the letter has not yet been emitted), and each letter's own token as
a positive control.

Behaviour first. i0 reaches SER 0.019, i1 reaches 0.192 against the published 0.1875, and
base reaches 0.924 with only 0.6% of generations producing eight lines. The i1
reproduction matters because the parallel CoT experiment failed to reproduce on the same
software stack, so the acrostic result rests on a model confirmed to still encode.

### E1: the derived payload occupies the representation of the supplied one

A 24-way logistic probe on the decision-point residual stream, fitted on examples 0 to 89
and scored on examples 90 to 179 so that no example appears in both, with pre_0 excluded
because for i0 that position is a copy of the prompt. Majority-class rate 0.095.

| layer | i0 to i0 | i0 to i1 | i1 to i1 | i1 to i0 | to base | base to base |
|-------|----------|----------|----------|----------|---------|--------------|
| 4  | 0.108 | 0.108 | 0.092 | 0.056 | 0.078 to 0.083 | 0.100 |
| 24 | 0.938 | 0.613 | 0.602 | 0.725 | 0.079 to 0.087 | 0.113 |
| 28 | 0.973 | 0.740 | 0.648 | 0.875 | 0.092 to 0.106 | 0.106 |

A probe trained on the told-the-secret model recovers the derived payload at 0.740 against
that model's own held-out ceiling of 0.648, and the reverse direction gives 0.875. Transfer
is at ceiling in both directions on disjoint examples, so the two conditions share a linear
representation and internalisation amounts to the model supplying itself the instruction
rather than building a separate pathway. Shuffled-label controls run 0.058 to 0.073 and
macro-averaged recall tracks raw accuracy at 0.777 to 0.823, so this is not an artefact of
the skewed letter distribution. A layer sweep at every second layer finds nothing above
chance until layer 22 and an abrupt jump at 24, so the letter decision crystallises in the
last four or five layers rather than forming gradually.

The base column is the sharper result. The base model was teacher-forced on i1's own
generated text, so it reads the identical tokens with the letters visibly present, and
every probe reads it at chance while its own held-out accuracy is 0.106. The representation
the adapters use is absent from the base model rather than merely weaker in it. One
asymmetry remains unexplained: probes fitted on base activations score 0.419 on i0 and
0.325 on i1 despite base itself being at chance, which most likely reflects a generic
next-token letter direction shared across models rather than anything encoding-specific.

### E3: the payload is resolved one letter at a time, even when it is supplied

Probes at layer 28, the peak identified by E1. At the end of the prompt, before any text
exists, only the first letter is readable: 0.99 against a 0.27 majority for both conditions,
while slots 2 to 7 sit at or below their majority rates in both. At each decision point the
imminent letter reads 0.971 for i0 and 0.774 for i1, one line ahead falls to 0.240 and
0.271, and two ahead is at chance.

The i0 row carries the argument. That model is handed all eight letters in the prompt and
still does not aggregate them into a plan at the final prompt position; it retrieves the
letter it needs when it needs it. So the locality is not a consequence of having to derive
the payload, it is how the model operates even when the answer is written in front of it.
This gives a mechanism for the locally-controllable-succeeds and globally-resolved-fails
pattern reported in the parent repo: a scheme whose value is fixed only at line completion
has nothing to condition on at the token that commits to the first letter. Two qualifications.
The one-line lookahead at 0.24 to 0.27 is above the 0.095 baseline, so encoding is not purely
local. And the secret tokens remain in context throughout, so the correct claim is that the
payload is not carried forward as a global plan, not that it is unavailable.

The positive control, the letter's own token, reads 0.703 for i0 and 0.532 for i1 at layer
28, lower than the imminent-letter probe. This is expected at the final layer, where the
residual stream is dominated by what comes next rather than by the current token; the same
control read 0.70 at layer 14 for i0.

### E6: per-letter directions are near-orthogonal

Mean-difference directions at the decision point, layer 28, for the 18 letters with at
least 20 instances. Mean pairwise cosine is -0.041, so the letters occupy close to
orthogonal directions rather than sharing a single axis, and the slight negative is the
expected arithmetic of mean-centred vectors. Mean direction norm is 87.2. Contrasting i1
against the base model teacher-forced on identical text over 180 matched pairs gives a
difference vector of norm 157.3, larger than the letter-to-letter variation. Without a
no-encoding control condition on the same adapter this vector is what these adapters do to
activations rather than a demonstrated encoding feature, so calling it an encoding-mode
direction is an interpretation the present data does not establish.

### Method corrections made during this run

Two intermediate results were wrong and are recorded here because both are easy traps.

The first transfer numbers were inflated by example leakage. Probes were fitted on the
source condition's full 180 examples and scored on the target's full 180, and since every
condition uses the same examples with the same payloads, "this prompt at this position
means this letter" could be memorised from prompt-derived features and reused. The tell was
transfer exceeding the target's own cross-validated ceiling, and at layers with no signal at
all: i1 to base_tf scored 0.225 at layer 4 while both within-condition accuracies were below
0.08. All reported figures now fit on examples 0 to 89 and score on 90 to 179.

Cosine similarity between probe weight vectors was initially computed after per-condition
PCA, which places the two vectors in different bases and makes the comparison meaningless.
Recomputed in the raw residual basis at layer 28, the mean per-letter cosine between the i0
and i1 probes is 0.242 across 24 letters, well above the chance value of about 0.017 in 3584
dimensions but far from alignment. The honest reading is that the probes find overlapping
rather than identical directions, and that cosine between multiclass coefficients is a weak
instrument because a decision boundary survives substantial rotation. Transfer accuracy is
the load-bearing measure; cosine corroborates it.

A third choice worth noting rather than correcting: the first cache of the base condition
was nearly empty, because an unadapted model writes paragraphs rather than eight
newline-separated sentences and only 0.6% of its generations produced eight lines. Rather
than discard the floor, the base model was teacher-forced on i1's generated text and cached
at the same sites. This is the better control anyway, since it asks whether the letter is
represented when the base model reads identical tokens, and it supplies 180 matched pairs
for the direction analysis.

## Next steps

The present results are read-only: probes and mean-difference directions establish what is
represented and where, not that the model uses it. The causal follow-ups are cheap now that
the peak layer is known. Patching a decision-point activation from a donor generation into a
recipient at layers 24 to 28 tests whether the letter is used rather than merely present,
and whether the effect stays local to one line. Steering along a per-letter direction tests
whether the acrostic can be redirected without disturbing the text. Projecting out the
adapter-versus-base difference vector tests whether encoding can be switched off while
fluency survives.

Two experiments remain unstarted. E2 needs the acrostic I₂ to I₅ adapters, which were never
uploaded and must be retrained at 3B, and asks whether Caesar, XOR and majority vote fail at
the same computational step despite their similar symbol error rates. E4 would test whether
an activation probe recovers payloads that a text monitor and paraphrase defence miss; the
E1 probes are directly reusable for it.
