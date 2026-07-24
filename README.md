# Interpretability of Encoded Reasoning

*SPAR 2026 fellowship research, supervised by Dr. Robert Krzyzanowski at Poseidon Research.*

Two earlier repos established behaviour. [WFJKK/Finetuning-steganography](https://github.com/WFJKK/Finetuning-steganography) showed that supervised fine-tuning can internalise an acrostic encoding, so that a model derives and emits a hidden payload with no instruction at inference time, and mapped the point on a ladder of transformations where that internalisation breaks down. [WFJKK/trained-cot-controllability](https://github.com/WFJKK/trained-cot-controllability) showed partial internalisation of a 1-bit payload into the chain of thought.

Both measure what the model does. This repo asks what the model represents. The central question is whether the distinction those results rest on, capability with the rule supplied against internalisation with the rule withheld, corresponds to anything inside the network, and where in the network the encoding lives.

## Summary of findings

The derived payload occupies the same linear representation as the supplied one. A probe trained to read the letter from a model that is told the secret reads a model that derives it for itself at close to that model's own ceiling, in both directions and on disjoint examples. Internalisation, at least here, amounts to the model supplying itself the instruction rather than building separate machinery.

The payload is resolved one letter at a time. Only the letter about to be emitted is decodable at the point of decision; letters one line ahead are weak and two lines ahead are at chance. This holds even for the model that is handed all eight letters in its prompt, so the locality is not a consequence of having to derive the payload. It gives a mechanism for the pattern reported in the parent repo, that locally controllable encodings are learnable and globally resolved ones are not: a scheme whose value is fixed only at the end of a line has nothing to condition on at the token that commits to its first letter.

The letter is carried by the activation state, not by a single extractable direction. Transplanting the whole activation at the decision point flips the emitted letter up to 94% of the time, and does so only at the correct site and only in the layers where the letter is decodable. But adding a per-letter mean-difference direction moves the letter to the target at most 13% of the time, and projecting the adapter-versus-base direction out of the residual stream changes nothing a random direction of the same norm does not also change. Detection by probe is easy here; control by direction is not.

## Naming

Two I-ladders appear in this work and they do not correspond rung for rung, so every mention is qualified. The acrostic ladder, from Finetuning-steganography, withholds progressively more: prose uses I₀ to I₆ and code uses the rung names.

| rung | code | rule |
| ---- | ---- | ---- |
| I₀ | `stage1` | secret supplied in the prompt |
| I₁ | `v0` | first letters of the prompt words, derived by the model |
| I₂ | `v1a` | reversed |
| I₃ | `v1b` | Caesar shift of +3 |
| I₄ | `v2` | XOR with neighbours |
| I₅ | `v3` | majority vote |
| I₆ | `v4` | per-word MD5 hash |

The CoT ladder, from trained-cot-controllability, supplies the payload bit and the encode instruction at every rung and withholds only the codebook. It is a weaker notion of internalisation, not an equivalent rung.

## Setup

All acrostic work uses Qwen2.5-7B-Instruct in 4-bit NF4, 180 held-out prompts, eight-letter payloads, greedy decoding. Three conditions are built from the same prompts and the same target payload so that only the source of the payload varies: i0 supplies it as `<secret>`, i1 derives it from the prompt words, base has no adapter. The i1 model is the parent repo's stacked construction, stage1 merged into the base and the v0 adapter applied on top.

Activations are cached once and reused by every read-only analysis, at 17 sites per example across all 29 layers: the end of the prompt, the token immediately before each of the eight acrostic letters, and each letter's own token as a positive control. The token before a letter is the decision point, where the model has committed to nothing yet; a probe at the letter's own token would be reading its own input.

Behaviour, for reference: i0 reaches SER 0.019, i1 reaches 0.192 against the published 0.1875, and base reaches 0.924 with only 0.6% of its generations producing eight lines at all. Because base cannot produce the format, its floor condition is instead the base model teacher-forced on i1's own generated text, which reads identical tokens with the letters visibly present and yields 180 matched pairs.

## E1: the derived payload occupies the representation of the supplied one

A 24-way logistic probe on the decision-point residual stream, fitted on examples 0 to 89 and scored on examples 90 to 179 so that no example appears in both, with the first line excluded because for i0 that position is a copy of the prompt. Majority-class rate 0.095.

| layer | i0 to i0 | i0 to i1 | i1 to i1 | i1 to i0 | either to base | base to base |
|-------|----------|----------|----------|----------|----------------|--------------|
| 4  | 0.108 | 0.108 | 0.092 | 0.056 | 0.078 to 0.083 | 0.100 |
| 24 | 0.938 | 0.613 | 0.602 | 0.725 | 0.079 to 0.087 | 0.113 |
| 28 | 0.973 | 0.740 | 0.648 | 0.875 | 0.092 to 0.106 | 0.106 |

Transfer is at ceiling in both directions. Shuffled-label controls run 0.058 to 0.073, macro-averaged recall tracks raw accuracy at 0.777 to 0.823, so this is not an artefact of the skewed letter distribution, and a sweep at every second layer finds nothing above chance until layer 22 and an abrupt jump at 24. The letter decision crystallises in the last four or five layers rather than forming gradually.

Cosine between the i0 and i1 probe weight vectors, computed in the raw residual basis at layer 28, averages 0.242 across 24 letters, against about 0.017 by chance in 3584 dimensions. The probes find overlapping rather than identical directions. Transfer accuracy is the load-bearing measure; cosine between multiclass coefficients is weak evidence either way, since a decision boundary survives substantial rotation.

The floor for this comparison is not 0.095 but 0.325, for the reason given under E9.

## E3: the payload is resolved one letter at a time

Probes at layer 28, the peak identified by E1.

| | i0 | i1 |
|---|---|---|
| end of prompt, letter 1 | 0.99 | 0.99 |
| end of prompt, letters 3 to 8 | 0.09 to 0.19 | 0.07 to 0.15 |
| decision point, imminent letter | 0.971 | 0.774 |
| decision point, one line ahead | 0.240 | 0.271 |
| decision point, two lines ahead | 0.140 | 0.151 |

Majority rates are 0.27 for the first slot and about 0.10 to 0.19 elsewhere. The i0 column carries the argument: that model is handed all eight letters in its prompt and still does not aggregate them at the final prompt position, so it retrieves each letter when it needs it rather than holding a plan.

Two qualifications. One-line lookahead at 0.24 to 0.27 is above baseline, so encoding is not purely local. And the secret tokens remain in context, so the correct claim is that the payload is not carried forward as a global plan, not that it is unavailable. The positive control, the letter's own token, reads 0.703 for i0 and 0.532 for i1 at layer 28, lower than the imminent-letter probe because at the final layer the residual stream is dominated by what comes next; the same control reads 0.70 at layer 14.

## E7: patching shows the representation is used, and used locally

Read-only probes show information is present, not that the model consults it. Here the recipient's own text is teacher-forced up to its decision point for line k, so the patch site is always the final position of the prefix, a donor example's cached activation for its own line k is written into layers 24 to 28, and generation continues. Fifty pairs per line, chosen so the donor and recipient letters differ.

| line | none | patch | mismatched site | early layer |
|------|------|-------|-----------------|-------------|
| 2 | flip 0.00, keep 0.96 | flip 0.94, keep 0.00 | flip 0.04 | flip 0.00, keep 0.96 |
| 4 | flip 0.04, keep 0.68 | flip 0.60, keep 0.08 | flip 0.08 | flip 0.06, keep 0.66 |
| 6 | flip 0.02, keep 0.66 | flip 0.52, keep 0.02 | flip 0.06 | flip 0.02, keep 0.66 |

The activation at the decision point causally determines the letter. The effect is specific to the correct site rather than to perturbation in general, and it is absent at a layer where E1 found no decodable signal. The declining flip rate across lines is unexplained; accumulated context resisting a single patched vector is the obvious candidate.

Downstream letters are largely undisturbed, but the locality claim needs care: the no-patch condition preserves the tail on only 0.48, 0.50 and 0.84 of trials at lines 2, 4 and 6, because i1 makes errors anyway. Against those baselines, patching gives 0.28, 0.44 and 0.84. Disruption is modest relative to a noisy baseline, and the clean version of the test would condition on trials where the no-patch run preserved the tail.

## E8: the encoding is not carried by a single direction

Projecting the adapter-versus-base direction out of the residual stream at every position, for whole free generations, against a random direction of the same norm.

| condition | SER | lines | words per line |
|-----------|-----|-------|----------------|
| none | 0.219 | 8.0 | 14.1 |
| ablate encoding direction | 0.197 | 8.0 | 14.0 |
| ablate random direction | 0.216 | 8.0 | 14.2 |

A null. Steering is nearly as weak: adding a per-letter direction at the decision point moves the emitted letter to the target on 0.000, 0.033, 0.067, 0.067 and 0.133 of trials as the scale goes 0, 20, 50, 100, 200, while the original letter falls only from 0.667 to 0.600. At n=30 the top point is four examples, so it is a trend rather than a measurement.

Set against E7, this is informative rather than a failed experiment. The whole activation state controls the letter; averaged directions extracted from it barely do. That is consistent with E6, which found the per-letter directions to be near-orthogonal rather than variations on one axis, and it means an activation-level defence built on a single steering direction would not work on this model.

## E6: per-letter directions

Mean-difference directions at the decision point, layer 28, for the 18 letters with at least 20 instances. Mean pairwise cosine is -0.041, so the letters occupy close to orthogonal directions; the slight negative is the expected arithmetic of mean-centred vectors. Mean norm is 87.2. The i1-minus-base direction over 180 matched pairs has norm 157.3. Without a no-encoding control on the same adapter, that vector is what these adapters do to activations rather than a demonstrated encoding feature, and E8 shows it does not behave like a switch.

## E9: the base model's apparent transfer is next-token predictability

E1 left an anomaly. A probe fitted on base activations scores 0.106 on held-out base activations, the majority rate, yet scores 0.419 on i0 and 0.325 on i1. Four diagnostics resolve it. Shuffled labels collapse the transfer to 0.14 and 0.10, so it is genuinely label-dependent. Removing PCA raises it to 0.475, so it is not a projection artefact. The probe predicts 14 to 16 distinct letters with macro recall 0.27 to 0.33, so it has not collapsed onto frequent classes. Decisively, on i1, whose output disagrees with its payload about 19% of the time, the probe scores 0.363 against the letter actually written and 0.325 against the intended payload; on i0, where the two coincide, it scores 0.421 and 0.419.

The probe tracks the letter that is coming, not the code. A base model reading coherent text has some information about the next word, hence its first letter, along directions shared across models that share base weights and unembedding. On base's own activations the separation is too small to beat the majority class; applied to i0, where the adapters have amplified letter commitment enormously, the same roughly correct direction classifies well.

This sharpens E1 rather than weakening it. The generic next-letter component is worth 0.325, so the right statement is that i0 to i1 transfer at 0.740 beats a generic floor of 0.325, not that it beats chance at 0.095.

## E2: where the ladder fails, and a design flaw

The I₃ (Caesar) and I₄ (XOR) adapters were never uploaded and were retrained at 3B on 1000 examples for 6 epochs. Both reproduce: SER 0.784 against a published 0.833, and 0.774 against 0.847, both at zero exact matches. They are now on the Hub under `acrostics_ladder_3b_n1000`.

Probes at layer 36 carry three label sets: the letter actually emitted, the raw prompt-derived payload, and the transformed payload the rung should produce.

| rung | emitted | raw | transformed | majority |
|------|---------|-----|-------------|----------|
| I₃ Caesar | 0.392 | 0.215 | 0.214 | 0.104 to 0.150 |
| I₄ XOR | 0.465 | 0.203 | 0.232 | 0.094 to 0.106 |

The emitted letter decodes well, so the probe works on models that fail the task. Both payloads sit at about twice the majority rate, far below the 0.648 the working 7B model achieves. Partial credit is ruled out: the emitted letter coincides with the raw payload on 3 to 4% of positions and with the transformed payload on 11 to 12%, well under the 0.21 the probes reach. So a trace of the payload is present, but the failing rungs do not hold it strongly enough for an operation over it to have a substrate, which is what E3 predicts.

The experiment has a flaw worth recording. A Caesar shift is a per-letter bijection, so the raw and transformed label sets are the same information relabelled, and no linear probe can distinguish them; the observed 0.215 and 0.214 are that identity, not evidence. The Caesar arm cannot separate extraction from transformation as designed. XOR is not bijective and its two label sets do differ, at 0.203 and 0.232. There is also no 3B base control, so part of the 0.21 may be the generic component E9 identified.

## The CoT result does not reproduce

The acrostic pivot happened because the chain-of-thought track failed to reproduce. Retraining CoT I₁ with the source repo's own script, unmodified, on an A100-SXM4-40GB under transformers 5.14.1 and torch 2.11.0:

| condition | published | seed 42 | seed 1, epochs 1 to 3 |
|-----------|-----------|---------|------------------------|
| I₀ codebook shown | 0.008 | 0.025 | |
| I₀ codebook stripped | 0.653 | 0.644 | |
| I₁ codebook stripped | 0.305 | 0.492 | 0.492, 0.559, 0.525 |

The I₀ conditions reproduce; I₁ is at chance across two seeds and every epoch checkpoint. A larger sample confirms it: 176 generations on unused held-out problems give BER 0.494 with a standard error of 0.038. Coverage stays at 0.97 to 1.00 and training loss falls to 0.029, so the model fits the data and emits connectives fluently; the payload is not riding on the choice. The data files are byte-identical and the hyperparameters unchanged, which leaves the software stack as the prime suspect, since the original used transformers 5.6.2 and `warmup_ratio` is deprecated between the two versions. Evidence is in `experiments/e1/results/cot_nonreproduction/`. Pending resolution the 0.305 figure should be treated as provisional.

## Method corrections made during this work

Two intermediate results were wrong and are recorded because both are easy traps.

The first transfer numbers were inflated by example leakage. Probes were fitted on the source condition's full 180 examples and scored on the target's full 180, and since every condition uses the same examples with the same payloads, "this prompt at this position means this letter" could be memorised from prompt-derived features and reused. The tell was transfer exceeding the target's own cross-validated ceiling, and doing so at layers with no signal at all. All reported figures now use disjoint example splits.

Cosine similarity between probe weight vectors was initially computed after per-condition PCA, which places the two vectors in different bases and makes the comparison meaningless. It is now computed in the raw residual basis.

A third choice is worth noting rather than correcting. The base condition could not generate the format, so the floor was obtained by teacher-forcing the base model on i1's text instead. This is the better control anyway, since it asks whether the letter is represented when the base model reads identical tokens, and it supplies matched pairs for the direction analyses.

## Layout

```
common/acrostic_common.py    prompts, chat rendering, letter extraction, model loading
common/cot_common.py         the same for the CoT track
e13/cache_acrostic.py        generation and activation caching (gen | cache)
e13/probe_acrostic.py        E1 transfer, E3 just-in-time, E6 directions
e13/patch_acrostic.py        E7 activation patching
e13/e8_steer_ablate.py       E8 steering and projection ablation
e13/e9_base_anomaly.py       E9 diagnostics
e13/e2_ladder.py             E2 ladder rungs, self-contained (gen | cache | probe)
run_acrostic.sh              driver for the shared cache and the read-only analyses
experiments/e13/results/     all results and run logs
```

Activations are gigabytes and are regenerated rather than stored; only results, probe weights and logs are committed. Adapters live on `WFJKK/poseidon-sft-adapters`.

## Next steps

The Caesar arm of E2 needs redesigning around a non-bijective transformation, and E2 needs a base control at 3B. E7's locality test should be conditioned on trials where the unpatched run preserves the downstream letters. The declining flip rate across line positions is unexplained. Replication at other model sizes is cheap, since `stage1` and `v0` adapters exist from 1.5B to 72B and the pipeline is unchanged, and the behavioural phase transition between 1.5B and 3B reported in the parent repo makes the representational question there worth asking.

Two experiments remain unstarted. E4 would test whether an activation probe recovers payloads that a text monitor and a paraphrase defence miss, reusing the E1 probes directly, and has the clearest safety payoff. E5, per-layer LoRA ablation to localise where each scheme lives, is deferred.
