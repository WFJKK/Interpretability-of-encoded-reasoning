# Interpretability of Encoded Reasoning

*Part of SPAR 2026 fellowship research supervised by Dr. Robert Krzyzanowski at Poseidon Research.*

White-box follow-up to two black-box results. [WFJKK/Finetuning-steganography](https://github.com/WFJKK/Finetuning-steganography) showed that SFT can internalise an output-channel encoding and mapped where that internalisation breaks down; [WFJKK/trained-cot-controllability](https://github.com/WFJKK/trained-cot-controllability) showed that a 1-bit payload partially internalises into the chain of thought. Both measure behaviour. This repo asks whether the distinction those results rest on, capability with the rule supplied against internalisation with the rule withheld, is also a distinction inside the network, and where in the network the encoding lives.

The models are the ones already trained in those repos. Every experiment here is inference plus hooks plus logistic regression, so the compute is a small fraction of what the training sweeps cost.

## Experiments

| id | question | models | status |
| -- | -------- | ------ | ------ |
| E1 | Does a probe trained on the supplied-codebook model read the internalised model? | CoT I₀, I₁, base | scripts written |
| E2 | Where on the acrostic ladder does the computation fail: extraction, transformation, or readout? | acrostic 3B I₁ to I₅ | planned |
| E3 | Is the payload precomputed at end of prompt or computed just in time at each sentence start? | acrostic 7B I₀, I₁ | planned |
| E4 | Does an activation probe recover payloads that a text monitor and paraphrase defence miss? | all local SFT models | planned |
| E6 | Is there a direction for encoding, and is it shared across schemes? | acrostic and permutation 7B | planned |

E5 (per-layer LoRA ablation localisation) is deferred.

## Assets and facts verified before writing any code

Checked directly against the source repos, since several of these change the design.

1. The CoT model is Qwen3-8B in bf16 with no quantisation and LoRA on q/k/v/o/gate/up/down. The acrostic and permutation models are Qwen2.5-Instruct in 4-bit NF4. Activations are therefore not comparable across the two families, and every probe must run in the precision the model was trained and evaluated in. Acrostics and permutation share a base (Qwen2.5-7B-Instruct, NF4), so cross-scheme comparison in E6 is meaningful for that pair only.
2. In the CoT setup the payload bit is stated in the prompt at both I₀ and I₁; only the codebook block differs. Bit decodability is therefore a prompt copy and not evidence of encoding. E1 probes the pending connective choice, with the bit kept as a control that should decode everywhere including in the base model.
3. `strip_codebook` applied to an I₀ prompt reproduces the I₁ prompt byte for byte (verified). The i0_stripped condition therefore isolates the adapter from the prompt format.
4. The ladder data stores the transformed payload in the `secret` field, and the raw payload is the first letters of the prompt words. Both label sets for E2 are available without reimplementing any transform.
5. The v1b transform is a Caesar shift of +3, not +1 as the ladder table states. Verified on 200 training rows, 200/200 match for +3 and 0/200 for +1. The source README needs a correction.
6. The acrostic data directory contains rungs v0, v1a, v1b, v2, v3, v4, v5 and v6, and `results/vladder_3b` holds results for all of them at 500 and 1000 examples. The README table documents only a subset.
7. The eval pool in the CoT repo is the leading 15% of GSM8K train, of which only the first 60 problems were used for `eval_ood`. The remaining problems in that pool are unused by training and by evaluation, so they are free probe data. E1 draws from `--pool-start 60` onward.
8. The blackbox repo is entirely API-based through OpenRouter and Anthropic, so no activation access to any model in it. E4's white-box arm runs on the local SFT models only; the frontier numbers stay as motivation.

Still to confirm on the machine, not resolvable from a browser: which adapters are actually present in `WFJKK/poseidon-sft-adapters`, and whether the CoT I₀ and I₁ adapters still exist anywhere (they are git-ignored in the source repo and retrain in about 11 minutes each).

## E1 protocol

Four conditions on the same base model:

| condition | adapter | codebook in prompt | published BER |
| --------- | ------- | ------------------ | ------------- |
| base_shown | none | yes | 0.967 |
| i0_shown | I₀ | yes | 0.008 |
| i0_stripped | I₀ | no | 0.653 |
| i1_stripped | I₁ | no | 0.305 |

The probe is logistic regression on the residual stream at the token immediately before the first connective in the reasoning, predicting which connective the model is about to emit. Reported per layer: within-condition cross-validated accuracy, cross-condition transfer accuracy, cosine between probe weight vectors, and a shuffled-label control.

The reading of the outcomes. Transfer from i0_shown to i1_stripped close to the within-condition ceiling, with high cosine, means the internalised codebook occupies the same representation as the supplied one, so internalisation amounts to the model supplying its own instruction. Transfer at the base_shown level means SFT built a separate pathway. The i0_stripped condition separates a failure caused by the change in prompt format from a failure caused by the change in adapter, because it shares the prompt format of i1_stripped and the adapter of i0_shown.

Generation is required rather than teacher forcing on the training data, because the question is what the model represents when it makes the choice itself. At I₁ the realised connective disagrees with the intended bit on about 30% of trials, and that dissociation is what makes the label informative; `--correct-only` restricts to agreeing trials as a secondary analysis.

## Layout

```
common/cot_common.py            prompts, chat rendering, connective decoding, model loading
                                (mirrors trained-cot-controllability exactly)
e1/e1_probe_transfer.py         gen | cache | probe
run_e1.sh                       driver for all four conditions
experiments/e1/results/         probe outputs, committed
```

Activations are large and stay in `/dev/shm`; only results JSON, probe weights and plots are committed.

## Running E1

```
pip install torch transformers peft accelerate scikit-learn

# smoke test first, 8 generations
python e1/e1_probe_transfer.py gen --condition i0_shown --limit 8 \
    --adapter-i0 /dev/shm/adapter-cot-cause-i0 --out /dev/shm/e1/smoke.jsonl

# full run, all four conditions
nohup bash run_e1.sh > /dev/shm/e1.log 2>&1 &
tail -5 /dev/shm/e1.log
```
