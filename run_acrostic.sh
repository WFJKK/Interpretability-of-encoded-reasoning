#!/bin/bash
# E1 + E3 + E6 from one shared activation cache, acrostic 7B I0/I1/base.
# Resumable: gen appends and skips written ids, cache skips existing .pt files.
# Launch:  nohup bash run_acrostic.sh > /dev/shm/acr.log 2>&1 &
set -u
FT=${FT:-/workspace/Finetuning-steganography/acrostics}
ROOT=${ROOT:-/dev/shm/acr}
S1=${S1:-$FT/adapters/qwen2.5-7b/stage1/full/final}
V0=${V0:-$FT/adapters/qwen2.5-7b/v0/full/final}
MERGED=${MERGED:-/workspace/merged-7b-stage1}
DATA=${DATA:-$FT/data/news/v0_8bit/test.jsonl}
LAYER=${LAYER:-14}

mkdir -p "$ROOT/gen" "$ROOT/acts" experiments/e13/results
for d in "$S1" "$V0"; do
  [ -f "$d/adapter_config.json" ] || { echo "FATAL: no adapter at $d"; exit 1; }
done

for C in i0 i1 base; do
  echo "######## [$C] generate ########"
  python e13/cache_acrostic.py gen --condition "$C" --data "$DATA" \
      --stage1-adapter "$S1" --v0-adapter "$V0" --merged-dir "$MERGED" \
      --out "$ROOT/gen/$C.jsonl" || exit 1
  echo "######## [$C] cache ########"
  python e13/cache_acrostic.py cache --condition "$C" --data "$DATA" \
      --stage1-adapter "$S1" --v0-adapter "$V0" --merged-dir "$MERGED" \
      --gen "$ROOT/gen/$C.jsonl" --outdir "$ROOT/acts/$C" || exit 1
done

echo "######## E1 transfer ########"
python e13/probe_acrostic.py transfer --cacheroot "$ROOT/acts" \
    --out experiments/e13/results/e1_transfer.json

echo "######## E3 just-in-time ########"
for C in i0 i1; do
  python e13/probe_acrostic.py jit --cacheroot "$ROOT/acts" --condition "$C" \
      --layer "$LAYER" --out "experiments/e13/results/e3_jit_$C.json"
done

echo "######## E6 directions ########"
python e13/probe_acrostic.py directions --cacheroot "$ROOT/acts" --condition i1 \
    --layer "$LAYER" --out experiments/e13/results/e6_directions.json
echo "======== done ========"
