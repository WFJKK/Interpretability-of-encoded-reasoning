#!/bin/bash
# E1 driver: generate, cache and probe all four conditions.
# Resumable: rerunning skips generations already written and caches already saved.
# Launch:  nohup bash run_e1.sh > /dev/shm/e1.log 2>&1 &
set -u
I0=${I0:-/dev/shm/adapter-cot-cause-i0}
I1=${I1:-/dev/shm/adapter-cot-cause-i1}
N=${N:-300}                 # problems per condition, x2 payload bits
ROOT=${ROOT:-/dev/shm/e1}
mkdir -p "$ROOT/gen" "$ROOT/acts" experiments/e1/results

for f in "$I0" "$I1"; do
  [ -d "$f" ] || { echo "FATAL: missing adapter $f (retrain with run_cot.sh)"; exit 1; }
done

for C in base_shown i0_shown i0_stripped i1_stripped; do
  echo "######## [$C] generate ########"
  python e1/e1_probe_transfer.py gen --condition "$C" --n-problems "$N" \
      --adapter-i0 "$I0" --adapter-i1 "$I1" --out "$ROOT/gen/$C.jsonl" || exit 1
  echo "######## [$C] cache ########"
  python e1/e1_probe_transfer.py cache --condition "$C" --gen "$ROOT/gen/$C.jsonl" \
      --adapter-i0 "$I0" --adapter-i1 "$I1" --outdir "$ROOT/acts/$C" || exit 1
done

echo "######## probes ########"
for SITE in pre conn lastprompt; do
  python e1/e1_probe_transfer.py probe --cacheroot "$ROOT/acts" --site "$SITE" \
      --label connective --out "experiments/e1/results/probe_${SITE}_connective.json"
done
python e1/e1_probe_transfer.py probe --cacheroot "$ROOT/acts" --site pre \
    --label bit --out experiments/e1/results/probe_pre_bit_control.json
echo "======== E1 done ========"
