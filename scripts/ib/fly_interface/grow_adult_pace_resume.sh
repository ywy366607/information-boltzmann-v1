#!/bin/bash
# Continuation of the adult in the world that sets its own pace, toward the end of the route: break the 14-tick waste
# (one word owning up to 14 ticks while the input waits).  Thinking overlaps the next words (a pipeline), so the
# measure is prediction per tick: the tempo ladder walks the window's mean interval range down
#   [4,16] -> [2,16] -> [2,8] -> [1,4]
# one rung each time the probe's arrival CE at the current fastest mean (mu = MU_MIN, or fixed 1 tick at the last rung)
# stops improving by 0.02.  Words are pulses; every tick's expectation scores with the arrival hazard.  No flux, no go,
# no CTM, no homeostasis.  Chunks of 4,096 words; after each, the pace probe (fixed weights, fixed text).
# One torch process at a time.  STOP file in the output dir stops it.
# Usage: grow_adult_pace_resume.sh [pulse|hold] [chunks]
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; S=scratch/fly_stream
FORM=${1:-pulse}; CHUNKS=${2:-6}
out=$S/adult_pace_$FORM; mkdir -p $out
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
HOLD=--no-hold-input; [ "$FORM" = hold ] && HOLD=--hold-input
LADDER=("4 16" "2 16" "2 8" "1 4")
rung=$(grep -o 'rung [0-9]*' $out/status 2>/dev/null | tail -1 | awk '{print $2}'); rung=${rung:-0}
grep -q 'plateau: mu_min -> 2' $out/status 2>/dev/null && [ $rung -eq 0 ] && rung=1
best=99
done_chunks=$(grep -c ' train 0' $out/status 2>/dev/null); done_chunks=${done_chunks:-0}
for chunk in $(seq $((done_chunks + 1)) $((done_chunks + CHUNKS))); do
  [ -f $out/STOP ] && break
  read MU_MIN MU_MAX <<< "${LADDER[$rung]}"
  RESUME=""; [ -f $out/last.pt ] || RESUME="--resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt"
  $PY -u scripts/ib/train_fly_bptt_stream.py $RESUME --extend-budget --additional-tokens 4096 \
    --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph \
    --go-neuron none --stream-ticks 8 $HOLD --stream-pace-mean $MU_MAX --stream-pace-min $MU_MIN \
    --no-homeostasis --no-ctm-loss --vram-limit-mib 4050 \
    --output $out --checkpoint-dir $out >> $out/train.log 2>&1
  code=$?; echo "chunk $chunk rung $rung range $MU_MIN-$MU_MAX train $code" >> $out/status; rm -f $out/best.pt
  [ $code -ne 0 ] && break
  $PY -u scripts/ib/fly_interface/pace_zero_shot.py $out $out/pace_probe.jsonl > $out/pace_probe_$chunk.txt 2>&1
  # plateau on the current rung's fastest condition (same 1,024 validation words every chunk)
  cond=geo${MU_MIN}_pulse; [ $MU_MIN -eq 1 ] && cond=fixed1
  v=$($PY -c "import json;r=[json.loads(l) for l in open('$out/pace_probe.jsonl')];print([x for x in r if x['condition']=='$cond'][-1]['arrival_ce_min_median_mean_max'][2])" 2>/dev/null || echo 99)
  echo "chunk $chunk probe $cond $v" >> $out/status
  if $PY -c "import sys;sys.exit(0 if float('$v') > float('$best') - 0.02 else 1)"; then
    if [ $rung -lt 3 ]; then rung=$((rung + 1)); best=99; echo "plateau: rung $rung (${LADDER[$rung]})" >> $out/status; fi
  else
    best=$v
  fi
done
echo done >> $out/status
