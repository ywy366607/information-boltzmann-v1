#!/bin/bash
# The adult (settle-14 lineage, before the go neuron) in a world that sets its own pace: every word's interval is
# memoryless (geometric), the window's mean interval log-uniform in [MU_MIN, 16]; every tick's expectation scores
# with the arrival hazard.  Words are pulses (a word is an event; the silence between words is input too).  No flux
# admission, no go neuron, no CTM, no homeostasis.  Chunks of 4,096 words; after each, the pace probe (fixed weights,
# fixed text).  MU_MIN starts at 4 and steps once to 2 when the probe's mu = 8 arrival CE stops improving by 0.02.
# One torch process at a time.  STOP file in the output dir stops it.
# Usage: grow_adult_pace.sh [pulse|hold] [chunks]
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; S=scratch/fly_stream
FORM=${1:-pulse}; CHUNKS=${2:-6}
out=$S/adult_pace_$FORM; mkdir -p $out
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
HOLD=--no-hold-input; [ "$FORM" = hold ] && HOLD=--hold-input
MU_MAX=16; MU_MIN=4; best=7.77                  # the zero-shot mu = 8 arrival CE
for chunk in $(seq 1 $CHUNKS); do
  [ -f $out/STOP ] && break
  RESUME=""; [ -f $out/last.pt ] || RESUME="--resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt"
  $PY -u scripts/ib/train_fly_bptt_stream.py $RESUME --extend-budget --additional-tokens 4096 \
    --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph \
    --go-neuron none --stream-ticks 8 $HOLD --stream-pace-mean $MU_MAX --stream-pace-min $MU_MIN \
    --no-homeostasis --no-ctm-loss --vram-limit-mib 4050 \
    --output $out --checkpoint-dir $out >> $out/train.log 2>&1
  code=$?; echo "chunk $chunk mu_min $MU_MIN train $code" >> $out/status; rm -f $out/best.pt
  [ $code -ne 0 ] && break
  $PY -u scripts/ib/fly_interface/pace_zero_shot.py $out $out/pace_probe.jsonl > $out/pace_probe_$chunk.txt 2>&1
  # plateau: the probe's mu = 8 arrival CE (same 1,024 validation words every chunk)
  v=$($PY -c "import json;r=[json.loads(l) for l in open('$out/pace_probe.jsonl')];print([x for x in r if x['condition']=='geo8_pulse'][-1]['arrival_ce_min_median_mean_max'][2])" 2>/dev/null || echo 99)
  echo "chunk $chunk probe_geo8 $v" >> $out/status
  if $PY -c "import sys;sys.exit(0 if float('$v') > float('$best') - 0.02 else 1)"; then
    [ $MU_MIN -gt 2 ] && MU_MIN=2 && echo "plateau: mu_min -> 2" >> $out/status
  fi
  $PY -c "import sys;sys.exit(0 if float('$v') < float('$best') else 1)" && best=$v
done
echo done >> $out/status
