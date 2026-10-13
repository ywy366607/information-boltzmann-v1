#!/bin/bash
# D3 (GOAL: docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261013.md, section D3): D1's setup (commits sampled
# from motor cells, exact expectation over commit times, listener-prior miss, habit prior; pulse world; eyes open) but
# the policy gradient reaches the brain through the motor spikes (--act-brain-grad), as GRPO updates the whole network.
# The answers' values stay detached (beliefs are trained by the language objective only).  Compare with D1's d1_habit.
# Arm d3_precision adds the policy precision encoded by the brain's own dopamine cells (--act-dopamine-precision,
# GOAL section 'D3b'); d3_braingrad is its control.  Before the run: the tests (incl. the GPU ones) on a free GPU.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; root=/e/fly_stream_overflow/d3_braingrad
ARMS=${ARMS:-"d3_braingrad d3_precision"}
CHUNKS=${1:-3}
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
COMMON="--extend-budget --additional-tokens 4096 --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 \
  --window 32 --no-graph --go-neuron none --stream-ticks 8 --no-hold-input --stream-pace-mean 16 --stream-pace-min 2 \
  --no-homeostasis --no-ctm-loss --kc-dendritic-only --kc-renorm --connectome-v2 --dopamine-modulation \
  --dopamine-site pre --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 \
  --ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.0 --miss-prior --ai-epistemic 0.0 \
  --ai-pragmatic 1.0 --ai-sigreg 0.01 --eye-forced-open --act-brain-grad --vram-limit-mib 3800"
mkdir -p $root
$PY -m pytest -q tests/test_fly_adaptive_admission.py > $root/tests.txt 2>&1 || { echo "$(date +%H:%M) tests FAILED" >> $root/status; exit 1; }
echo "$(date +%H:%M) tests passed" >> $root/status
extra() { [ "$1" = d3_precision ] && echo "--act-dopamine-precision"; }
# smoke of the precision arm (calibrates the dopamine tone; fails loudly if the dopamine cells are silent on text)
smoke=$root/smoke_precision; rm -rf $smoke
if [ ! -f $root/d3_precision/last.pt ]; then
  if $PY -u scripts/ib/train_fly_bptt_stream.py --resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt --al-reset \
       $COMMON --act-dopamine-precision --additional-tokens 64 --validate-every-tokens 64 --eval-tokens 32 \
       --output $smoke --checkpoint-dir $smoke > $root/smoke_precision.log 2>&1; then
    grep -h "dopamine tone" $root/smoke_precision.log >> $root/status
  else
    echo "$(date +%H:%M) precision smoke FAILED (see smoke_precision.log): running the control only" >> $root/status
    ARMS=d3_braingrad
  fi
  rm -rf $smoke
fi
for chunk in $(seq 1 $CHUNKS); do
  for arm in $ARMS; do
    [ -f $root/STOP ] && break 2
    grep -q "chunk $chunk $arm train 0" $root/status 2>/dev/null && continue
    out=$root/$arm; mkdir -p $out
    RESUME=""; AL=""; [ -f $out/last.pt ] || { RESUME="--resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt"; AL="--al-reset"; }
    $PY -u scripts/ib/train_fly_bptt_stream.py $RESUME $AL $COMMON $(extra $arm) --output $out --checkpoint-dir $out >> $out/train.log 2>&1
    code=$?; echo "$(date +%H:%M) chunk $chunk $arm train $code" >> $root/status; rm -f $out/best.pt
    [ $code -ne 0 ] && break 2
  done
done
for arm in $ARMS; do
  ACT_NOISY= ACT_MODES=open $PY -u scripts/ib/fly_interface/act_probe.py $root/$arm $root/$arm/act_probe.jsonl > $root/$arm/act_probe.txt 2>&1
  $PY -u scripts/ib/fly_interface/rarity_decode_probe.py $root/$arm $root/$arm/rarity_decode.json > $root/$arm/rarity_decode.txt 2>&1
  $PY -u scripts/ib/fly_interface/word_novelty_probe.py $root/$arm $root/$arm/word_novelty.json > $root/$arm/word_novelty.txt 2>&1
  $PY -u scripts/ib/fly_interface/commit_tail.py $root/$arm $root/$arm/commit_tail.json > $root/$arm/commit_tail.txt 2>&1
done
echo "$(date +%H:%M) done" >> $root/status
