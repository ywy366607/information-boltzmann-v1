#!/bin/bash
# D3 (GOAL: docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261013.md, section D3): D1's setup (commits sampled
# from motor cells, exact expectation over commit times, listener-prior miss, habit prior; pulse world; eyes open) but
# the policy gradient reaches the brain through the motor spikes (--act-brain-grad), as GRPO updates the whole network.
# The answers' values stay detached (beliefs are trained by the language objective only).  Compare with D1's d1_habit.
# Arm d3_precision adds the policy precision encoded by the brain's own dopamine cells (--act-dopamine-precision,
# GOAL section 'D3b'); arm d3_credit adds credit beyond the window at O(1) (--carried-credit, GOAL section 'Credit
# beyond the 32-word window'); d3_braingrad is the control of both.  Before the run: the tests (incl. the GPU ones)
# on a free GPU -- if they fail only the control runs -- and a smoke (start + resume) of each new arm.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; root=/e/fly_stream_overflow/d3_braingrad
ARMS=${ARMS:-"d3_braingrad d3_precision d3_credit"}
CHUNKS=${1:-3}
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
COMMON="--extend-budget --additional-tokens 4096 --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 \
  --window 32 --no-graph --go-neuron none --stream-ticks 8 --no-hold-input --stream-pace-mean 16 --stream-pace-min 2 \
  --no-homeostasis --no-ctm-loss --kc-dendritic-only --kc-renorm --connectome-v2 --dopamine-modulation \
  --dopamine-site pre --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 \
  --ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.0 --miss-prior --ai-epistemic 0.0 \
  --ai-pragmatic 1.0 --ai-sigreg 0.01 --eye-forced-open --act-brain-grad --vram-limit-mib 3800"
extra() { case $1 in d3_precision) echo "--act-dopamine-precision";; d3_credit) echo "--carried-credit";; esac; }
mkdir -p $root
if $PY -m pytest -q tests/test_fly_adaptive_admission.py > $root/tests.txt 2>&1; then
  echo "$(date +%H:%M) tests passed" >> $root/status
else
  echo "$(date +%H:%M) tests FAILED (see tests.txt): running the control only" >> $root/status; ARMS=d3_braingrad
fi
# a smoke of every new arm (from the adult, 64 words, then a resume of that): a failing arm is dropped
keep=""
for arm in $ARMS; do
  if [ -z "$(extra $arm)" ] || [ -f $root/$arm/last.pt ]; then keep="$keep $arm"; continue; fi
  smoke=$root/smoke_$arm; rm -rf $smoke
  S="$COMMON $(extra $arm) --additional-tokens 64 --validate-every-tokens 64 --eval-tokens 32 --output $smoke --checkpoint-dir $smoke"
  if $PY -u scripts/ib/train_fly_bptt_stream.py --resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt --al-reset $S > $root/smoke_$arm.log 2>&1 \
     && $PY -u scripts/ib/train_fly_bptt_stream.py $S >> $root/smoke_$arm.log 2>&1; then
    keep="$keep $arm"; grep -h "dopamine tone\|Carried credit" $root/smoke_$arm.log | head -2 >> $root/status
  else
    echo "$(date +%H:%M) $arm smoke FAILED (see smoke_$arm.log): dropped" >> $root/status
  fi
  rm -rf $smoke
done
ARMS=$keep
echo "$(date +%H:%M) arms:$ARMS" >> $root/status
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
  $PY -u scripts/ib/fly_interface/memory_revisit.py $root/$arm $root/$arm/memory_revisit.json > $root/$arm/memory_revisit.txt 2>&1
done
# the same O(1)-memory probe on D1 (before the brain gradient) and on D2's arms, for comparison
for f in /e/fly_stream_overflow/d1_commits/d1_habit /e/fly_stream_overflow/d2_accumulate/d2_fixed /e/fly_stream_overflow/d2_accumulate/d2_control; do
  [ -f $f/last.pt ] && [ ! -f $f/memory_revisit.json ] && $PY -u scripts/ib/fly_interface/memory_revisit.py $f $f/memory_revisit.json > $f/memory_revisit.txt 2>&1
done
echo "$(date +%H:%M) done" >> $root/status
