#!/bin/bash
# D2 (GOAL: docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261013.md, section D2): does evolution grow evidence
# accumulation once short-term plasticity may learn and the Kenyon subtypes have their own physiology?  Sustained noisy
# input (each word heard every tick, sigma ~ U(0, 2)), commits from motor cells (the D1 setup: listener-prior miss,
# habit prior, readout learns), eyes open.  Arms: d2_fixed (--learn-stp --kc-subtypes), d2_control (no fixes).
# Chunks of 4,096 words, arms alternating (one torch process at a time); then the act probe on each.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; root=/e/fly_stream_overflow/d2_accumulate
ARMS=${ARMS:-"d2_fixed d2_control"}
CHUNKS=${1:-3}
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
COMMON="--extend-budget --additional-tokens 4096 --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 \
  --window 32 --no-graph --go-neuron none --stream-ticks 8 --hold-input --sense-noise 2.0 \
  --stream-pace-mean 16 --stream-pace-min 2 \
  --no-homeostasis --no-ctm-loss --kc-dendritic-only --kc-renorm --connectome-v2 --dopamine-modulation \
  --dopamine-site pre --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 \
  --ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior --ai-epistemic 0.0 \
  --ai-pragmatic 1.0 --ai-sigreg 0.01 --eye-forced-open --vram-limit-mib 3800"
for chunk in $(seq 1 $CHUNKS); do
  for arm in $ARMS; do
    [ -f $root/STOP ] && break 2
    grep -q "chunk $chunk $arm train 0" $root/status 2>/dev/null && continue
    out=$root/$arm; mkdir -p $out
    case $arm in
      d2_fixed)   EXTRA="--learn-stp --kc-subtypes";;
      d2_control) EXTRA="";;
    esac
    RESUME=""; AL=""; [ -f $out/last.pt ] || { RESUME="--resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt"; AL="--al-reset"; }
    $PY -u scripts/ib/train_fly_bptt_stream.py $RESUME $AL $COMMON $EXTRA --output $out --checkpoint-dir $out >> $out/train.log 2>&1
    code=$?; echo "$(date +%H:%M) chunk $chunk $arm train $code" >> $root/status; rm -f $out/best.pt
    [ $code -ne 0 ] && break 2
  done
done
for arm in $ARMS; do
  ACT_NOISY= ACT_MODES=open ACT_MUS=8,4 $PY -u scripts/ib/fly_interface/act_probe.py $root/$arm $root/$arm/act_probe.jsonl > $root/$arm/act_probe.txt 2>&1
done
echo "$(date +%H:%M) done" >> $root/status
