#!/bin/bash
# M1 (GOAL: docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261013.md): the adult with the connectome v2 fixes
# in a world-paced text stream (mu ~ log-uniform [2, 16] per window), eyes only: survival-weighted CE + SIGReg on the
# read latents + the epistemic value of looking (information gain).  Ablation arm eyes_noepi: no epistemic term (the
# language objective alone decides the eyes).  Control arm: eyes forced open.  Chunks of 4,096 words, arms
# alternating (one torch process at a time); after the last chunk the eyes probe on each.  STOP in the root stops it.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; root=/e/fly_stream_overflow/m1_eyes; mkdir -p $root/eyes $root/eyes_noepi $root/open
CHUNKS=${1:-3}
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
COMMON="--extend-budget --additional-tokens 4096 --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 \
  --window 32 --no-graph --go-neuron none --stream-ticks 8 --no-hold-input --stream-pace-mean 16 --stream-pace-min 2 \
  --no-homeostasis --no-ctm-loss --kc-dendritic-only --kc-renorm --connectome-v2 --dopamine-modulation \
  --dopamine-site pre --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 \
  --ai --ai-pragmatic 0.0 --ai-sigreg 0.01 --vram-limit-mib 3800"
for chunk in $(seq 1 $CHUNKS); do
  for arm in eyes eyes_noepi open; do
    [ -f $root/STOP ] && break 2
    out=$root/$arm
    case $arm in eyes) EXTRA="--ai-epistemic 1.0";; eyes_noepi) EXTRA="--ai-epistemic 0.0";; open) EXTRA="--ai-epistemic 1.0 --eye-forced-open";; esac
    RESUME=""; AL=""; [ -f $out/last.pt ] || { RESUME="--resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt"; AL="--al-reset"; }
    $PY -u scripts/ib/train_fly_bptt_stream.py $RESUME $AL $COMMON $EXTRA --output $out --checkpoint-dir $out >> $out/train.log 2>&1
    code=$?; echo "$(date +%H:%M) chunk $chunk $arm train $code" >> $root/status; rm -f $out/best.pt
    [ $code -ne 0 ] && break 2
  done
done
for arm in eyes eyes_noepi open; do
  $PY -u scripts/ib/fly_interface/eyes_probe.py $root/$arm $root/$arm/eyes_probe.jsonl > $root/$arm/eyes_probe.txt 2>&1
done
echo "$(date +%H:%M) done" >> $root/status
