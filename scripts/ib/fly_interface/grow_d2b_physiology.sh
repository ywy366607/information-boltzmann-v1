#!/bin/bash
# D2b (GOAL section 'D2b'): D2 continued with the class physiology on its own relative rate (--lr-physiology 5e-3,
# 25x the shared 2e-4).  D2 moved STP by <= 0.016 in log / logit space in two chunks: a step-budget null, not an
# answer.  Both arms continue their own D2 lineage (same world, same flags) for 3 chunks; then the act probe.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; root=/e/fly_stream_overflow/d2b_physiology; src=/e/fly_stream_overflow/d2_accumulate
ARMS=${ARMS:-"d2b_fixed d2b_control"}
CHUNKS=${1:-3}
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
COMMON="--extend-budget --additional-tokens 4096 --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 \
  --window 32 --no-graph --go-neuron none --stream-ticks 8 --hold-input --sense-noise 2.0 \
  --stream-pace-mean 16 --stream-pace-min 2 \
  --no-homeostasis --no-ctm-loss --kc-dendritic-only --kc-renorm --connectome-v2 --dopamine-modulation \
  --dopamine-site pre --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 \
  --ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior --ai-epistemic 0.0 \
  --ai-pragmatic 1.0 --ai-sigreg 0.01 --eye-forced-open --vram-limit-mib 3800 --lr-physiology 5e-3"
for chunk in $(seq 1 $CHUNKS); do
  for arm in $ARMS; do
    [ -f $root/STOP ] && break 2
    grep -q "chunk $chunk $arm train 0" $root/status 2>/dev/null && continue
    out=$root/$arm; mkdir -p $out
    case $arm in
      d2b_fixed)   EXTRA="--learn-stp --kc-subtypes";;
      d2b_control) EXTRA="";;
    esac
    RESUME=""; AL=""; [ -f $out/last.pt ] || RESUME="--resume $src/${arm/d2b_/d2_}/last.pt"
    $PY -u scripts/ib/train_fly_bptt_stream.py $RESUME $AL $COMMON $EXTRA --output $out --checkpoint-dir $out >> $out/train.log 2>&1
    code=$?; echo "$(date +%H:%M) chunk $chunk $arm train $code" >> $root/status; rm -f $out/best.pt
    [ $code -ne 0 ] && break 2
  done
done
for arm in $ARMS; do
  ACT_NOISY= ACT_MODES=open ACT_MUS=8,4 $PY -u scripts/ib/fly_interface/act_probe.py $root/$arm $root/$arm/act_probe.jsonl > $root/$arm/act_probe.txt 2>&1
done
echo "$(date +%H:%M) done" >> $root/status
