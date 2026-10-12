#!/bin/bash
# M3 (GOAL: docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261013.md, sections M3 and 'M3 world revised'): both
# acts in the sustained noisy world (each word heard every tick as a noisy sample, sigma ~ U(0, 2)).  Binary eyes,
# questions kept open while the eyes are closed (depth 2), the noisy TV (0.25 per window, 8 words, 16-token alphabet).
# Arms: full, open (eyes forced open), passive (eyes forced open, no pragmatic: the language reference), noepi, noprag,
# depth0 (questions close at the next main onset whatever the eyes do: attributes closures to the kept-open rule).
# Eyes: a look sampled from the policy in training, open probability 0.9 at the start.
# Chunks of 4,096 words, arms alternating (one torch process at a time); then the act probe on every arm.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; root=/e/fly_stream_overflow/m3_sustained
ARMS=${ARMS:-"full open passive noepi noprag depth0"}
CHUNKS=${1:-3}
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
COMMON="--extend-budget --additional-tokens 4096 --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 \
  --window 32 --no-graph --go-neuron none --stream-ticks 8 --stream-pace-mean 16 --stream-pace-min 2 \
  --no-homeostasis --no-ctm-loss --kc-dendritic-only --kc-renorm --connectome-v2 --dopamine-modulation \
  --dopamine-site pre --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 \
  --ai --eye-binary --eye-init 0.9 --ai-depth 2 --ai-sigreg 0.01 --noise-rate 0.25 --hold-input --sense-noise 2.0 \
  --vram-limit-mib 3800"
for chunk in $(seq 1 $CHUNKS); do
  for arm in $ARMS; do
    [ -f $root/STOP ] && break 2
    grep -q "chunk $chunk $arm train 0" $root/status 2>/dev/null && continue      # already done (pilot)
    out=$root/$arm; mkdir -p $out
    case $arm in
      full)    EXTRA="--ai-epistemic 1.0 --ai-pragmatic 1.0";;
      open)    EXTRA="--ai-epistemic 1.0 --ai-pragmatic 1.0 --eye-forced-open";;
      passive) EXTRA="--ai-epistemic 1.0 --ai-pragmatic 0.0 --eye-forced-open";;
      noepi)   EXTRA="--ai-epistemic 0.0 --ai-pragmatic 1.0";;
      noprag)  EXTRA="--ai-epistemic 1.0 --ai-pragmatic 0.0";;
      depth0)  EXTRA="--ai-epistemic 1.0 --ai-pragmatic 1.0 --ai-depth 0";;
    esac
    RESUME=""; AL=""; [ -f $out/last.pt ] || { RESUME="--resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt"; AL="--al-reset"; }
    $PY -u scripts/ib/train_fly_bptt_stream.py $RESUME $AL $COMMON $EXTRA --output $out --checkpoint-dir $out >> $out/train.log 2>&1
    code=$?; echo "$(date +%H:%M) chunk $chunk $arm train $code" >> $root/status; rm -f $out/best.pt
    [ $code -ne 0 ] && break 2
  done
done
for arm in $ARMS; do
  MODES=own,open; case $arm in open|passive) MODES=open;; esac      # forced-open arms: their eyes never acted
  ACT_SCORE=512 ACT_MODES=$MODES $PY -u scripts/ib/fly_interface/act_probe.py $root/$arm $root/$arm/act_probe.jsonl > $root/$arm/act_probe.txt 2>&1
done
echo "$(date +%H:%M) done" >> $root/status
