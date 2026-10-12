#!/bin/bash
# M2 (GOAL: docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261013.md): output only.  Every arriving word is a
# question ("what comes next?"); the output commits the read standing then (value -CE) or the next word arrives first
# (-log V).  Eyes forced open; otherwise identical to M1's forced-open arm, which is the no-pragmatic ablation.
# Chunks of 4,096 words; then the act probe on this arm and on M1's forced-open arm.  STOP in the root stops it.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; root=/e/fly_stream_overflow/m2_output; out=$root/out; mkdir -p $out
CHUNKS=${1:-3}
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
COMMON="--extend-budget --additional-tokens 4096 --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 \
  --window 32 --no-graph --go-neuron none --stream-ticks 8 --no-hold-input --stream-pace-mean 16 --stream-pace-min 2 \
  --no-homeostasis --no-ctm-loss --kc-dendritic-only --kc-renorm --connectome-v2 --dopamine-modulation \
  --dopamine-site pre --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 \
  --ai --ai-epistemic 1.0 --ai-pragmatic 1.0 --ai-sigreg 0.01 --eye-forced-open --vram-limit-mib 3800"
for chunk in $(seq 1 $CHUNKS); do
  [ -f $root/STOP ] && break
  RESUME=""; AL=""; [ -f $out/last.pt ] || { RESUME="--resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt"; AL="--al-reset"; }
  $PY -u scripts/ib/train_fly_bptt_stream.py $RESUME $AL $COMMON --output $out --checkpoint-dir $out >> $out/train.log 2>&1
  code=$?; echo "$(date +%H:%M) chunk $chunk out train $code" >> $root/status; rm -f $out/best.pt
  [ $code -ne 0 ] && break
done
ACT_NOISY= ACT_MODES=open $PY -u scripts/ib/fly_interface/act_probe.py $out $out/act_probe.jsonl > $out/act_probe.txt 2>&1
ACT_NOISY= ACT_MODES=open $PY -u scripts/ib/fly_interface/act_probe.py /e/fly_stream_overflow/m1_eyes/open $root/m1_open_act_probe.jsonl > $root/m1_open_act_probe.txt 2>&1
echo "$(date +%H:%M) done" >> $root/status
