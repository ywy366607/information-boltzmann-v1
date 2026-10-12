#!/bin/bash
# Queued after grow_odour_arms.sh (waits for its 'done'): (1) tops the active arm up to 5 chunks (it was restarted once);
# (2) the electrode-free arm 'taste_only': identical to the active arm except that contact drives only the taste
# receptors -- no direct drive of the dopamine cells; whether the dopamine cells answer sugar is the brain's own
# (evolved) business; (3) the novelty -> dopamine probe on the adult (with the fixes), the active and the taste-only
# arms, and the heavy-tail test of their act intervals.  One torch process at a time.  STOP in the base dir stops.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; BASE=/e/fly_stream_overflow/odour_arms
until grep -q '^done' $BASE/status 2>/dev/null; do sleep 60; done
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
COMMON="--extend-budget --additional-tokens 4096 --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 \
  --window 32 --no-graph --go-neuron none --stream-ticks 8 --no-hold-input --stream-pace-mean 16 --stream-pace-min 4 \
  --no-homeostasis --no-ctm-loss --kc-dendritic-only --kc-renorm --dopamine-modulation --dopamine-site pre \
  --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 --vram-limit-mib 4050"
run_arm() {   # arm, target chunks, extra flags
  local arm=$1 target=$2; shift 2
  local out=$BASE/$arm; mkdir -p $out
  while true; do
    [ -f $BASE/STOP ] && exit 0
    local done_chunks=$(grep -c ' train 0' $out/status 2>/dev/null); done_chunks=${done_chunks:-0}
    [ $done_chunks -ge $target ] && break
    local chunk=$((done_chunks + 1))
    local RESUME=""; [ -f $out/last.pt ] || RESUME="--resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt"
    $PY -u scripts/ib/train_fly_bptt_stream.py $RESUME $COMMON "$@" --output $out --checkpoint-dir $out >> $out/train.log 2>&1
    local code=$?; echo "$(date +%H:%M) chunk $chunk train $code" >> $out/status; rm -f $out/best.pt
    [ $code -ne 0 ] && exit 1
    $PY -u scripts/ib/fly_interface/pace_zero_shot.py $out $out/pace_probe.jsonl > $out/pace_probe_$chunk.txt 2>&1
    echo "$(date +%H:%M) chunk $chunk probe $?" >> $out/status
    $PY scripts/ib/fly_interface/odour_summary.py $out/odour.jsonl > $out/odour_summary.txt 2>&1
  done
}
run_arm active 5 --odour-world --act-lambda 0.0003
run_arm taste_only 5 --odour-world --act-lambda 0.0003 --no-outcome-dopamine
# probes
R=results/published/fly_consequence_20261012
for arm in active taste_only; do
  $PY -u scripts/ib/fly_interface/novelty_probe.py $BASE/$arm $R/novelty_probe.jsonl > $BASE/$arm/novelty_probe.txt 2>&1
  $PY scripts/ib/fly_interface/act_intervals.py $BASE/$arm/odour_acts.jsonl > $BASE/$arm/act_intervals.json 2>&1
done
echo "$(date +%H:%M) taste-only chain done" >> $BASE/status
