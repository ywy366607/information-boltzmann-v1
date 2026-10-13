#!/bin/bash
# R1 (GOAL: docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261013.md, section R1): a newborn on the corrected
# engine (Shiu-validated synapse scale, equal-area inhibition, spike semantics; no STP; graded APL / CT1; literature
# transmitter of the antennal-lobe local neurons; the connectome fixes from the first tick) learns the language
# stream.  Chunk 1 is the birth, followed by the gate probes; then chunks to CHUNKS x 4096 words, probes and a
# snapshot at chunk 5 (20k words) and at the end.  One GPU process at a time; do not edit core modules while it runs
# (each chunk reloads the code).
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; root=/e/fly_stream_overflow/r1_rebirth; ckpt=/e/ib_checkpoints/r1_rebirth
CHUNKS=${1:-25}
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
R1="--window 32 --no-graph --go-neuron none --stream-ticks 8 --no-hold-input --stream-pace-mean 16 --stream-pace-min 2 \
  --no-homeostasis --no-ctm-loss --kc-dendritic-only --connectome-v2 --dopamine-modulation --dopamine-site pre \
  --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 --kc-subtypes --modulatory-classes --intrinsic \
  --graded-types APL,CT1 --graded-gain 0.1 --lr-physiology 5e-3 --carried-credit --vram-limit-mib 3800 \
  --graph data/malecns_v1/fly_reservoir_coba_shiu.npz --E-E 0.55 --E-I -0.15 --spike-consumes-drive \
  --refractory-ticks 1 --no-stp --surrogate-mode threshold --detach-reset --transmission-mode incoming \
  --writer-baseline-clock physical --validate-every-tokens 4096 --eval-tokens 512 --replay-tokens 128 \
  --output $root --checkpoint-dir $ckpt"
probes() {   # $1 = label
  $PY -u scripts/ib/fly_interface/cell_health_probe.py $ckpt $root/cell_health_$1.json > $root/cell_health_$1.txt 2>&1
  $PY -u scripts/ib/fly_interface/kenyon_coding_probe.py $ckpt > $root/coding_$1.txt 2>&1
  TASTE_RATE=100 TASTE_ON=200 TASTE_AFTER=0 $PY -u scripts/ib/fly_interface/taste_pulse_probe.py $ckpt $root/taste_$1.json > $root/taste_$1.txt 2>&1
}
mkdir -p $root $ckpt
if [ ! -f $ckpt/last.pt ]; then
  $PY -u scripts/ib/train_fly_bptt_stream.py --from-scratch --additional-tokens 4096 $R1 >> $root/train.log 2>&1 \
    || { echo "$(date +%H:%M) birth FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) born: chunk 1 done" >> $root/status
  probes c1
  echo "$(date +%H:%M) gate probes c1 written" >> $root/status
fi
done_chunks=$(grep -c "chunk [0-9]* done" $root/status)
for ((c=done_chunks+1; c<=CHUNKS; c++)); do
  $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 $R1 >> $root/train.log 2>&1 \
    || { echo "$(date +%H:%M) chunk $c FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) chunk $c done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1)" >> $root/status
  if [ $c -eq 5 ] || [ $c -eq $CHUNKS ]; then
    cp $ckpt/last.pt $ckpt/c$c.pt
    probes c$c
    echo "$(date +%H:%M) probes c$c written, snapshot c$c.pt" >> $root/status
  fi
done
echo "$(date +%H:%M) R1 finished" >> $root/status
