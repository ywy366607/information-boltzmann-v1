#!/bin/bash
# Champion replica (user 2026-10-13): the champion lineage's early life -- old birth graph (echo-state scale), old
# engine defaults, event world (one pulse, 4 settle ticks), window 8, CTM loss, per-parameter Adam on everything,
# frozen GPT-2 embedding -- grown from scratch to 12k words.  Does the champion's line also latch / have a negative
# brain contribution early?  Measured at 4k and 12k exactly as the new individuals (latched cells, decoder alone vs
# brain, prequential gain over the fixed unigram).
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; root=/e/fly_stream_overflow/champion_replica; ckpt=/e/ib_checkpoints/champion_replica
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
F="--profile none --graph data/malecns_v1/fly_reservoir_coba.npz --settle-ticks 4 --window 8 --ctm-loss --surrogate-mode threshold --detach-reset --transmission-mode incoming --writer-baseline-clock physical --go-neuron none --no-graph --validate-every-tokens 4096 --eval-tokens 512 --replay-tokens 128 --vram-limit-mib 3800 --output $root --checkpoint-dir $ckpt"
measure() {
  $PY -u scripts/ib/fly_interface/text_leak_probe.py $ckpt > $root/leak_$1.txt 2>&1
  for sw in none noedges; do
    NLL_TICKS=5 NLL_PLACES=2 NLL_SCORE=512 SWAP=$([ $sw = none ] || echo $sw) BIRTH=$ckpt/birth.pt $PY -u scripts/ib/fly_interface/swap_nll_probe.py $ckpt 2>&1 | grep "^{" >> $root/brain_$1.txt
  done
  echo "$(date +%H:%M) $1: $(grep -o 'latched ([^;]*' $root/leak_$1.txt | head -1); $(grep 'brain mean' $root/leak_$1.txt); $(tr '\n' ' ' < $root/brain_$1.txt | grep -o '"swap": [^,]*, [^}]*nll_places_min_median_mean_max": \[[^]]*\]' | tr '\n' ' ')" >> $root/status
}
mkdir -p $root $ckpt
$PY -u scripts/ib/train_fly_bptt_stream.py --from-scratch --additional-tokens 4096 $F >> $root/train.log 2>&1 || { echo "birth FAILED" >> $root/status; exit 1; }
echo "$(date +%H:%M) c1 done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1)" >> $root/status
measure c1
for c in 2 3; do
  $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 $F >> $root/train.log 2>&1 || { echo "chunk $c FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) c$c done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1)" >> $root/status
done
measure c3
echo "$(date +%H:%M) replica done" >> $root/status
