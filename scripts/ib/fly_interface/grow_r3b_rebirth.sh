#!/bin/bash
# R3b: R3 with evidence-gated plasticity (2026-10-13): a synapse or class physiology parameter steps only when the running mean of its gradient is significant (> 3 standard errors, synaptic tagging / consolidation), slow pull to birth (tau 4096 updates), dopamine plasticity on the phasic change (adapted), conductance cap never cuts a birth synapse.  Cause: in R3 Adam turned gradient noise into a random walk and latching is an absorbing side of it (latched 6 -> 1,159 -> 9,520; learned substrate carried no language by ablation).  Early stop: latched cells > 200 at any chunk.
# receptor type, visual / mechanosensory per cell; bounded transduction conductance around spontaneous firing, held for
# the word).  The brain learns which receptors the text uses (user 2026-10-13: shared senses stay learnable, as in
# sensory substitution).  Principle from here on: hand-fix only mechanism problems; numbers are learned; learned
# structure is judged by causal ablation, not by fly-likeness.  Gate probes record, they do not tune.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; root=/e/fly_stream_overflow/r3b_rebirth; ckpt=/e/ib_checkpoints/r3b_rebirth
CHUNKS=${1:-25}
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
R2="--window 32 --no-graph --go-neuron none --stream-ticks 8 --hold-input --writer receptor_all --stream-pace-mean 16 --stream-pace-min 2 --no-homeostasis --no-ctm-loss --kc-dendritic-only --connectome-v2 --pin-modulatory-types --dopamine-modulation --dopamine-site pre --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 --type-classes --intrinsic --intrinsic-jitter 0.1 --graded-types APL,CT1 --graded-gain 0.025 --lr-physiology 5e-3 --carried-credit --evidence-plasticity --pull-tau 4096 --dopamine-adapted --vram-limit-mib 3800 --graph data/malecns_v1/fly_reservoir_coba_r2.npz --E-E 0.55 --E-I -0.15 --spike-consumes-drive --refractory-ticks 1 --inhibitory-tau-ratio 4 --no-stp --alif-beta-init 0.2 --surrogate-mode threshold --detach-reset --transmission-mode incoming --writer-baseline-clock physical --validate-every-tokens 4096 --eval-tokens 512 --replay-tokens 128 --output $root --checkpoint-dir $ckpt"
probes() {   # $1 = label
  $PY -u scripts/ib/fly_interface/cell_health_probe.py $ckpt $root/cell_health_$1.json > $root/cell_health_$1.txt 2>&1
  $PY -u scripts/ib/fly_interface/kenyon_coding_probe.py $ckpt > $root/coding_$1.txt 2>&1
  TASTE_RATE=100 TASTE_ON=200 TASTE_AFTER=0 $PY -u scripts/ib/fly_interface/taste_pulse_probe.py $ckpt $root/taste_$1.json > $root/taste_$1.txt 2>&1
  $PY -u scripts/ib/fly_interface/text_leak_probe.py $ckpt > $root/leak_$1.txt 2>&1
}
gate() {     # $1 = label: latched cells (>200 Hz, words withheld) after every chunk; stop when > 200
  $PY -u scripts/ib/fly_interface/text_leak_probe.py $ckpt > $root/leak_$1.txt 2>&1
  n=$(grep -o "latched (> 200 Hz without words, >= 2 of 4 places): [0-9]*" $root/leak_$1.txt | grep -o "[0-9]*$")
  echo "$(date +%H:%M) gate $1: latched ${n:-?}, $(grep 'brain mean' $root/leak_$1.txt)" >> $root/status
  if [ -z "$n" ] || [ "$n" -gt 200 ]; then echo "$(date +%H:%M) EARLY STOP at $1: latched ${n:-?} > 200" >> $root/status; exit 2; fi
}
mkdir -p $root $ckpt
if [ ! -f $ckpt/last.pt ]; then
  $PY -u scripts/ib/train_fly_bptt_stream.py --from-scratch --additional-tokens 4096 $R2 >> $root/train.log 2>&1 \
    || { echo "$(date +%H:%M) birth FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) born: chunk 1 done" >> $root/status
  probes c1
  echo "$(date +%H:%M) gate probes c1 written" >> $root/status
  gate c1
fi
done_chunks=$(grep -c "chunk [0-9]* done" $root/status)
for ((c=done_chunks+1; c<=CHUNKS; c++)); do
  $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 $R2 >> $root/train.log 2>&1 \
    || { echo "$(date +%H:%M) chunk $c FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) chunk $c done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1)" >> $root/status
  gate c$c
  if [ $c -eq 5 ] || [ $c -eq $CHUNKS ]; then
    cp $ckpt/last.pt $ckpt/c$c.pt
    probes c$c
    # does the learned substrate carry the language?  (put the synapses / everything back to birth)
    for sw in none edges_e,edges_i model; do
      SWAP=$([ $sw = none ] || echo $sw) BIRTH=$ckpt/birth.pt $PY -u scripts/ib/fly_interface/swap_nll_probe.py $ckpt 2>&1 | grep "^{" >> $root/swap_nll_c$c.txt
    done
    echo "$(date +%H:%M) probes c$c written, snapshot c$c.pt" >> $root/status
  fi
done
echo "$(date +%H:%M) R3b finished" >> $root/status
