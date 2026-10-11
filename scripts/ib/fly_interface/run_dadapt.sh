#!/bin/bash
# B: adapted dopamine receptors (each target sees its own relative change, half-saturating at the tonic level) with a
# working starting sensitivity 2 (half the stability edge: 4 makes BPTT gradients diverge, 3 does not), presynaptic
# plasticity site.  Forks damod2_gb
# (4096 tokens) like damod_long (post) and dasite_pre (pre); segments of 8192 tokens, measured after each.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe
out=scratch/fly_stream/dadapt
mkdir -p $out; [ -f $out/last.pt ] || cp scratch/fly_stream/damod2_gb/last.pt scratch/fly_stream/damod2_gb/config.json $out/
for seg in ${SEGS:-1}; do
  [ -f $out/STOP ] && break
  init=""; [ "$seg" = 1 ] && init="--dopamine-sensitivity-init 2.0"
  FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8 $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget \
    --additional-tokens 8192 --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph \
    --go-neuron virtual --read-bank --predictive-writer --dopamine-modulation --dopamine-site pre --dopamine-adapted $init \
    --mixed-world --output $out --checkpoint-dir $out >> $out.log 2>&1
  echo "segment $seg train exit $?" >> scratch/fly_stream/status_dadapt
  rm -f $out/best.pt
  for probe in dopamine_response dopamine_states dan_input precision_probe; do
    PYTHONIOENCODING=utf-8 $PY -u scratch/fly_stream/$probe.py $out > $out/seg${seg}_$probe.txt 2>&1
    cp $out/$probe.json $out/seg${seg}_$probe.json
  done
  echo "segment $seg measured" >> scratch/fly_stream/status_dadapt
done
echo done >> scratch/fly_stream/status_dadapt
