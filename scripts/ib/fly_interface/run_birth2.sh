#!/bin/bash
# Grown from birth, second try: both newborns start from the same gestated state (homeostatic intrinsic plasticity has
# woken the silent cells with fixed weights), Kenyon axo-axonal synapses pinned at 0, homeostasis on for life, receptor
# affinity K from the gestated tonic dopamine (passed as $1).  One arm with the O(1) online dopamine rule.  16,384 tokens
# each, same data and worlds; then the measurements.  STOP file in an arm dir to stop it.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe
K=$1
for arm in control online; do
  out=scratch/fly_stream/b2_$arm
  [ -f $out/STOP ] && continue
  mkdir -p $out; [ -f $out/last.pt ] || cp scratch/fly_stream/gestated/last.pt scratch/fly_stream/gestated/config.json $out/
  extra=""; [ $arm = online ] && extra="--online-da-plasticity"
  FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8 $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 16384 \
    --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph \
    --go-neuron virtual --read-bank --predictive-writer --dopamine-modulation --dopamine-site pre --dopamine-affinity $K \
    --kc-dendritic-only --homeostasis $extra --mixed-world --output $out --checkpoint-dir $out > $out.log 2>&1
  echo "$arm train exit $?" >> scratch/fly_stream/status_b2
  rm -f $out/best.pt
done
for arm in control online; do
  out=scratch/fly_stream/b2_$arm
  for probe in online_revisit mb_role kc_recurrence dopamine_response precision_probe; do
    PYTHONIOENCODING=utf-8 $PY -u scratch/fly_stream/$probe.py $out > $out/$probe.txt 2>&1
    echo "$arm $probe exit $?" >> scratch/fly_stream/status_b2
  done
done
echo done >> scratch/fly_stream/status_b2
