#!/bin/bash
# Layer 1 comparison: two forks of dasite_pre (pre segment 1), same data, 8192 tokens each; control vs the O(1) online
# dopamine rule at the Kenyon terminals; both arms with Kenyon axo-axonal synapses pinned at 0 (--kc-dendritic-only), so the
# difference is the online rule alone.  Then the measurements on both.  STOP file in either fork dir to stop.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe
for arm in control online; do
  out=scratch/fly_stream/l1_$arm
  mkdir -p $out; [ -f $out/last.pt ] || cp scratch/fly_stream/dasite_pre/last.pt scratch/fly_stream/dasite_pre/config.json $out/
  [ -f $out/STOP ] && continue
  extra=""; [ $arm = online ] && extra="--online-da-plasticity"
  FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8 $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget \
    --additional-tokens 8192 --validate-every-tokens 2048 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph \
    --go-neuron virtual --read-bank --predictive-writer --dopamine-modulation --dopamine-site pre --kc-dendritic-only $extra --mixed-world \
    --output $out --checkpoint-dir $out > $out.log 2>&1
  echo "$arm train exit $?" >> scratch/fly_stream/status_l1
  rm -f $out/best.pt
done
for arm in control online; do
  out=scratch/fly_stream/l1_$arm
  for probe in online_revisit dopamine_response dopamine_states precision_probe kc_recurrence; do
    PYTHONIOENCODING=utf-8 $PY -u scratch/fly_stream/$probe.py $out > $out/$probe.txt 2>&1
    echo "$arm $probe exit $?" >> scratch/fly_stream/status_l1
  done
done
echo done >> scratch/fly_stream/status_l1
