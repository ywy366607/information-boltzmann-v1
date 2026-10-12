#!/bin/bash
# The online arm stopped at 8,224 tokens (VRAM budget: cached blocks, now released near the limit); finish it to the control's 16,416 with the same world sequence
# (skip the 256 windows of worlds already drawn), then measure both arms.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe
out=scratch/fly_stream/b2_online
FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8 $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 8160 --vram-limit-mib 4050 \
  --world-skip 257 --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph \
  --go-neuron virtual --read-bank --predictive-writer --dopamine-modulation --dopamine-site pre --dopamine-affinity 0.000554 \
  --kc-dendritic-only --homeostasis --online-da-plasticity --mixed-world --output $out --checkpoint-dir $out >> scratch/fly_stream/b2_online.log 2>&1
echo "online resume exit $?" >> scratch/fly_stream/status_b2
rm -f $out/best.pt
for arm in control online; do
  o=scratch/fly_stream/b2_$arm
  for probe in online_revisit mb_role kc_recurrence dopamine_response precision_probe; do
    PYTHONIOENCODING=utf-8 $PY -u scratch/fly_stream/$probe.py $o > $o/$probe.txt 2>&1
    echo "$arm $probe exit $?" >> scratch/fly_stream/status_b2
  done
done
echo done >> scratch/fly_stream/status_b2
