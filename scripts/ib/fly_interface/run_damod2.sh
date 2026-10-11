#!/bin/bash
# Dopamine as a connectome neuromodulator (no imposed intrinsic current, no ad-hoc precision read) + predictive writer,
# mixed world; paired with pc_gb.  Checkpoint kept.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; SRC=E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt
out=scratch/fly_stream/damod2_gb
FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8 $PY -u scripts/ib/train_fly_bptt_stream.py --resume $SRC --extend-budget \
  --additional-tokens 4096 --validate-every-tokens 1024 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph \
  --go-neuron virtual --read-bank --predictive-writer --dopamine-modulation --mixed-world --output $out --checkpoint-dir $out > $out.log 2>&1
echo "damod2 exit $?" >> scratch/fly_stream/status_damod2
rm -f $out/best.pt
echo done >> scratch/fly_stream/status_damod2
