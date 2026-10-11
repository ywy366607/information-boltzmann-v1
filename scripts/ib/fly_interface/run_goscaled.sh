#!/bin/bash
# Virtual go (scale-matched lr for its scalars) + read bank, 4096 tokens; paired with base_gb; checkpoint kept.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; SRC=E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt
out=scratch/fly_stream/goscaled_gb
PYTHONIOENCODING=utf-8 $PY -u scripts/ib/train_fly_bptt_stream.py --resume $SRC --extend-budget --additional-tokens 4096 \
  --validate-every-tokens 1024 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph --go-neuron virtual --read-bank \
  --output $out --checkpoint-dir $out > $out.log 2>&1
echo "goscaled exit $?" >> scratch/fly_stream/status_gs
rm -f $out/best.pt
echo done >> scratch/fly_stream/status_gs
