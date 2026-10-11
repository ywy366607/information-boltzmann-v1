#!/bin/bash
# Step 2: mixed world + released active cells + neuromodulatory precision (attention + dopamine-set plasticity);
# paired with mixed_gb (step 1).  Checkpoint kept for the step-3 observations.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; SRC=E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt
out=scratch/fly_stream/precision_gb
FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8 $PY -u scripts/ib/train_fly_bptt_stream.py --resume $SRC --extend-budget \
  --additional-tokens 4096 --validate-every-tokens 1024 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph \
  --go-neuron virtual --read-bank --intrinsic --precision --mixed-world --output $out --checkpoint-dir $out > $out.log 2>&1
echo "precision exit $?" >> scratch/fly_stream/status_prec
rm -f $out/best.pt
echo done >> scratch/fly_stream/status_prec
