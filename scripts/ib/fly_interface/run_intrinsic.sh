#!/bin/bash
# Base (virtual go + read bank) vs base + released spontaneously active cells; final checkpoints KEPT for analysis.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; SRC=E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt
COMMON="--resume $SRC --extend-budget --additional-tokens 4096 --validate-every-tokens 1024 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph --go-neuron virtual --read-bank"
run() {
  out=scratch/fly_stream/$1
  PYTHONIOENCODING=utf-8 $PY -u scripts/ib/train_fly_bptt_stream.py $COMMON $2 --output $out --checkpoint-dir $out > $out.log 2>&1
  echo "$1 exit $?" >> scratch/fly_stream/status_intr
  rm -f $out/best.pt
}
run base_gb ""
run intrinsic_gb "--intrinsic"
echo done >> scratch/fly_stream/status_intr
