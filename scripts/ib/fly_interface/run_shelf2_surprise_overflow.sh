#!/bin/bash
# Shelf round 2, surprise arm: identical to scratch/fly_stream/run_shelf.sh but output on E: (D: is full).
cd /d/information-boltzmann-v1
out=/e/fly_stream_overflow/shelf2_surprise
FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8 /d/conda_envs/vox/python.exe -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 \
  --validate-every-tokens 1000000 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph \
  --go-neuron virtual --read-bank --predictive-writer --dopamine-modulation --dopamine-site pre \
  --bookshelf surprise --bookshelf-seed 0 --output $out --checkpoint-dir $out > $out.log 2>&1
echo "surprise exit $?" >> /e/fly_stream_overflow/status
