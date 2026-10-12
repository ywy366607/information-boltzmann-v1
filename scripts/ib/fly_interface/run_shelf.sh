#!/bin/bash
# Curious shelf v1, small experiment: does choosing what to read by learning progress work?  Three arms from the adult
# lineage dasite_pre (fast reader), self-paced reading only, the same 12 books and seeds: progress / random / surprise.
# 2,048 tokens each (32 segments of 64), the book just read probed after every segment.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe
for sel in progress random surprise; do
  out=scratch/fly_stream/${SHELF_PREFIX:-shelf}_$sel
  mkdir -p $out; [ -f $out/last.pt ] || cp scratch/fly_stream/dasite_pre/last.pt scratch/fly_stream/dasite_pre/config.json $out/
  [ -f $out/STOP ] && continue
  FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8 $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens ${SHELF_TOKENS:-2048} \
    --validate-every-tokens 1000000 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph \
    --go-neuron virtual --read-bank --predictive-writer --dopamine-modulation --dopamine-site pre \
    --bookshelf $sel --bookshelf-seed 0 --output $out --checkpoint-dir $out > $out.log 2>&1
  echo "$sel exit $?" >> scratch/fly_stream/${SHELF_PREFIX:-shelf}_status
  rm -f $out/best.pt
done
echo done >> scratch/fly_stream/${SHELF_PREFIX:-shelf}_status
