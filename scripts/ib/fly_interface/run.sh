#!/bin/bash
# Streaming-clock forks of the CTM-S14 champion under the four-pillar active protocol (source checkpoint untouched).
# One process at a time; each arm's checkpoints are deleted after it finishes (reports kept).
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; SRC=E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt
COMMON="--resume $SRC --extend-budget --additional-tokens 4096 --validate-every-tokens 1024 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph"
run() {  # name, extra flags
  out=scratch/fly_stream/$1
  PYTHONIOENCODING=utf-8 $PY -u scripts/ib/train_fly_bptt_stream.py $COMMON $2 --output $out --checkpoint-dir $out > $out.log 2>&1
  echo "$1 exit $?" >> scratch/fly_stream/status
  rm -f $out/*.pt $out/segments/*/*.pt
}
run ctrl_adaptive ""
run stream4 "--stream-ticks 4"
run stream3 "--stream-ticks 3"
run stream3_gamma "--stream-ticks 3 --use-read-gamma-trace"
run stream3_hold_gamma "--stream-ticks 3 --hold-input --use-read-gamma-trace"
echo done >> scratch/fly_stream/status
