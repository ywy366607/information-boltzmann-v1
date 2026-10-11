#!/bin/bash
# Longer paired forks (16384 tokens, active evaluation every 2048): does learning close the streaming gap?
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; SRC=E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt
COMMON="--resume $SRC --extend-budget --additional-tokens 16384 --validate-every-tokens 2048 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph"
run() {
  out=scratch/fly_stream/long_$1
  PYTHONIOENCODING=utf-8 $PY -u scripts/ib/train_fly_bptt_stream.py $COMMON $2 --output $out --checkpoint-dir $out > $out.log 2>&1
  echo "long_$1 exit $?" >> scratch/fly_stream/status_long
  rm -f $out/*.pt $out/segments/*/*.pt
}
run ctrl_adaptive ""
run lag4 "--stream-ticks 4 --read-lag 1"
run lag3 "--stream-ticks 3 --read-lag 1"
echo done >> scratch/fly_stream/status_long
