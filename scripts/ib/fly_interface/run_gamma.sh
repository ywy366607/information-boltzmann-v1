#!/bin/bash
# Integrator readout, fair start: uniform short decays (near-instantaneous to ~5-tick spread), 4096-token forks,
# paired with the earlier ctrl_adaptive / stream4 arms (same cursor, same protocol).
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; SRC=E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt
COMMON="--resume $SRC --extend-budget --additional-tokens 4096 --validate-every-tokens 1024 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph --use-read-gamma-trace"
run() {
  out=scratch/fly_stream/$1
  PYTHONIOENCODING=utf-8 $PY -u scripts/ib/train_fly_bptt_stream.py $COMMON $2 --output $out --checkpoint-dir $out > $out.log 2>&1
  echo "$1 exit $?" >> scratch/fly_stream/status_gamma
  rm -f $out/*.pt $out/segments/*/*.pt
}
run stream4_g03 "--stream-ticks 4 --init-read-gamma 0.3"
run stream4_g05 "--stream-ticks 4 --init-read-gamma 0.5"
run stream4_g07 "--stream-ticks 4 --init-read-gamma 0.7"
run adaptive_g05 "--init-read-gamma 0.5"
echo done >> scratch/fly_stream/status_gamma
