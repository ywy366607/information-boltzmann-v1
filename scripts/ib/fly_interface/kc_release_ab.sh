#!/bin/bash
# After D4: does the Kenyon -> dopamine-terminal release (attach_kenyon_terminal_release) carry surprise to dopamine?
# From m_main c6, two living arms on the same 2048 words (same world, learning on): base / --kc-dan-release.
# (2) surprise -> phasic dopamine in each stream (live_da_report); the cost or gain of the release on prediction, paired
# by window (live_gate_report: release - base); the taste lines with fixed weights (taste_pulse_probe) on both.
# Then the mainline resumes.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
until grep -q "d4_acts finished\|FAILED\|STOP" /e/fly_stream_overflow/d4_acts/status; do sleep 60; done
root=/e/fly_stream_overflow/kc_release_ab; mkdir -p $root
for arm in base release; do
  o=$root/$arm; k=/e/ib_checkpoints/kc_release_ab_$arm; rm -rf $o $k; mkdir -p $o $k
  cp /e/ib_checkpoints/m_main/c6.pt $k/last.pt; cp /e/ib_checkpoints/m_main/birth.pt $k/birth.pt
  for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp /e/fly_stream_overflow/m_main/$f $o/ 2>/dev/null; done
  $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 2048 --output $o --checkpoint-dir $k \
      --score-log $o/scores.jsonl --log-word-dopamine $([ $arm = release ] && echo --kc-dan-release) > $o/train.log 2>&1 \
      || echo "$(date +%H:%M) $arm FAILED" >> $root/status
  echo "$(date +%H:%M) $arm: $(grep 'ACTIVE EVAL' $o/train.log | tail -1); DA $($PY $FI/live_da_report.py $o/scores.jsonl)" >> $root/status
  $PY -u $FI/taste_pulse_probe.py $k $o/taste.json > $o/taste.txt 2>&1
  echo "$(date +%H:%M) $arm taste: $(grep 'extra Hz' $o/taste.txt | tr -s ' ' | tr '\n' ';')" >> $root/status
  rm -rf $k
done
echo "$(date +%H:%M) release - base (paired by window, < 0: release helps): $($PY $FI/live_gate_report.py $root/base/scores.jsonl $root/release/scores.jsonl release)" >> $root/status
echo "$(date +%H:%M) kc_release_ab done; resuming m_main" >> /e/fly_stream_overflow/m_main/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
