#!/bin/bash
# The release arm (from m_main c6, 2048 words, --kc-dan-release), its taste lines and paired comparison with the base
# arm, then the mainline resumes.  (Three earlier tries failed on resume paths, now fixed and smoke-tested.)
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface; ab=/e/fly_stream_overflow/kc_release_ab
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
arm=release; o=$ab/$arm; k=/e/ib_checkpoints/kc_release_ab_$arm; rm -rf $o $k; mkdir -p $o $k
cp /e/ib_checkpoints/m_main/c6.pt $k/last.pt; cp /e/ib_checkpoints/m_main/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp $ab/base/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 2048 --output $o --checkpoint-dir $k \
    --score-log $o/scores.jsonl --log-word-dopamine --kc-dan-release > $o/train.log 2>&1 \
    || echo "$(date +%H:%M) $arm FAILED (fourth)" >> $ab/status
echo "$(date +%H:%M) $arm: $(grep 'ACTIVE EVAL' $o/train.log | tail -1); DA $($PY $FI/live_da_report.py $o/scores.jsonl)" >> $ab/status
$PY -u $FI/taste_pulse_probe.py $k $o/taste.json > $o/taste.txt 2>&1
echo "$(date +%H:%M) $arm taste: $(grep 'extra Hz' $o/taste.txt | tr -s ' ' | tr '\n' ';')" >> $ab/status
rm -rf $k
echo "$(date +%H:%M) release - base (paired by window, < 0: release helps): $($PY $FI/live_gate_report.py $ab/base/scores.jsonl $o/scores.jsonl release)" >> $ab/status
echo "$(date +%H:%M) release arm done; resuming m_main" >> /e/fly_stream_overflow/m_main/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
