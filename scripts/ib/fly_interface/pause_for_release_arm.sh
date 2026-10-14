#!/bin/bash
# After m_main's gate c6 (paired with D4's c6 gate: the same words): pause between chunks, rerun the release arm of
# the A/B from c6 (the base arm is done), compare paired by window, then resume the mainline.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
root=/e/fly_stream_overflow/m_main; ckpt=/e/ib_checkpoints/m_main; ab=/e/fly_stream_overflow/kc_release_ab
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
until grep -q "gate c6 (living)\|FAILED\|STOP" $root/status; do sleep 60; done
grep -q "FAILED\|STOP" $root/status && exit 1
until [ -f $root/scores_c8.jsonl ]; do sleep 10; done
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*grow_mainline_live.sh m_main*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*' -and \$_.CommandLine -like '*m_main*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
sleep 15
cp $ckpt/c7.pt $ckpt/last.pt; rm -f $root/scores_c8.jsonl
echo "$(date +%H:%M) paused after c7 (chunk 8 discarded) for the release arm" >> $root/status
arm=release; o=$ab/$arm; k=/e/ib_checkpoints/kc_release_ab_$arm; rm -rf $o $k; mkdir -p $o $k
cp $ckpt/c6.pt $k/last.pt; cp $ckpt/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp $ab/base/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 2048 --output $o --checkpoint-dir $k \
    --score-log $o/scores.jsonl --log-word-dopamine --kc-dan-release > $o/train.log 2>&1 \
    || echo "$(date +%H:%M) $arm FAILED (third)" >> $ab/status
echo "$(date +%H:%M) $arm: $(grep 'ACTIVE EVAL' $o/train.log | tail -1); DA $($PY $FI/live_da_report.py $o/scores.jsonl)" >> $ab/status
$PY -u $FI/taste_pulse_probe.py $k $o/taste.json > $o/taste.txt 2>&1
echo "$(date +%H:%M) $arm taste: $(grep 'extra Hz' $o/taste.txt | tr -s ' ' | tr '\n' ';')" >> $ab/status
rm -rf $k
echo "$(date +%H:%M) release - base (paired by window, < 0: release helps): $($PY $FI/live_gate_report.py $ab/base/scores.jsonl $o/scores.jsonl release)" >> $ab/status
echo "$(date +%H:%M) release arm done; resuming m_main" >> $root/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
