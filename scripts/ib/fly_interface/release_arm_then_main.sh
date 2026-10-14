#!/bin/bash
# After the mother smoke: the release arm of the A/B (rerun: the first try failed on the weight-migration check), its
# paired comparison with the base arm, then the mainline resumes.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
while powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*mother_smoke*' }) { exit 0 } else { exit 1 }"; do sleep 20; done
ms=/e/fly_stream_overflow/mother_smoke
echo "$(date +%H:%M) mother smoke ended: $(grep -c . $ms/mother.jsonl 2>/dev/null) turns logged; $(grep -E 'Traceback|Error' $ms/train.log | head -1)" >> /e/fly_stream_overflow/kc_release_ab/status
rm -rf /e/ib_checkpoints/mother_smoke
root=/e/fly_stream_overflow/kc_release_ab; arm=release
o=$root/$arm; k=/e/ib_checkpoints/kc_release_ab_$arm; rm -rf $o $k; mkdir -p $o $k
cp /e/ib_checkpoints/m_main/c6.pt $k/last.pt; cp /e/ib_checkpoints/m_main/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp /e/fly_stream_overflow/m_main/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 2048 --output $o --checkpoint-dir $k \
    --score-log $o/scores.jsonl --log-word-dopamine --kc-dan-release > $o/train.log 2>&1 \
    || echo "$(date +%H:%M) $arm FAILED again" >> $root/status
echo "$(date +%H:%M) $arm: $(grep 'ACTIVE EVAL' $o/train.log | tail -1); DA $($PY $FI/live_da_report.py $o/scores.jsonl)" >> $root/status
$PY -u $FI/taste_pulse_probe.py $k $o/taste.json > $o/taste.txt 2>&1
echo "$(date +%H:%M) $arm taste: $(grep 'extra Hz' $o/taste.txt | tr -s ' ' | tr '\n' ';')" >> $root/status
rm -rf $k /e/ib_checkpoints/kc_release_ab_base
echo "$(date +%H:%M) release - base (paired by window, < 0: release helps): $($PY $FI/live_gate_report.py $root/base/scores.jsonl $root/release/scores.jsonl release)" >> $root/status
echo "$(date +%H:%M) release A/B and mother smoke done; resuming m_main" >> /e/fly_stream_overflow/m_main/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
