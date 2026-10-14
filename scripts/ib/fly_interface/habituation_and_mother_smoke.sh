#!/bin/bash
# One training fits in host RAM (other apps hold ~14.5 GB), so the slot is shared in turn:
# 1. wait until d5_eyes has trained chunk 9 and gated it, then stop the d5 chain (it has just begun chunk 10);
# 2. (2) surprise -> dopamine: the Kenyon release with habituation (alpha'3 novelty) from m_main c6, the same 2,048
#    words as the release A/B (kc_release_ab: base and release arms), paired by window against both;
# 3. mother world v2 smoke (speaking as the act, the Gemini mouth): 256 words from d4_acts c8;
# 4. d5_eyes resumes (chunk 10 again from its c9).
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
d5=/e/fly_stream_overflow/d5_eyes; ab=/e/fly_stream_overflow/kc_release_ab
until grep -q "gate c9 acts (living)" $d5/status; do sleep 30; done
# only the chain's own bash (its exact invocation) and its trainer: tool shells that merely mention the names never match
for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_d4_acts.sh d5_eyes *') -or (\$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/d5_eyes --checkpoint-dir*') } | % { \$_.ProcessId }"); do
  powershell -c "Stop-Process -Id $p -Force" 2>/dev/null
done
sleep 5
cp /e/ib_checkpoints/d5_eyes/c9.pt /e/ib_checkpoints/d5_eyes/last.pt
# the discarded chunk 10: its scores and act trace rows go
end9=$(tail -n 1 $d5/scores_c9.jsonl | $PY -c "import json,sys; print(json.loads(sys.stdin.read())['trained'])")
rm -f $d5/scores_c10.jsonl
$PY - $d5/ai_trace.jsonl $end9 <<'PY'
import json, sys
path, end = sys.argv[1], int(sys.argv[2])
rows = [l for l in open(path) if json.loads(l).get('bptt_train_tokens', 0) <= end]
open(path, 'w').writelines(rows)
PY
echo "$(date +%H:%M) paused after c9 (chunk 10 discarded) for the habituation arm and the mother smoke" >> $d5/status

# 2. the habituation arm
o=$ab/habituation; k=/e/ib_checkpoints/kc_release_ab_habituation; rm -rf $o $k; mkdir -p $o $k
cp /e/ib_checkpoints/m_main/c6.pt $k/last.pt; cp /e/ib_checkpoints/m_main/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp $ab/base/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 2048 --output $o --checkpoint-dir $k \
    --score-log $o/scores.jsonl --log-word-dopamine --kc-dan-release --kc-habituation-tau 4096 --kc-habituation-use 0.2 \
    > $o/train.log 2>&1 || echo "$(date +%H:%M) habituation FAILED" >> $ab/status
echo "$(date +%H:%M) habituation: $(grep 'ACTIVE EVAL' $o/train.log | tail -1); DA $($PY $FI/live_da_report.py $o/scores.jsonl)" >> $ab/status
echo "$(date +%H:%M) habituation - base: $($PY $FI/live_gate_report.py $ab/base/scores.jsonl $o/scores.jsonl habituation)" >> $ab/status
echo "$(date +%H:%M) habituation - release: $($PY $FI/live_gate_report.py $ab/release/scores.jsonl $o/scores.jsonl habituation)" >> $ab/status
$PY -u $FI/taste_pulse_probe.py $k $o/taste.json > $o/taste.txt 2>&1
echo "$(date +%H:%M) habituation taste: $(grep 'extra Hz' $o/taste.txt | tr -s ' ' | tr '\n' ';')" >> $ab/status
rm -rf $k

# 3. the mother world v2 smoke
o=/e/fly_stream_overflow/mother_v2_smoke; k=/e/ib_checkpoints/mother_v2_smoke; rm -rf $o $k; mkdir -p $o $k
cp /e/ib_checkpoints/d4_acts/c8.pt $k/last.pt; cp /e/ib_checkpoints/d4_acts/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp /e/fly_stream_overflow/d4_acts/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 256 --output $o --checkpoint-dir $k \
    --score-log $o/scores.jsonl --skip-eval --mother-world gemini --mother-speak-act > $o/train.log 2>&1 \
    && echo "$(date +%H:%M) mother v2 smoke done: $(wc -l < $o/mother.jsonl) turns" >> $d5/status \
    || echo "$(date +%H:%M) mother v2 smoke FAILED" >> $d5/status
rm -rf $k

# 4. d5 resumes
echo "$(date +%H:%M) habituation arm and mother smoke done; d5_eyes resumes" >> $d5/status
SRC=d4_acts FROM=8 ACTS="--ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior --ai-epistemic 0.0 --ai-pragmatic 1.0 --ai-sigreg 0.01 --act-brain-grad --noise-rate 0.25 --ai-depth 2 --eye-binary --eye-init 0.9" \
  nohup bash $FI/grow_d4_acts.sh d5_eyes 12 > /dev/null 2>&1 &
