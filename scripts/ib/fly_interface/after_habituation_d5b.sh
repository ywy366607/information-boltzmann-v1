#!/bin/bash
# After habituation_and_mother_smoke.sh (one training fits in host RAM):
# 1. it relaunches d5_eyes, whose eyes closed for good in chunk 9 (dark room: no prior on the eye policy, p -> 0 is
#    absorbing); stop that chain at once;
# 2. what reading is worth in the mainline: a blind arm from m_main c7 on chunk 8's words (that chunk trained, its save
#    failed; its scores are the brain arm), paired by window;
# 3. d5b: D5 again from d4_acts c8 with an exploration floor on the eyes (--eye-floor 0.05), gates with a blind arm
#    (grow_acts_v2.sh), chunks 9-11;
# 4. m_main resumes (its own chain, grow_mainline_live.sh, from c7).
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
d5=/e/fly_stream_overflow/d5_eyes; mm=/e/fly_stream_overflow/m_main
until grep -q "habituation arm and mother smoke done; d5_eyes resumes" $d5/status; do sleep 30; done
sleep 20
for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_d4_acts.sh d5_eyes *') -or (\$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/d5_eyes --checkpoint-dir*') } | % { \$_.ProcessId }"); do
  powershell -c "Stop-Process -Id $p -Force" 2>/dev/null
done
rm -f $d5/scores_c10.jsonl
echo "$(date +%H:%M) d5_eyes stopped for good: its eyes closed at ~35,200 words and never reopened (dark room); continued as d5b (eye floor 0.05)" >> $d5/status

# 2. the blind arm on the mainline
g=$mm/gate_c7; o=$g/blind; k=/e/ib_checkpoints/m_main_gate; mkdir -p $g; rm -rf $o $k; mkdir -p $o $k
head -n 32 $mm/scores_c8_failed_save.jsonl > $g/brain_scores.jsonl
cp /e/ib_checkpoints/m_main/c7.pt $k/last.pt; cp /e/ib_checkpoints/m_main/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp $mm/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 1024 --output $o --checkpoint-dir $k \
    --score-log $o/scores.jsonl --skip-eval --ablate blind > $o/train.log 2>&1 \
  && echo "$(date +%H:%M) gate c7 (living): (1b) reading, blind - brain on chunk 8's words: $($PY $FI/live_gate_report.py $g/brain_scores.jsonl $o/scores.jsonl blind)" >> $mm/status \
  || echo "$(date +%H:%M) gate c7 blind arm FAILED" >> $mm/status
rm -rf $k

# 3. d5b
SRC=d4_acts FROM=8 ACTS="--ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior --ai-epistemic 0.0 --ai-pragmatic 1.0 --ai-sigreg 0.01 --act-brain-grad --noise-rate 0.25 --ai-depth 2 --eye-binary --eye-init 0.9 --eye-floor 0.05" \
  bash $FI/grow_acts_v2.sh d5b 11

# 4. the mainline
echo "$(date +%H:%M) d5b done; m_main resumes from c7" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
