#!/bin/bash
# queue_reading_value.sh, re-ordered for the DeepSeek mouth (one training fits in host RAM), in turn:
# 1. the blind arm on m_main c7 (already running when this starts): wait for it, report reading = blind - brain;
# 2. does a word's Kenyon code repeat when the word repeats (m_main c7)?
# 3. the champion's reading value in its own regime: brain and blind arms, 512 words each;
# 4. the mother world v2 with the DeepSeek mouth: 1,024 words from d4_acts c8, speaking act on (turn ends, decisions);
# 5. d5b: D5 again from d4_acts c8 with --eye-floor 0.05, gates with a blind arm (grow_acts_v2.sh), chunks 9-11;
# 6. m_main resumes (grow_mainline_live.sh, from c7).
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; g=$mm/gate_c7
running() { powershell -c "@(Get-CimInstance Win32_Process | ? { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*$1*' }).Count" | tr -d '\r'; }

# 1.
while [ "$(running 'fly_stream_overflow/m_main/gate_c7/blind --checkpoint-dir')" != "0" ]; do sleep 20; done
rm -rf /e/ib_checkpoints/m_main_gate
echo "$(date +%H:%M) gate c7 (living): (1b) reading, blind - brain on chunk 8's words: $($PY $FI/live_gate_report.py $g/brain_scores.jsonl $g/blind/scores.jsonl blind)" >> $mm/status

# 2.
$PY -u $FI/kenyon_repeat_probe.py /e/ib_checkpoints/m_main 1500 $mm/kenyon_repeat_c7.json > $mm/kenyon_repeat_c7.txt 2>&1 \
  && echo "$(date +%H:%M) c7 Kenyon repeat: $(tail -n 1 $mm/kenyon_repeat_c7.txt)" >> $mm/status \
  || echo "$(date +%H:%M) c7 Kenyon repeat probe FAILED" >> $mm/status

# 3.
cr=/e/fly_stream_overflow/champion_reading; rm -rf $cr; mkdir -p $cr
for arm in brain blind; do
  o=$cr/$arm; k=/e/ib_checkpoints/champion_reading_$arm; rm -rf $o $k; mkdir -p $o $k
  $PY -u scripts/ib/train_fly_bptt_stream.py --resume /e/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt --al-reset \
      --extend-budget --additional-tokens 512 --no-graph --skip-eval --output $o --checkpoint-dir $k \
      --score-log $o/scores.jsonl $([ $arm = blind ] && echo --ablate blind) > $o/train.log 2>&1 \
    || echo "$(date +%H:%M) champion $arm arm FAILED" >> $cr/status
  rm -rf $k
done
echo "$(date +%H:%M) champion reading (blind - brain, 512 words): $($PY $FI/live_gate_report.py $cr/brain/scores.jsonl $cr/blind/scores.jsonl blind)" >> $cr/status
cat $cr/status >> $mm/status

# 4.
o=/e/fly_stream_overflow/mother_v2_deepseek; k=/e/ib_checkpoints/mother_v2_deepseek; rm -rf $o $k; mkdir -p $o $k
cp /e/ib_checkpoints/d4_acts/c8.pt $k/last.pt; cp /e/ib_checkpoints/d4_acts/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp /e/fly_stream_overflow/d4_acts/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 1024 --output $o --checkpoint-dir $k \
    --score-log $o/scores.jsonl --skip-eval --mother-world deepseek --mother-speak-act > $o/train.log 2>&1 \
  && echo "$(date +%H:%M) mother v2 (DeepSeek) done: $(grep -c . $o/mother.jsonl) transcript lines; monitor $(tail -n 1 $o/scores.jsonl | $PY -c "import json,sys; print(json.dumps(json.loads(sys.stdin.read()).get('monitor')))")" >> $o/status \
  || echo "$(date +%H:%M) mother v2 (DeepSeek) FAILED" >> $o/status
mv $k/last.pt $k/mother_1024.pt 2>/dev/null; rm -f $k/birth.pt $k/*.tmp*   # keep only the raised checkpoint (birth = d4_acts/birth.pt)
cat $o/status >> $mm/status

# 5.
SRC=d4_acts FROM=8 ACTS="--ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior --ai-epistemic 0.0 --ai-pragmatic 1.0 --ai-sigreg 0.01 --act-brain-grad --noise-rate 0.25 --ai-depth 2 --eye-binary --eye-init 0.9 --eye-floor 0.05" \
  bash $FI/grow_acts_v2.sh d5b 11

# 6.
echo "$(date +%H:%M) d5b done; m_main resumes from c7" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
