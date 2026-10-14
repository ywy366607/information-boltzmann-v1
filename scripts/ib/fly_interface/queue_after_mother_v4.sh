#!/bin/bash
# The queue from the DeepSeek mother run on (already running when this starts; one training fits in host RAM):
# 4. wait for the mother run, record it, keep only its raised checkpoint;
# 3c. what drives the Kenyon cells while reading (m_main c7);
# 5. d5b (eye floor 0.05, blind arms in the gates), chunks 9-11;  6. m_main resumes.
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main
running() { powershell -c "@(Get-CimInstance Win32_Process | ? { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*$1*' }).Count" | tr -d '\r\n '; }

# 4.
o=/e/fly_stream_overflow/mother_v2_deepseek; k=/e/ib_checkpoints/mother_v2_deepseek
while [ "$(running 'mother_v2_deepseek --checkpoint-dir')" != "0" ]; do sleep 20; done
if grep -q "BPTT 33792/33792" $o/train.log; then
  echo "$(date +%H:%M) mother v2 (DeepSeek) done: $(grep -c . $o/mother.jsonl) transcript lines; monitor $(tail -n 1 $o/scores.jsonl | $PY -c "import json,sys; print(json.dumps(json.loads(sys.stdin.read()).get('monitor')))")" >> $o/status
else
  echo "$(date +%H:%M) mother v2 (DeepSeek) FAILED or stopped (see train.log)" >> $o/status
fi
mv $k/last.pt $k/mother_1024.pt 2>/dev/null; rm -f $k/birth.pt $k/*.tmp*   # keep only the raised checkpoint (birth = d4_acts/birth.pt)
cat $o/status >> $mm/status

# 3c.
$PY -u $FI/kenyon_input_probe.py /e/ib_checkpoints/m_main 400 $mm/kenyon_input_c7.json > $mm/kenyon_input_c7.txt 2>&1 \
  && echo "$(date +%H:%M) c7 Kenyon input: $(tail -n 1 $mm/kenyon_input_c7.txt)" >> $mm/status \
  || echo "$(date +%H:%M) c7 Kenyon input probe FAILED" >> $mm/status

# 5.
SRC=d4_acts FROM=8 ACTS="--ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior --ai-epistemic 0.0 --ai-pragmatic 1.0 --ai-sigreg 0.01 --act-brain-grad --noise-rate 0.25 --ai-depth 2 --eye-binary --eye-init 0.9 --eye-floor 0.05" \
  bash $FI/grow_acts_v2.sh d5b 11

# 6.
echo "$(date +%H:%M) d5b done; m_main resumes from c7" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
