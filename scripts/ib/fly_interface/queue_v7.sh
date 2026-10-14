#!/bin/bash
# The epistemic mother run is already running (started by queue v6, whose bash was stopped). One training at a time:
# 1. wait for it, record it, keep only its raised checkpoint (on D);
# 2. the corrected Kenyon input probe (graded APL output counted as transmitted, m_main c7);
# 3. d5b from d4_acts c8 (eye floor 0.05, blind arms in the gates), chunks 9-11, checkpoints on D;
# 4. m_main resumes (from c7, on E).
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main
running() { powershell -c "@(Get-CimInstance Win32_Process | ? { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*$1*' }).Count" | tr -d '\r\n '; }

o=/e/fly_stream_overflow/mother_v2_epistemic; k=/d/ib_checkpoints/mother_v2_epistemic
while [ "$(running 'mother_v2_epistemic --checkpoint-dir')" != "0" ]; do sleep 20; done
if grep -q "BPTT 33792/33792" $o/train.log; then
  echo "$(date +%H:%M) epistemic mother (DeepSeek, contingent) done: $(grep -c . $o/mother.jsonl) transcript lines; monitor $(tail -n 1 $o/scores.jsonl | $PY -c "import json,sys; print(json.dumps(json.loads(sys.stdin.read()).get('monitor')))")" >> $o/status
else
  echo "$(date +%H:%M) epistemic mother FAILED or stopped (see train.log)" >> $o/status
fi
mv $k/last.pt $k/epistemic_1024.pt 2>/dev/null; rm -f $k/birth.pt $k/*.tmp*
cat $o/status >> $mm/status

$PY -u $FI/kenyon_input_probe.py /e/ib_checkpoints/m_main 400 $mm/kenyon_input_c7_v2.json > $mm/kenyon_input_c7_v2.txt 2>&1 \
  && echo "$(date +%H:%M) c7 Kenyon input (graded output counted): $(tail -n 1 $mm/kenyon_input_c7_v2.txt)" >> $mm/status \
  || echo "$(date +%H:%M) c7 Kenyon input probe v2 FAILED" >> $mm/status

CKROOT=/d/ib_checkpoints SRC=d4_acts FROM=8 ACTS="--ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior --ai-epistemic 0.0 --ai-pragmatic 1.0 --ai-sigreg 0.01 --act-brain-grad --noise-rate 0.25 --ai-depth 2 --eye-binary --eye-init 0.9 --eye-floor 0.05" \
  bash $FI/grow_acts_v2.sh d5b 11
echo "$(date +%H:%M) d5b done; m_main resumes from c7" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
