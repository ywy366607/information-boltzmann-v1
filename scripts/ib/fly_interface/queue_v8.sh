#!/bin/bash
# 1. stop queue v7 and d5b (d5b had just begun its chunk 9; it restarts from its fork);
# 2. the mother world with the DISCOUNTED epistemic credit (--mother-credit-tau 32): 1,024 words from d4_acts c8,
#    contingent mother, DeepSeek mouth -- the same start as the three earlier mother runs; one change against the
#    flat 64-tick horizon run (mother_v2_epistemic);
# 3. d5b from d4_acts c8 (eye floor 0.05, blind arms in the gates), chunks 9-11, checkpoints on D;  4. m_main resumes.
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main
for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and (\$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/queue_v7.sh*' -or \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_acts_v2.sh d5b *')) -or (\$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/d5b*') } | % { \$_.ProcessId }" | tr -d '\r'); do
  powershell -c "Stop-Process -Id $p -Force" 2>/dev/null
done
sleep 5
rm -rf /e/fly_stream_overflow/d5b /d/ib_checkpoints/d5b /d/ib_checkpoints/d5b_gate

o=/e/fly_stream_overflow/mother_v2_discount; k=/d/ib_checkpoints/mother_v2_discount; rm -rf $o $k; mkdir -p $o $k
cp /e/ib_checkpoints/d4_acts/c8.pt $k/last.pt; cp /e/ib_checkpoints/d4_acts/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp /e/fly_stream_overflow/d4_acts/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 1024 --output $o --checkpoint-dir $k \
    --score-log $o/scores.jsonl --skip-eval --mother-world deepseek --mother-speak-act --mother-credit epistemic \
    --mother-credit-tau 32 > $o/train.log 2>&1 \
  && echo "$(date +%H:%M) discounted epistemic mother (tau 32 ticks) done: $(grep -c . $o/mother.jsonl) transcript lines; monitor $(tail -n 1 $o/scores.jsonl | $PY -c "import json,sys; print(json.dumps(json.loads(sys.stdin.read()).get('monitor')))")" >> $o/status \
  || echo "$(date +%H:%M) discounted epistemic mother FAILED" >> $o/status
mv $k/last.pt $k/discount_1024.pt 2>/dev/null; rm -f $k/birth.pt $k/*.tmp*
cat $o/status >> $mm/status

CKROOT=/d/ib_checkpoints SRC=d4_acts FROM=8 ACTS="--ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior --ai-epistemic 0.0 --ai-pragmatic 1.0 --ai-sigreg 0.01 --act-brain-grad --noise-rate 0.25 --ai-depth 2 --eye-binary --eye-init 0.9 --eye-floor 0.05" \
  bash $FI/grow_acts_v2.sh d5b 11
echo "$(date +%H:%M) d5b done; m_main resumes from c7" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
