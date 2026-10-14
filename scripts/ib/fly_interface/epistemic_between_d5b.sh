#!/bin/bash
# Between d5b's chunks (one training fits in host RAM):
# 1. wait until d5b has trained chunk 9 and gated it; stop queue v5 and the d5b chain (it has just begun chunk 10);
# 2. the mother world with the epistemic speaking credit (--mother-credit epistemic, 64-tick horizon): 1,024 words from
#    d4_acts c8, contingent mother, DeepSeek mouth -- the same start as the two earlier mother runs;
# 3. d5b resumes (chunks 10-11, from its c9), then m_main resumes.
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; d=/e/fly_stream_overflow/d5b
until grep -q "gate c9 acts (living)" $d/status 2>/dev/null; do sleep 30; done
for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and (\$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/queue_contingent_v5.sh*' -or \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_acts_v2.sh d5b *')) -or (\$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/d5b*') } | % { \$_.ProcessId }" | tr -d '\r'); do
  powershell -c "Stop-Process -Id $p -Force" 2>/dev/null
done
sleep 5
cp /e/ib_checkpoints/d5b/c9.pt /e/ib_checkpoints/d5b/last.pt; rm -f $d/scores_c10.jsonl
end9=$(tail -n 1 $d/scores_c9.jsonl | $PY -c "import json,sys; print(json.loads(sys.stdin.read())['trained'])")
$PY - $d/ai_trace.jsonl $end9 <<'PY'
import json, sys
path, end = sys.argv[1], int(sys.argv[2])
rows = [l for l in open(path) if json.loads(l).get('bptt_train_tokens', 0) <= end]
open(path, 'w').writelines(rows)
PY
echo "$(date +%H:%M) paused after c9 (chunk 10 discarded) for the epistemic mother run" >> $d/status

o=/e/fly_stream_overflow/mother_v2_epistemic; k=/e/ib_checkpoints/mother_v2_epistemic; rm -rf $o $k; mkdir -p $o $k
cp /e/ib_checkpoints/d4_acts/c8.pt $k/last.pt; cp /e/ib_checkpoints/d4_acts/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp /e/fly_stream_overflow/d4_acts/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 1024 --output $o --checkpoint-dir $k \
    --score-log $o/scores.jsonl --skip-eval --mother-world deepseek --mother-speak-act --mother-credit epistemic \
    > $o/train.log 2>&1 \
  && echo "$(date +%H:%M) epistemic mother (DeepSeek, contingent) done: $(grep -c . $o/mother.jsonl) transcript lines; monitor $(tail -n 1 $o/scores.jsonl | $PY -c "import json,sys; print(json.dumps(json.loads(sys.stdin.read()).get('monitor')))")" >> $o/status \
  || echo "$(date +%H:%M) epistemic mother FAILED" >> $o/status
mv $k/last.pt $k/epistemic_1024.pt 2>/dev/null; rm -f $k/birth.pt $k/*.tmp*
cat $o/status >> $mm/status

echo "$(date +%H:%M) epistemic mother run done; d5b resumes" >> $d/status
SRC=d4_acts FROM=8 ACTS="--ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior --ai-epistemic 0.0 --ai-pragmatic 1.0 --ai-sigreg 0.01 --act-brain-grad --noise-rate 0.25 --ai-depth 2 --eye-binary --eye-init 0.9 --eye-floor 0.05" \
  bash $FI/grow_acts_v2.sh d5b 11
echo "$(date +%H:%M) d5b done; m_main resumes from c7" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
