#!/bin/bash
# Between d5b's chunks (one process at a time, host RAM):
# 1. wait until d5b has trained chunk 9 and gated it; stop queue v9 and the d5b chain (it has just begun chunk 10);
# 2. word_margin_probe on m_main c7: how far below threshold an isolated word leaves receptors, ALPN and Kenyon cells;
# 3. d5b resumes (chunks 10-11 from its c9, checkpoints on D), then m_main resumes.
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; d=/e/fly_stream_overflow/d5b; ck=/d/ib_checkpoints/d5b
until grep -q "gate c9 acts (living)" $d/status 2>/dev/null; do sleep 30; done
for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and (\$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/queue_v9.sh*' -or \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_acts_v2.sh d5b *')) -or (\$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/d5b*') } | % { \$_.ProcessId }" | tr -d '\r'); do
  powershell -c "Stop-Process -Id $p -Force" 2>/dev/null
done
sleep 5
cp $ck/c9.pt $ck/last.pt; rm -f $d/scores_c10.jsonl $ck/last.pt.tmp*
end9=$(tail -n 1 $d/scores_c9.jsonl | $PY -c "import json,sys; print(json.loads(sys.stdin.read())['trained'])")
$PY - $d/ai_trace.jsonl $end9 <<'PY'
import json, sys
path, end = sys.argv[1], int(sys.argv[2])
rows = [l for l in open(path) if json.loads(l).get('bptt_train_tokens', 0) <= end]
open(path, 'w').writelines(rows)
PY
echo "$(date +%H:%M) paused after c9 (chunk 10 discarded) for the word margin probe" >> $d/status
$PY -u $FI/word_margin_probe.py /e/ib_checkpoints/m_main 200 32 $mm/word_margin_c7.json > $mm/word_margin_c7.txt 2>&1 \
  && echo "$(date +%H:%M) c7 word margin (isolated words, gap 32): $(tail -n 1 $mm/word_margin_c7.txt)" >> $mm/status \
  || echo "$(date +%H:%M) c7 word margin probe FAILED" >> $mm/status
echo "$(date +%H:%M) probe done; d5b resumes" >> $d/status
CKROOT=/d/ib_checkpoints SRC=d4_acts FROM=8 ACTS="--ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior --ai-epistemic 0.0 --ai-pragmatic 1.0 --ai-sigreg 0.01 --act-brain-grad --noise-rate 0.25 --ai-depth 2 --eye-binary --eye-init 0.9 --eye-floor 0.05" \
  bash $FI/grow_acts_v2.sh d5b 11
echo "$(date +%H:%M) d5b done; m_main resumes from c7" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
