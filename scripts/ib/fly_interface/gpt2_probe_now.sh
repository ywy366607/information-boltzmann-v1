#!/bin/bash
# P3 again with GPT-2 as the measure of meaning (user: no co-occurrence when a better measure exists).  d5b has just
# begun chunk 10 (its queue, semantic_after_gap.sh, was stopped): stop it, restore c9, run kenyon_gpt2_probe.py on
# m_main c7, then resume d5b to c11 and m_main as the earlier queues would.
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; d=/e/fly_stream_overflow/d5b; ck=/d/ib_checkpoints/d5b
for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_acts_v2.sh d5b *') -or (\$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/d5b*') } | % { \$_.ProcessId }" | tr -d '\r'); do
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
echo "$(date +%H:%M) paused again after c9 for the GPT-2 meaning probe (P3)" >> $d/status
$PY -u $FI/kenyon_gpt2_probe.py /e/ib_checkpoints/m_main 3000 $mm/kenyon_gpt2_c7.json > $mm/kenyon_gpt2_c7.txt 2>&1 \
  && echo "$(date +%H:%M) c7 Kenyon meaning (P3, GPT-2 layer 6, no shared token): $(tail -n 1 $mm/kenyon_gpt2_c7.txt)" >> $mm/status \
  || echo "$(date +%H:%M) c7 Kenyon GPT-2 probe FAILED" >> $mm/status
echo "$(date +%H:%M) GPT-2 probe done; d5b resumes" >> $d/status
CKROOT=/d/ib_checkpoints SRC=d4_acts FROM=8 ACTS="--ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior --ai-epistemic 0.0 --ai-pragmatic 1.0 --ai-sigreg 0.01 --act-brain-grad --noise-rate 0.25 --ai-depth 2 --eye-binary --eye-init 0.9 --eye-floor 0.05" \
  bash $FI/grow_acts_v2.sh d5b 11
echo "$(date +%H:%M) d5b done; m_main resumes from c7" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
