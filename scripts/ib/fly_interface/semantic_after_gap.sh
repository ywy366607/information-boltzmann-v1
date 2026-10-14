#!/bin/bash
# After context_gap_after_probes.sh has run the context gap sweep (8, 16 ticks), the moment it restarts
# d5b: stop it again (chunk 10 has only just begun), run kenyon_semantic_probe.py (P2, P3 of the
# representation-level surprise hypothesis), then resume d5b to c11 and m_main exactly as that queue would.
# Question (user): does the integrated Kenyon code carry meaning beyond the token (semantic, not token, surprise)?
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; d=/e/fly_stream_overflow/d5b; ck=/d/ib_checkpoints/d5b
until grep -q "gap sweep done; d5b resumes" $d/status 2>/dev/null; do sleep 20; done
sleep 10
for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and (\$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/context_gap_after_probes.sh*' -or \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_acts_v2.sh d5b *')) -or (\$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/d5b*') } | % { \$_.ProcessId }" | tr -d '\r'); do
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
echo "$(date +%H:%M) paused again after c9 for the Kenyon semantic probe (P2, P3)" >> $d/status
$PY -u $FI/kenyon_semantic_probe.py /e/ib_checkpoints/m_main 3000 $mm/kenyon_semantic_c7.json > $mm/kenyon_semantic_c7.txt 2>&1 \
  && echo "$(date +%H:%M) c7 Kenyon semantic (P2 novelty/prediction error vs rarity; P3 same meaning, different words): $(tail -n 1 $mm/kenyon_semantic_c7.txt)" >> $mm/status \
  || echo "$(date +%H:%M) c7 Kenyon semantic probe FAILED" >> $mm/status
echo "$(date +%H:%M) semantic probe done; d5b resumes" >> $d/status
CKROOT=/d/ib_checkpoints SRC=d4_acts FROM=8 ACTS="--ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior --ai-epistemic 0.0 --ai-pragmatic 1.0 --ai-sigreg 0.01 --act-brain-grad --noise-rate 0.25 --ai-depth 2 --eye-binary --eye-init 0.9 --eye-floor 0.05" \
  bash $FI/grow_acts_v2.sh d5b 11
echo "$(date +%H:%M) d5b done; m_main resumes from c7" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
