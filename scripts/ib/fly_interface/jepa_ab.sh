#!/bin/bash
# Change 2 (the JEPA term, --stream-jepa): a mechanism test, no brain-contribution gate.  Control = m_main's own chunk
# 11 from c10 (the chain is running it); as soon as it is saved, stop the chain (its gate c10 is skipped this once),
# run the treatment = chunk 11 from c10 with --stream-jepa on the same words, then on both c11s: P2 (Kenyon novelty /
# next-pattern prediction error vs rarity and the brain's CE) and P3 (GPT-2 meaning beyond the senses); then resume
# m_main.  Predictions (design, FLY_ACTIVE_INFERENCE_GOAL_20261014_ACTS.md): next-pattern error below the mean
# pattern 0.698; P3 partial rho above 0.05; novelty / error predict the CE residual (rho > 0.1); (1) not harmed.
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; ck=/e/ib_checkpoints/m_main
o=/e/fly_stream_overflow/m_main_jepa; k=/e/ib_checkpoints/m_main_jepa
until grep -q "chunk 11 done" $mm/status 2>/dev/null && [ -f $ck/c11.pt ] && [ "$(stat -c %s $ck/c11.pt)" = "$(stat -c %s $ck/last.pt)" ]; do sleep 20; done
sleep 10
for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_mainline_live.sh m_main *') -or (\$_.Name -eq 'python.exe' -and (\$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main *' -or \$_.CommandLine -like '*fly_interface*E:/ib_checkpoints/m_main *' -or \$_.CommandLine -like '*fly_interface*E:/ib_checkpoints/m_main' -or \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main/gate_*')) } | % { \$_.ProcessId }" | tr -d '\r'); do
  powershell -c "Stop-Process -Id $p -Force" 2>/dev/null
done
sleep 5
cp $ck/c11.pt $ck/last.pt; rm -f $mm/scores_c12.jsonl $ck/last.pt.tmp*; rm -rf /e/ib_checkpoints/m_main_gate
echo "$(date +%H:%M) paused right after chunk 11 (gate c10 skipped) for change 2: treatment = chunk 11 from c10 with the stream JEPA term" >> $mm/status
rm -rf $o $k; mkdir -p $o $k
cp $ck/c10.pt $k/last.pt; cp $ck/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp $mm/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 --output $o --checkpoint-dir $k \
    --score-log $o/scores_c11.jsonl --log-word-dopamine --stream-jepa > $o/train.log 2>&1 \
  && echo "$(date +%H:%M) JEPA treatment chunk 11 done: $(grep 'ACTIVE EVAL' $o/train.log | tail -1); skipped updates $(grep -c 'skipped the update' $o/train.log)" >> $mm/status \
  || echo "$(date +%H:%M) JEPA treatment chunk 11 FAILED" >> $mm/status
grep -q "Stream JEPA term on" $o/train.log || echo "$(date +%H:%M) WARNING: JEPA treatment log lacks the attach line" >> $mm/status
echo "$(date +%H:%M) JEPA treatment c11 (1) vs control, paired: $($PY $FI/live_gate_report.py $mm/scores_c11.jsonl $o/scores_c11.jsonl jepa)" >> $mm/status
for arm in control treatment; do
  if [ $arm = control ]; then src=$ck; out=$mm; else src=$k; out=$o; fi
  $PY -u $FI/kenyon_semantic_probe.py $src 3000 $out/kenyon_semantic_c11.json > $out/kenyon_semantic_c11.txt 2>&1 \
    && echo "$(date +%H:%M) $arm c11 P2: $(grep '^P2' $out/kenyon_semantic_c11.txt)" >> $mm/status \
    || echo "$(date +%H:%M) $arm c11 P2 probe FAILED" >> $mm/status
  $PY -u $FI/kenyon_gpt2_probe.py $src 3000 $out/kenyon_gpt2_c11.json > $out/kenyon_gpt2_c11.txt 2>&1 \
    && echo "$(date +%H:%M) $arm c11 P3 (GPT-2): $(tail -n 1 $out/kenyon_gpt2_c11.txt)" >> $mm/status \
    || echo "$(date +%H:%M) $arm c11 P3 probe FAILED" >> $mm/status
done
rm -f $k/birth.pt $k/last.pt.tmp*
echo "$(date +%H:%M) change 2 (JEPA) A/B done; m_main resumes from c11" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
