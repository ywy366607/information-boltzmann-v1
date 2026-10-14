#!/bin/bash
# Connectome fix (user: "fix it"): APL -> Kenyon inhibition on the soma only for its calyx synapses
# (--apl-kc-dendritic-only; 84.5% of them sit on the Kenyon axons, yet the point model put all on the membrane).
# Mechanism test after the JEPA A/B: control = m_main's own chunk 12 from c11 (the chain runs it once jepa_ab.sh
# resumes it); as soon as it is saved, stop the chain (gate c11 skipped this once), run the treatment = chunk 12 from
# c11 with the fix on the same words, then on both c12s: the word margin probe (isolated words), the Kenyon repeat
# probe (same word / same last 2-3 words), P3 (GPT-2 meaning beyond the senses), (2) and the paired NLL; then resume.
# Predictions: Kenyon level up from ~3%, single-word margin up from 0.33 theta, same-word repeatability up, the context
# code kept, (1) not harmed.
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; ck=/e/ib_checkpoints/m_main
o=/e/fly_stream_overflow/m_main_apl; k=/e/ib_checkpoints/m_main_apl
until grep -q "chunk 12 done" $mm/status 2>/dev/null && [ -f $ck/c12.pt ] && [ "$(stat -c %s $ck/c12.pt)" = "$(stat -c %s $ck/last.pt)" ]; do sleep 20; done
sleep 10
for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_mainline_live.sh m_main *') -or (\$_.Name -eq 'python.exe' -and (\$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main *' -or \$_.CommandLine -like '*fly_interface*E:/ib_checkpoints/m_main *' -or \$_.CommandLine -like '*fly_interface*E:/ib_checkpoints/m_main' -or \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main/gate_*')) } | % { \$_.ProcessId }" | tr -d '\r'); do
  powershell -c "Stop-Process -Id $p -Force" 2>/dev/null
done
sleep 5
cp $ck/c12.pt $ck/last.pt; rm -f $mm/scores_c13.jsonl $ck/last.pt.tmp*; rm -rf /e/ib_checkpoints/m_main_gate
echo "$(date +%H:%M) paused right after chunk 12 (gate c11 skipped) for the APL -> Kenyon fix: treatment = chunk 12 from c11 with --apl-kc-dendritic-only" >> $mm/status
rm -rf $o $k; mkdir -p $o $k
cp $ck/c11.pt $k/last.pt; cp $ck/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp $mm/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 --output $o --checkpoint-dir $k \
    --score-log $o/scores_c12.jsonl --log-word-dopamine --apl-kc-dendritic-only > $o/train.log 2>&1 \
  && echo "$(date +%H:%M) APL-fix treatment chunk 12 done: $(grep 'ACTIVE EVAL' $o/train.log | tail -1); $(grep 'APL -> Kenyon' $o/train.log | tail -1); skipped updates $(grep -c 'skipped the update' $o/train.log)" >> $mm/status \
  || echo "$(date +%H:%M) APL-fix treatment chunk 12 FAILED" >> $mm/status
echo "$(date +%H:%M) APL-fix c12 (1) vs control, paired: $($PY $FI/live_gate_report.py $mm/scores_c12.jsonl $o/scores_c12.jsonl aplfix)" >> $mm/status
echo "$(date +%H:%M) APL-fix c12 (2) surprise~DA: $($PY $FI/live_da_report.py $o/scores_c12.jsonl)" >> $mm/status
echo "$(date +%H:%M) control c12 (2) surprise~DA: $($PY $FI/live_da_report.py $mm/scores_c12.jsonl)" >> $mm/status
for arm in control treatment; do
  if [ $arm = control ]; then src=$ck; out=$mm; else src=$k; out=$o; fi
  $PY -u $FI/word_margin_probe.py $src 200 32 $out/word_margin_c12.json > $out/word_margin_c12.txt 2>&1 \
    && echo "$(date +%H:%M) $arm c12 word margin: $(tail -n 1 $out/word_margin_c12.txt)" >> $mm/status \
    || echo "$(date +%H:%M) $arm c12 word margin FAILED" >> $mm/status
  $PY -u $FI/kenyon_repeat_probe.py $src 3000 $out/kenyon_context_c12.json 0 > $out/kenyon_context_c12.txt 2>&1 \
    && echo "$(date +%H:%M) $arm c12 Kenyon repeat (same word / last 2 / last 3): $(tail -n 1 $out/kenyon_context_c12.txt)" >> $mm/status \
    || echo "$(date +%H:%M) $arm c12 Kenyon repeat FAILED" >> $mm/status
  $PY -u $FI/kenyon_gpt2_probe.py $src 3000 $out/kenyon_gpt2_c12.json > $out/kenyon_gpt2_c12.txt 2>&1 \
    && echo "$(date +%H:%M) $arm c12 P3 (GPT-2): $(tail -n 1 $out/kenyon_gpt2_c12.txt)" >> $mm/status \
    || echo "$(date +%H:%M) $arm c12 P3 FAILED" >> $mm/status
done
rm -f $k/birth.pt $k/last.pt.tmp*
echo "$(date +%H:%M) APL fix A/B done; m_main resumes from c12" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
