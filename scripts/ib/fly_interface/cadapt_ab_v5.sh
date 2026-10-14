#!/bin/bash
# Change 1 treatment, v5 (v4 waited for the c10 gate, AG and probes, ~1 h; user: a mechanism test needs no
# brain-contribution gate, so start as soon as chunk 10 is saved; this time the mainline skips gate c9 / AG /
# probes c10).  From v4:  v3 could not start: the chain had pruned c7 (it keeps c1, every 10th and the latest two).
# So the pair moves to c9: m_main's own chunk 10 from c9 (running since 21:30, non-finite windows now skip) is the
# control; when it has trained and gated it (gate c9, AG and probes), stop the chain (chunk 11 has only just begun), restore c10, run
# the treatment = chunk 10 from c9 with --compartment-adapted on the same words, its MBON probe and (2) report, the
# control's MBON probe on c10, then resume m_main.
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; ck=/e/ib_checkpoints/m_main
o=/e/fly_stream_overflow/m_main_cadapt; k=/e/ib_checkpoints/m_main_cadapt
# chunk 10 trained and copied to c10.pt (the chain is then in its latch probe or gate: stopped)
until grep -q "chunk 10 done" $mm/status 2>/dev/null && [ -f $ck/c10.pt ] && [ "$(stat -c %s $ck/c10.pt)" = "$(stat -c %s $ck/last.pt)" ]; do sleep 20; done
sleep 10
for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_mainline_live.sh m_main *') -or (\$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main *') -or (\$_.Name -eq 'python.exe' -and (\$_.CommandLine -like '*fly_interface*E:/ib_checkpoints/m_main *' -or \$_.CommandLine -like '*fly_interface*E:/ib_checkpoints/m_main' -or \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main/gate_*')) } | % { \$_.ProcessId }" | tr -d '\r'); do
  powershell -c "Stop-Process -Id $p -Force" 2>/dev/null
done
sleep 5
cp $ck/c10.pt $ck/last.pt; rm -f $mm/scores_c11.jsonl $ck/last.pt.tmp*; rm -rf /e/ib_checkpoints/m_main_gate
echo "$(date +%H:%M) paused right after chunk 10 (gate c9 / AG / probes c10 skipped) for change 1: treatment = chunk 10 from c9 with compartment receptors adapted" >> $mm/status
rm -rf $o $k; mkdir -p $o $k
cp $ck/c9.pt $k/last.pt; cp $ck/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp $mm/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 --output $o --checkpoint-dir $k \
    --score-log $o/scores_c10.jsonl --log-word-dopamine --compartment-adapted > $o/train.log 2>&1 \
  && echo "$(date +%H:%M) treatment chunk 10 (compartment adapted) done: $(grep 'ACTIVE EVAL' $o/train.log | tail -1); skipped updates $(grep -c 'skipped the update' $o/train.log)" >> $mm/status \
  || echo "$(date +%H:%M) treatment chunk 10 FAILED (v5)" >> $mm/status
grep -q "Compartment receptors adapt" $o/train.log || echo "$(date +%H:%M) WARNING: treatment log lacks the adapted-receptor line" >> $mm/status
$PY -u $FI/mbon_input_probe.py $k 3000 $o/mbon_input_c10.json > $o/mbon_input_c10.txt 2>&1 \
  && echo "$(date +%H:%M) treatment c10 MBON probe: $(tail -n 1 $o/mbon_input_c10.txt)" >> $mm/status \
  || echo "$(date +%H:%M) treatment c10 MBON probe FAILED (v5)" >> $mm/status
$PY -u $FI/mbon_input_probe.py $ck 3000 $mm/mbon_input_c10.json > $mm/mbon_input_c10.txt 2>&1 \
  && echo "$(date +%H:%M) control c10 MBON probe: $(tail -n 1 $mm/mbon_input_c10.txt)" >> $mm/status \
  || echo "$(date +%H:%M) control c10 MBON probe FAILED (v5)" >> $mm/status
echo "$(date +%H:%M) treatment c10 (2) surprise~DA: $($PY $FI/live_da_report.py $o/scores_c10.jsonl)" >> $mm/status
echo "$(date +%H:%M) control c10 (2) surprise~DA: $($PY $FI/live_da_report.py $mm/scores_c10.jsonl)" >> $mm/status
rm -f $k/birth.pt $k/last.pt.tmp*
echo "$(date +%H:%M) change 1 treatment v5 done; m_main resumes from c10" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
