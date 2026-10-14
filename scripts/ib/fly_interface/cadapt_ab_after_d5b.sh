#!/bin/bash
# Change 1 (readout first): the compartment memory's receptors adapt to their own slow dopamine (--compartment-adapted).
# When d5b is done and m_main resumes from c7: stop m_main at once (its chunk 8 has only just begun), restore c7, and
# 1. zero-shot: MBON probe on c7 with the rule off and on (the same weights; does phi learn within the probe stream?);
# 2. the treatment: one chunk (4096 words) from c7 with --compartment-adapted, pre-update scores and word dopamine
#    logged, then its MBON probe and its surprise -> DA report;
# 3. resume m_main: its own chunk 8 from c7 is the control, paired by window (same text, same start).
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; ck=/e/ib_checkpoints/m_main
o=/e/fly_stream_overflow/m_main_cadapt; k=/e/ib_checkpoints/m_main_cadapt
until grep -q "d5b done; m_main resumes from c7" $mm/status 2>/dev/null; do sleep 30; done
sleep 10
for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_mainline_live.sh m_main *') -or (\$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main *') -or (\$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main') } | % { \$_.ProcessId }" | tr -d '\r'); do
  powershell -c "Stop-Process -Id $p -Force" 2>/dev/null
done
sleep 5
cp $ck/c7.pt $ck/last.pt; rm -f $mm/scores_c8.jsonl $ck/last.pt.tmp*
echo "$(date +%H:%M) paused at c7 for change 1 (compartment receptors adapt): zero-shot probes, then one treatment chunk" >> $mm/status
for a in 0 1; do
  MBON_COMP_ADAPTED=$a $PY -u $FI/mbon_input_probe.py $ck 3000 $mm/mbon_input_c7_adapted$a.json > $mm/mbon_input_c7_adapted$a.txt 2>&1 \
    && echo "$(date +%H:%M) c7 MBON probe, compartment adapted=$a (zero-shot): $(tail -n 1 $mm/mbon_input_c7_adapted$a.txt)" >> $mm/status \
    || echo "$(date +%H:%M) c7 MBON probe adapted=$a FAILED" >> $mm/status
done
rm -rf $o $k; mkdir -p $o $k
cp $ck/c7.pt $k/last.pt; cp $ck/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp $mm/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 --output $o --checkpoint-dir $k \
    --score-log $o/scores_c8.jsonl --log-word-dopamine --compartment-adapted > $o/train.log 2>&1 \
  && echo "$(date +%H:%M) treatment chunk 8 (compartment adapted) done: $(grep 'ACTIVE EVAL' $o/train.log | tail -1)" >> $mm/status \
  || echo "$(date +%H:%M) treatment chunk 8 FAILED" >> $mm/status
grep -q "Compartment receptors adapt" $o/train.log || echo "$(date +%H:%M) WARNING: treatment log lacks the adapted-receptor line" >> $mm/status
$PY -u $FI/mbon_input_probe.py $k 3000 $o/mbon_input_c8.json > $o/mbon_input_c8.txt 2>&1 \
  && echo "$(date +%H:%M) treatment c8 MBON probe: $(tail -n 1 $o/mbon_input_c8.txt)" >> $mm/status \
  || echo "$(date +%H:%M) treatment c8 MBON probe FAILED" >> $mm/status
echo "$(date +%H:%M) treatment c8 (2) surprise~DA: $($PY $FI/live_da_report.py $o/scores_c8.jsonl)" >> $mm/status
rm -f $k/birth.pt $k/last.pt.tmp*
echo "$(date +%H:%M) change 1 treatment done; m_main resumes from c7 (its chunk 8 is the control)" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
