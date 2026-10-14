#!/bin/bash
# Change 1 treatment, requeued (cadapt_ab_after_d5b.sh fired early on an old "d5b done" line from 14:20 and its
# treatment ran out of GPU memory beside d5b; its zero-shot probes are valid).  This one waits until no d5b chain or
# trainer process is left (by process, not by a status line), then:
# 1. one treatment chunk (4096 words) from m_main c7 with --compartment-adapted, then its MBON probe and (2) report;
# 2. m_main resumes from c7: its own chunk 8 is the control, paired by window.
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; ck=/e/ib_checkpoints/m_main
o=/e/fly_stream_overflow/m_main_cadapt; k=/e/ib_checkpoints/m_main_cadapt
busy() {
  powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_acts_v2.sh d5b *') -or (\$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*') } | % { \$_.ProcessId }" | tr -d '\r' | grep -q .
}
while busy; do sleep 60; done
sleep 10
echo "$(date +%H:%M) d5b chain finished; change 1 treatment starts from c7" >> $mm/status
rm -rf $o $k; mkdir -p $o $k
cp $ck/c7.pt $k/last.pt; cp $ck/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp $mm/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 --output $o --checkpoint-dir $k \
    --score-log $o/scores_c8.jsonl --log-word-dopamine --compartment-adapted > $o/train.log 2>&1 \
  && echo "$(date +%H:%M) treatment chunk 8 (compartment adapted) done: $(grep 'ACTIVE EVAL' $o/train.log | tail -1)" >> $mm/status \
  || echo "$(date +%H:%M) treatment chunk 8 FAILED (v2)" >> $mm/status
grep -q "Compartment receptors adapt" $o/train.log || echo "$(date +%H:%M) WARNING: treatment log lacks the adapted-receptor line" >> $mm/status
$PY -u $FI/mbon_input_probe.py $k 3000 $o/mbon_input_c8.json > $o/mbon_input_c8.txt 2>&1 \
  && echo "$(date +%H:%M) treatment c8 MBON probe: $(tail -n 1 $o/mbon_input_c8.txt)" >> $mm/status \
  || echo "$(date +%H:%M) treatment c8 MBON probe FAILED (v2)" >> $mm/status
echo "$(date +%H:%M) treatment c8 (2) surprise~DA: $($PY $FI/live_da_report.py $o/scores_c8.jsonl)" >> $mm/status
rm -f $k/birth.pt $k/last.pt.tmp*
echo "$(date +%H:%M) change 1 treatment v2 done; m_main resumes from c7 (its chunk 8 is the control)" >> $mm/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
