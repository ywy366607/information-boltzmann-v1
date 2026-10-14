#!/bin/bash
# v3 of the compartment-rule comparison (user: speed decides how soon we are back on the mainline): one merged probe
# (snapshot_probe.py, ~5 min) instead of three, and the control's trainer is waited for by process.
# 1. when the control (chunk 12 from c11 without the rule, already running) ends: snapshot + (2) on it;
# 2. the mainline continues from c11 WITH the rule (its own chunk 12, progress, not a test cost);
# 3. when it is saved: snapshot + (2) + paired NLL against the control, then the chain goes on.
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; ck=/e/ib_checkpoints/m_main
o=/e/fly_stream_overflow/m_main_nofix; k=/e/ib_checkpoints/m_main_nofix
running() { powershell -c "Get-CimInstance Win32_Process | ? { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*m_main_nofix*' } | % { \$_.ProcessId }" | tr -d '\r' | grep -q .; }
while running; do sleep 15; done
grep -q "ACTIVE EVAL 49152" $o/train.log && echo "$(date +%H:%M) control chunk 12 (no compartment rule) done: $(grep 'ACTIVE EVAL' $o/train.log | tail -1)" >> $mm/status \
  || echo "$(date +%H:%M) control chunk 12 did not finish (see $o/train.log)" >> $mm/status
$PY -u $FI/snapshot_probe.py $k $o/snapshot_c12.json > $o/snapshot_c12.txt 2>&1 \
  && echo "$(date +%H:%M) control c12 snapshot: $(tail -n 1 $o/snapshot_c12.txt)" >> $mm/status \
  || echo "$(date +%H:%M) control c12 snapshot FAILED" >> $mm/status
echo "$(date +%H:%M) control c12 (2) surprise~DA: $($PY $FI/live_da_report.py $o/scores_c12.jsonl)" >> $mm/status
rm -f $k/birth.pt $k/last.pt.tmp*
echo "$(date +%H:%M) the mainline resumes from c11 WITH the compartment rule (--compartment-dendritic)" >> $mm/status
EXTRA="--compartment-dendritic" nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
until grep -q "chunk 12 done" $mm/status 2>/dev/null && [ -f $ck/c12.pt ] && [ "$(stat -c %s $ck/c12.pt)" = "$(stat -c %s $ck/last.pt)" ]; do sleep 15; done
grep -q "Compartment rule" $mm/train.log && echo "$(date +%H:%M) $(grep 'Compartment rule' $mm/train.log | tail -1)" >> $mm/status \
  || echo "$(date +%H:%M) WARNING: the mainline log lacks the compartment-rule line" >> $mm/status
sleep 5
for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_mainline_live.sh m_main *') -or (\$_.Name -eq 'python.exe' -and (\$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main *' -or \$_.CommandLine -like '*fly_interface*E:/ib_checkpoints/m_main *' -or \$_.CommandLine -like '*fly_interface*E:/ib_checkpoints/m_main' -or \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main/gate_*')) } | % { \$_.ProcessId }" | tr -d '\r'); do
  powershell -c "Stop-Process -Id $p -Force" 2>/dev/null
done
sleep 5
cp $ck/c12.pt $ck/last.pt; rm -f $mm/scores_c13.jsonl $ck/last.pt.tmp*; rm -rf /e/ib_checkpoints/m_main_gate
echo "$(date +%H:%M) compartment rule c12 vs control, paired NLL (rule - control): $($PY $FI/live_gate_report.py $o/scores_c12.jsonl $mm/scores_c12.jsonl compartment)" >> $mm/status
echo "$(date +%H:%M) compartment rule c12 (2) surprise~DA: $($PY $FI/live_da_report.py $mm/scores_c12.jsonl)" >> $mm/status
$PY -u $FI/snapshot_probe.py $ck $mm/snapshot_c12.json > $mm/snapshot_c12.txt 2>&1 \
  && echo "$(date +%H:%M) compartment rule c12 snapshot: $(tail -n 1 $mm/snapshot_c12.txt)" >> $mm/status \
  || echo "$(date +%H:%M) compartment rule c12 snapshot FAILED" >> $mm/status
echo "$(date +%H:%M) compartment-rule comparison done; the mainline goes on with the rule (gate c11 skipped this once)" >> $mm/status
EXTRA="--compartment-dendritic" nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
