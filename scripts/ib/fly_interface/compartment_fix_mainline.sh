#!/bin/bash
# The general compartment rule on the mainline (user: "fix them first"): every edge into Kenyon cells, MBONs, DANs, PNs
# and DNs scaled once by the share of its synapses outside the target's axonal territory (--compartment-dendritic;
# APL -> KC 0.155, PN 0.93 / 0.81, MBON 0.90 / 0.59, DN 0.91 / 0.87 excitatory / inhibitory).
# 1. when m_main's chunk 12 is saved: stop the chain (gate c11 skipped), and run the control = chunk 13 from c12 WITHOUT
#    the rule (m_main_nofix), with its probes (word margin, Kenyon repeat, P3) and (2);
# 2. the chain resumes with EXTRA=--compartment-dendritic (applied at this resume, inherited after);
# 3. when the mainline's chunk 13 is saved: the same probes on it and the paired NLL against the control; the chain
#    goes on.
# Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; ck=/e/ib_checkpoints/m_main
o=/e/fly_stream_overflow/m_main_nofix; k=/e/ib_checkpoints/m_main_nofix
stop_chain() {
  for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_mainline_live.sh m_main *') -or (\$_.Name -eq 'python.exe' -and (\$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main *' -or \$_.CommandLine -like '*fly_interface*E:/ib_checkpoints/m_main *' -or \$_.CommandLine -like '*fly_interface*E:/ib_checkpoints/m_main' -or \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main/gate_*')) } | % { \$_.ProcessId }" | tr -d '\r'); do
    powershell -c "Stop-Process -Id $p -Force" 2>/dev/null
  done
  sleep 5
}
probes() {   # $1 = checkpoint dir, $2 = output dir, $3 = label
  $PY -u $FI/word_margin_probe.py $1 200 32 $2/word_margin_c13.json > $2/word_margin_c13.txt 2>&1 \
    && echo "$(date +%H:%M) $3 c13 word margin: $(tail -n 1 $2/word_margin_c13.txt)" >> $mm/status
  $PY -u $FI/kenyon_repeat_probe.py $1 3000 $2/kenyon_context_c13.json 0 > $2/kenyon_context_c13.txt 2>&1 \
    && echo "$(date +%H:%M) $3 c13 Kenyon repeat: $(tail -n 1 $2/kenyon_context_c13.txt)" >> $mm/status
  $PY -u $FI/kenyon_gpt2_probe.py $1 3000 $2/kenyon_gpt2_c13.json > $2/kenyon_gpt2_c13.txt 2>&1 \
    && echo "$(date +%H:%M) $3 c13 P3 (GPT-2): $(tail -n 1 $2/kenyon_gpt2_c13.txt)" >> $mm/status
  echo "$(date +%H:%M) $3 c13 (2) surprise~DA: $($PY $FI/live_da_report.py $2/scores_c13.jsonl)" >> $mm/status
}
until grep -q "chunk 12 done" $mm/status 2>/dev/null && [ -f $ck/c12.pt ] && [ "$(stat -c %s $ck/c12.pt)" = "$(stat -c %s $ck/last.pt)" ]; do sleep 20; done
sleep 10; stop_chain
cp $ck/c12.pt $ck/last.pt; rm -f $mm/scores_c13.jsonl $ck/last.pt.tmp*; rm -rf /e/ib_checkpoints/m_main_gate
echo "$(date +%H:%M) paused right after chunk 12 (gate c11 skipped): control chunk 13 from c12 without the compartment rule" >> $mm/status
rm -rf $o $k; mkdir -p $o $k
cp $ck/c12.pt $k/last.pt; cp $ck/birth.pt $k/birth.pt
for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp $mm/$f $o/ 2>/dev/null; done
$PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 --output $o --checkpoint-dir $k \
    --score-log $o/scores_c13.jsonl --log-word-dopamine > $o/train.log 2>&1 \
  && echo "$(date +%H:%M) control chunk 13 (no compartment rule) done: $(grep 'ACTIVE EVAL' $o/train.log | tail -1)" >> $mm/status \
  || echo "$(date +%H:%M) control chunk 13 FAILED" >> $mm/status
probes $k $o control
rm -f $k/birth.pt $k/last.pt.tmp*
echo "$(date +%H:%M) the mainline resumes from c12 WITH the compartment rule (--compartment-dendritic)" >> $mm/status
EXTRA="--compartment-dendritic" nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
until grep -q "chunk 13 done" $mm/status 2>/dev/null && [ -f $ck/c13.pt ] && [ "$(stat -c %s $ck/c13.pt)" = "$(stat -c %s $ck/last.pt)" ]; do sleep 20; done
grep -q "Compartment rule" $mm/train.log && echo "$(date +%H:%M) $(grep 'Compartment rule' $mm/train.log | tail -1)" >> $mm/status \
  || echo "$(date +%H:%M) WARNING: the mainline log lacks the compartment-rule line" >> $mm/status
sleep 10; stop_chain
cp $ck/c13.pt $ck/last.pt; rm -f $mm/scores_c14.jsonl $ck/last.pt.tmp*; rm -rf /e/ib_checkpoints/m_main_gate
echo "$(date +%H:%M) paused right after chunk 13 (gate c12 skipped) for the compartment-rule probes" >> $mm/status
echo "$(date +%H:%M) compartment rule c13 (1) vs control, paired: $($PY $FI/live_gate_report.py $o/scores_c13.jsonl $mm/scores_c13.jsonl compartment)" >> $mm/status
probes $ck $mm fixed
echo "$(date +%H:%M) compartment-rule comparison done; the mainline goes on with the rule" >> $mm/status
EXTRA="--compartment-dendritic" nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
