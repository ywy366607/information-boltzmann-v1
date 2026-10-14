#!/bin/bash
# The compartment rule checked on its own before any training (user: "pause everything, look at the fixed brain
# alone"): m_main c11, zero-shot, the rule off vs on (ZS_COMPARTMENT applies it to the trained weights at load).
# Cell health per group (silent / saturated / text-following), the word margin (isolated words) and the Kenyon repeat
# probe (same word / same last 2-3 words, coding level).
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; ck=/e/ib_checkpoints/m_main; o=$mm/compartment_sanity; mkdir -p $o
for zs in 1 0; do
  if [ $zs = 1 ]; then export ZS_COMPARTMENT=1; else unset ZS_COMPARTMENT; fi
  $PY -u $FI/cell_health_probe.py $ck $o/cell_health_rule$zs.json > $o/cell_health_rule$zs.txt 2>&1
  echo "$(date +%H:%M) compartment sanity, rule=$zs: cell health written" >> $mm/status
  $PY -u $FI/word_margin_probe.py $ck 200 32 $o/word_margin_rule$zs.json > $o/word_margin_rule$zs.txt 2>&1
  $PY -u $FI/kenyon_repeat_probe.py $ck 3000 $o/kenyon_context_rule$zs.json 0 > $o/kenyon_context_rule$zs.txt 2>&1
  echo "$(date +%H:%M) compartment sanity, rule=$zs: margin and repeat written" >> $mm/status
done
echo "$(date +%H:%M) compartment sanity done" >> $mm/status
