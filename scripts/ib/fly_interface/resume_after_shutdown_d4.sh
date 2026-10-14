#!/bin/bash
# After the 2026-10-14 shutdown: the surprise path probe on m_main c6, then D4 (resumes from its status), then m_main.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface; root=/e/fly_stream_overflow/m_main; ckpt=/e/ib_checkpoints/m_main
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
$PY -u $FI/surprise_path_probe.py $ckpt $root/surprise_path_c6.json > $root/surprise_path_c6.txt 2>&1
echo "$(date +%H:%M) surprise path c6: $(grep -E 'rho_' $root/surprise_path_c6.txt | tr -s ' ' | tr '\n' ';')" >> $root/status
SRC=m_main FROM=5 bash $FI/grow_d4_acts.sh d4_acts 8
echo "$(date +%H:%M) D4 done; resuming" >> $root/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
