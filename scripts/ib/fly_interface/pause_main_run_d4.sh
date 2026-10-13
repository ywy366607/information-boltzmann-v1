#!/bin/bash
# After m_main's gate c5 (with the AG arms): pause the mainline between chunks, run the surprise path probe and the D4
# act fork from c5 (3 chunks), then resume the mainline.  The mainline chain resumes by itself from its status (chunks
# done are counted there; last.pt is restored to the last finished chunk, the half-trained chunk discarded).
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
root=/e/fly_stream_overflow/m_main; ckpt=/e/ib_checkpoints/m_main
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
until grep -q "gate c5 AG\|FAILED\|STOP" $root/status; do sleep 60; done
grep -q "FAILED\|STOP" $root/status && exit 1
until [ -f $root/scores_c7.jsonl ]; do sleep 10; done        # chunk 7 has started: chunk 6 and its gate are complete
# stop the chain (bash running grow_mainline_live.sh) first, then its trainer; never this shell or PowerShell itself
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*grow_mainline_live.sh m_main*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*train_fly_bptt_stream.py*' -and \$_.CommandLine -like '*m_main*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
sleep 15
cp $ckpt/c6.pt $ckpt/last.pt; rm -f $root/scores_c7.jsonl
echo "$(date +%H:%M) paused after c6 (chunk 7 discarded) for the surprise path probe and D4" >> $root/status
$PY -u $FI/surprise_path_probe.py $ckpt $root/surprise_path_c6.json > $root/surprise_path_c6.txt 2>&1
echo "$(date +%H:%M) surprise path c6: $(grep -E 'rho_' $root/surprise_path_c6.txt | tr -s ' ' | tr '\n' ';')" >> $root/status
SRC=m_main FROM=5 bash $FI/grow_d4_acts.sh d4_acts 8
echo "$(date +%H:%M) D4 done; resuming" >> $root/status
nohup bash $FI/grow_mainline_live.sh m_main 40 > /dev/null 2>&1 &
