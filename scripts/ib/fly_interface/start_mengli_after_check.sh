#!/bin/bash
# After the compartment-rule comparison (compartment_fix_mainline_v3.sh): if the rule costs no more than 0.05 nats per
# word against the control (paired over the chunk's windows), stop the mainline chain that v3 restarts (m_main stays
# at c12 as the testbed) and begin Mengli's raising lineage (grow_mengli.sh).  Otherwise leave the mainline running and
# write why to the status.  Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
mm=/e/fly_stream_overflow/m_main; ck=/e/ib_checkpoints/m_main
until grep -q "compartment-rule comparison done" $mm/status 2>/dev/null; do sleep 20; done
sleep 20
cost=$(grep "compartment rule c12 vs control, paired NLL" $mm/status | tail -1 | $PY -c "
import sys, json, re
line = sys.stdin.read(); m = re.search(r'(\{.*\})', line)
print(json.loads(m.group(1))['mean'] if m else 'nan')")
ok=$($PY -c "import math; c=float('$cost'); print(1 if math.isfinite(c) and c <= 0.05 else 0)")
if [ "$ok" != 1 ]; then
  echo "$(date +%H:%M) Mengli NOT started: the compartment rule's paired cost is $cost nats (> 0.05 or missing); the mainline keeps running" >> $mm/status
  exit 0
fi
for p in $(powershell -c "Get-CimInstance Win32_Process | ? { (\$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*bash.exe*\" scripts/ib/fly_interface/grow_mainline_live.sh m_main *') -or (\$_.Name -eq 'python.exe' -and (\$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main *' -or \$_.CommandLine -like '*fly_interface*E:/ib_checkpoints/m_main *' -or \$_.CommandLine -like '*fly_interface*E:/ib_checkpoints/m_main' -or \$_.CommandLine -like '*train_fly_bptt_stream.py*fly_stream_overflow/m_main/gate_*')) } | % { \$_.ProcessId }" | tr -d '\r'); do
  powershell -c "Stop-Process -Id $p -Force" 2>/dev/null
done
sleep 5
cp $ck/c12.pt $ck/last.pt; rm -f $mm/scores_c13.jsonl $ck/last.pt.tmp*; rm -rf /e/ib_checkpoints/m_main_gate
echo "$(date +%H:%M) the compartment rule costs $cost nats (<= 0.05): adopted; m_main paused at c12 as the testbed; Mengli's raising lineage begins (grow_mengli.sh)" >> $mm/status
nohup bash $FI/grow_mengli.sh 200 > /dev/null 2>&1 &
