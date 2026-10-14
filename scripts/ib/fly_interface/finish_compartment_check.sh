#!/bin/bash
# The end of the compartment-rule comparison by hand (the v3 queue waited for a chunk number the miscount never wrote):
# the snapshot on the rule's c12, then Mengli's lineage (the paired cost was -0.0035 +- 0.0036 nats: no harm).
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mm=/e/fly_stream_overflow/m_main; ck=/e/ib_checkpoints/m_main
$PY -u $FI/snapshot_probe.py $ck $mm/snapshot_c12.json > $mm/snapshot_c12.txt 2>&1 \
  && echo "$(date +%H:%M) compartment rule c12 snapshot: $(tail -n 1 $mm/snapshot_c12.txt)" >> $mm/status \
  || echo "$(date +%H:%M) compartment rule c12 snapshot FAILED" >> $mm/status
echo "$(date +%H:%M) the compartment rule is adopted (no harm: -0.0035 +- 0.0036 nats); m_main paused at c12 as the testbed; Mengli's raising lineage begins" >> $mm/status
nohup bash $FI/grow_mengli.sh 200 > /dev/null 2>&1 &
