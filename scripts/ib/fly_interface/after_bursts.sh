#!/bin/bash
# After the growth chain stops: bursts in the adult's history on two clocks.  One torch process at a time.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; S=scratch/fly_stream; B=E:/ib_checkpoints/q8_fly_ctm_settle14_100k
until grep -q "^done" $S/grow_status 2>/dev/null; do sleep 30; done
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
rm -f $S/burst_simple.jsonl
for tpt in 1 15; do
  $PY -u scripts/ib/fly_interface/burst_simple.py $B/segments/learning_mode_before_100000/best.pt $tpt $B/last.pt >> $S/bursts_adult.log 2>&1
  $PY -u scripts/ib/fly_interface/burst_simple.py $B/last.pt $tpt >> $S/bursts_adult.log 2>&1
  $PY -u scripts/ib/fly_interface/burst_simple.py $S/dasite_pre/last.pt $tpt >> $S/bursts_adult.log 2>&1
  $PY -u scripts/ib/fly_interface/burst_simple.py $S/regime_kc/last.pt $tpt >> $S/bursts_adult.log 2>&1
done
rm -rf $S/_burst_tmp
echo done >> $S/bursts_status
