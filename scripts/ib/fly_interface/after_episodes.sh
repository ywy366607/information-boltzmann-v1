#!/bin/bash
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; S=scratch/fly_stream
until grep -qE "^done|no checkpoint" $S/after_kc_status 2>/dev/null; do sleep 20; done
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
$PY -m pytest -q tests/test_fly_adaptive_admission.py > $S/tests_comp.log 2>&1; echo "tests $?" >> $S/after_episodes_status
$PY -u scripts/ib/fly_interface/taste_episodes.py $S/regime_kc 40 > $S/taste_episodes.log 2>&1; echo "episodes $?" >> $S/after_episodes_status
echo done >> $S/after_episodes_status
