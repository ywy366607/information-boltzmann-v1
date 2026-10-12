#!/bin/bash
# After the regime gestation: the zero-shot checks, one torch process at a time.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; S=scratch/fly_stream; F=$S/regime_gestated
until grep -qE "gestated state written|Traceback" $S/regime_gestation.log; do sleep 20; done; sleep 30
[ -f $F/last.pt ] || { echo "no gestated checkpoint" > $S/after_regime_status; exit 1; }
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
$PY -u $S/membrane_tails.py $F > $S/regime_tails.log 2>&1; echo "tails $?" >> $S/after_regime_status; mv $S/membrane_tails.json $S/membrane_tails_regime.json 2>/dev/null
$PY -u $S/word_path.py $F > $S/regime_word_path.log 2>&1; echo "word_path $?" >> $S/after_regime_status; mv $S/word_path.json $S/word_path_regime.json 2>/dev/null
$PY -u $S/taste_phase0.py $F > $S/regime_phase0.log 2>&1; echo "phase0 $?" >> $S/after_regime_status
$PY -u $S/port_check.py $F > $S/regime_port.log 2>&1; echo "port $?" >> $S/after_regime_status
echo done >> $S/after_regime_status
