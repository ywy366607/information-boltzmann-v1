#!/bin/bash
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; S=scratch/fly_stream; F=$S/regime_tied
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
$PY -u scripts/ib/fly_interface/gestation_regime.py $S/regime_gestated $F 200000 0 tie > $S/tied_gestation.log 2>&1; echo "gestation $?" >> $S/after_tied_status
[ -f $F/last.pt ] || { echo "no checkpoint" >> $S/after_tied_status; exit 1; }
$PY -u $S/word_path.py $F > $S/tied_word_path.log 2>&1; echo "word_path $?" >> $S/after_tied_status; mv $S/word_path.json $S/word_path_tied.json 2>/dev/null
$PY -u $S/taste_phase0.py $F > $S/tied_phase0.log 2>&1; echo "phase0 $?" >> $S/after_tied_status
$PY -u $S/membrane_tails.py $F > $S/tied_tails.log 2>&1; echo "tails $?" >> $S/after_tied_status; mv $S/membrane_tails.json $S/membrane_tails_tied.json 2>/dev/null
echo done >> $S/after_tied_status
