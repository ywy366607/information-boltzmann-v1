#!/bin/bash
# After D1 and its probes: GPU tests, a split smoke (from the adult, then a resume of the split lineage), then D2.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; log=/e/fly_stream_overflow/d2_gate.status
until grep -q "probes done" "$1" 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M) D1 probes done" >> $log
$PY -m pytest -q tests/test_fly_adaptive_admission.py > /e/fly_stream_overflow/d2_gate_tests.txt 2>&1 || { echo "$(date +%H:%M) tests FAILED" >> $log; exit 1; }
echo "$(date +%H:%M) tests passed" >> $log
smoke=/e/fly_stream_overflow/smoke_split; rm -rf $smoke
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
A="--extend-budget --additional-tokens 64 --validate-every-tokens 64 --eval-tokens 32 --replay-tokens 32 --window 32 --no-graph \
  --go-neuron none --stream-ticks 8 --hold-input --sense-noise 2.0 --stream-pace-mean 16 --stream-pace-min 2 --no-homeostasis \
  --no-ctm-loss --kc-dendritic-only --kc-renorm --connectome-v2 --dopamine-modulation --dopamine-site pre --compartment-memory \
  --comp-rate 2.0 --relative-edge-rate 0.02 --ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior \
  --ai-epistemic 0.0 --ai-pragmatic 1.0 --ai-sigreg 0.01 --eye-forced-open --learn-stp --kc-subtypes --vram-limit-mib 3800 \
  --output $smoke --checkpoint-dir $smoke"
$PY -u scripts/ib/train_fly_bptt_stream.py --resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt --al-reset $A > $smoke.log1 2>&1 || { echo "$(date +%H:%M) split smoke 1 FAILED" >> $log; exit 1; }
$PY -u scripts/ib/train_fly_bptt_stream.py $A > $smoke.log2 2>&1 || { echo "$(date +%H:%M) split smoke 2 (resume) FAILED" >> $log; exit 1; }
grep -h "Kenyon subtypes" $smoke.log1 $smoke.log2 >> $log
rm -rf $smoke
echo "$(date +%H:%M) split smoke ok; launching D2" >> $log
bash scripts/ib/fly_interface/grow_d2_accumulate.sh 3 > /e/fly_stream_overflow/d2_accumulate.log 2>&1
echo "$(date +%H:%M) D2 chain exit $?" >> $log
