#!/bin/bash
# After M1: the full test suite on the GPU, an M3 smoke run (train 64 words, probe), then M2 and M3 back to back.
# One torch process at a time.  Stops at the first failure (see $log).
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; log=/e/fly_stream_overflow/m2_m3.status
until grep -q done /e/fly_stream_overflow/m1_eyes/status 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M) M1 done" >> $log
$PY -m pytest -q tests/test_fly_adaptive_admission.py > /e/fly_stream_overflow/m2_m3_tests.txt 2>&1 || { echo "$(date +%H:%M) tests FAILED" >> $log; exit 1; }
echo "$(date +%H:%M) tests passed" >> $log
smoke=/e/fly_stream_overflow/smoke_m3; rm -rf $smoke
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
$PY -u scripts/ib/train_fly_bptt_stream.py --resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt --al-reset \
  --extend-budget --additional-tokens 128 --validate-every-tokens 128 --eval-tokens 32 --replay-tokens 32 \
  --window 32 --no-graph --go-neuron none --stream-ticks 8 --no-hold-input --stream-pace-mean 16 --stream-pace-min 2 \
  --no-homeostasis --no-ctm-loss --kc-dendritic-only --kc-renorm --connectome-v2 --dopamine-modulation \
  --dopamine-site pre --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 \
  --ai --eye-binary --eye-init 0.9 --ai-depth 2 --ai-sigreg 0.01 --noise-rate 1.0 --ai-epistemic 1.0 --ai-pragmatic 1.0 \
  --vram-limit-mib 3800 --output $smoke --checkpoint-dir $smoke > $smoke.log 2>&1 || { echo "$(date +%H:%M) M3 smoke train FAILED" >> $log; exit 1; }
ACT_BURN=32 ACT_SCORE=96 ACT_MUS=2 ACT_FIXED= ACT_NOISY=2 $PY -u scripts/ib/fly_interface/act_probe.py $smoke $smoke/act_probe.jsonl > $smoke.probe.txt 2>&1 || { echo "$(date +%H:%M) M3 smoke probe FAILED" >> $log; exit 1; }
echo "$(date +%H:%M) M3 smoke ok" >> $log
rm -f $smoke/last.pt
bash scripts/ib/fly_interface/grow_m2_output.sh 3 > /e/fly_stream_overflow/m2_output.log 2>&1
echo "$(date +%H:%M) M2 chain exit $?" >> $log
bash scripts/ib/fly_interface/grow_m3_both.sh 3 > /e/fly_stream_overflow/m3_both.log 2>&1
echo "$(date +%H:%M) M3 chain exit $?" >> $log
