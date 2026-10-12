#!/bin/bash
# Queued after grow_taste_only.sh (waits for its 'taste-only chain done'): the self-model arm = the active arm (odour
# world, act head, outcome dopamine as in the active arm) + the brain predicts its other senses one tick ahead (odour
# levels per receptor type, sweet, bitter) + the world sends gusts (world-caused odour rises identical to an approach).
# Test: prediction error on self-caused rises (its own approach) vs world-caused rises (gusts).
# Gate: a 64-word smoke run per sense weight (1, 0.1, 0.01); the first with a finite sense error and a total gradient
# norm below 5 is used; if none passes, nothing long runs.  One torch process at a time.  STOP in the base dir stops.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; BASE=/e/fly_stream_overflow/odour_arms
until grep -q 'taste-only chain done' $BASE/status 2>/dev/null; do sleep 60; done
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
COMMON="--extend-budget --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 \
  --window 32 --no-graph --go-neuron none --stream-ticks 8 --no-hold-input --stream-pace-mean 16 --stream-pace-min 4 \
  --no-homeostasis --no-ctm-loss --kc-dendritic-only --kc-renorm --dopamine-modulation --dopamine-site pre \
  --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 --vram-limit-mib 4050 \
  --odour-world --act-lambda 0.0003 --sense-prediction"
LAMBDA=""
for lam in 1.0 0.1 0.01; do
  smoke=$BASE/smoke_sense; rm -rf $smoke
  $PY -u scripts/ib/train_fly_bptt_stream.py --resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt $COMMON \
    --additional-tokens 64 --validate-every-tokens 64 --eval-tokens 32 --replay-tokens 32 --sense-lambda $lam \
    --output $smoke --checkpoint-dir $smoke > $BASE/smoke_sense_$lam.log 2>&1
  ok=$($PY -c "
import json, math
r = [json.loads(l) for l in open('E:/fly_stream_overflow/odour_arms/smoke_sense/metrics.jsonl') if 'sense_err' in l]
print(1 if r and all(math.isfinite(x['sense_err']) and x['grad_norm_before_clip'] < 5 for x in r) else 0)" 2>/dev/null)
  echo "$(date +%H:%M) smoke sense lambda $lam ok=$ok" >> $BASE/status
  rm -rf $smoke
  if [ "$ok" = "1" ]; then LAMBDA=$lam; break; fi
done
[ -z "$LAMBDA" ] && { echo "$(date +%H:%M) self-model arm not started (smoke failed)" >> $BASE/status; exit 1; }
out=$BASE/self_model; mkdir -p $out
for chunk in 1 2 3 4 5; do
  [ -f $BASE/STOP ] && exit 0
  RESUME=""; [ -f $out/last.pt ] || RESUME="--resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt"
  $PY -u scripts/ib/train_fly_bptt_stream.py $RESUME $COMMON --additional-tokens 4096 --sense-lambda $LAMBDA \
    --output $out --checkpoint-dir $out >> $out/train.log 2>&1
  code=$?; echo "$(date +%H:%M) chunk $chunk train $code" >> $out/status; rm -f $out/best.pt
  [ $code -ne 0 ] && exit 1
  $PY -u scripts/ib/fly_interface/pace_zero_shot.py $out $out/pace_probe.jsonl > $out/pace_probe_$chunk.txt 2>&1
  $PY scripts/ib/fly_interface/odour_summary.py $out/odour.jsonl > $out/odour_summary.txt 2>&1
done
$PY scripts/ib/fly_interface/act_intervals.py $out/odour_acts.jsonl > $out/act_intervals.json 2>&1
echo "$(date +%H:%M) self-model chain done (sense lambda $LAMBDA)" >> $BASE/status
