#!/bin/bash
# Exploration pilots (2,048 words each, from the adult, the active-arm configuration): does the fly start to approach,
# and once it has tasted, does it tell sweet from bitter?
#   levy -- heavy-tailed spontaneous pushes into the act integrators (scale 0.005: a jump to the bound ~1/300 ticks)
#   info -- evolution's fitness counts the lifetime value of tasting an odour of unknown valence
#   both -- the two together
# Reference: the active arm's first chunk (avoid everything, never approach).  One torch process at a time.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; BASE=/e/fly_stream_overflow/explore_pilots; mkdir -p $BASE
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
COMMON="--resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt --extend-budget --additional-tokens 2048 \
  --validate-every-tokens 2048 --eval-tokens 128 --replay-tokens 64 --window 32 --no-graph --go-neuron none \
  --stream-ticks 8 --no-hold-input --stream-pace-mean 16 --stream-pace-min 4 --no-homeostasis --no-ctm-loss \
  --kc-dendritic-only --kc-renorm --dopamine-modulation --dopamine-site pre --compartment-memory --comp-rate 2.0 \
  --relative-edge-rate 0.02 --vram-limit-mib 4050 --odour-world --act-lambda 0.0003"
for arm in levy info both; do
  out=$BASE/$arm; rm -rf $out; mkdir -p $out
  case $arm in
    levy) EXTRA="--levy-scale 0.005";;
    info) EXTRA="--info-value";;
    both) EXTRA="--levy-scale 0.005 --info-value";;
  esac
  $PY -u scripts/ib/train_fly_bptt_stream.py $COMMON $EXTRA --output $out --checkpoint-dir $out > $out/train.log 2>&1
  echo "$(date +%H:%M) $arm train $?" >> $BASE/status
  $PY scripts/ib/fly_interface/odour_summary.py $out/odour.jsonl 512 > $out/odour_summary.txt 2>&1
  $PY scripts/ib/fly_interface/act_intervals.py $out/odour_acts.jsonl > $out/act_intervals.json 2>&1
  rm -f $out/last.pt $out/best.pt
done
echo done >> $BASE/status
