#!/bin/bash
# The adult (settle-14 lineage) with the connectome fixes (Kenyon axo-axonal input cut, Kenyon inputs renormalized over
# dendritic partners), dopamine modulation and compartment memory (rate 2), relative plasticity (2% of the birth weight
# per step), text at the world's pace (mean interval log-uniform in [4, 16], pulses), no flux / go / CTM / homeostasis.
#   active:  + the closed-loop odour world and the act head (act fitness weight 3e-4)
#   control: the same without the odour world
# The two arms alternate chunk by chunk (4,096 words each; one torch process at a time).  After each chunk: the pace
# probe (language, fixed text) for that arm; the encounter summary for the active arm.  STOP in the base dir stops.
# Usage: grow_odour_arms.sh [chunks per arm]
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; BASE=/e/fly_stream_overflow/odour_arms; mkdir -p $BASE
CHUNKS=${1:-5}
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
COMMON="--extend-budget --additional-tokens 4096 --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 \
  --window 32 --no-graph --go-neuron none --stream-ticks 8 --no-hold-input --stream-pace-mean 16 --stream-pace-min 4 \
  --no-homeostasis --no-ctm-loss --kc-dendritic-only --kc-renorm --dopamine-modulation --dopamine-site pre \
  --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 --vram-limit-mib 4050"
for chunk in $(seq 1 $CHUNKS); do
  for arm in active control; do
    [ -f $BASE/STOP ] && exit 0
    out=$BASE/$arm; mkdir -p $out
    done_chunks=$(grep -c ' train 0' $out/status 2>/dev/null); done_chunks=${done_chunks:-0}
    [ $done_chunks -ge $chunk ] && continue
    RESUME=""; [ -f $out/last.pt ] || RESUME="--resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt"
    EXTRA="--no-odour-world"; [ $arm = active ] && EXTRA="--odour-world --act-lambda 0.0003"
    $PY -u scripts/ib/train_fly_bptt_stream.py $RESUME $COMMON $EXTRA --output $out --checkpoint-dir $out >> $out/train.log 2>&1
    code=$?; echo "$(date +%H:%M) chunk $chunk train $code" >> $out/status; rm -f $out/best.pt
    [ $code -ne 0 ] && exit 1
    $PY -u scripts/ib/fly_interface/pace_zero_shot.py $out $out/pace_probe.jsonl > $out/pace_probe_$chunk.txt 2>&1
    echo "$(date +%H:%M) chunk $chunk probe $?" >> $out/status
    if [ $arm = active ]; then
      $PY scripts/ib/fly_interface/odour_summary.py $out/odour.jsonl > $out/odour_summary.txt 2>&1
    fi
  done
done
echo done >> $BASE/status
