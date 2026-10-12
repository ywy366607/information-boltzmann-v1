#!/bin/bash
# Grow the regime newborn: read and learn (BPTT) in chunks of 4,096 tokens; after each chunk measure population bursts
# and the word path.  One torch process at a time.  STOP file in the output dir stops it.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; S=scratch/fly_stream; out=$S/grow_regime
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mkdir -p $out; [ -f $out/last.pt ] || cp $S/regime_kc/last.pt $S/regime_kc/config.json $out/
for chunk in 1 2 3 4; do
  [ -f $out/STOP ] && break
  $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 \
    --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph \
    --go-neuron virtual --read-bank --predictive-writer --dopamine-modulation --dopamine-site pre --dopamine-affinity 0.00457 \
    --kc-dendritic-only --mixed-world --vram-limit-mib 4050 --output $out --checkpoint-dir $out >> $out.log 2>&1
  code=$?; echo "chunk $chunk train $code" >> $S/grow_status; rm -f $out/best.pt
  [ $code -ne 0 ] && break
  $PY -u $S/kc_bursts.py $out > $out/bursts_$chunk.txt 2>&1; mv $out/kc_bursts_r1.json $out/kc_bursts_chunk$chunk.json 2>/dev/null
  $PY -u $S/word_path.py $out > $out/word_path_$chunk.txt 2>&1; mv $S/word_path.json $out/word_path_chunk$chunk.json 2>/dev/null
  echo "chunk $chunk probes done" >> $S/grow_status
done
echo done >> $S/grow_status
