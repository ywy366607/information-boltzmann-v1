#!/bin/bash
# Same as run_damod_long.sh, but dopamine scales learning at the presynaptic terminal (--dopamine-site pre):
# forks damod2_gb (4096 tokens already); 4 x 8192 tokens, measured after every segment. STOP file to stop.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe
out=scratch/fly_stream/dasite_pre
mkdir -p $out; [ -f $out/last.pt ] || cp scratch/fly_stream/damod2_gb/last.pt scratch/fly_stream/damod2_gb/config.json $out/
for seg in 1 2 3 4; do
  [ -f $out/STOP ] && break
  FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8 $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget \
    --additional-tokens 8192 --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 --window 32 --no-graph \
    --go-neuron virtual --read-bank --predictive-writer --dopamine-modulation --dopamine-site pre --mixed-world \
    --output $out --checkpoint-dir $out >> $out.log 2>&1
  echo "segment $seg train exit $?" >> scratch/fly_stream/status_dasite_pre
  rm -f $out/best.pt
  [ -f $out/STOP ] && break
  PYTHONIOENCODING=utf-8 $PY -u scratch/fly_stream/dopamine_response.py $out > $out/seg${seg}_response.txt 2>&1
  cp $out/dopamine_response.json $out/seg${seg}_dopamine_response.json
  PYTHONIOENCODING=utf-8 $PY -u scratch/fly_stream/dopamine_states.py $out > $out/seg${seg}_states.txt 2>&1
  cp $out/dopamine_states.json $out/seg${seg}_dopamine_states.json
  echo "segment $seg measured" >> scratch/fly_stream/status_dasite_pre
done
echo done >> scratch/fly_stream/status_dasite_pre
