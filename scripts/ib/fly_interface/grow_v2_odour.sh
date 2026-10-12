#!/bin/bash
# Proof of learnability on the corrected connectome: the adult with the audit's blocking fixes (Kenyon axo-axonal cut
# + renormalization, Kenyon->dopamine terminals, sensory terminals and octopamine/serotonin fast edges pinned, the
# antennal lobe back to birth once), the fixed compartment-memory rule, relative plasticity, the odour world with the
# lifetime information value of untasted odours.  Chunks of 4,096 words; after each: encounter summary, Kenyon
# specificity, the memory chain probe (memory -> MBON -> output), the language pace probe.
# Does evolution sharpen the Kenyon code and carry the memory to behaviour?  STOP in the output dir stops it.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; out=/e/fly_stream_overflow/v2_odour; mkdir -p $out
CHUNKS=${1:-4}
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
COMMON="--extend-budget --additional-tokens 4096 --validate-every-tokens 4096 --eval-tokens 256 --replay-tokens 128 \
  --window 32 --no-graph --go-neuron none --stream-ticks 8 --no-hold-input --stream-pace-mean 16 --stream-pace-min 4 \
  --no-homeostasis --no-ctm-loss --kc-dendritic-only --kc-renorm --connectome-v2 --dopamine-modulation \
  --dopamine-site pre --compartment-memory --comp-rate 2.0 --relative-edge-rate 0.02 --odour-world \
  --act-lambda 0.0003 --info-value --vram-limit-mib 4090"
for chunk in $(seq 1 $CHUNKS); do
  [ -f $out/STOP ] && break
  RESUME=""; AL=""; [ -f $out/last.pt ] || { RESUME="--resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt"; AL="--al-reset"; }
  $PY -u scripts/ib/train_fly_bptt_stream.py $RESUME $AL $COMMON --output $out --checkpoint-dir $out >> $out/train.log 2>&1
  code=$?; echo "$(date +%H:%M) chunk $chunk train $code" >> $out/status; rm -f $out/best.pt
  [ $code -ne 0 ] && break
  $PY scripts/ib/fly_interface/odour_summary.py $out/odour.jsonl 1024 > $out/odour_summary.txt 2>&1
  CONNECTOME_V2=1 $PY -u scripts/ib/fly_interface/kc_specificity.py $out $out/kc_specificity.jsonl > /dev/null 2>&1
  CONNECTOME_V2=1 $PY -u scripts/ib/fly_interface/memory_chain_probe.py $out $out/memory_chain.jsonl > /dev/null 2>&1
  $PY -u scripts/ib/fly_interface/pace_zero_shot.py $out $out/pace_probe.jsonl > $out/pace_probe_$chunk.txt 2>&1
  echo "$(date +%H:%M) chunk $chunk probes done" >> $out/status
done
echo done >> $out/status
