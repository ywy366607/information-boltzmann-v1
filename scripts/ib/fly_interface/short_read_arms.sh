#!/bin/bash
# Where is the belief read? (user 2026-10-13: try reading the prediction where it lives)  Two births of the mainline
# profile, 4096 words each: 'out' reads the prediction from the output cells (as before); 'mb' also from the Kenyon
# cells and MBONs (acts would stay on the output cells).  Measured in the training world: latched cells and the brain
# contribution (decoder alone vs brain).  First: R3 c5 re-measured in its own world (the fixed-8-tick probe was off-world).
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; out=/e/fly_stream_overflow/short_read
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
mkdir -p $out
brain() {   # $1 = fork dir, $2 = birth.pt, $3 = output file
  for sw in none noedges; do
    NLL_WORLD=train NLL_PLACES=2 NLL_SCORE=512 SWAP=$([ $sw = none ] || echo $sw) BIRTH=$2 $PY -u scripts/ib/fly_interface/swap_nll_probe.py $1 2>&1 | grep "^{" >> $3
  done
  $PY -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; b,o=r[-2]['nll_places_min_median_mean_max'][2],r[-1]['nll_places_min_median_mean_max'][2]; print(f'brain {b:.3f} decoder-alone {o:.3f} contribution {o-b:+.3f} unigram {r[-2][\"unigram\"]:.3f}')" $3
}
mkdir -p /e/ib_checkpoints/r3_c5view; cp /e/ib_checkpoints/r3_rebirth/c5.pt /e/ib_checkpoints/r3_c5view/last.pt
echo "$(date +%H:%M) R3 c5 in its own world: $(brain /e/ib_checkpoints/r3_c5view /e/ib_checkpoints/r3_rebirth/birth.pt $out/r3_c5_brain.txt)" >> $out/status
rm -rf /e/ib_checkpoints/r3_c5view
for arm in mb out; do
  extra=""; [ $arm = mb ] && extra="--read-extra-classes Kenyon_Cell,MBON"
  root=$out/$arm; ckpt=/e/ib_checkpoints/short_read_$arm
  rm -rf $root $ckpt; mkdir -p $root $ckpt
  $PY -u scripts/ib/train_fly_bptt_stream.py --from-scratch --additional-tokens 4096 $extra --output $root --checkpoint-dir $ckpt > $root/train.log 2>&1 \
    || { echo "$(date +%H:%M) $arm FAILED" >> $out/status; continue; }
  $PY -u scripts/ib/fly_interface/text_leak_probe.py $ckpt > $root/leak.txt 2>&1
  n=$(grep -o "latched (> 200 Hz without words, >= 2 of 4 places): [0-9]*" $root/leak.txt | grep -o "[0-9]*$")
  echo "$(date +%H:%M) $arm: latched ${n:-?}; $(grep 'brain mean' $root/leak.txt); $(brain $ckpt $ckpt/birth.pt $root/brain.txt); train $(grep 'ACTIVE EVAL' $root/train.log | tail -1)" >> $out/status
  rm -f $ckpt/last.pt $ckpt/best.pt
done
echo "$(date +%H:%M) read arms done" >> $out/status
