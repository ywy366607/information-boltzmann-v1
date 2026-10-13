#!/bin/bash
# One chain for every mainline individual.  Every switch comes from the trainer's PROFILES['mainline'] at birth and
# from the checkpoint's own config afterwards; a run names only what it changes (EXTRA, kept identical every chunk).
# Usage: EXTRA="--no-predictive-writer" bash grow_mainline.sh <name> <chunks>
# Gates after every chunk (recorded in status): latched cells (> 200 Hz, words withheld; stop when > 200) and the
# brain contribution (decoder alone vs brain).  Probes at c1, c5 and the last chunk.  Never edit while it runs:
# bash reads this file as it executes (copy it to a new name instead).
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; NAME=${1:?name}; CHUNKS=${2:-25}
root=/e/fly_stream_overflow/$NAME; ckpt=/e/ib_checkpoints/$NAME
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
PATHS="--output $root --checkpoint-dir $ckpt"
probes() {
  $PY -u scripts/ib/fly_interface/cell_health_probe.py $ckpt $root/cell_health_$1.json > $root/cell_health_$1.txt 2>&1
  $PY -u scripts/ib/fly_interface/kenyon_coding_probe.py $ckpt > $root/coding_$1.txt 2>&1
  for p in wait_oracle rarity_decode_probe gamma_tone_probe; do
    $PY -u scripts/ib/fly_interface/$p.py $ckpt $root/${p}_$1.json > $root/${p}_$1.txt 2>&1
  done
}
gate() {
  $PY -u scripts/ib/fly_interface/text_leak_probe.py $ckpt > $root/leak_$1.txt 2>&1
  n=$(grep -o "latched (> 200 Hz without words, >= 2 of 4 places): [0-9]*" $root/leak_$1.txt | grep -o "[0-9]*$")
  for sw in none noedges; do
    NLL_PLACES=2 NLL_SCORE=512 SWAP=$([ $sw = none ] || echo $sw) BIRTH=$ckpt/birth.pt $PY -u scripts/ib/fly_interface/swap_nll_probe.py $ckpt 2>&1 | grep "^{" >> $root/brain_$1.txt
  done
  c=$($PY -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; b,o=r[-2]['nll_places_min_median_mean_max'][2],r[-1]['nll_places_min_median_mean_max'][2]; print(f'brain {b:.3f} decoder-alone {o:.3f} contribution {o-b:+.3f} unigram {r[-2][\"unigram\"]:.3f}')" $root/brain_$1.txt)
  echo "$(date +%H:%M) gate $1: latched ${n:-?}; $(grep 'brain mean' $root/leak_$1.txt); $c" >> $root/status
  if [ -z "$n" ] || [ "$n" -gt 200 ]; then echo "$(date +%H:%M) EARLY STOP at $1: latched ${n:-?} > 200" >> $root/status; exit 2; fi
}
mkdir -p $root $ckpt
if [ ! -f $ckpt/last.pt ]; then
  $PY -u scripts/ib/train_fly_bptt_stream.py --from-scratch --additional-tokens 4096 $PATHS $EXTRA >> $root/train.log 2>&1 \
    || { echo "$(date +%H:%M) birth FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) born: chunk 1 done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1)" >> $root/status
  cp $ckpt/last.pt $ckpt/c1.pt; probes c1; gate c1
fi
done_chunks=$(grep -c "chunk [0-9]* done" $root/status)
for ((c=done_chunks+1; c<=CHUNKS; c++)); do
  $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 $PATHS $EXTRA >> $root/train.log 2>&1 \
    || { echo "$(date +%H:%M) chunk $c FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) chunk $c done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1)" >> $root/status
  if [ $c -eq 5 ] || [ $c -eq $CHUNKS ]; then cp $ckpt/last.pt $ckpt/c$c.pt; probes c$c; fi
  gate c$c
done
echo "$(date +%H:%M) $NAME finished" >> $root/status
