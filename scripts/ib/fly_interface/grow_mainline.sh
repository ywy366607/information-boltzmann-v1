#!/bin/bash
# One chain for every mainline individual.  Every switch comes from the trainer's PROFILES['mainline'] at birth and
# from the checkpoint's own config afterwards; a run names only what it changes (EXTRA, kept identical every chunk).
# Usage: EXTRA="--no-predictive-writer" bash grow_mainline.sh <name> <chunks>
# Gates after every chunk, in the mainline's causal order (status): (1) brain contribution, (2) surprise -> phasic
# dopamine, (L) what the latched cells are worth; every 5 chunks (3) readiness for acts (probes).  The only early stop:
# brain contribution <= 0 at five gates in a row after 40k words.  Probes at c1, c5 and the last chunk.  Never edit while it runs:
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
gate() {     # $1 = label, $2 = chunk number.  The mainline's causal order, every chunk:
  # (1) the brain carries the prediction: brain contribution = NLL(decoder alone, no synapses) - NLL(brain), training world
  # (2) surprise reaches dopamine: Spearman of word surprisal with phasic dopamine (same word, next word)
  # (L) latching by cause: NLL with the latched cells silenced - NLL(brain) (> 0: they are used, as the champion's +0.10)
  LATCHED_OUT=$root/latched_$1.npy $PY -u scripts/ib/fly_interface/text_leak_probe.py $ckpt > $root/leak_$1.txt 2>&1
  n=$(grep -o "latched (> 200 Hz without words, >= 2 of 4 places): [0-9]*" $root/leak_$1.txt | grep -o "[0-9]*$")
  for sw in none noedges silenced; do
    env $([ $sw = silenced ] && echo ZS_SILENCE_MASK=$root/latched_$1.npy) NLL_WORLD=train NLL_PLACES=2 NLL_SCORE=512       SWAP=$([ $sw = noedges ] && echo noedges) BIRTH=$ckpt/birth.pt $PY -u scripts/ib/fly_interface/swap_nll_probe.py $ckpt 2>&1 | grep "^{" >> $root/brain_$1.txt
  done
  DA_PLACES=2 DA_SCORE=512 $PY -u scripts/ib/fly_interface/surprise_dopamine_probe.py $ckpt $root/surprise_da_$1.json > /dev/null 2>&1
  line=$($PY - $root/brain_$1.txt $root/surprise_da_$1.json "$n" <<'PY'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])][-3:]
b, o, s = (x['nll_places_min_median_mean_max'][2] for x in r)
try:
    d = json.load(open(sys.argv[2])); da = f"(2) surprise~phasic DA {d['rho_surprise_phasic'][2]:+.3f} next {d['rho_surprise_next_phasic'][2]:+.3f} DA cells {d['rho_surprise_dan_rate'][2]:+.3f}, phasic sd {d['phasic_sd_words'][2]:.3f}"
except Exception:
    da = '(2) n/a'
print(f"(1) brain contribution {o - b:+.3f} (brain {b:.3f}, decoder alone {o:.3f}, unigram {r[0]['unigram']:.3f}); {da}; (L) latched {sys.argv[3]}, worth {s - b:+.3f}")
PY
)
  echo "$(date +%H:%M) gate $1: $line" >> $root/status
  # the only early stop: after 40k words, the brain contribution <= 0 at five gates in a row (the brain does not learn
  # language at all after the changes: a big problem to trace)
  if [ "$2" -ge 10 ]; then
    neg=$(grep -o "(1) brain contribution [-+0-9.]*" $root/status | tail -5 | awk '{if ($4+0 <= 0) c++} END {print c+0}')
    if [ "$neg" -ge 5 ]; then echo "$(date +%H:%M) STOP at $1: brain contribution <= 0 at 5 gates in a row after 40k words" >> $root/status; exit 2; fi
  fi
}
mkdir -p $root $ckpt
if [ ! -f $ckpt/last.pt ]; then
  $PY -u scripts/ib/train_fly_bptt_stream.py --from-scratch --additional-tokens 4096 $PATHS $EXTRA >> $root/train.log 2>&1 \
    || { echo "$(date +%H:%M) birth FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) born: chunk 1 done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1)" >> $root/status
  cp $ckpt/last.pt $ckpt/c1.pt; probes c1; gate c1 1
fi
done_chunks=$(grep -c "chunk [0-9]* done" $root/status)
for ((c=done_chunks+1; c<=CHUNKS; c++)); do
  $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 $PATHS $EXTRA >> $root/train.log 2>&1 \
    || { echo "$(date +%H:%M) chunk $c FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) chunk $c done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1)" >> $root/status
  # (3) readiness for acts every 5 chunks: difficulty readable early from the motor cells, value of thinking, gamma tone
  if [ $((c % 5)) -eq 0 ] || [ $c -eq $CHUNKS ]; then cp $ckpt/last.pt $ckpt/c$c.pt; probes c$c; [ $((c - 5)) -gt 5 ] && rm -f $ckpt/c$((c - 5)).pt; fi
  gate c$c $c
done
echo "$(date +%H:%M) $NAME finished" >> $root/status
