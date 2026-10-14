#!/bin/bash
# The mainline chain with living causal gates (2026-10-14: the brain is never frozen for a gate).
# Usage: EXTRA="..." GATE_WORDS=1024 bash grow_mainline_live.sh <name> <chunks>
# Every switch comes from PROFILES['mainline'] at birth and from the checkpoint's own config afterwards (EXTRA: only
# what a run changes, identical every chunk).  Every chunk logs its pre-update word scores and per-word dopamine.
# Gate for chunk c-1, evaluated after chunk c trains (its first GATE_WORDS words are the living brain arm):
#   (1) brain contribution = NLL(noedges arm) - NLL(brain), the same words and world, learning on, paired by window
#   (2) surprise -> phasic dopamine, Spearman on chunk c's own stream (no probe)
#   (L) latch worth = NLL(latched cells silenced arm) - NLL(brain), when >= 20 cells latch (mask: words-withheld probe)
#   (3) every 5 chunks: readiness for acts (probes) and fresh B / AG for brain and noedges arms
# The only early stop: brain contribution <= 0 at five gates in a row after 40k words.
# Never edit while it runs: bash reads this file as it executes (copy it to a new name instead).
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; NAME=${1:?name}; CHUNKS=${2:-25}; W=${GATE_WORDS:-1024}
root=/e/fly_stream_overflow/$NAME; ckpt=/e/ib_checkpoints/$NAME
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
FI=scripts/ib/fly_interface
LOGS="--log-word-dopamine"
latch_mask() {   # right after chunk $1 trains (last.pt = $1): the cells that stay > 200 Hz with the words withheld
  LATCHED_OUT=$root/latched_$1.npy $PY -u $FI/text_leak_probe.py $ckpt > $root/leak_$1.txt 2>&1
}
probes() {
  $PY -u $FI/cell_health_probe.py $ckpt $root/cell_health_$1.json > $root/cell_health_$1.txt 2>&1
  $PY -u $FI/kenyon_coding_probe.py $ckpt > $root/coding_$1.txt 2>&1
  for p in wait_oracle rarity_decode_probe gamma_tone_probe; do
    $PY -u $FI/$p.py $ckpt $root/${p}_$1.json > $root/${p}_$1.txt 2>&1
  done
}
arm() {   # $1 = source cN, $2 = arm name, $3 = ablation ('' = brain), $4 = 'eval' for fresh B
  local o=$root/gate_$1/$2 k=/e/ib_checkpoints/${NAME}_gate
  rm -rf $o $k; mkdir -p $o $k
  cp $ckpt/$1.pt $k/last.pt; cp $ckpt/birth.pt $k/birth.pt
  for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl; do cp $root/$f $o/ 2>/dev/null; done
  $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens $W --output $o --checkpoint-dir $k \
      --score-log $o/scores.jsonl $3 $([ "$4" = eval ] || echo --skip-eval) $EXTRA > $o/train.log 2>&1 \
      || echo "$(date +%H:%M) gate $1 arm $2 FAILED" >> $root/status
  rm -rf $k
}
gate() {  # $1 = gated checkpoint (c-1), $2 = the chunk trained from it (c), $3 = c (number)
  local g=$root/gate_$1; mkdir -p $g
  head -n $((W / 32)) $root/scores_$2.jsonl > $g/brain_scores.jsonl
  arm $1 noedges "--ablate noedges"
  c1=$($PY $FI/live_gate_report.py $g/brain_scores.jsonl $g/noedges/scores.jsonl noedges)
  n=$(grep -o "latched (> 200 Hz without words, >= 2 of 4 places): [0-9]*" $root/leak_$1.txt | grep -o "[0-9]*$")
  cl='{}'
  if [ "${n:-0}" -ge 20 ]; then
    arm $1 silenced "--ablate $root/latched_$1.npy"
    cl=$($PY $FI/live_gate_report.py $g/brain_scores.jsonl $g/silenced/scores.jsonl silenced)
  fi
  da=$($PY $FI/live_da_report.py $root/scores_$2.jsonl)
  if [ $(($3 % 5)) -eq 0 ]; then       # fresh B / AG, both arms learning
    arm $1 brain_ag "" eval; arm $1 noedges_ag "--ablate noedges" eval
  fi
  line=$($PY - "$c1" "$cl" "$da" "${n:-0}" <<'PY'
import json, sys
c, l, d = (json.loads(x) for x in sys.argv[1:4])
s = lambda r: f"{r['mean']:+.3f} +- {r['se']:.3f} [min {r['min_median_mean_max'][0]:+.2f} med {r['min_median_mean_max'][1]:+.2f} max {r['min_median_mean_max'][3]:+.2f}]" if 'mean' in r else 'n/a'
print(f"(1) brain contribution {s(c)} (brain {c.get('brain_nll')}, noedges {c.get('arm_nll')}, unigram {c.get('unigram')}, "
      f"{c.get('words')} words); (2) surprise~phasic DA {d.get('rho_surprise_same')} next {d.get('rho_surprise_next')} "
      f"DA cells {d.get('rho_surprise_cells')} per-1024 {d.get('rho_same_per_1024_min_median_mean_max')}; "
      f"(L) latched {sys.argv[4]}, worth {s(l)}")
PY
)
  echo "$(date +%H:%M) gate $1 (living): $line" >> $root/status
  if [ -f $g/brain_ag/train.log ]; then
    echo "$(date +%H:%M) gate $1 AG: brain $(grep 'ACTIVE EVAL' $g/brain_ag/train.log | tail -1 | grep -o 'fresh B.*'); noedges $(grep 'ACTIVE EVAL' $g/noedges_ag/train.log | tail -1 | grep -o 'fresh B.*')" >> $root/status
  fi
  if [ "$3" -ge 11 ]; then
    neg=$(grep -o "(1) brain contribution [-+0-9.]*" $root/status | tail -5 | awk '{if ($4+0 <= 0) c++} END {print c+0}')
    if [ "$neg" -ge 5 ]; then echo "$(date +%H:%M) STOP at $1: brain contribution <= 0 at 5 living gates in a row after 40k words" >> $root/status; exit 2; fi
  fi
}
mkdir -p $root $ckpt
if [ ! -f $ckpt/last.pt ]; then
  $PY -u scripts/ib/train_fly_bptt_stream.py --from-scratch --additional-tokens 4096 --output $root --checkpoint-dir $ckpt \
      --score-log $root/scores_c1.jsonl $LOGS $EXTRA >> $root/train.log 2>&1 \
    || { echo "$(date +%H:%M) birth FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) born: chunk 1 done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1)" >> $root/status
  cp $ckpt/last.pt $ckpt/c1.pt; latch_mask c1; probes c1
fi
done_chunks=$(grep -cE "^[0-9:]+ (born: )?chunk [0-9]+ done" $root/status)   # only the chain's own lines (a "treatment chunk N done" line once shifted the count)
for ((c=done_chunks+1; c<=CHUNKS; c++)); do
  $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 --output $root --checkpoint-dir $ckpt \
      --score-log $root/scores_c$c.jsonl $LOGS $EXTRA >> $root/train.log 2>&1 \
    || { echo "$(date +%H:%M) chunk $c FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) chunk $c done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1)" >> $root/status
  cp $ckpt/last.pt $ckpt/c$c.pt; latch_mask c$c
  gate c$((c - 1)) c$c $((c - 1))
  if [ $((c % 5)) -eq 0 ] || [ $c -eq $CHUNKS ]; then probes c$c; fi
  # keep the gated past bounded: c1, every 10th, and the latest two
  p=$((c - 2)); [ $p -gt 1 ] && [ $((p % 10)) -ne 0 ] && rm -f $ckpt/c$p.pt
done
echo "$(date +%H:%M) $NAME finished" >> $root/status
