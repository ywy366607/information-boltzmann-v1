#!/bin/bash
# grow_d4_acts.sh plus a blind arm in every gate (2026-10-14): the same individual, still learning, with the words never
# reaching the senses (--ablate blind).  brain - blind is what READING is worth; noedges (every synapse cut) is worse
# than the unigram, so brain - noedges overstates it.  D5 found the eyes-closed brain still +0.12 over the unigram.
# D4 (docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261014_ACTS.md): acts on a brain that carries prediction.
# A fork of a mainline run at chunk FROM (its checkpoint, status and scores), continued with commits sampled from the
# motor cells (exact expectation over commit times, listener-prior miss, habit prior, the policy gradient reaching the
# brain), eyes open, the mainline's world.  The same living gates as the mainline, plus (3) from the living stream:
# live_act_report.py on the chunk's ai_trace (answered fraction, latency, its entropy, Spearman(latency, CE)).
# Usage: SRC=m_main FROM=5 ACTS="..." bash grow_d4_acts.sh <name> <chunks (absolute, > FROM)>
# Never edit while it runs.
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
  arm $1 blind "--ablate blind"
  cb=$($PY $FI/live_gate_report.py $g/brain_scores.jsonl $g/blind/scores.jsonl blind)
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
  line=$($PY - "$c1" "$cl" "$da" "${n:-0}" "$cb" <<'PY'
import json, sys
c, l, d = (json.loads(x) for x in sys.argv[1:4])
b = json.loads(sys.argv[5])
s = lambda r: f"{r['mean']:+.3f} +- {r['se']:.3f} [min {r['min_median_mean_max'][0]:+.2f} med {r['min_median_mean_max'][1]:+.2f} max {r['min_median_mean_max'][3]:+.2f}]" if 'mean' in r else 'n/a'
print(f"(1) brain contribution {s(c)} (brain {c.get('brain_nll')}, noedges {c.get('arm_nll')}, unigram {c.get('unigram')}, "
      f"{c.get('words')} words); (1b) reading (brain - blind) {s(b)} (blind {b.get('arm_nll')}); (2) surprise~phasic DA {d.get('rho_surprise_same')} next {d.get('rho_surprise_next')} "
      f"DA cells {d.get('rho_surprise_cells')} rarity {d.get('rho_surprise_rarity_same')} residual {d.get('rho_surprise_residual_same')} "
      f"per-1024 {d.get('rho_same_per_1024_min_median_mean_max')}; "
      f"(L) latched {sys.argv[4]}, worth {s(l)}")
PY
)
  echo "$(date +%H:%M) gate $1 (living): $line" >> $root/status
  if [ -f $root/ai_trace.jsonl ]; then   # (3) the acts in the living stream, chunk $2's windows only
    first=$(head -n 1 $root/scores_$2.jsonl | $PY -c "import json,sys; print(json.loads(sys.stdin.read())['trained'] - 32)")
    echo "$(date +%H:%M) gate $2 acts (living): $($PY $FI/live_act_report.py $root/ai_trace.jsonl $first)" >> $root/status
  fi
  if [ -f $g/brain_ag/train.log ]; then
    echo "$(date +%H:%M) gate $1 AG: brain $(grep 'ACTIVE EVAL' $g/brain_ag/train.log | tail -1 | grep -o 'fresh B.*'); noedges $(grep 'ACTIVE EVAL' $g/noedges_ag/train.log | tail -1 | grep -o 'fresh B.*')" >> $root/status
  fi
  if [ "$3" -ge 11 ]; then
    neg=$(grep -o "(1) brain contribution [-+0-9.]*" $root/status | tail -5 | awk '{if ($4+0 <= 0) c++} END {print c+0}')
    if [ "$neg" -ge 5 ]; then echo "$(date +%H:%M) STOP at $1: brain contribution <= 0 at 5 living gates in a row after 40k words" >> $root/status; exit 2; fi
  fi
}
ACTS=${ACTS:-"--ai --motor-acts --act-tau 8 --act-habit 0.2 --act-gamma-inv 0.05 --miss-prior --ai-epistemic 0.0 --ai-pragmatic 1.0 --ai-sigreg 0.01 --eye-forced-open --act-brain-grad"}
EXTRA="$EXTRA $ACTS"
mkdir -p $root $ckpt
if [ ! -f $ckpt/last.pt ]; then          # the fork: the source's chunk FROM becomes this run's past
  src=/e/fly_stream_overflow/${SRC:?SRC}; sck=/e/ib_checkpoints/$SRC
  cp $sck/c$FROM.pt $ckpt/last.pt; cp $sck/c$FROM.pt $ckpt/c$FROM.pt; cp $sck/birth.pt $ckpt/birth.pt
  for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl latched_c$FROM.npy leak_c$FROM.txt; do cp $src/$f $root/ 2>/dev/null; done
  grep -E "born: chunk 1 done|chunk [0-9]+ done" $src/status | awk -v n=$FROM '{match($0, /chunk [0-9]+/); c=substr($0, RSTART+6, RLENGTH-6)+0; if (c <= n) print}' > $root/status
  echo "$(date +%H:%M) forked from $SRC c$FROM with acts: $ACTS" >> $root/status
fi
if [ ! -f $ckpt/last.pt ]; then
  $PY -u scripts/ib/train_fly_bptt_stream.py --from-scratch --additional-tokens 4096 --output $root --checkpoint-dir $ckpt \
      --score-log $root/scores_c1.jsonl $LOGS $EXTRA >> $root/train.log 2>&1 \
    || { echo "$(date +%H:%M) birth FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) born: chunk 1 done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1)" >> $root/status
  cp $ckpt/last.pt $ckpt/c1.pt; latch_mask c1; probes c1
fi
done_chunks=$(grep -c "chunk [0-9]* done" $root/status)
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
