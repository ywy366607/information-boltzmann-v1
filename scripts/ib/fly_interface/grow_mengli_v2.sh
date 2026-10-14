#!/bin/bash
# v2 (2026-10-15, from chunk 3): the mother's voice on Johnston's organ (--mother-voice, JO-A/B at 1.5 x rheobase while
# her word is on the senses) and the child's turn ends only at a line break, a yield or 16 words (raising.json
# mother.child_end_of_turn), user.  Same lineage, same counters.
# Mengli's raising lineage (user, 2026-10-14): one individual, never reset, raised in the mother world chunk after
# chunk -- the brain, its acts, the conversation (mother_state.json), the mother's memory (data/harness/world.json) and
# the family all carry on.  A/B arms fork from this lineage instead of restarting from an old checkpoint.
# Start: d5b c11 (the mainline brain + acts trained on the text stream: look / think / speak muscles, eye floor), with
# the general compartment rule applied at the first resume (--compartment-dendritic, checked zero-shot healthy).
# Mother world: DeepSeek mouth, contingent, speaking act, epistemic credit discounted per tick (tau 32, the best
# behaved of the four earlier runs), beats of 8 ticks.  Eyes continuous (the live pass does not sample binary eyes),
# no noise stretches (the mother's words are what is worth looking at); word intervals easier than on the text
# (user: each window's mean interval log-uniform in [4, 8] ticks instead of [2, 16]; no reply pause).
# Every chunk: 1024 words; checkpoints c1, every 10th and the latest two (on D); every 5 chunks: the snapshot probe
# (Kenyon code, meaning, margins) on the text, and (2) surprise -> DA from the chunk's own words.
# Usage: bash grow_mengli.sh [chunks=200]   Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface; CHUNKS=${1:-200}
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
root=/e/fly_stream_overflow/mengli; ckpt=/d/ib_checkpoints/mengli
MOTHER="--mother-world deepseek --mother-speak-act --mother-credit epistemic --mother-credit-ticks 64 --mother-credit-tau 32 --mother-voice --mother-voice-level 1.5"
FIXED="--compartment-dendritic --no-eye-binary --noise-rate 0 --log-word-dopamine --stream-pace-mean 8 --stream-pace-min 4"
mkdir -p $root $ckpt
if [ ! -f $ckpt/last.pt ]; then
  cp /d/ib_checkpoints/d5b/c11.pt $ckpt/last.pt; cp /d/ib_checkpoints/d5b/birth.pt $ckpt/birth.pt
  for f in config.json progress.json metrics.jsonl lifelong_evaluation.jsonl ai_trace.jsonl; do cp /e/fly_stream_overflow/d5b/$f $root/ 2>/dev/null; done
  echo "$(date '+%m-%d %H:%M') Mengli's lineage begins from d5b c11 (+ the compartment rule); mother: DeepSeek, contingent, speaking act, epistemic credit tau 32" >> $root/status
fi
done_chunks=$(grep -cE "^[0-9-]+ [0-9:]+ chunk [0-9]+ done" $root/status)   # only the chain's own lines
for ((c=done_chunks+1; c<=CHUNKS; c++)); do
  $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 1024 --output $root --checkpoint-dir $ckpt \
      --score-log $root/scores_c$c.jsonl $MOTHER $FIXED >> $root/train.log 2>&1 \
    || { echo "$(date '+%m-%d %H:%M') chunk $c FAILED" >> $root/status; exit 1; }
  mon=$($PY -c "import json; d=json.load(open('data/harness/monitor.json', encoding='utf-8'))['monitor']; print(json.dumps({k: d.get(k) for k in ('fly_turns', 'fly_silent_turns', 'fly_mean_turn_words', 'fly_ttr', 'fly_family_words', 'mother_family_words', 'mother_calls', 'mouth_cache_hit_rate')}))" 2>/dev/null)
  echo "$(date '+%m-%d %H:%M') chunk $c done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1 | grep -o 'fresh B.*'); epistemic $(tail -n 1 $root/scores_c$c.jsonl | $PY -c "import json,sys; r=json.loads(sys.stdin.read()); e=r.get('epistemic') or {}; print({k: e.get(k) for k in ('speak_said_return', 'speak_silent_return', 'ig_mother_mean')})" 2>/dev/null); monitor $mon" >> $root/status
  cp $ckpt/last.pt $ckpt/c$c.pt
  p=$((c - 2)); [ $p -gt 1 ] && [ $((p % 10)) -ne 0 ] && rm -f $ckpt/c$p.pt
  if [ $((c % 5)) -eq 0 ]; then
    $PY -u $FI/snapshot_probe.py $ckpt $root/snapshot_c$c.json > $root/snapshot_c$c.txt 2>&1 \
      && echo "$(date '+%m-%d %H:%M') c$c snapshot: $(tail -n 1 $root/snapshot_c$c.txt)" >> $root/status
    echo "$(date '+%m-%d %H:%M') c$c (2) surprise~DA: $(cat $root/scores_c$((c-4)).jsonl $root/scores_c$((c-3)).jsonl $root/scores_c$((c-2)).jsonl $root/scores_c$((c-1)).jsonl $root/scores_c$c.jsonl > $root/scores_last5.jsonl; $PY $FI/live_da_report.py $root/scores_last5.jsonl)" >> $root/status
  fi
done
echo "$(date '+%m-%d %H:%M') Mengli's lineage: $CHUNKS chunks done" >> $root/status
