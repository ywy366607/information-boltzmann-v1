#!/bin/bash
# One-off (2026-10-15): grow_mengli.sh was stopped during chunk 2 to switch to v2; when chunk 2's trainer exits, do the
# chain's bookkeeping for it (status line, c2.pt).  Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; root=/e/fly_stream_overflow/mengli; ckpt=/d/ib_checkpoints/mengli; c=2
while powershell -c "Get-Process -Id 14608 -ErrorAction SilentlyContinue" | grep -q python; do sleep 15; done
if grep -q "ACTIVE EVAL 47104]" $root/train.log; then
  mon=$($PY -c "import json; d=json.load(open('data/harness/monitor.json', encoding='utf-8'))['monitor']; print(json.dumps({k: d.get(k) for k in ('fly_turns', 'fly_silent_turns', 'fly_mean_turn_words', 'fly_ttr', 'fly_family_words', 'mother_family_words', 'mother_calls', 'mouth_cache_hit_rate')}))" 2>/dev/null)
  echo "$(date '+%m-%d %H:%M') chunk $c done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1 | grep -o 'fresh B.*'); epistemic $(tail -n 1 $root/scores_c$c.jsonl | $PY -c "import json,sys; r=json.loads(sys.stdin.read()); e=r.get('epistemic') or {}; print({k: e.get(k) for k in ('speak_said_return', 'speak_silent_return', 'ig_mother_mean')})" 2>/dev/null); monitor $mon" >> $root/status
  cp $ckpt/last.pt $ckpt/c$c.pt
  echo "c2 finished" > $root/c2_finished.flag
else
  echo "$(date '+%m-%d %H:%M') chunk $c FAILED (stopped chain, see train.log)" >> $root/status
  echo "c2 failed" > $root/c2_finished.flag
fi
