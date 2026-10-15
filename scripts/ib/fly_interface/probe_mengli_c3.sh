#!/bin/bash
# One-off (2026-10-15, user): pause Mengli's lineage after chunk 3 for the probe (voice reach, hop decay, corollary
# discharge, familiarity of 'Mengli', Kenyon / dopamine roles: mengli_probe.py), then resume grow_mengli_v2.sh.
# The GPU (4 GB) cannot hold the probe beside training.  Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface; root=/e/fly_stream_overflow/mengli; ckpt=/d/ib_checkpoints/mengli; c=3
export PYTHONIOENCODING=utf-8 FLY_REUSE_TRANSMISSION=0
while powershell -c "Get-Process -Id 26032 -ErrorAction SilentlyContinue" | grep -q python; do sleep 15; done
if grep -q "ACTIVE EVAL 48128]" $root/train.log; then
  mon=$($PY -c "import json; d=json.load(open('data/harness/monitor.json', encoding='utf-8'))['monitor']; print(json.dumps({k: d.get(k) for k in ('fly_turns', 'fly_silent_turns', 'fly_mean_turn_words', 'fly_ttr', 'fly_family_words', 'mother_family_words', 'mother_calls', 'mouth_cache_hit_rate')}))" 2>/dev/null)
  echo "$(date '+%m-%d %H:%M') chunk $c done: $(grep 'ACTIVE EVAL' $root/train.log | tail -1 | grep -o 'fresh B.*'); epistemic $(tail -n 1 $root/scores_c$c.jsonl | $PY -c "import json,sys; r=json.loads(sys.stdin.read()); e=r.get('epistemic') or {}; print({k: e.get(k) for k in ('speak_said_return', 'speak_silent_return', 'ig_mother_mean')})" 2>/dev/null); monitor $mon" >> $root/status
  cp $ckpt/last.pt $ckpt/c$c.pt
else
  echo "$(date '+%m-%d %H:%M') chunk $c FAILED (stopped chain, see train.log)" >> $root/status; exit 1
fi
timeout 2400 $PY -u $FI/mengli_probe.py $ckpt $root 600 E:/fly_stream_overflow/mengli/probe_c3.json > $root/probe_c3.txt 2>&1   && echo "$(date '+%m-%d %H:%M') c3 probe done (probe_c3.json)" >> $root/status   || echo "$(date '+%m-%d %H:%M') c3 probe FAILED (probe_c3.txt)" >> $root/status
PROBE_PARTS=F timeout 1200 $PY -u $FI/mengli_probe.py $ckpt/c1.pt $root 600 E:/fly_stream_overflow/mengli/probe_c1_F.json > $root/probe_c1_F.txt 2>&1   && echo "$(date '+%m-%d %H:%M') c1 familiarity probe done" >> $root/status   || echo "$(date '+%m-%d %H:%M') c1 familiarity probe FAILED (probe_c1_F.txt)" >> $root/status
echo "$(date '+%m-%d %H:%M') lineage resumes (grow_mengli_v2.sh)" >> $root/status
nohup bash $FI/grow_mengli_v2.sh 200 > /dev/null 2>&1 &
