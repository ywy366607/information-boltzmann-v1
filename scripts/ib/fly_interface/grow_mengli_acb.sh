#!/bin/bash
# Mengli, one change per chunk (2026-10-15, docs/information_boltzmann/MENGLI_TRACES_AND_HOMEOSTASIS_DESIGN_20261015.md):
#   chunk 6  + pacemaker homeostasis (A)
#   chunk 7  + the speaking credit as a continuous O(1) trace in average-reward form (C; the plain discount biased it to silence)
#   chunk 8  + her answer as a dopamine pulse (B)
#   then plain chunks (every switch is inherited).
# Waits until DeepSeek answers first.  grow_mengli.py runs the probes after every chunk.  Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface; root=E:/fly_stream_overflow/mengli; ckpt=D:/ib_checkpoints/mengli
export PYTHONIOENCODING=utf-8 FLY_REUSE_TRANSMISSION=0
note() { echo "$(date '+%m-%d %H:%M') [chain] $1" >> $root/status; }
note "waiting for DeepSeek before chunk 6 (A: pacemaker homeostasis)"
until $PY -c "
import os, sys, requests, winreg
try:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as k:
        key = os.environ.get('DEEPSEEK_API_KEY') or winreg.QueryValueEx(k, 'DEEPSEEK_API_KEY')[0]
except Exception:
    key = os.environ.get('DEEPSEEK_API_KEY')
try:
    r = requests.post('https://api.deepseek.com/chat/completions', timeout=30,
                      json={'model': 'deepseek-flash', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 3},
                      headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    sys.exit(0 if r.status_code == 200 else 1)
except Exception:
    sys.exit(1)
"; do sleep 120; done
note "DeepSeek answers"
completed() { $PY -c "import json,sys; sys.exit(0 if json.load(open('$root/chunk_plan.json'))['status'] == 'completed' else 1)"; }
for extra in "--pacemaker-homeostasis" "--mother-credit-trace --mother-credit-differential" "--mother-reward" "" ""; do
  [ -f $root/STOP ] && { note "STOP"; exit 0; }
  note "next chunk, added: ${extra:-nothing (inherited)}"
  $PY -u $FI/grow_mengli.py --output $root --checkpoint-dir $ckpt --chunk-tokens 8192 --chunks 1 --extra="$extra" >> $root/chain.log 2>&1
  if ! completed; then note "chunk not completed (see status / train.log); chain stops"; exit 1; fi
  c=$($PY -c "import json; print(json.load(open('$root/chunk_plan.json'))['chunk'])")
  cp $ckpt/last.pt $ckpt/c$c.pt
  # keep the disk: the newest three chunk checkpoints and birth / c1
  ls $ckpt/c*.pt | grep -v '/c1.pt' | sort -V | head -n -3 | xargs -r rm -f
done
note "chain done"
