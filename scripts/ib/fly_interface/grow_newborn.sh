#!/bin/bash
# The newborn (2026-10-16): born from scratch into the mother world, then chunks of 8192 words with probes after each.
# Plan and audit: docs/information_boltzmann/FLY_EVENT_THREE_FACTOR_DESIGN_20261015.md (third review, launch working
# point, gate 2) and MENGLI_AUDIT_20261015.md.  Mouth: DeepSeek, SiliconFlow GLM-5.3-Flash as the reserve.
# Never edit while it runs (bash reads it by offset).
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface; root=E:/fly_stream_overflow/newborn; ckpt=E:/ib_checkpoints/newborn
export PYTHONIOENCODING=utf-8 FLY_REUSE_TRANSMISSION=0
mkdir -p $root $ckpt
note() { echo "$(date '+%m-%d %H:%M') [chain] $1" >> $root/status; }
# the mother world's switches are never inherited: every chunk passes them (grow_mengli adds deepseek / speak act / voice)
WORLD="--mother-fallback siliconflow --mother-credit-trace --mother-credit-event-clock --mother-credit-brain-tau --mother-reward --sense-gains-learnable --stream-pace-mean 16 --stream-pace-min 12"
if [ ! -f $root/born ]; then
  [ -f $ckpt/last.pt ] && { note "a checkpoint without the born marker: an interrupted birth; remove $ckpt and $root to start over"; exit 1; }
  note "birth: from scratch into the mother world (x2 time scale, 16 ticks per word, window 8)"
  $PY -u scripts/ib/train_fly_bptt_stream.py --from-scratch --output $root --checkpoint-dir $ckpt --score-log $root/scores_c1.jsonl \
    --skip-eval --checkpoint-every-tokens 1024 --additional-tokens 1024 --window 8 \
    --ai --motor-acts --ai-pragmatic 1 --ai-depth 2 --ai-sigreg 0.01 --act-brain-grad --eye-floor 0.05 --no-eye-binary --noise-rate 0 --ai-epistemic 1 --epistemic-belief \
    --mother-world deepseek --mother-speak-act --mother-credit epistemic --mother-voice --mother-voice-level 1.5 $WORLD \
    --pacemaker-homeostasis --homeostat-skip-dans --dan-kappa 1.05 --intrinsic-mbon-kappa 1.0 \
    --comp-tonic-forget 65536 --comp-trace-tau 40 --comp-timing --comp-unit-traces --comp-burst-relative --comp-burst-k 2 --comp-rate-calibrate \
    --compartment-adapted --input-gain-kenyon-pooled --compartment-dendritic --log-word-dopamine --time-scale 2 >> $root/train.log 2>&1
  rc=$?
  # a checkpoint is saved after the first window, so its existence proves nothing: the exit code and the full budget
  $PY -c "import json,sys; p=json.load(open('$root/progress.json')); sys.exit(0 if p['status'] == 'completed' and int(p['bptt_train_tokens']) == 1024 else 1)"     && [ $rc -eq 0 ] || { note "birth failed (exit $rc; see train.log)"; exit 1; }
  touch $root/born
  echo '{"chunk": 1, "status": "completed"}' > $root/chunk_plan.json
  echo "$(date '+%m-%d %H:%M ')chunk 1 done: birth, 1024 words in the mother world" >> $root/status
  cp $ckpt/last.pt $ckpt/c1.pt
fi
completed() { $PY -c "import json,sys; sys.exit(0 if json.load(open('$root/chunk_plan.json'))['status'] == 'completed' else 1)"; }
TOTAL_CHUNKS=9      # birth + 8 chunks of 8192 words; a restart runs only what is left
while true; do
  $PY -c "import json,sys; p=json.load(open('$root/chunk_plan.json')); sys.exit(0 if p['status'] == 'completed' and int(p['chunk']) >= $TOTAL_CHUNKS else 1)" && break
  [ -f $root/STOP ] && { note "STOP"; exit 0; }
  $PY -u $FI/grow_mengli.py --output $root --checkpoint-dir $ckpt --chunk-tokens 8192 --chunks 1 --extra="$WORLD" >> $root/chain.log 2>&1
  if ! completed; then note "chunk not completed (see status / train.log); chain stops"; exit 1; fi
  c=$($PY -c "import json; print(json.load(open('$root/chunk_plan.json'))['chunk'])")
  cp $ckpt/last.pt $ckpt/c$c.pt
  ls $ckpt/c*.pt | grep -v '/c1.pt' | sort -V | head -n -2 | xargs -r rm -f
  # gate 2 as a developmental outcome (fourth review): from c2 on, if the ordinary dopamine still carries nothing
  # (surprisal rho < 0.1), shows no decline over repeats (trend > -0.1) and B is not above other words (< 1.2), the
  # hypothesis that life makes the teacher informative failed: stop and fix the threshold (per word, not per tick)
  if [ "$c" -ge 2 ]; then
    verdict=$($PY - "$root" "$c" <<'PYEOF'
import json, subprocess, sys
root, c = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(f'{root}/da_info_c{c}.json', encoding='utf-8'))
    rho = d['spearman drive~surprisal']; trend = d['repeat trend (spearman occurrence~relative drive)']
    out = subprocess.run([sys.executable, 'scripts/ib/fly_interface/live_da_report.py', f'{root}/scores_c{c}.jsonl'],
                         capture_output=True, text=True).stdout.strip().splitlines()[-1]
    b = json.loads(out).get('B_phasic_onset_over_other')
    dead = rho < 0.1 and (trend is None or trend > -0.1) and (b is None or b < 1.2)
    print(('DEAD' if dead else 'ALIVE') + f' rho {rho} trend {trend} B {b}')
except Exception as e:
    print(f'UNKNOWN {type(e).__name__}')
PYEOF
)
    note "c$c gate 2: $verdict"
    case "$verdict" in DEAD*) touch $root/STOP; note "gate 2 failed at c$c: STOP (next: threshold judged per word)";; esac
  fi
done
note "chain done"
