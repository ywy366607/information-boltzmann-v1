#!/bin/bash
# The newborn, v2 (2026-10-16): the same individual (born 10-15 21:45, birth paused at 152 words by a STOP).
#   1. birth completed to 1024 words in the mother world (resumed, never restarted)
#   2. chunks of 8192 words (grow_mengli.py: the mother world's switches per chunk, probes after each)
#   3. from c2, the living gate for chunk c-1 after chunk c trains:
#      (1) brain contribution on her normal replies = excess(noedges arm) - excess(brain), excess = NLL - unigram
#          (the arm lives its own conversation from c-1 for GATE_WORDS words, learning on)
#      (4) the brain on her normal replies against the fixed word frequency (whole chunk)
#      (L) latched cells (text_leak_probe); cell health per group
#      gate 2 (compartment dopamine, familiarity) and the acts come from grow_mengli's probes
#   Stop rule from c2: gate 2 dead (surprisal rho < 0.1, repeat trend > -0.1, B ratio < 1.2).
# Plan and audits: docs/information_boltzmann/FLY_EVENT_THREE_FACTOR_DESIGN_20261015.md.  Never edit while it runs.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; FI=scripts/ib/fly_interface; root=E:/fly_stream_overflow/newborn; ckpt=E:/ib_checkpoints/newborn
W=${GATE_WORDS:-1024}; TOTAL_CHUNKS=${TOTAL_CHUNKS:-9}
export PYTHONIOENCODING=utf-8 FLY_REUSE_TRANSMISSION=0 FLY_COMPILE_TICK=0
note() { echo "$(date '+%m-%d %H:%M') [chain] $1" >> $root/status; }
# the mother world's switches are never inherited: every run passes them (grow_mengli adds deepseek / speak act / voice)
WORLD="--mother-fallback siliconflow --mother-credit-trace --mother-credit-event-clock --mother-credit-brain-tau --mother-reward --sense-gains-learnable ${EXTRA:-}"
MOTHER="--mother-world deepseek --mother-speak-act --mother-credit epistemic --mother-voice --mother-voice-level 1.5 --compartment-dendritic --no-eye-binary --noise-rate 0 --log-word-dopamine"
trained() { $PY -c "import json; print(int(json.load(open('$root/progress.json'))['bptt_train_tokens']))"; }

if [ ! -f $root/born ]; then
  [ -f $ckpt/last.pt ] || { note "no newborn checkpoint: run grow_newborn.sh for a birth"; exit 1; }
  t=$(trained); left=$((1024 - t))
  note "birth continues: $t -> 1024 words (the same individual)"
  if [ $left -gt 0 ]; then
    # Birth continuation is memory-sensitive: avoid a one-time TorchDynamo guard compilation while preserving the
    # exact physical tick semantics. Explicit budgets also make the paused 152-word resume independent of inherited CLI.
    FLY_COMPILE_TICK=0 $PY -u scripts/ib/train_fly_bptt_stream.py --resume $ckpt/last.pt --extend-budget --additional-tokens $left --window 8 --eval-tokens 512 --replay-tokens 128 \
      --output $root --checkpoint-dir $ckpt --score-log $root/scores_c1.jsonl --skip-eval --checkpoint-every-tokens 1024 \
      $MOTHER $WORLD >> $root/train.log 2>&1
    rc=$?
  else
    rc=0
  fi
  $PY -c "import json,sys; p=json.load(open('$root/progress.json')); sys.exit(0 if p['status'] == 'completed' and int(p['bptt_train_tokens']) == 1024 else 1)" \
    && [ $rc -eq 0 ] || { note "birth not completed (exit $rc; see train.log)"; exit 1; }
  touch $root/born
  echo '{"chunk": 1, "status": "completed"}' > $root/chunk_plan.json
  echo "$(date '+%m-%d %H:%M ')chunk 1 done: birth, 1024 words in the mother world" >> $root/status
  cp $ckpt/last.pt $ckpt/c1.pt
  note "c1 (4) her replies vs fixed word frequency: $($PY $FI/live_mother_gate_report.py $root/scores_c1.jsonl)"
fi

gate() {   # $1 = c (the chunk just trained); gates c-1 with chunk c's first W words as the living brain arm
  local c=$1 p=$(($1 - 1)) g=$root/gate_c$(($1 - 1)) k=/e/ib_checkpoints/newborn_gate
  [ -f $ckpt/c$p.pt ] || { note "gate c$p: no c$p.pt"; return; }
  rm -rf $g $k; mkdir -p $g/noedges $k
  head -n $((W / 8)) $root/scores_c$c.jsonl > $g/brain_scores.jsonl          # window 8: W/8 rows
  cp $ckpt/c$p.pt $k/last.pt
  for f in config.json progress.json metrics.jsonl; do cp $root/$f $g/noedges/ 2>/dev/null; done
  $PY -u scripts/ib/train_fly_bptt_stream.py --resume $k/last.pt --extend-budget --additional-tokens $W --output $g/noedges \
    --checkpoint-dir $k --score-log $g/noedges/scores.jsonl --skip-eval --checkpoint-every-tokens 100000 --ablate noedges \
    $MOTHER $WORLD > $g/noedges/train.log 2>&1 || note "gate c$p noedges arm FAILED"
  rm -rf $k
  # the dashboard's gate line (HH:MM gate cN (living): (1) brain contribution ...)
  echo "$(date +%H:%M) gate c$p (living): $($PY $FI/live_mother_gate_report.py $g/brain_scores.jsonl $g/noedges/scores.jsonl noedges --line)" >> $root/status
  note "c$c (4) her replies vs fixed word frequency: $($PY $FI/live_mother_gate_report.py $root/scores_c$c.jsonl)"
  $PY -u $FI/text_leak_probe.py $ckpt > $root/leak_c$c.txt 2>&1
  note "c$c (L) $(grep -o 'latched (> 200 Hz without words, >= 2 of 4 places): [0-9]*' $root/leak_c$c.txt | tail -1)"
  $PY -u $FI/cell_health_probe.py $ckpt $root/cell_health_c$c.json > $root/cell_health_c$c.txt 2>&1
  note "c$c cell health: $(grep -i 'dead\|saturated' $root/cell_health_c$c.txt | head -3 | tr '\n' ' ' | cut -c1-400)"
}

completed() { $PY -c "import json,sys; sys.exit(0 if json.load(open('$root/chunk_plan.json'))['status'] == 'completed' else 1)"; }
while true; do
  $PY -c "import json,sys; p=json.load(open('$root/chunk_plan.json')); sys.exit(0 if p['status'] == 'completed' and int(p['chunk']) >= $TOTAL_CHUNKS else 1)" && break
  [ -f $root/STOP ] && { note "STOP"; exit 0; }
  $PY -u $FI/grow_mengli.py --output $root --checkpoint-dir $ckpt --chunk-tokens 8192 --chunks 1 --extra="$WORLD" >> $root/chain.log 2>&1
  if ! completed; then note "chunk not completed (see status / train.log); chain stops"; exit 1; fi
  c=$($PY -c "import json; print(json.load(open('$root/chunk_plan.json'))['chunk'])")
  cp $ckpt/last.pt $ckpt/c$c.pt
  [ "$c" -ge 2 ] && gate $c
  ls $ckpt/c*.pt | grep -v '/c1.pt' | sort -V | head -n -2 | xargs -r rm -f
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
    case "$verdict" in DEAD*) touch $root/STOP; note "gate 2 failed at c$c: STOP (next: the per-word / cAMP burst judgement)";; esac
  fi
done
note "chain done"
