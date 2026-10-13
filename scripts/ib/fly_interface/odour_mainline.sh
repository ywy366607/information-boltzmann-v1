#!/bin/bash
# The odour world on a mainline newborn (user 2026-10-13: acts that need no language -- the fly's own olfactory
# learning).  Birth from the mainline profile (text through the non-chemical senses), 4096 words of text, the act head
# calibrated on this individual, then the odour world with taste-only outcomes (no dopamine electrode: valence reaches
# PAM / PPL1 through the fly's own taste lines) for CHUNKS chunks.  Acceptance (written 10-12): on revisits
# P(approach|sweet) - P(approach|bitter) >= 0.1 in the last chunk; avoidance of bitter above first encounters; acts not
# collapsed onto one choice; language cost < 0.05.
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; NAME=${1:-odour_mainline}; CHUNKS=${2:-3}
root=/e/fly_stream_overflow/$NAME; ckpt=/e/ib_checkpoints/$NAME
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
P="--output $root --checkpoint-dir $ckpt"
W="--odour-world --no-outcome-dopamine --act-init scratch/fly_stream/act_init_$NAME.pt"
mkdir -p $root $ckpt
if [ ! -f $ckpt/last.pt ]; then
  $PY -u scripts/ib/train_fly_bptt_stream.py --from-scratch --additional-tokens 4096 $P >> $root/train.log 2>&1 || { echo "birth FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) born (text 4096)" >> $root/status
  $PY -u scripts/ib/fly_interface/act_calibrate.py $ckpt > $root/act_calibrate.txt 2>&1 && cp scratch/fly_stream/act_init.pt scratch/fly_stream/act_init_$NAME.pt \
    || { echo "act calibration FAILED" >> $root/status; exit 1; }
  echo "$(date +%H:%M) act head calibrated: $(tail -3 $root/act_calibrate.txt | tr '\n' ' ')" >> $root/status
fi
done_chunks=$(grep -c "odour chunk [0-9]* done" $root/status)
for ((c=done_chunks+1; c<=CHUNKS; c++)); do
  $PY -u scripts/ib/train_fly_bptt_stream.py --extend-budget --additional-tokens 4096 $P $W >> $root/train.log 2>&1 || { echo "odour chunk $c FAILED" >> $root/status; exit 1; }
  $PY - $root/odour.jsonl >> $root/status <<'PYEOF'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
last = [r for r in rows if r['bptt_train_tokens'] == rows[-1]['bptt_train_tokens']]
def share(sel, act):
    s = [r for r in last if sel(r)]
    return (sum(r['act'] == act for r in s) / len(s), len(s)) if s else (float('nan'), 0)
acts = collections.Counter(r['act'] for r in last)
rev = lambda v: (lambda r: r['valence'] == v and r['prior_outcomes'] > 0)
first = lambda v: (lambda r: r['valence'] == v and r['prior_outcomes'] == 0)
a_s, n_s = share(rev('sweet'), 'approach'); a_b, n_b = share(rev('bitter'), 'approach')
v_bf, _ = share(first('bitter'), 'avoid'); v_br, _ = share(rev('bitter'), 'avoid')
print(f"odour chunk done: encounters {len(last)}, acts {dict(acts)}, revisit approach sweet-bitter {a_s - a_b:+.3f} "
      f"(n {n_s}/{n_b}), avoid bitter first {v_bf:.2f} revisit {v_br:.2f}, fitness {sum(r['fitness'] for r in last) / max(len(last), 1):+.3f}")
PYEOF
  sed -i "\$s/^odour chunk done/$(date +%H:%M) odour chunk $c done/" $root/status
done
echo "$(date +%H:%M) $NAME finished" >> $root/status
