#!/bin/bash
# Short arms (4096 words from birth, R3 substrate): does the synapse rule decide R3's runaway and the brain's use?
#   adam   = R3 as run (per-parameter Adam, relative steps, class physiology learned)
#   eg     = normalized exponentiated gradient + KL to the birth connectome; class physiology frozen; phasic dopamine
#   frozen = no synapse learning, class physiology frozen, phasic dopamine (control: what does synapse learning add?)
# Each: latched cells (>200 Hz, words withheld) and the brain contribution (decoder alone vs brain, NLL).
cd /d/information-boltzmann-v1
PY=/d/conda_envs/vox/python.exe; out=/e/fly_stream_overflow/short_eg
export FLY_REUSE_TRANSMISSION=0 PYTHONIOENCODING=utf-8
R2=$(grep '^R2=' scripts/ib/fly_interface/grow_r3_rebirth.sh | sed 's/^R2="//; s/"$//' | sed 's/--output \$root --checkpoint-dir \$ckpt//')
mkdir -p $out
for arm in ${ARMS:-eg frozen adam}; do
  case $arm in
    adam)   extra="" ;;
    eg)     extra="--eg-plasticity --eg-tau 4096 --lr-physiology 0 --dopamine-adapted" ;;
    frozen) extra="--lr-synapse 0 --lr-physiology 0 --dopamine-adapted" ;;
  esac
  root=$out/$arm; ckpt=/e/ib_checkpoints/short_eg_$arm
  rm -rf $root $ckpt; mkdir -p $root $ckpt
  echo "$(date +%H:%M) $arm start" >> $out/status
  $PY -u scripts/ib/train_fly_bptt_stream.py --from-scratch --additional-tokens 4096 $R2 $extra --output $root --checkpoint-dir $ckpt > $root/train.log 2>&1 \
    || { echo "$(date +%H:%M) $arm FAILED" >> $out/status; continue; }
  $PY -u scripts/ib/fly_interface/text_leak_probe.py $ckpt > $root/leak.txt 2>&1
  for sw in none noedges; do
    NLL_PLACES=2 NLL_SCORE=512 SWAP=$([ $sw = none ] || echo $sw) BIRTH=$ckpt/birth.pt $PY -u scripts/ib/fly_interface/swap_nll_probe.py $ckpt 2>&1 | grep "^{" >> $root/brain.txt
  done
  n=$(grep -o "latched (> 200 Hz without words, >= 2 of 4 places): [0-9]*" $root/leak.txt | grep -o "[0-9]*$")
  c=$($PY -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; b,o=r[0]['nll_places_min_median_mean_max'][2],r[1]['nll_places_min_median_mean_max'][2]; print(f'brain {b:.3f} decoder-alone {o:.3f} contribution {o-b:+.3f} unigram {r[0][\"unigram\"]:.3f}')" $root/brain.txt)
  echo "$(date +%H:%M) $arm: latched ${n:-?}; $(grep 'brain mean' $root/leak.txt); $c; train $(grep 'ACTIVE EVAL' $root/train.log | tail -1)" >> $out/status
  [ $arm = eg ] || rm -f $ckpt/*.pt            # disk: keep only the EG individual
done
echo "$(date +%H:%M) all arms done" >> $out/status
