# M1 — eyes only (2026-10-13)

Adult (`q8_fly_ctm_settle14_100k`) with the connectome v2 fixes, world pace mu ~ log-uniform [2, 16] per window,
3 chunks x 4,096 words per arm. Objective: survival-weighted CE + 0.01 SIGReg + epistemic value
(information gain 0.5 log(1 + e/(1+n))). Arms: `eyes` (epistemic 1), `eyes_noepi` (epistemic 0), `open`
(eyes forced open). Probe: `scripts/ib/fly_interface/eyes_probe.py` (same 1,024 validation words, same intervals per mu).

## Eyes (probe, mode own, eye gain at word onsets: min | median | mean | max)
| arm | mu=16 | mu=8 | mu=4 | mu=2 | open fraction |
|---|---|---|---|---|---|
| eyes | 0.998 / 0.9997 / 0.9996 / 0.9999 | 0.998 / 0.9997 / 0.9997 / 0.9999 | 0.999 / 0.9997 / 0.9997 / 0.9999 | 0.9996 / 0.9998 / 0.9998 / 0.9999 | 1.000 everywhere |
| eyes_noepi | 0.735 / 0.988 / 0.974 / 0.999 | 0.878 / 0.991 / 0.986 / 0.999 | 0.967 / 0.992 / 0.991 / 0.998 | 0.978 / 0.995 / 0.995 / 0.998 | 1.000 everywhere |

Training trace: eyes 0.95 -> 0.989 (chunk 1) -> 0.9995 (chunk 3); eyes_noepi 0.945 -> 0.985; no onset ever below 0.5.

## Closing vs difficulty and pace
- Spearman(eye, CE of the word about to be written) |rho| <= 0.065; hardest vs easiest quartile differ by <= 0.001.
  The untrained head on the forced-open arm gives |rho| up to 0.107 (same membranes), so this is baseline level.
- Pace: without the epistemic term the gain is lowest at the slowest pace (min 0.73 at mu = 16 vs 0.98 at mu = 2;
  training rho(eye, mu) = -0.30); with it, flat at ~1.

## Language cost (paired per word, 95% bootstrap)
- Same weights, own eyes - forced open, survival CE: -0.0005 .. -0.0008 (eyes), -0.0004 .. -0.0025 (eyes_noepi).
- Across arms, eyes - open arm: survival -0.0009 .. +0.0036, arrival -0.0011 .. +0.0040 (all intervals include 0);
  eyes_noepi - open arm: survival +0.0018 .. +0.0071.
- Training CE, chunk 3: eyes 6.920, eyes_noepi 6.921, open 6.917.

## Verdict against the acceptance criteria
- Acceptance 1 (no dark room): met. Eyes open at 100% of onsets at every pace, with or without the epistemic term.
- Acceptance 2 (closing to think): not met in M1, as predicted before the run. With no output act, missing a word can
  only remove context, so there is nothing to think for; the eyes have no reason to close.
- Acceptance 5 (language cost <= 0.05 nats): met (<= 0.004, within noise).
- Weak point: with the epistemic term the eye saturates (gain 0.9996); for M3 the eyes are a sampled 0/1 policy
  starting at p = 0.9.
