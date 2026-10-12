# M2 — output only (2026-10-13)

Adult with the connectome v2 fixes, pulse input, world pace mu ~ log-uniform [2, 16], eyes forced open, 3 chunks x
4,096 words. Every arriving word is a question ("what comes next?"): the output commits the read standing then
(value -CE), or the next word arrives first (-log V = -10.82). Arm `out`: pragmatic weight 1. The no-pragmatic
reference is M1's forced-open arm (identical otherwise). First launch stopped after 23 windows: the output crossed the
bound every tick and the sigmoid straight-through slope vanished there (absorbing habit); relaunched with the
straight-through slope taken at the bound everywhere. Probe: `scripts/ib/fly_interface/act_probe.py` (eyes open).

## Output
| condition | answered (out / no-pragmatic) | latency min/median/mean/max (out) | answer value (out / no-pragmatic) | commits per word (out) |
|---|---|---|---|---|
| fixed16 | 1.000 / 0.087 | 1 / 1 / 1 / 1 | -7.77 / -10.49 | 16.0 |
| geo16 | 1.000 / 0.182 | 1 / 1 / 1 / 1 | -7.80 / -10.27 | 16.1 |
| geo8 | 1.000 / 0.067 | 1 / 1 / 1 / 1 | -7.81 / -10.62 | 8.2 |
| geo4 | 1.000 / 0.020 | 1 / 1 / 1 / 1 | -7.80 / -10.76 | 4.0 |
| geo2 | 1.000 / 0.003 | 1 / 1 / 1 / 1 | -7.83 / -10.81 | 2.0 |

Output emerged from the pragmatic value alone and never blocks input (the world never waits; every word scored).
Without the pragmatic term the output almost never commits (acceptance 6b).

## Why every answer comes at tick 1: what waiting could buy at all (hindsight oracle, no-pragmatic arm)
- Mean CE by tick since onset (fixed16): 7.757, 7.565, 7.494, 7.478, 7.468, ... -> the best mean gain over tick 1 is
  0.31 nats, against a miss risk of (10.82 - CE)/mu per tick; at mu = 16 committing at tick 1/2/3 is worth about
  -7.91/-7.93/-8.00 (adult curve).
- The gain is larger for EASY words (0.67) than hard ones (0.27); the hindsight-best tick does not track difficulty:
  rho(best tick, CE at tick 1) = -0.10 .. +0.04 at every pace.
- So with pulse input no policy can make latency scale with difficulty: next-word difficulty is aleatoric and a pulse
  delivers its evidence at once (the read is settled by tick 2). Acceptance 3 is unreachable in this world, not a
  failure of the head.

## Language cost (paired per word, 95% bootstrap, out - no-pragmatic)
| | fixed16 | geo16 | geo8 | geo4 | geo2 |
|---|---|---|---|---|---|
| survival CE | +0.161 [+0.140, +0.181] | +0.101 [+0.076, +0.127] | +0.090 [+0.062, +0.120] | +0.047 [+0.025, +0.071] | -0.019 [-0.042, +0.003] |
| arrival CE | +0.118 | +0.088 | +0.091 | +0.046 | -0.026 |

Cause: the answer's value (-CE at the commit tick) also trained the predictions, so the brain traded later-tick
accuracy for quick answers (its CE-by-tick curve flattened: best gain 0.14 vs 0.31). In active inference preferences
select acts under the beliefs; they do not rewrite the beliefs. Fix (committed for M3): the answers' values are
evaluated under the beliefs, detached; gradients reach the brain only through the acts.

## Verdict
- Output is learned and never blocks input; no fixed 14 ticks (acceptance 3, second half): met.
- Latency scales with difficulty (acceptance 3, first half): not met; shown unreachable with pulse input (oracle).
- No pragmatic term -> almost no output (acceptance 6b): met.
- Language cost <= 0.05 (acceptance 5): not met at mu >= 4 (+0.05 .. +0.16); cause found and fixed.
- Exit: sustained noisy input (a word heard over its interval as noisy samples), where waiting has value for unclear
  words (the fly mushroom body accumulates evidence over sustained odour input; DasGupta et al. 2014).
