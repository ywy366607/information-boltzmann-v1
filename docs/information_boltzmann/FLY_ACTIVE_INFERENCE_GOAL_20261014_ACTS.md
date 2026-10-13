# Acts on a brain that carries prediction (plan and predictions, written before the runs, 2026-10-14)

## Why now
The earlier act experiments (M1-M3, D1-D3, in FLY_ACTIVE_INFERENCE_GOAL_20261013.md) all ran on brains that did not
carry the prediction. Passive evolution had starved the mushroom body, and text never reached the Kenyon cells. The
acts collapsed: eyes always open, answers at tick 1. Commit timing could not depend on a state that held nothing about
the current word.

In the mainline now (living gates, m_olf):
- text reaches the Kenyon cells through smell, with sparse coding (0.9-1.2% per word, overlap 0.00);
- the living brain carries +0.29 +- 0.06 nats at c2;
- taste reaches dopamine with the right sign (sugar -> PAM up, bitter and salt -> PPL1 up).

The precondition the act runs lacked is met, or is being met in m_main.

## Plan (causal order step 3; one change at a time)
- **Readiness at m_main c5.** Run the probes the chain already takes: wait_oracle (does thinking pay, and does its
  value track difficulty), rarity_decode (is the current word's difficulty readable at tick 1 on the motor side), and
  gamma tone. Act only if the oracle's best tick > 1 for the hard quartile.
- **D4 fork from m_main.** Commits from motor cells: `--ai --act-brain-grad --miss-prior --act-habit 0.2
  --act-gamma-inv 0.05`. Eyes open; the same text, world and living gates. Control: the same fork without
  `--act-brain-grad`, so only the readout learns.
- **D5 (after D4).** Eyes as a sampled act with a background speaker (the M3 revised world).
- **Then the mother world.** Speaking changes the next input.

## Predictions for D4 (written before the run)
1. Diversity: per-word latency entropy >= 0.5 nats with the habit prior.
2. Adaptivity: expected latency grows with the tick-1 CE (rho > 0.1) and tracks the oracle's best tick. The earlier
   adult could not do this; this brain carries the word in the mushroom body.
3. The brain is still used: the living noedges gate on the fork stays >= the mainline's at the same age (the act
   objective must not replace the brain by a shortcut).
4. Language cost <= 0.05 nats against the mainline on the same words.

If 2 fails while the brain carries the prediction: check whether the oracle says thinking pays for this brain. If it
does not, the cause is aleatoric difficulty, not the act machinery.

## Speed (2026-10-14): measured, and the next step
Measured with cProfile and the torch profiler (FLY_TORCH_PROFILE, one window). Per window, CPU self time is 8.7 s and
CUDA kernel time 3.0 s: the run is launch-bound, at about 550 kernel launches per tick (forward, recompute and
backward). Per tick the forward costs about 14 ms of host time:

| part | host time per tick |
|---|---|
| the COBA step | 5.9 ms |
| checkpoint pack hooks | 2.2 ms |
| writer | 1.5 ms |
| carried credit | 1.7 ms |
| compartment memory, dopamine and the rest | 2-3 ms |

The backward with its recompute is 52% of the time.

Done:
- **Dopamine class sensitivity once per window**: was a sparse product plus backward on every tick. 18% faster, equal
  within run-to-run noise.
- **A held word's writer projections once per segment**: about 10% of GPU kernel time; no wall-clock change while
  launch-bound.

Cut:
- **Compiling the carried-credit steps** gave no gain: the compile guards cost what the launches did. Reverted.

Next: CUDA graphs. They are the only fix for launch-bound ticks; expected 2-3x. Two designs:
- **(a) A forward and a recompute+backward graph per segment length (1-16 ticks).** Simple semantics, but every graph
  needs static gradient buffers for all parameters (the edges alone are tens of MB). 16 pairs do not fit in 4 GB.
- **(b) Fixed 16-tick segments that span word boundaries**, with per-tick write masks and embeddings as tensors, and
  the once-per-word expectation as a masked update. One graph pair. But the expectation (softmax over 50k words) then
  runs on every tick unless it is moved out of the segment.

To do in a dedicated session with the GPU free:
1. Implement (b) behind a switch.
2. Verify gradients against eager on the full connectome (paired runs within run-to-run noise).
3. Then deploy between chunks.
