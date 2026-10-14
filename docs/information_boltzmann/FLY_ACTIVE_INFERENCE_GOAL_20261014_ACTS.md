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

## m_main: the mainline from birth with living gates (2026-10-14, night)

Run: `grow_mainline_live.sh m_main 40`, PROFILES['mainline'] (receptor_lang, input gain size^0.5, EG, evidence-gated
physiology, dopamine, compartment memory). Each gate: 1024 words, paired by window, learning on.

| gate | (1) brain contribution, mean +- se [min, median, max] | fresh B brain / noedges | (2) surprise~phasic DA (next) | DA cells | (L) latched, worth |
|---|---|---|---|---|---|
| c1 | +0.091 +- 0.036 [-0.41, +0.08, +0.58] | -- | +0.014 (+0.017) | -0.010 | 111, -0.019 +- 0.007 |
| c2 | +0.276 +- 0.064 [-0.14, +0.13, +1.22] | -- | +0.000 (-0.013) | -0.060 | 0 |
| c3 | +0.211 +- 0.045 [-0.33, +0.23, +0.79] | -- | -0.035 (-0.043) | -0.031 | 6 |
| c4 | +0.295 +- 0.047 [-0.07, +0.25, +0.87] | -- | -0.044 (-0.047) | -0.062 | 141, -0.009 +- 0.005 |
| c5 | **+0.485 +- 0.038 [+0.15, +0.45, +0.91]** | **6.887 / 7.128 (+0.24)** | +0.016 (+0.009); all four 1024-blocks > 0 | -0.050 | 412, **+0.040 +- 0.018** |

**Reproducible.** m_olf, an independent newborn, gave +0.087 (c1) and +0.290 (c2).

**(1) holds.** At c5 every window is positive, and the brain is worth 0.49 nats on the stream and 0.24 on fresh text.

**(L) latched cells became useful.** Their worth went from mildly harmful to +0.04 at c5, the same sign as the
champion's +0.10.

**(2) has no stable sign yet.** It is negative at c3-c4 and positive at c5; the dopamine cells' own firing is
consistently negative. The surprise path probe (c6) locates the break.

**Readiness at c5** (diagnostic probes, fixed weights):
- Thinking now pays: CE by tick goes 7.71 -> 7.53 (at m_olf c2 it was 7.80 -> 7.75).
- With the listener-prior miss, the best common answer tick is 7. It beats tick 1 by 0.10 nats, and the hardest
  quartile gains 0.52.
- The value of waiting is starting to be readable: output membranes at tick 4 R^2 0.10, muscles 0.05; at c2 both
  were ~0.
- Kenyon coding: 3.7% per word, consecutive-word overlap 0.02. MBONs fire at 28.5 Hz.

The precondition the earlier act runs lacked is now present. D4 forks from c5.

## (2) Where surprise stops on its way to dopamine (m_main c6, surprise_path_probe.py; diagnostic, fixed weights)

Spearman of each word's surprisal with each layer, [min | median | mean | max] over 4 places x 512 words:

| layer | rho |
|---|---|
| writer innovation | 0.83 [0.77, 0.84, 0.83, 0.87] |
| olfactory receptors | 0.52 |
| ALPN | 0.26 |
| Kenyon | 0.13 |
| MBON | 0.06 |
| DAN spikes | -0.03 |
| DAN v/theta | +0.02 |
| phasic DA | -0.04 |

**Cause.** Surprise is strong at the senses: the predictive writer's innovation is the prediction error. Along the
olfactory path the correlation roughly halves with each synaptic layer, and it is lost between the MBONs and the
dopamine cells. Their other inputs dominate. Nothing in the objective asks the dopamine cells to carry it either: the
benefit of surprise-gated plasticity shows only across windows, which BPTT does not see.

**Exit, the fly's own.** The novelty circuit: the alpha'3 compartment's MBON fires to novel odours and habituates
through dopamine-dependent plasticity (Hattori et al. 2017), fed by Kenyon cells, which already carry surprise (0.13).
Check that the circuit is present and wired in the model, and that the compartment memory can habituate it to
familiar words. Only after that consider letting the credit of dopamine-gated plasticity reach the DAN inputs.

## D4, first chunk (lost to a shutdown at 117/128 windows; act trace kept)

Living stream, 3,744 words:
- answered 0.72; latency 2.0 ticks; timing entropy 1.06;
- rho(latency, CE) -0.01 [-0.06, 0.02]; rho(latency, previous CE) +0.03;
- the commit hazard's spread grew from 0.12 to 0.23 over the chunk.

The acts are becoming state-dependent, but they are not yet adaptive to difficulty. Rerun from c5 for 3 chunks.

## D4 results (fork of m_main c5, 3 chunks, living stream)

| gate | brain contribution | fresh B at chunk end | answered | latency | timing entropy |
|---|---|---|---|---|---|
| c5 (acts in chunk 6) | +0.454 +- 0.035 (mainline at c5: +0.485) | 6.938 | 0.71 | 2.0 | 1.05 |
| c6 (chunk 7) | +0.148 +- 0.035 | 7.517 | 0.58 | 2.3 | 1.14 |
| c7 (chunk 8) | +0.346 +- 0.056 | 7.742 | 0.62 | 2.2 | 1.07 |

**The brain stays used with acts.** On the same words as the mainline's c5 gate, the language cost was 0.03. The c6
dip awaits the mainline's c6 gate, which uses the same words, for a paired comparison: text segment or the act
objective?

**(3) The definition's test** (chunk 8, with every tick's CE in the trace). The pooled correlations are artefacts of
the interval: a constant-hazard null gives larger ones (latency~wait gain +0.38, ~best tick +0.66 vs the policy's
+0.28 / +0.43). Within words of the same interval, per 1024-word block [min, median, mean, max]:
- latency ~ wait gain: [-0.04, +0.07, +0.03, +0.07];
- latency ~ best tick: [-0.03, +0.06, +0.05, +0.11];
- timing value over the constant-hazard null: [+0.005, +0.007, +0.011, +0.020] nats per word, positive in every block.

The acts read the state a little: commits come later where waiting pays, and the timing beats a state-blind policy
in every block. The signal is weak; "learned when to speak" is not reached yet.

**Why so weak, from the definitions.** Most of the next word's difficulty is aleatoric. The value of waiting is
readable only at R^2 0.10, and a word's ticks gain 0.18 nats on average. Pragmatic value in a listening stream is the
information an answer adds over the listener's prior, and speaking changes nothing in this world. The mother world is
where speaking has consequences (the next input depends on what was said), so the value of timing becomes large and
learnable.

Report fixed: `live_act_report.py` now gives the within-interval correlations (average ranks) and the timing value
over the null.

**D4 vs the mainline, same words** (both chunk 7, trained from c6):

| gate c6 | brain contribution | brain NLL | unigram |
|---|---|---|---|
| mainline | +0.161 +- 0.037 | 7.2172 | 7.3638 |
| D4 | +0.148 +- 0.035 | 7.2247 | 7.3638 |

The dip from c5 (+0.49) is the text segment: both arms dropped together, so the acts did not cause it.

The acts' language cost, paired by window over the whole chunk (4,096 words): **+0.016 +- 0.004 nats**. It is real
and within the 0.05 acceptance.

### Speed by mechanism (cProfile, 256 words = 8 windows = 2,771 ticks, host time, cumulative)

| mechanism | s | % | per tick |
|---|---|---|---|
| compartment memory, carried credit (comp_step) | 2.95 | 2.3 | 1.06 ms |
| dopamine: carried credit / occupancy / class sensitivity (now once per window) | 1.66 / 0.87 / 0.76 | 1.3 / 0.7 / 0.6 | 0.60 / 0.32 / 0.28 ms |
| writer incl. predictive (write_input) | 4.16 | 3.2 | 1.50 ms |
| predictive writer, expectation (once per segment) | 0.22 | 0.2 | 0.08 ms |
| EG update (once per window) | 0.76 | 0.6 | — |
| evidence gate (once per window) | 0.20 | 0.2 | — |
| COBA step incl. synapses | 16.3 | 12.5 | 5.88 ms |
| backward incl. recompute | 68.4 | 52.6 | 24.7 ms |

The added mechanisms together cost about 9%. The time is in the backward with its recompute and in about 550 launches
per tick, so CUDA graphs are the lever.

## (2) Kenyon -> dopamine-terminal release: A/B from m_main c6 (2026-10-14)

Both arms trained 2,048 words from m_main c6 on the same stream, learning on. The release arm adds the KC -> DAN
axo-axonal synapses (Cervantes-Sandoval 2017; 145,727 edges onto 348 dopamine cells) as local release, with a learned
gain per type that starts at 0.1.

| | base | release |
|---|---|---|
| surprise ~ phasic DA, same word | -0.0035 | +0.0052 |
| next word | +0.0002 | +0.0169 |
| DA cells | -0.0042 | -0.0210 |
| per 1,024 words [min, med, mean, max] | [0.021, 0.024, 0.024, 0.027] | [0.002, 0.021, 0.021, 0.039] |
| phasic DA sd | 1.08 | 2.50 |
| fresh B NLL | 7.5986 | 7.5994 |

Paired by window, release - base: **+0.0053 +- 0.0025 nats** (64 windows, [min -0.034, med +0.004, max +0.053]).
That is a small cost.

Taste: under release the PPL1 signs are no longer clean. Bitter gives PPL1 -0.26 Hz (base 0.0) and water gives
+0.26 Hz.

**Verdict: not adopted.** The release more than doubles the phasic variance, but none of the added variance follows
surprise.

**Cause.** The release follows Kenyon activity, and Kenyon activity codes which word is present, not how unexpected
it is. Its surprise correlation is only 0.13. Novelty in the fly is a difference: the α'3 compartment habituates to a
familiar odour, so its MBON/DAN output falls with familiarity (Hattori 2017). Adding more Kenyon drive adds the input
but never subtracts the expectation.

**Exit.** Look for the subtraction:
- MBON->DAN feedback sign and strength in α'3 (MBON16/17/28 <-> PPL104);
- whether the α'3 KC->MBON synapses depress with repetition in our plasticity (habituation).

If the subtraction is there but too weak, the fix is a mechanism, not a gain.

## D5: eyes as an act with a background speaker (launched 2026-10-14 12:15)

The user's priority is active reading, thinking and speaking in the listening stream; the mother world comes after.

**Setup.** Fork d5_eyes from D4 c8 with the same acts, plus:
- `--eye-binary --eye-init 0.9`: the eyes are a 0/1 sampled act;
- `--ai-depth 2`: a question stays open across closed eyes;
- `--noise-rate 0.25`: in a quarter of the windows a background speaker says 8 random words from a 16-word alphabet.

**Control.** d5_open: the same world with the eyes forced open. It runs afterwards on the same stream, because host
RAM (16 GB, about 6 GB per run) allows only two runs at once.

**Predictions (written before the run):**
1. Look: the eyes close more during the background speaker than during the main text
   (`eye_open_noise < eye_open_text` by at least 0.1 within 2 chunks), at a language cost <= 0.02 nats against
   d5_open on the same words.
2. Think: commits still land later than tick 1 where waiting pays (ce_at_commit - tick1 < 0, as in D4 c8: -0.08).
3. Speak: the timing value over the constant-hazard null stays >= D4.
4. The brain stays used: the living noedges gate stays positive.

**If 1 fails.** Check whether closing ever paid: the hindsight value of skipping the noise, computed from the CE of
the main text after a stretch with the eyes forced open vs forced closed.

## Mother world v2 code: speaking as the act (written, smoke pending)

- `fly_mother_world.live_window(speak_act=True)`: in the fly's turn its motor commits decide at each arrival. A commit
  on any decision tick says the sampled word; no commit is `MotherWorld.yield_turn()`.
- The live pass's segments are aligned with the training pass, so the packets are the same.
- `fly_active_inference.speaking_value`: a score-function credit, with return = -mean CE of the next 16 words and a
  running-mean baseline.
- Trainer flag: `--mother-speak-act`.

The default path is unchanged. The first smoke (local mother) died while host RAM was full. It will rerun after
D5, because active read/think/speak is the precondition for the mother world.

## (3) "Think": why the hard words do not get more thinking, and what to test instead (2026-10-14)

D4 prediction 2 said the expected latency would grow with the tick-1 CE. It did not: within intervals, Spearman
(latency, CE) is about -0.01 at c7-c8.

**Cause.** The hindsight oracle on D4 c8 (`wait_oracle_c8`, pace geo(mu=8), missed answers priced at the listener
prior) shows what waiting is worth, by difficulty quartile:

| difficulty quartile (tick-1 CE) | easiest | 2 | 3 | hardest |
|---|---|---|---|---|
| gain of the best tick over tick 1 (nats) | 0.415 | 0.346 | 0.164 | 0.158 |
| fraction of words where waiting pays | 0.74 | 0.80 | 0.61 | 0.65 |

Waiting pays most on the easy words. Hard words are mostly rare words (surprise ~ rarity, rho 0.97), and their
difficulty is aleatoric: no amount of settling makes an unpredictable word predictable. The reads of easy words keep
sharpening as the brain settles; CE by tick goes from 7.60 to 7.39 over 8 ticks. This is the M2 finding again, now on a
brain that carries the prediction.

**Exit.** The criterion for learned thinking is that latency tracks each word's own wait value, not its difficulty.
- `live_act_report.py` already reports `rho_latency_wait_gain` within intervals: +0.30 at D4 c8.
- It also reports `ce_at_commit_minus_tick1`: -0.08 nats at c8, so commits land where the read has improved.
- These are the measures for "think", alongside the timing value over the constant-hazard null.
- In the mother world, thinking gets a second value. The word said comes from the read at the commit tick, so a
  settled read says a better word.
