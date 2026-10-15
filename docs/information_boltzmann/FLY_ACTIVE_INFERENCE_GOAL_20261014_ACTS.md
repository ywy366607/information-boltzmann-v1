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

## D5 chunk 9: the eyes closed for good (dark room), and what reading is worth (2026-10-14)

**Result.** Over the first 60% of chunk 9 (D5, forked from D4 c8) the eyes stayed open:
- mean eye at onset 0.54-1.0;
- open on noise and on text alike (0.96-1.0).

At about 35,200 words they closed. They stayed at 0.000 for the rest of the chunk, on noise and text alike. "Look" did
not differentiate.

Living gate c8, on chunk 9's first 1,024 words, with the eyes still open:
- (1) brain - noedges +0.452 +- 0.040;
- (L) latched cells worth +0.073 +- 0.026;
- (2) surprise ~ phasic DA -0.019.

Acts at c9:
- answered 0.84;
- latency 1.6 ticks;
- `rho_latency_wait_gain` +0.29 pooled, but +0.014 within intervals;
- timing value over the null -0.0003. Weaker than D4.

**Cause 1: no prior on the eye policy.** The commits carry a habit prior (KL to 0.2), which keeps them sampled. The
eyes carry none. Once p(open) is near 0 the sampled look is never open, and the brain never again meets what looking
gives: p -> 0 is absorbing.

**Cause 2: looking is worth little to this brain.** Gain over the fixed unigram per window, from `scores_c9`:

| | windows | min | median | mean | max | se |
|---|---|---|---|---|---|---|
| eyes open (<= 35,200) | 76 | -1.31 | +0.325 | +0.176 | +1.10 | 0.062 |
| eyes closed (> 35,328) | 48 | -1.18 | +0.234 | +0.117 | +0.79 | 0.066 |

The blind brain still beats the fixed unigram, because it keeps learning the current text's word frequencies online.
What reading the current word adds is about 0.06 nats, and that is within the noise. Closing the eyes cost almost
nothing.

**What this means for gate (1).** The noedges arm is worse than the unigram (8.08 vs 7.82 at c8). So brain - noedges
measures "a brain" against "a broken brain", not the use of what is read. The gates now add a **blind arm**
(`--ablate blind`): the words never reach the senses, exactly as with closed eyes, while every synapse lives and
learns. brain - blind is what reading is worth. Chain `grow_acts_v2.sh`.

**Exit.**
1. `--eye-floor 0.05`: the sampled open probability stays in [0.05, 0.95]. A sensory gate is never shut for good, and
   a closed habit still meets the value of looking. d5b runs this from D4 c8 with blind arms in the gates.
2. The blind gate on the mainline: m_main c7, on chunk 8's words.
3. If reading stays worth about 0.06, the eyes cannot learn to look before the brain makes more of what it reads.
   That is the brain's job (1), not the eyes'.

## (2) Habituation of the Kenyon release: negative, and the precondition it exposes (2026-10-14)

**The arm.** From m_main c6 on the same 2,048 words as the release A/B, with `--kc-dan-release --kc-habituation-tau
4096 --kc-habituation-use 0.2`.

| | base | release | habituation |
|---|---|---|---|
| surprise ~ phasic DA, same word | -0.0035 | +0.0052 | -0.0054 |
| rarity (unigram CE) ~ DA | +0.0015 | +0.0264 | +0.0151 |
| residual surprise ~ DA | -0.018 | -0.049 | -0.065 |
| phasic DA sd | 1.08 | 2.50 | 1.06 |
| fresh B NLL | 7.5986 | 7.5994 | 7.5891 |

Paired by window:
- habituation - base: -0.0022 +- 0.0028;
- habituation - release: -0.0074 +- 0.0028.

The habituation arm costs no language, but surprise still does not reach dopamine.

**Cause (from the numbers).** The phasic variance fell back to the base level: the release was depressed away
everywhere, not selectively.
- A Kenyon cell is active for 3.7% of words, so it fires about every 27 words.
- Over the recovery time (4,096 ticks, about 800 words) it is used about 30 times.
- Its availability settles near 0.1 for every cell, familiar or not.

Per-cell habituation becomes per-word familiarity only if **a word's Kenyon code repeats when the word repeats**.
Consecutive different words overlap at chance (0.02 vs the level 0.037), but the same word's codes across its
occurrences were never measured. If they do not repeat, the code is context-driven, and familiarity cannot be read
from it at all.

**Exit.**
1. Measure it first: for words occurring several times, the Kenyon overlap between occurrences of the same word, against
   different words at the same distance. Only if the same-word overlap is well above chance is habituation the right
   mechanism, with a recovery scaled to the cell's firing rate.
2. If it is near chance, the mushroom body does not hold word identity stably. That is a (1) problem as well: memory in
   KC -> MBON synapses needs a repeatable code.

## What reading is worth, and where word identity is lost (2026-10-14, living blind arms)

**Reading value** = blind arm - brain arm, paired by window. In the blind arm the words never reach the senses; every
synapse lives and learns.

| individual | words | reading value | [min, med, mean, max] | brain | blind | unigram |
|---|---|---|---|---|---|---|
| m_main c7 (chunk 8's words) | 1,024 | **+0.227 +- 0.028** | [-0.06, +0.22, +0.23, +0.63] | 7.178 | 7.405 | 7.524 |
| champion (its own regime) | 512 | **+0.857 +- 0.066** | [+0.36, +0.94, +0.86, +1.28] | 6.782 | 7.640 | 7.719 |

- The mainline brain does use what it reads, about 0.23 nats per word. The D5 estimate of about 0.06 compared
  different stretches of text, unpaired, and is withdrawn. D5's eyes closed mainly because the eye policy had no prior.
- The blind mainline brain still beats the fixed unigram by 0.12: it learns the current text's word frequencies
  online.
- The champion reads about 3.8 times more (0.86). Its own training grew the text route, even on the old birth scale.
  Its blind arm is only 0.08 better than the unigram.

**Where word identity is lost: Kenyon cells** (`kenyon_repeat_probe.py`, m_main c7, 1,468 words, 780 same-word
pairs). Overlap |a & b| / |a| of the active sets per word:

| layer | coding level | same word (mean) | different words (mean) | same - different |
|---|---|---|---|---|
| olfactory receptors | 0.21 | 0.67 | 0.25 | +0.42 |
| ALPN | 0.09 | 0.48 | 0.25 | +0.23 |
| Kenyon | 0.03 | 0.15 | 0.10 | +0.05 |
| MBON | 0.14 | 0.54 | 0.53 | +0.01 |

The projection neurons carry the word, but the Kenyon cells barely do. Different words already overlap at 3x chance,
so a common pool of Kenyon cells fires whatever the word is.

This explains:
- why habituation could not become familiarity: it depressed a random pool;
- why the mushroom body adds little memory of words: KC -> MBON learning needs a repeatable code.

**Next.** `kenyon_input_probe.py`: what drives the Kenyon cells while reading, split by presynaptic class, and whether
the ALPN -> Kenyon drive itself repeats for the same word. The fly's own answer is a few strong claw inputs from
projection neurons plus APL normalization. If our Kenyon cells are driven mostly by something else (background,
other classes), that is the mechanism to fix.

**Kenyon input probe** (m_main c7, 368 words). Synaptic drive into the Kenyon cells by presynaptic class, as a signed
share of the absolute total:
- All ticks: an unannotated-class source -0.976 (inhibition; most likely APL, whose class is missing in the
  annotations), ALPN +0.023, MBON -0.001.
- Ticks when a Kenyon cell fires: unannotated -0.962, ALPN +0.036.
- The ALPN -> Kenyon drive itself repeats for the same word: correlation 0.44 (same word) vs 0.21 (different words).

The word's projection-neuron input does reach the Kenyon cells, and it repeats. But it is only a few percent of their
synaptic drive, under a wall of inhibition, so it does not decide which Kenyon cells fire. Their spikes then come from
whatever else drives them: intrinsic or homeostatic current, and fluctuations. That is why the code does not repeat.

**Next probe:** the Kenyon current budget per tick (PN excitation, APL inhibition, intrinsic / homeostatic current,
threshold), the PN -> Kenyon weights now against birth, and the APL output scale.

In the fly a Kenyon cell needs several coincident claw inputs from projection neurons, and APL normalizes the
population. Our PN input is far below that.

**Theory for the Kenyon fix: Vafaii, Galor & Yates, "Brain-like variational inference" (arXiv:2410.19315v2, iP-VAE).**
The model:
- membrane u = log rate, r = exp(u), Poisson spike counts z;
- free energy F = ||x - Phi z||^2 / 2 + beta KL(rates), where the KL term is the coding rate;
- natural-gradient dynamics give u <- u + Phi^T x - Phi^T Phi z: feedforward drive minus "explaining away", plus a
  homeostatic leak.

Inhibition there removes what co-active cells already explain: it is lateral competition proportional to their
overlap, which acts as divisive normalization. The result is a sparse code that repeats with the input.

Our Kenyon cells sit under a wall of inhibition (about 97% of their synaptic drive) with the PN drive at 2-4%.

**Hypothesis to test first** (the Kenyon current budget probe): heavy APL inhibition plus Kenyon homeostasis. The
homeostasis raises intrinsic current to hold the firing rate, so the cells fire from that current and from
fluctuations, independent of the input. Two mechanisms cancelling each other, as in
feedback-check-mechanism-interactions.

**If it holds**, the fly's own fix is APL as feedback normalization scaled by Kenyon activity (Papadopoulou et al.
2011), not a constant subtraction.

The same paper places the prediction error x - Phi z inside the recurrent dynamics, with no dedicated error cells.
That is the form surprise should take for (2).

**Limits of the paper.** It is a single-layer image model trained by BPTT through its inference steps, not a
connectome, and it has no action. We take the principle, not its neuron model.

**Correction (2026-10-14 15:00).** The Kenyon input probe counted every presynaptic cell's output as 0/1 (ring > 0).
APL is graded: it transmits a small graded output every tick (about 0.014 per tick, from the coding probe). The probe
counted that as a full spike every tick, so "~97% inhibition" overstates APL by a large factor. **Withdrawn.** The probe
now uses the transmitted output itself, and it will be rerun.

The homeostasis hypothesis also fails on its premise: m_main has homeostasis off, and its intrinsic drive covers only
clock and modulatory types (pattern LNv / DN / LPN / PAM / PPL / OA / 5-HT), not the Kenyon cells.

**What stands:** a word's Kenyon code repeats only weakly (0.15 same word vs 0.10 different words), and its ALPN ->
Kenyon drive repeats (0.44 vs 0.21). The cause is open until the corrected budget is in.

**Corrected Kenyon input budget** (graded output counted as transmitted, m_main c7, 368 words). Signed share of the
absolute synaptic drive into Kenyon cells:

| ticks | unannotated (inhibition, likely APL) | ALPN |
|---|---|---|
| all ticks | -0.567 | +0.413 |
| when a Kenyon cell fires | -0.550 | +0.420 |

The projection-neuron input is a large share, roughly balanced against inhibition. It is not drowned.

Word identity thins at every stage:

| stage | same word | different words |
|---|---|---|
| receptor active sets | 0.67 | 0.25 |
| ALPN active sets | 0.48 | 0.25 |
| ALPN -> Kenyon drive (correlation) | 0.44 | 0.21 |
| Kenyon active sets (top ~3%) | 0.15 | 0.10 |

A sparse threshold keeps only the strongest few percent of cells. That turns a moderate input correlation (0.44) into
a small overlap of the active sets (0.15).

**Likely cause: temporal mixing.** In a continuous word stream each word's response carries the activity of the words
before it. In the fly, odours are usually separated by clean air.

**Exits.**
1. Measure the code's repeatability against the interval since the previous word. If longer gaps give more
   repeatable codes, the cause is mixing.
2. If it is mixing, use the iP-VAE principle: inhibition removes what is already explained, here the predictable
   carry-over of the previous words. That is the innovation, input minus the brain's own expectation (the predictive
   writer), and it is also the form of surprise that (2) needs.

**Temporal mixing, tested** (`kenyon_repeat_probe.py` with a gap of quiet ticks after every word, m_main c7, 968 words,
461 same-word pairs). Mean overlap of the active sets:

| layer | no gap: level / same / different | 32-tick gap: level / same / different |
|---|---|---|
| olfactory receptors | 0.21 / 0.66 / 0.25 | 0.10 / **0.92 / 0.15** |
| ALPN | 0.094 / 0.47 / 0.26 | 0.017 / **0.75 / 0.07** |
| Kenyon | 0.031 / 0.15 / 0.10 | **0.000 (silent)** |
| MBON | 0.15 / 0.54 / 0.52 | 0.058 / 0.52 / 0.50 |

**1. Mixing confirmed.** Separated words have repeatable, word-specific receptor and PN codes (ALPN 0.75 vs 0.07). In
the continuous stream each word's response carries the words before it.

**2. A single word drives no Kenyon cell.** With the gap the Kenyon cells are silent. In the stream they fire only from
the summed activity of several consecutive words, so their code is a mixture of the recent words. It cannot repeat
with a word, and familiarity or surprise cannot be read from it.

An isolated word activates about 1.7% of the ALPN (9.4% in the stream). In the fly an odour activates a sizeable share
of the projection neurons, and a Kenyon cell fires on several coincident claw inputs. Our word input is too sparse and
too weak to reach threshold alone.

**Exit.** Make one word's PN pattern drive a small, repeatable set of Kenyon cells on its own. The candidate stages:
- the word -> receptor write strength;
- the PN -> Kenyon synaptic efficacy;
- the Kenyon threshold and the APL inhibition.

**Next probe:** for isolated words, how far below threshold the Kenyon membranes stay, and which stage loses the most.
Then change one stage.

**Reframing (user, 2026-10-14).** Integration over several words is not a defect in itself. A Kenyon cell summing the
last few words and firing when the sum warrants is a **context code**, and a context code is what next-word
prediction needs. Same-word overlap was the wrong test for it.

The right test is whether the same context (the same last two or three words) gives the same Kenyon code.
`kenyon_repeat_probe.py` now reports same-bigram and same-trigram overlaps against controls at the same distances.

- **If the context code repeats:** the Kenyon cells are fine. (2) failed for a different reason: within 2,048 words
  the same contexts seldom recur, so habituation never had repeats to work on. Habituation over a context code gives
  sequence novelty ("this sequence is new"), which is closer to surprise than word rarity is.
- **If it does not repeat either:** the code follows slow state drift, and the Kenyon stage needs the fix.

## Three directions from the user (2026-10-14, evening)

**1. A JEPA objective for the brain.** The brain is trained only on next-token CE. CE forces it to separate every
surface difference, GPT-2 word fragments included. The mainline config says `lambda_jepa: 0.1`, but the paced stream
branch switches that term off (`last_jepa_loss = None`), so it never acted.

A JEPA term would instead predict the next input's **representation**:
- the brain's own PN / Kenyon pattern for the next word, with SIGReg against collapse. Correction (checked against
  the abstracts after the user asked): LeJEPA (arXiv 2511.08544) uses **no stop-gradient and no teacher-student / EMA**;
  SIGReg alone prevents collapse and gradients flow into both sides. LeVLJEPA (arXiv 2607.00784) **does** stop the
  gradient on its targets, but has no momentum encoder (no EMA) and no teacher. Our existing latent predictor
  (`fly_bptt_learning.py`, `z_target = graph_observer.encoders(S_q_next).detach()`) stops the gradient, with no EMA,
  which is the LeVLJEPA choice rather than LeJEPA's. Default for the new term: LeJEPA (no stop-gradient, SIGReg);
  stop-gradient is the one-change comparison arm;
- meanings that differ only in wording get close representations;
- the distance between predicted and actual representation is a semantic surprise, which is what (2) lacks: not
  "this word is rare" but "this meaning was unexpected".

VL-JEPA (ICLR 2026) predicts text embeddings instead of tokens and reports better results with half the trainable
parameters. CE stays, because Mengli must still choose words to speak. The JEPA term is added to it.

**First test:** whether contexts with similar meanings already give similar Kenyon codes, as a fly's similar odours
give overlapping codes. Then derive the term, write predictions, and run a living A/B.

**2. Mengli is always at the beginning.** All four mother-world runs started from the same D4 c8 for 1,024 words.
That was right for comparing credits, but she never accumulated interaction. Learning to talk needs one **continuous
raising lineage**, never reset, chunk after chunk: the conversation, the mother's memory and the family all persist,
and A/B arms fork from the lineage instead of restarting.

Missing so far: the mouth's history restarts empty in every process. Persist the transcript tail and the cache anchor
in the run directory and restore them on resume.

**3. LeVLJEPA** (arXiv 2607.00784): vision-language pretraining with no negatives, using cross-modal prediction with
stop-gradient targets (no momentum encoder or EMA) and SIGReg per modality. When Mengli has several senses (text through smell, taste, later
vision), predicting one sense's representation from another follows the same pattern. Later.

## Hypothesis (user): surprise should live on the integrated representation, not on the token

**The two halves are one mechanism.**
- **Integration builds the representation.** Kenyon membranes accumulate drive over several words, so the Kenyon
  code is a context code. The c7 data so far agrees:
  - at gap 0, same-word Kenyon overlap is 0.099 vs 0.075 for different words, so word identity is weak;
  - at gap 32, an isolated word drives no Kenyon cell.
- **Representation prediction gives the error.** If the Kenyon code is the brain's working representation, surprise
  is the gap between the predicted and the actual Kenyon pattern. That is a JEPA error, and semantic rather than
  token-level.
- **The current DA signal is token-level.** It follows token CE, and that is why it reduces to rarity (rho 0.97).

**Fly substrate.** DANs receive Kenyon terminal release (axo-axonal, already implemented) and MBON feedback. The
MBONs read the same Kenyon code through learned weights. The DAN signal is therefore "Kenyon drive minus what the
MBON loop expected": representation-level surprise from the fly's own circuit. Habituation per cell failed because
single cells do not repeat per word. If the code repeats per context, familiarity should be counted per context
pattern.

**Predictions**, written before the probes:
- P1 (queued context probe): Kenyon overlap for the same last 2 / 3 words is clearly above the same-last-word 0.10
  and above its shuffled control.
  - If not, the Kenyon code is not yet a reliable representation. The fix is a training target that makes it one:
    the JEPA term (LeJEPA: no stop-gradient, SIGReg), before any surprise read-out.
- P2: a Kenyon-pattern prediction error (predicted from the state one word earlier) correlates with token rarity
  well below 0.97, and still predicts the brain's own CE residual.
- P3: contexts that are neighbours in meaning (GPT-2 embedding of the last words) have higher Kenyon overlap than
  random context pairs.

**Order.**
1. P1.
2. If P1 holds: P2 and P3 on c7.
3. Then one change: DA driven by the Kenyon-level prediction error instead of the token error. Judge it by the living
   (2) gate: phasic DA tracks surprise with no cost to (1).
4. If P1 fails: add the JEPA term first, then repeat P1.

**Corollary (user): a representation-level surprise favours coherence and little silence.**
- Meaning is built by integrating consecutive words. One word followed by a long silence builds nothing: at gap 32 no
  Kenyon cell fires.
- So once surprise and credit live on the integrated code, silence has a natural price: the context decays, and the
  next words become harder to predict.
- This is an anti-dark-room pressure. A token-level objective can be lowered by hearing nothing; a semantic one
  cannot, because the representation it must predict is lost.
- It also gives "think" (waiting) a natural upper bound: the integration time.

**Measured next** (`context_gap_after_probes.sh`, right after the queued probes): the context repeat probe with quiet
gaps of 8 ticks (one beat, the mother's pause before every reply) and 16 ticks, to compare with gaps 0 and 32. If one
beat already erases the context code, the mother world breaks the child's context at every turn boundary. Then:
- shorten the pause, or let speech overlap as in real conversation (the brain never waits for input);
- the child's own speech heard back (self-hearing) also bridges the gap.

## P1 result (m_main c7, 2,968 words, gap 0): the Kenyon code is a context code over about 3 words

Overlap of the active sets between successive occurrences of the same context, against controls at the same
distance whose contexts differ. Medians are shown; the full [min|median|mean|max] is in the m_main status.

| layer | same last 1 | same last 2 | same last 3 | different (1/2/3) | ratio 1 → 2 → 3 |
|---|---|---|---|---|---|
| receptors | 0.669 | 0.742 | 0.823 | 0.221 / 0.232 / 0.225 | 3.0 → 3.2 → 3.7 |
| ALPN | 0.462 | 0.665 | 0.737 | 0.236 / 0.234 / 0.229 | 2.0 → 2.8 → 3.2 |
| **Kenyon** | **0.103** | **0.328** | **0.470** | 0.078 / 0.082 / 0.080 | **1.3 → 4.0 → 5.9** |
| MBON | 0.533 | 0.556 | 0.600 | 0.500 / 0.526 / 0.533 | 1.07 → 1.06 → 1.13 |

**Kenyon (P1 holds).** The Kenyon layer depends on context more than any other layer. Same word alone: 1.3x chance;
same last 3 words: 5.9x.

**Margin probe agrees** (isolated words, gap 32, 192 words). One word lifts the Kenyon top 1% only to h/θ 0.33
(median; [0.19 | 0.33 | 0.32 | 0.54]). The top 1% of receptors reaches threshold, and of ALPN 0.98. So a Kenyon
cell needs about three words of accumulated drive, which is the 3-word context. This is the user's
potential-accumulation account, measured.

**New defect: the MBONs do not read the context code.**
- Their overlap is the same for the same and for different contexts (0.53–0.60 vs 0.50–0.53), and 14% of them fire
  for every word.
- The fly's own expectation loop (Kenyon → MBON → DAN) therefore cannot yet say "this context was expected".
- Representation-level surprise through the fly's circuit needs context-specific MBONs. Making them so is the job of
  DA-gated Kenyon → MBON plasticity, and is a precondition to check before switching the DA signal.

**Next.** P2 / P3 are queued: `kenyon_semantic_probe.py` via `semantic_after_gap.sh`, after the gap sweep.
- P2: Kenyon novelty and ridge prediction error vs rarity, vs the brain's CE and its residual.
- P3: same meaning with no shared token (PPMI-SVD contexts from the training text; the local GPT-2 cache has only
  the tokenizer) vs random.

**d5b at the same time:**
- gate c8: (1) +0.436 ± 0.037; reading +0.103 ± 0.019; (2) still rarity (0.02, residual −0.001); latch +0.064 ± 0.030.
- c9 acts: latency follows wait gain (rho 0.23) only between word lengths, not within them (−0.01).

## Gap sweep (m_main c7): one beat of silence between words leaves the Kenyon layer almost silent

Quiet ticks are inserted after **every** word; the active set is read over each word's own 8 ticks.

| gap (ticks) | receptor level | ALPN level | Kenyon level | Kenyon same last 3 words (mean) | different (mean) |
|---|---|---|---|---|---|
| 0 | 0.207 | 0.093 | **0.030** | 0.458 | 0.102 |
| 8 (one beat) | 0.139 | 0.030 | **0.001** | 0.230 | 0.000 |
| 16 | 0.112 | 0.021 | **0.000** | 0.750 (of the few that fire) | 0.000 |
| 32 | 0.097 | 0.017 | 0 | — | — |

**Reading:**
- With one beat of silence between words, Kenyon activity drops 30-fold. The drive of earlier words decays within
  8 ticks, so the Kenyon integration window is shorter than one beat. The context code exists only in continuous
  speech, as the user predicted.
- The few cells that still fire do so for repeated contexts: sparse but reliable.
- MBON overlap is again flat (0.50 vs 0.50) at every gap.

**Consequence for the mother world.** Within a turn, words are continuous and Kenyon integration works. The pause
before every reply (1 beat), and the waits, erase the integrated context at every turn boundary. Across a turn
boundary, context must be held by memory, not by sensory integration: the latch (L) and slow state are what should
bridge it, as human working memory bridges conversational gaps of about one word's length.

**Functional test, before changing anything.** CE on the first words after a turn boundary vs on the words within a
turn in the mother transcripts, and `world_eval` gap cost on c7. The brain may bridge gaps by memory even though the
Kenyon layer goes silent. If it does not, the options, one at a time:
1. a shorter pause (half a beat) in the mother world;
2. the latch carrying the context.

## P2 result (m_main c7, 2,968 words): Kenyon novelty is not rarity, and not yet the brain's surprise either

| measure | vs rarity | vs brain CE | vs CE residual after rarity |
|---|---|---|---|
| brain CE (token surprise) | **0.983** | — | — |
| Kenyon novelty (1 − best cosine to the last 256 words) | −0.119 | −0.102 | 0.031 |
| Kenyon prediction error (ridge from Kenyon + ALPN at t−1, test half) | −0.019 | −0.018 | −0.033 |

**Distributions and controls:**
- Novelty: [0.28 | 0.69 | 0.68 | 0.90].
- Prediction error: [0.29 | 0.69 | 0.69 | 1.08], against 0.698 for predicting the mean pattern. A linear read of
  the current state barely predicts the next Kenyon pattern.
- Novelty and prediction error agree with each other (rho 0.54).
- Kenyon is silent on 0.07% of words.

**Reading:**
- The Kenyon-level signal is a different quantity from token surprise. Token surprise is rarity (0.98); Kenyon
  novelty is not (−0.12). So the representation level does escape rarity.
- But it is unrelated to what the brain finds hard to predict. That fits the MBON finding: the brain's prediction
  does not run through the Kenyon code (MBONs do not read it), so Kenyon novelty and the CE are unrelated.
- The next Kenyon pattern is linearly unpredictable from the current state. A JEPA-style term is exactly what would
  make it predictable: train the brain so that its state predicts its own next Kenyon / PN representation.

**P3, first pass (PPMI co-occurrence) is withdrawn.** GPT-2 small is now downloaded (`F:/ml_cache/huggingface`, 526 MB).
Co-occurrence is the weaker measure, so P3 is redone with GPT-2 only (`kenyon_gpt2_probe.py`, layer 6, last 3 tokens
and 512-token context; partial correlation beyond receptor + ALPN overlap). The PPMI numbers stay in the status file
for the record but are not used.

## P3 result (GPT-2 small, layer 6, m_main c7): the Kenyon code carries no meaning beyond what the senses already share

**Setup.** 600 anchor words, paired with:
- the nearest context that shares **no** token among the last 3 words (GPT-2 cosine, median 0.46);
- a random context (median −0.02).

Overlap ratio (neighbour / random, means):

| GPT-2 context | receptors | ALPN | Kenyon | MBON | rho(cosine, Kenyon) beyond receptor + ALPN overlap (20k random pairs) |
|---|---|---|---|---|---|
| last 3 tokens | **1.62** | 1.36 | 1.28 | 1.04 | **0.006** |
| 512-token context | 1.49 | 1.31 | 1.20 | 1.04 | **0.011** |

**Reading:**
- Words that mean the same already arrive with similar receptor patterns: the writer's embedding is semantic
  (1.62x).
- The similarity then shrinks layer by layer: receptors 1.62, ALPN 1.36, Kenyon 1.28, MBON 1.04.
- Beyond what the senses already share, the Kenyon code adds nothing (rho 0.006 / 0.011).
- So the brain's own integration builds a **context** code, but no **meaning** of its own. P3 fails.

**Where this leaves the hypothesis (P1–P3 together):**

| prediction | result |
|---|---|
| P1: Kenyon is a context code | holds, over about 3 words, and only in continuous speech |
| P2: Kenyon novelty is not rarity | holds |
| P2: Kenyon novelty relates to the brain's own CE | fails |
| P3: Kenyon adds meaning beyond the senses | fails |
| MBONs read the Kenyon code | fail |

**Two causes, one per change:**
1. **The readout loop.** MBONs do not read the Kenyon code, although compartment memory (DA-gated Kenyon → MBON,
   phi) is on in m_main. `mbon_input_probe.py` is queued after d5b c10 to find out why. It splits the MBON drive by
   class (compartment synapses at comp_weight × (1 + phi)) and asks whether the Kenyon → MBON drive is
   context-specific.
2. **The objective.** Nothing in training asks the Kenyon code to be predictable or semantic: CE is token-level, and
   the paced branch never applies lambda_jepa. The JEPA term (LeJEPA: no stop-gradient, SIGReg) is the change that
   asks for it. Its prediction: after training, P3's partial rho > 0 and the next-pattern prediction error falls
   below the mean-pattern 0.698.

## Design: the JEPA term in the stream branch (theory only; built after the MBON probe decides the order)

**Representation.** Per word t, the Kenyon activity transmitted over the word's ticks: surrogate spikes summed per
cell, differentiable through BPTT. A learned linear projection P maps it to z_t ∈ R^32, and z_t is
standardised over the window. The Kenyon layer is chosen because it is the integrated context code (P1). The
receptors would only repeat the input.

**Predictor.** A linear map Q from the brain's read at the end of word t−1 (the same read that predicts the token)
gives ẑ_t. The prediction therefore has to come from the brain's state. No separate network may learn it.

**Loss.**

  L_jepa = mean_t ||ẑ_t − z_t||² + λ_s · SIGReg({z_t})        (LeJEPA: no stop-gradient, no EMA)

added to the CE as λ_jepa · L_jepa, with λ_jepa set so that L_jepa is about 10% of the CE at the start.
- Gradients flow into both sides: the brain learns to predict its next Kenyon representation, and the Kenyon code
  learns to be predictable.
- SIGReg keeps z isotropic. Without it, both sides could collapse z to a constant.

**Batch size.** A window has 32 words, and SIGReg on 32 samples is noisy in high dimension. Hence z is 32-dimensional,
and SIGReg pools the last 4 windows' z, as LeWM pools over time. The older windows are detached, so this pool is the
only stop-gradient in the design.

**Predictions, written before the run.** One change (λ_jepa > 0 in the stream branch) vs the unchanged lineage, from
the same checkpoint, with a living gate:
1. Brain contribution (1) is not reduced by more than its error.
2. The next-pattern prediction error (P2 probe) falls below the mean-pattern 0.698.
3. The P3 partial rho (Kenyon vs GPT-2, beyond receptor + ALPN overlap) rises above 0.05.
4. Kenyon novelty and prediction error start to predict the brain's own CE residual (rho > 0.1).

**Failure branches:**
- If (1) drops: the term competes with the token objective; lower λ_jepa.
- If (2) holds but (3) does not: predictability without meaning; the Kenyon code is shaped only by its own
  predictability. Next would be the cross-sense version (LeVLJEPA: predict one sense from another).

## MBON probe (m_main c7, 2,968 words): the readout is naive because the compartment memory cannot learn

**Drive into the MBONs** (shares of |drive|, all ticks):

| source | share |
|---|---|
| unannotated cells (`None`) | −0.603 |
| MBON → MBON | −0.219 |
| Kenyon (compartment synapses) | **+0.166**, the only real excitation |
| CX | +0.011 |
| ALPN | +0.002 |

**The Kenyon → MBON drive is almost the same for every word.** Correlation of the per-MBON drive vector:
- same last 3 words: [0.19 | 0.97 | 0.94 | 1.00];
- random pairs: [0.07 | 0.93 | 0.89 | 0.99].

MBON firing: same contexts 0.55 vs random 0.47. The median MBON fires on 0% of ticks; the mean is 1.9%. (The "14% fire
every word" read earlier was the per-word coding level, not the same cells.)

**The plastic memory phi is nearly untouched.** [−0.84 | −0.009 | −0.067 | 0]; 18% are below −0.1 and 2% below
−0.5. The learned rule's constants barely moved from birth (eta −2.00 → −2.00, forget 0.020 → 0.027, trace about
30 ticks).

**Cause: receptor saturation at the compartments.**
- Tonic dopamine at the MBONs is [0 | 0.156 | 0.77 | 8.2], **14.5 × the affinity K = 0.0108** at the median. K was
  calibrated on the mean of all targets; other targets have a median of 0.0135.
- So occupancy sits at 0.93 for both the fast and the slow pool, and the teaching signal
  relu(occ(fast) − occ(slow)) is [0 | 0 | 0.002 | 0.029].
- The relative dopamine changes are there at this instant ((fast − slow) / slow from −1.0 to +0.12), but saturated
  occupancy hides them.
- **This is a mechanism inconsistency.** m_main runs `dopamine_adapted = True`: every target's receptors adapt to their
  own slow level, for threshold modulation and plasticity scaling. The compartment rule alone uses the fixed K.

**The fly's view.** Naive MBONs are broadly tuned in the fly too; odour or context specificity is learned, with
dopamine depressing the active Kenyon → MBON synapses (familiarity, alpha'3). Our MBONs are naive because that
learning cannot happen.

**Change 1, readout first (fly-native, one change).** A flag `--compartment-adapted`, default off, so running chains
are unchanged: the compartment rule's receptors adapt as every other target's do,
K_j = max(slow_j, K).
- occ(slow) becomes 0.5, so the window factor is the presynaptic trace alone.
- The teaching signal becomes relu(f/(f+K_j) − 0.5), about the relative burst / 4.
- At the median MBON the signal grows about 2x; at the dopamine-dense compartments, up to about 100x.
- Below K nothing changes (newborn safety).

**Predictions for one chunk from c7, paired by window against m_main's own chunk 8:**
1. (1) is not reduced beyond its error.
2. phi spreads: the fraction below −0.1 rises well above 18%.
3. Familiarity appears: the MBON drive at the second occurrence of a 3-word context falls below the first.
4. Same-context vs random MBON firing separates beyond 0.55 vs 0.47.
5. (2): the DA residual after rarity turns positive.

**If not:**
- If phi spreads but no familiarity appears: the bursts are not paired with the right Kenyon activity (timing), so
  check the trace.
- If phi does not spread: the bursts are too small. Then the dopamine cells' own input (KC → DAN release) comes next,
  as a separate change.

**Built and reviewed (subagent).**
- With the flag off: bit-identical, so running chains are unchanged.
- With the flag on: the manual carried-credit slopes match autograd. The flag is saved in the config and inherited on
  resume. The probes read it via `world_eval.load`.
- One effect to keep in mind when reading results: once adapted, the window occ(slow)/0.5 is pinned at 1 (it was
  about 1.87), so learning and forgetting per tick both drop by about 1.9x. Net teaching at the median MBON is
  about +2.5x, and forgetting is slower.

**Queued:** `cadapt_ab_after_d5b.sh`, after d5b c11.
1. Zero-shot MBON probe on c7 with the rule off vs on (the same weights; does phi spread and does familiarity appear
   within the probe stream?).
2. One treatment chunk from c7 with `--compartment-adapted`, then its MBON probe and (2) report.
3. m_main resumes; its own chunk 8 from c7 is the control, paired by window.

### Change 1, zero-shot (m_main c7, same weights, the rule off vs on over the probe's 2,968-word stream)

| | off | on (adapted) |
|---|---|---|
| phi at end, median | −0.016 | **−0.184** |
| phi below −0.1 | 26% | **61%** |
| Kenyon → MBON drive corr, same last 3 vs random (median) | 0.971 vs 0.925 (gap 0.046) | 0.955 vs 0.887 (gap **0.068**) |
| MBON firing corr, same vs random | 0.548 vs 0.468 | 0.537 vs 0.465 |
| familiarity: log drive ratio at a context's repeat (median / mean) | −0.005 / +0.008 | **−0.027 / −0.032** |
| same ratio, random pairs (median / mean) | +0.041 / −0.018 | +0.061 / −0.014 |

**Reading:**
- Prediction 2 holds within one probe stream: phi spreads (61% vs 26% below −0.1).
- The Kenyon → MBON drive becomes a little more context-specific: the gap between same-context and random grows from
  0.046 to 0.068.
- A small familiarity effect appears in the predicted direction: a repeated context drives about 3% less than its
  previous occurrence, while random pairs do not.
- MBON firing itself does not change yet.
- Zero-shot means the weights never adjusted to the new memory, and 257 repeat pairs give no error bar here. The
  treatment chunk (learning on) is the real test.

**Queue incident.** The first A/B queue waited for a status line, "d5b done; m_main resumes from c7", which already
existed from 14:20, so it fired at once (16:46):
- the zero-shot probes ran beside d5b (valid);
- the treatment chunk ran out of GPU memory;
- the m_main chain it then started beside d5b was stopped. m_main is still at c7.

`cadapt_ab_v2.sh` now waits for the d5b processes themselves to end. **Lesson: queue on a process, or on a line
stamped after the queue starts, never on a line that may already exist.**

### Evening (while the user was away): two runs died on a non-finite gradient; fixed; the change-1 pair moved to c9

**What happened:**
- **d5b finished.**
  - gate c10: (1) +0.926 ± 0.099; reading +0.120 ± 0.022; (2) −0.004 (residual −0.023); latch −0.001 ± 0.013.
  - c11 acts: answered 0.81–0.97; latency entropy median 0.73; within-k wait-gain rho −0.017; timing value over null
    0.0025.
- **The treatment chunk from c7 (adapted)** died 1,536 words in on a non-finite gradient: forward and loss finite,
  every parameter's gradient inf/NaN.
- **m_main's own chunks 8 and 9 ran** (gates c7 +0.371 ± 0.058, c8 +0.611 ± 0.068).
- **m_main chunk 10 died the same way** at 40,576 words, without the change, and so did a c8 gate arm. The chain then
  stopped for two hours.

**Cause.**
- This is the known blow-up of BPTT through 256 ticks, noted at the clip: the backward explodes past float range in
  one window, while the gradient norm before it is ordinary (about 2 to 4).
- It is not specific to change 1, since the control hit it too. Change 1 may raise the rate (the treatment died
  sooner), so its count is now a measured cost.
- None had occurred in m_main c1–c9 or in d5b.

**Fix (no change to semantics for finite windows).** A window whose gradient is non-finite, while its state is finite,
skips its parameter update instead of killing the run.
- Default on: `FLY_SKIP_NONFINITE`.
- Counted as `skipped_updates` and in the treatment's status line.
- A non-finite state still stops the run.

**The pair moved to c9.** The chain had pruned c7 (it keeps c1, every 10th and the latest two), so the rerun from c7
could not start. Now:
- control = m_main's chunk 10 from c9 (resumed 21:30);
- treatment = chunk 10 from c9 with `--compartment-adapted`, on the same words (`cadapt_ab_v4.sh`, after the c10
  gate, AG and probes);
- then both MBON probes and both (2) reports, and m_main resumes.

### Change 1 result: one chunk from c9 on the same 4,096 words; treatment (adapted) vs control (m_main's own chunk 10)

| | control c10 | treatment c10 (adapted) |
|---|---|---|
| (1) NLL, treatment − control (paired over 128 windows) | | **+0.0022 ± 0.0024** (no harm) |
| skipped updates (non-finite) | 0 | 0 |
| phi median / fraction below −0.1 | −0.083 / 47% | **−0.437 / 74%** |
| Kenyon compartment share of MBON drive | 0.166 | 0.115 |
| MBON firing, mean fraction of ticks | 1.17% | 0.95% |
| Kenyon → MBON drive corr, same last 3 vs random (median) | 0.965 vs 0.927 (gap 0.038) | 0.953 vs 0.899 (gap 0.054) |
| MBON firing corr, same vs random (median) | 0.714 vs 0.642 (gap 0.072) | 0.583 vs 0.497 (gap 0.086) |
| familiarity, log drive ratio at a repeat vs random (mean) | +0.010 vs +0.012 | +0.000 vs +0.003 |
| (2) rho(surprise, phasic DA) / residual after rarity | −0.003 / +0.024 | −0.009 / +0.017 |
| phasic DA sd (word level) | 0.94 | **0.14** |

**Predictions:**

| prediction | result |
|---|---|
| 1. (1) not harmed | holds |
| 2. phi spreads | holds |
| 3. familiarity appears | fails |
| 4. MBONs separate contexts | marginal (+0.014) |
| 5. the DA residual turns positive | fails |
| (not predicted) phasic DA | 7x smaller |

**Cause, from the failure branch "phi spreads but no familiarity".** The memory now learns, but its depression is
global rather than pattern-specific: median phi −0.44, Kenyon drive to the MBONs down a third, MBON firing down 20%.
- The dopamine that teaches it is not novelty-gated. Bursts depress whatever Kenyon cells were active, and over a
  text stream every cell is active sometimes.
- The 3-word Kenyon context code rarely recurs exactly (P1: about 9% of 3-grams repeat in 3,000 words). So the
  memory becomes a usage map of the Kenyon cells, not a record of which contexts were seen.
- With the drive through the KC → MBON → DAN loop weaker, the DAN bursts shrink (phasic sd 0.94 → 0.14).

**Familiarity needs something that recurs.** In the fly, the same odour recurs. Here it would be the same meaning,
which needs a semantic Kenyon code (P3 failed: none beyond the senses).

**Decision: change 1 is not adopted into the mainline.**
- It works as a mechanism and costs nothing in prediction.
- But it shrinks the DA bursts 7-fold and buys no familiarity while the code it learns from is non-semantic.

**Next: change 2, the JEPA term, on the unchanged mainline** (design above). It is the precondition change 1 lacks.
Change 1 is retested on top once P3's partial rho is above 0.05.

### Change 2 built: `--stream-jepa` (default off, reviewed)

**Implementation.**
- Per word, the Kenyon membranes at the end of its ticks (the integrated context code) go through `jepa_proj` to give
  z_t ∈ R^32.
- `jepa_pred` maps the read after word t−1 to the prediction ẑ_t.
- Loss: 0.95 · mean (ẑ − z)² + 0.05 · N · SIGReg(z), pooled with the last 3 windows' detached z. No stop-gradient,
  no EMA. Weight 0.5 in the total loss.
- Logged: `jepa_pred`, `jepa_sigreg`, `jepa_pred_over_mean_pattern`, `jepa_z_std` (about 1 when healthy, falling
  toward 0 at collapse).

**Review fixes:**
1. The new parameters had to be allowed to start fresh Adam moments at a resume; without that, the resume crashed.
2. SIGReg is multiplied by the sample count, as in LeJEPA. Unscaled, it is about 0.01 for a Gaussian and at most 0.2
   at full collapse, so z could shrink to a constant for free. Scaled, it is 0.5 for a Gaussian and 26 at collapse.
3. A weight of 0 is no longer turned into 0.5.

**A/B** (`jepa_ab.sh`), mechanism test:
- control = m_main's chunk 11 from c10; treatment = chunk 11 from c10 with `--stream-jepa`, on the same words;
- then P2 and P3 on both c11s.

## Static connectome audit: can the fly's own circuit carry iP-VAE's "explaining away"? (`mb_feedback_audit.py`, `neuprint_apl_kc_rois.py`)

**Kenyon input, raw synapses** (shares):

| source | share |
|---|---|
| Kenyon → Kenyon | 0.530 (97% axo-axonal in the lobes; 2.9% in the calyx, already pinned by `kc_dendritic_only`) |
| ALPN | 0.180 (all in the calyx; median 97 per KC) |
| DAN | 0.103 (on the axons in the lobes) |
| APL | 0.090 (median 50 per KC) |
| DPM | 0.015 |
| MBON | 0.005 |

**APL.**
- Of APL → KC synapses, **only 15.5% sit in the calyx**, on the Kenyon dendrites (30,407 of 196,200; per-pair median
  13%). The other 85% sit on the Kenyon axons in the lobes and peduncle.
- KC → APL is 13% calyx.
- The loop is reciprocal per cell: APL → KC vs KC → APL, rho 0.61.
- APL → ALPN (13,070 synapses) and ALPN → APL sit essentially all in the calyx, on the PN boutons.

**Feedback from the mushroom body to the PNs is tiny** (KC 0.4%, APL 0.3% of ALPN input).

**Phi^T Phi** (cosine overlap of two Kenyon cells' PN input): median 0, mean 0.04; 8.4% of pairs above 0.2.

**Two consequences:**
1. **A point-model artifact, not fixed until now.** In the model, all APL → KC synapses act on the Kenyon membrane,
   so the Kenyon cells integrate about 6.5x the dendritic inhibition they really get.
   - The earlier input-budget probe found the Kenyon drive to be about −0.55 from APL-like cells against +0.42 from
     PNs.
   - With 85% of that inhibition axonal, the dendritic budget would be roughly −0.09 vs +0.42.
   - This fits the margin probe (one word lifts the Kenyon top 1% only to 0.33 θ) and the weak word code. It is the
     same kind of artifact as Kenyon → Kenyon and Kenyon → DAN, already fixed.
   - Fix (fly-native, connectome correctness), the same pattern as `attach_kenyon_dendritic_only`: scale each
     APL → KC edge by its calyx share and pin the edges with no calyx synapse.
   - The axonal part's real role is presynaptic inhibition of the Kenyon output (gain on KC → MBON). That is a later,
     separate change: a terminal-local factor like the release.
2. **iP-VAE in the fly's own terms.** x = PN bouton output in the calyx; z = Kenyon spikes; Phi^T = PN → KC weights.
   - The explaining-away term Phi^T Phi z needs inhibition onto KC i in proportion to what the KCs sharing its PN
     boutons are doing.
   - The calyx APL is non-spiking and local: its branches take input in a microglomerulus (from the PN bouton and the
     claws around it) and inhibit there, both the claws and the bouton itself.
   - Inhibition local to a bouton is therefore shared by exactly the KCs that share that bouton:
     inh_i ≈ Σ_g Phi_gi (Σ_j Phi_gj z_j) = (Phi^T Phi z)_i.
   - So a **local calyx APL** is the fly's substrate for explaining away. It is computable with the existing PN → KC
     edge list: two sparse products, no new weights.
   - Its residual, the Kenyon activity not explained by co-active cells, is the representation-level surprise that
     KC → DAN release can carry to the dopamine cells.

**Order, one change at a time, after the JEPA A/B:**
1. The APL → KC dendritic-only fix (correctness). Predicted: Kenyon level up from 3%, the single-word margin up from
   0.33 θ, same-word repeatability up, and the context code (P1) kept.
2. The local calyx APL (Phi^T Phi z explaining away).
3. Kenyon residual → DA through the release, judged by the living (2) gate.

### APL → Kenyon fix built: `--apl-kc-dendritic-only` (user: "fix it")

**Why it was missed.**
- The 10-12 audit flagged APL as a compartmentalized giant modelled as one point (row 8, Medium) and listed two fixes:
  a graded output, or a split by region.
- 10-13 did the graded output only. The ROI query needed for the split stayed on the to-do list, so the 85% axonal
  share was not measured until today.

**The fix.**
- Mirrors `attach_kenyon_dendritic_only`: each APL → KC inhibitory edge is scaled once by its calyx share (per pair,
  from neuPrint), and the 819 axonal-only edges are pinned at 0.
- Checked on the graph: 4,633 edges, mean share 0.143, weight-weighted share 0.155 (= neuPrint's 15.5%).
- Saved in the config; later resumes only pin.

**A/B** (`apl_ab.sh`, after the JEPA A/B):
- control = m_main's chunk 12 from c11; treatment = chunk 12 from c11 with the fix;
- word margin, Kenyon repeat (P1) and P3 on both; (1) paired; (2) on both.

**Whether this could make the JEPA term unnecessary:** see the reply of 2026-10-14 23:00 in the status/doc trail.
- This fix should restore what the senses deliver (word repeatability, excitability).
- It does not by itself add meaning beyond the senses. But the senses already carry some meaning (receptors 1.62x
  for GPT-2 neighbours), and the Kenyon layer lost most of it (1.28x). If the fix lets the Kenyon code keep the
  senses' meaning, familiarity of meaning may work without JEPA.
- So: measure P3 after the fix before deciding on JEPA.

### JEPA A/B stopped: the treatment's z collapsed from the start

**What happened.**
- From the first logged window, z std was 0.02 and SIGReg × N sat at 26, its full-collapse value. Prediction error
  over the mean pattern was 1.3–22, worse than the mean.
- The raw Kenyon membranes are small, so z = proj(h) began near 0. There the characteristic function is about 1 and
  SIGReg has no gradient, so z never left.
- Paired NLL vs control: +0.0048 ± 0.0029.

**Fix (built, flag-gated).** Standardize each Kenyon cell's membrane over the window's words before the projection,
so that z starts at std ≈ 1 (checked: 0.99).

**Rerun after the APL fix decision.** The JEPA probes were stopped: the treatment's would have measured a collapsed
term, and the control's are not needed now.

## Systematic compartment audit (`compartment_audit.py`, neuPrint, user: "check every circuit we use")

**Method.**
- For each target group and each input class: the share of synapses that land in the target cell's output-enriched
  primary ROIs, i.e. ROIs holding a larger share of that cell's output sites than of its input sites ("mixed"; > 1.5x
  is "axonal").
- The first criterion (more output than input sites in the ROI) found nothing in the mushroom body, because Kenyon
  axons receive heavy axo-axonal input.
- Validation against known truths: KC → KC 0.88 (truth 0.97); APL → KC 0.73 (truth 0.845). So "mixed" is a slightly
  low estimate of the axonal share.

**Point-model artifacts on the route, by size** (share of the target's input × share axonal):

| target ← source | share of input | axonal ("mixed") | status |
|---|---|---|---|
| KC ← KC | 0.551 | 0.88 | fixed (`kc_dendritic_only`) |
| DAN ← KC | 0.588 | 0.87 | fixed (`connectome_v2` pin) |
| KC ← DAN | 0.108 | 0.91 | modulatory channel (dopamine at the terminals), fine |
| **KC ← APL** | 0.094 | 0.73 (truth 0.845) | **fix in A/B now** |
| **ALPN ← cb_intrinsic** (LH and others onto PN boutons) | 0.096 | 0.89 | **open**: 8.5% of PN input |
| **MBON ← cb_intrinsic** | 0.224 | 0.40 | **open**: about 9% of MBON input |
| **DN ← VNC intrinsic, DN, AN** | 0.032 / 0.130 / 0.124 | 0.84 / 0.16 / 0.15 | **open**: about 7.7% of DN input |
| ALPN ← ALPN | 0.105 | 0.18 | open, small |
| KC ← DPM, MBON | 0.016, 0.005 | 0.93, 0.92 | open, about 2% of KC input |
| DAN ← MBON, DAN, DPM | 0.024, 0.020, 0.009 | 0.38, 0.71, 0.89 | open, about 3% of DAN input |
| ALPN ← KC, APL | 0.005, 0.003 | 0.99, 0.99 | open, small (presynaptic on PN boutons in the calyx) |

**Fine:**
- KC → MBON (0.04 axonal);
- olfactory and ALLN → ALPN;
- motor neurons (vnc 2.6% axonal; cb 8%, mixed 32%: unclear).

APL and DPM inputs are mixed everywhere: they are non-spiking, local cells, for which the point concept itself fails.

**Plan.**
1. One general mechanism instead of pair-by-pair pins: scale every edge by the fraction of its synapses in the target
   cell's non-axonal ROIs (per pair, from neuPrint). It would replace `kc_dendritic_only`, the KC → DAN pin and the
   APL fix with one rule.
2. Built and A/B'd after the APL result (which tests the single largest open case first). Sizes per edge come from a
   per-pair query by target group.

### The general compartment rule, applied to the mainline (user: "fix them first", instead of one A/B per pair)

**`--compartment-dendritic`.** Every edge into Kenyon cells, MBONs, DANs, PNs and DNs is scaled once by the share of
its synapses outside the target's axonal territory, from per-pair neuPrint tables (`neuprint_dendritic_share.py`):
- the calyx for Kenyon targets (anatomical truth);
- output-enriched primary ROIs for the other targets.

Edges with no synapse in the dendritic territory are pinned at 0. Kenyon → Kenyon stays with `kc_dendritic_only`, and
the separate APL flag is skipped when this one is on. Pinned: 231,886 excitatory and 89,006 inhibitory edges.

Weighted dendritic share kept per target:

| target | excitatory | inhibitory |
|---|---|---|
| KC | 0.82 | 0.23 (APL → KC 0.155) |
| MBON | 0.90 | 0.59 |
| DAN | 0.32 (mostly KC → DAN, already pinned) | 0.75 |
| PN | 0.93 | 0.81 |
| DN | 0.91 | 0.87 |

**Queue `compartment_fix_mainline.sh`:**
1. control = chunk 13 from c12 without the rule;
2. the mainline continues from c12 with the rule;
3. on both c13s: word margin, Kenyon repeat (P1), P3 and (2); paired NLL.

### Zero-shot sanity of the compartment rule (user: "pause everything, look at the fixed brain alone"; m_main c11, rule off vs on)

The rule was applied at load to the trained weights. Weighted dendritic share: 0.807 excitatory, 0.720 inhibitory.

**Per group** (rate per tick, silent fraction):

| group | rule off | rule on |
|---|---|---|
| whole brain | 7.0 Hz, 80% silent | 5.9 Hz, 81% silent |
| Kenyon | 0.0044, 39% silent, v/θ 0.34 | **0.0067**, 32% silent, v/θ 0.43 |
| ALPN | 0.0124 | 0.0127 |
| MBON | 0.0126, 61% | 0.0149, 66% |
| DAN | 0.0119, 36% | **0.0186**, 25% (PAM 0.011 → 0.019; PPL1 0.021 → 0.017) |
| descending | 0.0865, 26% | **0.0458**, 43% |
| read cells | 0.102, 26% | **0.073**, 36% |
| APL | 0.87 "saturated" | 0.94 "saturated" (a readout artifact of the graded APL in both) |

**No group dies or saturates because of the rule.** Gustatory and chemosensory cells are silent in both: they get no
text.

**Kenyon code** (3,000 words):

| | rule off | rule on |
|---|---|---|
| level | 0.035 | 0.053 |
| same word vs different (median) | 0.113 vs 0.086 | 0.149 vs 0.118 |
| same last 3 vs different (median) | 0.448 vs 0.084 | 0.434 vs 0.127 |

The Kenyon layer is more active and broader, but **not more word-specific**: the same/different ratio is 1.32 vs 1.27,
and context specificity drops from 5.3x to 3.4x.

**Isolated-word margin:** Kenyon top 1% at h/θ 0.346 → 0.301, lower. The excitatory cut onto Kenyon cells includes
DAN → KC fast edges on the axons. These had given a tonic somatic drive; removed, they no longer prop the margin.

**My earlier hypothesis was wrong.** The excess APL inhibition is not why one word cannot drive a Kenyon cell:
removing it does not raise the margin. The limit is the PN drive itself. An isolated word activates about 1.2% of the
PNs, too few claws for a Kenyon cell's coincidence threshold. This is the fly's own design (a KC needs several of its
about 6 claws), not a defect.

**The read surface is quieter** (DN −47%, read cells −29%). The weights were trained on the wrong wiring, so training
on the corrected connectome has to re-adapt them; the paired NLL will show the cost.

**Verdict: healthy; resume.** `compartment_fix_mainline_v2.sh`:
1. control = chunk 12 from c11 without the rule;
2. the mainline continues from c11 with the rule;
3. probes and paired NLL on both.

### Speed (user: "speed decides how soon we are back on the mainline")

**Where the time went.** About 70 min per comparison:
- a 4,096-word chunk is 20–25 min (about 4.5 words/s on a 4 GB GPU);
- the control and the treatment run serially (one GPU, tight RAM);
- the three probes reload and re-read each time, about 15 min per arm.

**New rules:**
1. A mechanism test goes zero-shot first (load, look; about 15 min).
2. Training comparisons use 1,024 words when the question is not long-term adaptation (about 6 min, 32 paired
   windows).
3. One merged probe, `snapshot_probe.py`: one load, P1, P3 and the word margin in about 5 min.
4. The control is the mainline itself wherever possible, so its chunk is progress, not a test cost.

**Target: about 15–20 min per mechanism comparison.** `compartment_fix_mainline_v3.sh` already uses the merged probe.

## Mengli's raising lineage, integrated (`grow_mengli.sh`; starts via `start_mengli_after_check.sh`)

**Why the abilities move to the mother world** (user: "in OWT, could we never measure the abilities?"):
- Speaking has no consequence in a text stream, so its timing can never be learned there.
- Looking and thinking have value in the text, but small (timing value 0.004–0.015 nats). That is why d5b's acts
  learn so slowly.
- Familiarity needs recurrence: exact 3-word contexts recur about 9% in OWT. That is probably why change 1 found no
  familiarity, so that result is reopened in the mother world.
- From now on, OWT (m_main, paused at c12) is the testbed for "no harm" checks of mechanisms, and the mother world is
  where abilities are formed and judged.

**What the lineage integrates:**

| piece | source |
|---|---|
| Brain | d5b c11: the mainline lineage (receptor_lang, gain 0.5) plus 11 chunks of acts (look / think / speak muscles), gate c10 +0.926 |
| Connectome fixes | `kc_dendritic_only`, `connectome_v2` (KC → DAN pin, sensory terminals, modulatory fast edges), **the general compartment rule** (APL → KC, PN boutons, MBON / DN axons), applied at the first resume |
| Learning safety | non-finite gradient windows skip their update |
| Mother | DeepSeek mouth, contingent; speaking act; epistemic credit discounted per tick (tau 32: the best behaved of the four runs); beats of 8 ticks |
| Continuity | `mother_state.json`: the transcript tail, the cache anchor and the silence count survive each chunk; a new chunk resumes with her calling "Mengli?", not a new greeting |
| World memory | `data/harness/world.json` (Katz the cat, Papa's drum, Aunt Ling's Sundays, teddy); Aunt Ling (the heart) consolidates it |
| Eyes | continuous (the live pass does not sample binary eyes), no noise stretches |

**Checks.** Every chunk is 1,024 words and logs the monitor and the epistemic returns. Every 5 chunks: the snapshot
probe (Kenyon code, meaning, margins) and (2) on the chunk's own words.

**Lessons from the text side that apply here:**
1. The Kenyon code is a context code over about 3 words and lives only in continuous speech: one beat of silence
   empties it. Measure the CE of the first words after a turn boundary in the transcripts before changing the pause.
2. Familiarity needs recurrence. Retest the compartment receptor adaptation (change 1) as a fork of this lineage,
   where names and things recur.
3. The JEPA term (fixed: standardized membranes) and the local calyx APL (Phi^T Phi z explaining away) are the next
   forks, one at a time, judged by snapshot P3 and (2).

**Start condition** (`start_mengli_after_check.sh`): the compartment rule's paired cost against its control is at most
0.05 nats per word. Then m_main is paused at c12 and the lineage begins.

### Per-tick gradient clipping; no imposed reply pause (user, 2026-10-15 00:05)

**Why windows blew up although there is a gradient clip.**
- The clip acts after the backward. The backward itself runs through 256 ticks, and in a loop with gain above 1
  (latched cells, Kenyon → MBON → DAN) the error grows exponentially past float range.
- inf / NaN cannot be rescaled, so the clip cannot help.

**Fix: `clip_state_gradients`.** At every tick, each physical-state tensor (h, ge, gi, b, x, u, dopamine, compartment
memory and trace, spike ring) has its incoming gradient capped at norm 1,000, with non-finite entries zeroed, on the
way back.
- It is branch-free, with no host sync per tick.
- Below the cap the gradient is multiplied by exactly 1, so the forward and every ordinary update are unchanged.
- Counted as `step_grad_clips`; `FLY_STEP_GRAD_CLIP=0` switches it off.
- Test: a gain-3 chain over 200 steps gives inf without the clip and finite with it (194 clips).
- The skip-on-non-finite stays as the last guard.

**No imposed reply pause.** `raising.json` `mother.pause_beats_before_reply` goes from 1 to 0.
- The one-beat silence before every reply emptied the Kenyon context (gap sweep).
- The words already come at variable intervals (pace 2–16 ticks, held on the senses), the mouth's latency never
  reaches the child, and the mother's replies differ in length.
- The child's own silences still last a beat: that is her act.
- Logged in `mother_notes.md`.

### The compartment rule's comparison: no harm, and the first positive (2) residual

**A counting bug, fixed.** `grow_mainline_live.sh` counted "chunk N done" lines, and the 22:57 "JEPA treatment chunk
11 done" line matched.
- So the chain named the rule chunk 13 (saved as c13.pt, scores_c13), pruned c11, and the v3 queue waited forever for
  "chunk 12 done".
- Repaired: renumbered to c12 (status line annotated). The scores file had kept 68 windows of the stopped 22:59
  no-rule run; they are split off into `scores_c12_partial_norule_2259.jsonl`. The paired NLL keys windows by
  position, the later rows win, so it was unaffected.
- The counters in `grow_mainline_live.sh` and `grow_mengli.sh` now match only the chain's own lines (`grep -E`).

**Result** (chunk 12 from c11, the same 4,096 words, 128 paired windows):

| | control (no rule) | rule |
|---|---|---|
| NLL, rule − control | | **−0.0035 ± 0.0036** [−0.098 \| −0.004 \| −0.0035 \| +0.089]: no harm, if anything better |
| (2) rho(surprise, phasic DA) | −0.028 | +0.007 |
| (2) residual after rarity | −0.014 | **+0.107** |
| (2) rarity part | −0.029 | −0.022 |
| phasic sd | 0.114 | 0.125 |

The residual is the first clearly positive one: DA follows the part of surprise that is not rarity. It is one chunk
and the per-1,024-word median is still about −0.01, so it has to repeat before it counts. Mengli's lineage checks
(2) every 5 chunks.

**Adopted.** m_main is paused at c12 as the testbed. Mengli's lineage starts once the snapshot on the rule's c12 is
done (`finish_compartment_check.sh`).

## 2026-10-15 00:45 — Mengli: the mother's voice and the child's end of turn (from chunk 3)

**Why Mengli spoke so little.** In chunks 1–2 her turns averaged 1.6–1.8 words:
- 13 of 33 turns ended at her own '.', '?' or '!';
- 20 of 33 ended when she yielded.

A sentence end took the floor away from her, and she babbles these marks often.

**Change 1: the child's end of turn** (`raising.json` `mother.child_end_of_turn = ["\n", "\n\n"]`). Her turn now ends:
- when she yields (her own act);
- at a line break;
- at 16 words (`max_fly_words`, the cap against endless talk).

For an LLM, learning when to stop is hard. For her, stopping is already an act: a yield is the muscles not committing.

**Change 2: the mother's voice** (`--mother-voice`, `attach_mother_voice`). Nothing in her senses said whose word it was:
- her own words are heard next (self-hearing);
- they arrive through the same embedding as the mother's words.

The fly hears through Johnston's organ: JO-A and JO-B follow sound and near-field vibration such as courtship song. While a word of the mother's is the input on the senses, these 138 cells receive a steady current of 1.5 × their rheobase. Her own words carry no such current.

The voice is the same in all three passes:
- the live pass (`live_window`);
- the training pass (`forward_window`, `voices` from `forced['speakers']`);
- the epistemic replay (`replay_ce`, the record's `voices`).

The first input of a window takes the voice of the previous window's last word (`learner.last_voice`, kept in `mother_state.json`).

**Biology.**
- *Babies* hear their own voice mostly through bone conduction, with a different timbre. Their auditory cortex is damped while they speak (speech-induced suppression, via an efference copy of the motor command). The mother's voice arrives through the air, from a place, in her own timbre.
- *Insects* rely on corollary discharge. The cricket's CDI interneuron inhibits the auditory pathway on every chirp. Flies send efference copies to their visual neurons during voluntary turns.

Here the self side already exists, as her own act: commit/yield is in the brain and self-hearing CE is on. The voice adds the other side: the mother's sound marked in the senses.

**Watch** (status line every chunk, (2) every 5):
- mean turn words and silent turns;
- end reasons (end_word against end_yield and end_breath);
- the epistemic returns;
- whether familiarity (MBON) separates the mother's words from her own.

## 2026-10-15 01:00 — Mengli probe after chunk 3 (`mengli_probe.py`): predictions before the run

The user asked five questions: does Johnston's organ work, does a signal still die after two hops, is there a
corollary discharge, does the familiarity memory catch the repeated "Mengli", and do the Kenyon and dopamine cells
matter?

The probe is a diagnostic on a copy of c3: weights frozen, the online compartment rule still running. It replays her
own last 600 tokens of conversation, holding each word for 6 ticks. Every effect is a paired branch taken from the same
state. Results are reported as [min | median | mean | max]. The lineage pauses for it, because the 4 GB GPU cannot hold
the probe beside training.

**Predictions**

1. **(V) The voice**
   - JO itself will fire strongly.
   - Hop 1 (315 first-order cells: SAD and AMMC) rises.
   - From hop 2 on the change is near 0, if the birth synapse scale still holds (one synapse = 0.35% of threshold).
   - Kenyon, MBON and DAN show about 0.
   - If so, the voice is felt only at the ear. Nothing downstream can use it until learning strengthens the route.
2. **(H) Two-hop decay of the text input**
   - The same profile: the cells the writer drives and hop 1 rise, the change falls to about 0 by hop 2–3.
   - The Kenyon cells are the exception, because their claws were rescaled by the compartment rule.
3. **(C) Corollary discharge**
   - The static route gives excitation at hop 1 and net inhibition at hops 2–3 (`efference_route_probe.py`).
   - In activity, expect a small command-specific effect on the auditory cells: first-order rising a little at tick 1–2
     and nothing clear after.
   - A clear specific inhibition at ticks 2–4 would mean the route already carries a corollary discharge.
4. **(F) Familiarity**
   - Earlier, on OWT, phi saturated and depression was global (change 1 not adopted).
   - So phi of the edges used by " Meng"/"li" will differ little from the rest, and its correlation with word count
     stays weak.
   - Comparing c1 with c3 shows whether three chunks of "Mengli" left a trace.
5. **(K) Kenyon and dopamine**
   - Silencing the Kenyon cells for one word costs little, because the MBONs barely read them (the MBON probe).
   - Silencing them for the whole stream costs more.
   - Silencing the dopamine cells for one word costs about 0. Sustained, the cost is small but not zero: DA modulates
     thresholds and phi.

## 2026-10-15 02:30 — Why Mengli falls silent: the speaking credit favours silence by construction

**Observed.** Her silent turns rose 26% → 58% → 86% (chunks 1–3), fell to 30% in chunk 4, then reached 100% in
chunk 5: 554 turns, 0 words, and the mother only calls "Mengli?". This is the dark room.

**Credit per decision, chunk means of the window means:**

| chunk | said | silent | of which: yield after speech | of which: pure silence |
|---|---|---|---|---|
| 1 | 0.39 | 0.49 | 0.53 | 0.31 |
| 2 | 0.15 | 0.23 | 0.22 | 0.16 |
| 3 | 0.11 | 0.19 | 0.22 | 0.12 |

Silence out-earns speech in every chunk. The causes are in the design, not the brain.

1. **The reply's credit goes to the wrong decision.** The epistemic return of a decision is the learning progress of
   the mother's words arriving within 64 ticks after it, discounted with tau 32.
   - After a "say", the next words are her own, which earn no learning progress.
   - The mother's reply comes only after the yield, so for the "say" it is far away and discounted.
   - The yield right before the reply collects the reply.
   - So at every decision point silence looks better, although it was the speech that bought the reply.
2. **Continuity is not in the objective.** One quiet beat silences the Kenyon layer 30-fold (gap sweep). After a
   silence the context is gone and the mother's next words are harder to predict.
   - Learning progress does not price that loss. If anything, a word that is more surprising has more to teach.
   - The accuracy (expected surprise) part of expected free energy, "keep my world predictable", is missing.
   - The `predict` credit has an accuracy term, yet it also went silent: 76% in mother_v2_contingent. It prices her
     own words at the unigram prior (expensive), while silence buys the most predictable words of all, the call
     "Mengli?".
3. **The brain side is secondary.** The brain cannot hold context through a pause: no persistent activity at the
   Kenyon level. That makes silence truly costly, but the credit never sees the cost.

**Way out (one change at a time).**
- The credit of a decision should compare the futures of speaking and of keeping silent. Two options:
  - per turn: the reply is credited to the turn's speech, not only to its last yield;
  - the accuracy of the mother's following words (CE level) next to their learning progress.
- Then the brain's persistence through pauses, from the fly's own recurrent loops (MBON–DAN, central complex).

**Coordination.** A second agent's goal (`MENGLI_GOAL_20261015.md`: true silence, state continuation, delayed credit)
targets the same problem. Its `grow_mengli.py` chunk 6 failed at 02:21 with a CUDA memory-allocation error; nothing
trains now. The user decides who owns the lineage.

## 2026-10-15 02:40 — Mengli probe results

The probe ran on `last.pt` (50,944 words, the end of chunk 5), plus the familiarity part alone on c1.

It replays her own last 600 tokens. 296 of them are " Meng"/"li": she was silent, and the mother kept calling.

All results are paired branches from the same state. Rates are spikes per cell per tick, given as a change against a
baseline. Distributions are written as mean ± se [min | median | max].

**(V) The voice reaches the ear and stops there.** Prediction 1 holds.

| group | rate without voice | change with voice | relative |
|---|---|---|---|
| JO-A/B (138) | 0.0137 | +0.0725 [+0.052 \| +0.073 \| +0.094] | ×6 |
| auditory first order (283) | 0.0097 | +0.0039 | +40% |
| hop 1 from JO (979) | 0.0169 | +0.0019 | +11% |
| auditory second order | 0.0113 | +0.0005 | +4% |
| hop 2–4 | | 0.0000 | 0 |
| Kenyon, MBON, DAN | | 0.0000 | 0 |
| output cells | 0.0743 | +0.0007 | +1% |

Nothing that could tell "her" from "me" (MB, dopamine, acts) receives the voice.

**(H) A sensory signal still dies after two hops.** Prediction 2 holds.

| group | rate in a quiet interval | change with the word written | relative |
|---|---|---|---|
| cells the writer drives (hop 0) | 0.0003 | +0.0153 | |
| hop 1 | 0.0070 | +0.0006 | +9% |
| hop 2 | 0.0031 | +0.0001 | +3% |
| hop 3 | | 0 | |
| Kenyon | | +0.0002 | +9%, 62% of words |
| MBON | | +0.0003 | +2% |
| DAN | | 0 | |
| output | | +0.0057 | +8% |

The learned direct path to the output cells carries the word. The deep brain hardly hears it.

**(C) Corollary discharge: inconclusive, the test was too weak.**
- The command raised output firing by only +0.0008 at tick 1 and did not move the commit hazard at all.
- The auditory cells did not change: 0 ± 0.0001 at every tick.
- Rerun with a command strong enough to make the act commit before judging.

**(F) Familiarity does not catch "Mengli".**

| | phi on fired KC edges, Mengli | phi on fired KC edges, other words | rho(phi, word count) | MBON effect of phi, Mengli | MBON effect of phi, other words |
|---|---|---|---|---|---|
| c1 | −0.099 | −0.107 | +0.07 | −0.0018 | −0.0028 |
| c5 | −0.105 | −0.108 | +0.15 | −0.0005 | −0.0020 |

- Mengli's synapses are, if anything, less depressed than the rest.
- The MBON effect of phi shrinks with word count (rho +0.48): the opposite of familiarity.
- phi is near 0 for most edges (median −0.005) and deep for a few (min −0.84).

**(K) Kenyon and dopamine cells do not help prediction now.** Changes in next-word CE on the mother's words (intact
CE 1.06):

| ablation | change in CE |
|---|---|
| KC silenced for one word | +0.0001 |
| DAN silenced for one word | −0.0004 |
| KC silenced throughout | −0.025 ± 0.013 [−1.61 \| −0.012 \| +1.43] |
| DAN silenced throughout | −0.029 ± 0.013 [−1.39 \| −0.009 \| +1.47] |

Silencing either set, if anything, helps by about 2 se. Prediction 5 was wrong on the sustained part.

**Cause and way out.** One cause links all of these: signals do not travel. The birth synapse scale, one synapse = 0.35% of
threshold, is still the bottleneck (see the memory note on the birth scale).
- The voice cannot reach a decision.
- The text barely reaches the Kenyon cells.
- The MB therefore codes almost nothing about the conversation, so phi and DA have nothing to work on.
- A corollary discharge cannot form along routes that carry no signal.

The deep brain (KC, MBON, DAN) is effectively disconnected from the conversation. Fixing the speaking credit alone will
not make it carry the conversation.

The way out is the pending rebirth decision: a birth scale under which signals travel 3–4 hops (1–3.9% of threshold
per synapse reached DA, MB and motor zero-shot). It should be paired with the credit fix and checked zero-shot first,
with this same probe: V, H and F should become non-zero downstream.

## 2026-10-15 02:50 — Rebirth question, zero-shot on the conversation (`birth_scale_conversation.py`): predictions

**Setup.**
- Starting point: the birth checkpoint, with the general compartment rule applied.
- Variants: every edge set to a uniform efficacy per synapse of 0.35% (as born), 1%, 2% or 3.9% of the threshold.
- Input: Mengli's last 400 tokens of conversation, with the voice on for the mother's words.

**Predictions.**
- **Born (0.35%):** like the adult.
  - The voice and the text stop at hop 1–2.
  - Kenyon, MBON and DAN show about 0.
- **1–2%:** the conversation reaches KC, MBON, PAM and PPL1, and the voice reaches beyond the first-order auditory cells.
- **3.9%:** risk of runaway activity under continuous input.
  - It is only safe if the saturated fraction (cells firing on more than 50% of ticks) stays small and the late rate
    stays near the early rate.

**Choice rule.** The smallest scale at which V and H are non-zero in KC, MBON and DAN, with no saturation drift.

**Results (03:15).** Run on birth.pt with the compartment rule, 400 tokens, 2,400 ticks per scale. Values are means of
paired rate differences per cell per tick; the rate during a quiet interval is in parentheses.

| scale | text → KC | text → MBON | text → PPL1 | text → PAM | voice → aud1 | voice → KC/MBON/DAN | all-cell rate | silent cells | saturated cells | MBON rate, first → last quarter |
|---|---|---|---|---|---|---|---|---|---|---|
| born 0.35% | +0.0002 (0.0013) | 0 | 0 | +0.0001 | +0.0017 | 0 | 0.0030 | 89% | 0 | 0.0003 → 0.0005 |
| 1% | +0.0038 (0.012) | +0.0023 (0.022) | +0.0002 | 0 | +0.0038 | 0 | 0.0042 | 83% | 0 | 0.021 → 0.034 |
| 2% | +0.0084 (0.021) | +0.0071 (0.052) | +0.0011 | 0 | +0.0057 | 0 | 0.0060 | 75% | 0 | 0.046 → 0.075 |
| 3.9% | +0.0154 (0.031) | +0.0136 (0.087) | +0.0010 | 0 | +0.0080 | 0 | 0.0087 | 63% | 0 | 0.073 → 0.115 |

**Against the predictions.**
- Born behaves as predicted.
- From 1% on, the text reaches the Kenyon cells and the MBONs (KC +31% to +50%, MBON +10% to +16%). It reaches PPL1
  only weakly and PAM not at all.
- Prediction wrong: the voice still stops at the first-order auditory cells at every scale. This may be the fly's own
  anatomy rather than a defect. The Drosophila mushroom-body calyx gets mainly olfactory input; song goes
  JO → AMMC/WED → AVLP → courtship and premotor circuits. The voice would then have to tell self from mother through
  the auditory-premotor route, which training must open, not through the MB.
- Prediction wrong: 3.9% does not run away. No cell saturates, and the overall rate is the same in the first and last
  quarter.
- The MBON rate, however, drifts up by about 60% within 400 tokens at every scale above born. This needs a longer
  stability check before a long run.

**Reading.**
- 2% is the smallest scale at which the conversation clearly reaches the MB (KC +39%, MBON +14%) with no saturation.
- PAM (reward) is not reached by text at any scale. It should come from taste or valence routes, not from words.

**Rebirth candidate.**
- Birth at 2% per synapse plus the compartment rule.
- First a 2,000-token zero-shot stability check of the MBON drift.
- Then a new lineage from birth, with the speaking-credit fix as a separate, controlled change.

Waiting for the user's decision on lineage ownership: a second agent is working on Mengli, and there is one GPU.

**Stability at 2% over 2,000 tokens (12,000 ticks).** The drift plateaus.

| | first quarter | last quarter |
|---|---|---|
| all cells | 0.0061 | 0.0060 |
| KC | 0.0309 | 0.0301 |
| MBON | 0.065 | 0.075 |
| PPL1 | 0.040 | 0.039 |
| output | 0.0106 | 0.0105 |

- The MBON rate levels off at about 0.075, which it had already reached within the first 400 tokens.
- 4 cells saturate (0.0024%); 72% of cells never fire.
- Each word still reaches the mushroom body: KC +0.008, MBON +0.0087.

**2% passes the zero-shot checks.** The rebirth plan is ready: birth at 2% plus the compartment rule. The
speaking-credit fix comes after it, as its own controlled change.

## 2026-10-15 03:45 — Correction: the rebirth proposal rested on a wrong premise

**What was wrong.** `birth_scale_conversation.py` converted weights with the original builder's g = 0.0058, where one
synapse is about 0.35% of threshold. Mengli and every lineage since 10-13 are born on the Shiu-scale **r2** graph:
- the median edge is already 3.6% of threshold;
- the median |w| is 8.6× the original graph.

So the variants labelled 1%, 2% and 3.9% were global gains on r2:

| label | median gain | p10–p90 gain |
|---|---|---|
| 1% | ×2.8 | ×1.6–5.1 |
| 2% | ×5.5 | ×3.2–10.3 |
| 3.9% | ×10.8 | ×6.2–20 |

These are far beyond the biological scale. "Born" was r2 as it is.

**What stands.**
- On r2 as born, the conversation barely reaches the mushroom body: KC +0.0002 over a quiet rate of 0.0013; MBON, PAM
  and PPL1 about 0.
- The trained c5 shows the same: text dies after two hops, and the voice stops at the first-order auditory cells.
- The engine stays stable even at ×10 gains: no saturation over 2,400 ticks.
- The memory note on the birth synapse scale was written before r2 and is out of date for the current lineages. The
  synapse scale is **not** the reason the conversation misses the MB.

**Withdrawn.**
- The `--birth-synapse-efficacy` option is reverted on branch `claude/rebirth-birth-scale` (d0bc31c). Nothing was
  trained with it.
- No rebirth.

**Open cause.** Why does text reach the output cells but not the Kenyon cells on a Shiu-scale graph? Candidates to test
zero-shot before any change:
1. **Input strength.** receptor_lang writes each word weakly into 14k sensory cells: +0.03 spikes per tick per driven
   cell, and the mainline input gain is 0.5. Few PN claws coincide, which matches the earlier finding that an isolated
   word lifts the Kenyon top 1% only to about 0.33 θ.
2. **Anatomy.** Only the olfactory share of the text reaches the calyx, through PNs.
3. **Inhibition.** APL / PN gain at the calyx.

The next step is a dose–response of the writer gain (not the synapses) against KC / MBON reach and stability, with the
same probe.

## 2026-10-15 04:10 — Writer-gain dose–response, zero-shot on Mengli c5 (synapses unchanged)

**Setup.** The text drive is multiplied by ×1, ×2, ×4 or ×8 on every written tick. 400 tokens, paired word-vs-quiet
branches. Rates are spikes per cell per tick: the extra caused by the word, with the quiet-interval rate in parentheses.

| gain | written cells | KC | MBON | PAM | PPL1 | output | all-cell rate, first → last quarter | saturated |
|---|---|---|---|---|---|---|---|---|
| ×1 | +0.022 | +0.0021 (0.0053) | +0.0011 (0.025) | +0.0002 | +0.0001 | +0.0091 | 0.0066 → 0.0066 | 0.002% |
| ×2 | +0.060 | +0.0053 (0.010) | +0.0037 (0.036) | +0.0002 | +0.0003 | +0.020 | 0.0119 → 0.0117 | 0.002% |
| ×4 | +0.118 | +0.0090 (0.016) | +0.0058 (0.045) | +0.0005 | +0.0003 | +0.034 | 0.0192 → 0.0187 | 0.002% |
| ×8 | +0.194 | +0.0155 (0.022) | +0.0108 (0.055) | +0.0005 | +0.0001 | +0.046 | 0.0281 → 0.0274 | 0.002% |

**Reading.**

1. **Input strength is a lever.** The words' reach into the mushroom body grows roughly in proportion to the writer
   gain: KC ×7 and MBON ×10 from ×1 to ×8. The engine stays stable: rates are flat across the stream and nothing
   saturates.
2. **The written cells fire very weakly at ×1:** 0.022 spikes per tick. At 5 ms per tick that is about 4 Hz evoked,
   against tens to hundreds of Hz for a real receptor at its stimulus. The mainline also halves the input (gain 0.5).
   The principled invariant is "a word must make its receptor cells fire clearly", not a copied rate. ×4–×8 reaches
   20–40 Hz.
3. **The dopamine cells (PAM, PPL1) do not respond to text at any gain.** That is as it should be. Dopamine should
   follow surprise or value, not the presence of input, which is gate (2) and not a reach problem.
4. **Discrepancy to resolve.** At ×1 the KC word effect here (+0.0021 over 0.0053) is 10× the one in `mengli_probe.py`
   on the same checkpoint (+0.0002 over 0.0022). The streams differ (400 vs 600 tokens), and `mengli_probe.py` has since
   been edited by the second agent. Check before relying on either number.

**Next (training, one change, with control).** From the same checkpoint, writer gain ×4 against ×1 for 1,024 words,
paired by window. Judged by:
- the living brain contribution (noedges arm);
- (2) surprise → phasic DA;
- KC/MBON reach afterwards.

This needs the GPU, and therefore the lineage-ownership decision.

## 2026-10-15 05:20 — Writer gain ×4 vs ×1, living A/B from Mengli c5 (branch `claude/writer-gain-ab`)

**Setup.** One change: a fixed ×4 on the text writer drive. Both arms start from the same checkpoint and read the same
1,024 OWT words, each with its own living noedges arm. All learning is on and windows are paired. Values are nats per
word, as mean ± se [min | median | mean | max] over 32 windows.

**(1) Brain contribution, noedges − brain**

| arm | contribution | NLL brain | NLL noedges |
|---|---|---|---|
| ×1 | **+0.413 ± 0.049** [−0.40 \| +0.43 \| +0.41 \| +0.89] | 7.038 | 7.450 |
| ×4 | +0.287 ± 0.052 [−0.40 \| +0.27 \| +0.29 \| +0.97] | 7.163 | 7.450 |

The unigram on the same words is 7.361.

**×4 − ×1 on the brain arms:** +0.126 ± 0.031 nats worse [−0.26 \| +0.09 \| +0.13 \| +0.52].

**(2) Surprise ~ phasic DA**
- ×1: rho 0.011; residual after rarity −0.10.
- ×4: rho −0.010; residual +0.035.
- Neither is different from 0.

**Reading.**
- **Gate (1) on text is clearly positive for Mengli's c5 brain:** +0.41 nats per word over the noedges arm, and
  0.32 below the unigram.
- **×4 on the adult is not adopted.** It costs 0.13 nats per word within 1,024 words.
- **Cause:** the adult's weights and readout are calibrated to the ×1 drive. ×4 moves the operating point (output
  rate 0.08 → 0.12), so its recurrent prediction worsens even though the words reach the mushroom body more
  strongly (zero-shot dose–response). The noedges arms are identical (7.4503 vs 7.4504), so the cost lies entirely in
  the recurrent brain.
- **Way out:** the senses' strength cannot be changed suddenly on an adult. Two options:
  - a newborn comparison, ×4 vs ×1 from birth, where the brain grows up with strong senses;
  - a slow ramp, if the lineage must be kept.
- (2) stays at about 0 either way. Dopamine does not follow surprise yet; the next mechanism question is separate from
  reach.

## 2026-10-15 05:45 — Items 1, 3 and 4 re-run this session (Mengli c5)

**Item 1. m_olf living gate, segment 1 vs 2** (paired by window, 1,024 words each)

| segment | brain contribution [min \| median \| mean \| max] | brain vs unigram |
|---|---|---|
| live_c1 | +0.087 ± 0.037 [−0.42 \| +0.08 \| +0.087 \| +0.57] (noedges rerun) | 7.450 vs 7.541 |
| live_c2 | +0.290 ± 0.062 [−0.13 \| +0.15 \| +0.29 \| +1.17] | 7.655 vs 7.946 |

- The change of +0.20 is real: about 2.8 se, combined se 0.072.
- The frozen gate had read c1 as −0.047. That was an artifact of freezing.
- The latch is worth −0.026 ± 0.015 in c1 and −0.022 ± 0.005 in c2: no value.
- Mengli c5 on text: +0.413 ± 0.049 (writer-gain A/B, ×1 arm).

**Item 3. Speed profile** (`scratch/profile_mainline_tick.sh`, 256 words, current working tree)
- 4.2 s per 32-word window (`forward_window`, cumulative). GPU utilisation is 27% on average, so the run is host-bound.
- Per mechanism, over 1,899 ticks:

| mechanism | function | cumulative time |
|---|---|---|
| writer (predictive) | `forward_with_state` | 2.0 s |
| carried compartment credit | `comp_step` (new, `fly_carried_credit.py`) | 1.67 s |
| dopamine | `dopamine_step` | 0.98 s |
| EG | `_eg_apply` | 0.70 s |
| act muscles | `ai_step` | 1.37 s |

- Gradient checkpointing pack hooks: 4.4 s cumulative over 267k calls.
- Hidden overhead: about 540k grad-mode switches.
- Speed-ups that keep the semantics:
  - batch the per-tick hooks;
  - avoid the grad-mode context churn in the checkpointed segments.

**Item 4. Taste pulse probe** (`taste_pulse_probe.py`, fixed weights). Extra Hz per cell over 40 ticks:

| taste | MN9 | PAM | PPL1 | MBON |
|---|---|---|---|---|
| sugar | **0.0** | +0.49 | +1.04 | +0.47 |
| bitter | +2.08 | −0.47 | +1.04 | +0.43 |
| water | 0 | +0.04 | 0 | −0.26 |
| high salt | 0 | +0.73 | +0.78 | +0.86 |

- **Negative result.** The innate taste lines measured on 10-13 (r2 rebirth smoke: sugar → MN9 +68 Hz, bitter → PPL1
  +45 Hz) are almost gone in the adult Mengli:
  - sugar no longer drives MN9;
  - bitter barely drives PPL1;
  - valence survives only as a sign: sugar raises PAM (+0.49), bitter lowers it (−0.47).
- **Probable cause.** The BPTT training on text reshaped the taste routes, which carried no gradient. EG plasticity
  pulls toward birth only with tau 4,096 updates. On 10-13 the decision was to gate births on the senses surviving
  ("sugar → MN9 ≥ +20 Hz at chunks 5 and 25"); this adult fails that gate.
- **Way out.**
  - Check when the routes died: d5b birth → c11 → Mengli c1 / c5, the same probe on each.
  - If training kills them, protect the innate taste synapses: pin them, as the connectome pins already do, or lean
    harder on the pull to birth.
  - This matters for the mother world: taste is the only built-in value signal for later acts.

## 2026-10-15 03:50 — m_main long run resumed (item 5): first living gate after the resume

The chain is `grow_mainline_live_wt.sh` (isolated worktree, commit 97b6654).

**Chunk 13.** It reached 53,248 words in 18 minutes.
- Fresh B NLL 6.996, AG 0.558.
- A-revisit 6.582.

**Gate c12 (living, paired by window, 1,024 words)**
- **(1) Brain contribution: +0.797 ± 0.058** [min +0.28 | median +0.79 | mean +0.797 | max +1.55].
  - Every window is positive.
  - Brain 6.986, noedges 7.783, unigram on the same words 7.676: the brain beats the unigram by 0.69.
- **(2) Surprise ~ phasic DA:** −0.011, next −0.006, DA cells −0.001; per 1,024 words [−0.035 | −0.029 | −0.028 | −0.018].
  Still about 0; dopamine does not follow surprise.
- **(L)** No cell latches.

**Reading.**
- Gate (1) is clearly positive and growing in the long run. It went +0.09 → +0.29 (m_olf c1 → c2), then +0.41
  (Mengli c5 on text), and now +0.80 at m_main c12.
- The causal order now stops at (2): surprise does not reach dopamine. The acts (3) depend on it, because the speaking
  and looking credits read dopamine as precision.
- (2) is the next mechanism question, together with the lost taste lines (item 4), since taste is the dopamine
  system's innate teacher.

## 2026-10-15 04:00 — (2) surprise → dopamine: theory before the next test

This draws on what was known before plus today's measurements.

**Known.**
- Surprise is strong at the senses: the writer innovation has rho 0.83.
- It roughly halves at each synapse along the olfactory path: receptors 0.52 → ALPN 0.26 → KC 0.13 → MBON 0.06 →
  DAN about 0 (m_main c6).
- Kenyon → dopamine-terminal release raised the variance of phasic DA, not its tie to surprise.
- Its habituation erased the release everywhere: the Kenyon code is a context code over about 3 words, so it never
  repeats word by word.

**New today.**
- The text arrives very weakly. A written cell fires about 0.02 per tick, roughly 4 Hz.
- The deep brain therefore receives the word mostly as a small change on top of its own background. The Kenyon and
  MBON responses to a word scale with the writer drive (dose–response).
- The dopamine cells do not respond to text at any gain, in PAM or PPL1. Their input is dominated by other lines,
  through state-dependent recurrent loops.
- The innate taste lines, the dopamine system's own teacher, are mostly lost in the adults.

**Hypothesis.** Surprise is diluted twice:
- a weak carrier at the input, at each written cell;
- averaging per layer, which halves it at each synapse.

Without a teacher, the dopamine cells' input weights are never shaped to read it, and BPTT on next-word CE gives
them no reason to. The fly's own answer is novelty DANs (PPL1-α'3/α'2), driven by MBON-α'3 and habituated by the
compartment memory. That needs a word code that repeats, which only a context of about 3 words provides.

**Test, zero-shot, when the GPU is free between chains.** On m_main c13, correlate surprise with DA at the writer gain
of the live run (×1) and at ×4.
- Does surprise survive further down the path when the carrier is stronger (rho per layer)?
- Does MBON-α'3 respond more to a context seen for the first time than to a repeated one (3-word contexts)?

If both hold, the route exists and only the senses' strength is missing. That leads to a newborn comparison, ×4 vs ×1
from birth (the adult cannot take ×4 suddenly, as the A/B showed). If MBON-α'3 does not discriminate, the
compartment memory does not habituate on contexts. That has to be fixed before any dopamine credit.

## 2026-10-15 04:45 — m_main living gates c13, c14

| gate | (1) brain contribution [min \| median \| mean \| max] | (2) surprise ~ phasic DA, same / next word | (2) per 1,024 words [min \| median \| mean \| max] | (L) latched cells, worth |
|---|---|---|---|---|
| c12 | +0.797 ± 0.058 [+0.28 \| +0.79 \| +0.797 \| +1.55] | −0.011 / −0.006 | [−0.035 \| −0.029 \| −0.028 \| −0.018] | 0, n/a |
| c13 | +0.679 ± 0.054 [+0.18 \| +0.66 \| +0.679 \| +1.35] | +0.017 / +0.022 | [−0.014 \| +0.023 \| +0.022 \| +0.056] | 618, **+0.033 ± 0.010** [−0.07 \| +0.02 \| +0.033 \| +0.12] |
| c14 | +0.854 ± 0.056 [+0.27 \| +0.83 \| +0.854 \| +1.48] | +0.034 / +0.031 | [−0.005 \| +0.023 \| +0.030 \| +0.079] | 684, +0.031 ± 0.015 [−0.07 \| +0.02 \| +0.031 \| +0.34] |

- **(1)** Every window is positive at all three gates. The brain beats the unigram by 0.58–0.74 nats.
- **(2)** It creeps up, −0.011 → +0.017 → +0.034, and at c14 three of the four 1,024-word blocks are positive. It is
  small; it has to keep rising before it counts.
- **(L)** The latch is worth something for the first time:
  - +0.033 ± 0.010 at c13 (3.3 se) and +0.031 ± 0.015 at c14 (2 se), with 618–684 cells latched;
  - earlier it was −0.02 in m_olf and −0.019 in m_main c1.
- **Fresh B / AG.** Fresh B NLL went 6.996 (c13) → 6.943 (c14) → 7.181 (c15). The first gate with fresh B / AG is due
  at c15.

## 2026-10-15 05:35 — m_main gate c15: the first fresh B / AG of the resumed run, and its taste probe

**(1) Brain contribution: +0.715 ± 0.038** [+0.34 | +0.76 | +0.715 | +1.12]. Every window is positive.

**AG** (both arms learning):

| | brain | noedges | brain advantage |
|---|---|---|---|
| fresh B NLL | 6.947 | 7.214 | **0.27** |
| A-revisit | 6.407 | 7.318 | **0.91** |

The brain generalises to unseen text and keeps far more of what it read.

**(2) Surprise ~ DA** −0.015 (next −0.000), per 1,024 words [−0.054 | −0.019 | −0.015 | +0.034]. The rise at
c13/c14 was noise around 0; (2) is still not met.

**(L) Latch** +0.021 ± 0.014 (666 cells). Small, positive three gates in a row.

**Taste** (taste_pulse_probe, m_main c15), extra Hz per cell over 40 ticks:

| | MN9 | PAM | PPL1 |
|---|---|---|---|
| sugar | −2.08 | −0.21 | −1.30 |
| bitter | 0 | +0.41 | +2.6 |
| water | −2.08 | | |
| salt | 0 | | |

- The innate lines are lost in m_main as in Mengli (birth smoke 10-13: sugar → MN9 +68, bitter → PPL1 +45).
- Text training erases the taste routes. They carry no gradient, and the pull to birth (tau 4,096) is too weak.
- Only a faint bitter → PPL1 remains.
- This is the most likely structural reason (2) stays at 0: the dopamine system has lost its teacher.

**Way out (one change, controlled).** Protect the innate taste synapses, receptor → … → MN9 / PAM / PPL1 routes,
from BPTT and EG, as the connectome pins protect fixed edges. Keep them at birth weights, or give them a strong pull
to birth.
- **Check zero-shot first:** restore only those edges to their birth weights in the c15 copy. If the taste lines come
  back without hurting text, the protection is the fix. If not, the loss lies elsewhere, in thresholds or the class
  physiology learned by evidence plasticity.
- This is a GPU test, to run between chunks or after the run.

**Taste-restore probe, queued (`ZS_RESTORE_TASTE`, world_eval).** On c17, before the chain goes on, the taste pulse
probe is run four ways:
- as is;
- with the output edges of the taste receptors restored to birth weights (k = 1);
- with the edges of every cell within 1 hop of the receptors also restored (k = 2);
- within 2 hops (k = 3).

Edges pinned at 0 by the connectome rules stay 0. Gate c16 is skipped for this pause.

**Predictions.**
- If k = 1 already brings back sugar → MN9 (birth ≥ +20 Hz), the loss sits at the receptors' own synapses. A narrow
  protection fixes it.
- If only k = 2–3 does, it sits in the shared interneurons (GNG). Protecting them also touches the text route, which
  needs a text-cost check.
- If no k does, the loss is in the learned physiology: thresholds and class parameters from evidence plasticity, not
  in the weights.

**Taste-restore results (c17), extra Hz per cell:**

| restored | sugar → MN9 | bitter → PPL1 | bitter → PAM |
|---|---|---|---|
| none | 0 | +2.1 | +0.41 |
| receptor outputs (8,166 edges) | 0 | +3.4 | +0.40 |
| + 1 hop (237,660 edges) | 0 | +3.7 | +2.12 |
| + 2 hops (7.9 M edges) | 0 | +5.0 | +0.36 |

Restoring the weights, even across most of the brain, does not bring sugar → MN9 back. So the loss is not in the
weights, or it was never a loss.

**Comparability.** The 10-13 numbers (+68 / +45 Hz) came from the smoke newborn with **100 Hz Poisson receptors**.
`taste_pulse_probe.py` drives the receptors with a tonic current (gain × θ, 20 ticks) from the reading state. The two
are not comparable. The same probe runs next on m_main's own birth checkpoint, at the next chunk boundary (PAUSE). Only
then is "lost in training" a finding.

## 2026-10-15 06:25 — Correction: the taste lines were not lost in training

The same `taste_pulse_probe.py` on m_main's **birth** checkpoint gives sugar → MN9 −2.1 Hz and bitter → PPL1
+3.4 Hz. That is the same as the adults: c15 gave −2.1 / +2.6, c17 0 / +2.1, Mengli c5 0 / +1.0.

So the "loss" was an artifact of comparing different stimuli. The 10-13 +68 / +45 Hz came from 100 Hz Poisson
receptors on the smoke newborn. This probe's tonic pulse from the reading state hardly moves MN9 or PPL1 even at
birth.

Withdrawn:
- "innate taste lines erased by text training";
- the protection proposal;
- the hypothesis that this is why (2) stays at 0.

The restore probe agrees: restoring the birth weights changed nothing.

Open: the two taste probes disagree at birth (Poisson vs tonic drive). Before the taste routes are used as a dopamine
teacher, settle the stimulus with a dose–response: the tonic gain, and Poisson rates of 50, 100 and 200 Hz, on birth
and adult alike.

**Gate c17 (living):**
- (1) brain contribution +0.511 ± 0.050 [+0.07 | +0.49 | +0.511 | +1.24]. Every window is positive.
- (2) +0.025 (next +0.033), per 1,024 words [−0.016 | +0.011 | +0.011 | +0.037].
- (L) −0.005 ± 0.015, with 710 cells latched.

## 2026-10-15 06:40 — (3) Do the acts time themselves? (Mengli, `live_act_report.py`, the living stream's own trace)

m_main has no acts. Mengli has the motor acts: look, think (waiting) and speak/commit. The data is her `ai_trace.jsonl`
in the mother world, 4,832 words (chunks 1–5). The ranges are per 1,024-word block, [min | median | mean | max].

| measure | Mengli, mother world | the same trace from 0 (incl. the d5b text history, 18,208 words) |
|---|---|---|
| answered fraction | [0.03 \| 0.45 \| 0.37 \| 0.54] | [0.04 \| 0.84 \| 0.71 \| 0.99] |
| expected latency (ticks) | [2.5 \| 2.6 \| 3.1 \| 4.8] | [1.1 \| 1.7 \| 2.0 \| 4.7] |
| latency entropy (timing diversity) | [1.04 \| 1.17 \| 1.25 \| 1.62] | [0.19 \| 0.78 \| 0.83 \| 1.58] |
| rho(latency, gain from waiting) | **[+0.30 \| +0.36 \| +0.40 \| +0.58]** | [−0.02 \| +0.15 \| +0.21 \| +0.59] |
| rho(latency, best tick of the read) | **[+0.38 \| +0.52 \| +0.57 \| +0.87]** | [+0.03 \| +0.32 \| +0.32 \| +0.87] |
| rho(latency, the word's CE) | [−0.55 \| −0.14 \| −0.23 \| −0.08] | [−0.51 \| −0.02 \| −0.06 \| +0.05] |

**Reading.**
- In the mother world the commit timing follows the value of waiting, in every block. It commits later when waiting
  would improve the read, and close to the read's best tick (rho +0.57).
- Its timing is more diverse than on text: entropy 1.25 against 0.83.
- This is the first sign in the causal order that an act reads the brain's own uncertainty and times itself.
- It is correlational, from the living stream. The causal test is the act's living ablation: a fixed-timing arm on
  the same conversation.
- The negative rho with the word's CE means harder words get committed earlier. Together with the falling answered
  fraction (Mengli's silence), this fits the speaking-credit bias found tonight: yielding pays.

**Correction to the reading above, from the report's own controls.**

| measure | [min \| median \| mean \| max] |
|---|---|
| rho(latency, gain from waiting), within the same interval length k | [+0.04 \| +0.07 \| +0.07 \| +0.11] |
| rho(latency, best tick), within k | [−0.03 \| +0.04 \| +0.02 \| +0.06] |
| timing value over a null timing | [−0.010 \| +0.005 \| +0.025 \| +0.102] nats |

- The strong raw correlations (+0.40, +0.57) come mostly from interval length. A longer interval means both a later
  commit and a later best tick.
- Within the same interval the act follows the value of waiting only weakly (+0.07) and the best tick barely at all
  (+0.02).
- Its timing is worth +0.025 nats over a null timing on average; the median is only +0.005.
- **(3) is therefore not shown.** The acts are diverse (entropy 1.25), but they do not yet time themselves by
  uncertainty beyond the interval.
- This is consistent with the order: without (2), dopamine as precision carries no surprise, so the acts have nothing
  to read.

## 2026-10-15 06:50 — (2) zero-shot tests queued for the next chunk boundary

The tests run on m_main's latest checkpoint, at writer gain ×1 and ×4 (`ZS_WRITER_GAIN`, world_eval):
- `surprise_path_probe.py`: Spearman of surprisal with each layer along the olfactory path, down to phasic DA;
- `word_novelty_probe.py`: do new words drive MBON-α'3 / PPL1-α'3, and does the response habituate over 8
  presentations?

**Predictions.**
- **×1** looks like c6: receptors about 0.5 → KC about 0.1 → DAN about 0.
- **×4** raises rho from the ALPN to the MBONs, because the carrier is stronger (the dose–response raised KC and MBON
  reach ×4–7). If rho at the dopamine cells stays about 0 even then, the break lies between the MBONs and the DANs:
  their input weights or the other lines dominate. Strength is not the bottleneck.
- **Novelty.** If MBON-α'3 responds to new words and habituates while PPL1-α'3 does not follow, the MBON → DAN link
  is the missing piece. If MBON-α'3 does not respond at all, the novelty code is missing upstream.

## 2026-10-15 07:00 — (2) results: where surprise stops (m_main c19), and gate c18

**Gate c18 (living):**
- (1) +0.382 ± 0.069 [−0.27 | +0.31 | +0.382 | +1.40]. A harder stretch of text: unigram 8.01.
- (2) **+0.053** (next +0.066), per 1,024 words [−0.018 | +0.016 | +0.026 | +0.089]. The highest so far, still small.
- (L) **+0.047 ± 0.017** with 631 latched cells (2.8 se).

**Surprise along the path.** Spearman with surprisal, [min | median | mean | max] over 4 places:

| layer | writer ×1 | writer ×4 |
|---|---|---|
| writer innovation | [0.81 \| 0.87 \| 0.86 \| 0.90] | [0.83 \| 0.88 \| 0.88 \| 0.92] |
| olfactory receptors | [0.48 \| 0.53 \| 0.54 \| 0.61] | [0.42 \| 0.47 \| 0.47 \| 0.52] |
| ALPN | [0.23 \| 0.28 \| 0.29 \| 0.38] | **[0.33 \| 0.39 \| 0.39 \| 0.45]** |
| Kenyon | [0.08 \| 0.14 \| 0.14 \| 0.21] | **[0.14 \| 0.16 \| 0.17 \| 0.24]** |
| MBON | [0.04 \| 0.08 \| 0.08 \| 0.13] | [0.04 \| 0.08 \| 0.09 \| 0.13] |
| DAN spikes | [−0.04 \| −0.02 \| −0.02 \| −0.00] | [−0.04 \| +0.01 \| +0.01 \| +0.08] |
| DAN v/θ | [−0.04 \| −0.02 \| −0.02 \| +0.01] | [−0.10 \| −0.08 \| −0.07 \| −0.01] |
| phasic DA | [−0.04 \| +0.02 \| +0.02 \| +0.07] | [−0.09 \| +0.02 \| +0.01 \| +0.08] |

- The prediction holds. A stronger carrier raises surprise in the ALPN (+0.10) and the Kenyon cells (+0.03).
- The MBONs stay at 0.08 and the dopamine cells at about 0.
- **The path breaks at Kenyon → MBON (0.17 → 0.09) and MBON → DAN (0.09 → 0), not in the senses' strength.**
- A ×4 writer is not the fix for (2).

**Word novelty (`word_novelty_probe.py`).** Six unseen words, each 8 times among text:
- MBON-α'3 (MBON16/17, 6 cells), PPL104 (α'3 dopamine, 2 cells) and phi on the α'3 edges are **exactly 0** for every
  word and every presentation, at ×1 and ×4.
- The read's CE on arrival is 11.5–13.3 nats, and it does not fall over the 8 presentations (slope −0.006).
- The novelty circuit does not respond at all. A response of exactly 0 most likely means these cells are silent, or
  their inputs never reach them.
- This could not be checked now: the static check of their inputs ran out of RAM beside the training. It is the next
  test at the next pause:
  - their baseline firing;
  - their Kenyon input share;
  - the MBON-α'3 → PPL104 edges.

**Cause and exit.**
- Surprise reaches the Kenyon cells but the MB output layer barely reads it, and the dopamine cells not at all.
- The fly's own route for novelty, KC → MBON-α'3 → PPL1-α'3, is silent in this model.
- **Next:**
  1. Find out why α'3 is silent: thresholds, inputs pinned by a rule, or class physiology driven to silence.
  2. Fix it as a mechanism.
  3. Then rerun this probe: novelty response plus habituation.
- Only after that does dopamine-as-precision have a surprise to carry for the acts (3).

## 2026-10-15 07:35 — Why the α'3 novelty circuit is silent (`alpha3_probe.py`, m_main c20, 512 words of text)

| | MBON-α'3 (6) | PPL104 (2) | all MBONs (97) | all PPL1 (16) |
|---|---|---|---|---|
| firing per tick | **0** | **0** | 0.012 | 0.004 |
| median v/θ, [min \| median \| mean \| max] over words | [−0.69 \| −0.53 \| −0.53 \| −0.36] | [−1.18 \| −1.00 \| −0.99 \| −0.68] | [−0.37 \| −0.24 \| −0.23 \| −0.03] | [−1.09 \| −0.99 \| −0.98 \| −0.75] |
| excitatory static edges (pinned at 0) | 1,833 (1,755) | 1,555 (1,139) | 73,955 (67,399) | 25,897 (21,471) |
| excitatory weight: static + compartment | 0.9 + 18.1 | 4.0 | 163 + 970 | 84 |
| inhibitory weight | 11.5 | 10.5 | 707 | 186 |

- φ on the α'3 Kenyon edges is [−0.81 \| −0.38 \| 0], so their efficacy is depressed by 38% at the median.
- There are 11 excitatory MBON-α'3 → PPL104 edges.

**Reading.**
- Both α'3 cells sit deep below rest: MBON-α'3 at −0.53 θ, PPL104 at −1.0 θ.
- Inhibition dominates, and their Kenyon input has been depressed by the compartment memory: φ median −0.38.
- PPL1 as a whole sits at −1.0 θ as well and hardly fires (0.004 per tick). The PPL1 cluster, the punishment and
  novelty dopamine, is held under. That fits the flat (2) at every gate.
- The novelty route exists in the wiring: KC → MBON-α'3 through 1,614 compartment edges, and MBON-α'3 → PPL104
  through 11 edges. The cells are driven out of their operating range.

**Candidate causes to separate (zero-shot, one at a time, at the next pause):**
1. φ depression. Set φ to 0 on the α'3 edges and see whether MBON-α'3 fires.
2. Tonic inhibition. Is the inhibition of MBON-α'3 / PPL1 from APL or a learned class? Split its weight by presynaptic
   type.
3. Learned physiology. Compare PPL1's threshold and leak against birth.
4. The same probe on birth.pt. If α'3 is silent there too, it is a birth (connectome or rule) problem. If not, training
   drove it there.

**Also fixed.** The gate c19 noedges arm crashed: `UnboundLocalError 'evidence'` on an update skipped for non-finite
values. Fixed in the worktree (40536aa); gate c19 is lost.

## 2026-10-15 08:20 — α'3 / PPL1 cause separated: training silenced the PPL1 dopamine cells

`alpha3_probe.py` was run three ways: on m_main c21 as is, with φ on the α'3 edges reset to 0, and on m_main's birth.
The figures are firing per tick and the median v/θ.

| | MBON-α'3 | PPL104 (α'3 DA) | all PPL1 | all MBONs |
|---|---|---|---|---|
| birth | 0 / −0.07 | **0.017 / +0.60** | **0.017 / +0.77** | 0.0001 / +0.01 |
| c21 | 0 / −0.55 | 0 / −1.06 | 0.002 / −1.02 | 0.014 / −0.20 |
| c21 with φ_α'3 = 0 | 0 / −0.50 | 0 / −1.06 | 0.002 / −1.02 | 0.014 / −0.19 |

**Inhibitory weight onto them, top presynaptic types:**

| target | birth | c21 |
|---|---|---|
| PPL1 | SMP056 12.5, CRE027 11.4, DPM 9.7 | **FB4G 18.6**, SMP116 16.8, CRE027 15.7, SMP084 13.7, **CRE052 12.6**, LHMB1 12.4 |
| PPL104 | mALB1 1.7, … (all < 1.8) | **CRE052 4.5** |

**Findings.**
1. **Training silenced the PPL1 dopamine cells.**
   - At birth they are tonically active (+0.77 θ); PPL104 is above threshold and fires.
   - By c21 they sit at −1.0 θ, held down by inhibition that grew in training: new or strengthened FB4G, CRE052,
     SMP116, SMP084.
   - BPTT on next-word CE found it useful to shut the punishment/novelty dopamine off. Its phasic modulation of
     thresholds and plasticity presumably cost prediction in the short window.
   - This is a likely structural reason (2) stays at 0: the cells that should carry novelty and surprise are clamped.
2. **φ is not the cause.** Resetting the α'3 compartment memory changes nothing.
3. **MBON-α'3 is silent even at birth** (−0.07 θ, undriven). That is a separate birth-level gap, which the
   KC-reach problem above explains.

**Way out.** The fly's own principle: dopamine cells are not free parameters of the prediction objective.
- Option 1: protect the inhibitory synapses onto the DANs from BPTT, keeping them at birth weights, or keep them under
  a strong pull to birth. One change, controlled.
- Option 2, better: let them learn only through their own local rule, not through the CE gradient.
- Check zero-shot first: restore only the inhibitory inputs onto the DANs to birth weights on c21. Then re-run
  `alpha3_probe` and `surprise_path_probe`:
  - do PPL1 fire again?
  - does surprise reach phasic DA?
  - what does it cost the text?

The same check matters most for Mengli: her BPTT in the mother world faces the same pressure.

**Also.**
- The gate c20 noedges arm failed with a NaN score (the JSON write refused NaN). This needs a look. The c20 brain
  contribution is lost.
- m_main is asked to pause at the next chunk boundary. It has done its job:
  - (1) is established;
  - the PPL1 silencing is found.

  The active parts (3) can only be tested in the mother world, so the GPU goes to Mengli.
