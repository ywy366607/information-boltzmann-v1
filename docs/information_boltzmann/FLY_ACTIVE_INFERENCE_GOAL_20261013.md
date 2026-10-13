# GOAL: unified perception and action (active inference) — 2026-10-13

**One sentence.** In a text stream whose pace the world sets, the fly brain learns by itself *when to look, when to
think, when to speak*: perception and output are both acts, selected under one active-inference objective; the rhythm
is not imposed, it emerges from feedback.

## World
- Words arrive at the world's own pace (memoryless intervals); the world never waits for the brain.
- Every arriving word is scored, whether or not the brain was looking.

## Two acts (one act head: learned read of all output-cell membranes, leaky integration to a bound)
1. **Perception — eyes open / closed**: the precision (gain) of the text input. Closed: less interference, room to
   think; but words arriving meanwhile are missed — context is lost and later predictions pay (a real cost; unlike the
   rejected go neuron, which made the world wait).
2. **Output — commit now / keep thinking**: speak the expectation standing on the output surface.

## Objective (evolution's fitness for BPTT; nothing injected into the brain)
- Perception: VFE = latent prediction error (JEPA) + SIGReg (an exact VFE: Arnez & Gomez-Villa 2026, arXiv 2607.13612).
- Acts: EFE = pragmatic (accuracy of committed outputs) + epistemic (count-based novelty; the adult core already
  encodes familiarity, and the mushroom-body alpha'3 loop gives within-life familiarity).

## The feedback (user)
| Behaviour | Consequence | Corrected by |
|---|---|---|
| eyes always closed | no new information, context lost, predictions worsen | loss + curiosity |
| eyes always open under a fast world | noise mixes in, thinking and output suffer | output accuracy |
| always thinking, never output | predictions never tested, uncertainty accumulates, no pragmatic value | pragmatic |
| always output, never thinking | low accuracy; monotonous output carries no information | pragmatic + epistemic |

## Acceptance (falsifiable)
1. No dark room: eyes open most of the time at a normal pace.
2. Close eyes to think: under a fast world, brief eye closures around hard items, and output quality better than with
   eyes forced open.
3. Output latency scales with difficulty; output never blocks input; no fixed 14 ticks.
4. No noisy TV: on unlearnable noise, curiosity habituates and the eyes close.
5. Language cost <= 0.05 nats.
6. Ablations: no epistemic term -> dark room or never closing; no pragmatic term -> never output.

## Milestones
- M0 infrastructure: segmented checkpoints for long words (memory); the act head generalized to eye + output acts.
- M1 eyes only: perception VFE + epistemic value.
- M2 output only: questions in the text stream; answer now vs keep thinking.
- M3 both: inverted-U rhythms, eyes-closed thinking, ablations.
- M4 dialogue world: a caregiver ("mother") sets questions and feedback.

## Folded in / archived
Bookshelf -> perception choice; self-generated text -> output; odour world -> test bench for mushroom-body memory only.
Carried over: connectome v2 fixes, relative plasticity, the fixed compartment-memory rule, the act head, survival-weighted
reading, the information-value idea (-> epistemic term).

## M1 predictions (written before the run, 2026-10-13)
Setup: the adult (connectome v2 fixes), world pace mu ~ log-uniform [2, 16] per window, eyes only: objective =
survival-weighted CE + 0.01 SIGReg on the read latents - 1.0 x epistemic (mean over onsets of e x novelty). Control:
the same with the eyes forced open. 3 chunks of 4,096 words each. Probe: scripts/ib/fly_interface/eyes_probe.py.
- P1 (acceptance 1): the eyes stay open. From the definition: with no output act, missing a word can only remove
  context from every later prediction, and the epistemic term rewards looking; both gradients push e up. Expect open
  fraction >= 0.99 at every pace and mean e rising from 0.95.
- P2: no closing with difficulty. The eye is decided before the word is written, from the state; it has no reason to
  close on hard words. At the fastest pace (mu = 2) a small dip on frequent, low-novelty words is the only possible
  closing (overlap with the previous word's processing), and it should be weak (|rho| < 0.1).
- P3 (acceptance 5): language CE within 0.05 nats of the forced-open control, in the same weights (own vs open) and
  between the arms.
- P4 (acceptance 2): NOT met in M1 by construction. Closing to think needs a consequence of thinking that undisturbed
  time improves: an output whose quality is scored when it is committed (M2/M3). If the eyes nevertheless close on hard
  items in M1, that is a finding to explain (interference between overlapping words), not the acceptance.
Revisions before launch (colleague review): the epistemic value is the information gain of seeing the word with gain e,
0.5 log(1 + e / (1 + n)) (concave in e, fading with familiarity), not e x novelty; closed eyes are exactly a quiet tick
(the sensory adaptation also sees e x the word: before, a missed word leaked through adaptation); an ablation arm with
no epistemic term (eyes_noepi) tests whether the language objective alone keeps the eyes open; the CE comparison is
paired per word (same text, same intervals) with a bootstrap interval; the forced-open arm's untrained head gives the
baseline correlations (the eye and the CE are read from the same membranes).

## M2 predictions (written before the run, 2026-10-13)
Question = every arriving word: "what comes next?".  The answer is the read standing when the output commits (value
-CE); the question closes when the next word arrives (eyes open in M2), unanswered = -log V.  The world never waits, so
output cannot block input.  Arm `out`: as M1's forced-open arm plus the pragmatic term (weight 1); M1's forced-open arm
is the no-pragmatic ablation (its output channel gets no gradient).
Optimal stopping from the definition: keep thinking while E[CE gain of one more tick] > hazard x (log V - CE_t);
hazard = 1/mu.  Thinking gains on the adult are ~0.1-0.15 nats/tick early (CE by tick 7.91, 7.74, 7.61, 7.54, ...).
- P1: answered fraction high (>= 0.9 at mu >= 4); never blocking input (by construction).
- P2 (acceptance 3): latency grows with difficulty: an easy word (CE ~3) risks ~(log V - 3)/mu ~ 0.5 nats per tick of
  waiting at mu = 16, so it is answered at once; a hard word (CE ~10) risks ~0.05, so it waits.  rho(latency, CE at
  tick 1) > 0.2 in fixed16; the tick-1 hazard of the easiest quartile > the hardest quartile's.
- P3: latency shrinks with the world's pace (the pace is visible in the brain's own input rhythm).
- P4: no pragmatic term -> almost no output (the init rate, ~1/300 ticks): acceptance 6b.
- P5: language CE within 0.05 nats of the no-pragmatic arm.

## M3 design and predictions (written before the run, 2026-10-13)
Why M3 needs one more rule (first principles, from the definition of a question): if a question closes when its
answer arrives, closing the eyes can never help an answer -- the deadline is the world's.  But a question is only
answered *by being told*: it closes when the brain next LOOKS.  Closed eyes keep the question open; the brain can keep
thinking (the read's CE falls ~0.1 nats per tick over the first ticks) at the price of the missed word's context.
This is gaze aversion while thinking (people look away to answer hard questions; Glenberg et al. 1998).  Rules
(fly_active_inference.pragmatic_value): question i opens at onset i; it closes at the first onset where the eyes are
open, or after `ai_depth` = 2 closed onsets; a commit answers the oldest open question, valued -CE of the read then
against that question's word; closing unanswered = -log V.  Eyes are a 0/1 decision (straight-through, like the output
commit) so a question kept open cannot be answered by peeking at a half-seen word.
Gradient check of the bias: when the eyes are open, the straight-through estimate of the value of closing uses reads
that did see the next word (pessimistic); it matters only for a question still unanswered at the arrival, where it
still says "closing beats a sure miss" (-CE of a stale read > -log V).  So the direction is right where it counts.
Noisy TV: with probability 0.25 per 32-word window, 8 words drawn from a fixed 16-token alphabet.
Arms (3 chunks of 4,096 words each, mu ~ log-uniform [2, 16]): full (epistemic 1, pragmatic 1); open (eyes forced
open, otherwise full); passive (eyes forced open, pragmatic 0 -- the language reference); noepi (epistemic 0); noprag
(pragmatic 0).  Probe: scripts/ib/fly_interface/act_probe.py (fixed16, geo 16/8/4/2, noisy geo 4/2; own vs open).
- P1 (acceptance 1): eyes open >= 90% of onsets at mu >= 8.
- P2 (acceptance 2): closures concentrate where a question is still unanswered at the next arrival, i.e. at fast pace
  and on hard questions: closed fraction (unanswered) >> closed fraction (answered); higher at mu = 2 than at 16.
  Answer value own > open (same weights) at mu = 2, 4, and full > open arm.
- P3 (acceptance 3): latency grows with difficulty (rho > 0.2 in fixed16), shorter at fast pace; never blocks input.
- P4 (acceptance 4): on noise the information gain fades (the 16 words are seen hundreds of times); the eyes close more
  on noise than on text once inside a noise stretch (late noise words).  Risk: closing has no value on noise unless
  the noise disturbs the context -- if eyes stay open on noise, the cause is that seeing noise costs nothing.
- P5 (acceptance 5): language CE (clean text, geo mu) of full within 0.05 nats of passive.
- P6 (acceptance 6): noprag never outputs (init commit rate) and never closes (no value in keeping a question);
  noepi: no epistemic push to look -- closes more than full; whether it reaches the dark room depends on how much the
  language CE alone values context.
Revisions before launch (colleague review of M3):
- In a single-thread text stream the next word IS the answer to the open question, so looking is always informative
  and closing can never pay (it trades a fresh question for a stale one and loses context).  Closing can pay only when
  the input is irrelevant to the current question.  So the noisy TV is a background speaker: noise words ask nothing
  and close nothing; a question asked just before a noise stretch stays open across it (its answer is the main
  speaker's next word); looking at the noise only disturbs the pending answer and the main context.  One world
  feature, two acceptance tests: closing to think (2) and the noisy TV (4).  Stretches never touch a window's edge.
- The eye is a sampled policy in training (Bernoulli(p), as active inference samples policies: real closures with real
  outcomes) and starts at p = 0.9 (twice the gradient of 0.95); the probe samples with a fixed seed.
- New pace and noise schedules every chunk (seeded by the lifetime so far).
- Arm depth0 (questions close at the next main onset whatever the eyes do) attributes closures to the kept-open rule.
- Expected weak points, stated now: in clean text P2 may fail (closing has no value there); on noise the eyes may
  stay open because predicting the 16-word alphabet requires knowing a stretch is running (looking tells it).
Revision during M2 (first launch stopped after 23 windows): the output integrator went to crossing the bound on every
tick, and the straight-through slope of sigmoid((g - bound)/0.25) is ~0 there, so the habit was absorbing (it could
never learn to wait again).  Both acts now pass gradients with the soft crossing's slope at the bound, everywhere
(output: (g - bound)/(4 x 0.25); eyes: 0.25 G).  Theory note: for this adult, answering at once is near-optimal on
average (CE 7.91 -> 7.74 from tick 1 to 2 vs a 2.9-nat miss risk per 1/mu: at mu = 16 the value of committing at tick
1, 2, 3 is -7.91, -7.93, -8.00), so difficulty scaling can only come from the words where thinking pays or where any
answer is worse than silence (CE > log V); the head must be able to keep learning to find them.

## M3 world revised: sustained noisy input (theory, written before the run, 2026-10-13)
M2 (pulse input) converged to answering every word at tick 1.  Cause, from the definitions: deliberation time has
value only if (a) the uncertainty is reducible (epistemic, not aleatoric), (b) something is there to work with --
evidence still arriving, or a recurrence still refining, (c) the gain differs by item in a way the state can read,
and for closing the eyes (d) input interferes with the computation in progress.  Next-word difficulty is mostly
aleatoric (a word with CE 10 is unpredictable, not hard to compute) and a pulse delivers all its evidence at once;
the read is settled by tick 2.  So with pulse input acceptance 2 and 3 are unreachable by ANY policy (to be checked
by the hindsight oracle in the M2 probe: gain of the best tick over tick 1, and whether the best tick tracks
difficulty).  From the fly itself: the mushroom body accumulates evidence over sustained noisy odour input, and harder
discriminations take longer (DasGupta et al. 2014, FoxP in Kenyon cells sets the accumulation time).  Real senses
deliver a word over time, not as one sample.  The principled exit is therefore the input format, not a task:
- Sustained noisy input: a word is heard on every tick of its interval as a noisy sample, embedding + sigma *
  |embedding| / sqrt(d) * xi_t, with the clarity drawn per word, sigma ~ U(0, sigma_max).  Sensory adaptation
  (tau ~ 20 ticks) still high-passes the drive.  Eyes act on every tick (a closed tick loses one sample).
- Question rule, per tick (it reduces to the onset rule for pulse input): a question closes the first tick its
  answer word is RECEIVED (written with the eyes open), or after `depth` further main words; within a tick the eye
  acts before the write and the commit after the read.
Predictions (sustained noisy input, arms as before):
- P2' (acceptance 3): answer latency grows with the current word's noise (rho(latency, sigma) > 0.2 in fixed16);
  the hindsight-best tick grows with sigma too (if not, the brain cannot accumulate: say so).
- P3' (acceptance 2): at fast pace a noisy word may still be unidentified when the next word starts; a brief closure
  at the next word's first tick(s) costs only a sample or two of that word (it is sustained) and keeps the question
  open: closures concentrate on open questions about noisy words at fast pace; answer value own > forced open.
- P4' (acceptance 4): inside background stretches the eyes close (looking only disturbs the open question and the
  main context; the 16-word alphabet's information gain fades).
- The rest as in the M3 predictions above.  Zero-shot check first (no training): CE by tick for clear vs noisy words
  at fixed16 must fall with ticks for noisy words, or accumulation is absent and the run is pointless.
Pilot (passive arm, 1 chunk, sustained noisy input), before the M3 results: no accumulation was grown -- CE by tick at
fixed16 is flat for noisy words (8.71 -> 8.68) and for clear ones (8.06 -> 8.00), while the same brain under pulse
input improves 7.76 -> 7.45.  So sustained input does not add evidence for this brain; it disturbs the settling that
is its thinking.  Revised prediction for M3 (written before its results): the rational eye policy is "glance, then
close to think": look at a word's first tick(s) and close for the rest of its interval (a clear word is fully given by
a glance; the remaining ticks become pulse-like settling, worth up to ~0.3 nats).  Signatures: eye open fraction high
at word onsets, low within words; perception and answer CE better than the forced-open arm; at fast pace fewer
within-word closures (no time to think).  If the eyes stay open within words, the straight-through gradient through
the drive did not find the settling benefit (cause to check: the benefit only shows ticks later, through BPTT).

## Behaviour diversity first (user, 2026-10-13, after M1-M3)
User: before adaptivity, behaviour must be diverse (LLMs learn adaptive thinking length because GRPO samples many
rollouts and compares them); sources must follow first principles and the fly connectome; no "virtual" neurons unless
necessary.  Correction: the adult's 6-8 ticks per word was found best by the CTM loss before flux admission existed,
so thinking has NLL value; my worlds made waiting worthless (memoryless pace, never pausing).
Diagnosis of M1-M3: every act was a deterministic threshold of an external integrator trained by straight-through
gradients along ONE trajectory: the alternative (closing, waiting) was never tried, only linearly extrapolated, and the
policies collapsed (eyes 100% open, answers at tick 1).
Measured (M2 checkpoint, motor_activity_probe.py): 2,333 output cells, ~10% spike per tick, 865 never fire; only 7% of
the spike pattern's variance is word-locked (membranes 51%) -- the motor output is mostly intrinsic fluctuation.
Design (to review, then implement):
1. Acts are the brain's own output cells: muscle activation m = filter(motor spikes) (a body property, tau ~ 8 ticks;
   the spikes keep the brain's atan surrogate); act logits = W m + b (W learned: which cells drive which act); the
   act is SAMPLED: look ~ Bernoulli(sigmoid(eye logit)) at write ticks, commit ~ Bernoulli(sigmoid(out logit)) every
   tick.  No integrator, no bound, no straight-through.
2. Learning from compared outcomes (group-relative): commits do not change the brain, so K sampled commit sequences
   on one brain trajectory are scored and compared for free; eyes change the trajectory, so K real branches of the
   window (same text, same world) with independently sampled eyes.  Advantage = J_k - mean_k J; policy loss =
   -sum_k A_k sum_t log p(act_t^k); perception CE trains the brain on every branch.
3. World with heavy-tailed pace and pauses: interval K ~ discrete Pareto (P(K >= k) = k^-alpha), so the hazard falls
   with elapsed time and waiting becomes cheap after a few ticks; the survival estimator becomes sum_{t<=K} h(t) CE_t
   with the hazard h(t) = P(K=t)/P(K>=t) (unbiased for the expected arrival CE).
4. Diversity measured before training: K branches from one state -> fraction of differing acts, per-word latency
   spread, inter-act interval tail (power law vs exponential), and whether the spread concentrates on hard items.
Second source, deferred until measured: probabilistic transmitter release (binomial over a terminal's release sites;
needs per-edge synapse counts and one extra sparse scatter per tick).
Review and user notes (2026-10-13): (1) commits do not change the brain, so no branches are needed: along ONE
trajectory every tick is scored as "if I answered now" (as CTM scores every tick) -- the soft hazards give the exact
expectation over all commit times (Rao-Blackwell; better than sampling); (2) eyes change the trajectory, but behaviour
sampled along time already contrasts closed and open stretches -- no branches unless that fails; (3) collapse is
prevented by active inference's own policy prior, q(pi) ~ E exp(-gamma G): 1/gamma KL(policy || habit) per decision,
the policy-level twin of SIGReg (the belief-level KL to an isotropic prior); (4) the value of not answering is the
listener's prior (unigram CE of the answer): an answer is worth the information it adds beyond what the listener
expects.  Zero-training oracle (wait_oracle.py, M1 forced-open brain, results/.../diversity/wait_oracle.json):
tick-1 CE 7.76 is WORSE than the unigram prior 7.66; by tick 3 it is 7.49.  With the prior as the miss value the best
common answer tick is 3-4 at every pace (geometric or heavy-tailed), 64-78% of words are best answered after tick 1,
and the hardest quartile wants to wait (97%, gain 0.74-1.17 nats): difficulty predicts the value of thinking.  With
-log V the best common tick is 1 (that was M2).  The pace's tail mattered little; the miss value decided it.
Acts from motor cells: the readout sees each muscle's standardized activation (calibrate_muscles: mean and spread on
128 words, spread floored at the median over cells that move); W starts at 0, so the acts start at their habits.

## D1: commits only, sampled from motor cells (predictions written before the run)
Pulse input, pace mu ~ log-uniform [2, 16], eyes forced open, miss = listener prior, readout learns (brain detached),
3 chunks.  Arms: d1_habit (1/gamma = 0.05, habit hazard 0.2) and d1_nohabit (1/gamma = 0: expected to collapse).
Probe: act_probe.py (fixed16 and geo; exact expectations under the hazards).
- Diversity: latency entropy per word stays >= 0.5 nats with the habit prior; without it the hazards drift to 0 / 1
  (entropy falls) -- the ablation that shows the prior is what keeps behaviour alive.
- Adaptivity: expected latency grows with difficulty (rho(expected latency, tick-1 CE) > 0.1; hardest quartile later
  than easiest), tracks the oracle's best tick, and the expected answer value beats answering at tick 1.
- Language: unchanged (brain detached): |survival CE - M1 forced-open| <= 0.02.
- Risk: only 7% of the motor spike variance is word-locked; if the readout cannot see difficulty, latency stays flat
  -> next step is to let the policy gradient reach the brain (act_brain_grad).

## Connectome audit: why the brain cannot accumulate (read from the model, no simulation; user: "look, don't test")
- Short-term plasticity was never learned (learn_stp False in the adult's whole life): every synapse of all 27
  classes has u0 = 0.25, tau_rec = 200 ticks, tau_fac = 100 -- the initial values.  Under sustained firing the
  resources settle at x* = 1/(1 + u r tau_rec) ~ 1/6 at r = 0.1/tick: sustained signals are cut ~6x, only changes pass.
  The whole brain is a transient detector -- it settles by tick 2 after a pulse, and sustained input is interference,
  not evidence (the sustained-input pilot).  Real synapses differ by type (depressing and facilitating).
- Kenyon cells had no physiology of their own: all 4,064 sit in cb_intrinsic with 32,160 central-brain neurons and
  share its membrane time constant (43 ticks), threshold, adaptation and STP.  The alpha/beta core (488 cells; FoxP,
  the evidence accumulators of DasGupta et al. 2014 / Groschner et al. 2018) could not differ from anything.  The
  connectome records wiring, not channels: the per-neuron tau_m in the graph does not single them out either (40.6 vs
  39-50 for other subtypes).  A point-model coarse-graining artifact on the active route.
- Known: mushroom-body output reaches descending neurons with 0.07% of their input; the consequences of an answer do
  not enter the brain through any receptor.
Fixes (user: "全部改好"): --learn-stp (evolution sets each class's STP) and --kc-subtypes (alpha/beta core, alpha/beta
surface+middle, alpha/beta posterior, alpha'/beta', gamma get their own class parameters, initialised from
cb_intrinsic so the dynamics are unchanged at the split; the optimizer's moments carried from the class they left).

## D2: does evolution grow accumulation once it may? (predictions written before the run)
Sustained noisy input (each word heard every tick, sigma ~ U(0, 2)), pace mu ~ log-uniform [2, 16], commits from motor
cells as in D1 (listener-prior miss, habit prior, readout learns), eyes open.  Arms: d2_fixed (--learn-stp
--kc-subtypes) and d2_control (same world, no fixes).  3 chunks each.
- Accumulation: in d2_fixed the CE of noisy words falls with ticks at fixed16 (best mean gain on the noisy quartile
  >= 0.15 nats; the pilot without fixes had 0.03); d2_control stays flat.
- The fixes are used: STP moves away from the uniform depression in some classes (u0 or tau_rec), and the alpha/beta
  core's time constants or STP diverge from the other Kenyon subtypes.
- Adaptivity: the commit's expected latency grows with the word's noise (rho(latency, sigma) > 0.1) in d2_fixed only.
- Language: d2_fixed no worse than d2_control on clean text (<= 0.05 nats).
If accumulation does not appear with the fixes, the next suspects are the sensory adaptation (tau 20 ticks, a
high-pass at the writer) and the weak MB -> DN route; say which before changing anything.

## Why the fly's own circuit is not used (static analyses, weights only; 2026-10-13)
- static_route_analysis.py (D1 weights vs birth): the EXIT works -- from the alpha/beta core the route runs Kenyon ->
  MBON / DAN -> convergence (SMP/SIP/CRE) -> central complex / LAL -> descending neurons, as in the fly, and training
  strengthened it.  The ENTRY starved: text spreads into the Kenyon cells with 5-14% of its mass at birth but 0.5% ->
  0.01% after passive-reading evolution; 0% of the text's influence on the outputs passes through them.
- rarity_decode_probe.py (CV R^2): the core knows the current word's rarity at tick 1 (0.82); the motor side only by
  tick 4 (membranes 0.60, muscles 0.32).  The NEXT word's difficulty is unreadable anywhere (0.07, core included): it
  is the next word's rarity (r = 0.97), aleatoric.  The value of thinking longer is partly readable from the muscles
  at tick 4 (0.22), not at tick 1.
- word_novelty_probe.py: new words evoke no dopamine (PPL104, PAM, PPL1 ~ 0), no alpha'3 novelty response, no
  habituation, no change of phi -- the mushroom body's novelty system is silent for text.
- taste_route_check.py: the innate taste -> dopamine lines barely exist in the model (sweet receptors -> PAM 0.1% of
  the spread at birth, ~0.01% trained; no sweet / bitter specificity at the dopamine cells), while olfactory receptors
  -> Kenyon cells are strong (15-34%).  Caveat: the "sweet" / "bitter" receptor sets of the odour world (15 / 39 cells)
  are not backed by receptor annotations (receptorType only marks ppk23 / ppk25 / IR52b); of 1,428 gustatory cells
  only ~220 enter through the labial nerve (the labellum's sugar and bitter receptors).
Reading: passive next-word evolution rewired the brain around the read (shortcuts), starved the mushroom body of input
and erased the weak innate outcome lines.  The circuit for "think, then act on what was learned" is anatomically there
but has no input and no teacher.

## Consequence loop (design, to review with the user before code)
When: a question's outcome is revealed when its answer word is RECEIVED (the question closes with the eyes open).
What: the information the answer added over the listener's prior, gain = prior CE(answer) - answer CE (nats); an
unanswered question gives nothing (silence has no taste).
How: through receptors only (no dopamine electrode): gain > 0 drives the sugar receptors, gain < 0 the bitter ones, for
4 ticks, amplitude proportional to |gain| (clipped at 2 nats); the drive is computed from the realized commits word by
word and passed into the next word's checkpoint segment (replayed exactly on the recompute); no gradient through the
world.
Who learns: within life the dopamine-gated compartment memory (Kenyon -> MBON) and the dopamine-scaled plasticity;
across the lineage BPTT as before.
Preconditions, in order: (1) the right receptors -- identify the labellar sugar (Gr5a/Gr64) and bitter (Gr66a) GRN
types from the connectome's own annotations (neuPrint / FlyWire matches), not a guess; (2) the taste -> dopamine line
must carry the outcome: if it is still ~0.1% with the right receptors, audit it like the Kenyon cells (normalisation
over thousands of partners at the dopamine cells, sensory-terminal artifacts) and fix what blocks it; (3) then close
the loop and check: dopamine responds to good vs bad answers, phi changes, the text -> Kenyon entry strengthens over
chunks, and commit timing starts to depend on state.
Open question for the adult: the trained weights erased routes that the newborn has (entry 10x, taste lines 10x).  If
(2) cannot restore them in the adult, the plasticity criterion says: rebirth with the corrected connectome, acts and the
consequence loop present from the start.

## D3: the reward's gradient reaches the brain (user, 2026-10-13; predictions written before the run)
User: if an adult cannot learn what it did not learn before maturity, our implementation lost plasticity -- biology
keeps learning; GRPO grows adaptive thinking length because thinking longer improves the reward and the reward trains
the whole network.  D1 had sampling and compared outcomes but kept the brain detached from the policy gradient (only the
readout learned), so nothing asked the brain to make "am I done thinking?" visible to the motor side (readable only
at tick 4, R^2 0.22).  The relative plasticity (2% of the birth weight per step) lets erased routes regrow if a gradient
asks.  D3 = D1 + --act-brain-grad: the gradient of the expected answer value w.r.t. the commit hazards flows through
the motor spikes (atan surrogate) into the brain; the answers' values stay detached (no wishful beliefs); language CE
trains the predictions as before.  One arm, 3 chunks, pulse world, compared with d1_habit.
- The expected latency starts to track the value of waiting (rho(expected latency, oracle tick) > 0.1 at fixed16; D1:
  -0.03) and hard / easy words separate.
- The value of waiting becomes readable earlier from the muscles (R^2 at tick 1 > 0.05; D1: 0.01).
- Answer value above D1's at fixed16; language cost <= 0.05 nats vs D1.
Revision before D3 (user): the habit KL is the analogue of GRPO's KL to the reference policy, which long-reasoning
RL drops (DAPO, Dr. GRPO); D1 showed it inert (d1_habit and d1_nohabit indistinguishable, no collapse without it), so
D3 runs without it (1/gamma = 0).  The relative plasticity stays: it scales each synapse's step to 2% of its birth
weight (step geometry, no pull back to birth -- the trained weights drifted far, e.g. the mushroom-body entry fell
10-1000x) and was introduced because absolute Adam steps had saturated / silenced cells (surrogate gradients ~0).
If behaviour collapses once the brain gets the gradient (GRPO's entropy collapse), that is the evidence for an
entropy-keeping term, added then.

## D3b: the fly's own signals for diversity and value (user: "add them all; use what the fly has and learned itself")
Measured first: D1's waiting is not heavy-tailed (inter-commit intervals from its own hazards: CV 1.01, q99 29 ticks,
Hill exponent over the top 5% 3.6, vs 0.91 / 25 / 4.1 for a geometric of equal mean).  Sampling alone gives one
geometric-like clock; nothing in the policy varies slowly.
1. Policy precision encoded by dopamine (Friston et al. 2014; q(pi) ~ E exp(-gamma G)): z = ln E + gamma (W m + b),
   ln E the habit (now a frozen constant, not scaled; the learned bias is part of the value), gamma = (d + eps) /
   (d_ref + eps), d = the population mean spike of the 392 dopamine cells low-passed on the brain's own slow dopamine
   clock (sigmoid(dopamine_logit_slow) = 0.9982, ~556 ticks), d_ref its mean on text (calibrated with the muscles).
   The population tone, not the dopamine delivered at the output cells: those get 0.5% of all dopamine (553 of 2333
   receive any).  gamma is read, never trained: it is one gain redundant with |W|; a gradient into the dopamine cells
   could only learn to burst into determinism and would open the three-factor plasticity windows brain-wide (review).
   No new cell, no new parameter.  Arm d3_precision = D3 + this; d3_braingrad is its control.
   Predictions: if the dopamine tone fluctuates (gamma CV over words > 0.2), the waiting becomes a mixture of clocks
   -- Hill exponent < 3 and CV > 1.1 (commit_tail.py), latency entropy per word not lower than the control; if gamma's
   CV < 0.1 the arm equals the control and the cause is a flat dopamine tone (exit: which dopamine types fluctuate,
   and whether the online three-factor rule / no-homeostasis flatten it).  Language cost <= 0.05 vs the control;
   report it split by gamma tercile (high tonic dopamine also opens plasticity windows).
2. Epistemic value of looking from the brain's own belief, not a counter outside it: the Bayesian surprise (Itti &
   Baldi) of each word, KL(prediction after the word's interval || prediction just before it), evaluated under the
   beliefs, weighted by the eye gain, per word as the pragmatic value (--epistemic-belief).  The entropy of the
   prediction was rejected in review: it counts unlearnable noise as information and would reward watching the noisy
   TV (goal 4); a word the brain has learned to ignore moves nothing and is worth ~0.  Used in eye experiments only
   (D3 keeps the eyes open).
3. The fly's familiarity signals.  Long-term familiarity (word rarity) is the brain's own: decodable from the core at
   tick 1 (R^2 0.82), at the muscles only by tick 4 (0.32); D3's brain gradient is what can route it earlier (no
   electrode).  Within-life familiarity (alpha'3 novelty loop) is unusable now: PPL104 does not respond to new words
   (0.0), MBON-alpha'3 at noise level, new words' CE drops 0.017 nats per presentation.  Cause, static: the text
   reaching alpha'/beta' Kenyon cells fell ~70x during passive training (hop 3: 1.1% -> 0.015%), MBON-alpha'3 ~30x,
   PPL104 ~40x, and the MBON16 -> PPL104 feedback left PPL104's top inputs.  Exit: D2 gives the Kenyon subtypes their
   own physiology; D3 adds the gradient; word_novelty_probe runs after each, to see whether the loop recovers.
4. The epistemic value of acts (answering to learn) needs the consequence loop (an answer's outcome observed through
   receptors); it stays with that design.

## Credit beyond the 32-word window (user: "should be O(1), not only BPTT"; review 2026-10-13)
The window is ~215 ticks (64-512).  Within-word answer timing (<= 16 ticks) is inside it: D1's missing difficulty
adaptation is not a window effect.  What loses credit at the boundary and matters for look / think / speak: the
compartment memory phi (a look or a thought pays when the content recurs, often > 32 words later -- where learning
progress and the epistemic value live), STP recovery (the later cost of a thinking burst, tau_rec ~200), and questions
carried over the boundary (only with closed eyes / depth > 0: at depth 0 with open eyes the window-end miss is exact).
The slow dopamine tone is detached by design; ALIF, muscles and writer adaptation are much shorter than the window.
The O(1) learner already exists and is on in every D run: the compartment memory phi on the 61,210 Kenyon -> MBON
synapses, written online by phasic dopamine x a Kenyon trace (FLY_ONLINE_DA_LEARNING_DESIGN.md sections 9-11; its
predecessor at Kenyon terminals measured live - frozen revisit savings ~0.002 nats in l1 / b2: nothing, because passive
reading had no consequences and the mushroom body was unused).  So the missing piece is not a learner but what it
learns from: the text entry to the Kenyon cells (down ~70x) and dopamine that carries something (outcome, novelty).
Lessons from the retired whole-brain e-prop (approximate credit on 165k neurons plateaued): keep online credit local
and exact on few targets, as a residual next to BPTT, never a competitor on the same parameters; long reach comes from
bootstrapping (a critic), not from longer traces.
Order: (0) measure phi's own contribution in the D-series (memory_revisit.py: live vs frozen phi, slow weights fixed;
D1, D2 arms and D3 arms) -- the yardstick for everything below.  (1) Carried questions with exact readout credit,
when eyes or depth come back: the question's P = alive * S passes to the next window as a leaf, g = dJ_next/dP, the
old window's readout gets -(lambda/n)(g - miss) dP/dtheta (one step stale).  (2) A readout eligibility longer than
the window, e <- l e + (a - h)[m, 1] (the acts are sampled: unbiased score function), Delta W ~ delta e with delta read
from the dopamine cells; the critic is the mushroom body's own MBON -> DAN loop (occ(fast) - occ(slow) is already a
temporal difference), read per compartment after measuring each loop's sign and gain -- only once dopamine carries
the outcome (consequence loop) or novelty (alpha'3 recovers); until then delta ~ 0.
Review also found a real bug, fixed: motor-mode binary eyes took the straight-through slope on the first muscle, not
the eye's logit, so the eye readout never learned (M3-style eye runs only; D1-D3 keep the eyes open).

### Carried credit, implemented (user: "wherever credit needs more than the window, add the O(1) rule too")
information_boltzmann/core/fly_carried_credit.py, --carried-credit.  Every slow state that outlives the window and has
trained parameters gets a forward sensitivity S = ds/dtheta, advanced each tick by the state's own local recursion
(spikes and arriving dopamine held fixed): the compartment memory phi and the Kenyon trace (comp_eta, the forgetting
and trace clocks, and the dopamine clocks through the phasic dopamine that writes phi: 5 x 61,210 + 4,064 numbers),
the fast / slow dopamine concentrations (3 x N), STP's u and x (5 x N, per-class parameters, only with --learn-stp).
The window's opening states are leaves; after the backward, G = dL/ds_0 and the parameters get G . S_0 on top of their
in-window BPTT gradient.  Test: with spikes and arrivals fixed, BPTT through two windows equals window-2 BPTT + G . S_0
to 1e-4 for all eight parameters (CPU); a GPU test runs it in the stream.  Off: nothing changes (no fields, no leaves).
D3 gets the arm d3_credit (= control + --carried-credit).  Predictions: the carried credit is not small next to the
in-window credit for the slow clocks -- dopamine_logit_slow (tau ~556 > window) carried / in-window norm > 0.5 -- and
for the memory's own parameters it is the larger part whenever the memory is written at all; but the memory is barely
written (text -> Kenyon entry down ~70x), so behaviour and language equal the control's (|diff| < 0.02 nats), and
memory_revisit live - frozen stays < 0.01 nats.  What it is for: once the entry and dopamine carry something, the
parameters that decide what the memory keeps learn from consequences arriving long after the window.

## D2b: the physiology needs its own step (found during D2, before its end)
After two D2 chunks the STP parameters had moved at most 0.016 (mean 0.005) in log / logit space -- tau_rec 200 ->
~203.  Cause, arithmetic: Adam moves a parameter ~lr per step; the class physiology lives in log / logit space, so
the shared lr 2e-4 is a 0.02% relative change per step, ~128 steps per chunk -> <= 2.6% per chunk even with a
consistent sign (noisy gradients cancel most of it).  The synapses take 2% of their birth weight per step (relative
plasticity), 100x more; the read-bank gains already needed 50x for the same reason.  So D2 cannot answer whether
evolution grows accumulation: a step-budget null.  D2b continues both D2 lineages for 3 chunks with
--lr-physiology 5e-3 (0.5% per step; not the synapses' 2%, because one class parameter moves thousands of cells at
once).  Predictions: tau_rec / u0 of the fixed arm move by > 0.1 in log / logit space and differ between Kenyon
subtypes; the noisy-word CE falls with ticks more in the fixed arm (ce_by_tick_noisy slope); if the parameters move
but nothing changes in behaviour, the cause is elsewhere (the read, or the noise level); if thresholds run away
(silence / saturation), 5e-3 is too large for log_threshold and it needs its own smaller rate.
Also: the same arithmetic holds for every log-space clock in D1-D3 (dopamine, memory, membrane): none of them could
evolve within these runs at 2e-4.

### Pruned before running (2026-10-13 12:10, look before testing)
- d3_precision not trained.  gamma_tone_probe.py on D1 (2,000 words, zero-shot): the dopamine population's spike is
  white (raw CV 0.67 per tick, autocorrelation ~7 ticks), so on the brain's own slow clock (~557 ticks) gamma stays in
  [0.991, 1.008], CV 0.24% over words, 0.008% over 512-tick blocks; with D1's commit value (sd 0.46) it would move the
  hazard by sd 0.0002.  The prediction's null branch, before training.  Cause: the population tone carries no slow
  state; the exit is per dopamine type (which compartments, if any, have slow structure), not this readout.
- d3_credit not trained now: at the shared 2e-4 the clocks it gives credit to cannot move (D2b's arithmetic) and the
  memory is barely written; it runs once --lr-physiology and the Kenyon entry are in place.
- D2 chunk 3 stopped (a step-budget null); D2b continues both lineages from their latest checkpoints.

### Correction: the Kenyon entry is not starved; the share measure misled (static, 2026-10-13 12:20)
The "text -> Kenyon entry fell ~70x" above is a SHARE of the linearly propagated mass, and all excitatory weights grew
4.4x in training (sum 27,134 -> 120,006), so every share of a route that grew less looks starved.  In absolute
weights (D1 checkpoint): text-driven cells -> ALPN x1.21, ORN -> ALPN x1.20, ALPN -> KC x5.5, ALPN -> LH x4.5,
non-PN -> KC x7.3, KC -> KC x0.15 (95% pinned at 0: the axo-axonal fix).  A Kenyon cell's median excitatory input is
0.156 at birth, 0.189 trained, and its PN share rose from 26% to 89%.  So the entry did not wither; the PN input to
the mushroom body is a smaller part of a brain that grew louder elsewhere (hop-1 share on ALPN 7.0% -> 1.5%).
Consequences: (a) the alpha'3 / PPL104 silence is not explained by the entry; the cause must be found in activity --
rarity_decode_probe now also reads the Kenyon cells (membranes, spikes since onset, fraction active per word) in
D3's probes and on D1; (b) "rebirth to restore the entry" is off the table on this evidence; (c) static shares are
only comparable within one weight set -- across birth / trained, compare absolute weights or simulate.

## Consequence loop, precondition 1 done: the receptors (neuPrint + literature, 2026-10-13)
neuPrint male-cns v1.0 carries no taste modality for the gustatory neurons (receptorType only ppk23 / ppk25 /
IR52b on leg and wing bristles; 274 have a flywireType).  The modality comes from Vicente et al. 2025 (bioRxiv
10.1101/2025.08.25.671814), who typed every GRN in this very maleCNS volume and matched receptor-GAL4 projections:
sugar = LB3b, LB3c (Gr64f; LB3b also Ir56b, low salt) and dorsal_tpGRN (Gr5a); bitter = LB1a-d (Gr33a); LB1e = Ir94e
(mild aversion, glutamate); water = LB3a (ppk28); high salt = LB3d (Ir7c / ppk23); LB2, LB4: no receptor match.
The first taste_route_check used guessed types (dorsal_tpGRN, a Gr5a sugar cell, counted as bitter) and shares; its
"~0.1% at birth" is void.  Redone in absolute mass (D1 weights, per unit receptor activity, mass on the dopamine
cells at hop 2 / 3): bitter 8.5e-4 / 1.0e-2, sugar 2.6e-4 / 4.1e-3, water 2.1e-4 / 2.9e-3, ORN 1.1e-4 / 1.6e-3,
text 6e-5 / 1.0e-3: the taste lines reach dopamine better per unit than odour or text.  Valence is weakly separated
(top target PAM01 for sugar and bitter alike; PAM/PPL1 at hop 3: sugar 2.9, bitter 2.3).
Precondition 2 (activity): taste_pulse_probe.py drives each receptor set (1.5 x its thresholds, 20 ticks) from a
reading state and counts the extra spikes of PAM, PPL1, each dopamine type, MBONs, Kenyon cells and the change of
phi, vs a control fork; queued on D1 and D3 once the GPU is free.  Predictions: the receptors fire (else the gain is
too small: raise it); PAM responds to sugar more than PPL1 does and PPL1 to bitter more than PAM does (if both go to
PAM01 alike, the valence lines are not separated in this brain and the loop would teach the same thing for good and
bad answers -- then the cause is looked for in the SEZ relays before closing the loop).

### Valence is separated; the "weak separation" was the unsigned measure (static, signed, 2026-10-13 12:25)
The route checks propagated |w| for both excitatory and inhibitory edges, so an inhibited target counted as driven.
Signed (A = g_e w_e - g_i w_i), per unit receptor activity, at hop 2 (the direct relays, GNG / DNg cells):
            sugar -> PAM      sugar -> PPL1     bitter -> PAM     bitter -> PPL1
  birth     +1.2e-5           -1.7e-5           +6.2e-6           +4.9e-6
  trained   +1.4e-4           -1.7e-5           -1.3e-4           +9.0e-5
Sugar excites PAM and inhibits PPL1 (PPL106); bitter excites PPL1 (and PPL2) and, after training, inhibits PAM01 /
PAM08 -- the fly's valence logic (reward PAM, punishment PPL1), already present and strengthened by training (sugar ->
PAM x12, bitter -> PPL1 x18).  At hops 3-4 both turn net negative on all dopamine cells (long paths are dominated by
inhibition in the linear picture; thresholds decide in activity).  So the loop can teach different things for good
and bad answers; the activity check (taste_pulse_probe) is still the precondition.  Method lesson: static routes must
be signed and absolute (two wrong conclusions today came from shares and from |w|).

### Queue cut to measurements (user: "don't queue piles of training; what can be read from code or measured?")
- Read from checkpoints (CPU): the compartment memory IS written -- D1 phi sd 0.20, 45% of the 61k synapses
  |phi| > 0.01, max 0.82; 30% of Kenyon cells with a live trace (D2: sd 0.24-0.26, 39%).  The earlier "barely
  written" was wrong; only the rule's own parameters are stuck (comp_eta -2.00 -> -1.98: the step size).  Whether the
  memory is USED is memory_revisit's question (live vs frozen phi), a measurement.
- D2b training dropped; stp_sweep_probe.py measures zero-shot whether ANY STP setting (tau_rec x4 / x0.25, u0 0.1 /
  0.5, tau_fac x4) makes a noisy word's CE fall with ticks on d2_fixed.  Only if one does has evolution a direction
  worth a training run.
- Dropped: D2 act probes (the sweep's baseline gives the noisy CE-by-tick), d3_precision, d3_credit.
- Kept: D3 control (mainline, half done), its probes, D1 Kenyon probe, taste pulse probes on D1 and D3.

### D3 control stopped, D3 full instead (12:23)
The running D3 control had none of today's fixes in its configuration (no STP learning, no Kenyon subtypes, the
physiology at the shared 2e-4): only synapses could change, so a null would not say whether the brain gradient fails
or the timing clocks were frozen.  Stopped after chunk 1.  Order now: zero-shot measurements (STP sweep on d2_fixed,
memory revisit, taste pulses and the Kenyon probe on D1), then ONE arm d3_full = D3 + --learn-stp --kc-subtypes
--lr-physiology 5e-3 --carried-credit (smoked first, start + resume), compared with D1.  Predictions as in D3, plus:
the STP / threshold / memory-clock parameters move > 0.1 in log space within 3 chunks without runaway (firing rates
of the read cells within 2x of D1's); carried credit for the dopamine and memory clocks is >= half the in-window one.
