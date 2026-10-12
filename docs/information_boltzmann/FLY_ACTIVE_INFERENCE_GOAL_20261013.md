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
