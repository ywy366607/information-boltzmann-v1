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
