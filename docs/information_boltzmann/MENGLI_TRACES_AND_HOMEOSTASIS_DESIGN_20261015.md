# Mengli: consequences through the brain's own traces, and responsiveness kept alive (design, 2026-10-15)

This design follows the user and GPT's direction. The short window keeps doing what fits the machine:
- immediate prediction, through 32-word BPTT;
- delayed learning, through local eligibility traces carried across windows (finite state) together with later
  dopamine;
- responsiveness, through a continuously running local homeostasis, so the short-window optimisation cannot
  permanently silence the learning circuits.

The 64-tick hard cutoff of the speaking return goes away.

## What we measured that asks for this

- **m_main (OWT, BPTT only), c12 → c21.**
  - The PPL1 dopamine cells went from +0.77 θ at birth to −1.0 θ; PPL104 stopped firing.
  - The silencing is spread over several mechanisms:
    - the pacemaker drive (`intrinsic_kappa`, 439 cells) fell from a median of 1.21 to 0.39;
    - inhibition onto the DANs grew (FB4G, CRE052, SMP116, SMP084);
    - the class physiology changed.
  - Restoring either the inhibition or the drive alone does not bring PPL1 back: −0.55 θ and −0.92 θ.
  - Surprise never reached phasic DA at any gate.
- **Mengli c5 (mother world).** PPL1 still fires (0.024 per tick) and PPL104 too (0.037 per tick). The same pressure
  is present in her BPTT, so the protection should come before the damage.
- **Speaking credit.** In every chunk silence earned more than speech (chunks 1–3). The reply's credit lands on the last
  yield, and the return reaches only the act readout.

## Mechanism A — responsiveness homeostasis (per cell, O(1), outside the gradient)

**Fly basis.** Intrinsic homeostatic plasticity. Cells regulate their own excitability towards a set point on a slow
clock, as in Drosophila motoneurons, the Kenyon cells (Apostolopoulou & Lin 2020) and the clock neurons. Dopamine
neurons pacemake from their own conductances; that is not a free parameter of a prediction objective.

**State.** Per cell, a slow average of its own membrane relative to threshold, m = EMA(v/θ), with time constant τ_h.

**Set point.** Each cell's own m at birth (measured once over the first chunk of life, then fixed): the individual's
operating point, not a hand-set rate.

**Update, every tick, no gradient.** The drive b moves by η (m* − m), with b clamped to a band.
- Population bursts appeared under the 10-12 regime homeostasis, which regulated every neuron's threshold quantile.
- Here the scope is narrower: only the modulatory and pacemaker cells (the 439 intrinsic cells: PAM, PPL, OA, 5-HT,
  DPM), plus the MBONs if a probe shows them pinned.
- BPTT keeps its gradient on everything else. `intrinsic_kappa` leaves the BPTT parameter list; the homeostat owns it.

**Prediction (falsifiable).**
- (a) After 8k words in the mother world, PPL1 v/θ stays within ±0.3 of birth. A control arm without A drifts down
  like m_main.
- (b) The surprise → phasic DA correlation is higher with A than without. If it is not, the dopamine cells are alive
  but read nothing, and B is the missing part.
- (c) Normal-reply NLL is no worse than control (paired by window, ±0.02 nats).

## Mechanism B — consequences arrive as dopamine onto the brain's own traces

**Fly basis.** The three-factor rule already in the brain:
- Kenyon activity leaves an eligibility trace (`comp_trace`, its own learnable time constant, carried across windows
  in O(1));
- dopamine arriving later converts trace × DA into a change of the Kenyon → MBON synapse (φ);
- MBONs steer the premotor and output cells that the acts read.

That is how a fly links an odour to a consequence seconds later. There is no fixed horizon: the trace decays on its
own clock.

**What is the consequence for speaking?** The mother answers contingently, and her normal reply teaches the child
something.
- In the fly, social and sensory consequences reach PAM / PPL1 through the senses.
- Here the mother's voice already enters Johnston's organ.
- The missing link is that her answer, as a consequence of the child's act, reaches dopamine.

**The proposal.**
- A normal mother reply's learning progress (known after the window's update, as now) is delivered at the start of the
  next window as a phasic dopamine pulse on the reward cells (PAM). The sign carries the value: progress drives
  them, loss slightly inhibits them.
- The pulse lasts a few ticks, at most 1 × rheobase, applied as an external drive.
- It then works only through the existing traces: the compartment rule and the carried dopamine credit. No new
  learning rule.
- The act readout's own delayed credit stays, but becomes a continuous trace: an exponential with the trace's time
  constant, carried across windows, with no 64-tick cutoff.

**Why this beats a scalar "speak more" bonus.**
- Nothing tells the brain to speak.
- The brain learns which of its own states (Kenyon codes, MBON outputs) preceded a teaching reply. Only through them
  does it learn when speaking pays.
- It also gives dopamine a meaning beyond the stream, which (2) has lacked.

**Prediction (falsifiable).**
- (d) φ changes on the Kenyon edges active in the ticks before her own speech that drew a normal reply, more than on
  edges active before silence.
- (e) The speaking hazard rises after states that historically drew teaching replies (correlation of hazard with the
  trace-weighted past return > 0 within interval length).
- (f) The silent-turn fraction stops rising (chunks 1–5 went 26% → 100%), without any forced speaking.
- If (d) holds but (e) does not, the MBON → output route that the acts read is too weak. That points to the
  MBON → premotor coupling, not to the reward.

## Order and controls

One change at a time, each against the lineage's own control fork on the same conversation seed:
1. A alone (a responsiveness guard; it should not change behaviour quickly).
2. B on top of A.

Each runs for an 8k-word chunk. The probes after each chunk:
- `alpha3_probe` (DA cells alive);
- `surprise_path_probe` (surprise per layer);
- `mengli_probe` F/K (familiarity, Kenyon/DA ablation);
- `live_act_report` (timing within interval);
- the cell activity audit (latching).

All of it goes to the dashboard.

## Speed

- A is one EMA and one add per pacemaker cell per tick: about 439 elements, negligible.
- B is a few-tick external drive on about 300 cells, once per window.
- The continuous act trace replaces the per-decision pending list (a list of tensors, O(decisions)) with one decaying
  vector per muscle: O(1).

## Revisions after the code review (2026-10-15) and implementation

**A — pacemaker homeostasis (`--pacemaker-homeostasis`, `attach_pacemaker_homeostasis`).**
- The signal is each cell's own firing rate, counted every tick of the training pass. Mean v/θ was dropped: a firing
  cell's v/θ barely moves with its drive.
- κ is updated once per window, after the update and the epistemic replay. The next window's live, training and
  replay passes therefore share one κ.
- The set point is each cell's mean rate over the first 16 windows after switching on.
- The update is η = 0.05 × log(target / EMA rate), with the EMA over 16 windows; κ is clamped to [0, 3].
- κ leaves the BPTT optimizer; the resume coverage check is patched for it.
- Stats per window go to the score log (`homeo`).

**B — her answer as a consequence (`--mother-reward`, `attach_reply_reward`).**
- The pulse is given in the live pass at the first word of a normal mother reply that follows the child's speech, not
  at the next window's start. At the next window's start the Kenyon trace (τ about 50 ticks) would have decayed to
  about 2%.
- The pulse is 1 × rheobase on the 316 PAM cells for the first 4 ticks of that word's interval.
- It is keyed per word, like the voice: `forced['reply_onsets']` → `learner.last_reward`. It is applied identically
  in the live pass, the training pass and `replay_ce` (`speak_record['rewards']`).

**Calibration (`reward_pulse_probe.py`, Mengli c5 state, 20 places, 40 ticks).** Δφ is the change of the compartment
memory; "on traces" is its share on edges whose Kenyon trace was standing. Values are medians.

| level × rheobase | PAM extra spikes / cell / tick | phasic DA extra | Σ\|Δφ\| | share of Δφ on standing traces |
|---|---|---|---|---|
| 0.5 | 0.0027 | +0.029 | 14 | 0.97 |
| **1.0** | 0.0061 | +0.053 | 29 | 0.98 |
| 2.0 | 0.0101 | +0.095 | 59 | 0.98 |
| 4.0 | 0.0165 | +0.157 | 123 | 0.98 |

- The pulse works through the brain's own three-factor path, linearly in its level.
- About 98% of the written φ lands on the synapses whose Kenyon trace was standing at the answer. That is exactly the
  "which of my states preceded her answer" credit.
- 1.0 is the start level.
