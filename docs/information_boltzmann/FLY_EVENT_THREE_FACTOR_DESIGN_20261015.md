# Event-driven three-factor learning (design, 2026-10-15, for review)

**User:** "BPTT is biologically almost impossible; the brain's own rule should exploit sparseness." The user is right.
Biological three-factor learning is event driven and lazy:
- a synapse's eligibility is local and decays passively between events;
- weight change is the integral of the third factor times the eligibility, settled only when something happens (a
  presynaptic spike or a third-factor arrival). NEST's dopamine-modulated STDP synapse works this way.

Its cost scales with spikes, not synapses. About 1–2% of cells fire per tick here.

## What we know before designing

1. **Measurements:**
   - More frequent local updates learn much faster. On 1,024 identical OWT words, from Mengli's state, an 8-word BPTT
     window (128 updates) beats a 32-word window (32 updates) by 0.50 ± 0.03 nats per word overall and by 0.17 in the
     last 256 words, at equal wall time.
   - The profile (one 8-word window): CPU 1.69 s against GPU 0.57 s. About 29k kernel launches and 122 ms of syncs, so
     the run is launch-bound. Backward is about half of GPU time.
2. **The earlier online run (2026-10-04, `FLY_ONLINE_PLATEAU_20261004.md`) plateaued for three identified reasons:**
   - 99% of updates were below the weight's float32 ULP, so they were rounded away;
   - the dopamine cells were silent, so the third factor carried nothing;
   - a crude credit approximation with random low-rank feedback.
3. **Gate (2) is about 0:** surprise does not reach phasic dopamine in any lineage so far. A three-factor learner
   inherits whatever its third factor carries.

## The rule (per synapse i → j, only on synapses whose post or pre site receives dopamine; others keep the slow learner)

- **Eligibility:** e_ij jumps at a presynaptic spike by ψ_j(t) (post depolarisation, the surrogate width already used)
  and decays as exp(−Δt/τ_e) between events. It is stored per synapse with a last-update tick (lazy decay).
  - τ_e is learnable per class.
  - For the post-depolarisation factor, the cheap form is a per-post exponential trace of ψ_j read at pre-spike time.
- **Third factor:** the phasic dopamine already delivered per neuron (`dopamine[0] − dopamine[1]`, from 390 DANs with
  delays; compartment-specific by wiring). It is not one global error. The other modulators (OA, 5-HT) are a later
  addition.
- **Weight:**
  - Δw_ij = η_class · ∫ p(t) e_ij(t) dt, settled at the next presynaptic spike or every K ticks for synapses with a
    standing trace.
  - Accumulated in an fp32 residual buffer and written when the change is representable, keeping the remainder: no
    sub-ULP loss.
- **Cost:** O(spikes × out-degree) per tick for jumps and settlement, the same traversal as the forward transmission.
  State is one float plus one tick stamp per plastic synapse (about 25M × 8 B = 200 MB).

## The prerequisite: dopamine must carry something (gate 2)

The design depends on surprise or value reaching the DANs. Three fly-native candidates, to be tested zero-shot before
any learning run:
- (a) A prediction-error route: MBON → DAN feedback, which exists in the wiring, with the familiarity / novelty
  compartments driven (α'3 is open).
- (b) The reply-reward pulse (B) as value, already calibrated.
- (c) The brain's own Bayesian surprise as a modulatory input to the DANs. This is the risky one: it is an engineered
  route and needs a fly argument first.

## Transition

BPTT stays for now, as short windows (8 words): the developmental scaffold for fast states. The event three-factor
learner runs alongside, on the dopamine-receiving synapses. Its share of the learning is measured by the paired
ablation "learner off".

## Falsifiable predictions

1. **Cost:** event settlement adds less than 20% of a forward tick.
2. **Lossless:** less than 1% of the intended update is lost to rounding (measured against an fp64 shadow on a sample).
3. **Usefulness:** with the scaffold on, turning the three-factor learner off costs more than 0.02 nats per word after
   4k words. If it costs 0 while gate (2) is about 0, the learner is not the problem: the third factor is.
4. **Gate 2 first:** at least one of (a)–(c) gives a surprise → phasic-DA correlation of 0.2 or more zero-shot.
   Otherwise the three-factor learner would only re-sharpen what dopamine already says (nothing).

## Review 2 adopted, and step 1 measured (2026-10-15 18:00)

The review said not to start a new rule but to extend the existing Kenyon → MBON compartment memory (φ, presynaptic,
the fly's own site):
- timing-signed (Kenyon then dopamine depresses, dopamine then Kenyon potentiates, signs fixed, magnitudes learnable);
- φ separate from w;
- learner and informativeness of dopamine tested apart.

The review also flagged that the third factor is dense, so event machinery comes only later.

**`pairing_probe.py`.** An optogenetic-style pairing on Mengli's state: word X held 32 ticks, the compartment's
DANs driven for 8 ticks, 10 pairings, Z interleaved unpaired, then 256 words of conversation and a retest.
MBON11 / PPL101 (γ1pedc). PPL101 sits at −0.8 to −1.1 θ (the BPTT-silenced PPL1), so only 16× rheobase makes it
spike once (phasic DA at MBON11 0.42 → 1.3–1.7).

| rule | φ_X after pairing | after 256 words | φ_Z | backward pairing |
|---|---|---|---|---|
| existing (trace-gated forgetting, κ learned 0.02 → 0.036) | −0.187 | **−0.020 (gone)** | −0.127 (not specific) | −0.114 (wrong sign) |
| timing rule + no use-forgetting + tonic τ 8192 | −0.110 | −0.015 | −0.004 | **+0.050** |
| timing rule + no use-forgetting + tonic τ 65536 | −0.148 | **−0.106 (71% kept)** | ≈0 | +0.062 → +0.041 |

**Findings:**
- **BPTT shaped a short-sighted memory.** Its 32-word horizon cannot see what a 256-word memory is worth, so it raised
  the use-dependent forgetting rate.
- **Specificity, direction and persistence pass with the fly's rule.** `--comp-timing` / `attach_comp_timing`, plus
  active forgetting on a long clock.
- **Not yet passed: MBON11's output barely moves** (−7% against the control drift; the target was ≥ 20%). The MBON's
  drive is mostly not Kenyon input, and one pairing moves φ by about 0.015, where a fly pairing depresses strongly.
  - Next: MBON input composition (KC share) and the potentiation / depression learning-rate start.
  - Then the novelty loop (α'3).
- **Carried credit** does not yet include the potentiation path (its `keep` and its η row). That matters only once
  BPTT trains η_pot.

## Step 1 at birth (2026-10-15 evening): why a newborn's pairing wrote nothing

Same protocol on the birth individual (`_birth_probe`, MBON11 / PPL101). Up to 16× rheobase, all arms came out
identical. Three causes were found, each measured.

1. **Saturated receptor.**
   - The newborn's pacemaking DANs keep MBON11's dopamine at 0.3–0.6, against K = 0.0108 (50×).
   - The fixed-K occupancy then reads a 16× pulse (fast concentration 0.5 → 5.9) as a phasic of 0.02.
   - Fix (exists, now in the newborn plan): `--compartment-adapted`, K_j = max(slow_j, K). The 4× pulse then reads
     0.37, against a threshold of 0.30.
2. **Rate-normalised eligibility.**
   - e ← λe + (1 − λ)s gives one Kenyon spike 0.02, so a 16× pairing moved φ by 0.01.
   - Fix: `--comp-unit-traces` / `attach_comp_unit_traces`. e ← λe + (1 − λe)s: one spike lifts it to about 1, and
     it saturates at 1 (presynaptic Ca²⁺/cAMP). The dopamine trace works the same way.
   - Rates restart at −1 / +1 (learnable). With adapted receptors one tick changes φ by at most 1, so the (1 ± φ)
     bounds hold.
   - Carried credit covers it: the CPU test is parametrised on both forms and matches BPTT exactly.
3. **The k = 2 burst threshold sits near the occupancy ceiling.**
   - Adapted occupancy phasic is at most 0.5.
   - The probe's silence → word rhythm already gives 0.30 at word onset, and the threshold (mean + 2 sd) reached
     0.40.
   - More basically, the newborn's own dopamine at MBON11 during ordinary words is larger than any forced pulse:
     relative (f − s)/(s + K) mean 1.9, p99 27, max 66. A 16× PPL101 pulse peaks at 8.3, because the two PPL101 cells
     are refractory-limited and their spontaneous synchronous volleys already reach that rate.
   - A threshold at mean + 2 sd therefore blocks the most informative dopamine the newborn has.
   - Measured (k = 0.25, unit traces, adapted receptors, 16×, TESTS=3, κ_DAN unchanged): direction appears at once.
     φ_X after pairing: none −0.12, forward −0.41, backward +0.72. But after 256 words the unpaired arm itself
     drifted to −0.83, so the spontaneous dopamine's writes erase the pairing.
4. **The DANs' spontaneous regime is the root (zero-shot κ sweep for the 338 DANs, review 2026-10-15 night).**
   - Full sweep (zero-shot, birth, 32 ordinary words; MBON11 relative dopamine (f − s)/(s + K); PPL101 spontaneous
     rate; 4× and 16× pulse peaks):

     | κ_DAN | PPL101 spikes / cell / 100 ticks | ISI CV | spont. p99 | 4× peak | 16× peak | 4× / p99 |
     |---|---|---|---|---|---|---|
     | 0.85 | 0 | — | 0.31 | 102 | 104 | 330 |
     | 0.95 | 0 | — | 4.0 | 45 | 56 | 11 |
     | 1.0 | 0 | — | 5.9 | 38 | 49 | 6.5 |
     | 1.05 | 0 | — | 6.7 | 25 | 34 | 3.8 |
     | 1.1 | 0.10 | — | 10.3 | 10.6 | 15 | 1.0 |
     | 1.2 (birth) | 0.29 | 0.06–0.11 | 15.4 | 4.9 | 9.1 | 0.3 |

   - Correction: at κ 1.2, PPL101 does not fire near its maximum. It fires about once per 350 ticks, but almost
     perfectly regularly, and both cells fire together. The volleys arrive at an MBON whose slow dopamine has decayed
     in between, so in relative terms each one is as large as a burst. The "every 6 ticks" seen earlier was under the
     pulse.
   - The headroom for a burst appears below κ ≈ 1.1. The spontaneous signal at MBON11 then comes from the other DANs
     onto it (PPL102, PAM10/11, small weights).
   - κ 0.95 and 0.85: PPL101 fires **0** spikes during ordinary words. The words do not drive the DANs at birth, so
     the spontaneous signal goes to about 0 (relative p99 4.0 at 0.95, 0.31 at 0.85). A forced pulse then reads
     36–104. That is clean for a probe, but the dopamine carries nothing in life.
   - With `--pacemaker-homeostasis` the set point is the birth rate. A DAN silent at birth stays silent for life (the
     homeostat does not push a cell silent at its set point), which repeats the BPTT-silenced PPL1 of m_main.
   - So `--dan-kappa 0.5` (committed by another agent in 38f5399) is **not adopted**. It silences the third factor.
   - Way forward, falsifiable: an irregular low-rate regime between the two.
     - (a) κ about 1.0–1.05: PPL101 rate below 1/3 of the pacemaking rate, ISI CV above 0.5 in the stream.
     - (b) Channel noise on the pacemakers: the irregularity comes from the cell, not from a hand-set rate.
     - (c) The DANs' input drive (gate 2): words must reach the DANs at all.
     - Pass: spontaneous relative p99 below the 4× pulse peak, the unpaired arm's φ drift under 0.1 per 256 words,
       and forward/backward signs held at retest.

**Another agent's probe table (38f5399, κ_DAN 0.5, 16×), re-read.** It does not show what its text claimed:

| arm | pre φ_X | post φ_X | post φ_Z | retest φ_X | MBON X |
|---|---|---|---|---|---|
| none | 0.1105 | −0.3753 | +0.2921 | 0.0937 | +1.2% |
| forward | 0.1105 | −0.3853 | +0.2323 | 0.1885 | +7.9% |
| backward | 0.1105 | +0.4013 | +0.0543 | −0.1285 | −11.9% |

- Forward ≈ none at post (−0.385 vs −0.375): **no forward depression**. At retest forward is *above* none.
- Backward potentiates at post, but is gone and reversed by the retest.
- The MBON moves against φ in both paired arms (φ up, output down; φ ≈ none, output up). Single-trial readouts drift
  about 30% in the control, so these MBON numbers are noise.
- The unpaired arm itself moves φ_X by −0.49 during the pairing block: large non-specific writes.

Also fixed:
- `pairing_probe`: `TESTS=n` averages n test trials and reports [min, median, mean, max]. One trial drifted 28% in the
  unpaired arm.
- `BWD_LEAD` for backward pairing, with the silence lengthened to fit it.
- `world_eval.load` now restores a newborn's compartment rules (tonic forgetting, timing, unit traces), MBON κ and time
  scale from its config. Before this, loading the newborn checkpoint failed the unexpected-keys assertion.

**New in the trainer:** `--time-scale k` (newborn; time constants / k, conductances and sensory drive × k; inherited on
resume), `--comp-unit-traces`.

## Step 1 at birth, continued (2026-10-15 night)

**κ_DAN 1.05, k 0.25, unit traces, 4×, TESTS=3** (φ_X; pre −0.258 in all arms):

| arm | post | retest (256 words) |
|---|---|---|
| none | +0.228 | −0.799 |
| forward | −0.576 | −0.740 |
| backward | +0.692 | −0.830 |

- The direction is right at post: forward −0.80 and backward +0.46 relative to none.
- But the control itself moved +0.49 during the block, and every arm ends at about −0.8 after conversation. The
  ordinary words' dopamine follows their Kenyon spikes (Kenyon then dopamine), so with a low threshold every word
  depresses its edges.
- Cause of the low threshold's failure: in occupancy terms the spontaneous threshold (m + 2 sd = 0.47) equals the 4×
  pulse's peak (0.466), because adapted occupancy is capped at 0.5. In contrast terms they are 4.0 and 25.

**Fix: `--comp-burst-relative` / `attach_comp_relative_burst`.**
- The burst is judged on the linear contrast (f − s)/(s + K), the same Weber signal `dopamine_adapted` gives every
  other target.
- The amount written stays the bounded occupancy, ramped over one sd above mean + k sd, so k is learnable.
- Carried credit is exact: the CPU test is parametrised on both trace forms, and the ramp's interior is exercised.
- Predictions (k 2, κ_DAN 1.05):
  - none drifts under 0.1 over the block and over 256 words;
  - forward below none and backward above none at post, signs kept at retest;
  - 2× still writes (pulse contrast about 25 against threshold about 4).

**Open for review: the pacemaker homeostat's set point for event-driven cells.**
- The homeostat moves κ by η (log target − log rate), with the target set to the cell's own rate over its first 16
  windows.
- A DAN near threshold (κ ≤ 1.05) is silent at birth, so its target is about 0.
- Every evoked burst then lowers its κ: for a cell evoked to 1 spike per 1000 ticks, each window steps by
  0.05 × (log(0.0005) − log(0.0015)) ≈ −0.055. After 16 windows κ is down by about 0.9. Use makes it less excitable,
  which is anti-responsive.
- Candidates, each falsifiable:
  - (a) Sense the cell's mean depolarisation v/θ instead of its rate (excitability homeostasis). Evoked spikes reset v,
    so use keeps the cell near threshold.
  - (b) Keep the rate sensor but give the DANs a set point no lower than their evoked floor.
  - (c) DANs out of the homeostat, with κ fixed at birth (out of the gradient as now). Their synapses can still drift
    under BPTT.
- Test: 4096 words with each option; the DANs' κ trajectory and the 4× pulse contrast at the end.
