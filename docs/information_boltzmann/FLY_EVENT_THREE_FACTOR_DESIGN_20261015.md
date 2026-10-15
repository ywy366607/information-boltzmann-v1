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
