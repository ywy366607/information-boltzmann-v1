# Mengli audit before training (2026-10-15)

**Rule (user):** audit everything first. Anything that needs calibration becomes a reasonable initialisation plus a
learnable mechanism. When the audit is all clear, it goes to review, and only then to training. No train-then-probe.
There are two audit classes:
- the connectome itself (done earlier: point-model artifacts);
- temporal reachability (new).

**Status:** ✅ fixed in code (not yet in life) · 🔍 open · ⚠ decision needed

| # | item | evidence | fix (first principles, fly mechanism, init + learnable) | status |
|---|---|---|---|---|
| T1 | A word ends before it reaches the mushroom body | Words last 4–8 ticks. Receptor → Kenyon takes 15–25 ticks; → MBON ≥ 22 (`reach_audit.py`). At 32 ticks per word the name's Kenyon Jaccard goes 0.078 → 0.74 and a'b' 15% → 41%. | The world's clock at speech rate: the mother's words last about 300 ms = 50–60 ticks (initial value; the world's own, not the brain's). The child's own timing is her act, already learnable. Ramp, because the adult is calibrated to 8 ticks. | 🔍 |
| T2 | The phasic dopamine clock (fast τ 36 ticks) smears 4–9 words; gate (2) correlates per word | `reach_audit` clocks | Measure gate (2) on the dopamine's own clock (and after T1 a word is longer than τ_fast). τ_fast is learnable already. | 🔍 |
| T3 | Speaking-credit τ 32 is a separate hand-set number from the Kenyon trace τ (26, learnable) | config | Credit decay = the brain's own eligibility τ (comp_logit_trace), one clock | 🔍 |
| T4 | BPTT window vs path latency | A mushroom-body round trip is about 40 ticks. 32 words × 32 ticks = 1,024 ticks. | Keep 32 words; save memory by finer tick checkpointing, not by a shorter window (a shorter window loses language context) | 🔍 |
| M1 | Compartment memory: learning and forgetting both trace-gated, so no frequency coding | φ −0.104 for the name and for other words | Active forgetting at tonic dopamine, learnable τ (init 2048) — d9b6e06 | ✅ |
| M2 | Speaking credit: the plain discount biases against speaking (~22%), so silence became absorbing | Shapley: w −8.14 of −8.87; predicted gap matches | Average-reward (differential) credit — 3ce88c8 | ✅ |
| M3 | Active inference: speaking has no epistemic value; ai_epistemic 0; dopamine precision off; m_main has no acts | config | Epistemic value = the fly's novelty circuit (MBON-α'3 / PPL1-α'3) on the act's consequence; γ = dopamine tone (`act_dopamine_precision`) | 🔍 needs M4 |
| M4 | MBON-α'3 never fires (birth too) | Inhibition is 8× excitation (DPM GABA tonic, 0.13 per tick; LHCENT8, APL, LHMB1). With DPM silenced v/θ −0.33 → −0.04, still silent. a'b' Kenyon rate about 0.003 per tick on long words. The VP (thermo/hygro) PNs that feed a'b' are silent. | Open: (a) DPM's tonic rate (its pacemaker κ: A homeostasis set point?) and DPM → MBON GABA sign; (b) a'b' drive after T1; (c) comp_weight initial efficacy onto α'3 | 🔍 |
| S1 | Senses weak: about 4 Hz evoked at ×1 | Writer gain dose–response | Re-measure after T1 (long words may suffice); else receptor gain as learnable physiology | 🔍 after T1 |
| H1 | GPU: about 0.8 GB taken outside; training OOMs even at the old 8-tick config | mem_get_info 3.3 GB free | Find the holder | ⚠ user's machine |
| W1 | The mother's mouth: DeepSeek is down | status page | SiliconFlow `glm-5.3-flash`, key from the user env, stable-prefix prompt | 🔍 |
| A | Pacemaker homeostasis (rate set point per cell, κ out of the gradient) | m_main PPL1 silenced by BPTT | 710edd6 | ✅ |
| B | Her answer as a PAM pulse | calibration: 98% of φ on standing traces | 710edd6; level / ticks to become learnable? | 🔍 |

## T1 refined (2026-10-15 13:40): late arrival vs no code; discriminability

Probe: `kc_code_probe.py`, run on the lineage's early, varied conversation (`KC_FROM=start`, 600 tokens). Pairs within 4
words are excluded, because neighbours share lingering activity. Values are the Kenyon set per word; "nearest-neighbour
identification" asks whether the most similar non-neighbour set belongs to the same token (chance 1.9%).

| word held | read window | active | Jaccard same / different | identification |
|---|---|---|---|---|
| 8 ticks | the word's ticks | 3.4% | 0.067 / 0.054 | 16% |
| 8 ticks | delayed by 13 ticks (pipelined) | 3.4% | 0.082 / 0.052 | 30% |
| 32 ticks | the word's ticks | 16% | 0.51 / 0.21 | **99%** |

- **Pipelining is partly there.** Aligning the read window to the latency doubles identification at 8 ticks, so part
  of the gap is late arrival.
- **Most of it is a missing code.** At 8 ticks a word leaves a weak, smeared set (30%). At 32 ticks words are clearly
  distinguishable (99%), even though more cells are recruited (different-token overlap 0.21).
- **Separate the three clocks** (user / GPT):
  - stimulus duration (how long a word is held);
  - arrival interval;
  - the internal step.
- **Absolute time is free.** A tick is a computing step, not 5 ms. What must hold is the ratio: an input has to last as
  many internal steps as the brain's own chain needs (about 15–25 here). The other way to meet it is to shorten the
  chain itself (learnable time constants, delays).
- **Both credit spans are recorded:** ticks and tokens.
