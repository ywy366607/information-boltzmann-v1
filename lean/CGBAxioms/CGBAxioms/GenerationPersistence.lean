import CGBAxioms.GenerationGrowth

/-!
# Activity without input: driven circulation, adaptation, latches (generation chain, 2026-10-27)

Three different things keep a body active when the input says nothing, and they must not be confused
(eco-sphere audit: 154 of 211 dead bodies of the espop2 species never moved; their motor started only
after sensing food). Each gets its own necessity statement.

**A. Spontaneous circulation needs a drive.** A self-repeating act (run, tumble, run; speak, listen,
speak) is a cycle of states with net circulation. `cycle_rates_balance` (Kolmogorov's criterion, one
direction): under detailed balance with a positive weight, the product of rates around any cycle equals
the product in the reverse direction, so no cycle carries a net current. `driven_cycle_not_balanced`:
a cycle whose forward and reverse rate products differ rules out every positive detailed-balance
weight. With G6DB (all irreversibility enters at the boundary) a body's own default activity can
circulate only through events that consume what crosses the boundary (intake) or a store that the
boundary refills.

**B. An adapting reference only steers.** Integral feedback (the chemotaxis methylation of E. coli)
updates a reference `m` by `γ (a - a₀)` with activity `a = f (s - m)`. `integral_feedback_adapts`:
at every fixed point the activity is `a₀`, whatever the constant input `s`; `adapted_gap_zero`: the
gap `a - a₀` that drives the act is then zero. In a constant world (a dry place stays dry) a body whose
acts are driven only by this gap, or only by sensed food, rests; with no intake where it rests, H1
(`trap_irrecoverable`) makes that rest irrecoverable. Search in a uniform world is the job of A.

**C. A latch needs loop gain at least one.** A latch is a positive-feedback loop with two rest states
(set by one cause, reset by another). `contraction_unique_rest`: a loop map with gain `L < 1`
everywhere has at most one rest state, so tonic (driven, spontaneous) activity without enough gain is
monostable and never latches. `latch_needs_unit_gain`: two distinct rest states of a differentiable
loop map force slope one somewhere between them. The wiring gives the loops (the maleCNS connectome:
45k cells in strong reciprocal excitatory pairs, 30k in mutual inhibition, the head-direction ring
EPG <-> PEN among them); whether a loop latches is decided by its gain, which growth and plasticity
set. G23/G24 add that the held state, beyond the equilibrium memory bound, needs drive.
-/

namespace CGB.Generation

open Finset

section Circulation

variable {X : Type*}

/-- **A (Kolmogorov, one direction)**: detailed balance with a positive weight balances the rate
products around every cycle `c 0 → c 1 → … → c (n-1) → c 0`. -/
theorem cycle_rates_balance {n : ℕ} (K : X → X → ℝ) (π : X → ℝ) (hπ : ∀ x, 0 < π x)
    (hdb : ∀ x y, π x * K x y = π y * K y x) (c : Fin n → X) :
    ∏ i, K (c i) (c (finRotate n i)) = ∏ i, K (c (finRotate n i)) (c i) := by
  have h1 : ∏ i, (π (c i) * K (c i) (c (finRotate n i))) =
      ∏ i, (π (c (finRotate n i)) * K (c (finRotate n i)) (c i)) :=
    prod_congr rfl (fun i _ => hdb _ _)
  rw [prod_mul_distrib, prod_mul_distrib] at h1
  have hrot : ∏ i, π (c (finRotate n i)) = ∏ i, π (c i) :=
    Equiv.prod_comp (finRotate n) (fun i => π (c i))
  rw [hrot] at h1
  have hpos : (∏ i, π (c i)) ≠ 0 := (prod_pos (fun i _ => hπ (c i))).ne'
  exact mul_left_cancel₀ hpos h1

/-- **A**: a cycle with unequal forward and reverse rate products (net circulation) admits no
positive detailed-balance weight: self-repeating activity needs a drive. -/
theorem driven_cycle_not_balanced {n : ℕ} (K : X → X → ℝ) (c : Fin n → X)
    (hcirc : ∏ i, K (c i) (c (finRotate n i)) ≠ ∏ i, K (c (finRotate n i)) (c i)) :
    ¬ ∃ π : X → ℝ, (∀ x, 0 < π x) ∧ ∀ x y, π x * K x y = π y * K y x := by
  rintro ⟨π, hπ, hdb⟩
  exact hcirc (cycle_rates_balance K π hπ hdb c)

end Circulation

section Adaptation

/-- **B (integral feedback adapts perfectly)**: if the reference update `m ↦ m + γ (f (s - m) - a₀)`
with `γ ≠ 0` has `m` as a fixed point, the activity there is `a₀`, whatever the input `s`. -/
theorem integral_feedback_adapts (f : ℝ → ℝ) (γ a₀ s m : ℝ) (hγ : γ ≠ 0)
    (hfix : m + γ * (f (s - m) - a₀) = m) : f (s - m) = a₀ := by
  have h : γ * (f (s - m) - a₀) = 0 := by linarith
  rcases mul_eq_zero.mp h with h | h
  · exact absurd h hγ
  · linarith

/-- **B**: an act driven only by the gap `g (a - a₀)` with `g 0 = 0` stops at every adapted fixed
point of a constant input. -/
theorem adapted_gap_zero (f g : ℝ → ℝ) (hg : g 0 = 0) (γ a₀ s m : ℝ) (hγ : γ ≠ 0)
    (hfix : m + γ * (f (s - m) - a₀) = m) : g (f (s - m) - a₀) = 0 := by
  rw [integral_feedback_adapts f γ a₀ s m hγ hfix, sub_self, hg]

end Adaptation

section Latch

/-- **C**: a loop map with gain `L < 1` has at most one rest state: tonic activity without enough
loop gain is monostable. -/
theorem contraction_unique_rest (f : ℝ → ℝ) (L : ℝ) (hL : L < 1)
    (hlip : ∀ x y, |f x - f y| ≤ L * |x - y|) (x y : ℝ) (hx : f x = x) (hy : f y = y) : x = y := by
  have h := hlip x y
  rw [hx, hy] at h
  have h0 : |x - y| ≤ 0 := by nlinarith [abs_nonneg (x - y)]
  have : |x - y| = 0 := le_antisymm h0 (abs_nonneg _)
  exact sub_eq_zero.mp (abs_eq_zero.mp this)

/-- **C**: two distinct rest states of a differentiable loop map force loop gain (slope) exactly one
somewhere between them: a latch needs unit gain. -/
theorem latch_needs_unit_gain (f : ℝ → ℝ) (hf : Differentiable ℝ f) (x y : ℝ) (hxy : x < y)
    (hx : f x = x) (hy : f y = y) : ∃ z ∈ Set.Ioo x y, deriv f z = 1 := by
  obtain ⟨z, hz, hd⟩ := exists_deriv_eq_slope f hxy hf.continuous.continuousOn
    (fun w _ => (hf w).differentiableWithinAt)
  refine ⟨z, hz, ?_⟩
  rw [hd, hx, hy, div_self (sub_ne_zero.mpr hxy.ne')]

end Latch

end CGB.Generation
