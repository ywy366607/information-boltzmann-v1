import CGBAxioms.GenerationGrowth

/-!
# An equilibrium amplifier: ordered filling of a conserved shell (generation chain, G25)

Chemotaxis needs an all-or-none response to a tiny difference (a gain of order ten in
*E. coli*), and ignition needs the same switch closed into a loop. In a rule-table medium the
switch already exists at equilibrium. Take a closed shell of a low page `L` (`NL` units,
energy `φL`) and a high page `H` (`NH` units, energy `φH = φL + Δ`). Its units share one
potential `λ`, set by the shell's content, so their occupancies are `occupancy (λ - φ)` (G18).

* `occupancy_le_exp`, `one_sub_occupancy_le_exp`: both tails are exponentially small.
* `high_off_below_threshold` (G25a): while `L` is at most half full (`λ ≤ φL`), `H` is
  occupied at most `e^{-Δ}`.
* `switch_width` (G25b): between two potentials, the shell's content grows by at most
  `NH` times the change of `H`'s occupancy plus `NL e^{-(l1 - φL)}`. Taking `H` from occupancy
  `ε` to `1 - ε` (so `l1 - φL = Δ + logit ε`) costs at most `NH + NL e^{-Δ} (1-ε)/ε` content,
  while the switch sits at content `≈ NL`: a relative change `≈ NH/NL` of the input flips `H`.

This amplifies **content** (the input is a number of quanta put into a closed shell). Against a
**potential** (a reservoir's chemical potential, which is how every sense enters, P3) there is
no equilibrium gain at all:

* `equilibrium_hill_one` (G25c): with additive energies each unit's log-odds of being occupied
  is exactly `μ - φ`, so its Hill coefficient with respect to the potential is exactly 1,
  whatever the structure (the equilibrium law is a product over units). Ultrasensitivity to a
  sensed potential (Hill > 1, which chemotaxis needs) requires drive or a non-additive
  interaction energy, the same requirement as for the latch (G23). The amplifier of
  chemotaxis and the latch of ignition are one substrate: a driven or cooperative switch.
-/

namespace CGB.Generation

theorem occupancy_le_exp (y : ℝ) : occupancy y ≤ Real.exp y := by
  unfold occupancy
  have h1 : 0 < Real.exp (-y) := Real.exp_pos _
  rw [div_le_iff₀ (by positivity), mul_add, mul_one, ← Real.exp_add, add_neg_cancel,
    Real.exp_zero]
  linarith [Real.exp_pos y]

theorem one_sub_occupancy_le_exp (y : ℝ) : 1 - occupancy y ≤ Real.exp (-y) := by
  have h : 1 - occupancy y = occupancy (-y) := by
    unfold occupancy
    have h1 : 0 < 1 + Real.exp (-y) := by positivity
    have h2 : 0 < 1 + Real.exp (- -y) := by positivity
    rw [neg_neg]
    field_simp
    rw [show Real.exp (-y) = (Real.exp y)⁻¹ by rw [Real.exp_neg]]
    field_simp
    ring
  rw [h]; exact occupancy_le_exp (-y)

theorem occupancy_le_one (y : ℝ) : occupancy y ≤ 1 := by
  unfold occupancy
  rw [div_le_one (by positivity)]
  have := Real.exp_pos (-y); linarith

/-- **G25a (H stays off below threshold)**: while the low page is at most half full, the high
page is occupied at most `e^{-Δ}`. -/
theorem high_off_below_threshold {l φL Δ : ℝ} (hl : l ≤ φL) :
    occupancy (l - (φL + Δ)) ≤ Real.exp (-Δ) :=
  (occupancy_le_exp _).trans (Real.exp_le_exp.mpr (by linarith))

/-- Content of the shell at potential `λ`. -/
noncomputable def shellContent (NL NH φL φH l : ℝ) : ℝ :=
  NL * occupancy (l - φL) + NH * occupancy (l - φH)

/-- **G25b (switch width)**: raising the potential from `l1` to `l2` adds at most `NH` times the
gain in `H`'s occupancy plus `NL e^{-(l1 - φL)}` content. -/
theorem switch_width {NL NH φL φH l1 l2 : ℝ} (hNL : 0 ≤ NL) :
    shellContent NL NH φL φH l2 - shellContent NL NH φL φH l1 ≤
      NH * (occupancy (l2 - φH) - occupancy (l1 - φH)) + NL * Real.exp (-(l1 - φL)) := by
  unfold shellContent
  have h1 := occupancy_le_one (l2 - φL)
  have h2 := one_sub_occupancy_le_exp (l1 - φL)
  have : NL * (occupancy (l2 - φL) - occupancy (l1 - φL)) ≤ NL * Real.exp (-(l1 - φL)) :=
    mul_le_mul_of_nonneg_left (by linarith) hNL
  nlinarith

/-- **G25c (equilibrium Hill coefficient is one)**: the log-odds of occupation is exactly the
potential above the unit's energy, so it moves one-for-one with the sensed potential. -/
theorem equilibrium_hill_one (μ φ : ℝ) :
    Real.log (occupancy (μ - φ) / (1 - occupancy (μ - φ))) = μ - φ := by
  rw [occupancy_odds, Real.log_exp]

/-- **G25d (cooperativity from multiple quanta)**: a state holding `s` units of content, with
energy `φs`, has equilibrium weight `exp (s μ - φs)` relative to the empty state, so its log-odds
moves `s`-for-one with the potential: its Hill coefficient is `s`. Binary units give 1 (G25c);
a unit that takes up several quanta at once is cooperative at equilibrium, as in Hill's
haemoglobin. (Found by selection in the multi-state chemotaxis run before it was written down.) -/
theorem quanta_hill (s μ φs : ℝ) :
    Real.log (Real.exp (s * μ - φs) / Real.exp (s * 0 - φs)) = s * μ := by
  rw [← Real.exp_sub, Real.log_exp]; ring

end CGB.Generation
