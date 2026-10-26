import CGBAxioms.GenerationAmplifier

/-!
# When must a forager compare now with the past? (generation chain, C-1)

A forager at a patch with local gain rate `q` either stays (earning `q`) or leaves for the rest
of the world (worth `c`, the environment's average rate). The best is `max q c`, attained by the
leave rule "stay iff `q ≥ c`" (marginal value theorem, Charnov 1976; the leave threshold of G12
`exists_leave_threshold` is this `c`).

* `fixed_threshold_optimal`: in a world whose average `c` never changes, the fixed threshold
  `τ = c` is optimal at every patch. Reading only the present level is enough; no comparison
  with the past is needed (what selection found in stationary worlds).
* `no_fixed_leave_threshold` (**C-1**): if the average takes two different values `c₁ < c₂`
  (a world whose richness changes), every fixed threshold loses at some patch in one of them.
  The threshold must follow the current average, which is not sensed directly but must be
  estimated from the recent past (G10, G11, G13): the forager compares now with the past.
-/

namespace CGB.Generation

/-- The value of the leave rule with threshold `τ` at a patch of rate `q` when leaving is worth
`c`. -/
noncomputable def leaveValue (τ c q : ℝ) : ℝ := if τ ≤ q then q else c

theorem leaveValue_le (τ c q : ℝ) : leaveValue τ c q ≤ max q c := by
  unfold leaveValue; split_ifs <;> simp

/-- In a stationary world the fixed threshold `τ = c` is optimal everywhere. -/
theorem fixed_threshold_optimal (c q : ℝ) : leaveValue c c q = max q c := by
  unfold leaveValue
  split_ifs with h
  · exact (max_eq_left h).symm
  · exact (max_eq_right (le_of_lt (not_le.mp h))).symm

/-- **C-1 (a changing world needs comparison)**: with two different averages `c₁ < c₂`, every
fixed threshold `τ` loses strictly at some patch rate `q` between them, in one of the two worlds. -/
theorem no_fixed_leave_threshold {c₁ c₂ : ℝ} (hc : c₁ < c₂) (τ : ℝ) :
    ∃ q, c₁ < q ∧ q < c₂ ∧
      (leaveValue τ c₁ q < max q c₁ ∨ leaveValue τ c₂ q < max q c₂) := by
  by_cases hτ : τ ≤ (c₁ + c₂) / 2
  · -- the threshold is low: at a patch just below `c₂` it stays, though leaving pays `c₂`
    refine ⟨(c₁ + c₂) / 2, by linarith, by linarith, Or.inr ?_⟩
    unfold leaveValue
    rw [if_pos hτ, max_eq_right (by linarith)]
    linarith
  · -- the threshold is high: at a patch above `c₁` it leaves, though staying pays more
    push Not at hτ
    refine ⟨(c₁ + (c₁ + c₂) / 2) / 2 + ((c₁ + c₂) / 2 - c₁) / 2, by linarith, by linarith, Or.inl ?_⟩
    unfold leaveValue
    rw [if_neg (by linarith), max_eq_left (by linarith)]
    linarith

end CGB.Generation
