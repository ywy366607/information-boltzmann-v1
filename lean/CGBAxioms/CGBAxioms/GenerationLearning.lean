import CGBAxioms.GenerationLeaveRule

/-!
# When does a world reward learning? (generation chain, L-1 to L-3)

C-1 (`GenerationLeaveRule`) says when a forager must compare now with the past within one life.
Lifted to generations: an innate rule is a threshold `τ` fixed by the genome for every life; a
learner observes its world first (at a cost `κ ≥ 0`) and then uses the world's own average.

* L-1 `innate_optimal_of_constant`, `learner_loses_of_constant`: if every generation meets the
  same average `c`, the innate threshold `τ = c` is optimal at every patch and a learner that
  pays any positive cost does strictly worse: stable worlds make instincts, not learners.
* L-2 `innate_loss_quarter`, `learner_beats_innate`: if generations meet two averages
  `c₁ < c₂`, every innate threshold loses at least `(c₂ - c₁) / 4` at some patch in one of them;
  a learner whose cost is below that wins there. Worlds that change between generations but not
  within a life make learners.
* L-3 `history_useless_of_independent`: if what comes next is independent of what was seen
  (no temporal correlation within the life), any rule chosen from the history earns on average no
  more than the best fixed rule: memory and learning need a world whose past predicts its future.

These are conditional theorems about worlds (phenomena), not admission obligations.
-/

namespace CGB.Generation

/-- **L-1**: a constant average across generations makes the innate threshold optimal. -/
theorem innate_optimal_of_constant (c : ℝ) (q : ℝ) : leaveValue c c q = max q c :=
  fixed_threshold_optimal c q

/-- **L-1**: in a constant world a learner that pays `κ > 0` loses to the instinct at every patch. -/
theorem learner_loses_of_constant {κ : ℝ} (hκ : 0 < κ) (c q : ℝ) :
    max q c - κ < leaveValue c c q := by
  rw [fixed_threshold_optimal]; linarith

/-- **L-2**: with two different averages every innate threshold loses at least a quarter of their
gap at some patch in one of the two worlds. -/
theorem innate_loss_quarter {c₁ c₂ : ℝ} (hc : c₁ < c₂) (τ : ℝ) :
    ∃ q, (leaveValue τ c₁ q ≤ max q c₁ - (c₂ - c₁) / 4) ∨
      (leaveValue τ c₂ q ≤ max q c₂ - (c₂ - c₁) / 4) := by
  by_cases hτ : τ ≤ (c₁ + c₂) / 2
  · refine ⟨(c₁ + c₂) / 2, Or.inr ?_⟩
    unfold leaveValue
    rw [if_pos hτ, max_eq_right (by linarith)]
    linarith
  · push Not at hτ
    refine ⟨(3 * c₁ + c₂) / 4 + (c₂ - c₁) / 4, Or.inl ?_⟩
    unfold leaveValue
    rw [if_neg (by linarith), max_eq_left (by linarith)]
    linarith

/-- **L-2**: a learner (who uses the world's own average after paying `κ`) beats every instinct
somewhere when its cost is below a quarter of the gap. -/
theorem learner_beats_innate {c₁ c₂ κ : ℝ} (hc : c₁ < c₂) (hκ : κ < (c₂ - c₁) / 4) (τ : ℝ) :
    ∃ q, (leaveValue τ c₁ q < max q c₁ - κ) ∨ (leaveValue τ c₂ q < max q c₂ - κ) := by
  obtain ⟨q, h | h⟩ := innate_loss_quarter hc τ
  · exact ⟨q, Or.inl (by linarith)⟩
  · exact ⟨q, Or.inr (by linarith)⟩

/-- **L-3**: if the next world state `c` (law `p`) is independent of the history `e` (law `r`)
from which the rule `σ e` is chosen, the expected value is at most that of the best fixed rule. -/
theorem history_useless_of_independent {E Cw R : Type*} [Fintype E] [Fintype Cw] [Nonempty R]
    (r : E → ℝ) (p : Cw → ℝ) (hr : ∀ e, 0 ≤ r e) (hr1 : ∑ e, r e = 1) (hp : ∀ c, 0 ≤ p c)
    (v : R → Cw → ℝ) (σ : E → R) {M : ℝ} (hM : ∀ ρ, ∑ c, p c * v ρ c ≤ M) :
    ∑ e, ∑ c, r e * p c * v (σ e) c ≤ M := by
  have : ∀ e, ∑ c, r e * p c * v (σ e) c = r e * ∑ c, p c * v (σ e) c := fun e => by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun c _ => by ring
  simp_rw [this]
  calc ∑ e, r e * ∑ c, p c * v (σ e) c ≤ ∑ e, r e * M :=
        Finset.sum_le_sum fun e _ => mul_le_mul_of_nonneg_left (hM (σ e)) (hr e)
    _ = M := by rw [← Finset.sum_mul, hr1, one_mul]

end CGB.Generation
