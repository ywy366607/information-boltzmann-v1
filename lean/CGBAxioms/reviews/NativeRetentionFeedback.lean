import Mathlib.Tactic

/-! Binding: experience_consolidation.py and the native observation cycle.
Read-dose Adam and retention utility are separate persisted coordinates.
Finite removal is evaluated on an arrived external target at common weights;
its sign is not an original-write return or a guarantee of future improvement.
-/
namespace CGB.NativeRetentionFeedback
noncomputable section

/-- The existing rule reads u; changing only read dose d cannot revise u. -/
def doseOnly (s : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (d, s.2)

theorem missing_writer_preserves_retention (s : ℝ × ℝ) (d : ℝ) :
    (doseOnly s d).2 = s.2 := rfl

theorem missing_writer_preserves_choice {A : Type*}
    (choose : ℝ → A) (s : ℝ × ℝ) (d : ℝ) :
    choose (doseOnly s d).2 = choose s.2 := rfl

/-- Common target, parameters, start state and all other inputs are premises. -/
def benefit (without withRecord : ℝ) : ℝ := without - withRecord

theorem positive_benefit_iff (without withRecord : ℝ) :
    0 < benefit without withRecord ↔ withRecord < without := by
  unfold benefit
  exact sub_pos

/-- A useful record can be overdosed: lowering its dose helps, deleting hurts. -/
theorem finite_use_is_not_dose_gradient :
    (0 : ℚ) < (0-1)^2 - (3/2-1)^2 ∧
    (0 : ℚ) < 2*(3/2-1) := by norm_num

def update (decay old observed : ℝ) : ℝ :=
  decay*old + (1-decay)*observed

theorem same_identity_observation_changes_evidence
    (decay old observed : ℝ) (hd : decay < 1) (h : old ≠ observed) :
    update decay old observed ≠ old := by
  unfold update
  intro equal
  have product : (1-decay)*(observed-old) = 0 := by nlinarith [equal]
  rcases mul_eq_zero.mp product with hzero | hzero
  · linarith
  · apply h
    linarith

/-- Actual existing first-moment decay .9: four contrary uses reverse one use. -/
theorem contrary_uses_reopen :
    update (9/10) (update (9/10) (update (9/10)
      (update (9/10) (update (9/10) 0 (1/10)) (-1/10)) (-1/10)) (-1/10)) (-1/10) < 0 := by
  norm_num [update]

/-- A recycled address cannot be paid as the original participating identity. -/
def settle (same : Bool) (decay old observed : ℝ) : ℝ :=
  if same then update decay old observed else old

theorem stale_identity_unpaid (decay old observed : ℝ) :
    settle false decay old observed = old := rfl

/-- The current choice precedes the arrived label: only the successor sees it. -/
def transition {A : Type*} (choose : ℝ → A) (decay old observed : ℝ) : A × ℝ :=
  (choose old, update decay old observed)

theorem current_choice_has_no_future_label {A : Type*}
    (choose : ℝ → A) (decay old x y : ℝ) :
    (transition choose decay old x).1 = (transition choose decay old y).1 := rfl

end
end CGB.NativeRetentionFeedback
