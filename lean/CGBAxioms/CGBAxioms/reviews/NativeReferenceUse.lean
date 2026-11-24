import Mathlib.Tactic

namespace CGB.NativeReferenceUse

/-- Alternate routes that leave the effective predictor unchanged owe no credit. -/
theorem redundant_route {Evidence Outcome : Type}
    (score : Evidence → Outcome → ℝ) (full cut : Evidence) (actual : Outcome)
    (same : full = cut) : score full actual - score cut actual = 0 := by
  rw [same, sub_self]

/-- Commit first. The new scoring label is only an input to the next selection. -/
theorem label_after_selection {Credit Refs Event : Type}
    (select : Credit → Refs) (advance : Credit → Event → Credit)
    (old : Credit) (labelA labelB : Event) :
    (select old, advance old labelA).1 = (select old, advance old labelB).1 := rfl

end CGB.NativeReferenceUse
