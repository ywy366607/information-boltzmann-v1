import Mathlib.Tactic

/-! Actual body binding: persistent_rhythm.read_plans reads the scalar
native_value_bank; latent_action_planner.value uses its weighted mean.
The root-only shadow in root_option_calibration_shadow.py evaluates paired
action plans on those same states. Response hypotheses use the original
uniform mean. The actor receives 10*gamma times their future-value delta.
The old q-only patch is recorded separately from this explicit new lift.
-/
namespace CGB.NativePersistentCalibrationLift
noncomputable section

/-- Same successor weights transport the paired correction exactly. -/
theorem successor_transport {I : Type*} [Fintype I]
    (w base old new : I → ℝ) :
    (∑ i, w i*(base i+(new i-old i))) - (∑ i, w i*base i) =
    ∑ i, w i*(new i-old i) := by
  simp_rw [mul_add]
  rw [Finset.sum_add_distrib]
  ring

/-- The actual critic-unit lift changes action contrasts with the inherited
gain conversion and gamma, even when the scalar base already differs. -/
theorem root_contrast (a b da db gamma : ℝ) :
    ((a+10*gamma*da)-(b+10*gamma*db))-(a-b) =
    10*gamma*(da-db) := by ring

/-- New and old first-choice contributions use the same actual action G.
Their difference is precisely the probability displacement weighted by G;
the unchanged habit reference cancels. -/
theorem contribution_difference {I : Type*} [Fintype I]
    (new old habit gain : I → ℝ) :
    (∑ i, (new i-habit i)*gain i)-(∑ i, (old i-habit i)*gain i) =
    ∑ i, (new i-old i)*gain i := by
  rw [← Finset.sum_sub_distrib]
  congr 1
  funext i
  ring

/-- `_imagine` takes an outer uniform response mean. A replacement of one
inner successor value carries its inner mass divided by that response count. -/
theorem response_mean_replacement (sumValue innerMass err responses : ℝ)
    (hn : responses ≠ 0) :
    (sumValue+innerMass*err)/responses-sumValue/responses =
    (innerMass/responses)*err := by
  field_simp
  ring

/-- Averaging original normalized successor distributions preserves mass.
The implementation aggregates repeated content keys before this division. -/
theorem averaged_node_mass_normalized {H J : Type*}
    [Fintype H] [Fintype J] [Nonempty H]
    (w : H → J → ℝ) (normal : ∀ h, ∑ j, w h j = 1) :
    (∑ j, (∑ h, w h j)/(Fintype.card H : ℝ)) = 1 := by
  rw [← Finset.sum_div, Finset.sum_comm]
  simp_rw [normal]
  simp

/-- Repeated action aliases duplicate numerator and response count,
preserving the same physical root's coefficient. -/
theorem aliases_preserve_mass (mass responses aliases : ℝ)
    (hr : responses ≠ 0) (ha : aliases ≠ 0) :
    (aliases*mass)/(aliases*responses)=mass/responses := by
  field_simp

end
end CGB.NativePersistentCalibrationLift
