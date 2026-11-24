import Mathlib.Tactic

/-! Actual-body binding: latent_action_planner.value supplies original successor
mass w and frozen predicted v for a (source, action) root. flush_teachers has
already evaluated t = prospective_decision.value at two such points before the
real actor selects. root_native_correction replaces only these contributions.
The fitted bank and observed world evidence stay fixed until real settlement.
The patch uses wood units 10*U and converts to critic units 100*U in _value.
-/
namespace CGB.RootNativeCorrection
noncomputable section

def patched (covered : Bool) (v t : ℝ) : ℝ :=
  if covered then t else v

theorem evaluated_contribution_replaced (w v t : ℝ) :
    w*v + w*(t-v) = w*t := by ring

theorem selected_error_removed (v t : ℝ) :
    t - patched true v t = 0 := by simp [patched]

theorem untouched_error_preserved (v t : ℝ) :
    t - patched false v t = t-v := by simp [patched]

theorem remaining_error {I : Type*} [Fintype I]
    (covered : I → Bool) (w v t : I → ℝ) :
    (∑ i, w i*t i) - (∑ i, w i*patched (covered i) (v i) (t i)) =
      ∑ i, if covered i then 0 else w i*(t i-v i) := by
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  cases h : covered i <;> simp [patched, h] <;> ring

theorem critic_wood_units (gamma mass residual : ℝ) :
    (10 * gamma * mass * residual) / 10 = gamma*mass*residual := by ring

theorem logit_difference (ha hb precision qa qb da db : ℝ) :
    ((ha+precision*(qa+da))-(hb+precision*(qb+db))) -
      ((ha+precision*qa)-(hb+precision*qb)) = precision*(da-db) := by ring

theorem common_shift_preserves_difference (qa qb c : ℝ) :
    (qa+c)-(qb+c) = qa-qb := by ring

end
end CGB.RootNativeCorrection
