import Mathlib.Tactic

/-! Binding: prospective_decision.boundary/lift_boundary/preview_bind/value.
V is the actual full decision vector; Z is the existing encoder space.
anchor is the raw successor posterior, fitted to actual source/action/mother
and raw internal_next labels. R is Wᵀ(WWᵀ)⁻¹. The right-inverse premise is
checked numerically against the actual current encoder before admission.
The live scope test remains on V. All question/action probabilities and finite
U are the native ones; future word endpoints are predicted program executions.
-/
namespace CGB.NativeProspectiveDecision
noncomputable section
variable {V Z : Type*} [AddCommGroup V] [Module ℝ V] [AddCommGroup Z] [Module ℝ Z]

def lift (E : V →ₗ[ℝ] Z) (R : Z →ₗ[ℝ] V) (h : V) (z : Z) := h + R (z-E h)

theorem lifted_boundary_has_forecast (E : V →ₗ[ℝ] Z) (R : Z →ₗ[ℝ] V)
    (right : ∀ z, E (R z)=z) (h : V) (z : Z) :
    E (lift E R h z)=z := by
  simp [lift, map_add, right]

theorem observed_coordinate_recovers_anchor (E : V →ₗ[ℝ] Z) (R : Z →ₗ[ℝ] V)
    (h : V) : lift E R h (E h)=h := by simp [lift]

theorem raw_distinctions_in_kernel_survive (E : V →ₗ[ℝ] Z) (R : Z →ₗ[ℝ] V)
    (h : V) (n : V) (same : E n=0) (z : Z) :
    lift E R (h+n) z=lift E R h z+n := by
  simp only [lift, map_add, same, add_zero]
  abel

/-- The future scope uses the raw forecast. Reusing only E(h) identifies
distinct scope projections; carrying the raw anchor retains that information. -/
theorem scope_changes_with_retained_raw_direction
    (E : V →ₗ[ℝ] Z) (R : Z →ₗ[ℝ] V) (scope : V →ₗ[ℝ] ℝ)
    (h n : V) (same : E n=0) (z : Z) :
    scope (lift E R (h+n) z)-scope (lift E R h z)=scope n := by
  rw [raw_distinctions_in_kernel_survive E R h n same z, map_add]
  abel

end
end CGB.NativeProspectiveDecision
