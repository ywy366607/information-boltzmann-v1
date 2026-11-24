import Mathlib.Tactic

/-! Actual binding: successor_address_geometry.snapshot/addresses. Each live
relation is a cell defined by its original raw scope halfspaces, cosine
threshold and winner rule. A fresh cell completes the partition. The finite
pushforward below counts the SAME posterior draws before/after address lookup.
This validates the map; convergence of numerical mass is recorded separately.
-/
namespace CGB.NativeSuccessorAddress
noncomputable section

variable {V Z : Type*} [AddCommGroup V] [Module ℝ V] [AddCommGroup Z] [Module ℝ Z]

theorem lifted_halfspace_affine (E : V →ₗ[ℝ] Z) (R : Z →ₗ[ℝ] V)
    (scope : V →ₗ[ℝ] ℝ) (h : V) (z : Z) :
    scope (h + R (z-E h)) =
      scope h + (scope.comp R) (z-E h) := by
  simp

def mass {K A : Type*} [Fintype K] [DecidableEq A]
    (w : K → ℝ) (address : K → A) (a : A) :=
  ∑ k, if address k=a then w k else 0

theorem address_mass_normalized {K A : Type*} [Fintype K] [Fintype A] [DecidableEq A]
    (w : K → ℝ) (address : K → A) (normalized : ∑ k, w k=1) :
    ∑ a, mass w address a = 1 := by
  unfold mass
  rw [Finset.sum_comm]
  simpa [eq_comm] using normalized

theorem cell_constant_read_exact {K A : Type*} [Fintype K] [Fintype A] [DecidableEq A]
    (w : K → ℝ) (address : K → A) (v : A → ℝ) :
    ∑ a, mass w address a * v a = ∑ k, w k*v (address k) := by
  unfold mass
  simp only [Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k _
  simp [ite_mul,eq_comm]

end
end CGB.NativeSuccessorAddress
