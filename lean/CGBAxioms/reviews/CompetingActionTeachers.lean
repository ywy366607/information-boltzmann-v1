import Mathlib.Tactic

/-! Body binding: latent_action_planner.value retains w*mu contributions for
each actual (source, action) root key in CausalProgramEngine._relation_value.
competing_action_teachers retains per-root coverage and assigns at most two
native calls to the highest-valued distinct AVAILABLE root endpoints. The
current values remain frozen; native labels join the existing pending bank,
which real finite-U settlement commits. Numerical contracts check IDs, masks,
coverage, permutation, both retention schemas and checkpoint continuation.

The useful quantity is the ERROR IN THE DIFFERENCE of two action values.
Below q are frozen predicted root values, v corresponding reference values,
and e are applicable absolute error bounds. No guarantee on unobserved v is
inferred from the empirical native residual; that residual is recorded.
-/
namespace CGB.CompetingActionTeachers

theorem margin_error (qa qb va vb ea eb : ℝ)
    (ha : |va-qa| ≤ ea) (hb : |vb-qb| ≤ eb) :
    |(va-vb)-(qa-qb)| ≤ ea+eb := by
  calc
    |(va-vb)-(qa-qb)| = |(va-qa) + -(vb-qb)| := by congr 1; ring
    _ ≤ |va-qa| + |-(vb-qb)| := abs_add_le _ _
    _ = |va-qa| + |vb-qb| := by rw [abs_neg]
    _ ≤ ea+eb := add_le_add ha hb

theorem robust_order (qa qb va vb ea eb : ℝ)
    (ha : |va-qa| ≤ ea) (hb : |vb-qb| ≤ eb)
    (margin : ea+eb < qa-qb) : vb < va := by
  have h := (abs_le.mp (margin_error qa qb va vb ea eb ha hb)).1
  linarith

-- A native label's action-value sensitivity is its original mass times
-- gamma. This is the finite weighted contribution, not a new action reward.
theorem weighted_replacement (q gamma mass mu v : ℝ) :
    (q + gamma*mass*(v-mu))-q = gamma*mass*(v-mu) := by ring

theorem shared_teacher_cap {α : Type*} (menu : List α) :
    (menu.take 2).length ≤ 2 := by simp

end CGB.CompetingActionTeachers
