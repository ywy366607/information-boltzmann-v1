import Mathlib.Tactic

/-! Binding: scripts/option_basis_calibration.py.
V is the weighted bank-row space (at most 64 rows); W is the space of
represented option intercepts (at most three). A is the inherited weighted
Gram plus the inherited positive ridge; H is the source-weighted one-hot
design. L is its transpose. The inherited positive-ridge result supplies the
inverse A⁻¹. These statements prove the Schur solution and its invariance to
option-wise shifts of the guarded prior, with fixed target labels.
-/
namespace CGB.NativeOptionBasisCalibration
noncomputable section
variable {V W : Type*} [AddCommGroup V] [Module ℝ V]
  [AddCommGroup W] [Module ℝ W]

/-- The production class list is exactly the represented options. Each
class has a row, and the inherited source weights are strictly positive;
therefore its weighted one-hot design has no invisible intercept. -/
theorem represented_option_design_injective
    {I C : Type*} (option : I → C) (weight : I → ℝ)
    (represented : Function.Surjective option) (nonzero : ∀ i, weight i ≠ 0) :
    Function.Injective (fun b : C → ℝ => fun i => weight i*b (option i)) := by
  intro x y h
  funext c
  obtain ⟨i, hi⟩ := represented c
  have row : weight i*x c = weight i*y c := by
    simpa [hi] using congrFun h i
  exact (mul_left_cancel₀ (nonzero i)) row

/-- Native Schur elimination: alpha0=A⁻¹y, Z=A⁻¹H,
beta solves (L Z) beta=L alpha0. -/
theorem schur_normal_equations
    (A : V →ₗ[ℝ] V) (H Z : W →ₗ[ℝ] V) (L : V →ₗ[ℝ] W)
    (y a0 : V) (beta : W)
    (ha : A a0 = y) (hz : ∀ b, A (Z b) = H b)
    (hb : L (Z beta) = L a0) :
    A (a0-Z beta)+H beta = y ∧ L (a0-Z beta) = 0 := by
  constructor
  · simp [ha, hz]
  · simp [hb]

/-- Changing prior by C shift changes the fitted intercept by -shift;
the kernel coefficients retain the same normal equations. -/
theorem shifted_prior_normal_equations
    (A : V →ₗ[ℝ] V) (H : W →ₗ[ℝ] V) (L : V →ₗ[ℝ] W)
    (y alpha : V) (beta shift : W)
    (hn : A alpha+H beta=y) (ho : L alpha=0) :
    A alpha+H (beta-shift)=y-H shift ∧ L alpha=0 := by
  constructor
  · simp only [map_sub]; rw [← hn]; abel
  · exact ho

/-- The actual query prediction is invariant to that same prior shift.
Here c is the query option-design row and k the weighted cross-kernel row. -/
theorem query_prior_invariance
    (c : W →ₗ[ℝ] ℝ) (k : V →ₗ[ℝ] ℝ)
    (p : ℝ) (alpha : V) (beta shift : W) :
    (p+c shift)+c (beta-shift)+k alpha = p+c beta+k alpha := by
  simp only [map_sub]; ring

/-- A positive quadratic form composed with an injective option design
stays positive. For A SPD, this is the Schur form Hᵀ A⁻¹ H, hence the
small option system is identifiable for each represented class. -/
theorem schur_positive
    (H : W →ₗ[ℝ] V) (Q : V → ℝ)
    (hi : Function.Injective H) (hq : ∀ v, v ≠ 0 → 0 < Q v)
    (b : W) (hb : b ≠ 0) : 0 < Q (H b) := by
  apply hq
  intro hz
  apply hb
  apply hi
  simpa using hz

/-- The content/intercept fit minimizes squared weighted residual plus
the inherited RKHS penalty. This expansion is the objective increment
at the two normal equations (residual=lambda*alpha, Hᵀalpha=0).
The remaining terms are nonnegative for a PSD kernel and lambda >= 0. -/
theorem regularized_objective_increment
    {X Y : Type*} [NormedAddCommGroup X] [InnerProductSpace ℝ X]
    [AddCommGroup Y] [Module ℝ Y]
    (G : X →ₗ[ℝ] X) (H : Y →ₗ[ℝ] X)
    (symmetric : ∀ x y, inner ℝ x (G y) = inner ℝ y (G x))
    (a d : X) (e : Y) (lambda : ℝ)
    (orthogonal : inner ℝ a (H e) = 0) :
    (inner ℝ (lambda • a-(G d+H e)) (lambda • a-(G d+H e)) +
      lambda*inner ℝ (a+d) (G (a+d))) -
    (inner ℝ (lambda • a) (lambda • a)+lambda*inner ℝ a (G a)) =
    inner ℝ (G d+H e) (G d+H e)+lambda*inner ℝ d (G d) := by
  have cross : inner ℝ (G d) a = inner ℝ d (G a) := by
    calc
      _ = inner ℝ a (G d) := real_inner_comm _ _
      _ = _ := symmetric a d
  have zero : inner ℝ (H e) a = 0 := by
    calc
      _ = inner ℝ a (H e) := real_inner_comm _ _
      _ = 0 := orthogonal
  simp only [map_add, inner_add_left, inner_add_right, inner_sub_left,
    inner_sub_right, inner_smul_left, inner_smul_right, cross, zero,
    orthogonal, symmetric a d, starRingEnd_apply, star_trivial]
  ring

end
end CGB.NativeOptionBasisCalibration
