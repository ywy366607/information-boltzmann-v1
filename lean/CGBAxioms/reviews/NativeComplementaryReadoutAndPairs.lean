import Mathlib.Tactic

/-!
Actual port binding (cycle 2537):
E = encoder.weight : R^256 -> R^128.
J = D_h(momentum_to_boundary(decision_momentum_read(P,h))) at fixed P.
C = E^T (E E^T)^(-1), the production encoded-state right inverse.
The audit computes J restricted to ker E in eight saved native conditions.
Full rank there supplies a kernel inverse K. This file states precisely
which reconstruction follows from that condition, at the actual dimensions.
The numerical rank premise and this algebraic consequence are separate
artifacts: cgb_joint_target_coordinates_20261122.json stores singular values.

The final finite identity binds the symmetric content-weighted commutator
energy in adaptive_hamiltonian.potential to independent_pair_potential.
Antisymmetric commutators give symmetric squared pair energy and zero diagonal.
-/
namespace CGB.NativeComplementaryReadoutAndPairs
noncomputable section

abbrev BodyCondition := Fin 256 → ℝ
abbrev EncodedCondition := Fin 128 → ℝ

/-- The exact additional condition on the second readout is kernel separation. -/
theorem joint_readout_injective_iff
    (E J : BodyCondition →ₗ[ℝ] EncodedCondition) :
    Function.Injective (fun x => (E x, J x)) ↔
      ∀ x, E x = 0 → J x = 0 → x = 0 := by
  constructor
  · intro h x he hj
    apply h
    simp [he, hj]
  · intro h x y hxy
    have he : E x = E y := congrArg Prod.fst hxy
    have hj : J x = J y := congrArg Prod.snd hxy
    have hz := h (x-y) (by simp [he]) (by simp [hj])
    exact sub_eq_zero.mp hz

/-- Reconstruct the encoder-visible part and then its residual using P-read.
K only needs to invert J on ker E, rather than on all body coordinates. -/
theorem reconstruct_from_complementary_readouts
    (E J : BodyCondition →ₗ[ℝ] EncodedCondition)
    (C K : EncodedCondition →ₗ[ℝ] BodyCondition)
    (rightInverse : ∀ z, E (C z) = z)
    (kernelInverse : ∀ d, E d = 0 → K (J d) = d)
    (x : BodyCondition) :
    C (E x) + K (J x - J (C (E x))) = x := by
  have hd : E (x-C (E x)) = 0 := by simp [rightInverse]
  have hk := kernelInverse (x-C (E x)) hd
  rw [map_sub] at hk
  rw [hk]
  abel

/-- The existing degree-two kernel changes directional magnitudes within
fixed factors on unit-bounded packet features. Here a=||v||², n=||q||²,
b²=<q,v>²; the final premise is Cauchy--Schwarz. -/
theorem polynomial_kernel_tangent_bounds (a n b : ℝ)
    (ha : 0 ≤ a) (hn1 : n ≤ 1)
    (hcs : b^2 ≤ n*a) :
    2*a ≤ 2*(1+n)*a+2*b^2 ∧ 2*(1+n)*a+2*b^2 ≤ 6*a := by
  constructor
  · nlinarith [sq_nonneg b]
  · nlinarith [mul_nonneg (sub_nonneg.mpr hn1) ha]

/-- On a fixed active-role stratum, the normalized packet has constant norm.
Its tangents are perpendicular to the packet; the kernel multiplies all
squared tangent lengths by the same positive factor, preserving their ratio. -/
theorem polynomial_kernel_perpendicular_ratio (n a c : ℝ)
    (hn : 0 ≤ n) (hc : c ≠ 0) :
    (2*(1+n)*a)/(2*(1+n)*c) = a/c := by
  have h : 2*(1+n) ≠ 0 := by positivity
  field_simp


/-- For the scalar pair energy, summing each unordered pair once halves the
ordered sum. Head gates remain content dependent and symmetric. -/
theorem ordered_pair_energy_eq_twice_upper
    (n : ℕ) (f : Fin n → Fin n → ℝ)
    (symm : ∀ i j, f i j = f j i) (diag : ∀ i, f i i = 0) :
    (∑ i, ∑ j, f i j) = 2 * ∑ i, ∑ j, if i < j then f i j else 0 := by
  have term : ∀ i j, f i j =
      (if i < j then f i j else 0) + (if j < i then f j i else 0) := by
    intro i j
    rcases lt_trichotomy i j with hij | hij | hij
    · simp [hij, not_lt_of_gt hij]
    · subst j; simp [diag]
    · simp [hij, not_lt_of_gt hij, symm]
  have split : (∑ i, ∑ j, f i j) =
      (∑ i, ∑ j, if i < j then f i j else 0) +
      (∑ i, ∑ j, if j < i then f j i else 0) := by
    calc
      _ = ∑ i, ∑ j, ((if i < j then f i j else 0) +
          (if j < i then f j i else 0)) := by
        apply Finset.sum_congr rfl
        intro i _
        apply Finset.sum_congr rfl
        intro j _
        exact term i j
      _ = _ := by simp only [Finset.sum_add_distrib]
  rw [split]
  conv_lhs => rhs; rw [Finset.sum_comm]
  ring

end
end CGB.NativeComplementaryReadoutAndPairs
