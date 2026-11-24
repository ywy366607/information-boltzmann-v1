import Mathlib.Tactic

/-! Reuse GenerationRoutingCredit.softmax_read_direction_deriv for the
derivative. Here p0,p1,p2 are the actual three-choice actor probabilities,
g are its fixed already measured conditional G2 outcomes, and d are the
actual precision-scaled root-critic increments. Two legal choices set p2=0.
The body binding numerically checks every pair's (g_i-g_j)(d_i-d_j).
-/
namespace CGB.NativeValueDeformationOrder
noncomputable section

def covariance3 (p0 p1 p2 g0 g1 g2 d0 d1 d2 : ℝ) :=
  p0*g0*d0+p1*g1*d1+p2*g2*d2 -
    (p0*g0+p1*g1+p2*g2)*(p0*d0+p1*d1+p2*d2)

theorem covariance_pairs (p0 p1 p2 g0 g1 g2 d0 d1 d2 : ℝ)
    (hp : p0+p1+p2=1) :
    covariance3 p0 p1 p2 g0 g1 g2 d0 d1 d2 =
      p0*p1*((g0-g1)*(d0-d1))+p0*p2*((g0-g2)*(d0-d2))+
      p1*p2*((g1-g2)*(d1-d2)) := by
  unfold covariance3
  linear_combination -(p0*g0*d0+p1*g1*d1+p2*g2*d2)*hp

theorem same_order_nonnegative (p0 p1 p2 g0 g1 g2 d0 d1 d2 : ℝ)
    (hp : p0+p1+p2=1) (h0 : 0 ≤ p0) (h1 : 0 ≤ p1) (h2 : 0 ≤ p2)
    (h01 : 0 ≤ (g0-g1)*(d0-d1))
    (h02 : 0 ≤ (g0-g2)*(d0-d2))
    (h12 : 0 ≤ (g1-g2)*(d1-d2)) :
    0 ≤ covariance3 p0 p1 p2 g0 g1 g2 d0 d1 d2 := by
  rw [covariance_pairs p0 p1 p2 g0 g1 g2 d0 d1 d2 hp]
  exact add_nonneg
    (add_nonneg (mul_nonneg (mul_nonneg h0 h1) h01)
      (mul_nonneg (mul_nonneg h0 h2) h02))
    (mul_nonneg (mul_nonneg h1 h2) h12)

theorem log_odds_crossing (base slope strength : ℝ) (hs : slope ≠ 0) :
    base+strength*slope=0 ↔ strength = -base/slope := by
  constructor
  · intro h
    apply (eq_div_iff hs).mpr
    linarith
  · intro h
    have hh := (eq_div_iff hs).mp h
    linarith

end
end CGB.NativeValueDeformationOrder
