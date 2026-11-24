import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-! Local, smooth, active-branch model of SADL under Euclidean parameter
gradient flow. r=log(q+eps)-log(k+eps)>0. g2 is the squared parameter
supply gradient, den=k+eps>0, b=<grad k, grad L0>. w also includes changing
stopped references. The log chain rule gives dr=w-damping*r.
Paper excitation-energy/spectral propositions are inherited, not re-proved.
-/
namespace CGB.SupplyDemandDynamics

theorem log_deficit_derivative {q k : ℝ → ℝ} {dq dk x eps : ℝ}
    (hq : HasDerivAt q dq x) (hk : HasDerivAt k dk x)
    (hqp : 0 < q x+eps) (hkp : 0 < k x+eps) :
    HasDerivAt (fun t => Real.log (q t+eps)-Real.log (k t+eps))
      (dq/(q x+eps)-dk/(k x+eps)) x := by
  exact ((hq.add_const eps).log (ne_of_gt hqp)).sub
    ((hk.add_const eps).log (ne_of_gt hkp))

theorem local_supply_rate (b lam r g2 den : ℝ) :
    -b + (2*lam*r/den)*g2 = -b + (2*lam*g2/den)*r := by ring

theorem drift_damping_decomposition (dq qden b ref lam r g2 den : ℝ)
    (hd : den ≠ 0) :
    dq/qden-((-b+2*lam*r*g2/den)+ref)/den =
      (dq/qden+(b-ref)/den)-(2*lam*g2/den^2)*r := by
  field_simp [hd]
  ring

theorem deficit_power_balance (r dr w damping : ℝ)
    (h : dr = w-damping*r) :
    r*dr = r*w-damping*r^2 := by rw [h]; ring

theorem frozen_reference_dissipation (r dr damping : ℝ)
    (hd : 0 ≤ damping) (h : dr = -damping*r) : r*dr ≤ 0 := by
  rw [h]
  nlinarith [sq_nonneg r]

theorem moving_reference_power_bound (r dr w bound damping floor : ℝ)
    (hr : 0 ≤ r) (hw : w ≤ bound) (hd : floor ≤ damping)
    (h : dr = w-damping*r) :
    r*dr ≤ r*bound-floor*r^2 := by
  rw [deficit_power_balance r dr w damping h]
  have h₁ := mul_le_mul_of_nonneg_left hw hr
  have h₂ := mul_le_mul_of_nonneg_right hd (sq_nonneg r)
  linarith

theorem positive_damping (lam g2 den : ℝ) (hl : 0 < lam)
    (hg : 0 < g2) (hd : den ≠ 0) : 0 < 2*lam*g2/den^2 := by positivity

end CGB.SupplyDemandDynamics
