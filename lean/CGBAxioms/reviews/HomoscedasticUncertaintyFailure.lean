import Init

/-! Formal analysis of when Kendall-style homoscedastic uncertainty weighting
  fails or loses robustness in multi-objective language + regularizer training.

  In continuous-variable form (Kendall et al. 2018), task losses L_1, L_2 > 0
  are combined via:
      F(w, s_1, s_2) = (1/2) * (e^{-s_1} L_1 + s_1 + e^{-s_2} L_2 + s_2)
  At the exact stationary point e^{s_k^*} = L_k, the effective gradient weights
  are lambda_k^* = e^{-s_k^*} = 1 / L_k.

  To keep the Lean 4 proof zero-dependency (`import Init`) and exact over
  positive rationals/naturals (avoiding floating-point approximations), we
  prove the exact structural identities and counterexamples on the inverse-loss
  weight law `lambda_k = 1 / L_k` (expressed cross-multiplied over Nat/Int)
  and on the bounded-clamping degeneration.

  Five structural results:
  1. `multiplicative_scale_invariance`: Rescaling L_k -> c * L_k cancels from
     the weighted gradient term (1 / (c * L_k)) * (c * g_k) = g_k / L_k.
  2. `zero_floor_hijacks_entropy_floor_loss`: When task 1 (language CE) has an
     irreducible entropy floor H_0 > 0 (so L_1 >= H_0) while task 2 (SIGReg /
     MSE) shrinks to L_2 <= eps with H_0 >= M * eps, the effective weight ratio
     lambda_2 / lambda_1 = L_1 / L_2 is at least M (unboundedly large as eps -> 0).
  3. `additive_floor_breaks_invariance`: Adding an irreducible entropy floor
     H_0 > 0 strictly depresses the effective weight on task 1's reducible
     excess Delta = L_1 - H_0 relative to a zero-floor loss of equal reducible
     excess Delta.
  4. `signed_loss_unbounded_below`: If any loss term satisfies L <= 0 (e.g.
     signed policy score or shifted log-likelihood), scaling by an arbitrary
     positive weight w > 0 makes w * L arbitrarily negative (no finite optimum).
  5. `clamping_degenerates_to_manual_weight`: If log-variance is clamped from
     below by s_min (weight capped at W_max = e^{-s_min}), then whenever
     L < 1 / W_max the optimal weight pins at W_max, reducing back to a fixed
     manual weight.
-/
namespace CGB.HomoscedasticUncertaintyFailure

/-- Exact cross-multiplied comparison of effective weights `w1 = 1/L1` vs `w2 = 1/L2`:
    `w2 >= M * w1` iff `L1 >= M * L2` (for positive L1, L2). -/
def weightRatioAtLeast (L1 L2 M : Nat) : Prop :=
  M * L2 ≤ L1

/-- 1. Pure multiplicative rescaling preserves relative gradient magnitudes:
       comparing `(c * g1) / (c * L1)` vs `g1 / L1` (cross-multiplied). -/
theorem multiplicative_scale_invariance (c L1 g1 : Int) (hc : c ≠ 0) :
    (c * g1) * L1 = g1 * (c * L1) := by
  have _ := hc
  rw [Int.mul_comm c g1, Int.mul_assoc]

/-- 2. Zero-floor regularizer (SIGReg / MSE -> eps) hijacks an entropy-floored
       loss (language CE >= H0) by an arbitrarily large factor M. -/
theorem zero_floor_hijacks_entropy_floor_loss
    (L_ce H0 L_reg eps M : Nat)
    (h_ce_floor : H0 ≤ L_ce)
    (h_reg_small : L_reg ≤ eps)
    (h_gap : M * eps ≤ H0) :
    weightRatioAtLeast L_ce L_reg M := by
  unfold weightRatioAtLeast
  have hmul : M * L_reg ≤ M * eps := Nat.mul_le_mul_left M h_reg_small
  omega

/-- 3. Additive entropy floor H0 > 0 strictly breaks invariance:
       Even when reducible excess `delta` is identical for CE (`H0 + delta`)
       and regularizer (`delta`), the regularizer receives strictly larger
       effective weight `1/delta > 1/(H0 + delta)` (cross-multiplied: `delta < H0 + delta`). -/
theorem additive_floor_breaks_invariance (H0 delta : Nat)
    (hH0 : 0 < H0) :
    delta < H0 + delta := by
  omega

/-- 4. Negative / signed loss (L < 0) makes `w * L` unbounded below for
       arbitrarily large precision `w > 0` (no finite optimal variance). -/
theorem signed_loss_unbounded_below (L : Int) (hneg : L < 0) (B : Int) :
    ∃ w : Int, w > 0 ∧ w * L ≤ B := by
  refine ⟨if B < 0 then -B + 1 else 1, ?_, ?_⟩
  · split <;> omega
  · split
    · rename_i hB
      have h1 : L ≤ -1 := by omega
      have h2 : (-B + 1) > 0 := by omega
      have hmul : (-B + 1) * L ≤ (-B + 1) * (-1) :=
        Int.mul_le_mul_of_nonneg_left h1 (by omega)
      omega
    · rename_i hB
      have h1 : 1 * L = L := by omega
      omega

/-- Clamped optimal weight rule (precision capped at `W_max` when `L` drops
    below the floor `L_min`, represented via cross-multiplication `W_max * L < 1`
    in scaled units `scale > 0`: `W_max * L < scale`). -/
def clampedWeight (W_max L scale : Nat) : Nat :=
  if W_max * L < scale then W_max else scale / L

/-- 5. Whenever a regularizer drops below the clamped variance floor
       (`W_max * L < scale`), clamped homoscedastic weighting degenerates
       identically to the fixed manual weight `W_max`. -/
theorem clamping_degenerates_to_manual_weight (W_max L scale : Nat)
    (hsmall : W_max * L < scale) :
    clampedWeight W_max L scale = W_max := by
  unfold clampedWeight
  simp [hsmall]

end CGB.HomoscedasticUncertaintyFailure
