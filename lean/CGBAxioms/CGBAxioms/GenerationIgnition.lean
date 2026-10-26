import CGBAxioms.GenerationGrowth

/-!
# Ignition needs drive or interaction (generation chain, G23)

Ignition (global-workspace theory) is all-or-none, self-sustaining activation: the number
of active units has two peaks (off and ignited) with a valley between them. In equilibrium
with additive single-unit energies (the energy of M2/M3 media), `N` identical two-state units
at a fixed reservoir potential have `k` active with weight `choose N k * r ^ k`. That sequence
is log-concave, so it has no valley: no ignition. Ignition therefore needs either a
non-additive interaction energy (the `κ * a * b` term of G18, carried inside a unit that holds
both ends) or a drive that keeps the dynamics away from equilibrium (an autocatalytic loop fed
by a boundary reservoir, P3). Its end needs a slow negative feedback: reservoir depletion (G22).
-/

namespace CGB.Generation

/-- Binomial coefficients are log-concave. -/
theorem choose_log_concave (N k : ℕ) :
    N.choose k * N.choose (k + 2) ≤ N.choose (k + 1) ^ 2 := by
  have h1 := Nat.choose_succ_right_eq N k
  have h2 := Nat.choose_succ_right_eq N (k + 1)
  rcases Nat.lt_or_ge N (k + 1) with hN | hN
  · have : N.choose (k + 2) = 0 := Nat.choose_eq_zero_of_lt (by omega)
    simp [this]
  · obtain ⟨m, rfl⟩ : ∃ m, N = k + 1 + m := ⟨N - (k + 1), by omega⟩
    have e1 : k + 1 + m - k = m + 1 := by omega
    have e2 : k + 1 + m - (k + 1) = m := by omega
    rw [e1] at h1
    rw [e2] at h2
    -- h1 : C(k+1) * (k+1) = C(k) * (m+1),  h2 : C(k+2) * (k+2) = C(k+1) * m
    set a := (k + 1 + m).choose k
    set b := (k + 1 + m).choose (k + 1)
    set c := (k + 1 + m).choose (k + 2)
    have hm : m * (k + 1) ≤ (m + 1) * (k + 2) := by nlinarith
    have key : a * c * ((k + 1) * (k + 2)) ≤ b ^ 2 * ((k + 1) * (k + 2)) :=
      calc a * c * ((k + 1) * (k + 2)) = a * b * (m * (k + 1)) := by
              rw [show a * c * ((k + 1) * (k + 2)) = a * (k + 1) * (c * (k + 2)) by ring, h2]; ring
        _ ≤ a * b * ((m + 1) * (k + 2)) := Nat.mul_le_mul_left _ hm
        _ = b ^ 2 * ((k + 1) * (k + 2)) := by
              rw [show b ^ 2 * ((k + 1) * (k + 2)) = b * (k + 1) * (b * (k + 2)) by ring, h1]; ring
    exact Nat.le_of_mul_le_mul_right key (Nat.mul_pos (Nat.succ_pos _) (Nat.succ_pos _))

/-- Weight of `k` active units among `N` at activity ratio `r`. -/
noncomputable def activeWeight (N : ℕ) (r : ℝ) (k : ℕ) : ℝ := (N.choose k : ℝ) * r ^ k

theorem activeWeight_log_concave (N : ℕ) {r : ℝ} (hr : 0 ≤ r) (k : ℕ) :
    activeWeight N r k * activeWeight N r (k + 2) ≤ activeWeight N r (k + 1) ^ 2 := by
  unfold activeWeight
  have hc : ((N.choose k : ℝ) * N.choose (k + 2)) ≤ (N.choose (k + 1) : ℝ) ^ 2 := by
    exact_mod_cast choose_log_concave N k
  have hp : 0 ≤ r ^ (2 * k + 2) := pow_nonneg hr _
  calc (N.choose k : ℝ) * r ^ k * ((N.choose (k + 2) : ℝ) * r ^ (k + 2))
      = ((N.choose k : ℝ) * N.choose (k + 2)) * r ^ (2 * k + 2) := by ring
    _ ≤ (N.choose (k + 1) : ℝ) ^ 2 * r ^ (2 * k + 2) := mul_le_mul_of_nonneg_right hc hp
    _ = ((N.choose (k + 1) : ℝ) * r ^ (k + 1)) ^ 2 := by ring

/-- **G23 (no ignition in equilibrium with additive energies)**: the distribution of the
number of active units has no strict valley, so it cannot have an "off" peak and an
"ignited" peak separated by a gap. -/
theorem no_valley (N : ℕ) {r : ℝ} (hr : 0 ≤ r) (k : ℕ) :
    ¬ (activeWeight N r (k + 1) < activeWeight N r k ∧
       activeWeight N r (k + 1) < activeWeight N r (k + 2)) := by
  rintro ⟨h1, h2⟩
  have h0 : 0 ≤ activeWeight N r (k + 1) := by
    unfold activeWeight; positivity
  have := activeWeight_log_concave N hr k
  nlinarith

/-! ## G24: an equilibrium memory writes and forgets at tied rates

A two-state unit with energy gap `Δ` and symmetric prefactor `κ` relaxes at the total rate
`κ (e^{-Δ/2} + e^{Δ/2})`. A stimulus enters only as reservoir work `μ` (P3), shifting the gap
to `Δ - μ`. The write rate under the stimulus is then at most `e^{|μ|/2}` times the forget rate
without it, whatever `κ` and `Δ` are. After a pulse of length `tp` and a delay `T` the retained
deviation is at most `e^{|μ|/2 - 1} tp / T`: weak brief stimuli cannot be held over long
delays by equilibrium relaxation. Holding them needs a driven latch.
-/

/-- Total relaxation rate of a two-state unit with gap `Δ`. -/
noncomputable def relaxRate (κ Δ : ℝ) : ℝ := κ * (Real.exp (-Δ / 2) + Real.exp (Δ / 2))

theorem pulse_rate_le {κ : ℝ} (hκ : 0 ≤ κ) (Δ μ : ℝ) :
    relaxRate κ (Δ - μ) ≤ Real.exp (|μ| / 2) * relaxRate κ Δ := by
  unfold relaxRate
  have e1 : Real.exp (-(Δ - μ) / 2) ≤ Real.exp (|μ| / 2) * Real.exp (-Δ / 2) := by
    rw [← Real.exp_add]; apply Real.exp_le_exp.mpr
    have := le_abs_self μ; linarith
  have e2 : Real.exp ((Δ - μ) / 2) ≤ Real.exp (|μ| / 2) * Real.exp (Δ / 2) := by
    rw [← Real.exp_add]; apply Real.exp_le_exp.mpr
    have := neg_abs_le μ; linarith
  calc κ * (Real.exp (-(Δ - μ) / 2) + Real.exp ((Δ - μ) / 2))
      ≤ κ * (Real.exp (|μ| / 2) * Real.exp (-Δ / 2) + Real.exp (|μ| / 2) * Real.exp (Δ / 2)) :=
        mul_le_mul_of_nonneg_left (add_le_add e1 e2) hκ
    _ = Real.exp (|μ| / 2) * (κ * (Real.exp (-Δ / 2) + Real.exp (Δ / 2))) := by ring

/-- **G24 (equilibrium memory bound)**: loading at rate `lp ≤ e^{|μ|/2} l0` for time `tp`,
then relaxing at rate `l0` for time `T`, keeps at most `e^{|μ|/2 - 1} tp / T` of the
deviation, for every choice of `l0`. -/
theorem equilibrium_memory_bound {l0 lp tp T μ : ℝ}
    (htp : 0 ≤ tp) (hT : 0 < T) (hp : lp ≤ Real.exp (|μ| / 2) * l0) :
    (1 - Real.exp (-(lp * tp))) * Real.exp (-(l0 * T)) ≤ Real.exp (|μ| / 2 - 1) * tp / T := by
  have hw : 1 - Real.exp (-(lp * tp)) ≤ lp * tp := by
    have := Real.add_one_le_exp (-(lp * tp)); linarith
  have hd : 0 < Real.exp (-(l0 * T)) := Real.exp_pos _
  -- x e^{-x} ≤ e^{-1}
  have hx : l0 * T * Real.exp (-(l0 * T)) ≤ Real.exp (-1) := by
    have h := Real.add_one_le_exp (l0 * T - 1)
    have h2 : Real.exp (l0 * T - 1) * Real.exp (-(l0 * T)) = Real.exp (-1) := by
      rw [← Real.exp_add]; ring_nf
    nlinarith [Real.exp_pos (-(l0 * T))]
  have hm : Real.exp (|μ| / 2 - 1) = Real.exp (|μ| / 2) * Real.exp (-1) := by
    rw [← Real.exp_add]; ring_nf
  have hE : 0 ≤ Real.exp (|μ| / 2) := (Real.exp_pos _).le
  rw [hm, le_div_iff₀ hT]
  calc (1 - Real.exp (-(lp * tp))) * Real.exp (-(l0 * T)) * T
      ≤ lp * tp * Real.exp (-(l0 * T)) * T := by gcongr
    _ ≤ Real.exp (|μ| / 2) * l0 * tp * Real.exp (-(l0 * T)) * T := by gcongr
    _ = Real.exp (|μ| / 2) * tp * (l0 * T * Real.exp (-(l0 * T))) := by ring
    _ ≤ Real.exp (|μ| / 2) * tp * Real.exp (-1) := by gcongr
    _ = Real.exp (|μ| / 2) * Real.exp (-1) * tp := by ring

end CGB.Generation
