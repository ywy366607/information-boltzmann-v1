import CGBAxioms.GenerationSelfProduction
import CGBAxioms.EnergyBudget

/-!
# Conditional maintenance, not immortality
Maintenance means longer expected structural lifetime with drive than without
(`CGB_AXIOMS_V0_20261022.md`, section 4). The lifetime comparison needs no all-time
floor; the cumulative-risk budget is a separate, stronger sufficient condition.
An infinite input stream does not imply an immortal individual. P1--P3 supply no
uniform block loss: prove the local bounds for the actual processes.
For continuity, use absorbing death/first-hit labels: `aliveInd` can include revival.
`Medium.hold_vanishes` only concerns leaving one microstate, not death.
No `Life` change, `completeLives` claim, or GDN/physical-world premise is asserted.
-/

open Finset

namespace CGB.SurvivalBounds

/-- Reuse the existing geometric bound; no nonnegativity of `q` is needed for
this upper-tail conclusion. For survival, each index can denote a completed block. -/
theorem decay_eventually_lt (q : ℕ → ℝ) (p : ℝ) (hp : 0 < p) (hp1 : p ≤ 1)
    (hq0 : q 0 ≤ 1) (hrec : ∀ n, q (n + 1) ≤ (1 - p) * q n)
    (τ : ℝ) (hτ : 0 < τ) : ∃ N, ∀ n, N ≤ n → q n < τ := by
  obtain ⟨N, hN⟩ := exists_pow_lt_of_lt_one hτ (by linarith : 1 - p < 1)
  refine ⟨N, fun n hn => ?_⟩
  exact (SelfProduction.uniform_block_learning_failure_decay p hp1 q hq0 hrec n).trans_lt
    ((pow_le_pow_of_le_one (sub_nonneg.mpr hp1) (by linarith) hn).trans_lt hN)

/-- Uniform positive irreversible block loss rules out a fixed positive floor. -/
theorem no_uniform_positive_lower (q : ℕ → ℝ) (p : ℝ) (hp : 0 < p) (hp1 : p ≤ 1)
    (hq0 : q 0 ≤ 1) (hrec : ∀ n, q (n + 1) ≤ (1 - p) * q n)
    (τ : ℝ) (hτ : 0 < τ) : ¬ ∀ n, τ ≤ q n := by
  obtain ⟨N, hN⟩ := decay_eventually_lt q p hp hp1 hq0 hrec τ hτ
  exact fun h => (not_le.mpr (hN N le_rfl)) (h N)

/-- Lower counterpart to the reused geometric upper bound. -/
theorem geometric_lower (q : ℕ → ℝ) (p : ℝ) (hp1 : p ≤ 1) (hq0 : 1 ≤ q 0)
    (hrec : ∀ n, (1 - p) * q n ≤ q (n + 1)) : ∀ n, (1 - p) ^ n ≤ q n := by
  intro n
  induction n with
  | zero => simpa using hq0
  | succ n ih =>
    calc
      (1 - p) ^ (n + 1) = (1 - p) * (1 - p) ^ n := by rw [pow_succ]; ring
      _ ≤ (1 - p) * q n := mul_le_mul_of_nonneg_left ih (sub_nonneg.mpr hp1)
      _ ≤ q (n + 1) := hrec n

/-- Telescoping lower bound; no independence, absorption, or sign premise is
needed for this algebraic statement. Nonnegative `risk` gives its risk reading. -/
theorem cumulative_risk_lower (q risk : ℕ → ℝ)
    (hstep : ∀ n, q n - risk n ≤ q (n + 1)) (n : ℕ) :
    q 0 - ∑ k ∈ range n, risk k ≤ q n := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [sum_range_succ]
    linarith [hstep n]

/-- A prefix budget allows strictly positive but summably bounded per-step risk.
It does not require `δ < 1`; that premise is needed only for a positive floor. -/
theorem risk_budget_uniform_lower (q risk : ℕ → ℝ) (δ : ℝ) (hq0 : q 0 = 1)
    (hstep : ∀ n, q n - risk n ≤ q (n + 1))
    (hbudget : ∀ n, (∑ k ∈ range n, risk k) ≤ δ) : ∀ n, 1 - δ ≤ q n := by
  intro n
  have h := cumulative_risk_lower q risk hstep n
  rw [hq0] at h
  linarith [hbudget n]

/-! ### Finite-window transfer
The errors below concern the registered life/continuity observable under each
process's own closed-loop interaction, not merely NLL or a forcibly shared input
sequence. No finite code space or physical refinement follows automatically;
whole-program admissibility and real source/energy certificates remain obligations.
The window `T` refines expected quantities, not training truncation or state reset.
No clock/reserve state, globally exact infinite-precision trajectory implementation,
or infinite-horizon all-state contraction is required by the transfer theorem.
-/

/-- Uniform scalar decay bounds every finite cut tail, reusing the existing
pointwise geometric bound rather than repeating its power induction. -/
theorem geometric_tail_bound (q : ℕ → ℝ) (p : ℝ) (hp : 0 < p) (hp1 : p ≤ 1)
    (hq0 : q 0 ≤ 1) (hrec : ∀ n, q (n + 1) ≤ (1 - p) * q n) (T L : ℕ) :
    (∑ k ∈ range L, q (T + k)) ≤ (1 - p) ^ T / p := by
  have hgeom : (∑ k ∈ range L, (1 - p) ^ k) * p = 1 - (1 - p) ^ L := by
    simpa using geom_sum_mul_neg (1 - p) L
  calc
    _ ≤ ∑ k ∈ range L, (1 - p) ^ (T + k) := Finset.sum_le_sum fun k _ =>
      SelfProduction.uniform_block_learning_failure_decay p hp1 q hq0 hrec (T + k)
    _ = (1 - p) ^ T * ∑ k ∈ range L, (1 - p) ^ k := by
      simp only [pow_add, Finset.mul_sum]
    _ ≤ (1 - p) ^ T / p := by
      apply (le_div_iff₀ hp).2
      rw [mul_assoc, hgeom]
      calc
        _ ≤ (1 - p) ^ T * 1 := mul_le_mul_of_nonneg_left
          (by linarith [pow_nonneg (sub_nonneg.mpr hp1) L])
          (pow_nonneg (sub_nonneg.mpr hp1) T)
        _ = (1 - p) ^ T := mul_one _

/-- A prefix advantage exceeding the entire cut-tail allowance persists at all
later horizons; normal needs only nonnegative scores, not contraction. -/
theorem prefix_gap_persists (normal cut : ℕ → ℝ) (T : ℕ) (tailBound : ℝ)
    (hnormal : ∀ n, 0 ≤ normal n)
    (htail : ∀ L, (∑ k ∈ range L, cut (T + k)) ≤ tailBound)
    (hgap : tailBound < (∑ t ∈ range T, normal t) - ∑ t ∈ range T, cut t) :
    ∀ N, T ≤ N → (∑ t ∈ range N, cut t) < ∑ t ∈ range N, normal t := by
  intro N hN
  obtain ⟨L, rfl⟩ := Nat.exists_eq_add_of_le hN
  have hn : 0 ≤ ∑ k ∈ range L, normal (T + k) :=
    Finset.sum_nonneg fun k _ => hnormal (T + k)
  simp only [Finset.sum_range_add]
  linarith [htail L]

/-- Finite-window cumulative errors plus an actual cut-tail certificate transfer
reference lifetime advantage to every actual horizon beyond the window. -/
theorem finite_window_lifetime_transfer (qN qC rN rC : ℕ → ℝ)
    (T : ℕ) (E_N E_C tailBound : ℝ)
    (hnormal : ∀ n, 0 ≤ rN n)
    (htail : ∀ L, (∑ k ∈ range L, rC (T + k)) ≤ tailBound)
    (herrorN : |(∑ t ∈ range T, rN t) - ∑ t ∈ range T, qN t| ≤ E_N)
    (herrorC : |(∑ t ∈ range T, rC t) - ∑ t ∈ range T, qC t| ≤ E_C)
    (hgap : E_N + E_C + tailBound <
      (∑ t ∈ range T, qN t) - ∑ t ∈ range T, qC t) :
    ∀ N, T ≤ N → (∑ t ∈ range N, rC t) < ∑ t ∈ range N, rN t := by
  apply prefix_gap_persists rN rC T tailBound hnormal htail
  have hn := (abs_le.mp herrorN).1
  have hc := (abs_le.mp herrorC).2
  linarith

section Process

variable {S O : Type*} [Fintype S] [DecidableEq S] [Fintype O]

/-- General expected-step identity extracted from the rearrangement in
`Energetics.expected_first_law`. The observable is not an energy `Account`. -/
theorem expected_step (P : Interaction.Machine S ℕ O) (initial : S) (v : S → ℝ)
    (t : ℕ) :
    Energetics.mean (Energetics.law P initial (t + 1)) v =
      ∑ s, Energetics.law P initial t s * (∑ o, ∑ s', P.step t s o s' * v s') := by
  simp only [Energetics.law, EventChannels.Kernel.stateLaw,
    EventChannels.Kernel.forget, Energetics.events, Energetics.mean]
  simp only [Finset.sum_mul, mul_assoc, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro s _
  rw [Finset.sum_comm]

theorem mean_initial (P : Interaction.Machine S ℕ O) (initial : S) (v : S → ℝ) :
    Energetics.mean (Energetics.law P initial 0) v = v initial := by
  classical
  simp [Energetics.mean, Energetics.law, EventChannels.Kernel.stateLaw]

/-- A bounded observable gives a bounded score under the actual process law. -/
theorem mean_unit_interval (P : Interaction.Machine S ℕ O) (initial : S)
    (v : S → ℝ) (hv : ∀ s, 0 ≤ v s ∧ v s ≤ 1) (t : ℕ) :
    0 ≤ Energetics.mean (Energetics.law P initial t) v ∧
      Energetics.mean (Energetics.law P initial t) v ≤ 1 := by
  obtain ⟨hn, hsum⟩ := Energetics.law_probability P initial t
  constructor
  · exact Finset.sum_nonneg fun s _ => mul_nonneg (hn s) (hv s).1
  · calc
      _ ≤ ∑ s, Energetics.law P initial t s * 1 :=
        Finset.sum_le_sum fun s _ => mul_le_mul_of_nonneg_left (hv s).2 (hn s)
      _ = 1 := by simpa using hsum

/-- A local row contraction transfers to the original ensemble score. -/
theorem mean_contraction (P : Interaction.Machine S ℕ O) (initial : S)
    (v : S → ℝ) (p : ℝ) (t : ℕ)
    (hrow : ∀ s, (∑ o, ∑ s', P.step t s o s' * v s') ≤ (1 - p) * v s) :
    Energetics.mean (Energetics.law P initial (t + 1)) v ≤
      (1 - p) * Energetics.mean (Energetics.law P initial t) v := by
  rw [expected_step]
  calc
    _ ≤ ∑ s, Energetics.law P initial t s * ((1 - p) * v s) :=
      Finset.sum_le_sum fun s _ => mul_le_mul_of_nonneg_left (hrow s)
        ((Energetics.law_probability P initial t).1 s)
    _ = _ := by simp only [Energetics.mean, Finset.mul_sum]; congr 1; funext s; ring

/-- A local additive risk allowance transfers without independence assumptions. -/
theorem mean_risk_step (P : Interaction.Machine S ℕ O) (initial : S)
    (v : S → ℝ) (risk : ℝ) (t : ℕ)
    (hrow : ∀ s, v s - risk ≤ ∑ o, ∑ s', P.step t s o s' * v s') :
    Energetics.mean (Energetics.law P initial t) v - risk ≤
      Energetics.mean (Energetics.law P initial (t + 1)) v := by
  rw [expected_step]
  calc
    _ = ∑ s, Energetics.law P initial t s * (v s - risk) := by
      simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, Energetics.mean,
        (Energetics.law_probability P initial t).2, one_mul]
    _ ≤ _ := Finset.sum_le_sum fun s _ => mul_le_mul_of_nonneg_left (hrow s)
      ((Energetics.law_probability P initial t).1 s)

/-- Lower multiplicative row bounds transfer along the same actual law. -/
theorem mean_retention_step (P : Interaction.Machine S ℕ O) (initial : S)
    (v : S → ℝ) (p : ℝ) (t : ℕ)
    (hrow : ∀ s, (1 - p) * v s ≤ ∑ o, ∑ s', P.step t s o s' * v s') :
    (1 - p) * Energetics.mean (Energetics.law P initial t) v ≤
      Energetics.mean (Energetics.law P initial (t + 1)) v := by
  rw [expected_step]
  calc
    _ = ∑ s, Energetics.law P initial t s * ((1 - p) * v s) := by
      simp only [Energetics.mean, Finset.mul_sum]; congr 1; funext s; ring
    _ ≤ _ := Finset.sum_le_sum fun s _ => mul_le_mul_of_nonneg_left (hrow s)
      ((Energetics.law_probability P initial t).1 s)

/-- Truncated expected lifetime for a continuity indicator with absorbing death;
for fractional `a`, this is instead the corresponding weighted duration. -/
noncomputable def expectedLifetime (P : Interaction.Machine S ℕ O) (initial : S)
    (a : S → ℝ) (T : ℕ) : ℝ :=
  ∑ t ∈ range T, Energetics.mean (Energetics.law P initial t) a

/-- Positive maintenance without immortality: the same initial individual and
observable have strictly longer truncated lifetime under the normal process.
The advantage is strict at tick 1 and nonnegative at every tick. -/
theorem expectedLifetime_strict_of_row_gap (normal cut : Interaction.Machine S ℕ O)
    (initial : S) (a : S → ℝ) (ha : ∀ s, 0 ≤ a s ∧ a s ≤ 1) (hinit : a initial = 1)
    (pN pC : ℝ) (_hpN : 0 ≤ pN) (hgap : pN < pC) (hpC1 : pC ≤ 1)
    (hnormal : ∀ t s, (1 - pN) * a s ≤ ∑ o, ∑ s', normal.step t s o s' * a s')
    (hcut : ∀ t s, (∑ o, ∑ s', cut.step t s o s' * a s') ≤ (1 - pC) * a s)
    (T : ℕ) (hT : 2 ≤ T) :
    expectedLifetime cut initial a T < expectedLifetime normal initial a T := by
  have hpN1 : pN ≤ 1 := hgap.le.trans hpC1
  have hN := geometric_lower (fun t => Energetics.mean (Energetics.law normal initial t) a)
    pN hpN1 (by rw [mean_initial, hinit])
    (fun t => mean_retention_step normal initial a pN t (hnormal t))
  have hC := SelfProduction.uniform_block_learning_failure_decay pC hpC1
    (fun t => Energetics.mean (Energetics.law cut initial t) a)
    (mean_unit_interval cut initial a ha 0).2
    (fun t => mean_contraction cut initial a pC t (hcut t))
  have hpoint (t : ℕ) : Energetics.mean (Energetics.law cut initial t) a ≤
      Energetics.mean (Energetics.law normal initial t) a :=
    (hC t).trans ((pow_le_pow_left₀ (sub_nonneg.mpr hpC1) (by linarith) t).trans (hN t))
  have hfirst : Energetics.mean (Energetics.law cut initial 1) a <
      Energetics.mean (Energetics.law normal initial 1) a := by
    have hn := hN 1
    have hc := hC 1
    simp only [pow_one] at hn hc
    linarith
  exact Finset.sum_lt_sum (fun t _ => hpoint t)
    ⟨1, Finset.mem_range.mpr (by omega), hfirst⟩

/-- Stronger, separate all-time-floor criterion: SAME initial state and observable,
normal prefix risk budget versus uniform cut contraction. The row premises must
be verified on those two processes. No infinite-path measure is assumed. -/
theorem normal_cut_eventual_gap (normal cut : Interaction.Machine S ℕ O)
    (initial : S) (v : S → ℝ) (hv : ∀ s, 0 ≤ v s ∧ v s ≤ 1)
    (hinit : v initial = 1) (risk : ℕ → ℝ) (δ p : ℝ) (hδ : δ < 1)
    (hp : 0 < p) (hp1 : p ≤ 1)
    (hbudget : ∀ n, (∑ k ∈ range n, risk k) ≤ δ)
    (hnormal : ∀ t s, v s - risk t ≤ ∑ o, ∑ s', normal.step t s o s' * v s')
    (hcut : ∀ t s, (∑ o, ∑ s', cut.step t s o s' * v s') ≤ (1 - p) * v s) :
    (∀ n, 1 - δ ≤ Energetics.mean (Energetics.law normal initial n) v) ∧
      ∃ N, ∀ n, N ≤ n →
        (1 - δ) / 2 < Energetics.mean (Energetics.law normal initial n) v -
          Energetics.mean (Energetics.law cut initial n) v := by
  have hlower := risk_budget_uniform_lower
    (fun n => Energetics.mean (Energetics.law normal initial n) v) risk δ
    ((mean_initial normal initial v).trans hinit)
    (fun t => mean_risk_step normal initial v (risk t) t (hnormal t)) hbudget
  refine ⟨hlower, ?_⟩
  obtain ⟨N, hN⟩ := decay_eventually_lt
    (fun n => Energetics.mean (Energetics.law cut initial n) v) p hp hp1
    (mean_unit_interval cut initial v hv 0).2
    (fun t => mean_contraction cut initial v p t (hcut t))
    ((1 - δ) / 2) (by linarith)
  exact ⟨N, fun n hn => by linarith [hlower n, hN n hn]⟩

end Process
end CGB.SurvivalBounds
