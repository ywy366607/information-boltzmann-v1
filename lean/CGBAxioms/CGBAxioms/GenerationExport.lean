import CGBAxioms.Generation

/-!
# Export and selective export (generation chain, G16-G17)

P1a makes the body finite; P2a gives the content ledger
`content (t+1) + out t = content t + in t` (`MaterialLedger.balance`). The life
definition needs intake that does not stop (`not_persists_without_incoming`).

* G16: a finite body that keeps taking in must export at the same long-run rate;
  a write-only body takes in at most its capacity over its whole life.
* G17: if export does not depend on what a unit holds, a stored item survives
  `n` exports with probability at most `(1 - q₀)^n → 0`; keeping an item beyond
  the turnover time needs its export probability to go to zero, i.e. to depend
  on the unit's state. By G1/G1E a rate depends on state only through local
  energy differences, so the selection is by energy.
-/

namespace CGB.Generation

open Finset

/-- **G16a (export is forced)**: with the content ledger and a capacity bound,
cumulative export is at least cumulative intake minus the capacity. -/
theorem export_forced (c inn out : ℕ → ℕ) (cap : ℕ)
    (ledger : ∀ t, c (t + 1) + out t = c t + inn t) (hcap : ∀ t, c t ≤ cap) (T : ℕ) :
    (∑ t ∈ range T, inn t) ≤ (∑ t ∈ range T, out t) + cap := by
  have key : ∀ T, c T + ∑ t ∈ range T, out t = c 0 + ∑ t ∈ range T, inn t := by
    intro T
    induction T with
    | zero => simp
    | succ n ih =>
      rw [sum_range_succ, sum_range_succ]
      have := ledger n
      omega
  have := key T
  have := hcap T
  omega

/-- **G16b (write-only bodies starve)**: with no export, total lifetime intake is at
most the capacity. -/
theorem write_only_intake_bounded (c inn : ℕ → ℕ) (cap : ℕ)
    (ledger : ∀ t, c (t + 1) = c t + inn t) (hcap : ∀ t, c t ≤ cap) (T : ℕ) :
    (∑ t ∈ range T, inn t) ≤ cap := by
  have := export_forced c inn (fun _ => 0) cap (fun t => by simpa using ledger t) hcap T
  simpa using this

/-- **G16c**: if intake does not stop (its total exceeds every bound), export does not
stop either. -/
theorem endless_intake_endless_export (c inn out : ℕ → ℕ) (cap : ℕ)
    (ledger : ∀ t, c (t + 1) + out t = c t + inn t) (hcap : ∀ t, c t ≤ cap)
    (hin : ∀ B, ∃ T, B < ∑ t ∈ range T, inn t) (B : ℕ) :
    ∃ T, B < ∑ t ∈ range T, out t := by
  obtain ⟨T, hT⟩ := hin (B + cap)
  have := export_forced c inn out cap ledger hcap T
  exact ⟨T, by omega⟩

/-- Survival probability of one stored item over `n` exports, the `k`-th removing it
with probability `q k`. -/
def survive (q : ℕ → ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => survive q n * (1 - q n)

/-- **G17a (unselective export forgets)**: if every export removes the item with
probability at least `q₀ > 0` (for example `1/C` when the exported unit is drawn
without regard to what it holds), survival is at most `(1 - q₀)^n`. -/
theorem survive_le_pow (q : ℕ → ℝ) (q₀ : ℝ) (hq : ∀ k, q₀ ≤ q k) (h1 : ∀ k, q k ≤ 1) (n : ℕ) :
    survive q n ≤ (1 - q₀) ^ n := by
  have hs : ∀ n, 0 ≤ survive q n := by
    intro n; induction n with
    | zero => simp [survive]
    | succ n ih => simp only [survive]; exact mul_nonneg ih (by linarith [h1 n])
  induction n with
  | zero => simp [survive]
  | succ n ih =>
    simp only [survive, pow_succ]
    exact mul_le_mul ih (by linarith [hq n]) (by linarith [h1 n]) ((hs n).trans ih)

theorem unselective_export_forgets (q : ℕ → ℝ) (q₀ : ℝ) (h0 : 0 < q₀)
    (hq : ∀ k, q₀ ≤ q k) (h1 : ∀ k, q k ≤ 1) (ε : ℝ) (hε : 0 < ε) :
    ∃ n, survive q n < ε := by
  have hq1 : q₀ ≤ 1 := (hq 0).trans (h1 0)
  obtain ⟨n, hn⟩ := exists_pow_lt_of_lt_one hε (by linarith : 1 - q₀ < 1)
  exact ⟨n, (survive_le_pow q q₀ hq h1 n).trans_lt hn⟩

/-- **G17b (retention needs selectivity)**: if an item must survive every number of
exports with probability at least `p > 0`, its per-export removal probability has
no positive lower bound: export must eventually spare it, so it cannot be drawn
without regard to state (whose removal probability is a fixed `1/C`). -/
theorem retention_needs_selectivity (q : ℕ → ℝ) (h1 : ∀ k, q k ≤ 1) (p : ℝ) (hp : 0 < p)
    (hret : ∀ n, p ≤ survive q n) : ∀ q₀ > 0, ∃ k, q k < q₀ := by
  intro q₀ h0
  by_contra hcon
  simp only [not_exists, not_lt] at hcon
  obtain ⟨n, hn⟩ := unselective_export_forgets q q₀ h0 hcon h1 p hp
  exact absurd (hret n) (not_le.mpr hn)

end CGB.Generation
