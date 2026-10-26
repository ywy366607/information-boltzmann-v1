import CGBAxioms.GenerationTable
import CGBAxioms.GenerationExport
import CGBAxioms.Life

/-!
# Life necessities as admission obligations (generation chain, gate layer L)

The axioms P1–P3 say what a medium is. Being alive asks more, and the chain has already
proved several things every living organism must have: it senses (`Lives.exposed`), it
exchanges content (`Lives.metabolizes`), it exports (G16 `export_forced`), its export is
state-selective (G17 `retention_needs_selectivity`), it remembers (`Lives.remembers`, G19).
This module turns the first two into witness lemmas for pair-reaction media, so that every
admitted organism carries a Lean proof of them, and proves that no exact microstate is held
forever (`kernel_self_le`, `hold_vanishes`): memory can only live in sets of states.
-/

open Finset

namespace CGB

namespace Medium

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]
  (M : Medium U Λ A I)

/-- One state-changing event of rate `r` makes the chain leave `x` within a tick with
probability at least `r / R`. -/
theorem kernel_self_le {R : ℝ} (hR : 0 < R) (s : U → I) (x : U → Λ) (a : A)
    (hne : M.act a s x ≠ x) : M.kernel R s x x ≤ 1 - M.rate a s x / R := by
  have h0 : (M.jump s).rate x x = 0 := by
    simp [jump]
  have hle : M.rate a s x ≤ (M.jump s).rate x (M.act a s x) := by
    simp only [jump]
    have := Finset.single_le_sum (f := fun b => if M.act b s x = M.act a s x ∧ x ≠ M.act a s x
      then M.rate b s x else 0) (fun b _ => by split_ifs <;> simp [M.rate_nonneg]) (Finset.mem_univ a)
    simpa [Ne.symm hne] using this
  have hsum : (M.jump s).rate x (M.act a s x) ≤ ∑ z, (M.jump s).rate x z :=
    Finset.single_le_sum (fun z _ => (M.jump s).rate_nonneg x z) (Finset.mem_univ _)
  simp only [kernel, h0, zero_div, zero_add, ite_true]
  have : M.rate a s x / R ≤ (∑ z, (M.jump s).rate x z) / R :=
    div_le_div_of_nonneg_right (hle.trans hsum) hR.le
  linarith

/-- **No exact latch**: if some state-changing event has rate at least `η > 0` under every
stimulus, then the probability of sitting in `x` for `n` ticks under any stimulus sequence
is at most `(1 - η / R) ^ n`, which tends to zero (G17 applied to one microstate). Memory
that lasts must be carried by a set of states, such as the basin of a latch. -/
theorem hold_vanishes {R η : ℝ} (hc : M.Clocked R) (hη : 0 < η) (x : U → Λ)
    (hev : ∀ s, ∃ a, M.act a s x ≠ x ∧ η ≤ M.rate a s x) (sched : ℕ → U → I) (n : ℕ) :
    ∏ t ∈ range n, M.kernel R (sched t) x x ≤ (1 - η / R) ^ n := by
  calc ∏ t ∈ range n, M.kernel R (sched t) x x ≤ ∏ _t ∈ range n, (1 - η / R) := by
        refine Finset.prod_le_prod₀ (fun t _ => M.kernel_nonneg hc _ x x) (fun t _ => ?_)
        obtain ⟨a, hne, ha⟩ := hev (sched t)
        have h1 := M.kernel_self_le hc.1 (sched t) x a hne
        have h2 : η / R ≤ M.rate a (sched t) x / R := div_le_div_of_nonneg_right ha hc.1.le
        linarith
    _ = (1 - η / R) ^ n := by simp

end Medium

namespace PairCRN

variable {U Λ I R : Type*} [Fintype U] [DecidableEq U] [Fintype R] [DecidableEq R]
  (C : PairCRN U Λ I R)

omit [Fintype R] [DecidableEq R] in
/-- Every rule with a positive prefactor fires on every ordered pair of distinct units at a
positive rate, under every stimulus. -/
theorem rate_pos {r : R} (hκ : 0 < C.κ r) {u v : U} (huv : u ≠ v) (s : U → I) (x : U → Λ) :
    0 < C.medium.rate (r, u, v) s x := by
  simp only [medium, huv, ite_false]
  exact mul_pos hκ (Real.exp_pos _)

omit [Fintype R] [DecidableEq R] in
/-- **Life gate L-sense** (`Lives.exposed`): a pair medium is exposed as soon as one rule
with a positive prefactor moves content through a boundary reservoir whose potential differs
between two stimuli. -/
theorem exposed_of_witness [Fintype Λ] [DecidableEq Λ] {r : R} (hκ : 0 < C.κ r) {u v : U}
    (huv : u ≠ v) (s s' : U → I) (x : U → Λ) (hdc : C.dC r u v x ≠ 0)
    (hres : C.reservoir s u v ≠ C.reservoir s' u v) : C.medium.Exposed := by
  refine ⟨(r, u, v), s, s', x, Or.inl fun h => hres ?_⟩
  simp only [medium, huv, ite_false] at h
  have h1 := Real.exp_injective (mul_left_cancel₀ hκ.ne' h)
  simp only [dE] at h1
  have h2 : C.reservoir s u v * C.dC r u v x = C.reservoir s' u v * C.dC r u v x := by linarith
  exact mul_right_cancel₀ hdc h2

/-- The content an event takes in is the change of total content. -/
theorem total_pairAct {r : R} {u v : U} (huv : u ≠ v) (x : U → Λ) :
    (C.medium.total (C.pairAct r u v x) : ℝ) - C.medium.total x = C.dC r u v x := by
  have := C.sum_pairAct (fun _ l => (C.content l : ℝ)) r huv x
  simp only [Medium.total, medium, Nat.cast_sum] at this ⊢
  rw [this]; simp only [dC]

/-- **Life gate L-metabolism** (`Lives.metabolizes`): one content-changing event with a
positive prefactor is a witness. -/
theorem metabolizes_of_witness {r : R} (hκ : 0 < C.κ r) {u v : U} (huv : u ≠ v) (s : U → I)
    (x : U → Λ) (hdc : C.dC r u v x ≠ 0) :
    ∃ a s x, 0 < C.medium.rate a s x ∧ C.medium.total (C.medium.act a s x) ≠ C.medium.total x := by
  refine ⟨(r, u, v), s, x, C.rate_pos hκ huv s x, fun h => hdc ?_⟩
  have h' : (C.medium.total (C.pairAct r u v x) : ℝ) = C.medium.total x := by exact_mod_cast h
  have := C.total_pairAct huv x (r := r)
  linarith

end PairCRN

namespace tableCRN

variable {U Λ I : Type*} [Fintype U] [DecidableEq U] [DecidableEq Λ] {n : ℕ}
  (spec : Fin n → RuleSpec Λ) (ident : U → ℕ) (bid : Finset ℕ) (content : Λ → ℕ)
  (φ : ℕ → Λ → ℝ) (μ : I → ℝ) (birth : ℕ → Λ) (calm : I)

/-- A state in which units `u` and `v` hold the source states of rule `k`. -/
def witness (k : Fin n) (u : U) : U → Λ := Function.update (fun _ => (spec k).a.2) u (spec k).a.1

theorem dC_witness (k : Fin n) {u v : U} (huv : u ≠ v) (hu : ident u = (spec k).i)
    (hv : ident v = (spec k).j) :
    (tableCRN spec ident bid content φ μ birth calm).dC (k, false) u v (witness spec k u) =
      (content (spec k).b.1 : ℝ) + content (spec k).b.2 - content (spec k).a.1 -
        content (spec k).a.2 := by
  have hxu : witness spec k u u = (spec k).a.1 := by simp [witness]
  have hxv : witness spec k u v = (spec k).a.2 := by simp [witness, Ne.symm huv]
  simp only [PairCRN.dC, tableCRN, RuleSpec.apply, hxu, hxv, hu, hv, Bool.false_eq_true,
    if_false, true_and, Prod.mk.eta, if_true]

/-- **L-metabolism for rule tables**: one rule with a positive prefactor whose two sides
carry different content, on two distinct units of its pages. -/
theorem metabolizes_of_rule (k : Fin n) (hκ : 0 < (spec k).κ) {u v : U} (huv : u ≠ v)
    (hu : ident u = (spec k).i) (hv : ident v = (spec k).j)
    (hc : content (spec k).a.1 + content (spec k).a.2 ≠ content (spec k).b.1 + content (spec k).b.2)
    (s : U → I) :
    ∃ a s x, 0 < (tableCRN spec ident bid content φ μ birth calm).medium.rate a s x ∧
      (tableCRN spec ident bid content φ μ birth calm).medium.total
        ((tableCRN spec ident bid content φ μ birth calm).medium.act a s x) ≠
      (tableCRN spec ident bid content φ μ birth calm).medium.total x := by
  refine PairCRN.metabolizes_of_witness _ (r := (k, false)) hκ huv s (witness spec k u) ?_
  rw [dC_witness spec ident bid content φ μ birth calm k huv hu hv]
  intro h
  apply hc
  have h' : ((content (spec k).a.1 + content (spec k).a.2 : ℕ) : ℝ) =
      ((content (spec k).b.1 + content (spec k).b.2 : ℕ) : ℝ) := by push_cast; linarith
  exact_mod_cast h'

/-- **L-sense for rule tables**: the same witness on a rule touching a boundary page, and
two stimuli at different reservoir potentials. -/
theorem exposed_of_rule [Fintype Λ] (k : Fin n) (hκ : 0 < (spec k).κ) {u v : U} (huv : u ≠ v)
    (hu : ident u = (spec k).i) (hv : ident v = (spec k).j)
    (hc : content (spec k).a.1 + content (spec k).a.2 ≠ content (spec k).b.1 + content (spec k).b.2)
    (hb : (spec k).i ∈ bid ∨ (spec k).j ∈ bid) (c c' : I) (hμ : μ c ≠ μ c') :
    (tableCRN spec ident bid content φ μ birth calm).medium.Exposed := by
  refine PairCRN.exposed_of_witness _ (r := (k, false)) hκ huv (fun _ => c) (fun _ => c')
    (witness spec k u) ?_ ?_
  · rw [dC_witness spec ident bid content φ μ birth calm k huv hu hv]
    intro h
    apply hc
    have h' : ((content (spec k).a.1 + content (spec k).a.2 : ℕ) : ℝ) =
        ((content (spec k).b.1 + content (spec k).b.2 : ℕ) : ℝ) := by push_cast; linarith
    exact_mod_cast h'
  · simp only [PairCRN.reservoir, tableCRN, hu, hv]
    rcases hb with hb | hb
    · simpa [hb] using hμ
    · by_cases hi : (spec k).i ∈ bid
      · simpa [hi] using hμ
      · simpa [hi, hb] using hμ

end tableCRN

end CGB
