import CGBAxioms.EnergyBudget
import Mathlib.Data.Fintype.Powerset
import Mathlib.Tactic.DeriveFintype

/-!
# Finite, causal and paid candidate-model learning

Models predict possible boundary outcomes of actions. Observations may exclude
models, but never supply the unknown true-model identity to the controller.
The candidate set and finite write reserve are actual internal state. A depleted
write reserve holds plasticity while the world and perception continue.

This is an interaction/learning interface. A full CGB medium, encoded genome,
physical sensing/computation costs and safe informative exploration still need
their own realization evidence; no organism capability experiment is run here.
-/

open Finset

namespace CGB.CausalLearning

structure Memory (H : Type*) [Fintype H] (B : ℕ) where
  candidates : Finset H
  reserve : Fin (B + 1)
  deriving Fintype, DecidableEq

variable {H C A O : Type*} [Fintype H] [DecidableEq H]

/-- A complete candidate support table; its description is not free genome data. -/
abbrev Support (H C A O : Type*) := H → C → A → O → Bool

def screen (possible : Support H C A O) (v : Finset H) (c : C) (a : A) (o : O) : Finset H :=
  v.filter (fun h => possible h c a o = true)

def fee (possible : Support H C A O) {B : ℕ} (m : Memory H B) (c : C) (a : A) (o : O) : ℕ :=
  if 0 < m.reserve.val ∧ screen possible m.candidates c a o ≠ m.candidates then 1 else 0

def advance (possible : Support H C A O) {B : ℕ} (m : Memory H B)
    (c : C) (a : A) (o : O) : Memory H B :=
  if hw : 0 < m.reserve.val ∧ screen possible m.candidates c a o ≠ m.candidates then
    ⟨screen possible m.candidates c a o, ⟨m.reserve.val - 1, by have := m.reserve.isLt; omega⟩⟩
  else m

def initialMemory (H : Type*) [Fintype H] (B : ℕ) : Memory H B :=
  ⟨univ, ⟨B, by omega⟩⟩

omit [DecidableEq H] in
/-- Candidate masks have |H| bits of state capacity, plus the explicit reserve.
This cardinality does not certify a rulebook's 1-percent description budget. -/
theorem memory_cardinality (B : ℕ) :
    Fintype.card (Memory H B) = 2 ^ Fintype.card H * (B + 1) := by
  let e : Memory H B ≃ (Finset H × Fin (B + 1)) :=
    { toFun := fun m => (m.candidates, m.reserve)
      invFun := fun p => ⟨p.1, p.2⟩
      left_inv := by intro m; cases m; rfl
      right_inv := by intro p; cases p; rfl }
  rw [Fintype.card_congr e, Fintype.card_prod, Fintype.card_finset, Fintype.card_fin]

/-- Each effective write strictly removes candidates and spends one declared unit
of finite reserve. Insufficient reserve cannot create an unpaid update. -/
theorem write_contract (possible : Support H C A O) {B : ℕ}
    (m : Memory H B) (c : C) (a : A) (o : O) :
    (advance possible m c a o).candidates ⊆ m.candidates ∧
    (advance possible m c a o).reserve.val + fee possible m c a o = m.reserve.val ∧
    (advance possible m c a o).candidates.card + fee possible m c a o ≤ m.candidates.card := by
  by_cases hw : 0 < m.reserve.val ∧ screen possible m.candidates c a o ≠ m.candidates
  · have ha : advance possible m c a o =
        ⟨screen possible m.candidates c a o,
          ⟨m.reserve.val - 1, by have := m.reserve.isLt; omega⟩⟩ := dite_eq_left hw
    have hf : fee possible m c a o = 1 := ite_eq_left hw
    rw [ha, hf]
    change screen possible m.candidates c a o ⊆ m.candidates ∧
      (m.reserve.val - 1) + 1 = m.reserve.val ∧
      (screen possible m.candidates c a o).card + 1 ≤ m.candidates.card
    have hs : screen possible m.candidates c a o ⊆ m.candidates := Finset.filter_subset _ _
    have hc : (screen possible m.candidates c a o).card < m.candidates.card :=
      Finset.card_lt_card (Finset.ssubset_iff_subset_ne.mpr ⟨hs, hw.2⟩)
    exact ⟨hs, by omega, by omega⟩
  · have ha : advance possible m c a o = m := dite_eq_right hw
    have hf : fee possible m c a o = 0 := ite_eq_right hw
    rw [ha, hf]
    simp

theorem retain_consistent (possible : Support H C A O) {B : ℕ}
    (m : Memory H B) (c : C) (a : A) (o : O) (h : H)
    (hm : h ∈ m.candidates) (ho : possible h c a o = true) :
    h ∈ (advance possible m c a o).candidates := by
  unfold advance
  split_ifs <;> simp_all [screen]

/-- A predictable observation provides no exclusion evidence. Observing more
does not by itself imply identification or useful action learning. -/
theorem agreement_no_write (possible : Support H C A O) {B : ℕ}
    (m : Memory H B) (c : C) (a : A) (o : O)
    (ha : ∀ h ∈ m.candidates, possible h c a o = true) :
    advance possible m c a o = m ∧ fee possible m c a o = 0 := by
  have hs : screen possible m.candidates c a o = m.candidates := by
    exact Finset.filter_eq_self.mpr ha
  simp [advance, fee, hs]

theorem empty_write_reserve (possible : Support H C A O) {B : ℕ}
    (m : Memory H B) (hz : m.reserve.val = 0) (c : C) (a : A) (o : O) :
    advance possible m c a o = m ∧ fee possible m c a o = 0 := by
  simp [advance, fee, hz]

/-- A funded observation can remove a false candidate using the observation
itself, without consulting a true-model identifier. -/
theorem paid_elimination (possible : Support H C A O) {B : ℕ}
    (m : Memory H B) (c : C) (a : A) (o : O) (h : H)
    (hr : 0 < m.reserve.val) (hm : h ∈ m.candidates) (ho : possible h c a o = false) :
    h ∉ (advance possible m c a o).candidates ∧ fee possible m c a o = 1 := by
  have hn : screen possible m.candidates c a o ≠ m.candidates := by
    intro he
    have hh : h ∈ screen possible m.candidates c a o := he.symm ▸ hm
    simp [screen, ho] at hh
  have ha : advance possible m c a o =
      ⟨screen possible m.candidates c a o,
        ⟨m.reserve.val - 1, by have := m.reserve.isLt; omega⟩⟩ := dite_eq_left ⟨hr, hn⟩
  have hf : fee possible m c a o = 1 := ite_eq_left ⟨hr, hn⟩
  rw [ha, hf]
  simp [screen, ho]

/-- Analysis may count writes along a path; that counter is not learner state.
This holds for arbitrary observations, including model-misspecified histories. -/
theorem path_budget (possible : Support H C A O) {B : ℕ}
    (m : ℕ → Memory H B) (context : ℕ → C) (action : ℕ → A) (observed : ℕ → O)
    (hs : ∀ t, m (t + 1) = advance possible (m t) (context t) (action t) (observed t))
    (n : ℕ) :
    (m n).reserve.val + (∑ t ∈ range n, fee possible (m t) (context t) (action t) (observed t)) =
      (m 0).reserve.val ∧
    (m n).candidates.card +
        (∑ t ∈ range n, fee possible (m t) (context t) (action t) (observed t)) ≤
      (m 0).candidates.card := by
  induction n with
  | zero => simp
  | succ n ih =>
    have hc := write_contract possible (m n) (context n) (action n) (observed n)
    rw [← hs n] at hc
    simp only [Finset.sum_range_succ]
    constructor <;> omega

variable {S : Type*} [Fintype S] [DecidableEq S] [Fintype A] [DecidableEq A] [Fintype O]
variable (P : Interaction.Machine S (ℕ × A) O) (sensor : S → C)
  (possible : Support H C A O) {B : ℕ} (choose : Memory H B → C → A)

/-- Actions are emitted before their outcomes update internal memory. World
time belongs to P; the policy reads only the boundary sensor and its memory. -/
noncomputable def closed : Interaction.Machine (S × Memory H B) ℕ (A × O) where
  step t z out z' :=
    if out.1 = choose z.2 (sensor z.1) ∧
        z'.2 = advance possible z.2 (sensor z.1) out.1 out.2 then
      P.step (t, out.1) z.1 out.2 z'.1 else 0
  nonneg t z out z' := by
    split_ifs
    · exact P.nonneg _ _ _ _
    · exact le_rfl
  sum_one t z := by
    simp only [Fintype.sum_prod_type, ite_and]
    simp only [Finset.sum_ite_irrel, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte,
      Finset.sum_const_zero]
    exact P.sum_one (t, choose z.2 (sensor z.1)) z.1

omit [DecidableEq S] in
private theorem delivered (t : ℕ) (z z' : S × Memory H B) (out : A × O)
    (hp : 0 < (closed P sensor possible choose).step t z out z') :
    out.1 = choose z.2 (sensor z.1) ∧
    z'.2 = advance possible z.2 (sensor z.1) out.1 out.2 ∧
    0 < P.step (t, out.1) z.1 out.2 z'.1 := by
  change 0 < (if out.1 = choose z.2 (sensor z.1) ∧
    z'.2 = advance possible z.2 (sensor z.1) out.1 out.2 then
      P.step (t, out.1) z.1 out.2 z'.1 else 0) at hp
  split_ifs at hp with h
  · exact ⟨h.1, h.2, hp⟩
  · norm_num at hp

/-- Realizability is an explicit environmental premise, not a life axiom.
Time variation must be represented by sensed context or admitted support. -/
def Faithful (h : H) : Prop :=
  ∀ t s a o s', 0 < P.step (t, a) s o s' → possible h (sensor s) a o = true

omit [DecidableEq S] in
theorem true_model_step (h : H) (hf : Faithful P sensor possible h)
    (t : ℕ) (z z' : S × Memory H B) (out : A × O)
    (hm : h ∈ z.2.candidates)
    (hp : 0 < (closed P sensor possible choose).step t z out z') :
    h ∈ z'.2.candidates := by
  obtain ⟨_, hnext, hworld⟩ := delivered P sensor possible choose t z z' out hp
  rw [hnext]
  exact retain_consistent possible z.2 (sensor z.1) out.1 out.2 h hm
    (hf t z.1 out.1 out.2 z'.1 hworld)

private theorem law_invariant {X E : Type*} [Fintype X] [DecidableEq X] [Fintype E]
    (Q : Interaction.Machine X ℕ E) (initial : X) (Good : X → Prop) (hi : Good initial)
    (hs : ∀ t x e y, Good x → 0 < Q.step t x e y → Good y)
    (n : ℕ) (x : X) (hx : ¬ Good x) : Energetics.law Q initial n x = 0 := by
  classical
  induction n generalizing x with
  | zero =>
    have hn : x ≠ initial := by intro he; subst x; exact hx hi
    simp [Energetics.law, EventChannels.Kernel.stateLaw, hn]
  | succ n ih =>
    change (∑ y : X, Energetics.law Q initial n y * (∑ e : E, Q.step n y e x)) = 0
    apply Finset.sum_eq_zero
    intro y _
    by_cases hy : Good y
    · have hz (e : E) : Q.step n y e x = 0 := by
        by_cases hp : 0 < Q.step n y e x
        · exact False.elim (hx (hs n y e x hy hp))
        · exact le_antisymm (le_of_not_gt hp) (Q.nonneg n y e x)
      simp [hz]
    · rw [ih y hy, zero_mul]

/-- Positive-probability states retain the faithful candidate at every time;
the controller definition itself never receives this candidate's identity. -/
theorem true_model_law (h : H) (hf : Faithful P sensor possible h)
    (s₀ : S) (n : ℕ) (z : S × Memory H B)
    (hp : 0 < Energetics.law (closed P sensor possible choose)
      (s₀, initialMemory H B) n z) : h ∈ z.2.candidates := by
  classical
  by_contra hz
  have hi : h ∈ (initialMemory H B).candidates := by simp [initialMemory]
  have hn := law_invariant (closed P sensor possible choose) (s₀, initialMemory H B)
    (fun z => h ∈ z.2.candidates) hi
    (fun t x out y hx hs => true_model_step P sensor possible choose h hf t x y out hx hs)
    n z hz
  linarith

omit [DecidableEq S] in
/-- Even with zero write reserve, each physical action/outcome law continues.
Only the deterministic memory successor is attached to the physical event. -/
theorem physical_marginal (t : ℕ) (z : S × Memory H B) (out : A × O) (s' : S) :
    (∑ m' : Memory H B, (closed P sensor possible choose).step t z out (s', m')) =
      if out.1 = choose z.2 (sensor z.1) then P.step (t, out.1) z.1 out.2 s' else 0 := by
  simp [closed, ite_and]

/-- Robustness quantifies over all candidates and all admitted outcomes. A
safe informative action may fail to exist; that is a separate world premise. -/
def Robust (good : C → A → O → Prop) (v : Finset H) (c : C) (a : A) : Prop :=
  ∀ h ∈ v, ∀ o, possible h c a o = true → good c a o

omit [DecidableEq S] in
theorem safe_outcome (good : C → A → O → Prop) (h : H)
    (hf : Faithful P sensor possible h)
    (hsafe : ∀ m c, m.candidates.Nonempty → Robust possible good m.candidates c (choose m c))
    (t : ℕ) (z z' : S × Memory H B) (out : A × O)
    (hm : h ∈ z.2.candidates)
    (hp : 0 < (closed P sensor possible choose).step t z out z') :
    good (sensor z.1) out.1 out.2 := by
  obtain ⟨ha, _, hw⟩ := delivered P sensor possible choose t z z' out hp
  have hc := hsafe z.2 (sensor z.1) ⟨h, hm⟩ h hm out.2
  rw [← ha] at hc
  exact hc (hf t z.1 out.1 out.2 z'.1 hw)

/-- A computable finite list of actions certified by every remaining model. -/
def certifiedActions (good : C → A → O → Bool) (v : Finset H) (c : C) : Finset A :=
  if v.Nonempty then
    univ.filter (fun a => ∀ h ∈ v, ∀ o, possible h c a o = true → good c a o = true)
  else ∅

/-- Concrete candidate-dependent control. The fallback is explicitly uncertified
when no robust action exists; it needs its own physical emergency policy. -/
def certifiedOr [LinearOrder A] (good : C → A → O → Bool) (fallback : A)
    (m : Memory H B) (c : C) : A :=
  if h : (certifiedActions possible good m.candidates c).Nonempty then
    (certifiedActions possible good m.candidates c).min' h else fallback

omit [DecidableEq A] in
theorem certified_choice_sound [LinearOrder A] (good : C → A → O → Bool) (fallback : A)
    (m : Memory H B) (c : C)
    (hs : (certifiedActions possible good m.candidates c).Nonempty) :
    Robust possible (fun c a o => good c a o = true) m.candidates c
      (certifiedOr possible good fallback m c) := by
  have hv : m.candidates.Nonempty := by
    by_contra hn
    simp [certifiedActions, hn] at hs
  have he : certifiedOr possible good fallback m c =
      (certifiedActions possible good m.candidates c).min' hs := dite_eq_left hs
  rw [he]
  have hm := Finset.min'_mem (certifiedActions possible good m.candidates c) hs
  have hset : certifiedActions possible good m.candidates c =
      univ.filter (fun a : A => ∀ h ∈ m.candidates, ∀ o,
        possible h c a o = true → good c a o = true) := ite_eq_left hv
  have hf := Eq.mp (congrArg (fun v : Finset A =>
    (certifiedActions possible good m.candidates c).min' hs ∈ v) hset) hm
  exact (Finset.mem_filter.mp hf).2

omit [Fintype H] [DecidableEq H] [Fintype A] [DecidableEq A] [Fintype O] in
theorem no_safe_exploration (good : C → A → O → Prop) (v : Finset H) (c : C)
    (hb : ∀ a, ∃ h ∈ v, ∃ o, possible h c a o = true ∧ ¬good c a o) :
    ¬∃ a, Robust possible good v c a := by
  rintro ⟨a, ha⟩
  obtain ⟨h, hv, o, hp, hn⟩ := hb a
  exact hn (ha h hv o hp)

/-- A declared unit of write energy is spent only by an effective state update.
It is not a universal microscopic energy cost or a free-energy-from-information claim. -/
def charge (_t : ℕ) (z : S × Memory H B) (out : A × O) (_z' : S × Memory H B) : ℝ :=
  (fee possible z.2 (sensor z.1) out.1 out.2 : ℝ)

noncomputable def writeAccount : Energetics.Account (closed P sensor possible choose) where
  stored _ z := (z.2.reserve.val : ℝ)
  drive _ _ _ _ := 0
  load _ _ _ _ := 0
  matter _ _ _ _ := 0
  heat t z out z' := -charge sensor possible t z out z'
  balance t z out z' hp := by
    obtain ⟨_, hm, _⟩ := delivered P sensor possible choose t z z' out hp
    have hb := (write_contract possible z.2 (sensor z.1) out.1 out.2).2.1
    rw [← hm] at hb
    have hr : (z'.2.reserve.val : ℝ) + (fee possible z.2 (sensor z.1) out.1 out.2 : ℝ) =
        (z.2.reserve.val : ℝ) := by exact_mod_cast hb
    simp only [charge, add_zero]
    linarith

/-- Exact mean finite-prefix payment on the actual action/outcome process.
World perception continues after write funds run out. -/
theorem paid_mean_budget (initial : S × Memory H B) (n : ℕ) :
    Energetics.mean (Energetics.law (closed P sensor possible choose) initial n)
        ((writeAccount P sensor possible choose).stored n) +
      ∑ t ∈ range n, Energetics.flux (closed P sensor possible choose) initial
        (charge sensor possible) t = (initial.2.reserve.val : ℝ) := by
  have hz (t : ℕ) : Energetics.flux (closed P sensor possible choose) initial
      (fun _ _ _ _ => 0) t = 0 := by simp [Energetics.flux]
  have hq (t : ℕ) : Energetics.flux (closed P sensor possible choose) initial
      (writeAccount P sensor possible choose).heat t =
        -Energetics.flux (closed P sensor possible choose) initial (charge sensor possible) t := by
    simp only [Energetics.flux, writeAccount, mul_neg, Finset.sum_neg_distrib]
  induction n with
  | zero =>
    simp only [Finset.range_zero, Finset.sum_empty, add_zero]
    change (∑ z : S × Memory H B, (if z = initial then (1 : ℝ) else 0) *
      (z.2.reserve.val : ℝ)) = _
    simp only [ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
  | succ n ih =>
    have hb := Energetics.expected_first_law (closed P sensor possible choose)
      (writeAccount P sensor possible choose) initial n
    have hl : Energetics.flux (closed P sensor possible choose) initial
        (writeAccount P sensor possible choose).load n = 0 := hz n
    have hd : Energetics.flux (closed P sensor possible choose) initial
        (writeAccount P sensor possible choose).drive n = 0 := hz n
    have hm : Energetics.flux (closed P sensor possible choose) initial
        (writeAccount P sensor possible choose).matter n = 0 := hz n
    rw [hl, hd, hm, hq n] at hb
    simp only [Finset.sum_range_succ]
    linarith

/-- Mount a physical account only AFTER certifying it for this learned-action
process. Its stored energy must exclude the separate write reserve. -/
noncomputable def totalAccount (physical : Energetics.Account (closed P sensor possible choose))
    (stored : ℕ → S → Finset H → ℝ)
    (separate : ∀ t z, physical.stored t z = stored t z.1 z.2.candidates) :
    Energetics.Account (closed P sensor possible choose) where
  stored t z := stored t z.1 z.2.candidates + (z.2.reserve.val : ℝ)
  drive := physical.drive
  load := physical.load
  matter := physical.matter
  heat t z out z' := physical.heat t z out z' - charge sensor possible t z out z'
  balance t z out z' hp := by
    have he := physical.balance t z out z' hp
    rw [separate, separate] at he
    have hw := (writeAccount P sensor possible choose).balance t z out z' hp
    simp only [writeAccount, add_zero] at hw
    linarith

end CGB.CausalLearning
