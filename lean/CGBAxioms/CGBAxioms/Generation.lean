import CGBAxioms.Inference
import CGBAxioms.LanguageLifeKernel
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-!
# Generation, layer 1: what the axioms force on the equilibrium law

The axioms constrain; this file starts deriving. Under P1b an event reads and writes
only its participants, and participants are pairwise in contact. Under P2b the calm
dynamics has a positive detailed-balance weight `π` (`energy_of_p2b`: `E = -log π`).

* `jump_rate_local` (G1): the total rate of a transition depends only on the changed
  set `D`, the new values on `D`, and the old state and stimulus on `D ∪ scope D`, where
  `scope D` is the set of units in contact with every unit of `D`.
* `equilibrium_ratio_local` (G1E): hence every equilibrium ratio `π y / π x` across one
  transition with a positive reverse rate is a function of that same local data.
* `content_fixed_by_closed_single_unit` (G2): under P2a a closed event that changes one
  unit cannot change its content; content moves inside the body only by transfers that
  change at least two units.
* `JumpProcess.reversible_mul_invariant` (G3): multiplying a detailed-balance weight by
  any positive function that is constant along every move keeps detailed balance. Local
  energy differences therefore fix `π` only up to functions of the move invariants; a
  global coupling can enter through them (for example a conserved total under transfers).
* `local_markov_of_single_unit_moves` (G4): if every value of unit `u` is reachable by a
  calm move that changes only `u`, then the conditional ratio
  `π (x with u := λ) / π x` depends only on `x` on `{u} ∪ scope {u}`: the equilibrium law is
  locally Markov at `u` for the contact relation.

Not claimed: that P1b and P2b alone give a decomposition into local potentials (G3 shows
how it fails), the converse of G3 (that move invariants are the only source of non-local
terms), or where single-unit moves come from. Detailed balance fixes rate ratios only;
the symmetric conductance remains free (`ConstraintAudit.reversible_conductance_representation`).
-/

open Finset

namespace CGB

namespace JumpProcess

variable {S : Type*}

/-- **G3**: a positive factor that is constant along every move preserves detailed
balance. -/
theorem reversible_mul_invariant (P : JumpProcess S) {π w : S → ℝ} (hp : ∀ x, 0 < π x)
    (hdb : ∀ x y, π x * P.rate x y = π y * P.rate y x) (hw : ∀ x, 0 < w x)
    (hinv : ∀ x y, 0 < P.rate x y → w x = w y) :
    (∀ x, 0 < π x * w x) ∧ ∀ x y, π x * w x * P.rate x y = π y * w y * P.rate y x := by
  refine ⟨fun x => mul_pos (hp x) (hw x), fun x y => ?_⟩
  rcases (P.rate_nonneg x y).lt_or_eq with hxy | hxy
  · rw [hinv x y hxy]
    linear_combination w y * hdb x y
  · have hyx : P.rate y x = 0 := by
      have := hdb x y
      rw [← hxy, mul_zero] at this
      exact (mul_eq_zero.1 this.symm).resolve_left (hp y).ne'
    rw [← hxy, hyx]
    ring

variable {Q P : Type*}

/-- The dynamics with the slow part `P` frozen: only moves that keep `P` survive. -/
def freezeSlow [DecidableEq P] (K : JumpProcess (Q × P)) : JumpProcess (Q × P) where
  rate x y := if x.2 = y.2 then K.rate x y else 0
  rate_nonneg x y := by split_ifs <;> simp [K.rate_nonneg]

/-- **G13 (prediction is the boundary's Gibbs conditional)**: if the joint calm dynamics
has detailed-balance weight `π`, then with the slow interior `P` frozen, for each fixed
`p` the conditional `π (·, p) / Σ π (·, p)` is a detailed-balance (hence stationary) law of
the fast part's own moves. What an undriven fast boundary expresses about the next input,
given its slow interior, is this conditional. Convergence to it (irreducibility, mixing
time) and the time-scale separation itself are not claimed. -/
theorem freezeSlow_conditional_balance [Fintype Q] [DecidableEq P] (K : JumpProcess (Q × P))
    {π : Q × P → ℝ} (hp : ∀ x, 0 < π x) (hdb : ∀ x y, π x * K.rate x y = π y * K.rate y x)
    (p : P) (q q' : Q) :
    π (q, p) / (∑ r, π (r, p)) * (freezeSlow K).rate (q, p) (q', p) =
      π (q', p) / (∑ r, π (r, p)) * (freezeSlow K).rate (q', p) (q, p) := by
  simp only [freezeSlow, if_true]
  rw [div_mul_eq_mul_div, div_mul_eq_mul_div, hdb]


end JumpProcess

namespace Medium

variable {U Λ A I : Type*} (M : Medium U Λ A I)

/-- Units in contact with every unit of `D`: where an event that changes all of `D` may
read. -/
def scope (D : Set U) : Set U := {v | ∀ u ∈ D, M.near u v}

/-- Under P1b an event that changes all of `D` reads only inside `scope D`. -/
lemma parts_subset_scope (h : M.P1b) {a : A} {D : Set U} (hD : ∀ u ∈ D, u ∈ M.parts a) :
    ∀ v ∈ M.parts a, v ∈ M.scope D :=
  fun v hv u hu => h.2.1 a u (hD u hu) v hv

/-- **G2**: under P2a a closed event that changes a single unit keeps that unit's content,
so content moves inside the body only by events that change at least two units. -/
theorem content_fixed_by_closed_single_unit [Fintype U] (h : M.P2a) {a : A} {s : U → I}
    {x : U → Λ} (u : U) (ha : M.Closed a) (hr : 0 < M.rate a s x)
    (hone : ∀ v, v ≠ u → M.act a s x v = x v) :
    M.content (M.act a s x u) = M.content (x u) := by
  classical
  have ht := h a s x ha hr
  simp only [total] at ht
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ u),
    ← Finset.add_sum_erase _ _ (Finset.mem_univ u)] at ht
  have hrest : ∑ v ∈ univ.erase u, M.content (M.act a s x v) =
      ∑ v ∈ univ.erase u, M.content (x v) :=
    Finset.sum_congr rfl fun v hv => by rw [hone v (Finset.ne_of_mem_erase hv)]
  omega

variable [Fintype A] [DecidableEq (U → Λ)]

/-- **G1**: the total rate of a transition `x → y` with changed set `D` depends only on the
new values on `D` and on the old state and stimulus on `D ∪ scope D`. -/
theorem jump_rate_local (h : M.P1b) {s s' : U → I} {x y x' y' : U → Λ} (D : Set U)
    (hD : ∀ u, u ∈ D ↔ x u ≠ y u) (hy : ∀ u, u ∉ D → y' u = x' u)
    (hyD : ∀ u ∈ D, y' u = y u) (hx : ∀ u, u ∈ D ∨ u ∈ M.scope D → x' u = x u)
    (hs : ∀ u, u ∈ D ∨ u ∈ M.scope D → s' u = s u) :
    (M.jump s').rate x' y' = (M.jump s).rate x y := by
  classical
  have hne : x' ≠ y' ↔ x ≠ y := by
    constructor
    · rintro hne rfl
      exact hne (funext fun u => by
        by_cases hu : u ∈ D
        · exact absurd rfl ((hD u).1 hu)
        · exact (hy u hu).symm)
    · intro hne heq
      obtain ⟨u, hu⟩ := Function.ne_iff.1 hne
      have huD := (hD u).2 hu
      exact hu (by rw [← hx u (Or.inl huD), ← hyD u huD, heq])
  simp only [jump]
  refine Finset.sum_congr rfl fun a _ => ?_
  by_cases hsub : ∀ u ∈ D, u ∈ M.parts a
  · have hsc := M.parts_subset_scope h hsub
    obtain ⟨hr, hact⟩ := h.2.2.2 a s' s x' x fun u hu =>
      ⟨hx u (Or.inr (hsc u hu)), hs u (Or.inr (hsc u hu))⟩
    have hyy : ∀ u ∈ M.parts a, y' u = y u := fun u hu => by
      by_cases huD : u ∈ D
      · exact hyD u huD
      · rw [hy u huD, hx u (Or.inr (hsc u hu))]
        exact not_not.1 fun hn => huD ((hD u).2 hn)
    have hcond : M.act a s' x' = y' ↔ M.act a s x = y := by
      constructor
      · intro he
        funext u
        by_cases hu : u ∈ M.parts a
        · rw [← hact u hu, he, hyy u hu]
        · rw [h.2.2.1 a s x u hu]
          exact not_not.1 fun hn => hu (hsub u ((hD u).2 hn))
      · intro he
        funext u
        by_cases hu : u ∈ M.parts a
        · rw [hact u hu, he, hyy u hu]
        · rw [h.2.2.1 a s' x' u hu]
          exact (hy u fun huD => hu (hsub u huD)).symm
    simp only [hcond, hne, hr]
  · push Not at hsub
    obtain ⟨u, huD, hu⟩ := hsub
    have h1 : M.act a s x ≠ y := fun he =>
      (hD u).1 huD (by rw [← he, h.2.2.1 a s x u hu])
    have h2 : M.act a s' x' ≠ y' := fun he =>
      (hD u).1 huD (by rw [← hx u (Or.inl huD), ← hyD u huD, ← he, h.2.2.1 a s' x' u hu])
    simp [h1, h2]

/-- **G1E**: across one calm transition with a positive reverse rate, the equilibrium
ratio `π y / π x` depends only on the same local data as G1. -/
theorem equilibrium_ratio_local (h : M.P1b) {π : (U → Λ) → ℝ} (hp : ∀ x, 0 < π x)
    (hdb : ∀ x y, π x * (M.jump M.calm).rate x y = π y * (M.jump M.calm).rate y x)
    {x y x' y' : U → Λ} (D : Set U)
    (hD : ∀ u, u ∈ D ↔ x u ≠ y u) (hy : ∀ u, u ∉ D → y' u = x' u)
    (hyD : ∀ u ∈ D, y' u = y u) (hx : ∀ u, u ∈ D ∨ u ∈ M.scope D → x' u = x u)
    (hpos : 0 < (M.jump M.calm).rate y x) :
    π y' / π x' = π y / π x := by
  classical
  have hfw := M.jump_rate_local (s := M.calm) (s' := M.calm) h D hD hy hyD hx fun _ _ => rfl
  have hbw : (M.jump M.calm).rate y' x' = (M.jump M.calm).rate y x := by
    refine M.jump_rate_local (s := M.calm) (s' := M.calm) h D (fun u => by rw [hD u]; exact ne_comm)
      (fun u hu => (hy u hu).symm) (fun u hu => hx u (Or.inl hu)) (fun u hu => ?_)
      fun _ _ => rfl
    by_cases huD : u ∈ D
    · exact hyD u huD
    · rw [hy u huD, hx u hu]
      exact not_not.1 fun hn => huD ((hD u).2 hn)
  have ratio : ∀ {a b}, 0 < (M.jump M.calm).rate b a →
      π b / π a = (M.jump M.calm).rate a b / (M.jump M.calm).rate b a := fun {a b} hr => by
    rw [div_eq_div_iff (hp a).ne' hr.ne']
    linear_combination (hdb a b).symm
  rw [ratio (hbw ▸ hpos), ratio hpos, hfw, hbw]

/-- **G4**: if every value of unit `u` can be left by a calm move that changes only `u`, the
equilibrium law is locally Markov at `u`: the ratio `π (x with u := l) / π x` depends only
on `x` on `{u} ∪ scope {u}`. -/
theorem local_markov_of_single_unit_moves [DecidableEq U] (h : M.P1b) {π : (U → Λ) → ℝ}
    (hp : ∀ x, 0 < π x)
    (hdb : ∀ x y, π x * (M.jump M.calm).rate x y = π y * (M.jump M.calm).rate y x) (u : U)
    (hmove : ∀ (x : U → Λ) (l : Λ), l ≠ x u →
      0 < (M.jump M.calm).rate (Function.update x u l) x)
    {x x' : U → Λ} (hx : ∀ v, v = u ∨ v ∈ M.scope {u} → x' v = x v) (l : Λ) :
    π (Function.update x' u l) / π x' = π (Function.update x u l) / π x := by
  by_cases hl : l = x u
  · have hl' : l = x' u := by rw [hx u (Or.inl rfl), hl]
    rw [show Function.update x u l = x by rw [hl]; exact Function.update_eq_self u x,
      show Function.update x' u l = x' by rw [hl']; exact Function.update_eq_self u x',
      div_self (hp x).ne', div_self (hp x').ne']
  · refine M.equilibrium_ratio_local h hp hdb {u} (fun v => ?_) (fun v hv => ?_)
      (fun v hv => ?_) (fun v hv => hx v (by simpa using hv)) (hmove x l hl)
    · by_cases hv : v = u
      · subst hv
        simp [Ne.symm hl]
      · simp [hv, Function.update_of_ne hv]
    · exact Function.update_of_ne (by simpa using hv) _ _
    · rw [Set.mem_singleton_iff.1 hv]
      simp

/-- **P2b from weighted micro-reversibility** (generalizes `p2b_of_micro`, which has
uniform weights): if events pair up by an involution `rev` so that every effective calm
event is undone by its partner and the pair satisfies detailed balance with a positive
weight `π`, then P2b holds with that weight. This is the constructor an energy-based
medium needs. -/
theorem p2b_of_micro_weighted (π : (U → Λ) → ℝ) (hp : ∀ x, 0 < π x)
    (rev : A → A) (hinv : Function.Involutive rev)
    (hback : ∀ a x, M.act a M.calm x ≠ x →
      M.act (rev a) M.calm (M.act a M.calm x) = x ∧
        π x * M.rate a M.calm x = π (M.act a M.calm x) * M.rate (rev a) M.calm (M.act a M.calm x)) :
    M.P2b := by
  refine ⟨π, hp, fun x y => ?_⟩
  simp only [jump, Finset.mul_sum]
  rw [← Equiv.sum_comp (hinv.toPerm rev)
    (fun a => π y * if M.act a M.calm y = x ∧ y ≠ x then M.rate a M.calm y else 0)]
  refine Finset.sum_congr rfl fun a _ => ?_
  have e : (hinv.toPerm rev) a = rev a := rfl
  rw [e]
  by_cases h1 : M.act a M.calm x = y ∧ x ≠ y
  · obtain ⟨hxy, hne⟩ := h1
    have hb := hback a x (by rw [hxy]; exact Ne.symm hne)
    rw [hxy] at hb
    rw [if_pos ⟨hxy, hne⟩, if_pos ⟨hb.1, Ne.symm hne⟩]
    exact hb.2
  · rw [if_neg h1]
    by_cases h2 : M.act (rev a) M.calm y = x ∧ y ≠ x
    · obtain ⟨hyx, hne⟩ := h2
      have hb := hback (rev a) y (by rw [hyx]; exact Ne.symm hne)
      rw [hyx, hinv a] at hb
      exact absurd ⟨hb.1, Ne.symm hne⟩ h1
    · rw [if_neg h2, mul_zero, mul_zero]

/-! ## Layer 2: the interior is always in local detailed balance -/

/-- **G6**: under P1b and P3 a transition whose changed set and its scope avoid the
boundary has the same rate under every stimulus as in the calm environment. -/
theorem rate_interior_stimulus_free (h : M.P1b) (h3 : M.OnlyBoundarySenses) (s : U → I)
    {x y : U → Λ} (D : Set U) (hD : ∀ u, u ∈ D ↔ x u ≠ y u)
    (hint : ∀ u, u ∈ D ∨ u ∈ M.scope D → u ∉ M.boundary) :
    (M.jump s).rate x y = (M.jump M.calm).rate x y := by
  classical
  let s' : U → I := fun u => if u ∈ M.boundary then M.calm u else s u
  have h1 : (M.jump s').rate x y = (M.jump s).rate x y :=
    M.jump_rate_local h D hD (fun u hu => (not_not.1 fun hn => hu ((hD u).2 (Ne.symm hn))))
      (fun _ _ => rfl) (fun _ _ => rfl) fun u hu => by simp [s', hint u hu]
  rw [← h1]
  exact M.jump_blanket h3 (fun u hu => by simp [s', hu]) x y

/-- **G6DB**: hence, with P2b, interior transitions satisfy detailed balance with the calm
weight under every stimulus: all irreversibility enters at the boundary. -/
theorem interior_detailed_balance (h : M.P1b) (h3 : M.OnlyBoundarySenses)
    {π : (U → Λ) → ℝ}
    (hdb : ∀ x y, π x * (M.jump M.calm).rate x y = π y * (M.jump M.calm).rate y x)
    (s : U → I) {x y : U → Λ} (D : Set U) (hD : ∀ u, u ∈ D ↔ x u ≠ y u)
    (hint : ∀ u, u ∈ D ∨ u ∈ M.scope D → u ∉ M.boundary) :
    π x * (M.jump s).rate x y = π y * (M.jump s).rate y x := by
  rw [M.rate_interior_stimulus_free h h3 s D hD hint,
    M.rate_interior_stimulus_free h h3 s D (fun u => by rw [hD u]; exact ne_comm) hint]
  exact hdb x y

/-- **G7**: no equilibrium coupling beyond contact. If a second change (on `E`, e.g. a
plastic variable) lies outside `D ∪ scope D`, it does not alter the equilibrium preference
across the move on `D`; the mixed second difference of `log π` vanishes. -/
theorem coupling_vanishes_beyond_scope (h : M.P1b) {π : (U → Λ) → ℝ} (hp : ∀ x, 0 < π x)
    (hdb : ∀ x y, π x * (M.jump M.calm).rate x y = π y * (M.jump M.calm).rate y x)
    {x y x' y' : U → Λ} (D : Set U) (hD : ∀ u, u ∈ D ↔ x u ≠ y u)
    (hy : ∀ u, u ∉ D → y' u = x' u) (hyD : ∀ u ∈ D, y' u = y u)
    (hfar : ∀ u, x' u ≠ x u → u ∉ D ∧ u ∉ M.scope D)
    (hpos : 0 < (M.jump M.calm).rate y x) :
    π y' / π x' = π y / π x :=
  M.equilibrium_ratio_local h hp hdb D hD hy hyD (fun u hu => not_not.1 fun hn =>
    (hu.elim (hfar u hn).1 (hfar u hn).2)) hpos

/-- **G9 (modulation needs the boundary)**: if the rate of a transition (for example a
plastic move: its learning rate or, through the forward/backward pair, its learning
direction) differs between two stimuli, then some unit of the changed set or its contact
scope is a boundary unit. Interior plasticity can therefore depend on the world only
through local states it reads; world-driven gating, sign or rate control needs a
boundary (modulator) unit at the plastic site. Contrapositive of G6. -/
theorem modulation_needs_boundary (h : M.P1b) (h3 : M.OnlyBoundarySenses) {s s' : U → I}
    {x y : U → Λ} (D : Set U) (hD : ∀ u, u ∈ D ↔ x u ≠ y u)
    (hdiff : (M.jump s).rate x y ≠ (M.jump s').rate x y) :
    ∃ u, (u ∈ D ∨ u ∈ M.scope D) ∧ u ∈ M.boundary := by
  by_contra hno
  push Not at hno
  exact hdiff ((M.rate_interior_stimulus_free h h3 s D hD hno).trans
    (M.rate_interior_stimulus_free h h3 s' D hD hno).symm)

/-! ## Why learning rates must be modulated (minimal world class)

World: a hidden value drifting as a random walk (step variance `q`) and observed with
noise (variance `r`). A learner with a fixed gain `K` (update `m ← m + K (y − m)`) has
stationary prediction-error variance `V(K) = (q + K² r) / (K (2 − K))` for `0 < K < 2`
(from `e' = (1 − K) e + w − K v`). Parametrize the world by `P > 0` with
`q = P² / (P + r)` (`P` is the Kalman steady-state prior variance). -/

/-- **G10a**: the fixed-gain error exceeds the optimum `P` by an exact square; the unique
optimal fixed gain is the Kalman gain `P / (P + r)`. -/
theorem fixed_gain_error (r P K : ℝ) (hr : 0 < r) (hP : 0 < P) (hK0 : 0 < K) (hK2 : K < 2) :
    (P ^ 2 / (P + r) + K ^ 2 * r) / (K * (2 - K)) - P =
      ((P + r) * K - P) ^ 2 / (K * (2 - K) * (P + r)) := by
  have h1 : K * (2 - K) ≠ 0 := (mul_pos hK0 (by linarith)).ne'
  have h2 : P + r ≠ 0 := by positivity
  field_simp
  ring

/-- **G10b (necessity of modulation, minimal form)**: in two worlds with different
volatility (`P₁ ≠ P₂`), no fixed gain is optimal in both: in at least one world its error
is strictly above that world's optimum. A learner that persists across such worlds must
change its learning rate with the world; by G9 that change needs boundary participation
or local states informed by activity. -/
theorem no_fixed_gain_for_two_volatilities (r P₁ P₂ K : ℝ) (hr : 0 < r) (hP₁ : 0 < P₁)
    (hP₂ : 0 < P₂) (hne : P₁ ≠ P₂) (hK0 : 0 < K) (hK2 : K < 2) :
    P₁ < (P₁ ^ 2 / (P₁ + r) + K ^ 2 * r) / (K * (2 - K)) ∨
      P₂ < (P₂ ^ 2 / (P₂ + r) + K ^ 2 * r) / (K * (2 - K)) := by
  have hden : ∀ P, 0 < P → 0 < K * (2 - K) * (P + r) := fun P hP =>
    mul_pos (mul_pos hK0 (by linarith)) (by linarith)
  by_contra hno
  push Not at hno
  obtain ⟨h₁, h₂⟩ := hno
  have z : ∀ P, 0 < P → (P ^ 2 / (P + r) + K ^ 2 * r) / (K * (2 - K)) ≤ P →
      (P + r) * K - P = 0 := fun P hP hle => by
    have e := fixed_gain_error r P K hr hP hK0 hK2
    have hsq : ((P + r) * K - P) ^ 2 / (K * (2 - K) * (P + r)) ≤ 0 := by linarith
    have hsq' : ((P + r) * K - P) ^ 2 ≤ 0 :=
      (div_le_iff₀ (hden P hP)).1 hsq |>.trans (by rw [zero_mul])
    exact pow_eq_zero_iff (n := 2) (by norm_num) |>.1 (le_antisymm hsq' (sq_nonneg _))
  have e₁ := z P₁ hP₁ h₁
  have e₂ := z P₂ hP₂ h₂
  apply hne
  have : (P₁ - P₂) * (K - 1) = 0 := by linear_combination e₁ - e₂
  rcases mul_eq_zero.1 this with h | h
  · linarith
  · have hK : K = 1 := by linarith
    subst hK
    nlinarith

/-! ## Interoceptive memory is necessary (forage world, minimal form)

A source is rich or poor (fixed while the organism stays) with prior odds `a = (1-π)/π`;
a rich source yields intake with probability `pr`, a poor one with `pp < pr`. After `n`
consecutive ticks without intake, Bayes gives the rich posterior `1 / (1 + a ρⁿ)` with
`ρ = (1-pp)/(1-pr) > 1`, and the best predicted intake `pp + post · (pr - pp)`. -/

/-- Best predicted intake after `n` consecutive misses. -/
noncomputable def missPrediction (a ρ pr pp : ℝ) (n : ℕ) : ℝ :=
  pp + (pr - pp) / (1 + a * ρ ^ n)

/-- The miss posterior is Bayes' rule: with prior `π`, miss probabilities `qr < qp`
(`qr = 1 - pr`, `qp = 1 - pp`), `n` misses give rich posterior `1 / (1 + a ρⁿ)` with
`a = (1-π)/π` and `ρ = qp/qr`. -/
theorem miss_posterior_bayes {π qr qp : ℝ} (hπ0 : 0 < π) (hπ1 : π < 1) (hr : 0 < qr)
    (hp : 0 < qp) (n : ℕ) :
    π * qr ^ n / (π * qr ^ n + (1 - π) * qp ^ n) = 1 / (1 + (1 - π) / π * (qp / qr) ^ n) := by
  have h1 : 0 < qr ^ n := pow_pos hr n
  have h2 : 0 < qp ^ n := pow_pos hp n
  have h3 : 0 < 1 - π := by linarith
  rw [div_pow]
  field_simp

/-- **G11a**: each further miss strictly lowers the best prediction. -/
theorem missPrediction_strictAnti {a ρ pr pp : ℝ} (ha : 0 < a) (hρ : 1 < ρ) (hp : pp < pr) :
    StrictAnti (missPrediction a ρ pr pp) := by
  apply strictAnti_nat_of_succ_lt
  intro n
  unfold missPrediction
  have h0 : 0 < ρ ^ n := pow_pos (by linarith) n
  have hlt : ρ ^ n < ρ ^ (n + 1) := pow_lt_pow_right₀ hρ (Nat.lt_succ_self n)
  have d1 : 0 < 1 + a * ρ ^ n := by positivity
  have d2 : 1 + a * ρ ^ n < 1 + a * ρ ^ (n + 1) := by nlinarith
  have : (pr - pp) / (1 + a * ρ ^ (n + 1)) < (pr - pp) / (1 + a * ρ ^ n) :=
    div_lt_div_of_pos_left (by linarith) d1 d2
  linarith

/-- **G11b (interoceptive memory)**: to predict intake within `ε` after every history of up
to `N` consecutive misses, where `2ε` is below every gap between those predictions, an
organism's state must take at least `N + 1` distinct values on those intake histories
(via `CausalState.separated_histories_capacity`). In stage 1 the heard bytes carry no
information about intake, so these distinctions must be stored from intake itself; by G6
the interior cannot re-read the stimulus later, so they must be held in states. -/
theorem intake_memory_capacity {S : Type*} [Fintype S] {a ρ pr pp ε : ℝ} (N : ℕ)
    (code : List Bool → S) (decode : S → ℝ)
    (accurate : ∀ n : Fin (N + 1),
      dist (missPrediction a ρ pr pp n) (decode (code (List.replicate n false))) ≤ ε)
    (gap : ∀ i j : Fin (N + 1), i ≠ j →
      ε + ε < dist (missPrediction a ρ pr pp i) (missPrediction a ρ pr pp j)) :
    N + 1 ≤ Fintype.card S := by
  have h := LanguageLifeKernel.CausalState.separated_histories_capacity
    (fun h : List Bool => missPrediction a ρ pr pp (h.count false))
    (fun n : Fin (N + 1) => List.replicate n false) code (fun s _ => decode s)
    (fun _ suffix => suffix = []) ε
    (fun n suffix hs => by subst hs; simpa [List.count_replicate] using accurate n)
    (fun i j hij => ⟨[], rfl, rfl, by simpa [List.count_replicate] using gap i j hij⟩)
  simpa using h.2

/-- The gap condition is satisfiable: strict monotonicity separates any finite family, so
some positive precision forces `N + 1` states. -/
theorem intake_gap_exists {a ρ pr pp : ℝ} (ha : 0 < a) (hρ : 1 < ρ) (hp : pp < pr) (N : ℕ) :
    ∃ ε > 0, ∀ i j : Fin (N + 1), i ≠ j →
      ε + ε < dist (missPrediction a ρ pr pp i) (missPrediction a ρ pr pp j) := by
  have hs := missPrediction_strictAnti ha hρ hp
  let gaps := (Finset.univ.filter fun p : Fin (N + 1) × Fin (N + 1) => p.1 ≠ p.2).image
    fun p => dist (missPrediction a ρ pr pp p.1) (missPrediction a ρ pr pp p.2)
  have pos : ∀ g ∈ gaps, 0 < g := by
    intro g hg
    obtain ⟨p, hp', rfl⟩ := Finset.mem_image.1 hg
    have hne : (p.1 : ℕ) ≠ p.2 := fun h => (Finset.mem_filter.1 hp').2 (Fin.ext h)
    exact dist_pos.2 (hs.injective.ne hne)
  by_cases he : gaps.Nonempty
  · refine ⟨gaps.min' he / 4, by have := pos _ (gaps.min'_mem he); positivity, fun i j hij => ?_⟩
    have hm : gaps.min' he ≤ dist (missPrediction a ρ pr pp i) (missPrediction a ρ pr pp j) :=
      gaps.min'_le _ (Finset.mem_image.2 ⟨(i, j), by simp [hij], rfl⟩)
    have := pos _ (gaps.min'_mem he)
    linarith
  · refine ⟨1, one_pos, fun i j hij => absurd ⟨_, Finset.mem_image.2 ⟨(i, j), by simp [hij], rfl⟩⟩ he⟩

/-- The leave threshold exists: whenever going elsewhere is worth more than a known-poor
source (`pp < c`), enough misses push the best prediction below `c`. -/
theorem exists_leave_threshold {a ρ pr pp c : ℝ} (ha : 0 < a) (hρ : 1 < ρ) (hp : pp < pr)
    (hc : pp < c) : ∃ n, missPrediction a ρ pr pp n < c := by
  obtain ⟨n, hn⟩ := ((tendsto_pow_atTop_atTop_of_one_lt hρ).eventually_gt_atTop
    ((pr - pp) / (a * (c - pp)))).exists
  refine ⟨n, ?_⟩
  have hcp : 0 < c - pp := by linarith
  have hac : 0 < a * (c - pp) := mul_pos ha hcp
  have h1 : pr - pp < a * (c - pp) * ρ ^ n := by
    rwa [div_lt_iff₀ hac, mul_comm] at hn
  have hpos : 0 < 1 + a * ρ ^ n := by have := pow_pos (by linarith : (0:ℝ) < ρ) n; positivity
  unfold missPrediction
  have : (pr - pp) / (1 + a * ρ ^ n) < c - pp := by
    rw [div_lt_iff₀ hpos]
    nlinarith
  linarith

/-- Optimal foraging action after a history of misses: `1` (stay) iff the best predicted
intake is at least the value `c` of going elsewhere, else `0` (leave). -/
noncomputable def stayDecision (a ρ pr pp c : ℝ) (h : List Bool) : ℝ :=
  if c ≤ missPrediction a ρ pr pp (h.count false) then 1 else 0

/-- **G12 (a miss counter is necessary for optimal foraging)**: let `nStar` be the first
miss count whose best prediction falls below `c` (all earlier counts stay at or above it).
Any state code whose readout gives the optimal stay/leave action, for histories of up to
`nStar` misses and every continuation of further misses, takes at least `nStar + 1`
distinct values: histories with `i < j ≤ nStar` misses must be told apart, because after
`nStar - j` more misses one must leave and the other must stay. The precision is fixed
by the world (`c`, `pr`, `pp`, prior), not chosen. -/
theorem miss_counter_necessary {S : Type*} [Fintype S] {a ρ pr pp c : ℝ} (nStar : ℕ)
    (hbelow : missPrediction a ρ pr pp nStar < c)
    (habove : ∀ n < nStar, c ≤ missPrediction a ρ pr pp n)
    (code : List Bool → S) (decode : S → List Bool → ℝ)
    (exact : ∀ (n : Fin (nStar + 1)) (k : ℕ), n + k ≤ nStar →
      decode (code (List.replicate n false)) (List.replicate k false) =
        stayDecision a ρ pr pp c (List.replicate n false ++ List.replicate k false)) :
    nStar + 1 ≤ Fintype.card S := by
  have h := LanguageLifeKernel.CausalState.separated_histories_capacity
    (stayDecision a ρ pr pp c) (fun n : Fin (nStar + 1) => List.replicate n false) code decode
    (fun n suffix => ∃ k, suffix = List.replicate k false ∧ (n : ℕ) + k ≤ nStar) (1 / 3)
    (fun n suffix ⟨k, hs, hk⟩ => by subst hs; rw [exact n k hk, dist_self]; norm_num)
    (fun i j hij => by
      wlog hlt : (i : ℕ) < j generalizing i j
      · obtain ⟨s, h1, h2, h3⟩ := this j i (Ne.symm hij)
          (lt_of_le_of_ne (not_lt.1 hlt) (fun e => hij (Fin.ext e.symm)))
        exact ⟨s, h2, h1, by rwa [dist_comm]⟩
      refine ⟨List.replicate (nStar - j) false, ⟨_, rfl, by omega⟩, ⟨_, rfl, by omega⟩, ?_⟩
      have hi : c ≤ missPrediction a ρ pr pp (i + (nStar - j)) := habove _ (by omega)
      have hj : (j : ℕ) + (nStar - j) = nStar := by omega
      simp only [stayDecision, List.count_append, List.count_replicate_self]
      rw [if_pos hi, hj, if_neg (not_le.2 hbelow), Real.dist_eq]
      norm_num)
  simpa using h.2

/-- Merging two histories that lie on opposite sides of the threshold costs expected
intake: with positive weights, a single shared action earns strictly less than acting
optimally on each. -/
theorem merging_across_threshold_loses {w₁ w₂ p₁ p₂ c : ℝ} (hw₁ : 0 < w₁) (hw₂ : 0 < w₂)
    (h₁ : c < p₁) (h₂ : p₂ < c) (stay : Bool) :
    w₁ * (if stay then p₁ else c) + w₂ * (if stay then p₂ else c) < w₁ * p₁ + w₂ * c := by
  cases stay <;> simp <;> nlinarith

/-! ## The learning rule (delta rule) from the two-phase protocol

Readout (G13): the boundary takes value `c` with probability `softmax z c`, where the
logit `z c` collects the couplings from the slow interior. A plastic coupling `w` from an
interior unit with state `x` to boundary value `b` adds `x * w` to `z b`. -/

/-- Boundary readout law. -/
noncomputable def softmax {B : Type*} [Fintype B] (z : B → ℝ) (c : B) : ℝ :=
  Real.exp (z c) / ∑ d, Real.exp (z d)

/-- **G14a (two-phase drift)**: in the drive phase the boundary is clamped to the arrived
value `b'` and the site moves by `+λ x` when the boundary shows `b`; in the free phase the
boundary samples the readout law and the site moves by `-μ x` when it shows `b`. The
expected drift per cycle is `x (λ [b = b'] - μ softmax z b)`. -/
theorem two_phase_drift {B : Type*} [Fintype B] [DecidableEq B] (z : B → ℝ) (b b' : B)
    (x lam mu : ℝ) :
    lam * x * (if b' = b then 1 else 0) - mu * ∑ c, softmax z c * (x * (if c = b then 1 else 0)) =
      x * (lam * (if b' = b then 1 else 0) - mu * softmax z b) := by
  simp only [mul_ite, mul_one, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, if_true]
  split_ifs <;> ring

/-- The readout law sums to one. -/
theorem softmax_sum {B : Type*} [Fintype B] [Nonempty B] (z : B → ℝ) :
    ∑ c, softmax z c = 1 := by
  simp only [softmax, div_eq_mul_inv, ← Finset.sum_mul]
  exact mul_inv_cancel₀ (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty).ne'

/-- **G14b**: the log likelihood of the arrived value `b'` changes, per unit change of that
coupling, by `x ([b' = b] - softmax z b)`. -/
theorem log_softmax_deriv {B : Type*} [Fintype B] [DecidableEq B] (z : B → ℝ) (b b' : B)
    (x : ℝ) :
    HasDerivAt (fun t => Real.log (softmax (fun c => z c + if c = b then t * x else 0) b'))
      (x * ((if b' = b then 1 else 0) - softmax z b)) 0 := by
  have hS : ∀ t : ℝ, 0 < ∑ d, Real.exp (z d + if d = b then t * x else 0) := fun t =>
    Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨b, Finset.mem_univ _⟩
  have hform : (fun t => Real.log (softmax (fun c => z c + if c = b then t * x else 0) b')) =
      fun t => (z b' + if b' = b then t * x else 0) -
        Real.log (∑ d, Real.exp (z d + if d = b then t * x else 0)) := by
    funext t
    simp only [softmax]
    rw [Real.log_div (Real.exp_pos _).ne' (hS t).ne', Real.log_exp]
  rw [hform]
  have hlin : ∀ d, HasDerivAt (fun t : ℝ => z d + if d = b then t * x else 0)
      (if d = b then x else 0) 0 := fun d => by
    split_ifs
    · simpa using ((hasDerivAt_id (0 : ℝ)).mul_const x).const_add (z d)
    · simpa using hasDerivAt_const (0 : ℝ) (z d)
  have hsum : HasDerivAt (fun t => ∑ d, Real.exp (z d + if d = b then t * x else 0))
      (∑ d, Real.exp (z d) * (if d = b then x else 0)) 0 := by
    have := HasDerivAt.fun_sum (u := Finset.univ) (fun d _ => (hlin d).exp)
    simpa using this
  have hlog := hsum.log (hS 0).ne'
  have := (hlin b').sub hlog
  simp only [zero_mul, ite_self, add_zero, mul_ite, mul_zero, Finset.sum_ite_eq',
    Finset.mem_univ, ite_true] at this
  convert this using 1
  have hS0 : 0 < ∑ d, Real.exp (z d) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨b, Finset.mem_univ _⟩
  simp only [softmax]
  split_ifs <;> field_simp <;> ring

/-- **G14c (the phases must balance)**: if, for every readout state and arrived value, the
expected drift vector over boundary values is a positive multiple of the log-likelihood
gradient vector, the drive and free phases carry equal weight (`λ = μ`): summed over
boundary values the gradient vanishes while the drift sums to `λ - μ`. Sleep-like free
sampling must balance waking clamped learning. -/
theorem phases_must_balance {B : Type*} [Fintype B] [DecidableEq B] [Nonempty B]
    (lam mu : ℝ)
    (prop : ∀ (z : B → ℝ) (b' : B), ∃ κ > 0, ∀ b,
      lam * (if b' = b then 1 else 0) - mu * softmax z b =
        κ * ((if b' = b then 1 else 0) - softmax z b)) :
    lam = mu := by
  obtain ⟨b'⟩ := ‹Nonempty B›
  obtain ⟨κ, -, h⟩ := prop (fun _ => 0) b'
  have hs := Finset.sum_congr rfl (fun b (_ : b ∈ Finset.univ) => h b)
  simp only [Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_ite_eq, Finset.mem_univ,
    if_true, softmax_sum] at hs
  linarith

/-! ## Why the readout is one particle among labels (the softmax premise of G14)

The world delivers exactly one symbol per tick, so the target is a categorical law. -/

/-- **R2 (leakage loses on every outcome)**: if the free boundary puts mass `ℓ > 0` on
configurations that are no symbol, every symbol's predicted probability is `(1 - ℓ) q b`
instead of `q b`, and the log likelihood of whatever arrives is strictly lower. An optimal
predictive boundary keeps its free dynamics inside the one-symbol sector. -/
theorem leakage_loses {ℓ q : ℝ} (hℓ : 0 < ℓ) (hℓ1 : ℓ < 1) (hq : 0 < q) :
    Real.log ((1 - ℓ) * q) < Real.log q := by
  apply Real.log_lt_log (mul_pos (by linarith) hq)
  nlinarith

/-- **R1 (conservation makes the readout one-hot)**: if every label unit holds content `0`
or `1` and the readout's total content is `1` (an invariant shell of the closed free
dynamics under P2a, `total_of_reachable`), exactly one label is on. -/
theorem one_unit_one_label {B : Type*} [Fintype B] [DecidableEq B] (n : B → ℕ)
    (h01 : ∀ b, n b ≤ 1) (htot : ∑ b, n b = 1) : ∃! b, n b = 1 := by
  have hex : ∃ b, n b = 1 := by
    by_contra hno
    push Not at hno
    have : ∀ b, n b = 0 := fun b => by have := h01 b; have := hno b; omega
    simp [this] at htot
  obtain ⟨b, hb⟩ := hex
  refine ⟨b, hb, fun c hc => ?_⟩
  by_contra hne
  have h2 : n b + n c ≤ ∑ d, n d := by
    rw [← Finset.sum_pair (Ne.symm hne)]
    exact Finset.sum_le_sum_of_subset (Finset.subset_univ _)
  omega

/-- In the one-particle sector a local label energy gives the softmax readout: with
weights `exp (z b)` on the configurations `label b on`, the normalized law is `softmax z`
(G13 restricted to the sector). -/
theorem sector_law_is_softmax {B : Type*} [Fintype B] (z : B → ℝ) (b : B) :
    Real.exp (z b) / ∑ c, Real.exp (z c) = softmax z b := rfl

/-! ## Actions: the motor readout and reward-modulated learning

The world reads the motor boundary (`ReadsBoundary`); by R1/R2 and G13 the action is one
motor particle among action labels with equilibrium law `softmax z` given the interior.
No target action is ever supplied; the boundary senses intake (reward), so by G9 its sign
reaches motor plasticity only through a boundary modulator. Both phases are weighted by
the same advantage `R a - bl` (balanced phases, G14c). -/

/-- **G15a (reward-modulated two-phase drift is the policy gradient)**: if the motor
particle was clamped at the action taken `a ~ softmax z`, the drive phase moved the
site by `(R a - bl) x` when label `b` was on, and the free phase by `-(R a - bl) x softmax z b`,
the expected drift equals `x softmax z b (R b - J)` with `J = Σ softmax z a * R a`, for
every baseline `bl`. -/
theorem reward_two_phase_drift {B : Type*} [Fintype B] [DecidableEq B] [Nonempty B]
    (z R : B → ℝ) (bl : ℝ) (b : B) :
    ∑ a, softmax z a * ((R a - bl) * ((if a = b then 1 else 0) - softmax z b)) =
      softmax z b * (R b - ∑ a, softmax z a * R a) := by
  have hs := softmax_sum z
  have h1 : ∀ a, softmax z a * ((R a - bl) * ((if a = b then 1 else 0) - softmax z b)) =
      ((if a = b then softmax z a * (R a - bl) else 0) - softmax z b * (softmax z a * R a)) +
        softmax z b * bl * softmax z a := by
    intro a; split_ifs <;> ring
  rw [Finset.sum_congr rfl fun a _ => h1 a, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    Finset.sum_ite_eq', ← Finset.mul_sum, ← Finset.mul_sum, hs]
  simp only [Finset.mem_univ, if_true]
  ring

/-- **G15b**: that drift is exactly the derivative of the expected reward `J` with respect
to the logit of `b`. -/
theorem expected_reward_deriv {B : Type*} [Fintype B] [DecidableEq B] (z R : B → ℝ) (b : B) :
    HasDerivAt (fun t => ∑ a, softmax (fun c => z c + if c = b then t else 0) a * R a)
      (softmax z b * (R b - ∑ a, softmax z a * R a)) 0 := by
  set D : ℝ → ℝ := fun t => ∑ d, Real.exp (z d + if d = b then t else 0)
  set N : ℝ → ℝ := fun t => ∑ d, Real.exp (z d + if d = b then t else 0) * R d
  have hD : ∀ t, 0 < D t := fun t =>
    Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨b, Finset.mem_univ _⟩
  have hratio : ∀ w : B → ℝ, ∑ a, softmax w a * R a =
      (∑ d, Real.exp (w d) * R d) / ∑ d, Real.exp (w d) := by
    intro w
    simp only [softmax, div_eq_mul_inv, Finset.sum_mul]
    exact Finset.sum_congr rfl fun a _ => by ring
  have hform : (fun t => ∑ a, softmax (fun c => z c + if c = b then t else 0) a * R a) =
      fun t => N t / D t := by
    funext t
    exact hratio _
  rw [hform]
  have hlin : ∀ d, HasDerivAt (fun t : ℝ => z d + if d = b then t else 0)
      (if d = b then 1 else 0) 0 := fun d => by
    split_ifs
    · simpa using (hasDerivAt_id (0 : ℝ)).const_add (z d)
    · simpa using hasDerivAt_const (0 : ℝ) (z d)
  have hD' : HasDerivAt D (Real.exp (z b)) 0 := by
    have := HasDerivAt.fun_sum (u := Finset.univ) (fun d _ => (hlin d).exp)
    simpa [D, mul_ite] using this
  have hN' : HasDerivAt N (Real.exp (z b) * R b) 0 := by
    have := HasDerivAt.fun_sum (u := Finset.univ) (fun d _ => ((hlin d).exp).mul_const (R d))
    simpa [N, mul_ite, ite_mul] using this
  convert hN'.div hD' (hD 0).ne' using 1
  have hD0 : D 0 = ∑ d, Real.exp (z d) := by simp [D]
  have hN0 : N 0 = ∑ d, Real.exp (z d) * R d := by simp [N]
  have hpos : 0 < ∑ d, Real.exp (z d) := hD0 ▸ hD 0
  rw [hD0, hN0, hratio z]
  simp only [softmax]
  field_simp

/-- **G15c (outcome-blind modulation learns nothing)**: if the modulator's weight `m` does
not depend on the action's outcome, the expected two-phase drift is zero for every site.
So a policy can only climb through a modulator that reads intake, which by G9 is a
boundary unit. -/
theorem outcome_blind_drift_zero {B : Type*} [Fintype B] [DecidableEq B] [Nonempty B]
    (z : B → ℝ) (m : ℝ) (b : B) :
    ∑ a, softmax z a * (m * ((if a = b then 1 else 0) - softmax z b)) = 0 := by
  have h := reward_two_phase_drift z (fun _ => m) 0 b
  have hs := softmax_sum z
  simp only [sub_zero] at h
  rw [h, ← Finset.sum_mul, hs]
  ring

/-- **G15d (reward phases must balance)**: with drive weight `lam` and free weight `mu`,
if the expected drift is a multiple of the expected-reward gradient for every reward
table, then `lam = mu`. -/
theorem reward_phases_must_balance {B : Type*} [Fintype B] [DecidableEq B] [Nonempty B]
    (z : B → ℝ) (bl lam mu : ℝ) (b : B)
    (h : ∀ R : B → ℝ, ∃ κ : ℝ,
      ∑ a, softmax z a * ((R a - bl) * (lam * (if a = b then 1 else 0) - mu * softmax z b)) =
        κ * (softmax z b * (R b - ∑ a, softmax z a * R a))) : lam = mu := by
  obtain ⟨κ, hκ⟩ := h (fun _ => bl + 1)
  have hs := softmax_sum z
  have hp : 0 < softmax z b := div_pos (Real.exp_pos _)
    (Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨b, Finset.mem_univ _⟩)
  have hJ : ∑ a, softmax z a * (bl + 1) = bl + 1 := by rw [← Finset.sum_mul, hs, one_mul]
  have ht : ∀ a, softmax z a * ((bl + 1 - bl) * (lam * (if a = b then 1 else 0) -
      mu * softmax z b)) = lam * (if a = b then softmax z a else 0) -
        mu * softmax z b * softmax z a := by
    intro a; split_ifs <;> ring
  rw [Finset.sum_congr rfl fun a _ => ht a, Finset.sum_sub_distrib, ← Finset.mul_sum,
    ← Finset.mul_sum, Finset.sum_ite_eq', hs, hJ] at hκ
  simp only [Finset.mem_univ, if_true] at hκ
  have h0 : (lam - mu) * softmax z b = 0 := by linarith
  have := (mul_eq_zero.mp h0).resolve_right hp.ne'
  linarith

/-! ## Layer 2b: the integration bottleneck

A single-unit move of `v` is produced by events whose participants form a clique
containing `v`. If `v`'s neighbours split into parts with no contact between different
parts, every such event reads at most one part, so the total up and down rates are sums of
single-part functions. Detailed balance asks their ratio to equal the equilibrium factor.
When that factor is a product over parts (for instance pairwise couplings `J_i x_i`):

* two parts: always realizable (`two_part_balance`, a crossed construction);
* three parts, each with a non-constant factor: impossible with a positive down rate
  (`three_part_balance_impossible`).

So a unit can be energetically sensitive to at most two mutually non-contacting groups of
neighbours; integrating more inputs needs intermediate units (binary-tree convergence).
These are the algebraic cores; the bridge from `Medium.jump` to the part-wise sums is
stated in the generation note, not yet formalized. -/

/-- **G8a** (two parts suffice): with positive part factors `φ₁, φ₂` and constants
`a, b ≥ 0`, up rates `a φ₁ + b φ₂` and down rates `b / φ₁ + a / φ₂` (each term reads one
part) have ratio exactly `φ₁ φ₂`. -/
theorem two_part_balance {α β : Type*} (φ₁ : α → ℝ) (φ₂ : β → ℝ) (h₁ : ∀ a, 0 < φ₁ a)
    (h₂ : ∀ b, 0 < φ₂ b) (a b : ℝ) (x : α) (y : β) :
    a * φ₁ x + b * φ₂ y = φ₁ x * φ₂ y * (b / φ₁ x + a / φ₂ y) := by
  field_simp [(h₁ x).ne', (h₂ y).ne']
  ring

/-- **G8b** (three parts are impossible): if up rates `f₁+f₂+f₃` and down rates `g₁+g₂+g₃`
(each term reading one part) satisfy detailed balance with a product factor `φ₁ φ₂ φ₃` that
is non-constant on every part, the total down rate vanishes at a chosen state. No sign
assumption is needed; with nonnegative rates every part's down rate vanishes there. -/
theorem three_part_balance_impossible {α β γ : Type*}
    (f₁ g₁ φ₁ : α → ℝ) (f₂ g₂ φ₂ : β → ℝ) (f₃ g₃ φ₃ : γ → ℝ)
    (hφ₁ : ∀ a, 0 < φ₁ a) (hφ₂ : ∀ b, 0 < φ₂ b) (hφ₃ : ∀ c, 0 < φ₃ c)
    (hbal : ∀ a b c, f₁ a + f₂ b + f₃ c = φ₁ a * φ₂ b * φ₃ c * (g₁ a + g₂ b + g₃ c))
    {a₀ a₁ : α} {b₀ b₁ : β} {c₀ c₁ : γ}
    (d₁ : φ₁ a₀ ≠ φ₁ a₁) (d₂ : φ₂ b₀ ≠ φ₂ b₁) (d₃ : φ₃ c₀ ≠ φ₃ c₁) :
    g₁ a₀ + g₂ b₀ + g₃ c₀ = 0 := by
  -- mixed second difference over two parts at a fixed third part value vanishes
  have mix : ∀ p₀ p₁ q₀ q₁ r₀ r₁ s₀ s₁ t : ℝ,
      (p₁ * r₁ * (q₁ + s₁ + t) - p₁ * r₀ * (q₁ + s₀ + t)) -
        (p₀ * r₁ * (q₀ + s₁ + t) - p₀ * r₀ * (q₀ + s₀ + t)) =
      (p₁ * q₁ - p₀ * q₀) * (r₁ - r₀) + (p₁ - p₀) * (r₁ * s₁ - r₀ * s₀)
        + t * (p₁ - p₀) * (r₁ - r₀) := fun _ _ _ _ _ _ _ _ _ => by ring
  have zero12 : ∀ c, (φ₁ a₁ * g₁ a₁ - φ₁ a₀ * g₁ a₀) * (φ₂ b₁ - φ₂ b₀) +
      (φ₁ a₁ - φ₁ a₀) * (φ₂ b₁ * g₂ b₁ - φ₂ b₀ * g₂ b₀) +
      g₃ c * (φ₁ a₁ - φ₁ a₀) * (φ₂ b₁ - φ₂ b₀) = 0 := fun c => by
    have e := mix (φ₁ a₀) (φ₁ a₁) (g₁ a₀) (g₁ a₁) (φ₂ b₀) (φ₂ b₁) (g₂ b₀) (g₂ b₁) (g₃ c)
    have h11 := hbal a₁ b₁ c; have h10 := hbal a₁ b₀ c
    have h01 := hbal a₀ b₁ c; have h00 := hbal a₀ b₀ c
    have hc := (hφ₃ c).ne'
    rw [← e]
    apply mul_left_cancel₀ hc
    linear_combination h10 + h01 - h11 - h00
  have zero13 : ∀ b, (φ₁ a₁ * g₁ a₁ - φ₁ a₀ * g₁ a₀) * (φ₃ c₁ - φ₃ c₀) +
      (φ₁ a₁ - φ₁ a₀) * (φ₃ c₁ * g₃ c₁ - φ₃ c₀ * g₃ c₀) +
      g₂ b * (φ₁ a₁ - φ₁ a₀) * (φ₃ c₁ - φ₃ c₀) = 0 := fun b => by
    have e := mix (φ₁ a₀) (φ₁ a₁) (g₁ a₀) (g₁ a₁) (φ₃ c₀) (φ₃ c₁) (g₃ c₀) (g₃ c₁) (g₂ b)
    have h11 := hbal a₁ b c₁; have h10 := hbal a₁ b c₀
    have h01 := hbal a₀ b c₁; have h00 := hbal a₀ b c₀
    have hb := (hφ₂ b).ne'
    rw [← e]
    apply mul_left_cancel₀ hb
    linear_combination h10 + h01 - h11 - h00
  have zero23 : ∀ a, (φ₂ b₁ * g₂ b₁ - φ₂ b₀ * g₂ b₀) * (φ₃ c₁ - φ₃ c₀) +
      (φ₂ b₁ - φ₂ b₀) * (φ₃ c₁ * g₃ c₁ - φ₃ c₀ * g₃ c₀) +
      g₁ a * (φ₂ b₁ - φ₂ b₀) * (φ₃ c₁ - φ₃ c₀) = 0 := fun a => by
    have e := mix (φ₂ b₀) (φ₂ b₁) (g₂ b₀) (g₂ b₁) (φ₃ c₀) (φ₃ c₁) (g₃ c₀) (g₃ c₁) (g₁ a)
    have h11 := hbal a b₁ c₁; have h10 := hbal a b₁ c₀
    have h01 := hbal a b₀ c₁; have h00 := hbal a b₀ c₀
    have ha := (hφ₁ a).ne'
    rw [← e]
    apply mul_left_cancel₀ ha
    linear_combination h10 + h01 - h11 - h00
  have n1 : φ₁ a₁ - φ₁ a₀ ≠ 0 := sub_ne_zero.2 (Ne.symm d₁)
  have n2 : φ₂ b₁ - φ₂ b₀ ≠ 0 := sub_ne_zero.2 (Ne.symm d₂)
  have n3 : φ₃ c₁ - φ₃ c₀ ≠ 0 := sub_ne_zero.2 (Ne.symm d₃)
  -- each down rate takes the same value at the two chosen points
  have k3 : g₃ c₁ = g₃ c₀ := by
    have := (zero12 c₁).trans (zero12 c₀).symm
    have h : (g₃ c₁ - g₃ c₀) * ((φ₁ a₁ - φ₁ a₀) * (φ₂ b₁ - φ₂ b₀)) = 0 := by linear_combination this
    exact sub_eq_zero.1 ((mul_eq_zero.1 h).resolve_right (mul_ne_zero n1 n2))
  have k2 : g₂ b₁ = g₂ b₀ := by
    have := (zero13 b₁).trans (zero13 b₀).symm
    have h : (g₂ b₁ - g₂ b₀) * ((φ₁ a₁ - φ₁ a₀) * (φ₃ c₁ - φ₃ c₀)) = 0 := by linear_combination this
    exact sub_eq_zero.1 ((mul_eq_zero.1 h).resolve_right (mul_ne_zero n1 n3))
  have k1 : g₁ a₁ = g₁ a₀ := by
    have := (zero23 a₁).trans (zero23 a₀).symm
    have h : (g₁ a₁ - g₁ a₀) * ((φ₂ b₁ - φ₂ b₀) * (φ₃ c₁ - φ₃ c₀)) = 0 := by linear_combination this
    exact sub_eq_zero.1 ((mul_eq_zero.1 h).resolve_right (mul_ne_zero n2 n3))
  have h := zero12 c₀
  rw [k1, k2] at h
  have h' : (g₁ a₀ + g₂ b₀ + g₃ c₀) * ((φ₁ a₁ - φ₁ a₀) * (φ₂ b₁ - φ₂ b₀)) = 0 := by
    linear_combination h
  exact (mul_eq_zero.1 h').resolve_right (mul_ne_zero n1 n2)

end Medium

end CGB
