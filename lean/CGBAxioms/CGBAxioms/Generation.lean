import CGBAxioms.Inference

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

/-- **G8b** (three parts are impossible): if up rates `f₁+f₂+f₃` and nonnegative down rates
`g₁+g₂+g₃` (each term reading one part) satisfy detailed balance with a product factor
`φ₁ φ₂ φ₃` that is non-constant on every part, the down rate vanishes somewhere. -/
theorem three_part_balance_impossible {α β γ : Type*}
    (f₁ g₁ φ₁ : α → ℝ) (f₂ g₂ φ₂ : β → ℝ) (f₃ g₃ φ₃ : γ → ℝ)
    (hg₁ : ∀ a, 0 ≤ g₁ a) (hg₂ : ∀ b, 0 ≤ g₂ b) (hg₃ : ∀ c, 0 ≤ g₃ c)
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
