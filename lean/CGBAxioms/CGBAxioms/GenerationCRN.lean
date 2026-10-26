import CGBAxioms.GenerationMedium

/-!
# Well-mixed pair reaction media are admitted (generation chain, M2)

One compartment (every unit touches every other: a single clique, which by G8 can
integrate anything). Events are a rule applied to an ordered pair of distinct units.
Rules see only the identity pages and states of the two units. The energy change of an
event is the change of single-unit energies minus the reservoir work `μ * Δc`: content that
an event takes in or gives out is exchanged with the reservoir of its boundary participant,
whose potential `μ` is set by that unit's stimulus. Rates are `κ r * exp (-ΔE / 2)` (M1a).
With `μ calm = 0` the calm dynamics is detailed balanced; different reservoir potentials
give cycles a net affinity, so a constant environment can drive a steady current.

Revision 2026-10-26: the first version put the stimulus into a state energy `ψ (s u) l` of
the boundary unit. Then every constant stimulus admitted the potential `Σ φ + Σ ψ`, so the
dynamics was detailed balanced in every constant environment: no steady metabolism was
possible, and a token entering at one reservoir and leaving at another did no net work. The theorem `PairCRN.axioms` proves the six
clauses once from finite, decidable obligations on the rule table.
-/

open Finset

namespace CGB

/-- A well-mixed pair reaction medium. `bid` are the identity pages on the boundary. -/
structure PairCRN (U Λ I R : Type*) where
  ident : U → ℕ
  bid : Finset ℕ
  content : Λ → ℕ
  /-- single-unit energy by identity page -/
  φ : ℕ → Λ → ℝ
  /-- reservoir potential of a boundary unit under its stimulus -/
  μ : I → ℝ
  /-- effect of a rule on the ordered pair, given their identity pages -/
  eff : R → ℕ → ℕ → Λ × Λ → Λ × Λ
  κ : R → ℝ
  κ_nonneg : ∀ r, 0 ≤ κ r
  rev : R → R
  /-- birth state by identity page -/
  birth : ℕ → Λ
  calm : I

namespace PairCRN

variable {U Λ I R : Type*} [Fintype U] [DecidableEq U] [Fintype R] [DecidableEq R]
  (C : PairCRN U Λ I R)

/-- The boundary units. -/
def boundary : Finset U := univ.filter fun u => C.ident u ∈ C.bid

/-- Energy of unit `u` in state `l`. -/
noncomputable def unitE (u : U) (l : Λ) : ℝ := C.φ (C.ident u) l

/-- The reservoir potential an event on `(u, v)` exchanges content with. -/
noncomputable def reservoir (s : U → I) (u v : U) : ℝ :=
  if C.ident u ∈ C.bid then C.μ (s u) else if C.ident v ∈ C.bid then C.μ (s v) else 0

/-- Content taken in by the event. -/
noncomputable def dC (r : R) (u v : U) (x : U → Λ) : ℝ :=
  let p := C.eff r (C.ident u) (C.ident v) (x u, x v)
  (C.content p.1 : ℝ) + C.content p.2 - C.content (x u) - C.content (x v)

/-- The effect of rule `r` on the pair `(u, v)`. -/
def pairAct (r : R) (u v : U) (x : U → Λ) : U → Λ :=
  if u = v then x else
    let p := C.eff r (C.ident u) (C.ident v) (x u, x v)
    Function.update (Function.update x u p.1) v p.2

/-- Energy change of the event, reservoir work included. -/
noncomputable def dE (r : R) (u v : U) (s : U → I) (x : U → Λ) : ℝ :=
  let p := C.eff r (C.ident u) (C.ident v) (x u, x v)
  C.unitE u p.1 + C.unitE v p.2 - C.unitE u (x u) - C.unitE v (x v) -
    C.reservoir s u v * C.dC r u v x

/-- The medium. -/
noncomputable def medium : Medium U Λ (R × U × U) I where
  near _ _ := True
  ident := C.ident
  boundary := C.boundary
  content := C.content
  parts e := {e.2.1, e.2.2}
  act e _ x := C.pairAct e.1 e.2.1 e.2.2 x
  rate e s x := if e.2.1 = e.2.2 then 0 else C.κ e.1 * Real.exp (-C.dE e.1 e.2.1 e.2.2 s x / 2)
  rate_nonneg e s x := by
    split_ifs
    · exact le_rfl
    · exact mul_nonneg (C.κ_nonneg _) (Real.exp_pos _).le
  init u := C.birth (C.ident u)
  calm _ := C.calm

/-- Total single-unit energy. -/
noncomputable def totalE (x : U → Λ) : ℝ := ∑ u, C.unitE u (x u)

/-- A pair event changes a unit-wise sum only at its two units. -/
lemma sum_pairAct (g : U → Λ → ℝ) (r : R) {u v : U} (huv : u ≠ v) (x : U → Λ) :
    ∑ w, g w (C.pairAct r u v x w) - ∑ w, g w (x w) =
      g u (C.eff r (C.ident u) (C.ident v) (x u, x v)).1 +
        g v (C.eff r (C.ident u) (C.ident v) (x u, x v)).2 - g u (x u) - g v (x v) := by
  classical
  unfold pairAct
  simp only [huv, if_false]
  set p := C.eff r (C.ident u) (C.ident v) (x u, x v)
  rw [← Finset.sum_sub_distrib]
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ u),
    ← Finset.add_sum_erase _ _ (Finset.mem_erase.2 ⟨Ne.symm huv, Finset.mem_univ v⟩)]
  have hrest : ∑ w ∈ (univ.erase u).erase v,
      (g w (Function.update (Function.update x u p.1) v p.2 w) - g w (x w)) = 0 := by
    refine Finset.sum_eq_zero fun w hw => ?_
    obtain ⟨hwv, hw'⟩ := Finset.mem_erase.1 hw
    obtain ⟨hwu, -⟩ := Finset.mem_erase.1 hw'
    simp [Function.update_of_ne hwv, Function.update_of_ne hwu]
  rw [hrest]
  simp [Function.update_of_ne huv, Function.update_of_ne (Ne.symm huv)]
  ring

lemma totalE_pairAct (r : R) {u v : U} (huv : u ≠ v) (s : U → I) (x : U → Λ)
    (hs : C.reservoir s u v = 0) :
    C.totalE (C.pairAct r u v x) - C.totalE x = C.dE r u v s x := by
  rw [totalE, totalE, C.sum_pairAct (fun w l => C.unitE w l) r huv x]
  simp only [dE, hs, zero_mul, sub_zero]

/-! ## The six clauses from table obligations -/

theorem p1a (h : (∀ n, {l : Λ | C.content l ≤ n}.Finite) ∧ (∃ K, ∀ l, C.content l ≤ K) ∧
    ∃! l, C.content l = 0) : C.medium.P1a := h

theorem p1b : C.medium.P1b := by
  classical
  refine ⟨fun _ _ _ _ _ => trivial, ?_, ?_⟩
  · rintro ⟨r, u, v⟩ s x w hw
    simp only [medium, Finset.mem_insert, Finset.mem_singleton, not_or] at hw
    simp only [medium, pairAct]
    split_ifs
    · rfl
    · simp [Function.update_of_ne hw.1, Function.update_of_ne hw.2]
  · rintro ⟨r, u, v⟩ s s' x y hxy
    have hu := hxy u (by simp [medium])
    have hv := hxy v (by simp [medium])
    refine ⟨?_, fun w hw => ?_⟩
    · simp only [medium, dE, unitE, reservoir, dC, hu.1, hv.1, hu.2, hv.2]
    · simp only [medium, Finset.mem_insert, Finset.mem_singleton] at hw
      simp only [medium, pairAct, hu.1, hv.1]
      split_ifs with huv
      · exact (hxy w (by rcases hw with rfl | rfl <;> simp [medium])).1
      · rcases hw with rfl | rfl
        · by_cases hwv : w = v
          · exact absurd hwv huv
          · simp [Function.update_of_ne hwv]
        · simp

theorem p2a (hcons : ∀ r i j p, i ∉ C.bid → j ∉ C.bid →
      C.content (C.eff r i j p).1 + C.content (C.eff r i j p).2 = C.content p.1 + C.content p.2) :
    C.medium.P2a := by
  classical
  rintro ⟨r, u, v⟩ s x hc _
  have hu : C.ident u ∉ C.bid := fun h => Finset.disjoint_left.mp hc
    (show u ∈ C.medium.parts (r, u, v) by simp [medium])
    (show u ∈ C.medium.boundary by simp [medium, boundary, h])
  have hv : C.ident v ∉ C.bid := fun h => Finset.disjoint_left.mp hc
    (show v ∈ C.medium.parts (r, u, v) by simp [medium])
    (show v ∈ C.medium.boundary by simp [medium, boundary, h])
  change ∑ w, C.content (C.pairAct r u v x w) = ∑ w, C.content (x w)
  by_cases huv : u = v
  · simp [pairAct, huv]
  have h := C.sum_pairAct (fun _ l => (C.content l : ℝ)) r huv x
  have key := hcons r _ _ (x u, x v) hu hv
  have keyR : (C.content (C.eff r (C.ident u) (C.ident v) (x u, x v)).1 : ℝ) +
      C.content (C.eff r (C.ident u) (C.ident v) (x u, x v)).2 = C.content (x u) + C.content (x v) := by
    exact_mod_cast key
  have : ((∑ w, C.content (C.pairAct r u v x w) : ℕ) : ℝ) = ((∑ w, C.content (x w) : ℕ) : ℝ) := by
    push_cast
    linarith
  exact_mod_cast this

theorem p2b [DecidableEq (U → Λ)] (hcalm : C.μ C.calm = 0)
    (hinv : Function.Involutive C.rev) (hκ : ∀ r, C.κ (C.rev r) = C.κ r)
    (hback : ∀ r i j p, C.eff r i j p ≠ p → C.eff (C.rev r) i j (C.eff r i j p) = p) :
    C.medium.P2b := by
  classical
  refine C.medium.p2b_of_energy C.totalE (fun e => C.κ e.1)
    (fun e => (C.rev e.1, e.2)) (fun e => by simp [hinv e.1]) (fun e => hκ e.1) ?_ ?_
  · rintro ⟨r, u, v⟩ x hne
    simp only [medium, pairAct] at hne ⊢
    by_cases huv : u = v
    · simp [huv] at hne
    simp only [huv, if_false] at hne ⊢
    set p := C.eff r (C.ident u) (C.ident v) (x u, x v)
    have hp : p ≠ (x u, x v) := by
      intro h
      apply hne
      rw [h]
      simp
    have hb := hback r _ _ (x u, x v) hp
    simp only [Function.update_self, Function.update_of_ne huv]
    have hq : C.eff (C.rev r) (C.ident u) (C.ident v) (p.1, p.2) = (x u, x v) := by
      rw [Prod.mk.eta]; exact hb
    rw [hq]
    funext w
    by_cases hwv : w = v
    · subst hwv; simp
    · by_cases hwu : w = u
      · subst hwu; simp [Function.update_of_ne hwv]
      · simp [Function.update_of_ne hwv, Function.update_of_ne hwu]
  · rintro ⟨r, u, v⟩ x hne
    have huv : u ≠ v := by
      rintro rfl
      exact hne (by simp [medium, pairAct])
    have h := C.totalE_pairAct r huv (fun _ => C.calm) x (by simp [reservoir, hcalm])
    simp only [medium, huv, if_false]
    rw [h]

theorem p2c : C.medium.P2c := by
  classical
  intro σ hσ
  have hid : ∀ u, C.ident (σ u) = C.ident u := hσ.2.1
  refine ⟨Equiv.prodCongr (Equiv.refl R) (Equiv.prodCongr σ σ), ?_⟩
  rintro ⟨r, u, v⟩ s x
  have hE : ∀ w l, C.unitE (σ w) l = C.unitE w l := by
    intro w l; simp [unitE, hid]
  have hR : C.reservoir (s ∘ σ.symm) (σ u) (σ v) = C.reservoir s u v := by
    simp [reservoir, hid]
  refine ⟨?_, ?_⟩
  · simp only [medium, Equiv.prodCongr_apply, Equiv.refl_apply, Prod.map, EmbeddingLike.apply_eq_iff_eq]
    split_ifs
    · rfl
    · simp only [dE, dC, Function.comp_apply, Equiv.symm_apply_apply, hid, hE, hR]
  · simp only [medium, Equiv.prodCongr_apply, Equiv.refl_apply, Prod.map, pairAct,
      EmbeddingLike.apply_eq_iff_eq]
    split_ifs
    · rfl
    · simp only [Function.comp_apply, Equiv.symm_apply_apply, hid]
      rw [Function.update_comp_equiv, Function.update_comp_equiv]
      simp

theorem p3 : C.medium.P3 := by
  classical
  refine ⟨?_, fun σ hσ => ?_⟩
  · rintro ⟨r, u, v⟩ s s' x hs
    have hR : C.reservoir s u v = C.reservoir s' u v := by
      unfold reservoir
      split_ifs with h1 h2
      · rw [hs u (by simp [medium, boundary, h1])]
      · rw [hs v (by simp [medium, boundary, h2])]
      · rfl
    refine ⟨?_, rfl⟩
    simp only [medium, dE, hR]
  · funext u
    have : C.ident (σ.symm u) = C.ident u := by
      have := hσ.2.1 (σ.symm u)
      rw [Equiv.apply_symm_apply] at this
      exact this.symm
    simp [medium, this]

/-- The single compartment's contact is homogeneous (an optional model property). -/
theorem homogeneous : C.medium.HomogeneousContact :=
  fun u v => ⟨Equiv.swap u v, fun _ _ => Iff.rfl, by simp⟩

/-- **M2**: a well-mixed pair reaction medium is a model of the CGB axioms as soon as its
finite rule table passes the local obligations: content levels (P1a), conservation of
closed rules (P2a), reversible rules with symmetric prefactors and a zero calm reservoir
potential (P2b). -/
theorem axioms [DecidableEq (U → Λ)] (hcalm : C.μ C.calm = 0)
    (h1a : (∀ n, {l : Λ | C.content l ≤ n}.Finite) ∧ (∃ K, ∀ l, C.content l ≤ K) ∧
      ∃! l, C.content l = 0)
    (hcons : ∀ r i j p, i ∉ C.bid → j ∉ C.bid →
      C.content (C.eff r i j p).1 + C.content (C.eff r i j p).2 = C.content p.1 + C.content p.2)
    (hinv : Function.Involutive C.rev) (hκ : ∀ r, C.κ (C.rev r) = C.κ r)
    (hback : ∀ r i j p, C.eff r i j p ≠ p → C.eff (C.rev r) i j (C.eff r i j p) = p) :
    C.medium.Axioms :=
  ⟨C.p1a h1a, C.p1b, C.p2a hcons, C.p2b hcalm hinv hκ hback, C.p2c, C.p3⟩

end PairCRN

end CGB
