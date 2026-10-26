import CGBAxioms.GenerationMedium

/-!
# Well-mixed pair reaction media are admitted (generation chain, M2)

One compartment (every unit touches every other: a single clique, which by G8 can
integrate anything). Events are a rule applied to an ordered pair of distinct units.
Rules see only the identity pages and states of the two units; the energy is a sum of
single-unit terms, boundary units adding a reservoir term that reads their stimulus.
Rates are `κ r * exp (-ΔE / 2)` (M1a). The theorem `PairCRN.axioms` proves the six
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
  /-- reservoir term of a boundary unit under its stimulus -/
  ψ : I → Λ → ℝ
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

/-- Energy of unit `u` in state `l` under stimulus `s`. -/
noncomputable def unitE (s : U → I) (u : U) (l : Λ) : ℝ :=
  C.φ (C.ident u) l + if C.ident u ∈ C.bid then C.ψ (s u) l else 0

/-- The effect of rule `r` on the pair `(u, v)`. -/
def pairAct (r : R) (u v : U) (x : U → Λ) : U → Λ :=
  if u = v then x else
    let p := C.eff r (C.ident u) (C.ident v) (x u, x v)
    Function.update (Function.update x u p.1) v p.2

/-- Energy change of the event. -/
noncomputable def dE (r : R) (u v : U) (s : U → I) (x : U → Λ) : ℝ :=
  let p := C.eff r (C.ident u) (C.ident v) (x u, x v)
  C.unitE s u p.1 + C.unitE s v p.2 - C.unitE s u (x u) - C.unitE s v (x v)

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

/-- Total energy under stimulus `s`. -/
noncomputable def totalE (s : U → I) (x : U → Λ) : ℝ := ∑ u, C.unitE s u (x u)

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

lemma totalE_pairAct (r : R) {u v : U} (huv : u ≠ v) (s : U → I) (x : U → Λ) :
    C.totalE s (C.pairAct r u v x) - C.totalE s x = C.dE r u v s x := by
  rw [totalE, totalE, C.sum_pairAct (fun w l => C.unitE s w l) r huv x]
  rfl

/-! ## The six clauses from table obligations -/

theorem p1a (h : (∀ n, {l : Λ | C.content l ≤ n}.Finite) ∧ (∃ K, ∀ l, C.content l ≤ K) ∧
    ∃! l, C.content l = 0) : C.medium.P1a := h

theorem p1b : C.medium.P1b := by
  classical
  refine ⟨fun u v => ⟨Equiv.swap u v, fun _ _ => Iff.rfl, by simp⟩, fun _ _ _ _ _ => trivial,
    ?_, ?_⟩
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
    · simp only [medium, dE, unitE, hu.1, hv.1, hu.2, hv.2]
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

theorem p2b [DecidableEq (U → Λ)] (hinv : Function.Involutive C.rev) (hκ : ∀ r, C.κ (C.rev r) = C.κ r)
    (hback : ∀ r i j p, C.eff r i j p ≠ p → C.eff (C.rev r) i j (C.eff r i j p) = p) :
    C.medium.P2b := by
  classical
  refine C.medium.p2b_of_energy (C.totalE fun _ => C.calm) (fun e => C.κ e.1)
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
    have h := C.totalE_pairAct r huv (fun _ => C.calm) x
    simp only [medium, huv, if_false]
    rw [h]

theorem p2c : C.medium.P2c := by
  classical
  intro σ hσ
  have hid : ∀ u, C.ident (σ u) = C.ident u := hσ.2.1
  refine ⟨Equiv.prodCongr (Equiv.refl R) (Equiv.prodCongr σ σ), ?_⟩
  rintro ⟨r, u, v⟩ s x
  have hE : ∀ w l, C.unitE (s ∘ σ.symm) (σ w) l = C.unitE s w l := by
    intro w l; simp [unitE, hid]
  refine ⟨?_, ?_⟩
  · simp only [medium, Equiv.prodCongr_apply, Equiv.refl_apply, Prod.map, EmbeddingLike.apply_eq_iff_eq]
    split_ifs
    · rfl
    · simp only [dE, Function.comp_apply, Equiv.symm_apply_apply, hid, hE]
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
    have hE : ∀ w l, C.unitE s w l = C.unitE s' w l := by
      intro w l
      unfold unitE
      split_ifs with h
      · rw [hs w (by simp [medium, boundary, h])]
      · rfl
    refine ⟨?_, rfl⟩
    simp only [medium, dE, hE]
  · funext u
    have : C.ident (σ.symm u) = C.ident u := by
      have := hσ.2.1 (σ.symm u)
      rw [Equiv.apply_symm_apply] at this
      exact this.symm
    simp [medium, this]

/-- **M2**: a well-mixed pair reaction medium is a model of the CGB axioms as soon as its
finite rule table passes the local obligations: content levels (P1a), conservation of
closed rules (P2a), and reversible rules with symmetric prefactors (P2b). -/
theorem axioms [DecidableEq (U → Λ)]
    (h1a : (∀ n, {l : Λ | C.content l ≤ n}.Finite) ∧ (∃ K, ∀ l, C.content l ≤ K) ∧
      ∃! l, C.content l = 0)
    (hcons : ∀ r i j p, i ∉ C.bid → j ∉ C.bid →
      C.content (C.eff r i j p).1 + C.content (C.eff r i j p).2 = C.content p.1 + C.content p.2)
    (hinv : Function.Involutive C.rev) (hκ : ∀ r, C.κ (C.rev r) = C.κ r)
    (hback : ∀ r i j p, C.eff r i j p ≠ p → C.eff (C.rev r) i j (C.eff r i j p) = p) :
    C.medium.Axioms :=
  ⟨C.p1a h1a, C.p1b, C.p2a hcons, C.p2b hinv hκ hback, C.p2c, C.p3⟩

end PairCRN

end CGB
