import CGBAxioms.GenerationCRN

/-!
# Lattice media: cells with neighbour contact (generation chain, M4; nesting T-E1)

The joint medium of an organism nested in its world. Units are cells of a world with a given
symmetric, irreflexive adjacency. A cell's local state may be large: in the E. coli world it
holds the cell's food and waste and, if a bacterium is there, the bacterium's whole inner
state (the inner medium, one level down). Events are

* **single-cell rules** `eff1`: the cell's own reactions (the bacterium's inner chemistry,
  sensing, eating, excreting, all inside one cell);
* **neighbour rules** `eff2` on ordered adjacent pairs (diffusion of food and waste, a bacterium
  moving to the next cell).

Energies add over cells; an event's energy change subtracts the reservoir work of its first
boundary participant. Rates are `conductance × exp (-ΔE / 2)` with a conductance that may
depend on the participants' states but is equal on an event and its reverse. (A lumped inner
pair rule has rate `κ · n_src · e^{-ΔE_unit/2} = κ √(n_src n_dst) · e^{-ΔE_cell/2}`, with the
counts' mixing entropy in the cell energy; so the inner medium fits this form.)

* `Medium.p2b_of_conductance`: M1a with state-dependent symmetric conductances.
* `LatticeCRN.axioms` (**M4**): such a medium satisfies the six clauses as soon as its finite
  rule tables pass local obligations (content levels, conservation by interior rules,
  reversibility, symmetric conductances, a zero calm potential).
-/

open Finset

namespace CGB

namespace Medium

variable {U Λ A I : Type*} [Fintype U] [Fintype A] [DecidableEq (U → Λ)]
  (M : Medium U Λ A I)

/-- **M1a with conductances**: rates `κ a x · exp (-(E y - E x) / 2)` with `κ` equal on an event
and its reverse (read at the respective source states) give P2b. -/
theorem p2b_of_conductance (E : (U → Λ) → ℝ) (κ : A → (U → Λ) → ℝ) (rev : A → A)
    (hinv : Function.Involutive rev)
    (hback : ∀ a x, M.act a M.calm x ≠ x → M.act (rev a) M.calm (M.act a M.calm x) = x)
    (hκ : ∀ a x, M.act a M.calm x ≠ x → κ (rev a) (M.act a M.calm x) = κ a x)
    (hrate : ∀ a x, M.act a M.calm x ≠ x →
      M.rate a M.calm x = κ a x * Real.exp (-(E (M.act a M.calm x) - E x) / 2)) :
    M.P2b := by
  refine M.p2b_of_micro_weighted (fun x => Real.exp (-E x)) (fun _ => Real.exp_pos _) rev hinv
    fun a x hne => ⟨hback a x hne, ?_⟩
  have hy : M.act (rev a) M.calm (M.act a M.calm x) ≠ M.act a M.calm x := by
    rw [hback a x hne]; exact fun h => hne h.symm
  rw [hrate a x hne, hrate (rev a) _ hy, hback a x hne, hκ a x hne]
  generalize M.act a M.calm x = y
  have e1 : Real.exp (-E x) * (κ a x * Real.exp (-(E y - E x) / 2)) =
      κ a x * Real.exp (-(E x + E y) / 2) := by
    rw [mul_left_comm, ← Real.exp_add]; congr 2; ring
  have e2 : Real.exp (-E y) * (κ a x * Real.exp (-(E x - E y) / 2)) =
      κ a x * Real.exp (-(E x + E y) / 2) := by
    rw [mul_left_comm, ← Real.exp_add]; congr 2; ring
  rw [e1, e2]

end Medium

/-- A lattice reaction medium. -/
structure LatticeCRN (U Λ I R₁ R₂ : Type*) where
  adj : U → U → Prop
  adj_symm : ∀ u v, adj u v → adj v u
  adj_irrefl : ∀ u, ¬ adj u u
  ident : U → ℕ
  bid : Finset ℕ
  content : Λ → ℕ
  /-- cell energy by identity page -/
  φ : ℕ → Λ → ℝ
  /-- reservoir potential of a boundary cell under its stimulus -/
  μ : I → ℝ
  eff1 : R₁ → ℕ → Λ → Λ
  eff2 : R₂ → ℕ → ℕ → Λ × Λ → Λ × Λ
  c1 : R₁ → ℕ → Λ → ℝ
  c2 : R₂ → ℕ → ℕ → Λ × Λ → ℝ
  c1_nonneg : ∀ r i l, 0 ≤ c1 r i l
  c2_nonneg : ∀ r i j p, 0 ≤ c2 r i j p
  rev1 : R₁ → R₁
  rev2 : R₂ → R₂
  birth : ℕ → Λ
  calm : I

namespace LatticeCRN

open Classical

variable {U Λ I R₁ R₂ : Type*} [Fintype U] [DecidableEq U] (C : LatticeCRN U Λ I R₁ R₂)

def boundary : Finset U := univ.filter fun u => C.ident u ∈ C.bid

noncomputable def unitE (u : U) (l : Λ) : ℝ := C.φ (C.ident u) l

noncomputable def res (s : U → I) (u : U) : ℝ := if C.ident u ∈ C.bid then C.μ (s u) else 0

noncomputable def reservoir (s : U → I) (u v : U) : ℝ :=
  if C.ident u ∈ C.bid then C.μ (s u) else if C.ident v ∈ C.bid then C.μ (s v) else 0

/-- Effect of a single-cell rule. -/
noncomputable def act1 (r : R₁) (u : U) (x : U → Λ) : U → Λ :=
  Function.update x u (C.eff1 r (C.ident u) (x u))

/-- Effect of a neighbour rule (none unless the cells are adjacent). -/
noncomputable def act2 (r : R₂) (u v : U) (x : U → Λ) : U → Λ :=
  if C.adj u v then
    let p := C.eff2 r (C.ident u) (C.ident v) (x u, x v)
    Function.update (Function.update x u p.1) v p.2
  else x

noncomputable def dE1 (r : R₁) (u : U) (s : U → I) (x : U → Λ) : ℝ :=
  C.unitE u (C.eff1 r (C.ident u) (x u)) - C.unitE u (x u) -
    C.res s u * ((C.content (C.eff1 r (C.ident u) (x u)) : ℝ) - C.content (x u))

noncomputable def dE2 (r : R₂) (u v : U) (s : U → I) (x : U → Λ) : ℝ :=
  let p := C.eff2 r (C.ident u) (C.ident v) (x u, x v)
  C.unitE u p.1 + C.unitE v p.2 - C.unitE u (x u) - C.unitE v (x v) -
    C.reservoir s u v * ((C.content p.1 : ℝ) + C.content p.2 - C.content (x u) - C.content (x v))

/-- The medium. Events: `inl (r, u)` a single-cell rule at `u`; `inr (r, u, v)` a neighbour rule
on the ordered pair (no participants and no effect unless adjacent). -/
noncomputable def medium : Medium U Λ ((R₁ × U) ⊕ (R₂ × U × U)) I where
  near u v := u = v ∨ C.adj u v
  ident := C.ident
  boundary := C.boundary
  content := C.content
  parts
    | .inl (_, u) => {u}
    | .inr (_, u, v) => if C.adj u v then {u, v} else ∅
  act
    | .inl (r, u) => fun _ x => C.act1 r u x
    | .inr (r, u, v) => fun _ x => C.act2 r u v x
  rate
    | .inl (r, u) => fun s x => C.c1 r (C.ident u) (x u) * Real.exp (-C.dE1 r u s x / 2)
    | .inr (r, u, v) => fun s x =>
        if C.adj u v then C.c2 r (C.ident u) (C.ident v) (x u, x v) * Real.exp (-C.dE2 r u v s x / 2)
        else 0
  rate_nonneg e s x := by
    rcases e with ⟨r, u⟩ | ⟨r, u, v⟩
    · exact mul_nonneg (C.c1_nonneg _ _ _) (Real.exp_pos _).le
    · dsimp only
      split_ifs
      · exact mul_nonneg (C.c2_nonneg _ _ _ _) (Real.exp_pos _).le
      · exact le_rfl
  init u := C.birth (C.ident u)
  calm _ := C.calm

/-- Total cell energy. -/
noncomputable def totalE (x : U → Λ) : ℝ := ∑ u, C.unitE u (x u)

lemma sum_update (g : U → Λ → ℝ) (u : U) (l : Λ) (x : U → Λ) :
    ∑ w, g w (Function.update x u l w) - ∑ w, g w (x w) = g u l - g u (x u) := by
  rw [← Finset.sum_sub_distrib, ← Finset.add_sum_erase _ _ (Finset.mem_univ u)]
  have hrest : ∑ w ∈ univ.erase u, (g w (Function.update x u l w) - g w (x w)) = 0 :=
    Finset.sum_eq_zero fun w hw => by
      simp [Function.update_of_ne (Finset.mem_erase.1 hw).1]
  rw [hrest]; simp

lemma sum_update2 (g : U → Λ → ℝ) {u v : U} (huv : u ≠ v) (a b : Λ) (x : U → Λ) :
    ∑ w, g w (Function.update (Function.update x u a) v b w) - ∑ w, g w (x w) =
      g u a + g v b - g u (x u) - g v (x v) := by
  have h1 := sum_update g v b (Function.update x u a)
  have h2 := sum_update g u a x
  simp only [Function.update_of_ne (Ne.symm huv)] at h1
  linarith

lemma adj_ne {u v : U} (h : C.adj u v) : u ≠ v := fun e => C.adj_irrefl u (e ▸ h)

/-! ## The six clauses from table obligations -/

theorem p1a (h : (∀ n, {l : Λ | C.content l ≤ n}.Finite) ∧ (∃ K, ∀ l, C.content l ≤ K) ∧
    ∃! l, C.content l = 0) : C.medium.P1a := h

theorem p1b : C.medium.P1b := by
  refine ⟨?_, ?_, ?_⟩
  · rintro (⟨r, u⟩ | ⟨r, u, v⟩) a ha b hb
    · simp only [medium, Finset.mem_singleton] at ha hb
      exact Or.inl (ha.trans hb.symm)
    · simp only [medium] at ha hb
      split_ifs at ha hb with hadj
      · simp only [Finset.mem_insert, Finset.mem_singleton] at ha hb
        rcases ha with rfl | rfl <;> rcases hb with rfl | rfl
        · exact Or.inl rfl
        · exact Or.inr hadj
        · exact Or.inr (C.adj_symm _ _ hadj)
        · exact Or.inl rfl
      · simp at ha
  · rintro (⟨r, u⟩ | ⟨r, u, v⟩) s x w hw
    · simp only [medium, Finset.mem_singleton] at hw
      simp [medium, act1, Function.update_of_ne hw]
    · simp only [medium] at hw ⊢
      unfold act2
      split_ifs with hadj
      · rw [if_pos hadj] at hw
        simp only [Finset.mem_insert, Finset.mem_singleton, not_or] at hw
        simp [Function.update_of_ne hw.1, Function.update_of_ne hw.2]
      · rfl
  · rintro (⟨r, u⟩ | ⟨r, u, v⟩) s s' x y hxy
    · have hu := hxy u (by simp [medium])
      refine ⟨?_, fun w hw => ?_⟩
      · simp only [medium, dE1, res, hu.1, hu.2]
      · simp only [medium, Finset.mem_singleton] at hw
        subst hw
        simp [medium, act1, hu.1]
    · simp only [medium]
      by_cases hadj : C.adj u v
      · have hu := hxy u (by simp [medium, hadj])
        have hv := hxy v (by simp [medium, hadj])
        refine ⟨?_, fun w hw => ?_⟩
        · simp only [hadj, if_true, dE2, reservoir, hu.1, hv.1, hu.2, hv.2]
        · simp only [medium, hadj, if_true, Finset.mem_insert, Finset.mem_singleton] at hw
          simp only [act2, hadj, if_true, hu.1, hv.1]
          rcases hw with rfl | rfl
          · simp [Function.update_of_ne (C.adj_ne hadj)]
          · simp
      · refine ⟨by simp [hadj], fun w hw => ?_⟩
        simp [medium, hadj] at hw

theorem p2a (hcons1 : ∀ r i l, i ∉ C.bid → C.content (C.eff1 r i l) = C.content l)
    (hcons2 : ∀ r i j p, i ∉ C.bid → j ∉ C.bid →
      C.content (C.eff2 r i j p).1 + C.content (C.eff2 r i j p).2 = C.content p.1 + C.content p.2) :
    C.medium.P2a := by
  rintro (⟨r, u⟩ | ⟨r, u, v⟩) s x hc hr
  · have hu : C.ident u ∉ C.bid := fun h => Finset.disjoint_left.mp hc
      (show u ∈ C.medium.parts (.inl (r, u)) by simp [medium])
      (show u ∈ C.medium.boundary by simp [medium, boundary, h])
    change ∑ w, C.content (C.act1 r u x w) = ∑ w, C.content (x w)
    have h := sum_update (fun _ l => (C.content l : ℝ)) u (C.eff1 r (C.ident u) (x u)) x
    have key : (C.content (C.eff1 r (C.ident u) (x u)) : ℝ) = C.content (x u) := by
      exact_mod_cast hcons1 r _ _ hu
    have : ((∑ w, C.content (C.act1 r u x w) : ℕ) : ℝ) = ((∑ w, C.content (x w) : ℕ) : ℝ) := by
      push_cast; unfold act1; linarith
    exact_mod_cast this
  · by_cases hadj : C.adj u v
    · have hu : C.ident u ∉ C.bid := fun h => Finset.disjoint_left.mp hc
        (show u ∈ C.medium.parts (.inr (r, u, v)) by simp [medium, hadj])
        (show u ∈ C.medium.boundary by simp [medium, boundary, h])
      have hv : C.ident v ∉ C.bid := fun h => Finset.disjoint_left.mp hc
        (show v ∈ C.medium.parts (.inr (r, u, v)) by simp [medium, hadj])
        (show v ∈ C.medium.boundary by simp [medium, boundary, h])
      change ∑ w, C.content (C.act2 r u v x w) = ∑ w, C.content (x w)
      set p := C.eff2 r (C.ident u) (C.ident v) (x u, x v)
      have h := sum_update2 (fun _ l => (C.content l : ℝ)) (C.adj_ne hadj) p.1 p.2 x
      have key : (C.content p.1 : ℝ) + C.content p.2 = C.content (x u) + C.content (x v) := by
        exact_mod_cast hcons2 r _ _ (x u, x v) hu hv
      have : ((∑ w, C.content (C.act2 r u v x w) : ℕ) : ℝ) =
          ((∑ w, C.content (x w) : ℕ) : ℝ) := by
        push_cast; simp only [act2, hadj, if_true]; linarith
      exact_mod_cast this
    · change ∑ w, C.content (C.act2 r u v x w) = ∑ w, C.content (x w)
      simp [act2, hadj]

/-- The reverse of an event. -/
def revE : (R₁ × U) ⊕ (R₂ × U × U) → (R₁ × U) ⊕ (R₂ × U × U)
  | .inl (r, u) => .inl (C.rev1 r, u)
  | .inr (r, u, v) => .inr (C.rev2 r, u, v)

/-- The conductance of an event at a state. -/
noncomputable def cond : (R₁ × U) ⊕ (R₂ × U × U) → (U → Λ) → ℝ
  | .inl (r, u), x => C.c1 r (C.ident u) (x u)
  | .inr (r, u, v), x => if C.adj u v then C.c2 r (C.ident u) (C.ident v) (x u, x v) else 0

lemma act1_ne {r : R₁} {u : U} {x : U → Λ} (h : C.act1 r u x ≠ x) :
    C.eff1 r (C.ident u) (x u) ≠ x u := fun e => h (by simp [act1, e])

lemma act2_ne {r : R₂} {u v : U} {x : U → Λ} (hadj : C.adj u v) (h : C.act2 r u v x ≠ x) :
    C.eff2 r (C.ident u) (C.ident v) (x u, x v) ≠ (x u, x v) := by
  intro e
  apply h
  simp only [act2, hadj, if_true, e]
  funext w
  by_cases hwv : w = v
  · subst hwv; simp
  · by_cases hwu : w = u
    · subst hwu; simp [Function.update_of_ne hwv]
    · simp [Function.update_of_ne hwv, Function.update_of_ne hwu]

theorem p2b [DecidableEq (U → Λ)] [Fintype R₁] [Fintype R₂] (hcalm : C.μ C.calm = 0)
    (hinv1 : Function.Involutive C.rev1) (hinv2 : Function.Involutive C.rev2)
    (hback1 : ∀ r i l, C.eff1 r i l ≠ l → C.eff1 (C.rev1 r) i (C.eff1 r i l) = l)
    (hback2 : ∀ r i j p, C.eff2 r i j p ≠ p → C.eff2 (C.rev2 r) i j (C.eff2 r i j p) = p)
    (hc1 : ∀ r i l, C.eff1 r i l ≠ l → C.c1 (C.rev1 r) i (C.eff1 r i l) = C.c1 r i l)
    (hc2 : ∀ r i j p, C.eff2 r i j p ≠ p → C.c2 (C.rev2 r) i j (C.eff2 r i j p) = C.c2 r i j p) :
    C.medium.P2b := by
  refine C.medium.p2b_of_conductance C.totalE C.cond C.revE ?_ ?_ ?_ ?_
  · rintro (⟨r, u⟩ | ⟨r, u, v⟩)
    · simp [revE, hinv1 r]
    · simp [revE, hinv2 r]
  · rintro (⟨r, u⟩ | ⟨r, u, v⟩) x hne
    · change C.act1 (C.rev1 r) u (C.act1 r u x) = x
      have he := C.act1_ne hne
      have hb := hback1 r _ _ he
      funext w
      by_cases hwu : w = u
      · subst hwu; simp [act1, hb]
      · simp [act1, Function.update_of_ne hwu]
    · change C.act2 (C.rev2 r) u v (C.act2 r u v x) = x
      by_cases hadj : C.adj u v
      · have he := C.act2_ne hadj hne
        have hb := hback2 r _ _ _ he
        have huv := C.adj_ne hadj
        set p := C.eff2 r (C.ident u) (C.ident v) (x u, x v)
        have hq : C.eff2 (C.rev2 r) (C.ident u) (C.ident v)
            ((C.act2 r u v x) u, (C.act2 r u v x) v) = (x u, x v) := by
          simp only [act2, hadj, if_true, Function.update_self, Function.update_of_ne huv]
          rw [Prod.mk.eta]; exact hb
        simp only [act2, hadj, if_true] at hq ⊢
        rw [hq]
        funext w
        by_cases hwv : w = v
        · subst hwv; simp
        · by_cases hwu : w = u
          · subst hwu; simp [Function.update_of_ne hwv]
          · simp [Function.update_of_ne hwv, Function.update_of_ne hwu]
      · exact absurd (by simp [medium, act2, hadj]) hne
  · rintro (⟨r, u⟩ | ⟨r, u, v⟩) x hne
    · change C.c1 (C.rev1 r) (C.ident u) ((C.act1 r u x) u) = C.c1 r (C.ident u) (x u)
      simp only [act1, Function.update_self]
      exact hc1 r _ _ (C.act1_ne hne)
    · by_cases hadj : C.adj u v
      · have he := C.act2_ne hadj hne
        have huv := C.adj_ne hadj
        change (if C.adj u v then C.c2 (C.rev2 r) (C.ident u) (C.ident v)
            ((C.act2 r u v x) u, (C.act2 r u v x) v) else 0) =
          if C.adj u v then C.c2 r (C.ident u) (C.ident v) (x u, x v) else 0
        simp only [hadj, if_true, act2, Function.update_self, Function.update_of_ne huv]
        rw [Prod.mk.eta]; exact hc2 r _ _ _ he
      · exact absurd (by simp [medium, act2, hadj]) hne
  · rintro (⟨r, u⟩ | ⟨r, u, v⟩) x hne
    · change C.c1 r (C.ident u) (x u) * Real.exp (-C.dE1 r u (fun _ => C.calm) x / 2) =
        C.c1 r (C.ident u) (x u) * Real.exp (-(C.totalE (C.act1 r u x) - C.totalE x) / 2)
      have h := sum_update (fun w l => C.unitE w l) u (C.eff1 r (C.ident u) (x u)) x
      have hd : C.dE1 r u (fun _ => C.calm) x = C.totalE (C.act1 r u x) - C.totalE x := by
        simp only [dE1, res, hcalm, ite_self, zero_mul, sub_zero, totalE, act1]
        linarith
      rw [hd]
    · by_cases hadj : C.adj u v
      · have huv := C.adj_ne hadj
        change (if C.adj u v then C.c2 r (C.ident u) (C.ident v) (x u, x v) *
            Real.exp (-C.dE2 r u v (fun _ => C.calm) x / 2) else 0) =
          (if C.adj u v then C.c2 r (C.ident u) (C.ident v) (x u, x v) else 0) *
            Real.exp (-(C.totalE (C.act2 r u v x) - C.totalE x) / 2)
        simp only [hadj, if_true]
        set p := C.eff2 r (C.ident u) (C.ident v) (x u, x v)
        have h := sum_update2 (fun w l => C.unitE w l) huv p.1 p.2 x
        have hd : C.dE2 r u v (fun _ => C.calm) x = C.totalE (C.act2 r u v x) - C.totalE x := by
          simp only [dE2, reservoir, hcalm, ite_self, zero_mul, sub_zero, totalE, act2, hadj,
            if_true]
          linarith
        rw [hd]
      · exact absurd (by simp [medium, act2, hadj]) hne

lemma adj_aut {σ : Equiv.Perm U} (hσ : C.medium.IsAut σ) (u v : U) :
    C.adj (σ u) (σ v) ↔ C.adj u v := by
  have hn := hσ.1 u v
  simp only [medium] at hn
  by_cases huv : u = v
  · subst huv; simp [C.adj_irrefl]
  · have hσuv : σ u ≠ σ v := fun e => huv (σ.injective e)
    simpa [huv, hσuv] using hn

theorem p2c : C.medium.P2c := by
  intro σ hσ
  have hid : ∀ u, C.ident (σ u) = C.ident u := hσ.2.1
  let τ : (R₁ × U) ⊕ (R₂ × U × U) ≃ (R₁ × U) ⊕ (R₂ × U × U) :=
    Equiv.sumCongr (Equiv.prodCongr (Equiv.refl R₁) σ)
      (Equiv.prodCongr (Equiv.refl R₂) (Equiv.prodCongr σ σ))
  refine ⟨τ, ?_⟩
  rintro (⟨r, u⟩ | ⟨r, u, v⟩) s x
  · have hres : C.res (s ∘ σ.symm) (σ u) = C.res s u := by simp [res, hid]
    refine ⟨?_, ?_⟩
    · simp only [τ, medium, Equiv.sumCongr_apply, Sum.map_inl, Equiv.prodCongr_apply,
        Equiv.refl_apply, Prod.map, dE1, unitE, Function.comp_apply, Equiv.symm_apply_apply, hid,
        hres]
    · simp only [τ, medium, Equiv.sumCongr_apply, Sum.map_inl, Equiv.prodCongr_apply,
        Equiv.refl_apply, Prod.map, act1, Function.comp_apply, Equiv.symm_apply_apply, hid]
      rw [Function.update_comp_equiv]
      simp
  · have hadj := C.adj_aut hσ u v
    have hres : C.reservoir (s ∘ σ.symm) (σ u) (σ v) = C.reservoir s u v := by
      simp [reservoir, hid]
    refine ⟨?_, ?_⟩
    · simp only [τ, medium, Equiv.sumCongr_apply, Sum.map_inr, Equiv.prodCongr_apply,
        Equiv.refl_apply, Prod.map, hadj]
      split_ifs
      · simp only [dE2, unitE, Function.comp_apply, Equiv.symm_apply_apply, hid, hres]
      · rfl
    · simp only [τ, medium, Equiv.sumCongr_apply, Sum.map_inr, Equiv.prodCongr_apply,
        Equiv.refl_apply, Prod.map, act2, hadj]
      split_ifs
      · simp only [Function.comp_apply, Equiv.symm_apply_apply, hid]
        rw [Function.update_comp_equiv, Function.update_comp_equiv]
        simp
      · rfl

theorem p3 : C.medium.P3 := by
  refine ⟨?_, fun σ hσ => ?_⟩
  · rintro (⟨r, u⟩ | ⟨r, u, v⟩) s s' x hs
    · have hR : C.res s u = C.res s' u := by
        unfold res
        split_ifs with h1
        · rw [hs u (by simp [medium, boundary, h1])]
        · rfl
      exact ⟨by simp only [medium, dE1, hR], rfl⟩
    · have hR : C.reservoir s u v = C.reservoir s' u v := by
        unfold reservoir
        split_ifs with h1 h2
        · rw [hs u (by simp [medium, boundary, h1])]
        · rw [hs v (by simp [medium, boundary, h2])]
        · rfl
      exact ⟨by simp only [medium, dE2, hR], rfl⟩
  · funext u
    have : C.ident (σ.symm u) = C.ident u := by
      have := hσ.2.1 (σ.symm u)
      rw [Equiv.apply_symm_apply] at this
      exact this.symm
    simp [medium, this]

/-- **M4 (lattice media are admitted)**: a lattice reaction medium satisfies the six clauses
when its rule tables pass local obligations: content levels (P1a), conservation by rules among
interior cells (P2a), a zero calm potential, involutive reverses that undo every effective rule,
and conductances equal on a rule and its reverse (P2b). P1b, P2c and P3 hold by construction. -/
theorem axioms [DecidableEq (U → Λ)] [Fintype R₁] [Fintype R₂]
    (hP1a : (∀ n, {l : Λ | C.content l ≤ n}.Finite) ∧ (∃ K, ∀ l, C.content l ≤ K) ∧
      ∃! l, C.content l = 0)
    (hcons1 : ∀ r i l, i ∉ C.bid → C.content (C.eff1 r i l) = C.content l)
    (hcons2 : ∀ r i j p, i ∉ C.bid → j ∉ C.bid →
      C.content (C.eff2 r i j p).1 + C.content (C.eff2 r i j p).2 = C.content p.1 + C.content p.2)
    (hcalm : C.μ C.calm = 0)
    (hinv1 : Function.Involutive C.rev1) (hinv2 : Function.Involutive C.rev2)
    (hback1 : ∀ r i l, C.eff1 r i l ≠ l → C.eff1 (C.rev1 r) i (C.eff1 r i l) = l)
    (hback2 : ∀ r i j p, C.eff2 r i j p ≠ p → C.eff2 (C.rev2 r) i j (C.eff2 r i j p) = p)
    (hc1 : ∀ r i l, C.eff1 r i l ≠ l → C.c1 (C.rev1 r) i (C.eff1 r i l) = C.c1 r i l)
    (hc2 : ∀ r i j p, C.eff2 r i j p ≠ p → C.c2 (C.rev2 r) i j (C.eff2 r i j p) = C.c2 r i j p) :
    C.medium.Axioms :=
  ⟨C.p1a hP1a, C.p1b, C.p2a hcons1 hcons2,
    C.p2b hcalm hinv1 hinv2 hback1 hback2 hc1 hc2, C.p2c, C.p3⟩

end LatticeCRN

end CGB
