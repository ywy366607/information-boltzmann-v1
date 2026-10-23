import CGBAxioms.Medium

/-!
# Consistency models, independence witnesses and the bottleneck theorem

**Consistency.**
* `twinFlip`: closed. Two identical units, each flipping between two conformations.
* `seed`: open, driven and growing. Four sites on a ring (a homogeneous space): a source
  and a sink on the boundary, opposite each other, and two identical interior sites. Life
  starts as one occupied site at the source; under food content is taken in and spreads,
  so the number of units grows (`seed_grows`). All six clauses hold, the reflection that
  swaps the interior sites is a genuine symmetry, and under food the open dynamics has no
  detailed balance (`seed_driven`), while in the calm environment it does.

**Independence.** For each clause a medium violating it while satisfying the other five;
P1a and P3 have one witness per part. Every witness except the P2c one has a genuine
symmetry (the swap of two identical units).

| clause | witness | how it fails |
|---|---|---|
| P1a, finite under bounds | `twinTape` | a tape of unbounded length at fixed content |
| P1a, capacity | `twinBag` | a unit that could hold unboundedly much content |
| P1a, one vacuum | `twinBlank` | several free states |
| P1b | `farFlip` | a unit's rate reads a unit it does not touch |
| P2a | `twinToggle` | content created from nothing |
| P2b | `twinCycle` | each unit cycles through three conformations one way |
| P2c | `triLookup` | one pair of identical units stores its own conductance |
| P3, birth state | `pairPreset` | the birth state is set asymmetric by hand |
| P3, sensing | `innerClock` | an interior unit senses the environment directly |

**Bottleneck.** `xhop_pages`: with all units in contact, P2c forces every pairwise
conductance to depend only on the two units' identity pages. A rulebook with `k` pages has
at most `k²` free conductances, a lookup table has `|U|²`; ρ decides which.
-/

open Finset

namespace CGB

namespace Models

open Medium

/-! ## Exclusion hopping: one unit of content moves to an empty site -/

section XHop

variable {U : Type*} [DecidableEq U]

/-- Move the content of `u` to the empty site `v`. -/
def xhopAct (u v : U) (x : U → Bool) : U → Bool :=
  if u ≠ v ∧ x u = true ∧ x v = false then x ∘ Equiv.swap u v else x

lemma xhopAct_of_ok {u v : U} {x : U → Bool} (h : u ≠ v ∧ x u = true ∧ x v = false) :
    xhopAct u v x = x ∘ Equiv.swap u v := ite_eq_left h

lemma xhopAct_of_not {u v : U} {x : U → Bool} (h : ¬ (u ≠ v ∧ x u = true ∧ x v = false)) :
    xhopAct u v x = x := ite_eq_right h

lemma xhopAct_rev_ok {u v : U} {x : U → Bool} (h : u ≠ v ∧ x u = true ∧ x v = false) :
    v ≠ u ∧ xhopAct u v x v = true ∧ xhopAct u v x u = false := by
  rw [xhopAct_of_ok h]
  exact ⟨h.1.symm, by simp [h.2.1], by simp [h.2.2]⟩

lemma xhopAct_back {u v : U} {x : U → Bool} (h : u ≠ v ∧ x u = true ∧ x v = false) :
    xhopAct v u (xhopAct u v x) = x := by
  rw [xhopAct_of_ok (xhopAct_rev_ok h), xhopAct_of_ok h]
  funext w
  simp [Equiv.swap_comm v u]

lemma xhopAct_sum [Fintype U] (f : Bool → ℕ) (u v : U) (x : U → Bool) :
    ∑ w, f (xhopAct u v x w) = ∑ w, f (x w) := by
  by_cases h : u ≠ v ∧ x u = true ∧ x v = false
  · rw [xhopAct_of_ok h]
    exact Equiv.sum_comp (Equiv.swap u v) (fun w => f (x w))
  · rw [xhopAct_of_not h]

lemma xhopAct_outside {u v w : U} (x : U → Bool) (hu : w ≠ u) (hv : w ≠ v) :
    xhopAct u v x w = x w := by
  unfold xhopAct
  split_ifs <;> simp [Equiv.swap_apply_of_ne_of_ne hu hv]

lemma xhopAct_congr {u v w : U} {x y : U → Bool} (hu : x u = y u) (hv : x v = y v)
    (hw : w = u ∨ w = v) : xhopAct u v x w = xhopAct u v y w := by
  simp only [xhopAct, hu, hv]
  rcases hw with rfl | rfl <;> split_ifs <;> simp [hu, hv]

lemma symm_swap (σ : Equiv.Perm U) (u v w : U) :
    σ.symm (Equiv.swap (σ u) (σ v) w) = Equiv.swap u v (σ.symm w) := by
  by_cases hu : w = σ u
  · subst hu; simp
  · by_cases hv : w = σ v
    · subst hv; simp
    · rw [Equiv.swap_apply_of_ne_of_ne hu hv, Equiv.swap_apply_of_ne_of_ne
        (fun h => hu ((Equiv.symm_apply_eq σ).1 h)) (fun h => hv ((Equiv.symm_apply_eq σ).1 h))]

lemma xhopAct_perm (σ : Equiv.Perm U) (u v : U) (x : U → Bool) :
    xhopAct (σ u) (σ v) (x ∘ σ.symm) = xhopAct u v x ∘ σ.symm := by
  have hc : (σ u ≠ σ v ∧ (x ∘ σ.symm) (σ u) = true ∧ (x ∘ σ.symm) (σ v) = false) ↔
      (u ≠ v ∧ x u = true ∧ x v = false) := by simp
  by_cases h : u ≠ v ∧ x u = true ∧ x v = false
  · rw [xhopAct_of_ok (hc.2 h), xhopAct_of_ok h]
    funext w
    simp [symm_swap]
  · rw [xhopAct_of_not (mt hc.1 h), xhopAct_of_not h]

/-- The content of `a` alone, moved by a hop, can only have gone along `a → b`. -/
lemma xhopAct_ind {a b p q : U} (hab : a ≠ b)
    (h : xhopAct p q (fun w => decide (w = a)) = fun w => decide (w = b)) : p = a ∧ q = b := by
  by_cases hc : p ≠ q ∧ decide (p = a) = true ∧ decide (q = a) = false
  · rw [xhopAct_of_ok hc] at h
    have hp : p = a := by simpa using hc.2.1
    have hq := congrFun h q
    simp only [Function.comp_apply, Equiv.swap_apply_right] at hq
    exact ⟨hp, by simpa [hp] using hq.symm⟩
  · rw [xhopAct_of_not hc] at h
    simpa [hab] using congrFun h a

end XHop

lemma ident_swap {U : Type*} [DecidableEq U] {ident : U → ℕ} {a b : U}
    (h : ident a = ident b) (w : U) : ident (Equiv.swap a b w) = ident w := by
  by_cases ha : w = a
  · subst ha; simp [h]
  · by_cases hb : w = b
    · subst hb; simp [h]
    · rw [Equiv.swap_apply_of_ne_of_ne ha hb]

/-- Homogeneous contact by equality: any swap is a symmetry. -/
lemma eq_homogeneous {U : Type*} [DecidableEq U] (u v : U) :
    ∃ σ : Equiv.Perm U, (∀ a b, σ a = σ b ↔ a = b) ∧ σ u = v :=
  ⟨Equiv.swap u v, fun _ _ => (Equiv.swap u v).injective.eq_iff, Equiv.swap_apply_left u v⟩

/-- Occupancy content: a site is empty (the vacuum) or holds one unit. -/
lemma bool_p1a {U A I : Type*} {M : Medium U Bool A I} (h : M.content = Bool.toNat) :
    M.P1a := by
  rw [Medium.P1a, h]
  exact ⟨fun _ => Set.toFinite _, ⟨1, by decide⟩, ⟨false, rfl, by decide⟩⟩

/-! ## Exclusion hopping with all units in contact -/

section XHopMedium

variable {U : Type*} [Fintype U] [DecidableEq U]

/-- Content hops to empty sites at conductance `c u v`; no boundary, no sensing. -/
noncomputable def xhop (U : Type*) [DecidableEq U] (c : U → U → ℝ) (hc : ∀ u v, 0 ≤ c u v)
    (ident : U → ℕ) (init : U → Bool) : Medium U Bool (U × U) Unit where
  near _ _ := True
  ident := ident
  boundary := ∅
  content := Bool.toNat
  parts a := {a.1, a.2}
  act a _ x := xhopAct a.1 a.2 x
  rate a _ x := if a.1 ≠ a.2 ∧ x a.1 = true ∧ x a.2 = false then c a.1 a.2 else 0
  rate_nonneg a _ x := by
    split_ifs
    exacts [hc _ _, le_rfl]
  init := init
  calm _ := ()

variable {c : U → U → ℝ} {hc : ∀ u v, 0 ≤ c u v} {ident : U → ℕ} {init : U → Bool}

omit [Fintype U] in
lemma xhop_isAut {σ : Equiv.Perm U} (hid : ∀ u, ident (σ u) = ident u) :
    (xhop U c hc ident init).IsAut σ :=
  ⟨fun _ _ => Iff.rfl, hid, fun _ => by simp [xhop]⟩

omit [Fintype U] in
lemma xhop_p1b : (xhop U c hc ident init).P1b := by
  refine ⟨fun u v => ⟨Equiv.swap u v, fun _ _ => Iff.rfl, Equiv.swap_apply_left u v⟩,
    fun _ _ _ _ _ => trivial, fun a _ x w hw => ?_, fun a s s' x y hxy => ?_⟩
  · simp only [xhop, Finset.mem_insert, Finset.mem_singleton, not_or] at hw
    exact xhopAct_outside x hw.1 hw.2
  · have h1 : x a.1 = y a.1 := (hxy _ (by simp [xhop])).1
    have h2 : x a.2 = y a.2 := (hxy _ (by simp [xhop])).1
    refine ⟨by simp only [xhop, h1, h2], fun w hw => ?_⟩
    simp only [xhop, Finset.mem_insert, Finset.mem_singleton] at hw
    exact xhopAct_congr h1 h2 hw

lemma xhop_p2a : (xhop U c hc ident init).P2a :=
  fun a _ x _ _ => xhopAct_sum Bool.toNat a.1 a.2 x

lemma xhop_p2b (hsym : ∀ u v, c u v = c v u) : (xhop U c hc ident init).P2b := by
  refine p2b_of_micro _ Prod.swap (fun a => Prod.swap_swap a) fun a x hne => ?_
  have h : a.1 ≠ a.2 ∧ x a.1 = true ∧ x a.2 = false := by
    by_contra h
    exact hne (xhopAct_of_not h)
  refine ⟨xhopAct_back h, ?_⟩
  simp only [xhop, Prod.fst_swap, Prod.snd_swap]
  rw [ite_eq_left (xhopAct_rev_ok h), ite_eq_left h, hsym]

omit [Fintype U] in
lemma xhop_p2c (hinv : ∀ σ, (xhop U c hc ident init).IsAut σ → ∀ u v, c (σ u) (σ v) = c u v) :
    (xhop U c hc ident init).P2c := by
  intro σ hσ
  refine ⟨Equiv.prodCongr σ σ, fun a s x => ⟨?_, xhopAct_perm σ a.1 a.2 x⟩⟩
  simp [xhop, hinv σ hσ]

omit [Fintype U] in
lemma xhop_senses : (xhop U c hc ident init).OnlyBoundarySenses :=
  fun _ _ _ _ _ => ⟨rfl, rfl⟩

omit [Fintype U] in
/-- **Bottleneck theorem.** With all units in contact, P2c forces each conductance to
depend only on the identity pages of its two ends. -/
theorem xhop_pages (h : (xhop U c hc ident init).P2c) {u v u' v' : U} (huv : u ≠ v)
    (huv' : u' ≠ v') (hu : ident u = ident u') (hv : ident v = ident v') : c u v = c u' v' := by
  obtain ⟨v1, hv1⟩ : ∃ v1, v1 = Equiv.swap u u' v := ⟨_, rfl⟩
  have hv1u' : u' ≠ v1 := by
    intro h
    apply huv
    have h' : Equiv.swap u u' v = Equiv.swap u u' u := by
      rw [← hv1, ← h, Equiv.swap_apply_left]
    exact ((Equiv.swap u u').injective h').symm
  have hid1 : ident v1 = ident v' := by rw [hv1, ident_swap hu, hv]
  let σ : Equiv.Perm U := Equiv.swap v1 v' * Equiv.swap u u'
  have hσu : σ u = u' := by
    simp only [σ, Equiv.Perm.mul_apply, Equiv.swap_apply_left]
    exact Equiv.swap_apply_of_ne_of_ne hv1u' huv'
  have hσv : σ v = v' := by
    simp only [σ, Equiv.Perm.mul_apply, ← hv1, Equiv.swap_apply_left]
  have hid : ∀ w, ident (σ w) = ident w := fun w => by
    simp only [σ, Equiv.Perm.mul_apply]
    rw [ident_swap hid1, ident_swap hu]
  obtain ⟨τ, hτ⟩ := h σ (xhop_isAut hid)
  obtain ⟨hr, ha⟩ := hτ (u, v) (fun _ => ()) (fun w => decide (w = u))
  have hx : (fun w => decide (w = u)) ∘ σ.symm = fun w => decide (w = u') := by
    funext w
    simp [Equiv.symm_apply_eq, hσu]
  have hy : (xhop U c hc ident init).act (u, v) (fun _ => ()) (fun w => decide (w = u)) ∘ σ.symm
      = fun w => decide (w = v') := by
    have hok : u ≠ v ∧ decide (u = u) = true ∧ decide (v = u) = false := by simp [huv, huv.symm]
    simp only [xhop]
    rw [xhopAct_of_ok hok]
    funext w
    simp only [Function.comp_apply, Equiv.swap_apply_eq_iff, decide_eq_decide]
    rw [Equiv.symm_apply_eq, Equiv.swap_apply_left, hσv]
  rw [hx] at ha hr
  rw [hy] at ha
  obtain ⟨h1, h2⟩ := xhopAct_ind huv' ha
  rw [show τ (u, v) = (u', v') from Prod.ext h1 h2] at hr
  simp [xhop, huv, huv', Ne.symm huv, Ne.symm huv'] at hr
  exact hr.symm

end XHopMedium

/-! ## Identical units out of contact, each with its own local events -/

section Solo

variable {U Λ E : Type*} [DecidableEq U]

/-- Identical units out of contact; each changes its own state by local events `E`. -/
noncomputable def solo (U : Type*) [DecidableEq U] (content : Λ → ℕ) (step : E → Λ → Λ)
    (r : E → Λ → ℝ) (hr : ∀ e ℓ, 0 ≤ r e ℓ) (ℓ₀ : Λ) : Medium U Λ (U × E) Unit where
  near u v := u = v
  ident _ := 0
  boundary := ∅
  content := content
  parts a := {a.1}
  act a _ x := Function.update x a.1 (step a.2 (x a.1))
  rate a _ x := r a.2 (x a.1)
  rate_nonneg _ _ _ := hr _ _
  init _ := ℓ₀
  calm _ := ()

variable {content : Λ → ℕ} {step : E → Λ → Λ} {r : E → Λ → ℝ} {hr : ∀ e ℓ, 0 ≤ r e ℓ}
  {ℓ₀ : Λ}

/-- Every permutation of units is a symmetry of `solo`. -/
lemma solo_isAut (σ : Equiv.Perm U) : (solo U content step r hr ℓ₀).IsAut σ :=
  ⟨fun _ _ => σ.injective.eq_iff, fun _ => rfl, fun _ => by simp [solo]⟩

lemma solo_p1b : (solo U content step r hr ℓ₀).P1b := by
  refine ⟨eq_homogeneous, fun a u hu v hv => ?_, fun a _ x u hu => ?_,
    fun a s s' x y hxy => ?_⟩
  · simp only [solo, Finset.mem_singleton] at hu hv
    exact hu.trans hv.symm
  · simp only [solo, Finset.mem_singleton] at hu
    simp [solo, Function.update_of_ne hu]
  · have h : x a.1 = y a.1 := (hxy _ (by simp [solo])).1
    refine ⟨by simp only [solo, h], fun u hu => ?_⟩
    simp only [solo, Finset.mem_singleton] at hu
    subst hu
    simp [solo, h]

/-- Local content conservation of each event gives P2a. -/
lemma solo_p2a [Fintype U] (h : ∀ e ℓ, content (step e ℓ) = content ℓ) :
    (solo U content step r hr ℓ₀).P2a := by
  intro a _ x _ _
  refine Finset.sum_congr rfl fun w _ => ?_
  by_cases hw : w = a.1
  · subst hw; simp [solo, h]
  · simp [solo, Function.update_of_ne hw]

lemma solo_p2c : (solo U content step r hr ℓ₀).P2c := by
  intro σ _
  refine ⟨Equiv.prodCongr σ (Equiv.refl E), fun a s x => ⟨?_, ?_⟩⟩
  · simp [solo]
  · simp only [solo, Equiv.prodCongr_apply, Prod.map_fst, Prod.map_snd, Equiv.refl_apply,
      Function.comp_apply, Equiv.symm_apply_apply]
    rw [Function.update_comp_equiv]
    simp

lemma solo_p3 : (solo U content step r hr ℓ₀).P3 :=
  ⟨fun _ _ _ _ _ => ⟨rfl, rfl⟩, fun _ _ => rfl⟩

/-- Local microscopic reversibility of each unit gives P2b. -/
lemma solo_p2b [Fintype U] [Fintype E] [DecidableEq (U → Λ)] (rev : E → E)
    (hinv : Function.Involutive rev)
    (hloc : ∀ e ℓ, step e ℓ ≠ ℓ → step (rev e) (step e ℓ) = ℓ ∧ r (rev e) (step e ℓ) = r e ℓ) :
    (solo U content step r hr ℓ₀).P2b := by
  refine p2b_of_micro _ (fun a => (a.1, rev a.2)) (fun a => by simp [hinv a.2])
    fun a x hne => ?_
  have hs : step a.2 (x a.1) ≠ x a.1 := fun h => hne (by simp [solo, h])
  obtain ⟨h1, h2⟩ := hloc a.2 (x a.1) hs
  exact ⟨by simp [solo, h1], by simp [solo, h2]⟩

end Solo

/-! ## Two conformations of one unit of content -/

/-- Exchange the two conformations `1 ↔ 2` of one unit of content; the vacuum `0` stays. -/
def flip12 (ℓ : Fin 3) : Fin 3 := if ℓ = 1 then 2 else if ℓ = 2 then 1 else ℓ

/-- Content cost on `Fin 3`: the vacuum is free, both conformations hold one unit. -/
def flipCost (ℓ : Fin 3) : ℕ := if ℓ = 0 then 0 else 1

lemma flip12_flip12 (ℓ : Fin 3) : flip12 (flip12 ℓ) = ℓ := by revert ℓ; decide

lemma flipCost_flip12 (ℓ : Fin 3) : flipCost (flip12 ℓ) = flipCost ℓ := by revert ℓ; decide

lemma flipCost_p1a {U A I : Type*} {M : Medium U (Fin 3) A I} (h : M.content = flipCost) :
    M.P1a := by
  rw [Medium.P1a, h]
  exact ⟨fun _ => Set.toFinite _, ⟨1, by decide⟩, ⟨0, rfl, by decide⟩⟩

/-! ## Consistency (closed) -/

/-- Two identical units, each flipping its conformation; closed and reversible. -/
noncomputable def twinFlip : Medium (Fin 2) (Fin 3) (Fin 2 × Unit) Unit :=
  solo (Fin 2) flipCost (fun _ => flip12) (fun _ _ => 1) (fun _ _ => zero_le_one) 1

theorem twinFlip_axioms : twinFlip.Axioms :=
  ⟨flipCost_p1a rfl, solo_p1b, solo_p2a fun _ => flipCost_flip12,
    solo_p2b id (fun _ => rfl) fun _ ℓ _ => ⟨flip12_flip12 ℓ, rfl⟩, solo_p2c, solo_p3⟩

theorem twinFlip_swap_isAut : twinFlip.IsAut (Equiv.swap 0 1) := solo_isAut _

/-- **A rock**: two identical units that never change. It satisfies all six clauses, so
"unpaid structure decays" (T0c, T4) does not follow from them: decay needs a further
premise (ergodicity at finite temperature). -/
noncomputable def rock : Medium (Fin 2) (Fin 3) (Fin 2 × Unit) Unit :=
  solo (Fin 2) flipCost (fun _ => flip12) (fun _ _ => 0) (fun _ _ => le_rfl) 1

theorem rock_axioms : rock.Axioms :=
  ⟨flipCost_p1a rfl, solo_p1b, solo_p2a fun _ => flipCost_flip12,
    solo_p2b id (fun _ => rfl) fun _ ℓ _ => ⟨flip12_flip12 ℓ, rfl⟩, solo_p2c, solo_p3⟩

/-- The rock is frozen: no transition ever has positive rate, in any environment. -/
theorem rock_frozen (s : Fin 2 → Unit) (x y : Fin 2 → Fin 3) : (rock.jump s).rate x y = 0 := by
  simp only [jump]
  exact Finset.sum_eq_zero fun a _ => by split_ifs <;> simp [rock, solo]

/-! ## Consistency (open, driven, growing): the seed -/

/-- Contact on a ring of four sites: each site touches itself and its two neighbours. -/
def ringAdj (u v : Fin 4) : Prop := v = u ∨ v = u + 1 ∨ u = v + 1

instance : DecidableRel ringAdj := fun _ _ => by unfold ringAdj; infer_instance

lemma ringAdj_refl (u : Fin 4) : ringAdj u u := by revert u; decide

lemma ringAdj_symm {u v : Fin 4} (h : ringAdj u v) : ringAdj v u := by revert u v; decide

lemma ringAdj_shift (k a b : Fin 4) : ringAdj (a + k) (b + k) ↔ ringAdj a b := by
  revert k a b; decide

/-- Identity pages: source `1`, sink `2`, the two interior sites share page `0`. -/
def ringIdent (w : Fin 4) : ℕ := if w = 0 then 1 else if w = 2 then 2 else 0

/-- Four sites on a ring; source `0` and sink `2` on the boundary, opposite each other; `1`
and `3` interior. A site is empty or holds one unit. The source takes content in when its
stimulus says food is present, the sink excretes when its drain is open, and content hops
to empty sites in contact. In the calm environment neither reservoir exchanges. Life
starts as one unit at the source. -/
noncomputable def seed : Medium (Fin 4) Bool ((Fin 4 × Fin 4) ⊕ Bool) Bool where
  near := ringAdj
  ident := ringIdent
  boundary := {0, 2}
  content := Bool.toNat
  parts
    | .inl a => if ringAdj a.1 a.2 then {a.1, a.2} else ∅
    | .inr true => {0}
    | .inr false => {2}
  act
    | .inl a, _, x => if ringAdj a.1 a.2 then xhopAct a.1 a.2 x else x
    | .inr true, s, x => if s 0 = true then Function.update x 0 true else x
    | .inr false, s, x => if s 2 = true then Function.update x 2 false else x
  rate
    | .inl a, _, x =>
      if ringAdj a.1 a.2 ∧ a.1 ≠ a.2 ∧ x a.1 = true ∧ x a.2 = false then 1 else 0
    | .inr true, s, x => if s 0 = true ∧ x 0 = false then 1 else 0
    | .inr false, s, x => if s 2 = true ∧ x 2 = true then 1 else 0
  rate_nonneg := by
    rintro (a | _ | _) s x <;> simp only <;> (try split_ifs) <;> norm_num
  init w := decide (w = 0)
  calm _ := false

lemma seed_p1b : seed.P1b := by
  refine ⟨fun u v => ⟨Equiv.addRight (v - u), fun a b => ringAdj_shift _ a b, by simp⟩,
    ?_, ?_, ?_⟩
  · rintro (a | _ | _) u hu v hv
    · by_cases ha : ringAdj a.1 a.2
      · simp only [seed, ha, ite_true, Finset.mem_insert, Finset.mem_singleton] at hu hv
        change ringAdj u v
        rcases hu with rfl | rfl <;> rcases hv with rfl | rfl
        exacts [ringAdj_refl _, ha, ringAdj_symm ha, ringAdj_refl _]
      · simp [seed, ha] at hu
    · simp only [seed, Finset.mem_singleton] at hu hv
      subst hu hv; exact ringAdj_refl _
    · simp only [seed, Finset.mem_singleton] at hu hv
      subst hu hv; exact ringAdj_refl _
  · rintro (a | _ | _) s x w hw
    · by_cases ha : ringAdj a.1 a.2
      · simp only [seed, ha, ite_true, Finset.mem_insert, Finset.mem_singleton, not_or] at hw
        simp only [seed, ha, ite_true]
        exact xhopAct_outside x hw.1 hw.2
      · simp [seed, ha]
    · simp only [seed, Finset.mem_singleton] at hw
      simp only [seed]
      split_ifs <;> simp [Function.update_of_ne hw]
    · simp only [seed, Finset.mem_singleton] at hw
      simp only [seed]
      split_ifs <;> simp [Function.update_of_ne hw]
  · rintro (a | _ | _) s s' x y hxy
    · by_cases ha : ringAdj a.1 a.2
      · have h1 : x a.1 = y a.1 := (hxy _ (by simp [seed, ha])).1
        have h2 : x a.2 = y a.2 := (hxy _ (by simp [seed, ha])).1
        refine ⟨by simp only [seed, h1, h2], fun w hw => ?_⟩
        simp only [seed, ha, ite_true, Finset.mem_insert, Finset.mem_singleton] at hw
        simp only [seed, ha, ite_true]
        exact xhopAct_congr h1 h2 hw
      · exact ⟨by simp [seed, ha], fun w hw => by simp [seed, ha] at hw⟩
    · obtain ⟨h, hs⟩ := hxy 2 (by simp [seed])
      refine ⟨by simp [seed, h, hs], fun w hw => ?_⟩
      simp only [seed, Finset.mem_singleton] at hw
      subst hw; simp only [seed, hs]; split_ifs <;> simp [h]
    · obtain ⟨h, hs⟩ := hxy 0 (by simp [seed])
      refine ⟨by simp [seed, h, hs], fun w hw => ?_⟩
      simp only [seed, Finset.mem_singleton] at hw
      subst hw; simp only [seed, hs]; split_ifs <;> simp [h]

lemma seed_p2a : seed.P2a := by
  rintro (a | _ | _) s x hc _
  · change ∑ w, Bool.toNat ((if ringAdj a.1 a.2 then xhopAct a.1 a.2 x else x) w) =
      ∑ w, Bool.toNat (x w)
    split_ifs
    exacts [xhopAct_sum _ _ _ _, rfl]
  · simp [Closed, seed] at hc
  · simp [Closed, seed] at hc

lemma seed_p2b : seed.P2b := by
  refine p2b_of_micro _ (Sum.map Prod.swap id) (fun a => by cases a <;> simp) ?_
  rintro (a | _ | _) x hne
  · have ha : ringAdj a.1 a.2 := by
      by_contra ha
      exact hne (by simp [seed, ha])
    have h : a.1 ≠ a.2 ∧ x a.1 = true ∧ x a.2 = false := by
      by_contra h
      exact hne (by simp only [seed, ha, ite_true]; exact xhopAct_of_not h)
    refine ⟨?_, ?_⟩
    · simp only [seed, Sum.map_inl, Prod.fst_swap, Prod.snd_swap, ha, ringAdj_symm ha,
        ite_true]
      exact xhopAct_back h
    · simp only [seed, Sum.map_inl, Prod.fst_swap, Prod.snd_swap, ha, ite_true]
      rw [ite_eq_left ⟨ringAdj_symm ha, xhopAct_rev_ok h⟩, ite_eq_left ⟨trivial, h⟩]
  · exact absurd (by simp [seed]) hne
  · exact absurd (by simp [seed]) hne

/-- Every symmetry of the seed fixes the source and the sink. -/
lemma seed_aut_fix {σ : Equiv.Perm (Fin 4)} (hσ : seed.IsAut σ) :
    σ.symm 0 = 0 ∧ σ.symm 2 = 2 ∧ σ 0 = 0 ∧ σ 2 = 2 := by
  have h0 : σ 0 = 0 := by
    have key : ∀ w, ringIdent w = 1 → w = 0 := by decide
    exact key _ (hσ.2.1 0)
  have h2 : σ 2 = 2 := by
    have key : ∀ w, ringIdent w = 2 → w = 2 := by decide
    exact key _ (hσ.2.1 2)
  exact ⟨by rw [Equiv.symm_apply_eq, h0], by rw [Equiv.symm_apply_eq, h2], h0, h2⟩

lemma seed_p2c : seed.P2c := by
  intro σ hσ
  obtain ⟨h0', h2', h0, h2⟩ := seed_aut_fix hσ
  have hadj : ∀ u v, ringAdj (σ u) (σ v) ↔ ringAdj u v := hσ.1
  refine ⟨Equiv.sumCongr (Equiv.prodCongr σ σ) (Equiv.refl Bool), ?_⟩
  rintro (a | _ | _) s x
  · refine ⟨?_, ?_⟩
    · simp [seed, hadj]
    · simp only [seed, Equiv.sumCongr_apply, Sum.map_inl, Equiv.prodCongr_apply,
        Prod.map_fst, Prod.map_snd, hadj]
      split_ifs
      exacts [xhopAct_perm σ a.1 a.2 x, rfl]
  · refine ⟨by simp [seed, h2'], ?_⟩
    simp only [seed, Equiv.sumCongr_apply, Sum.map_inr, Equiv.refl_apply, Function.comp_apply,
      h2']
    split_ifs
    · rw [Function.update_comp_equiv]
      simp [h2]
    · rfl
  · refine ⟨by simp [seed, h0'], ?_⟩
    simp only [seed, Equiv.sumCongr_apply, Sum.map_inr, Equiv.refl_apply, Function.comp_apply,
      h0']
    split_ifs
    · rw [Function.update_comp_equiv]
      simp [h0]
    · rfl

lemma seed_p3 : seed.P3 := by
  refine ⟨?_, fun σ hσ => ?_⟩
  · rintro (a | _ | _) s s' x hs
    · exact ⟨rfl, rfl⟩
    · have h := hs 2 (by simp [seed])
      exact ⟨by simp [seed, h], by simp [seed, h]⟩
    · have h := hs 0 (by simp [seed])
      exact ⟨by simp [seed, h], by simp [seed, h]⟩
  · obtain ⟨h0', -, h0, -⟩ := seed_aut_fix hσ
    funext w
    simp [seed, Equiv.symm_apply_eq, h0]

theorem seed_axioms : seed.Axioms :=
  ⟨bool_p1a rfl, seed_p1b, seed_p2a, seed_p2b, seed_p2c, seed_p3⟩

/-- The reflection swapping the interior sites is a genuine symmetry. -/
theorem seed_swap_isAut : seed.IsAut (Equiv.swap 1 3) :=
  ⟨by show ∀ u v : Fin 4, ringAdj _ _ ↔ ringAdj u v; decide, by decide, by decide⟩

/-- Food at the source and an open drain at the sink. -/
def food : Fin 4 → Bool := fun _ => true

/-- The seed is driven: under food its open dynamics has no detailed balance. Excreted
content never comes back through the sink. In the calm environment it is reversible. -/
theorem seed_driven : ¬ (seed.jump food).Reversible := by
  rintro ⟨π, hpos, hdb⟩
  let e : Fin 4 → Bool := fun _ => false
  let t : Fin 4 → Bool := fun w => decide (w = 2)
  have hte : 0 < (seed.jump food).rate t e := by
    have hterm : seed.act (.inr false) food t = e ∧ t ≠ e := by decide
    refine lt_of_lt_of_le ?_ (Finset.single_le_sum (f := fun a =>
      if seed.act a food t = e ∧ t ≠ e then seed.rate a food t else 0)
      (fun a _ => by split_ifs; exacts [seed.rate_nonneg a food t, le_rfl])
      (Finset.mem_univ (.inr false)))
    simp only [ite_eq_left hterm]
    simp [seed, t, food]
  have het : (seed.jump food).rate e t = 0 := by
    have none : ∀ a, seed.act a food e ≠ t := by decide
    simp only [jump]
    exact Finset.sum_eq_zero fun a _ => ite_eq_right fun h => none a h.1
  have := hdb t e
  rw [het, mul_zero] at this
  exact (mul_pos (hpos t) hte).ne' this

/-- One event of the open dynamics with positive rate. -/
def OpenStep {U Λ A I : Type*} (M : Medium U Λ A I) (s : U → I) (x y : U → Λ) : Prop :=
  ∃ a, 0 < M.rate a s x ∧ M.act a s x = y

/-- **Growth.** From one unit at the source, under food, the seed reaches a state with
three units: intake and spreading create units at empty sites. -/
theorem seed_grows : Relation.ReflTransGen (OpenStep seed food) seed.init
    (fun w => decide (w ≠ 2)) ∧ seed.total seed.init = 1 ∧
    seed.total (fun w => decide (w ≠ 2)) = 3 := by
  refine ⟨?_, by decide, by decide⟩
  have step : ∀ a x y, 0 < seed.rate a food x → seed.act a food x = y →
      OpenStep seed food x y := fun a x y h1 h2 => ⟨a, h1, h2⟩
  refine .tail (.tail (.tail (.single (step (.inl (0, 1)) _ (fun w => decide (w = 1))
    (by simp [seed, ringAdj]) (by decide)))
    (step (.inr true) _ (fun w => decide (w = 0 ∨ w = 1)) (by simp [seed, food]) (by decide)))
    (step (.inl (0, 3)) _ (fun w => decide (w = 1 ∨ w = 3)) (by simp [seed, ringAdj])
      (by decide)))
    (step (.inr true) _ _ (by simp [seed, food]) (by decide))

/-! ## P1a: one witness per part -/

/-- Content cost of a tape: empty is free, any written tape costs one unit. -/
def tapeCost (ℓ : ℕ) : ℕ := if ℓ = 0 then 0 else 1

/-- Two identical tapes that lengthen and shorten at fixed content: unboundedly many states
under a content bound (free information). -/
noncomputable def twinTape : Medium (Fin 2) ℕ (Fin 2 × Bool) Unit :=
  solo (Fin 2) tapeCost (fun b ℓ => if b then (if 1 ≤ ℓ then ℓ + 1 else ℓ) else
      (if 2 ≤ ℓ then ℓ - 1 else ℓ))
    (fun b ℓ => if b then (if 1 ≤ ℓ then 1 else 0) else (if 2 ≤ ℓ then 1 else 0))
    (fun b ℓ => by split_ifs <;> norm_num) 1

theorem twinTape_witness :
    ¬ twinTape.P1a ∧ twinTape.P1b ∧ twinTape.P2a ∧ twinTape.P2b ∧ twinTape.P2c ∧
      twinTape.P3 ∧ twinTape.IsAut (Equiv.swap 0 1) ∧
      (∃ K, ∀ ℓ, twinTape.content ℓ ≤ K) ∧ (∃! ℓ, twinTape.content ℓ = 0) := by
  refine ⟨fun h => ?_, solo_p1b, solo_p2a fun b ℓ => ?_, ?_, solo_p2c, solo_p3, solo_isAut _,
    ⟨1, fun ℓ => by simp only [twinTape, solo, tapeCost]; split_ifs <;> norm_num⟩,
    ⟨0, by simp [twinTape, solo, tapeCost], fun ℓ hℓ => by
      simp only [twinTape, solo, tapeCost] at hℓ; split_ifs at hℓ with h0; exact h0⟩⟩
  · have hfin := h.1 1
    rw [show {ℓ : ℕ | twinTape.content ℓ ≤ 1} = Set.univ from
      Set.eq_univ_of_forall fun ℓ => by
        simp only [Set.mem_ofPred_eq, twinTape, solo, tapeCost]; split_ifs <;> norm_num] at hfin
    exact Set.infinite_univ hfin
  · cases b <;> simp only [tapeCost, Bool.false_eq_true, ite_false, ite_true] <;>
      split_ifs <;> omega
  · refine solo_p2b (fun b => !b) (fun b => Bool.not_not b) fun b ℓ hne => ?_
    cases b
    · simp only [Bool.false_eq_true, ite_false] at hne ⊢
      split_ifs at hne ⊢ <;> first | omega | (simp_all; omega) | simp_all
    · simp only [ite_true] at hne ⊢
      split_ifs at hne ⊢ <;> first | omega | simp_all

/-- Content cost of a bag: empty is free, a bag of `n` items costs `n + 1`. -/
def bagCost : Option (ℕ × Bool) → ℕ
  | none => 0
  | some (n, _) => n + 1

/-- Turn a bag over; its content stays. -/
def bagTurn : Option (ℕ × Bool) → Option (ℕ × Bool)
  | none => none
  | some (n, b) => some (n, !b)

/-- Two identical bags that could each hold unboundedly much: no capacity bound. -/
noncomputable def twinBag : Medium (Fin 2) (Option (ℕ × Bool)) (Fin 2 × Unit) Unit :=
  solo (Fin 2) bagCost (fun _ => bagTurn) (fun _ _ => 1) (fun _ _ => zero_le_one) (some (0, false))

theorem twinBag_witness :
    ¬ twinBag.P1a ∧ twinBag.P1b ∧ twinBag.P2a ∧ twinBag.P2b ∧ twinBag.P2c ∧
      twinBag.P3 ∧ twinBag.IsAut (Equiv.swap 0 1) ∧
      (∀ n, {ℓ | twinBag.content ℓ ≤ n}.Finite) ∧ (∃! ℓ, twinBag.content ℓ = 0) := by
  have hturn : ∀ ℓ, bagTurn (bagTurn ℓ) = ℓ := by
    rintro (_ | ⟨n, b⟩) <;> simp [bagTurn]
  refine ⟨fun h => ?_, solo_p1b, solo_p2a fun _ ℓ => ?_,
    solo_p2b id (fun _ => rfl) fun _ ℓ _ => ⟨hturn ℓ, rfl⟩, solo_p2c, solo_p3, solo_isAut _,
    fun K => ?_, ⟨none, rfl, fun ℓ hℓ => ?_⟩⟩
  · obtain ⟨K, hK⟩ := h.2.1
    have := hK (some (K, false))
    simp [twinBag, solo, bagCost] at this
  · rcases ℓ with _ | ⟨n, b⟩ <;> rfl
  · refine (Set.finite_range (fun p : Option (Fin K × Bool) =>
      p.map fun q => (q.1.val, q.2))).subset fun ℓ hℓ => ?_
    rcases ℓ with _ | ⟨n, b⟩
    · exact ⟨none, rfl⟩
    · have hn : n < K := by
        simp only [Set.mem_ofPred_eq, twinBag, solo, bagCost] at hℓ; omega
      exact ⟨some (⟨n, hn⟩, b), rfl⟩
  · rcases ℓ with _ | ⟨n, b⟩
    · rfl
    · simp [twinBag, solo, bagCost] at hℓ

/-- Two identical units whose three states are all free: no unique vacuum. -/
noncomputable def twinBlank : Medium (Fin 2) (Fin 3) (Fin 2 × Unit) Unit :=
  solo (Fin 2) (fun _ => 0) (fun _ => flip12) (fun _ _ => 1) (fun _ _ => zero_le_one) 1

theorem twinBlank_witness :
    ¬ twinBlank.P1a ∧ twinBlank.P1b ∧ twinBlank.P2a ∧ twinBlank.P2b ∧ twinBlank.P2c ∧
      twinBlank.P3 ∧ twinBlank.IsAut (Equiv.swap 0 1) ∧
      (∀ n, {ℓ | twinBlank.content ℓ ≤ n}.Finite) ∧ (∃ K, ∀ ℓ, twinBlank.content ℓ ≤ K) := by
  refine ⟨fun h => ?_, solo_p1b, solo_p2a fun _ _ => rfl,
    solo_p2b id (fun _ => rfl) fun _ ℓ _ => ⟨flip12_flip12 ℓ, rfl⟩, solo_p2c, solo_p3,
    solo_isAut _, fun _ => Set.toFinite _, ⟨0, fun _ => le_rfl⟩⟩
  obtain ⟨ℓ, -, huniq⟩ := h.2.2
  have h0 := huniq 0 rfl
  have h1 := huniq 1 rfl
  exact absurd (h0.trans h1.symm) (by decide)

/-! ## P2a: content from nothing -/

/-- Two identical sites, each switching between empty and occupied by itself. -/
noncomputable def twinToggle : Medium (Fin 2) Bool (Fin 2 × Unit) Unit :=
  solo (Fin 2) Bool.toNat (fun _ b => !b) (fun _ _ => 1) (fun _ _ => zero_le_one) false

theorem twinToggle_witness :
    ¬ twinToggle.P2a ∧ twinToggle.P1a ∧ twinToggle.P1b ∧ twinToggle.P2b ∧ twinToggle.P2c ∧
      twinToggle.P3 ∧ twinToggle.IsAut (Equiv.swap 0 1) := by
  refine ⟨fun h => ?_, bool_p1a rfl, solo_p1b,
    solo_p2b id (fun _ => rfl) (fun _ b _ => ⟨Bool.not_not b, rfl⟩), solo_p2c, solo_p3,
    solo_isAut _⟩
  have := h (0, ()) (fun _ => ()) (fun _ => false) (by simp [Closed, twinToggle, solo])
    (by simp [twinToggle, solo])
  simp [total, twinToggle, solo, Fin.sum_univ_two] at this

/-! ## P2b: one-way internal cycles -/

/-- A vacuum `0` and three conformations `1 → 2 → 3 → 1` of one unit of content. -/
def cycleStep (ℓ : Fin 4) : Fin 4 := if ℓ = 0 then 0 else if ℓ = 3 then 1 else ℓ + 1

/-- Content cost: the vacuum is free, every conformation holds one unit. -/
def conformCost (ℓ : Fin 4) : ℕ := if ℓ = 0 then 0 else 1

/-- Two identical units, each cycling its conformations one way. Content is conserved and
every permutation of units is a symmetry, yet there is no detailed balance: the chirality
sits in the rulebook, not in space. -/
noncomputable def twinCycle : Medium (Fin 2) (Fin 4) (Fin 2 × Unit) Unit :=
  solo (Fin 2) conformCost (fun _ => cycleStep) (fun _ _ => 1) (fun _ _ => zero_le_one) 1

theorem twinCycle_not_p2b : ¬ twinCycle.P2b := by
  rintro ⟨π, hpos, hdb⟩
  let x : Fin 2 → Fin 4 := fun _ => 1
  let y : Fin 2 → Fin 4 := fun w => if w = 0 then 2 else 1
  have h1 : (twinCycle.jump twinCycle.calm).rate x y = 1 := by
    have only : ∀ b : Fin 2 × Unit, twinCycle.act b twinCycle.calm x = y → b = (0, ()) := by
      decide
    simp only [jump]
    rw [Finset.sum_eq_single (0, ())]
    · rw [ite_eq_left ⟨by decide, by decide⟩]
      simp [twinCycle, solo]
    · intro b _ hb
      rw [ite_eq_right fun h => hb (only b h.1)]
    · simp
  have h0 : (twinCycle.jump twinCycle.calm).rate y x = 0 := by
    have none : ∀ b : Fin 2 × Unit, twinCycle.act b twinCycle.calm y ≠ x := by decide
    simp only [jump]
    exact Finset.sum_eq_zero fun b _ => ite_eq_right fun h => none b h.1
  have := hdb x y
  rw [h1, h0, mul_one, mul_zero] at this
  exact (hpos x).ne' this

theorem twinCycle_witness :
    ¬ twinCycle.P2b ∧ twinCycle.P1a ∧ twinCycle.P1b ∧ twinCycle.P2a ∧ twinCycle.P2c ∧
      twinCycle.P3 ∧ twinCycle.IsAut (Equiv.swap 0 1) := by
  refine ⟨twinCycle_not_p2b, ?_, solo_p1b, solo_p2a (by decide), solo_p2c, solo_p3,
    solo_isAut _⟩
  change (∀ n, {ℓ | conformCost ℓ ≤ n}.Finite) ∧ (∃ K, ∀ ℓ, conformCost ℓ ≤ K) ∧
    ∃! ℓ, conformCost ℓ = 0
  exact ⟨fun _ => Set.toFinite _, ⟨1, by decide⟩, ⟨0, rfl, by decide⟩⟩

/-! ## P1b: action at a distance -/

/-- Two identical units not in contact; each flips its conformation faster when the
*other* unit is in conformation `2`. -/
noncomputable def farFlip : Medium (Fin 2) (Fin 3) (Fin 2) Unit where
  near u v := u = v
  ident _ := 0
  boundary := ∅
  content := flipCost
  parts u := {u}
  act u _ x := Function.update x u (flip12 (x u))
  rate u _ x := if x (u + 1) = 2 then 2 else 1
  rate_nonneg u _ x := by split_ifs <;> norm_num
  init _ := 1
  calm _ := ()

lemma perm_fin2_succ (σ : Equiv.Perm (Fin 2)) (u : Fin 2) : σ (u + 1) = σ u + 1 := by
  have hne : σ (u + 1) ≠ σ u := σ.injective.ne (by revert u; decide)
  revert hne
  generalize σ (u + 1) = a
  generalize σ u = b
  revert a b
  decide

theorem farFlip_witness :
    ¬ farFlip.P1b ∧ farFlip.P1a ∧ farFlip.P2a ∧ farFlip.P2b ∧ farFlip.P2c ∧ farFlip.P3 ∧
      farFlip.IsAut (Equiv.swap 0 1) := by
  refine ⟨fun h => ?_, flipCost_p1a rfl, fun u _ x _ _ => ?_, ?_,
    fun σ _ => ⟨σ, fun u s x => ⟨?_, ?_⟩⟩, ⟨fun _ _ _ _ _ => ⟨rfl, rfl⟩, fun _ _ => rfl⟩,
    ⟨fun _ _ => (Equiv.swap 0 1).injective.eq_iff, fun _ => rfl, fun _ => by simp [farFlip]⟩⟩
  · have := (h.2.2.2 0 (fun _ => ()) (fun _ => ()) (fun _ => 1)
      (Function.update (fun _ => 1) 1 2) (by simp [farFlip])).1
    simp [farFlip] at this
  · refine Finset.sum_congr rfl fun w _ => ?_
    by_cases hw : w = u
    · subst hw; simp [farFlip, flipCost_flip12]
    · simp [farFlip, Function.update_of_ne hw]
  · refine p2b_of_micro _ id (fun _ => rfl) fun u x _ => ⟨?_, ?_⟩
    · simp [farFlip, flip12_flip12]
    · simp [farFlip]
  · have : σ.symm (σ u + 1) = u + 1 := by rw [← perm_fin2_succ, Equiv.symm_apply_apply]
    simp [farFlip, this]
  · simp only [farFlip, Function.comp_apply, Equiv.symm_apply_apply]
    rw [Function.update_comp_equiv]
    simp

/-! ## P2c: a lookup table -/

/-- Three sites with the same identity; the pair `{0, 2}` stores its own conductance. -/
noncomputable def triLookup : Medium (Fin 3) Bool (Fin 3 × Fin 3) Unit :=
  xhop (Fin 3) (fun u v => if (u = 0 ∧ v = 2) ∨ (u = 2 ∧ v = 0) then 2 else 1)
    (fun _ _ => by split_ifs <;> norm_num) (fun _ => 0) (fun _ => false)

theorem triLookup_witness :
    ¬ triLookup.P2c ∧ triLookup.P1a ∧ triLookup.P1b ∧ triLookup.P2a ∧ triLookup.P2b ∧
      triLookup.P3 := by
  refine ⟨fun h => ?_, bool_p1a rfl, xhop_p1b, xhop_p2a,
    xhop_p2b fun u v => by fin_cases u <;> fin_cases v <;> simp,
    ⟨xhop_senses, fun _ _ => rfl⟩⟩
  have := xhop_pages h (u := 0) (v := 2) (u' := 1) (v' := 2) (by decide) (by decide) rfl rfl
  norm_num at this

/-! ## P3: a hand-set birth state, and an interior sensor -/

/-- Two identical sites with the birth state set asymmetric by hand. -/
noncomputable def pairPreset : Medium (Fin 2) Bool (Fin 2 × Fin 2) Unit :=
  xhop (Fin 2) (fun _ _ => 1) (fun _ _ => zero_le_one) (fun _ => 0) (fun w => decide (w = 0))

theorem pairPreset_witness :
    ¬ pairPreset.P3 ∧ pairPreset.P1a ∧ pairPreset.P1b ∧ pairPreset.P2a ∧ pairPreset.P2b ∧
      pairPreset.P2c ∧ pairPreset.IsAut (Equiv.swap 0 1) := by
  have haut : pairPreset.IsAut (Equiv.swap 0 1) := xhop_isAut fun _ => rfl
  refine ⟨fun h => ?_, bool_p1a rfl, xhop_p1b, xhop_p2a, xhop_p2b fun _ _ => rfl,
    xhop_p2c fun _ _ _ _ => rfl, haut⟩
  have := congrFun (h.2 _ haut) 0
  simp [pairPreset, xhop] at this

/-- Two identical interior units (no boundary) whose flip rate reads their own stimulus:
a clock or a time program inside the body. -/
noncomputable def innerClock : Medium (Fin 2) (Fin 3) (Fin 2) Bool where
  near u v := u = v
  ident _ := 0
  boundary := ∅
  content := flipCost
  parts u := {u}
  act u _ x := Function.update x u (flip12 (x u))
  rate u s _ := if s u = true then 2 else 1
  rate_nonneg u s _ := by split_ifs <;> norm_num
  init _ := 1
  calm _ := false

theorem innerClock_witness :
    ¬ innerClock.P3 ∧ innerClock.P1a ∧ innerClock.P1b ∧ innerClock.P2a ∧ innerClock.P2b ∧
      innerClock.P2c ∧ innerClock.IsAut (Equiv.swap 0 1) := by
  refine ⟨fun h => ?_, flipCost_p1a rfl, ⟨eq_homogeneous, fun a u hu v hv => ?_,
    fun a _ x w hw => ?_, fun a s s' x y hxy => ?_⟩, fun u _ x _ _ => ?_, ?_,
    fun σ _ => ⟨σ, fun u s x => ⟨by simp [innerClock], ?_⟩⟩,
    ⟨fun _ _ => (Equiv.swap 0 1).injective.eq_iff, fun _ => rfl,
      fun _ => by simp [innerClock]⟩⟩
  · have := (h.1 0 (fun _ => false) (fun _ => true) (fun _ => 1) (by simp [innerClock])).1
    simp [innerClock] at this
  · simp only [innerClock, Finset.mem_singleton] at hu hv
    exact hu.trans hv.symm
  · simp only [innerClock, Finset.mem_singleton] at hw
    simp [innerClock, Function.update_of_ne hw]
  · obtain ⟨hx, hs⟩ := hxy a (by simp [innerClock])
    refine ⟨by simp [innerClock, hs], fun w hw => ?_⟩
    simp only [innerClock, Finset.mem_singleton] at hw
    subst hw; simp [innerClock, hx]
  · refine Finset.sum_congr rfl fun w _ => ?_
    by_cases hw : w = u
    · subst hw; simp [innerClock, flipCost_flip12]
    · simp [innerClock, Function.update_of_ne hw]
  · refine p2b_of_micro _ id (fun _ => rfl) fun u x _ => ⟨?_, rfl⟩
    simp [innerClock, flip12_flip12]
  · simp only [innerClock, Function.comp_apply, Equiv.symm_apply_apply]
    rw [Function.update_comp_equiv]
    simp

end Models

end CGB
