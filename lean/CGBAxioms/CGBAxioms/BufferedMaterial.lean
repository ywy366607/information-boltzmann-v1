import CGBAxioms.EnergyBudget

/-!
# Finite identity-preserving buffered material transport

Twelve fixed carriers suffice for a body containing at most four units, a
four-unit accessible source and at least four returned carriers. Recycling
uses OLD returned carriers, so there is no same-tick two-hop shortcut.
This is an interpretation construction, not a minimal carrier-count claim.
-/

open Finset

namespace CGB.BufferedMaterial
open MaterialSource

abbrev Token := Fin 12
abbrev Owners := Token → Place

instance : Fintype Place where
  elems := {.body, .source, .sink, .outside}
  complete p := by cases p <;> decide

def Stock (b : ℕ) (o : Owners) : Prop :=
  count o .body = b ∧ count o .source = 4 ∧ count o .outside = 0

private def canonical (b : ℕ) (c : Token) : Place :=
  if c.val < b then .body else if c.val < b + 4 then .source else .sink

private theorem canonical_stock (b : ℕ) (hb : b ≤ 4) : Stock b (canonical b) := by
  have he : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 := by omega
  rcases he with rfl | rfl | rfl | rfl | rfl <;> unfold Stock <;> decide

private theorem total (o : Owners) :
    count o .body + count o .source + count o .sink + count o .outside = 12 := by
  simp only [count, ← Finset.sum_add_distrib]
  calc
    _ = ∑ _c : Token, 1 := by
      apply Finset.sum_congr rfl
      intro c _
      cases o c <;> simp
    _ = 12 := by decide

private theorem select (o : Owners) (p : Place) (n : ℕ) (hn : n ≤ count o p) :
    ∃ a : Finset Token, (∀ c ∈ a, o c = p) ∧ a.card = n := by
  have hc : (univ.filter (fun c => o c = p)).card = count o p := by
    simp [count]
  obtain ⟨a, ha, hn⟩ := Finset.exists_subset_card_eq (n := n) (s := univ.filter
    (fun c => o c = p)) (by omega)
  exact ⟨a, fun c hc => (Finset.mem_filter.mp (ha hc)).2, hn⟩

/-- Local existence with every intake/release/recharge coming from old stock.
The source is replenished by exactly the amount it actually delivers. -/
theorem transport_exists (o : Owners) (b b' i d : ℕ) (hs : Stock b o)
    (hb : b ≤ 4) (hi : i ≤ 4) (hd : d ≤ b) (bal : b' + d = b + i) :
    ∃ o' : Owners, Stock b' o' ∧ Allowed o o' ∧
      moved o o' .source .body = i ∧ moved o o' .body .sink = d ∧
      moved o o' .sink .source = i := by
  have ht := total o
  have hk : 4 ≤ count o .sink := by rcases hs with ⟨h1,h2,h3⟩; omega
  obtain ⟨a, ha, hna⟩ := select o .source i (by rw [hs.2.1]; exact hi)
  obtain ⟨v, hv, hnv⟩ := select o .body d (by rw [hs.1]; exact hd)
  obtain ⟨w, hw, hnw⟩ := select o .sink i (by omega)
  let o' : Owners := fun c => if c ∈ a then .body else if c ∈ v then .sink
    else if c ∈ w then .source else o c
  have row (c : Token) :
      (o c = o' c ∨ (o c = .source ∧ o' c = .body) ∨
        (o c = .body ∧ o' c = .sink) ∨ (o c = .sink ∧ o' c = .source)) ∧
      (if o c = .source ∧ o' c = .body then 1 else 0) = (if c ∈ a then 1 else 0) ∧
      (if o c = .body ∧ o' c = .sink then 1 else 0) = (if c ∈ v then 1 else 0) ∧
      (if o c = .sink ∧ o' c = .source then 1 else 0) = (if c ∈ w then 1 else 0) ∧
      (o' c = .outside ↔ o c = .outside) := by
    by_cases h1 : c ∈ a <;> by_cases h2 : c ∈ v <;> by_cases h3 : c ∈ w
    all_goals try have hA := ha c h1
    all_goals try have hV := hv c h2
    all_goals try have hW := hw c h3
    all_goals cases ho : o c <;> simp_all [o']
  have allowed : Allowed o o' := by
    intro c
    rcases (row c).1 with h | h | h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr (Or.inl h))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl h)))))
  have hI : moved o o' .source .body = i := by
    calc
      _ = ∑ c : Token, if c ∈ a then 1 else 0 :=
        Finset.sum_congr rfl (fun c _ => (row c).2.1)
      _ = i := by simp [hna]
  have hD : moved o o' .body .sink = d := by
    calc
      _ = ∑ c : Token, if c ∈ v then 1 else 0 :=
        Finset.sum_congr rfl (fun c _ => (row c).2.2.1)
      _ = d := by simp [hnv]
  have hR : moved o o' .sink .source = i := by
    calc
      _ = ∑ c : Token, if c ∈ w then 1 else 0 :=
        Finset.sum_congr rfl (fun c _ => (row c).2.2.2.1)
      _ = i := by simp [hnw]
  have ho : count o' .outside = 0 := by
    simpa only [count, (row _).2.2.2.2] using hs.2.2
  have hx : moved o o' .source .outside = 0 ∧ moved o o' .outside .source = 0 := by
    constructor <;> apply Finset.sum_eq_zero
    · intro c _
      have hr := (row c).2.2.2.2
      by_cases hp : o c = .source <;> simp_all
    · intro c _
      have hr := (row c).2.2.2.2
      by_cases hp : o c = .outside
      · have hr' : o' c = .outside := hr.mpr hp
        simp [hp, hr']
      · simp [hp]
  have hbal := balance o o' allowed
  refine ⟨o', ⟨?_, ?_, ho⟩, allowed, hI, hD, hR⟩
  · rw [hI, hD, hs.1] at hbal; omega
  · rw [hI, hR, hx.1, hx.2, hs.2.1] at hbal; omega

variable {S E : Type*} [Fintype S] [DecidableEq S] [Fintype E]

/-- Bounds and supported local budgets, supplied by a concrete process.
They do not assert recovery, life, learning or physical universality. -/
structure Rules (P : Interaction.Machine S ℕ E) where
  content : S → ℕ
  incoming : ℕ → S → E → S → ℕ
  outgoing : ℕ → S → E → S → ℕ
  bounded : ∀ s, content s ≤ 4
  intake_bound : ∀ t s e s', 0 < P.step t s e s' → incoming t s e s' ≤ 4
  release_bound : ∀ t s e s', 0 < P.step t s e s' → outgoing t s e s' ≤ content s
  balance : ∀ t s e s', 0 < P.step t s e s' →
    content s' + outgoing t s e s' = content s + incoming t s e s'

variable {P : Interaction.Machine S ℕ E} (B : Rules P)

abbrev State := {r : S × Owners // Stock (B.content r.1) r.2}

noncomputable instance : Fintype (State B) := Fintype.ofFinite _
noncomputable instance : DecidableEq (State B) := Classical.decEq _

def project (r : State B) : S := r.val.1
def owner (r : State B) : Owners := r.val.2

noncomputable def seed (s : S) : State B :=
  ⟨(s, canonical (B.content s)), canonical_stock _ (B.bounded s)⟩

private theorem available (t : ℕ) (r : State B) (e : E) (s : S)
    (hp : 0 < P.step t (project B r) e s) :
    ∃ o', Stock (B.content s) o' ∧ Allowed (owner B r) o' ∧
      moved (owner B r) o' .source .body = B.incoming t (project B r) e s ∧
      moved (owner B r) o' .body .sink = B.outgoing t (project B r) e s ∧
      moved (owner B r) o' .sink .source = B.incoming t (project B r) e s :=
  transport_exists (owner B r) _ _ _ _ r.property (B.bounded _) (B.intake_bound _ _ _ _ hp)
    (B.release_bound _ _ _ _ hp) (B.balance _ _ _ _ hp)

noncomputable def advance (t : ℕ) (r : State B) (e : E) (s : S) : State B :=
  if hp : 0 < P.step t (project B r) e s then
    ⟨(s, Classical.choose (available B t r e s hp)),
      (Classical.choose_spec (available B t r e s hp)).1⟩
  else seed B s

theorem advance_project (t : ℕ) (r : State B) (e : E) (s : S) :
    project B (advance B t r e s) = s := by
  classical
  unfold advance
  split_ifs <;> rfl

private theorem lift_sum {R : Type*} [Fintype R] [DecidableEq R]
    (f : R → S) (g : S → R) (hg : ∀ s, f (g s) = s) (v : S → ℝ) :
    (∑ r, if r = g (f r) then v (f r) else 0) = ∑ s, v s := by
  classical
  calc
    _ = ∑ r, ∑ s, if r = g s then v s else 0 := by
      apply Finset.sum_congr rfl
      intro r _
      symm
      apply Finset.sum_eq_single (f r)
      · intro s _ hs
        have hn : r ≠ g s := by
          intro he
          have hf := congrArg f he
          rw [hg] at hf
          exact hs hf.symm
        simp [hn]
      · simp
    _ = _ := by rw [Finset.sum_comm]; simp

noncomputable def physical : Interaction.Machine (State B) ℕ E where
  step t r e r' := P.step t (project B r) e (project B r') *
    if r' = advance B t r e (project B r') then 1 else 0
  nonneg t r e r' := mul_nonneg (P.nonneg _ _ _ _) (by split_ifs <;> norm_num)
  sum_one t r := by
    classical
    simp only [mul_ite, mul_one, mul_zero]
    simp_rw [lift_sum (project B) (advance B t r _) (advance_project B t r _)]
    exact P.sum_one t (project B r)

theorem refines : Interaction.Refines (physical B) P (project B) := by
  classical
  intro t r e s
  have h := lift_sum (project B) (advance B t r e) (advance_project B t r e)
    (fun z => if z = s then P.step t (project B r) e z else 0)
  simp only [physical, mul_ite, mul_one, mul_zero]
  calc
    _ = ∑ r', if r' = advance B t r e (project B r') then
        (if project B r' = s then P.step t (project B r) e (project B r') else 0) else 0 := by
      apply Finset.sum_congr rfl
      intro r' _
      split_ifs <;> rfl
    _ = _ := by simpa only [Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte] using h

theorem delivery (t : ℕ) (r r' : State B) (e : E)
    (hp : 0 < (physical B).step t r e r') :
    Allowed (owner B r) (owner B r') ∧
      moved (owner B r) (owner B r') .source .body =
        B.incoming t (project B r) e (project B r') ∧
      moved (owner B r) (owner B r') .body .sink =
        B.outgoing t (project B r) e (project B r') ∧
      moved (owner B r) (owner B r') .sink .source =
        B.incoming t (project B r) e (project B r') := by
  classical
  have he : r' = advance B t r e (project B r') := by
    by_contra hn
    simp [physical, hn] at hp
  have hq : 0 < P.step t (project B r) e (project B r') := by
    change 0 < P.step t (project B r) e (project B r') *
      (if r' = advance B t r e (project B r') then 1 else 0) at hp
    rw [if_pos he, mul_one] at hp
    exact hp
  have hh := (Classical.choose_spec (available B t r e (project B r') hq)).2
  have ho : owner B r' = Classical.choose (available B t r e (project B r') hq) := by
    calc
      _ = owner B (advance B t r e (project B r')) := congrArg (owner B) he
      _ = _ := by simp only [advance, hq, ↓reduceDIte, owner]
  rw [ho]
  exact hh

/-- All original states, all supported events, and the same supplied clock. -/
theorem realized (clock : ℝ) (hc : 0 < clock) :
    MaterialSource.Realization (physical B) P (owner B) (project B) B.content
      B.incoming B.outgoing clock := by
  refine ⟨⟨hc, rfl, refines B⟩, ?_, ?_⟩
  · intro r; exact r.property.1
  · intro t r e r' hp
    have h := delivery B t r r' e hp
    exact ⟨h.1, h.2.1, h.2.2.1⟩

end CGB.BufferedMaterial
