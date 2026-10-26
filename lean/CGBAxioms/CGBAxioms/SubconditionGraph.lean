import CGBAxioms.ConstraintAudit
import CGBAxioms.Models

set_option linter.style.header false

/-!
# Structural Graph of the 14 Atomic Sub-conditions

This file establishes the formal dependency and redundancy relations among
the 14 atomic sub-conditions (A01–A14) from the 6 compound CGB axioms.

Key results formally proven:
1. Under `[Finite Λ]`, A01 (`finiteSublevels`) and A02 (`boundedContent`) are
   consequences of this additional finite-state premise, not globally redundant.
2. A03 (`uniqueVacuum`) is strictly independent of `[Finite Λ]`.
3. P1b's read domain decouples into rate reading (A07) and effect reading (A08):
   - `farFlip` isolates and violates A07 while satisfying A08, A06, A05, A04.
   - `farAct` isolates and violates A08 while satisfying A07, A06, A05, A04.
4. P3's boundary sensing decouples into boundary rate (A12) and boundary effect (A13):
   - `innerClock` isolates and violates A12 while satisfying A13 and A14.
   - `innerActClock` isolates and violates A13 while satisfying A12 and A14.
   - `pairPreset` isolates and violates A14 while satisfying A12 and A13.
5. Conditional collapse lemmas (complete graph, universal participants, all-boundary).
6. A nine-field ReducedCore candidate; recovery of Atomized additionally needs
   finite local states, homogeneous contact AND rigid structural data.

The witnesses below certify the displayed subsets of conditions only. They do not
constitute a full 14-condition irredundancy or physical-necessity certificate.
-/

open Finset

namespace CGB.Medium

variable {U Λ A I : Type*} (M : Medium U Λ A I)

/-! ## 1. Named predicates for the atomic sub-conditions -/

def ParticipantsContact (M : Medium U Λ A I) : Prop :=
  ∀ a, ∀ u ∈ M.parts a, ∀ v ∈ M.parts a, M.near u v

def WritesParticipants (M : Medium U Λ A I) : Prop :=
  ∀ a s x u, u ∉ M.parts a → M.act a s x u = x u

def RateReadsParticipants (M : Medium U Λ A I) : Prop :=
  ∀ a s s' x y, (∀ u ∈ M.parts a, x u = y u ∧ s u = s' u) → M.rate a s x = M.rate a s' y

def EffectReadsParticipants (M : Medium U Λ A I) : Prop :=
  ∀ a s s' x y, (∀ u ∈ M.parts a, x u = y u ∧ s u = s' u) →
    ∀ u ∈ M.parts a, M.act a s x u = M.act a s' y u

def BoundaryRate (M : Medium U Λ A I) : Prop :=
  ∀ a s s' x, (∀ u ∈ M.boundary, s u = s' u) → M.rate a s x = M.rate a s' x

def BoundaryEffect (M : Medium U Λ A I) : Prop :=
  ∀ a s s' x, (∀ u ∈ M.boundary, s u = s' u) → M.act a s x = M.act a s' x

def InitialInvariant (M : Medium U Λ A I) : Prop :=
  ∀ σ, M.IsAut σ → M.init ∘ σ.symm = M.init

theorem p1b_iff_atoms :
    M.P1b ↔ M.ParticipantsContact ∧ M.WritesParticipants ∧
      M.RateReadsParticipants ∧ M.EffectReadsParticipants := by
  constructor
  · intro h
    exact ⟨h.1, h.2.1,
      fun a s s' x y hh => (h.2.2 a s s' x y hh).1,
      fun a s s' x y hh => (h.2.2 a s s' x y hh).2⟩
  · rintro ⟨h2, h3, h4, h5⟩
    exact ⟨h2, h3, fun a s s' x y hh => ⟨h4 a s s' x y hh, h5 a s s' x y hh⟩⟩

theorem onlyBoundarySenses_iff_atoms :
    M.OnlyBoundarySenses ↔ M.BoundaryRate ∧ M.BoundaryEffect := by
  constructor
  · intro h
    exact ⟨fun a s s' x hh => (h a s s' x hh).1, fun a s s' x hh => (h a s s' x hh).2⟩
  · rintro ⟨h1, h2⟩
    exact fun a s s' x hh => ⟨h1 a s s' x hh, h2 a s s' x hh⟩

theorem p3_iff_atoms :
    M.P3 ↔ M.BoundaryRate ∧ M.BoundaryEffect ∧ M.InitialInvariant := by
  rw [P3, onlyBoundarySenses_iff_atoms]
  tauto

/-! ## 2. Redundancy and Independence in Finite-State Specialization -/

/-- In any finite local state space, sublevel sets are automatically finite (A01 is redundant). -/
theorem a01_of_finite [Finite Λ] (content : Λ → ℕ) :
    ∀ n, {l : Λ | content l ≤ n}.Finite := by
  intro n
  exact Set.toFinite _

/-- In any finite local state space, content is automatically bounded (A02 is redundant). -/
theorem a02_of_finite [Finite Λ] (content : Λ → ℕ) :
    ∃ K, ∀ l, content l ≤ K := by
  cases nonempty_fintype Λ
  by_cases h : Nonempty Λ
  · have : (Finset.univ : Finset Λ).Nonempty := Finset.univ_nonempty_iff.mpr h
    obtain ⟨lmax, -, hmax⟩ := Finset.exists_max_image Finset.univ content this
    exact ⟨content lmax, fun l => hmax l (Finset.mem_univ l)⟩
  · exact ⟨0, fun l => (h ⟨l⟩).elim⟩

/-- Under `[Finite Λ]`, A01 and A02 hold unconditionally for ANY content function. -/
theorem a01_and_a02_redundant_of_finite [Finite Λ] (content : Λ → ℕ) :
    (∀ n, {l : Λ | content l ≤ n}.Finite) ∧ (∃ K, ∀ l, content l ≤ K) :=
  ⟨a01_of_finite content, a02_of_finite content⟩

/-- A03 (uniqueVacuum) is strictly independent of finiteness: on `Fin 2`, we can have
zero vacuums, multiple vacuums, or a unique vacuum. -/
theorem uniqueVacuum_independent_of_finite :
    -- Zero vacuums on Fin 2
    (¬ ∃! l : Fin 2, (fun _ : Fin 2 => (1 : ℕ)) l = 0) ∧
    -- Multiple vacuums on Fin 2
    (¬ ∃! l : Fin 2, (fun _ : Fin 2 => (0 : ℕ)) l = 0) ∧
    -- Unique vacuum on Fin 2
    (∃! l : Fin 2, (fun l : Fin 2 => if l = 0 then (0 : ℕ) else 1) l = 0) := by
  refine ⟨?_, ?_, ?_⟩
  · rintro ⟨l, hl, -⟩
    revert l hl
    decide
  · rintro ⟨l, -, huniq⟩
    have e0 := huniq 0 rfl
    have e1 := huniq 1 rfl
    have e : (1 : Fin 2) = 0 := e1.trans e0.symm
    revert e
    decide
  · refine ⟨0, by decide, fun l hl => ?_⟩
    revert l hl
    decide

/-! ## 3. Decoupling P1b: Rate vs Effect Reading -/

open Models

/-- `farFlip` violates rateReadsParticipants (A07). -/
theorem farFlip_not_rateReads : ¬ farFlip.RateReadsParticipants := by
  intro h
  have hh : ∀ u ∈ farFlip.parts 0,
      (fun _ : Fin 2 => (1 : Fin 3)) u = (Function.update (fun _ => (1 : Fin 3)) 1 2) u ∧
      (fun _ : Fin 2 => ()) u = (fun _ => ()) u := by
    intro u hu
    simp only [farFlip, Finset.mem_singleton] at hu
    subst hu
    decide
  have eq := h 0 (fun _ => ()) (fun _ => ()) (fun _ => 1)
    (Function.update (fun _ => 1) 1 2) hh
  simp [farFlip] at eq

/-- `farFlip` SATISFIES effectReadsParticipants (A08). -/
theorem farFlip_satisfies_effectReads : farFlip.EffectReadsParticipants := by
  intro a s s' x y hparts u hu
  simp only [farFlip, Finset.mem_singleton] at hu
  have hx : x a = y a := (hparts a (by simp [farFlip])).1
  subst hu
  simp [farFlip, hx]

/-- `farFlip` satisfies writesParticipants (A06). -/
theorem farFlip_satisfies_writes : farFlip.WritesParticipants := by
  intro a s x u hu
  simp only [farFlip, Finset.mem_singleton] at hu
  simp [farFlip, Function.update_of_ne hu]

/-- `farFlip` satisfies participantsContact (A05). -/
theorem farFlip_satisfies_partsContact : farFlip.ParticipantsContact := by
  intro a u hu v hv
  simp only [farFlip, Finset.mem_singleton] at hu hv
  subst hu; subst hv
  simp [farFlip]

/-- `farFlip` satisfies homogeneousContact (A04). -/
theorem farFlip_satisfies_homoContact : farFlip.HomogeneousContact :=
  eq_homogeneous

/-- Decoupling summary for `farFlip`: it isolates and violates A07 while
satisfying A08, A06, A05, A04. -/
theorem farFlip_isolates_a07 :
    ¬ farFlip.RateReadsParticipants ∧
      farFlip.EffectReadsParticipants ∧
      farFlip.WritesParticipants ∧
      farFlip.ParticipantsContact ∧
      farFlip.HomogeneousContact :=
  ⟨farFlip_not_rateReads, farFlip_satisfies_effectReads,
   farFlip_satisfies_writes, farFlip_satisfies_partsContact,
   farFlip_satisfies_homoContact⟩

/-- Dual witness `farAct`: rate reads only participants (constant 1), but effect reads outside. -/
noncomputable def farAct : Medium (Fin 2) (Fin 3) (Fin 2) Unit where
  near u v := u = v
  ident _ := 0
  boundary := ∅
  content := flipCost
  parts u := {u}
  act u _ x := Function.update x u (if x (u + 1) = 2 then flip12 (x u) else x u)
  rate _ _ _ := 1
  rate_nonneg _ _ _ := by norm_num
  init _ := 1
  calm _ := ()

theorem farAct_not_effectReads : ¬ farAct.EffectReadsParticipants := by
  intro h
  have hh : ∀ u ∈ farAct.parts 0,
      (fun _ : Fin 2 => (1 : Fin 3)) u = (Function.update (fun _ => (1 : Fin 3)) 1 2) u ∧
      (fun _ : Fin 2 => ()) u = (fun _ => ()) u := by
    intro u hu
    simp only [farAct, Finset.mem_singleton] at hu
    subst hu
    decide
  have eq := h 0 (fun _ => ()) (fun _ => ()) (fun _ => 1)
    (Function.update (fun _ => 1) 1 2) hh 0 (by simp [farAct])
  simp [farAct] at eq
  revert eq
  decide

theorem farAct_satisfies_rateReads : farAct.RateReadsParticipants := by
  intro a s s' x y _
  rfl

theorem farAct_satisfies_writes : farAct.WritesParticipants := by
  intro a s x u hu
  simp only [farAct, Finset.mem_singleton] at hu
  simp [farAct, Function.update_of_ne hu]

theorem farAct_satisfies_partsContact : farAct.ParticipantsContact := by
  intro a u hu v hv
  simp only [farAct, Finset.mem_singleton] at hu hv
  subst hu; subst hv
  simp [farAct]

theorem farAct_satisfies_homoContact : farAct.HomogeneousContact :=
  eq_homogeneous

/-- Decoupling summary for `farAct`: it isolates and violates A08 while
satisfying A07, A06, A05, A04. -/
theorem farAct_isolates_a08 :
    ¬ farAct.EffectReadsParticipants ∧
      farAct.RateReadsParticipants ∧
      farAct.WritesParticipants ∧
      farAct.ParticipantsContact ∧
      farAct.HomogeneousContact :=
  ⟨farAct_not_effectReads, farAct_satisfies_rateReads,
   farAct_satisfies_writes, farAct_satisfies_partsContact,
   farAct_satisfies_homoContact⟩

/-! ## 4. Decoupling P3: Boundary Rate vs Boundary Effect vs Initial Invariance -/

/-- `innerClock` violates boundaryRate (A12). -/
theorem innerClock_not_boundaryRate : ¬ innerClock.BoundaryRate := by
  intro h
  have hh : ∀ u ∈ innerClock.boundary, (fun _ : Fin 2 => false) u = (fun _ => true) u := by
    intro u hu
    simp [innerClock] at hu
  have eq := h 0 (fun _ => false) (fun _ => true) (fun _ => 1) hh
  simp [innerClock] at eq

/-- `innerClock` SATISFIES boundaryEffect (A13). -/
theorem innerClock_satisfies_boundaryEffect : innerClock.BoundaryEffect := by
  intro a s s' x _
  rfl

/-- `innerClock` SATISFIES initialInvariant (A14). -/
theorem innerClock_satisfies_initialInvariant : innerClock.InitialInvariant := by
  intro σ _
  funext u
  rfl

/-- Decoupling summary for `innerClock`: it isolates and violates A12 while
satisfying A13 and A14. -/
theorem innerClock_isolates_a12 :
    ¬ innerClock.BoundaryRate ∧
      innerClock.BoundaryEffect ∧
      innerClock.InitialInvariant :=
  ⟨innerClock_not_boundaryRate, innerClock_satisfies_boundaryEffect,
   innerClock_satisfies_initialInvariant⟩

/-- Dual witness `innerActClock`: rate ignores stimulus (constant 1),
but effect reads internal stimulus. -/
noncomputable def innerActClock : Medium (Fin 2) (Fin 3) (Fin 2) Bool where
  near u v := u = v
  ident _ := 0
  boundary := ∅
  content := flipCost
  parts u := {u}
  act u s x := Function.update x u (if s u then flip12 (x u) else x u)
  rate _ _ _ := 1
  rate_nonneg _ _ _ := by norm_num
  init _ := 1
  calm _ := false

theorem innerActClock_not_boundaryEffect : ¬ innerActClock.BoundaryEffect := by
  intro h
  have hh : ∀ u ∈ innerActClock.boundary, (fun _ : Fin 2 => false) u = (fun _ => true) u := by
    intro u hu
    simp [innerActClock] at hu
  have eq := congrFun (h 0 (fun _ => false) (fun _ => true) (fun _ => 1) hh) 0
  simp [innerActClock] at eq
  revert eq
  decide

theorem innerActClock_satisfies_boundaryRate : innerActClock.BoundaryRate := by
  intro a s s' x _
  rfl

theorem innerActClock_satisfies_initialInvariant : innerActClock.InitialInvariant := by
  intro σ _
  funext u
  rfl

/-- Decoupling summary for `innerActClock`: it isolates and violates A13 while
satisfying A12 and A14. -/
theorem innerActClock_isolates_a13 :
    ¬ innerActClock.BoundaryEffect ∧
      innerActClock.BoundaryRate ∧
      innerActClock.InitialInvariant :=
  ⟨innerActClock_not_boundaryEffect, innerActClock_satisfies_boundaryRate,
   innerActClock_satisfies_initialInvariant⟩

/-- `pairPreset` isolates A14: satisfies boundaryRate (A12) and boundaryEffect (A13),
but violates A14. -/
theorem pairPreset_isolates_a14 :
    ¬ pairPreset.InitialInvariant ∧
      pairPreset.BoundaryRate ∧
      pairPreset.BoundaryEffect := by
  have haut : pairPreset.IsAut (Equiv.swap 0 1) := xhop_isAut fun _ => rfl
  refine ⟨fun h => ?_, ?_, ?_⟩
  · have eq := congrFun (h (Equiv.swap 0 1) haut) 0
    revert eq
    decide
  · intro a s s' x hh
    have hboth := xhop_senses a s s' x hh
    exact hboth.1
  · intro a s s' x hh
    have hboth := xhop_senses a s s' x hh
    exact hboth.2

/-! ## 5. Regime Collapse Lemmas (Vacuous Satisfaction) -/

/-- Under a complete graph (`near u v ↔ True`), homogeneousContact (A04) and
participantsContact (A05) are vacuously satisfied. -/
theorem complete_graph_satisfies_a04_a05
    (hnear : ∀ u v : U, M.near u v) :
    M.HomogeneousContact ∧ M.ParticipantsContact := by
  classical
  refine ⟨fun u v => ⟨Equiv.swap u v, fun a b => by simp [hnear], Equiv.swap_apply_left u v⟩,
          fun a u hu v hv => hnear u v⟩

/-- Under universal participants (`∀ a, parts a = univ`), writesParticipants (A06),
rateReadsParticipants (A07), and effectReadsParticipants (A08) are vacuously satisfied. -/
theorem universal_parts_satisfies_a06_a07_a08 [Fintype U]
    (hparts : ∀ a, M.parts a = univ) :
    M.WritesParticipants ∧ M.RateReadsParticipants ∧ M.EffectReadsParticipants := by
  refine ⟨fun a s x u hu => ?_, fun a s s' x y hh => ?_, fun a s s' x y hh u hu => ?_⟩
  · exfalso; apply hu; simp [hparts]
  · have hx : x = y := funext fun u => (hh u (by simp [hparts])).1
    have hs : s = s' := funext fun u => (hh u (by simp [hparts])).2
    subst hx; subst hs; rfl
  · have hx : x = y := funext fun w => (hh w (by simp [hparts])).1
    have hs : s = s' := funext fun w => (hh w (by simp [hparts])).2
    subst hx; subst hs; rfl

/-- When boundary = univ, boundaryRate (A12) and boundaryEffect (A13) are vacuously satisfied. -/
theorem all_boundary_satisfies_a12_a13 [Fintype U]
    (hb : M.boundary = univ) :
    M.BoundaryRate ∧ M.BoundaryEffect := by
  have hsenses := M.onlyBoundarySenses_of_all_boundary hb
  rw [onlyBoundarySenses_iff_atoms] at hsenses
  exact hsenses

/-! ## 6. Non-homogeneous Graphs and A04 Incompatibility -/

/-- A simple 3-node path graph: 0 - 1 - 2. Node 1 has degree 2, nodes 0 and 2 have degree 1. -/
def path3Near (u v : Fin 3) : Prop :=
  (u = 0 ∧ v = 1) ∨ (u = 1 ∧ v = 0) ∨
  (u = 1 ∧ v = 2) ∨ (u = 2 ∧ v = 1) ∨ (u = v)

instance (u v : Fin 3) : Decidable (path3Near u v) := by
  unfold path3Near
  infer_instance

/-- This specific three-site contact relation is not vertex-transitive.
No identification of contact with a realized biological wiring graph is assumed. -/
theorem path3_not_homogeneous :
    ¬ (∀ u v : Fin 3, ∃ σ : Equiv.Perm (Fin 3),
        (∀ a b, path3Near (σ a) (σ b) ↔ path3Near a b) ∧ σ u = v) := by
  intro h
  obtain ⟨σ, hσ, h01⟩ := h 0 1
  have hnot02 : ¬ path3Near 0 2 := by decide
  have hnot1s2 : ¬ path3Near 1 (σ 2) := by
    have hiff := hσ 0 2
    rw [h01] at hiff
    exact fun hh => hnot02 (hiff.mp hh)
  generalize h2 : σ 2 = v
  fin_cases v
  · dsimp at h2
    apply hnot1s2
    rw [h2]
    decide
  · dsimp at h2
    have heq : σ 2 = σ 0 := by rw [h2, h01]
    have hcontra : (2 : Fin 3) = 0 := σ.injective heq
    revert hcontra
    decide
  · dsimp at h2
    apply hnot1s2
    rw [h2]
    decide

/-! ## 7. A Reduced Core Candidate -/

/-- A selected nine-field subset, with no irredundancy or physical-necessity claim:
1. Unique vacuum (A03)
2. Locality (A05, A06, A07, A08)
3. Thermodynamics (A09, A10)
4. Boundary interface (A12, A13)
-/
structure ReducedCore [Fintype U] [Fintype A] [DecidableEq (U → Λ)] : Prop where
  uniqueVacuum : ∃! l, M.content l = 0
  participantsContact : M.ParticipantsContact
  writesParticipants : M.WritesParticipants
  rateReadsParticipants : M.RateReadsParticipants
  effectReadsParticipants : M.EffectReadsParticipants
  closedConservation : M.P2a
  calmBalance : M.P2b
  boundaryRate : M.BoundaryRate
  boundaryEffect : M.BoundaryEffect

theorem atomized_implies_reducedCore [Fintype U] [Fintype A] [DecidableEq (U → Λ)] :
    M.Atomized → M.ReducedCore := by
  intro h
  exact ⟨h.uniqueVacuum, h.participantsContact, h.writesParticipants,
         h.rateReadsParticipants, h.effectReadsParticipants,
         h.closedConservation, h.calmBalance, h.boundaryRate, h.boundaryEffect⟩

theorem reducedCore_implies_atomized_of_regimes [Fintype U] [Fintype A] [DecidableEq (U → Λ)]
    [Finite Λ] (hrigid : M.Rigid) :
    M.ReducedCore → M.Atomized := by
  intro h
  obtain ⟨h01, h02⟩ := a01_and_a02_redundant_of_finite M.content
  exact ⟨h01, h02, h.uniqueVacuum, h.participantsContact, h.writesParticipants,
         h.rateReadsParticipants, h.effectReadsParticipants,
         h.closedConservation, h.calmBalance,
         p2c_of_rigid M hrigid,
         h.boundaryRate, h.boundaryEffect,
         initial_invariant_of_rigid M hrigid⟩

/-- Exact recovery under finite local states and rigid structural data (contact
homogeneity is no longer part of the axioms since 2026-10-26). -/
theorem reducedCore_iff_atomized_of_regimes [Fintype U] [Fintype A]
    [DecidableEq (U → Λ)] [Finite Λ] (hrigid : M.Rigid) :
    M.ReducedCore ↔ M.Atomized :=
  ⟨M.reducedCore_implies_atomized_of_regimes hrigid,
    M.atomized_implies_reducedCore⟩

/-- A finite rigid contact carrier with no events. This is a logical countermodel,
not a biological model or a capability experiment. -/
def path3Rigid : Medium (Fin 3) (Fin 2) Empty Unit where
  near := path3Near
  ident u := u.val
  boundary := ∅
  content l := l.val
  parts a := a.elim
  act a := a.elim
  rate a := a.elim
  rate_nonneg a := a.elim
  init _ := 0
  calm _ := ()

theorem path3Rigid_reducedCore : path3Rigid.ReducedCore := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨0, rfl, fun l hl => ?_⟩
    exact Fin.ext hl
  · intro a; exact a.elim
  · intro a; exact a.elim
  · intro a; exact a.elim
  · intro a; exact a.elim
  · intro a; exact a.elim
  · refine ⟨fun _ => 1, fun _ => by norm_num, ?_⟩
    intro x y
    simp [jump]
  · intro a; exact a.elim
  · intro a; exact a.elim

/-- Since contact homogeneity left P1b (2026-10-26), the rigid three-unit chain, whose
contact is not homogeneous (its ends have one contact, its middle two), is a model;
before, it was the countermodel showing that finiteness and rigidity do not recover A04. -/
theorem path3Rigid_atomized_not_homogeneous :
    path3Rigid.Atomized ∧ ¬ path3Rigid.HomogeneousContact := by
  refine ⟨?_, path3_not_homogeneous⟩
  exact (path3Rigid.reducedCore_implies_atomized_of_regimes
    (rigid_of_injective_identity path3Rigid (fun _ _ h => Fin.ext h))) path3Rigid_reducedCore

end CGB.Medium
