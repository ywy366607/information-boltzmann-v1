import CGBAxioms.GenerationLattice

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

/-!
# Voxel chemistry: every local reaction table is admitted (generation chain, M5)

The law layer of a developing body, proved once. Units are voxels of a world with a symmetric
neighbour relation. A voxel holds a bounded count of each species. Species carry no role: only a
positive content weight. A rule is a stoichiometry: single-voxel reactions and two-voxel reactions
take `before` out of each side and put `after` in, and do nothing unless the counts allow it.
A species listed on both sides of a reaction is a catalyst, so a charge that hops only where
a cell's neurite segment is present on both sides is an ordinary rule, and so is a growth step.

`VoxelChem.axioms`: the six clauses hold for the whole family as soon as

* every species has positive weight;
* interior rules conserve weight;
* each rule has a reverse with swapped stoichiometry and equal prefactor;
* the calm potential is zero.

The energy `φ`, the rule tables and the prefactors stay free, so a genome is a table and nothing
here says which table serves language. As `ConstraintFreedom.arbitrary_driven_tables_admitted`
already warns, admission alone says nothing about capability.
-/

open Finset
open scoped Classical

namespace CGB

/-- One side of a stoichiometry. -/
structure VoxelRxn (S : Type*) where
  before : S → ℕ
  after : S → ℕ

namespace VoxelRxn

variable {S : Type*}

/-- The reverse stoichiometry. -/
def swap (c : VoxelRxn S) : VoxelRxn S := ⟨c.after, c.before⟩

@[simp]
theorem swap_swap (c : VoxelRxn S) : c.swap.swap = c := rfl

end VoxelRxn

/-- Bounded species counts in one voxel. -/
abbrev VoxelState (S : Type*) (cap : ℕ) := {l : S → ℕ // ∀ s, l s ≤ cap}

namespace VoxelState

variable {S : Type*} {cap : ℕ}

/-- A reaction can run when its inputs are present and its products fit. -/
def Fires (c : VoxelRxn S) (l : VoxelState S cap) : Prop :=
  (∀ s, c.before s ≤ l.1 s) ∧ ∀ s, l.1 s - c.before s + c.after s ≤ cap

/-- Run a reaction, or do nothing when it cannot run. -/
noncomputable def fire (c : VoxelRxn S) (l : VoxelState S cap) : VoxelState S cap :=
  if h : Fires c l then ⟨fun s => l.1 s - c.before s + c.after s, h.2⟩ else l

theorem fire_of_fires {c : VoxelRxn S} {l : VoxelState S cap} (h : Fires c l) (s : S) :
    (fire c l).1 s = l.1 s - c.before s + c.after s := by
  simp [fire, h]

theorem fire_of_not {c : VoxelRxn S} {l : VoxelState S cap} (h : ¬ Fires c l) : fire c l = l := by
  simp [fire, h]

/-- A reaction that ran can run backwards. -/
theorem fires_swap {c : VoxelRxn S} {l : VoxelState S cap} (h : Fires c l) :
    Fires c.swap (fire c l) := by
  refine ⟨fun s => ?_, fun s => ?_⟩
  · simp only [VoxelRxn.swap]
    rw [fire_of_fires h]
    omega
  · simp only [VoxelRxn.swap]
    rw [fire_of_fires h]
    have h1 := h.1 s
    have h2 := l.2 s
    omega

/-- Running backwards restores the voxel. -/
theorem fire_swap_fire {c : VoxelRxn S} {l : VoxelState S cap} (h : Fires c l) :
    fire c.swap (fire c l) = l := by
  apply Subtype.ext
  funext s
  rw [fire_of_fires (fires_swap h)]
  simp only [VoxelRxn.swap]
  rw [fire_of_fires h]
  have h1 := h.1 s
  omega

instance [Finite S] : Finite (VoxelState S cap) := by
  classical
  let f : VoxelState S cap → S → Fin (cap + 1) :=
    fun l s => ⟨l.1 s, Nat.lt_succ_of_le (l.2 s)⟩
  refine Finite.of_injective f fun l m h => ?_
  apply Subtype.ext
  funext s
  have := congrFun h s
  simpa [f] using this

/-- Total weight in a voxel. -/
def weight [Fintype S] (w : S → ℕ) (l : VoxelState S cap) : ℕ := ∑ s, w s * l.1 s

/-- The weight of a stoichiometric vector. -/
def rxnWeight [Fintype S] (w : S → ℕ) (n : S → ℕ) : ℕ := ∑ s, w s * n s

theorem weight_fire [Fintype S] (w : S → ℕ) {c : VoxelRxn S} {l : VoxelState S cap}
    (h : Fires c l) :
    weight w (fire c l) + rxnWeight w c.before = weight w l + rxnWeight w c.after := by
  unfold weight rxnWeight
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [fire_of_fires h, ← Nat.mul_add, ← Nat.mul_add]
  congr 1
  have h1 := h.1 s
  omega

end VoxelState

/-- A two-voxel stoichiometry. -/
structure VoxelRxn₂ (S : Type*) where
  left : VoxelRxn S
  right : VoxelRxn S

namespace VoxelRxn₂

variable {S : Type*} {cap : ℕ}

def swap (c : VoxelRxn₂ S) : VoxelRxn₂ S := ⟨c.left.swap, c.right.swap⟩

/-- A two-voxel reaction runs only when both sides can. -/
def Fires (c : VoxelRxn₂ S) (p : VoxelState S cap × VoxelState S cap) : Prop :=
  VoxelState.Fires c.left p.1 ∧ VoxelState.Fires c.right p.2

noncomputable def fire (c : VoxelRxn₂ S) (p : VoxelState S cap × VoxelState S cap) :
    VoxelState S cap × VoxelState S cap :=
  if Fires c p then (VoxelState.fire c.left p.1, VoxelState.fire c.right p.2) else p

theorem fire_of_fires {c : VoxelRxn₂ S} {p : VoxelState S cap × VoxelState S cap}
    (h : Fires c p) : fire c p = (VoxelState.fire c.left p.1, VoxelState.fire c.right p.2) := by
  simp [fire, h]

theorem fire_swap_fire {c : VoxelRxn₂ S} {p : VoxelState S cap × VoxelState S cap}
    (h : Fires c p) : fire c.swap (fire c p) = p := by
  have hf := fire_of_fires h
  have hs : Fires c.swap (fire c p) := by
    rw [hf]
    exact ⟨VoxelState.fires_swap h.1, VoxelState.fires_swap h.2⟩
  have hl := VoxelState.fire_swap_fire h.1
  have hr := VoxelState.fire_swap_fire h.2
  calc
    fire c.swap (fire c p)
        = (VoxelState.fire c.left.swap (fire c p).1, VoxelState.fire c.right.swap (fire c p).2) := by
          exact fire_of_fires hs
    _ = p := by
          rw [hf]
          exact Prod.ext hl hr

end VoxelRxn₂

/-- A chemistry on voxels. The tables `rxn₁`, `rxn₂` depend on the identity pages, so a genome
can differ by place. -/
structure VoxelChem (U S I R₁ R₂ : Type*) (cap : ℕ) where
  adj : U → U → Prop
  adj_symm : ∀ u v, adj u v → adj v u
  adj_irrefl : ∀ u, ¬ adj u u
  ident : U → ℕ
  bid : Finset ℕ
  /-- content weight of a species; the only thing a species is given -/
  w : S → ℕ
  φ : ℕ → VoxelState S cap → ℝ
  μ : I → ℝ
  rxn₁ : R₁ → ℕ → VoxelRxn S
  rxn₂ : R₂ → ℕ → ℕ → VoxelRxn₂ S
  κ₁ : R₁ → ℕ → ℝ
  κ₂ : R₂ → ℕ → ℕ → ℝ
  κ₁_nonneg : ∀ r i, 0 ≤ κ₁ r i
  κ₂_nonneg : ∀ r i j, 0 ≤ κ₂ r i j
  /-- symmetric state dependence of a prefactor, such as mass action -/
  ξ₁ : VoxelState S cap → VoxelState S cap → ℝ
  ξ₂ : VoxelState S cap × VoxelState S cap → VoxelState S cap × VoxelState S cap → ℝ
  ξ₁_nonneg : ∀ l m, 0 ≤ ξ₁ l m
  ξ₂_nonneg : ∀ p q, 0 ≤ ξ₂ p q
  ξ₁_symm : ∀ l m, ξ₁ l m = ξ₁ m l
  ξ₂_symm : ∀ p q, ξ₂ p q = ξ₂ q p
  rev₁ : R₁ → R₁
  rev₂ : R₂ → R₂
  birth : ℕ → VoxelState S cap
  calm : I

namespace VoxelChem

variable {U S I R₁ R₂ : Type*} {cap : ℕ} [Fintype S] (C : VoxelChem U S I R₁ R₂ cap)

/-- The chemistry as a lattice medium. -/
noncomputable def lattice : LatticeCRN U (VoxelState S cap) I R₁ R₂ where
  adj := C.adj
  adj_symm := C.adj_symm
  adj_irrefl := C.adj_irrefl
  ident := C.ident
  bid := C.bid
  content := VoxelState.weight C.w
  φ := C.φ
  μ := C.μ
  eff1 r i l := VoxelState.fire (C.rxn₁ r i) l
  eff2 r i j p := (C.rxn₂ r i j).fire p
  c1 r i l := C.κ₁ r i * C.ξ₁ l (VoxelState.fire (C.rxn₁ r i) l)
  c2 r i j p := C.κ₂ r i j * C.ξ₂ p ((C.rxn₂ r i j).fire p)
  c1_nonneg r i l := mul_nonneg (C.κ₁_nonneg r i) (C.ξ₁_nonneg _ _)
  c2_nonneg r i j p := mul_nonneg (C.κ₂_nonneg r i j) (C.ξ₂_nonneg _ _)
  rev1 := C.rev₁
  rev2 := C.rev₂
  birth := C.birth
  calm := C.calm

/-- The empty voxel. -/
def vacuum : VoxelState S cap := ⟨fun _ => 0, fun _ => Nat.zero_le _⟩

theorem weight_eq_zero (hw : ∀ s, 0 < C.w s) {l : VoxelState S cap}
    (h : VoxelState.weight C.w l = 0) : l = vacuum := by
  apply Subtype.ext
  funext s
  have hs := (Finset.sum_eq_zero_iff.mp h) s (Finset.mem_univ s)
  simpa [vacuum, (hw s).ne'] using hs

theorem p1a_data (hw : ∀ s, 0 < C.w s) :
    (∀ n, {l : VoxelState S cap | VoxelState.weight C.w l ≤ n}.Finite) ∧
      (∃ K, ∀ l : VoxelState S cap, VoxelState.weight C.w l ≤ K) ∧
      ∃! l : VoxelState S cap, VoxelState.weight C.w l = 0 := by
  refine ⟨fun n => Set.toFinite _, ?_, vacuum, ?_, fun l hl => C.weight_eq_zero hw hl⟩
  · refine ⟨∑ s, C.w s * cap, fun l => ?_⟩
    exact Finset.sum_le_sum fun s _ => Nat.mul_le_mul_left _ (l.2 s)
  · simp [VoxelState.weight, vacuum]

theorem fires_of_fire_ne₁ {c : VoxelRxn S} {l : VoxelState S cap}
    (h : VoxelState.fire c l ≠ l) : VoxelState.Fires c l := by
  by_contra hn
  exact h (VoxelState.fire_of_not hn)

theorem fires_of_fire_ne₂ {c : VoxelRxn₂ S} {p : VoxelState S cap × VoxelState S cap}
    (h : c.fire p ≠ p) : c.Fires p := by
  by_contra hn
  exact h (by simp [VoxelRxn₂.fire, hn])

/-- **M5**: every voxel chemistry that conserves weight in the interior and has reverses is
admitted. -/
theorem axioms [Fintype U] [DecidableEq U] [Fintype R₁] [DecidableEq R₁] [Fintype R₂]
    [DecidableEq R₂] [DecidableEq (U → VoxelState S cap)]
    (hw : ∀ s, 0 < C.w s)
    (hcons₁ : ∀ r i, i ∉ C.bid →
      VoxelState.rxnWeight C.w (C.rxn₁ r i).before = VoxelState.rxnWeight C.w (C.rxn₁ r i).after)
    (hcons₂ : ∀ r i j, i ∉ C.bid → j ∉ C.bid →
      VoxelState.rxnWeight C.w (C.rxn₂ r i j).left.before +
          VoxelState.rxnWeight C.w (C.rxn₂ r i j).right.before =
        VoxelState.rxnWeight C.w (C.rxn₂ r i j).left.after +
          VoxelState.rxnWeight C.w (C.rxn₂ r i j).right.after)
    (hcalm : C.μ C.calm = 0)
    (hinv₁ : Function.Involutive C.rev₁) (hinv₂ : Function.Involutive C.rev₂)
    (hrev₁ : ∀ r i, C.rxn₁ (C.rev₁ r) i = (C.rxn₁ r i).swap)
    (hrev₂ : ∀ r i j, C.rxn₂ (C.rev₂ r) i j = (C.rxn₂ r i j).swap)
    (hκ₁ : ∀ r i, C.κ₁ (C.rev₁ r) i = C.κ₁ r i)
    (hκ₂ : ∀ r i j, C.κ₂ (C.rev₂ r) i j = C.κ₂ r i j) :
    C.lattice.medium.Axioms := by
  refine LatticeCRN.axioms C.lattice (C.p1a_data hw) ?_ ?_ hcalm hinv₁ hinv₂ ?_ ?_ ?_ ?_
  · intro r i l hi
    change VoxelState.weight C.w (VoxelState.fire (C.rxn₁ r i) l) = VoxelState.weight C.w l
    by_cases h : VoxelState.Fires (C.rxn₁ r i) l
    · have := VoxelState.weight_fire C.w h
      rw [hcons₁ r i hi] at this
      omega
    · rw [VoxelState.fire_of_not h]
  · intro r i j p hi hj
    change VoxelState.weight C.w ((C.rxn₂ r i j).fire p).1 +
        VoxelState.weight C.w ((C.rxn₂ r i j).fire p).2 =
      VoxelState.weight C.w p.1 + VoxelState.weight C.w p.2
    by_cases h : (C.rxn₂ r i j).Fires p
    · have hl := VoxelState.weight_fire C.w h.1
      have hr := VoxelState.weight_fire C.w h.2
      have hc := hcons₂ r i j hi hj
      have hf : (C.rxn₂ r i j).fire p =
          (VoxelState.fire (C.rxn₂ r i j).left p.1, VoxelState.fire (C.rxn₂ r i j).right p.2) := by
        simp [VoxelRxn₂.fire, h]
      rw [hf]
      simp only
      omega
    · have hf : (C.rxn₂ r i j).fire p = p := by simp [VoxelRxn₂.fire, h]
      rw [hf]
  · intro r i l hne
    change VoxelState.fire (C.rxn₁ (C.rev₁ r) i) (VoxelState.fire (C.rxn₁ r i) l) = l
    rw [hrev₁]
    exact VoxelState.fire_swap_fire (fires_of_fire_ne₁ hne)
  · intro r i j p hne
    change (C.rxn₂ (C.rev₂ r) i j).fire ((C.rxn₂ r i j).fire p) = p
    rw [hrev₂]
    exact VoxelRxn₂.fire_swap_fire (fires_of_fire_ne₂ hne)
  · intro r i l hne
    change C.κ₁ (C.rev₁ r) i *
        C.ξ₁ (VoxelState.fire (C.rxn₁ r i) l)
          (VoxelState.fire (C.rxn₁ (C.rev₁ r) i) (VoxelState.fire (C.rxn₁ r i) l)) =
      C.κ₁ r i * C.ξ₁ l (VoxelState.fire (C.rxn₁ r i) l)
    rw [hrev₁, VoxelState.fire_swap_fire (fires_of_fire_ne₁ hne), hκ₁, C.ξ₁_symm]
  · intro r i j p hne
    change C.κ₂ (C.rev₂ r) i j *
        C.ξ₂ ((C.rxn₂ r i j).fire p) ((C.rxn₂ (C.rev₂ r) i j).fire ((C.rxn₂ r i j).fire p)) =
      C.κ₂ r i j * C.ξ₂ p ((C.rxn₂ r i j).fire p)
    rw [hrev₂, VoxelRxn₂.fire_swap_fire (fires_of_fire_ne₂ hne), hκ₂, C.ξ₂_symm]

end VoxelChem

/-! ## One row form

Every rule of a table can be written as one row `a + c ⇌ b + d`. A catalyst is a species on both
sides (`c = d`), and a hop between voxels is a pair of rows, one on each side, trading a species
for a hole. Interior conservation then reduces to one equality per row, which a program can check
on the table. -/

namespace VoxelRxn

variable {S : Type*}

/-- The row `a + c → b + d`. -/
noncomputable def pair (a c b d : S) : VoxelRxn S :=
  ⟨Pi.single a 1 + Pi.single c 1, Pi.single b 1 + Pi.single d 1⟩

theorem swap_pair (a c b d : S) : (pair a c b d).swap = pair b d a c := rfl

end VoxelRxn

namespace VoxelState

variable {S : Type*} [Fintype S]

theorem rxnWeight_pair (w : S → ℕ) (a c : S) :
    rxnWeight w (Pi.single a 1 + Pi.single c 1) = w a + w c := by
  unfold rxnWeight
  simp [Pi.single_apply, mul_add, Finset.sum_add_distrib]

/-- A row conserves weight exactly when its two sides weigh the same. -/
theorem row_conserves (w : S → ℕ) {a c b d : S} (h : w a + w c = w b + w d) :
    rxnWeight w (VoxelRxn.pair a c b d).before = rxnWeight w (VoxelRxn.pair a c b d).after := by
  simp only [VoxelRxn.pair]
  rw [rxnWeight_pair, rxnWeight_pair, h]

end VoxelState

end CGB
