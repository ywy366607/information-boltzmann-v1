import CGBAxioms.Symmetry

/-!
# The CGB medium and the six axiom clauses

The full language of `docs/information_boltzmann/CGB_AXIOMS_V0_20261022.md` §1–2.
A *medium* is a finite set of units (sites in space) with local states, a contact relation,
an identity page per unit, a boundary, a content cost per local state, and events with
participants, effects and nonnegative rates. Every unit also receives a *stimulus* from
the environment; the environment is a stream of stimuli `ε : ℕ → U → I`. Growth needs no
extra machinery: a unit in the vacuum state is an empty site, and a unit is born when its
site leaves the vacuum.

The six clauses P1a–P3 are predicates on a medium, never Lean `axiom`s.

* `finite_reachable` (**T0a**): conservation implies finiteness for the closed dynamics,
  without the capacity bound; `finite_states`: with the capacity bound (needed once the
  medium is open) the whole state space is finite.
* `p2b_of_micro`: microscopic reversibility with equal forward/backward rates ⇒ P2b.
* Parameters, not axioms: `rho` (rulebook pages per unit, the genome bottleneck) and
  `boundaryFraction`.
* `jump s` is the full open dynamics under a fixed stimulus `s`; `calm` is the equilibrium
  environment, under which detailed balance must hold.
-/

open Finset

namespace CGB

/-- A medium on units `U` with local states `Λ`, events `A` and stimuli `I`. -/
structure Medium (U Λ A I : Type*) where
  /-- contact: which units can take part in one event together -/
  near : U → U → Prop
  /-- identity: the rulebook page a unit reads -/
  ident : U → ℕ
  /-- boundary units (sensory organs, body interface) -/
  boundary : Finset U
  /-- conserved content each local state costs -/
  content : Λ → ℕ
  /-- participants of an event -/
  parts : A → Finset U
  /-- effect of an event, given the stimulus and the global state -/
  act : A → (U → I) → (U → Λ) → (U → Λ)
  /-- rate of an event, given the stimulus and the global state -/
  rate : A → (U → I) → (U → Λ) → ℝ
  rate_nonneg : ∀ a s x, 0 ≤ rate a s x
  /-- birth state -/
  init : U → Λ
  /-- the equilibrium environment: all boundary reservoirs at the same potential -/
  calm : U → I

namespace Medium

variable {U Λ A I : Type*} (M : Medium U Λ A I)

/-- Total content of a global state. -/
def total [Fintype U] (x : U → Λ) : ℕ := ∑ u, M.content (x u)

/-- An event is closed if no participant is on the boundary. -/
def Closed (a : A) : Prop := Disjoint (M.parts a) M.boundary

instance [DecidableEq U] (a : A) : Decidable (M.Closed a) := by
  unfold Closed; infer_instance

/-- Symmetries of the given structure: permutations of units preserving contact,
identity and boundary. -/
def IsAut (σ : Equiv.Perm U) : Prop :=
  (∀ u v, M.near (σ u) (σ v) ↔ M.near u v) ∧ (∀ u, M.ident (σ u) = M.ident u) ∧
    ∀ u, σ u ∈ M.boundary ↔ u ∈ M.boundary

/-- The full (open) dynamics under stimulus `s`, as a jump process on global states. -/
noncomputable def jump [Fintype A] [DecidableEq (U → Λ)] (s : U → I) : JumpProcess (U → Λ) where
  rate x y := ∑ a, if M.act a s x = y ∧ x ≠ y then M.rate a s x else 0
  rate_nonneg x y := Finset.sum_nonneg fun a _ => by
    split_ifs
    exacts [M.rate_nonneg a s x, le_rfl]

/-! ## The six clauses -/

/-- **P1a** states cost content: only finitely many local states fit under any content
bound, a unit (a finite region) holds at most a bounded content, and exactly one state (the
vacuum) is free. -/
def P1a : Prop :=
  (∀ n, {ℓ : Λ | M.content ℓ ≤ n}.Finite) ∧ (∃ K, ∀ ℓ, M.content ℓ ≤ K) ∧
    ∃! ℓ, M.content ℓ = 0

/-- **P1b** influence needs contact: participants are in contact, an event changes only
its participants, and its rate and effect read only its participants' states and stimuli.

Revision 2026-10-26: contact homogeneity (every unit looks like every other) was removed
from P1b. It entered no derivation of the generation chain, it cannot stop per-unit rules
(`ConstraintAudit.rigid_of_injective_identity`), and it forces a regular contact graph
(`GenerationAxiomAudit.homogeneous_regular`), excluding trees, hubs, finite grids with a
surface and line graphs of unequal compartments. It survives as the optional model property
`HomogeneousContact`. -/
def P1b : Prop :=
  (∀ a, ∀ u ∈ M.parts a, ∀ v ∈ M.parts a, M.near u v) ∧
  (∀ a s x u, u ∉ M.parts a → M.act a s x u = x u) ∧
  ∀ a s s' x y, (∀ u ∈ M.parts a, x u = y u ∧ s u = s' u) →
    M.rate a s x = M.rate a s' y ∧ ∀ u ∈ M.parts a, M.act a s x u = M.act a s' y u

/-- **P2a** conservation: closed events with positive rate preserve total content. -/
def P2a [Fintype U] : Prop :=
  ∀ a s x, M.Closed a → 0 < M.rate a s x → M.total (M.act a s x) = M.total x

/-- **P2b** time-reversal symmetry: in the equilibrium environment the whole dynamics,
boundary exchange included, satisfies detailed balance. Irreversibility can then enter
only through the stimulus. -/
def P2b [Fintype A] [DecidableEq (U → Λ)] : Prop := (M.jump M.calm).Reversible

/-- **P2c** one rulebook: the rules commute with every symmetry of contact, identity and
boundary, so they can tell units apart only by identity, position, state and stimulus. -/
def P2c : Prop :=
  ∀ σ, M.IsAut σ → ∃ τ : Equiv.Perm A, ∀ a s x,
    M.rate (τ a) (s ∘ σ.symm) (x ∘ σ.symm) = M.rate a s x ∧
      M.act (τ a) (s ∘ σ.symm) (x ∘ σ.symm) = M.act a s x ∘ σ.symm

/-- Only boundary units sense the environment. -/
def OnlyBoundarySenses : Prop :=
  ∀ a s s' x, (∀ u ∈ M.boundary, s u = s' u) →
    M.rate a s x = M.rate a s' x ∧ M.act a s x = M.act a s' x

/-- **P3** symmetry breaking only from boundary, identity/position and history: the
environment (which breaks time-translation symmetry) acts only through the boundary, and
the birth state carries no asymmetry beyond the given structure. -/
def P3 : Prop := M.OnlyBoundarySenses ∧ ∀ σ, M.IsAut σ → M.init ∘ σ.symm = M.init

/-- Optional model property (removed from P1b on 2026-10-26): contact comes from a homogeneous
space, every unit looks like every other. -/
def HomogeneousContact : Prop :=
  ∀ u v, ∃ σ : Equiv.Perm U, (∀ a b, M.near (σ a) (σ b) ↔ M.near a b) ∧ σ u = v

/-- A model of the CGB axioms. -/
structure Axioms [Fintype U] [Fintype A] [DecidableEq (U → Λ)] : Prop where
  p1a : M.P1a
  p1b : M.P1b
  p2a : M.P2a
  p2b : M.P2b
  p2c : M.P2c
  p3 : M.P3

/-! ## Parameters: the genome bottleneck -/

/-- The rulebook pages in use. -/
def pages [Fintype U] : Finset ℕ := univ.image M.ident

/-- **ρ**: rulebook pages per unit. A parameter, not an axiom; it sets how much P2c bites
(see `Models.xhop_pages`). -/
noncomputable def rho [Fintype U] : ℝ := (M.pages.card : ℝ) / Fintype.card U

/-- Boundary units per unit. A parameter, not an axiom; it sets how much P2a bites. -/
noncomputable def boundaryFraction [Fintype U] : ℝ := (M.boundary.card : ℝ) / Fintype.card U

/-! ## First consequences -/

/-- One closed event with positive rate under stimulus `s`. -/
def Step (s : U → I) (x y : U → Λ) : Prop :=
  ∃ a, M.Closed a ∧ 0 < M.rate a s x ∧ M.act a s x = y

/-- P2a confines the closed dynamics to a content shell. -/
theorem total_of_reachable [Fintype U] (h2 : M.P2a) {s : U → I} {x y : U → Λ}
    (hy : Relation.ReflTransGen (M.Step s) x y) : M.total y = M.total x := by
  induction hy with
  | refl => rfl
  | tail _ hst ih =>
    obtain ⟨a, ha, hr, rfl⟩ := hst
    rw [h2 a _ _ ha hr, ih]

/-- **T0a** conservation implies finiteness: under P2a and the first part of P1a, the
closed dynamics reaches only finitely many states. The capacity bound is not needed here;
it is needed once content can enter through the boundary. -/
theorem finite_reachable [Fintype U] (h1 : ∀ n, {ℓ : Λ | M.content ℓ ≤ n}.Finite)
    (h2 : M.P2a) (s : U → I) (x : U → Λ) :
    {y | Relation.ReflTransGen (M.Step s) x y}.Finite := by
  refine (Set.Finite.pi fun _ => h1 (M.total x)).subset fun y hy => Set.mem_univ_pi.2 fun u => ?_
  change M.content (y u) ≤ M.total x
  rw [← M.total_of_reachable h2 hy]
  exact Finset.single_le_sum (f := fun u => M.content (y u)) (fun _ _ => Nat.zero_le _)
    (Finset.mem_univ u)

/-- With the capacity bound the whole state space is finite: the premise "finite state
space" of the life definition. -/
theorem finite_states [Finite U] (h : M.P1a) : Finite (U → Λ) := by
  obtain ⟨K, hK⟩ := h.2.1
  have : Finite Λ := by
    have hfin := h.1 K
    rw [show {ℓ : Λ | M.content ℓ ≤ K} = Set.univ from Set.eq_univ_of_forall hK] at hfin
    exact Set.finite_univ_iff.1 hfin
  infer_instance

/-- **T-E** for media: P2b gives an energy function of the equilibrium dynamics. -/
theorem energy_of_p2b [Fintype A] [DecidableEq (U → Λ)] (h : M.P2b) :
    ∃ E : (U → Λ) → ℝ, ∀ x y, 0 < (M.jump M.calm).rate y x →
      (M.jump M.calm).rate x y / (M.jump M.calm).rate y x = Real.exp (-(E y - E x)) :=
  JumpProcess.energy_of_reversible _ h

/-- Microscopic reversibility: if in the equilibrium environment every effective event is
undone by its reverse event at the same rate, P2b holds (with uniform weights). -/
theorem p2b_of_micro [Fintype A] [DecidableEq (U → Λ)]
    (rev : A → A) (hinv : Function.Involutive rev)
    (hback : ∀ a x, M.act a M.calm x ≠ x →
      M.act (rev a) M.calm (M.act a M.calm x) = x ∧
        M.rate (rev a) M.calm (M.act a M.calm x) = M.rate a M.calm x) :
    M.P2b := by
  refine ⟨fun _ => 1, fun _ => one_pos, fun x y => ?_⟩
  simp only [one_mul, jump]
  rw [← Equiv.sum_comp (hinv.toPerm rev)
    (fun a => if M.act a M.calm y = x ∧ y ≠ x then M.rate a M.calm y else 0)]
  refine Finset.sum_congr rfl fun a _ => ?_
  have e : (hinv.toPerm rev) a = rev a := rfl
  split_ifs with h1 h2 h2
  · obtain ⟨hxy, hne⟩ := h1
    have hb := hback a x (by rw [hxy]; exact Ne.symm hne)
    rw [hxy] at hb
    rw [e, hb.2]
  · obtain ⟨hxy, hne⟩ := h1
    have hb := hback a x (by rw [hxy]; exact Ne.symm hne)
    rw [hxy] at hb
    exact absurd ⟨by rw [e]; exact hb.1, Ne.symm hne⟩ h2
  · rw [e] at h2
    obtain ⟨hyx, hne⟩ := h2
    have hb := hback (rev a) y (by rw [hyx]; exact Ne.symm hne)
    rw [hyx, hinv a] at hb
    exact absurd ⟨hb.1, Ne.symm hne⟩ h1
  · rfl

end Medium

end CGB
