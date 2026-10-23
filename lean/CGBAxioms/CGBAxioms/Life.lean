import CGBAxioms.Medium

/-!
# The infinite-stream life statement

The life definition (`CGB_AXIOMS_V0_20261022.md` §4.1–4.2) splits in two: its premises are
the axioms, its emergent part is a theorem to prove. This file states that theorem.

An organism (a medium) lives in closed loop with a world: the world has its own state space,
much larger than the organism's ("a nearly infinite stream"), decides what each unit senses,
and moves in response to the organism's boundary (its body). Time is discrete: one step is
one tick of a uniformization rate that is the same for every stimulus.

Formalized target properties:
* **L6 maintenance** (`Persists`): the organism holds content with probability at least
  `1 - δ` (`δ < 1`) at every time, and at some time strictly more often than it would in
  the calm (equilibrium) environment: its persistence is paid for by the drive (doc §4,
  "maintenance").
* **L2 memory** (`Remembers`): after the world stops informing (every unit senses the
  world's `quiet` stimulus, which may still carry energy), the organism's state keeps at
  least `ε > 0` nats of information about the last input, *beyond* what its own state held
  when it sensed that input (conditional mutual information), at every later time.

Not yet formalized: L1 (self-generated purpose), L3 (world model), L4 (action selection),
L5 (niche construction).

Known gap: without a clause forcing unpaid structure to decay (ergodicity at finite
temperature), a "sponge" that absorbs content when food is present and never moves again
plausibly satisfies `Persists` and `Remembers`. See the axiom document, §2.1.
-/

open Finset

namespace CGB

/-- A world for organisms whose units `U` have states `Λ` and sense stimuli `I`. -/
structure World (U Λ I : Type*) where
  /-- world states -/
  W : Type
  [fin : Fintype W]
  [dec : DecidableEq W]
  /-- what each unit senses in a world state -/
  sense : W → U → I
  /-- probability of the next world state, given the organism's state -/
  moves : W → (U → Λ) → W → ℝ
  moves_nonneg : ∀ w x w', 0 ≤ moves w x w'
  moves_sum : ∀ w x, ∑ w', moves w x w' = 1
  /-- world state at the organism's birth -/
  start : W
  /-- what units sense once the world stops informing (it may still carry energy) -/
  quiet : U → I

attribute [instance] World.fin World.dec

namespace Medium

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]
  [Fintype I] (M : Medium U Λ A I)

/-- Uniformization rate, the same for every stimulus: exceeds every total exit rate. -/
noncomputable def unif : ℝ := 1 + ∑ s : U → I, ∑ x, ∑ y, (M.jump s).rate x y

/-- One step of the uniformized organism chain under stimulus `s`. -/
noncomputable def kernel (s : U → I) (x y : U → Λ) : ℝ :=
  (M.jump s).rate x y / M.unif +
    if x = y then 1 - (∑ z, (M.jump s).rate x z) / M.unif else 0

lemma unif_pos : 0 < M.unif := by
  have : 0 ≤ ∑ s : U → I, ∑ x, ∑ y, (M.jump s).rate x y :=
    Finset.sum_nonneg fun s _ => Finset.sum_nonneg fun x _ =>
      Finset.sum_nonneg fun y _ => (M.jump s).rate_nonneg x y
  unfold unif; linarith

lemma kernel_nonneg (s : U → I) (x y : U → Λ) : 0 ≤ M.kernel s x y := by
  have hR := M.unif_pos
  have hout : ∑ z, (M.jump s).rate x z ≤ M.unif := by
    have h1 : ∑ z, (M.jump s).rate x z ≤ ∑ x', ∑ z, (M.jump s).rate x' z :=
      Finset.single_le_sum (f := fun x' => ∑ z, (M.jump s).rate x' z)
        (fun x' _ => Finset.sum_nonneg fun z _ => (M.jump s).rate_nonneg x' z) (mem_univ x)
    have h2 : ∑ x', ∑ z, (M.jump s).rate x' z ≤
        ∑ s' : U → I, ∑ x', ∑ z, (M.jump s').rate x' z :=
      Finset.single_le_sum (f := fun s' => ∑ x', ∑ z, (M.jump s').rate x' z)
        (fun s' _ => Finset.sum_nonneg fun x' _ =>
          Finset.sum_nonneg fun z _ => (M.jump s').rate_nonneg x' z) (mem_univ s)
    unfold unif; linarith
  have hfrac : (∑ z, (M.jump s).rate x z) / M.unif ≤ 1 := (div_le_one hR).2 hout
  refine add_nonneg (div_nonneg ((M.jump s).rate_nonneg x y) hR.le) ?_
  split_ifs <;> linarith

/-- The uniformized kernel is stochastic. -/
lemma kernel_sum (s : U → I) (x : U → Λ) : ∑ y, M.kernel s x y = 1 := by
  simp only [kernel, Finset.sum_add_distrib, Finset.sum_ite_eq, Finset.mem_univ, ite_true,
    div_eq_mul_inv]
  rw [← Finset.sum_mul]
  ring

variable (Wd : World U Λ I)

/-- The world reacts to the organism only through its boundary (the body). -/
def ReadsBoundary : Prop :=
  ∀ w x y w', (∀ u ∈ M.boundary, x u = y u) → Wd.moves w x w' = Wd.moves w y w'

/-- Joint law at time `n` of (organism state, world state), when at step `k` the units sense
`sched k w` in world state `w`. -/
noncomputable def law (sched : ℕ → Wd.W → U → I) : ℕ → (U → Λ) → Wd.W → ℝ
  | 0 => fun x w => if x = M.init ∧ w = Wd.start then 1 else 0
  | n + 1 => fun y w' =>
      ∑ x, ∑ w, law sched n x w * Wd.moves w x w' * M.kernel (sched n w) x y

omit [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A] [Fintype I] in
lemma sum4 {α β : Type*} [Fintype α] [Fintype β] (F : α → β → α → β → ℝ) :
    ∑ y, ∑ w', ∑ x, ∑ w, F x w y w' = ∑ x, ∑ w, ∑ y, ∑ w', F x w y w' := by
  have := Finset.sum_comm (s := (univ : Finset (α × β))) (t := (univ : Finset (α × β)))
    (f := fun q p => F p.1 p.2 q.1 q.2)
  simpa only [Fintype.sum_prod_type] using this

/-- The law is a probability distribution at every time. -/
lemma law_sum (sched : ℕ → Wd.W → U → I) (n : ℕ) : ∑ x, ∑ w, M.law Wd sched n x w = 1 := by
  induction n with
  | zero =>
    simp only [law]
    rw [Finset.sum_eq_single M.init (fun x _ hx => by simp [hx]) (by simp)]
    simp
  | succ n ih =>
    have inner : ∀ x w, ∑ y, ∑ w', M.law Wd sched n x w * Wd.moves w x w' *
        M.kernel (sched n w) x y = M.law Wd sched n x w := by
      intro x w
      rw [Finset.sum_comm]
      simp only [← Finset.mul_sum, M.kernel_sum, mul_one, Wd.moves_sum]
    simp only [law]
    calc _ = ∑ x, ∑ w, ∑ y, ∑ w', M.law Wd sched n x w * Wd.moves w x w' *
          M.kernel (sched n w) x y :=
          sum4 (fun x w y w' => M.law Wd sched n x w * Wd.moves w x w' * M.kernel (sched n w) x y)
      _ = 1 := by simp only [inner, ih]

/-- Probability that the organism holds content at time `n` under schedule `sched`. -/
noncomputable def aliveProb (sched : ℕ → Wd.W → U → I) (n : ℕ) : ℝ :=
  ∑ x, ∑ w, if 0 < M.total x then M.law Wd sched n x w else 0

/-- **L6 maintenance**: alive with probability at least `1 - δ` (with `δ < 1`) at every time in the
world, and at some time strictly more often than in the calm environment. -/
def Persists (δ : ℝ) : Prop :=
  δ < 1 ∧ (∀ n, 1 - δ ≤ M.aliveProb Wd (fun _ => Wd.sense) n) ∧
    ∃ n, M.aliveProb Wd (fun _ _ => M.calm) n < M.aliveProb Wd (fun _ => Wd.sense) n

/-- Law at time `n₀ + m` of (organism state, world state), tagged with the organism state
`x₀` and world state `w₀` at time `n₀`. At step `n₀` the organism senses `w₀`; afterwards
it senses only the world's `quiet` stimulus. -/
noncomputable def memLaw (n₀ : ℕ) : ℕ → (U → Λ) → Wd.W → (U → Λ) → Wd.W → ℝ
  | 0 => fun x w x₀ w₀ => if x = x₀ ∧ w = w₀ then M.law Wd (fun _ => Wd.sense) n₀ x w else 0
  | m + 1 => fun y w' x₀ w₀ =>
      ∑ x, ∑ w, memLaw n₀ m x w x₀ w₀ * Wd.moves w x w' *
        M.kernel (if m = 0 then Wd.sense w else Wd.quiet) x y

/-- Conditional mutual information `I(B; C | A)` of a joint distribution (in nats). -/
noncomputable def condMutualInfo {α β γ : Type*} [Fintype α] [Fintype β] [Fintype γ]
    (p : α → β → γ → ℝ) : ℝ :=
  ∑ a, ∑ b, ∑ c, if p a b c = 0 then 0 else
    p a b c * Real.log (p a b c * (∑ b', ∑ c', p a b' c') /
      ((∑ c', p a b c') * ∑ b', p a b' c))

/-- **L2 memory**: after the world stops informing, the organism keeps at least `ε > 0`
nats about the last input beyond what its own state already held: `I(X_later; W₀ | X₀)`.
Given `X₀`, the organism's later states depend on `W₀` only through what it sensed. -/
def Remembers : Prop :=
  ∃ n₀ ε, 0 < ε ∧ ∀ m, ε ≤
    condMutualInfo (fun x₀ w₀ x => ∑ w, M.memLaw Wd n₀ (m + 1) x w x₀ w₀)

/-- An organism lives in a world: all six clauses, genome bottleneck at most `ρ`, a world
that touches only the body, maintenance and memory. -/
structure Lives (ρ δ : ℝ) : Prop where
  axioms : M.Axioms
  bottleneck : M.rho ≤ ρ
  body : M.ReadsBoundary Wd
  persists : M.Persists Wd δ
  remembers : M.Remembers Wd

end Medium

/-- **Main existence conjecture** (statement only): in the world `Wd`, for a body of units
`U` with local states `Λ`, some rulebook with bottleneck at most `ρ` lives. -/
def LifeConjecture {U Λ I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ]
    [Fintype I] (Wd : World U Λ I) (ρ δ : ℝ) : Prop :=
  ∃ (A : Type) (_ : Fintype A) (M : Medium U Λ A I), M.Lives Wd ρ δ

end CGB
