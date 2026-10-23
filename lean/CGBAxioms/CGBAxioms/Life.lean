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
* **L1/L4/L6 self-maintenance through the loop** (`SelfMaintains`): the organism changes
  the stream, and the changed stream keeps it alive. Formally, from some time on it is
  alive more often (by a fixed margin) in the closed loop than in a loop-cut world, where
  the world reacts to a statistically identical but causally disconnected organism
  (`lawCut`). Its viable region is then defined by itself, not by a designer.
* **L2 memory** (`Remembers`): after the world stops informing (every unit senses the
  world's `quiet` stimulus, which may still carry energy), the organism's state keeps at
  least `ε > 0` nats of information about the last input, *beyond* what its own state held
  when it sensed that input (conditional mutual information), at every later time.

Not yet formalized: L1 (self-generated purpose), L3 (world model), L4 (action selection),
L5 (niche construction).

Necessary conditions, proved: an organism that ignores its stimuli
(`not_selfMaintains_of_unexposed`) or exchanges no content with the world
(`not_selfMaintains_of_closedMatter`) cannot self-maintain. So a rock, a second-order
controller or an agent whose existence is not at stake never lives; exposure and
metabolism are consequences of self-maintenance, not extra assumptions.

Not yet formalized: L1 beyond self-maintenance, L3 (world model), L4 (action selection),
L5 (niche construction). A passive "sponge" that only eats when food is present is the
open test case for L2 learning and L3.
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

/-- `1` if the organism holds content (has not dissolved into the vacuum), else `0`. -/
noncomputable def aliveInd (x : U → Λ) : ℝ := if 0 < M.total x then 1 else 0

/-- Probability that the organism holds content at time `n` under schedule `sched`. -/
noncomputable def aliveProb (sched : ℕ → Wd.W → U → I) (n : ℕ) : ℝ :=
  ∑ x, ∑ w, M.aliveInd x * M.law Wd sched n x w

omit [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A] [Fintype I] in
/-- One joint step, weighted by a test function `a` that the organism kernel preserves. -/
lemma stepSum {α β : Type*} [Fintype α] [Fintype β] (a : α → ℝ) (L : α → β → ℝ)
    (μ : α → β → β → ℝ) (K : β → α → α → ℝ) (hμ : ∀ x w, ∑ w', μ x w w' = 1)
    (hK : ∀ w x, ∑ y, a y * K w x y = a x) :
    ∑ y, ∑ w', a y * ∑ x, ∑ w, L x w * μ x w w' * K w x y = ∑ x, ∑ w, a x * L x w := by
  simp only [Finset.mul_sum]
  calc _ = ∑ x, ∑ w, ∑ y, ∑ w', a y * (L x w * μ x w w' * K w x y) :=
        sum4 (fun x w y w' => a y * (L x w * μ x w w' * K w x y))
    _ = _ := by
      refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun w _ => ?_
      calc ∑ y, ∑ w', a y * (L x w * μ x w w' * K w x y)
          = L x w * ((∑ y, a y * K w x y) * ∑ w', μ x w w') := by
            rw [Finset.sum_mul_sum, Finset.mul_sum]
            refine Finset.sum_congr rfl fun y _ => ?_
            rw [Finset.mul_sum]
            exact Finset.sum_congr rfl fun w' _ => by ring
        _ = a x * L x w := by rw [hK, hμ]; ring

omit [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A] [Fintype I] in
/-- One joint step of the organism's marginal, when its kernel ignores the world. -/
lemma margStep {α β : Type*} [Fintype α] [Fintype β] (L : α → β → ℝ) (μ : α → β → β → ℝ)
    (K : α → α → ℝ) (hμ : ∀ x w, ∑ w', μ x w w' = 1) (y : α) :
    ∑ w', ∑ x, ∑ w, L x w * μ x w w' * K x y = ∑ x, (∑ w, L x w) * K x y := by
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.sum_comm, Finset.sum_mul]
  refine Finset.sum_congr rfl fun w _ => ?_
  rw [← Finset.sum_mul, ← Finset.mul_sum, hμ, mul_one]

/-- How the world moves in the loop-cut world: driven by an independent organism with the
same marginal law `∑ w, L x w`. -/
noncomputable def cutMove (L : (U → Λ) → Wd.W → ℝ) (_x : U → Λ) (w w' : Wd.W) : ℝ :=
  ∑ x', (∑ w'', L x' w'') * Wd.moves w x' w'

omit [DecidableEq Λ] [Fintype I] in
lemma cutMove_sum {L : (U → Λ) → Wd.W → ℝ} (hL : ∑ x, ∑ w, L x w = 1) (x : U → Λ)
    (w : Wd.W) : ∑ w', cutMove Wd L x w w' = 1 := by
  simp only [cutMove]
  rw [Finset.sum_comm]
  simp only [← Finset.mul_sum, Wd.moves_sum, mul_one]
  exact hL

/-- Law of the loop-cut world: the organism senses the world as before, but the world
reacts to a statistically identical, causally disconnected organism. -/
noncomputable def lawCut (sched : ℕ → Wd.W → U → I) : ℕ → (U → Λ) → Wd.W → ℝ
  | 0 => fun x w => if x = M.init ∧ w = Wd.start then 1 else 0
  | n + 1 => fun y w' =>
      ∑ x, ∑ w, lawCut sched n x w * cutMove Wd (lawCut sched n) x w w' *
        M.kernel (sched n w) x y

lemma lawCut_sum (sched : ℕ → Wd.W → U → I) (n : ℕ) :
    ∑ x, ∑ w, M.lawCut Wd sched n x w = 1 := by
  induction n with
  | zero =>
    simp only [lawCut]
    rw [Finset.sum_eq_single M.init (fun x _ hx => by simp [hx]) (by simp)]
    simp
  | succ n ih =>
    have h := stepSum (fun _ => (1 : ℝ)) (M.lawCut Wd sched n)
      (cutMove Wd (M.lawCut Wd sched n)) (fun w => M.kernel (sched n w))
      (cutMove_sum Wd ih) (fun w x => by simp [M.kernel_sum])
    simp only [one_mul] at h
    simp only [lawCut]
    rw [h, ih]

/-- Probability of being alive at time `n` in the loop-cut world. -/
noncomputable def aliveProbCut (sched : ℕ → Wd.W → U → I) (n : ℕ) : ℝ :=
  ∑ x, ∑ w, M.aliveInd x * M.lawCut Wd sched n x w

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

/-- The organism's rates or effects depend on what it senses. -/
def Exposed : Prop := ∃ a s s' x, M.rate a s x ≠ M.rate a s' x ∨ M.act a s x ≠ M.act a s' x

/-- **Self-maintenance through the loop**: from some time on, the organism is alive more
often, by a fixed margin `ε`, in the closed loop than in the loop-cut world. What keeps it
alive is its own influence on the stream it receives. -/
def SelfMaintains : Prop :=
  ∃ n₀ ε, 0 < ε ∧ ∀ n, n₀ ≤ n →
    M.aliveProbCut Wd (fun _ => Wd.sense) n + ε ≤ M.aliveProb Wd (fun _ => Wd.sense) n

/-- An organism that ignores its stimuli cannot self-maintain: its own law does not depend
on the world, so cutting the loop changes nothing. -/
theorem not_selfMaintains_of_unexposed (h : ¬ M.Exposed) : ¬ M.SelfMaintains Wd := by
  simp only [Exposed, not_exists, not_or, not_not] at h
  have hrate : ∀ s s' x y, (M.jump s).rate x y = (M.jump s').rate x y := fun s s' x y => by
    simp only [jump]
    exact Finset.sum_congr rfl fun a _ => by rw [(h a s s' x).1, (h a s s' x).2]
  have hK : ∀ w, M.kernel (Wd.sense w) = M.kernel Wd.quiet := fun w => by
    funext x y
    simp only [kernel, hrate (Wd.sense w) Wd.quiet]
  have hm : ∀ n x, ∑ w, M.law Wd (fun _ => Wd.sense) n x w =
      ∑ w, M.lawCut Wd (fun _ => Wd.sense) n x w := by
    intro n
    induction n with
    | zero => intro x; rfl
    | succ n ih =>
      intro y
      simp only [law, lawCut, hK]
      rw [margStep _ (fun x w w' => Wd.moves w x w') _ (fun x w => Wd.moves_sum w x) y,
        margStep _ (cutMove Wd (M.lawCut Wd (fun _ => Wd.sense) n)) _
          (cutMove_sum Wd (M.lawCut_sum Wd _ n)) y]
      exact Finset.sum_congr rfl fun x _ => by rw [ih x]
  have heq : ∀ n, M.aliveProb Wd (fun _ => Wd.sense) n =
      M.aliveProbCut Wd (fun _ => Wd.sense) n := fun n => by
    simp only [aliveProb, aliveProbCut, ← Finset.mul_sum, hm]
  rintro ⟨n₀, ε, hε, hn⟩
  have := hn n₀ le_rfl
  rw [heq] at this
  linarith

/-- An organism that exchanges no content with the world cannot self-maintain: whether it
holds content never changes, in the loop or out of it. -/
theorem not_selfMaintains_of_closedMatter
    (h : ∀ a s x, 0 < M.rate a s x → M.total (M.act a s x) = M.total x) :
    ¬ M.SelfMaintains Wd := by
  have hK : ∀ s x, ∑ y, M.aliveInd y * M.kernel s x y = M.aliveInd x := by
    intro s x
    have hz : ∀ y, M.aliveInd y ≠ M.aliveInd x → M.kernel s x y = 0 := by
      intro y hy
      have hne : x ≠ y := fun e => hy (by rw [e])
      have htot : M.total y ≠ M.total x := fun e => hy (by simp only [aliveInd, e])
      have hj : (M.jump s).rate x y = 0 := by
        simp only [jump]
        refine Finset.sum_eq_zero fun a _ => ?_
        split_ifs with ha
        · rcases (M.rate_nonneg a s x).lt_or_eq with hp | hp
          · exact absurd (ha.1 ▸ h a s x hp) htot
          · exact hp.symm
        · rfl
      simp [kernel, hj, hne]
    calc ∑ y, M.aliveInd y * M.kernel s x y = ∑ y, M.aliveInd x * M.kernel s x y :=
          Finset.sum_congr rfl fun y _ => by
            by_cases hy : M.aliveInd y = M.aliveInd x
            · rw [hy]
            · rw [hz y hy, mul_zero, mul_zero]
      _ = M.aliveInd x := by rw [← Finset.mul_sum, M.kernel_sum, mul_one]
  have hc : ∀ n, M.aliveProb Wd (fun _ => Wd.sense) n = M.aliveProb Wd (fun _ => Wd.sense) 0 := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih =>
      rw [← ih]
      exact stepSum _ _ _ (fun w => M.kernel (Wd.sense w)) (fun x w => Wd.moves_sum w x)
        (fun w x => hK _ x)
  have hcut : ∀ n, M.aliveProbCut Wd (fun _ => Wd.sense) n =
      M.aliveProbCut Wd (fun _ => Wd.sense) 0 := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih =>
      rw [← ih]
      exact stepSum _ _ _ (fun w => M.kernel (Wd.sense w))
        (cutMove_sum Wd (M.lawCut_sum Wd _ n)) (fun w x => hK _ x)
  rintro ⟨n₀, ε, hε, hn⟩
  have := hn n₀ le_rfl
  rw [hc, hcut, show M.aliveProbCut Wd (fun _ => Wd.sense) 0 =
    M.aliveProb Wd (fun _ => Wd.sense) 0 from rfl] at this
  linarith

/-- **L2 memory**: after the world stops informing, the organism keeps at least `ε > 0`
nats about the last input beyond what its own state already held: `I(X_later; W₀ | X₀)`.
Given `X₀`, the organism's later states depend on `W₀` only through what it sensed. -/
def Remembers : Prop :=
  ∃ n₀ ε, 0 < ε ∧ ∀ m, ε ≤
    condMutualInfo (fun x₀ w₀ x => ∑ w, M.memLaw Wd n₀ (m + 1) x w x₀ w₀)

/-- An organism lives in a world: all six clauses, genome bottleneck at most `ρ`, a world
that touches only the body and has at least `κ` times as many states as the organism (a
nearly infinite stream), maintenance, self-maintenance through the loop, and memory. -/
structure Lives (ρ δ : ℝ) (κ : ℕ) : Prop where
  axioms : M.Axioms
  bottleneck : M.rho ≤ ρ
  body : M.ReadsBoundary Wd
  rich : κ * Fintype.card (U → Λ) ≤ Fintype.card Wd.W
  persists : M.Persists Wd δ
  selfMaintains : M.SelfMaintains Wd
  remembers : M.Remembers Wd

/-- A living organism necessarily senses its world. -/
theorem Lives.exposed {ρ δ : ℝ} {κ : ℕ} (h : M.Lives Wd ρ δ κ) : M.Exposed :=
  by_contra fun hn => M.not_selfMaintains_of_unexposed Wd hn h.selfMaintains

/-- A living organism necessarily exchanges content with its world (metabolism). -/
theorem Lives.metabolizes {ρ δ : ℝ} {κ : ℕ} (h : M.Lives Wd ρ δ κ) :
    ∃ a s x, 0 < M.rate a s x ∧ M.total (M.act a s x) ≠ M.total x := by
  by_contra hn
  exact M.not_selfMaintains_of_closedMatter Wd
    (fun a s x hp => by_contra fun hne => hn ⟨a, s, x, hp, hne⟩) h.selfMaintains

end Medium

/-- **Main existence conjecture** (statement only): in the world `Wd`, for a body of units
`U` with local states `Λ`, some rulebook with bottleneck at most `ρ` lives. -/
def LifeConjecture {U Λ I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ]
    [Fintype I] (Wd : World U Λ I) (ρ δ : ℝ) (κ : ℕ) : Prop :=
  ∃ (A : Type) (_ : Fintype A) (M : Medium U Λ A I), M.Lives Wd ρ δ κ

end CGB
