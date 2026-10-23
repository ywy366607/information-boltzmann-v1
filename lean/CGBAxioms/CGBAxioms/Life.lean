import CGBAxioms.Medium

/-!
# The infinite-stream life statement

**Historical menu semantics (versioned 2026-10-23).** The `recover`, `aliveInd`,
`Lives` and `LifeConjecture` below retain the old favourable-stimulus menu for
compatibility. They do not measure current-world reachability. New operational
targets live in `OperationalLife.lean`, using actual joint first-hit probabilities
and a finite encoded rulebook. No equivalence of the two targets is asserted.
Descriptions of negentropy, anaesthesia and death below are historical motivation;
the implementation observes content intake and a sensory schedule, not a full
world energy ledger or an irreversible probability-threshold crossing.

The life definition (`CGB_AXIOMS_V0_20261022.md` §4.1–4.2) splits in two: its premises are
the axioms, its emergent part is a theorem to prove. This file states that theorem.

An organism (a medium) lives in closed loop with a world: the world has its own state space,
much larger than the organism's ("a nearly infinite stream"), decides what each unit senses,
and moves in response to the organism's boundary (its body). Time is discrete and owned by
the world: one step is one tick of the world's clock, and the organism's rates must stay
below it (`Clocked`, finite rates).

Formalized target properties:
* **Alive** (user 2026-10-23): life takes in negentropy, and death is the loss of
  continuity. A state is *recoverable* if some path of the organism's own events leads to
  an intake; losing that is irreversible (`irrecoverable_absorbing`). Possibility is not
  enough: under detailed balance every step can be undone, so the way back always exists,
  only with vanishing probability (entropy). So a non-empty state is *alive* when, within
  the world's horizon `H`, an actual intake (a jump that raises content) happens with
  probability at least the world's threshold `p > 0`, with stimuli chosen as favourably as
  the world can produce outside the calm environment (`recover`, `aliveInd`). Pushed
  towards the cliff, that probability falls. An organism with no intake event at all (a
  rock, a body frozen in every environment) was never alive (`never_alive_without_intake`).
* **L6, self-defined viability** (`Persists`): alive with probability at least `1 - δ`
  (`δ < 1`) at every time in the world; still so through any interruption of `T` steps in
  which it senses only the calm environment (anaesthesia: pushed to the cliff edge, not
  over it); past a critical interruption `T* > T` it stays dead even when the world returns
  (the critical point); and permanently cut off, it is from some time on dead (it needs
  negentropy).
* **L1/L4/L6 self-maintenance through the loop** (`SelfMaintains`): the organism changes
  the stream, and the changed stream keeps it alive: from some time on it is alive more
  often (by a fixed margin) in the closed loop than in a loop-cut world, where the world
  reacts to a statistically identical but causally disconnected organism (`lawCut`).
* **L2 memory** (`Remembers`): after the world stops informing (every unit senses the
  world's `quiet` stimulus, which may still carry energy), the organism's state keeps at
  least `ε > 0` nats of information about the last input, *beyond* what its own state held
  when it sensed that input (conditional mutual information), at every later time. Known
  gap: an organism that freezes under `quiet` meets this trivially.

Necessary conditions, proved: an organism that ignores its stimuli
(`not_selfMaintains_of_unexposed`) or whose events never change its content
(`not_selfMaintains_of_closedMatter`) cannot self-maintain. With "alive" defined through
intake the second is close to definitional; the first is not.

Not yet formalized: L1 beyond self-maintenance, L3 (world model), L4 (action selection),
L5 (niche construction). Active-inference rungs are in `Inference.lean`.
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
  /-- clock rate: one step of time is one tick; every rate is measured against it -/
  clock : ℝ
  clock_pos : 0 < clock
  /-- time scale of recovery: steps within which an organism must be able to eat again -/
  horizon : ℕ
  /-- probability of recovery below which an organism counts as dead -/
  recovery : ℝ
  recovery_pos : 0 < recovery

attribute [instance] World.fin World.dec

namespace Medium

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]
  (M : Medium U Λ A I)

/-- One step of the organism chain under stimulus `s`, at clock rate `R`: an event of
rate `r` fires within one tick with probability `r / R`. -/
noncomputable def kernel (R : ℝ) (s : U → I) (x y : U → Λ) : ℝ :=
  (M.jump s).rate x y / R + if x = y then 1 - (∑ z, (M.jump s).rate x z) / R else 0

/-- The organism's total exit rates never exceed the clock (finite rates, T0b). -/
def Clocked (R : ℝ) : Prop := 0 < R ∧ ∀ s x, ∑ y, (M.jump s).rate x y ≤ R

lemma kernel_nonneg {R : ℝ} (h : M.Clocked R) (s : U → I) (x y : U → Λ) :
    0 ≤ M.kernel R s x y := by
  have hfrac : (∑ z, (M.jump s).rate x z) / R ≤ 1 := (div_le_one h.1).2 (h.2 s x)
  refine add_nonneg (div_nonneg ((M.jump s).rate_nonneg x y) h.1.le) ?_
  split_ifs <;> linarith

/-- The kernel is stochastic. -/
lemma kernel_sum (R : ℝ) (s : U → I) (x : U → Λ) : ∑ y, M.kernel R s x y = 1 := by
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
      ∑ x, ∑ w, law sched n x w * Wd.moves w x w' * M.kernel Wd.clock (sched n w) x y

omit [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A] in
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
        M.kernel Wd.clock (sched n w) x y = M.law Wd sched n x w := by
      intro x w
      rw [Finset.sum_comm]
      simp only [← Finset.mul_sum, M.kernel_sum, mul_one, Wd.moves_sum]
    simp only [law]
    calc _ = ∑ x, ∑ w, ∑ y, ∑ w', M.law Wd sched n x w * Wd.moves w x w' *
          M.kernel Wd.clock (sched n w) x y :=
          sum4 (fun x w y w' =>
            M.law Wd sched n x w * Wd.moves w x w' * M.kernel Wd.clock (sched n w) x y)
      _ = 1 := by simp only [inner, ih]

omit [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A] in
/-- One event of the organism alone, under some stimulus. -/
def OrgStep (x y : U → Λ) : Prop := ∃ a s, 0 < M.rate a s x ∧ M.act a s x = y

omit [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A] in
/-- An intake is possible: some event brings content in. -/
def CanTakeIn (x : U → Λ) : Prop :=
  ∃ a s, 0 < M.rate a s x ∧ M.total x < M.total (M.act a s x)

omit [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A] in
/-- **Recoverable**: along some path of its own events under suitable stimuli, the organism
can still take content in again. -/
def Recoverable (x : U → Λ) : Prop :=
  ∃ y, Relation.ReflTransGen M.OrgStep x y ∧ M.CanTakeIn y

omit [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A] in
/-- **Losing recoverability is irreversible**: no event leads from an irrecoverable state
back to a recoverable one. -/
theorem irrecoverable_absorbing {x y : U → Λ} (hd : ¬ M.Recoverable x) (h : M.OrgStep x y) :
    ¬ M.Recoverable y :=
  fun ⟨z, hz, hi⟩ => hd ⟨z, Relation.ReflTransGen.head h hz, hi⟩

open Classical in
/-- Best probability that an actual intake (a jump that raises content) happens within `k`
ticks, the stimulus at each tick chosen as favourably as the world can produce outside the
calm environment. -/
noncomputable def recover : ℕ → (U → Λ) → ℝ
  | 0 => fun _ => 0
  | k + 1 => fun x => ⨆ w : Wd.W, if Wd.sense w = M.calm then 0 else
      ∑ y, M.kernel Wd.clock (Wd.sense w) x y *
        (if M.total x < M.total y then 1 else recover k y)

open Classical in
/-- `1` if the organism is alive: it holds content and, within the world's horizon, eats
again with at least the world's recovery probability; else `0`. -/
noncomputable def aliveInd (x : U → Λ) : ℝ :=
  if 0 < M.total x ∧ Wd.recovery ≤ M.recover Wd Wd.horizon x then 1 else 0

/-- Without any possible intake, no step raises content. -/
lemma kernel_up_zero (h : ∀ x, ¬ M.CanTakeIn x) (R : ℝ) (s : U → I) {x y : U → Λ}
    (hlt : M.total x < M.total y) : M.kernel R s x y = 0 := by
  have hne : x ≠ y := fun e => (lt_irrefl _ (e ▸ hlt))
  have hj : (M.jump s).rate x y = 0 := by
    simp only [jump]
    refine Finset.sum_eq_zero fun a _ => ?_
    split_ifs with ha
    · rcases (M.rate_nonneg a s x).lt_or_eq with hp | hp
      · exact absurd ⟨a, s, hp, ha.1 ▸ hlt⟩ (h x)
      · exact hp.symm
    · rfl
  simp [kernel, hj, hne]

open Classical in
/-- Without any possible intake, recovery is impossible. -/
lemma recover_eq_zero (h : ∀ x, ¬ M.CanTakeIn x) (k : ℕ) (x : U → Λ) :
    M.recover Wd k x = 0 := by
  induction k generalizing x with
  | zero => rfl
  | succ k ih =>
    have hterm : ∀ w, (if Wd.sense w = M.calm then (0 : ℝ) else
        ∑ y, M.kernel Wd.clock (Wd.sense w) x y *
          (if M.total x < M.total y then 1 else M.recover Wd k y)) = 0 := fun w => by
      split_ifs
      · rfl
      · refine Finset.sum_eq_zero fun y _ => ?_
        by_cases hlt : M.total x < M.total y
        · rw [M.kernel_up_zero h _ _ hlt, zero_mul]
        · rw [ite_eq_right hlt, ih, mul_zero]
    have : Nonempty Wd.W := ⟨Wd.start⟩
    simp only [recover, hterm]
    exact ciSup_const

lemma aliveInd_eq_zero (h : ∀ x, ¬ M.CanTakeIn x) (x : U → Λ) : M.aliveInd Wd x = 0 := by
  have := Wd.recovery_pos
  simp only [aliveInd, M.recover_eq_zero Wd h]
  rw [ite_eq_right_iff]
  intro h'
  linarith [h'.2]

/-- Probability that the organism is alive at time `n` under schedule `sched`. -/
noncomputable def aliveProb (sched : ℕ → Wd.W → U → I) (n : ℕ) : ℝ :=
  ∑ x, ∑ w, M.aliveInd Wd x * M.law Wd sched n x w

omit [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A] in
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

omit [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A] in
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

omit [DecidableEq Λ] in
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
        M.kernel Wd.clock (sched n w) x y

lemma lawCut_sum (sched : ℕ → Wd.W → U → I) (n : ℕ) :
    ∑ x, ∑ w, M.lawCut Wd sched n x w = 1 := by
  induction n with
  | zero =>
    simp only [lawCut]
    rw [Finset.sum_eq_single M.init (fun x _ hx => by simp [hx]) (by simp)]
    simp
  | succ n ih =>
    have h := stepSum (fun _ => (1 : ℝ)) (M.lawCut Wd sched n)
      (cutMove Wd (M.lawCut Wd sched n)) (fun w => M.kernel Wd.clock (sched n w))
      (cutMove_sum Wd ih) (fun w x => by simp [M.kernel_sum])
    simp only [one_mul] at h
    simp only [lawCut]
    rw [h, ih]

/-- Probability of being alive at time `n` in the loop-cut world. -/
noncomputable def aliveProbCut (sched : ℕ → Wd.W → U → I) (n : ℕ) : ℝ :=
  ∑ x, ∑ w, M.aliveInd Wd x * M.lawCut Wd sched n x w

/-- An organism that can never take anything in is never alive, whatever it senses. -/
theorem never_alive_without_intake (h : ∀ x, ¬ M.CanTakeIn x) (sched : ℕ → Wd.W → U → I)
    (n : ℕ) : M.aliveProb Wd sched n = 0 := by
  simp [aliveProb, M.aliveInd_eq_zero Wd h]

/-- An interruption: from step `n₀`, for `T` steps, the organism senses only the calm
environment (anaesthesia, starvation); otherwise it senses the world. -/
def interrupt (n₀ T : ℕ) : ℕ → Wd.W → U → I :=
  fun k w => if n₀ ≤ k ∧ k < n₀ + T then M.calm else Wd.sense w

/-- **L6, self-defined viability**: alive with probability at least `1 - δ` (`δ < 1`) at
every time in the world; still so through any interruption of `T ≥ 1` steps (homeostasis:
pushed to the cliff edge, not over it); past some critical interruption `T* > T`, at any
time, it stays dead even when the world returns (the critical point); and permanently cut
off from the world it is dead from some time on (it needs negentropy). -/
def Persists (δ : ℝ) (T : ℕ) : Prop :=
  δ < 1 ∧ 0 < T ∧ (∀ n, 1 - δ ≤ M.aliveProb Wd (fun _ => Wd.sense) n) ∧
    (∀ n₀ n, 1 - δ ≤ M.aliveProb Wd (M.interrupt Wd n₀ T) n) ∧
    (∃ Tc, T < Tc ∧ ∀ n₀ n, n₀ + Tc ≤ n → M.aliveProb Wd (M.interrupt Wd n₀ Tc) n < 1 - δ) ∧
    ∃ n₀, ∀ n, n₀ ≤ n → M.aliveProb Wd (fun _ _ => M.calm) n < 1 - δ

/-- Without any possible intake there is no persistence. -/
theorem not_persists_of_no_intake (h : ∀ x, ¬ M.CanTakeIn x) (δ : ℝ) (T : ℕ) :
    ¬ M.Persists Wd δ T := by
  rintro ⟨hδ, -, hp, -⟩
  have := hp 0
  rw [M.never_alive_without_intake Wd h] at this
  linarith

/-- Law at time `n₀ + m` of (organism state, world state), tagged with the organism state
`x₀` and world state `w₀` at time `n₀`. At step `n₀` the organism senses `w₀`; afterwards
it senses only the world's `quiet` stimulus. -/
noncomputable def memLaw (n₀ : ℕ) : ℕ → (U → Λ) → Wd.W → (U → Λ) → Wd.W → ℝ
  | 0 => fun x w x₀ w₀ => if x = x₀ ∧ w = w₀ then M.law Wd (fun _ => Wd.sense) n₀ x w else 0
  | m + 1 => fun y w' x₀ w₀ =>
      ∑ x, ∑ w, memLaw n₀ m x w x₀ w₀ * Wd.moves w x w' *
        M.kernel Wd.clock (if m = 0 then Wd.sense w else Wd.quiet) x y

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
  have hK : ∀ w, M.kernel Wd.clock (Wd.sense w) = M.kernel Wd.clock Wd.quiet := fun w => by
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
  have hno : ∀ x, ¬ M.CanTakeIn x := fun x ⟨a, s, hr, hlt⟩ => by
    rw [h a s x hr] at hlt
    exact lt_irrefl _ hlt
  have hcut : ∀ n, M.aliveProbCut Wd (fun _ => Wd.sense) n = 0 := fun n => by
    simp [aliveProbCut, M.aliveInd_eq_zero Wd hno]
  rintro ⟨n₀, ε, hε, hn⟩
  have := hn n₀ le_rfl
  rw [hcut, M.never_alive_without_intake Wd hno] at this
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
structure Lives (ρ δ : ℝ) (κ T : ℕ) : Prop where
  axioms : M.Axioms
  clocked : M.Clocked Wd.clock
  bottleneck : M.rho ≤ ρ
  body : M.ReadsBoundary Wd
  rich : κ * Fintype.card (U → Λ) ≤ Fintype.card Wd.W
  persists : M.Persists Wd δ T
  selfMaintains : M.SelfMaintains Wd
  remembers : M.Remembers Wd

/-- A living organism necessarily senses its world. -/
theorem Lives.exposed {ρ δ : ℝ} {κ T : ℕ} (h : M.Lives Wd ρ δ κ T) : M.Exposed :=
  by_contra fun hn => M.not_selfMaintains_of_unexposed Wd hn h.selfMaintains

/-- A living organism necessarily exchanges content with its world (metabolism). -/
theorem Lives.metabolizes {ρ δ : ℝ} {κ T : ℕ} (h : M.Lives Wd ρ δ κ T) :
    ∃ a s x, 0 < M.rate a s x ∧ M.total (M.act a s x) ≠ M.total x := by
  by_contra hn
  exact M.not_selfMaintains_of_closedMatter Wd
    (fun a s x hp => by_contra fun hne => hn ⟨a, s, x, hp, hne⟩) h.selfMaintains

end Medium

/-- **Main existence conjecture** (statement only): in the world `Wd`, for a body of units
`U` with local states `Λ`, some rulebook with bottleneck at most `ρ` lives. -/
def LifeConjecture {U Λ I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ]
    [Fintype I] (Wd : World U Λ I) (ρ δ : ℝ) (κ T : ℕ) : Prop :=
  ∃ (A : Type) (_ : Fintype A) (M : Medium U Λ A I), M.Lives Wd ρ δ κ T

end CGB
