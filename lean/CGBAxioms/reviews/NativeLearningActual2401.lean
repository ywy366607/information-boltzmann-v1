import Mathlib.Tactic
import Mathlib.Data.Fintype.Sum

/-!
Current-body learning projection, not a new optimizer or a capability model.
Sources: adaptive_hamiltonian.StateBridge/MomentumReadBridge/consume,
language_navigation_v2.Engine._advance, budgeted_adamw.BudgetedAdamW.step,
generative_relations.GenerativeRelationBank.commit and shared_unknown_bindings.commit.
Arbitrary finite coordinates stand for real tensor entries. Actual layouts are
32 layers, 12 local dynamics parameters, 8192 S coordinates and 256 P-read
coordinates per layer. Floating-point execution is checked separately.
-/
namespace CGB.NativeLearning

noncomputable section

def credit {P X : Type*} [Fintype X]
    (eligibility : P → X → ℝ) (error : X → ℝ) : P → ℝ :=
  fun p => ∑ x, eligibility p x * error x

/-- The saved read trace's actual [parameter=12,layer=32,read=256] layout. -/
def blockReadEligibility (e : Fin 12 → Fin 32 → Fin 256 → ℝ)
    (p : Fin 32 × Fin 12) (x : Fin 32 × Fin 256) : ℝ :=
  if p.1 = x.1 then e p.2 p.1 x.2 else 0

theorem saved_read_layout_contracts_per_layer
    (e : Fin 12 → Fin 32 → Fin 256 → ℝ) (error : (Fin 32 × Fin 256) → ℝ)
    (layer : Fin 32) (parameter : Fin 12) :
    credit (blockReadEligibility e) error (layer, parameter) =
      ∑ coordinate, e parameter layer coordinate * error (layer, coordinate) := by
  classical
  simp [credit, blockReadEligibility, Fintype.sum_prod_type]

def transportedEligibility {P X Y : Type*} [Fintype X]
    (jacobian : Y → X → ℝ) (e : P → X → ℝ)
    (injection : P → Y → ℝ) : P → Y → ℝ :=
  fun p y => (∑ x, jacobian y x * e p x) + injection p y

def transportedError {X Y : Type*} [Fintype Y]
    (jacobian : Y → X → ℝ) (error : Y → ℝ) : X → ℝ :=
  fun x => ∑ y, jacobian y x * error y

/-- The JVP eligibility and later error contract to the same chain-rule term.
The Jacobian is of the actual local transition; independence is not assumed. -/
theorem transported_credit {P X Y : Type*} [Fintype X] [Fintype Y]
    (j : Y → X → ℝ) (e : P → X → ℝ) (injection : P → Y → ℝ)
    (error : Y → ℝ) (p : P) :
    credit (transportedEligibility j e injection) error p =
      credit e (transportedError j error) p + credit injection error p := by
  classical
  unfold credit transportedEligibility transportedError
  simp_rw [add_mul, Finset.sum_add_distrib, Finset.sum_mul, Finset.mul_sum]
  congr 1
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro x hx
  apply Finset.sum_congr rfl
  intro y hy
  ring

def combinedGradient {I : Type*}
    (instant currentRead delayedRead actor : I → ℝ) (advantage : ℝ) : I → ℝ :=
  fun i => instant i + currentRead i + delayedRead i - advantage * actor i

structure TraceState (P X R : Type*) where
  sensitivityS : P → X → ℝ
  sensitivityP : P → X → ℝ
  pendingRead : P → R → ℝ

/-- consume() clears only solver sensitivities. The pending read is a clone
stored in internal_pending, not an alias of these buffers. -/
def consumeTrace {P X R : Type*} (s : TraceState P X R) : TraceState P X R :=
  { s with sensitivityS := fun _ _ => 0, sensitivityP := fun _ _ => 0 }

theorem consume_preserves_delayed_credit {P X R : Type*} [Fintype R]
    (s : TraceState P X R) (laterError : R → ℝ) :
    credit (consumeTrace s).pendingRead laterError = credit s.pendingRead laterError := rfl

structure AdamCoordinate where
  weight : ℝ
  first : ℝ
  second : ℝ
  count : ℕ

def nextFirst (beta old gradient : ℝ) : ℝ := beta * old + (1 - beta) * gradient
def nextSecond (beta old gradient : ℝ) : ℝ := beta * old + (1 - beta) * gradient ^ 2

def adamProposal (first second beta1 beta2 eps : ℝ) (count : ℕ) : ℝ :=
  (first / (1 - beta1 ^ count)) / (Real.sqrt (second / (1 - beta2 ^ count)) + eps)

/-- allocation is the actual broadcast cell factor; gate multiplies decay too. -/
def executedWrite (weight proposal lr decay gate allocation : ℝ) : ℝ :=
  weight - lr * gate * (allocation * proposal + decay * weight)

/-- Same zero-initialized finite-sample EMA noise mass as noise_weight(). -/
def noiseMass (beta : ℝ) (count : ℕ) : ℝ :=
  (1 - beta) / (1 + beta) * (1 + beta ^ count) / (1 - beta ^ count)

def clampUnit (x : ℝ) : ℝ := max 0 (min 1 x)

/-- The finite-precision floor and neutral cold-start branch are explicit.
The 1e-12 tolerance is the Python branch, not a theoretical noise threshold. -/
def coherenceGate (signal energy noise tiny : ℝ) : ℝ :=
  if noise ≥ 1 - (1 : ℝ) / 1000000000000 then 1 else
    if energy > 0 then
      clampUnit ((signal - noise * energy) / max tiny ((1 - noise) * energy)) else 0

def recentRmsUpper (firstPower priorRms beta : ℝ) (count : ℕ) : ℝ :=
  (Real.sqrt firstPower + beta ^ count * priorRms) / (1 - beta ^ count)

def allocationPhase (old cue beta : ℝ) : ℝ :=
  max (-1) (min 1 (beta * old + (1 - beta) * cue))

/-- Four physical cells of a real 64x64 parameter tile. Non-tiled parameters
use allocation=1; these are compute cells, not four synthetic agents. -/
def cellAllocation (phase : ℝ) (i j : Fin 2) : ℝ :=
  if i = j then 1 + phase else 1 - phase

theorem coherence_gate_bounds (signal energy noise tiny : ℝ) :
    0 ≤ coherenceGate signal energy noise tiny ∧
      coherenceGate signal energy noise tiny ≤ 1 := by
  unfold coherenceGate clampUnit
  split_ifs <;> constructor <;> first | norm_num | exact le_max_left _ _ |
    exact max_le (by norm_num) (min_le_left _ _)

theorem allocation_row_budget (phase : ℝ) (i : Fin 2) :
    cellAllocation phase i 0 + cellAllocation phase i 1 = 2 := by
  fin_cases i <;> norm_num [cellAllocation]

theorem allocation_column_budget (phase : ℝ) (j : Fin 2) :
    cellAllocation phase 0 j + cellAllocation phase 1 j = 2 := by
  fin_cases j <;> norm_num [cellAllocation]

def adamCoordinateStep (old : AdamCoordinate) (gradient beta1 beta2 eps lr decay
    gate allocation : ℝ) : AdamCoordinate :=
  let m := nextFirst beta1 old.first gradient
  let v := nextSecond beta2 old.second gradient
  let n := old.count + 1
  ⟨executedWrite old.weight (adamProposal m v beta1 beta2 eps n) lr decay gate allocation,
   m, v, n⟩

theorem closed_write_still_updates_observer (old : AdamCoordinate)
    (gradient beta1 beta2 eps lr decay allocation : ℝ) :
    let new := adamCoordinateStep old gradient beta1 beta2 eps lr decay 0 allocation
    new.weight = old.weight ∧ new.first = nextFirst beta1 old.first gradient ∧
      new.second = nextSecond beta2 old.second gradient ∧ new.count = old.count + 1 := by
  simp [adamCoordinateStep, executedWrite]

/-- A regulator of the actual write changes the proposal's effect. This
identity does not assert that the full body is the paper's percolation model. -/
theorem regulator_proposal_mixed_difference (w d lr decay k a dd dk : ℝ) :
    executedWrite w (d + dd) lr decay (k + dk) a -
      executedWrite w (d + dd) lr decay k a -
      executedWrite w d lr decay (k + dk) a + executedWrite w d lr decay k a =
      -lr * a * dd * dk := by
  unfold executedWrite
  ring

structure Scope where
  identity : ℕ
  generation : ℕ
  version : ℕ
  deriving DecidableEq, Repr

structure Admission where
  observed : Bool
  encoded : Bool
  occupied : Bool
  episodeCount : ℕ
  actualStamp : ℕ
  claimedStamp : ℕ
  motherValid : Bool
  motherStamp : ℕ
  deriving Repr

/-- Actual primary commit gate. Split/share plans may subsequently update
other scopes; no false assertion of a single-slot total write is made. -/
def canCommit (current expected : Scope) (a : Admission) : Bool :=
  a.observed && !a.encoded && decide (current = expected) && a.occupied &&
  a.episodeCount == 1 && a.actualStamp == a.claimedStamp && a.motherValid &&
  a.motherStamp == a.claimedStamp

theorem hypothetical_not_committed (current expected : Scope) (a : Admission)
    (h : a.observed = false) : canCommit current expected a = false := by
  simp [canCommit, h]

theorem merged_episode_not_individually_committed (current expected : Scope) (a : Admission)
    (h : a.episodeCount ≠ 1) : canCommit current expected a = false := by
  simp [canCommit, h]

def commitUnknownVersion (primaryCommit : Bool) (version : ℕ) : ℕ :=
  if primaryCommit then version + 1 else version

theorem rejected_primary_keeps_unknown_version (version : ℕ) :
    commitUnknownVersion false version = version := rfl

/-- The existing table logit-probe settlement uses this version/occupancy gate. -/
def readParticipationValid (recordValid occupied : Bool) (saved current : ℕ) : Bool :=
  recordValid && decide (saved = current) && occupied

theorem recycled_slot_rejects_old_credit (valid occupied : Bool) (saved current : ℕ)
    (h : saved ≠ current) : readParticipationValid valid occupied saved current = false := by
  simp [readParticipationValid, h]

structure LearningReceipt where
  event : ℕ
  merged : Bool
  relationVersionBefore : ℕ
  relationVersionAfter : ℕ
  primaryCommitted : Bool
  unknownCommitted : Bool
  unknownVersionBefore : ℕ
  unknownVersionAfter : Option ℕ
  currentReadCredits : ℕ
  delayedReadCredits : ℕ
  deriving Repr

/-- Check the delivered primary/unknown commit correspondence; do not confuse
the functional revise() calls with persistence. Delayed credits are retained
as actual counts, not inferred from successful commits. -/
def learningReceiptValid (r : LearningReceipt) : Bool :=
  (if r.primaryCommitted then !r.merged &&
    r.relationVersionAfter == r.relationVersionBefore + 1 else true) &&
  r.unknownCommitted == r.primaryCommitted &&
  (if r.primaryCommitted then r.unknownVersionAfter == some (r.unknownVersionBefore + 1)
   else r.unknownVersionAfter == none)

end
end CGB.NativeLearning


namespace CGB.NativeLearning.Actual2401
set_option linter.style.longLine false
def receipts : List LearningReceipt := [
⟨2824, false, 36, 37, true, true, 0, some 1, 1, 1⟩,
⟨2828, false, 29, 30, true, true, 0, some 1, 1, 1⟩,
⟨2832, false, 14, 15, true, true, 0, some 1, 1, 1⟩,
⟨2834, false, 38, 39, true, true, 0, some 1, 1, 1⟩,
⟨2836, false, 8, 9, true, true, 0, some 1, 1, 1⟩,
⟨2840, false, 54, 55, true, true, 4, some 5, 1, 1⟩,
⟨2842, false, 7, 8, true, true, 0, some 1, 1, 1⟩,
⟨2849, false, 40, 41, true, true, 0, some 1, 1, 1⟩,
⟨2851, false, 52, 53, true, true, 0, some 1, 1, 1⟩,
⟨2854, false, 36, 37, true, true, 0, some 1, 1, 1⟩,
⟨2857, false, 54, 55, true, true, 0, some 1, 1, 1⟩,
⟨2863, false, 18, 19, true, true, 0, some 1, 1, 1⟩,
⟨2865, false, 15, 16, true, true, 0, some 1, 1, 1⟩,
⟨2867, false, 14, 15, true, true, 0, some 1, 1, 1⟩,
⟨2869, false, 9, 10, true, true, 0, some 1, 1, 1⟩,
⟨2871, false, 42, 43, true, true, 0, some 1, 1, 1⟩,
⟨2873, false, 0, 1, true, true, 2, some 3, 1, 1⟩,
⟨2877, false, 38, 39, true, true, 0, some 1, 1, 1⟩,
⟨2879, true, 173, 173, false, false, 0, none, 1, 1⟩,
⟨2881, true, 44, 44, false, false, 3, none, 1, 0⟩,
⟨2883, false, 58, 59, true, true, 0, some 1, 1, 0⟩,
⟨2887, false, 59, 60, true, true, 0, some 1, 1, 1⟩,
⟨2889, false, 56, 57, true, true, 0, some 1, 1, 1⟩,
⟨2892, true, 173, 173, false, false, 0, none, 1, 1⟩,
⟨2896, false, 6, 7, true, true, 0, some 1, 1, 0⟩,
⟨2899, false, 10, 11, true, true, 0, some 1, 1, 1⟩,
⟨2901, false, 11, 12, true, true, 0, some 1, 1, 1⟩,
⟨2903, true, 39, 39, false, false, 2, none, 1, 1⟩,
⟨2905, false, 20, 21, true, true, 1, some 2, 1, 0⟩,
⟨2907, true, 173, 173, false, false, 1, none, 1, 1⟩,
⟨2909, false, 3, 4, true, true, 0, some 1, 1, 0⟩,
⟨2914, false, 22, 23, true, true, 0, some 1, 1, 1⟩,
⟨2916, false, 18, 19, true, true, 0, some 1, 1, 1⟩,
⟨2919, false, 12, 13, true, true, 0, some 1, 1, 1⟩,
⟨2921, false, 4, 5, true, true, 0, some 1, 1, 1⟩,
⟨2923, false, 62, 63, true, true, 0, some 1, 1, 1⟩,
⟨2925, false, 4, 5, true, true, 0, some 1, 1, 1⟩,
⟨2927, false, 44, 45, true, true, 0, some 1, 1, 1⟩,
⟨2929, false, 13, 14, true, true, 0, some 1, 1, 1⟩,
⟨2931, false, 6, 7, true, true, 1, some 2, 1, 1⟩,
⟨2936, false, 47, 48, true, true, 1, some 2, 1, 1⟩]
theorem actual_learning_receipts_valid :
    receipts.all learningReceiptValid = true := by native_decide
end CGB.NativeLearning.Actual2401
