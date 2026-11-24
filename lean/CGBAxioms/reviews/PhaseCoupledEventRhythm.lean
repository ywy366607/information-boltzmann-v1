import Init

/-! Formal guarantees for the Phase-Coupled Event-Driven Multi-Rate Rhythm.

  The oscillator and consolidation pressure are driven by real bodily
  observables (BFSS momentum activity, prediction surprise, self-chosen
  internal steps), never a fixed `step % N` counter.

  This module proves six structural invariants:
  1. Immediate (0-step) preemption of Theta deliberation and SWR replay on surprise.
  2. Immediate (0-step) preemption of Theta deliberation on self-chosen internal steps.
  3. Bounded liveness (no starvation) for Theta phase windows under a strict
     positive floor `omega_min > 0`.
  4. Bounded liveness (no starvation) for hippocampal SWR consolidation under
     a strict positive pressure floor `alpha_min > 0`.
  5. Exact singleton REINFORCE score identity for Gamma-flow focused speech
     (`proposal_budget = 1`), eliminating redundant `imagine()` evaluation.
  6. Bounded H-JEPA theme-hold horizon invariant (`held ≤ max_hold`).
-/
namespace CGB.PhaseCoupledEventRhythm

structure GateInputs where
  thetaWrapped : Bool
  surpriseExceeded : Bool
  boundaryInternalSteps : Nat
  internalThreshold : Nat
  swrPressureReached : Bool
  swrCoupledReady : Bool
  firstCycle : Bool
  deriving DecidableEq, Repr

def thetaOpen (g : GateInputs) : Bool :=
  g.firstCycle || g.surpriseExceeded || (g.boundaryInternalSteps ≥ g.internalThreshold) || g.thetaWrapped

def swrOpen (g : GateInputs) : Bool :=
  g.surpriseExceeded || g.swrPressureReached || (thetaOpen g && g.swrCoupledReady)

theorem surprise_preempts_theta_and_swr (g : GateInputs)
    (h : g.surpriseExceeded = true) :
    thetaOpen g = true ∧ swrOpen g = true := by
  unfold thetaOpen swrOpen
  rw [h]
  simp

theorem internal_step_preempts_theta (g : GateInputs)
    (h : g.boundaryInternalSteps ≥ g.internalThreshold) :
    thetaOpen g = true := by
  unfold thetaOpen
  have hd : decide (g.boundaryInternalSteps ≥ g.internalThreshold) = true := decide_eq_true h
  simp [hd]

/-- Cumulative phase growth over `n` calm steps with floor `omega_min`. -/
def advanceCumulative (phase : Nat) (incs : List Nat) : Nat :=
  incs.foldl (· + ·) phase

theorem advanceCumulative_ge_floor (phase omegaMin : Nat) (incs : List Nat)
    (hfloor : ∀ d ∈ incs, omegaMin ≤ d) :
    phase + incs.length * omegaMin ≤ advanceCumulative phase incs := by
  induction incs generalizing phase with
  | nil =>
      simp [advanceCumulative]
  | cons d rest ih =>
      simp only [advanceCumulative, List.foldl_cons, List.length_cons, Nat.succ_mul]
      have hd : omegaMin ≤ d := hfloor d (by simp)
      have hrest : ∀ x ∈ rest, omegaMin ≤ x := fun x hx => hfloor x (by simp [hx])
      have hsub := ih (phase + d) hrest
      simp only [advanceCumulative] at hsub
      omega

theorem theta_bounded_liveness (phase period omegaMin : Nat) (incs : List Nat)
    (hfloor : ∀ d ∈ incs, omegaMin ≤ d)
    (hspan : period ≤ incs.length * omegaMin) :
    period ≤ advanceCumulative phase incs := by
  have hgrow := advanceCumulative_ge_floor phase omegaMin incs hfloor
  omega

theorem swr_bounded_liveness (pressure cap alphaMin : Nat) (incs : List Nat)
    (hfloor : ∀ d ∈ incs, alphaMin ≤ d)
    (hspan : cap ≤ incs.length * alphaMin) :
    cap ≤ advanceCumulative pressure incs := by
  have hgrow := advanceCumulative_ge_floor pressure alphaMin incs hfloor
  omega

/-- Exact singleton proposal REINFORCE score identity (in integer/rational model). -/
def proposalLogProb (choiceLogp : Int) (proposalSum : Int) (selectionLogp : Int) : Int :=
  choiceLogp + proposalSum + selectionLogp

theorem gamma_single_proposal_selection_identity (choiceLogp proposalLogp : Int) :
    proposalLogProb choiceLogp proposalLogp 0 = choiceLogp + proposalLogp := by
  unfold proposalLogProb
  omega

/-- H-JEPA theme-hold state machine preserves `held ≤ maxHold`. -/
def nextHeld (held maxHold : Nat) (canHold : Bool) : Nat :=
  if canHold && decide (held < maxHold) then held + 1 else 0

theorem theme_hold_bounded_horizon (held maxHold : Nat) (canHold : Bool) :
    nextHeld held maxHold canHold ≤ maxHold := by
  unfold nextHeld
  by_cases hc : canHold = true
  · by_cases hlt : held < maxHold
    · simp [hc, hlt]
      omega
    · simp [hc, hlt]
  · simp [hc]

theorem theme_hold_forces_revision_at_cap (maxHold : Nat) (canHold : Bool) :
    nextHeld maxHold maxHold canHold = 0 := by
  unfold nextHeld
  simp

end CGB.PhaseCoupledEventRhythm
