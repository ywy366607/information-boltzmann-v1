import Mathlib.Tactic

/-! Body binding: the complete state includes ExternalTask's schedule stage,
S/P, parameters, optimizer, relation/pending identities and world position.
Q advances every non-arrival stage; R reaches the next actual world_arrived
handoff. Clock is -d/d eta of exp(-eta * dispatch_seconds); credit is a
declared directional derivative; mixed is -d/d eta d/d delta. Operations
are ordered kernel composition, not five nodes assigned to five phases.
The old prefix probability/tangent identities are reused separately.
-/
namespace CGB.NativeClockCreditComposition

structure Jet (R : Type*) where
  mass : R
  clock : R
  credit : R
  mixed : R

def compose {R : Type*} [Ring R] (a b : Jet R) : Jet R where
  mass := a.mass*b.mass
  clock := a.clock*b.mass+a.mass*b.clock
  credit := a.credit*b.mass+a.mass*b.credit
  mixed := a.mixed*b.mass+a.credit*b.clock+a.clock*b.credit+a.mass*b.mixed

/-- Works for noncommuting operators, including finite matrix bindings. -/
theorem compose_assoc {R : Type*} [Ring R] (a b c : Jet R) :
    compose (compose a b) c = compose a (compose b c) := by
  cases a
  cases b
  cases c
  unfold compose
  congr 1 <;> noncomm_ring

/-- The mixed term includes both cross-stage paths. -/
theorem mixed_cross_paths {R : Type*} [Ring R] (a b : Jet R) :
    (compose a b).mixed-(a.mixed*b.mass+a.mass*b.mixed) =
      a.credit*b.clock+a.clock*b.credit := by
  simp only [compose]
  noncomm_ring

def mark (p t dp dt : ℝ) : Jet ℝ where
  mass := p
  clock := p*t
  credit := dp
  mixed := dp*t+p*dt

theorem marked_clock_adds (p q t u dp dq : ℝ) :
    (compose (mark p t dp 0) (mark q u dq 0)).clock = p*q*(t+u) := by
  simp only [compose, mark]
  ring

theorem marked_credit_clock (p q t u dp dq : ℝ) :
    (compose (mark p t dp 0) (mark q u dq 0)).mixed =
      (dp*q+p*dq)*(t+u) := by
  simp only [compose, mark]
  ring

/-- Equal endpoint weights can carry distinct clock observables. -/
theorem endpoint_clock_difference (p t u dp : ℝ) :
    (mark p t dp 0).clock-(mark p u dp 0).clock = p*(t-u) := by
  simp only [mark]
  ring

end CGB.NativeClockCreditComposition
