import CGBAxioms.InformationEconomics
import CGBAxioms.GenerationColumnMemory

/-!
# Column refinement: one body, several readable projections

This file does not construct a column organism.  It records the proof
obligation an implementation must meet before it can call its activity,
causal memory, contact state, eligibility trace, and local demand one CGB
body.  All of them are functions of the same `Medium` state.  In particular,
there is no separately advanced trace page or global learning-rate page.

The interface deliberately stops before declaring an arbitrary contact value
to be a conserved, observable transfer.  `LanguageLifeKernel` proves why that
typing and observability must be exhibited by the concrete medium rather than
inferred from total conservation alone.
-/

namespace CGB.Generation

open CGB.InformationEconomics

abbrev Body (U Λ : Type*) := U → Λ

/-- Site and edge projections which read, but do not supplement, one body. -/
structure LocalColumnProjections {U Λ A I H C S : Type*}
    (M : Medium U Λ A I) where
  activity : Body U Λ → U → H
  carrier : Body U Λ → U → C
  contact : Body U Λ → U → U → S
  activity_local : ∀ {x y u}, x u = y u → activity x u = activity y u
  carrier_local : ∀ {x y u}, x u = y u → carrier x u = carrier y u
  contact_local : ∀ {x y u v}, x u = y u → x v = y v →
    contact x u v = contact y u v

/-- A causal state is valid only when the actual body update realizes both its
required readout and its recursive successor.  The event selector reads the
already encoded body and current arrival, so it cannot be a hidden history
store. -/
structure BodyCausalCarrier {U Λ A I History Arrival Readout Carrier : Type*}
    (M : Medium U Λ A I) where
  encode : History → Body U Λ
  advance : History → Arrival → History
  stimulus : Arrival → U → I
  event : Body U Λ → Arrival → A
  state : Body U Λ → Carrier
  required : History → Readout
  read : Carrier → Readout
  next : Carrier → Arrival → Carrier
  body_step : ∀ h a,
    encode (advance h a) = M.act (event (encode h) a) (stimulus a) (encode h)
  read_correct : ∀ h, read (state (encode h)) = required h
  next_correct : ∀ h a,
    next (state (encode h)) a = state (encode (advance h a))

/-- The minimum common-state contract for a column medium.  `trace` and
`demand` are reconstructed from already present carrier/contact readings;
they therefore cannot secretly introduce a fourth memory subsystem. -/
structure ColumnRefinement {U Λ A I H C S History Arrival Readout Carrier Trace Demand : Type*}
    (M : Medium U Λ A I) where
  projections : LocalColumnProjections (H := H) (C := C) (S := S) M
  causal : BodyCausalCarrier (History := History) (Arrival := Arrival)
    (Readout := Readout) (Carrier := Carrier) M
  assemble : (U → C) → Carrier
  carrier_from_local : ∀ x,
    causal.state x = assemble (fun u => projections.carrier x u)
  trace : Body U Λ → U → U → Trace
  trace_read : C → S → Trace
  trace_from_local : ∀ x u v,
    trace x u v = trace_read (projections.carrier x u) (projections.contact x u v)
  demand : Body U Λ → U → Demand
  demand_read : C → Demand
  demand_from_local : ∀ x u,
    demand x u = demand_read (projections.carrier x u)

namespace LocalColumnProjections

variable {U Λ A I H C S : Type*}
  {M : Medium U Λ A I} (P : LocalColumnProjections (H := H) (C := C) (S := S) M)

/-- P1b makes a nonparticipant's activity readout unchanged under the same
physical event.  No independent activity update is permitted. -/
theorem activity_unchanged_outside {a : A} {s : U → I} {x : Body U Λ} {u : U}
    (hp1b : M.P1b) (hu : u ∉ M.parts a) :
    P.activity (M.act a s x) u = P.activity x u :=
  P.activity_local (hp1b.2.1 a s x u hu)

/-- The same event leaves a nonparticipant's local causal reading unchanged. -/
theorem carrier_unchanged_outside {a : A} {s : U → I} {x : Body U Λ} {u : U}
    (hp1b : M.P1b) (hu : u ∉ M.parts a) :
    P.carrier (M.act a s x) u = P.carrier x u :=
  P.carrier_local (hp1b.2.1 a s x u hu)

/-- A contact reading changes only when at least one endpoint participates. -/
theorem contact_unchanged_outside {a : A} {s : U → I} {x : Body U Λ} {u v : U}
    (hp1b : M.P1b) (hu : u ∉ M.parts a) (hv : v ∉ M.parts a) :
    P.contact (M.act a s x) u v = P.contact x u v :=
  P.contact_local (hp1b.2.1 a s x u hu) (hp1b.2.1 a s x v hv)

end LocalColumnProjections

namespace BodyCausalCarrier

variable {U Λ A I History Arrival Readout Carrier : Type*}
  {M : Medium U Λ A I}
  (K : BodyCausalCarrier (History := History) (Arrival := Arrival)
    (Readout := Readout) (Carrier := Carrier) M)

/-- The carrier fields meet the existing exact readout-and-successor contract.
This is a use of the actual body step, not a second recurrence. -/
theorem satisfies_causal_carrier_contract :
    ∃ read : Carrier → Readout, ∃ next : Carrier → Arrival → Carrier,
      (∀ h, read (K.state (K.encode h)) = K.required h) ∧
      (∀ h a, next (K.state (K.encode h)) a =
        K.state (K.encode (K.advance h a))) :=
  ⟨K.read, K.next, K.read_correct, K.next_correct⟩

/-- Equivalently, any two histories merged by the carrier agree on every
declared readout and on the carrier successor for each arrival. -/
theorem merged_history_preserves_read_and_successor :
    (∀ h h', K.state (K.encode h) = K.state (K.encode h') →
      K.required h = K.required h') ∧
    (∀ h h', K.state (K.encode h) = K.state (K.encode h') → ∀ a,
      K.state (K.encode (K.advance h a)) =
        K.state (K.encode (K.advance h' a))) := by
  constructor
  · intro h h' he
    rw [← K.read_correct h, ← K.read_correct h', he]
  · intro h h' he a
    rw [← K.next_correct h a, ← K.next_correct h' a, he]

end BodyCausalCarrier

end CGB.Generation
