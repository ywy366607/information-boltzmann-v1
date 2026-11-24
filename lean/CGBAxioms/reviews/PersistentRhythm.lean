import Mathlib.Tactic

/-! Body binding (2026-11-20).

`persistent_rhythm.ExternalTask` runs the ORIGINAL external step on the same
thread. Its yield points are `world_arrived`, and RETURNS of the original
`_advance` (after backward, optimizer, recurrent S and captured P commit).
The latter is where the original autonomous selector may advance quiet S/P.
No live graph is moved across an optimizer update. The numerical contracts
exercise this exact production call path and its delayed action eligibility.

The projections below are the carried recurrent matrix, momentum matrix,
parameters, real-memory clock, inbox and pre-action credit receipt. W,Q,R
are the actual packed tensor coordinate counts, not new modeled organs.
`receive` corresponds to world.take's body-independent output, and `quiet`
to the original coupled S/P integrator. An arrival can commute with a quiet
step under these read/write scopes. Learning reads S/P and therefore retains
its actual causal order. We do not assert that quiet and learning commute.

The existing context no-interference result is indexed in
GenerationContextAssociation.disjoint_context_read_unchanged; here its
read/write premise is instantiated explicitly to the inbox/S/P ports.
-/
namespace CGB.PersistentRhythm

structure Boundary (Q W R : Nat) where
  s : Fin Q → ℝ
  p : Fin Q → ℝ
  theta : Fin W → ℝ
  observed : Nat
  receipt : Fin R → ℝ

structure Process (Q W R : Nat) where
  body : Boundary Q W R
  inbox : List Nat
  world : Nat

def quiet {Q W R : Nat} (flow : (Fin Q → ℝ) → (Fin Q → ℝ) →
    ((Fin Q → ℝ) × (Fin Q → ℝ))) (x : Process Q W R) : Process Q W R :=
  {x with body := {x.body with
    s := (flow x.body.s x.body.p).1
    p := (flow x.body.s x.body.p).2}}

def receive {Q W R : Nat} (message : Nat) (x : Process Q W R) : Process Q W R :=
  {x with inbox := x.inbox ++ [message], world := x.world + 1}

theorem receive_quiet_same_ports {Q W R : Nat}
    (flow : (Fin Q → ℝ) → (Fin Q → ℝ) → ((Fin Q → ℝ) × (Fin Q → ℝ)))
    (message : Nat) (x : Process Q W R) :
    receive message (quiet flow x) = quiet flow (receive message x) := by
  rfl

theorem quiet_preserves_real_evidence_and_credit {Q W R : Nat}
    (flow : (Fin Q → ℝ) → (Fin Q → ℝ) → ((Fin Q → ℝ) × (Fin Q → ℝ)))
    (x : Process Q W R) :
    (quiet flow x).body.observed = x.body.observed ∧
    (quiet flow x).body.theta = x.body.theta ∧
    (quiet flow x).body.receipt = x.body.receipt ∧
    (quiet flow x).inbox = x.inbox := by
  exact ⟨rfl, rfl, rfl, rfl⟩

-- A completed _advance publishes the input of the next original objective.
-- A quiet operation at that yield changes exactly that next input.
theorem next_objective_reads_evolved_state {Q W R : Nat}
    (flow : (Fin Q → ℝ) → (Fin Q → ℝ) → ((Fin Q → ℝ) × (Fin Q → ℝ)))
    (objective : (Fin Q → ℝ) → ℝ) (x : Process Q W R) :
    objective (quiet flow x).body.s = objective (flow x.body.s x.body.p).1 := rfl

-- Production poll stores detached selector-score eligibility, then the real
-- later actor factor modulates its SUM once, through add_boundary_credit.
theorem delayed_factor_of_accumulated_scores (eta delta ea eb : ℝ) :
    eta * delta * (ea + eb) = eta * delta * ea + eta * delta * eb := by ring

structure Plan (D : Nat) where
  raw : Fin D → ℝ
  relationIdentity : Nat
  relationVersion : Nat
  unknownIdentity : Nat
  previousValue : ℝ

def readPlan {D W : Nat} (encode : (Fin W → ℝ) → (Fin D → ℝ) → ℝ)
    (theta : Fin W → ℝ) (p : Plan D) : ℝ := encode theta p.raw

theorem read_reencodes_current_parameters {D W : Nat}
    (encode : (Fin W → ℝ) → (Fin D → ℝ) → ℝ)
    (theta : Fin W → ℝ) (p : Plan D) (v : ℝ) :
    readPlan encode theta {p with previousValue := v} = readPlan encode theta p := rfl

-- Pending labels are separate from the frozen fitting bank throughout the
-- choice. Only the original actual-settlement transition publishes them.
structure Bank (A : Type) where
  fitted : List A
  pending : List A
  revision : Nat

def enqueue {A : Type} (b : Bank A) (label : A) : Bank A :=
  {b with pending := label :: b.pending}

theorem pending_label_preserves_choice_bank {A : Type} (b : Bank A) (label : A) :
    (enqueue b label).fitted = b.fitted ∧ (enqueue b label).revision = b.revision :=
  ⟨rfl, rfl⟩

def commit {A : Type} (b : Bank A) : Bank A :=
  ⟨b.fitted ++ b.pending, [], b.revision + 1⟩

theorem settlement_consumes_pending_once {A : Type} (b : Bank A) :
    (commit b).pending = [] ∧ (commit b).revision = b.revision + 1 := ⟨rfl, rfl⟩

end CGB.PersistentRhythm
