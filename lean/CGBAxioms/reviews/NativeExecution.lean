import Mathlib.Tactic

/-!
Abstraction of the *executed* Python program and autonomous dispatch receipts.
Ports are those of causal_relation_program.py, not names for five graph vertices.
Variables are local to one invocation; invocation identity is kept separately.
Tensor coordinates remain opaque: shape checking is not a dynamical equivalence
proof and does not turn a vector into an oscillator phase.
-/
namespace CGB.NativeExecution

inductive Role where
  | condition | action | mother | decision | value
  deriving Inhabited, DecidableEq, Repr

structure Variable where
  identity : String
  role : Role
  deriving Inhabited, DecidableEq, Repr

inductive Operation where
  | motherResponse | ambientTransport | bodyTransition | continuation
  deriving Inhabited, DecidableEq, Repr

def Operation.inputs : Operation → List Role
  | .motherResponse => [.condition, .action]
  | .ambientTransport => [.condition, .action, .mother]
  | .bodyTransition => [.condition, .action]
  | .continuation => [.decision]

def Operation.output : Operation → Role
  | .motherResponse => .mother
  | .ambientTransport | .bodyTransition => .decision
  | .continuation => .value

structure Call where
  operation : Operation
  arguments : List Variable
  result : Variable
  deriving Inhabited, Repr

def callValid (known : List Variable) (c : Call) : Bool :=
  c.arguments.map Variable.role == c.operation.inputs &&
  c.result.role == c.operation.output && c.arguments.all known.contains

/-- Mirrors execute(): existing bindings win over alternative predictions. -/
def extend (known : List Variable) (v : Variable) : List Variable :=
  if known.contains v = true then known else v :: known

def programValid : List Variable → List Call → Bool
  | _, [] => true
  | known, c :: cs => callValid known c && programValid (extend known c.result) cs

theorem bound_result_not_replaced (known : List Variable) (v : Variable)
    (h : known.contains v = true) : extend known v = known := by
  unfold extend
  rw [ite_eq_left h]

structure ValueShape where
  binding : Variable
  dimensions : List Nat
  deriving Inhabited, Repr

def expectedShape (batch : Nat) : Role → List Nat
  | .condition | .action => [batch, 256]
  | .mother | .decision => [batch, 128]
  | .value => [batch]

structure Execution where
  event : Nat
  ordinal : Nat
  batch : Nat
  calls : List Call
  values : List ValueShape
  alternatives : List ValueShape
  deriving Inhabited, Repr

def executionValid (x : Execution) : Bool :=
  let known := x.values.filterMap (fun v =>
    if v.binding.role == .condition || v.binding.role == .action
      then some v.binding else none)
  x.batch > 0 && programValid known x.calls &&
  x.values.all (fun v => v.dimensions == expectedShape x.batch v.binding.role) &&
  x.alternatives.all (fun v => v.dimensions == expectedShape x.batch v.binding.role) &&
  x.alternatives.map ValueShape.binding == x.calls.map Call.result

inductive EventKind where
  | internal | release | external
  deriving Inhabited, DecidableEq, Repr

structure EventReceipt where
  event : Nat
  kind : EventKind
  updates : Nat
  optimizerCalls : Nat
  worldCalls : Nat
  externalBytes : Nat
  newEvidence : Bool
  deriving Inhabited, Repr

/-- Source: autonomous_cycle.dispatch/poll, actual world.take and optimizer.step.
External updates are checked against observed calls, not fixed to four here.
-/
def receiptValid (x : EventReceipt) : Bool :=
  x.updates == x.optimizerCalls && match x.kind with
  | .internal | .release =>
      x.updates == 0 && x.worldCalls == 0 && x.externalBytes == 0 && !x.newEvidence
  | .external => x.worldCalls == 1 && x.newEvidence

/-- The release is a scheduling decision; the *next* dispatch contacts the world. -/
def dispatchValid : Bool → List EventReceipt → Bool
  | _, [] => true
  | ready, x :: xs => receiptValid x && match x.kind with
    | .external => ready && dispatchValid false xs
    | .release => !ready && dispatchValid true xs
    | .internal => !ready && dispatchValid false xs

/-- An observation must factor through the recorded abstraction to be recoverable. -/
def Recoverable {S A Q : Type*} (abstract : S → A) (observable : S → Q) : Prop :=
  ∃ measurement : A → Q, ∀ s, measurement (abstract s) = observable s

theorem recoverable_respects_fibres {S A Q : Type*}
    (abstract : S → A) (observable : S → Q) (h : Recoverable abstract observable)
    {a b : S} (same : abstract a = abstract b) : observable a = observable b := by
  obtain ⟨measurement, correct⟩ := h
  rw [← correct a, ← correct b, same]

end CGB.NativeExecution
