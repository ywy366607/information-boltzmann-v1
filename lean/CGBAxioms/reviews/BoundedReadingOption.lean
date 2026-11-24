import Init

/-! Binding: B=the real carried body, `learn`=one completed objective/_advance,
    α=one actual 480B StreamPiece, j=completed 960B blocks. Python binds these
    to world_learning_cycle, keeps the first action receipt, and appends two
    real pieces only after both current updates finish. No dynamics or fixed
    point assumption about B is made here. -/
namespace CGB.BoundedReadingOption

def learnPieces {B α : Type} (learn : B → α → B) (body : B) (xs : List α) : B :=
  xs.foldl learn body

theorem append_preserves_body {B α : Type} (learn : B → α → B)
    (body : B) (first next : List α) :
    learnPieces learn body (first ++ next) =
    learnPieces learn (learnPieces learn body first) next := by
  simp [learnPieces, List.foldl_append]

def continueReading (j cap : Nat) (surprise changed eof : Bool) : Bool :=
  !surprise && !changed && !eof && decide (j < cap)

theorem surprise_stops (j cap : Nat) (changed eof : Bool) :
    continueReading j cap true changed eof = false := by
  simp [continueReading]

theorem changed_target_stops (j cap : Nat) (surprise eof : Bool) :
    continueReading j cap surprise true eof = false := by
  simp [continueReading]

theorem cap_stops (cap : Nat) (surprise changed eof : Bool) :
    continueReading cap cap surprise changed eof = false := by
  simp [continueReading]

theorem admitted_next_block_bounded (j cap : Nat) (s c e : Bool)
    (h : continueReading j cap s c e = true) : j + 1 ≤ cap := by
  simp [continueReading, Bool.and_eq_true] at h
  omega

structure Receipt where
  blocks : Nat
  actorOwners : Nat
  settlements : Nat
  updates : Nat
  bytes : Nat
  deriving Repr, DecidableEq

def openAction : Receipt := ⟨0, 1, 0, 0, 0⟩
def finishBlock (r : Receipt) : Receipt :=
  { r with blocks := r.blocks+1, updates := r.updates+2, bytes := r.bytes+960 }
def closeAction (r : Receipt) : Receipt :=
  if r.settlements = 0 then { r with settlements := 1 } else r

theorem block_preserves_owner (r : Receipt) :
    (finishBlock r).actorOwners = r.actorOwners ∧
    (finishBlock r).settlements = r.settlements := by simp [finishBlock]

theorem block_accounting (r : Receipt)
    (hu : r.updates = 2*r.blocks) (hb : r.bytes = 960*r.blocks) :
    (finishBlock r).updates = 2*(finishBlock r).blocks ∧
    (finishBlock r).bytes = 960*(finishBlock r).blocks := by
  simp [finishBlock] at *
  omega

theorem one_terminal_settlement (r : Receipt) (h : r.settlements = 0) :
    (closeAction r).settlements = 1 := by simp [closeAction, h]

theorem repeated_close_pays_once (r : Receipt) :
    closeAction (closeAction r) = closeAction r := by
  by_cases h : r.settlements = 0
  · simp [closeAction, h]
  · simp [closeAction, h]

/- Exponents specify coefficients for any scalar interpretation of γ. -/
def terminalExponent (blocks : Nat) := blocks-1
def bootstrapExponent (blocks : Nat) := blocks
theorem terminal_before_bootstrap (d : Nat) (h : 0 < d) :
    terminalExponent d + 1 = bootstrapExponent d := by
  simp [terminalExponent, bootstrapExponent]
  omega
theorem inherited_single_block_clock :
    terminalExponent 1 = 0 ∧ bootstrapExponent 1 = 1 := by decide

def repeatedCost (k h l : Nat) := k*(h+l)
def optionCost (k h l c : Nat) := h+k*l+k*c

theorem saved_work_identity (k h l c : Nat) :
    optionCost (k+1) h l c + k*h = repeatedCost (k+1) h l + (k+1)*c := by
  simp [optionCost, repeatedCost, Nat.mul_add, Nat.add_mul,
        Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

theorem faster_when_saved_work_exceeds_checks (k h l c : Nat)
    (hcost : (k+1)*c < k*h) :
    optionCost (k+1) h l c < repeatedCost (k+1) h l := by
  have hid := saved_work_identity k h l c
  omega

end CGB.BoundedReadingOption
