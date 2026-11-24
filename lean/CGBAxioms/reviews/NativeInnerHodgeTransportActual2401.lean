import Mathlib.Tactic
namespace CGB.NativeInnerHodgeTransportActual2401
open Matrix
/-- The certified support mask follows the same eight actual fact identities
    as their 128-coordinate response rows. -/
theorem transport_mask (M : Matrix (Fin 8) (Fin 4) ℝ)
    (R : Matrix (Fin 8) (Fin 128) ℝ) (P : Matrix (Fin 8) (Fin 8) ℝ)
    (orthogonal : P.transpose * P = 1) :
    (P * M).transpose * (P * R) = M.transpose * R := by
  simp only [Matrix.transpose_mul, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc P.transpose P R, orthogonal, Matrix.one_mul]
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old0 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new0 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm0 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new0[e]!)[j]! == (old0[e]!)[perm0[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old1 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new1 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm1 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new1[e]!)[j]! == (old1[e]!)[perm1[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old2 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new2 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm2 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new2[e]!)[j]! == (old2[e]!)[perm2[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old3 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new3 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm3 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new3[e]!)[j]! == (old3[e]!)[perm3[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old4 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new4 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm4 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new4[e]!)[j]! == (old4[e]!)[perm4[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old5 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new5 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm5 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new5[e]!)[j]! == (old5[e]!)[perm5[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old6 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new6 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm6 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new6[e]!)[j]! == (old6[e]!)[perm6[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old7 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new7 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm7 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new7[e]!)[j]! == (old7[e]!)[perm7[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old8 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new8 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm8 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new8[e]!)[j]! == (old8[e]!)[perm8[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old9 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new9 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm9 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new9[e]!)[j]! == (old9[e]!)[perm9[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old10 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new10 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm10 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new10[e]!)[j]! == (old10[e]!)[perm10[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old11 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new11 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm11 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new11[e]!)[j]! == (old11[e]!)[perm11[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old12 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new12 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm12 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new12[e]!)[j]! == (old12[e]!)[perm12[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old13 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new13 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm13 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new13[e]!)[j]! == (old13[e]!)[perm13[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old14 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new14 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm14 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new14[e]!)[j]! == (old14[e]!)[perm14[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old15 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new15 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm15 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new15[e]!)[j]! == (old15[e]!)[perm15[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old16 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new16 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm16 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new16[e]!)[j]! == (old16[e]!)[perm16[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old17 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new17 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm17 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new17[e]!)[j]! == (old17[e]!)[perm17[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old18 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new18 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm18 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new18[e]!)[j]! == (old18[e]!)[perm18[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old19 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new19 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm19 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new19[e]!)[j]! == (old19[e]!)[perm19[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old20 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new20 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm20 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new20[e]!)[j]! == (old20[e]!)[perm20[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old21 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new21 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm21 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new21[e]!)[j]! == (old21[e]!)[perm21[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old22 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new22 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm22 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new22[e]!)[j]! == (old22[e]!)[perm22[j]!]!)) = true := by native_decide
example : ([1, 2, 3, 4, 0, 5, 6, 7] : List Nat).Perm (List.range 8) := by decide
def old23 : Array (Array Nat) := #[#[1, 0, 0, 0, 0, 0, 1, 1], #[1, 0, 0, 0, 0, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def new23 : Array (Array Nat) := #[#[0, 0, 0, 0, 1, 0, 1, 1], #[0, 0, 0, 0, 1, 0, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1]]
def perm23 : Array Nat := #[1, 2, 3, 4, 0, 5, 6, 7]
example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new23[e]!)[j]! == (old23[e]!)[perm23[j]!]!)) = true := by native_decide
end CGB.NativeInnerHodgeTransportActual2401
