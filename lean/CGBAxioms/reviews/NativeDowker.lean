import Mathlib.Tactic

/-!
Dowker measurement complex of the actual relation/typed-unknown incidence.
A simplex has a concrete common-unknown witness. This is an associated
measurement complex, not an assertion that the runtime explicitly stores cells
or obeys topological oscillator equations. No five-phase cartoon is used.
-/
namespace CGB.NativeDowker

def Supported {R U : Type*} (depends : R → U → Prop) (vertices : Finset R) : Prop :=
  ∃ unknown, ∀ relation ∈ vertices, depends relation unknown

theorem support_downward_closed {R U : Type*} (depends : R → U → Prop)
    (small large : Finset R) (subset : small ⊆ large)
    (h : Supported depends large) : Supported depends small := by
  obtain ⟨unknown, witness⟩ := h
  exact ⟨unknown, fun relation member => witness relation (subset member)⟩

structure Cell where
  vertices : List Nat
  witness : Nat
  deriving Inhabited, Repr

abbrev Boundary := List (Nat × Int)

/-- Compact certificates avoid compiling thousands of literal list cells. -/
def decodeNats (s : String) : List Nat :=
  if s.isEmpty then [] else (s.splitOn ",").map (fun x => x.toNat?.getD 0)

def decodeCells (s : String) : Array Cell :=
  (s.splitOn "|").toArray.map (fun row =>
    let fields := row.splitOn ":"
    ⟨decodeNats (fields[1]?.getD ""), (fields[0]?.getD "").toNat?.getD 0⟩)

def deriveBoundary (low high : Array Cell) : Array Boundary :=
  high.map (fun c => (List.range c.vertices.length).map (fun j =>
    let face := c.vertices.eraseIdx j
    let index := (low.toList.findIdx? (fun lower => lower.vertices == face)).getD low.size
    (index, if j % 2 == 0 then 1 else -1)))

def uniqueCells (cells : Array Cell) : Bool :=
  decide ((cells.toList.map Cell.vertices).Nodup)

def combinations : Nat → List Nat → List (List Nat)
  | 0, _ => [[]]
  | _ + 1, [] => []
  | n + 1, x :: xs => (combinations n xs).map (x :: ·) ++ combinations (n + 1) xs

def sortedDistinct : List Nat → Bool
  | [] | [_] => true
  | x :: y :: xs => x < y && sortedDistinct (y :: xs)

def validCell (supports : Array (List Nat)) (vertexCount size : Nat) (c : Cell) : Bool :=
  c.vertices.length == size && sortedDistinct c.vertices &&
  c.vertices.all (· < vertexCount) && c.witness < supports.size &&
  c.vertices.all (supports[c.witness]!.contains ·)

def completeCells (supports : Array (List Nat)) (size : Nat) (cells : Array Cell) : Bool :=
  supports.all (fun support => (combinations size support).all
    (fun vertices => cells.any (fun c => c.vertices == vertices)))

def boundaryCorrect (low high : Array Cell) (boundary : Array Boundary) : Bool :=
  high.size == boundary.size && (List.range high.size).all (fun i =>
    let c := high[i]!
    let b := boundary[i]!
    b.length == c.vertices.length && (List.range b.length).all (fun j =>
      let term := b[j]!
      term.1 < low.size &&
      low[term.1]!.vertices == c.vertices.eraseIdx j &&
      term.2 == (if j % 2 == 0 then 1 else -1)))

def expandBoundary (low : Array Boundary) (b : Boundary) : Boundary :=
  b.flatMap (fun (index, coefficient) =>
    low[index]!.map (fun (endpoint, value) => (endpoint, coefficient * value)))

def boundarySquareZero (low high : Array Boundary) : Bool :=
  high.all (fun b =>
    let terms := expandBoundary low b
    terms.all (fun (endpoint, _) =>
      (terms.foldl (fun acc term => if term.1 == endpoint then acc + term.2 else acc) 0) == 0))

end CGB.NativeDowker
