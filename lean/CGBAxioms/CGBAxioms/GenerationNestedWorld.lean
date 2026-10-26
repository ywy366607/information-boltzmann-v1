import CGBAxioms.GenerationLattice
import CGBAxioms.GenerationNesting
import CGBAxioms.GenerationTable

/-!
# An organism nested in its world (T-E2)

The joint medium of a bacterium-like organism and its world, as a lattice medium (M4). A cell
holds a food pool and a waste pool (each `0..V` quanta) and at most one organism. The organism's
inner state is its inner units' states (`Fin n → Λi`, the inner medium one level down) and a
heading. Its body costs `B > 0` content (the body is matter, so an empty cell is the unique
vacuum).

Cells form any world with a symmetric neighbour relation (`axioms_ring`, `axioms_torus` for a
`G × G` torus); identity pages are `9 * kind + position code`, so a cell can tell the neighbour
ahead for each heading (chirality, P2c) and its kind (plain 0, food source 1, waste sink 2,
birth place 3). Source and sink cells are the world's own boundary; a source receives whole
pellets (`pellet` quanta) at the potential of its stimulus, so food rain and the user's clicks are
stimuli at the world's boundary.

Rules (each with its reverse):
* single cell: an inner pair rule on two inner units; an exchange of one inner unit with the
  cell's food or waste pool (sensing, eating, excreting); a turn from heading `a` to `b` while the
  motor is not in its run state (tumble); a food pellet entering or leaving at a source cell,
  waste at a sink cell;
* neighbour cells: one quantum of food or waste diffusing; the organism moving to an empty
  neighbour along its heading while one fuel rule fires inside it (moving forward spends fuel,
  moving back would return it: self-propulsion is driven).

`NestedWorld.axioms`: the joint medium satisfies the six clauses when the inner rules and the
fuel rule conserve content and every rate prefactor is nonnegative.
-/

open Finset

namespace CGB

/-- Shift a pool level by `d`, if it stays in `0..V`. -/
def poolShift {V : ℕ} (k : Fin (V + 1)) (d : ℤ) : Option (Fin (V + 1)) :=
  if h : 0 ≤ (k : ℤ) + d ∧ (k : ℤ) + d ≤ V then
    some ⟨((k : ℤ) + d).toNat, by omega⟩
  else none

theorem poolShift_back {V : ℕ} {k k' : Fin (V + 1)} {d : ℤ} (h : poolShift k d = some k') :
    poolShift k' (-d) = some k := by
  unfold poolShift at h ⊢
  split_ifs at h with h1
  cases h
  have hk : ((⟨((k : ℤ) + d).toNat, by omega⟩ : Fin (V + 1)) : ℤ) = (k : ℤ) + d := by
    simp only [Fin.val_mk]; omega
  rw [dif_pos (by rw [hk]; omega)]
  congr 1
  ext
  simp only [Fin.val_mk]
  rw [hk]; omega

theorem poolShift_val {V : ℕ} {k k' : Fin (V + 1)} {d : ℤ} (h : poolShift k d = some k') :
    (k' : ℤ) = k + d := by
  unfold poolShift at h
  split_ifs at h with h1
  cases h
  simp only [Fin.val_mk]; omega

/-- The data of a nested world. -/
structure NestedWorld (Λi Hd : Type*) (n nR nX : ℕ) where
  /-- pool capacity (food and waste, quanta per cell) -/
  V : ℕ
  /-- content of the organism's body -/
  B : ℕ
  B_pos : 0 < B
  /-- identity page of each inner unit -/
  pageIn : Fin n → ℕ
  contentIn : Λi → ℕ
  /-- inner state energies by inner page -/
  φIn : ℕ → Λi → ℝ
  /-- the inner pair rules -/
  spec : Fin nR → RuleSpec Λi
  /-- pool exchange `j`: an inner unit on page `xpage j` goes `xa j → xb j`; the food pool
  (`xfood j`) or the waste pool takes the content difference -/
  xpage : Fin nX → ℕ
  xa : Fin nX → Λi
  xb : Fin nX → Λi
  xfood : Fin nX → Bool
  xκ : Fin nX → ℝ
  xκ_nonneg : ∀ j, 0 ≤ xκ j
  /-- `xgated j`: the exchange goes through the membrane (conductance ∝ intact membrane units);
  an ungated exchange is physics the organism does not control (wear of the membrane itself) -/
  xgated : Fin nX → Bool
  /-- the motor unit and its run state -/
  motor : Fin n
  run : Λi
  /-- the inner rule that fires when the organism moves (fuel) -/
  fuel : RuleSpec Λi
  κflip : ℝ
  κmove : ℝ
  κdiff : ℝ
  κin : ℝ
  κflip_nonneg : 0 ≤ κflip
  κmove_nonneg : 0 ≤ κmove
  κdiff_nonneg : 0 ≤ κdiff
  κin_nonneg : 0 ≤ κin
  /-- quanta in one food pellet supplied by the world's boundary -/
  pellet : ℕ
  /-- `ahead i j h`: the cell with page `j` is the neighbour ahead of the cell with page `i` for
  heading `h` (read from identity pages only, so P2c holds) -/
  ahead : ℕ → ℕ → Hd → Bool
  /-- the membrane: exchanges through the boundary go through intact units of this page (state
  `gateOn`); their conductance is proportional to how many there are (`gateCount`) -/
  gatePage : ℕ
  gateOn : Λi
  /-- the empty body (every inner unit in a state without content): it can dissolve into the
  cell's waste pool, releasing the body's content `B`; the reverse forms an empty body from `B`
  quanta of waste -/
  yEmpty : Fin n → Λi
  κdis : ℝ
  κdis_nonneg : 0 ≤ κdis

namespace NestedWorld

variable {Λi Hd : Type*} [DecidableEq Λi] [DecidableEq Hd] {n nR nX : ℕ} (W : NestedWorld Λi Hd n nR nX)

/-- A cell: food, waste, and possibly an organism (inner states and heading). -/
abbrev Cell := Fin (W.V + 1) × Fin (W.V + 1) × Option ((Fin n → Λi) × Hd)

/-- Single-cell rules: inner pair rule, pool exchange, turn (heading a to b), boundary supply,
dissolution of an empty body with heading `h` (and its reverse). -/
abbrev NR1 (Hd : Type*) (n nR nX : ℕ) :=
  (Fin nR × Fin n × Fin n × Bool) ⊕ (Fin nX × Fin n × Bool) ⊕ (Hd × Hd) ⊕
    ((Bool × Bool) ⊕ (Hd × Bool))

/-- Neighbour rules: diffusion of food or waste, movement with fuel units. -/
abbrev NR2 (n : ℕ) := (Bool × Bool) ⊕ (Fin n × Fin n × Bool)

/-- Content of an organism. -/
def bugContent (y : Fin n → Λi) : ℕ := W.B + ∑ u, W.contentIn (y u)

def content (l : W.Cell) : ℕ :=
  l.1 + l.2.1 + match l.2.2 with
    | none => 0
    | some (y, _) => W.bugContent y

/-- Intact membrane units of an organism. -/
def gateCount (y : Fin n → Λi) : ℕ :=
  (Finset.univ.filter fun v => W.pageIn v = W.gatePage ∧ y v = W.gateOn).card

/-- Intact membrane units in a cell (none without an organism). -/
def cellGate : W.Cell → ℕ
  | (_, _, some (y, _)) => W.gateCount y
  | _ => 0

/-- Inner energy of an organism. -/
noncomputable def bugE (y : Fin n → Λi) : ℝ := ∑ u, W.φIn (W.pageIn u) (y u)

noncomputable def φ (_i : ℕ) (l : W.Cell) : ℝ :=
  Generation.cellF W.V l.1 + Generation.cellF W.V l.2.1 + match l.2.2 with
    | none => 0
    | some (y, _) => W.bugE y

/-- An inner pair rule applied to two inner units. -/
def innerAct (k : Fin nR) (u v : Fin n) (dir : Bool) (y : Fin n → Λi) : Fin n → Λi :=
  if u = v then y else
    let p := (W.spec k).apply dir (W.pageIn u) (W.pageIn v) (y u, y v)
    Function.update (Function.update y u p.1) v p.2

/-- The fuel rule applied to two inner units. -/
def fuelAct (e d : Fin n) (dir : Bool) (y : Fin n → Λi) : Fin n → Λi :=
  if e = d then y else
    let p := W.fuel.apply dir (W.pageIn e) (W.pageIn d) (y e, y d)
    Function.update (Function.update y e p.1) d p.2

/-- Effect of a single-cell rule on a cell with page `i`. -/
def eff1 : NR1 Hd n nR nX → ℕ → W.Cell → W.Cell
  | .inl (k, u, v, dir), _, l => match l with
      | (f, w, some (y, h)) => (f, w, some (W.innerAct k u v dir y, h))
      | l => l
  | .inr (.inl (j, u, dir)), _, l => match l with
      | (f, w, some (y, h)) =>
          let a := if dir then W.xb j else W.xa j
          let b := if dir then W.xa j else W.xb j
          let d : ℤ := (W.contentIn a : ℤ) - W.contentIn b
          if W.pageIn u = W.xpage j ∧ y u = a then
            if W.xfood j then
              match poolShift f d with
              | some f' => (f', w, some (Function.update y u b, h))
              | none => l
            else
              match poolShift w d with
              | some w' => (f, w', some (Function.update y u b, h))
              | none => l
          else l
      | l => l
  | .inr (.inr (.inl (a, b))), _, l => match l with
      | (f, w, some (y, h)) => if y W.motor = W.run ∨ h ≠ a then l else (f, w, some (y, b))
      | l => l
  | .inr (.inr (.inr (.inl (food, dir)))), i, l =>
      if food then
        let d : ℤ := if dir then -(W.pellet : ℤ) else W.pellet
        if i / 9 = 1 then match poolShift l.1 d with
          | some f' => (f', l.2.1, l.2.2)
          | none => l
        else l
      else
        let d : ℤ := if dir then -1 else 1
        if i / 9 = 2 then match poolShift l.2.1 d with
          | some w' => (l.1, w', l.2.2)
          | none => l
        else l
  | .inr (.inr (.inr (.inr (hd, dir)))), _, l =>
      if dir then
        match l with
        | (f, w, none) => match poolShift w (-(W.B : ℤ)) with
          | some w' => (f, w', some (W.yEmpty, hd))
          | none => l
        | l => l
      else
        match l with
        | (f, w, some (y, h)) =>
            if y = W.yEmpty ∧ h = hd then
              match poolShift w (W.B : ℤ) with
              | some w' => (f, w', none)
              | none => l
            else l
        | l => l

/-- Effect of a neighbour rule on the ordered pair of cells with pages `i`, `j`. -/
def eff2 : NR2 n → ℕ → ℕ → W.Cell × W.Cell → W.Cell × W.Cell
  | .inl (food, dir), _, _, (l, l') =>
      let d : ℤ := if dir then 1 else -1
      if food then
        match poolShift l.1 d, poolShift l'.1 (-d) with
        | some f, some f' => ((f, l.2.1, l.2.2), (f', l'.2.1, l'.2.2))
        | _, _ => (l, l')
      else
        match poolShift l.2.1 d, poolShift l'.2.1 (-d) with
        | some w, some w' => ((l.1, w, l.2.2), (l'.1, w', l'.2.2))
        | _, _ => (l, l')
  | .inr (e, d, dir), i, j, (l, l') =>
      if dir then
        match l.2.2, l'.2.2 with
        | none, some (y, h) =>
            let y' := W.fuelAct e d true y
            if W.ahead i j h = true ∧ y' ≠ y ∧ y W.motor = W.run ∧ y' W.motor = W.run then
              ((l.1, l.2.1, some (y', h)), (l'.1, l'.2.1, none))
            else (l, l')
        | _, _ => (l, l')
      else
        match l.2.2, l'.2.2 with
        | some (y, h), none =>
            let y' := W.fuelAct e d false y
            if W.ahead i j h = true ∧ y' ≠ y ∧ y W.motor = W.run ∧ y' W.motor = W.run then
              ((l.1, l.2.1, none), (l'.1, l'.2.1, some (y', h)))
            else (l, l')
        | _, _ => (l, l')

def rev1 : NR1 Hd n nR nX → NR1 Hd n nR nX
  | .inl (k, u, v, dir) => .inl (k, u, v, !dir)
  | .inr (.inl (j, u, dir)) => .inr (.inl (j, u, !dir))
  | .inr (.inr (.inl (a, b))) => .inr (.inr (.inl (b, a)))
  | .inr (.inr (.inr (.inl (food, dir)))) => .inr (.inr (.inr (.inl (food, !dir))))
  | .inr (.inr (.inr (.inr (hd, dir)))) => .inr (.inr (.inr (.inr (hd, !dir))))

def rev2 : NR2 n → NR2 n
  | .inl (food, dir) => .inl (food, !dir)
  | .inr (e, d, dir) => .inr (e, d, !dir)

def c1 : NR1 Hd n nR nX → ℕ → W.Cell → ℝ
  | .inl (k, _, _, _), _, _ => (W.spec k).κ
  | .inr (.inl (j, _, _)), _, l => W.xκ j * (if W.xgated j then (W.cellGate l : ℝ) else 1)
  | .inr (.inr (.inl _)), _, _ => W.κflip
  | .inr (.inr (.inr (.inl _))), _, _ => W.κin
  | .inr (.inr (.inr (.inr _))), _, _ => W.κdis

def c2 : NR2 n → ℕ → ℕ → W.Cell × W.Cell → ℝ
  | .inl _, _, _, _ => W.κdiff
  | .inr _, _, _, _ => W.κmove

/-! ## Inner rules: content and reversal -/

theorem apply_content {cIn : Λi → ℕ} (s : RuleSpec Λi)
    (hs : cIn s.a.1 + cIn s.a.2 = cIn s.b.1 + cIn s.b.2) (dir : Bool) (i j : ℕ) (p : Λi × Λi) :
    cIn (s.apply dir i j p).1 + cIn (s.apply dir i j p).2 = cIn p.1 + cIn p.2 := by
  cases dir
  · simp only [RuleSpec.apply, Bool.false_eq_true, if_false]
    split_ifs with hm
    · rw [hm.2.2]; exact hs.symm
    · rfl
  · simp only [RuleSpec.apply, if_true]
    split_ifs with hm
    · rw [hm.2.2]; exact hs
    · rfl

theorem sum_pair_update [Fintype (Fin n)] (g : Λi → ℕ) {u v : Fin n} (huv : u ≠ v) (a b : Λi)
    (y : Fin n → Λi) (h : g a + g b = g (y u) + g (y v)) :
    ∑ w, g (Function.update (Function.update y u a) v b w) = ∑ w, g (y w) := by
  have := LatticeCRN.sum_update2 (fun _ l => (g l : ℝ)) huv a b y
  have hR : (∑ w, (g (Function.update (Function.update y u a) v b w) : ℝ)) =
      ∑ w, (g (y w) : ℝ) := by
    have hh : (g a : ℝ) + g b = g (y u) + g (y v) := by exact_mod_cast h
    linarith
  exact_mod_cast hR

theorem pair_back {u v : Fin n} (huv : u ≠ v) (y : Fin n → Λi) (p : Λi × Λi)
    (q : Λi × Λi) (hq : q = (y u, y v)) :
    Function.update (Function.update (Function.update (Function.update y u p.1) v p.2) u q.1) v q.2 = y := by
  subst hq
  funext w
  by_cases hwv : w = v
  · subst hwv; simp
  · by_cases hwu : w = u
    · subst hwu; simp [Function.update_of_ne hwv]
    · simp [Function.update_of_ne hwv, Function.update_of_ne hwu]

theorem innerAct_content (hspec : ∀ k, W.contentIn (W.spec k).a.1 + W.contentIn (W.spec k).a.2 =
      W.contentIn (W.spec k).b.1 + W.contentIn (W.spec k).b.2)
    (k : Fin nR) (u v : Fin n) (dir : Bool) (y : Fin n → Λi) :
    ∑ w, W.contentIn (W.innerAct k u v dir y w) = ∑ w, W.contentIn (y w) := by
  unfold innerAct
  split_ifs with huv
  · rfl
  · exact sum_pair_update _ huv _ _ y (apply_content _ (hspec k) dir _ _ _)

theorem fuelAct_content (hfuel : W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 =
      W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2)
    (e d : Fin n) (dir : Bool) (y : Fin n → Λi) :
    ∑ w, W.contentIn (W.fuelAct e d dir y w) = ∑ w, W.contentIn (y w) := by
  unfold fuelAct
  split_ifs with hed
  · rfl
  · exact sum_pair_update _ hed _ _ y (apply_content _ hfuel dir _ _ _)

theorem innerAct_back (k : Fin nR) (u v : Fin n) (dir : Bool) (y : Fin n → Λi)
    (hne : W.innerAct k u v dir y ≠ y) : W.innerAct k u v (!dir) (W.innerAct k u v dir y) = y := by
  unfold innerAct at hne ⊢
  split_ifs at hne ⊢ with huv
  · exact absurd rfl hne
  · set p := (W.spec k).apply dir (W.pageIn u) (W.pageIn v) (y u, y v)
    have hp : p ≠ (y u, y v) := by
      intro e; apply hne; rw [e]; exact pair_back huv y (y u, y v) (y u, y v) rfl ▸ by
        funext w
        by_cases hwv : w = v
        · subst hwv; simp
        · by_cases hwu : w = u
          · subst hwu; simp [Function.update_of_ne hwv]
          · simp [Function.update_of_ne hwv, Function.update_of_ne hwu]
    have hb := RuleSpec.apply_back _ dir _ _ _ hp
    simp only [Function.update_self, Function.update_of_ne huv]
    rw [Prod.mk.eta, hb]
    exact pair_back huv y p _ rfl

theorem fuelAct_back (e d : Fin n) (dir : Bool) (y : Fin n → Λi)
    (hne : W.fuelAct e d dir y ≠ y) : W.fuelAct e d (!dir) (W.fuelAct e d dir y) = y := by
  unfold fuelAct at hne ⊢
  split_ifs at hne ⊢ with hed
  · exact absurd rfl hne
  · set p := W.fuel.apply dir (W.pageIn e) (W.pageIn d) (y e, y d)
    have hp : p ≠ (y e, y d) := by
      intro h; apply hne; rw [h]
      funext w
      by_cases hwd : w = d
      · subst hwd; simp
      · by_cases hwe : w = e
        · subst hwe; simp [Function.update_of_ne hwd]
        · simp [Function.update_of_ne hwd, Function.update_of_ne hwe]
    have hb := RuleSpec.apply_back _ dir _ _ _ hp
    simp only [Function.update_self, Function.update_of_ne hed]
    rw [Prod.mk.eta, hb]
    exact pair_back hed y p _ rfl

/-! ## The joint medium -/

/-- Identity pages are `9 * kind + local position code` (`< 9`): kind 0 plain, 1 food source,
2 waste sink (the world's own boundary), 3 birth place. -/
def bidW : Finset ℕ := Finset.Icc 9 26

/-- Birth: an organism in state `y0` with heading `h0` on cells of kind 3, every other cell
empty. -/
def birthCell (y0 : Fin n → Λi) (h0 : Hd) (i : ℕ) : W.Cell :=
  if i / 9 = 3 then (0, 0, some (y0, h0)) else (0, 0, none)

theorem c1_nonneg (r : NR1 Hd n nR nX) (i : ℕ) (l : W.Cell) : 0 ≤ W.c1 r i l := by
  rcases r with ⟨k, -, -, -⟩ | ⟨j, -, -⟩ | ⟨-, -⟩ | ⟨-, -⟩ | ⟨-, -⟩
  · exact (W.spec k).κ_nonneg
  · exact mul_nonneg (W.xκ_nonneg j) (by split_ifs <;> positivity)
  · exact W.κflip_nonneg
  · exact W.κin_nonneg
  · exact W.κdis_nonneg

theorem c2_nonneg (r : NR2 n) (i j : ℕ) (p : W.Cell × W.Cell) : 0 ≤ W.c2 r i j p := by
  rcases r with ⟨-, -⟩ | ⟨-, -, -⟩
  · exact W.κdiff_nonneg
  · exact W.κmove_nonneg

/-- The joint medium of organisms and their world on cells `U` with a symmetric, irreflexive
neighbour relation. -/
noncomputable def lattice {U : Type*} (adj : U → U → Prop) (hsymm : ∀ u v, adj u v → adj v u)
    (hirr : ∀ u, ¬ adj u u) (page : U → ℕ) (y0 : Fin n → Λi) (h0 : Hd) :
    LatticeCRN U W.Cell ℝ (NR1 Hd n nR nX) (NR2 n) where
  adj := adj
  adj_symm := hsymm
  adj_irrefl := hirr
  ident := page
  bid := bidW
  content := W.content
  φ := W.φ
  μ := id
  eff1 := W.eff1
  eff2 := W.eff2
  c1 := W.c1
  c2 := W.c2
  c1_nonneg := W.c1_nonneg
  c2_nonneg := W.c2_nonneg
  rev1 := rev1
  rev2 := rev2
  birth := W.birthCell y0 h0
  calm := 0

/-! ## Obligations -/

theorem content_p1a [Fintype Λi] [Fintype Hd] :
    (∀ m, {l : W.Cell | W.content l ≤ m}.Finite) ∧ (∃ K, ∀ l, W.content l ≤ K) ∧
      ∃! l, W.content l = 0 := by
  classical
  refine ⟨fun _ => Set.toFinite _, ?_, ?_⟩
  · refine ⟨W.V + W.V + (W.B + n * Finset.univ.sup W.contentIn), fun l => ?_⟩
    obtain ⟨f, w, b⟩ := l
    have hf := f.isLt
    have hw := w.isLt
    rcases b with _ | ⟨y, h⟩
    · simp only [content]; omega
    · simp only [content, bugContent]
      have hs : ∑ u, W.contentIn (y u) ≤ n * Finset.univ.sup W.contentIn := by
        calc ∑ u, W.contentIn (y u) ≤ ∑ _u : Fin n, Finset.univ.sup W.contentIn :=
              Finset.sum_le_sum fun u _ => Finset.le_sup (Finset.mem_univ (y u))
          _ = n * Finset.univ.sup W.contentIn := by simp
      omega
  · refine ⟨(0, 0, none), by simp [content], fun l hl => ?_⟩
    obtain ⟨f, w, b⟩ := l
    rcases b with _ | ⟨y, h⟩
    · simp only [content] at hl
      have hf : f.val = 0 := by omega
      have hw : w.val = 0 := by omega
      have e1 : f = 0 := Fin.ext (by rw [hf]; rfl)
      have e2 : w = 0 := Fin.ext (by rw [hw]; rfl)
      rw [e1, e2]
    · simp only [content, bugContent] at hl
      have := W.B_pos
      omega

theorem not_bid {i : ℕ} (hi : i ∉ bidW) : i / 9 ≠ 1 ∧ i / 9 ≠ 2 := by
  simp only [bidW, Finset.mem_Icc, not_and_or, not_le] at hi
  omega

theorem sum_update_nat (g : Λi → ℕ) (u : Fin n) (b : Λi) (y : Fin n → Λi) :
    ((∑ w, g (Function.update y u b w) : ℕ) : ℤ) = ∑ w, (g (y w) : ℤ) - g (y u) + g b := by
  have := LatticeCRN.sum_update (fun _ l => (g l : ℝ)) u b y
  have hR : ((∑ w, g (Function.update y u b w) : ℕ) : ℝ) = ∑ w, (g (y w) : ℝ) - g (y u) + g b := by
    push_cast; linarith
  have : ((∑ w, g (Function.update y u b w) : ℕ) : ℝ) =
      (((∑ w, (g (y w) : ℤ) - g (y u) + g b : ℤ)) : ℝ) := by
    rw [hR]; push_cast; ring
  exact_mod_cast this

theorem cons1 (hspec : ∀ k, W.contentIn (W.spec k).a.1 + W.contentIn (W.spec k).a.2 =
      W.contentIn (W.spec k).b.1 + W.contentIn (W.spec k).b.2)
    (hEmpty : ∀ u, W.contentIn (W.yEmpty u) = 0)
    (r : NR1 Hd n nR nX) (i : ℕ) (l : W.Cell) (hi : i ∉ bidW) :
    W.content (W.eff1 r i l) = W.content l := by
  obtain ⟨f, w, b⟩ := l
  rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨ha, hb⟩ | ⟨food, dir⟩ | ⟨hd, dir⟩
  · rcases b with _ | ⟨y, h⟩
    · rfl
    · simp only [eff1, content, bugContent, W.innerAct_content hspec]
  · rcases b with _ | ⟨y, h⟩
    · rfl
    · simp only [eff1]
      generalize (if dir = true then W.xb j else W.xa j) = a
      generalize (if dir = true then W.xa j else W.xb j) = b
      by_cases hm : W.pageIn u = W.xpage j ∧ y u = a
      · rw [if_pos hm]
        have hsum := sum_update_nat W.contentIn u b y
        rw [hm.2] at hsum
        by_cases hfood : W.xfood j = true
        · rw [if_pos hfood]
          rcases hps : poolShift f ((W.contentIn a : ℤ) - W.contentIn b) with _ | f'
          · rfl
          · have hv := poolShift_val hps
            simp only [content, bugContent]
            zify at hv ⊢
            push_cast at hsum; linarith
        · rw [if_neg hfood]
          rcases hps : poolShift w ((W.contentIn a : ℤ) - W.contentIn b) with _ | w'
          · rfl
          · have hv := poolShift_val hps
            simp only [content, bugContent]
            zify at hv ⊢
            push_cast at hsum; linarith
      · rw [if_neg hm]
  · rcases b with _ | ⟨y, h⟩
    · rfl
    · simp only [eff1]; split_ifs <;> rfl
  · have hk := not_bid hi
    cases food <;> simp [eff1, hk.1, hk.2]
  · have h0 : ∑ u, W.contentIn (W.yEmpty u) = 0 := by simp [hEmpty]
    rcases b with _ | ⟨y, h⟩ <;> cases dir
    · rfl
    · rcases hps : poolShift w (-(W.B : ℤ)) with _ | w'
      · simp only [eff1, if_true, hps]
      · have hv := poolShift_val hps
        simp only [eff1, if_true, hps, content, bugContent, h0, add_zero]
        omega
    · by_cases hc : y = W.yEmpty ∧ h = hd
      · rcases hps : poolShift w (W.B : ℤ) with _ | w'
        · simp only [eff1, Bool.false_eq_true, if_false, if_pos hc, hps]
        · have hv := poolShift_val hps
          obtain ⟨rfl, rfl⟩ := hc
          simp only [eff1, Bool.false_eq_true, if_false, and_self, if_true, hps, content, bugContent, h0, add_zero]
          omega
      · simp only [eff1, Bool.false_eq_true, if_false, if_neg hc]
    · rfl

theorem cons2 (hfuel : W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 =
      W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2)
    (r : NR2 n) (i j : ℕ) (p : W.Cell × W.Cell) :
    W.content (W.eff2 r i j p).1 + W.content (W.eff2 r i j p).2 =
      W.content p.1 + W.content p.2 := by
  obtain ⟨⟨f, w, b⟩, ⟨f', w', b'⟩⟩ := p
  rcases r with ⟨food, dir⟩ | ⟨e, d, dir⟩
  · simp only [eff2]
    generalize (if dir = true then (1 : ℤ) else -1) = δ
    cases food
    · simp only [Bool.false_eq_true, if_false]
      rcases h1 : poolShift w δ with _ | g <;> rcases h2 : poolShift w' (-δ) with _ | g'
      · rfl
      · rfl
      · rfl
      · simp only [content]
        have := poolShift_val h1
        have := poolShift_val h2
        omega
    · simp only [if_true]
      rcases h1 : poolShift f δ with _ | g <;> rcases h2 : poolShift f' (-δ) with _ | g'
      · rfl
      · rfl
      · rfl
      · simp only [content]
        have := poolShift_val h1
        have := poolShift_val h2
        omega
  · simp only [eff2]
    cases dir
    · rcases b with _ | ⟨y, h⟩ <;> rcases b' with _ | ⟨y', h'⟩
      all_goals simp only [Bool.false_eq_true, if_false]
      · split_ifs
        · simp only [content, bugContent, W.fuelAct_content hfuel]; ring
        · rfl
    · rcases b with _ | ⟨y, h⟩ <;> rcases b' with _ | ⟨y', h'⟩
      all_goals simp only [if_true]
      · split_ifs
        · simp only [content, bugContent, W.fuelAct_content hfuel]; ring
        · rfl

/-! ## Reversal -/

/-- The pool exchange with explicit source and target states. -/
def xchEff (pg : ℕ) (food : Bool) (a b : Λi) (u : Fin n) : W.Cell → W.Cell
  | (f, w, some (y, h)) =>
      if W.pageIn u = pg ∧ y u = a then
        if food then
          match poolShift f ((W.contentIn a : ℤ) - W.contentIn b) with
          | some f' => (f', w, some (Function.update y u b, h))
          | none => (f, w, some (y, h))
        else
          match poolShift w ((W.contentIn a : ℤ) - W.contentIn b) with
          | some w' => (f, w', some (Function.update y u b, h))
          | none => (f, w, some (y, h))
      else (f, w, some (y, h))
  | l => l

theorem eff1_xch (j : Fin nX) (u : Fin n) (dir : Bool) (i : ℕ) (l : W.Cell) :
    W.eff1 (.inr (.inl (j, u, dir))) i l =
      W.xchEff (W.xpage j) (W.xfood j) (if dir then W.xb j else W.xa j)
        (if dir then W.xa j else W.xb j) u l := by
  obtain ⟨f, w, _ | ⟨y, h⟩⟩ := l <;> rfl

theorem gateCount_update {u : Fin n} (hu : W.pageIn u ≠ W.gatePage) (b : Λi) (y : Fin n → Λi) :
    W.gateCount (Function.update y u b) = W.gateCount y := by
  unfold gateCount
  congr 1
  apply Finset.filter_congr
  intro v _
  by_cases hv : v = u
  · subst hv; simp [hu]
  · simp [Function.update_of_ne hv]

/-- An exchange through a page other than the membrane leaves the membrane as it is. -/
theorem xchEff_gate {pg : ℕ} (hpg : pg ≠ W.gatePage) (food : Bool) (a b : Λi) (u : Fin n)
    (l : W.Cell) : W.cellGate (W.xchEff pg food a b u l) = W.cellGate l := by
  obtain ⟨f, w, _ | ⟨y, h⟩⟩ := l
  · rfl
  · by_cases hm : W.pageIn u = pg ∧ y u = a
    · have hu : W.pageIn u ≠ W.gatePage := hm.1 ▸ hpg
      cases food
      · rcases hp : poolShift w ((W.contentIn a : ℤ) - W.contentIn b) with _ | w' <;>
          simp [xchEff, hm, hp, cellGate, W.gateCount_update hu]
      · rcases hp : poolShift f ((W.contentIn a : ℤ) - W.contentIn b) with _ | f' <;>
          simp [xchEff, hm, hp, cellGate, W.gateCount_update hu]
    · simp [xchEff, hm]

theorem xchEff_back (pg : ℕ) (food : Bool) (a b : Λi) (u : Fin n) (l : W.Cell)
    (hne : W.xchEff pg food a b u l ≠ l) : W.xchEff pg food b a u (W.xchEff pg food a b u l) = l := by
  obtain ⟨f, w, _ | ⟨y, h⟩⟩ := l
  · exact absurd rfl hne
  · by_cases hm : W.pageIn u = pg ∧ y u = a
    · have hy : Function.update (Function.update y u b) u a = y := by
        rw [Function.update_idem, ← hm.2, Function.update_eq_self]
      have hc : W.pageIn u = pg ∧ Function.update y u b u = b := ⟨hm.1, by simp⟩
      cases food
      · rcases hps : poolShift w ((W.contentIn a : ℤ) - W.contentIn b) with _ | w'
        · simp [xchEff, hm, hps] at hne
        · have hb := poolShift_back hps
          rw [neg_sub] at hb
          simp [xchEff, hm, hps, hc, hb, hy]
      · rcases hps : poolShift f ((W.contentIn a : ℤ) - W.contentIn b) with _ | f'
        · simp [xchEff, hm, hps] at hne
        · have hb := poolShift_back hps
          rw [neg_sub] at hb
          simp [xchEff, hm, hps, hc, hb, hy]
    · simp [xchEff, hm] at hne

theorem back1 (r : NR1 Hd n nR nX) (i : ℕ) (l : W.Cell) (hne : W.eff1 r i l ≠ l) :
    W.eff1 (rev1 r) i (W.eff1 r i l) = l := by
  rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨ha, hb⟩ | ⟨food, dir⟩ | ⟨hd, dir⟩
  · obtain ⟨f, w, _ | ⟨y, h⟩⟩ := l
    · exact absurd rfl hne
    · have hy : W.innerAct k u v dir y ≠ y := fun e => hne (by simp [eff1, e])
      simp [eff1, rev1, W.innerAct_back k u v dir y hy]
  · rw [eff1_xch] at hne ⊢
    change W.eff1 (.inr (.inl (j, u, !dir))) i _ = l
    rw [eff1_xch]
    have h1 : (if (!dir) = true then W.xb j else W.xa j) = (if dir = true then W.xa j else W.xb j) := by
      cases dir <;> rfl
    have h2 : (if (!dir) = true then W.xa j else W.xb j) = (if dir = true then W.xb j else W.xa j) := by
      cases dir <;> rfl
    rw [h1, h2]
    exact W.xchEff_back _ _ _ _ u l hne
  · obtain ⟨f, w, _ | ⟨y, h⟩⟩ := l
    · exact absurd rfl hne
    · by_cases hc : y W.motor = W.run ∨ h ≠ ha
      · simp [eff1, hc] at hne
      · push Not at hc
        have hab : hb ≠ ha := fun e => hne (by simp [eff1, hc, e])
        simp [eff1, rev1, hc, hab]
  · obtain ⟨f, w, b⟩ := l
    cases food
    · by_cases hk : i / 9 = 2
      · rcases hps : poolShift w (if dir = true then -1 else 1) with _ | w'
        · simp [eff1, hk, hps] at hne
        · have hb := poolShift_back hps
          have hb' : poolShift w' (if dir = false then (-1 : ℤ) else 1) = some w := by
            cases dir <;> simpa using hb
          simp [eff1, rev1, hk, hps, hb']
      · simp [eff1, hk] at hne
    · by_cases hk : i / 9 = 1
      · rcases hps : poolShift f (if dir = true then -(W.pellet : ℤ) else W.pellet) with _ | f'
        · simp [eff1, hk, hps] at hne
        · have hb := poolShift_back hps
          have hb' : poolShift f' (if dir = false then -(W.pellet : ℤ) else W.pellet) = some f := by
            cases dir <;> simpa using hb
          simp [eff1, rev1, hk, hps, hb']
      · simp [eff1, hk] at hne
  · obtain ⟨f, w, b⟩ := l
    cases dir
    · rcases b with _ | ⟨y, h⟩
      · simp [eff1] at hne
      · by_cases hc : y = W.yEmpty ∧ h = hd
        · obtain ⟨rfl, rfl⟩ := hc
          rcases hps : poolShift w (W.B : ℤ) with _ | w'
          · simp [eff1, hps] at hne
          · have hb := poolShift_back hps
            simp [eff1, rev1, hps, hb]
        · simp [eff1, hc] at hne
    · rcases b with _ | ⟨y, h⟩
      · rcases hps : poolShift w (-(W.B : ℤ)) with _ | w'
        · simp [eff1, hps] at hne
        · have hb := poolShift_back hps
          rw [neg_neg] at hb
          simp [eff1, rev1, hps, hb]
      · simp [eff1] at hne

/-- Diffusion with an explicit shift. -/
def diffEff (food : Bool) (δ : ℤ) : W.Cell × W.Cell → W.Cell × W.Cell
  | (l, l') =>
      if food then
        match poolShift l.1 δ, poolShift l'.1 (-δ) with
        | some f, some f' => ((f, l.2.1, l.2.2), (f', l'.2.1, l'.2.2))
        | _, _ => (l, l')
      else
        match poolShift l.2.1 δ, poolShift l'.2.1 (-δ) with
        | some w, some w' => ((l.1, w, l.2.2), (l'.1, w', l'.2.2))
        | _, _ => (l, l')

theorem eff2_diff (food dir : Bool) (i j : ℕ) (p : W.Cell × W.Cell) :
    W.eff2 (.inl (food, dir)) i j p = W.diffEff food (if dir then 1 else -1) p := by
  obtain ⟨l, l'⟩ := p; rfl

theorem diffEff_back (food : Bool) (δ : ℤ) (p : W.Cell × W.Cell) (hne : W.diffEff food δ p ≠ p) :
    W.diffEff food (-δ) (W.diffEff food δ p) = p := by
  obtain ⟨⟨f, w, b⟩, ⟨f', w', b'⟩⟩ := p
  cases food
  · rcases h1 : poolShift w δ with _ | g
    · simp [diffEff, h1] at hne
    · rcases h2 : poolShift w' (-δ) with _ | g'
      · simp [diffEff, h1, h2] at hne
      · have hb1 := poolShift_back h1
        have hb2 := poolShift_back h2
        rw [neg_neg] at hb2
        simp [diffEff, h1, h2, hb1, hb2]
  · rcases h1 : poolShift f δ with _ | g
    · simp [diffEff, h1] at hne
    · rcases h2 : poolShift f' (-δ) with _ | g'
      · simp [diffEff, h1, h2] at hne
      · have hb1 := poolShift_back h1
        have hb2 := poolShift_back h2
        rw [neg_neg] at hb2
        simp [diffEff, h1, h2, hb1, hb2]

theorem back2 (r : NR2 n) (i j : ℕ) (p : W.Cell × W.Cell) (hne : W.eff2 r i j p ≠ p) :
    W.eff2 (rev2 r) i j (W.eff2 r i j p) = p := by
  rcases r with ⟨food, dir⟩ | ⟨e, d, dir⟩
  · rw [eff2_diff] at hne ⊢
    change W.eff2 (.inl (food, !dir)) i j _ = p
    rw [eff2_diff]
    have hδ : (if (!dir) = true then (1 : ℤ) else -1) = -(if dir = true then 1 else -1) := by
      cases dir <;> rfl
    rw [hδ]
    exact W.diffEff_back food _ p hne
  · obtain ⟨⟨f, w, b⟩, ⟨f', w', b'⟩⟩ := p
    cases dir
    · rcases b with _ | ⟨y, h⟩ <;> rcases b' with _ | ⟨y', h'⟩
      · simp [eff2] at hne
      · simp [eff2] at hne
      · by_cases hc : W.ahead i j h = true ∧ W.fuelAct e d false y ≠ y ∧ y W.motor = W.run ∧
            W.fuelAct e d false y W.motor = W.run
        · have hy := W.fuelAct_back e d false y hc.2.1
          simp only [Bool.not_false] at hy
          have hc' : W.ahead i j h = true ∧ W.fuelAct e d true (W.fuelAct e d false y) ≠ W.fuelAct e d false y ∧
              W.fuelAct e d false y W.motor = W.run ∧
              W.fuelAct e d true (W.fuelAct e d false y) W.motor = W.run := by
            rw [hy]; exact ⟨hc.1, fun h' => hc.2.1 h'.symm, hc.2.2.2, hc.2.2.1⟩
          simp only [eff2, rev2, Bool.not_false, Bool.false_eq_true, if_false, if_true, if_pos hc]
          rw [if_pos hc', hy]
        · simp [eff2, hc] at hne
      · simp [eff2] at hne
    · rcases b with _ | ⟨y', h'⟩ <;> rcases b' with _ | ⟨y, h⟩
      · simp [eff2] at hne
      · by_cases hc : W.ahead i j h = true ∧ W.fuelAct e d true y ≠ y ∧ y W.motor = W.run ∧
            W.fuelAct e d true y W.motor = W.run
        · have hy := W.fuelAct_back e d true y hc.2.1
          simp only [Bool.not_true] at hy
          have hc' : W.ahead i j h = true ∧ W.fuelAct e d false (W.fuelAct e d true y) ≠ W.fuelAct e d true y ∧
              W.fuelAct e d true y W.motor = W.run ∧
              W.fuelAct e d false (W.fuelAct e d true y) W.motor = W.run := by
            rw [hy]; exact ⟨hc.1, fun h' => hc.2.1 h'.symm, hc.2.2.2, hc.2.2.1⟩
          simp only [eff2, rev2, Bool.not_true, Bool.false_eq_true, if_false, if_true, if_pos hc]
          rw [if_pos hc', hy]
        · simp [eff2, hc] at hne
      · simp [eff2] at hne
      · simp [eff2] at hne

open Classical in
/-- **T-E2 (an organism nested in its world is admitted)**: the joint medium of organisms and
their world satisfies the six axiom clauses as soon as the organism's inner pair rules and its
fuel rule conserve content and no exchange goes through the membrane page itself. Sensing, eating, excreting (exchange with the cell's pools),
tumbling, diffusion, driven movement and the world's own sources and sinks need no further
obligation. -/
theorem axioms [Fintype Λi] [Fintype Hd] {U : Type*} [Fintype U] [DecidableEq U]
    (adj : U → U → Prop) (hsymm : ∀ u v, adj u v → adj v u) (hirr : ∀ u, ¬ adj u u)
    (page : U → ℕ) (y0 : Fin n → Λi) (h0 : Hd)
    (hspec : ∀ k, W.contentIn (W.spec k).a.1 + W.contentIn (W.spec k).a.2 =
      W.contentIn (W.spec k).b.1 + W.contentIn (W.spec k).b.2)
    (hfuel : W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 =
      W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2)
    (hgate : ∀ j, W.xgated j = true → W.xpage j ≠ W.gatePage)
    (hEmpty : ∀ u, W.contentIn (W.yEmpty u) = 0) :
    (W.lattice adj hsymm hirr page y0 h0).medium.Axioms := by
  classical
  refine LatticeCRN.axioms (W.lattice adj hsymm hirr page y0 h0) W.content_p1a
    (fun r i l hi => W.cons1 hspec hEmpty r i l hi) (fun r i j p _ _ => W.cons2 hfuel r i j p) rfl
    ?_ ?_ (fun r i l hne => W.back1 r i l hne) (fun r i j p hne => W.back2 r i j p hne) ?_ ?_
  · intro r; rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨a, b⟩ | ⟨food, dir⟩ | ⟨hd, dir⟩ <;> simp [lattice, rev1]
  · intro r; rcases r with ⟨food, dir⟩ | ⟨e, d, dir⟩ <;> simp [lattice, rev2]
  · intro r i l _
    rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨a, b⟩ | ⟨food, dir⟩ | ⟨hd, dir⟩
    · rfl
    · change W.xκ j * (if W.xgated j then (W.cellGate (W.eff1 (.inr (.inl (j, u, dir))) i l) : ℝ) else 1) =
        W.xκ j * (if W.xgated j then (W.cellGate l : ℝ) else 1)
      by_cases hg : W.xgated j = true
      · rw [if_pos hg, if_pos hg, eff1_xch, W.xchEff_gate (hgate j hg)]
      · rw [if_neg hg, if_neg hg]
    · rfl
    · rfl
    · rfl
  · intro r i j p _; rcases r with ⟨food, dir⟩ | ⟨e, d, dir⟩ <;> rfl

/-! ## The membrane: why it is needed, and what losing it means -/

/-- Content held by the organism in a cell (its inner units; the body `B` is constant). -/
def orgIn : W.Cell → ℕ
  | (_, _, some (y, _)) => ∑ u, W.contentIn (y u)
  | _ => 0

theorem orgIn_congr {l l' : W.Cell} (h : l'.2.2 = l.2.2) : W.orgIn l' = W.orgIn l := by
  obtain ⟨_, _, b⟩ := l
  obtain ⟨_, _, b'⟩ := l'
  simp only at h
  subst h
  rcases b' with _ | ⟨y, h⟩ <;> rfl

/-- The world's supply rules never touch the organism. -/
theorem eff1_supply_bug (food dir : Bool) (i : ℕ) (l : W.Cell) :
    (W.eff1 (.inr (.inr (.inr (.inl (food, dir))))) i l).2.2 = l.2.2 := by
  obtain ⟨f, w, b⟩ := l
  cases food <;> simp only [eff1, Bool.false_eq_true, if_false, if_true] <;> split_ifs <;>
    (try split) <;> rfl

/-- **Only exchanges feed the organism**: every single-cell event that raises the content held by
the organism is a pool exchange; a gated one has conductance proportional to the intact membrane.
-/
theorem intake_is_exchange
    (hspec : ∀ k, W.contentIn (W.spec k).a.1 + W.contentIn (W.spec k).a.2 =
      W.contentIn (W.spec k).b.1 + W.contentIn (W.spec k).b.2)
    (hEmpty : ∀ u, W.contentIn (W.yEmpty u) = 0)
    (r : NR1 Hd n nR nX) (i : ℕ) (l : W.Cell) (hup : W.orgIn l < W.orgIn (W.eff1 r i l)) :
    ∃ j u dir, r = .inr (.inl (j, u, dir)) ∧
      W.c1 r i l = W.xκ j * (if W.xgated j then (W.cellGate l : ℝ) else 1) := by
  rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨a, b⟩ | ⟨food, dir⟩ | ⟨hd, dir⟩
  · exfalso
    obtain ⟨f, w, _ | ⟨y, h⟩⟩ := l
    · simp [eff1, orgIn] at hup
    · simp only [eff1, orgIn, W.innerAct_content hspec] at hup
      exact lt_irrefl _ hup
  · exact ⟨j, u, dir, rfl, rfl⟩
  · exfalso
    obtain ⟨f, w, _ | ⟨y, h⟩⟩ := l
    · simp [eff1, orgIn] at hup
    · simp only [eff1] at hup
      split_ifs at hup <;> simp [orgIn] at hup
  · exfalso
    rw [W.orgIn_congr (W.eff1_supply_bug food dir i l)] at hup
    exact lt_irrefl _ hup
  · exfalso
    obtain ⟨f, w, b⟩ := l
    rcases b with _ | ⟨y, h⟩ <;> cases dir <;>
      simp only [eff1, Bool.false_eq_true, if_false, if_true] at hup <;> (try split_ifs at hup) <;>
      (try split at hup) <;> simp_all [orgIn]

/-- **Death** (`Part.CanTakeIn` fails through the organism's own doing): with no intact membrane
unit, every event that would raise the organism's content has rate zero, except an ungated
exchange: physics the organism does not control (the reverse of a leak, membrane re-forming from
the cell's pools). Every rate of the medium is a positive multiple of `exp`, so
`Part.Recoverable` alone never fails; without the membrane factor, intake would not depend on the
body at all, structure would have no use and selection could drop its upkeep. -/
theorem dead_intake_is_physics
    (hspec : ∀ k, W.contentIn (W.spec k).a.1 + W.contentIn (W.spec k).a.2 =
      W.contentIn (W.spec k).b.1 + W.contentIn (W.spec k).b.2)
    (hEmpty : ∀ u, W.contentIn (W.yEmpty u) = 0)
    (r : NR1 Hd n nR nX) (i : ℕ) (l : W.Cell) (hdead : W.cellGate l = 0)
    (hup : W.orgIn l < W.orgIn (W.eff1 r i l)) :
    W.c1 r i l = 0 ∨ ∃ j u dir, r = .inr (.inl (j, u, dir)) ∧ W.xgated j = false := by
  obtain ⟨j, u, dir, hr, hc⟩ := W.intake_is_exchange hspec hEmpty r i l hup
  by_cases hg : W.xgated j = true
  · left; rw [hc, if_pos hg, hdead]; simp
  · right; exact ⟨j, u, dir, hr, by simpa using hg⟩

/-- **A dead body does not give itself away**: every single-cell event that lowers the content
held by the organism is a pool exchange; with no intact membrane the gated ones have rate zero, so
what leaves a dead body leaves only by an ungated exchange (physics, a leak). With G26c
(`corpses_must_go`), a world of endless mortal lives needs such leaks, or organisms that take
dead bodies apart (decomposers, events this table does not yet have). -/
theorem dead_output_is_physics
    (hspec : ∀ k, W.contentIn (W.spec k).a.1 + W.contentIn (W.spec k).a.2 =
      W.contentIn (W.spec k).b.1 + W.contentIn (W.spec k).b.2)
    (hEmpty : ∀ u, W.contentIn (W.yEmpty u) = 0)
    (r : NR1 Hd n nR nX) (i : ℕ) (l : W.Cell) (hdead : W.cellGate l = 0)
    (hdown : W.orgIn (W.eff1 r i l) < W.orgIn l) :
    W.c1 r i l = 0 ∨ ∃ j u dir, r = .inr (.inl (j, u, dir)) ∧ W.xgated j = false := by
  rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨a, b⟩ | ⟨food, dir⟩ | ⟨hd, dir⟩
  · exfalso
    obtain ⟨f, w, _ | ⟨y, h⟩⟩ := l
    · simp [eff1, orgIn] at hdown
    · simp only [eff1, orgIn, W.innerAct_content hspec] at hdown
      exact lt_irrefl _ hdown
  · by_cases hg : W.xgated j = true
    · left; simp [c1, hg, hdead]
    · right; exact ⟨j, u, dir, rfl, by simpa using hg⟩
  · exfalso
    obtain ⟨f, w, _ | ⟨y, h⟩⟩ := l
    · simp [eff1, orgIn] at hdown
    · simp only [eff1] at hdown
      split_ifs at hdown <;> simp [orgIn] at hdown
  · exfalso
    rw [W.orgIn_congr (W.eff1_supply_bug food dir i l)] at hdown
    exact lt_irrefl _ hdown
  · exfalso
    obtain ⟨f, w, b⟩ := l
    rcases b with _ | ⟨y, h⟩ <;> cases dir <;>
      simp only [eff1, Bool.false_eq_true, if_false, if_true] at hdown <;> (try split_ifs at hdown) <;>
      (try split at hdown) <;> simp_all [orgIn]

/-- **Only dissolution empties a place** (with G26c `corpses_must_go`, why the table needs it):
every single-cell event that turns a cell holding a body into a cell without one is the
dissolution of an empty body. (A move only carries the body to the next cell.) Without this
event no place ever loses its body, and a world of endless mortal lives fills with corpses. By
P2b the reverse exists too: an empty body forming from `B` quanta of waste, an entropy-suppressed
spontaneous generation. -/
theorem frees_only_by_dissolving (r : NR1 Hd n nR nX) (i : ℕ) (l : W.Cell)
    (hb : l.2.2 ≠ none) (hfree : (W.eff1 r i l).2.2 = none) :
    ∃ hd, r = .inr (.inr (.inr (.inr (hd, false)))) := by
  obtain ⟨f, w, b⟩ := l
  rcases b with _ | ⟨y, h⟩
  · exact absurd rfl hb
  rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨a, c⟩ | ⟨food, dir⟩ | ⟨hd, dir⟩
  · simp [eff1] at hfree
  · rw [eff1_xch] at hfree
    simp only [xchEff] at hfree
    split_ifs at hfree <;> (try split at hfree) <;> simp_all
  · simp only [eff1] at hfree
    split_ifs at hfree <;> simp_all
  · rw [W.eff1_supply_bug] at hfree
    simp at hfree
  · cases dir
    · exact ⟨hd, rfl⟩
    · simp [eff1] at hfree

/-! ## Leaks: every held quantum needs an exit the organism cannot close

In a closed box, content that a part can hold for ever is lost to the rest (the infinite stream
`stream_never_dry` needs the world to keep flowing; with a shared pool the holders only grow,
G19a). The axioms give no permanent lock (P2b: every event has a positive reverse), but a hold can
be made as long as the organism likes in three ways, each closed below:
* an exit that needs a partner unit is shut when the partner is gone (`partner_exit_blocked`);
* an exit through the membrane is shut when the membrane is gone (`gated_exit_blocked`);
* an exit from a deep enough energy well is as slow as wanted (the factor `exp (-Φ)` below).
An ungated one-unit exit into the waste pool, with energies bounded by a physical scale `Φ`, has a
rate bounded below whatever the organism does (`leak_bound`). So every page that can hold content
needs such a leak, and the energy scale is physics, not selection. -/

/-- A pair rule whose partner state is absent leaves the organism as it is. -/
theorem partner_exit_blocked (k : Fin nR) (u v : Fin n) (dir : Bool) (y : Fin n → Λi)
    (hno : ∀ w, W.pageIn w = (W.spec k).j →
      y w ≠ (if dir then (W.spec k).b else (W.spec k).a).2) :
    W.innerAct k u v dir y = y := by
  unfold innerAct
  split_ifs with huv
  · rfl
  · have hp : (W.spec k).apply dir (W.pageIn u) (W.pageIn v) (y u, y v) = (y u, y v) := by
      unfold RuleSpec.apply
      rw [if_neg]
      rintro ⟨-, hj, hpe⟩
      exact hno v hj (by rw [← hpe])
    simp only [hp, Function.update_eq_self]

/-- A gated exchange has no conductance without an intact membrane. -/
theorem gated_exit_blocked (j : Fin nX) (hg : W.xgated j = true) (u : Fin n) (dir : Bool) (i : ℕ)
    (l : W.Cell) (hd : W.cellGate l = 0) : W.c1 (.inr (.inl (j, u, dir))) i l = 0 := by
  simp [c1, hg, hd]

theorem poolShift_one {V : ℕ} (w : Fin (V + 1)) (hw : (w : ℕ) < V) :
    ∃ w' : Fin (V + 1), poolShift w 1 = some w' ∧ (w' : ℕ) = w + 1 := by
  refine ⟨⟨w + 1, by omega⟩, ?_, rfl⟩
  unfold poolShift
  rw [dif_pos (by omega)]
  congr 1

/-- **Leak bound**: an ungated exchange that moves one quantum from a unit into the waste pool
fires at rate at least `κ * exp (-Φ - log V / 2)` whenever the unit holds that quantum and the
pool is not full, whatever the rest of the organism and the reservoir potential `μ` (the
exchange conserves the cell's content, so `μ` drops out). The bound depends only on physics. -/
theorem leak_bound (Φ : ℝ) (hΦ : ∀ p s, |W.φIn p s| ≤ Φ) (j : Fin nX) (hg : W.xgated j = false)
    (hf : W.xfood j = false) (hc : W.contentIn (W.xa j) = W.contentIn (W.xb j) + 1) (u : Fin n)
    (i : ℕ) (f w : Fin (W.V + 1)) (y : Fin n → Λi) (h : Hd)
    (hu : W.pageIn u = W.xpage j ∧ y u = W.xa j) (hw : (w : ℕ) < W.V) (μ : ℝ) :
    W.xκ j * Real.exp (-Φ - Real.log W.V / 2) ≤
      W.c1 (.inr (.inl (j, u, false))) i (f, w, some (y, h)) *
        Real.exp (-(W.φ i (W.eff1 (.inr (.inl (j, u, false))) i (f, w, some (y, h))) -
          W.φ i (f, w, some (y, h)) - μ * ((W.content (W.eff1 (.inr (.inl (j, u, false))) i
            (f, w, some (y, h))) : ℝ) - W.content (f, w, some (y, h)))) / 2) := by
  obtain ⟨w', hw', hwv⟩ := poolShift_one w hw
  have hd : ((W.contentIn (W.xa j) : ℤ) - W.contentIn (W.xb j)) = 1 := by rw [hc]; push_cast; ring
  have heff : W.eff1 (.inr (.inl (j, u, false))) i (f, w, some (y, h)) =
      (f, w', some (Function.update y u (W.xb j), h)) := by
    simp only [eff1, Bool.false_eq_true, if_false, hu, and_self, if_true, hf, hd, hw']
  rw [heff]
  have hcont : (W.content (f, w', some (Function.update y u (W.xb j), h)) : ℝ) =
      W.content (f, w, some (y, h)) := by
    have hs := sum_update_nat W.contentIn u (W.xb j) y
    rw [hu.2, hc] at hs
    simp only [content, bugContent]
    have : (((f : ℕ) + (w' : ℕ) + (W.B + ∑ v, W.contentIn (Function.update y u (W.xb j) v)) : ℕ) : ℤ) =
        (((f : ℕ) + (w : ℕ) + (W.B + ∑ v, W.contentIn (y v)) : ℕ) : ℤ) := by
      push_cast [hwv] at hs ⊢
      rw [hs]; push_cast; ring
    exact_mod_cast this
  have hE : ∑ v, W.φIn (W.pageIn v) (Function.update y u (W.xb j) v) - ∑ v, W.φIn (W.pageIn v) (y v) =
      W.φIn (W.pageIn u) (W.xb j) - W.φIn (W.pageIn u) (y u) :=
    LatticeCRN.sum_update (fun v l => W.φIn (W.pageIn v) l) u (W.xb j) y
  have hstep := Generation.cell_step W.V w hw
  have hphi : W.φ i (f, w', some (Function.update y u (W.xb j), h)) - W.φ i (f, w, some (y, h)) =
      Real.log (((w : ℕ) + 1 : ℝ) / (W.V - (w : ℕ))) +
        (W.φIn (W.pageIn u) (W.xb j) - W.φIn (W.pageIn u) (y u)) := by
    simp only [φ, bugE]
    rw [show ((w' : ℕ)) = (w : ℕ) + 1 from hwv]
    push_cast at hstep ⊢
    linarith
  have hb1 := hΦ (W.pageIn u) (W.xb j)
  have hb2 := hΦ (W.pageIn u) (y u)
  rw [abs_le] at hb1 hb2
  have hV : (1 : ℝ) ≤ W.V := by exact_mod_cast (show 1 ≤ W.V by omega)
  have hwV : ((w : ℕ) : ℝ) + 1 ≤ W.V := by exact_mod_cast hw
  have hden : (1 : ℝ) ≤ W.V - (w : ℕ) := by linarith
  have hr0 : (0 : ℝ) < ((w : ℕ) + 1 : ℝ) / (W.V - (w : ℕ)) := by positivity
  have hrV : ((w : ℕ) + 1 : ℝ) / (W.V - (w : ℕ)) ≤ W.V := by
    rw [div_le_iff₀ (by linarith)]; nlinarith
  have hlog := Real.log_le_log hr0 hrV
  have hc1 : W.c1 (.inr (.inl (j, u, false))) i (f, w, some (y, h)) = W.xκ j := by
    simp [c1, hg]
  rw [hc1, hcont, sub_self, mul_zero, sub_zero, hphi]
  apply mul_le_mul_of_nonneg_left _ (W.xκ_nonneg j)
  apply Real.exp_le_exp.mpr
  linarith

/-! ## Two instances: a ring and a torus -/

/-- Neighbours on a ring of `L` cells. -/
def ringAdj (L : ℕ) (u v : Fin L) : Prop :=
  u ≠ v ∧ ((u.val + 1) % L = v.val ∨ (v.val + 1) % L = u.val)

theorem ringAdj_symm (L : ℕ) (u v : Fin L) (h : ringAdj L u v) : ringAdj L v u :=
  ⟨h.1.symm, h.2.symm⟩

/-- Neighbours on a `G × G` torus (four-neighbourhood). -/
def torusAdj (G : ℕ) (u v : Fin G × Fin G) : Prop :=
  u ≠ v ∧ ((u.1 = v.1 ∧ ((u.2.val + 1) % G = v.2.val ∨ (v.2.val + 1) % G = u.2.val)) ∨
    (u.2 = v.2 ∧ ((u.1.val + 1) % G = v.1.val ∨ (v.1.val + 1) % G = u.1.val)))

theorem torusAdj_symm (G : ℕ) (u v : Fin G × Fin G) (h : torusAdj G u v) : torusAdj G v u := by
  refine ⟨h.1.symm, ?_⟩
  rcases h.2 with ⟨h1, h2⟩ | ⟨h1, h2⟩
  · exact Or.inl ⟨h1.symm, h2.symm⟩
  · exact Or.inr ⟨h1.symm, h2.symm⟩

/-- Page of a torus cell: `9 * kind + 3 * (x % 3) + y % 3` (kind and chirality code). -/
def torusPage (G : ℕ) (kind : Fin G × Fin G → ℕ) (u : Fin G × Fin G) : ℕ :=
  9 * kind u + 3 * (u.1.val % 3) + u.2.val % 3

/-- The neighbour ahead, read from the chirality codes of the two pages (heading 0: +x, 1: +y,
2: -x, 3: -y). With `G` a multiple of 3 it picks exactly the neighbour in that direction. -/
def torusAhead (i j : ℕ) (h : Fin 4) : Bool :=
  let ix := i % 9 / 3
  let iy := i % 3
  let jx := j % 9 / 3
  let jy := j % 3
  match h with
  | 0 => jx == (ix + 1) % 3 && jy == iy
  | 1 => jy == (iy + 1) % 3 && jx == ix
  | 2 => jx == (ix + 2) % 3 && jy == iy
  | 3 => jy == (iy + 2) % 3 && jx == ix

open Classical in
/-- **The two-dimensional world is a model of the axioms**: organisms on a torus, food pellets
from the world's boundary, diffusion, waste sinks, driven movement in four headings. -/
theorem axioms_torus [Fintype Λi] [Fintype Hd] (G : ℕ) (kind : Fin G × Fin G → ℕ)
    (y0 : Fin n → Λi) (h0 : Hd)
    (hspec : ∀ k, W.contentIn (W.spec k).a.1 + W.contentIn (W.spec k).a.2 =
      W.contentIn (W.spec k).b.1 + W.contentIn (W.spec k).b.2)
    (hfuel : W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 =
      W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2) (hgate : ∀ j, W.xgated j = true → W.xpage j ≠ W.gatePage)
    (hEmpty : ∀ u, W.contentIn (W.yEmpty u) = 0) :
    (W.lattice (torusAdj G) (torusAdj_symm G) (fun _ h => h.1 rfl) (torusPage G kind) y0 h0).medium.Axioms :=
  W.axioms _ _ _ _ y0 h0 hspec hfuel hgate hEmpty

open Classical in
/-- The one-dimensional ring is a model too. -/
theorem axioms_ring [Fintype Λi] [Fintype Hd] (L : ℕ) (page : Fin L → ℕ) (y0 : Fin n → Λi) (h0 : Hd)
    (hspec : ∀ k, W.contentIn (W.spec k).a.1 + W.contentIn (W.spec k).a.2 =
      W.contentIn (W.spec k).b.1 + W.contentIn (W.spec k).b.2)
    (hfuel : W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 =
      W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2) (hgate : ∀ j, W.xgated j = true → W.xpage j ≠ W.gatePage)
    (hEmpty : ∀ u, W.contentIn (W.yEmpty u) = 0) :
    (W.lattice (ringAdj L) (ringAdj_symm L) (fun _ h => h.1 rfl) page y0 h0).medium.Axioms :=
  W.axioms _ _ _ _ y0 h0 hspec hfuel hgate hEmpty

end NestedWorld

end CGB
