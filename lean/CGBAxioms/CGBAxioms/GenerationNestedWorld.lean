import CGBAxioms.GenerationLattice
import CGBAxioms.GenerationNesting
import CGBAxioms.GenerationTable
import CGBAxioms.NestedLife

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
* neighbour cells: one quantum of food or waste diffusing; the organism moving along its heading
  while one fuel rule fires inside it (moving forward spends fuel, moving back would return it:
  self-propulsion is driven). A move exchanges the mover with whatever occupies the cell ahead: the
  vacuum (an empty cell) or another body, which takes the mover's place unchanged
  (`move_exchanges`). The vacuum is not special: restricting moves to empty cells would be an extra
  rule, and it jams the inside of every colony grown by division (eco-sphere audit 2026-10-27).

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
structure NestedWorld (Λi Hd : Type*) (n nR nX nB : ℕ) where
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
  /-- pool coupling of pair rule `k`: in its forward direction (`a ↦ b`) the cell's food and waste
  pools change by `sdf k` and `sdw k` quanta. `0, 0` is a rule inside the body; any other pair
  crosses the boundary (an antiport, a symport, a pump: one grammar for all of them), and if
  `sgated k` it goes through the membrane -/
  sdf : Fin nR → ℤ
  sdw : Fin nR → ℤ
  sgated : Fin nR → Bool
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
  /-- chemical energy of a food quantum relative to a waste quantum: food carries free energy, not
  only matter (the injection at the boundary is free energy) -/
  foodE : ℝ
  /-- `ahead i j h`: the cell with page `j` is the neighbour ahead of the cell with page `i` for
  heading `h` (read from identity pages only, so P2c holds) -/
  ahead : ℕ → ℕ → Hd → Bool
  /-- the membrane: exchanges through the boundary go through intact units of this page (state
  `gateOn`); their conductance is proportional to how many there are (`gateCount`) -/
  gatePage : ℕ
  gateOn : Λi
  /-- dissolution: a body whose membrane is wholly broken dissolves into the cell's waste pool
  with everything it holds (`B` plus its inner content); the reverse forms such a body from waste.
  The conductance is `κdis` for a body without intact membrane, 0 otherwise. -/
  κdis : ℝ
  κdis_nonneg : 0 ≤ κdis
  /-- division by recipe `rb`: a living body (intact membrane) puts a newborn in state `yb rb` into
  an empty neighbour, paying for the newborn's content (`B` plus its inner content) with units of
  page `divPage rb` going `divFrom rb → divTo rb`, one quantum each; the reverse is two neighbours
  fusing. Recipes are how a lineage passes itself on: when inner states carry the unit's chemistry
  (its genotype), a recipe drains units of the parent's genotype and gives the newborn a genotype
  of its own; the recipes' conductances are the mutation kernel (they depend only on the event,
  so reversibility holds for every kernel) -/
  divPage : Fin nB → ℕ
  divFrom : Fin nB → Λi
  divTo : Fin nB → Λi
  yb : Fin nB → Fin n → Λi
  κdiv : Fin nB → ℝ
  κdiv_nonneg : ∀ rb, 0 ≤ κdiv rb

namespace NestedWorld

variable {Λi Hd : Type*} [DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ} (W : NestedWorld Λi Hd n nR nX nB)

/-- A cell: food, waste, and possibly an organism (inner states and heading). -/
abbrev Cell := Fin (W.V + 1) × Fin (W.V + 1) × Option ((Fin n → Λi) × Hd)

/-- Single-cell rules: inner pair rule, pool exchange, turn (heading a to b), boundary supply,
dissolution of a body in inner state `y` with heading `h` (and its reverse). -/
abbrev NR1 (Λi Hd : Type*) (n nR nX : ℕ) :=
  (Fin nR × Fin n × Fin n × Bool) ⊕ (Fin nX × Fin n × Bool) ⊕ (Hd × Hd) ⊕
    ((Bool × Bool) ⊕ (Hd × (Fin n → Λi) × Bool))

/-- Neighbour rules: diffusion of food or waste, movement with fuel units. -/
abbrev NR2 (Hd : Type*) (n nB : ℕ) := (Bool × Bool) ⊕ (Fin n × Fin n × Bool) ⊕ (Finset (Fin n) × Fin nB × Hd × Bool)

/-- Content of an organism, as an integer (what dissolution releases). -/
def bodyQ (y : Fin n → Λi) : ℤ := W.B + ∑ u, (W.contentIn (y u) : ℤ)

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
  Generation.cellF W.V l.1 + W.foodE * (l.1 : ℕ) + Generation.cellF W.V l.2.1 + match l.2.2 with
    | none => 0
    | some (y, _) => W.bugE y

/-- Pool changes (food, waste) of pair rule `k` in direction `dir`. -/
def sdir (k : Fin nR) (dir : Bool) : ℤ × ℤ := if dir then (-W.sdf k, -W.sdw k) else (W.sdf k, W.sdw k)

theorem sdir_not (k : Fin nR) (dir : Bool) :
    W.sdir k (!dir) = (-(W.sdir k dir).1, -(W.sdir k dir).2) := by
  cases dir <;> simp [sdir]

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
def eff1 : NR1 Λi Hd n nR nX → ℕ → W.Cell → W.Cell
  | .inl (k, u, v, dir), _, l => match l with
      | (f, w, some (y, h)) =>
          if W.innerAct k u v dir y = y then l else
          match poolShift f (W.sdir k dir).1, poolShift w (W.sdir k dir).2 with
          | some f', some w' => (f', w', some (W.innerAct k u v dir y, h))
          | _, _ => l
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
  | .inr (.inr (.inr (.inr (hd, yd, dir)))), _, l =>
      if dir then
        match l with
        | (f, w, none) => match poolShift w (-W.bodyQ yd) with
          | some w' => (f, w', some (yd, hd))
          | none => l
        | l => l
      else
        match l with
        | (f, w, some (y, h)) =>
            if y = yd ∧ h = hd then
              match poolShift w (W.bodyQ yd) with
              | some w' => (f, w', none)
              | none => l
            else l
        | l => l

/-- Units of `S` set to state `b`. -/
def drain (S : Finset (Fin n)) (b : Λi) (y : Fin n → Λi) : Fin n → Λi := fun v => if v ∈ S then b else y v

/-- Every unit of `S` is on the division page in state `a`. -/
def canDrain (rb : Fin nB) (S : Finset (Fin n)) (a : Λi) (y : Fin n → Λi) : Prop :=
  ∀ v ∈ S, W.pageIn v = W.divPage rb ∧ y v = a

instance (rb : Fin nB) (S : Finset (Fin n)) (a : Λi) (y : Fin n → Λi) : Decidable (W.canDrain rb S a y) := by
  unfold canDrain; infer_instance

/-- The content a newborn of recipe `rb` needs: its body and its inner content. -/
def bornQ (rb : Fin nB) : ℕ := W.B + ∑ v, W.contentIn (W.yb rb v)

omit [DecidableEq Hd] in
theorem drain_content (S : Finset (Fin n)) (a b : Λi) (y : Fin n → Λi) (hy : ∀ v ∈ S, y v = a) :
    ∑ v, W.contentIn (drain S b y v) + S.card * W.contentIn a =
      ∑ v, W.contentIn (y v) + S.card * W.contentIn b := by
  have hpt : ∀ v, W.contentIn (drain S b y v) + (if v ∈ S then W.contentIn a else 0) =
      W.contentIn (y v) + (if v ∈ S then W.contentIn b else 0) := fun v => by
    unfold drain
    by_cases hv : v ∈ S
    · simp [hv, hy v hv]; ring
    · simp [hv]
  have hs : ∀ k : ℕ, ∑ v : Fin n, (if v ∈ S then k else 0) = S.card * k := fun k => by
    rw [Finset.sum_ite_mem, Finset.univ_inter, Finset.sum_const, smul_eq_mul]
  rw [← hs, ← hs, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun v _ => hpt v

omit [DecidableEq Λi] [DecidableEq Hd] in
theorem drain_drain (S : Finset (Fin n)) (a b : Λi) (y : Fin n → Λi) (hy : ∀ v ∈ S, y v = a) :
    drain S a (drain S b y) = y := by
  funext v
  unfold drain
  by_cases hv : v ∈ S
  · simp [hv, hy v hv]
  · simp [hv]

theorem canDrain_drain {rb : Fin nB} {S : Finset (Fin n)} {a : Λi} {y : Fin n → Λi} (h : W.canDrain rb S a y) (b : Λi) :
    W.canDrain rb S b (drain S b y) := fun v hv => ⟨(h v hv).1, by simp [drain, hv]⟩

theorem gateCount_drain {rb : Fin nB} (hdg : W.divPage rb ≠ W.gatePage) {S : Finset (Fin n)} {a : Λi}
    {y : Fin n → Λi} (h : W.canDrain rb S a y) (b : Λi) : W.gateCount (drain S b y) = W.gateCount y := by
  unfold gateCount
  congr 1
  apply Finset.filter_congr
  intro v _
  by_cases hv : v ∈ S
  · have hp := (h v hv).1
    have : W.pageIn v ≠ W.gatePage := hp ▸ hdg
    simp [drain, hv, this]
  · simp [drain, hv]

/-- Effect of a neighbour rule on the ordered pair of cells with pages `i`, `j`. -/
def eff2 : NR2 Hd n nB → ℕ → ℕ → W.Cell × W.Cell → W.Cell × W.Cell
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
  | .inr (.inl (e, d, dir)), i, j, (l, l') =>
      if dir then
        match l.2.2, l'.2.2 with
        | none, some (y, h) =>
            let y' := W.fuelAct e d true y
            if W.ahead i j h = true ∧ y' ≠ y ∧ y W.motor = W.run ∧ y' W.motor = W.run then
              ((l.1, l.2.1, some (y', h)), (l'.1, l'.2.1, none))
            else (l, l')
        | some (z, hz), some (y, h) =>
            let y' := W.fuelAct e d true y
            if W.ahead i j h = true ∧ y' ≠ y ∧ y W.motor = W.run ∧ y' W.motor = W.run then
              ((l.1, l.2.1, some (y', h)), (l'.1, l'.2.1, some (z, hz)))
            else (l, l')
        | _, _ => (l, l')
      else
        match l.2.2, l'.2.2 with
        | some (y, h), none =>
            let y' := W.fuelAct e d false y
            if W.ahead i j h = true ∧ y' ≠ y ∧ y W.motor = W.run ∧ y' W.motor = W.run then
              ((l.1, l.2.1, none), (l'.1, l'.2.1, some (y', h)))
            else (l, l')
        | some (y, h), some (z, hz) =>
            let y' := W.fuelAct e d false y
            if W.ahead i j h = true ∧ y' ≠ y ∧ y W.motor = W.run ∧ y' W.motor = W.run then
              ((l.1, l.2.1, some (z, hz)), (l'.1, l'.2.1, some (y', h)))
            else (l, l')
        | _, _ => (l, l')
  | .inr (.inr (S, rb, h2, dir)), _, _, (l, l') =>
      if dir then
        match l.2.2, l'.2.2 with
        | some (y, h), some (y2, h2') =>
            if y2 = (W.yb rb) ∧ h2' = h2 ∧ S.card = (W.bornQ rb) ∧ W.canDrain rb S (W.divTo rb) y then
              ((l.1, l.2.1, some (drain S (W.divFrom rb) y, h)), (l'.1, l'.2.1, none))
            else (l, l')
        | _, _ => (l, l')
      else
        match l.2.2, l'.2.2 with
        | some (y, h), none =>
            if S.card = (W.bornQ rb) ∧ W.canDrain rb S (W.divFrom rb) y then
              ((l.1, l.2.1, some (drain S (W.divTo rb) y, h)), (l'.1, l'.2.1, some ((W.yb rb), h2)))
            else (l, l')
        | _, _ => (l, l')

/-- **A move is an exchange**: whatever occupies the cell ahead (vacuum or a body), a running mover
whose fuel rule fires takes its place and the occupant takes the mover's, unchanged; pools stay. -/
theorem move_exchanges (e d : Fin n) (i j : ℕ) (f w f' w' : Fin (W.V + 1))
    (o : Option ((Fin n → Λi) × Hd)) (y : Fin n → Λi) (h : Hd)
    (hc : W.ahead i j h = true ∧ W.fuelAct e d true y ≠ y ∧ y W.motor = W.run ∧
      W.fuelAct e d true y W.motor = W.run) :
    W.eff2 (.inr (.inl (e, d, true))) i j ((f, w, o), (f', w', some (y, h))) =
      ((f, w, some (W.fuelAct e d true y, h)), (f', w', o)) := by
  rcases o with _ | ⟨z, hz⟩ <;> simp [eff2, hc]

def rev1 : NR1 Λi Hd n nR nX → NR1 Λi Hd n nR nX
  | .inl (k, u, v, dir) => .inl (k, u, v, !dir)
  | .inr (.inl (j, u, dir)) => .inr (.inl (j, u, !dir))
  | .inr (.inr (.inl (a, b))) => .inr (.inr (.inl (b, a)))
  | .inr (.inr (.inr (.inl (food, dir)))) => .inr (.inr (.inr (.inl (food, !dir))))
  | .inr (.inr (.inr (.inr (hd, yd, dir)))) => .inr (.inr (.inr (.inr (hd, yd, !dir))))

def rev2 : NR2 Hd n nB → NR2 Hd n nB
  | .inl (food, dir) => .inl (food, !dir)
  | .inr (.inl (e, d, dir)) => .inr (.inl (e, d, !dir))
  | .inr (.inr (S, rb, h2, dir)) => .inr (.inr (S, rb, h2, !dir))

/-- Pool multiplicity factor of a change of a pool from `a` to `b` quanta (NESTING_SYSTEM_MAP 7.1:
lumped rates are `κ √(source multiplicity · target multiplicity) e^{-ΔE/2}`): symmetric in `a`, `b`,
equal to 1 when the pool does not change. A one-quantum step `m → m + 1` gets `√((m+1)(V-m))`,
which turns the Boltzmann factor of `cellF` into mass action: a pool gains at rate `∝ V - m` (free
places) and loses at rate `∝ m + 1` (its quanta), every quantum moving on its own. -/
noncomputable def poolCond (V a b : ℕ) : ℝ :=
  Real.sqrt (∏ k ∈ Finset.Ico (min a b) (max a b), (((k : ℝ) + 1) * ((V : ℝ) - k)))

theorem poolCond_symm (V a b : ℕ) : poolCond V a b = poolCond V b a := by
  unfold poolCond; rw [min_comm, max_comm]

theorem poolCond_nonneg (V a b : ℕ) : 0 ≤ poolCond V a b := Real.sqrt_nonneg _

/-- Both pools of a cell. -/
noncomputable def poolFactor (l l' : W.Cell) : ℝ :=
  poolCond W.V l.1 l'.1 * poolCond W.V l.2.1 l'.2.1

theorem poolFactor_symm (l l' : W.Cell) : W.poolFactor l l' = W.poolFactor l' l := by
  unfold poolFactor; rw [poolCond_symm, poolCond_symm W.V l.2.1]

theorem poolFactor_nonneg (l l' : W.Cell) : 0 ≤ W.poolFactor l l' :=
  mul_nonneg (poolCond_nonneg _ _ _) (poolCond_nonneg _ _ _)

theorem poolCond_pos (V a b : ℕ) (ha : a ≤ V) (hb : b ≤ V) : 0 < poolCond V a b := by
  unfold poolCond
  apply Real.sqrt_pos.mpr
  apply Finset.prod_pos
  intro k hk
  rw [Finset.mem_Ico] at hk
  have : k < V := lt_of_lt_of_le hk.2 (max_le ha hb)
  have : (k : ℝ) < V := by exact_mod_cast this
  exact mul_pos (by positivity) (by linarith)

/-- Pools are bounded by `V`, so every pool change has positive multiplicity. -/
theorem poolFactor_pos (l l' : W.Cell) : 0 < W.poolFactor l l' :=
  mul_pos (poolCond_pos _ _ _ (Nat.lt_succ_iff.mp l.1.2) (Nat.lt_succ_iff.mp l'.1.2))
    (poolCond_pos _ _ _ (Nat.lt_succ_iff.mp l.2.1.2) (Nat.lt_succ_iff.mp l'.2.1.2))

theorem poolCond_self (V a : ℕ) : poolCond V a a = 1 := by
  simp [poolCond]

theorem poolCond_step (V m : ℕ) : poolCond V m (m + 1) = Real.sqrt (((m : ℝ) + 1) * ((V : ℝ) - m)) := by
  unfold poolCond
  rw [min_eq_left (Nat.le_succ m), max_eq_right (Nat.le_succ m), Nat.Ico_succ_singleton,
    Finset.prod_singleton]

/-- **Mass action** from 7.1: with the entropic pool energy `cellF`, the lumped conductance makes a
one-quantum gain of a pool with `m` quanta fire at a rate `∝ V - m` (its free places) ... -/
theorem mass_action_up (V m : ℕ) (hm : m < V) (D : ℝ) :
    poolCond V m (m + 1) *
        Real.exp (-(Generation.cellF V (m + 1) - Generation.cellF V m + D) / 2) =
      ((V : ℝ) - m) * Real.exp (-D / 2) := by
  have ha : (0 : ℝ) < (m : ℝ) + 1 := by positivity
  have hb : (0 : ℝ) < (V : ℝ) - m := by
    have : (m : ℝ) < V := by exact_mod_cast hm
    linarith
  rw [poolCond_step, Generation.cell_step V m hm, ← Real.exp_log (mul_pos ha hb), ← Real.exp_half,
    Real.log_mul ha.ne' hb.ne', Real.log_div ha.ne' hb.ne']
  conv_rhs => rw [← Real.exp_log hb]
  rw [← Real.exp_add, ← Real.exp_add]
  congr 1
  ring

/-- ... and a one-quantum loss of a pool with `m + 1` quanta at a rate `∝ m + 1` (its quanta). -/
theorem mass_action_down (V m : ℕ) (hm : m < V) (D : ℝ) :
    poolCond V (m + 1) m *
        Real.exp (-(Generation.cellF V m - Generation.cellF V (m + 1) + D) / 2) =
      ((m : ℝ) + 1) * Real.exp (-D / 2) := by
  have ha : (0 : ℝ) < (m : ℝ) + 1 := by positivity
  have hb : (0 : ℝ) < (V : ℝ) - m := by
    have : (m : ℝ) < V := by exact_mod_cast hm
    linarith
  have hs : Generation.cellF V m - Generation.cellF V (m + 1) =
      -(Generation.cellF V (m + 1) - Generation.cellF V m) := by ring
  rw [poolCond_symm, poolCond_step, hs, Generation.cell_step V m hm, ← Real.exp_log (mul_pos ha hb),
    ← Real.exp_half, Real.log_mul ha.ne' hb.ne', Real.log_div ha.ne' hb.ne']
  conv_rhs => rw [← Real.exp_log ha]
  rw [← Real.exp_add, ← Real.exp_add]
  congr 1
  ring

/-- **Uniport selectivity is fixed by free energy**: a reversible exchange that moves one food
quantum into the body (inner energy change `D`, food energy included) fills at food `m + 1` and
drains back at food `q`; the ratio of the two mass-action rates is `(m + 1) / (V - q) · e^{-D}`,
whatever its conductance `κ`. Selection can make the door slow or fast, never selective. -/
theorem uniport_selectivity (V m q : ℕ) (hm : m < V) (hq : q < V) {κ : ℝ} (hκ : 0 < κ) (D : ℝ) :
    (κ * poolCond V (m + 1) m * Real.exp (-(Generation.cellF V m - Generation.cellF V (m + 1) + D) / 2)) /
      (κ * poolCond V q (q + 1) * Real.exp (-(Generation.cellF V (q + 1) - Generation.cellF V q + -D) / 2)) =
      ((m : ℝ) + 1) / ((V : ℝ) - q) * Real.exp (-D) := by
  have hb : (0 : ℝ) < (V : ℝ) - q := by
    have : (q : ℝ) < V := by exact_mod_cast hq
    linarith
  rw [mul_assoc, mul_assoc, mass_action_down V m hm, mass_action_up V q hq]
  have he : Real.exp (-D) = Real.exp (-D / 2) / Real.exp (-(-D) / 2) := by
    rw [← Real.exp_sub]; congr 1; ring
  rw [he]
  field_simp

/-- **Antiport selectivity**: the same exchange coupled to the waste pool (one food quantum in, one
waste quantum out, inner energy change `D`) fills at food `m + 1`, waste `w` and drains at food `q`,
waste `z + 1`; the ratio is `(m + 1)(V - w) / ((V - q)(z + 1)) · e^{-D}`. Where the world keeps
waste low (`z` small), the door drains `~ V / (z + 1)` times less than a uniport with the same
`D`: dissipating the food-to-waste step at the door makes it one-way. -/
theorem antiport_selectivity (V m q w z : ℕ) (hm : m < V) (hq : q < V) (hw : w < V) (hz : z < V)
    {κ : ℝ} (hκ : 0 < κ) (D : ℝ) :
    (κ * (poolCond V (m + 1) m * poolCond V w (w + 1)) *
        Real.exp (-((Generation.cellF V m - Generation.cellF V (m + 1)) +
          ((Generation.cellF V (w + 1) - Generation.cellF V w) + D)) / 2)) /
      (κ * (poolCond V q (q + 1) * poolCond V (z + 1) z) *
        Real.exp (-((Generation.cellF V (q + 1) - Generation.cellF V q) +
          ((Generation.cellF V z - Generation.cellF V (z + 1)) + -D)) / 2)) =
      ((m : ℝ) + 1) * ((V : ℝ) - w) / (((V : ℝ) - q) * ((z : ℝ) + 1)) * Real.exp (-D) := by
  have hb : (0 : ℝ) < (V : ℝ) - q := by
    have : (q : ℝ) < V := by exact_mod_cast hq
    linarith
  have hbw : (0 : ℝ) < (V : ℝ) - w := by
    have : (w : ℝ) < V := by exact_mod_cast hw
    linarith
  have split1 : ∀ (A B C : ℝ), Real.exp (-(A + (B + C)) / 2) =
      Real.exp (-(A + 0) / 2) * Real.exp (-(B + C) / 2) := fun A B C => by
    rw [← Real.exp_add]; congr 1; ring
  rw [split1, split1]
  have hn : κ * (poolCond V (m + 1) m * poolCond V w (w + 1)) *
      (Real.exp (-(Generation.cellF V m - Generation.cellF V (m + 1) + 0) / 2) *
        Real.exp (-(Generation.cellF V (w + 1) - Generation.cellF V w + D) / 2)) =
      κ * (((m : ℝ) + 1) * Real.exp (-0 / 2)) * (((V : ℝ) - w) * Real.exp (-D / 2)) := by
    rw [← mass_action_down V m hm 0, ← mass_action_up V w hw D]; ring
  have hd : κ * (poolCond V q (q + 1) * poolCond V (z + 1) z) *
      (Real.exp (-(Generation.cellF V (q + 1) - Generation.cellF V q + 0) / 2) *
        Real.exp (-(Generation.cellF V z - Generation.cellF V (z + 1) + -D) / 2)) =
      κ * (((V : ℝ) - q) * Real.exp (-0 / 2)) * (((z : ℝ) + 1) * Real.exp (-(-D) / 2)) := by
    rw [← mass_action_up V q hq 0, ← mass_action_down V z hz (-D)]; ring
  rw [hn, hd]
  have he : Real.exp (-D) = Real.exp (-D / 2) / Real.exp (-(-D) / 2) := by
    rw [← Real.exp_sub]; congr 1; ring
  rw [he]
  simp only [neg_zero, zero_div, Real.exp_zero, mul_one]
  field_simp

noncomputable def c1 : NR1 Λi Hd n nR nX → ℕ → W.Cell → ℝ
  | .inl (k, u, v, dir), i, l => (W.spec k).κ * (if W.sgated k then (W.cellGate l : ℝ) else 1) *
      W.poolFactor l (W.eff1 (.inl (k, u, v, dir)) i l)
  | .inr (.inl (j, u, dir)), i, l => W.xκ j * (if W.xgated j then (W.cellGate l : ℝ) else 1) *
      W.poolFactor l (W.eff1 (.inr (.inl (j, u, dir))) i l)
  | .inr (.inr (.inl _)), _, _ => W.κflip
  | .inr (.inr (.inr (.inl (food, dir)))), i, l =>
      W.κin * W.poolFactor l (W.eff1 (.inr (.inr (.inr (.inl (food, dir))))) i l)
  | .inr (.inr (.inr (.inr (hd, yd, dir)))), i, l =>
      (if W.gateCount yd = 0 then W.κdis else 0) *
        Real.exp (-|W.φ i (W.eff1 (.inr (.inr (.inr (.inr (hd, yd, dir))))) i l) - W.φ i l| / 2)

noncomputable def c2 : NR2 Hd n nB → ℕ → ℕ → W.Cell × W.Cell → ℝ
  | .inl r, i, j, p => W.κdiff * (W.poolFactor p.1 (W.eff2 (.inl r) i j p).1 *
      W.poolFactor p.2 (W.eff2 (.inl r) i j p).2)
  | .inr (.inl _), _, _, _ => W.κmove
  | .inr (.inr (S, rb, h2, dir)), i, j, p =>
      W.κdiv rb * (if 0 < W.cellGate p.1 then 1 else 0) *
        Real.exp (-|(W.φ i (W.eff2 (.inr (.inr (S, rb, h2, dir))) i j p).1 +
          W.φ j (W.eff2 (.inr (.inr (S, rb, h2, dir))) i j p).2) - (W.φ i p.1 + W.φ j p.2)| / 2)

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

theorem gateCount_update {u : Fin n} (hu : W.pageIn u ≠ W.gatePage) (b : Λi) (y : Fin n → Λi) :
    W.gateCount (Function.update y u b) = W.gateCount y := by
  unfold gateCount
  congr 1
  apply Finset.filter_congr
  intro v _
  by_cases hv : v = u
  · subst hv; simp [hu]
  · simp [Function.update_of_ne hv]

/-- Content balance of the pair rules with the pools they couple to (P1b for coupled rules). -/
def SpecBalanced : Prop := ∀ k, (W.contentIn (W.spec k).a.1 : ℤ) + W.contentIn (W.spec k).a.2 =
    W.contentIn (W.spec k).b.1 + W.contentIn (W.spec k).b.2 + W.sdf k + W.sdw k

/-- A pair rule that fired moved the content the pools gave up into the body (or back). -/
theorem innerAct_delta (hspec : W.SpecBalanced) (k : Fin nR) (u v : Fin n) (dir : Bool)
    (y : Fin n → Λi) (hne : W.innerAct k u v dir y ≠ y) :
    ((∑ w, W.contentIn (W.innerAct k u v dir y w) : ℕ) : ℤ) + (W.sdir k dir).1 + (W.sdir k dir).2 =
      ((∑ w, W.contentIn (y w) : ℕ) : ℤ) := by
  unfold innerAct at hne ⊢
  split_ifs at hne ⊢ with huv
  · exact absurd rfl hne
  · set p := (W.spec k).apply dir (W.pageIn u) (W.pageIn v) (y u, y v) with hpdef
    have hp : p ≠ (y u, y v) := by
      intro e; apply hne; rw [e]
      funext w
      by_cases hwv : w = v
      · subst hwv; simp
      · by_cases hwu : w = u
        · subst hwu; simp [Function.update_of_ne hwv]
        · simp [Function.update_of_ne hwv, Function.update_of_ne hwu]
    have hm : W.pageIn u = (W.spec k).i ∧ W.pageIn v = (W.spec k).j ∧
        (y u, y v) = (if dir then (W.spec k).b else (W.spec k).a) := by
      by_contra hm; apply hp; rw [hpdef]; unfold RuleSpec.apply; rw [if_neg hm]
    have hpv : p = (if dir then (W.spec k).a else (W.spec k).b) := by
      rw [hpdef]; unfold RuleSpec.apply; rw [if_pos hm]
    have hs := LatticeCRN.sum_update2 (fun _ l => (W.contentIn l : ℝ)) huv p.1 p.2 y
    have hu : y u = (if dir then (W.spec k).b else (W.spec k).a).1 := by rw [← hm.2.2]
    have hv : y v = (if dir then (W.spec k).b else (W.spec k).a).2 := by rw [← hm.2.2]
    have hk := hspec k
    have hR : ((∑ w, W.contentIn (Function.update (Function.update y u p.1) v p.2 w) : ℕ) : ℝ) +
        (W.sdir k dir).1 + (W.sdir k dir).2 = ((∑ w, W.contentIn (y w) : ℕ) : ℝ) := by
      rw [hpv]
      push_cast
      rw [hu, hv, hpv] at hs
      have hk' : ((W.contentIn (W.spec k).a.1 : ℤ) : ℝ) + ((W.contentIn (W.spec k).a.2 : ℤ) : ℝ) =
          ((W.contentIn (W.spec k).b.1 : ℤ) : ℝ) + ((W.contentIn (W.spec k).b.2 : ℤ) : ℝ) +
            ((W.sdf k : ℤ) : ℝ) + ((W.sdw k : ℤ) : ℝ) := by exact_mod_cast congrArg (fun z : ℤ => (z : ℝ)) hk
      push_cast at hk'
      cases dir <;> simp only [sdir, Bool.false_eq_true, if_false, if_true] at hs ⊢ <;> push_cast <;> linarith
    exact_mod_cast hR

/-- A pair rule away from the membrane page leaves the membrane as it is. -/
theorem gateCount_innerAct (k : Fin nR) (u v : Fin n) (dir : Bool) (y : Fin n → Λi)
    (hi : (W.spec k).i ≠ W.gatePage) (hj : (W.spec k).j ≠ W.gatePage) :
    W.gateCount (W.innerAct k u v dir y) = W.gateCount y := by
  unfold innerAct
  split_ifs with huv
  · rfl
  · by_cases hm : W.pageIn u = (W.spec k).i ∧ W.pageIn v = (W.spec k).j ∧
        (y u, y v) = (if dir then (W.spec k).b else (W.spec k).a)
    · dsimp only
      rw [W.gateCount_update (by rw [hm.2.1]; exact hj), W.gateCount_update (by rw [hm.1]; exact hi)]
    · have hp : (W.spec k).apply dir (W.pageIn u) (W.pageIn v) (y u, y v) = (y u, y v) := by
        unfold RuleSpec.apply; rw [if_neg hm]
      simp only [hp, Function.update_eq_self]

/-! ## The joint medium -/

/-- Identity pages are `9 * kind + local position code` (`< 9`): kind 0 plain, 1 food source,
2 waste sink (the world's own boundary), 3 birth place. -/
def bidW : Finset ℕ := Finset.Icc 9 26

/-- Birth: an organism in state `y0` with heading `h0` on cells of kind 3, every other cell
empty. -/
def birthCell (y0 : Fin n → Λi) (h0 : Hd) (i : ℕ) : W.Cell :=
  if i / 9 = 3 then (0, 0, some (y0, h0)) else (0, 0, none)

theorem c1_nonneg (r : NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell) : 0 ≤ W.c1 r i l := by
  rcases r with ⟨k, u, v, dir⟩ | ⟨j, -, -⟩ | ⟨-, -⟩ | ⟨-, -⟩ | ⟨-, yd, -⟩
  · exact mul_nonneg (mul_nonneg (W.spec k).κ_nonneg (by split_ifs <;> positivity)) (W.poolFactor_nonneg _ _)
  · exact mul_nonneg (mul_nonneg (W.xκ_nonneg j) (by split_ifs <;> positivity)) (W.poolFactor_nonneg _ _)
  · exact W.κflip_nonneg
  · exact mul_nonneg W.κin_nonneg (W.poolFactor_nonneg _ _)
  · simp only [c1]
    refine mul_nonneg ?_ (Real.exp_pos _).le
    split_ifs
    · exact W.κdis_nonneg
    · exact le_rfl

theorem c2_nonneg (r : NR2 Hd n nB) (i j : ℕ) (p : W.Cell × W.Cell) : 0 ≤ W.c2 r i j p := by
  rcases r with r | ⟨-, -, -⟩ | ⟨S, rb, h2, dir⟩
  · exact mul_nonneg W.κdiff_nonneg (mul_nonneg (W.poolFactor_nonneg _ _) (W.poolFactor_nonneg _ _))
  · exact W.κmove_nonneg
  · simp only [c2]
    refine mul_nonneg (mul_nonneg (W.κdiv_nonneg rb) ?_) (Real.exp_pos _).le
    split_ifs <;> norm_num

/-- The joint medium of organisms and their world on cells `U` with a symmetric, irreflexive
neighbour relation. -/
noncomputable def lattice {U : Type*} (adj : U → U → Prop) (hsymm : ∀ u v, adj u v → adj v u)
    (hirr : ∀ u, ¬ adj u u) (page : U → ℕ) (y0 : Fin n → Λi) (h0 : Hd) :
    LatticeCRN U W.Cell ℝ (NR1 Λi Hd n nR nX) (NR2 Hd n nB) where
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

theorem cons1 (hspec : W.SpecBalanced)
    (r : NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell) (hi : i ∉ bidW) :
    W.content (W.eff1 r i l) = W.content l := by
  obtain ⟨f, w, b⟩ := l
  rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨ha, hb⟩ | ⟨food, dir⟩ | ⟨hd, yd, dir⟩
  · rcases b with _ | ⟨y, h⟩
    · rfl
    · simp only [eff1]
      split_ifs with hy
      · rfl
      · have hd := W.innerAct_delta hspec k u v dir y hy
        rcases hf : poolShift f (W.sdir k dir).1 with _ | f' <;>
          rcases hw : poolShift w (W.sdir k dir).2 with _ | w'
        · rfl
        · rfl
        · rfl
        · have hv1 := poolShift_val hf
          have hv2 := poolShift_val hw
          simp only [content, bugContent]
          zify at hv1 hv2 ⊢
          push_cast at hd ⊢
          linarith
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
  · have hq : W.bodyQ yd = ((W.bugContent yd : ℕ) : ℤ) := by
      simp only [bodyQ, bugContent]; push_cast; ring
    rcases b with _ | ⟨y, h⟩ <;> cases dir
    · rfl
    · rcases hps : poolShift w (-W.bodyQ yd) with _ | w'
      · simp only [eff1, if_true, hps]
      · have hv := poolShift_val hps
        simp only [eff1, if_true, hps, content]
        rw [hq] at hv
        omega
    · by_cases hc : y = yd ∧ h = hd
      · rcases hps : poolShift w (W.bodyQ yd) with _ | w'
        · simp only [eff1, Bool.false_eq_true, if_false, if_pos hc, hps]
        · have hv := poolShift_val hps
          obtain ⟨rfl, rfl⟩ := hc
          simp only [eff1, Bool.false_eq_true, if_false, and_self, if_true, hps, content]
          rw [hq] at hv
          omega
      · simp only [eff1, Bool.false_eq_true, if_false, if_neg hc]
    · rfl

theorem cons2 (hfuel : W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 =
      W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2)
    (hdiv : ∀ rb, W.contentIn (W.divFrom rb) = W.contentIn (W.divTo rb) + 1)
    (r : NR2 Hd n nB) (i j : ℕ) (p : W.Cell × W.Cell) :
    W.content (W.eff2 r i j p).1 + W.content (W.eff2 r i j p).2 =
      W.content p.1 + W.content p.2 := by
  obtain ⟨⟨f, w, b⟩, ⟨f', w', b'⟩⟩ := p
  rcases r with ⟨food, dir⟩ | ⟨e, d, dir⟩ | ⟨S, rb, h2, dir⟩
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
      · split_ifs
        · simp only [content, bugContent, W.fuelAct_content hfuel]; ring
        · rfl
    · rcases b with _ | ⟨y, h⟩ <;> rcases b' with _ | ⟨y', h'⟩
      all_goals simp only [if_true]
      · split_ifs
        · simp only [content, bugContent, W.fuelAct_content hfuel]; ring
        · rfl
      · split_ifs
        · simp only [content, bugContent, W.fuelAct_content hfuel]; ring
        · rfl
  · simp only [eff2]
    cases dir
    · rcases b with _ | ⟨y, h⟩ <;> rcases b' with _ | ⟨y', h'⟩
      all_goals simp only [Bool.false_eq_true, if_false]
      · split_ifs with hc
        · have hd := W.drain_content S (W.divFrom rb) (W.divTo rb) y (fun v hv => (hc.2 v hv).2)
          simp only [content, bugContent, bornQ] at hd hc ⊢
          rw [hdiv rb] at hd
          nlinarith [hc.1]
        · rfl
    · rcases b with _ | ⟨y, h⟩ <;> rcases b' with _ | ⟨y', h'⟩
      all_goals simp only [if_true]
      · split_ifs with hc
        · obtain ⟨rfl, -, hcard, hcan⟩ := hc
          have hd := W.drain_content S (W.divTo rb) (W.divFrom rb) y (fun v hv => (hcan v hv).2)
          simp only [content, bugContent, bornQ] at hd hcard ⊢
          rw [hdiv rb] at hd
          nlinarith [hcard]
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

theorem back1 (r : NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell) (hne : W.eff1 r i l ≠ l) :
    W.eff1 (rev1 r) i (W.eff1 r i l) = l := by
  rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨ha, hb⟩ | ⟨food, dir⟩ | ⟨hd, yd, dir⟩
  · obtain ⟨f, w, _ | ⟨y, h⟩⟩ := l
    · exact absurd rfl hne
    · have hy : W.innerAct k u v dir y ≠ y := fun e => hne (by simp [eff1, e])
      rcases hf : poolShift f (W.sdir k dir).1 with _ | f'
      · exact absurd (by simp [eff1, hy, hf]) hne
      rcases hw : poolShift w (W.sdir k dir).2 with _ | w'
      · exact absurd (by simp [eff1, hy, hf, hw]) hne
      have hb := W.innerAct_back k u v dir y hy
      have hy' : W.innerAct k u v (!dir) (W.innerAct k u v dir y) ≠ W.innerAct k u v dir y := by
        rw [hb]; exact Ne.symm hy
      have hf' := poolShift_back hf
      have hw' := poolShift_back hw
      rw [← (W.sdir_not k dir ▸ rfl : (W.sdir k (!dir)).1 = -(W.sdir k dir).1)] at hf'
      rw [← (W.sdir_not k dir ▸ rfl : (W.sdir k (!dir)).2 = -(W.sdir k dir).2)] at hw'
      simp only [eff1, rev1, if_neg hy, hf, hw, hf', hw', hb, if_neg (Ne.symm hy)]
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
      · by_cases hc : y = yd ∧ h = hd
        · obtain ⟨rfl, rfl⟩ := hc
          rcases hps : poolShift w (W.bodyQ y) with _ | w'
          · simp [eff1, hps] at hne
          · have hb := poolShift_back hps
            simp [eff1, rev1, hps, hb]
        · simp [eff1, hc] at hne
    · rcases b with _ | ⟨y, h⟩
      · rcases hps : poolShift w (-W.bodyQ yd) with _ | w'
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

theorem back2 (r : NR2 Hd n nB) (i j : ℕ) (p : W.Cell × W.Cell) (hne : W.eff2 r i j p ≠ p) :
    W.eff2 (rev2 r) i j (W.eff2 r i j p) = p := by
  rcases r with ⟨food, dir⟩ | ⟨e, d, dir⟩ | ⟨S, rb, h2, dir⟩
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

  · obtain ⟨⟨f, w, b⟩, ⟨f', w', b'⟩⟩ := p
    cases dir
    · rcases b with _ | ⟨y, h⟩ <;> rcases b' with _ | ⟨y', h'⟩
      · simp [eff2] at hne
      · simp [eff2] at hne
      · by_cases hc : S.card = (W.bornQ rb) ∧ W.canDrain rb S (W.divFrom rb) y
        · have hy := drain_drain S (W.divFrom rb) (W.divTo rb) y (fun v hv => (hc.2 v hv).2)
          have hc' : (W.yb rb) = (W.yb rb) ∧ h2 = h2 ∧ S.card = (W.bornQ rb) ∧
              W.canDrain rb S (W.divTo rb) (drain S (W.divTo rb) y) := ⟨rfl, rfl, hc.1, W.canDrain_drain hc.2 _⟩
          simp only [eff2, rev2, Bool.not_false, Bool.false_eq_true, if_false, if_true, if_pos hc]
          rw [if_pos (by simpa using hc'), hy]
        · simp [eff2, hc] at hne
      · simp [eff2] at hne
    · rcases b with _ | ⟨y, h⟩ <;> rcases b' with _ | ⟨y', h'⟩
      · simp [eff2] at hne
      · simp [eff2] at hne
      · simp [eff2] at hne
      · by_cases hc : y' = (W.yb rb) ∧ h' = h2 ∧ S.card = (W.bornQ rb) ∧ W.canDrain rb S (W.divTo rb) y
        · obtain ⟨rfl, rfl, hcard, hcan⟩ := hc
          have hy := drain_drain S (W.divTo rb) (W.divFrom rb) y (fun v hv => (hcan v hv).2)
          have hc' : S.card = (W.bornQ rb) ∧ W.canDrain rb S (W.divFrom rb) (drain S (W.divFrom rb) y) :=
            ⟨hcard, W.canDrain_drain hcan _⟩
          simp only [eff2, rev2, Bool.not_true, Bool.false_eq_true, if_false, if_true, and_self, hcard, hcan,
            true_and]
          rw [if_pos hc'.2, hy]
        · simp [eff2, hc] at hne

theorem div_gate (rb : Fin nB) (hdg : W.divPage rb ≠ W.gatePage) (S : Finset (Fin n)) (h2 : Hd) (dir : Bool) (i j : ℕ)
    (p : W.Cell × W.Cell) : W.cellGate (W.eff2 (.inr (.inr (S, rb, h2, dir))) i j p).1 = W.cellGate p.1 := by
  obtain ⟨⟨f, w, b⟩, ⟨f', w', b'⟩⟩ := p
  cases dir
  · rcases b with _ | ⟨y, h⟩ <;> rcases b' with _ | ⟨y', h'⟩ <;> simp only [eff2, Bool.false_eq_true, if_false]
    split_ifs with hc
    · simp [cellGate, W.gateCount_drain hdg hc.2]
    · rfl
  · rcases b with _ | ⟨y, h⟩ <;> rcases b' with _ | ⟨y', h'⟩ <;> simp only [eff2, if_true]
    split_ifs with hc
    · simp [cellGate, W.gateCount_drain hdg hc.2.2.2]
    · rfl

theorem cellGate_inner (k : Fin nR) (u v : Fin n) (dir : Bool) (i : ℕ) (l : W.Cell)
    (hi : (W.spec k).i ≠ W.gatePage) (hj : (W.spec k).j ≠ W.gatePage) :
    W.cellGate (W.eff1 (.inl (k, u, v, dir)) i l) = W.cellGate l := by
  obtain ⟨f, w, _ | ⟨y, h⟩⟩ := l
  · rfl
  · simp only [eff1]
    split_ifs with hy
    · rfl
    · rcases hf : poolShift f (W.sdir k dir).1 with _ | f' <;>
        rcases hw : poolShift w (W.sdir k dir).2 with _ | w'
      · rfl
      · rfl
      · rfl
      · exact W.gateCount_innerAct k u v dir y hi hj

open Classical in
/-- **T-E2 (an organism nested in its world is admitted)**: the joint medium of organisms and
their world satisfies the six axiom clauses as soon as the organism's inner pair rules and its
fuel rule conserve content and no exchange goes through the membrane page itself. Sensing, eating, excreting (exchange with the cell's pools),
tumbling, diffusion, driven movement and the world's own sources and sinks need no further
obligation. -/
theorem axioms [Fintype Λi] [Fintype Hd] {U : Type*} [Fintype U] [DecidableEq U]
    (adj : U → U → Prop) (hsymm : ∀ u v, adj u v → adj v u) (hirr : ∀ u, ¬ adj u u)
    (page : U → ℕ) (y0 : Fin n → Λi) (h0 : Hd)
    (hspec : W.SpecBalanced)
    (hfuel : W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 =
      W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2)
    (hgate : ∀ j, W.xgated j = true → W.xpage j ≠ W.gatePage)
    (hsg : ∀ k, W.sgated k = true → (W.spec k).i ≠ W.gatePage ∧ (W.spec k).j ≠ W.gatePage)
    (hdiv : ∀ rb, W.contentIn (W.divFrom rb) = W.contentIn (W.divTo rb) + 1) (hdg : ∀ rb, W.divPage rb ≠ W.gatePage) :
    (W.lattice adj hsymm hirr page y0 h0).medium.Axioms := by
  classical
  refine LatticeCRN.axioms (W.lattice adj hsymm hirr page y0 h0) W.content_p1a
    (fun r i l hi => W.cons1 hspec r i l hi) (fun r i j p _ _ => W.cons2 hfuel hdiv r i j p) rfl
    ?_ ?_ (fun r i l hne => W.back1 r i l hne) (fun r i j p hne => W.back2 r i j p hne) ?_ ?_
  · intro r; rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨a, b⟩ | ⟨food, dir⟩ | ⟨hd, yd, dir⟩ <;> simp [lattice, rev1]
  · intro r; rcases r with ⟨food, dir⟩ | ⟨e, d, dir⟩ | ⟨S, rb, h2, dir⟩ <;> simp [lattice, rev2]
  · intro r i l hne
    rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨a, b⟩ | ⟨food, dir⟩ | ⟨hd, yd, dir⟩
    · have hback := W.back1 (.inl (k, u, v, dir)) i l hne
      change (W.spec k).κ * (if W.sgated k then (W.cellGate (W.eff1 (.inl (k, u, v, dir)) i l) : ℝ) else 1) *
          W.poolFactor (W.eff1 (.inl (k, u, v, dir)) i l)
            (W.eff1 (.inl (k, u, v, !dir)) i (W.eff1 (.inl (k, u, v, dir)) i l)) =
        (W.spec k).κ * (if W.sgated k then (W.cellGate l : ℝ) else 1) *
          W.poolFactor l (W.eff1 (.inl (k, u, v, dir)) i l)
      rw [show W.eff1 (.inl (k, u, v, !dir)) i (W.eff1 (.inl (k, u, v, dir)) i l) = l from hback,
        W.poolFactor_symm (W.eff1 _ i l) l]
      by_cases hg : W.sgated k = true
      · rw [if_pos hg, if_pos hg, W.cellGate_inner k u v dir i l (hsg k hg).1 (hsg k hg).2]
      · rw [if_neg hg, if_neg hg]
    · have hback := W.back1 (.inr (.inl (j, u, dir))) i l hne
      change W.xκ j * (if W.xgated j then (W.cellGate (W.eff1 (.inr (.inl (j, u, dir))) i l) : ℝ) else 1) *
          W.poolFactor (W.eff1 (.inr (.inl (j, u, dir))) i l)
            (W.eff1 (.inr (.inl (j, u, !dir))) i (W.eff1 (.inr (.inl (j, u, dir))) i l)) =
        W.xκ j * (if W.xgated j then (W.cellGate l : ℝ) else 1) *
          W.poolFactor l (W.eff1 (.inr (.inl (j, u, dir))) i l)
      rw [show W.eff1 (.inr (.inl (j, u, !dir))) i (W.eff1 (.inr (.inl (j, u, dir))) i l) = l from hback,
        W.poolFactor_symm (W.eff1 _ i l) l]
      by_cases hg : W.xgated j = true
      · rw [if_pos hg, if_pos hg, eff1_xch, W.xchEff_gate (hgate j hg)]
      · rw [if_neg hg, if_neg hg]
    · rfl
    · have hback := W.back1 (.inr (.inr (.inr (.inl (food, dir))))) i l hne
      change W.κin * W.poolFactor (W.eff1 (.inr (.inr (.inr (.inl (food, dir))))) i l)
          (W.eff1 (.inr (.inr (.inr (.inl (food, !dir))))) i (W.eff1 (.inr (.inr (.inr (.inl (food, dir))))) i l)) =
        W.κin * W.poolFactor l (W.eff1 (.inr (.inr (.inr (.inl (food, dir))))) i l)
      rw [show W.eff1 (.inr (.inr (.inr (.inl (food, !dir))))) i
          (W.eff1 (.inr (.inr (.inr (.inl (food, dir))))) i l) = l from hback, W.poolFactor_symm]
    · have hback := W.back1 (.inr (.inr (.inr (.inr (hd, yd, dir))))) i l hne
      change (if W.gateCount yd = 0 then W.κdis else 0) *
          Real.exp (-|W.φ i (W.eff1 (.inr (.inr (.inr (.inr (hd, yd, !dir))))) i
            (W.eff1 (.inr (.inr (.inr (.inr (hd, yd, dir))))) i l)) -
            W.φ i (W.eff1 (.inr (.inr (.inr (.inr (hd, yd, dir))))) i l)| / 2) =
        (if W.gateCount yd = 0 then W.κdis else 0) *
          Real.exp (-|W.φ i (W.eff1 (.inr (.inr (.inr (.inr (hd, yd, dir))))) i l) - W.φ i l| / 2)
      rw [show W.eff1 (.inr (.inr (.inr (.inr (hd, yd, !dir))))) i
          (W.eff1 (.inr (.inr (.inr (.inr (hd, yd, dir))))) i l) = l from hback, abs_sub_comm]
  · intro r i j p hne
    rcases r with ⟨food, dir⟩ | ⟨e, d, dir⟩ | ⟨S, rb, h2, dir⟩
    · have hback := W.back2 (.inl (food, dir)) i j p hne
      change W.κdiff * (W.poolFactor (W.eff2 (.inl (food, dir)) i j p).1
            (W.eff2 (.inl (food, !dir)) i j (W.eff2 (.inl (food, dir)) i j p)).1 *
          W.poolFactor (W.eff2 (.inl (food, dir)) i j p).2
            (W.eff2 (.inl (food, !dir)) i j (W.eff2 (.inl (food, dir)) i j p)).2) =
        W.κdiff * (W.poolFactor p.1 (W.eff2 (.inl (food, dir)) i j p).1 *
          W.poolFactor p.2 (W.eff2 (.inl (food, dir)) i j p).2)
      rw [show W.eff2 (.inl (food, !dir)) i j (W.eff2 (.inl (food, dir)) i j p) = p from hback,
        W.poolFactor_symm _ p.1, W.poolFactor_symm _ p.2]
    · rfl
    · have hback := W.back2 (.inr (.inr (S, rb, h2, dir))) i j p hne
      have hg := W.div_gate rb (hdg rb) S h2 dir i j p
      change (W.κdiv rb) * (if 0 < W.cellGate (W.eff2 (.inr (.inr (S, rb, h2, dir))) i j p).1 then 1 else 0) *
          Real.exp (-|(W.φ i (W.eff2 (.inr (.inr (S, rb, h2, !dir))) i j
              (W.eff2 (.inr (.inr (S, rb, h2, dir))) i j p)).1 +
            W.φ j (W.eff2 (.inr (.inr (S, rb, h2, !dir))) i j (W.eff2 (.inr (.inr (S, rb, h2, dir))) i j p)).2) -
            (W.φ i (W.eff2 (.inr (.inr (S, rb, h2, dir))) i j p).1 +
              W.φ j (W.eff2 (.inr (.inr (S, rb, h2, dir))) i j p).2)| / 2) =
        (W.κdiv rb) * (if 0 < W.cellGate p.1 then 1 else 0) *
          Real.exp (-|(W.φ i (W.eff2 (.inr (.inr (S, rb, h2, dir))) i j p).1 +
            W.φ j (W.eff2 (.inr (.inr (S, rb, h2, dir))) i j p).2) - (W.φ i p.1 + W.φ j p.2)| / 2)
      rw [show W.eff2 (.inr (.inr (S, rb, h2, !dir))) i j (W.eff2 (.inr (.inr (S, rb, h2, dir))) i j p) = p from hback,
        hg, abs_sub_comm]

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

/-- A pair rule that does not couple to the pools (`sdf + sdw = 0`) keeps the organism's content. -/
theorem orgIn_inner (hspec : W.SpecBalanced) (k : Fin nR) (u v : Fin n) (dir : Bool) (i : ℕ)
    (l : W.Cell) (h0 : W.sdf k + W.sdw k = 0) :
    W.orgIn (W.eff1 (.inl (k, u, v, dir)) i l) = W.orgIn l := by
  obtain ⟨f, w, _ | ⟨y, h⟩⟩ := l
  · rfl
  · simp only [eff1]
    split_ifs with hy
    · rfl
    · rcases hf : poolShift f (W.sdir k dir).1 with _ | f' <;>
        rcases hw : poolShift w (W.sdir k dir).2 with _ | w'
      · rfl
      · rfl
      · rfl
      · have hd := W.innerAct_delta hspec k u v dir y hy
        have hz : (W.sdir k dir).1 + (W.sdir k dir).2 = 0 := by
          cases dir <;> simp only [sdir, Bool.false_eq_true, if_false, if_true] <;> omega
        simp only [orgIn]
        omega

/-- The world's supply rules never touch the organism. -/
theorem eff1_supply_bug (food dir : Bool) (i : ℕ) (l : W.Cell) :
    (W.eff1 (.inr (.inr (.inr (.inl (food, dir))))) i l).2.2 = l.2.2 := by
  obtain ⟨f, w, b⟩ := l
  cases food <;> simp only [eff1, Bool.false_eq_true, if_false, if_true] <;> split_ifs <;>
    (try split) <;> rfl

/-- **Only events across the boundary feed the organism**: every single-cell event that raises the
content held by the organism is a pool exchange or a pair rule coupled to the pools; a gated one has
conductance proportional to the intact membrane.
-/
theorem intake_is_exchange
    (hspec : W.SpecBalanced)
    (r : NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell) (hb : l.2.2 ≠ none)
    (hup : W.orgIn l < W.orgIn (W.eff1 r i l)) :
    (∃ j u dir, r = .inr (.inl (j, u, dir)) ∧
      W.c1 r i l = W.xκ j * (if W.xgated j then (W.cellGate l : ℝ) else 1) * W.poolFactor l (W.eff1 r i l)) ∨
    (∃ k u v dir, r = .inl (k, u, v, dir) ∧ W.sdf k + W.sdw k ≠ 0 ∧
      W.c1 r i l = (W.spec k).κ * (if W.sgated k then (W.cellGate l : ℝ) else 1) *
        W.poolFactor l (W.eff1 r i l)) := by
  rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨a, b⟩ | ⟨food, dir⟩ | ⟨hd, yd, dir⟩
  · refine Or.inr ⟨k, u, v, dir, rfl, fun h0 => ?_, rfl⟩
    rw [W.orgIn_inner hspec k u v dir i l h0] at hup
    exact lt_irrefl _ hup
  · exact Or.inl ⟨j, u, dir, rfl, rfl⟩
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
    rcases b with _ | ⟨y, h⟩
    · exact hb rfl
    cases dir
    · simp only [eff1, Bool.false_eq_true, if_false] at hup
      split_ifs at hup
      · split at hup <;> simp [orgIn] at hup
      · exact lt_irrefl _ hup
    · simp [eff1] at hup

/-- **Death** (`Part.CanTakeIn` fails through the organism's own doing): with no intact membrane
unit, every event that would raise the organism's content has rate zero, except an ungated
exchange: physics the organism does not control (the reverse of a leak, membrane re-forming from
the cell's pools). Every rate of the medium is a positive multiple of `exp`, so
`Part.Recoverable` alone never fails; without the membrane factor, intake would not depend on the
body at all, structure would have no use and selection could drop its upkeep. -/
theorem dead_intake_is_physics
    (hspec : W.SpecBalanced)
    (r : NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell) (hb : l.2.2 ≠ none) (hdead : W.cellGate l = 0)
    (hup : W.orgIn l < W.orgIn (W.eff1 r i l)) :
    W.c1 r i l = 0 ∨ (∃ j u dir, r = .inr (.inl (j, u, dir)) ∧ W.xgated j = false) ∨
      ∃ k u v dir, r = .inl (k, u, v, dir) ∧ W.sgated k = false := by
  rcases W.intake_is_exchange hspec r i l hb hup with ⟨j, u, dir, hr, hc⟩ | ⟨k, u, v, dir, hr, -, hc⟩
  · by_cases hg : W.xgated j = true
    · left; rw [hc, if_pos hg, hdead]; simp
    · right; left; exact ⟨j, u, dir, hr, by simpa using hg⟩
  · by_cases hg : W.sgated k = true
    · left; rw [hc, if_pos hg, hdead]; simp
    · right; right; exact ⟨k, u, v, dir, hr, by simpa using hg⟩

/-- **A dead body does not give itself away**: every single-cell event that lowers the content
held by the organism is a pool exchange; with no intact membrane the gated ones have rate zero, so
what leaves a dead body leaves only by an ungated exchange (physics, a leak). With G26c
(`corpses_must_go`), a world of endless mortal lives needs such leaks, or organisms that take
dead bodies apart (decomposers, events this table does not yet have). -/
theorem dead_output_is_physics
    (hspec : W.SpecBalanced)
    (r : NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell) (hdead : W.cellGate l = 0)
    (hdown : W.orgIn (W.eff1 r i l) < W.orgIn l) :
    W.c1 r i l = 0 ∨ (∃ j u dir, r = .inr (.inl (j, u, dir)) ∧ W.xgated j = false) ∨
      (∃ k u v dir, r = .inl (k, u, v, dir) ∧ W.sdf k + W.sdw k ≠ 0 ∧ W.sgated k = false) ∨
      ∃ hd yd, r = .inr (.inr (.inr (.inr (hd, yd, false)))) := by
  rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨a, b⟩ | ⟨food, dir⟩ | ⟨hd, yd, dir⟩
  · by_cases h0 : W.sdf k + W.sdw k = 0
    · exfalso
      rw [W.orgIn_inner hspec k u v dir i l h0] at hdown
      exact lt_irrefl _ hdown
    · by_cases hg : W.sgated k = true
      · left; simp [c1, hg, hdead]
      · right; right; left; exact ⟨k, u, v, dir, rfl, h0, by simpa using hg⟩
  · by_cases hg : W.xgated j = true
    · left; simp [c1, hg, hdead]
    · right; left; exact ⟨j, u, dir, rfl, by simpa using hg⟩
  · exfalso
    obtain ⟨f, w, _ | ⟨y, h⟩⟩ := l
    · simp [eff1, orgIn] at hdown
    · simp only [eff1] at hdown
      split_ifs at hdown <;> simp [orgIn] at hdown
  · exfalso
    rw [W.orgIn_congr (W.eff1_supply_bug food dir i l)] at hdown
    exact lt_irrefl _ hdown
  · cases dir
    · right; right; right; exact ⟨hd, yd, rfl⟩
    · exfalso
      obtain ⟨f, w, b⟩ := l
      rcases b with _ | ⟨y, h⟩
      · simp [orgIn] at hdown
      · simp [eff1] at hdown

/-- **Only dissolution empties a place** (with G26c `corpses_must_go`, why the table needs it):
every single-cell event that turns a cell holding a body into a cell without one is the
dissolution of a body. (A move only carries the body to the next cell.) Without this event no
place ever loses its body, and a world of endless mortal lives fills with corpses. Only a body
without intact membrane dissolves (its conductance is zero otherwise), with everything it holds;
by P2b the reverse exists too: a dead body forming from waste, suppressed by the entropy of
gathering `B` and more quanta. -/
theorem frees_only_by_dissolving (r : NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell)
    (hb : l.2.2 ≠ none) (hfree : (W.eff1 r i l).2.2 = none) :
    ∃ hd yd, r = .inr (.inr (.inr (.inr (hd, yd, false)))) := by
  obtain ⟨f, w, b⟩ := l
  rcases b with _ | ⟨y, h⟩
  · exact absurd rfl hb
  rcases r with ⟨k, u, v, dir⟩ | ⟨j, u, dir⟩ | ⟨a, c⟩ | ⟨food, dir⟩ | ⟨hd, yd, dir⟩
  · simp only [eff1] at hfree
    split_ifs at hfree <;> (try split at hfree) <;> simp_all
  · rw [eff1_xch] at hfree
    simp only [xchEff] at hfree
    split_ifs at hfree <;> (try split at hfree) <;> simp_all
  · simp only [eff1] at hfree
    split_ifs at hfree <;> simp_all
  · rw [W.eff1_supply_bug] at hfree
    simp at hfree
  · cases dir
    · exact ⟨hd, yd, rfl⟩
    · simp [eff1] at hfree

/-- Bodies in a cell. -/
def bodies (l : W.Cell) : ℕ := if l.2.2.isSome then 1 else 0

/-- **Births come only from division** (the dual of `frees_only_by_dissolving`): a neighbour
event after which the pair holds more bodies is a division. Diffusion keeps the bodies where
they are, a move carries one body, a fusion removes one. With division, the endless stream of
mortal lives of G26c can be fed from inside the world. -/
theorem births_only_by_division (r : NR2 Hd n nB) (i j : ℕ) (p : W.Cell × W.Cell)
    (hup : W.bodies p.1 + W.bodies p.2 < W.bodies (W.eff2 r i j p).1 + W.bodies (W.eff2 r i j p).2) :
    ∃ S rb h2, r = .inr (.inr (S, rb, h2, false)) := by
  obtain ⟨⟨f, w, b⟩, ⟨f', w', b'⟩⟩ := p
  rcases r with ⟨food, dir⟩ | ⟨e, d, dir⟩ | ⟨S, rb, h2, dir⟩
  · exfalso
    simp only [eff2] at hup
    cases food <;> simp only [Bool.false_eq_true, if_false, if_true] at hup <;>
      (repeat' split at hup) <;> simp_all [bodies]
  · exfalso
    simp only [eff2] at hup
    cases dir <;> simp only [Bool.false_eq_true, if_false, if_true] at hup <;>
      (repeat' split at hup) <;> (try split_ifs at hup) <;> simp_all [bodies]
  · cases dir
    · exact ⟨S, rb, h2, rfl⟩
    · exfalso
      simp only [eff2, if_true] at hup
      rcases b with _ | ⟨y, h⟩ <;> rcases b' with _ | ⟨y', h'⟩ <;>
        simp only at hup <;> (try split_ifs at hup) <;> simp_all [bodies]

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
fires at rate at least `κ * exp (-Φ)` whenever the unit holds that quantum and the pool is not
full, whatever the rest of the organism and the reservoir potential `μ` (the exchange conserves the
cell's content, so `μ` drops out). The bound depends only on physics; under mass action the rate is
`κ (V - w) e^{-ΔE_in/2}`, at least one free place. -/
theorem leak_bound (Φ : ℝ) (hΦ : ∀ p s, |W.φIn p s| ≤ Φ) (j : Fin nX) (hg : W.xgated j = false)
    (hf : W.xfood j = false) (hc : W.contentIn (W.xa j) = W.contentIn (W.xb j) + 1) (u : Fin n)
    (i : ℕ) (f w : Fin (W.V + 1)) (y : Fin n → Λi) (h : Hd)
    (hu : W.pageIn u = W.xpage j ∧ y u = W.xa j) (hw : (w : ℕ) < W.V) (μ : ℝ) :
    W.xκ j * Real.exp (-Φ) ≤
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
  have hc1 : W.c1 (.inr (.inl (j, u, false))) i (f, w, some (y, h)) =
      W.xκ j * poolCond W.V w ((w : ℕ) + 1) := by
    simp only [c1, hg, Bool.false_eq_true, if_false, mul_one, heff, poolFactor, poolCond_self, one_mul, hwv]
  rw [hc1, hcont, sub_self, mul_zero, sub_zero, hphi, ← hstep, mul_assoc, mass_action_up W.V w hw]
  apply mul_le_mul_of_nonneg_left _ (W.xκ_nonneg j)
  have hE : Real.exp (-Φ) ≤
      Real.exp (-(W.φIn (W.pageIn u) (W.xb j) - W.φIn (W.pageIn u) (y u)) / 2) :=
    Real.exp_le_exp.mpr (by linarith)
  exact hE.trans (le_mul_of_one_le_left (Real.exp_pos _).le hden)

/-! ## Nested life: sensing and metabolism, once for every organism of the class

The life notions of a part of a medium (`NestedLife`): the organism is the body in a cell `u0`
(`orgPart`), its content the inner content `orgIn`. Both obligations follow from one decidable
table condition: a gated exchange that takes one quantum from the food pool, with an intact
membrane. Every nested organism whose table has such an exchange senses and metabolizes. -/

open Classical in
/-- The organism in cell `u0` as a part of the joint medium. -/
noncomputable def orgPart {U : Type*} (u0 : U) : Part U W.Cell (Option ((Fin n → Λi) × Hd)) :=
  ⟨fun x => (x u0).2.2, fun x => W.orgIn (x u0)⟩

theorem intake_eff (j : Fin nX) (hf : W.xfood j = true)
    (hc : W.contentIn (W.xb j) = W.contentIn (W.xa j) + 1) (u : Fin n) (i : ℕ) (y : Fin n → Λi) (h : Hd)
    (hu : W.pageIn u = W.xpage j ∧ y u = W.xa j) (f w : Fin (W.V + 1)) (hf1 : 1 ≤ (f : ℕ)) :
    W.eff1 (.inr (.inl (j, u, false))) i (f, w, some (y, h)) =
      (⟨(f : ℕ) - 1, by omega⟩, w, some (Function.update y u (W.xb j), h)) := by
  have hd : ((W.contentIn (W.xa j) : ℤ) - W.contentIn (W.xb j)) = -1 := by rw [hc]; push_cast; ring
  have hps : poolShift f (-1) = some ⟨(f : ℕ) - 1, by omega⟩ := by
    unfold poolShift
    rw [dif_pos (by omega)]
    congr 1
    ext
    simp only [Fin.val_mk]
    omega
  simp only [eff1, Bool.false_eq_true, if_false, hu, and_self, if_true, hf, hd, hps]

theorem intake_content (j : Fin nX) (hc : W.contentIn (W.xb j) = W.contentIn (W.xa j) + 1)
    (u : Fin n) (y : Fin n → Λi) (hu : y u = W.xa j) :
    ∑ v, W.contentIn (Function.update y u (W.xb j) v) = ∑ v, W.contentIn (y v) + 1 := by
  have hs := sum_update_nat W.contentIn u (W.xb j) y
  rw [hu, hc] at hs
  push_cast at hs
  zify
  push_cast
  linarith

/-- A cell holding food `f`, no waste, and the organism. -/
def withOrg {U : Type*} [DecidableEq U] (u0 : U) (f : Fin (W.V + 1)) (y : Fin n → Λi) (h : Hd) :
    U → W.Cell :=
  Function.update (fun _ => ((0 : Fin (W.V + 1)), (0 : Fin (W.V + 1)), none)) u0 (f, 0, some (y, h))

open Classical in
/-- **Nested metabolism** (`Part.Metabolizes`) for every organism whose table has a gated food
intake of one quantum, in a state with an intact membrane and a unit ready to take in. -/
theorem metabolizes_of_intake [Fintype Λi] [Fintype Hd] {U : Type*} [Fintype U] [DecidableEq U]
    (adj : U → U → Prop) (hsymm : ∀ u v, adj u v → adj v u) (hirr : ∀ u, ¬ adj u u)
    (page : U → ℕ) (y0 : Fin n → Λi) (h0 : Hd) (u0 : U)
    (j : Fin nX) (hg : W.xgated j = true) (hf : W.xfood j = true) (hk : 0 < W.xκ j)
    (hc : W.contentIn (W.xb j) = W.contentIn (W.xa j) + 1) (u : Fin n) (y : Fin n → Λi) (h : Hd)
    (hu : W.pageIn u = W.xpage j ∧ y u = W.xa j) (hgate : 0 < W.gateCount y) (hV : 1 ≤ W.V) :
    (W.orgPart u0).Metabolizes (W.lattice adj hsymm hirr page y0 h0).medium := by
  set C := W.lattice adj hsymm hirr page y0 h0
  let f1 : Fin (W.V + 1) := ⟨1, by omega⟩
  refine ⟨.inl (.inr (.inl (j, u, false)), u0), fun _ => 0, W.withOrg u0 f1 y h, ?_, ?_⟩
  · change 0 < C.c1 _ (C.ident u0) (W.withOrg u0 f1 y h u0) * Real.exp _
    refine mul_pos ?_ (Real.exp_pos _)
    simp only [withOrg, Function.update_self]
    change 0 < W.xκ j * (if W.xgated j then (W.gateCount y : ℝ) else 1) * W.poolFactor _ _
    rw [if_pos hg]
    exact mul_pos (mul_pos hk (by exact_mod_cast hgate)) (W.poolFactor_pos _ _)
  · change W.orgIn (C.act1 _ u0 (W.withOrg u0 f1 y h) u0) ≠ W.orgIn (W.withOrg u0 f1 y h u0)
    simp only [LatticeCRN.act1, Function.update_self, withOrg]
    change W.orgIn (W.eff1 (.inr (.inl (j, u, false))) (page u0) (f1, 0, some (y, h))) ≠
      W.orgIn (f1, 0, some (y, h))
    rw [W.intake_eff j hf hc u _ y h hu f1 0 (by simp [f1])]
    simp only [orgIn, W.intake_content j hc u y hu.2]
    omega

open Classical in
/-- **Nested sensing** (`Part.Senses`): the same intake event, from the same organism, fires at
different rates when only the cell's food differs (one quantum or two). -/
theorem senses_of_intake [Fintype Λi] [Fintype Hd] {U : Type*} [Fintype U] [DecidableEq U]
    (adj : U → U → Prop) (hsymm : ∀ u v, adj u v → adj v u) (hirr : ∀ u, ¬ adj u u)
    (page : U → ℕ) (y0 : Fin n → Λi) (h0 : Hd) (u0 : U)
    (j : Fin nX) (hg : W.xgated j = true) (hf : W.xfood j = true) (hk : 0 < W.xκ j)
    (hc : W.contentIn (W.xb j) = W.contentIn (W.xa j) + 1) (u : Fin n) (y : Fin n → Λi) (h : Hd)
    (hu : W.pageIn u = W.xpage j ∧ y u = W.xa j) (hgate : 0 < W.gateCount y) (hV : 2 ≤ W.V) :
    (W.orgPart u0).Senses (W.lattice adj hsymm hirr page y0 h0).medium := by
  set C := W.lattice adj hsymm hirr page y0 h0
  let f1 : Fin (W.V + 1) := ⟨1, by omega⟩
  let f2 : Fin (W.V + 1) := ⟨2, by omega⟩
  have hxb : W.xb j ≠ W.xa j := fun e => by rw [e] at hc; omega
  refine ⟨.inl (.inr (.inl (j, u, false)), u0), fun _ => 0, W.withOrg u0 f1 y h, W.withOrg u0 f2 y h,
    ?_, ?_, ?_⟩
  · simp [orgPart, withOrg]
  · change (C.act1 _ u0 (W.withOrg u0 f1 y h) u0).2.2 ≠ (W.withOrg u0 f1 y h u0).2.2
    simp only [LatticeCRN.act1, Function.update_self, withOrg]
    change (W.eff1 (.inr (.inl (j, u, false))) (page u0) (f1, 0, some (y, h))).2.2 ≠ some (y, h)
    rw [W.intake_eff j hf hc u _ y h hu f1 0 (by simp [f1])]
    intro e
    simp only [Option.some.injEq, Prod.mk.injEq] at e
    have := congrFun e.1 u
    rw [Function.update_self, hu.2] at this
    exact hxb this
  · -- the two rates: the same conductance, different energy changes
    have hcell : ∀ f : Fin (W.V + 1), W.withOrg u0 f y h u0 = (f, 0, some (y, h)) := fun f => by
      simp [withOrg]
    change C.c1 _ (C.ident u0) (W.withOrg u0 f1 y h u0) * Real.exp (-C.dE1 _ u0 _ (W.withOrg u0 f1 y h) / 2) ≠
      C.c1 _ (C.ident u0) (W.withOrg u0 f2 y h u0) * Real.exp (-C.dE1 _ u0 _ (W.withOrg u0 f2 y h) / 2)
    -- energy changes
    have hdE : ∀ f : Fin (W.V + 1), 1 ≤ (f : ℕ) → C.dE1 (.inr (.inl (j, u, false))) u0 (fun _ => 0)
        (W.withOrg u0 f y h) = Generation.cellF W.V ((f : ℕ) - 1) - Generation.cellF W.V f - W.foodE +
          (W.bugE (Function.update y u (W.xb j)) - W.bugE y) := fun f hf1 => by
      simp only [LatticeCRN.dE1, LatticeCRN.unitE, LatticeCRN.res]
      rw [hcell]
      change W.φ (page u0) (W.eff1 (.inr (.inl (j, u, false))) (page u0) (f, 0, some (y, h))) -
          W.φ (page u0) (f, 0, some (y, h)) -
          (if page u0 ∈ bidW then (0 : ℝ) else 0) *
            ((W.content (W.eff1 (.inr (.inl (j, u, false))) (page u0) (f, 0, some (y, h))) : ℝ) -
              (W.content (f, 0, some (y, h)) : ℝ)) = _
      rw [W.intake_eff j hf hc u _ y h hu f 0 hf1]
      simp only [φ, ite_self, zero_mul, sub_zero]
      have : (((f : ℕ) - 1 : ℕ) : ℝ) = (f : ℕ) - 1 := by rw [Nat.cast_sub hf1]; simp
      simp only [Fin.val_mk, this]
      ring
    -- mass action: the same intake fires at a rate proportional to the food quanta
    have e4 : ∀ a b c d : ℝ, a - b - c + d = a - b + (-c + d) := fun _ _ _ _ => by ring
    have hrate : ∀ (f : Fin (W.V + 1)) (m : ℕ), (f : ℕ) = m + 1 → m < W.V →
        C.c1 (.inr (.inl (j, u, false))) (C.ident u0) (W.withOrg u0 f y h u0) *
          Real.exp (-C.dE1 (.inr (.inl (j, u, false))) u0 (fun _ => 0) (W.withOrg u0 f y h) / 2) =
        W.xκ j * W.gateCount y * (((m : ℝ) + 1) *
          Real.exp (-(-W.foodE + (W.bugE (Function.update y u (W.xb j)) - W.bugE y)) / 2)) := by
      intro f m hfm hm
      rw [hdE f (by omega), hcell]
      change W.xκ j * (if W.xgated j then (W.gateCount y : ℝ) else 1) *
          W.poolFactor (f, 0, some (y, h))
            (W.eff1 (.inr (.inl (j, u, false))) (page u0) (f, 0, some (y, h))) * _ = _
      rw [if_pos hg, W.intake_eff j hf hc u _ y h hu f 0 (by omega)]
      simp only [poolFactor, poolCond_self, mul_one, Fin.val_mk, hfm, Nat.add_sub_cancel]
      rw [← mass_action_down W.V m hm (-W.foodE + (W.bugE (Function.update y u (W.xb j)) - W.bugE y)), e4]
      ring
    rw [hrate f1 0 rfl (by omega), hrate f2 1 rfl (by omega)]
    have hpos : 0 < W.xκ j * (W.gateCount y : ℝ) := mul_pos hk (by exact_mod_cast hgate)
    apply ne_of_lt
    apply mul_lt_mul_of_pos_left _ hpos
    apply mul_lt_mul_of_pos_right (by push_cast; norm_num) (Real.exp_pos _)

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
    (hspec : W.SpecBalanced)
    (hfuel : W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 =
      W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2) (hgate : ∀ j, W.xgated j = true → W.xpage j ≠ W.gatePage)
    (hsg : ∀ k, W.sgated k = true → (W.spec k).i ≠ W.gatePage ∧ (W.spec k).j ≠ W.gatePage)
    (hdiv : ∀ rb, W.contentIn (W.divFrom rb) = W.contentIn (W.divTo rb) + 1) (hdg : ∀ rb, W.divPage rb ≠ W.gatePage) :
    (W.lattice (torusAdj G) (torusAdj_symm G) (fun _ h => h.1 rfl) (torusPage G kind) y0 h0).medium.Axioms :=
  W.axioms _ _ _ _ y0 h0 hspec hfuel hgate hsg hdiv hdg

open Classical in
/-- The one-dimensional ring is a model too. -/
theorem axioms_ring [Fintype Λi] [Fintype Hd] (L : ℕ) (page : Fin L → ℕ) (y0 : Fin n → Λi) (h0 : Hd)
    (hspec : W.SpecBalanced)
    (hfuel : W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 =
      W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2) (hgate : ∀ j, W.xgated j = true → W.xpage j ≠ W.gatePage)
    (hsg : ∀ k, W.sgated k = true → (W.spec k).i ≠ W.gatePage ∧ (W.spec k).j ≠ W.gatePage)
    (hdiv : ∀ rb, W.contentIn (W.divFrom rb) = W.contentIn (W.divTo rb) + 1) (hdg : ∀ rb, W.divPage rb ≠ W.gatePage) :
    (W.lattice (ringAdj L) (ringAdj_symm L) (fun _ h => h.1 rfl) page y0 h0).medium.Axioms :=
  W.axioms _ _ _ _ y0 h0 hspec hfuel hgate hsg hdiv hdg

end NestedWorld

end CGB
