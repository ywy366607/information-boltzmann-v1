import CGBAxioms.GenerationExport

/-!
# Growth of a connection unit (generation chain, G18)

A connection unit sits between two compartments and carries its two end activities
`a b : Fin 2` and its slot occupancy `w`. Its energy is any function `E w a b`.

* G18a: every function of two binary end states splits uniquely into a constant, a
  term of `a` alone, a term of `b` alone and `κ * a * b`; the only part that couples
  the two ends is the product (Hebbian co-activity), so growth can depend on the two
  ends jointly only through `a * b`.
* G18b (reciprocity): the mixed difference "how the slot changes the co-activity
  preference" equals "how co-activity changes the slot preference".
* G18c: the equilibrium occupancy of one slot with field `h` is `1 / (1 + exp (-h))`,
  strictly increasing: more use, more structure (G17's state-selective export).
-/

namespace CGB.Generation

/-- The interaction coefficient of a function of two binary variables. -/
def interaction (f : Fin 2 → Fin 2 → ℝ) : ℝ := f 1 1 - f 1 0 - f 0 1 + f 0 0

/-- **G18a**: the Möbius decomposition. -/
theorem binary_pair_decomposition (f : Fin 2 → Fin 2 → ℝ) (a b : Fin 2) :
    f a b = f 0 0 + (f 1 0 - f 0 0) * (a : ℝ) + (f 0 1 - f 0 0) * (b : ℝ) +
      interaction f * (a : ℝ) * (b : ℝ) := by
  fin_cases a <;> fin_cases b <;> simp [interaction] <;> ring

/-- **G18a (uniqueness)**: if `f` is written with unary parts and a product
coefficient `κ`, then `κ` is the interaction coefficient. -/
theorem interaction_unique (f : Fin 2 → Fin 2 → ℝ) (c u v κ : ℝ)
    (h : ∀ a b : Fin 2, f a b = c + u * (a : ℝ) + v * (b : ℝ) + κ * (a : ℝ) * (b : ℝ)) :
    κ = interaction f := by
  simp only [interaction, h]
  simp
  ring

/-- **G18b (reciprocity)**: for an energy `E w x` of a slot `w` and a co-activity `x`,
the effect of `x` on the slot's flip equals the effect of the slot on `x`'s flip. -/
theorem slot_reciprocity (E : Fin 2 → Fin 2 → ℝ) :
    (E 1 1 - E 0 1) - (E 1 0 - E 0 0) = (E 1 1 - E 1 0) - (E 0 1 - E 0 0) := by
  ring

/-- Equilibrium occupancy of one slot with log-odds `h`. -/
noncomputable def occupancy (h : ℝ) : ℝ := 1 / (1 + Real.exp (-h))

/-- The detailed-balanced build/dismantle ratio fixes the occupancy:
`occ / (1 - occ) = exp h`. -/
theorem occupancy_odds (h : ℝ) : occupancy h / (1 - occupancy h) = Real.exp h := by
  have he : 0 < Real.exp (-h) := Real.exp_pos _
  have h1 : 1 - occupancy h = Real.exp (-h) / (1 + Real.exp (-h)) := by
    unfold occupancy; field_simp; ring
  rw [h1]
  unfold occupancy
  rw [Real.exp_neg]
  field_simp

/-- **G18c**: occupancy is strictly increasing in the field. -/
theorem occupancy_strictMono : StrictMono occupancy := by
  intro x y hxy
  unfold occupancy
  have hx : 0 < 1 + Real.exp (-x) := by positivity
  have hy : 0 < 1 + Real.exp (-y) := by positivity
  have : Real.exp (-y) < Real.exp (-x) := Real.exp_lt_exp.mpr (by linarith)
  rw [div_lt_div_iff₀ hx hy]
  linarith

/-! ## G19: what the forced growth law generates

With independent ends the co-activity is a product of mean rates, and with one shared
content pool the field is `κ r_i r_j - θ`. -/

/-- **G19a (one pool grows a rich club)**: with one shared pool, the strong set is an
upper set in the rates: if `i` is strongly connected to `j` and `k` fires at least as
much as `i`, then `k` is too. Neighbourhoods are nested (a threshold graph). -/
theorem one_pool_nested {κ θ τ : ℝ} (hκ : 0 ≤ κ) {ri rk rj : ℝ} (hrj : 0 ≤ rj) (hik : ri ≤ rk)
    (h : τ ≤ occupancy (κ * ri * rj - θ)) : τ ≤ occupancy (κ * rk * rj - θ) := by
  refine h.trans (occupancy_strictMono.monotone ?_)
  have : κ * ri * rj ≤ κ * rk * rj :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hik hκ) hrj
  linarith

/-- The co-activity statistic of a stream given as a joint law `P prev cur` over pairs
of consecutive symbols, for end states `φ i` that depend on the pair. -/
def coactivity {B ι : Type*} [Fintype B] (P : B → B → ℝ) (φ : ι → B → B → ℝ) (i j : ι) : ℝ :=
  ∑ a, ∑ b, P a b * (φ i a b * φ j a b)

/-- **G19b (memoryless ends see only symbol frequencies)**: if every end state depends
on the current symbol only, the co-activity, hence the grown structure, depends on the
stream only through the frequencies of the current symbol. -/
theorem memoryless_coactivity {B ι : Type*} [Fintype B] (P : B → B → ℝ) (ψ : ι → B → ℝ)
    (i j : ι) :
    coactivity P (fun k _ b => ψ k b) i j = ∑ b, (∑ a, P a b) * (ψ i b * ψ j b) := by
  unfold coactivity
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun b _ => by rw [Finset.sum_mul]

theorem memoryless_same_frequencies_same_structure {B ι : Type*} [Fintype B]
    (P Q : B → B → ℝ) (hPQ : ∀ b, ∑ a, P a b = ∑ a, Q a b) (ψ : ι → B → ℝ) (i j : ι) :
    coactivity P (fun k _ b => ψ k b) i j = coactivity Q (fun k _ b => ψ k b) i j := by
  rw [memoryless_coactivity, memoryless_coactivity]
  exact Finset.sum_congr rfl fun b _ => by rw [hPQ b]

/-- Two binary streams with the same symbol frequencies (1/2 each) but opposite
order: one alternates, one repeats. -/
noncomputable def alternating : Fin 2 → Fin 2 → ℝ := fun a b => if a = b then 0 else 1 / 2
noncomputable def repeating : Fin 2 → Fin 2 → ℝ := fun a b => if a = b then 1 / 2 else 0

/-- **G19c (sequence structure needs memory at the ends)**: the two streams have equal
frequencies, so memoryless ends grow the same structure (G19b); an end that holds the
previous symbol (`i` active on the previous 1, `j` on the current 1) separates them. -/
theorem memory_separates :
    (∀ b, ∑ a, alternating a b = ∑ a, repeating a b) ∧
    coactivity alternating (fun (k : Fin 2) a b => if k = 0 then (a : ℝ) else (b : ℝ)) 0 1 ≠
      coactivity repeating (fun (k : Fin 2) a b => if k = 0 then (a : ℝ) else (b : ℝ)) 0 1 := by
  refine ⟨fun b => ?_, ?_⟩
  · fin_cases b <;> simp [alternating, repeating, Fin.sum_univ_two]
  · simp [coactivity, alternating, repeating, Fin.sum_univ_two]

/-! ## G20: the form of memory

A carrier of the past (G11, G19c, `delayed_credit_requires_history_carrier`) is a
population of two-state units (or two compartments exchanging content). -/

/-- One tick of a population of two-state units: up with probability `kp`, down with
`km`; `p` is the fraction up. -/
def twoStateStep (kp km p : ℝ) : ℝ := p + kp * (1 - p) - km * p

def twoStateIter (kp km p₀ : ℝ) : ℕ → ℝ
  | 0 => p₀
  | t + 1 => twoStateStep kp km (twoStateIter kp km p₀ t)

/-- **G20a (two-state memory is geometric)**: the deviation from the fixed point
`p* = kp / (kp + km)` shrinks by the same factor `1 - kp - km` every tick, so the
trace of a past input is exponential; detailed balance fixes `p*`, not the form. -/
theorem two_state_geometric (kp km p₀ : ℝ) (h : kp + km ≠ 0) (t : ℕ) :
    twoStateIter kp km p₀ t - kp / (kp + km) =
      (1 - kp - km) ^ t * (p₀ - kp / (kp + km)) := by
  induction t with
  | zero => simp [twoStateIter]
  | succ n ih =>
    simp only [twoStateIter, twoStateStep, pow_succ]
    have e : twoStateIter kp km p₀ n = kp / (kp + km) + (1 - kp - km) ^ n * (p₀ - kp / (kp + km)) := by
      linarith
    rw [e]
    field_simp
    ring

/-- **G20b (symmetric exchange)**: two compartments exchanging content at the same rate
`D` both ways (G6DB) keep their total and their difference shrinks by `1 - 2D`. -/
theorem symmetric_exchange (D x y : ℝ) :
    let x' := x - D * (x - y)
    let y' := y + D * (x - y)
    x' + y' = x + y ∧ x' - y' = (1 - 2 * D) * (x - y) := by
  constructor <;> ring

/-- **G21 (no differentiation without fluctuation)**: deterministic dynamics `f` that
commute with a relabelling `σ` (P2c) keep a `σ`-symmetric birth state (P3) symmetric
forever. Identical units born identically and driven identically stay identical; their
differentiation needs fluctuations, i.e. the stochastic medium, not its mean field. -/
theorem symmetric_stays_symmetric {X : Type*} (f σ : X → X) (hf : ∀ x, f (σ x) = σ (f x))
    (x₀ : X) (h₀ : σ x₀ = x₀) (t : ℕ) : σ (f^[t] x₀) = f^[t] x₀ := by
  induction t with
  | zero => simpa using h₀
  | succ n ih => rw [Function.iterate_succ_apply', ← hf, ih]

/-! ## G22: finite local reservoirs (refractoriness, fatigue, depression, scaling)

One law at every time scale: a local reservoir is a two-state population (available /
used); at its fixed point the available fraction is `occupancy (log supply - log demand)`
(G20a's `kp / (kp + km)`), so more use leaves less available. -/

/-- The fixed point of G20a is the occupancy of the log odds `log kp - log km`. -/
theorem fixed_point_is_occupancy {kp km : ℝ} (hp : 0 < kp) (hm : 0 < km) :
    kp / (kp + km) = occupancy (Real.log kp - Real.log km) := by
  unfold occupancy
  rw [show -(Real.log kp - Real.log km) = Real.log km - Real.log kp by ring,
    Real.exp_sub, Real.exp_log hp, Real.exp_log hm]
  field_simp

/-- **G22a (use depletes)**: the available fraction strictly falls as demand rises. -/
theorem available_strictAnti {kp : ℝ} (hp : 0 < kp) {m₁ m₂ : ℝ} (h1 : 0 < m₁) (h : m₁ < m₂) :
    kp / (kp + m₂) < kp / (kp + m₁) := by
  apply div_lt_div_of_pos_left hp (by linarith) (by linarith)

/-- **G22b (equilibrium inhibition retreats from co-activity)**: a slot species whose
energy raises co-activity's cost (`κ < 0`) has occupancy strictly falling in co-activity. -/
theorem inhibitory_slot_antitone {κ θ : ℝ} (hκ : κ < 0) : StrictAnti (fun c => occupancy (κ * c - θ)) := by
  intro x y hxy
  exact occupancy_strictMono (by nlinarith)

/-- **G22c (no rich club needs local potentials)**: if all compartments share one potential
and the strong set is not nested in rate order, contradiction (G19a). So a medium whose
grown structure is not a rate-ordered threshold graph has compartment-local potentials. -/
theorem shared_potential_forces_nesting {κ θ τ : ℝ} (hκ : 0 ≤ κ) {ri rk rj : ℝ} (hrj : 0 ≤ rj)
    (hik : ri ≤ rk) (hstrong : τ ≤ occupancy (κ * ri * rj - θ))
    (hnot : occupancy (κ * rk * rj - θ) < τ) : False :=
  absurd (one_pool_nested hκ hrj hik hstrong) (not_le.mpr hnot)

end CGB.Generation
