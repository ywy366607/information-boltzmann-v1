import CGBAxioms.GenerationGDNLife
import CGBAxioms.GenerationAssociationCapacity

/-!
# Context association: what a language memory must write, and where

`GDNLife.assimilate` writes an arrival at that arrival's own address and adds
its value.  This file records the consequence: between time releases, the
natural and precision arrays are a sum over arrivals, so they cannot hold the
order of two arrivals.  A readout of that state therefore cannot serve any
requirement that separates two orderings.

The second part states the smallest alternative on the same conjugate
arithmetic.  The arrival is written at the address that was read to predict
it, so a cell accumulates what followed its context.  Nothing here adds a
history page: the cell totals are the state, and the list decomposition is a
mathematical description only.

The third part states what a recurrent context must satisfy to act as an
address.  An address is useful only if the same recent bytes reach it again.
A contraction on arrivals gives that repeatability; a coordinate that the
arrivals leave untouched keeps its old value for every common suffix.  A
contraction also has at most one rest point under a held arrival, so the
address-forming part cannot be the part that latches.

These are necessary and structural statements.  They do not prove language
capability, and they do not choose a concrete address map.
-/

namespace CGB.GDNLife

variable {k d : ℕ}

theorem state_ext (s t : State k d)
    (h1 : s.natural = t.natural) (h2 : s.precision = t.precision)
    (h3 : s.velocity = t.velocity) (h4 : s.velocityPrecision = t.velocityPrecision)
    (h5 : s.lastByte = t.lastByte)
    (h6 : s.lastObservationPrecision = t.lastObservationPrecision)
    (h7 : s.head = t.head) (h8 : s.bias = t.bias) : s = t := by
  cases s
  cases t
  simp_all

/-- Natural parameters after a list of arrivals are the initial array plus one
summand per arrival.  Order enters only through a list sum. -/
theorem foldl_assimilate_natural (M : Model k d) (l : List Arrival)
    (s : State k d) (j : Fin k) (i : Fin d) :
    (l.foldl (assimilate M) s).natural j i =
      s.natural j i + (l.map fun a => incoming M a j * M.value a.byte i).sum := by
  induction l generalizing s with
  | nil => simp
  | cons a l ih =>
      simp only [List.foldl_cons, List.map_cons, List.sum_cons]
      rw [ih]
      simp only [assimilate]
      ring

theorem foldl_assimilate_precision (M : Model k d) (l : List Arrival)
    (s : State k d) (j : Fin k) :
    (l.foldl (assimilate M) s).precision j =
      s.precision j + (l.map fun a => incoming M a j).sum := by
  induction l generalizing s with
  | nil => simp
  | cons a l ih =>
      simp only [List.foldl_cons, List.map_cons, List.sum_cons]
      rw [ih]
      simp only [assimilate]
      ring

theorem foldl_assimilate_velocity (M : Model k d) (l : List Arrival) (s : State k d) :
    (l.foldl (assimilate M) s).velocity = s.velocity ∧
      (l.foldl (assimilate M) s).velocityPrecision = s.velocityPrecision ∧
      (l.foldl (assimilate M) s).head = s.head ∧
      (l.foldl (assimilate M) s).bias = s.bias := by
  induction l generalizing s with
  | nil => simp
  | cons a l ih =>
      simpa [assimilate] using ih (assimilate M s a)

/-- A positive-confidence arrival replaces both last-arrival fields. -/
theorem assimilate_last_fields (M : Model k d) (s : State k d) (c : Arrival)
    (hc : 0 < c.observationPrecision) :
    (assimilate M s c).lastByte = some c.byte ∧
      (assimilate M s c).lastObservationPrecision = some c.observationPrecision := by
  constructor
  · cases h : s.lastByte <;> simp [assimilate, h, hc]
  · cases h : s.lastObservationPrecision <;> simp [assimilate, h, hc]

/-- **Order blindness.**  Any reordering of the arrivals before a common
positive-confidence arrival reaches exactly the same complete state.  Elapsed
time acts only through `prior`, which is the identity at zero elapsed time, so
this is the state reached across the zero-elapsed bytes of one packet. -/
theorem assimilate_order_blind (M : Model k d) (s : State k d) {l l' : List Arrival}
    (hp : l.Perm l') (c : Arrival) (hc : 0 < c.observationPrecision) :
    (l ++ [c]).foldl (assimilate M) s = (l' ++ [c]).foldl (assimilate M) s := by
  simp only [List.foldl_append, List.foldl_cons, List.foldl_nil]
  obtain ⟨hv, hvp, hh, hb⟩ := foldl_assimilate_velocity M l s
  obtain ⟨hv', hvp', hh', hb'⟩ := foldl_assimilate_velocity M l' s
  obtain ⟨hl, ho⟩ := assimilate_last_fields M (l.foldl (assimilate M) s) c hc
  obtain ⟨hl', ho'⟩ := assimilate_last_fields M (l'.foldl (assimilate M) s) c hc
  apply state_ext
  · funext j i
    simp only [assimilate]
    rw [foldl_assimilate_natural, foldl_assimilate_natural,
      (hp.map fun a => incoming M a j * M.value a.byte i).sum_eq]
  · funext j
    simp only [assimilate]
    rw [foldl_assimilate_precision, foldl_assimilate_precision,
      (hp.map fun a => incoming M a j).sum_eq]
  · simpa [assimilate] using hv.trans hv'.symm
  · simpa [assimilate] using hvp.trans hvp'.symm
  · rw [hl, hl']
  · rw [ho, ho']
  · simpa [assimilate] using hh.trans hh'.symm
  · simpa [assimilate] using hb.trans hb'.symm

/-- Order-blind states issue the same byte distribution. -/
theorem order_blind_same_allocation (M : Model k d) (s : State k d) {l l' : List Arrival}
    (hp : l.Perm l') (c : Arrival) (hc : 0 < c.observationPrecision) :
    allocation M ((l ++ [c]).foldl (assimilate M) s) =
      allocation M ((l' ++ [c]).foldl (assimilate M) s) := by
  rw [assimilate_order_blind M s hp c hc]

/-- **Necessity.**  No readout of this state can serve a requirement that
separates two orderings.  This is the state-merging half of
`causal_carrier_contract_iff`, instantiated at the actual GDN transition. -/
theorem no_read_separates_order {C : Type*} (M : Model k d) (s : State k d)
    {l l' : List Arrival} (hp : l.Perm l') (c : Arrival)
    (hc : 0 < c.observationPrecision) (required : List Arrival → C)
    (hreq : required (l ++ [c]) ≠ required (l' ++ [c])) :
    ¬ ∃ read : State k d → C, ∀ h, read (h.foldl (assimilate M) s) = required h := by
  rintro ⟨read, hread⟩
  apply hreq
  rw [← hread, ← hread, assimilate_order_blind M s hp c hc]

end CGB.GDNLife

namespace CGB.Generation.ContextAssociation

open Finset

abbrev Byte := Fin 256

/-- The same conjugate arithmetic with one change of key: the address belongs
to the context that predicted the arrival, not to the arrival. -/
structure Model (Ctx : Type*) (k d : ℕ) where
  value : Byte → Fin d → ℝ
  address : Ctx → Fin k → ℝ
  address_nonneg : ∀ c j, 0 ≤ address c j
  evidence : Byte → ℝ
  evidence_nonneg : ∀ b, 0 ≤ evidence b
  advance : Ctx → Byte → Ctx

structure State (Ctx : Type*) (k d : ℕ) where
  natural : Fin k → Fin d → ℝ
  precision : Fin k → ℝ
  context : Ctx

variable {Ctx : Type*} {k d : ℕ}

noncomputable def mean (s : State Ctx k d) (j : Fin k) (i : Fin d) : ℝ :=
  s.natural j i / s.precision j

/-- A read through one context's address. -/
noncomputable def read (M : Model Ctx k d) (s : State Ctx k d) (c : Ctx) (i : Fin d) : ℝ :=
  ∑ j, M.address c j * mean s j i

/-- Prediction is read from `s.context`; the arrival is then written there. -/
noncomputable def write (M : Model Ctx k d) (s : State Ctx k d) (b : Byte) : State Ctx k d where
  natural := fun j i => s.natural j i + M.address s.context j * M.evidence b * M.value b i
  precision := fun j => s.precision j + M.address s.context j * M.evidence b
  context := M.advance s.context b

/-- What a cell received along a list of arrivals from a starting context. -/
noncomputable def cellNatural (M : Model Ctx k d) (j : Fin k) (i : Fin d) : Ctx → List Byte → ℝ
  | _, [] => 0
  | c, b :: l => M.address c j * M.evidence b * M.value b i + cellNatural M j i (M.advance c b) l

noncomputable def cellPrecision (M : Model Ctx k d) (j : Fin k) : Ctx → List Byte → ℝ
  | _, [] => 0
  | c, b :: l => M.address c j * M.evidence b + cellPrecision M j (M.advance c b) l

/-- A cell's natural total is the evidence-weighted sum of the values that
followed its contexts.  This is a decomposition of the stored number. -/
theorem foldl_write_natural (M : Model Ctx k d) (l : List Byte) (s : State Ctx k d)
    (j : Fin k) (i : Fin d) :
    (l.foldl (write M) s).natural j i = s.natural j i + cellNatural M j i s.context l := by
  induction l generalizing s with
  | nil => simp [cellNatural]
  | cons b l ih =>
      simp only [List.foldl_cons, cellNatural]
      rw [ih]
      simp only [write]
      ring

theorem foldl_write_precision (M : Model Ctx k d) (l : List Byte) (s : State Ctx k d)
    (j : Fin k) :
    (l.foldl (write M) s).precision j = s.precision j + cellPrecision M j s.context l := by
  induction l generalizing s with
  | nil => simp [cellPrecision]
  | cons b l ih =>
      simp only [List.foldl_cons, cellPrecision]
      rw [ih]
      simp only [write]
      ring

/-- **Conditional statistic.**  A cell's read is the precision-weighted mean of
what followed its contexts, combined with its prior. -/
theorem cell_mean_is_conditional_statistic (M : Model Ctx k d) (l : List Byte)
    (s : State Ctx k d) (j : Fin k) (i : Fin d) :
    mean (l.foldl (write M) s) j i =
      (s.natural j i + cellNatural M j i s.context l) /
        (s.precision j + cellPrecision M j s.context l) := by
  simp only [mean, foldl_write_natural, foldl_write_precision]

/-- A hard-addressed write is the Bayesian innovation step at that cell. -/
theorem hard_cell_innovation (M : Model Ctx k d) (s : State Ctx k d) (b : Byte)
    (j : Fin k) (i : Fin d) (hj : M.address s.context j = 1) (hp : 0 < s.precision j) :
    mean (write M s b) j i =
      mean s j i + (M.evidence b / (s.precision j + M.evidence b)) *
        (M.value b i - mean s j i) := by
  have hnat : s.precision j * mean s j i = s.natural j i := by
    simp only [mean]
    field_simp
  have h := precision_assimilation_is_innovation (s.precision j) (M.evidence b)
    (mean s j i) (M.value b i) hp (M.evidence_nonneg b)
  rw [hnat] at h
  simp only [mean, write, hj, one_mul] at h ⊢
  exact h

/-- Writes leave every cell outside the writing context's support unchanged. -/
theorem write_outside_support (M : Model Ctx k d) (s : State Ctx k d) (b : Byte)
    (j : Fin k) (i : Fin d) (hj : M.address s.context j = 0) :
    mean (write M s b) j i = mean s j i := by
  simp [mean, write, hj]

/-- **No interference.**  A context whose support is disjoint from the writer
reads exactly what it read before.  Its conditional statistic is its own. -/
theorem disjoint_context_read_unchanged (M : Model Ctx k d) (s : State Ctx k d) (b : Byte)
    (c : Ctx) (i : Fin d) (hdisjoint : ∀ j, M.address s.context j * M.address c j = 0) :
    read M (write M s b) c i = read M s c i := by
  simp only [read]
  apply Finset.sum_congr rfl
  intro j _
  rcases mul_eq_zero.mp (hdisjoint j) with h | h
  · rw [write_outside_support M s b j i h]
  · simp [h]

/-- **Order separation.**  Unlike `assimilate_order_blind`, the context write
holds the order of two arrivals whenever the cell of one successor context
sees the value of the other arrival. -/
theorem context_write_separates_order (M : Model Ctx k d) (s : State Ctx k d)
    (a b : Byte) (j : Fin k) (i : Fin d)
    (hstart : M.address s.context j = 0)
    (ha : M.address (M.advance s.context a) j = 1)
    (hb : M.address (M.advance s.context b) j = 0)
    (he : 0 < M.evidence b) (hv : M.value b i ≠ 0) :
    ([a, b].foldl (write M) s).natural j i ≠ ([b, a].foldl (write M) s).natural j i := by
  simp only [List.foldl_cons, List.foldl_nil, write, hstart, ha, hb]
  intro h
  have : M.evidence b * M.value b i = 0 := by nlinarith
  rcases mul_eq_zero.mp this with h0 | h0
  · exact he.ne' h0
  · exact hv h0

/-- **Address must refine what the future separates.**  Two contexts sharing an
address share one prediction, and a common prediction pays at least this
squared excess across two different required means.  With
`collision_of_excess_keys`, this is why the key count bounds the separable
contexts. -/
theorem same_address_same_read (M : Model Ctx k d) (s : State Ctx k d) (c c' : Ctx)
    (h : M.address c = M.address c') (i : Fin d) : read M s c i = read M s c' i := by
  simp [read, h]

theorem common_prediction_excess (p m m' : ℝ) :
    (m - m') ^ 2 / 2 ≤ (p - m) ^ 2 + (p - m') ^ 2 := by
  nlinarith [sq_nonneg (2 * p - m - m')]

/-- **Fading context.**  If each arrival contracts distance, a common suffix
of arrivals shrinks the difference geometrically, whatever the older history. -/
theorem common_suffix_distance {X : Type*} [PseudoMetricSpace X]
    (adv : X → Byte → X) {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hc : ∀ b x y, dist (adv x b) (adv y b) ≤ ρ * dist x y)
    (l : List Byte) (x y : X) :
    dist (l.foldl adv x) (l.foldl adv y) ≤ ρ ^ l.length * dist x y := by
  induction l generalizing x y with
  | nil => simp
  | cons b l ih =>
      simp only [List.foldl_cons, List.length_cons]
      calc dist (l.foldl adv (adv x b)) (l.foldl adv (adv y b))
          ≤ ρ ^ l.length * dist (adv x b) (adv y b) := ih _ _
        _ ≤ ρ ^ l.length * (ρ * dist x y) :=
            mul_le_mul_of_nonneg_left (hc b x y) (pow_nonneg hρ _)
        _ = ρ ^ (l.length + 1) * dist x y := by ring

/-- **Repeatable address.**  A Lipschitz address of a fading context returns
to within this bound after a common suffix. -/
theorem common_suffix_address_bound {X Y : Type*} [PseudoMetricSpace X] [PseudoMetricSpace Y]
    (adv : X → Byte → X) (q : X → Y) {ρ L : ℝ} (hρ : 0 ≤ ρ) (hL : 0 ≤ L)
    (hc : ∀ b x y, dist (adv x b) (adv y b) ≤ ρ * dist x y)
    (hq : ∀ x y, dist (q x) (q y) ≤ L * dist x y)
    (l : List Byte) (x y : X) :
    dist (q (l.foldl adv x)) (q (l.foldl adv y)) ≤ L * ρ ^ l.length * dist x y := by
  calc dist (q (l.foldl adv x)) (q (l.foldl adv y))
      ≤ L * dist (l.foldl adv x) (l.foldl adv y) := hq _ _
    _ ≤ L * (ρ ^ l.length * dist x y) :=
        mul_le_mul_of_nonneg_left (common_suffix_distance adv hρ hc l x y) hL
    _ = L * ρ ^ l.length * dist x y := by ring

/-- **A held coordinate never fades.**  If arrivals leave a coordinate
untouched, two contexts differing there still differ after every common suffix,
so an address reading it is not a function of recent language.  This is the
`torch.where(local_write_support, proposed, activity)` branch at unrouted sites. -/
theorem held_coordinate_survives_suffix {X Y : Type*} (adv : X → Byte → X) (proj : X → Y)
    (l : List Byte) (hheld : ∀ x, ∀ b ∈ l, proj (adv x b) = proj x) (x y : X)
    (hxy : proj x ≠ proj y) : proj (l.foldl adv x) ≠ proj (l.foldl adv y) := by
  have key : ∀ (l : List Byte), (∀ x, ∀ b ∈ l, proj (adv x b) = proj x) →
      ∀ x, proj (l.foldl adv x) = proj x := by
    intro l
    induction l with
    | nil => simp
    | cons b l ih =>
        intro h x
        simp only [List.foldl_cons]
        rw [ih (fun x c hc => h x c (List.mem_cons_of_mem b hc)), h x b List.mem_cons_self]
  rw [key l hheld x, key l hheld y]
  exact hxy

/-- **Separation of roles.**  A contraction under a held arrival has at most
one rest point, so an address-forming state cannot also hold two latched
alternatives.  Persistence of learned distinctions must live elsewhere. -/
theorem contraction_has_no_two_latches {X : Type*} [MetricSpace X]
    (adv : X → Byte → X) {ρ : ℝ} (hρ : ρ < 1)
    (hc : ∀ b x y, dist (adv x b) (adv y b) ≤ ρ * dist x y)
    (b : Byte) (x y : X) (hx : adv x b = x) (hy : adv y b = y) : x = y := by
  have h := hc b x y
  rw [hx, hy] at h
  have hnn : 0 ≤ dist x y := dist_nonneg
  have : dist x y = 0 := by nlinarith
  exact dist_eq_zero.mp this

end CGB.Generation.ContextAssociation
