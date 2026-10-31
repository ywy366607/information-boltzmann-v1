import CGBAxioms.GenerationSelfProduction
import CGBAxioms.GenerationInformationProjection

/-!
# Real-valued GDN carrier on one open trajectory

The body is a fixed-dimensional real state, not a finite enumeration. This is
an analytic reference for the single-observation byte port in
`information_boltzmann/core/cgb_prequential_bridge.py`: motion updating is off;
process variance releases precision, assimilation adds natural parameters,
and the affine head uses the feature saved before the byte arrived.

`Model` fixes the byte encoder/address/likelihood maps. They can be extracted
from a fixed encoder and address network; they do not learn in this module.
The causal query is the last arrived byte's address (a fixed seed query at
birth). This instantiates one query, not all arbitrary Python callbacks.
Velocity is carried unchanged, as in this byte-port path. Floating-point
refinement, learned queries, autonomous policy learning, physical maintenance
and a world supplying repeated future benefit remain separate obligations.
-/

namespace CGB.GDNLife

open Finset CGB.SelfProduction

abbrev Byte := Fin 256

/-- No history buffer: the previous byte determines the fixed encoder's last
observation. The head parameters are part of the very same real body. -/
structure State (k d : ℕ) where
  natural : Fin k → Fin d → ℝ
  precision : Fin k → ℝ
  velocity : Fin 2 → ℝ
  velocityPrecision : ℝ
  lastByte : Option Byte
  lastObservationPrecision : Option ℝ
  head : Byte → Fin d → ℝ
  bias : Byte → ℝ

/-- Finite arrays of encoder/address values, not a table of body transitions.
All state evolution below is explicit arithmetic on real tensors. -/
structure Model (k d : ℕ) where
  value : Byte → Fin d → ℝ
  addressLogits : Byte → Fin k → ℝ
  seedLogits : Fin k → ℝ
  likelihoodPrecision : Byte → ℝ
  likelihood_nonneg : ∀ b, 0 ≤ likelihoodPrecision b
  processVariance : ℝ
  process_nonneg : 0 ≤ processVariance
  headStep : ℝ
  precisionFloor : ℝ
  floor_pos : 0 < precisionFloor

inductive Source where
  | external
  | self

/-- Time and source are inputs, not additional symbols in the 256-byte
scoring alphabet. Elapsed time is known before prediction; the byte is not. -/
structure Arrival where
  elapsed : ℝ
  elapsed_nonneg : 0 ≤ elapsed
  byte : Byte
  observationPrecision : ℝ
  observation_nonneg : 0 ≤ observationPrecision
  source : Source

variable {k d : ℕ}

noncomputable def address (M : Model k d) (b : Byte) : Fin k → ℝ :=
  CGB.Medium.softmax (M.addressLogits b)

theorem address_pos (M : Model k d) (b : Byte) (j : Fin k) :
    0 < address M b j := by
  unfold address CGB.Medium.softmax
  exact div_pos (Real.exp_pos _)
    (Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨j, Finset.mem_univ _⟩)

noncomputable def query (M : Model k d) (s : State k d) : Fin k → ℝ :=
  match s.lastByte with
  | none => CGB.Medium.softmax M.seedLogits
  | some b => address M b

noncomputable def mean (M : Model k d) (s : State k d) (j : Fin k) (i : Fin d) : ℝ :=
  s.natural j i / max (s.precision j) M.precisionFloor

noncomputable def feature (M : Model k d) (s : State k d) : Fin d → ℝ :=
  fun i => ∑ j, mean M s j i * query M s j

noncomputable def logits (s : State k d) (f : Fin d → ℝ) : Byte → ℝ :=
  fun b => s.bias b + ∑ i, s.head b i * f i

noncomputable def allocation (M : Model k d) (s : State k d) : Byte → ℝ :=
  CGB.Medium.softmax (logits s (feature M s))

/-- The action-score factor for one affine language-head coordinate, read from
the body immediately before a byte is emitted.  It is a local likelihood
identity, not a stored Jacobian or a second eligibility array. -/
noncomputable def actionHeadScore (M : Model k d) (s : State k d)
    (emitted coordinate : Byte) (i : Fin d) : ℝ :=
  feature M s i * ((if emitted = coordinate then 1 else 0) - allocation M s coordinate)

/-- The existing G14 log-softmax law instantiated at the GDN language head.
Thus the first two factors of delayed action credit are fixed by the pre-action
body and emitted byte; only their persistence until an external outcome is a
separate causal-carrier obligation. -/
theorem action_head_score_deriv (M : Model k d) (s : State k d)
    (emitted coordinate : Byte) (i : Fin d) :
    HasDerivAt
      (fun θ => Real.log (CGB.Medium.softmax
        (fun b => logits s (feature M s) b +
          if b = coordinate then θ * feature M s i else 0) emitted))
      (actionHeadScore M s emitted coordinate i) 0 := by
  simpa [actionHeadScore, allocation] using
    (CGB.Medium.log_softmax_deriv (logits s (feature M s)) coordinate emitted
      (feature M s i))

def Positive (s : State k d) : Prop := ∀ j, 0 < s.precision j

/-- Exactly the byte port's zero-time no-op and clamped process prior.
`eta / precision` is not confused with the stored natural parameter. -/
noncomputable def prior (M : Model k d) (s : State k d) (elapsed : ℝ) : State k d := by
  classical
  let p := fun j => 1 / (1 / max (s.precision j) M.precisionFloor +
    elapsed * M.processVariance)
  exact if elapsed = 0 then s else
    { s with precision := p, natural := fun j i => p j * mean M s j i }

/-- The prepared feature is held through observation and learning. -/
structure Prepared (k d : ℕ) where
  state : State k d
  feature : Fin d → ℝ

noncomputable def prepare (M : Model k d) (s : State k d) (elapsed : ℝ) : Prepared k d :=
  let p := prior M s elapsed
  ⟨p, feature M p⟩

noncomputable def incoming (M : Model k d) (a : Arrival) (j : Fin k) : ℝ :=
  address M a.byte j * (a.observationPrecision * M.likelihoodPrecision a.byte)

noncomputable def assimilate (M : Model k d) (s : State k d) (a : Arrival) : State k d := by
  classical
  exact { s with
    natural := fun j i => s.natural j i + incoming M a j * M.value a.byte i
    precision := fun j => s.precision j + incoming M a j
    lastByte := match s.lastByte with
      | none => some a.byte
      | some b => if 0 < a.observationPrecision then some a.byte else some b
    lastObservationPrecision := match s.lastObservationPrecision with
      | none => some a.observationPrecision
      | some p => if 0 < a.observationPrecision then some a.observationPrecision else some p }

/-- A finite-difference probe of one fixed byte encoder direction.  This is
not a second memory or an online rule: it changes only the value presented by
the existing arrival transition.  Addressing, likelihood and the causal
last-byte query remain fixed. -/
noncomputable def valuePerturb (M : Model k d)
    (direction : Byte → Fin d → ℝ) (θ : ℝ) : Model k d :=
  { M with value := fun b i => M.value b i + θ * direction b i }

/-- The arrival-local sensitivity of a natural parameter to the encoder
direction.  It is carried by the same incoming flow that performs
assimilation; it is not an auxiliary reward or food variable. -/
noncomputable def valueEligibility (M : Model k d)
    (direction : Byte → Fin d → ℝ) (a : Arrival) (j : Fin k) (i : Fin d) : ℝ :=
  incoming M a j * direction a.byte i

/-- One basis perturbation of the address logits for a byte.  Like
`valuePerturb`, this is an analytic probe of the existing model fields, not
an added recurrent carrier. -/
noncomputable def addressLogitPerturb (M : Model k d) (b₀ : Byte)
    (j₀ : Fin k) (θ : ℝ) : Model k d :=
  { M with addressLogits := fun b j =>
      M.addressLogits b j + if b = b₀ ∧ j = j₀ then θ else 0 }

/-- The current-write contribution of one address-logit coordinate to one
natural parameter.  It is the soft-address complement to `valueEligibility`.
It describes a direction in the existing local write, not an extra trace. -/
noncomputable def addressEligibility (M : Model k d) (a : Arrival)
    (j₀ j : Fin k) (i : Fin d) : ℝ :=
  a.observationPrecision * M.likelihoodPrecision a.byte * address M a.byte j *
    ((if j = j₀ then 1 else 0) - address M a.byte j₀) * M.value a.byte i

/-- Exact softmax derivative for an address-logit basis coordinate.  This is
the G14 finite-alphabet score identity applied to the GDN address, rather
than an assumed attention-gradient rule. -/
theorem addressLogitPerturb_deriv (M : Model k d) (b : Byte)
    (j₀ j : Fin k) :
    HasDerivAt
      (fun θ => address (addressLogitPerturb M b j₀ θ) b j)
      (address M b j * ((if j = j₀ then 1 else 0) - address M b j₀)) 0 := by
  have hlog : HasDerivAt
      (fun θ => Real.log (address (addressLogitPerturb M b j₀ θ) b j))
      ((if j = j₀ then 1 else 0) - address M b j₀) 0 := by
    simpa [address, addressLogitPerturb] using
      (CGB.Medium.log_softmax_deriv (M.addressLogits b) j₀ j 1)
  have hexp := hlog.exp
  have hzero : address (addressLogitPerturb M b j₀ 0) b j = address M b j := by
    simp [address, addressLogitPerturb]
  have heq : (fun θ => Real.exp (Real.log (address (addressLogitPerturb M b j₀ θ) b j))) =
      (fun θ => address (addressLogitPerturb M b j₀ θ) b j) := by
    funext θ
    exact Real.exp_log (address_pos _ _ _)
  rw [heq] at hexp
  rw [hzero] at hexp
  simpa [Real.exp_log (address_pos M b j)] using hexp

/-- The exact address-side contribution to a GDN natural-parameter write.
The address and value paths are separate summands of an encoder's total
derivative; neither may be silently discarded. -/
theorem assimilate_addressLogitPerturb_natural_deriv (M : Model k d)
    (s : State k d) (a : Arrival) (j₀ j : Fin k) (i : Fin d) :
    HasDerivAt
      (fun θ => (assimilate (addressLogitPerturb M a.byte j₀ θ) s a).natural j i)
      (addressEligibility M a j₀ j i) 0 := by
  have h := addressLogitPerturb_deriv M a.byte j₀ j
  change HasDerivAt
    (fun θ => s.natural j i +
      incoming (addressLogitPerturb M a.byte j₀ θ) a j * M.value a.byte i)
    (addressEligibility M a j₀ j i) 0
  have hincoming : HasDerivAt
      (fun θ => incoming (addressLogitPerturb M a.byte j₀ θ) a j)
      (a.observationPrecision * M.likelihoodPrecision a.byte * address M a.byte j *
        ((if j = j₀ then 1 else 0) - address M a.byte j₀)) 0 := by
    change HasDerivAt
      (fun θ => address (addressLogitPerturb M a.byte j₀ θ) a.byte j *
        (a.observationPrecision * M.likelihoodPrecision a.byte)) _ 0
    convert h.mul_const (a.observationPrecision * M.likelihoodPrecision a.byte) using 1; ring
  convert (hincoming.mul_const (M.value a.byte i)).const_add (s.natural j i) using 1;
    simp [addressEligibility, mul_left_comm, mul_comm]

/-- The same address perturbation changes precision by the address-side
incoming derivative.  It is part of the future state sensitivity and cannot
be represented by the value-only result. -/
theorem assimilate_addressLogitPerturb_precision_deriv (M : Model k d)
    (s : State k d) (a : Arrival) (j₀ j : Fin k) :
    HasDerivAt
      (fun θ => (assimilate (addressLogitPerturb M a.byte j₀ θ) s a).precision j)
      (a.observationPrecision * M.likelihoodPrecision a.byte * address M a.byte j *
        ((if j = j₀ then 1 else 0) - address M a.byte j₀)) 0 := by
  have h := addressLogitPerturb_deriv M a.byte j₀ j
  change HasDerivAt
    (fun θ => s.precision j + incoming (addressLogitPerturb M a.byte j₀ θ) a j) _ 0
  have hincoming : HasDerivAt
      (fun θ => incoming (addressLogitPerturb M a.byte j₀ θ) a j)
      (a.observationPrecision * M.likelihoodPrecision a.byte * address M a.byte j *
        ((if j = j₀ then 1 else 0) - address M a.byte j₀)) 0 := by
    change HasDerivAt
      (fun θ => address (addressLogitPerturb M a.byte j₀ θ) a.byte j *
        (a.observationPrecision * M.likelihoodPrecision a.byte)) _ 0
    convert h.mul_const (a.observationPrecision * M.likelihoodPrecision a.byte) using 1; ring
  exact hincoming.const_add (s.precision j)

/-- The positive likelihood parametrization used by `ActiveInferenceGDN2`:
`softplus(raw)` is learned evidence confidence, while positivity is structural. -/
noncomputable def likelihoodSoftplus (x : ℝ) : ℝ := Real.log (1 + Real.exp x)

/-- The local slope of the learned likelihood logit. -/
noncomputable def likelihoodSlope (x : ℝ) : ℝ := Real.exp x / (1 + Real.exp x)

/-- Exact derivative of the actual softplus confidence parametrization. -/
theorem likelihoodSoftplus_deriv (r : ℝ) :
    HasDerivAt (fun θ : ℝ => likelihoodSoftplus (r + θ)) (likelihoodSlope r) 0 := by
  unfold likelihoodSoftplus likelihoodSlope
  have hsum := (hasDerivAt_const (0 : ℝ) 1).add
    ((hasDerivAt_const (0 : ℝ) r).add (hasDerivAt_id (0 : ℝ))).exp
  have hinner : HasDerivAt (fun θ : ℝ => 1 + Real.exp (r + θ)) (Real.exp r) 0 := by
    convert hsum using 1
    · ext θ
      change 1 + Real.exp (r + θ) = 1 + Real.exp (r + θ)
      rfl
    · simp
  have hlog := hinner.log (by
    simpa using (by positivity : (1 + Real.exp r) ≠ 0))
  convert hlog using 1
  simp

/-- A perturbation of the existing likelihood logit for one byte.  It changes
no state field, value, address or query; its positivity is inherited from the
same softplus parametrization used by the runtime. -/
noncomputable def likelihoodLogitPerturb (M : Model k d) (b₀ : Byte)
    (raw θ : ℝ) : Model k d :=
  { value := M.value
    addressLogits := M.addressLogits
    seedLogits := M.seedLogits
    likelihoodPrecision := fun b =>
      if b = b₀ then likelihoodSoftplus (raw + θ) else M.likelihoodPrecision b
    likelihood_nonneg := by
      intro b
      split_ifs with h
      · exact (Real.log_pos (by nlinarith [Real.exp_pos (raw + θ)])).le
      · exact M.likelihood_nonneg b
    processVariance := M.processVariance
    process_nonneg := M.process_nonneg
    headStep := M.headStep
    precisionFloor := M.precisionFloor
    floor_pos := M.floor_pos }

/-- The likelihood side of the current GDN write: a learned confidence logit
changes the natural parameter by its softplus slope times the existing
address, observation precision and value. -/
theorem assimilate_likelihoodLogitPerturb_natural_deriv (M : Model k d)
    (s : State k d) (a : Arrival) (raw : ℝ)
    (j : Fin k) (i : Fin d) :
    HasDerivAt
      (fun θ => (assimilate (likelihoodLogitPerturb M a.byte raw θ) s a).natural j i)
      (a.observationPrecision * address M a.byte j * M.value a.byte i * likelihoodSlope raw) 0 := by
  have hpi := likelihoodSoftplus_deriv raw
  simp only [assimilate, incoming, address, likelihoodLogitPerturb, ↓reduceIte]
  convert (hpi.const_mul (a.observationPrecision * address M a.byte j * M.value a.byte i)).const_add
    (s.natural j i) using 1
  · funext θ
    simp only [address]
    ring
  · simp only [address]

/-- The same learned confidence changes the stored precision.  Thus it cannot
be treated as a fixed write scale while claiming the encoder is trainable. -/
theorem assimilate_likelihoodLogitPerturb_precision_deriv (M : Model k d)
    (s : State k d) (a : Arrival) (raw : ℝ)
    (j : Fin k) :
    HasDerivAt
      (fun θ => (assimilate (likelihoodLogitPerturb M a.byte raw θ) s a).precision j)
      (a.observationPrecision * address M a.byte j * likelihoodSlope raw) 0 := by
  have hpi := likelihoodSoftplus_deriv raw
  simp only [assimilate, incoming, address, likelihoodLogitPerturb, ↓reduceIte]
  convert (hpi.const_mul (a.observationPrecision * address M a.byte j)).const_add
    (s.precision j) using 1
  · funext θ
    simp only [address]
    ring
  · simp only [address]

/-- When the raw logit represents the current confidence, the perturbation
passes through the actual arrival transition at zero exactly.  This establishes
that the two derivative theorems are tangents of the existing model, rather
than of an auxiliary learning rule. -/
theorem assimilate_likelihoodLogitPerturb_zero (M : Model k d) (s : State k d)
    (a : Arrival) (raw : ℝ)
    (hraw : M.likelihoodPrecision a.byte = likelihoodSoftplus raw) :
    assimilate (likelihoodLogitPerturb M a.byte raw 0) s a = assimilate M s a := by
  simp [assimilate, incoming, address, likelihoodLogitPerturb, hraw]

/-- Altering a byte value cannot alter the prediction prepared before that
byte arrives.  This is the causal boundary for its later credit. -/
theorem prepare_valuePerturb (M : Model k d) (direction : Byte → Fin d → ℝ)
    (θ elapsed : ℝ) (s : State k d) :
    prepare (valuePerturb M direction θ) s elapsed = prepare M s elapsed := by
  rfl

/-- Changing the byte encoder leaves the arrival weight unchanged: it does
not silently alter the address or likelihood path. -/
theorem incoming_valuePerturb (M : Model k d) (direction : Byte → Fin d → ℝ)
    (θ : ℝ) (a : Arrival) (j : Fin k) :
    incoming (valuePerturb M direction θ) a j = incoming M a j := by
  rfl

/-- Exact one-arrival finite difference for a byte encoder coordinate.  The
coefficient is an arrival-local eligibility seed.  It is a theorem about the
actual GDN assimilation arithmetic, not a claim that delayed future credit can
already be recovered without propagating this sensitivity. -/
theorem assimilate_valuePerturb_natural (M : Model k d)
    (direction : Byte → Fin d → ℝ) (θ : ℝ) (s : State k d) (a : Arrival)
    (j : Fin k) (i : Fin d) :
    (assimilate (valuePerturb M direction θ) s a).natural j i =
      (assimilate M s a).natural j i + θ * valueEligibility M direction a j i := by
  change s.natural j i + incoming (valuePerturb M direction θ) a j *
      (M.value a.byte i + θ * direction a.byte i) =
    s.natural j i + incoming M a j * M.value a.byte i +
      θ * valueEligibility M direction a j i
  rw [incoming_valuePerturb]
  simp only [valueEligibility]
  ring

/-- The same perturbation leaves precision unchanged.  A future learner can
therefore not attribute an encoder-value effect to a fabricated precision
change in this fixed-address GDN subcase. -/
theorem assimilate_valuePerturb_precision (M : Model k d)
    (direction : Byte → Fin d → ℝ) (θ : ℝ) (s : State k d) (a : Arrival)
    (j : Fin k) :
    (assimilate (valuePerturb M direction θ) s a).precision j =
      (assimilate M s a).precision j := by
  rfl

/-- Comparing two encoder coordinates gives the exact finite difference,
without invoking reverse-time differentiation. -/
theorem assimilate_valuePerturb_difference (M : Model k d)
    (direction : Byte → Fin d → ℝ) (θ₁ θ₂ : ℝ) (s : State k d) (a : Arrival)
    (j : Fin k) (i : Fin d) :
    (assimilate (valuePerturb M direction θ₁) s a).natural j i -
        (assimilate (valuePerturb M direction θ₂) s a).natural j i =
      (θ₁ - θ₂) * valueEligibility M direction a j i := by
  rw [assimilate_valuePerturb_natural, assimilate_valuePerturb_natural]
  ring

/-- No arriving flow means no encoder-value eligibility at that address. -/
theorem valueEligibility_zero_of_no_incoming (M : Model k d)
    (direction : Byte → Fin d → ℝ) (a : Arrival) (j : Fin k) (i : Fin d)
    (h : incoming M a j = 0) : valueEligibility M direction a j i = 0 := by
  simp [valueEligibility, h]

/-- The value-side sensitivity is carried by the same natural state across a
second arrival.  The second local term is added; no history buffer or reverse
computation is introduced. -/
theorem value_sensitivity_two_arrivals (M : Model k d) (s : State k d)
    (direction : Byte → Fin d → ℝ) (θ : ℝ) (a₀ a₁ : Arrival)
    (j : Fin k) (i : Fin d) :
    (assimilate (valuePerturb M direction θ)
      (assimilate (valuePerturb M direction θ) s a₀) a₁).natural j i =
      (assimilate M (assimilate M s a₀) a₁).natural j i +
        θ * (valueEligibility M direction a₀ j i + valueEligibility M direction a₁ j i) := by
  simp only [assimilate, incoming, address, valuePerturb, valueEligibility]
  ring

noncomputable def innovation (s : State k d) (f : Fin d → ℝ) (target : Byte) : Byte → ℝ :=
  fun b => (if b = target then 1 else 0) - CGB.Medium.softmax (logits s f) b

noncomputable def headUpdate (η : ℝ) (s : State k d) (f : Fin d → ℝ)
    (target : Byte) : State k d :=
  { s with
    head := fun b i => s.head b i + η * innovation s f target b * f i
    bias := fun b => s.bias b + η * innovation s f target b }

noncomputable def finish (M : Model k d) (p : Prepared k d) (a : Arrival) : State k d :=
  headUpdate M.headStep (assimilate M p.state a) p.feature a.byte

noncomputable def step (M : Model k d) (s : State k d) (a : Arrival) : State k d :=
  finish M (prepare M s a.elapsed) a

/-- Initial memory has positive precision and no observation history. -/
noncomputable def initial (p : ℝ) (head : Byte → Fin d → ℝ) (bias : Byte → ℝ) : State k d :=
  ⟨fun _ _ => 0, fun _ => p, fun _ => 0, p, none, none, head, bias⟩

theorem initial_positive (p : ℝ) (hp : 0 < p)
    (head : Byte → Fin d → ℝ) (bias : Byte → ℝ) :
    Positive (initial (k := k) p head bias) := fun _ => hp

theorem prior_zero (M : Model k d) (s : State k d) : prior M s 0 = s := by
  simp [prior]

/-- The exact multiplicative transport of an existing natural-state difference
through the GDN time prior.  This is a derived coefficient, not stored state. -/
noncomputable def priorNaturalScale (M : Model k d) (s : State k d)
    (elapsed : ℝ) (j : Fin k) : ℝ :=
  if elapsed = 0 then 1 else
    let p := 1 / (1 / max (s.precision j) M.precisionFloor + elapsed * M.processVariance)
    p / max (s.precision j) M.precisionFloor

/-- If a perturbation leaves a slot's precision unchanged, the prior transports
its natural-state difference by `priorNaturalScale`; no past input is reread. -/
theorem prior_natural_difference (M : Model k d) (s s' : State k d)
    (elapsed : ℝ) (j : Fin k) (i : Fin d)
    (hprecision : s'.precision j = s.precision j) :
    (prior M s' elapsed).natural j i - (prior M s elapsed).natural j i =
      priorNaturalScale M s elapsed j * (s'.natural j i - s.natural j i) := by
  by_cases h : elapsed = 0
  · subst elapsed
    rw [prior_zero, prior_zero]
    simp [priorNaturalScale]
  · simp [prior, h]
    have hmax : max (s'.precision j) M.precisionFloor = max (s.precision j) M.precisionFloor := by
      rw [hprecision]
    simp [priorNaturalScale, h]
    rw [hmax]
    simp only [mean]
    rw [hmax]
    ring

/-- A value-write sensitivity remains in the same state after the time prior.
The result is exact for the present GDN path: the prior rescales the existing
trace instead of allocating a separate history carrier. -/
theorem value_sensitivity_through_prior (M : Model k d) (s : State k d)
    (direction : Byte → Fin d → ℝ) (θ elapsed : ℝ) (a : Arrival)
    (j : Fin k) (i : Fin d) :
    (prior (valuePerturb M direction θ)
      (assimilate (valuePerturb M direction θ) s a) elapsed).natural j i -
      (prior M (assimilate M s a) elapsed).natural j i =
      priorNaturalScale M (assimilate M s a) elapsed j *
        (θ * valueEligibility M direction a j i) := by
  have hprecision :
      (assimilate (valuePerturb M direction θ) s a).precision j =
        (assimilate M s a).precision j :=
    assimilate_valuePerturb_precision M direction θ s a j
  have hprior := prior_natural_difference M
    (assimilate M s a) (assimilate (valuePerturb M direction θ) s a) elapsed j i hprecision
  rw [assimilate_valuePerturb_natural] at hprior
  simpa [prior, mean, valuePerturb] using hprior

/-- With unchanged precision and causal query, an existing natural-state
difference is read out linearly by the same GDN feature map.  This is the
readout side of forward sensitivity, not an auxiliary eligibility array. -/
theorem feature_difference_of_shared_query (M : Model k d) (s s' : State k d)
    (hprecision : ∀ j, s'.precision j = s.precision j)
    (hlast : s'.lastByte = s.lastByte) (i : Fin d) :
    feature M s' i - feature M s i =
      ∑ j, ((s'.natural j i - s.natural j i) / max (s.precision j) M.precisionFloor) *
        query M s j := by
  have hquery : query M s' = query M s := by
    simp [query, hlast]
  rw [feature, feature, hquery, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j _
  simp only [mean]
  rw [hprecision j]
  ring

/-- A perturbation of the byte-value encoder has no direct readout path.
Its only possible effect is the natural-state flow written at arrival. -/
theorem feature_valuePerturb (M : Model k d) (direction : Byte → Fin d → ℝ)
    (θ : ℝ) (s : State k d) :
    feature (valuePerturb M direction θ) s = feature M s := by
  rfl

/-- A positive-confidence arrival makes that byte the next causal query.
This is an existing `lastByte` state field, rather than a new trace. -/
theorem assimilate_lastByte_of_positive (M : Model k d) (s : State k d)
    (a : Arrival) (hpositive : 0 < a.observationPrecision) :
    (assimilate M s a).lastByte = some a.byte := by
  cases hlast : s.lastByte <;> simp [assimilate, hlast, hpositive]

/-- The exact forward eligibility of an encoder-value direction when the
just-written state is read on the following causal byte query.  It is a
derived view of the existing natural state: address-weighted local write,
divided by the existing precision, then read by the same address. -/
noncomputable def valueReadEligibility (M : Model k d)
    (direction : Byte → Fin d → ℝ) (s : State k d) (a : Arrival) (i : Fin d) : ℝ :=
  ∑ j, (valueEligibility M direction a j i /
      max ((assimilate M s a).precision j) M.precisionFloor) * address M a.byte j

/-- One byte-value encoder direction reaches the next causal readout exactly
through `valueReadEligibility`.  No reverse-time graph, copied history, or
separate eligibility state is used: the coefficient is reconstructed from the
same material state that already stores the write. -/
theorem value_read_eligibility_exact (M : Model k d)
    (direction : Byte → Fin d → ℝ) (θ : ℝ) (s : State k d) (a : Arrival)
    (hpositive : 0 < a.observationPrecision) (i : Fin d) :
    feature (valuePerturb M direction θ)
        (assimilate (valuePerturb M direction θ) s a) i =
      feature M (assimilate M s a) i +
        θ * valueReadEligibility M direction s a i := by
  rw [feature_valuePerturb]
  have hprecision : ∀ j,
      (assimilate (valuePerturb M direction θ) s a).precision j =
        (assimilate M s a).precision j := by
    intro j
    exact assimilate_valuePerturb_precision M direction θ s a j
  have hlast :
      (assimilate (valuePerturb M direction θ) s a).lastByte =
        (assimilate M s a).lastByte := by
    rfl
  have hread := feature_difference_of_shared_query M
    (assimilate M s a) (assimilate (valuePerturb M direction θ) s a)
    hprecision hlast i
  have hquery : query M (assimilate M s a) = address M a.byte := by
    simp [query, assimilate_lastByte_of_positive M s a hpositive]
  simp_rw [assimilate_valuePerturb_natural M direction θ s a] at hread
  rw [hquery] at hread
  have hsum :
      (∑ j, (((assimilate M s a).natural j i +
          θ * valueEligibility M direction a j i -
          (assimilate M s a).natural j i) /
          max ((assimilate M s a).precision j) M.precisionFloor) *
          address M a.byte j) =
        θ * valueReadEligibility M direction s a i := by
    rw [valueReadEligibility]
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [hsum] at hread
  linarith

/-- The same eligibility after an elapsed GDN prior.  The time coefficient is
the already-derived `priorNaturalScale`; it is reconstructed from state rather
than stored as an ever-growing history. -/
noncomputable def valueFutureReadEligibility (M : Model k d)
    (direction : Byte → Fin d → ℝ) (s : State k d) (a : Arrival)
    (elapsed : ℝ) (i : Fin d) : ℝ :=
  ∑ j, ((priorNaturalScale M (assimilate M s a) elapsed j *
      valueEligibility M direction a j i) /
      max ((prior M (assimilate M s a) elapsed).precision j) M.precisionFloor) *
      address M a.byte j

/-- The exact encoder-value eligibility at the next prepared readout.  A
write is transported by the GDN prior and read by the same causal query; no
backward graph or unbounded trace is introduced. -/
theorem value_future_read_eligibility_exact (M : Model k d)
    (direction : Byte → Fin d → ℝ) (θ elapsed : ℝ) (s : State k d) (a : Arrival)
    (hpositive : 0 < a.observationPrecision) (i : Fin d) :
    feature (valuePerturb M direction θ)
        (prior (valuePerturb M direction θ)
          (assimilate (valuePerturb M direction θ) s a) elapsed) i =
      feature M (prior M (assimilate M s a) elapsed) i +
        θ * valueFutureReadEligibility M direction s a elapsed i := by
  rw [feature_valuePerturb]
  have hassimilate_precision : ∀ j,
      (assimilate (valuePerturb M direction θ) s a).precision j =
        (assimilate M s a).precision j := by
    intro j
    exact assimilate_valuePerturb_precision M direction θ s a j
  have hprecision : ∀ j,
      (prior (valuePerturb M direction θ)
        (assimilate (valuePerturb M direction θ) s a) elapsed).precision j =
        (prior M (assimilate M s a) elapsed).precision j := by
    intro j
    by_cases hzero : elapsed = 0
    · subst elapsed
      simp [prior, hassimilate_precision j]
    · simp only [prior, hzero, ↓reduceIte]
      change
        1 / (1 / max ((assimilate (valuePerturb M direction θ) s a).precision j)
          M.precisionFloor + elapsed * M.processVariance) =
          1 / (1 / max ((assimilate M s a).precision j) M.precisionFloor +
            elapsed * M.processVariance)
      rw [hassimilate_precision j]
  have hlast_assimilate :
      (assimilate (valuePerturb M direction θ) s a).lastByte =
        (assimilate M s a).lastByte := by
    rfl
  have hlast :
      (prior (valuePerturb M direction θ)
        (assimilate (valuePerturb M direction θ) s a) elapsed).lastByte =
        (prior M (assimilate M s a) elapsed).lastByte := by
    by_cases hzero : elapsed = 0
    · subst elapsed
      simp [prior, hlast_assimilate]
    · simp [prior, hzero, hlast_assimilate]
  have hread := feature_difference_of_shared_query M
    (prior M (assimilate M s a) elapsed)
    (prior (valuePerturb M direction θ)
      (assimilate (valuePerturb M direction θ) s a) elapsed)
    hprecision hlast i
  have hquery : query M (prior M (assimilate M s a) elapsed) = address M a.byte := by
    have hlast_base :
        (prior M (assimilate M s a) elapsed).lastByte = some a.byte := by
      by_cases hzero : elapsed = 0
      · subst elapsed
        simpa [prior] using assimilate_lastByte_of_positive M s a hpositive
      · simpa [prior, hzero] using assimilate_lastByte_of_positive M s a hpositive
    simp [query, hlast_base]
  simp_rw [value_sensitivity_through_prior M s direction θ elapsed a] at hread
  rw [hquery] at hread
  have hsum :
      (∑ j, ((priorNaturalScale M (assimilate M s a) elapsed j *
          (θ * valueEligibility M direction a j i)) /
          max ((prior M (assimilate M s a) elapsed).precision j) M.precisionFloor) *
          address M a.byte j) =
        θ * valueFutureReadEligibility M direction s a elapsed i := by
    rw [valueFutureReadEligibility, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [hsum] at hread
  linarith

/-- Assimilation writes only the material memory fields and causal observation
cache; it does not create a second copy of the language readout parameters. -/
theorem assimilate_head (M : Model k d) (s : State k d) (a : Arrival) :
    (assimilate M s a).head = s.head := rfl

theorem assimilate_bias (M : Model k d) (s : State k d) (a : Arrival) :
    (assimilate M s a).bias = s.bias := rfl

/-- The time prior transports memory but retains the same head and bias. -/
theorem prior_head (M : Model k d) (s : State k d) (elapsed : ℝ) :
    (prior M s elapsed).head = s.head := by
  by_cases hzero : elapsed = 0 <;> simp [prior, hzero]

theorem prior_bias (M : Model k d) (s : State k d) (elapsed : ℝ) :
    (prior M s elapsed).bias = s.bias := by
  by_cases hzero : elapsed = 0 <;> simp [prior, hzero]

/-- Projection of the encoder's forward eligibility into one future byte
logit through the pre-existing affine language head. -/
noncomputable def valueFutureLogitEligibility (M : Model k d)
    (direction : Byte → Fin d → ℝ) (s : State k d) (a : Arrival)
    (elapsed : ℝ) (b : Byte) : ℝ :=
  ∑ i, (prior M (assimilate M s a) elapsed).head b i *
    valueFutureReadEligibility M direction s a elapsed i

/-- The next byte-logit displacement is exactly the forward eligibility read
through the unchanged affine head.  This is still an identity, before an
external outcome is used as a third factor. -/
theorem value_future_logit_eligibility_exact (M : Model k d)
    (direction : Byte → Fin d → ℝ) (θ elapsed : ℝ) (s : State k d) (a : Arrival)
    (hpositive : 0 < a.observationPrecision) (b : Byte) :
    logits
        (prior (valuePerturb M direction θ)
          (assimilate (valuePerturb M direction θ) s a) elapsed)
        (feature (valuePerturb M direction θ)
          (prior (valuePerturb M direction θ)
            (assimilate (valuePerturb M direction θ) s a) elapsed)) b =
      logits (prior M (assimilate M s a) elapsed)
        (feature M (prior M (assimilate M s a) elapsed)) b +
        θ * valueFutureLogitEligibility M direction s a elapsed b := by
  have hfeature := value_future_read_eligibility_exact M direction θ elapsed s a
    hpositive
  simp only [logits]
  simp_rw [prior_head, prior_bias, assimilate_head, assimilate_bias]
  simp_rw [hfeature]
  rw [valueFutureLogitEligibility]
  simp_rw [prior_head, assimilate_head]
  have hsum :
      (∑ i, s.head b i *
        (feature M (prior M (assimilate M s a) elapsed) i +
          θ * valueFutureReadEligibility M direction s a elapsed i)) =
        (∑ i, s.head b i * feature M (prior M (assimilate M s a) elapsed) i) +
          θ * ∑ i, s.head b i * valueFutureReadEligibility M direction s a elapsed i := by
    calc
      _ = ∑ i, (s.head b i * feature M (prior M (assimilate M s a) elapsed) i +
          θ * (s.head b i * valueFutureReadEligibility M direction s a elapsed i)) := by
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [hsum]
  ring

/-- G14 for a whole finite logit direction.  Coordinate G14 is the special
case in which `direction` has one nonzero component. -/
theorem byte_log_softmax_direction_deriv (z direction : Byte → ℝ) (target : Byte) :
    HasDerivAt
      (fun θ => Real.log (CGB.Medium.softmax (fun b => z b + θ * direction b) target))
      (direction target - ∑ b, CGB.Medium.softmax z b * direction b) 0 := by
  have hden : ∀ θ : ℝ, 0 < ∑ b, Real.exp (z b + θ * direction b) := fun θ =>
    Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨target, Finset.mem_univ _⟩
  have hform :
      (fun θ => Real.log (CGB.Medium.softmax
        (fun b => z b + θ * direction b) target)) =
        fun θ => z target + θ * direction target -
          Real.log (∑ b, Real.exp (z b + θ * direction b)) := by
    funext θ
    simp only [CGB.Medium.softmax]
    rw [Real.log_div (Real.exp_pos _).ne' (hden θ).ne', Real.log_exp]
  rw [hform]
  have hlin : ∀ b, HasDerivAt (fun θ : ℝ => z b + θ * direction b)
      (direction b) 0 := fun b => by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (direction b)).const_add (z b)
  have hsum : HasDerivAt (fun θ => ∑ b, Real.exp (z b + θ * direction b))
      (∑ b, Real.exp (z b) * direction b) 0 := by
    have h := HasDerivAt.fun_sum (u := Finset.univ) (fun b _ => (hlin b).exp)
    simpa using h
  have h := (hlin target).sub (hsum.log (hden 0).ne')
  simp only [zero_mul, add_zero] at h
  have hsoft :
      (∑ b, CGB.Medium.softmax z b * direction b) =
        (∑ b, Real.exp (z b) * direction b) / ∑ b, Real.exp (z b) := by
    simp only [CGB.Medium.softmax]
    calc
      (∑ b, Real.exp (z b) / (∑ c, Real.exp (z c)) * direction b) =
          ∑ b, (Real.exp (z b) * direction b) / (∑ c, Real.exp (z c)) := by
        apply Finset.sum_congr rfl
        intro b _
        ring
      _ = _ := by
        simp only [div_eq_mul_inv]
        rw [Finset.sum_mul]
  rw [hsoft]
  exact h

/-- The third-factor-ready score of a value direction: the observed next-byte
logit displacement minus its prediction-weighted free-phase displacement.
Nothing in this definition is an added trace; its two terms are read from the
same next GDN state and existing affine head. -/
noncomputable def valueFutureScore (M : Model k d)
    (direction : Byte → Fin d → ℝ) (s : State k d) (a : Arrival)
    (elapsed : ℝ) (target : Byte) : ℝ :=
  let next := prior M (assimilate M s a) elapsed
  let shift := valueFutureLogitEligibility M direction s a elapsed
  shift target - ∑ b, allocation M next b * shift b

/-- G14 applied to the exact next-byte GDN displacement: the derivative of
next-byte log likelihood with respect to an encoder-value direction is the
forward eligibility weighted by the observed-minus-predicted language error.
This derives the local two factors; a delayed external outcome may only enter
later as the already-required third factor. -/
theorem value_future_log_likelihood_deriv (M : Model k d)
    (direction : Byte → Fin d → ℝ) (elapsed : ℝ) (s : State k d) (a : Arrival)
    (hpositive : 0 < a.observationPrecision) (target : Byte) :
    HasDerivAt
      (fun θ => Real.log (allocation (valuePerturb M direction θ)
        (prior (valuePerturb M direction θ)
          (assimilate (valuePerturb M direction θ) s a) elapsed) target))
      (valueFutureScore M direction s a elapsed target) 0 := by
  let next := prior M (assimilate M s a) elapsed
  let shift := valueFutureLogitEligibility M direction s a elapsed
  have h := byte_log_softmax_direction_deriv
    (logits next (feature M next)) shift target
  change HasDerivAt
    (fun θ => Real.log (CGB.Medium.softmax
      (logits
        (prior (valuePerturb M direction θ)
          (assimilate (valuePerturb M direction θ) s a) elapsed)
        (feature (valuePerturb M direction θ)
          (prior (valuePerturb M direction θ)
            (assimilate (valuePerturb M direction θ) s a) elapsed))) target))
    (valueFutureScore M direction s a elapsed target) 0
  rw [show valueFutureScore M direction s a elapsed target =
      shift target - ∑ b, CGB.Medium.softmax (logits next (feature M next)) b * shift b by
    rfl]
  convert h using 1
  funext θ
  congr 2
  funext b
  exact value_future_logit_eligibility_exact M direction θ elapsed s a hpositive b

/-- The predicted free-phase language error at one feature coordinate.  It is
computed from the same next-state head and readout that produced the score. -/
noncomputable def nextFeatureLanguageError (M : Model k d)
    (s : State k d) (a : Arrival) (elapsed : ℝ) (target : Byte) (i : Fin d) : ℝ :=
  let next := prior M (assimilate M s a) elapsed
  next.head target i - ∑ b, allocation M next b * next.head b i

/-- Exact factorization of the encoder-value score.  The forward material
eligibility and the observed-minus-predicted language error are sufficient
statistics for this one-step value path; their product is not a hand-added
surrogate for BPTT. -/
theorem value_future_score_factor (M : Model k d)
    (direction : Byte → Fin d → ℝ) (s : State k d) (a : Arrival)
    (elapsed : ℝ) (target : Byte) :
    valueFutureScore M direction s a elapsed target =
      ∑ i, valueFutureReadEligibility M direction s a elapsed i *
        nextFeatureLanguageError M s a elapsed target i := by
  let next := prior M (assimilate M s a) elapsed
  let e := valueFutureReadEligibility M direction s a elapsed
  have htranspose :
      (∑ b, allocation M next b * ∑ i, next.head b i * e i) =
        ∑ i, e i * ∑ b, allocation M next b * next.head b i := by
    calc
      (∑ b, allocation M next b * ∑ i, next.head b i * e i) =
          ∑ b, ∑ i, allocation M next b * (next.head b i * e i) := by
        apply Finset.sum_congr rfl
        intro b _
        rw [Finset.mul_sum]
      _ = ∑ i, ∑ b, allocation M next b * (next.head b i * e i) := by
        rw [Finset.sum_comm]
      _ = ∑ i, e i * ∑ b, allocation M next b * next.head b i := by
        apply Finset.sum_congr rfl
        intro i _
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro b _
        ring
  change
    (∑ i, next.head target i * e i) -
      ∑ b, allocation M next b * ∑ i, next.head b i * e i =
      ∑ i, e i * (next.head target i -
        ∑ b, allocation M next b * next.head b i)
  have hright :
      (∑ i, e i * (next.head target i -
        ∑ b, allocation M next b * next.head b i)) =
        (∑ i, e i * next.head target i) -
          ∑ i, e i * ∑ b, allocation M next b * next.head b i := by
    calc
      _ = ∑ i, (e i * next.head target i -
          e i * ∑ b, allocation M next b * next.head b i) := by
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = _ := by rw [Finset.sum_sub_distrib]
  have htarget :
      (∑ i, next.head target i * e i) = ∑ i, e i * next.head target i := by
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hright, htranspose, htarget]

theorem address_nonneg (M : Model k d) (b : Byte) (j : Fin k) : 0 ≤ address M b j := by
  unfold address CGB.Medium.softmax
  exact div_nonneg (Real.exp_pos _).le (sum_nonneg fun _ _ => (Real.exp_pos _).le)

theorem incoming_nonneg (M : Model k d) (a : Arrival) (j : Fin k) :
    0 ≤ incoming M a j :=
  mul_nonneg (address_nonneg M a.byte j)
    (mul_nonneg a.observation_nonneg (M.likelihood_nonneg a.byte))

theorem prior_positive (M : Model k d) (s : State k d) (hs : Positive s)
    (elapsed : ℝ) (ht : 0 ≤ elapsed) : Positive (prior M s elapsed) := by
  classical
  intro j
  by_cases h : elapsed = 0
  · simpa [prior, h] using hs j
  · simp only [prior, h, ↓reduceIte]
    have hp : 0 < max (s.precision j) M.precisionFloor :=
      lt_of_lt_of_le M.floor_pos (le_max_right _ _)
    exact div_pos zero_lt_one (add_pos_of_pos_of_nonneg (div_pos zero_lt_one hp)
      (mul_nonneg ht M.process_nonneg))

theorem assimilate_positive (M : Model k d) (s : State k d) (hs : Positive s)
    (a : Arrival) : Positive (assimilate M s a) := by
  intro j
  exact add_pos_of_pos_of_nonneg (hs j) (incoming_nonneg M a j)

theorem step_positive (M : Model k d) (s : State k d) (hs : Positive s) (a : Arrival) :
    Positive (step M s a) :=
  assimilate_positive M _ (prior_positive M s hs a.elapsed a.elapsed_nonneg) a

/-- Zero-precision input retains an existing last observation, matching
`assimilate`'s live-observation mask rather than silently changing query. -/
theorem zero_precision_keeps_last_byte (M : Model k d) (s : State k d) (a : Arrival)
    (b : Byte) (hs : s.lastByte = some b) (ha : a.observationPrecision = 0) :
    (assimilate M s a).lastByte = some b := by
  simp [assimilate, hs, ha]

/-- The first input still installs the cache, even at zero precision. -/
theorem first_observation_sets_last_byte (M : Model k d) (s : State k d) (a : Arrival)
    (hs : s.lastByte = none) : (assimilate M s a).lastByte = some a.byte := by
  simp [assimilate, hs]

/-- Any live arrival writes its byte into the existing causal query field.
This is a state transition, not a separate eligibility or action-log array. -/
theorem positive_observation_sets_last_byte (M : Model k d) (s : State k d)
    (a : Arrival) (ha : 0 < a.observationPrecision) :
    (assimilate M s a).lastByte = some a.byte := by
  cases hs : s.lastByte <;> simp [assimilate, hs, ha]

/-- The complete GDN step preserves that live causal write through its local
head update.  A self-produced language byte can therefore use the same body
transition as an external byte; source provenance remains a boundary condition
for later outcome modulation. -/
theorem step_positive_observation_sets_last_byte (M : Model k d) (s : State k d)
    (a : Arrival) (ha : 0 < a.observationPrecision) :
    (step M s a).lastByte = some a.byte := by
  change (assimilate M (prior M s a.elapsed) a).lastByte = some a.byte
  exact positive_observation_sets_last_byte M _ a ha

/-- Distinct live language emissions produce distinct post-step bodies through
an existing causal field.  This establishes the positive premise needed by
`delayed_action_credit_requires_state_separation` for a byte-level speech act;
it does not claim that the distinction survives arbitrary later interference. -/
theorem distinct_live_bytes_separate_steps (M : Model k d) (s : State k d)
    (a₀ a₁ : Arrival) (h₀ : 0 < a₀.observationPrecision)
    (h₁ : 0 < a₁.observationPrecision) (different_byte : a₀.byte ≠ a₁.byte) :
    step M s a₀ ≠ step M s a₁ := by
  intro hstep
  have hlast := congrArg (fun t : State k d => t.lastByte) hstep
  rw [step_positive_observation_sets_last_byte M s a₀ h₀,
    step_positive_observation_sets_last_byte M s a₁ h₁] at hlast
  exact different_byte (Option.some.inj hlast)

/-- Source labels do not invent a speech fee or a different learning rule.
The world must still distinguish external provenance from self-produced text
when deciding whether an event is new evidence. -/
theorem step_source_irrelevant (M : Model k d) (s : State k d) (a : Arrival) (src : Source) :
    step M s { a with source := src } = step M s a := rfl

/-- Prediction cannot inspect the not-yet-arrived byte or its source. -/
theorem preparation_uses_only_elapsed (M : Model k d) (s : State k d) (a a' : Arrival)
    (h : a.elapsed = a'.elapsed) : prepare M s a.elapsed = prepare M s a'.elapsed := by
  rw [h]

/-- Local head learning does not modify the memory or its uncertainty. -/
theorem head_update_preserves_memory (η : ℝ) (s : State k d) (f : Fin d → ℝ) (b : Byte) :
    (headUpdate η s f b).natural = s.natural ∧
      (headUpdate η s f b).precision = s.precision := ⟨rfl, rfl⟩

/-- The actual affine parameter step induces a score step of size
`eta * (1 + squared feature norm)`, including the learned bias. -/
theorem head_update_logits (η : ℝ) (s : State k d) (f : Fin d → ℝ) (target b : Byte) :
    logits (headUpdate η s f target) f b = logits s f b +
      (η * (1 + ∑ i, f i * f i)) * innovation s f target b := by
  have hsum : (∑ i, (s.head b i + η * innovation s f target b * f i) * f i) =
      (∑ i, s.head b i * f i) + (η * innovation s f target b) * ∑ i, f i * f i := by
    rw [mul_sum, ← sum_add_distrib]
    apply sum_congr rfl
    intro i _
    ring
  simp only [logits, headUpdate]
  rw [hsum]
  ring

/-- Strict improvement on the saved feature, not a future-language claim. -/
theorem head_update_improves_saved_feature (η : ℝ) (hη : 0 < η)
    (s : State k d) (f : Fin d → ℝ) (target : Byte) :
    CGB.Medium.softmax (logits s f) target <
      CGB.Medium.softmax (logits (headUpdate η s f target) f) target := by
  have heff : 0 < η * (1 + ∑ i, f i * f i) :=
    mul_pos hη (by linarith [sum_nonneg (fun i (_ : i ∈ univ) => mul_self_nonneg (f i))])
  have hother : ∃ b : Byte, b ≠ target := by
    by_cases h : target = 0
    · exact ⟨1, by rw [h]; decide⟩
    · exact ⟨0, Ne.symm h⟩
  have h := CGB.SelfProduction.two_phase_finite_score_step_improves
    (logits s f) target _ heff hother
  have hlogits : logits (headUpdate η s f target) f = fun b => logits s f b +
      (η * (1 + ∑ i, f i * f i)) *
        ((if b = target then 1 else 0) - CGB.Medium.softmax (logits s f) b) := by
    funext b
    exact head_update_logits η s f target b
  rw [hlogits]
  exact h

/-- One explicit GDN instantiates the general carrier on real states.
The proof-side history here is the current causal summary; the dynamics and
readout are explicit, not a lookup or an additional memory system. -/
noncomputable def carrier {Act : Type*} (M : Model k d) (s₀ : State k d) (h₀ : Positive s₀)
    (policy : State k d → Act) (response : State k d → Act → Arrival) :
    RenewableLanguageGenerator (State k d) (State k d) Arrival Act (Byte → ℝ) where
  encode := id
  required := allocation M
  advanceHistory := step M
  fallback := allocation M s₀
  seed := s₀
  viable := Positive
  seed_viable := h₀
  readable := by intro s s' h; exact congrArg (allocation M) h
  transition_compatible := by intro s s' a h; exact congrArg (fun t => step M t a) h
  policy := policy
  world_response := response
  world_closed := by intro s hs; exact step_positive M s hs _

theorem carrier_next {Act : Type*} (M : Model k d) (s₀ : State k d) (h₀ : Positive s₀)
    (policy : State k d → Act) (response : State k d → Act → Arrival)
    (s : State k d) (a : Arrival) :
    renewableLanguageNext (carrier M s₀ h₀ policy response) s a = step M s a := by
  simpa [carrier] using
    (renewable_language_carrier_correct (carrier M s₀ h₀ policy response)).2 s a

/-- Actual input histories fold through this same real body, with no reset. -/
theorem carrier_fold {Act : Type*} (M : Model k d) (s₀ : State k d) (h₀ : Positive s₀)
    (policy : State k d → Act) (response : State k d → Act → Arrival) (inputs : List Arrival) :
    inputs.foldl (renewableLanguageNext (carrier M s₀ h₀ policy response)) s₀ =
      inputs.foldl (step M) s₀ := by
  have hnext : renewableLanguageNext (carrier M s₀ h₀ policy response) = step M := by
    funext s a
    exact carrier_next M s₀ h₀ policy response s a
  rw [hnext]

/-- Independent world state and elapsed-time events coexist with real GDN
state. This numeric invariant is not biological or material viability. -/
noncomputable def openWorld {Act World : Type*} (M : Model k d) (s₀ : State k d)
    (h₀ : Positive s₀) (policy : State k d → Act)
    (response : State k d → Act → Arrival) (w₀ : World)
    (boundary : World → State k d → Act → Arrival)
    (evolve : World → State k d → Act → Arrival → World) :
    RenewableOpenWorld (carrier M s₀ h₀ policy response) World where
  initial := w₀
  boundary := boundary
  evolve := evolve
  closed := by intro s w hs; exact step_positive M s hs _

/-- An actual continuous GDN, not a finite surrogate, has one infinite
open numerical trajectory. Future benefit, active maintenance and the full
life definition are not conclusions of positive precision. -/
theorem exists_open_real_gdn_trajectory {Act World : Type*}
    (M : Model k d) (s₀ : State k d) (h₀ : Positive s₀)
    (policy : State k d → Act) (response : State k d → Act → Arrival)
    (w₀ : World) (boundary : World → State k d → Act → Arrival)
    (evolve : World → State k d → Act → Arrival → World) :
    ∃ x : ℕ → State k d × World,
      x 0 = (s₀, w₀) ∧
      (∀ n, x (n + 1) =
        (step M (x n).1 (boundary (x n).2 (x n).1 (policy (x n).1)),
         evolve (x n).2 (x n).1 (policy (x n).1)
           (boundary (x n).2 (x n).1 (policy (x n).1)))) ∧
      (∀ n, Positive (x n).1) := by
  let G := carrier M s₀ h₀ policy response
  let W := openWorld M s₀ h₀ policy response w₀ boundary evolve
  let C := renewableOpenContract G W
  refine ⟨trajectory C, rfl, ?_, ?_⟩
  · intro n
    change (renewableLanguageNext G (trajectory C n).1
      (boundary (trajectory C n).2 (trajectory C n).1 (policy (trajectory C n).1)), _) = _
    rw [carrier_next]
    rfl
  · intro n
    obtain ⟨_, _, hpos⟩ := trajectory_viable C n
    exact hpos

/-- Assimilation does not change the head. Thus its local update is exactly
on the prediction-time feature, despite the changed memory underneath it. -/
theorem finish_saved_logits (M : Model k d) (p : Prepared k d) (a : Arrival) :
    logits (finish M p a) p.feature =
      logits (headUpdate M.headStep p.state p.feature a.byte) p.feature := by
  funext b
  simp only [finish, head_update_logits]
  rfl

noncomputable def nll (z : Byte → ℝ) (b : Byte) : ℝ :=
  -Real.log (CGB.Medium.softmax z b)

theorem byte_softmax_pos (z : Byte → ℝ) (b : Byte) : 0 < CGB.Medium.softmax z b := by
  unfold CGB.Medium.softmax
  exact div_pos (Real.exp_pos _)
    (sum_pos (fun _ _ => Real.exp_pos _) univ_nonempty)

/-- The saved-feature current-byte loss strictly decreases at any positive
head step. No assumption about the next byte is hidden in this result. -/
theorem finish_saved_nll_improves (M : Model k d) (hη : 0 < M.headStep)
    (p : Prepared k d) (a : Arrival) :
    nll (logits (finish M p a) p.feature) a.byte < nll (logits p.state p.feature) a.byte := by
  rw [finish_saved_logits]
  have h := head_update_improves_saved_feature M.headStep hη p.state p.feature a.byte
  have hl := Real.strictMonoOn_log (byte_softmax_pos _ _)
    (byte_softmax_pos _ _) h
  unfold nll
  linarith

/-- A distributional, not in-sample, sufficient condition. A known target
loss gain beats all possible other-byte loss increases when the *same*
future law assigns enough mass to the target. -/
theorem future_ce_improves_of_target_mass (law old new : Byte → ℝ) (target : Byte)
    (gain bound : ℝ) (hlaw0 : ∀ b, 0 ≤ law b) (hlaw : ∑ b, law b = 1)
    (htarget : nll new target - nll old target ≤ -gain)
    (hother : ∀ b, b ≠ target → nll new b - nll old b ≤ bound)
    (hdominant : bound < law target * (bound + gain)) :
    CGB.PredictiveHarvest.crossEntropy law (CGB.Medium.softmax new) <
      CGB.PredictiveHarvest.crossEntropy law (CGB.Medium.softmax old) := by
  have hpoint (b : Byte) : law b * (nll new b - nll old b) ≤
      law b * (bound - (if b = target then bound + gain else 0)) := by
    apply mul_le_mul_of_nonneg_left _ (hlaw0 b)
    by_cases hb : b = target
    · subst b
      simpa only [ite_true, sub_add_cancel_left, neg_le_neg_iff] using htarget
    · simpa only [hb, ite_false, sub_zero] using hother b hb
  have hsum := sum_le_sum (fun b (_ : b ∈ univ) => hpoint b)
  have hrhs : (∑ b, law b * (bound - (if b = target then bound + gain else 0))) =
      bound - law target * (bound + gain) := by
    simp only [mul_sub, sum_sub_distrib, ← sum_mul]
    simp [mul_ite, hlaw]
  rw [hrhs] at hsum
  have hlt : (∑ b, law b * (nll new b - nll old b)) < 0 := by linarith
  unfold CGB.PredictiveHarvest.crossEntropy
  simp only [nll, mul_sub, mul_neg, sum_sub_distrib, sum_neg_distrib] at hlt
  linarith

/-- A concrete head update can improve a declared future law even after the
memory/query changed: the saved-feature gain must exceed the readout drift,
and the law must be sufficiently persistent. These are numeric/world
conditions, not an assumption that future CE already improved. -/
theorem finish_future_ce_improves (M : Model k d) (p : Prepared k d) (a : Arrival)
    (law : Byte → ℝ) (δ κ : ℝ)
    (hlaw0 : ∀ b, 0 ≤ law b) (hlaw : ∑ b, law b = 1)
    (hdrift : ∀ b, |logits (finish M p a) (feature M (finish M p a)) b -
      logits (finish M p a) p.feature b| ≤ δ)
    (htotal : ∀ b, |logits (finish M p a) (feature M (finish M p a)) b -
      logits p.state p.feature b| ≤ κ)
    (hdominant : 2 * κ < law a.byte * (2 * κ +
      (nll (logits p.state p.feature) a.byte -
       nll (logits (finish M p a) p.feature) a.byte - 2 * δ))) :
    CGB.PredictiveHarvest.crossEntropy law (allocation M (finish M p a)) <
      CGB.PredictiveHarvest.crossEntropy law (CGB.Medium.softmax (logits p.state p.feature)) := by
  apply future_ce_improves_of_target_mass law (logits p.state p.feature)
    (logits (finish M p a) (feature M (finish M p a))) a.byte
    (nll (logits p.state p.feature) a.byte -
      nll (logits (finish M p a) p.feature) a.byte - 2 * δ) (2 * κ) hlaw0 hlaw
  · have h := CGB.SelfProduction.softmax_nll_logit_error_bound
      (logits (finish M p a) p.feature)
      (logits (finish M p a) (feature M (finish M p a))) a.byte δ hdrift
    change |nll _ a.byte - nll _ a.byte| ≤ _ at h
    have hu := (abs_le.mp h).2
    linarith
  · intro b _
    have h := CGB.SelfProduction.softmax_nll_logit_error_bound
      (logits p.state p.feature)
      (logits (finish M p a) (feature M (finish M p a))) b κ htotal
    exact (abs_le.mp h).2
  · exact hdominant

/-- The real family already embeds a real coordinate in an existing head
bias. No finite body-state enumeration is being smuggled into this carrier. -/
theorem real_bias_injective : Function.Injective
    (fun r : ℝ => initial (k := k) (d := d) 1 (fun _ _ => 0) (fun _ => r)) := by
  intro a b h
  exact congrArg (fun s : State k d => s.bias (0 : Byte)) h

theorem prior_logits (M : Model k d) (s : State k d) (elapsed : ℝ) (f : Fin d → ℝ) :
    logits (prior M s elapsed) f = logits s f := by
  classical
  funext b
  by_cases h : elapsed = 0 <;> simp [prior, h, logits]

/-- A positive real head step changes the body on every arriving byte.
This alone is not evidence of useful learning; unpredictable input also
moves parameters. Finite-precision stalling is a separate numerical issue. -/
theorem step_ne_of_positive_head_step (M : Model k d) (hη : 0 < M.headStep)
    (s : State k d) (a : Arrival) : step M s a ≠ s := by
  intro heq
  have h := finish_saved_nll_improves M hη (prepare M s a.elapsed) a
  change nll (logits (step M s a) (prepare M s a.elapsed).feature) a.byte <
    nll (logits (prior M s a.elapsed) (prepare M s a.elapsed).feature) a.byte at h
  rw [heq, prior_logits] at h
  exact (lt_irrefl _ h)

/-- The supplied policy/world are explicit generator data, not learned
conclusions. The process introduces no second body or auxiliary reserve. -/
structure Process (k d : ℕ) (Act World : Type*) where
  model : Model k d
  initial : State k d
  seed_positive : Positive initial
  policy : State k d → Act
  worldInitial : World
  boundary : World → State k d → Act → Arrival
  evolve : World → State k d → Act → Arrival → World

namespace Process

variable {Act World : Type*}

noncomputable def generator (P : Process k d Act World) :=
  carrier P.model P.initial P.seed_positive P.policy (P.boundary P.worldInitial)

noncomputable def world (P : Process k d Act World) :=
  openWorld P.model P.initial P.seed_positive P.policy
    (P.boundary P.worldInitial) P.worldInitial P.boundary P.evolve

noncomputable def contract (P : Process k d Act World) :=
  renewableOpenContract P.generator P.world

noncomputable def run (P : Process k d Act World) : ℕ → State k d × World :=
  trajectory P.contract

noncomputable def arrival (P : Process k d Act World) (n : ℕ) : Arrival :=
  P.boundary (P.run n).2 (P.run n).1 (P.policy (P.run n).1)

noncomputable def snapshot (P : Process k d Act World) (n : ℕ) : Prepared k d :=
  prepare P.model (P.run n).1 (P.arrival n).elapsed

noncomputable def scores (P : Process k d Act World) (n : ℕ) : Byte → ℝ :=
  logits (P.snapshot n).state (P.snapshot n).feature

/-- Legacy generic open-system signal.  It is retained for the reusable
opportunity theorem below, but is not a language-life intake definition; use
`externallyUseful` for the language route. -/
noncomputable def signal (P : Process k d Act World)
    (intake : World → State k d → Act → Arrival → ℕ) (n : ℕ) : ℕ :=
  intake (P.run n).2 (P.run n).1 (P.policy (P.run n).1) (P.arrival n)

theorem run_initial (P : Process k d Act World) : P.run 0 = (P.initial, P.worldInitial) := rfl

theorem run_step (P : Process k d Act World) (n : ℕ) :
    P.run (n + 1) = (step P.model (P.run n).1 (P.arrival n),
      P.evolve (P.run n).2 (P.run n).1 (P.policy (P.run n).1) (P.arrival n)) := by
  apply Prod.ext
  · exact carrier_next P.model P.initial P.seed_positive P.policy
      (P.boundary P.worldInitial) (P.run n).1 (P.arrival n)
  · rfl

theorem run_positive (P : Process k d Act World) (n : ℕ) : Positive (P.run n).1 := by
  obtain ⟨_, _, h⟩ := trajectory_viable P.contract n
  exact h

theorem run_body_step (P : Process k d Act World) (n : ℕ) :
    (P.run (n + 1)).1 = finish P.model (P.snapshot n) (P.arrival n) := by
  rw [P.run_step]
  rfl

/-- Two candidate language actions that emit different live self-produced
bytes write distinct next bodies through the ordinary GDN boundary transition.
The byte is the action's existing bodily consequence, not a separately stored
action record. -/
theorem self_speech_action_contrast_writes_body (P : Process k d Act World)
    (n : ℕ) (a₀ a₁ : Act)
    (_hself₀ : (P.boundary (P.run n).2 (P.run n).1 a₀).source = .self)
    (_hself₁ : (P.boundary (P.run n).2 (P.run n).1 a₁).source = .self)
    (h₀ : 0 < (P.boundary (P.run n).2 (P.run n).1 a₀).observationPrecision)
    (h₁ : 0 < (P.boundary (P.run n).2 (P.run n).1 a₁).observationPrecision)
    (hdifferent : (P.boundary (P.run n).2 (P.run n).1 a₀).byte ≠
      (P.boundary (P.run n).2 (P.run n).1 a₁).byte) :
    step P.model (P.run n).1 (P.boundary (P.run n).2 (P.run n).1 a₀) ≠
      step P.model (P.run n).1 (P.boundary (P.run n).2 (P.run n).1 a₁) := by
  exact distinct_live_bytes_separate_steps P.model (P.run n).1
    (P.boundary (P.run n).2 (P.run n).1 a₀)
    (P.boundary (P.run n).2 (P.run n).1 a₁) h₀ h₁ hdifferent

/-- Repetition refers to the world's actual next byte, not a byte chosen for
rescoring. The readout-drift budget includes the next physical-time prior. -/
theorem next_byte_improves (P : Process k d Act World) (n : ℕ) (δ : ℝ)
    (hrepeat : (P.arrival (n + 1)).byte = (P.arrival n).byte)
    (hdrift : ∀ b, |P.scores (n + 1) b -
      logits (finish P.model (P.snapshot n) (P.arrival n)) (P.snapshot n).feature b| ≤ δ)
    (hgap : 2 * δ < nll (P.scores n) (P.arrival n).byte -
      nll (logits (finish P.model (P.snapshot n) (P.arrival n))
        (P.snapshot n).feature) (P.arrival n).byte) :
    nll (P.scores (n + 1)) (P.arrival (n + 1)).byte <
      nll (P.scores n) (P.arrival (n + 1)).byte := by
  rw [hrepeat]
  have h := CGB.SelfProduction.softmax_nll_logit_error_bound
    (logits (finish P.model (P.snapshot n) (P.arrival n)) (P.snapshot n).feature)
    (P.scores (n + 1)) (P.arrival n).byte δ hdrift
  change |nll _ _ - nll _ _| ≤ _ at h
  have hu := (abs_le.mp h).2
  linarith

/-- A usable language event is not a paid token or an extra reserve.  It is
an externally sourced arrival whose actual, causally later language byte is
scored strictly better after the same GDN update.  The source tag is a world
provenance contract: self-produced bytes may be read and integrated, but do
not qualify as new external evidence merely by being copied. -/
noncomputable def externallyUseful (P : Process k d Act World) (n : ℕ) : Prop :=
  (P.arrival n).source = .external ∧
  ∃ δ : ℝ,
    (P.arrival (n + 1)).byte = (P.arrival n).byte ∧
    (∀ b, |P.scores (n + 1) b -
      logits (finish P.model (P.snapshot n) (P.arrival n)) (P.snapshot n).feature b| ≤ δ) ∧
    2 * δ < nll (P.scores n) (P.arrival n).byte -
      nll (logits (finish P.model (P.snapshot n) (P.arrival n))
        (P.snapshot n).feature) (P.arrival n).byte

/-- The useful-event certificate invokes the earlier causal result; it does
not inspect the next byte while preparing the current prediction. -/
theorem externallyUseful_improves (P : Process k d Act World) (n : ℕ)
    (h : P.externallyUseful n) :
    nll (P.scores (n + 1)) (P.arrival (n + 1)).byte <
      nll (P.scores n) (P.arrival (n + 1)).byte := by
  obtain ⟨_, δ, hrepeat, hdrift, hgap⟩ := h
  exact P.next_byte_improves n δ hrepeat hdrift hgap

/-- A copied self byte never becomes external evidence just by passing through
the same learning rule.  It can still affect the body through `step`. -/
theorem self_not_externallyUseful (P : Process k d Act World) (n : ℕ)
    (hself : (P.arrival n).source = .self) : ¬ P.externallyUseful n := by
  rintro ⟨hext, _⟩
  rw [hself] at hext
  cases hext

/-- Information carried by an external language encounter is the finite mutual
information between the arriving byte and a later external byte, conditional
on the *already fixed* GDN state and action.  The fixed body is deliberately
not placed in the finite sum: this quantity asks what the boundary can still
teach that particular body in that particular situation. -/
noncomputable def externalInformation (joint : Byte → Byte → ℝ) : ℝ :=
  CGB.Medium.condMutualInfo (fun (_ : Unit) now later => joint now later)

/-- A registered language boundary law.  This is a semantic contract on the
same world port, not a field of the organism and not an extra reward, food, or
reserve account.  `joint s a` must be the actual conditional law of one
externally sourced arriving byte and one later external byte after state `s`
and action `a`; normalization makes that interpretation explicit. -/
structure ExternalLanguageLaw (P : Process k d Act World) where
  joint : State k d → Act → Byte → Byte → ℝ
  nonneg : ∀ s a now later, 0 ≤ joint s a now later
  normalized : ∀ s a, (∑ now, ∑ later, joint s a now later) = 1

/-- The information opportunity available at one state/action, derived from
the registered external language law.  It shares the GDN's state but does not
add a scalar balance to it. -/
noncomputable def ExternalLanguageLaw.information
    {P : Process k d Act World} (L : ExternalLanguageLaw P)
    (s : State k d) (a : Act) : ℝ :=
  externalInformation (L.joint s a)

/-- A valid external language law has nonnegative information opportunity.
This is finite conditional mutual information, rather than an arbitrary
natural-number intake signal. -/
theorem externalInformation_nonneg (joint : Byte → Byte → ℝ)
    (hjoint : ∀ now later, 0 ≤ joint now later) :
    0 ≤ externalInformation joint := by
  unfold externalInformation
  exact CGB.InformationProjection.condMutualInfo_nonneg
    (fun (_ : Unit) now later => joint now later) (fun _ now later => hjoint now later)

/-- The state/action-indexed opportunity inherits CMI nonnegativity from the
registered external language law. -/
theorem ExternalLanguageLaw.information_nonneg
    {P : Process k d Act World} (L : ExternalLanguageLaw P)
    (s : State k d) (a : Act) : 0 ≤ L.information s a := by
  exact externalInformation_nonneg (L.joint s a) (L.nonneg s a)

/-- A finite registered family of legal *external* language continuations.
It is an evaluation contract supplied by the mother world, not a buffer held
by the GDN.  Restricting probes to external arrivals ensures that copying the
organism's own bytes cannot manufacture a future distinction. -/
structure ExternalFutureProbes (m : ℕ) where
  continuation : Fin m → List Arrival
  external : ∀ i a, a ∈ continuation i → a.source = .external

/-- What one GDN state still says about each registered future external
language continuation.  This is deliberately a prediction/readout signature,
not raw parameter equality: two numerically different states have the same
reserve when no registered future language interaction can tell them apart. -/
abbrev FutureLanguageSignature (m : ℕ) := Fin m → Byte → ℝ

noncomputable def futureLanguageSignature (M : Model k d)
    {m : ℕ} (Q : ExternalFutureProbes m) (s : State k d) :
    FutureLanguageSignature m :=
  fun i => allocation M ((Q.continuation i).foldl (step M) s)

/-- A state family retains its future language reserve through an update when
the new signatures still determine every old signature.  This is the exact
causal-carrier criterion applied to GDN future readouts; it adds no reserve
field to `State`. -/
noncomputable def futureReservePreserved (M : Model k d)
    {m n : ℕ} (Q : ExternalFutureProbes m)
    (before after : Fin n → State k d) : Prop :=
  ∀ i j,
    futureLanguageSignature M Q (after i) = futureLanguageSignature M Q (after j) →
      futureLanguageSignature M Q (before i) = futureLanguageSignature M Q (before j)

/-- A preserved future-language reserve cannot silently merge two histories
that a registered external continuation could previously distinguish.  This is
the long-delay retention condition for any causal difference already carried
by the body; it adds neither an action log nor a separate eligibility state. -/
theorem future_reserve_preserved_keeps_signature_distinction (M : Model k d)
    {m n : ℕ} (Q : ExternalFutureProbes m)
    (before after : Fin n → State k d) (i j : Fin n)
    (hpreserved : futureReservePreserved M Q before after)
    (hbefore : futureLanguageSignature M Q (before i) ≠
      futureLanguageSignature M Q (before j)) :
    futureLanguageSignature M Q (after i) ≠
      futureLanguageSignature M Q (after j) := by
  intro hafter
  exact hbefore (hpreserved i j hafter)

/-- The semantic form of information dissipation: a previously useful future
language distinction is merged by an update.  It is a relation between two
counterfactual histories, rather than a second energy or food account. -/
noncomputable def losesFutureLanguageDistinction (M : Model k d)
    {m n : ℕ} (Q : ExternalFutureProbes m)
    (before after : Fin n → State k d) : Prop :=
  ∃ i j,
    futureLanguageSignature M Q (before i) ≠ futureLanguageSignature M Q (before j) ∧
      futureLanguageSignature M Q (after i) = futureLanguageSignature M Q (after j)

/-- A finite audit can count how many distinct future-language signatures the
same state family exposes.  This count is an observable lower-resolution
report of reserve, not persistent organism state; its logarithm is the number
of bits only after a finite probe family and a representation precision have
been declared. -/
noncomputable def futureReserveClasses (M : Model k d)
    {m n : ℕ} (Q : ExternalFutureProbes m) (states : Fin n → State k d) : ℕ := by
  classical
  exact (Finset.univ.image fun i => futureLanguageSignature M Q (states i)).card

/-- The concrete GDN form of a possible forgetting/interference event: expose
the same external arrival to alternative prior histories, then ask whether it
merged a distinction needed by a registered future language probe. -/
noncomputable def arrivalLosesFutureLanguageDistinction (M : Model k d)
    {m n : ℕ} (Q : ExternalFutureProbes m) (a : Arrival)
    (states : Fin n → State k d) : Prop :=
  losesFutureLanguageDistinction M Q states (fun i => step M (states i) a)

/-- A future distinction is not dissipated exactly when all of the prior
future language signature remains readable from the post-update signature.
This reuses the existing causal-carrier theorem instead of introducing a
separate retention mechanism. -/
theorem future_reserve_recoverable_iff (M : Model k d)
    {m n : ℕ} (Q : ExternalFutureProbes m)
    (before after : Fin n → State k d) :
    (∃ restore : FutureLanguageSignature m → FutureLanguageSignature m,
      ∀ i, restore (futureLanguageSignature M Q (after i)) =
        futureLanguageSignature M Q (before i)) ↔
      futureReservePreserved M Q before after := by
  simpa only [futureReservePreserved] using
    (CGB.LanguageLifeKernel.history_readout_exists_iff
      (fun i => futureLanguageSignature M Q (after i))
      (fun i => futureLanguageSignature M Q (before i))
      (fun _ _ => 0))

/-- Merging a prior distinct future-language signature is precisely the
failure of that causal retention contract. -/
theorem loses_future_language_distinction_iff_not_preserved (M : Model k d)
    {m n : ℕ} (Q : ExternalFutureProbes m)
    (before after : Fin n → State k d) :
    losesFutureLanguageDistinction M Q before after ↔
      ¬ futureReservePreserved M Q before after := by
  classical
  constructor
  · rintro ⟨i, j, hbefore, hafter⟩ hpreserved
    exact hbefore (hpreserved i j hafter)
  · intro hnot
    unfold futureReservePreserved at hnot
    push Not at hnot
    obtain ⟨i, j, hafter, hbefore⟩ := hnot
    exact ⟨i, j, hbefore, hafter⟩

/-- Information-theoretic active necessity for the concrete GDN port.  If
settling into `passive` eliminates the conditional information available from
the external language boundary, while positive external information recurs on
the realised trajectory, the policy cannot settle into `passive`.

The premise is deliberately a boundary statement, not a synthetic hunger
counter.  To turn available information into a learning claim, it must also
meet the causal `externallyUseful` scoring certificate above. -/
theorem recurrent_external_information_forces_activity
    (P : Process k d Act World) (L : ExternalLanguageLaw P) (passive : Act)
    (hpassive : CGB.ActiveInference.eventuallyPassive
      (fun n => P.policy (P.run n).1) passive →
      ∃ N, ∀ n, N ≤ n →
        L.information (P.run n).1 (P.policy (P.run n).1) = 0)
    (hrecurrent : ∀ N, ∃ n, N ≤ n ∧ (P.arrival n).source = .external ∧
      0 < L.information (P.run n).1 (P.policy (P.run n).1)) :
    ¬ CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive := by
  intro hp
  obtain ⟨N, hN⟩ := hpassive hp
  obtain ⟨n, hn, _, hpositive⟩ := hrecurrent N
  rw [hN n hn] at hpositive
  exact lt_irrefl 0 hpositive

/-- If settling into a passive policy rules out recurrent externally useful
language, but such events actually recur on this same trajectory, the policy
cannot settle.  This is the language form of active necessity: no arbitrary
`intake`, opportunity counter, food pool, or maintenance currency is added. -/
theorem recurrent_external_evidence_forces_activity (P : Process k d Act World)
    (passive : Act)
    (hpassive : CGB.ActiveInference.eventuallyPassive
      (fun n => P.policy (P.run n).1) passive →
      ∃ N, ∀ n, N ≤ n → ¬ P.externallyUseful n)
    (hrecurrent : ∀ N, ∃ n, N ≤ n ∧ P.externallyUseful n) :
    ¬ CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive := by
  intro hp
  obtain ⟨N, hN⟩ := hpassive hp
  obtain ⟨n, hn, hevent⟩ := hrecurrent N
  exact hN n hn hevent

/-- Conditional language-level joint result for the real GDN.  Every
condition is about the actual language trajectory: source provenance,
recurrent post-update improvement, the causal effect of action on the next
arrival, and a non-uniform prequential margin.  The result intentionally says
nothing about physical food, material ingestion, or an auxiliary reserve
state.  It is still not a full `Medium.Lives` construction. -/
theorem scored_external_language_trajectory (P : Process k d Act World)
    (hη : 0 < P.model.headStep) (passive : Act)
    (hpassive : CGB.ActiveInference.eventuallyPassive
      (fun n => P.policy (P.run n).1) passive →
      ∃ N, ∀ n, N ≤ n → ¬ P.externallyUseful n)
    (hrecurrent : ∀ N, ∃ n, N ≤ n ∧ P.externallyUseful n)
    (haction : ∃ n a a', (P.boundary (P.run n).2 (P.run n).1 a).byte ≠
      (P.boundary (P.run n).2 (P.run n).1 a').byte)
    (hmargin : ∀ n, nll (P.scores n) (P.arrival n).byte < Real.log 256) :
    (∀ n, Positive (P.run n).1) ∧
    (¬ CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive) ∧
    (¬ ∃ predict : World → State k d → Byte, ∀ n a,
      predict (P.run n).2 (P.run n).1 = (P.boundary (P.run n).2 (P.run n).1 a).byte) ∧
    (∀ n, 0 < Real.log 256 - nll (P.scores n) (P.arrival n).byte) ∧
    (∀ N, ∃ n, N ≤ n ∧
      nll (P.scores (n + 1)) (P.arrival (n + 1)).byte <
        nll (P.scores n) (P.arrival (n + 1)).byte) ∧
    (¬ ∃ N, ∀ n, N ≤ n → (P.run (n + 1)).1 = (P.run n).1) := by
  refine ⟨P.run_positive,
    P.recurrent_external_evidence_forces_activity passive hpassive hrecurrent,
    ?_, ?_, ?_, ?_⟩
  · rintro ⟨predict, hpredict⟩
    obtain ⟨n, a, a', hne⟩ := haction
    exact hne ((hpredict n a).symm.trans (hpredict n a'))
  · intro n
    linarith [hmargin n]
  · intro N
    obtain ⟨n, hn, hevent⟩ := hrecurrent N
    exact ⟨n, hn, P.externallyUseful_improves n hevent⟩
  · rintro ⟨N, hfreeze⟩
    have h := hfreeze N (le_refl N)
    rw [P.run_step] at h
    exact step_ne_of_positive_head_step P.model hη (P.run N).1 (P.arrival N) h

/-- A conditional scored joint trajectory for this real GDN. Conditions are
on the actual world: registered intake, a finite passive opportunity budget
under the same intervenable transition, an action contrast, a nonnegative
predictive margin, and recurrent next-byte persistence with bounded
readout drift. Recurrent future score gain is *derived*, not supplied as
`hlearn`. Numeric positivity is still not material/biological viability.
This deterministic-world statement scores actual future bytes, not a claim
about the conditional law of an unspecified stochastic language world. -/
theorem scored_joint_trajectory (P : Process k d Act World) (hη : 0 < P.model.headStep)
    (intake : World → State k d → Act → Arrival → ℕ) (passive : Act)
    (hunbounded : CGB.ActiveInference.unbounded (P.signal intake))
    (opportunity : State k d × World → ℕ)
    (hspend : ∀ z, opportunity (P.contract.step z (P.contract.response z passive)) +
      intake z.2 z.1 passive (P.boundary z.2 z.1 passive) ≤ opportunity z)
    (haction : ∃ n a a', (P.boundary (P.run n).2 (P.run n).1 a).byte ≠
      (P.boundary (P.run n).2 (P.run n).1 a').byte)
    (m : ℝ) (hm : 0 ≤ m)
    (hpay : ∀ n, nll (P.scores n) (P.arrival n).byte < Real.log 256 - m)
    (hopportunity : ∀ N, ∃ n, N ≤ n ∧ ∃ δ : ℝ,
      (P.arrival (n + 1)).byte = (P.arrival n).byte ∧
      (∀ b, |P.scores (n + 1) b -
        logits (finish P.model (P.snapshot n) (P.arrival n)) (P.snapshot n).feature b| ≤ δ) ∧
      2 * δ < nll (P.scores n) (P.arrival n).byte -
        nll (logits (finish P.model (P.snapshot n) (P.arrival n))
          (P.snapshot n).feature) (P.arrival n).byte) :
    (∀ n, Positive (P.run n).1) ∧
    (¬ CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive) ∧
    (¬ ∃ predict : World → State k d → Byte, ∀ n a,
      predict (P.run n).2 (P.run n).1 = (P.boundary (P.run n).2 (P.run n).1 a).byte) ∧
    (∀ n, 0 < Real.log 256 - nll (P.scores n) (P.arrival n).byte ∧
      0 < Real.log 256 - nll (P.scores n) (P.arrival n).byte - m) ∧
    (∀ N, ∃ n, N ≤ n ∧
      nll (P.scores (n + 1)) (P.arrival (n + 1)).byte <
        nll (P.scores n) (P.arrival (n + 1)).byte) ∧
    (¬ ∃ N, ∀ n, N ≤ n → (P.run (n + 1)).1 = (P.run n).1) := by
  refine ⟨P.run_positive, ?_, ?_, ?_, ?_, ?_⟩
  · exact CGB.SelfProduction.passive_opportunity_forces_activity P.contract passive opportunity
      (fun z input => intake z.2 z.1 input.1 input.2) hspend hunbounded
  · rintro ⟨predict, hpredict⟩
    obtain ⟨n, a, a', hne⟩ := haction
    exact hne ((hpredict n a).symm.trans (hpredict n a'))
  · intro n
    constructor <;> linarith [hpay n]
  · intro N
    obtain ⟨n, hn, δ, hr, hd, hg⟩ := hopportunity N
    exact ⟨n, hn, P.next_byte_improves n δ hr hd hg⟩
  · rintro ⟨N, hfreeze⟩
    have h := hfreeze N (le_refl N)
    rw [P.run_step] at h
    exact step_ne_of_positive_head_step P.model hη (P.run N).1 (P.arrival N) h

end Process

end CGB.GDNLife
