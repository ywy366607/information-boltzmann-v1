import CGBAxioms.GenerationGDNLife
import CGBAxioms.GenerationGrowth

/-!
# Necessary bridges for a language-life existence theorem

Keep continuous GDN mathematics separate from its physical realization.
Finite hardware does not imply lookup-table execution. Conversely, an exact
codec of *all* real states into a finite microscopic body is impossible.
This is not a requirement to quantize the analytic model before proving its
properties: the later realization may use a reachable domain or an explicit
approximation relation.

Finite physical memory also cannot improve one fixed state loss strictly
forever while that loss never rises again. Continuing useful learning needs
renewed opportunities, a changing world objective, or intervening increases;
merely observing that a real-valued head never stops moving is insufficient.
-/

namespace CGB.LanguageExistence

open Finset

/-- A finite physical code cannot losslessly represent every real GDN state.
The real coordinate is an existing head bias, not an extra memory register. -/
theorem no_global_finite_gdn_codec {Physical : Type*} [Finite Physical] (k d : ℕ)
    (encode : GDNLife.State k d → Physical) (decode : Physical → GDNLife.State k d) :
    ¬ (∀ s, decode (encode s) = s) := by
  intro h
  have he : Function.Injective encode := Function.LeftInverse.injective h
  have hr := he.comp (GDNLife.real_bias_injective (k := k) (d := d))
  have hfinite : Finite ℝ := Finite.of_injective _ hr
  exact (Set.infinite_univ : (Set.univ : Set ℝ).Infinite)
    (Set.finite_univ_iff.mpr hfinite)

/-- P1a plus a finite number of physical sites rules out a global exact
real-state codec; it does not rule out a continuous macro model or finite
precision matrix arithmetic. -/
theorem p1a_no_global_real_codec {U Λ A I : Type*} [Finite U]
    (M : Medium U Λ A I) (hp : M.P1a) (k d : ℕ)
    (encode : GDNLife.State k d → (U → Λ)) (decode : (U → Λ) → GDNLife.State k d) :
    ¬ (∀ s, decode (encode s) = s) := by
  letI := M.finite_states hp
  exact no_global_finite_gdn_codec k d encode decode

/-- A nonincreasing real sequence with finite range eventually attains a
fixed value. No finiteness of the real number type is assumed. -/
theorem finite_range_nonincreasing_eventually_constant (f : ℕ → ℝ)
    (hf : (Set.range f).Finite) (hstep : ∀ n, f (n + 1) ≤ f n) :
    ∃ N, ∀ n, N ≤ n → f n = f N := by
  classical
  let values := hf.toFinset
  have hmem (n : ℕ) : f n ∈ values := by
    change f n ∈ hf.toFinset
    exact hf.mem_toFinset.mpr ⟨n, rfl⟩
  have hne : values.Nonempty := ⟨f 0, hmem 0⟩
  have hm := min'_mem values hne
  have hr : values.min' hne ∈ Set.range f := hf.mem_toFinset.mp hm
  obtain ⟨N, hN⟩ := hr
  have hleast (n : ℕ) : f N ≤ f n := by
    rw [hN]
    exact min'_le values _ (hmem n)
  have hdown (n : ℕ) (hn : N ≤ n) : f n ≤ f N := by
    induction n, hn using Nat.le_induction with
    | base => exact le_rfl
    | succ n hn ih => exact (hstep n).trans ih
  exact ⟨N, fun n hn => le_antisymm (hdown n hn) (hleast n)⟩

/-- A fixed loss on a finite body cannot keep strictly improving while
remaining nonincreasing at every step. This is a finite-body result, not a
claim that arbitrary real GDN trajectories must freeze. -/
theorem finite_body_stationary_loss_eventually_constant {State : Type*} [Finite State]
    (run : ℕ → State) (loss : State → ℝ)
    (hstep : ∀ n, loss (run (n + 1)) ≤ loss (run n)) :
    ∃ N, ∀ n, N ≤ n → loss (run n) = loss (run N) := by
  have hf : (Set.range (fun n => loss (run n))).Finite :=
    (Set.finite_range loss).subset (by rintro _ ⟨n, rfl⟩; exact ⟨run n, rfl⟩)
  exact finite_range_nonincreasing_eventually_constant _ hf hstep

/-- Even *eventually* nonincreasing stationary loss is incompatible with
arbitrarily late strict improvements on the same finite body. The theorem
does not forbid world-conditioned relearning after drift. -/
theorem recurrent_gain_excludes_eventual_stationary_descent
    {State : Type*} [Finite State] (run : ℕ → State) (loss : State → ℝ)
    (hgain : ∀ N, ∃ n, N ≤ n ∧ loss (run (n + 1)) < loss (run n)) :
    ¬ (∃ N, ∀ n, N ≤ n → loss (run (n + 1)) ≤ loss (run n)) := by
  rintro ⟨N, hmono⟩
  have hstep (n : ℕ) : loss (run (N + (n + 1))) ≤ loss (run (N + n)) := by
    simpa only [Nat.add_assoc] using hmono (N + n) (by omega)
  obtain ⟨K, hconstant⟩ := finite_body_stationary_loss_eventually_constant
    (fun n => run (N + n)) loss hstep
  obtain ⟨n, hn, hg⟩ := hgain (N + K)
  have hnN : N ≤ n := by omega
  have hk : K ≤ n - N := by omega
  have h₀ := hconstant (n - N) hk
  have h₁ := hconstant (n - N + 1) (by omega)
  have heq₀ : N + (n - N) = n := by omega
  have heq₁ : N + (n - N + 1) = n + 1 := by omega
  simp only [heq₀] at h₀
  simp only [heq₁] at h₁
  rw [h₀, h₁] at hg
  exact (lt_irrefl _ hg)

/-- The obstruction also applies on one actual trajectory: a finite body
cannot reproduce an endlessly strictly descending fixed real readout exactly.
It does not require encoding every mathematically possible real state. -/
theorem strict_descent_has_no_finite_exact_readout
    {State Physical : Type*} [Finite Physical]
    (run : ℕ → State) (loss : State → ℝ)
    (encode : State → Physical) (physicalLoss : Physical → ℝ)
    (hcorrect : ∀ n, physicalLoss (encode (run n)) = loss (run n))
    (hstrict : ∀ n, loss (run (n + 1)) < loss (run n)) : False := by
  have hstep (n : ℕ) : physicalLoss (encode (run (n + 1))) ≤
      physicalLoss (encode (run n)) := by
    rw [hcorrect, hcorrect]
    exact (hstrict n).le
  obtain ⟨N, hconstant⟩ := finite_body_stationary_loss_eventually_constant
    (fun n => encode (run n)) physicalLoss hstep
  have h := hconstant (N + 1) (by omega)
  rw [hcorrect, hcorrect] at h
  exact (ne_of_lt (hstrict N)) h

/-- A finite invariant domain can use the *same* arithmetic update, without
replacing it by a transition lookup table. The finite-domain and closure
proofs must be supplied for the actual process; they are not automatic for
GDN or consequences of bounded dimension. -/
def restrictedStep {State Input : Type*} (domain : Set State)
    (next : State → Input → State) (hclosed : ∀ s, s ∈ domain → ∀ u, next s u ∈ domain) :
    domain → Input → domain :=
  fun s u => ⟨next s.val u, hclosed s.val s.property u⟩

theorem restricted_step_refines {State Input : Type*} (domain : Set State)
    (next : State → Input → State) (hclosed : ∀ s, s ∈ domain → ∀ u, next s u ∈ domain)
    (s : domain) (u : Input) :
    (restrictedStep domain next hclosed s u).val = next s.val u := rfl

theorem restricted_fold_refines {State Input : Type*} (domain : Set State)
    (next : State → Input → State) (hclosed : ∀ s, s ∈ domain → ∀ u, next s u ∈ domain)
    (s : domain) (inputs : List Input) :
    (inputs.foldl (restrictedStep domain next hclosed) s).val = inputs.foldl next s.val := by
  induction inputs generalizing s with
  | nil => rfl
  | cons u inputs ih => exact ih (restrictedStep domain next hclosed s u)

/-- The physical-time prior never rewrites learned head bias. This is a
property of the existing GDN split, not a newly installed protected register. -/
theorem gdn_prior_preserves_bias {k d : ℕ} (M : GDNLife.Model k d)
    (s : GDNLife.State k d) (elapsed : ℝ) : (GDNLife.prior M s elapsed).bias = s.bias := by
  classical
  by_cases h : elapsed = 0 <;> simp [GDNLife.prior, h]

/-- With no new arrived byte, any sequence of physical-time priors preserves
the head bias. Real-time passage is not conflated with zero-time freezing. -/
theorem gdn_silent_continuation_bias {k d : ℕ} (M : GDNLife.Model k d)
    (s : GDNLife.State k d) (times : List ℝ) :
    (times.foldl (GDNLife.prior M) s).bias = s.bias := by
  induction times generalizing s with
  | nil => rfl
  | cons elapsed times ih =>
    simpa only [List.foldl_cons, gdn_prior_preserves_bias] using ih (GDNLife.prior M s elapsed)

/-- Alternative bytes arriving at the same prepared body install a
head-bias difference determined by the actual local update. No teacher,
extra history page, or externally written memory is supplied. -/
theorem gdn_alternative_arrival_bias {k d : ℕ} (M : GDNLife.Model k d)
    (p : GDNLife.Prepared k d) (a a' : GDNLife.Arrival) (b : GDNLife.Byte) :
    (GDNLife.finish M p a).bias b - (GDNLife.finish M p a').bias b =
      M.headStep * ((if b = a.byte then 1 else 0) - (if b = a'.byte then 1 else 0)) := by
  have ha : GDNLife.logits (GDNLife.assimilate M p.state a) p.feature =
      GDNLife.logits p.state p.feature := rfl
  have ha' : GDNLife.logits (GDNLife.assimilate M p.state a') p.feature =
      GDNLife.logits p.state p.feature := rfl
  change (p.state.bias b + M.headStep * ((if b = a.byte then 1 else 0) -
    CGB.Medium.softmax (GDNLife.logits (GDNLife.assimilate M p.state a) p.feature) b)) -
    (p.state.bias b + M.headStep * ((if b = a'.byte then 1 else 0) -
      CGB.Medium.softmax (GDNLife.logits (GDNLife.assimilate M p.state a') p.feature) b)) = _
  rw [ha, ha']
  ring

/-- A distinction written by different bytes survives arbitrarily many
silent physical-time steps in this real GDN. This supplies a concrete memory
mechanism, but does not identify it with the finite-medium CMI life clause. -/
theorem gdn_written_distinction_survives_silence {k d : ℕ} (M : GDNLife.Model k d)
    (hη : M.headStep ≠ 0) (p : GDNLife.Prepared k d)
    (a a' : GDNLife.Arrival) (hne : a.byte ≠ a'.byte) (times times' : List ℝ) :
    (times.foldl (GDNLife.prior M) (GDNLife.finish M p a)).bias a.byte -
      (times'.foldl (GDNLife.prior M) (GDNLife.finish M p a')).bias a.byte = M.headStep ∧
    (times.foldl (GDNLife.prior M) (GDNLife.finish M p a)) ≠
      (times'.foldl (GDNLife.prior M) (GDNLife.finish M p a')) := by
  have hdiff :
      (times.foldl (GDNLife.prior M) (GDNLife.finish M p a)).bias a.byte -
        (times'.foldl (GDNLife.prior M) (GDNLife.finish M p a')).bias a.byte = M.headStep := by
    rw [gdn_silent_continuation_bias, gdn_silent_continuation_bias]
    simpa only [ite_true, hne, ite_false, sub_zero, mul_one] using
      gdn_alternative_arrival_bias M p a a' a.byte
  refine ⟨hdiff, ?_⟩
  intro hsame
  rw [hsame, sub_self] at hdiff
  exact hη hdiff.symm

/-- The existing two-state population law gives exactly the actual GDN
precision-weighted mean update, with up/down probabilities determined by the
same local gain. This is not the mixed-key delta-write identity. -/
theorem precision_assimilation_is_two_state_step (p w m v : ℝ)
    (hp : 0 < p) (hw : 0 ≤ w) :
    (p * m + w * v) / (p + w) =
      Generation.twoStateStep ((w / (p + w)) * v) ((w / (p + w)) * (1 - v)) m := by
  rw [Generation.precision_assimilation_is_innovation p w m v hp hw]
  unfold Generation.twoStateStep
  ring

/-- For a bounded occupancy target the two rates are valid probabilities;
calm `v=1/2` balances them. A complete physical refinement must still expose
where the exchanged particles and precision dynamics reside. -/
theorem precision_two_state_rates_valid (p w v : ℝ) (hp : 0 < p) (hw : 0 ≤ w)
    (hv0 : 0 ≤ v) (hv1 : v ≤ 1) :
    0 ≤ (w / (p + w)) * v ∧ 0 ≤ (w / (p + w)) * (1 - v) ∧
    (w / (p + w)) * v + (w / (p + w)) * (1 - v) < 1 := by
  have hb := Generation.precision_assimilation_gain_bounds p w hp hw
  refine ⟨mul_nonneg hb.1 hv0, mul_nonneg hb.1 (by linarith), ?_⟩
  nlinarith [hb.2]

/-- Natural content between zero and precision represents a bounded
occupancy mean. It is a property of the same memory, not a new reservoir. -/
def UnitMemory {k d : ℕ} (s : GDNLife.State k d) : Prop :=
  ∀ j i, 0 ≤ s.natural j i ∧ s.natural j i ≤ s.precision j

theorem bounded_mean {k d : ℕ} (M : GDNLife.Model k d)
    (s : GDNLife.State k d) (hs : UnitMemory s) (j : Fin k) (i : Fin d) :
    0 ≤ GDNLife.mean M s j i ∧ GDNLife.mean M s j i ≤ 1 := by
  have hd : 0 < max (s.precision j) M.precisionFloor :=
    lt_of_lt_of_le M.floor_pos (le_max_right _ _)
  constructor
  · exact div_nonneg (hs j i).1 hd.le
  · apply (div_le_one hd).2
    exact (hs j i).2.trans (le_max_left _ _)

theorem prior_unit_memory {k d : ℕ} (M : GDNLife.Model k d)
    (s : GDNLife.State k d) (hp : GDNLife.Positive s) (hs : UnitMemory s)
    (elapsed : ℝ) (ht : 0 ≤ elapsed) : UnitMemory (GDNLife.prior M s elapsed) := by
  classical
  intro j i
  by_cases hz : elapsed = 0
  · simpa [GDNLife.prior, hz] using hs j i
  · have hmean := bounded_mean M s hs j i
    have hp' := GDNLife.prior_positive M s hp elapsed ht j
    simp only [GDNLife.prior, hz, ↓reduceIte] at hp' ⊢
    constructor
    · exact mul_nonneg hp'.le hmean.1
    · simpa only [mul_one] using mul_le_mul_of_nonneg_left hmean.2 hp'.le

theorem assimilate_unit_memory {k d : ℕ} (M : GDNLife.Model k d)
    (s : GDNLife.State k d) (hs : UnitMemory s) (a : GDNLife.Arrival)
    (hv : ∀ i, 0 ≤ M.value a.byte i ∧ M.value a.byte i ≤ 1) :
    UnitMemory (GDNLife.assimilate M s a) := by
  intro j i
  have hw := GDNLife.incoming_nonneg M a j
  constructor
  · exact add_nonneg (hs j i).1 (mul_nonneg hw (hv i).1)
  · change s.natural j i + GDNLife.incoming M a j * M.value a.byte i ≤
      s.precision j + GDNLife.incoming M a j
    nlinarith [(hs j i).2, mul_le_mul_of_nonneg_left (hv i).2 hw]

/-- Bounded byte embeddings imply bounded memory means through the *full*
time-prior/assimilation/head step. No assumption of contraction or a finite
real type is introduced. Head-parameter bounds remain a distinct obligation. -/
theorem step_unit_memory {k d : ℕ} (M : GDNLife.Model k d)
    (s : GDNLife.State k d) (hp : GDNLife.Positive s) (hs : UnitMemory s)
    (a : GDNLife.Arrival) (hv : ∀ i, 0 ≤ M.value a.byte i ∧ M.value a.byte i ≤ 1) :
    UnitMemory (GDNLife.step M s a) :=
  assimilate_unit_memory M _ (prior_unit_memory M s hp hs a.elapsed a.elapsed_nonneg) a hv

end CGB.LanguageExistence
