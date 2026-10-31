import CGBAxioms.GenerationSelfProduction
import CGBAxioms.Models
import CGBAxioms.GenerationInformationProjection

/-!
# Memory carried by a conserved observable, not a frozen body

Generalize the existing quiet-identity memory marginal. Quiet dynamics may
continue to rearrange the body. Only the registered observable must be
preserved. P2a supplies this for total content when positive quiet events
are closed; no identity kernel or added protected memory page is assumed.

The actual tagged observable law is preserved at every delay. Finite CMI
data processing and a supported counterfactual write derive the original
full-body memory clause, without assuming later memory or freezing. Actual
world write witnesses and viability still need instantiation; this is not full life.
-/

namespace CGB.ConservedMemory

open Finset

variable {U Λ A I Obs : Type*}
variable [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]
variable [DecidableEq Obs]

/-- Any test of a preserved observable has constant conditional expectation
under the specified stimulus. The body may still rearrange. -/
theorem observable_test_preserved (M : Medium U Λ A I) (R : ℝ) (hc : M.Clocked R)
    (stimulus : U → I) (observe : (U → Λ) → Obs) (test : Obs → ℝ)
    (hpreserve : ∀ x y, 0 < M.kernel R stimulus x y → observe y = observe x)
    (x : U → Λ) :
    (∑ y, test (observe y) * M.kernel R stimulus x y) = test (observe x) := by
  have hterm (y : U → Λ) : test (observe y) * M.kernel R stimulus x y =
      test (observe x) * M.kernel R stimulus x y := by
    by_cases hp : 0 < M.kernel R stimulus x y
    · rw [hpreserve x y hp]
    · have hz : M.kernel R stimulus x y = 0 :=
        le_antisymm (le_of_not_gt hp) (M.kernel_nonneg hc _ _ _)
      simp [hz]
  simp_rw [hterm]
  rw [← mul_sum, M.kernel_sum, mul_one]

/-- A stochastic kernel preserving an observable also preserves each of its
indicator test functions. The body itself need not stop moving. -/
theorem observable_indicator_preserved (M : Medium U Λ A I) (W : World U Λ I)
    (hc : M.Clocked W.clock) (observe : (U → Λ) → Obs)
    (hpreserve : ∀ x y, 0 < M.kernel W.clock W.quiet x y → observe y = observe x)
    (x : U → Λ) (r : Obs) :
    (∑ y, (if observe y = r then (1 : ℝ) else 0) * M.kernel W.clock W.quiet x y) =
      if observe x = r then 1 else 0 :=
  observable_test_preserved M W.clock hc W.quiet observe
    (fun v => if v = r then 1 else 0) hpreserve x

/-- Push the actual tagged memory law through an observable. World and
initial-body tags remain exactly the original ones. -/
noncomputable def observedMemory (M : Medium U Λ A I) (W : World U Λ I)
    (observe : (U → Λ) → Obs) (n₀ m : ℕ) (x₀ : U → Λ) (w₀ : W.W) (r : Obs) : ℝ := by
  classical
  exact ∑ x, ∑ w, (if observe x = r then (1 : ℝ) else 0) * M.memLaw W n₀ m x w x₀ w₀

/-- Quiet can be nontrivial: each future observable marginal equals the
one installed by the actual first observation, even as the world evolves. -/
theorem quiet_observable_memory_marginal (M : Medium U Λ A I) (W : World U Λ I)
    (hc : M.Clocked W.clock) (observe : (U → Λ) → Obs)
    (hpreserve : ∀ x y, 0 < M.kernel W.clock W.quiet x y → observe y = observe x)
    (n₀ : ℕ) (x₀ : U → Λ) (w₀ : W.W) (r : Obs) :
    ∀ m, observedMemory M W observe n₀ (m + 1) x₀ w₀ r =
      observedMemory M W observe n₀ 1 x₀ w₀ r := by
  classical
  have hstep (k : ℕ) (hk : k ≠ 0) :
      observedMemory M W observe n₀ (k + 1) x₀ w₀ r =
        observedMemory M W observe n₀ k x₀ w₀ r := by
    have h := Medium.stepSum (fun x => if observe x = r then (1 : ℝ) else 0)
      (fun x w => M.memLaw W n₀ k x w x₀ w₀)
      (fun x w w' => W.moves w x w')
      (fun _ x y => M.kernel W.clock W.quiet x y)
      (fun x w => W.moves_sum w x)
      (fun _ x => observable_indicator_preserved M W hc observe hpreserve x r)
    simpa only [observedMemory, Medium.memLaw, hk, ite_false] using h
  intro m
  induction m with
  | zero => rfl
  | succ m ih =>
    rw [hstep (m + 1) (by omega)]
    exact ih

/-- The tagged law used by the original memory definition is nonnegative
under the actual clock admission, including the observation and quiet steps. -/
theorem tagged_memory_nonneg (M : Medium U Λ A I) (W : World U Λ I)
    (hc : M.Clocked W.clock) (n₀ m : ℕ) (x x₀ : U → Λ) (w w₀ : W.W) :
    0 ≤ M.memLaw W n₀ m x w x₀ w₀ := by
  induction m generalizing x w with
  | zero =>
    simp only [Medium.memLaw]
    split_ifs
    · exact M.law_nonneg_of_clocked W hc _ _ _ _
    · exact le_rfl
  | succ m ih =>
    exact sum_nonneg fun y _ => sum_nonneg fun v _ =>
      mul_nonneg (mul_nonneg (ih y v) (W.moves_nonneg v y w))
        (M.kernel_nonneg hc _ _ _)

/-- The first actual write marginalizes the world's normalized successor
law; this is not a hand-assigned probability on a supported path. -/
theorem one_write_body_marginal (M : Medium U Λ A I) (W : World U Λ I)
    (n₀ : ℕ) (x₀ x : U → Λ) (w₀ : W.W) :
    (∑ w, M.memLaw W n₀ 1 x w x₀ w₀) =
      M.law W (fun _ => W.sense) n₀ x₀ w₀ * M.kernel W.clock (W.sense w₀) x₀ x := by
  classical
  simp only [Medium.memLaw, ite_true, ite_and, mul_ite, ite_mul, zero_mul, mul_zero]
  simp only [sum_ite_irrel, sum_const_zero, sum_ite_eq', mem_univ, ↓reduceIte]
  rw [← sum_mul, ← mul_sum, W.moves_sum, mul_one]

/-- A positive registered event that changes the body has a positive actual
uniformized transition, at the same world clock. -/
theorem registered_event_kernel_pos (M : Medium U Λ A I) (R : ℝ) (hR : 0 < R)
    (stimulus : U → I) (a : A) (x : U → Λ)
    (hr : 0 < M.rate a stimulus x) (hne : M.act a stimulus x ≠ x) :
    0 < M.kernel R stimulus x (M.act a stimulus x) := by
  classical
  have hxy : x ≠ M.act a stimulus x := Ne.symm hne
  have hle : M.rate a stimulus x ≤ (M.jump stimulus).rate x (M.act a stimulus x) := by
    change M.rate a stimulus x ≤ ∑ b,
      if M.act b stimulus x = M.act a stimulus x ∧ x ≠ M.act a stimulus x then
        M.rate b stimulus x else 0
    calc M.rate a stimulus x =
        (if M.act a stimulus x = M.act a stimulus x ∧ x ≠ M.act a stimulus x then
          M.rate a stimulus x else 0) := by simp [hxy]
      _ ≤ _ := single_le_sum (s := (univ : Finset A))
        (f := fun b => if M.act b stimulus x = M.act a stimulus x ∧ x ≠ M.act a stimulus x then
          M.rate b stimulus x else 0)
        (fun b _ => by
          split_ifs
          · exact M.rate_nonneg _ _ _
          · exact le_rfl) (mem_univ a)
  simp only [Medium.kernel, hxy, ite_false, add_zero]
  exact div_pos (hr.trans_le hle) hR

/-- The observable law is exactly the deterministic projection used by
finite conditional-information data processing. -/
theorem observed_memory_as_project (M : Medium U Λ A I) (W : World U Λ I)
    (observe : (U → Λ) → Obs) (n₀ m : ℕ) (x₀ : U → Λ) (w₀ : W.W) (r : Obs) :
    observedMemory M W observe n₀ m x₀ w₀ r =
      InformationProjection.project observe (fun x => ∑ w, M.memLaw W n₀ m x w x₀ w₀) r := by
  classical
  unfold observedMemory InformationProjection.project
  apply sum_congr rfl
  intro x _
  by_cases h : observe x = r <;> simp [h]

theorem observed_memory_nonneg (M : Medium U Λ A I) (W : World U Λ I)
    (hc : M.Clocked W.clock) (observe : (U → Λ) → Obs)
    (n₀ m : ℕ) (x₀ : U → Λ) (w₀ : W.W) (r : Obs) :
    0 ≤ observedMemory M W observe n₀ m x₀ w₀ r := by
  rw [observed_memory_as_project]
  exact InformationProjection.project_nonneg observe _
    (fun x => sum_nonneg fun w _ => tagged_memory_nonneg M W hc n₀ m x x₀ w w₀) r

/-- A first-write coarse row is its actual prefix mass times the observed
transition probability. -/
theorem one_write_observed_row (M : Medium U Λ A I) (W : World U Λ I)
    (observe : (U → Λ) → Obs) (n₀ : ℕ) (x₀ : U → Λ) (w₀ : W.W) (r : Obs) :
    observedMemory M W observe n₀ 1 x₀ w₀ r =
      M.law W (fun _ => W.sense) n₀ x₀ w₀ *
        ∑ y, if observe y = r then M.kernel W.clock (W.sense w₀) x₀ y else 0 := by
  classical
  rw [observed_memory_as_project]
  unfold InformationProjection.project
  simp_rw [one_write_body_marginal]
  rw [mul_sum]
  apply sum_congr rfl
  intro y _
  split_ifs <;> simp

/-- Summing the observed outcomes returns the same actual prefix mass. -/
theorem one_write_observed_mass [Fintype Obs]
    (M : Medium U Λ A I) (W : World U Λ I) (observe : (U → Λ) → Obs)
    (n₀ : ℕ) (x₀ : U → Λ) (w₀ : W.W) :
    (∑ r, observedMemory M W observe n₀ 1 x₀ w₀ r) =
      M.law W (fun _ => W.sense) n₀ x₀ w₀ := by
  simp_rw [observed_memory_as_project]
  rw [InformationProjection.sum_project]
  simp_rw [one_write_body_marginal]
  rw [← mul_sum, M.kernel_sum, mul_one]

/-- A supported actual transition from a positive prefix gives positive
observed write mass, with no assigned path weights. -/
theorem observed_write_pos_of_transition
    (M : Medium U Λ A I) (W : World U Λ I) (hc : M.Clocked W.clock)
    (observe : (U → Λ) → Obs) (n₀ : ℕ) (x₀ y : U → Λ) (w₀ : W.W)
    (hprefix : 0 < M.law W (fun _ => W.sense) n₀ x₀ w₀)
    (hstep : 0 < M.kernel W.clock (W.sense w₀) x₀ y) :
    0 < observedMemory M W observe n₀ 1 x₀ w₀ (observe y) := by
  classical
  rw [one_write_observed_row]
  apply mul_pos hprefix
  have hle := single_le_sum (s := (univ : Finset (U → Λ)))
    (f := fun z => if observe z = observe y then M.kernel W.clock (W.sense w₀) x₀ z else 0)
    (fun z _ => by
      split_ifs
      · exact M.kernel_nonneg hc _ _ _
      · exact le_rfl) (mem_univ y)
  exact hstep.trans_le (by simpa only [ite_true] using hle)

/-- Coarse information is a certified lower bound on the original full-body
CMI, including zeros in the actual probability support. -/
theorem observed_memory_information_le [Fintype Obs]
    (M : Medium U Λ A I) (W : World U Λ I) (hc : M.Clocked W.clock)
    (observe : (U → Λ) → Obs) (n₀ m : ℕ) :
    Medium.condMutualInfo (fun x₀ w₀ r => observedMemory M W observe n₀ m x₀ w₀ r) ≤
      Medium.condMutualInfo (fun x₀ w₀ x => ∑ w, M.memLaw W n₀ m x w x₀ w₀) := by
  have h := InformationProjection.condMutualInfo_project_le
    (fun x₀ w₀ x => ∑ w, M.memLaw W n₀ m x w x₀ w₀) observe
    (fun x₀ w₀ x => sum_nonneg fun w _ => tagged_memory_nonneg M W hc n₀ m x x₀ w w₀)
  have heq : (fun x₀ w₀ => InformationProjection.project observe
      (fun x => ∑ w, M.memLaw W n₀ m x w x₀ w₀)) =
      (fun x₀ w₀ r => observedMemory M W observe n₀ m x₀ w₀ r) := by
    funext x₀ w₀ r
    exact (observed_memory_as_project M W observe n₀ m x₀ w₀ r).symm
  rw [heq] at h
  exact h

/-- A positive actual information write into a conserved finite observable
implies the original full-body `Remembers`, without freezing the body or
assuming a positive CMI at all later times. -/
theorem remembers_of_conserved_observable [Fintype Obs]
    (M : Medium U Λ A I) (W : World U Λ I) (hc : M.Clocked W.clock)
    (observe : (U → Λ) → Obs)
    (hpreserve : ∀ x y, 0 < M.kernel W.clock W.quiet x y → observe y = observe x)
    (n₀ : ℕ)
    (hwrite : 0 < Medium.condMutualInfo
      (fun x₀ w₀ r => observedMemory M W observe n₀ 1 x₀ w₀ r)) : M.Remembers W := by
  refine ⟨n₀, _, hwrite, ?_⟩
  intro m
  have h := observed_memory_information_le M W hc observe n₀ (m + 1)
  have heq : (fun x₀ w₀ r => observedMemory M W observe n₀ (m + 1) x₀ w₀ r) =
      (fun x₀ w₀ r => observedMemory M W observe n₀ 1 x₀ w₀ r) := by
    funext x₀ w₀ r
    exact quiet_observable_memory_marginal M W hc observe hpreserve n₀ x₀ w₀ r m
  rw [heq] at h
  exact h

/-- An actual counterfactual write creates positive information: at the
same reachable body, one world input is quiet and another has a positive
transition into a different observable class. Sparse CMI strictness then
replaces any assumed positive-information write or future-memory premise. -/
theorem remembers_of_quiet_and_actual_write [Fintype Obs]
    (M : Medium U Λ A I) (W : World U Λ I) (hc : M.Clocked W.clock)
    (observe : (U → Λ) → Obs)
    (hquiet : ∀ x y, 0 < M.kernel W.clock W.quiet x y → observe y = observe x)
    (n₀ : ℕ) (x₀ y : U → Λ) (wquiet wwrite : W.W)
    (hsense : W.sense wquiet = W.quiet)
    (hprefix₀ : 0 < M.law W (fun _ => W.sense) n₀ x₀ wquiet)
    (hprefix₁ : 0 < M.law W (fun _ => W.sense) n₀ x₀ wwrite)
    (hwrite : 0 < M.kernel W.clock (W.sense wwrite) x₀ y)
    (hdifferent : observe y ≠ observe x₀) : M.Remembers W := by
  classical
  apply remembers_of_conserved_observable M W hc observe hquiet n₀
  apply InformationProjection.condMutualInfo_pos_of_missing_rectangle
    (fun a b c => observedMemory M W observe n₀ 1 a b c)
    (fun a b c => observed_memory_nonneg M W hc observe n₀ 1 a b c)
  refine ⟨x₀, wquiet, observe y, ?_, ?_, ?_⟩
  · rw [one_write_observed_row, hsense]
    have h := observable_indicator_preserved M W hc observe hquiet x₀ (observe y)
    have hne : observe x₀ ≠ observe y := Ne.symm hdifferent
    simp only [ite_mul, one_mul, zero_mul, hne, ite_false] at h
    rw [h, mul_zero]
  · rw [one_write_observed_mass]
    exact hprefix₀
  · have hp := observed_write_pos_of_transition M W hc observe n₀ x₀ y wwrite hprefix₁ hwrite
    have hle := single_le_sum (s := (univ : Finset W.W))
      (f := fun w => observedMemory M W observe n₀ 1 x₀ w (observe y))
      (fun w _ => observed_memory_nonneg M W hc observe n₀ 1 x₀ w (observe y)) (mem_univ wwrite)
    exact hp.trans_le hle

/-- A local event invariant lifts to the uniformized quiet kernel. This
also covers conserving hops which touch a boundary site without importing
or exporting content. -/
theorem event_observable_kernel (M : Medium U Λ A I) (R : ℝ) (stimulus : U → I)
    (observe : (U → Λ) → Obs)
    (hconserve : ∀ a x, 0 < M.rate a stimulus x → observe (M.act a stimulus x) = observe x)
    (x y : U → Λ) (hp : 0 < M.kernel R stimulus x y) : observe y = observe x := by
  classical
  by_contra hne
  have hxy : x ≠ y := by intro h; subst y; exact hne rfl
  have hrate : (M.jump stimulus).rate x y = 0 := by
    change (∑ a, if M.act a stimulus x = y ∧ x ≠ y then M.rate a stimulus x else 0) = 0
    apply sum_eq_zero
    intro a _
    by_cases ha : M.act a stimulus x = y ∧ x ≠ y
    · rw [if_pos ha]
      have hn : ¬ 0 < M.rate a stimulus x := by
        intro hr
        have htotal := hconserve a x hr
        rw [ha.1] at htotal
        exact hne htotal
      exact le_antisymm (le_of_not_gt hn) (M.rate_nonneg _ _ _)
    · simp [ha]
  simp only [Medium.kernel, hrate, hxy, ite_false, zero_div, add_zero] at hp
  exact (lt_irrefl _ hp)

/-- Quiet specialization of the local event invariant. -/
theorem quiet_event_observable_kernel (M : Medium U Λ A I) (W : World U Λ I)
    (observe : (U → Λ) → Obs)
    (hconserve : ∀ a x, 0 < M.rate a W.quiet x → observe (M.act a W.quiet x) = observe x)
    (x y : U → Λ) (hp : 0 < M.kernel W.clock W.quiet x y) : observe y = observe x :=
  event_observable_kernel M W.clock W.quiet observe hconserve x y hp

/-- Conservation is applied only to positive quiet events which really are
closed, not silently to arbitrary boundary-driven writes. -/
theorem quiet_total_kernel (M : Medium U Λ A I) (W : World U Λ I)
    (h2 : M.P2a)
    (hclosed : ∀ a x, 0 < M.rate a W.quiet x → M.Closed a)
    (x y : U → Λ) (hp : 0 < M.kernel W.clock W.quiet x y) : M.total y = M.total x :=
  quiet_event_observable_kernel M W M.total
    (fun a x hr => h2 a W.quiet x (hclosed a x hr) hr) x y hp

/-- P2a protects the total-content memory marginal under positive closed
quiet exchange. This does not require an identity transition kernel. -/
theorem quiet_total_memory_marginal (M : Medium U Λ A I) (W : World U Λ I)
    (hc : M.Clocked W.clock) (h2 : M.P2a)
    (hclosed : ∀ a x, 0 < M.rate a W.quiet x → M.Closed a)
    (n₀ : ℕ) (x₀ : U → Λ) (w₀ : W.W) (r : ℕ) (m : ℕ) :
    observedMemory M W M.total n₀ (m + 1) x₀ w₀ r =
      observedMemory M W M.total n₀ 1 x₀ w₀ r :=
  quiet_observable_memory_marginal M W hc M.total (quiet_total_kernel M W h2 hclosed)
    n₀ x₀ w₀ r m

/-- In the existing four-site physical seed, quiet shuts the two reservoir
ports but leaves the hopping dynamics intact. Even boundary-touching hops
conserve content. No new rates or states are installed. -/
theorem seed_calm_event_conserves
    (a : (Fin 4 × Fin 4) ⊕ Bool) (x : Fin 4 → Bool) :
    Models.seed.total (Models.seed.act a Models.seed.calm x) = Models.seed.total x := by
  rcases a with a | b
  · change (∑ u, Bool.toNat ((if Models.ringAdj a.1 a.2 then
        Models.xhopAct a.1 a.2 x else x) u)) = ∑ u, Bool.toNat (x u)
    split_ifs
    · exact Models.xhopAct_sum _ _ _ _
    · rfl
  · cases b <;> simp [Models.seed]

/-- A structural finite clock bound: at most eighteen registered events,
each of rate at most one. This is not a biological rate calibration. -/
theorem seed_event_clocked : Models.seed.EventClocked 18 := by
  refine ⟨by norm_num, fun s x => ?_⟩
  have hrate (a : (Fin 4 × Fin 4) ⊕ Bool) : Models.seed.rate a s x ≤ 1 := by
    rcases a with a | b
    · simp only [Models.seed]; split_ifs <;> norm_num
    · cases b <;> simp only [Models.seed] <;> split_ifs <;> norm_num
  calc ∑ a, Models.seed.rate a s x ≤ ∑ _a : (Fin 4 × Fin 4) ⊕ Bool, (1 : ℝ) :=
      sum_le_sum fun a _ => hrate a
    _ = 18 := by simp

/-- The same physical seed really moves at quiet: this is not the old
identity-kernel memory shortcut. -/
theorem seed_calm_has_nontrivial_event :
    ∃ (a : (Fin 4 × Fin 4) ⊕ Bool) (x : Fin 4 → Bool),
      0 < Models.seed.rate a Models.seed.calm x ∧ Models.seed.act a Models.seed.calm x ≠ x := by
  refine ⟨.inl (0, 1), Models.seed.init, ?_, ?_⟩
  · norm_num [Models.seed, Models.ringAdj]
  · intro h
    have h₀ := congrFun h 0
    norm_num [Models.seed, Models.ringAdj, Models.xhopAct] at h₀

/-- Content distinctions written through the original seed's reservoir
ports have invariant quiet marginals despite its nontrivial hopping. The
remaining one-write information/DPI and viability obligations are separate. -/
theorem seed_quiet_total_memory_marginal
    (W : World (Fin 4) Bool Bool) (hc : Models.seed.Clocked W.clock)
    (hq : W.quiet = Models.seed.calm)
    (n₀ : ℕ) (x₀ : Fin 4 → Bool) (w₀ : W.W) (r : ℕ) (m : ℕ) :
    observedMemory Models.seed W Models.seed.total n₀ (m + 1) x₀ w₀ r =
      observedMemory Models.seed W Models.seed.total n₀ 1 x₀ w₀ r := by
  apply quiet_observable_memory_marginal Models.seed W hc Models.seed.total
  intro x y hp
  apply quiet_event_observable_kernel Models.seed W Models.seed.total _ x y hp
  intro a z _
  rw [hq]
  exact seed_calm_event_conserves a z

/-- Historical persistence requires calm to change the viability observable,
not merely to move some microscopic states. The same joint law and clock are
used; no alternate interruption protocol is substituted. -/
theorem historical_persistence_requires_alive_change
    (M : Medium U Λ A I) (W : World U Λ I)
    (hstationary : ∀ x, (∑ y, M.aliveInd W y * M.kernel W.clock M.calm x y) = M.aliveInd W x)
    (δ : ℝ) (T : ℕ) : ¬ M.Persists W δ T := by
  let calm : ℕ → W.W → U → I := fun _ _ => M.calm
  have hconstant (n : ℕ) : M.aliveProb W calm n = M.aliveProb W calm 0 := by
    induction n with
    | zero => rfl
    | succ n ih =>
      have hs := Medium.stepSum (M.aliveInd W) (M.law W calm n)
        (fun x w w' => W.moves w x w') (fun _ x y => M.kernel W.clock M.calm x y)
        (fun x w => W.moves_sum w x) (fun _ x => hstationary x)
      have hstep : M.aliveProb W calm (n + 1) = M.aliveProb W calm n := by
        simpa only [Medium.aliveProb, Medium.law, calm] using hs
      exact hstep.trans ih
  intro hp
  obtain ⟨_, _, hnormal, _, _, N, hdead⟩ := hp
  have hstart := hnormal 0
  have hbirth : M.aliveProb W (fun _ => W.sense) 0 = M.aliveProb W calm 0 := by
    simp [Medium.aliveProb, Medium.law]
  rw [hbirth] at hstart
  have h := hdead N (le_refl N)
  change M.aliveProb W calm N < _ at h
  rw [hconstant] at h
  linarith

/-- If calm conserves total and viability reads *only* total, the original
long-disconnection death requirement is impossible. Thus a conserving
candidate needs organization-dependent recovery, or a real boundary loss;
adding nonzero stirring alone is not a life generator. -/
theorem conserving_calm_needs_organizational_viability
    (M : Medium U Λ A I) (W : World U Λ I) (hc : M.Clocked W.clock)
    (hcalm : ∀ a x, 0 < M.rate a M.calm x → M.total (M.act a M.calm x) = M.total x)
    (score : ℕ → ℝ) (hscore : ∀ x, M.aliveInd W x = score (M.total x))
    (δ : ℝ) (T : ℕ) : ¬ M.Persists W δ T := by
  apply historical_persistence_requires_alive_change M W _ δ T
  intro x
  have h := observable_test_preserved M W.clock hc M.calm M.total score
    (event_observable_kernel M W.clock M.calm M.total hcalm) x
  simpa only [hscore] using h

end CGB.ConservedMemory
