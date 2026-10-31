import CGBAxioms.InformationEconomics

/-!
# Historical finite-candidate language admission

This is a contract, not an existence theorem.  It forces the four information
economics bases to use one persistent state: causal readout/transition,
selective writing, predictive allocation, and action-conditioned active
inference.  Its `reserve` field is the finite candidate controller's write
credit; it is not the language organism's food, maintenance currency, or
information reserve.  The real GDN language route instead uses the same model
state and externally useful language events in `GenerationGDNLife`.
-/

namespace CGB.LanguageLifeAdmission

open CGB.CausalLearning
open CGB.ActiveInference
open CGB.InformationEconomics

structure LanguageCandidate
    (Hist Model Ctx Act Obs Byte Answer : Type*)
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte] where
  reserve : ℕ
  possible : Support Model Ctx Act Obs
  encode : Hist → Memory Model reserve
  required : Hist → Answer
  advanceHistory : Hist → Byte → Hist
  read : Memory Model reserve → Answer
  next : Memory Model reserve → Byte → Memory Model reserve
  read_correct : ∀ h, read (encode h) = required h
  next_correct : ∀ h b, next (encode h) b = encode (advanceHistory h b)
  context : Hist → Byte → Ctx
  action : Hist → Byte → Act
  outcome : Hist → Byte → Obs
  write_refines : ∀ h b,
    next (encode h) b =
      CausalLearning.advance possible (encode h)
        (context h b) (action h b) (outcome h b)
  stream : Hist → Byte → ℝ
  allocation : Memory Model reserve → Byte → ℝ
  stream_nonneg : ∀ h b, 0 ≤ stream h b
  stream_norm : ∀ h, ∑ b, stream h b = 1
  allocation_pos : ∀ s b, 0 < allocation s b
  allocation_norm : ∀ s, ∑ b, allocation s b = 1
  trajectory : ℕ → Hist
  signal : ℕ → ℕ
  policy : Memory Model reserve → Act
  passive : Act
  response : Act → Byte → ℝ
  predict : Byte → ℝ
  viable_signal : unbounded signal
  passive_bound : passiveBounded (fun t => policy (encode (trajectory t))) signal passive
  action_sensitive : actionSensitive response

/-- **Joint language-life admission.**  One shared encoded state satisfies all
four bases: causal compression, selective writing, predictive growth and the
active closure. -/
def joint_admission_spec
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageCandidate Hist Model Ctx Act Obs Byte Answer) : Prop :=
    (∃ (read : Memory Model L.reserve → Answer)
       (next : Memory Model L.reserve → Byte → Memory Model L.reserve),
      (∀ h, read (L.encode h) = L.required h) ∧
      (∀ h b, next (L.encode h) b = L.encode (L.advanceHistory h b))) ∧
    (∀ h b,
      ((∀ x ∈ (L.encode h).candidates,
          L.possible x (L.context h b) (L.action h b) (L.outcome h b) = true) →
        CausalLearning.advance L.possible (L.encode h)
          (L.context h b) (L.action h b) (L.outcome h b) = L.encode h ∧
        CausalLearning.fee L.possible (L.encode h)
          (L.context h b) (L.action h b) (L.outcome h b) = 0) ∧
      (0 < (L.encode h).reserve.val →
        ∀ x, x ∈ (L.encode h).candidates →
          L.possible x (L.context h b) (L.action h b) (L.outcome h b) = false →
          x ∉ (CausalLearning.advance L.possible (L.encode h)
            (L.context h b) (L.action h b) (L.outcome h b)).candidates ∧
          CausalLearning.fee L.possible (L.encode h)
            (L.context h b) (L.action h b) (L.outcome h b) = 1)) ∧
    (∀ h,
      PredictiveHarvest.growth (L.stream h) (L.allocation (L.encode h)) =
          Real.log (Fintype.card Byte) -
            PredictiveHarvest.crossEntropy (L.stream h) (L.allocation (L.encode h)) ∧
      PredictiveHarvest.growth (L.stream h) (L.allocation (L.encode h)) ≤
          Real.log (Fintype.card Byte) -
            PredictiveHarvest.entropy (L.stream h)) ∧
    ((¬ eventuallyPassive (fun t => L.policy (L.encode (L.trajectory t))) L.passive) ∧
      ¬ (∀ a b, L.predict b = L.response a b)) ∧
    (∀ h b,
      L.next (L.encode h) b =
        CausalLearning.advance L.possible (L.encode h)
          (L.context h b) (L.action h b) (L.outcome h b))

theorem joint_admission
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageCandidate Hist Model Ctx Act Obs Byte Answer) :
    joint_admission_spec L := by
  have hcausal :
      (∃ (read : Memory Model L.reserve → Answer)
         (next : Memory Model L.reserve → Byte → Memory Model L.reserve),
        (∀ h, read (L.encode h) = L.required h) ∧
        (∀ h b, next (L.encode h) b = L.encode (L.advanceHistory h b))) :=
    ⟨L.read, L.next, L.read_correct, L.next_correct⟩
  have hactive := active_closure
    (policy := fun t => L.policy (L.encode (L.trajectory t)))
    (signal := L.signal) (a₀ := L.passive)
    (response := L.response) (predict := L.predict)
    L.viable_signal L.passive_bound L.action_sensitive
  refine ⟨hcausal, ?_, ?_, hactive, L.write_refines⟩
  · intro h b
    exact selective_write_contract L.possible (L.encode h)
      (L.context h b) (L.action h b) (L.outcome h b)
  · intro h
    exact predictive_growth_contract (L.stream h) (L.allocation (L.encode h))
      (L.stream_nonneg h) (L.stream_norm h)
      (L.allocation_pos (L.encode h)) (L.allocation_norm (L.encode h))

end CGB.LanguageLifeAdmission
