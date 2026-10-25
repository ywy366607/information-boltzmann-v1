import CGBAxioms.RecoveryTransport

/-!
# Coarse contracts for separated sensory and resource ports

Complex bodies need not expose every carrier identity. A model may instead
register a factorized external port, a finite reserve rank, and a timed
refinement certificate. The obligations below are deliberately abstract:
they do not create a free resource or assert physical calibration.
-/

namespace CGB.PortContracts

structure PortSplit (I Sensor Resource : Type*) where
  encode : Sensor → Resource → I
  sensor : I → Sensor
  resource : I → Resource
  sensor_encode : ∀ s r, sensor (encode s r) = s
  resource_encode : ∀ s r, resource (encode s r) = r

def protocol {T Sensor Resource I : Type*} (P : PortSplit I Sensor Resource)
    (sensor : T → Sensor) (resource : T → Resource) : T → I :=
  fun t => P.encode (sensor t) (resource t)

theorem protocol_sensor {T Sensor Resource I : Type*} (P : PortSplit I Sensor Resource)
    (sensor : T → Sensor) (resource : T → Resource) (t : T) :
    P.sensor (protocol P sensor resource t) = sensor t :=
  P.sensor_encode _ _

theorem protocol_resource {T Sensor Resource I : Type*} (P : PortSplit I Sensor Resource)
    (sensor : T → Sensor) (resource : T → Resource) (t : T) :
    P.resource (protocol P sensor resource t) = resource t :=
  P.resource_encode _ _

structure RankedReserve (S : Type*) where
  empty : S → Prop
  rank : S → ℕ
  step : S → S
  rank_empty : ∀ s, empty s → rank s = 0
  step_empty : ∀ s, empty s → empty (step s)
  strict_loss : ∀ s, ¬empty s → rank (step s) < rank s

structure CoarseAdmission (I Sensor Resource S : Type*) where
  ports : PortSplit I Sensor Resource
  reserve : RankedReserve S

theorem admission_port_factorizes {I Sensor Resource S : Type*}
    (A : CoarseAdmission I Sensor Resource S) (s : Sensor) (r : Resource) :
    A.ports.sensor (protocol A.ports (fun _ => s) (fun _ => r) 0) = s ∧
    A.ports.resource (protocol A.ports (fun _ => s) (fun _ => r) 0) = r := by
  exact ⟨protocol_sensor _ _ _ _, protocol_resource _ _ _ _⟩

theorem admission_reserve_contract {I Sensor Resource S : Type*}
    (A : CoarseAdmission I Sensor Resource S) :
    (∀ s, A.reserve.empty s → A.reserve.rank s = 0) ∧
    (∀ s, A.reserve.empty s → A.reserve.empty (A.reserve.step s)) ∧
    (∀ s, ¬A.reserve.empty s → A.reserve.rank (A.reserve.step s) < A.reserve.rank s) := by
  exact ⟨A.reserve.rank_empty, A.reserve.step_empty, A.reserve.strict_loss⟩

structure RecoveryRefinement {R S E : Type*} [Fintype R] [DecidableEq R]
    [Fintype S] [DecidableEq S] [Fintype E]
    (P : Interaction.Machine R ℕ E) (Q : Interaction.Machine S ℕ E)
    (project : R → S) (markP : ℕ → R → E → R → Bool)
    (markQ : ℕ → S → E → S → Bool) where
  timed : Interaction.TimedRefinement P Q project 1 1
  mark_preserved : ∀ t r e r', 0 < P.step t r e r' →
    markP t r e r' = markQ t (project r) e (project r')

theorem recovery_preserved {R S E : Type*} [Fintype R] [DecidableEq R]
    [Fintype S] [DecidableEq S] [Fintype E]
    {P : Interaction.Machine R ℕ E} {Q : Interaction.Machine S ℕ E}
    {project : R → S} {markP : ℕ → R → E → R → Bool}
    {markQ : ℕ → S → E → S → Bool}
    (C : RecoveryRefinement P Q project markP markQ)
    (t k : ℕ) (r : R) :
    EventChannels.Kernel.firstHit (Energetics.events P) markP t k r =
      EventChannels.Kernel.firstHit (Energetics.events Q) markQ t k (project r) := by
  apply RecoveryTransport.firstHit_eq C.timed.representation
  exact C.mark_preserved

theorem elapsed_preserved {R S E : Type*} [Fintype R] [DecidableEq R]
    [Fintype S] [DecidableEq S] [Fintype E]
    {P : Interaction.Machine R ℕ E} {Q : Interaction.Machine S ℕ E}
    {project : R → S} {markP : ℕ → R → E → R → Bool}
    {markQ : ℕ → S → E → S → Bool}
    (C : RecoveryRefinement P Q project markP markQ) (n : ℕ) :
    (n : ℝ) / 1 = (n : ℝ) / 1 := by
  exact C.timed.elapsed_eq n

end CGB.PortContracts
