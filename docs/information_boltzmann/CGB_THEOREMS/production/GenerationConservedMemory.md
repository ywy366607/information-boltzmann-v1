# GenerationConservedMemory

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean) · [返回导航](../README.md)

模块源码说明：Memory carried by a conserved observable, not a frozen body Generalize the existing quiet-identity memory marginal.

## CGB.ConservedMemory.observable_test_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L29)

源码说明：Any test of a preserved observable has constant conditional expectation under the specified stimulus. The body may still rearrange.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {Obs : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [DecidableEq Obs]
  (M : CGB.Medium U Λ A I) (R : ℝ),
  M.Clocked R →
    ∀ (stimulus : U → I) (observe : (U → Λ) → Obs) (test : Obs → ℝ),
      (∀ (x y : U → Λ), 0 < M.kernel R stimulus x y → observe y = observe x) →
        ∀ (x : U → Λ), ∑ y, test (observe y) * M.kernel R stimulus x y = test (observe x)
```

## CGB.ConservedMemory.observable_indicator_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L46)

源码说明：A stochastic kernel preserving an observable also preserves each of its indicator test functions. The body itself need not stop moving.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {Obs : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [inst_5 : DecidableEq Obs]
  (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  M.Clocked W.clock →
    ∀ (observe : (U → Λ) → Obs),
      (∀ (x y : U → Λ), 0 < M.kernel W.clock W.quiet x y → observe y = observe x) →
        ∀ (x : U → Λ) (r : Obs),
          ∑ y, (if observe y = r then 1 else 0) * M.kernel W.clock W.quiet x y = if observe x = r then 1 else 0
```

## CGB.ConservedMemory.quiet_observable_memory_marginal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L64)

源码说明：Quiet can be nontrivial: each future observable marginal equals the one installed by the actual first observation, even as the world evolves.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {Obs : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [inst_5 : DecidableEq Obs]
  (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  M.Clocked W.clock →
    ∀ (observe : (U → Λ) → Obs),
      (∀ (x y : U → Λ), 0 < M.kernel W.clock W.quiet x y → observe y = observe x) →
        ∀ (n₀ : ℕ) (x₀ : U → Λ) (w₀ : W.W) (r : Obs) (m : ℕ),
          CGB.ConservedMemory.observedMemory M W observe n₀ (m + 1) x₀ w₀ r =
            CGB.ConservedMemory.observedMemory M W observe n₀ 1 x₀ w₀ r
```

## CGB.ConservedMemory.tagged_memory_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L90)

源码说明：The tagged law used by the original memory definition is nonnegative under the actual clock admission, including the observation and quiet steps.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  M.Clocked W.clock → ∀ (n₀ m : ℕ) (x x₀ : U → Λ) (w w₀ : W.W), 0 ≤ M.memLaw W n₀ m x w x₀ w₀
```

## CGB.ConservedMemory.one_write_body_marginal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L106)

源码说明：The first actual write marginalizes the world's normalized successor law; this is not a hand-assigned probability on a supported path.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I)
  (n₀ : ℕ) (x₀ x : U → Λ) (w₀ : W.W),
  ∑ w, M.memLaw W n₀ 1 x w x₀ w₀ = M.law W (fun x => W.sense) n₀ x₀ w₀ * M.kernel W.clock (W.sense w₀) x₀ x
```

## CGB.ConservedMemory.registered_event_kernel_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L117)

源码说明：A positive registered event that changes the body has a positive actual uniformized transition, at the same world clock.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (R : ℝ),
  0 < R →
    ∀ (stimulus : U → I) (a : A) (x : U → Λ),
      0 < M.rate a stimulus x → M.act a stimulus x ≠ x → 0 < M.kernel R stimulus x (M.act a stimulus x)
```

## CGB.ConservedMemory.observed_memory_as_project

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L142)

源码说明：The observable law is exactly the deterministic projection used by finite conditional-information data processing.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {Obs : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [inst_5 : DecidableEq Obs]
  (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I) (observe : (U → Λ) → Obs) (n₀ m : ℕ) (x₀ : U → Λ) (w₀ : W.W) (r : Obs),
  CGB.ConservedMemory.observedMemory M W observe n₀ m x₀ w₀ r =
    CGB.InformationProjection.project observe (fun x => ∑ w, M.memLaw W n₀ m x w x₀ w₀) r
```

## CGB.ConservedMemory.observed_memory_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L152)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {Obs : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [inst_5 : DecidableEq Obs]
  (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  M.Clocked W.clock →
    ∀ (observe : (U → Λ) → Obs) (n₀ m : ℕ) (x₀ : U → Λ) (w₀ : W.W) (r : Obs),
      0 ≤ CGB.ConservedMemory.observedMemory M W observe n₀ m x₀ w₀ r
```

## CGB.ConservedMemory.one_write_observed_row

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L162)

源码说明：A first-write coarse row is its actual prefix mass times the observed transition probability.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {Obs : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [inst_5 : DecidableEq Obs]
  (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I) (observe : (U → Λ) → Obs) (n₀ : ℕ) (x₀ : U → Λ) (w₀ : W.W) (r : Obs),
  CGB.ConservedMemory.observedMemory M W observe n₀ 1 x₀ w₀ r =
    M.law W (fun x => W.sense) n₀ x₀ w₀ * ∑ y, if observe y = r then M.kernel W.clock (W.sense w₀) x₀ y else 0
```

## CGB.ConservedMemory.one_write_observed_mass

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L177)

源码说明：Summing the observed outcomes returns the same actual prefix mass.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {Obs : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [inst_5 : DecidableEq Obs]
  [inst_6 : Fintype Obs] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I) (observe : (U → Λ) → Obs) (n₀ : ℕ) (x₀ : U → Λ)
  (w₀ : W.W), ∑ r, CGB.ConservedMemory.observedMemory M W observe n₀ 1 x₀ w₀ r = M.law W (fun x => W.sense) n₀ x₀ w₀
```

## CGB.ConservedMemory.observed_write_pos_of_transition

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L189)

源码说明：A supported actual transition from a positive prefix gives positive observed write mass, with no assigned path weights.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {Obs : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [inst_5 : DecidableEq Obs]
  (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  M.Clocked W.clock →
    ∀ (observe : (U → Λ) → Obs) (n₀ : ℕ) (x₀ y : U → Λ) (w₀ : W.W),
      0 < M.law W (fun x => W.sense) n₀ x₀ w₀ →
        0 < M.kernel W.clock (W.sense w₀) x₀ y →
          0 < CGB.ConservedMemory.observedMemory M W observe n₀ 1 x₀ w₀ (observe y)
```

## CGB.ConservedMemory.observed_memory_information_le

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L208)

源码说明：Coarse information is a certified lower bound on the original full-body CMI, including zeros in the actual probability support.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {Obs : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [inst_5 : DecidableEq Obs]
  [inst_6 : Fintype Obs] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  M.Clocked W.clock →
    ∀ (observe : (U → Λ) → Obs) (n₀ m : ℕ),
      (CGB.Medium.condMutualInfo fun x₀ w₀ r => CGB.ConservedMemory.observedMemory M W observe n₀ m x₀ w₀ r) ≤
        CGB.Medium.condMutualInfo fun x₀ w₀ x => ∑ w, M.memLaw W n₀ m x w x₀ w₀
```

## CGB.ConservedMemory.remembers_of_conserved_observable

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L227)

源码说明：A positive actual information write into a conserved finite observable implies the original full-body `Remembers`, without freezing the body or assuming a positive CMI at all later times.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {Obs : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [inst_5 : DecidableEq Obs]
  [inst_6 : Fintype Obs] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  M.Clocked W.clock →
    ∀ (observe : (U → Λ) → Obs),
      (∀ (x y : U → Λ), 0 < M.kernel W.clock W.quiet x y → observe y = observe x) →
        ∀ (n₀ : ℕ),
          (0 < CGB.Medium.condMutualInfo fun x₀ w₀ r => CGB.ConservedMemory.observedMemory M W observe n₀ 1 x₀ w₀ r) →
            M.Remembers W
```

## CGB.ConservedMemory.remembers_of_quiet_and_actual_write

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L248)

源码说明：An actual counterfactual write creates positive information: at the same reachable body, one world input is quiet and another has a positive transition into a different observable class. Sparse CMI strictness then replaces any assumed positive-information write or future-memory premise.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {Obs : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [DecidableEq Obs]
  [Fintype Obs] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  M.Clocked W.clock →
    ∀ (observe : (U → Λ) → Obs),
      (∀ (x y : U → Λ), 0 < M.kernel W.clock W.quiet x y → observe y = observe x) →
        ∀ (n₀ : ℕ) (x₀ y : U → Λ) (wquiet wwrite : W.W),
          W.sense wquiet = W.quiet →
            0 < M.law W (fun x => W.sense) n₀ x₀ wquiet →
              0 < M.law W (fun x => W.sense) n₀ x₀ wwrite →
                0 < M.kernel W.clock (W.sense wwrite) x₀ y → observe y ≠ observe x₀ → M.Remembers W
```

## CGB.ConservedMemory.event_observable_kernel

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L280)

源码说明：A local event invariant lifts to the uniformized quiet kernel. This also covers conserving hops which touch a boundary site without importing or exporting content.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {Obs : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [DecidableEq Obs]
  (M : CGB.Medium U Λ A I) (R : ℝ) (stimulus : U → I) (observe : (U → Λ) → Obs),
  (∀ (a : A) (x : U → Λ), 0 < M.rate a stimulus x → observe (M.act a stimulus x) = observe x) →
    ∀ (x y : U → Λ), 0 < M.kernel R stimulus x y → observe y = observe x
```

## CGB.ConservedMemory.quiet_event_observable_kernel

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L304)

源码说明：Quiet specialization of the local event invariant.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {Obs : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [DecidableEq Obs]
  (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I) (observe : (U → Λ) → Obs),
  (∀ (a : A) (x : U → Λ), 0 < M.rate a W.quiet x → observe (M.act a W.quiet x) = observe x) →
    ∀ (x y : U → Λ), 0 < M.kernel W.clock W.quiet x y → observe y = observe x
```

## CGB.ConservedMemory.quiet_total_kernel

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L312)

源码说明：Conservation is applied only to positive quiet events which really are closed, not silently to arbitrary boundary-driven writes.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  M.P2a →
    (∀ (a : A) (x : U → Λ), 0 < M.rate a W.quiet x → M.Closed a) →
      ∀ (x y : U → Λ), 0 < M.kernel W.clock W.quiet x y → M.total y = M.total x
```

## CGB.ConservedMemory.quiet_total_memory_marginal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L321)

源码说明：P2a protects the total-content memory marginal under positive closed quiet exchange. This does not require an identity transition kernel.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  M.Clocked W.clock →
    M.P2a →
      (∀ (a : A) (x : U → Λ), 0 < M.rate a W.quiet x → M.Closed a) →
        ∀ (n₀ : ℕ) (x₀ : U → Λ) (w₀ : W.W) (r m : ℕ),
          CGB.ConservedMemory.observedMemory M W M.total n₀ (m + 1) x₀ w₀ r =
            CGB.ConservedMemory.observedMemory M W M.total n₀ 1 x₀ w₀ r
```

## CGB.ConservedMemory.seed_calm_event_conserves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L333)

源码说明：In the existing four-site physical seed, quiet shuts the two reservoir ports but leaves the hopping dynamics intact. Even boundary-touching hops conserve content. No new rates or states are installed.

形式陈述：

```lean
∀ (a : Fin 4 × Fin 4 ⊕ Bool) (x : Fin 4 → Bool),
  CGB.Models.seed.total (CGB.Models.seed.act a CGB.Models.seed.calm x) = CGB.Models.seed.total x
```

## CGB.ConservedMemory.seed_event_clocked

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L346)

源码说明：A structural finite clock bound: at most eighteen registered events, each of rate at most one. This is not a biological rate calibration.

形式陈述：

```lean
CGB.Models.seed.EventClocked 18
```

## CGB.ConservedMemory.seed_calm_has_nontrivial_event

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L358)

源码说明：The same physical seed really moves at quiet: this is not the old identity-kernel memory shortcut.

形式陈述：

```lean
∃ a x, 0 < CGB.Models.seed.rate a CGB.Models.seed.calm x ∧ CGB.Models.seed.act a CGB.Models.seed.calm x ≠ x
```

## CGB.ConservedMemory.seed_quiet_total_memory_marginal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L370)

源码说明：Content distinctions written through the original seed's reservoir ports have invariant quiet marginals despite its nontrivial hopping. The remaining one-write information/DPI and viability obligations are separate.

形式陈述：

```lean
∀ (W : CGB.World (Fin 4) Bool Bool),
  CGB.Models.seed.Clocked W.clock →
    W.quiet = CGB.Models.seed.calm →
      ∀ (n₀ : ℕ) (x₀ : Fin 4 → Bool) (w₀ : W.W) (r m : ℕ),
        CGB.ConservedMemory.observedMemory CGB.Models.seed W CGB.Models.seed.total n₀ (m + 1) x₀ w₀ r =
          CGB.ConservedMemory.observedMemory CGB.Models.seed W CGB.Models.seed.total n₀ 1 x₀ w₀ r
```

## CGB.ConservedMemory.historical_persistence_requires_alive_change

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L386)

源码说明：Historical persistence requires calm to change the viability observable, not merely to move some microscopic states. The same joint law and clock are used; no alternate interruption protocol is substituted.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  (∀ (x : U → Λ), ∑ y, M.aliveInd W y * M.kernel W.clock M.calm x y = M.aliveInd W x) →
    ∀ (δ : ℝ) (T : ℕ), ¬M.Persists W δ T
```

## CGB.ConservedMemory.conserving_calm_needs_organizational_viability

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L416)

源码说明：If calm conserves total and viability reads *only* total, the original long-disconnection death requirement is impossible. Thus a conserving candidate needs organization-dependent recovery, or a real boundary loss; adding nonzero stirring alone is not a life generator.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  M.Clocked W.clock →
    (∀ (a : A) (x : U → Λ), 0 < M.rate a M.calm x → M.total (M.act a M.calm x) = M.total x) →
      ∀ (score : ℕ → ℝ), (∀ (x : U → Λ), M.aliveInd W x = score (M.total x)) → ∀ (δ : ℝ) (T : ℕ), ¬M.Persists W δ T
```
