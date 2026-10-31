# GenerationSeedMemory

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean) · [返回导航](../README.md)

模块源码说明：An actual memory world for the unchanged physical seed The first quiet tick moves the source particle and independently selects a quiet or food world.

## CGB.SeedMemory.clocked

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L53)

源码说明：The existing structural event bound admits this world's exact clock.

形式陈述：

```lean
CGB.Models.seed.Clocked CGB.SeedMemory.world.clock
```

## CGB.SeedMemory.law_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L57)

源码说明：Collapse only the initial Dirac law, without enumerating body histories.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I)
  (sched : ℕ → W.W → U → I) (y : U → Λ) (w' : W.W),
  M.law W sched 1 y w' = W.moves W.start M.init w' * M.kernel W.clock (sched 0 W.start) M.init y
```

## CGB.SeedMemory.first_hop_act

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L76)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.seed.act (Sum.inl (0, 1)) CGB.Models.seed.calm CGB.Models.seed.init = CGB.SeedMemory.beforeWrite
```

## CGB.SeedMemory.first_hop_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L81)

源码说明：A registered positive-rate hop yields a positive actual first transition.

形式陈述：

```lean
0 <
  CGB.Models.seed.kernel CGB.SeedMemory.world.clock CGB.Models.seed.calm CGB.Models.seed.init CGB.SeedMemory.beforeWrite
```

## CGB.SeedMemory.prefix_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L90)

源码说明：Both input tags have positive probability at the very same reachable body.

形式陈述：

```lean
∀ (b : Bool),
  0 <
    CGB.Models.seed.law CGB.SeedMemory.world (fun x => CGB.SeedMemory.world.sense) 1 CGB.SeedMemory.beforeWrite (some b)
```

## CGB.SeedMemory.write_act

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L97)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.seed.act (Sum.inr true) CGB.Models.food CGB.SeedMemory.beforeWrite = CGB.SeedMemory.afterWrite
```

## CGB.SeedMemory.write_totals

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L102)

源码说明：Food really increases total from one to two through the existing source.

形式陈述：

```lean
CGB.Models.seed.total CGB.SeedMemory.beforeWrite = 1 ∧ CGB.Models.seed.total CGB.SeedMemory.afterWrite = 2
```

## CGB.SeedMemory.write_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L106)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
0 <
  CGB.Models.seed.kernel CGB.SeedMemory.world.clock (CGB.SeedMemory.world.sense (some true)) CGB.SeedMemory.beforeWrite
    CGB.SeedMemory.afterWrite
```

## CGB.SeedMemory.quiet_preserves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L116)

源码说明：Positive quiet transitions preserve the readout even though hops continue.

形式陈述：

```lean
∀ (x y : Fin 4 → Bool),
  0 < CGB.Models.seed.kernel CGB.SeedMemory.world.clock CGB.SeedMemory.world.quiet x y →
    CGB.SeedMemory.observe y = CGB.SeedMemory.observe x
```

## CGB.SeedMemory.remembers

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L125)

源码说明：The original full-body CMI memory clause, derived from actual positive prefixes and source intake, with no assumed positive-information premise.

形式陈述：

```lean
CGB.Models.seed.Remembers CGB.SeedMemory.world
```

## CGB.SeedMemory.quiet_has_nontrivial_event

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L135)

源码说明：Quiet is not an identity dynamics: a positive-rate event changes the body.

形式陈述：

```lean
∃ a x, 0 < CGB.Models.seed.rate a CGB.SeedMemory.world.quiet x ∧ CGB.Models.seed.act a CGB.SeedMemory.world.quiet x ≠ x
```

## CGB.SeedMemory.partial_has_hop

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L142)

源码说明：Finite Boolean graph fact only: every partially occupied ring has an occupied-to-vacant contact. No real-rate decision procedure is involved.

形式陈述：

```lean
∀ (x : Fin 4 → Bool),
  0 < CGB.Models.seed.total x →
    CGB.Models.seed.total x < 4 → ∃ u v, CGB.Models.ringAdj u v ∧ u ≠ v ∧ x u = true ∧ x v = false
```

## CGB.SeedMemory.partial_rate_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L153)

源码说明：Every partial microstate has a state-changing rate-one hop under every stimulus, including quiet. The rates and seed state space are unchanged.

形式陈述：

```lean
∀ (x : Fin 4 → Bool),
  0 < CGB.Models.seed.total x →
    CGB.Models.seed.total x < 4 →
      ∀ (s : Fin 4 → Bool), ∃ a, CGB.Models.seed.act a s x ≠ x ∧ CGB.Models.seed.rate a s x = 1
```

## CGB.SeedMemory.partial_hold_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L169)

源码说明：The probability of never leaving a partial microstate has a geometric bound, while the coarse memory remains. This is not a bound on return visits.

形式陈述：

```lean
∀ (x : Fin 4 → Bool),
  0 < CGB.Models.seed.total x →
    CGB.Models.seed.total x < 4 →
      ∀ (sched : ℕ → Fin 4 → Bool) (n : ℕ),
        ∏ t ∈ Finset.range n, CGB.Models.seed.kernel CGB.SeedMemory.world.clock (sched t) x x ≤ (1 - 1 / 18) ^ n
```

## CGB.SeedMemory.physical_memory_component

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L181)

源码说明：One unchanged admitted physical medium jointly has actual memory and nontrivial quiet events in this world. No `Lives` conclusion is asserted.

形式陈述：

```lean
CGB.Models.seed.Axioms ∧
  CGB.Models.seed.Clocked CGB.SeedMemory.world.clock ∧
    CGB.Models.seed.Remembers CGB.SeedMemory.world ∧
      ∃ a x,
        0 < CGB.Models.seed.rate a CGB.SeedMemory.world.quiet x ∧ CGB.Models.seed.act a CGB.SeedMemory.world.quiet x ≠ x
```
