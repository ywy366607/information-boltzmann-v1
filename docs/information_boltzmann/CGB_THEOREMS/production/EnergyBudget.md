# EnergyBudget

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean) · [返回导航](../README.md)

模块源码说明：Energy balance and conditional mean availability budget Physical energy, work/heat ports, temperature and entropy require a carrier interpretation.

## CGB.Energetics.law_probability

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L28)

解释：交互机器从指定初态演化得到的每时刻状态分布非负且总和为一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype O]
  (P : CGB.Interaction.Machine S ℕ O) (initial : S) (t : ℕ),
  (∀ (s : S), 0 ≤ CGB.Energetics.law P initial t s) ∧ ∑ s, CGB.Energetics.law P initial t s = 1
```

## CGB.Energetics.expected_first_law

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L66)

解释：若每个受支持事件满足声明的能量收支，则可沿原过程对其取期望得到平均能量收支。

源码说明：Average the supported first law along the ORIGINAL process.

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype O]
  (P : CGB.Interaction.Machine S ℕ O) (a : CGB.Energetics.Account P) (initial : S) (t : ℕ),
  CGB.Energetics.mean (CGB.Energetics.law P initial (t + 1)) (a.stored (t + 1)) +
      CGB.Energetics.flux P initial a.load t =
    CGB.Energetics.mean (CGB.Energetics.law P initial t) (a.stored t) + CGB.Energetics.flux P initial a.drive t +
        CGB.Energetics.flux P initial a.matter t +
      CGB.Energetics.flux P initial a.heat t
```

## CGB.Energetics.availability_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L118)

解释：每拍平均自由能增量加输出功和 T 倍熵产，等于输入的可用量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype O]
  (P : CGB.Interaction.Machine S ℕ O) (a : CGB.Energetics.Account P) (initial : S) (T : ℝ) (entropy : ℕ → (S → ℝ) → ℝ)
  (carriedEntropy : ℕ → S → O → S → ℝ),
  0 < T →
    ∀ (t : ℕ),
      CGB.Energetics.freeEnergy P a initial T entropy (t + 1) - CGB.Energetics.freeEnergy P a initial T entropy t +
            CGB.Energetics.flux P initial a.load t +
          T * CGB.Energetics.production P a initial T entropy carriedEntropy t =
        CGB.Energetics.suppliedAvailability P a initial T carriedEntropy t
```

## CGB.Energetics.work_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L133)

解释：在给定可用量收支和储备下界等条件下，有限时段的输出功受供应与储备变化约束。

源码说明：One finite-prefix identity gives both the bounded-reserve work budget and the cyclic bound. A lower free-energy reserve and mean second law are explicit premises; no uniform per-tick loss or universal death condition is added.

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype O]
  (P : CGB.Interaction.Machine S ℕ O) (a : CGB.Energetics.Account P) (initial : S) (T : ℝ) (entropy : ℕ → (S → ℝ) → ℝ)
  (carriedEntropy : ℕ → S → O → S → ℝ),
  CGB.Energetics.Isothermal P a initial T entropy carriedEntropy →
    ∀ (n : ℕ),
      CGB.Energetics.freeEnergy P a initial T entropy n + ∑ t ∈ Finset.range n, CGB.Energetics.flux P initial a.load t +
            T * ∑ t ∈ Finset.range n, CGB.Energetics.production P a initial T entropy carriedEntropy t =
          CGB.Energetics.freeEnergy P a initial T entropy 0 +
            ∑ t ∈ Finset.range n, CGB.Energetics.suppliedAvailability P a initial T carriedEntropy t ∧
        (∀ lower ≤ CGB.Energetics.freeEnergy P a initial T entropy n,
            ∑ t ∈ Finset.range n, CGB.Energetics.flux P initial a.load t ≤
              CGB.Energetics.freeEnergy P a initial T entropy 0 - lower +
                ∑ t ∈ Finset.range n, CGB.Energetics.suppliedAvailability P a initial T carriedEntropy t) ∧
          (CGB.Energetics.freeEnergy P a initial T entropy n = CGB.Energetics.freeEnergy P a initial T entropy 0 →
            ∑ t ∈ Finset.range n, CGB.Energetics.flux P initial a.load t ≤
              ∑ t ∈ Finset.range n, CGB.Energetics.suppliedAvailability P a initial T carriedEntropy t)
```

## CGB.Energetics.no_sustained_use_on_finite_supply

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L169)

解释：在定理给定的正平均消耗、有限总供给与有界储备条件下，指定持续消耗不可能无限维持。

源码说明：A positive mean availability-use requirement (delivered work plus T times production) cannot persist on bounded total supply and reserve. This requirement is an explicit property of the mechanism, not a newly imposed maintenance cost for all living systems.

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype O]
  (P : CGB.Interaction.Machine S ℕ O) (a : CGB.Energetics.Account P) (initial : S) (T : ℝ) (entropy : ℕ → (S → ℝ) → ℝ)
  (carriedEntropy : ℕ → S → O → S → ℝ),
  CGB.Energetics.Isothermal P a initial T entropy carriedEntropy →
    ∀ (lower supplyBound ε : ℝ),
      0 < ε →
        (∀ (n : ℕ), lower ≤ CGB.Energetics.freeEnergy P a initial T entropy n) →
          (∀ (n : ℕ),
              ∑ t ∈ Finset.range n, CGB.Energetics.suppliedAvailability P a initial T carriedEntropy t ≤ supplyBound) →
            ¬∀ (n : ℕ),
                ↑n * ε ≤
                  ∑ t ∈ Finset.range n,
                    (CGB.Energetics.flux P initial a.load t +
                      T * CGB.Energetics.production P a initial T entropy carriedEntropy t)
```
