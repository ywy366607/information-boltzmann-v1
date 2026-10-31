# GenerationInformationEnergy

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean) · [返回导航](../README.md)

模块源码说明：Information and energy for a finite body in a stream (generation chain, 2026-10-28) Selected mathematical identities and conditional bounds for the information economics map.

## CGB.InformationEnergy.freeEnergy_eq_kl

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L53)

解释：有限状态下，按给定平衡权重定义的自由能恰等于分布相对于该权重的 KL 表达式。

源码说明：**N1**: the free energy in the landscape `-log π` is the relative entropy to `π`.

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (π q : S → ℝ), CGB.InformationEnergy.freeEnergy π q = CGB.InformationEnergy.kl q π
```

## CGB.InformationEnergy.freeEnergy_at_law

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L59)

解释：在该自由能定义下，将当前分布取为参考分布时自由能为零。

源码说明：The law itself has free energy zero.

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (π : S → ℝ), CGB.InformationEnergy.freeEnergy π π = 0
```

## CGB.InformationEnergy.kl_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L65)

解释：当 q 非负归一且 π 严格正并归一时，其有限状态 KL 散度非负。

源码说明：**N1**: free energy is never below that of the law (Gibbs' inequality).

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (q π : S → ℝ),
  (∀ (s : S), 0 ≤ q s) → ∑ s, q s = 1 → (∀ (s : S), 0 < π s) → ∑ s, π s = 1 → 0 ≤ CGB.InformationEnergy.kl q π
```

## CGB.InformationEnergy.snapshot_misses_hidden

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L94)

解释：在所定义的联合分布和只读取可见变量的测量核中，测量对联合状态的信息等于其对可见变量的信息。

源码说明：**N2**: a measurement that reads only the visible part carries no information about the hidden part beyond what it carries about the visible one.

形式陈述：

```lean
∀ {X : Type u_1} {Y : Type u_2} {M : Type u_3} [inst : Fintype X] [inst_1 : Fintype Y] [inst_2 : Fintype M]
  (p : X → Y → ℝ) (k : X → M → ℝ), CGB.InformationEnergy.infoJoint p k = CGB.InformationEnergy.infoVisible p k
```

## CGB.InformationEnergy.finite_memory_bounds_net_work

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L114)

解释：如果提取功上限为 kT·I 且超出容量的擦除成本至少为 kT·(I−cap)，净功不超过 kT·cap。

源码说明：**N3**: work drawn from acquired information `I` at most `kT · I`, erasure of everything beyond the capacity at least `kT` per unit: the net never exceeds `kT · cap`.

形式陈述：

```lean
∀ (W D I cap kT : ℝ), 0 ≤ kT → W ≤ kT * I → kT * (I - cap) ≤ D → W - D ≤ kT * cap
```

## CGB.InformationEnergy.passivity_ends

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L120)

解释：在正漂移的指定方差递推和正噪声条件下，对数表达式最终超过任意给定正数；这本身不决定是否行动。

源码说明：**N4**: with drift `q > 0` and no observation (W2's recursion at gain zero), the information of one observation, `½ log (1 + V / r)`, exceeds any cost `c > 0` after finitely many ticks.

形式陈述：

```lean
∀ (V : ℕ → ℝ) (q r c : ℝ),
  0 < q → 0 < r → 0 < c → 0 ≤ V 0 → (∀ (t : ℕ), V (t + 1) = V t + q) → ∃ t, c < Real.log (1 + V t / r) / 2
```

## CGB.InformationEnergy.mixture_regret_le

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L144)

解释：正先验权重的有限混合模型，对任一正概率假设的负对数损失至多比该假设多 log(1/w) 。

源码说明：**N5 (mixture regret)**: the Bayes mixture codes the sequence at most `log (1 / w θ₀)` worse than the source's law `θ₀` (pathwise, for every sequence and length). `P θ` is the probability the hypothesis `θ` gives to the whole sequence.

形式陈述：

```lean
∀ {Θ : Type u_1} [inst : Fintype Θ] (w P : Θ → ℝ),
  (∀ (θ : Θ), 0 < w θ) →
    (∀ (θ : Θ), 0 ≤ P θ) → ∀ (θ₀ : Θ), 0 < P θ₀ → -Real.log (∑ θ, w θ * P θ) ≤ -Real.log (P θ₀) + Real.log (1 / w θ₀)
```
