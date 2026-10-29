# GenerationGrowth

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean) · [返回导航](../README.md)

模块源码说明：Growth of a connection unit (generation chain, G18) A connection unit sits between two compartments and carries its two end activities `a b : Fin 2` and its slot occupancy `w`.

## CGB.Generation.binary_pair_decomposition

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L25)

解释：任意两个二值变量上的实函数可精确拆成常数、两个单变量项和一个乘积交互项。

源码说明：**G18a**: the Möbius decomposition.

形式陈述：

```lean
∀ (f : Fin 2 → Fin 2 → ℝ) (a b : Fin 2),
  f a b = f 0 0 + (f 1 0 - f 0 0) * ↑↑a + (f 0 1 - f 0 0) * ↑↑b + CGB.Generation.interaction f * ↑↑a * ↑↑b
```

## CGB.Generation.interaction_unique

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L32)

解释：若二值对函数已经写成常数、两个单变量项和乘积项，其乘积系数必等于四个函数值的混合差分。

源码说明：**G18a (uniqueness)**: if `f` is written with unary parts and a product coefficient `κ`, then `κ` is the interaction coefficient.

形式陈述：

```lean
∀ (f : Fin 2 → Fin 2 → ℝ) (c u v κ : ℝ),
  (∀ (a b : Fin 2), f a b = c + u * ↑↑a + v * ↑↑b + κ * ↑↑a * ↑↑b) → κ = CGB.Generation.interaction f
```

## CGB.Generation.slot_reciprocity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L41)

解释：二值槽位与二值共同活动的能量混合差分与求差次序无关。

源码说明：**G18b (reciprocity)**: for an energy `E w x` of a slot `w` and a co-activity `x`, the effect of `x` on the slot's flip equals the effect of the slot on `x`'s flip.

形式陈述：

```lean
∀ (E : Fin 2 → Fin 2 → ℝ), E 1 1 - E 0 1 - (E 1 0 - E 0 0) = E 1 1 - E 1 0 - (E 0 1 - E 0 0)
```

## CGB.Generation.occupancy_odds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L50)

解释：给定 logistic 占用函数的占用几率恰为 exp(h)。

源码说明：The detailed-balanced build/dismantle ratio fixes the occupancy: `occ / (1 - occ) = exp h`.

形式陈述：

```lean
∀ (h : ℝ), CGB.Generation.occupancy h / (1 - CGB.Generation.occupancy h) = Real.exp h
```

## CGB.Generation.occupancy_strictMono

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L60)

解释：给定 logistic 占用函数随场 h 严格单调增加。

源码说明：**G18c**: occupancy is strictly increasing in the field.

形式陈述：

```lean
StrictMono CGB.Generation.occupancy
```

## CGB.Generation.one_pool_nested

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L77)

解释：在指定共享势模型和非负耦合下，一个单位与某方向达到强连接阈值时，活动率更高的单位也达到该阈值。

源码说明：**G19a (one pool grows a rich club)**: with one shared pool, the strong set is an upper set in the rates: if `i` is strongly connected to `j` and `k` fires at least as much as `i`, then `k` is too. Neighbourhoods are nested (a threshold graph).

形式陈述：

```lean
∀ {κ θ τ : ℝ},
  0 ≤ κ →
    ∀ {ri rk rj : ℝ},
      0 ≤ rj → ri ≤ rk → τ ≤ CGB.Generation.occupancy (κ * ri * rj - θ) → τ ≤ CGB.Generation.occupancy (κ * rk * rj - θ)
```

## CGB.Generation.memoryless_coactivity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L92)

解释：若连接端状态只依赖当前符号，给定联合分布下的共同活动统计只依赖当前符号的边缘频率。

源码说明：**G19b (memoryless ends see only symbol frequencies)**: if every end state depends on the current symbol only, the co-activity, hence the grown structure, depends on the stream only through the frequencies of the current symbol.

形式陈述：

```lean
∀ {B : Type u_1} {ι : Type u_2} [inst : Fintype B] (P : B → B → ℝ) (ψ : ι → B → ℝ) (i j : ι),
  CGB.Generation.coactivity P (fun k x b => ψ k b) i j = ∑ b, (∑ a, P a b) * (ψ i b * ψ j b)
```

## CGB.Generation.memoryless_same_frequencies_same_structure

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L99)

解释：若两个符号对分布的当前符号边缘频率相同，则无历史记忆的端点得到相同共同活动统计。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {B : Type u_1} {ι : Type u_2} [inst : Fintype B] (P Q : B → B → ℝ),
  (∀ (b : B), ∑ a, P a b = ∑ a, Q a b) →
    ∀ (ψ : ι → B → ℝ) (i j : ι),
      CGB.Generation.coactivity P (fun k x b => ψ k b) i j = CGB.Generation.coactivity Q (fun k x b => ψ k b) i j
```

## CGB.Generation.memory_separates

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L113)

解释：构造的交替与重复二值流具有相同当前符号频率，但能被包含上一符号的共同活动统计区分。

源码说明：**G19c (sequence structure needs memory at the ends)**: the two streams have equal frequencies, so memoryless ends grow the same structure (G19b); an end that holds the previous symbol (`i` active on the previous 1, `j` on the current 1) separates them.

形式陈述：

```lean
(∀ (b : Fin 2), ∑ a, CGB.Generation.alternating a b = ∑ a, CGB.Generation.repeating a b) ∧
  CGB.Generation.coactivity CGB.Generation.alternating (fun k a b => if k = 0 then ↑↑a else ↑↑b) 0 1 ≠
    CGB.Generation.coactivity CGB.Generation.repeating (fun k a b => if k = 0 then ↑↑a else ↑↑b) 0 1
```

## CGB.Generation.two_state_geometric

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L137)

解释：两态递推中上转与下转率之和非零时，偏离固定点的量每步乘以同一个因子，得到几何形式。

源码说明：**G20a (two-state memory is geometric)**: the deviation from the fixed point `p* = kp / (kp + km)` shrinks by the same factor `1 - kp - km` every tick, so the trace of a past input is exponential; detailed balance fixes `p*`, not the form.

形式陈述：

```lean
∀ (kp km p₀ : ℝ),
  kp + km ≠ 0 →
    ∀ (t : ℕ), CGB.Generation.twoStateIter kp km p₀ t - kp / (kp + km) = (1 - kp - km) ^ t * (p₀ - kp / (kp + km))
```

## CGB.Generation.symmetric_exchange

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L152)

解释：两个量按指定对称交换更新后，总量保持不变，差值乘以 1−2D。

源码说明：**G20b (symmetric exchange)**: two compartments exchanging content at the same rate `D` both ways (G6DB) keep their total and their difference shrinks by `1 - 2D`.

形式陈述：

```lean
∀ (D x y : ℝ),
  have x' := x - D * (x - y);
  have y' := y + D * (x - y);
  x' + y' = x + y ∧ x' - y' = (1 - 2 * D) * (x - y)
```

## CGB.Generation.symmetric_stays_symmetric

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L162)

解释：若确定性更新与重标记交换且初始状态对重标记不变，则任意有限次更新后仍保持该对称性。

源码说明：**G21 (no differentiation without fluctuation)**: deterministic dynamics `f` that commute with a relabelling `σ` (P2c) keep a `σ`-symmetric birth state (P3) symmetric forever. Identical units born identically and driven identically stay identical; their differentiation needs fluctuations, i.e. the stochastic medium, not its mean field.

形式陈述：

```lean
∀ {X : Type u_1} (f σ : X → X),
  (∀ (x : X), f (σ x) = σ (f x)) → ∀ (x₀ : X), σ x₀ = x₀ → ∀ (t : ℕ), σ (f^[t] x₀) = f^[t] x₀
```

## CGB.Generation.fixed_point_is_occupancy

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L175)

解释：两个正转移率的比例固定点 kp/(kp+km) 等于对数比 log kp−log km 的 logistic 占用值。

源码说明：The fixed point of G20a is the occupancy of the log odds `log kp - log km`.

形式陈述：

```lean
∀ {kp km : ℝ}, 0 < kp → 0 < km → kp / (kp + km) = CGB.Generation.occupancy (Real.log kp - Real.log km)
```

## CGB.Generation.available_strictAnti

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L183)

解释：固定正供给下，正需求增加会使可用比例 kp/(kp+m) 严格下降。

源码说明：**G22a (use depletes)**: the available fraction strictly falls as demand rises.

形式陈述：

```lean
∀ {kp : ℝ}, 0 < kp → ∀ {m₁ m₂ : ℝ}, 0 < m₁ → m₁ < m₂ → kp / (kp + m₂) < kp / (kp + m₁)
```

## CGB.Generation.inhibitory_slot_antitone

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L189)

解释：耦合系数为负时，指定 logistic 占用率随共同活动强度严格下降。

源码说明：**G22b (equilibrium inhibition retreats from co-activity)**: a slot species whose energy raises co-activity's cost (`κ < 0`) has occupancy strictly falling in co-activity.

形式陈述：

```lean
∀ {κ θ : ℝ}, κ < 0 → StrictAnti fun c => CGB.Generation.occupancy (κ * c - θ)
```

## CGB.Generation.shared_potential_forces_nesting

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L196)

解释：在该共享势、非负耦合模型中，若较低活动单位达到强连接阈值而较高活动单位未达到，则产生矛盾。

源码说明：**G22c (no rich club needs local potentials)**: if all compartments share one potential and the strong set is not nested in rate order, contradiction (G19a). So a medium whose grown structure is not a rate-ordered threshold graph has compartment-local potentials.

形式陈述：

```lean
∀ {κ θ τ : ℝ},
  0 ≤ κ →
    ∀ {ri rk rj : ℝ},
      0 ≤ rj →
        ri ≤ rk →
          τ ≤ CGB.Generation.occupancy (κ * ri * rj - θ) → CGB.Generation.occupancy (κ * rk * rj - θ) < τ → False
```
