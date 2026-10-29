# GenerationNesting

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean) · [返回导航](../README.md)

模块源码说明：Nesting: the world is a medium too (generation chain, G26) An organism is a medium; the world it lives in obeys the same axioms, and so does a body that hosts organisms (mitochondria, gut bacteria) or a collective made o

## CGB.Generation.cell_potential

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L39)

解释：建立空间网格单元占有率与感受化学势的对数比值精确解析关系（定理G26a）。

源码说明：**G26a (the sensed potential of a world cell)**.

形式陈述：

```lean
∀ (μ : ℝ),
  CGB.Generation.occupancy μ / (1 - CGB.Generation.occupancy μ) = Real.exp μ ∧
    Real.log (CGB.Generation.occupancy μ / (1 - CGB.Generation.occupancy μ)) = μ
```

## CGB.Generation.site_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L45)

解释：在受限再生率下，单点在T拍后的残余养分加上已被捕食总养分不超过初值加上总再生量。

源码说明：Conservation at a site: food after `T` ticks plus everything eaten is at most the initial food plus the regeneration.

形式陈述：

```lean
∀ (f e g : ℕ → ℝ) (r : ℝ),
  (∀ (t : ℕ), f (t + 1) = f t - e t + g t) →
    (∀ (t : ℕ), g t ≤ r) → ∀ (T : ℕ), f T + ∑ t ∈ Finset.range T, e t ≤ f 0 + r * ↑T
```

## CGB.Generation.stay_starves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L58)

解释：若机体停留在有限再生单点且维持消耗大于再生速率，其体内能量储备在T拍后必呈线性下界衰减（定理G26b）。

源码说明：**G26b (staying put starves)**: with food never negative, regeneration at most `r`, and a reserve that gains at most what is eaten and pays `c` per tick, the reserve after `T` ticks is at most `R₀ + f₀ + (r - c) T`.

形式陈述：

```lean
∀ (f e g R : ℕ → ℝ) (r c : ℝ),
  (∀ (t : ℕ), f (t + 1) = f t - e t + g t) →
    (∀ (t : ℕ), g t ≤ r) →
      (∀ (t : ℕ), 0 ≤ f t) → (∀ (t : ℕ), R (t + 1) ≤ R t + e t - c) → ∀ (T : ℕ), R T ≤ R 0 + f 0 + (r - c) * ↑T
```

## CGB.Generation.stay_dies

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L74)

解释：当维持生存的基础能耗大于单点养分补给极限时，原地不动的生命机体必在有限时间内耗尽储备而死亡。

源码说明：The reserve is exhausted in finite time when living costs more than the site regenerates.

形式陈述：

```lean
∀ (f e g R : ℕ → ℝ) (r c : ℝ),
  (∀ (t : ℕ), f (t + 1) = f t - e t + g t) →
    (∀ (t : ℕ), g t ≤ r) → (∀ (t : ℕ), 0 ≤ f t) → (∀ (t : ℕ), R (t + 1) ≤ R t + e t - c) → r < c → ∃ T, R T < 0
```

## CGB.Generation.cell_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L91)

解释：导出容量为V的单元中增加一个微观量子所需的精确自由能代价为log((n+1)/(V-n))。

源码说明：Adding one quantum to a cell holding `n` of `V` costs exactly `log ((n + 1) / (V - n))`, so to a unit exchanging quanta with it the cell is a reservoir of potential `≈ log (n / V) = ln c`, and one exchange moves that potential only by `≈ 1/n`: the `PairCRN` reservoir and the receptors' `ln c` are the limit of exchanging with a large world cell (with G26a).

形式陈述：

```lean
∀ (V n : ℕ), n < V → CGB.Generation.cellF V (n + 1) - CGB.Generation.cellF V n = Real.log ((↑n + 1) / (↑V - ↑n))
```

## CGB.Generation.missProb_ge

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L118)

解释：给出当当前状态存在具有不同边界读数的后继态时预测误差率的正下界估计。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {W : Type u_1} {R : Type u_2} [inst : Fintype W] [inst_1 : DecidableEq R] (K : W → W → ℝ),
  (∀ (w w' : W), 0 ≤ K w w') →
    ∀ (r g : W → R) (x : W) {y₁ y₂ : W}, r y₁ ≠ r y₂ → min (K x y₁) (K x y₂) ≤ CGB.Generation.missProb K r g x
```

## CGB.Generation.stream_never_dry

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L140)

解释：在平稳遍历环境下若存在环境边界分歧，机体对下一刻观测的预测累积误差必随时间T呈线性下界增长（定理T-N7）。

源码说明：**T-N7 (the stream never runs dry)**: under a stationary law `π` of the world–organism process, if some state `x` (weight `π x`) has two possible successors with different boundary readings, then every predictor of the next reading from the full current state errs with probability at least `π x · min (K x y₁) (K x y₂) > 0` at every tick, so its expected number of errors over `T` ticks grows at least linearly in `T`.

形式陈述：

```lean
∀ {W : Type u_1} {R : Type u_2} [inst : Fintype W] [inst_1 : DecidableEq R] (K : W → W → ℝ),
  (∀ (w w' : W), 0 ≤ K w w') →
    ∀ (π : W → ℝ),
      (∀ (w : W), 0 ≤ π w) →
        ∀ (r g : W → R) (x : W) {y₁ y₂ : W},
          r y₁ ≠ r y₂ →
            ∀ (T : ℕ),
              ↑T * (π x * min (K x y₁) (K x y₂)) ≤ ∑ _t ∈ Finset.range T, ∑ w, π w * CGB.Generation.missProb K r g w
```

## CGB.Generation.corpses_must_go

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L161)

解释：在有限空间中若持续产生死亡个体，则必须存在尸体分解与清除机制以防止空间饱和终止流动（定理G26c）。

源码说明：**G26c (corpses must go: decomposition is necessary)**: in a world with finitely many places, if new deaths never stop (an endless stream of mortal lives: at every time some place that holds no corpse yet will later hold one), the set of places holding corpses cannot only grow: at some time a corpse is removed. A finite world whose dead bodies are never taken apart fills with them, and its stream ends. Who removes it is not the dead body (`NestedWorld.dead_output_is_physics`): physics (a leak) or another organism (a decomposer).

形式陈述：

```lean
∀ {U : Type u_3} [Fintype U] [DecidableEq U] (dead : ℕ → Finset U),
  (∀ (t : ℕ), ∃ t' > t, ∃ u ∉ dead t, u ∈ dead t') → ∃ t, ¬dead t ⊆ dead (t + 1)
```
