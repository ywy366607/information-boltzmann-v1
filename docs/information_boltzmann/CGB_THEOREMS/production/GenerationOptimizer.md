# GenerationOptimizer

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean) · [返回导航](../README.md)

模块源码说明：Optimizer interfaces forced by the generation chain This file does not classify Adam or Muon.

## CGB.Generation.coordinate_separable_has_no_pairwise_relation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L42)

解释：二值对函数若能拆为常数和两个单变量项，则其混合差分为零，不能表示这里定义的非零双变量关系。

源码说明：Coordinate-separable state cannot carry a nonzero pairwise interaction.

形式陈述：

```lean
∀ {F : Fin 2 → Fin 2 → ℝ}, CGB.Generation.CoordinateSeparable F → ¬CGB.Generation.PairwiseRelationCarrier F
```

## CGB.Generation.pairwise_relation_requires_cross_term

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L55)

解释：二值对函数的混合差分非零时，它不能写成仅有常数与两个单变量项的形式。

源码说明：Equivalently, a nonzero relation forces a cross-term in every unary decomposition. This is the minimal formal statement behind the claim that a diagonal preconditioner cannot represent arbitrary parameter relations.

形式陈述：

```lean
∀ {F : Fin 2 → Fin 2 → ℝ},
  CGB.Generation.PairwiseRelationCarrier F → ∀ (c u v : ℝ), ¬∀ (a b : Fin 2), F a b = c + u * ↑↑a + v * ↑↑b
```

## CGB.Generation.constant_gain_not_local_anti_dominant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L69)

解释：常数增益与局部需求无关，因而不满足需求增加时增益严格下降的定义。

源码说明：A demand-independent gain cannot satisfy local anti-dominance.

形式陈述：

```lean
∀ (k : ℝ), ¬CGB.Generation.LocalAntiDominance fun x x_1 => k
```

## CGB.Generation.reservoir_availability_is_local_anti_dominant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L78)

解释：正供给下，资源可用率 s/(s+m) 随正需求 m 增加而严格下降。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Generation.LocalAntiDominance CGB.Generation.reservoirAvailability
```

## CGB.Generation.reservoir_gain_is_local_anti_dominant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L87)

解释：资源可用率乘正的基础增益后，仍随局部需求严格下降。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {base : ℝ}, 0 < base → CGB.Generation.LocalAntiDominance (CGB.Generation.reservoirGain base)
```

## CGB.Generation.local_anti_dominance_requires_demand_readout

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L95)

解释：若增益满足这里的局部反垄断定义，它不可能对所有需求值保持不变。

源码说明：Any anti-dominant gain must read the compartment's local demand: it cannot be constant in that second argument.

形式陈述：

```lean
∀ {G : ℝ → ℝ → ℝ}, CGB.Generation.LocalAntiDominance G → ¬∀ (s m₁ m₂ : ℝ), G s m₁ = G s m₂
```

## CGB.Generation.anti_dominance_favors_underdemanded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L108)

解释：按局部反垄断定义，在正供给与正需求条件下，需求较小的隔室获得严格较大的增益。

源码说明：Under local anti-dominance, an under-demanded compartment receives strictly higher marginal availability than a dominant compartment with greater past demand. This is the formal rebalancing mechanism that prevents G19a rich-club collapse.

形式陈述：

```lean
∀ {G : ℝ → ℝ → ℝ}, CGB.Generation.LocalAntiDominance G → ∀ {s m₁ m₂ : ℝ}, 0 < s → 0 < m₁ → m₁ < m₂ → G s m₂ < G s m₁
```

## CGB.Generation.shared_global_gain_fails_anti_dominance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L119)

解释：对所有隔室和需求均返回同一个常数的共享增益不满足局部反垄断定义。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (g : ℝ), ¬CGB.Generation.LocalAntiDominance (CGB.Generation.SharedGlobalGain g)
```

## CGB.Generation.separable_and_global_fails_cgb

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L131)

解释：可分离的二值关系函数与共享常数增益组合不满足这里定义的两个候选接口。

源码说明：A coordinate-separable state paired with a shared global gain fails both CGB optimizer requirements.

形式陈述：

```lean
∀ {F : Fin 2 → Fin 2 → ℝ},
  CGB.Generation.CoordinateSeparable F →
    ∀ (g : ℝ), ¬CGB.Generation.CGBOptimizerCandidate F (CGB.Generation.SharedGlobalGain g)
```

## CGB.Generation.cgb_relation_carrier_interaction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L147)

解释：构造的二值乘积关系载体 κab 的混合差分恰为 κ。

源码说明：The CGB carrier has exact interaction `κ`.

形式陈述：

```lean
∀ (κ : ℝ), CGB.Generation.interaction (CGB.Generation.cgbRelationCarrier κ) = κ
```

## CGB.Generation.cgb_carrier_has_relation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L153)

解释：当 κ 非零时，构造的乘积载体具有这里定义的非零双变量关系。

源码说明：When coupling `κ ≠ 0`, the CGB carrier is a genuine pairwise relation carrier.

形式陈述：

```lean
∀ (κ : ℝ), κ ≠ 0 → CGB.Generation.PairwiseRelationCarrier (CGB.Generation.cgbRelationCarrier κ)
```

## CGB.Generation.cgb_optimizer_admitted

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L165)

解释：当关系系数非零、基础增益为正时，构造的关系载体与资源增益同时满足这两个抽象候选接口。

源码说明：The CGB optimizer satisfies both CGB admission requirements.

形式陈述：

```lean
∀ (κ base : ℝ),
  κ ≠ 0 →
    0 < base →
      CGB.Generation.CGBOptimizerCandidate (CGB.Generation.cgbOptimizer κ base).1 (CGB.Generation.cgbOptimizer κ base).2
```

## CGB.Generation.adam_class_has_positive_relational_error

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L178)

解释：当目标混合差分非零时，任何按定义坐标可分离的二值函数都有正关系误差；该类不是完整 Adam 轨迹。

源码说明：Any coordinate-separable (Adam-class) optimizer incurs strictly positive relational error whenever the target coupling is nonzero.

形式陈述：

```lean
∀ {F : Fin 2 → Fin 2 → ℝ},
  CGB.Generation.CoordinateSeparable F → ∀ {κ_target : ℝ}, κ_target ≠ 0 → 0 < CGB.Generation.relationalError F κ_target
```

## CGB.Generation.cgb_optimizer_zero_relational_error

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L190)

解释：把目标耦合系数直接用于构造关系载体时，该代理指标上的关系误差为零。

源码说明：The CGB optimizer achieves zero relational error on the target coupling.

形式陈述：

```lean
∀ (κ_target base : ℝ), CGB.Generation.relationalError (CGB.Generation.cgbOptimizer κ_target base).1 κ_target = 0
```

## CGB.Generation.cgb_strictly_beats_adam_on_relations

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L197)

解释：在目标耦合非零时，将目标系数直接写入的载体在关系误差代理指标上严格优于可分离函数，并未比较完整 Adam 训练。

源码说明：Formal Superiority over Adam: CGB optimizer strictly dominates any coordinate-separable optimizer on relational error.

形式陈述：

```lean
∀ {F_adam : Fin 2 → Fin 2 → ℝ},
  CGB.Generation.CoordinateSeparable F_adam →
    ∀ {κ_target : ℝ},
      κ_target ≠ 0 →
        ∀ (base : ℝ),
          CGB.Generation.relationalError (CGB.Generation.cgbOptimizer κ_target base).1 κ_target <
            CGB.Generation.relationalError F_adam κ_target
```

## CGB.Generation.muon_class_zero_tail_protection

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L217)

解释：该抽象共享常数增益对低、高需求隔室的增益差为零；这不是对完整 Muon 动力学的定理。

源码说明：Any shared global/spectral (Muon-class) optimizer provides exactly zero protective margin between compartments: the dominant and tail compartments compete under identical un-depleted gain.

形式陈述：

```lean
∀ (g s m_tail m_dom : ℝ), CGB.Generation.tailProtectionMargin (CGB.Generation.SharedGlobalGain g) s m_tail m_dom = 0
```

## CGB.Generation.cgb_optimizer_positive_tail_protection

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L225)

解释：在基础增益和供给为正、低需求严格小于高需求时，指定资源增益给低需求隔室的增益差为正。

源码说明：The CGB reservoir optimizer provides a strictly positive protective margin to the under-demanded compartment whenever the dominant compartment has greater demand.

形式陈述：

```lean
∀ (κ base s m_tail m_dom : ℝ),
  0 < base →
    0 < s →
      0 < m_tail →
        m_tail < m_dom → 0 < CGB.Generation.tailProtectionMargin (CGB.Generation.cgbOptimizer κ base).2 s m_tail m_dom
```

## CGB.Generation.cgb_strictly_beats_muon_on_tail_protection

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L236)

解释：在指定正性与需求排序条件下，资源增益的尾部保护差严格大于共享常数增益的零差；该对照不等于完整 Muon。

源码说明：Formal Superiority over Muon: CGB optimizer strictly dominates any shared global gain rule on the tail protection margin, actively suppressing rich-club runaway.

形式陈述：

```lean
∀ (g κ base s m_tail m_dom : ℝ),
  0 < base →
    0 < s →
      0 < m_tail →
        m_tail < m_dom →
          CGB.Generation.tailProtectionMargin (CGB.Generation.SharedGlobalGain g) s m_tail m_dom <
            CGB.Generation.tailProtectionMargin (CGB.Generation.cgbOptimizer κ base).2 s m_tail m_dom
```

## CGB.Generation.cgb_optimizer_pareto_superiority

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L260)

解释：对非零目标耦合和不同正需求，给定构造在关系误差及增益差两项代理指标上严格优于可分离函数加共享常数增益的组合。

源码说明：**Theorem: CGB Optimizer Strictly Pareto-Dominates the Adam-Muon Hybrid**. On any coupled heterogeneous benchmark with nonzero relation `κ* ≠ 0` and competing demands `m_tail < m_dom`, the CGB Hierarchical Reservoir Optimizer strictly outperforms any optimizer that combines coordinate-separable preconditioning (Adam) with shared global spectral gain (Muon).

形式陈述：

```lean
∀ {F_adam : Fin 2 → Fin 2 → ℝ},
  CGB.Generation.CoordinateSeparable F_adam →
    ∀ (g_muon : ℝ) {κ_target : ℝ},
      κ_target ≠ 0 →
        ∀ {base s m_tail m_dom : ℝ},
          0 < base →
            0 < s →
              0 < m_tail →
                m_tail < m_dom →
                  CGB.Generation.ParetoDominates (CGB.Generation.cgbOptimizer κ_target base)
                    (F_adam, CGB.Generation.SharedGlobalGain g_muon) κ_target base s m_tail m_dom
```
