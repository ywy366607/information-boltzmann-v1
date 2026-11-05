# GenerationLatchSlot

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean) · [返回导航](../README.md)

模块源码说明：Latch slot: renewal that does not ride on the input gate (generation chain, 2026-11-04) Motivation (`results/cgb_rich_world_20261104/RESULT.md`, `results/cgb_mode_task_20261103/`): the trained latch heads are erased by o

## CGB.Generation.LatchSlot.coupled_closed_gate_decays

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L48)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：耦合规则中关闭写入门只剩衰减，自我维持消失。

解释：Closed gate in the coupled rule: no renewal is left, only the decay `α`.

源码说明：Closed gate in the coupled rule: no renewal is left, only the decay `α`.

形式陈述：

```lean
∀ (α η g r : ℝ), CGB.Generation.HoldCorrect.fb (α * (1 - 0)) (0 * η) g r = α * r
```

## CGB.Generation.LatchSlot.coupled_small_gate_subcritical

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L54)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：耦合规则中小写入门必然亚临界（κ≤1）。

解释：**A small gate is subcritical** in the coupled rule: with `0 ≤ α`, `0 ≤ β` and `β η g ≤ 1 - α`, the renewal satisfies `c g ≤ 1 - a` (κ ≤ 1).

源码说明：**A small gate is subcritical** in the coupled rule: with `0 ≤ α`, `0 ≤ β` and `β η g ≤ 1 - α`, the renewal satisfies `c g ≤ 1 - a` (κ ≤ 1).

形式陈述：

```lean
∀ {α β η g : ℝ}, 0 ≤ α → 0 ≤ β → β * (η * g) ≤ 1 - α → β * η * g ≤ 1 - α * (1 - β)
```

## CGB.Generation.LatchSlot.coupled_small_gate_forgets

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L61)

分类：necessity；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：耦合规则中头无法关门挡文本而不失去锁存：内容每事件收缩。

解释：With a small gate the content is forgotten every event (`subcritical_forgets`).

源码说明：With a small gate the content is forgotten every event (`subcritical_forgets`).

形式陈述：

```lean
∀ {α β η g x : ℝ},
  0 ≤ α →
    0 < β →
      β ≤ 1 → 0 < η → 0 < g → β * (η * g) ≤ 1 - α → 0 < x → CGB.Generation.HoldCorrect.fb (α * (1 - β)) (β * η) g x < x
```

## CGB.Generation.LatchSlot.coupled_latch_gate_floor

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L69)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：耦合锁存的写入门有下限 β>(1-α)/(ηg-α)。

解释：**The gate floor of a coupled latch**: while it latches (κ > 1), `1 - α < β (η g - α)`.

源码说明：**The gate floor of a coupled latch**: while it latches (κ > 1), `1 - α < β (η g - α)`.

形式陈述：

```lean
∀ {α β η g : ℝ}, 1 - α * (1 - β) < β * η * g → 1 - α < β * (η * g - α)
```

## CGB.Generation.LatchSlot.coupled_disturbance_floor

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L77)

分类：necessity；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：耦合锁存收进的文本扰动有门控去不掉的下限，需把维持与输入分开。

解释：**The disturbance floor**: a latching coupled head admits from a value of size `V` at least `(1-α)(1-η)V/(η g - α)` per event (`D = β (1 - η) V`).

源码说明：**The disturbance floor**: a latching coupled head admits from a value of size `V` at least `(1-α)(1-η)V/(η g - α)` per event (`D = β (1 - η) V`).

形式陈述：

```lean
∀ {α β η g V : ℝ},
  η < 1 → 0 < V → α < η * g → 1 - α * (1 - β) < β * η * g → (1 - α) * ((1 - η) * V) / (η * g - α) < β * ((1 - η) * V)
```

## CGB.Generation.LatchSlot.slot_composite

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L87)

分类：design；适用范围：candidate；审核：draft；候选：LatchSlot

分类说明：衰减后接槽更新仍属同一 fb 族，参数重标度。

解释：Decay by `α` followed by the slot renewal is the same family with rescaled parameters.

源码说明：Decay by `α` followed by the slot renewal is the same family with rescaled parameters.

形式陈述：

```lean
∀ (a c g α r : ℝ), CGB.Generation.HoldCorrect.fb a c g (α * r) = CGB.Generation.HoldCorrect.fb (a * α) c (g * α) r
```

## CGB.Generation.LatchSlot.slot_supercritical_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L94)

分类：design；适用范围：candidate；审核：draft；候选：LatchSlot

分类说明：槽锁存当且仅当 α(1+ρ(ηg-1))>1，与写入门无关。

解释：**When the slot latches**: with `a = 1 - ρ`, `c = ρ η` after decay `α`, the composite map is supercritical (`1 - a α < c g α`) iff `α (1 + ρ (η g - 1)) > 1`. The write gate does not enter.

源码说明：**When the slot latches**: with `a = 1 - ρ`, `c = ρ η` after decay `α`, the composite map is supercritical (`1 - a α < c g α`) iff `α (1 + ρ (η g - 1)) > 1`. The write gate does not enter.

形式陈述：

```lean
∀ (ρ η g α : ℝ), 1 - (1 - ρ) * α < ρ * η * (g * α) ↔ 1 < α * (1 + ρ * (η * g - 1))
```

## CGB.Generation.LatchSlot.slot_supercritical_no_decay

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L100)

分类：design；适用范围：candidate；审核：draft；候选：LatchSlot

分类说明：无衰减时槽锁存当且仅当 ηg>1。

解释：Without decay the slot latches iff `η g > 1`, for any renewal rate `ρ > 0`.

源码说明：Without decay the slot latches iff `η g > 1`, for any renewal rate `ρ > 0`.

形式陈述：

```lean
∀ {ρ η g : ℝ}, 0 < ρ → (1 - (1 - ρ) * 1 < ρ * η * (g * 1) ↔ 1 < η * g)
```
