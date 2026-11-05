# GenerationActiveInference

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean) · [返回导航](../README.md)

模块源码说明：Active inference from finite supply (minimal information contract) This module does not add a physical entropy variable or a new organism state.

## CGB.ActiveInference.not_unbounded_of_bounded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L41)

分类：necessity；适用范围：world_contract；审核：reviewed

分类说明：动作敏感环境与无界信号下，排除了动作盲目预测器和最终被动策略的必要性定理。

解释：输入累积信号有上界的信号生成序列，经有界性检验，输出该信号序列必然无法满足生命体长期存活所需的发散无界信号条件。

源码说明：A bounded cumulative signal cannot satisfy the unbounded viability contract.

形式陈述：

```lean
∀ {signal : ℕ → ℕ},
  (∃ C, ∀ (T : ℕ), CGB.ActiveInference.cumulative signal T ≤ C) → ¬CGB.ActiveInference.unbounded signal
```

## CGB.ActiveInference.viable_not_eventually_passive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L52)

分类：necessity；适用范围：world_contract；审核：reviewed

分类说明：动作敏感环境与无界信号下，排除了动作盲目预测器和最终被动策略的必要性定理。

解释：输入长期策略、可用累积信号与初始参考动作，经生存必要性规约，输出若生存需要无界累积信号且所有最终被动策略仅产生有界信号，则存活策略绝不能最终停滞于被动状态。

源码说明：**Active necessity.** If viability requires an unbounded usable signal, and every eventually passive policy has a bounded signal in the declared world, then a viable policy cannot become eventually passive.

形式陈述：

```lean
∀ {A : Type u_1} (policy : ℕ → A) (signal : ℕ → ℕ) (a₀ : A),
  CGB.ActiveInference.unbounded signal →
    CGB.ActiveInference.passiveBounded policy signal a₀ → ¬CGB.ActiveInference.eventuallyPassive policy a₀
```

## CGB.ActiveInference.action_condition_required

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L68)

分类：necessity；适用范围：world_contract；审核：reviewed

分类说明：动作敏感环境与无界信号下，排除了动作盲目预测器和最终被动策略的必要性定理。

解释：输入动作敏感的环境条件响应算子与动作盲目的预测器，经反证法推导，输出动作盲目预测器无法在所有动作条件下精确拟合动作敏感环境的实际响应。

源码说明：If the world is action-sensitive, an action-blind predictor cannot match every action-conditioned response. The predictor need not be a new channel: the action is simply an argument of the existing causal state/readout.

形式陈述：

```lean
∀ {A : Type u_2} {I : Type u_3} (response : A → I → ℝ) (predict : I → ℝ),
  CGB.ActiveInference.actionSensitive response → (∀ (a : A) (i : I), predict i = response a i) → False
```

## CGB.ActiveInference.silence_condition_required

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L78)

分类：necessity；适用范围：world_contract；审核：reviewed

分类说明：当字节零的响应随行动变化时，行动无关预测无法匹配所有行动；并非每次沉默都提供新信息。

解释：输入字节级世界响应模型与无条件预测器，经动作干预导致零字节（沉默）概率变化的判据，输出盲目预测器无法拟合由自身动作干预引起的条件输出概率。

源码说明：Silence is an ordinary byte outcome. If the probability of byte `0` depends on the action, an action-blind predictor still cannot represent it.

形式陈述：

```lean
∀ {A : Type u_2} (response : A → Fin 256 → ℝ) (predict : Fin 256 → ℝ),
  (∃ a a', response a 0 ≠ response a' 0) → (∀ (a : A) (i : Fin 256), predict i = response a i) → False
```

## CGB.ActiveInference.active_closure

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L92)

分类：necessity；适用范围：world_contract；审核：reviewed

分类说明：联合结论依赖无界信号、被动有界以及行动敏感的世界合同。

解释：输入策略序列、可用生存信号、动作敏感响应算子与预测器，经主动推理生存闭包推演，输出长期维持生存排除了最终被动策略，且动作敏感环境排除了动作盲目预测器。

源码说明：**Active closure.** Under one declared world contract, viability rules out eventual passivity and action-sensitive future inputs rule out an action-blind predictor. This is the minimal logical closure of survival pressure, intervention and self-conditioned prediction; it does not assume that every tick produces a nonzero response.

形式陈述：

```lean
∀ {A : Type u_2} {I : Type u_3} (policy : ℕ → A) (signal : ℕ → ℕ) (a₀ : A) (response : A → I → ℝ) (predict : I → ℝ),
  CGB.ActiveInference.unbounded signal →
    CGB.ActiveInference.passiveBounded policy signal a₀ →
      CGB.ActiveInference.actionSensitive response →
        ¬CGB.ActiveInference.eventuallyPassive policy a₀ ∧ ¬∀ (a : A) (i : I), predict i = response a i
```
