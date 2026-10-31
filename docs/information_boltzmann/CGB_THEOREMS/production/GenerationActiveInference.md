# GenerationActiveInference

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean) · [返回导航](../README.md)

模块源码说明：Active inference from finite supply (minimal information contract) This module does not add a physical entropy variable or a new organism state.

## CGB.ActiveInference.not_unbounded_of_bounded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L41)

源码说明：A bounded cumulative signal cannot satisfy the unbounded viability contract.

形式陈述：

```lean
∀ {signal : ℕ → ℕ},
  (∃ C, ∀ (T : ℕ), CGB.ActiveInference.cumulative signal T ≤ C) → ¬CGB.ActiveInference.unbounded signal
```

## CGB.ActiveInference.viable_not_eventually_passive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L52)

源码说明：**Active necessity.** If viability requires an unbounded usable signal, and every eventually passive policy has a bounded signal in the declared world, then a viable policy cannot become eventually passive.

形式陈述：

```lean
∀ {A : Type u_1} (policy : ℕ → A) (signal : ℕ → ℕ) (a₀ : A),
  CGB.ActiveInference.unbounded signal →
    CGB.ActiveInference.passiveBounded policy signal a₀ → ¬CGB.ActiveInference.eventuallyPassive policy a₀
```

## CGB.ActiveInference.action_condition_required

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L68)

源码说明：If the world is action-sensitive, an action-blind predictor cannot match every action-conditioned response. The predictor need not be a new channel: the action is simply an argument of the existing causal state/readout.

形式陈述：

```lean
∀ {A : Type u_2} {I : Type u_3} (response : A → I → ℝ) (predict : I → ℝ),
  CGB.ActiveInference.actionSensitive response → (∀ (a : A) (i : I), predict i = response a i) → False
```

## CGB.ActiveInference.silence_condition_required

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L78)

源码说明：Silence is an ordinary byte outcome. If the probability of byte `0` depends on the action, an action-blind predictor still cannot represent it.

形式陈述：

```lean
∀ {A : Type u_2} (response : A → Fin 256 → ℝ) (predict : Fin 256 → ℝ),
  (∃ a a', response a 0 ≠ response a' 0) → (∀ (a : A) (i : Fin 256), predict i = response a i) → False
```

## CGB.ActiveInference.active_closure

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L92)

源码说明：**Active closure.** Under one declared world contract, viability rules out eventual passivity and action-sensitive future inputs rule out an action-blind predictor. This is the minimal logical closure of survival pressure, intervention and self-conditioned prediction; it does not assume that every tick produces a nonzero response.

形式陈述：

```lean
∀ {A : Type u_2} {I : Type u_3} (policy : ℕ → A) (signal : ℕ → ℕ) (a₀ : A) (response : A → I → ℝ) (predict : I → ℝ),
  CGB.ActiveInference.unbounded signal →
    CGB.ActiveInference.passiveBounded policy signal a₀ →
      CGB.ActiveInference.actionSensitive response →
        ¬CGB.ActiveInference.eventuallyPassive policy a₀ ∧ ¬∀ (a : A) (i : I), predict i = response a i
```
