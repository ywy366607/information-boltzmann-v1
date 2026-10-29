# Inference

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/Inference.lean) · [返回导航](../README.md)

模块源码说明：Active inference, first rungs: the blanket and foresight The life definition orders its emergent part as a ladder: memory → world model and prediction → choosing actions by what is predicted.

## CGB.Medium.jump_blanket

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Inference.lean#L50)

解释：马尔可夫跳跃介质中的局部事件速率仅依赖于局部毯内状态，毯外跃迁不影响局部概率流。

源码说明：Under P3, the full dynamics reads the stimulus only at the boundary.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype A] (M : CGB.Medium U Λ A I)
  [inst_1 : DecidableEq (U → Λ)],
  M.OnlyBoundarySenses →
    ∀ {s s' : U → I}, (∀ u ∈ M.boundary, s u = s' u) → ∀ (x y : U → Λ), (M.jump s).rate x y = (M.jump s').rate x y
```

## CGB.Medium.kernel_blanket

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Inference.lean#L56)

解释：时间离散化核算子在马尔可夫毯内局部化，外部状态的演化在给定毯条件后对内部条件独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I),
  M.OnlyBoundarySenses → ∀ (R : ℝ) {s s' : U → I}, (∀ u ∈ M.boundary, s u = s' u) → M.kernel R s = M.kernel R s'
```

## CGB.Medium.blanket_of_axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Inference.lean#L73)

解释：满足基础公理的局部介质自然存在马尔可夫毯结构，使得内部与外部在给定边界下条件独立。

源码说明：P3 gives the organism's half of the blanket; a world that touches only the body gives the other half.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I),
  M.Axioms → M.ReadsBoundary Wd → M.Blanket Wd
```

## CGB.Models.rock_cannot_eat

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Inference.lean#L119)

解释：惰性岩石模型定理：缺乏内部代谢反应网络的无机系统摄入速率恒为零。

源码说明：The rock satisfies all six axioms but can never take anything in: it was never alive.

形式陈述：

```lean
∀ (x : Fin 2 → Fin 3), ¬CGB.Models.rock.CanTakeIn x
```

## CGB.Models.rock_not_persists

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Inference.lean#L124)

解释：惰性岩石非生命定理：无法维持负熵流的被动系统不能在非平稳耗散环境中实现持久存在。

源码说明：So the rock persists in no world.

形式陈述：

```lean
∀ (Wd : CGB.World (Fin 2) (Fin 3) Unit) (δ : ℝ) (T : ℕ), ¬CGB.Models.rock.Persists Wd δ T
```

## CGB.finite_reweighting_improves_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Inference.lean#L134)

解释：有限样本重加权提高事件概率当且仅当该事件内部的加权期望严格大于全空间全局加权期望。

源码说明：Any positive finite reweighting improves an observed event's normalized probability exactly when its conditional mean multiplier exceeds the global mean. This is a decision criterion for a specified change, not a learning law, sampler guarantee, or claim that a shared potential is forced by the axioms.

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (event : Finset S) (w r : S → ℝ),
  (∀ (s : S), 0 < w s) →
    (∀ (s : S), 0 < r s) →
      event.Nonempty →
        ((∑ s ∈ event, w s) / ∑ s, w s < (∑ s ∈ event, w s * r s) / ∑ s, w s * r s ↔
          (∑ s, w s * r s) / ∑ s, w s < (∑ s ∈ event, w s * r s) / ∑ s ∈ event, w s)
```
