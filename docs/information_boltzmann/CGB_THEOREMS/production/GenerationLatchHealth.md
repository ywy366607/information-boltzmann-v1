# GenerationLatchHealth

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean) · [返回导航](../README.md)

模块源码说明：When is a latch healthy? (generation chain, latch health) A latch (ignition, G23/G24) is a basin of states held by positive feedback.

## CGB.Medium.trap_irrecoverable

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L27)

解释：若状态集合对个体自身事件封闭且其中没有任何可摄入状态，则集合内每个状态都无法通过自身事件恢复摄入。

源码说明：**H1 (lock-in is death)**: if the set `B` is closed under the organism's own events and no state in it can take content in, no state in it is recoverable.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  (B : Set (U → Λ)),
  (∀ (x y : U → Λ), x ∈ B → M.OrgStep x y → y ∈ B) → (∀ x ∈ B, ¬M.CanTakeIn x) → ∀ x ∈ B, ¬M.Recoverable x
```

## CGB.Medium.trap_insensitive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L41)

解释：在某状态下，如果所有事件的速率和作用均与刺激无关，那么该状态的一步转移核也与刺激无关。

源码说明：**H2 (latched means blind)**: where rates and effects do not depend on the stimulus, the one-tick kernel out of that state is the same under every stimulus.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] {R : ℝ} (x : U → Λ),
  (∀ (a : A) (s s' : U → I), M.rate a s x = M.rate a s' x) →
    (∀ (a : A) (s s' : U → I), M.act a s x = M.act a s' x) →
      ∀ (s s' : U → I) (y : U → Λ), M.kernel R s x y = M.kernel R s' x y
```

## CGB.PairCRN.pair_rate_ratio

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L59)

解释：在给定双体反应介质中，不同刺激下同一事件速率之比由储库势变化与内容变化的指数因子给出。

源码说明：**H1, quantitative**: a cue changes the rate of an event by exactly `exp((reservoir s - reservoir s') · dC / 2)`.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  (C : CGB.PairCRN U Λ I R) {r : R} {u v : U},
  u ≠ v →
    ∀ (s s' : U → I) (x : U → Λ),
      C.medium.rate (r, u, v) s x =
        C.medium.rate (r, u, v) s' x * Real.exp ((C.reservoir s u v - C.reservoir s' u v) * C.dC r u v x / 2)
```
