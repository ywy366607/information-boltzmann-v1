# GenerationLifeGate

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean) · [返回导航](../README.md)

模块源码说明：Life necessities as admission obligations (generation chain, gate layer L) The axioms P1–P3 say what a medium is.

## CGB.Medium.kernel_self_le

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L28)

解释：正时钟速率下，一个改变状态的事件使该状态的一步自保持概率不超过 1−事件速率/时钟速率。

源码说明：One state-changing event of rate `r` makes the chain leave `x` within a tick with probability at least `r / R`.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) {R : ℝ},
  0 < R → ∀ (s : U → I) (x : U → Λ) (a : A), M.act a s x ≠ x → M.kernel R s x x ≤ 1 - M.rate a s x / R
```

## CGB.Medium.hold_vanishes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L48)

解释：若每种刺激下该状态都存在至少 η>0 的离开事件，则连续保持该状态的概率具有指定几何上界。

源码说明：**No exact latch**: if some state-changing event has rate at least `η > 0` under every stimulus, then the probability of sitting in `x` for `n` ticks under any stimulus sequence is at most `(1 - η / R) ^ n`, which tends to zero (G17 applied to one microstate). Memory that lasts must be carried by a set of states, such as the basin of a latch.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) {R η : ℝ},
  M.Clocked R →
    0 < η →
      ∀ (x : U → Λ),
        (∀ (s : U → I), ∃ a, M.act a s x ≠ x ∧ η ≤ M.rate a s x) →
          ∀ (sched : ℕ → U → I) (n : ℕ), ∏ t ∈ Finset.range n, M.kernel R (sched t) x x ≤ (1 - η / R) ^ n
```

## CGB.PairCRN.rate_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L69)

解释：在双体反应模型中，只要反应系数为正且两个位置不同，该反应速率严格为正。

源码说明：Every rule with a positive prefactor fires on every ordered pair of distinct units at a positive rate, under every stimulus.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  (C : CGB.PairCRN U Λ I R) {r : R},
  0 < C.κ r → ∀ {u v : U}, u ≠ v → ∀ (s : U → I) (x : U → Λ), 0 < C.medium.rate (r, u, v) s x
```

## CGB.PairCRN.exposed_of_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L78)

解释：在指定双体介质中，正反应系数、非零内容变化和不同刺激对应的不同储库势可见证刺激敏感性。

源码说明：**Life gate L-sense** (`Lives.exposed`): a pair medium is exposed as soon as one rule with a positive prefactor moves content through a boundary reservoir whose potential differs between two stimuli.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  (C : CGB.PairCRN U Λ I R) [Fintype Λ] [DecidableEq Λ] {r : R},
  0 < C.κ r →
    ∀ {u v : U},
      u ≠ v → ∀ (s s' : U → I) (x : U → Λ), C.dC r u v x ≠ 0 → C.reservoir s u v ≠ C.reservoir s' u v → C.medium.Exposed
```

## CGB.PairCRN.total_pairAct

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L89)

解释：双体反应作用前后的总内容差，恰等于该反应定义的内容变化 dC。

源码说明：The content an event takes in is the change of total content.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U] [Fintype R]
  [DecidableEq R] (C : CGB.PairCRN U Λ I R) {r : R} {u v : U},
  u ≠ v → ∀ (x : U → Λ), ↑(C.medium.total (C.pairAct r u v x)) - ↑(C.medium.total x) = C.dC r u v x
```

## CGB.PairCRN.metabolizes_of_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L97)

解释：在指定双体介质中，一个正系数、两位置不同且内容变化非零的反应可见证内容交换。

源码说明：**Life gate L-metabolism** (`Lives.metabolizes`): one content-changing event with a positive prefactor is a witness.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U] [Fintype R]
  [DecidableEq R] (C : CGB.PairCRN U Λ I R) {r : R},
  0 < C.κ r →
    ∀ {u v : U},
      u ≠ v →
        ∀ (s : U → I) (x : U → Λ),
          C.dC r u v x ≠ 0 → ∃ a s x, 0 < C.medium.rate a s x ∧ C.medium.total (C.medium.act a s x) ≠ C.medium.total x
```

## CGB.tableCRN.dC_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L116)

解释：对于匹配指定规则两侧身份的不同位置，规则见证状态下的内容变化等于规则产物内容和减输入内容和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} [Fintype U] [inst : DecidableEq U] [inst_1 : DecidableEq Λ] {n : ℕ}
  (spec : Fin n → CGB.RuleSpec Λ) (ident : U → ℕ) (bid : Finset ℕ) (content : Λ → ℕ) (φ : ℕ → Λ → ℝ) (μ : I → ℝ)
  (birth : ℕ → Λ) (calm : I) (k : Fin n) {u v : U},
  u ≠ v →
    ident u = (spec k).i →
      ident v = (spec k).j →
        (CGB.tableCRN spec ident bid content φ μ birth calm).dC (k, false) u v (CGB.tableCRN.witness spec k u) =
          ↑(content (spec k).b.1) + ↑(content (spec k).b.2) - ↑(content (spec k).a.1) - ↑(content (spec k).a.2)
```

## CGB.tableCRN.metabolizes_of_rule

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L128)

解释：若规则系数为正、两位置身份匹配且规则前后内容和不同，则规则表介质存在正速率内容交换事件。

源码说明：**L-metabolism for rule tables**: one rule with a positive prefactor whose two sides carry different content, on two distinct units of its pages.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} [inst : Fintype U] [inst_1 : DecidableEq U] [inst_2 : DecidableEq Λ]
  {n : ℕ} (spec : Fin n → CGB.RuleSpec Λ) (ident : U → ℕ) (bid : Finset ℕ) (content : Λ → ℕ) (φ : ℕ → Λ → ℝ) (μ : I → ℝ)
  (birth : ℕ → Λ) (calm : I) (k : Fin n),
  0 < (spec k).κ →
    ∀ {u v : U},
      u ≠ v →
        ident u = (spec k).i →
          ident v = (spec k).j →
            content (spec k).a.1 + content (spec k).a.2 ≠ content (spec k).b.1 + content (spec k).b.2 →
              ∀ (s : U → I),
                ∃ a s x,
                  0 < (CGB.tableCRN spec ident bid content φ μ birth calm).medium.rate a s x ∧
                    (CGB.tableCRN spec ident bid content φ μ birth calm).medium.total
                        ((CGB.tableCRN spec ident bid content φ μ birth calm).medium.act a s x) ≠
                      (CGB.tableCRN spec ident bid content φ μ birth calm).medium.total x
```

## CGB.tableCRN.exposed_of_rule

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L146)

解释：若正系数内容变化规则触及边界，且两种刺激的储库势不同，则该规则表介质对刺激敏感。

源码说明：**L-sense for rule tables**: the same witness on a rule touching a boundary page, and two stimuli at different reservoir potentials.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} [inst : Fintype U] [inst_1 : DecidableEq U] [inst_2 : DecidableEq Λ]
  {n : ℕ} (spec : Fin n → CGB.RuleSpec Λ) (ident : U → ℕ) (bid : Finset ℕ) (content : Λ → ℕ) (φ : ℕ → Λ → ℝ) (μ : I → ℝ)
  (birth : ℕ → Λ) (calm : I) [Fintype Λ] (k : Fin n),
  0 < (spec k).κ →
    ∀ {u v : U},
      u ≠ v →
        ident u = (spec k).i →
          ident v = (spec k).j →
            content (spec k).a.1 + content (spec k).a.2 ≠ content (spec k).b.1 + content (spec k).b.2 →
              (spec k).i ∈ bid ∨ (spec k).j ∈ bid →
                ∀ (c c' : I), μ c ≠ μ c' → (CGB.tableCRN spec ident bid content φ μ birth calm).medium.Exposed
```
