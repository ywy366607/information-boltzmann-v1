# ConstraintFreedom

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/ConstraintFreedom.lean) · [返回导航](../README.md)

模块源码说明：Families admitted by the same clauses These are logical model families, not proposed brain mechanisms or capability experiments.

## CGB.ConstraintFreedom.arbitrarily_many_states_at_unit_capacity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintFreedom.lean#L19)

解释：对任意 n，存在 n+1 个局部状态的内容函数，使内容都不超过一且零内容状态唯一。

源码说明：P1a is qualitative: an arbitrarily large FINITE state alphabet may have capacity 1. A quantitative information/content relation is not encoded in it.

形式陈述：

```lean
∀ (n : ℕ), ∃ cost, (∀ (l : Fin (n + 1)), cost l ≤ 1) ∧ (∃! l, cost l = 0) ∧ Fintype.card (Fin (n + 1)) = n + 1
```

## CGB.ConstraintFreedom.arbitrary_driven_tables_admitted

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintFreedom.lean#L49)

解释：有限状态、唯一零态及非负速率下，任意更新表都能由指定开放单点介质满足相应公理。

源码说明：Thus the core alone does not determine a driven rule table. This does not negate life existence, nor waive encoded budgets or separate world constraints.

形式陈述：

```lean
∀ {Λ : Type u_1} {E : Type u_2} [Fintype Λ] [inst : DecidableEq Λ] [inst_1 : Fintype E] (cost : Λ → ℕ),
  (∃! l, cost l = 0) →
    ∀ (advance : E → Λ → Λ) (speed : E → Λ → ℝ) (hs : ∀ (e : E) (l : Λ), 0 ≤ speed e l) (birth : Λ),
      (CGB.ConstraintFreedom.openSite cost advance speed hs birth).Axioms
```

## CGB.ConstraintFreedom.openSite_reproduces_process

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintFreedom.lean#L80)

解释：指定开放单点构造可在非对角状态转移上复现原跳跃过程速率，并按定义处理对角项。

源码说明：With one event per target state, the exposed-site family reproduces the off-diagonal rates of ANY finite Markov jump process under its driven input. This is a statement about the permissiveness of the core representation, not an encoded-budget or life witness. Self-jumps are omitted just as in Medium.jump.

形式陈述：

```lean
∀ {Λ : Type u_1} [inst : Fintype Λ] [inst_1 : DecidableEq Λ] (cost : Λ → ℕ) (P : CGB.JumpProcess Λ) (birth x y : Λ),
  (((CGB.ConstraintFreedom.openSite cost (fun e x => e) (fun e l => P.rate l e) ⋯ birth).jump fun x => true).rate
      (fun x_1 => x) fun x => y) =
    if x = y then 0 else P.rate x y
```

## CGB.ConstraintFreedom.constant_protocol_can_be_driven

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintFreedom.lean#L88)

解释：固定的 food 刺激协议仍可对应非可逆的 seed 跳跃过程。

源码说明：A constant-in-time stimulus already breaks detailed balance in the existing seed witness. Time inhomogeneity and equilibrium reversal are different claims.

形式陈述：

```lean
(∀ (n : ℕ), (fun x => CGB.Models.food) n = CGB.Models.food) ∧ ¬(CGB.Models.seed.jump CGB.Models.food).Reversible
```
