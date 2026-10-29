# CausalLearning

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean) · [返回导航](../README.md)

模块源码说明：Finite, causal and paid candidate-model learning Models predict possible boundary outcomes of actions.

## CGB.CausalLearning.memory_cardinality

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L50)

解释：有限候选集合掩码与 0 到 B 的写入储备组成的记忆状态数恰为 2^|H|·(B+1)。

源码说明：Candidate masks have |H| bits of state capacity, plus the explicit reserve. This cardinality does not certify a rulebook's 1-percent description budget.

形式陈述：

```lean
∀ {H : Type u_1} [inst : Fintype H] (B : ℕ), Fintype.card (CGB.CausalLearning.Memory H B) = 2 ^ Fintype.card H * (B + 1)
```

## CGB.CausalLearning.write_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L61)

解释：一次候选筛选只会缩小候选集，且有效写入恰消耗一单位储备并至少移除一个候选。

源码说明：Each effective write strictly removes candidates and spends one declared unit of finite reserve. Insufficient reserve cannot create an unpaid update.

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H]
  (possible : CGB.CausalLearning.Support H C A O) {B : ℕ} (m : CGB.CausalLearning.Memory H B) (c : C) (a : A) (o : O),
  (CGB.CausalLearning.advance possible m c a o).candidates ⊆ m.candidates ∧
    ↑(CGB.CausalLearning.advance possible m c a o).reserve + CGB.CausalLearning.fee possible m c a o = ↑m.reserve ∧
      (CGB.CausalLearning.advance possible m c a o).candidates.card + CGB.CausalLearning.fee possible m c a o ≤
        m.candidates.card
```

## CGB.CausalLearning.retain_consistent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L84)

解释：当前候选若允许实际观测，则无论是否发生写入，更新后仍保留该候选。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H]
  (possible : CGB.CausalLearning.Support H C A O) {B : ℕ} (m : CGB.CausalLearning.Memory H B) (c : C) (a : A) (o : O),
  ∀ h ∈ m.candidates, possible h c a o = true → h ∈ (CGB.CausalLearning.advance possible m c a o).candidates
```

## CGB.CausalLearning.agreement_no_write

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L93)

解释：若所有当前候选都允许所见观测，则候选记忆保持不变且不收写入费用。

源码说明：A predictable observation provides no exclusion evidence. Observing more does not by itself imply identification or useful action learning.

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H]
  (possible : CGB.CausalLearning.Support H C A O) {B : ℕ} (m : CGB.CausalLearning.Memory H B) (c : C) (a : A) (o : O),
  (∀ h ∈ m.candidates, possible h c a o = true) →
    CGB.CausalLearning.advance possible m c a o = m ∧ CGB.CausalLearning.fee possible m c a o = 0
```

## CGB.CausalLearning.empty_write_reserve

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L101)

解释：写入储备为零时，候选记忆保持不变且费用为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H]
  (possible : CGB.CausalLearning.Support H C A O) {B : ℕ} (m : CGB.CausalLearning.Memory H B),
  ↑m.reserve = 0 →
    ∀ (c : C) (a : A) (o : O),
      CGB.CausalLearning.advance possible m c a o = m ∧ CGB.CausalLearning.fee possible m c a o = 0
```

## CGB.CausalLearning.paid_elimination

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L108)

解释：若储备为正且当前候选不允许所见观测，则更新移除该候选并消耗一单位储备。

源码说明：A funded observation can remove a false candidate using the observation itself, without consulting a true-model identifier.

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H]
  (possible : CGB.CausalLearning.Support H C A O) {B : ℕ} (m : CGB.CausalLearning.Memory H B) (c : C) (a : A) (o : O)
  (h : H),
  0 < ↑m.reserve →
    h ∈ m.candidates →
      possible h c a o = false →
        h ∉ (CGB.CausalLearning.advance possible m c a o).candidates ∧ CGB.CausalLearning.fee possible m c a o = 1
```

## CGB.CausalLearning.path_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L125)

解释：沿任意观测序列的候选更新，当前储备加累计写费等于初始储备，且当前候选数加累计写费不超过初始候选数。

源码说明：Analysis may count writes along a path; that counter is not learner state. This holds for arbitrary observations, including model-misspecified histories.

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H]
  (possible : CGB.CausalLearning.Support H C A O) {B : ℕ} (m : ℕ → CGB.CausalLearning.Memory H B) (context : ℕ → C)
  (action : ℕ → A) (observed : ℕ → O),
  (∀ (t : ℕ), m (t + 1) = CGB.CausalLearning.advance possible (m t) (context t) (action t) (observed t)) →
    ∀ (n : ℕ),
      ↑(m n).reserve + ∑ t ∈ Finset.range n, CGB.CausalLearning.fee possible (m t) (context t) (action t) (observed t) =
          ↑(m 0).reserve ∧
        (m n).candidates.card +
            ∑ t ∈ Finset.range n, CGB.CausalLearning.fee possible (m t) (context t) (action t) (observed t) ≤
          (m 0).candidates.card
```

## _private.CGBAxioms.CausalLearning.0.CGB.CausalLearning.delivered

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L164)

解释：闭环一步转移具有正概率时，输出动作来自当前策略、后继记忆按观测更新且相应物理转移为正概率。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H] {S : Type u_5}
  [inst_2 : Fintype S] [inst_3 : Fintype A] [inst_4 : DecidableEq A] [inst_5 : Fintype O]
  (P : CGB.Interaction.Machine S (ℕ × A) O) (sensor : S → C) (possible : CGB.CausalLearning.Support H C A O) {B : ℕ}
  (choose : CGB.CausalLearning.Memory H B → C → A) (t : ℕ) (z z' : S × CGB.CausalLearning.Memory H B) (out : A × O),
  0 < (CGB.CausalLearning.closed P sensor possible choose).step t z out z' →
    out.1 = choose z.2 (sensor z.1) ∧
      z'.2 = CGB.CausalLearning.advance possible z.2 (sensor z.1) out.1 out.2 ∧ 0 < P.step (t, out.1) z.1 out.2 z'.1
```

## CGB.CausalLearning.true_model_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L182)

解释：若某候选忠实覆盖实际物理转移且当前未被排除，则任何正概率闭环一步后它仍被保留。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H] {S : Type u_5}
  [inst_2 : Fintype S] [inst_3 : Fintype A] [inst_4 : DecidableEq A] [inst_5 : Fintype O]
  (P : CGB.Interaction.Machine S (ℕ × A) O) (sensor : S → C) (possible : CGB.CausalLearning.Support H C A O) {B : ℕ}
  (choose : CGB.CausalLearning.Memory H B → C → A) (h : H),
  CGB.CausalLearning.Faithful P sensor possible h →
    ∀ (t : ℕ) (z z' : S × CGB.CausalLearning.Memory H B) (out : A × O),
      h ∈ z.2.candidates →
        0 < (CGB.CausalLearning.closed P sensor possible choose).step t z out z' → h ∈ z'.2.candidates
```

## _private.CGBAxioms.CausalLearning.0.CGB.CausalLearning.law_invariant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L192)

解释：若性质在初态成立并对所有正概率一步保持，则任意时刻不满足该性质的状态概率为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {X : Type u_6} {E : Type u_7} [inst : Fintype X] [inst_1 : DecidableEq X] [inst_2 : Fintype E]
  (Q : CGB.Interaction.Machine X ℕ E) (initial : X) (Good : X → Prop),
  Good initial →
    (∀ (t : ℕ) (x : X) (e : E) (y : X), Good x → 0 < Q.step t x e y → Good y) →
      ∀ (n : ℕ) (x : X), ¬Good x → CGB.Energetics.law Q initial n x = 0
```

## CGB.CausalLearning.true_model_law

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L215)

解释：从包含全部候选的初始记忆出发，忠实候选在任意时刻的每个正概率可达状态中仍被保留。

源码说明：Positive-probability states retain the faithful candidate at every time; the controller definition itself never receives this candidate's identity.

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H] {S : Type u_5}
  [inst_2 : Fintype S] [inst_3 : DecidableEq S] [inst_4 : Fintype A] [inst_5 : DecidableEq A] [inst_6 : Fintype O]
  (P : CGB.Interaction.Machine S (ℕ × A) O) (sensor : S → C) (possible : CGB.CausalLearning.Support H C A O) {B : ℕ}
  (choose : CGB.CausalLearning.Memory H B → C → A) (h : H),
  CGB.CausalLearning.Faithful P sensor possible h →
    ∀ (s₀ : S) (n : ℕ) (z : S × CGB.CausalLearning.Memory H B),
      0 <
          CGB.Energetics.law (CGB.CausalLearning.closed P sensor possible choose)
            (s₀, CGB.CausalLearning.initialMemory H B) n z →
        h ∈ z.2.candidates
```

## CGB.CausalLearning.physical_marginal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L231)

解释：对后继记忆求和后，闭环物理动作与观测的转移概率仍等于被选动作对应的原物理核。

源码说明：Even with zero write reserve, each physical action/outcome law continues. Only the deterministic memory successor is attached to the physical event.

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H] {S : Type u_5}
  [inst_2 : Fintype S] [inst_3 : Fintype A] [inst_4 : DecidableEq A] [inst_5 : Fintype O]
  (P : CGB.Interaction.Machine S (ℕ × A) O) (sensor : S → C) (possible : CGB.CausalLearning.Support H C A O) {B : ℕ}
  (choose : CGB.CausalLearning.Memory H B → C → A) (t : ℕ) (z : S × CGB.CausalLearning.Memory H B) (out : A × O)
  (s' : S),
  ∑ m', (CGB.CausalLearning.closed P sensor possible choose).step t z out (s', m') =
    if out.1 = choose z.2 (sensor z.1) then P.step (t, out.1) z.1 out.2 s' else 0
```

## CGB.CausalLearning.safe_outcome

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L242)

解释：若所选动作对所有剩余候选均稳健安全且忠实候选仍在集合中，则任意正概率实际结果满足安全条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H] {S : Type u_5}
  [inst_2 : Fintype S] [inst_3 : Fintype A] [inst_4 : DecidableEq A] [inst_5 : Fintype O]
  (P : CGB.Interaction.Machine S (ℕ × A) O) (sensor : S → C) (possible : CGB.CausalLearning.Support H C A O) {B : ℕ}
  (choose : CGB.CausalLearning.Memory H B → C → A) (good : C → A → O → Prop) (h : H),
  CGB.CausalLearning.Faithful P sensor possible h →
    (∀ (m : CGB.CausalLearning.Memory H B) (c : C),
        m.candidates.Nonempty → CGB.CausalLearning.Robust possible good m.candidates c (choose m c)) →
      ∀ (t : ℕ) (z z' : S × CGB.CausalLearning.Memory H B) (out : A × O),
        h ∈ z.2.candidates →
          0 < (CGB.CausalLearning.closed P sensor possible choose).step t z out z' → good (sensor z.1) out.1 out.2
```

## CGB.CausalLearning.certified_choice_sound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L268)

解释：当经过所有剩余候选验证的安全动作集合非空时，选择其中最小动作满足稳健安全定义。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H]
  [inst_2 : Fintype A] [inst_3 : Fintype O] (possible : CGB.CausalLearning.Support H C A O) {B : ℕ}
  [inst_4 : LinearOrder A] (good : C → A → O → Bool) (fallback : A) (m : CGB.CausalLearning.Memory H B) (c : C),
  (CGB.CausalLearning.certifiedActions possible good m.candidates c).Nonempty →
    CGB.CausalLearning.Robust possible (fun c a o => good c a o = true) m.candidates c
      (CGB.CausalLearning.certifiedOr possible good fallback m c)
```

## CGB.CausalLearning.no_safe_exploration

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L288)

解释：若每个动作都被某剩余候选允许的某个不安全结果反驳，则不存在满足该稳健安全条件的动作。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} (possible : CGB.CausalLearning.Support H C A O)
  (good : C → A → O → Prop) (v : Finset H) (c : C),
  (∀ (a : A), ∃ h ∈ v, ∃ o, possible h c a o = true ∧ ¬good c a o) → ¬∃ a, CGB.CausalLearning.Robust possible good v c a
```

## CGB.CausalLearning.paid_mean_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L317)

解释：在实际闭环过程中，任意有限时刻的平均剩余写入储备加累计平均写费等于初始储备。

源码说明：Exact mean finite-prefix payment on the actual action/outcome process. World perception continues after write funds run out.

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H] {S : Type u_5}
  [inst_2 : Fintype S] [inst_3 : DecidableEq S] [inst_4 : Fintype A] [inst_5 : DecidableEq A] [inst_6 : Fintype O]
  (P : CGB.Interaction.Machine S (ℕ × A) O) (sensor : S → C) (possible : CGB.CausalLearning.Support H C A O) {B : ℕ}
  (choose : CGB.CausalLearning.Memory H B → C → A) (initial : S × CGB.CausalLearning.Memory H B) (n : ℕ),
  CGB.Energetics.mean (CGB.Energetics.law (CGB.CausalLearning.closed P sensor possible choose) initial n)
        ((CGB.CausalLearning.writeAccount P sensor possible choose).stored n) +
      ∑ t ∈ Finset.range n,
        CGB.Energetics.flux (CGB.CausalLearning.closed P sensor possible choose) initial
          (CGB.CausalLearning.charge sensor possible) t =
    ↑↑initial.2.reserve
```
