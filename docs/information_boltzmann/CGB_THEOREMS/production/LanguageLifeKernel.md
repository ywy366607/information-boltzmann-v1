# LanguageLifeKernel

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean) · [返回导航](../README.md)

模块源码说明：Compact algebraic reference kernel for the language-life model This is a finite exact reference object, not a neural-network capability claim.

## CGB.LanguageLifeKernel.incidence_ternary

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L22)

解释：该有向图定义中的每个节点—边关联系数只可能是 −1、0 或 1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {N E : ℕ} (g : CGB.LanguageLifeKernel.TernaryGraph N E) (e : Fin E) (n : Fin N),
  CGB.LanguageLifeKernel.incidence g e n = -1 ∨
    CGB.LanguageLifeKernel.incidence g e n = 0 ∨ CGB.LanguageLifeKernel.incidence g e n = 1
```

## CGB.LanguageLifeKernel.incidence_sum_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L28)

解释：每条边的节点关联系数之和为零，这是图上内部流动保持总量的基础。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {N E : ℕ} [NeZero N] (g : CGB.LanguageLifeKernel.TernaryGraph N E) (e : Fin E),
  ∑ n, CGB.LanguageLifeKernel.incidence g e n = 0
```

## CGB.LanguageLifeKernel.charge_total_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L38)

解释：对该有向图的任意整数边流，所有节点经关联矩阵更新后的电荷总和与更新前相等。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {N E : ℕ} [NeZero N] (g : CGB.LanguageLifeKernel.TernaryGraph N E) (charge : Fin N → ℤ) (flow : Fin E → ℤ),
  ∑ n, CGB.LanguageLifeKernel.chargeUpdate g charge flow n = ∑ n, charge n
```

## CGB.LanguageLifeKernel.reserve_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L50)

解释：若支出不超过原有储备与收入之和，储备更新后的余额加支出恰等于原储备加收入。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (reserve incoming outgoing : ℕ),
  outgoing ≤ reserve + incoming →
    CGB.LanguageLifeKernel.reserveUpdate reserve incoming outgoing + outgoing = reserve + incoming
```

## CGB.LanguageLifeKernel.step_clock_advances

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L66)

解释：语言核的离散时间标记在每次 step 后严格增加。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {N E : ℕ} (g : CGB.LanguageLifeKernel.TernaryGraph N E) (s : CGB.LanguageLifeKernel.State N) (flow : Fin E → ℤ)
  (incoming outgoing : ℕ), s.tick < (CGB.LanguageLifeKernel.step g s flow incoming outgoing).tick
```

## CGB.LanguageLifeKernel.step_charge_conserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L71)

解释：语言核一步图流更新前后，所有节点电荷总和相等。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {N E : ℕ} [NeZero N] (g : CGB.LanguageLifeKernel.TernaryGraph N E) (s : CGB.LanguageLifeKernel.State N)
  (flow : Fin E → ℤ) (incoming outgoing : ℕ),
  ∑ n, (CGB.LanguageLifeKernel.step g s flow incoming outgoing).charge n = ∑ n, s.charge n
```

## CGB.LanguageLifeKernel.step_reserve_paid

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L77)

解释：支出有足够储备和收入覆盖时，语言核一步更新后的储备加支出等于旧储备加收入。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {N E : ℕ} (g : CGB.LanguageLifeKernel.TernaryGraph N E) (s : CGB.LanguageLifeKernel.State N) (flow : Fin E → ℤ)
  (incoming outgoing : ℕ),
  outgoing ≤ s.reserve + incoming →
    (CGB.LanguageLifeKernel.step g s flow incoming outgoing).reserve + outgoing = s.reserve + incoming
```

## CGB.LanguageLifeKernel.no_visible_only_delayed_credit_rule

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L99)

解释：同一可见值对应两种不同的延迟信用目标时，只看当前可见值的确定规则无法同时满足两种目标。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {X C : Type} (rule : X → C) (visible : X) (credit₀ credit₁ : C),
  credit₀ ≠ credit₁ →
    ¬(rule visible = CGB.LanguageLifeKernel.delayedCreditTarget credit₀ credit₁ false ∧
        rule visible = CGB.LanguageLifeKernel.delayedCreditTarget credit₀ credit₁ true)
```

## CGB.LanguageLifeKernel.delayed_credit_requires_history_carrier

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L107)

解释：当同一个当前可见输入需要两个不同信用值时，任何只依赖该输入的确定规则都不可能同时输出两者。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {X C : Type} (rule : X → C) (visible : X) (credit₀ credit₁ : C),
  credit₀ ≠ credit₁ → ¬(rule visible = credit₀ ∧ rule visible = credit₁)
```

## CGB.LanguageLifeKernel.delayed_action_credit_requires_state_separation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L119)

源码说明：Delayed action credit needs a state distinction between alternatives whose later consequences require different credit. This does not require a separate action log: an existing action-induced body change, a self-produced utterance written into the same state, or a world-mediated state change may supply the needed distinction. If all are absent, no rule reading only the later state can recover which action occurred.

形式陈述：

```lean
∀ {S A C : Type} (advance : S → A → S) (rule : S → C) (s : S) (action₀ action₁ : A) (credit₀ credit₁ : C),
  advance s action₀ = advance s action₁ →
    credit₀ ≠ credit₁ → ¬(rule (advance s action₀) = credit₀ ∧ rule (advance s action₁) = credit₁)
```

## CGB.LanguageLifeKernel.conserved_transfer_target_trace

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L155)

解释：一次局部守恒转移的目标增加量恰等于转移量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : CGB.LanguageLifeKernel.LocalConservedTransfer), t.targetAfter - t.targetBefore = t.amount
```

## CGB.LanguageLifeKernel.conserved_transfer_source_trace

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L160)

解释：一次局部守恒转移的来源减少量恰等于转移量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : CGB.LanguageLifeKernel.LocalConservedTransfer), t.sourceBefore - t.sourceAfter = t.amount
```

## CGB.LanguageLifeKernel.conserved_transfer_total_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L165)

解释：一次局部守恒转移前后，来源与目标的总量相等。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : CGB.LanguageLifeKernel.LocalConservedTransfer), t.sourceAfter + t.targetAfter = t.sourceBefore + t.targetBefore
```

## CGB.LanguageLifeKernel.observable_transfer_recovers_amount

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L185)

解释：在该可观测守恒转移结构的读出前提下，目标端读数恰好等于转移量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : CGB.LanguageLifeKernel.ObservableConservedTransfer), t.targetReadout = t.amount
```

## CGB.LanguageLifeKernel.superposed_transfer_readout_is_net

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L189)

解释：同一目标叠加两条整数流后，其前后差值等于两流之和，单独分量不会由这个差值直接区分。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (targetBefore flow₁ flow₂ : ℤ), targetBefore + flow₁ + flow₂ - targetBefore = flow₁ + flow₂
```

## CGB.LanguageLifeKernel.conservation_does_not_identify_components

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L193)

解释：存在两组各自分量不同但总流量相等的分解，说明仅知道守恒总量不能唯一识别各分量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∃ flow₁ flow₂ other₁ other₂, flow₁ + flow₂ = other₁ + other₂ ∧ flow₁ ≠ other₁
```

## CGB.LanguageLifeKernel.leaky_transfer_readout_is_net

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L205)

解释：带泄漏的守恒转移中，载体前后差值等于输入减泄漏，不能仅凭净变化单独读出输入。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : CGB.LanguageLifeKernel.LeakyConservedTransfer), t.after - t.before = t.input - t.leaked
```

## CGB.LanguageLifeKernel.conserved_carrier_path_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L224)

解释：沿守恒载体轨迹，任意有限步的状态净变化等于这段时间输入减输出之和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p : CGB.LanguageLifeKernel.ConservedCarrierPath) (k : ℕ),
  p.value k - p.value 0 = ∑ n ∈ Finset.range k, (p.input n - p.output n)
```

## CGB.LanguageLifeKernel.conserved_carrier_path_no_output

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L233)

解释：沿守恒载体轨迹，如果各步输出均为零，则状态净变化等于累计输入。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p : CGB.LanguageLifeKernel.ConservedCarrierPath) (k : ℕ),
  (∀ (n : ℕ), p.output n = 0) → p.value k - p.value 0 = (Finset.range k).sum p.input
```

## CGB.LanguageLifeKernel.history_readout_exists_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L250)

解释：对所有历史存在准确读出规则，当且仅当编码相同的历史具有相同所需输出；默认输出用于构造读出。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {S : Type u_2} {C : Type u_3} (encode : H → S) (required : H → C) (fallback : C),
  (∃ read, ∀ (h : H), read (encode h) = required h) ↔ ∀ (h h' : H), encode h = encode h' → required h = required h'
```

## CGB.LanguageLifeKernel.history_transition_exists_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L258)

解释：存在只依赖当前编码与输入的确定状态更新，当且仅当编码相同的历史在同一后续输入下仍产生相同后继编码。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {S : Type u_2} {U : Type u_3} (encode : H → S) (advance : H → U → H) (fallback : S),
  (∃ next, ∀ (h : H) (u : U), next (encode h) u = encode (advance h u)) ↔
    ∀ (h h' : H), encode h = encode h' → ∀ (u : U), encode (advance h u) = encode (advance h' u)
```

## CGB.LanguageLifeKernel.external_causal_readout_exists_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L280)

解释：在给定检索接口和默认值下，存在能对所有历史和查询给出所需结果的读取规则，当且仅当不可区分的内部状态与检索结果不会要求不同输出。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {I : Type u_2} {E : Type u_3} {A : Type u_4} {V : Type u_5} {C : Type u_6} (internal : H → I)
  (external : H → E) (retrieve : I → E → A → V) (required : H → A → C) (fallback : C),
  (∃ read, ∀ (h : H) (a : A), read (internal h) a (retrieve (internal h) (external h) a) = required h a) ↔
    ∀ (h h' : H) (a : A),
      internal h = internal h' →
        retrieve (internal h) (external h) a = retrieve (internal h') (external h') a → required h a = required h' a
```

## CGB.LanguageLifeKernel.external_trace_requires_retrieved_distinction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L304)

解释：若确定读出规则能给出所有所需信用，当两段历史的内部状态相同但所需信用不同，检索出的外部记录必须不同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {I : Type u_2} {A : Type u_3} {V : Type u_4} {C : Type u_5} (internal : H → I) (returned : H → A → V)
  (required : H → A → C) (read : I → A → V → C),
  (∀ (h : H) (a : A), read (internal h) a (returned h a) = required h a) →
    ∀ (h h' : H) (a : A), internal h = internal h' → required h a ≠ required h' a → returned h a ≠ returned h' a
```

## CGB.LanguageLifeKernel.observational_record_does_not_fix_intervention

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L317)

解释：给出两个在观测输入一致时输出相同、但在一次干预下输出不同的规则，说明被动观测一致不足以确定干预结果。

源码说明：Two world equations agree on every recorded A=U trajectory but disagree under the intervention U=false, A=true. This is an exact logical countermodel, not a learning/capability experiment or an empirical causal-identification claim.

形式陈述：

```lean
have response := fun direct u a => if direct = true then a else u;
(∀ (u : Bool), response true u u = response false u u) ∧ response true false true ≠ response false false true
```

## CGB.LanguageLifeKernel.SensoryRecord.exact_record_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L366)

解释：该 16 状态感官记录编码与读出在合法状态上互逆，并保持合法性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
(∀ (r : Fin 16),
    CGB.LanguageLifeKernel.SensoryRecord.encode (CGB.LanguageLifeKernel.SensoryRecord.read r) = r ∧
      CGB.LanguageLifeKernel.SensoryRecord.admissible (CGB.LanguageLifeKernel.SensoryRecord.read r)) ∧
  (∀ (x : CGB.SensoryMaintenance.State),
      CGB.LanguageLifeKernel.SensoryRecord.admissible x →
        CGB.LanguageLifeKernel.SensoryRecord.read (CGB.LanguageLifeKernel.SensoryRecord.encode x) = x) ∧
    ∀ (s : CGB.SensoryMaintenance.Stim) (r : Fin 16),
      CGB.LanguageLifeKernel.SensoryRecord.read (CGB.LanguageLifeKernel.SensoryRecord.write s r) =
        CGB.SensoryMaintenance.next s (CGB.LanguageLifeKernel.SensoryRecord.read r)
```

## CGB.LanguageLifeKernel.SensoryRecord.action_content_distinguishes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L380)

解释：在该 16 状态见证中，若两条记录读出的策略动作和内容成本均相同，则两条记录相同。

源码说明：All sixteen records already have distinct action/content observations. Any exact representation preserving both must distinguish all sixteen.

形式陈述：

```lean
∀ (r r' : Fin 16),
  CGB.RenewableLearning.policy (CGB.LanguageLifeKernel.SensoryRecord.read r).1 =
      CGB.RenewableLearning.policy (CGB.LanguageLifeKernel.SensoryRecord.read r').1 →
    CGB.SensoryMaintenance.cost (CGB.LanguageLifeKernel.SensoryRecord.read r) =
        CGB.SensoryMaintenance.cost (CGB.LanguageLifeKernel.SensoryRecord.read r') →
      r = r'
```

## CGB.LanguageLifeKernel.SensoryRecord.record_state_lower_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L386)

解释：能够同时无损读取决策动作与维持成本的感知记录载体，其状态空间基数至少为16。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {R : Type u_1} [inst : Fintype R] (record : Fin 16 → R) (readAction : R → CGB.RenewableLearning.Action)
  (readContent : R → ℕ),
  (∀ (r : Fin 16),
      readAction (record r) = CGB.RenewableLearning.policy (CGB.LanguageLifeKernel.SensoryRecord.read r).1) →
    (∀ (r : Fin 16),
        readContent (record r) = CGB.SensoryMaintenance.cost (CGB.LanguageLifeKernel.SensoryRecord.read r)) →
      16 ≤ Fintype.card R
```

## CGB.LanguageLifeKernel.SensoryRecord.fold_refines

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L400)

解释：对编码记录连续写入感官输入，再读出，等于在原感官状态上连续执行对应更新。

源码说明：The finite record supports arbitrary-length replay by a forward fold, without consulting the world regime or keeping an autograd history.

形式陈述：

```lean
∀ (inputs : List CGB.SensoryMaintenance.Stim) (r : Fin 16),
  CGB.LanguageLifeKernel.SensoryRecord.read
      (List.foldl (fun code s => CGB.LanguageLifeKernel.SensoryRecord.write s code) r inputs) =
    List.foldl (fun x s => CGB.SensoryMaintenance.next s x) (CGB.LanguageLifeKernel.SensoryRecord.read r) inputs
```

## CGB.LanguageLifeKernel.SensoryRecord.joint_successor_reconstruct

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L412)

解释：原联合机体-世界演化过程中的每个正概率后继状态均可由内部感知记录精确编码与无损重构。

源码说明：Every positive-probability successor in the original joint body/world process is exactly representable, for either registered cut and any schedule. The environment and event label remain outside the body record.

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim) (t : ℕ) (r : Fin 16)
  (w : CGB.RenewableLearning.Env) (z' : CGB.SensoryMaintenance.S) (e : Option Unit),
  0 < (CGB.SensoryMaintenance.process cut sched).step t (fun x => CGB.LanguageLifeKernel.SensoryRecord.read r, w) e z' →
    CGB.LanguageLifeKernel.SensoryRecord.read (CGB.LanguageLifeKernel.SensoryRecord.encode (z'.1 ())) = z'.1 ()
```

## _private.CGBAxioms.LanguageLifeKernel.0.CGB.LanguageLifeKernel.CausalRecord.policy_cases

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L451)

解释：给出因果记录模块中细胞各离散构型对应决策策略动作的显式分情况查找表。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell),
  CGB.RenewableLearning.policy q =
    match q with
    | CGB.RenewableLearning.Cell.aSpent => 0
    | CGB.RenewableLearning.Cell.aReady => 0
    | CGB.RenewableLearning.Cell.aHeld => 0
    | CGB.RenewableLearning.Cell.bSpent => 1
    | CGB.RenewableLearning.Cell.bReady => 1
    | CGB.RenewableLearning.Cell.bHeld => 1
    | x => 2
```

## CGB.LanguageLifeKernel.CausalRecord.weights_exact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L458)

解释：对于非空主体状态，给定世界转移权重恰为因果记录整数权重除以二。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (w v : CGB.RenewableLearning.Env) (q : CGB.RenewableLearning.Cell),
  q ≠ CGB.RenewableLearning.Cell.vacuum →
    CGB.GraceRecovery.moves w (fun x => q) v =
      ↑(CGB.LanguageLifeKernel.CausalRecord.weight w (CGB.RenewableLearning.policy q) v) / 2
```

## CGB.LanguageLifeKernel.CausalRecord.sensor_alias_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L467)

解释：存在相同当前传感器读数但在相同动作下产生不同环境反馈后果的见证实例（传感器别名混淆）。

源码说明：Same present sensor, different consequences of the same actuator setting.

形式陈述：

```lean
CGB.RenewableLearning.sense CGB.RenewableLearning.Env.feedA () =
    CGB.RenewableLearning.sense CGB.RenewableLearning.Env.feedB () ∧
  CGB.LanguageLifeKernel.CausalRecord.observeWeight CGB.RenewableLearning.Env.feedA 0
        CGB.RenewableLearning.Signal.feed =
      2 ∧
    CGB.LanguageLifeKernel.CausalRecord.observeWeight CGB.RenewableLearning.Env.feedB 0
        CGB.RenewableLearning.Signal.feed =
      0
```

## CGB.LanguageLifeKernel.CausalRecord.update_sound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L478)

解释：对于非空主体状态，真实世界一步转移有正权重时，相应观测下的后继世界被因果记录更新保留。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (prior : Finset CGB.RenewableLearning.Env) (w v : CGB.RenewableLearning.Env) (q : CGB.RenewableLearning.Cell),
  q ≠ CGB.RenewableLearning.Cell.vacuum →
    w ∈ prior →
      0 < CGB.GraceRecovery.moves w (fun x => q) v →
        v ∈
          CGB.LanguageLifeKernel.CausalRecord.update prior (CGB.RenewableLearning.policy q)
            (CGB.RenewableLearning.sense v ())
```

## CGB.LanguageLifeKernel.CausalRecord.update_least

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L493)

解释：任何包含所有与先验、动作和观测相容后继世界的集合，都包含该因果记录更新结果。

源码说明：Any candidate set retaining every compatible successor contains this one. Thus the filter discards exactly the support exclusions licensed by the declared model and supplied evidence; it does not assert identifiability.

形式陈述：

```lean
∀ (prior posterior : Finset CGB.RenewableLearning.Env) (a : CGB.RenewableLearning.Action)
  (o : CGB.RenewableLearning.Signal),
  (∀ w ∈ prior,
      ∀ (v : CGB.RenewableLearning.Env),
        CGB.RenewableLearning.sense v () = o → 0 < CGB.LanguageLifeKernel.CausalRecord.weight w a v → v ∈ posterior) →
    CGB.LanguageLifeKernel.CausalRecord.update prior a o ⊆ posterior
```

## CGB.LanguageLifeKernel.CausalRecord.action_feedback_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L504)

解释：后续相同的传感器观测在前期采取不同动作时对应不同的因果环境后继状态解释。

源码说明：An identical later sensor has different interpretations when the earlier action differs. Keeping an existing action-bearing record can supply it; this does not mandate a separate action log or biological eligibility rule.

形式陈述：

```lean
CGB.LanguageLifeKernel.CausalRecord.update {CGB.RenewableLearning.Env.feedA, CGB.RenewableLearning.Env.feedB} 0
      CGB.RenewableLearning.Signal.miss =
    {CGB.RenewableLearning.Env.missB} ∧
  CGB.LanguageLifeKernel.CausalRecord.update {CGB.RenewableLearning.Env.feedA, CGB.RenewableLearning.Env.feedB} 1
      CGB.RenewableLearning.Signal.miss =
    {CGB.RenewableLearning.Env.missA}
```

## CGB.LanguageLifeKernel.PaidRecord.settlement_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L539)

解释：有偿记忆记录在结算时的剩余能量储备与扣除费用之和严格守恒，且合法候选假设集保持无损收缩。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {B : ℕ} (m : CGB.CausalLearning.Memory CGB.RenewableLearning.Env B) (a : CGB.RenewableLearning.Action)
  (o : CGB.RenewableLearning.Signal) (overhead : ℕ),
  ↑(CGB.LanguageLifeKernel.PaidRecord.settle m a o overhead).reserve +
        CGB.LanguageLifeKernel.PaidRecord.fee m a o overhead =
      ↑m.reserve ∧
    ∀ w ∈ m.candidates,
      CGB.LanguageLifeKernel.PaidRecord.possible w () a o = true →
        w ∈ (CGB.LanguageLifeKernel.PaidRecord.settle m a o overhead).candidates
```

## CGB.LanguageLifeKernel.PaidRecord.world_feedback_possible

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L558)

解释：非空主体在真实世界转移有正权重时，付费记录的可能性判据接受相应动作和后继观测。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (w v : CGB.RenewableLearning.Env) (q : CGB.RenewableLearning.Cell),
  q ≠ CGB.RenewableLearning.Cell.vacuum →
    0 < CGB.GraceRecovery.moves w (fun x => q) v →
      CGB.LanguageLifeKernel.PaidRecord.possible w () (CGB.RenewableLearning.policy q)
          (CGB.RenewableLearning.sense v ()) =
        true
```

## CGB.LanguageLifeKernel.PaidRecord.combined_write_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L574)

解释：综合有偿记录在单步演化中信用余额、写入能耗与手续费的综合支出受基础热力学第一定律预算严格约束。

源码说明：Add the recorder's separately endowed reserve to the ORIGINAL body's credit ledger. The original writes are not charged twice, and its credit is not silently reused to fund the added recorder. Capture/hold charges compose with this identity by the same prepayment subtraction.

形式陈述：

```lean
∀ {B : ℕ} (m : CGB.CausalLearning.Memory CGB.RenewableLearning.Env B) (s : CGB.SensoryMaintenance.Stim)
  (x : CGB.SensoryMaintenance.State) (a : CGB.RenewableLearning.Action) (o : CGB.RenewableLearning.Signal)
  (overhead : ℕ),
  CGB.RenewableLearning.credit (CGB.SensoryMaintenance.next s x).1 +
            ↑(CGB.LanguageLifeKernel.PaidRecord.settle m a o overhead).reserve +
          CGB.SensoryMaintenance.writes s x +
        CGB.SensoryMaintenance.lostCredit s x +
      CGB.LanguageLifeKernel.PaidRecord.fee m a o overhead =
    CGB.RenewableLearning.credit x.1 + ↑m.reserve + CGB.SensoryMaintenance.topups s x
```

## CGB.LanguageLifeKernel.PaidRecord.historical_frame_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L587)

解释：存在仅读取当前瞬间观测无法确定真实世界状态的历史因果歧义见证。

源码说明：Holding a source hypothesis is sound about its origin, not about a moving world. Also, substituting another action changes which source is compatible.

形式陈述：

```lean
CGB.LanguageLifeKernel.CausalRecord.weight CGB.RenewableLearning.Env.missA 0 CGB.RenewableLearning.Env.probeA = 2 ∧
  CGB.RenewableLearning.Env.probeA ∉ {CGB.RenewableLearning.Env.missA} ∧
    CGB.LanguageLifeKernel.PaidRecord.possible CGB.RenewableLearning.Env.feedA () 1 CGB.RenewableLearning.Signal.miss =
        true ∧
      CGB.LanguageLifeKernel.PaidRecord.possible CGB.RenewableLearning.Env.feedA () 0
          CGB.RenewableLearning.Signal.miss =
        false
```

## CGB.LanguageLifeKernel.PaidRecord.joint_source_possible

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L596)

解释：在联合状态演化下真实环境历史始终属于有偿记忆记录的可能候选集合。

源码说明：The ORIGINAL normal joint kernel, with any registered sensory schedule, licenses a receipt only for its pre-update body action and its successor's actual sensor. Whether that sensor is delivered is a separate port condition.

形式陈述：

```lean
∀ (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim) (t : ℕ) (z z' : CGB.SensoryMaintenance.S)
  (e : Option Unit),
  (z.1 ()).1 ≠ CGB.RenewableLearning.Cell.vacuum →
    0 < (CGB.SensoryMaintenance.process false sched).step t z e z' →
      CGB.LanguageLifeKernel.PaidRecord.possible z.2 () (CGB.RenewableLearning.policy (z.1 ()).1)
          (CGB.RenewableLearning.sense z'.2 ()) =
        true
```

## CGB.LanguageLifeKernel.PaidRecord.attach_refines

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L622)

解释：给已有交互机器附加更新记录后，投影到原状态仍细化原机器的行为。

源码说明：Projection recovers the whole original kernel, hence its registered finite-future observations by Interaction.Refines.sameBehavior. This theorem does not certify a particular recorder's sensor provenance or tariff physics.

形式陈述：

```lean
∀ {S : Type u_1} {I : Type u_2} {O : Type u_3} {R : Type u_4} [inst : Fintype S] [inst_1 : DecidableEq S]
  [inst_2 : Fintype O] [inst_3 : Fintype R] [inst_4 : DecidableEq R] (P : CGB.Interaction.Machine S I O)
  (advance : I → S → O → S → R → R),
  CGB.Interaction.Refines (CGB.LanguageLifeKernel.PaidRecord.attach P advance) P Prod.fst
```

## CGB.LanguageLifeKernel.LocalEdit.binary_factor_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L635)

解释：二元因子修改前后的交叉乘积差可精确分解为修改幅度 v−u 与联合表行列式 ad−bc 的乘积。

源码说明：A binary factor edit depends on joint contrast, not just either marginal. This algebraic identity specifies no history recurrence or world causal law.

形式陈述：

```lean
∀ {R : Type u_1} [inst : CommRing R] (a b c d u v : R),
  (b + v * d) * (a + b + u * c + u * d) - (b + u * d) * (a + b + v * c + v * d) = (v - u) * (a * d - b * c)
```

## CGB.LanguageLifeKernel.LocalEdit.accepts_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L648)

解释：当旧分布和候选分布总权重均为正时，整数交叉乘法接受判据等价于候选事件概率严格提高。

源码说明：With positive total masses, the executable comparison exactly detects improvement of the same event; inference and resource contracts are separate.

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] [inst_1 : DecidableEq S] (event : Finset S) (old candidate : S → ℕ),
  0 < ∑ s, old s →
    0 < ∑ s, candidate s →
      (CGB.LanguageLifeKernel.LocalEdit.accepts event old candidate = true ↔
        (∑ s ∈ event, ↑(old s)) / ∑ s, ↑(old s) < (∑ s ∈ event, ↑(candidate s)) / ∑ s, ↑(candidate s))
```

## CGB.LanguageLifeKernel.FirstHit.first_hit_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L680)

解释：首次命中表达式 1−(1−r)^t 可写成 r 乘有限几何级数。

源码说明：Conditional on a fixed lifetime, first-success probabilities telescope. The equality is algebraic and remains valid at rates zero and one.

形式陈述：

```lean
∀ (r : ℝ) (t : ℕ), 1 - (1 - r) ^ t = r * ∑ n ∈ Finset.range t, (1 - r) ^ n
```

## CGB.LanguageLifeKernel.FirstHit.truncation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L691)

解释：在权重非负且单步命中率介于零和一之间时，将寿命截断到有限时限不会增加首次命中成功率。

源码说明：Stop unfolding after `horizon` trials. The uncomputed success probability is at most the probability of both surviving the horizon and failing its trials. No tail error bound is assumed as a premise.

形式陈述：

```lean
∀ {I : Type u_1} [inst : Fintype I] (weight : I → ℝ) (lifetime : I → ℕ) (r : ℝ) (horizon : ℕ),
  (∀ (i : I), 0 ≤ weight i) →
    0 ≤ r →
      r ≤ 1 →
        CGB.LanguageLifeKernel.FirstHit.success weight (fun i => min (lifetime i) horizon) r ≤
            CGB.LanguageLifeKernel.FirstHit.success weight lifetime r ∧
          CGB.LanguageLifeKernel.FirstHit.success weight lifetime r ≤
            CGB.LanguageLifeKernel.FirstHit.success weight (fun i => min (lifetime i) horizon) r +
              (1 - r) ^ horizon * CGB.LanguageLifeKernel.FirstHit.survival weight lifetime horizon
```

## CGB.LanguageLifeKernel.FirstHit.probability_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L720)

解释：当权重非负且归一、单步命中率位于零和一之间时，加权首次命中成功率也位于零和一之间。

源码说明：A normalized nonnegative finite lifetime distribution gives a genuine probability, independently of the chosen support representation.

形式陈述：

```lean
∀ {I : Type u_1} [inst : Fintype I] (weight : I → ℝ) (lifetime : I → ℕ) (r : ℝ),
  (∀ (i : I), 0 ≤ weight i) →
    ∑ i, weight i = 1 →
      0 ≤ r →
        r ≤ 1 →
          0 ≤ CGB.LanguageLifeKernel.FirstHit.success weight lifetime r ∧
            CGB.LanguageLifeKernel.FirstHit.success weight lifetime r ≤ 1
```

## CGB.LanguageLifeKernel.CausalState.realizes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L768)

解释：因果状态编码能够读出历史所需输出，并且处理一个输入后与更新历史再编码一致。

源码说明：Canonical readout and single-input recursion realize the same contract.

形式陈述：

```lean
∀ {U : Type u_1} {O : Type u_2} (required : List U → O) (h : List U) (u : U),
  CGB.LanguageLifeKernel.CausalState.read required (CGB.LanguageLifeKernel.CausalState.encode required h) = required h ∧
    CGB.LanguageLifeKernel.CausalState.advance required (CGB.LanguageLifeKernel.CausalState.encode required h) u =
      CGB.LanguageLifeKernel.CausalState.encode required (h ++ [u])
```

## CGB.LanguageLifeKernel.CausalState.fold_refines

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L778)

解释：对因果状态逐个处理一个后缀，得到的状态等于先把后缀接到历史上再编码的状态。

源码说明：Arbitrarily long forward recursion; not a bounded-memory implementation.

形式陈述：

```lean
∀ {U : Type u_1} {O : Type u_2} (required : List U → O) (h suffix : List U),
  List.foldl (CGB.LanguageLifeKernel.CausalState.advance required)
      (CGB.LanguageLifeKernel.CausalState.encode required h) suffix =
    CGB.LanguageLifeKernel.CausalState.encode required (h ++ suffix)
```

## CGB.LanguageLifeKernel.CausalState.minimal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L787)

解释：任何准确实现指定历史输出的状态机都能满射到相应因果状态，表达该因果状态表示的最小性。

源码说明：Every exact deterministic realization projects onto the canonical states.

形式陈述：

```lean
∀ {U : Type u_1} {O : Type u_2} {S : Type u_3} (required : List U → O) (next : S → U → S) (initial : S) (out : S → O),
  (∀ (h : List U), out (List.foldl next initial h) = required h) →
    ∃ project,
      (∀ (h : List U), project (List.foldl next initial h) = CGB.LanguageLifeKernel.CausalState.encode required h) ∧
        Function.Surjective project
```

## CGB.LanguageLifeKernel.CausalState.finite_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L807)

解释：若有限状态机能对所有历史准确实现指定输出，则所需因果状态有限且数量不超过实现机器的状态数。

源码说明：Finiteness follows from a finite exact realization; it is not assumed.

形式陈述：

```lean
∀ {U : Type u_1} {O : Type u_2} {S : Type u_3} [inst : Fintype S] (required : List U → O) (next : S → U → S)
  (initial : S) (out : S → O),
  (∀ (h : List U), out (List.foldl next initial h) = required h) →
    Finite ↑(CGB.LanguageLifeKernel.CausalState.State required) ∧
      Nat.card ↑(CGB.LanguageLifeKernel.CausalState.State required) ≤ Fintype.card S
```

## CGB.LanguageLifeKernel.CausalState.finite_alphabets_unbounded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L817)

解释：即使输入为有限布尔字母表，要求判断真值与假值出现次数是否相等的历史也具有无限多个可区分因果状态。

源码说明：Finite input and output alphabets alone do not imply finite exact memory.

形式陈述：

```lean
Infinite ↑(CGB.LanguageLifeKernel.CausalState.State fun h => decide (List.count true h = List.count false h))
```

## CGB.LanguageLifeKernel.CausalState.separated_histories_capacity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L830)

解释：导出为区分非等价历史轨迹，因果状态记录载体所需容量的严格下界。

源码说明：Uniform precision requires distinct readable codes for separated futures. Codes precede suffix choice; query-dependent messages need fixed-query instances.

形式陈述：

```lean
∀ {U : Type u_4} {O : Type u_5} {I : Type u_6} {S : Type u_7} [inst : PseudoMetricSpace O] (required : List U → O)
  (history : I → List U) (code : List U → S) (decode : S → List U → O) (legal : I → List U → Prop) (ε : ℝ),
  (∀ (i : I) (suffix : List U),
      legal i suffix → dist (required (history i ++ suffix)) (decode (code (history i)) suffix) ≤ ε) →
    (∀ (i j : I),
        i ≠ j →
          ∃ suffix,
            legal i suffix ∧
              legal j suffix ∧ ε + ε < dist (required (history i ++ suffix)) (required (history j ++ suffix))) →
      Function.Injective (code ∘ history) ∧ ∀ [inst : Fintype I] [inst_1 : Fintype S], Fintype.card I ≤ Fintype.card S
```

## CGB.LanguageLifeKernel.Approximation.fold_error

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L865)

解释：若每步近似误差至多按因子 L 放大并增加 δ，有限输入序列后的总误差满足几何累积上界。

源码说明：Full one-step error certificates compose over any finite continuation.

形式陈述：

```lean
∀ {X : Type u_1} {Z : Type u_2} {U : Type u_3} (reference : X → U → X) (reduced : Z → U → Z) (error : X → Z → ℝ)
  (L δ : ℝ),
  0 ≤ L →
    (∀ (x : X) (z : Z) (u : U), error (reference x u) (reduced z u) ≤ L * error x z + δ) →
      ∀ (x : X) (z : Z) (inputs : List U),
        error (List.foldl reference x inputs) (List.foldl reduced z inputs) ≤
          L ^ inputs.length * error x z + δ * ∑ k ∈ Finset.range inputs.length, L ^ k
```

## CGB.LanguageLifeKernel.Approximation.invariant_error

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L884)

解释：若单步误差递推满足 Lε+δ≤ε，则从误差不超过 ε 出发，任意有限输入序列后误差仍不超过 ε。

源码说明：A locally closed error budget stays uniform in elapsed sequence length.

形式陈述：

```lean
∀ {X : Type u_1} {Z : Type u_2} {U : Type u_3} (reference : X → U → X) (reduced : Z → U → Z) (error : X → Z → ℝ)
  (L δ ε : ℝ),
  0 ≤ L →
    (∀ (x : X) (z : Z) (u : U), error (reference x u) (reduced z u) ≤ L * error x z + δ) →
      L * ε + δ ≤ ε →
        ∀ (x : X) (z : Z),
          error x z ≤ ε → ∀ (inputs : List U), error (List.foldl reference x inputs) (List.foldl reduced z inputs) ≤ ε
```
