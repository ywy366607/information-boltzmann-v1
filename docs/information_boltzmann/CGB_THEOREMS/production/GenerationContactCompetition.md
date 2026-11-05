# GenerationContactCompetition

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean) · [返回导航](../README.md)

模块源码说明：A shared finite contact reservoir `S` is already a bounded, typed local contact state.

## CGB.Generation.receipt_rate_in_unit_interval

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L31)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：非负回执标量通过 -expm1(-x) 映射后，输出严格落在 [0, 1] 合法转移速率区间内，满足共享池守恒与非负性公理的前置要求。

源码说明：A nonnegative receipt produces a legal exchange rate, even when its raw magnitude is unbounded. This is the precise range premise needed to apply the shared-pool conservation and nonnegativity laws to the executable ``-expm1(-x)`` conversion.

形式陈述：

```lean
∀ (x : ℝ), 0 ≤ x → 0 ≤ CGB.Generation.receiptRate x ∧ CGB.Generation.receiptRate x ≤ 1
```

## CGB.Generation.shared_contact_pool_total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L49)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：整个接触束遵循二态质量交换律；各槽位份额虽有差异，但其总和严格受限于同一隐式自由池的守恒约束。

源码说明：The whole contact bundle follows one two-state exchange. Coordinate shares may differ, but their sum can draw only from the same implicit pool.

形式陈述：

```lean
∀ {n : ℕ} (occupied target : Fin n → ℝ) (inflow release gate : ℝ),
  ∑ i, target i = 1 →
    ∑ i, CGB.Generation.sharedContactPoolStep occupied target inflow release gate i =
      ∑ i, occupied i + gate * (inflow * (1 - ∑ i, occupied i) - release * ∑ i, occupied i)
```

## CGB.Generation.full_shared_pool_blocks_admission

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L77)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当局部接触束的占用总和达到容量上限 1 时，任何正流入均无法再准入到任何接触槽位中，状态仅能通过释放通道发生改变。

源码说明：Once a local bundle is full, no positive admission can add to any of its contacts. Only its release channel can change it.

形式陈述：

```lean
∀ {n : ℕ} (occupied target : Fin n → ℝ) (inflow release gate : ℝ),
  ∑ j, occupied j = 1 →
    ∀ (i : Fin n),
      CGB.Generation.sharedContactPoolStep occupied target inflow release gate i =
        occupied i - gate * release * occupied i
```

## CGB.Generation.shared_pool_marginal_admission_is_anti_dominant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L89)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在正局部准入门控下，占用量更大的接触束具有严格更小的边际准入容量，构成防止赢家无限通吃的局部反垄断律。

源码说明：For a positive local admission gate, an already more occupied bundle has strictly less remaining marginal capacity. This is the local anti-dominance law required to prevent unlimited winner growth; it does not prescribe a global balancing scalar.

形式陈述：

```lean
∀ (gate inflow target q₁ q₂ : ℝ),
  0 < gate → 0 < inflow → 0 < target → q₁ < q₂ → gate * inflow * target * (1 - q₂) < gate * inflow * target * (1 - q₁)
```

## CGB.Generation.shared_pool_closed_gate_holds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L100)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当局部证据门关闭时，接触束各槽位状态完全保持，使得延迟证据可在不推进独立资格页或全局学习率的前提下调控现有 S 载体。

源码说明：When a local evidence gate is closed, the complete bundle holds exactly. This lets delayed evidence regulate the existing S carrier without advancing a separate trace or a global learning-rate state.

形式陈述：

```lean
∀ {n : ℕ} (occupied target : Fin n → ℝ) (inflow release : ℝ),
  CGB.Generation.sharedContactPoolStep occupied target inflow release 0 = occupied
```

## CGB.Generation.shared_contact_pool_vector_constant_release

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L114)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当逐坐标释放率取常数值时，向量化共享接触池退化并精确复现标量释放接口的动力学行为。

源码说明：Constant vector release recovers the original interface exactly.

形式陈述：

```lean
∀ {n : ℕ} (occupied target : Fin n → ℝ) (inflow release gate : ℝ),
  CGB.Generation.sharedContactPoolVectorStep occupied target (fun x => release) inflow gate =
    CGB.Generation.sharedContactPoolStep occupied target inflow release gate
```

## CGB.Generation.shared_contact_pool_vector_total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L122)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在目标向量归一化时，共享池各槽位后继占用总量精确等于旧占用总量加上门控有效流入并减去门控释放损耗。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ),
  ∑ i, target i = 1 →
    ∑ i, CGB.Generation.sharedContactPoolVectorStep occupied target release inflow gate i =
      ∑ i, occupied i + gate * inflow * (1 - ∑ i, occupied i) - gate * ∑ i, release i * occupied i
```

## CGB.Generation.shared_contact_pool_vector_conserves_total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L152)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在目标向量归一化条件下，共享池各槽位占用步进量与剩余空闲容量之和恒等于单位总量 1（总容量守恒）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ),
  ∑ i, target i = 1 →
    ∑ i, CGB.Generation.sharedContactPoolVectorStep occupied target release inflow gate i +
        CGB.Generation.sharedContactPoolFreeStep occupied release inflow gate =
      1
```

## CGB.Generation.shared_contact_pool_vector_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L161)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在初态非负且总和不超 1、目标非负、释放率处于 [0, 1]、流入非负及门控处于 [0, 1] 时，更新后的各槽位占用量严格非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ),
  (∀ (i : Fin n), 0 ≤ occupied i) →
    ∑ i, occupied i ≤ 1 →
      (∀ (i : Fin n), 0 ≤ target i) →
        (∀ (i : Fin n), 0 ≤ release i ∧ release i ≤ 1) →
          0 ≤ inflow →
            0 ≤ gate ∧ gate ≤ 1 →
              ∀ (i : Fin n), 0 ≤ CGB.Generation.sharedContactPoolVectorStep occupied target release inflow gate i
```

## CGB.Generation.shared_contact_pool_free_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L179)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在占用非负且总和不超 1、释放率非负、流入与门控处于 [0, 1] 的条件下，共享接触池更新后的剩余空闲容量严格非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied release : Fin n → ℝ) (inflow gate : ℝ),
  (∀ (i : Fin n), 0 ≤ occupied i) →
    ∑ i, occupied i ≤ 1 →
      (∀ (i : Fin n), 0 ≤ release i) →
        0 ≤ inflow ∧ inflow ≤ 1 →
          0 ≤ gate ∧ gate ≤ 1 → 0 ≤ CGB.Generation.sharedContactPoolFreeStep occupied release inflow gate
```

## CGB.Generation.shared_contact_pool_vector_total_le_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L191)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在占用非负且总和不超 1、目标归一、释放非负及参数规范在单位区间时，更新后的各槽位占用总量严格不超过 1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ),
  (∀ (i : Fin n), 0 ≤ occupied i) →
    ∑ i, occupied i ≤ 1 →
      ∑ i, target i = 1 →
        (∀ (i : Fin n), 0 ≤ release i) →
          0 ≤ inflow ∧ inflow ≤ 1 →
            0 ≤ gate ∧ gate ≤ 1 →
              ∑ i, CGB.Generation.sharedContactPoolVectorStep occupied target release inflow gate i ≤ 1
```

## CGB.Generation.full_shared_pool_vector_blocks_admission

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L203)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当共享池各槽位占用量之和达到单位容量 1 时，准入增量为零，各槽位后继状态仅由自身释放衰减 (1 - gate * release i) 决定。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ),
  ∑ j, occupied j = 1 →
    ∀ (i : Fin n),
      CGB.Generation.sharedContactPoolVectorStep occupied target release inflow gate i =
        (1 - gate * release i) * occupied i
```

## CGB.Generation.shared_pool_vector_closed_gate_holds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L212)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当门控系数为零时，共享接触池各槽位的向量更新步进严格保持原有占用分布不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow : ℝ),
  CGB.Generation.sharedContactPoolVectorStep occupied target release inflow 0 = occupied
```

## CGB.Generation.shared_contact_pool_vector_endpoint_agreement

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L220)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当两端点对同一共享状态应用相同输入时，绝不会在同一条边的接触池上写入不一致的状态副本。

源码说明：Endpoints applying the same inputs to the same shared state cannot write inconsistent copies of that edge's reservoir.

形式陈述：

```lean
∀ {n : ℕ} (left right target release : Fin n → ℝ) (inflow gate : ℝ),
  left = right →
    CGB.Generation.sharedContactPoolVectorStep left target release inflow gate =
      CGB.Generation.sharedContactPoolVectorStep right target release inflow gate
```

## CGB.Generation.shared_edge_response_gate_symmetric

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L229)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：共享边缘响应门控函数对左右两端响应具有严格对称性：输入互换后门控值不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (left right : ℝ), CGB.Generation.sharedEdgeResponseGate left right = CGB.Generation.sharedEdgeResponseGate right left
```

## CGB.Generation.shared_edge_response_gate_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L233)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当左右端点响应均处于单位区间 [0, 1] 时，共享边缘响应门控值严格保持在 [0, 1] 区间内。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (left right : ℝ),
  0 ≤ left ∧ left ≤ 1 →
    0 ≤ right ∧ right ≤ 1 →
      0 ≤ CGB.Generation.sharedEdgeResponseGate left right ∧ CGB.Generation.sharedEdgeResponseGate left right ≤ 1
```

## CGB.Generation.shared_edge_response_gate_closed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L238)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当两端点响应均为零时，共享边缘响应门控值精确等于零（双零完全关闭）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Generation.sharedEdgeResponseGate 0 0 = 0
```

## CGB.Generation.shared_edge_response_gate_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L241)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当一端响应达到饱和值 1 且另一端不超过 1 时，共享边缘响应门控值严格等于 1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ right ≤ 1, CGB.Generation.sharedEdgeResponseGate 1 right = 1
```
