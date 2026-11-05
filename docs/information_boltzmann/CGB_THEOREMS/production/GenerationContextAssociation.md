# GenerationContextAssociation

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean) · [返回导航](../README.md)

模块源码说明：Context association: what a language memory must write, and where `GDNLife.assimilate` writes an arrival at that arrival's own address and adds its value.

## CGB.GDNLife.state_ext

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L34)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：GDN 状态的外延相等性：当自然参数、精度、速度、最后字节及参数头全部分量相同时，两状态严格相等。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (s t : CGB.GDNLife.State k d),
  s.natural = t.natural →
    s.precision = t.precision →
      s.velocity = t.velocity →
        s.velocityPrecision = t.velocityPrecision →
          s.lastByte = t.lastByte →
            s.lastObservationPrecision = t.lastObservationPrecision → s.head = t.head → s.bias = t.bias → s = t
```

## CGB.GDNLife.foldl_assimilate_natural

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L46)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在 GDN 吸收动力学中，多步到达序列累积后的自然参数矩阵精确等于初态加上各步输入强度与特征值乘积的总和。

源码说明：Natural parameters after a list of arrivals are the initial array plus one summand per arrival. Order enters only through a list sum.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (l : List CGB.GDNLife.Arrival) (s : CGB.GDNLife.State k d) (j : Fin k)
  (i : Fin d),
  (List.foldl (CGB.GDNLife.assimilate M) s l).natural j i =
    s.natural j i + (List.map (fun a => CGB.GDNLife.incoming M a j * M.value a.byte i) l).sum
```

## CGB.GDNLife.foldl_assimilate_precision

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L58)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在 GDN 吸收动力学中，多步到达后各槽位的精度标量精确等于初始精度加上各步到达输入权重的总和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (l : List CGB.GDNLife.Arrival) (s : CGB.GDNLife.State k d) (j : Fin k),
  (List.foldl (CGB.GDNLife.assimilate M) s l).precision j =
    s.precision j + (List.map (fun a => CGB.GDNLife.incoming M a j) l).sum
```

## CGB.GDNLife.foldl_assimilate_velocity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L70)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在单纯到达吸收过程中，GDN 状态的速度、速度精度、先验头权重及偏置项保持不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (l : List CGB.GDNLife.Arrival) (s : CGB.GDNLife.State k d),
  (List.foldl (CGB.GDNLife.assimilate M) s l).velocity = s.velocity ∧
    (List.foldl (CGB.GDNLife.assimilate M) s l).velocityPrecision = s.velocityPrecision ∧
      (List.foldl (CGB.GDNLife.assimilate M) s l).head = s.head ∧
        (List.foldl (CGB.GDNLife.assimilate M) s l).bias = s.bias
```

## CGB.GDNLife.assimilate_last_fields

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L81)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在 GDN 物理模型中，正观测精度的到达事件被吸收后，后继状态的最后到达字节与对应精度精确记录当前到达值。

源码说明：A positive-confidence arrival replaces both last-arrival fields.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (c : CGB.GDNLife.Arrival),
  0 < c.observationPrecision →
    (CGB.GDNLife.assimilate M s c).lastByte = some c.byte ∧
      (CGB.GDNLife.assimilate M s c).lastObservationPrecision = some c.observationPrecision
```

## CGB.GDNLife.assimilate_order_blind

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L93)

分类：necessity；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：GDNLife 无上下文纯词袋吸收无法区分词序的不可行性证明。

解释：在无上下文的 GDN 吸收模型中，任意置换的到达序列在追加相同正精度到达后产生完全相同的吸收状态（词序盲目性）。

源码说明：**Order blindness.** Any reordering of the arrivals before a common positive-confidence arrival reaches exactly the same complete state. Elapsed time acts only through `prior`, which is the identity at zero elapsed time, so this is the state reached across the zero-elapsed bytes of one packet.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) {l l' : List CGB.GDNLife.Arrival},
  l.Perm l' →
    ∀ (c : CGB.GDNLife.Arrival),
      0 < c.observationPrecision →
        List.foldl (CGB.GDNLife.assimilate M) s (l ++ [c]) = List.foldl (CGB.GDNLife.assimilate M) s (l' ++ [c])
```

## CGB.GDNLife.order_blind_same_allocation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L118)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在无上下文 GDN 动力学中，置换序列吸收后导出的后继资源分配与预测概率完全相等。

源码说明：Order-blind states issue the same byte distribution.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) {l l' : List CGB.GDNLife.Arrival},
  l.Perm l' →
    ∀ (c : CGB.GDNLife.Arrival),
      0 < c.observationPrecision →
        CGB.GDNLife.allocation M (List.foldl (CGB.GDNLife.assimilate M) s (l ++ [c])) =
          CGB.GDNLife.allocation M (List.foldl (CGB.GDNLife.assimilate M) s (l' ++ [c]))
```

## CGB.GDNLife.no_read_separates_order

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L127)

分类：necessity；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：GDNLife 无上下文纯词袋吸收无法区分词序的不可行性证明。

解释：在 GDN 词序盲目模型中，若两置换序列要求不同输出，则不存在任何仅依赖当前吸收状态的读出函数能将二者区分。

源码说明：**Necessity.** No readout of this state can serve a requirement that separates two orderings. This is the state-merging half of `causal_carrier_contract_iff`, instantiated at the actual GDN transition.

形式陈述：

```lean
∀ {k d : ℕ} {C : Type u_1} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) {l l' : List CGB.GDNLife.Arrival},
  l.Perm l' →
    ∀ (c : CGB.GDNLife.Arrival),
      0 < c.observationPrecision →
        ∀ (required : List CGB.GDNLife.Arrival → C),
          required (l ++ [c]) ≠ required (l' ++ [c]) →
            ¬∃ read, ∀ (h : List CGB.GDNLife.Arrival), read (List.foldl (CGB.GDNLife.assimilate M) s h) = required h
```

## CGB.Generation.ContextAssociation.foldl_write_natural

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L185)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在上下文关联写入下，多步序列后的自然参数矩阵等于初始矩阵加上各步在对应上下文寻址调制下的自然参数增量之和。

源码说明：A cell's natural total is the evidence-weighted sum of the values that followed its contexts. This is a decomposition of the stored number.

形式陈述：

```lean
∀ {Ctx : Type u_1} {k d : ℕ} (M : CGB.Generation.ContextAssociation.Model Ctx k d)
  (l : List CGB.Generation.ContextAssociation.Byte) (s : CGB.Generation.ContextAssociation.State Ctx k d) (j : Fin k)
  (i : Fin d),
  (List.foldl (CGB.Generation.ContextAssociation.write M) s l).natural j i =
    s.natural j i + CGB.Generation.ContextAssociation.cellNatural M j i s.context l
```

## CGB.Generation.ContextAssociation.foldl_write_precision

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L196)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在上下文关联写入下，多步序列后的精度向量等于初始精度加上各步在对应上下文寻址调制下的精度增量之和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {k d : ℕ} (M : CGB.Generation.ContextAssociation.Model Ctx k d)
  (l : List CGB.Generation.ContextAssociation.Byte) (s : CGB.Generation.ContextAssociation.State Ctx k d) (j : Fin k),
  (List.foldl (CGB.Generation.ContextAssociation.write M) s l).precision j =
    s.precision j + CGB.Generation.ContextAssociation.cellPrecision M j s.context l
```

## CGB.Generation.ContextAssociation.cell_mean_is_conditional_statistic

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L209)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在上下文关联模型中，多步写入后各单元均值严格等于累积自然参数与累积精度的比值，构成条件充分统计量。

源码说明：**Conditional statistic.** A cell's read is the precision-weighted mean of what followed its contexts, combined with its prior.

形式陈述：

```lean
∀ {Ctx : Type u_1} {k d : ℕ} (M : CGB.Generation.ContextAssociation.Model Ctx k d)
  (l : List CGB.Generation.ContextAssociation.Byte) (s : CGB.Generation.ContextAssociation.State Ctx k d) (j : Fin k)
  (i : Fin d),
  CGB.Generation.ContextAssociation.mean (List.foldl (CGB.Generation.ContextAssociation.write M) s l) j i =
    (s.natural j i + CGB.Generation.ContextAssociation.cellNatural M j i s.context l) /
      (s.precision j + CGB.Generation.ContextAssociation.cellPrecision M j s.context l)
```

## CGB.Generation.ContextAssociation.hard_cell_innovation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L217)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当上下文地址对某槽位硬选通时，该槽位均值更新严格等于原均值加上由证据权重和原精度决定的卡尔曼型新息项。

源码说明：A hard-addressed write is the Bayesian innovation step at that cell.

形式陈述：

```lean
∀ {Ctx : Type u_1} {k d : ℕ} (M : CGB.Generation.ContextAssociation.Model Ctx k d)
  (s : CGB.Generation.ContextAssociation.State Ctx k d) (b : CGB.Generation.ContextAssociation.Byte) (j : Fin k)
  (i : Fin d),
  M.address s.context j = 1 →
    0 < s.precision j →
      CGB.Generation.ContextAssociation.mean (CGB.Generation.ContextAssociation.write M s b) j i =
        CGB.Generation.ContextAssociation.mean s j i +
          M.evidence b / (s.precision j + M.evidence b) * (M.value b i - CGB.Generation.ContextAssociation.mean s j i)
```

## CGB.Generation.ContextAssociation.write_outside_support

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L232)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当写入时的上下文地址在某槽位权重为零时，该槽位的均值在写入后严格保持原值不变。

源码说明：Writes leave every cell outside the writing context's support unchanged.

形式陈述：

```lean
∀ {Ctx : Type u_1} {k d : ℕ} (M : CGB.Generation.ContextAssociation.Model Ctx k d)
  (s : CGB.Generation.ContextAssociation.State Ctx k d) (b : CGB.Generation.ContextAssociation.Byte) (j : Fin k)
  (i : Fin d),
  M.address s.context j = 0 →
    CGB.Generation.ContextAssociation.mean (CGB.Generation.ContextAssociation.write M s b) j i =
      CGB.Generation.ContextAssociation.mean s j i
```

## CGB.Generation.ContextAssociation.disjoint_context_read_unchanged

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L239)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当写入时的上下文地址与读取上下文地址在所有槽位上正交无重叠时，关联读出值在写入后严格保持不变。

源码说明：**No interference.** A context whose support is disjoint from the writer reads exactly what it read before. Its conditional statistic is its own.

形式陈述：

```lean
∀ {Ctx : Type u_1} {k d : ℕ} (M : CGB.Generation.ContextAssociation.Model Ctx k d)
  (s : CGB.Generation.ContextAssociation.State Ctx k d) (b : CGB.Generation.ContextAssociation.Byte) (c : Ctx)
  (i : Fin d),
  (∀ (j : Fin k), M.address s.context j * M.address c j = 0) →
    CGB.Generation.ContextAssociation.read M (CGB.Generation.ContextAssociation.write M s b) c i =
      CGB.Generation.ContextAssociation.read M s c i
```

## CGB.Generation.ContextAssociation.context_write_separates_order

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L252)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在上下文关联模型中，当上下文寻址能够区分前序字符时，两字符按不同顺序写入后的自然参数矩阵产生严格差异，打破词序盲目性。

源码说明：**Order separation.** Unlike `assimilate_order_blind`, the context write holds the order of two arrivals whenever the cell of one successor context sees the value of the other arrival.

形式陈述：

```lean
∀ {Ctx : Type u_1} {k d : ℕ} (M : CGB.Generation.ContextAssociation.Model Ctx k d)
  (s : CGB.Generation.ContextAssociation.State Ctx k d) (a b : CGB.Generation.ContextAssociation.Byte) (j : Fin k)
  (i : Fin d),
  M.address s.context j = 0 →
    M.address (M.advance s.context a) j = 1 →
      M.address (M.advance s.context b) j = 0 →
        0 < M.evidence b →
          M.value b i ≠ 0 →
            (List.foldl (CGB.Generation.ContextAssociation.write M) s [a, b]).natural j i ≠
              (List.foldl (CGB.Generation.ContextAssociation.write M) s [b, a]).natural j i
```

## CGB.Generation.ContextAssociation.same_address_same_read

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L271)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在上下文关联模型中，具有完全相同寻址权重的两个上下文在相同状态下导出严格相等的读出向量。

源码说明：**Address must refine what the future separates.** Two contexts sharing an address share one prediction, and a common prediction pays at least this squared excess across two different required means. With `collision_of_excess_keys`, this is why the key count bounds the separable contexts.

形式陈述：

```lean
∀ {Ctx : Type u_1} {k d : ℕ} (M : CGB.Generation.ContextAssociation.Model Ctx k d)
  (s : CGB.Generation.ContextAssociation.State Ctx k d) (c c' : Ctx),
  M.address c = M.address c' →
    ∀ (i : Fin d), CGB.Generation.ContextAssociation.read M s c i = CGB.Generation.ContextAssociation.read M s c' i
```

## CGB.Generation.ContextAssociation.common_prediction_excess

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L275)

分类：side；适用范围：universal；审核：reviewed

分类说明：纯代数、数值计算或概率混合等式辅助引理，与语言身体机制无直接因果约束关系。

解释：两不同目标值与其共同预测之间二次误差的超额下界，满足平方差的一半受二者分别偏差平方和控制的实数代数不等式。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p m m' : ℝ), (m - m') ^ 2 / 2 ≤ (p - m) ^ 2 + (p - m') ^ 2
```

## CGB.Generation.ContextAssociation.common_suffix_distance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L281)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在单步常数 L 的度量空间递推中，任意有限输入序列诱导的多步复合转移保持几何距离放缩上界。

源码说明：**Fading context.** If each arrival contracts distance, a common suffix of arrivals shrinks the difference geometrically, whatever the older history.

形式陈述：

```lean
∀ {X : Type u_2} [inst : PseudoMetricSpace X] (adv : X → CGB.Generation.ContextAssociation.Byte → X) {ρ : ℝ},
  0 ≤ ρ →
    (∀ (b : CGB.Generation.ContextAssociation.Byte) (x y : X), dist (adv x b) (adv y b) ≤ ρ * dist x y) →
      ∀ (l : List CGB.Generation.ContextAssociation.Byte) (x y : X),
        dist (List.foldl adv x l) (List.foldl adv y l) ≤ ρ ^ l.length * dist x y
```

## CGB.Generation.ContextAssociation.common_suffix_address_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L298)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在满足利普希茨常数 L 的上下文递推动力学中，共同后缀序列推进后的距离受初始状态距离乘以 L 的长度幂次上界约束。

源码说明：**Repeatable address.** A Lipschitz address of a fading context returns to within this bound after a common suffix.

形式陈述：

```lean
∀ {X : Type u_2} {Y : Type u_3} [inst : PseudoMetricSpace X] [inst_1 : PseudoMetricSpace Y]
  (adv : X → CGB.Generation.ContextAssociation.Byte → X) (q : X → Y) {ρ L : ℝ},
  0 ≤ ρ →
    0 ≤ L →
      (∀ (b : CGB.Generation.ContextAssociation.Byte) (x y : X), dist (adv x b) (adv y b) ≤ ρ * dist x y) →
        (∀ (x y : X), dist (q x) (q y) ≤ L * dist x y) →
          ∀ (l : List CGB.Generation.ContextAssociation.Byte) (x y : X),
            dist (q (List.foldl adv x l)) (q (List.foldl adv y l)) ≤ L * ρ ^ l.length * dist x y
```

## CGB.Generation.ContextAssociation.held_coordinate_survives_suffix

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L314)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：若投影坐标在给定后缀序列的转移下保持守恒不变，则该坐标上可区分的两个初态经过整段序列演化后依然保持可区分。

源码说明：**A held coordinate never fades.** If arrivals leave a coordinate untouched, two contexts differing there still differ after every common suffix, so an address reading it is not a function of recent language. This is the `torch.where(local_write_support, proposed, activity)` branch at unrouted sites.

形式陈述：

```lean
∀ {X : Type u_2} {Y : Type u_3} (adv : X → CGB.Generation.ContextAssociation.Byte → X) (proj : X → Y)
  (l : List CGB.Generation.ContextAssociation.Byte),
  (∀ (x : X), ∀ b ∈ l, proj (adv x b) = proj x) →
    ∀ (x y : X), proj x ≠ proj y → proj (List.foldl adv x l) ≠ proj (List.foldl adv y l)
```

## CGB.Generation.ContextAssociation.contraction_has_no_two_latches

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L332)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在严格压缩比小于一的度量空间更新中，任意输入下不可能同时存在两个不同的自锁存不动点。

源码说明：**Separation of roles.** A contraction under a held arrival has at most one rest point, so an address-forming state cannot also hold two latched alternatives. Persistence of learned distinctions must live elsewhere.

形式陈述：

```lean
∀ {X : Type u_2} [inst : MetricSpace X] (adv : X → CGB.Generation.ContextAssociation.Byte → X) {ρ : ℝ},
  ρ < 1 →
    (∀ (b : CGB.Generation.ContextAssociation.Byte) (x y : X), dist (adv x b) (adv y b) ≤ ρ * dist x y) →
      ∀ (b : CGB.Generation.ContextAssociation.Byte) (x y : X), adv x b = x → adv y b = y → x = y
```
