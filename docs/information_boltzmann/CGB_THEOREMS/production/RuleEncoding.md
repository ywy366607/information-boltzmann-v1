# RuleEncoding

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean) · [返回导航](../README.md)

模块源码说明：A fixed finite rule language and an honest description budget This is a representation layer, not a seventh physical axiom.

## CGB.Description.code_roundtrip

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L52)

解释：通用描述规则书在自然数或比特编码序列与结构化对象之间满足无损双向编解码。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (b : CGB.Description.Book), Encodable.decode (Encodable.encode b) = some b
```

## CGB.Description.different_books_different_codes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L55)

解释：任意两本不同的通用描述规则书其生成的编码严格互异（单射性）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {a b : CGB.Description.Book}, Encodable.encode a = Encodable.encode b → a = b
```

## CGB.Description.bounded_books

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L60)

解释：占用比特数不超过B的所有合法描述规则书构成的集合其基数至多为2^B。

源码说明：At most 2^B distinct descriptions fit in B bits (an intentionally loose bound).

形式陈述：

```lean
∀ (B : ℕ) (s : Finset CGB.Description.Book), (∀ b ∈ s, CGB.Description.bits b ≤ B) → s.card ≤ 2 ^ B
```

## CGB.Description.one_percent_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L81)

解释：通用描述规则书所占比特数不超过总基因组容量1%的充要判定条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (b : CGB.Description.Book) (G : ℕ), CGB.Description.WithinBudget b G 1 100 ↔ 0 < G ∧ 100 * CGB.Description.bits b ≤ G
```

## CGB.Description.WiringReference.payload_length

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L94)

解释：规模为n个位点的布线拓扑参考图其有效负载数据长度严格等于n*n。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (g : CGB.Description.WiringReference n), g.payload.length = n * n
```

## CGB.Description.Book.findWithin_complete

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L119)

解释：在燃料不低于列表长度时，有限步谓词查找函数能够完全找到第一个满足条件的元素。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {D : Type u_1} (p : D → Bool) (xs : List D) (fuel : ℕ),
  xs.length ≤ fuel → CGB.Description.Book.findWithin p fuel xs = some (List.find? p xs)
```

## CGB.Description.Book.updateWithin_complete

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L153)

解释：在燃料充分条件下，带资源预算的局部状态更新函数与未设限的标准更新函数完全一致。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (b : CGB.Description.Book) (a : b.Action) (x : b.Site → b.State) (u : b.Site) (fuel : ℕ),
  (b.event a).ports.length ≤ fuel → b.updateWithin a x u fuel = some (b.update a x u)
```

## CGB.Description.Book.weight_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L161)

解释：描述规则书中每个离散动作的跃迁权重均严格非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (b : CGB.Description.Book) (a : b.Action), 0 ≤ b.weight a
```

## CGB.Description.Book.same_code_same_medium

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L189)

解释：规则书编码的一致性直接决定解释出的离散介质模型在数学上完全等价。

源码说明：Equality of codes entails equality of the whole interpreted instance. There is no arbitrary Medium carried beside a decorative short code.

形式陈述：

```lean
∀ {a b : CGB.Description.Book}, Encodable.encode a = Encodable.encode b → a.medium ≍ b.medium
```

## CGB.Description.Book.unchanged_outside

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L195)

解释：不在动作参与范围内的空间点在介质跃迁后保持状态不变，确立局域性是解释器的固有定理。

源码说明：Locality of effects is a theorem of the interpreter, not a user-supplied promise.

形式陈述：

```lean
∀ (b : CGB.Description.Book) (a : b.Action) (s : b.Site → b.Stim) (x : b.Site → b.State),
  ∀ u ∉ b.medium.parts a, b.medium.act a s x u = x u
```

## CGB.Description.Book.admits_some_clock

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L208)

解释：有限字母表与有限描述规则书必定存在某个统一的有界全局演化时钟准入速率。

源码说明：Finite alphabets and this finite rulebook admit some uniform clock. Whether it fits a *given* world's clock is still a separate admission obligation.

形式陈述：

```lean
∀ (b : CGB.Description.Book), ∃ R, b.medium.Clocked R
```
