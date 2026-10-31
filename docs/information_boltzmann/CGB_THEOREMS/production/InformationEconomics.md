# InformationEconomics

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/InformationEconomics.lean) · [返回导航](../README.md)

模块源码说明：Information economics: a small reusable basis The project has many mechanism-level theorems, but their common economic core is small.

## CGB.InformationEconomics.causal_carrier_contract_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InformationEconomics.lean#L33)

源码说明：**Causal carrier basis.** A finite state is a valid compressed carrier exactly when it supports both the required readout and the encoded successor for every future input.

形式陈述：

```lean
∀ {H : Type u_1} {S : Type u_2} {C : Type u_3} {U : Type u_4} (encode : H → S) (required : H → C) (advance : H → U → H)
  (fallbackC : C) (fallbackS : S),
  (∃ read next,
      (∀ (h : H), read (encode h) = required h) ∧ ∀ (h : H) (u : U), next (encode h) u = encode (advance h u)) ↔
    (∀ (h h' : H), encode h = encode h' → required h = required h') ∧
      ∀ (h h' : H), encode h = encode h' → ∀ (u : U), encode (advance h u) = encode (advance h' u)
```

## CGB.InformationEconomics.selective_write_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InformationEconomics.lean#L60)

源码说明：**Selective write basis.** Agreement over all live candidates consumes no write capacity; a funded observation that rules out a live candidate spends one unit and removes that candidate.

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H]
  (possible : CGB.CausalLearning.Support H C A O) {B : ℕ} (m : CGB.CausalLearning.Memory H B) (c : C) (a : A) (o : O),
  ((∀ h ∈ m.candidates, possible h c a o = true) →
      CGB.CausalLearning.advance possible m c a o = m ∧ CGB.CausalLearning.fee possible m c a o = 0) ∧
    (0 < ↑m.reserve →
      ∀ h ∈ m.candidates,
        possible h c a o = false →
          h ∉ (CGB.CausalLearning.advance possible m c a o).candidates ∧ CGB.CausalLearning.fee possible m c a o = 1)
```

## CGB.InformationEconomics.predictive_growth_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InformationEconomics.lean#L81)

源码说明：**Predictive-growth basis.** Under the declared reinvestment contract, growth decomposes into a fixed alphabet term minus cross-entropy and is bounded by the stream entropy. The contract is a niche assumption, not a general physical axiom.

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] [Nonempty B] (p c : B → ℝ),
  (∀ (b : B), 0 ≤ p b) →
    ∑ b, p b = 1 →
      (∀ (b : B), 0 < c b) →
        ∑ b, c b = 1 →
          CGB.PredictiveHarvest.growth p c = Real.log ↑(Fintype.card B) - CGB.PredictiveHarvest.crossEntropy p c ∧
            CGB.PredictiveHarvest.growth p c ≤ Real.log ↑(Fintype.card B) - CGB.PredictiveHarvest.entropy p
```
