# GenerationReciprocalCredit

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationReciprocalCredit.lean) · [返回导航](../README.md)

模块源码说明：Reciprocal local credit This is the common algebraic core for every slow map in a generated local medium.

## CGB.Generation.reciprocal_contact_response_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReciprocalCredit.lean#L41)

解释：The exact local forward response of one reciprocal contact. It is the pre/post eligibility product, derived from the actual contact read rather than stipulated as a biological analogy.

源码说明：The exact local forward response of one reciprocal contact. It is the pre/post eligibility product, derived from the actual contact read rather than stipulated as a biological analogy.

形式陈述：

```lean
∀ {Target : Type u_1} {Source : Type u_2} [inst : Fintype Target] [inst_1 : Fintype Source]
  [inst_2 : DecidableEq Target] [inst_3 : DecidableEq Source] (post : Target → ℝ) (contact : Target → Source → ℝ)
  (pre : Source → ℝ) (target : Target) (source : Source),
  HasDerivAt
    (fun θ =>
      CGB.Generation.reciprocalContactRead post (CGB.Generation.reciprocalContactPerturb contact target source θ) pre)
    (post target * pre source) 0
```

## CGB.Generation.reciprocal_contact_teaching_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReciprocalCredit.lean#L98)

解释：Scaling a local response by a fresh boundary teaching signal preserves its identity as the exact directional derivative. This is the abstract three-factor form: pre × post × outcome.

源码说明：Scaling a local response by a fresh boundary teaching signal preserves its identity as the exact directional derivative. This is the abstract three-factor form: pre × post × outcome.

形式陈述：

```lean
∀ {Target : Type u_1} {Source : Type u_2} [inst : Fintype Target] [inst_1 : Fintype Source]
  [inst_2 : DecidableEq Target] [inst_3 : DecidableEq Source] (post : Target → ℝ) (contact : Target → Source → ℝ)
  (pre : Source → ℝ) (target : Target) (source : Source) (teaching : ℝ),
  HasDerivAt
    (fun η =>
      CGB.Generation.reciprocalContactRead post
        (CGB.Generation.reciprocalContactPerturb contact target source (η * teaching)) pre)
    (teaching * post target * pre source) 0
```
