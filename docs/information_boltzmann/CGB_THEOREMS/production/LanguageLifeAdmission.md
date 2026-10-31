# LanguageLifeAdmission

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeAdmission.lean) · [返回导航](../README.md)

模块源码说明：Historical finite-candidate language admission This is a contract, not an existence theorem.

## CGB.LanguageLifeAdmission.joint_admission

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LanguageLifeAdmission.lean#L97)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.LanguageLifeAdmission.LanguageCandidate Hist Model Ctx Act Obs Byte Answer),
  CGB.LanguageLifeAdmission.joint_admission_spec L
```
