# NestedLife

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/NestedLife.lean) · [返回导航](../README.md)

模块源码说明：Life of a nested organism Once the organism is part of one medium with its world (`Nesting`), it has no outside stimulus: what it senses is the state of the world next to it, and what it takes in comes out of the world.

## CGB.Part.irrecoverable_absorbing

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/NestedLife.lean#L52)

解释：在不可逆吸收集内部，不可恢复状态在介质动力学步进下保持不可恢复，无法自发逃逸到可恢复集合。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {B : Type u_5} (M : CGB.Medium U Λ A I)
  (O : CGB.Part U Λ B) {x y : U → Λ}, ¬CGB.Part.Recoverable M O x → CGB.Part.Step M x y → ¬CGB.Part.Recoverable M O y
```

## CGB.Part.no_intake_of_closed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/NestedLife.lean#L58)

解释：若一个部件没有与外界进行代谢摄入的能力，则其在任意全局构型下都是不可恢复的。

源码说明：An organism that never exchanges content cannot take anything in, so nothing is recoverable for it: metabolism is necessary for nested life.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {B : Type u_5} (M : CGB.Medium U Λ A I)
  (O : CGB.Part U Λ B), ¬CGB.Part.Metabolizes M O → ∀ (x : U → Λ), ¬CGB.Part.Recoverable M O x
```
