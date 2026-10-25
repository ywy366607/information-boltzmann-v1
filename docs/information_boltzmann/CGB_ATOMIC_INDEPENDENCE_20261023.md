# CGB 逐项独立性与同体制约简证书（2026-10-23）

原14个固定子条件现均有“满足其余13项、仅违反目标项”的完整 Lean 证书。九项 `ReducedCore` 条件也各有一个局部状态有限、接触齐性、结构刚性的反例，满足其余13项并只违反目标项。因此，这组固定约简条件在声明体制内逐项不可删除。

这一结果补齐了前一轮复核指出的证书缺口。原六条公理与动力学未变；接触关系和真实接线的映射、完整规则预算及生命存在性继续分别验收。

## 证书怎样覆盖全部条件

`AtomicIndependence.lean` 给A01–A14统一索引，`Holds M c` 按原定义解释每个条件。`atomized_iff_all_conditions` 证明它与原 `Atomized` 完全对应，避免新证书证明一套较弱的替代条件。

每个单项反例都具有同一契约：

```text
Isolates M target :=
  (∀ c, c ≠ target → Holds M c) ∧ ¬ Holds M target
```

`all_atoms_independent` 对全部14个索引逐项提供该契约的模型。`twinFlip_all_conditions` 则独立提供全部条件共同成立的模型。

| 目标 | 模型 | 完整证书 |
|---|---|---|
| A01 | `twinTape` | `a01_independent` |
| A02 | `twinBag` | `a02_independent` |
| A03 | `twinBlank` | `a03_independent` |
| A04 | `path3Rigid` | `a04_independent` |
| A05 | `globalFlip univ` | `a05_independent` |
| A06 | `globalFlip ∅` | `a06_independent` |
| A07 | `farFlip` | `a07_independent` |
| A08 | `farAct` | `a08_independent` |
| A09 | `twinToggle` | `a09_independent` |
| A10 | `twinCycle` | `a10_independent` |
| A11 | `triLookup` | `a11_independent` |
| A12 | `innerClock` | `a12_independent` |
| A13 | `innerActClock` | `a13_independent` |
| A14 | `pairPreset` | `a14_independent` |

A01/A02使用一般语言允许的无限局部状态模型；有限特化下它们仍自动成立，二者结论的适用域已分开。A04使用三节点非齐性接触关系和空事件集，它只证明接触齐性不由其余条件推出。

A05/A06新增同一个正速率含量守恒对合：前者声明全体位点参与但接触关系只有自接触，后者声明空参与者却改写局部状态。两者均保留平衡、等变与边界条件。A08/A13则为先前的对偶模型补齐守恒、平衡、等变及其他父条款，完成全体系证书。

## 九项约简在同一体制内不可删除

`reducedCore_iff_selected_conditions` 证明九项选择准确对应现有结构。`every_reduced_condition_independent_in_regime` 对每个入选条件提供：

1. 有限局部状态空间；
2. 满足接触齐性；
3. 满足结构刚性；
4. 满足除目标项以外的全部13项；
5. 违反目标项。

这里使用 `rigidLabel` 给位点重新赋予单射身份。`holds_rigidLabel_iff` 与 `isolates_rigidLabel` 证明除等变与初态对称外，其余条件的真假均保持；新的等变与初态对称由刚性自动满足。动作、速率、接触、边界、内容函数、初态及刺激接口都未改变。

因此：

```text
有限局部状态 + 接触齐性 + 结构刚性
  ⊢ (ReducedCore ↔ Atomized)
```

这组固定九项中的任何一项都不能在上述体制下由其他八项代替；归约桥接所需的接触齐性仍明确保留在体制中。

## 结论如何用于下一步

固定列表的相对不可删性现已解决，可以据此检查具体载体的映射与扩张规律。若进一步固定身份页数、1%编码预算或生命目标，模型类会缩小，需要在该子域重新审查相互依赖；这些逻辑反例不自动满足那些附加要求。身份重标记也不宣称保持编码预算。

在进入真实生成检验之前，应先明确：`near` 对应什么物理接触、哪些状态变化代表生成接线、身份和几何如何完整计费、实现如何精化到规则书。这些映射给定后，才能检验具体模型是否落在已证体制内，并把自由动力学函数连接到真实果蝇数据。

相对不可删性不提供生命生成的充分性，也不决定唯一物理规律。生命目标继续构造见证；没有因本轮数学反例启动训练或能力实验。

## 验证与来源

- 全量构建与传递公理信任审计通过，审计器10项自检通过。
- 18个模块、1073个声明、658个内核定理条目，其中252个为源码直接声明的定理；本轮新增25个源码定理。生成声明与独立物理主张的数量分开计数。
- 前一轮934个声明的名称与签名全部保留；原Lean模块只有总导入入口与全量审计入口新增导入，原公理、动力学、规则编码和生命接口的源码哈希不变。
- 全库审计拒绝占位证明和未声明公理，传递依赖限于 `propext`、`Classical.choice`、`Quot.sound`。
- 可复跑命令（在 `lean/CGBAxioms` 中）：`python scripts/audit.py --output .lake/atomic_independence_20261023.json`。
- 发布记录：`results/published/cgb_atomic_independence_20261023.json`，包含14项证书签名、公理依赖、完整源码哈希与计数口径。
- 完整本地日志和签名清单：`lean/CGBAxioms/.lake/atomic_independence_20261023.*`。
- 本轮文档/入口更改前备份：`results/cgb_atomic_before_20261023_192146/`。
