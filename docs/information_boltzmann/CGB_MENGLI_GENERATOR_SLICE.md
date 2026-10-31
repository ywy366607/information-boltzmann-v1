# MengLi 最小生成核：自动反向依赖切片（勿手改）

来源：Lean 审计库存。此页不把相邻定理算作覆盖，不从名称猜测条件；它保留声明的直接形式前提和审计到的项目依赖。完整机器数据见同名 JSON。

库存：1575 条源码定理。首轮生命切片：112 个本地声明（54 定理、58 定义），7 个结构/世界边界。

## L0–L6 根与覆盖边界

| 需求 | 代表性根声明 | 覆盖状态 |
|---|---|---|
| L0/L6 | `CGB.SelfProduction.language_infinite_stream_life_complete`<br>`CGB.SelfProduction.language_controlled_trajectory_viable` | 条件合同；尚非共同动力学的无条件存在。 |
| L1 | `CGB.SelfProduction.language_self_consequence_model_exists` | 仅有自我后果模型的条件存在；目的涌现本身没有直接根定理。 |
| L2 | `CGB.InformationEconomics.causal_carrier_contract_iff`<br>`CGB.InformationEconomics.selective_write_contract`<br>`CGB.InformationEconomics.predictive_growth_contract`<br>`CGB.LanguageLifeKernel.delayed_credit_requires_history_carrier` | 机制链已形式化；尚待同一语言状态精化。 |
| L3 | `CGB.SelfProduction.language_self_consequence_model_exists` | 条件性行动后果模型；尚非经干预验收的学得世界模型。 |
| L4 | `CGB.ActiveInference.active_closure` | 给定供给界与行动敏感世界的条件结论。 |
| L5 | — | 没有精确的顶层 Lean 根声明；不能由邻近行动定理代填。 |

## 直接条件边

下列是根声明签名中明确写出的蕴含前提；构造 `Φ` 时必须逐项落实为公理结果、共同状态不变式或登记的世界合同。

| 服务需求 | 根声明 | 形式条件 |
|---|---|---|
| L2 | `predictive_growth_contract` | `(∀ (b : B), 0 ≤ p b)` |
| L2 | `predictive_growth_contract` | `∑ b, p b = 1` |
| L2 | `predictive_growth_contract` | `(∀ (b : B), 0 < c b)` |
| L2 | `predictive_growth_contract` | `∑ b, c b = 1` |
| L2 | `delayed_credit_requires_history_carrier` | `credit₀ ≠ credit₁` |
| L4 | `active_closure` | `CGB.ActiveInference.unbounded signal` |
| L4 | `active_closure` | `CGB.ActiveInference.passiveBounded policy signal a₀` |
| L4 | `active_closure` | `CGB.ActiveInference.actionSensitive response` |

## 跨需求脊柱

以下**定理**被至少两个独立需求切片共享。状态字段和编译辅助项留在 JSON，避免把文档重新膨胀为索引；这不是唯一的语义最小公理集。

| 声明 | 服务需求数 | 类别 | 源模块 |
|---|---:|---|---|
| `CGB.CausalLearning.retain_consistent` | 3 | theorem | `CGBAxioms.CausalLearning` |
| `CGB.LanguageLifeAdmission.LanguageCandidate.next_correct` | 3 | theorem | `CGBAxioms.LanguageLifeAdmission` |
| `CGB.LanguageLifeAdmission.LanguageCandidate.write_refines` | 3 | theorem | `CGBAxioms.LanguageLifeAdmission` |
| `CGB.SelfProduction.LanguageSelfProductionCandidate.faithful` | 3 | theorem | `CGBAxioms.GenerationSelfProduction` |
| `CGB.SelfProduction.LanguageSelfProductionCandidate.seed_nonempty` | 3 | theorem | `CGBAxioms.GenerationSelfProduction` |
| `CGB.SelfProduction.action_consequence_model_exists` | 3 | theorem | `CGBAxioms.GenerationSelfProduction` |
| `CGB.SelfProduction.language_self_consequence_model_exists` | 3 | theorem | `CGBAxioms.GenerationSelfProduction` |
| `CGB.SelfProduction.language_viable_next` | 3 | theorem | `CGBAxioms.GenerationSelfProduction` |
| `CGB.ActiveInference.action_condition_required` | 2 | theorem | `CGBAxioms.GenerationActiveInference` |
| `CGB.ActiveInference.active_closure` | 2 | theorem | `CGBAxioms.GenerationActiveInference` |
| `CGB.ActiveInference.not_unbounded_of_bounded` | 2 | theorem | `CGBAxioms.GenerationActiveInference` |
| `CGB.ActiveInference.viable_not_eventually_passive` | 2 | theorem | `CGBAxioms.GenerationActiveInference` |
| `CGB.CausalLearning.agreement_no_write` | 2 | theorem | `CGBAxioms.CausalLearning` |
| `CGB.CausalLearning.paid_elimination` | 2 | theorem | `CGBAxioms.CausalLearning` |
| `CGB.InformationEconomics.predictive_growth_contract` | 2 | theorem | `CGBAxioms.InformationEconomics` |
| `CGB.InformationEconomics.selective_write_contract` | 2 | theorem | `CGBAxioms.InformationEconomics` |
| `CGB.PredictiveHarvest.gibbs_inequality` | 2 | theorem | `CGBAxioms.GenerationPredictiveHarvest` |
| `CGB.PredictiveHarvest.gibbs_term` | 2 | theorem | `CGBAxioms.GenerationPredictiveHarvest` |
| `CGB.PredictiveHarvest.growth_eq_log_card_sub_crossEntropy` | 2 | theorem | `CGBAxioms.GenerationPredictiveHarvest` |
| `CGB.PredictiveHarvest.growth_le` | 2 | theorem | `CGBAxioms.GenerationPredictiveHarvest` |

## 共同 Φ 必须消化的合同字段

这四组来自切片中实际出现的结构字段和声明，不是新增模块。它们说明最小生成器必须把哪些既有合同投影到同一局部状态、边界事件和递推上。完整成员表在 JSON。

| 合同组 | 切片成员数 | 对 Φ 的义务 |
|---|---:|---|
| 因果载体 | 8 | `reserve/encode/read/next/advanceHistory/required` 必须成为同一状态 Z 的投影，不能另置记忆模块。 |
| 选择写入与可塑性 | 14 | 候选、费用、保留和写入精化必须是 Φ 对同一 Z 的局部变化，而非外置优化器状态。 |
| 边界、行动与世界后果 | 8 | `context/action/outcome/policy/possible/world_response` 必须成为边界事件和 Φ 输出，而非预装世界模型。 |
| 出生、连续性与主动性 | 7 | `seed/viable/signal` 必须成为可达初态、可存活区域和世界条件；不能偷换成奖励或食物账户。 |

## 未自动消去的边界

这些名称未出现在审计声明表中，通常是结构字段、类型类或外部世界条件。它们不能被误写成已经由 Lean 推出的机制。

- `CGB.CausalLearning.Memory`
- `CGB.CausalLearning.Memory.mk`
- `CGB.CausalLearning.Memory.rec`
- `CGB.LanguageLifeAdmission.LanguageCandidate`
- `CGB.SelfProduction.Contract`
- `CGB.SelfProduction.Contract.mk`
- `CGB.SelfProduction.LanguageSelfProductionCandidate`

## 使用

1. 新机制先选择其服务的 L 需求和根声明；
2. 检查该根的直接条件能否由同一 `Φ` 满足；
3. 仅在反例显示缺少可区分自由度时增加状态；
4. 运行 `python scripts/build_mengli_generator_slice.py --check`，防止索引更新后文档漂移。
