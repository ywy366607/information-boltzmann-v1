# CGB / 梦漓：项目入口

当前平台点：[2026-11-05 可续接的同体交互闭环](CGB_MILESTONE_CLOSED_LOOP_20261105.md)。后续代理先读该页，再按任务进入定义、定理或运行代码。历史文档保留其原始证据范围和稳定链接，不再由文件名中的“当前”“下一步”决定执行顺序。

## 定义、方法与当前工作

| 入口 | 用途 |
|---|---|
| [AGENTS.md](../../AGENTS.md) | 用户原始定义、实现主链、五系统与生成式研究方法；引用原文时沿其中来源查对 |
| [语言生命体活规格](CGB_LANGUAGE_LIFE_MODEL_V1_20261025.md) | 整体目标、系统边界和共同状态约束 |
| [本次里程碑](CGB_MILESTONE_CLOSED_LOOP_20261105.md) | 哪些已实现、哪些经过训练、最新零训练修复、恢复点和下一步 |
| [五行工作规范](CGB_FIVE_PHASE_DIAGNOSTIC_RULES.md) | 同一共同运动中的功能分析与整体验收 |
| [大小五行接口](CGB_NESTED_FIVE_PHASE_INTERFACE_20261105.md) | 25种内部作用、实际交接与脏腑经脉的候选映射；第9—10节有运行追踪和动作归属修复 |

## 定理入口

先检索 [已证定理索引](CGB_PROVEN_INDEX.md)，再查 [主题导航](CGB_THEOREMS/README.md) 和实际 Lean 陈述、前提及依赖。分类见 [定理基座](CGB_THEOREM_BASIS_20261030.md)；分类数据和释义分别保存在 `CGB_THEOREM_CLASSES.json`、`CGB_THEOREM_INTERPRETATIONS.json`。

本平台审核库存为1807条源码定理、136个生产模块；1665条分类为 reviewed，142条仍为 draft。库存数量不作为当前身体的满足数量。最大覆盖—最小生成仍是构造目标，参见 [相容核心](CGB_THEOREM_CONSISTENT_CORE_20261103.md)。

维护方式：

```powershell
# 仓库根目录；PATH 中应有匹配 lean-toolchain 的 lake/lean
python lean/CGBAxioms/scripts/audit.py --check-index
python scripts/incremental_sync_theorems.py --check
```

一次性释义生成脚本已移入 `scripts/archive/theorem_interpretations_20261105/`，不再作为当前维护入口。

## 当前执行代码

| 路径 | 职责 |
|---|---|
| `scripts/ib_local/gated_deltanet.py`、`gdn_fused.py` | 现有可训练语言基底与有限窗口GPU实现 |
| `ngram_fast_table.py` | 身体的一部分：有限字节快表 |
| `latent_navigation_v3.py`、`language_navigation_kanli_v4.py` | 有限经历记忆、具体话语与共同接口 |
| `language_navigation_kanli_v4p.py` 至 `v4t.py` | 当前预测、校准、结算等继承层；更早版本中的若干类仍被依赖 |
| `navigation_settlement.py` | 结算目标、观察/估计身份，以及动作标签完整性 |
| `navigation_checkpoint.py` | 完整持续状态的保存与恢复 |
| `run_learning_rooms.py` | 从原实际执行入口原样提升的单个体机制运行脚本 |
| `report_learning_rooms.py`、`compare_navigation_settlement.py` | 既有报告与同源结果比较 |

此表中的文件位于 `scripts/ib_local/`。用 `python -m scripts.ib_local.<模块>` 运行，避免将该目录下的文件误作标准库模块。

当前参数、身体、快表和经历均属于同一个体。母亲世界、3D介质、果蝇及优化器保留各自研究和证据；本次整理不删除这些路线，也不把它们强行合成当前载体。

## 历史文档状态

以下材料原位归档，保留原链接以便定理、实验与研究树追溯。只在复现对应版本或查找推导时阅读其中的执行指令。

| 历史组 | 原文入口 | 当前如何使用 |
|---|---|---|
| 旧柱与载体搜索 | `CGB_LANGUAGE_SYSTEM23_CONTINUOUS_20261101.md`、`CGB_LANGUAGE_BODY_SEARCH_20261102.md`、`CGB_E3_DESIGN_V1_20261102.md` | 失败与准入证据，不能替代当前运行身体 |
| 生成与总装探索 | `CGB_DESIGN_BRIEF_20261102.md`、`CGB_CLAUDE_HANDOFF_20261103.md`、`CGB_MINIMAL_GENERATOR_20261103.md` | 保留生成目标及取舍历史；执行状态以里程碑为准 |
| 锁存与召回实验 | `CGB_LATCH_TASKS_20261103.md`、`CGB_RETRIEVED_CREDIT_20261103.md` | 保留机制范围和净收益判断，不重新开启已关闭实验 |
| v2/v3导航 | `CGB_NAVIGATION_V2_PREREG_20261104.md`、`CGB_NAVIGATION_V3_ASSEMBLY_20261104.md`、`CGB_NAVIGATION_V3_WORLD_20261104.md`、`CGB_NAVIGATION_V3_TURNS_20261104.md` | 历史协议与测量纠正；过期PID队列已归档 |
| v4逐轮研究过程 | [坎离研究记录](CGB_NAVIGATION_V4_KANLI_20261104.md)、[交接过程](CGB_NAVIGATION_V4_HANDOVER_20261104.md) | 保存预登记、失败和修改原因；最新定位见交接第18节与里程碑 |
| 见证与结算的中间方案 | `CGB_CURRENT_WITNESS_20261105.md`、`CGB_REPRESENTATIVE_WITNESS_20261105.md`、`CGB_TYPED_WOOD_20261105.md` | 查清为什么采用共同续文评分，不将早期“下一步”当待办 |

原始日志、数据、完整检查点留在本地忽略目录。Git保存代码、Lean、精选报告及校验哈希；本次清理清单见 [工作区整理记录](../../results/published/cgb_workspace_cleanup_20261105.json)。
