# CGB / 梦漓 MengLi

面向无限信息流、有限状态与有限计算资源的持续学习系统。研究目标是让记忆、预测、学习、选择性保持与行动在同一身体—世界过程中共同形成，走向用户定义的无限流生命体。Alan T0 是整个项目完成后的命名愿景，CGB 为架构，MengLi（梦漓）为生命体。

**当前平台点：2026-11-05，可续接的同体交互闭环。** 已有真实语言学习与行动接线；最近在低语义机制世界完成观察收益记忆的512轮同源验证，过程改善、终点略好。最新动作归属修复通过接口及真实存档检查，尚未整圈训练。

## 从这里开始

- [项目导航与当前/历史文档分层](docs/information_boltzmann/README.md)
- [本次里程碑：结果、恢复点、版本边界与下一步](docs/information_boltzmann/CGB_MILESTONE_CLOSED_LOOP_20261105.md)
- [AGENTS.md：用户定义、主链、五系统与工作方法](AGENTS.md)
- [Lean定理索引](docs/information_boltzmann/CGB_PROVEN_INDEX.md) / [主题导航](docs/information_boltzmann/CGB_THEOREMS/README.md)
- [关系式五行规范](docs/information_boltzmann/CGB_FIVE_PHASE_DIAGNOSTIC_RULES.md) / [大小五行接口与实际周转](docs/information_boltzmann/CGB_NESTED_FIVE_PHASE_INTERFACE_20261105.md)

## 平台包含什么

同一Gated DeltaNet语言基底、有限字节快表与经历库，具体话语的内言和实际表达，世界返回、自听、同体在线学习、实测结算与历史取用。当前目标是沿这条共同运动继续生成功能，不按定理或生物名称逐个拼装模块。

| 已完成的观察收益记忆运行 | 同源控制 | 修复候选 |
|---|---:|---:|
| 终点均衡NLL | 0.409597 | 0.409162 |
| 固定40审计点平均NLL | 1.166549 | 0.926529 |

这两行属于单种子机制诊断。真实语言结果、完整协议、后续动作归属修复及未完成目标分别在里程碑中列明，避免把不同实验混作同一个成绩。

## 验证与运行

Python 3.10+、PyTorch；实验GPU环境需匹配本地CUDA/Triton。项目脚本从仓库根目录以模块方式运行：

```powershell
python -m pytest tests/test_navigation_observed_gain.py tests/test_navigation_settlement.py tests/test_navigation_residual_wood.py -q
python -m scripts.ib_local.run_learning_rooms --help
# Lean使用lean/CGBAxioms/lean-toolchain固定的工具链
python lean/CGBAxioms/scripts/audit.py --check-index
```

训练入口、同源检查点要求和完整命令见里程碑。数据、原始日志及检查点不进Git；精选证据在 `results/published/`，参数、优化器、记忆与来源状态必须一起恢复。

## 仓库布局

| 路径 | 用途 |
|---|---|
| `information_boltzmann/` | 介质、历史载体与共同算子 |
| `scripts/ib_local/` | 当前语言身体、训练/评估入口和数值工具 |
| `lean/CGBAxioms/` | 公理、定理、构造及审计 |
| `docs/information_boltzmann/` | 定义、当前规格、定理导航与历史记录 |
| `results/published/` | 精选报告和验证哈希 |
| `tests/` | 数值、因果、恢复与接口回归 |
| `scripts/archive/` | 已退出当前入口的一次性工具原文 |

本独立工作区继承自视觉/多模态项目。其源码与既有证据继续保留；[原README](docs/archive/FINE_GRAIN_VISION_README_20261105.md)已归档，视觉主图规格仍见 [NORTH_STAR](docs/NORTH_STAR.md)。CGB操作不擅自改变该独立研究线的要求。
