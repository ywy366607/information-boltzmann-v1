# CGB 开发记录与回退入口

本文件记录2026-10-22之后的接续开发。Git保存代码、推导、测试和精选报告；大型运行存档仍在实验盘，不能把Git快照称作完整个体备份。

## 基线：快活动—慢结构耦合之前

- 来源：Claude的构形生成脑主规划、核心方程、V1/V2/V3实验，及Codex随后接续的相遇、材料、运输和有限交换实现。基线提交是继承工作快照，不表示全部由本轮作者创作。
- 原HEAD：`0b80a66`。本次只提交CGB目录、相关测试、CGB规划/报告、生成脑原始规划与仓库约束；不收进无关的果蝇修复、CFE或其他工作目录修改。
- 备份时发现现有`research_tree.json`含多个不符合仓库注册格式的节点（状态名、ID长度、缺少动机及失效条件），提交检查拒绝了它。本轮未关闭检查、未把历史研究结论改成通过；原文件保留未提交，逐字另存`CGB_REGISTRY_SNAPSHOT_20261022.json`作为未验证的历史证据，便于Claude恢复。该副本不是正式注册表。
- 基线实际结果：有限交换与结构占据子核；97项相关检查通过；尚未接入真实快活动、语言后果和完整生成个体。
- 实验来源：`F:/fly_stream_overflow/cgb_spatial_calibrated_r1_20261021/medium_complete.npz`，SHA256 `c57b03a49418b37f3ca5bd0fc1dcefa892ce649637d5e007c5ea14c5fa447112`。
- 结构子核结果：`F:/fly_stream_overflow/cgb_structural_exchange_final_20261022/report.json`；200ms完整子核存档`exchange_complete.npz`的SHA256为`52b889cb0419b0e9c33eabb95119a201a22bd020aaf0f25c14c978ce89d1aaa1`。它不是原个体的替代存档。
- 下一步：沿既有电荷守恒与有限驱动态，推导并实现快活动向慢结构输送功的局部通路；明确边界做功、热流、反向过程及结构改变后的响应，不能将NLL或活动量直接当免费燃料。

## 如何回退与继续

基线提交`14bcfa173bd651f8e0302c04070d7458a15ff16a`，用Git标签`cgb-before-fast-work-20261022`定位；本轮完成版用`cgb-fast-work-20261022`定位。后续每个完整实现单独提交。优先在另一目录建立分支查看旧版：

```powershell
git worktree add -b codex/cgb-revisit-20261022 D:/cgb-revisit-20261022 cgb-before-fast-work-20261022
```

这会保留当前工作目录和后来历史。若只撤销某一提交，先检查工作区并审阅该提交，再在开发分支使用`git revert <提交号>`；不要对含未提交工作的目录直接执行硬重置。运行存档按报告的来源、版本和哈希另行加载。Git尚未上传远端，属于本机可回退备份。

## 本轮记录

2026-10-22 04:10（Asia/Shanghai附近；精确运行/提交时间见报告及Git）。

- 改动：新增`flow_exchange.py`及13项物理/恢复检查；原子存档按kind支持新类，原finite/structural档仍可加载；新增配对审计脚本，更新主规划、物理核心及归属表。
- 推导：[快场—慢结构预注册](CGB_FAST_WORK_PREREG_20261022.md)。固定电容、同质并联导通和热浴两态换能是明确本构。未使用目标梯度或免费活动奖励，未切换主个体。
- 命令：`D:/conda_envs/vox/python.exe scripts/ib/generative/flow_exchange_audit.py --archive F:/fly_stream_overflow/cgb_spatial_calibrated_r1_20261021/medium_complete.npz --out F:/fly_stream_overflow/cgb_flow_exchange_r2_20261022`。
- 最终验证：`pytest`运行flow_exchange、finite_exchange、structural_exchange、support_analysis、support_transport、material_reactions、structural_budget、birth、development、birth_calibration、tree_geometry这11个CGB测试文件，**110通过，10.45秒**；JUnit在最终运行目录`regression.xml`。
- 初轮及改进：CG解直接写荷量会累积求解余差。改为按反对称边电流推进，并单列传输的物理／数值热；最终内部输运总电荷余差约`1e-14`。初轮报告与源码也保留，不覆盖失败／改进证据。
- 实际结果、四臂指标、来源及存档SHA256见[结果文档](CGB_FAST_WORK_RESULTS_20261022.md)。输入经换能影响结构，结构影响下一快响应；完整恢复后20ms逐位一致。
- 尚未交付：这仍是被动互易电荷实现，不是全脑能量验收；有源定向传输、原脉冲／语言接口、选择性学习及世界后果尚待整合。
- 备份：当前分支`codex/cgb-physical-closure-20261022`。本轮标签与基线分开，可单独回退；实验盘另存包含这两次提交的增量Git bundle及清单，备份路径`F:/fly_stream_overflow/cgb_git_backup_20261022/`。实际bundle验证与哈希记录在该目录，只有验证成功后才称完成备份。

## 有源传输与端口（本轮结束后关机）

2026-10-22约04:20，Asia/Shanghai。

- 用户指令：“做完帮我关机”。完成当前一个有界接入／验证段后执行，不无限延长研究，也不恢复暂停主个体。
- 继承提交`eb720ed9647a1f2a826bee0ac33350989c9f823c`；新改动以`cgb-driven-ports-20261022`标签定位，旧版以`cgb-before-driven-ports-20261022`定位。
- 沿同一自由能新增有限P/D付款的极性传输，以及有限外部电容／源电导／端口反作用；实现`driven_exchange.py`、存档kind、14项新检查及真实书架字节配对审计。
- 执行：`D:/conda_envs/vox/python.exe scripts/ib/generative/driven_ports_audit.py --archive F:/fly_stream_overflow/cgb_spatial_calibrated_r1_20261021/medium_complete.npz --out F:/fly_stream_overflow/cgb_driven_ports_20261022`。
- 验证：12个相关CGB测试文件124项通过，JUnit随运行保存；4臂实际96ms、半程完整恢复、两个细化步长、付费的外部状态干预均完成。未启动语言训练。
- [实际结果及边界](CGB_DRIVEN_PORTS_RESULTS_20261022.md)。本轮交付材料／端口因果作用，没有把端口反馈改称完整语言主体闭环。
- 备份路径：`F:/fly_stream_overflow/cgb_driven_ports_backup_20261022/`，存Git bundle、完整子核状态、报告、书架和哈希清单。日志与Git先完成，备份验证后再发出关机指令；系统执行结果单列在该路径的关机请求记录中。

## 活动痕迹与捕获的联合实现

2026-10-22。用户恢复工作，要求沿Claude及主规划把学习过程与物理通路统一，并推进空间生长；本轮没有再次关机请求。

- 回退基线`e5b75bb`，标签`cgb-before-trace-growth-20261022`。痕迹实现单独提交并标`cgb-activity-trace-20261022`，空间扩展随后单独提交。
- 新增有限二态痕迹、带存储的实际传输事件、互易结合能、热恢复和两态通道群速率；沿原结构反应形成捕获／留存通路。新变量都有容量、储能和正反转化。
- `finite_exchange.py`提取共同结合能接口，`driven_exchange.py`提取共同正反尝试率，旧默认保持原数值路径。未向旧主个体写入新状态。
- 执行`trace_exchange_audit.py --archive F:/fly_stream_overflow/cgb_spatial_calibrated_r1_20261021/medium_complete.npz --out F:/fly_stream_overflow/cgb_trace_exchange_20261022`：4臂64ms、半程恢复、同文本切断对照完成。13项直接检查通过，JUnit为该目录`trace_tests.xml`。
- 联合实现更能保留结合支持，但写入要付款，总结构并未比不写痕迹更多；这不是失败数据删除，也没有把总量当作学习收益。详见[CGB_TRACE_EXCHANGE_RESULTS_20261022.md](CGB_TRACE_EXCHANGE_RESULTS_20261022.md)。
- 该阶段存档调度器在Git中保留仅到trace类型的实际版本；随后空间提交再加入spatial类型，便于Claude独立回退。
