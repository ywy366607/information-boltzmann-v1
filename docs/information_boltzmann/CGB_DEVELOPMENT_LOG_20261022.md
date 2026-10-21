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

## 空间材料反应、成核及种子检查

2026-10-22约05:45，Asia/Shanghai。痕迹提交`80478b3`已独立保存；本段用`cgb-spatial-material-20261022`标签定位。

- 把同一TraceExchange放入等体积三维空间：有限F/B/P/D扩散、局部结合能、双向表面催化与实际材料沉积／拆解。没有目标质心、指定分叉数或分枝方向调度。
- 主执行：`spatial_growth_audit.py --out F:/fly_stream_overflow/cgb_spatial_material_20261022`，预先记录四臂，共同9³网格／64个真实字节／256ms。看到低密度全域成核后，另写方案并执行`spatial_seed_control.py --primary F:/fly_stream_overflow/cgb_spatial_material_20261022 --out F:/fly_stream_overflow/cgb_spatial_seed_control_20261022`。
- 关键结果：无种子也形成高密度结构，现阶段是空间成核／沉积，不是神经突伸长验收；4臂、完整无种子诊断及失败出口均保留。[完整结果](CGB_SPATIAL_MATERIAL_RESULTS_20261022.md)。
- 验证：14个CGB文件150项通过，13.40秒；JUnit在主空间运行目录`regression.xml`。包括独立ODE、扩散算子空间极限、组合收支、相邻结合能、完整恢复和关闭扩展等价。初始2³扩散测试触及精度地板，换成非交换3³检验分裂误差，未改物理实现来凑验收。
- 观察室新增只读`/material`，读取实际`material_scene.json`；四臂／时间／显示阈值可切换，阈值不反馈给动力学。页面末态509.75材料／78.25底物／1266.68源功已与原报告核对；截图`F:/fly_stream_overflow/cgb_spatial_material_20261022/material_view.png`。旧出生候选页面保留。
- 仍缺：多谱系材料域、真实细胞接触与V3d调用、连通细长界面、完整有符号／预测贡献学习及连续身体世界。保留这些缺口，不按测试数量称整脑完成。
- 备份：`F:/fly_stream_overflow/cgb_trace_spatial_backup_20261022/`保存Git增量bundle、两阶段状态、源码／书架／报告／截图和哈希清单；正式主个体未恢复。本轮无新的关机授权，不自动关机。

## 下一步物理状态的必要性

空间代码提交`f448d91`之后，另将“交叉不相通／真正接合”的等材料端口响应推导写入[CGB_CONNECTIVITY_STATE_DERIVATION_20261022.md](CGB_CONNECTIVITY_STATE_DERIVATION_20261022.md)，同步主规划。这是下一轮连接端／导通域状态的来源，不新添未经推导的分叉动力学，不改变本轮执行代码和结果。


## 独立材料域、有限端与新接合

2026-10-22，基线48772fe。由连接状态必要性推导实施EndExchange，复用Trace/Spatial共同自由能和物料账。双端反应、同位置独立电荷、有限端竞争与接触生成已接入；实际材料移动和身份谱系继续待接。

- 实施前方案：CGB_END_PAIRING_PREREG_20261022.md；源码及该版本方案复制到运行目录。
- 主执行：`end_pairing_audit.py --out F:/fly_stream_overflow/cgb_end_pairing_20261022`，五臂各256ms。无种子仍沉积，不把它包装成神经突生长。
- 同末态因果：`end_contact_delivery.py --run F:/fly_stream_overflow/cgb_end_pairing_20261022`，实际256→256.5ms，sham逐位一致，只改新桥电导产生最大电荷差0.005087272，原档哈希不变。
- 15个CGB文件163项通过/18.33s，JUnit在运行目录regression.xml。默认慢合成通路测试最初触及自定幅度阈值，改用明确较快测试周转解析，无修改主实验常数。
- 状态134312字节，O(N+E)；各臂半程/末态及因果副本均完整保存并核对。正式主个体保持暂停。
- 新结果、原始证据、源码快照和Git阶段备份分开保留；后续提交定位见cgb-finite-ends-20261022标签。

- 观察室增加只读`/ends`，保留`/material`和原出生页面；实际页面与JSON核对，截图`F:/fly_stream_overflow/cgb_end_pairing_20261022/end_view.png`。本轮仅替换自己启动的只读viewer服务，保留原参数并加新结果入口；当前PID24464。无训练进程启动。


## 移动材料、存储载体与同温热浴

2026-10-22，基线abf1d04。新增MobileExchange，把既有反应图分辨成有限整数物料/二态桥；端库存与局部组成随实际材料坐标移动。实际新接触临时搜索，桥创建/配置释放/回收统一记账。

- 第一轮`mobile_material_audit.py --out F:/fly_stream_overflow/cgb_mobile_material_20261022`完整保存；12项直接与175项组合检查通过，但快场仍是历史确定性耗散。联查同温性发现需补配套电荷涨落，保留该轮源码/旧schema1，不作正式同温结果。
- 按相同kBT导出快场/世界端口OU协方差，第三RNG完整保存；新schema2明确区别，未给旧个体补随机数冒充连续恢复。第二轮`F:/fly_stream_overflow/cgb_mobile_thermal_20261022`完成。
- 共享端统计、一次相遇搜索、局部运动格箱及弹性能增量替换重复全图工作；`mobile_execution_equivalence.py --reference F:/fly_stream_overflow/cgb_mobile_thermal_20261022 --out F:/fly_stream_overflow/cgb_mobile_thermal_20261022/execution_equivalence.json`：五臂128→256ms物理数组/三RNG逐位相同，账本舍入≤2.51e-12。
- 当前执行`mobile_material_audit.py --out F:/fly_stream_overflow/cgb_mobile_bounded_20261022`。16个相关文件176项通过/17.34s；正式报告、全部几何/事件/源码/字节/恢复结果保留。
- 完整臂有19次原先不相遇材料对的装配，两条初始桥发展到28桥；0号连通部分3→23材料；初始两桥夹角π→1.46708rad，非角度目标控制。各干预只是一次配对实现，不由桥数大小声称学习收益。
- 新只读`/mobile`保留旧路由。实际初/末数值与图像核对，截图`F:/fly_stream_overflow/cgb_mobile_bounded_20261022/mobile_view.png`；viewer新PID12448，仅替换本线程原只读服务，无正式训练启动。
- 下一段：真实相遇的有向身份发生/强度物性接口、完整有符号/延迟学习与世界后果。小材料核未被叫作完整生命体。分阶段Git和完整证据备份另保存。


## 继承有向身份程序的材料物性接口

2026-10-22，基线cccdd06。实际相遇复用旧发生/强度身份分量：条件概率相对比对应识别G倾斜，强度身份配置输运系数。未将成年概率称真实生物结合能，也未重复类型几何门控。显式α保留U/熵不可辨识性，两种载体实测同轨迹不同热。

- `identity_material_audit.py --out F:/fly_stream_overflow/cgb_identity_material_20261022`完成五臂/189项初轮；原源码/方案/基因/状态保留。真实38453对发生重建误差2.22e-16、参考强度分解误差4.88e-7，原3000类型和码一致。
- 基础移动核五臂128→256ms对照原实际源码，数组、三RNG、账本逐位相同。
- 新增可丢弃的类型对完整求值缓存，最多4096条；身份只读，缓存不是物理/学习记忆，淘汰不改状态。`identity_execution_equivalence.py`五臂原实际源码对照全状态/账本逐位同。
- 当前`identity_material_audit.py --out F:/fly_stream_overflow/cgb_identity_cached_20261022`五臂完成，17个相关文件190项通过/19.47s。完整臂有12次起初不相遇材料对的装配，29末态桥、6对互惠桥；观察和学习收益分开。
- 完整基因数组825280字节，一群材料共享，旧模型文件无需在场也可完整续接；两轮共20份半程/末态档保留。
- `/identity`只读回放已核对共同过程和α=1热账1804.25/1804.01，截图F:/fly_stream_overflow/cgb_identity_cached_20261022/identity_view.png。仅替换本线程只读viewer，现PID10248；正式主个体未恢复。
- 下一步沿原主线接完整有符号时序/延迟学习、内生预测贡献信用、有限补给与世界行动；物性标定/生物解释不以测试数量替代。Git及完整来源备份另存。

## 同核学习生成元、真实流及原生感觉通道

2026-10-22，基线0f16741。沿原r/C/P/D与空间结构做真实流分支，不新增K/P池或外部奖励；正负学习通量由原反应生成元求和，参数淬变按ΔU/ΔG付款。

- `native_learning_stream.py --out F:/fly_stream_overflow/cgb_native_learning_20261022`：共享256真实OWT前缀，原生／切χ／固定慢存储三支各1024新A→256新B→256回访A，4ms/byte、dt2ms。原始NLL及每字节前序logit、曲线、物料/热功、源码/来源、相位游标及完整档保留；不改母本。
- 原生NLL12.172597/12.330182/12.428754，固定词频3.397642/3.543381/3.392655 nat/byte；源端滤波同类读出也差。切通路改变生成过程，但没有交付有用预测学习；短预算不判断容量，不调温度追分。
- 原生实际反应15977次，write+3356/write−3377，桥装配265/拆解266；物料/驱动/支持/端精确守恒，能量/自由能累计误差最大9.11e−10。χ=0副本显式工作34，记忆未擦除。
- `native_learning_resume.py`从原生A中点游标512、3328ms继续1024未来字节到7424ms；所有未来评分、全部物理状态/三RNG和外部会话逐位相同，报告及末档在resume/。
- `native_sensory_audit.py`按追加预注册，从现有有序OU场过程解析条件Gaussian通道；实际零噪声与噪声基数值核对。真实随后16字节的首个2ms场步，均匀源设计的信息上界均值0.00515802 nat，增益范数0.0202314、噪声协方差迹3.49027。输入后仪器后验不是主体后验或更新前预测；sham全部状态/三RNG/前序评分逐位相同。
- 21个CGB文件213项通过/23.65s（含此前未纳入该轮17文件清单的既有测试），JUnit regression_final.xml；感觉4项直接检查另通过。仅诊断观测和研究副本，无旧个体训练或关机。下一步从完整感觉时钟／有限功噪声推导换能工作点与预测表示，再接时序/延迟信用和世界作用。
