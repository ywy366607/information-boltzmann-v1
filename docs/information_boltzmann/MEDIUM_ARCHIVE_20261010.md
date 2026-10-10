# 3D 介质：封存说明（2026-10-10）

状态：**封存**。用户决定停止 3D 介质，转回果蝇大脑。本文件记录封存时的底线结论、为什么没有进展，以及将来重启时第一步必须做什么。

**范围更正（重要）**：以下结论只针对 2026-09-29 以后的 PlasticMedium3D 代（Gen1–3）。更早的 CBIM 介质已经证明能学、且碰撞与输运有明确因果：
OpenWebText 上 CBIM 场模型验证 NLL 7.19（1D）/ 7.26（3D），关闭碰撞 NLL 变差 +0.49 / +0.19；连续速度 S² 版本碰撞因果 +0.79、联合因果 +1.24 nats；
数独 Never-Reset Master Unified 格子准确率 55.5%（参数 8 倍少于 TRM，TRM 51.6%），25 万步正式训练 56.2% / 空格 37.9%。
所以失败的是这一代的实现，不是“介质”这条路。重启时首先要回答：PlasticMedium3D 相比 CBIM 丢了什么。已知线索（未核实）：
BPTT 窗口从 128 截断到 32（违反“永不截断”的记录）；碰撞从“连续速度方向 + 被转向的输运”变成固定格点边上的通道混合（实测 98.5% 静态线性）；
因果评价方式不同（CBIM 测关闭碰撞/输运的 ΔNLL，本代测冻结接口）。
时间：`torus3d.py` 始于 2026-09-29，介质文档始于 10-04；31 份 MEDIUM 文档，20 个结果目录，核心代码约 3700 行。

## 底线结论

冻结写入与读出（写入代理、读出探针与修正 MLP、时间读出），只让介质与线性解码器学习，介质自身几乎学不到东西
（K=4、读写分离、土、出生前发育、拆分写入、出口瓶颈、散射截面场，300 窗口，一个种子）：

| 条件 | 每 100 窗口增益（相对一元词频） |
|---|---|
| 全部可学 | 0.079 / 0.150 / 0.134 |
| 冻结接口，介质 + 线性解码器可学 | 0.008 / 0.007 / −0.014 |
| 介质也冻结，只训练解码器（储备池对照） | 0.023 / −0.080 / −0.198 |

此前所有增益来自介质外的接口（写口的外部 token 预测 + 新息、读出修正 MLP、时间读出），不是介质的计算。

## 原因（从代码可直接看出）

1. 介质没有主动元件：输运与碰撞是保范数旋转，响应模块向固定平衡点弛豫；任何活动按寿命约 0.87 衰减，不能保持状态，不能形成吸引子——结构上就是混响器。
2. 响应模块的门控作用在有符号、快速振荡的波场上（跳跃约 10/时间，门控时间常数约 0.45），看到的平均值约为零，因而静态；电导项对振荡波只能是阻尼，给不出振幅增益。
3. 基础设施（精确 VJP、CUDA 图、带 KL 信赖域的结构后验、Connes–Kreimer 记账、健康账本）先于“介质能自己学会任何东西”的证明；机制一个个叠加（河床、材料编码、导电可塑性、STP、Hopf 结接、引力、土、散射截面），没有一个单独证明过功能，测过的大多是惰性的。
4. 评价用短跑 NLL 增益，既碰到样本量上限（二元模型天花板），又分不清介质与接口的功劳。

## 本会话的可选改动（默认关闭，旧行为与旧个体不变）

- 碰撞：`--collision-preset boltzmann`（双线性成对碰撞 + 跨分量层），`cell_gain`（守恒的散射截面场），见 `COLLISION_OPERATOR.md`。
- 内在时间 `--internal-time-factor`；结构自引力 `--structural-gravity(-screening/-diffusion)`。
- 统一河床：`--soil-absorption`、`--channel-growth`、`--channel-mu`、`--prenatal-windows`；`PlasticMedium3D.soil_absorption / absorb_soil / through_flow / channel_growth_step`。
- 身体：`--port-layout separated`、`--write-split`、`--read-heads/--read-queries`。
- 堵路与接口：`--freeze none|interface|interface+medium`、`--brain-interface`（无读出修正 MLP、写口不做外部预测、禁止时间读出；只做过语法检查，未运行）。
- 学习器：`freeze_structure`、`skip_unstable_structure`（结构步无下降方向时跳过该窗口并计数）。
- 测试：`tests/test_collision_activity.py`（碰撞、土、通道生长、散射截面、拆分写入）。
- 测量与离线脚本（不入库）：`scratch/medium_distance_cut/`（need_matrix、bath_credit、port_delivery、analyze_brain、export_channels 等），
  `scratch/collision_pilot/`（dilution、growth_sim、unified_sim、chiral_wall、tweb、carve、physarum、pileup、motifs、viz/）。所有检查点已删除。
- 详细测量：`SPATIAL_DIFFERENTIATION_AUDIT_20261010.md` §1–§8。

## 重启方案（2026-10-10 用户确定）

重启 3D 介质时按 [MEDIUM_EXCITABLE_TISSUE_DESIGN.md](MEDIUM_EXCITABLE_TISSUE_DESIGN.md) 执行：可兴奋组织介质。神经元物理（COBA、ALIF、STP）成为每个格点、每个通道的可学物质参数；每个通道有自己的物质密度（多相），空腔不导电；突触是接触格点上的局部耦合；果蝇大脑作为搜索空间中的一点和出生时的初始条件。下面的旧建议作为参考保留。

## 重启时的第一步（旧建议）

1. 先定唯一验收标准并从第一天起使用：写入与读出冻结，只有介质与线性解码器可学，真实文本上明显胜过储备池对照。过不了这条，任何结构与可视化都不算进展。
2. 先对照 CBIM 找出本代丢失的东西（BPTT 128、连续速度与转向输运、碰撞形式），从已经证明能学的 CBIM 形式出发，而不是另起炉灶；只做最小版本，不带结构后验等基础设施。
3. 通过第 1 条后，再逐一加入：守恒密度场长出连接组（边导率 √(ρᵢρⱼ)，真空不传）、出生前发育、读写分离、手性方向；每加一项都要在冻结接口标准下证明有用，并以大脑功能（状态保持、积分、增益控制、门控、功能分区）验收，而不是送达或延迟。
