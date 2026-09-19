# CBIM Infinite-Stream Evaluation & Training Protocol v1 (CBIM-ISP-v1)

> **核心原则**：
> 1. **状态永不重置（Never Reset State）**：流体工作记忆是无限流物理状态，在推理和长程评测中严禁清空为零（$F=0$）或冷真空；跨网格必须使用连续谱投影重采样。
> 2. **反向时间与物理时间分离（Decoupled Gradient Horizons）**：梯度截断视界 $H_{\text{BPTT}}$ 是训练算法的计算近似，不等于物理系统的动力学记忆寿命；严禁混淆 BPTT 长度与模型认知容量。
> 3. **报告必须透明定界（Strict Regime Disambiguation）**：严禁将 Cold-start、Warm 256 词元与 NESS 4096 词元的损失混为一谈。所有公开评测报告必须包含标准元信息 Header。

---

## 一、 标准评测元信息 Header 规范

所有 CBIM 基准测试、消融实验和公开发表报告必须附带以下标准元数据 Header：

```yaml
CBIM-ISP-v1 Header:
  Architecture: "CBIM Three-Clock v1"       # Old TCD / Three-Clock / Next-Gen
  Grid: "(8, 8, 4)"                         # 空间网格分辨率与节点数 (256 nodes)
  Train Ponder K: 3                         # 训练内部思考步数 (固定值或采样分布)
  Eval Ponder K: 3                          # 推理思考步数 (可任意零样本外推 1..128)
  BPTT Horizon: 128                         # 显式反向传播展开长度 (32 / 128 / randomized)
  Microstep Gradient: "full"                # 微步梯度类型: full / reversible / truncated
  State Policy: "persistent"                # 状态继承策略: persistent (继承成熟流) / cold_reset
  Training Reset: "cross-chunk persistent"  # 训练步间状态继承: persistent / chunk_reset
  Warmup Tokens: 256                        # 评测前缀预热 Token 数 (0 / 256 / 512)
  Eval Stream: "OWT validation.npy"         # 评测数据流与具体切片索引
  Eval Length: 128                          # 打分评估视界 (128 / 1024 / 2048 / 4096)
  Metric: "Full NLL (nats)"                 # 评价指标 (nats / bpc / delta-causality)
  Compute: "41.5 tok/s, 1.8GB VRAM"         # 硬件推理速度与峰值显存
  Checkpoint: "Step 3000 (Best Val)"        # 检查点步数与选择规则
  Seed: 11                                  # 随机数种子
  State Regime: "warm_persistent"           # 运行域: cold_start / warm_persistent / ness_long_stream
```

---

## 二、 三大标准评测域（Evaluation Regimes）

为彻底杜绝“历史 6.44 nats 打赢 GDN”与“大盘 7.41 nats”造成的数字混淆，官方基准评测严格划分为三个正交测试域：

### 1. Cold-Start Regime（冷启动瞬态域）
- **状态定义**：$F_0 = 0$（冷真空），`Warmup Tokens = 0`；
- **测试目的**：评估系统从零物理激发开始吸收首批语义激波的瞬态响应能力，测试声学阻尼与 NESS 随机相位预热（Random-Phase Warm Start）的效果；
- **标准指标**：前 64~128 个词元的平均 NLL 及 Step-0 到 Step-10 的能量振荡峰值。

### 2. Warm Persistent Regime（暖机驻波域 —— GDN 对齐标准）
- **状态定义**：继承训练成熟态，或在目标文章前缀施加 **256 Token 的共享历史预热**（`Warmup Tokens = 256`）；
- **测试目的**：在流体场已建立起当前文章的语义相干驻波后，评测模型在后续 $128$ 词元视界内的即时因果推理能力与物理碰撞增益；
- **历史成绩锚点**：
  - Run 3 Continuous Q8 (Arm C): **$6.4409\text{ nats}$**（Site 3）；
  - Three-Clock Run B (Arm C): **$6.5725\text{ nats}$**（Site 3，因果增益 $+2.0228\text{ nats}$）；
  - **全网 4-Site 官方平均**：平均 NLL $\sim 7.51\text{ nats}$，平均碰撞因果贡献 $+0.786\text{ nats}$。

### 3. NESS Long-Stream Regime（非平衡稳态长程流域）
- **状态定义**：连续喂入 **$2048 \sim 4096+$ 个 Token**，状态全程不重置、不截断；
- **测试目的**：检验系统在无限长文本流下的数值稳定性、长期记忆衰减谱以及能量耗散平衡（NESS）；
- **标准指标**：
  - 全序列平稳 NLL（Run B 在 4096 词元大盘取得 $7.4100\text{ nats}$）；
  - 稳态场能量 $E_{\text{NESS}} \in [0.55, 0.85]$；
  - 晚期词元（Token 1500~2048）的平均 NLL 衰减趋势。

---

## 三、 对决 Gated DeltaNet (GDN) 的法定对齐规则

与循环神经网络或线性注意力的权威基线（如 GDN、GDN-2）进行对决时，必须执行“完全对齐”铁律：
1. **同数据切片**：使用完全相同的 OpenWebText 验证集划分与文档位点；
2. **同预热长度**：完全相同的 Warmup 长度（统一为 256 tokens）；
3. **同预测视界**：完全相同的打分视界（统一为后续 128 tokens）；
4. **同参数与计算预算**：报告同等训练 Token 数（如 384k tokens）下的检查点性能；
5. **明确推理计算量**：报告推理时的 FLOPs 与显存占用（CBIM $K=3$ 推理成本与标准 2 层线性注意力相当）。

### CBIM 专有物理自由度 Scaling 曲线：
除常规模型对比外，CBIM 报告两条传统 Transformer/RNN 无法生成的专有连续缩放曲线：
- **$L(K)$ 内部思考深度曲线**：在同一个权重下，测试 $K=1, 2, 3, 8, 16, 32, 64, 128$ 的零样本零外推损失收益；
- **$L(N_{\text{grid}})$ 神经算子分辨率曲线**：在同一个权重下，测试 32 到 2048 节点的零样本跨网格离散不变性。

---

## 四、 无限长流训练架构演进路线图（Infinite-Horizon Roadmap）

针对“模型状态活无限久，但反向传播受制于有限 GPU 显存”的本质矛盾，确立三层梯级解法：

```
+-----------------------------------------------------------------------------+
| Layer 3: Slow-Mode Eligibility Traces (q ≈ 0 慢模态长程正向积分资格迹)        |
|          只让守恒/近中性子空间 P_slow 携带长程梯度 (几百至上千 Token)       |
+-----------------------------------------------------------------------------+
                                       ▲
+-----------------------------------------------------------------------------+
| Layer 2: Randomized Long-Horizon BPTT (无偏随机几何分布截断)                 |
|          以期望值 E[H]=128 采样变长 BPTT 视界，彻底消除人为固定 32/128 边界  |
+-----------------------------------------------------------------------------+
                                       ▲
+-----------------------------------------------------------------------------+
| Layer 1: Reversible Pondering Backprop (内部时间可逆辛反传)                  |
|          (TC)^K 保持严格酉性与守恒，内部 K=64/128 无需显存激活检查点，精确反传 |
+-----------------------------------------------------------------------------+
```
