# LeJEPA 与 LeWM：可借鉴边界

## 结论

LeJEPA 不是一种专用世界模型架构，也不使用 zero-AdaLN。它提供架构无关的自监督目标：让同一输入的多视图表示一致，同时用 SIGReg 约束投影后的聚合分布接近各向同性高斯。官方最小实现直接使用普通 ViT/ResNet 等编码器。

LeWM 才加入时序动力学。它逐帧编码像素得到全局 latent，使用带因果遮罩的 Transformer 读取有限历史，并预测下一帧 latent。动作先经 MLP 编码，再在 predictor 的每一层通过 AdaLN-Zero 产生注意力与 MLP 的 shift、scale、gate；调制线性层的权重和偏置都从零开始。训练采用下一 latent MSE 与 SIGReg，推理时自回归 rollout，并用 MPC/CEM 搜索动作。官方配置默认历史长度为 3。模型内部只有状态序列 \(x\) 与单一动作条件 \(c\)：没有独立 Goal/Action/Horizon 条件通道，目标图像仅在 rollout 之后计算 MPC latent 代价。

V-JEPA2-AC 说明动作条件并不只有 AdaLN 一种正确形式。其官方 predictor 把每个时间步的动作 token、机器人状态 token 与该帧空间视觉 token 交错排列，再使用 block-causal attention。它适合动作/本体状态本身需要参与结构化推理的场景；LeWM 的 AdaLN-Zero 更像逐层动力学控制面。本项目采用二者分工：自然语言/结构化语义继续通过 MoT 交换，连续低维动作通过 zero-AdaLN 控制场演化；若以后动作变成离散工具或语言事件，再把动作同时表示为 MoT token。

## 与本项目的关系

LeWM 可以写成我们的同方差退化。若

\[
q(z_{t+1})=\mathcal N(\mu_q,\sigma^2 I),\qquad
p(z_{t+1}\mid b_t,a_t)=\mathcal N(\mu_p,\sigma^2 I),
\]

则条件 KL 的均值部分与 \(\lVert\mu_q-\mu_p\rVert_2^2\) 只差常数和比例；SIGReg 另行把聚合表示推向 \(\mathcal N(0,I)\)。[SIGReg–VFE 理论预印本](https://arxiv.org/abs/2607.13612)进一步证明：在编码噪声固定、总体各向同性高斯约束确实成立的条件下，这一目标可成为精确的信息瓶颈/VFE 退化，并可扩展到多步 expected free energy。该结论是条件性理论结果，论文把经验验证列为后续工作，也指出目标缺少 state-epistemic value；它不能把有限批次正则数值等同于逐样本 KL。

本项目已经显式学习 \(q(X_{t+1}\mid o_{\le t+1})\) 与状态条件 \(p(X_{t+1}\mid b_t,a_t,H_t)\) 的逐样本异方差 KL，并保留 RGB/文本/分割似然。因而 SIGReg 只保留为论文层面的“固定同方差、总体先验”退化对照；实验性转移 SIGReg 已从代码主路回退，避免双重先验压力抹平空间和模态结构。

## 采用与拒绝

采用：因果历史、下一信念预测、动作/时距的逐层 zero-AdaLN、训练时 teacher forcing 与推理时 prior-predictive rollout。

拒绝：另建独立 CLS 世界模型、让文本通过 AdaLN 取代 MoT、只优化 latent MSE、或永久丢弃全分辨率视觉场。

本项目中的对应形式应是

\[
b_t=q(X_t,H_t\mid o_{\le t}),\qquad
p(X_{t+1},H_{t+1}\mid b_t,a_t),
\]

其中固定数量 Slice 是每步瞬态计算工作区，终端 RGB、文本与分割似然在新观测出现时校正未来先验。历史版本的彩色数字任务没有动作且只给单帧，只能验证一阶 Markov 退化。当前代码已加入两帧有序历史、独立二维动作和同图未来后验；下一门槛是训练后证明历史/动作干预都能破坏对应未来预测，同时前三种能力不回归。

## Transolver 的跨时间方式

官方 NS 基准没有“跨时间搬网格地址”。输入 `fx` 是固定网格每点的 10 帧历史，模型每次输出下一帧一个标量场；训练滚动窗口写入真值，测试滚动窗口写入模型输出。Physics-Attention 在单层内用同一 `slice_weights` 读写，却仍能预测流场，因为 slice-token attention 学习的是固定欧拉网格上的非局部场算子。对本项目的直接启示是先补历史、动作和正确 rollout，而不是先发明 transport-Deslice。

## 可复用的完整套件

- Meta `eb_jepa`：图像、视频、动作条件视频 JEPA 与规划的教学式完整例程。
- Meta `vjepa2`：生产级视频编码器和 V-JEPA2-AC 动作/状态因果 token predictor。
- Meta `jepa-wms`：训练、动作/本体输入、可选解码、规划评测和权重的完整物理规划套件。
- `stable-worldmodel`：数据采集、训练、MPC/CEM 评估的统一平台，并包含 LeWM 训练入口。

这些仓库用于补齐世界模型实验协议，不替换本项目的全分辨率 Slice 主图。

## 独立条件通道诊断

实验曾把 Goal 与 Action 拆成私有 zero-AdaLN 路径，并把零精度路径的偏置固定为零。无 SIGReg 的 250 步诊断在完整银行上得到生成/当前/编辑/未来 0.892/0.824/0.821/0.227，未来主指标 0.137。历史干预方向正确（下降 0.022），但动作、时距与未来目标消融反而改善 0.030/0.042/0.027。私有通道实现随后回退，只保留负结果作为证据。

因此“不同语义分通道”不是世界模型能力的充分机制。LeWM 的低维 action AdaLN 在其全局 latent 任务中可用，但本项目的固定地址全分辨率演化需要动作与空间/Slice 表示显式交互。后续应优先测试 V-JEPA2-AC 式 action token × spatial token 因果耦合和配对反事实动作样本；完整动作条件必须在未来位置与分割两项上分别击败动作置零，才可进入主线。

最新受控实验表明，action token 加 Slice 相对几何只能在旧银行产生接近零的双正效应；解除 horizon/action 数据共线后，普通联合训练、冻结式分阶段训练和全局 PCGrad 都不能稳定同时保持静态能力与动作空间因果。v14 又把同一状态/历史下的五种动作组成配对观测能量并直接加入条件 VFE；能量对角只产生 +0.000373 的微弱优势，完整银行上动作位置/分割仍为负，因此该方案被否证为充分条件。

后续 v15 修复了 `(dx,dy)` 动作与 `(y,x)` 网格的轴顺序，但动作双指标仍不过门。v16 测试了动作条件的 Slice 分配 transport：固定欧拉点地址不动，只平流读分配并继续通过同一 Deslice 和终端观测似然训练。四层映射虽获得非零梯度，却只学到不足 0.1 像素且相互抵消的位移；完整动作位置/分割为负。诊断显示分配场与写入增量都近似空间均匀，因此这一 transport 形式被拒绝。

这与 Transolver 的固定地址演化不矛盾：真正需要演化的是固定地址上的空间解析场值，而不是移动地址，也不是平移近均匀的 Slice 概率。下一候选必须先分离持久信念的内容与固定坐标图表，再定义一次动作条件的欧拉场算子；不得把四层深度解释成四个时间步，也不得引入独立世界模型、未来私有读出或手写 RGB 平移。

v17–v18 进一步把动作写进与生成共用的 F2 prior action：只执行一次归一化 Slice-to-Slice 转移，再走原 Deslice。非饱和 v18 的转移矩阵已明显偏离恒等，且静态能力全部过门，整体动作双指标也转为微正；但单独关闭该转移对位置无影响，对分割反而略有帮助。因此缺口不在“有没有 VFE 动作公式”，而在瞬态 Slice 缺少跨时刻可配对身份。下一设计必须显式比较当前预测 Slice 与未来后验 Slice，或转向持久欧拉点场上的内容算子，不能把 transient Slice index 当作天然世界状态地址。

## 官方来源

- [LeJEPA 官方仓库与最小实现](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md)
- [LeJEPA 论文](https://arxiv.org/abs/2511.08544)
- [LeWM 论文](https://arxiv.org/html/2603.19312v3)
- [SIGReg 作为 VFE 的理论分析](https://arxiv.org/abs/2607.13612)
- [LeWM 当前官方仓库](https://github.com/Mengarr/lewm)
- [LeWM AdaLN-Zero predictor](https://github.com/Mengarr/lewm/blob/main/src/lewm/module.py)
- [LeWM rollout 实现](https://github.com/Mengarr/lewm/blob/main/src/lewm/jepa.py)
- [LeWM 训练目标](https://github.com/Mengarr/lewm/blob/main/train.py)
- [Transolver 官方 NS 训练与自回归代码](https://github.com/thuml/Transolver/blob/main/PDE-Solving-StandardBenchmark/exp_ns.py)
- [Transolver Physics-Attention](https://github.com/thuml/Transolver/blob/main/Physics_Attention.py)
- [V-JEPA2-AC action-conditioned predictor](https://github.com/facebookresearch/vjepa2/blob/main/src/models/ac_predictor.py)
- [Meta EB-JEPA 完整例程](https://github.com/facebookresearch/eb_jepa)
- [Meta JEPA-WMs 完整规划套件](https://github.com/facebookresearch/jepa-wms)
- [stable-worldmodel 与 LeWM 训练套件](https://github.com/galilai-group/stable-worldmodel)
