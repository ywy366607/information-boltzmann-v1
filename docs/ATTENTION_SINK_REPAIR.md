# TToken、注意力 sink 与高斯头修复

## 论文如何对应本项目

参考 [Fu et al., arXiv:2602.01203v3](https://arxiv.org/html/2602.01203v3)。论文把多头
注意力解释为头级专家与门控：当 sink 的 Value 为零时，非 sink 输出被
`g = 1 - attention_sink_mass` 缩放。微调时保留最活跃的共享头，对其余头的平均
门控施加 CV² 平衡正则。该正则是优化辅助项，并不自动等价于本项目的 VFE。

TToken 可以通过共享注意力影响头的任务分工，但它携带非零语义 Value，不能仅因
注意力很大就被称为零值 sink。我们的第二层某头对任务 token 的质量达到 0.987，
该 token 的 Value 范数为 2.979（全体均值 2.476）；它仍在传递内容。四层按
输出投影后贡献幅度计算的有效头数为 3.293/2.350/3.761/3.669。这不是论文的
gate-importance 指标，也不足以证明全网 head collapse。

## 已确认的实现缺口

1. **控制 token 被因果遮罩挡住。** 原先把它追加在 H 尾部，所有普通文本查询
   对它的直接注意力严格为零。`control_prefix_attention` 使其具有观测前缀的
   可见性；控制查询只能读取 prompt/控制与视觉，避免把答案带回下一层。
2. **高斯参数按错维度拆分。** 逐头 MLP 的输出本应是
   `[head, (mean, logvar), channel]`，原先直接 flatten 后全局 chunk。
   四头时，mean 只接到前两个头，logvar 只接到后两个头。修复先转成
   `[(mean, logvar), head, channel]` 再展平。逐头梯度测试覆盖先验和排列语义。
   此错误造成结构性职责分裂；它是否是生成失败的充分原因由匹配重训判断。
3. **32px 身份评估没有覆盖真实渲染。** 原 sliding evaluator 只搜索偶数
   box，32px 银行实际 box=11；绝对坐标取整还会改变模板。
   完美目标在旧指标上只有 0.411。新增 `paired_digit_scores` 使用相同地址和
   渲染参数比较十种字形；16/32/64px、十数字、九位置的完美目标测试全通过。
   这是配对身份指标，不能冒充平移不变 OCR，旧指标保留作历史对照。

## 保持图的一致性

所有修复仍在 X–Slice–H–Deslice 图内。TToken 不选择私有主干。独立零值 sink
是可选注意力门，每层每模态每头一个标量；不牺牲 TToken 的语义 Value。
GDN-2 的物理时距边界没有改变。旧权重未包含新配置时继续使用 legacy 布局和
原因果遮罩，避免静默改变已发布结果；新候选显式记录配置并独立重训。

## 实验与解释边界

- `audit_attention_write.py`：实际头贡献、任务 Value、逐层差异幅度、写路径干预。
- `attention_repair_{control,prefix,sink}.json`：同一初始权重、600 步、同数据顺序
  和余弦退火的三臂比较；sink 臂额外采用 0.01 头平衡系数。
- `train_gaussian_layout_probe.py`：固定中心/红色、十数字、三个静态模式共 30
  样本，从零对比旧/新高斯布局；两臂都使用可见控制 token，不加 sink。
- `audit_registered_digits.py`：同权重重新报告配对身份与完美目标上限。

此前“有效秩下降证明后层擦除身份”说得过强。当前轨迹的差异 RMS 从约 0.009
增至 0.131，而参与率有效秩从约 4.86 降至 1.89；这说明谱能量集中，不等于
信息消失。晚层 prior 减半/关闭、local 关闭均未恢复生成，不能据此直接上循环。
正确任务 token 也不必对所有错误 token 都严格领先：有些输入下任务目标相同。
互换是机制诊断，任务正确性和真实条件依赖才是训练目标，不应刻意损坏错误-token
输出以制造间隔。

## 600 步匹配续训结果

| 配置 | T2I IoU / 配对身份 | 重建 IoU / 配对身份 | 编辑 IoU / 配对身份 |
|---|---|---|---|
| 同预算原结构续训 | 0.274 / 0.178 | 0.971 / 0.989 | 0.707 / 0.911 |
| 控制前缀可见 | 0.274 / 0.167 | 0.974 / 0.989 | 0.725 / 0.933 |
| 控制前缀 + sink + 平衡 | 0.274 / 0.156 | 0.973 / 0.989 | 0.737 / 0.933 |

三臂仍使用旧高斯布局、相同 seed/初始化与数据顺序。没有一臂解决空图生成。
sink 以 -4 logit 接入以近似保持初始函数，训练后非 sink 门仍约 0.999；所以本轮
只排除这一温和接入配方的充分性，不能否决更充分训练的 sink/gated attention。
控制前缀让文本读取 TToken 的质量由严格 0 变为首层约 0.120，证明修复实际生效。
原统一候选的配对身份为生成/重建/编辑 0.189/0.956/0.744；生成失败仍存在，
重建和编辑的旧身份分数则明显低估了实际表现。

## 高斯布局的匹配重训

两臂均从相同随机初始化训练 1200 步：前 400 步仅生成，随后联合生成/重建/编辑；
30 个固定样本，32px、中心地址、红色源图，学习率 0.001 并余弦退火。
两臂都启用控制前缀，保留 F2 与同一终端似然，关闭 sink 和额外逐层监督。

| 重载选择权重 | 生成身份 / IoU | 重建身份 / IoU | 编辑身份 / IoU | I2T 类别准确率 |
|---|---|---|---|---|
| legacy 布局 | 0.900 / 0.736 | 1.000 / 0.961 | 1.000 / 0.967 | 0.700 |
| per_head 修复 | 0.900 / 0.863 | 1.000 / 0.953 | 1.000 / 0.950 | 0.700 |

修复使生成几何改善，但重建/编辑 IoU 略低，不能称为全指标提升。修复臂的第 400
和 1200 步生成身份曾达 1.000；宏分选出的重载权重为第 1000 步，身份 0.900，
必须以这一实际存档结果为准。旧布局也能学到九种数字，故此错误不是所有生成失败
的充分解释；更小数据覆盖和不同学习率可显著改变结果。后续联合收尾是额外预算，
与匹配对照分开保存在 `gaussian_layout_closure.json`，不能混作同预算改善证据。

## 最终固定集能力证明

修复臂从已存档权重再联合训练 800 步（初始 LR=0.0003），随后 400 步
（LR=0.00003），每段均余弦退火；无循环、无 sink、无新分类器。总运行预算
为 1200+800+400 步，中间按记录选择权重并重置优化器。

独立加载 `checkpoints/gaussian_layout_per_head_finish.pt`：

| 任务（每项十样本） | 配对数字身份 | RGB IoU | 文本答案 | 分割 IoU |
|---|---|---|---|---|
| 文生图/文生文 | 1.000 | 0.979 | 1.000 | 1.000 |
| 重建/图生文 | 1.000 | 0.993 | 1.000 | 1.000 |
| next-color 编辑 | 1.000 | 0.996 | 1.000 | 1.000 |

仅轮换生成提示中的数字，保留目标，生成身份降至 0；仅轮换重建/编辑的源图，
对应身份也降至 0，IoU 均降到约 0.30。编辑答案仍为同一个目标颜色，因此其文本
准确率不应因数字换图下降。结果见 `gaussian_layout_finish_causal.json`。

范围严格为**32px、中心位置、红色源图、十数字、三个静态任务**；编辑只覆盖红色
到下一颜色，不能证明完整换色规则。该检查点是单图固定集容量证明，不是完整银行
冠军，更不包含自然图、冻结 Pythia 或未来预测。下一步保持这些代码修复和新的
身份度量，逐步扩到四颜色/九位置后重训验收，再接回已规划的其余端口。

复现主命令：

```powershell
python scripts/train_gaussian_layout_probe.py --steps 1200
python scripts/train_gaussian_layout_probe.py --layouts per_head --steps 800 --warmup-steps 0 --lr 0.0003 --init checkpoints/gaussian_layout_per_head_probe.pt --tag closure --out results/published/gaussian_layout_closure.json
python scripts/train_gaussian_layout_probe.py --layouts per_head --steps 400 --warmup-steps 0 --lr 0.00003 --init checkpoints/gaussian_layout_per_head_closure.pt --tag finish --out results/published/gaussian_layout_finish.json
python scripts/train_gaussian_layout_probe.py --layouts per_head --init checkpoints/gaussian_layout_per_head_finish.pt --audit-only --out results/published/gaussian_layout_finish_causal.json
```
