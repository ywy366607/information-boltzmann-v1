# GDN 因果标记：实际算子精化，训练保持暂停

> **后续：** 24层cache交接、原第2/13层入口、固定支持读取与记录写入已展开为[联合接线合同](CGB_JOINT_CACHE_ADMISSION_20261108.md)。此前native方向预算不含经历库方向；新文档补齐这一范围，仍未宣称实际FLA/hook已满足全部前提。

## 这轮解决什么

把[有限参与方案](CGB_BOUNDED_CAUSAL_TAGS_20261108.md)中抽象的一步导数拆到当前安装的纯GDN算子，明确局部参与怎样产生方向、怎样经后层到达同一个语言边界。只修改探索Lean与文档/索引；不加载模型运行，不训练，不执行影子Adam步，不推进实际生命史。

同一次身体递推搬运参与的影响，真实结果到来后捕获它。方向状态是学习侧的精确参考，不往前向注入答案，也不要求生物突触资格等于Jacobian。这是少量方向的前向敏感度，不是独立于链式法则的生物理论，也不是为全部参数维护完整RTRL矩阵。

探索源码：[gdn_operator_tags_20261108.lean](../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean)。下文“已证”指实数算子的条件性方向导数，不自动等同GPU实现已通过。

**本轮核对完成：** 新算子文件22条，加上有限标记文件新增的3条来源分账，共新增25条探索定理；两文件共41条均通过逐文件Lean内核与逐条公理依赖检查，只依赖项目允许的 `propext / Classical.choice / Quot.sound`。未运行生产区全量审计，不增加生产区或身体覆盖。主检查点SHA256与此前归档审计一致，11个实际执行源码哈希保持。证明记录为 `scratch/bounded_causal_tags_20261108/operator_proof_result.json`，源码证据为同目录 `operator_source_manifest.json`、`source_preservation.json`。

## 真实源码与证明对照

安装源位于 `E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2`。配置为24层纯GDN、hidden1024、4头、每头256维、短卷积宽4、norm_eps=1e-6、swish门和SwiGLU，没有窗口Softmax注意力。

| 实际路径 | 共同源与传递 | 本轮证明 |
|---|---|---|
| `fla/layers/gated_deltanet.py:142–146`：q/k/v/a/b投影 | d(Wx)=dW·x+W·dx；冻结权重不等于冻结输入 | `bilinear_read_deriv` |
| `scripts/ib_local/pretrained_gdn.py:streaming_short_convolution` | q/k/v旧4格投影缓存移位，四点乘积求和及SiLU | `shift4_deriv`、`causal_conv4_deriv` |
| q/k归一化 | 平方和、分母及每个坐标一起变化 | `square_sum_deriv`、两种`normalizer_deriv`、`normalized_coordinate_deriv` |
| β门 | sigmoid响应乘以b投影方向 | `beta_deriv` |
| g与α | g=−exp(A_log)·softplus(a+dt_bias)，α=exp(g) | `log_retention_deriv`、`alpha_deriv`；softplus光滑分支 |
| S转移 | S、key、value、α、β共同变化 | `delta_next_deriv` |
| q读S | dq·S+q·dS，含固定head scale | `query_read_deriv` |
| RMS+swish门+o投影 | 范数、norm权重、门和读出共同参与 | `weighted_norm_deriv`、`swish_product_deriv`、投影通则 |
| 两次残差与SwiGLU | 保留输入方向，门/上/下投影共同传播 | `residual_deriv`、`swish_product_deriv`、投影通则 |
| 冻结后继层 | 冻结参数仍传递输入与自身cache方向 | `frozen_successor_carries_direction` |
| 原LM头与语言损失 | 原词表误差收缩logit方向，内层无私有词表头 | `boundary_nll_deriv`、`boundary_nll_eq_original` |

## 同一递推的完整来源

令 M=αS，旧预测 p=kᵀM，写入误差 r=v−p。实际更新 S⁺=M+βkrᵀ。

由各输入/参数的局部导数推出，而非假定最终导数：

```text
dM  = dα·S + α·dS
dp  = dkᵀ·M + kᵀ·dM
dr  = dv − dp
dS⁺ = dM + dβ·k·rᵀ + β·dk·rᵀ + β·k·drᵀ
do  = scale·(dqᵀ·S⁺ + qᵀ·dS⁺)
```

`delta_next_deriv`逐坐标证明S更新；读出、归一化、门和o投影再把方向送到残差与MLP。下层影响进入后层S/卷积时，后层方向缓存同步更新，不因后层参数冻结而丢弃。

后来真实词元y到达，原边界CE提供 `c = Σ_a p(a)·dz_a − dz_y`。这是原NLL沿所追踪方向的导数，不是给所有突触广播无区分的奖励标量。用于最小化的延迟步取负号。

`spatial_stages_deriv`另证明：给定各个局部阶段的联合cache导数，可以依次组装任意有限层数（包括24层），不必把整身导数当成前提。`frozen_stage_transport`明确冻结阶段的新参数源为零，输入/cache方向仍须通过。这里是组装定理；实际24块及现有hook到该接口的逐字段实例化仍在下一步清单中。

## 窗口内外不重复计账

当前预训练路径的计算窗为128**词元**，不要与旧字节模型的128字节混用。

同一线性化转移下，把参与源分为该次结算窗内和窗外：`e_next = A·e + b_old + b_recent`。有限标记文件新增：

- `one_step_source_transport`：新参与源与旧标记输运分开；一次旧事件不能每拍被伪装成新源再注入。
- `transport_sources_split`、`capture_sources_split`：源的分账传到结果端仍严格可加。

实际组合必须是“原窗口梯度＋窗外所选参与信用”，不能把包含窗内源的整份敏感度又加到BPTT上。固定槽的出生时间、参数方向身份、本次结算窗口都须登记；一次事件与共享参数每拍重复参与的来源不同。当前未安装调度器，不能称已有实现已无重复计账。

## 实际分支差异

短序列q_len≤64走 `fused_recurrent.py:82–83`：`q / (sqrt(Σq²) + 1e-6)`。

chunk走 `chunk.py:182–184` 调 `l2norm_fwd`：`q / sqrt(Σq² + 1e-6)`。

两个实数算子不完全相同。本轮分别证明分母导数，没有修改它们，也没有从源码差异推断实测退化原因。此前特定数据的数值一致性不证明二者对任意状态恒等，后续精化必须注明分支。

Torch softplus在输入大于默认threshold20时用线性分支。当前 `softplus_deriv`只证光滑log(1+exp x)分支；线性区可用恒等导数，但跨边界及PyTorch自定义反向未由本文件覆盖。fused q/k范数为零时，当前sqrt证明的正范数前提也未覆盖；整体向量归一化在该点需单独处理，不据此断言整体不可微。

## 下一步：运行步级组装

1. 把坐标规则组装为同一块的 `(激活,S,q/k/v卷积cache)` 到 `(输出,新cache)` 转移，再串24层，核对所有残差、范数与cache。
2. 明确softplus、零范数、fused/chunk分支及浮点舍入；实数证明不替代数值误差界。
3. 固定K、参数方向、出生/释放与窗内外来源。K=2仍只是候选预算，不声称覆盖全部参数信用。
4. 当前附加经历hook若保留，其检索/编码/注入及离散选址也是实际转移，必须精化。本文件是原生GDN精化，尚未包含hook，不能冒充实际带hook身体已过。

本阶段之后已继续完成联合组装与有限压缩合同。**11-08 Claude接手收尾：**本文件新增8条分支/误差定理，当前30条通过单文件内核及逐条公理依赖检查；与联合布局、压缩误差两文件共141条本轮重查，系列共182条。记录`scratch/bounded_causal_tags_20261108/layout_numerics_proof_result.json`。

新增范围：softplus在严格小于/大于20处的导数，误差小于分支间隔时的选择保持，正分母归一化方向界，recall clamp严格低分支，recurrent向量归一化在零点的导数及二次余项，以及正下界下倒数误差。阈值等号、clamp边界、实际浮点格式与执行误差仍未认证；recurrent零点导数包含1/eps增益，不能由可微推出稳定。

当前下一步是将这些条件接到具体数值程序，不新增模型前向或训练。主个体维持524更新、seen450757，第14题反馈未到，训练保持暂停。
