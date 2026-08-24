# 现状简报（给外部模型指导本仓库代理）

日期：2026-08-25。
架构唯一来源：[`NORTH_STAR.md`](NORTH_STAR.md)。实施计划：[`PYTHIA_INTEGRATION_PLAN.md`](PYTHIA_INTEGRATION_PLAN.md)。证据树：[`../research_tree.json`](../research_tree.json)。

本文是**操作快照**，不是新北极星。冲突时以 NORTH_STAR 为准。

---

## 0. 给指导模型的硬约束

指导本仓库代理时必须遵守：

1. **一个图、一个检查点。** 全分辨率可写点场 \(X\)，瞬态 Slice \(S\)，语言场 \(H\) 联合演化。`SliceRead → MoT(S,H) → Deslice → X`。不得外挂扩散/UNet/独立 VLM，不得用 `lm.generate()`。
2. **Pythia 只提供 tokenizer + 冻结 embedding + 冻结因果分布。** 请求 Pythia 必须精确 ID，失败直接报错，禁止静默 GPT-2/TinyLM。
3. **生成是语言条件的 F2 prior action 写回 \(X\)**（`prior_write=1` 时 `Deslice(μp−S)`），不是 Pythia 画图，也不是 `(dx,dy)` 动作。
4. **GDN-2 只允许 `tau>0`。** 不要为文本或 T2I 打开物理时间先验。
5. **视觉 query 不得读答案 token**（`prompt_mask`）。改答案后缀时 \(X^*\) 与更早 logits 必须不变。
6. **T2I 门已过的 Pythia 检查点不得覆盖、不得当试验垫。** 见 §3 保护名单。
7. **官方编辑门是 next-color 规则，不是 named-color。** 见 §5。不要把 `Paint the stroke red` 的分数当成能力闭环编辑。
8. 缓存必须在 `D:\ml_cache`（`ML_CACHE_ROOT`）。Windows。测试：`pytest -q`。

---

## 1. 产品目标（未完成）

一个检查点原生完成：T2T、I2T、IT2T、T2I、IT2I 编辑、重建、分割、下一帧（`tau>0`）。
当前 toy 能力闭环部分通过；**真实语言（Pythia token NLL + 图上 decode）未完成。** 对话/看图问答不算已具备。

### 最新静态能力闭环（2026-08-24）

旧编辑路线的根因已经定位并修正：纯生成视觉冠军本身不会当前帧重建；旧 T2I 使用 `omni_tasks.one_sample`，而编辑/重建使用 `capability_sample/grid_digit_mask`，实际是两套笔画几何；重建还错误地把 `Reconstruct current frame` 设成了 `pi_text=1` 内容证据。

新入口 `scripts/train_pythia_capabilities.py` 从统一能力冠军启动，所有端口共用 `capability_sample`。重建严格使用“图像存在、文本缺失、tau=0”。Pythia `text_in` 先以闭式 ridge 对齐到 toy 冠军已证明的 post-`text_in` 坐标；同色同位置十数字再用 10×10 RGB+分割观测能量识别，未增加模型、分类头或旁路。

新冠军 `omni_d64_pythia_capability_b3_edit_best.pt` 独立重载三门通过：T2I digit/color/IoU = **1.000/0.963/0.936**；重建 digit/color/IoU/seg = **0.911/0.903/0.883/0.975**；官方 next-color 编辑 digit/color/IoU/seg = **0.878/0.940/0.697/0.991**。编辑 source digit gap **0.789**、source IoU gap **0.631**、flood **0.028**。Pythia 冻结、无 GDN-2；stem/SliceRead/Deslice/seg head 全程冻结。

---

## 2. Pythia 接入进度

计划阶段：

| 阶段 | 内容 | 状态 |
|---|---|---|
| A | 精确加载、`forward_tokens`、因果 mask、结构测试 | **结构完成；训练接口已按审查修正** |
| B | 能力冠军视觉图 + 冻结 Pythia，闭合 T2I/重建/分割/编辑 | **完成；B3 三门通过** |
| C | 统一 collator，token NLL，T2T+I2T 再混其它端口 | **部分：T2T/IT2T 过，I2T 因果过但准确率未过** |
| D | 统一图 greedy decode（每 token 重跑 MoT） | **已实现；T2T/IT2T 过，I2T 未过** |

### 最新 token 结果（阶段 C/D，未获准为冠军）

入口为 `scripts/train_pythia_tokens.py`；统一 token 边界在
`fine_grain/token_tasks.py`。训练答案 token 只作因果 teacher forcing，
`visual_prompt_mask` 禁止答案或已生成 token 回写视觉场。解码每个 token 都重新
调用 `forward_tokens`，没有调用 `lm.generate()`。

候选 `omni_d64_pythia_token_candidate.pt` 独立重载结果：T2T token/greedy
**0.983/0.950**；官方 next-color IT2T **1.000/1.000**；I2T
**0.444/0.400**。I2T matched-vs-shuffled 中位 NLL 差 **1.289 nat/token**，
证明模型确实看图，但数字解码准确率不够。静态 T2I/current/edit 三门仍全部通过，
B3 保护检查点 SHA256 仍为
`64E79D3B7D8757A4C50AA12E880993A92312F4237986D840466B689FCA4F086B`。
因此没有创建 `omni_d64_pythia_token_best.pt`，候选不可称为统一冠军。

已否/收缩路径：只训零门 token interface 时 I2T 近随机；开放只读 terminal
SliceRead 与最后一层 H-only reader 后最高约 0.60，但与 T2T/IT2T rehearsal
存在冲突；固定 4×4 atlas 虽在线性探针上为 1.00，真实冻结 Pythia 训练反而更差，
故仅保留 `--terminal-atlas` 实验开关且默认关闭；prototype/ridge 对齐也默认关闭。

代码入口：

- `fine_grain/llm_backend.py`：`load_frozen_pythia` / `canonical_pythia_id`
- `fine_grain/omni_model.py`：`DualStreamOmni.forward_tokens`、`non_lm_state_dict`、`language_meta`
- `fine_grain/pythia_bridge.py`：加载生成冠军、优化相位 `interface` / `token_interface` / `language` / `language_rgb` / `edit_spatial` / `joint`
- `scripts/train_pythia_generation.py`：T2I/编辑训练
- `scripts/train_pythia_capabilities.py`：统一场景 B2/B3（T2I、重建、分割、官方 next-color）
- `scripts/train_pythia_tokens.py`：阶段 C/D token NLL、反事实门与逐 token 图上 decode
- `fine_grain/token_tasks.py`：统一 token collator、答案因果边界与 graph greedy
- `scripts/diagnose_pythia_language.py`：toy vs Pythia 语言敏感度
- 测试：`tests/test_pythia_omni.py`、`tests/test_pythia_generation.py`、`tests/test_pythia_capabilities.py`

本地权重：`EleutherAI/pythia-70m`（`d_llm=512`）已在 `D:\ml_cache`。优先 70m，结构稳定后再 160m。

---

## 3. 检查点（不要覆盖保护名单）

| 文件 | 角色 | 可覆盖？ |
|---|---|---|
| `checkpoints/omni_d64_pythia_language_best.pt` | **Pythia T2I 门通过** | **否** |
| `checkpoints/omni_d64_northstar_omni_active_f2_grid_best.pt` | toy 语言生成冠军，T2I 数字/颜色/位置 1.00 | 否 |
| `checkpoints/northstar_slice_capability_best.pt` | toy 能力闭环冠军（生成 0.930 / 当前 0.843 / 编辑 0.865 / 未来 0.375） | 否 |
| `checkpoints/v1_bayes_2000step_f2_vfe_run1_best.pt` | F2 识别 | 否 |
| `checkpoints/v1_bayes_2000step_upd_rms_run1_best.pt` | Champion B 识别 | 否 |
| `checkpoints/omni_d256_unified_fdesc_center_4k_best.pt` | d256 生成 4k | 否 |
| `checkpoints/omni_d256_unified_t2i_s0band_best.pt` | 历史 residual/band T2I，1px 未解决 | 否 |
| `checkpoints/omni_d64_pythia_named_edit_spatial_best.pt` | named-color `edit_spatial` 步 40，T2I 0.956；几何未过 | 否（阶段 1 起点） |
| `checkpoints/omni_d64_pythia_capability_b2_best.pt` | 统一场景 T2I + 当前重建/分割双门冠军 | **否** |
| `checkpoints/omni_d64_pythia_capability_b3_edit_best.pt` | **统一场景 T2I + 重建/分割 + 官方 next-color 三门冠军** | **否** |
| `checkpoints/omni_d64_pythia_edit_best.pt` | Pythia 编辑混训失败件，非冠军 | 可删/可覆盖 |

Pythia T2I 指标（90 格、四色循环、`results/published/pythia_language_reader.json`）：

- digit top1 **0.967**，打乱数字词 **0.011**
- color **0.988**，打乱颜色词 **0.00**
- paired IoU **0.960**，centroid **0.002**，flood **0.002**
- 门：digit≥0.95、颜色≥0.95、IoU≥0.85、flood≤0.10、两种 shuffle 差≥0.50

图结构：d=64，M=16，4 层，res=16，`surprise_mode=v1_bayes`，`prior_write=1`，`deslice_write=increment`，无 residual_read。Pythia 冻结。可训过的是 `text_in/out`、MoT **语言**专家、F2 `prior_head`。`language_rgb` 曾解开 `pix_head`，那是全局涂色捷径，不是合法编辑路径。stem / SliceRead / Deslice / 视觉专家来自 toy 生成冠军。

---

## 4. 关键诊断（不要再走已否路径）

Toy 冠军能过 T2I，是因为词表里 `7`/`green`/`top` 各是可训的 64 维行，和视觉一起训。接到 Pythia 后：

- 分词**对齐**（都是 13 token，数字在第 3 个：`Ġ7` vs `Ġ1`）。
- 只训 `text_in/text_out`、冻住 MoT：**失败**。换数字后 RGB rms 0.0037，H 差 0.005；注意力几乎均匀。数字 top1 停在 ~0.11，打乱数字词不变。
- 整网联合 + 全绿银行：**失败**。颜色 1.00 无意义（全绿），位置 0.62，数字仍 0.11。RGB BCE 靠「绿团画在格子里」就能降。
- **有效配方：** 解冻语言侧 MoT + F2 language prior；四色银行；digit-shuffle 损失 `relu(BCE_match − BCE_shuffle + m)`。颜色/位置先过因果门。数字卡在正好 0.90 = 全部 9 格的 **6 被模板判成 8**。6 过采样 + shuffle 改 6↔8 后 top1 到 0.967。

不要建议：只训投影、全绿银行、为 T2I 开 GDN-2、调用 `lm.generate()`、用 (dx,dy) 当 T2I。

---

## 5. 编辑：两件不同的事

**官方能力门（toy 冠军过了 0.865）** 只有：

```text
Change the stroke to the next color
```

目标色 = 调色板循环 `red→green→blue→yellow→red`。句子里**不出现**目标色名。IT2T 答色名，IT2I 换色。这是闭集规则，靠 toy 词表里的 `Change`/`next`/`color` 行。

**Named-color 编辑不是这个门。** 它只出现在旧 omni `i2i`：`Paint the stroke {red/green/...}`，且 50% 是 `Restore the thin stroke`。T2I 里的 `thin green stroke` 是生成条件，不是 IT2I。

Pythia 上 40% 混 next-color + named-color、保 T2I：编辑 color_acc **0.22**（随机），score **0.13**；T2I 从 0.967 掉到 ~0.93。失败件 `omni_d64_pythia_edit_best.pt`。原因：next-color 是规则；语言侧仍是「从黑图画」；编辑要从已有笔画换色。

若做 named-color（`Change the stroke to red`），必须**单列任务**，不得冒充官方编辑门。官方门仍是 next-color。任何编辑训练必须监视 T2I 门，digit top1 低于 0.92 不得存盘覆盖语言冠军。

**400 步 named-color 不能当作「冻结读图无效」的证据。** 那次训练实现没有跑预期实验：

1. `one_step` 已写 source-shuffle hinge，但编辑 batch 被传入 `shuffle_coef=0.0`，shuffle 只出现在评测。
2. `--modal-precision` 只创建了零初始化坐标，forward 没传 `image_precision`/`text_precision`，默认都是 1。
3. `language_rgb` 解冻 `pix_head`、冻结精度坐标和视觉读写。RGB 可通过改全局颜色解码下降，不必读源图几何。结果 flood 0.59、named digit 0.11、paired IoU 0.09。
4. 单批梯度：编辑在 stem/read/text/prior 上约为 T2I 的 6–8 倍，prior cosine −0.25、stem −0.18。50% rehearsal 不是优化意义上的 50:50；禁止直接 `joint`。

纠正后的契约：编辑 batch 用独立 `--edit-shuffle-coef`（从 0.1 起）；T2I `π_image=0`、named edit `π_image=1`、两者 `π_text=1`；相位 `edit_spatial` 训练精度坐标 + 语言侧 MoT/F2 prior，冻结 `pix_head`/stem/SliceRead/Deslice；edit 比例 10%–20%，接口学习率 `1e-4`，先 100 步。source shuffle 必须换 digit/place，保持同一目标颜色。100 步停止条件：T2I digit 始终 ≥0.95；named digit − shuffle ≥0.30；flood <0.10；paired IoU 明显高于 0.09。只有 shuffle 真正启用后仍无几何差，才以 `1e-5` 解冻 SliceRead、视觉 MoT query/output 和 Deslice；stem 与 pix_head 仍先冻结。

---

## 6. 对话 / I2T

仍不能宣称对话或完整 I2T。阶段 C/D 已接通真实 token NLL 与图上 greedy，
但当前 I2T token/greedy 仅 0.444/0.400，未过 0.80 准确门。

- `pythia-70m` 是 The Pile 基座，非 instruct。
- T2T 与 next-color IT2T 已在同一候选过 token/greedy，但不能掩盖 I2T 失败。
- 禁止 `model.lm.generate()`；现有 decode 每 token 回跑统一图。

---

## 7. 建议下一步（按优先级）

指导代理时选一条，不要并行拆图：

1. 以 `omni_d64_pythia_capability_b3_edit_best.pt` 为唯一静态视觉起点；不要回到 named-color、旧 `one_sample` T2I 银行或纯生成冠军。
2. **只修 I2T token accuracy。** 当前因果差已过，不要继续放大 shuffle margin；目标是在保住 T2T/IT2T 与静态三门时把 I2T token/greedy 从 0.444/0.400 提到至少 0.80。不得加数字分类头或输出词表旁路。
3. 图上 greedy 已实现；继续保持每 token 重跑同一 Slice–MoT 图，禁止 `lm.generate()`。
4. 不要重开 GDN-2/世界模型当当前门槛；I2T token 与 decode 闭合后再恢复 `tau>0`。

三门只读复验：

```text
python scripts/train_pythia_capabilities.py --device cpu --lm-device cpu --init checkpoints/omni_d64_pythia_capability_b3_edit_best.pt --load-language --no-lexical-ridge --eval-only --edit-steps 1 --out results/published/_eval_pythia_capability_b3_edit_best.json
```

复现 T2I 评测（只读 `--init`，不加载失败的 edit ckpt）：

```text
python scripts/train_pythia_generation.py --init checkpoints/omni_d64_pythia_language_best.pt --load-language --eval-only --out results/published/_eval_language_best.json
```

加 `--modal-precision` 会打开零初始化的 image/text precision 坐标。已复验：与默认评测同为 digit 0.967，不回归。训练时必须把 `image_precision`/`text_precision` 传入 forward。

`labels=None` 的 `forward_tokens` 只出 logits，`token_nll=None`（推理前缀全是已观测 token）。训练必须显式给 `labels`。`text_out`/`proj` 带零初始化残差门（`text_out_gate`/`proj_gate`）；`token_interface` phase 只训这两条和门，避免阶段 C 一上来破坏 T2I。保存门与官方 T2I 门相同（digit≥0.95 等）。named-color 与 next-color 银行不再混合。结果 JSON 必须包含 `run`（相位、modal precision、shuffle 系数、是否存盘）。

当前探针已跑完 100 步（`results/published/pythia_named_edit_spatial.json`）。source-shuffle **进了优化**（`source_shuffle_trained=true`，步 100 的编辑 batch 记录了 `source_shuffle=0.077`、`image_precision=1`）。保护检查点哈希未变。

100 步停止条件：

| 条件 | 结果 |
|---|---|
| T2I digit 始终 ≥0.95 | **否**。步 1=0.967，20=0.944，40=0.956，60=0.944，80=0.900，100=0.844。最佳存盘步 40，重载后 T2I 0.956 |
| named digit − shuffle ≥0.30 | **否**。全程 digit≈shuffle（0.03–0.19）；步 80 最大差 +0.055 |
| flood <0.10 | **是**（冻结 pix_head 后 0.04–0.08；旧 `language_rgb` 是 0.59） |
| paired IoU ≫ 0.09 | **否**。0.03–0.08 |

结论：这不再是「shuffle 被关掉」的假实验。冻结读图路径 + 只训语言侧/精度坐标，**没有**几何差值；颜色可以慢慢到 ~0.50 而不靠全图涂色。T2I 在 1e-4、15% edit 下仍会被啃。最佳仍是步 40 的 `omni_d64_pythia_named_edit_spatial_best.pt`。

第二阶段 `edit_read` 已从该检查点跑完 100 步（`results/published/pythia_named_edit_read.json`）。解冻 SliceRead / `Wq_v` / `Wo_v` / Deslice，`visual_lr=1e-5`；stem 与 pix_head 仍冻。`n_train` 287492 → 372440。10 个编辑步、90 个 T2I 步；步 100 记录了 `source_shuffle=0.087`、`image_precision=1`。T2I 官方门从未守住，**没有存盘**。语言冠军与 spatial 检查点未覆盖。

| 步 | T2I digit | named−shuffle | flood | IoU |
|---|---|---|---|---|
| 1 | 0.922 | 0.000 | 0.044 | 0.037 |
| 40 | 0.944 | −0.083 | 0.057 | 0.039 |
| 60 | 0.900 | +0.056 | 0.077 | 0.046 |
| 100 | 0.878 | −0.056 | 0.201 | 0.092 |

解冻 Deslice 后 flood 从 0.04 升到 0.20；几何差仍不存在。不要据此再开 `joint` 或解冻 pix_head。

```text
python scripts/train_pythia_generation.py --init checkpoints/omni_d64_pythia_named_edit_spatial_best.pt --load-language --modal-precision --opt-phase edit_read --edit-ratio 0.15 --edit-style named --edit-shuffle-coef 0.1 --interface-lr 1e-4 --visual-lr 1e-5 --steps-language 100 --eval-every 20 --ckpt checkpoints/omni_d64_pythia_named_edit_read_best.pt --out results/published/pythia_named_edit_read.json --device cuda
```

相位：`language` = text 专家 + prior；`rgb_likelihood` = 只训共享 RGB 均值/方差头；`edit_spatial` = 再加精度坐标、冻结 pix_head；`edit_read` = 再加 SliceRead / 视觉 query-output / Deslice；`generation_write` = 只开语言→视觉 K/V、F2 prior、Slice/Deslice query-output、精度坐标和共享 RGB 头，stem 与终端语言 reader 冻结；`language_rgb` = 语言演化与 pix_head 同时解冻，真实轮训已证会破坏编辑；不要默认 `joint`。

---

## 8. ShareGPT-4o 真实数据 pilot

真实数据入口已经落地，详细记录见 `docs/REAL_DATA_PILOT.md`。本地通过远程
Range/ZIP64 按需抽取了 **134 T2I、142 IT2I、512 I2T**，930 个引用图像均
可解码，不下载 262 GB 全库。完整扫描 OpenGV 57,289 条首轮单图会话后，全部
都是通用看图描述，**没有真正的 IT2T/VQA**。此前“128 IT2T”是 caption
paraphrase 分类器漏判，相关 IT2T 指标全部作废；不得再从这个数据集伪造 IT2T。
两套数据仍只作为同一个 X–Slice–H 图的边界条件和终端似然。

64×64 安全候选从 B3 启动，只更新统一 RGB 均值/方差似然头的 646 个参数。
180 步后 T2I/IT2I/重建 NLL 分别为 `0.0614/0.0893/0.0681`；IT2I
source-shuffle gap `+0.00589`，RGB MSE `0.2255→0.2238`；重建 RGB MSE
`0.1825→0.1784`。重新以 16×16 审计，T2I/current/edit 三门全过。

语言接口根因也已定位：安全 RGB 候选的 `proj_gate/text_out_gate` 都是 0，视觉
token 和 logits 对 source shuffle 完全不变，并非 Pythia 预训练不足。以已有
synthetic token reader 打开接口，再只合并互不重叠的安全 RGB 似然头，训练
410 条 I2T、留出 102 条。300 步后 held-out NLL `9.977→4.602`；平均
matched-vs-shuffled gap 保持为正（`+0.639→+0.173`），正 gap 样本占 52.9%。
这证明冻结 Pythia 可作为语言 prior，且视觉到语言消息路径已生效；但逐样本因果性
仍弱，不能晋级。

随后完成了更基础的 **5 样本真实图像过拟合门**。五条目标是短而唯一的自然描述，
训练显式监督 EOS；Pythia 冻结，只训 `token_reader`。step 0 为 NLL 6.947、
teacher token 1.8%、图上 greedy 0/5；step 200 已到 NLL 0.0845、token 100%、
greedy 5/5；step 600 为 NLL 0.0158、仍 5/5。循环换入另一张图后，五条回答全部
跟随新图改变且正确停止，排除了行序/固定语言模板。16×16 重载后三个 B3 静态门
仍全过。结论：架构和视觉→语言训练接口具备有限样本表达能力；当前瓶颈是泛化训练
协议，不是“连五张图都记不住”。旧真实训练没有 EOS 监督，现已补上，历史 NLL
结果仍按 legacy no-EOS 标记理解。

分级过拟合门随后扩到嵌套的 16、32 张真实图片。目标为从原答案确定性抽取的唯一
短描述（最多 8 词）并监督 EOS；它是容量诊断，不冒充开放域 caption。16 张达到
token 100%、graph-greedy 16/16、换图 16/16。32 张从该检查点启动时准确保留
16/32；均衡小批、低学习率 finishing 和成对 hard replay 后达到 NLL 0.0339、
token 99.7%、graph-greedy 32/32、换图 32/32、停止 32/32。Pythia 全程冻结，
每 token 回跑完整图。32 张候选在 16×16 独立重载后 T2I/current/edit 三门全过。
因此“模型连少量真实图片都无法绑定到文本”的假设已被否定；下一问题是 64/128
容量曲线与 held-out 泛化，而不是继续怀疑基础接口。

同一候选又加入固定 noisy 1px OCR：64×64、每个数字一张、box=14、真实
Bresenham 1px 线条。以真实 I2T:OCR=3:1 rehearsal 和成对 hard replay 后，
最终 checkpoint 达到真实 I2T 32/32、OCR 10/10、换图 42/42、EOS 42/42，
teacher token 99.4%。独立静态审计同时保持 1px 数字 T2I 生成 digit 1.000、
color 0.969、paired IoU 0.941、digit/color shuffle 0/0、flood 0.00165；
current 重建/分割与 next-color edit 也全过。故在“允许固定集过拟合”的标准下，
原 1px OCR 与生成已经能在一个检查点共存。它不等价于多字体OCR或自然图生成泛化。

自然照片 T2I 的最小容量门也已单列完成。`train_sharegpt4o_t2i_overfit.py`
使用两张 Freedom ShareGPT-4o 自然场景，输入始终是全零视觉场且
`image_precision=0`；Pythia 冻结并只提供 prompt embedding，冻结因果 decoder
不调用，RGB 仍由同一 Slice–MoT–Deslice 图单次写出。480 步达到 prompt
retrieval 2/2、PSNR 20.16 dB；轮换提示词后 MSE 从 0.00964 升至 0.19477，
gap 0.18513、输出 RMS 0.42735，因此不是无条件平均图。结果与图库见
`results/published/sharegpt4o_natural_t2i_overfit.json` 和
`present/figs/sharegpt4o_natural_t2i_overfit.png`。

该结果只回答“自然 RGB 能否过拟合”，**不是统一冠军**：无回放候选破坏旧
T2I/current/edit 三门；同步终端蒸馏虽把四图候选的旧 T2I 恢复到
digit/color/IoU `0.880/0.926/0.860`，自然图仅 14.23 dB，current/edit 仍未全过。
因此下一优化问题是同一检查点的多目标保持，而不是再外挂更强生图器或宣称自然图
生成已闭环。`omni_d64_pythia_i2t_ocr1px_joint_final.pt` 仍是已通过共存门的基线。

反例：旧版 250 步 broad-`auto` 虽降低真实图像 NLL，却破坏官方编辑门；旧版
混合语言试验中的 IT2T 又是误标 caption。当前 `auto` 仅把图像端映射到
`rgb_likelihood`、文本端映射到 `token_reader`，禁止默认进入
`language_rgb` 或把错误标签当能力证据。

保护 B3 哈希仍为
`64E79D3B7D8757A4C50AA12E880993A92312F4237986D840466B689FCA4F086B`。
安全候选 `omni_d64_pythia_sharegpt4o_r64_safe_candidate.pt` 哈希为
`CB4DA7423DEF230DB3F0379AB51AE1D831012AF3C36DE940C109956D0E1A2B79`，
I2T 候选 `omni_d64_pythia_sharegpt4o_i2t_candidate.pt` 哈希为
`D33679F911A31A9B35C47E142730A3461D1A71C27EF755B65E3F40FDD73C65AE`。
两者都仍是 candidate-only；保护 B3 未覆盖，I2T 候选复验静态三门全过。

下一步分两条门控支线：文本侧加入真正的图像条件问答/指令数据补 IT2T；视觉侧
从已过自然 T2I 容量门出发，使用明确的 B3 capability replay/梯度冲突处理，使
自然门与 T2I/current/edit 在同一检查点同时通过。共享语言或视觉写入一旦解冻，
必须按每个门保存最佳候选；禁止默认 joint、无回放 `language_rgb`，也禁止把单项
自然图过拟合候选升级为冠军。最终仍要求 held-out、多 seed 和全能力矩阵。

---

## 9. 环境

- Windows，PowerShell，Python 3.10+，包在仓库根 `pip install -e .`
- CUDA：GTX 1650 4GB 上 70m + d64/16² 可训
- 测试：`pytest tests/test_pythia_omni.py tests/test_pythia_generation.py -q`
- 缓存：`ML_CACHE_ROOT=D:\ml_cache`
