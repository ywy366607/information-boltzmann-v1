# 现状简报（给外部模型指导本仓库代理）

日期：2026-09-05。最新真实任务与混合分辨率进展见 §U7–U8；旧分节日期与成绩为历史记录。
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

### 首个统一 token 冠军（2026-08-29）

`omni_d64_pythia_token_best.pt`（SHA256
`2768A79451B774E1CD1AA2163AAC094099A52D686885120498CF643F49802830`）独立重载全门通过：
T2T token/greedy **1.000/1.000**（gap 9.50）；I2T token/greedy **0.822/0.811**（90 格全 bank、
matched-vs-shuffled 中位 gap **7.47 nat**）；官方 next-color IT2T **1.000/1.000**（gap 11.0）；
静态 T2I/current/edit 三门同过。Pythia 冻结、每 token 重跑同一图、无 `lm.generate()`、
可训参数仍是 `token_reader` 的 383,224 个。注意：token 评测 bank 是训练 bank 的注册子集
（固定 90 格审计，与 T2I/B3 同一 fixed-bank 标准），这是有限 bank 能力闭合，
不是 held-out 泛化。

**空间 held-out 审计（2026-08-29，
`results/published/pythia_token_heldout_audit.json`）**：把同样的 90 格场景按
整像素偏移重渲染后，冠军 I2T 为 control 0.822；水平 ±1px 0.722–0.767；垂直
±1px **0.411–0.467**；更大偏移 0.422–0.589。所有 jitter 下 matched-vs-shuffled
中位 gap 仍达 8.7–12.0 nat——模型仍看得见图，但 argmax 精度随位置退化，
垂直方向尤其敏感。结论：0.822 含有实质的位置记忆成分；后续语言支线应加
jitter 位置增广重训（协议类修复，与 E 系列同类）。

---

## 2. Pythia 接入进度

计划阶段：

| 阶段 | 内容 | 状态 |
|---|---|---|
| A | 精确加载、`forward_tokens`、因果 mask、结构测试 | **结构完成；训练接口已按审查修正** |
| B | 能力冠军视觉图 + 冻结 Pythia，闭合 T2I/重建/分割/编辑 | **完成；B3 三门通过** |
| C | 统一 collator，token NLL，T2T+I2T 再混其它端口 | **完成；三门 token gate 全过** |
| D | 统一图 greedy decode（每 token 重跑 MoT） | **完成；全 bank decode ≥0.811** |

### token 结果（冠军 `omni_d64_pythia_token_best.pt`，2026-08-29）

入口为 `scripts/train_pythia_tokens.py`；统一 token 边界在
`fine_grain/token_tasks.py`。训练答案 token 只作因果 teacher forcing，
`visual_prompt_mask` 禁止答案或已生成 token 回写视觉场。解码每个 token 都重新
调用 `forward_tokens`，没有调用 `lm.generate()`。

冠军独立重载（`results/published/pythia_token_champion_audit.json`）：T2T
token/greedy **1.000/1.000**；I2T **0.822/0.811**（gap 7.47）；官方 next-color
IT2T **1.000/1.000**。静态 T2I/current/edit 三门全过，B3 保护检查点 SHA256 仍为
`64E79D3B7D8757A4C50AA12E880993A92312F4237986D840466B689FCA4F086B`。

获胜配方（E1→E2→E2b，`results/published/pythia_token_e1_i2t_only.json` /
`pythia_token_e2_joint.json` / `pythia_token_e2b_finish.json`）：

1. **E1 纯 I2T 长跑**（B3 起点、2000 步、lr 3e-4、hard replay 4）：I2T 到
   0.667/0.689，静态门保持。历史 ~0.60 天花板主要是步数不足。
2. **E2 联合 + 回放**（E1 最佳起点、`--case-cycle i2t_heavy`、hard replay 4、
   resume-optimizer、lr 1e-4、900 步）：I2T 不再被 rehearsal 啃掉，反而升到
   0.778；T2T/IT2T 分别在 400/100 步内回到 1.000。旧「联合即干扰」已被协议修复打破。
3. **E2b 低学习率收尾**（E2 最佳起点、lr 3e-5、`--class-contrast-coef 10.0`、
   600 步）：step 100 出现三门全过（I2T 0.822），晋升为冠军。

已否/收缩路径（本轮新增）：`--initial-proj-trust 0.2`（视觉 token 提到
embedding 量级）反而更差（0.556），门幅度假说证伪——优化器维持 ~0.01 的小门
是局部最优而非步数问题；`token_reader_deep`（放开倒数第二层 H 专家）**破坏
X 不变性**（H_out 会进入下一层 Kt），已被确定性测试拒绝，未保留该相位。

代码入口：

- `fine_grain/llm_backend.py`：`load_frozen_pythia` / `canonical_pythia_id`
- `fine_grain/omni_model.py`：`DualStreamOmni.forward_tokens`、`non_lm_state_dict`、`language_meta`
- `fine_grain/pythia_bridge.py`：加载生成冠军、优化相位 `interface` / `token_interface` / `token_reader` / `language` / `language_rgb` / `edit_spatial` / `joint`
- `scripts/train_pythia_generation.py`：T2I/编辑训练
- `scripts/train_pythia_capabilities.py`：统一场景 B2/B3（T2I、重建、分割、官方 next-color）
- `scripts/train_pythia_tokens.py`：阶段 C/D token NLL、反事实门、hard replay、`--case-cycle`、resume-optimizer 与逐 token 图上 decode
- `fine_grain/token_tasks.py`：统一 token collator、答案因果边界与 graph greedy
- `scripts/diagnose_pythia_language.py`：toy vs Pythia 语言敏感度
- 测试：`tests/test_pythia_omni.py`、`tests/test_pythia_generation.py`、`tests/test_pythia_capabilities.py`、`tests/test_pythia_tokens.py`

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
| `checkpoints/omni_d64_pythia_token_best.pt` | **统一 token 冠军：T2T/I2T/IT2T + 图 decode + 静态三门** | **否** |
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

固定 bank 能力闭合已完成：统一 token 冠军在同一检查点上 T2T/I2T/IT2T 三门
token + 图上 greedy decode 全过（见 §2）。仍不能宣称的：

- **held-out / 开域对话**。token 评测 bank 是训练 bank 的注册子集；真实数据
  512 条 held-out I2T 的逐样本因果性仍弱（`docs/REAL_DATA_PILOT.md`）。
- **多轮对话 / instruct 行为**。`pythia-70m` 是 The Pile 基座，非 instruct。

禁止 `model.lm.generate()`；现有 decode 每 token 回跑统一图。

---

## 7. 建议下一步（按优先级）

I2T token 门已闭合（2026-08-29），指导代理时选一条，不要并行拆图：

1. **静态视觉唯一起点改为 `omni_d64_pythia_token_best.pt`**（它同时保有 B3
   三门）；B3 检查点仍受保护，只作历史证据。
2. **真实 IT2T/VQA 数据支线**：引入真正的图像条件问答/指令数据（ShareGPT-4o
   两个源都没有真 IT2T，不得再从 caption 伪造），带确定性 holdout，要求
   matched NLL 改善且逐样本 shuffle gap 显著为正。
3. **自然 T2I 支线（当前主线）**：N101 判别门已把问题定界——写路径能表达
   低频布局（PSNR 22.1、物体放置正确）但高频带宽渐近 0.46–0.48，低于 0.50
   内容门。下一步是**写端结构性改造**：写分配锐化 / 内容调制写入
   （读端 `readout.temp` 的写端对应物），并在改后复跑同一 A 阶梯对照渐近线；
   不要再用纯步数/损失权重冲击该门。任何自然候选不得覆盖冠军。
4. **`tau>0` 时序线可恢复**：token 闭合满足「I2T token 与 decode 闭合后再恢复」
   的前置条件；从 v29 内容坐标先验继续，仍要求静态门与因果符号双过。

冠军只读复验：

```text
ML_CACHE_ROOT=D:/ml_cache python scripts/train_pythia_tokens.py --init checkpoints/omni_d64_pythia_token_best.pt --eval-only --decode-limit -1 --out results/published/_eval_token_best.json --device cuda
```

注意 eval-only 记录里 `admitted=false` 是「selected==init」的语义；门数字
（gates=111、decode、static）才是复验对象。

三门只读复验（B3 历史冠军）：

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

相位：`language` = text 专家 + prior；`rgb_likelihood` = 只训共享 RGB 均值/方差头；`edit_spatial` = 再加精度坐标、冻结 pix_head；`edit_read` = 再加 SliceRead / 视觉 query-output / Deslice；`generation_write` = 只开语言→视觉 K/V、F2 prior、Slice/Deslice query-output、精度坐标和共享 RGB 头，stem 与终端语言 reader 冻结；`generation_capacity` = 诊断性地再开已有视觉 stem/KV/FFN/local，不是安全持续训练相位；`language_rgb` = 语言演化与 pix_head 同时解冻，真实轮训已证会破坏编辑；不要默认 `joint`。

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

自然照片 T2I 已改用 Freedom ShareGPT-4o 的**原始目标 PNG**直接展示；此前图库
把目标先缩成 16×16 再放大，造成“目标也像色块”的误导，现已修正。输入仍始终是
全零视觉场且 `image_precision=0`，Pythia 冻结、因果 decoder 不调用，RGB 由同一
Slice–MoT–Deslice 图单次写出。

修正后的内容审计推翻了旧“自然容量通过”结论。16×16 候选虽有 retrieval 2/2、
PSNR 20.16 dB 和提示词 gap +0.185，但边缘相对 MSE/相关为 `0.803/0.426`，未过
`≤0.75/≥0.50` 内容门。64×64、16 Slice 训练 1000 步后 PSNR 19.04 dB，边缘
相关仅 0.133，输出仍是平滑配色场。进一步开放已有视觉 stem/KV/FFN/local，扩到
64 Slice，关闭 Gaussian 方差捷径并强化边缘损失后，面罩低频轮廓开始出现，但
600 步 PSNR/边缘相关仍只有 `15.64/0.190`，岛屿与船等结构没有重建。因此当前
结论是：文本能选择不同低频视觉场，**自然图内容过拟合尚未完成**。

清晰目标/输出见 `present/figs/sharegpt4o_natural_t2i_overfit.png`；64×64 诊断见
`present/figs/sharegpt4o_natural_t2i_r64_m64_capacity.png` 和对应 JSON。旧静态门
也未保留，故任何自然候选都不得升级为冠军；已通过的 1px OCR+数字生成基线不变。

**协议阶梯判别门（2026-08-30，
`results/published/natural_t2i_a1_long_capacity.json` / `_a1b_continue.json` /
`_a1c_finish.json`）**：把 E 系列结论反向用于 N099——旧 600 步容量诊断
（edge corr 0.194）可能只是步数不足。同一两目标、64×64、64 Slice、
`generation_capacity`、`nll_coef=0`、`edge_coef=8`，按 A1（6000 步，
lr 3e-4/2e-4）→ A1b（6000 步，半 lr）→ A1c（3000 步，1/20 lr 收尾）共
15000 步：edge corr **0.194 → 0.351 → 0.422 → 0.448**，PSNR 15.6 → 22.1。
画廊显示粗布局、物体放置、逆光渐变乃至面罩眼缝都开始出现
（`present/figs/natural_t2i_a1c_finish.png`），但边缘相关在最后 3000 步只增
+0.026，渐近线估计 0.46–0.48，**未过 0.50 注册内容门**。

判定：N099 的"结构性失败"表述需要修订——写路径能写出正确的低频布局与
物体放置，协议修复贡献 2.3 倍边缘保真；但当前 Slice/Deslice 写入的**高频
带宽存在 expressivity 渐近**，协议手段不足以闭合内容门。下一个主线杠杆是
写端结构性改造（写分配锐化/内容调制写入，读端 `readout.temp` 的写端对应
物），不是继续加步数。capacity 相位按预期摧毁静态端口（digit 0.085），
诊断件不得晋级。

**写分配温度开门（2026-08-30，
`results/published/natural_t2i_b1_sharpening.json` / `_g4_fixed.json` /
`_g8_fixed.json`，research_tree N102）**：给 `DesliceWrite` 加 opt-in 的
写分配幂锐化 `w_write ∝ w^γ`（`γ=exp(raw)` 恒等初始化，`--write-sharpening`
学习 / `--write-gamma` 处方固定）。要点：

1. **学习型 γ 无效**（B1）：6000 步只走到 γ=1.06–1.26，edge 0.348 ≈ 基线
   0.351。标量参数梯度太弱，优化器不会自行探索幅度——与 `proj_gate` 同一
   教训，机制检验必须用处方剂量。
2. **剂量响应单调并过门**：edge corr @6000 步 = 0.351（γ=1）→ 0.393（γ=4）
   → **0.514（γ=8，过 0.50 注册门）**；edge MSE 0.863→0.728；PSNR 20.7→23.1。
   γ=8 画廊出现可辨认的面罩形状与眼缝
   （`present/figs/natural_t2i_g8_fixed.png`）。
3. **候选-only**：capacity 相位摧毁静态端口，自然内容门通过 ≠ 冠军；
   下一阶段是静态能力合并（同一检查点恢复 T2I/current/edit 且内容门保持），
   以及 γ>8 的边界与多 seed 复验。
4. **G8b 续训（半 lr，6000 步）**：13/13 评测点全过门，渐近线 edge corr
   **0.628**（γ=1 时 0.448）、edge MSE 0.598、PSNR 24.78
   （`results/published/natural_t2i_g8b_continue.json`）。γ=8 是有余量的稳态。
5. **分辨率假设已检验并否决**：γ 剂量响应在固定 N/M/网格上移动了门，而
   16×16→64×64 的升分辨率让门更难——约束在写端局域化，不在点数；剩余
   自由度预算是 N/M 配比与 γ 的联合扫描（M=128/256 × γ、固定 N/M 的
   32/128 网格门难度曲线）。

**笔画顺序写入判别门（2026-08-31，
`results/published/stroke_sequential_*_gate.json`，research_tree N103）**：
检验「画家假说」——持久场当画布，K=8 步逐笔写入（子笔画分解 + 尾部 no-op），
逐步累积监督。新增 `x_init` 持久场输入（`encode_X`/`forward_native`，缺席即
恒等）与 opt-in 步条件化先验投影（零初始化恒等，`--prior-step-condition`）。
2×2 矩阵（γ∈{1,8} × 步条件∈{off,on}），四格全部未过门，但失败方式高度有序：

1. **画布持久因果性完美**：末步给空白画布 → 输出为空（IoU 0.000–0.001）。
   每一步确实在读已积累的场。
2. **步索引从未被因果使用**：常数 t 消融与完整课表逐位相同（四格皆是），
   即使 t_coord 已训、步条件先验已训（投影 norm 仅 0.02–0.04，标量梯度弱）。
3. **写锐化改善逐步追踪（峰值 0.518 vs 0.408）但不改善合成**；逐步 IoU 在
   no-op 尾段先平台后回落——收敛到吸引子后又在轻微覆盖。
4. **机理结论**：设置点写 `X += proj(mu_p − S(X))` 要跨步合成，前提是
   `S(X)` 对画布内容是保真读回；与 N101 同源的弥散 assignment 使残差
   `mu_p − S` 表达不了「目标 − 画布」，故反复写收敛于同一吸引子而非累积。
   **顺序精化不是免费的，它被潜场读回保真度门槛挡住**——这是继读出
   协议修复（E 系列）、写带宽锐化（N102）之后定位的第三个独立瓶颈。

**读回保真诊断与修复尝试（2026-09-01/02，research_tree N104/N105）**：

1. **N104 零训练往返诊断**（`results/published/latent_roundtrip_diagnostic.json`）：
   `X_rt = blank + W(read(X))` 的解码距离在感知流形仅 **0.7–1.1 dB**、生成流形
   三层 <5 dB（仅 L2 达 10–16 dB），方差恢复 3.6–7.6%；锚点（场直接解码）
   12.2 dB——内容在场里，损失发生在一次读-写循环内部。B3 重建门认证的是
   全图像素解码路径，从未测过这个量。维度上界足够（1024 slice 数 > 768 像素
   数），损失是学出来的。
2. **幂等训练**（`results/published/readwrite_idempotence_gate.json`）：只训层内
   read+deslice.proj、目标 `W(read(X)) ≈ X − X_blank`——**RT 100 步内从 0.74
   跳到 13.3 dB**（Transolver 假说证实：读不是不能学，是从没被要求过），但
   静态三门全塌（读写算子是感知/生成共享的）。50% rehearsal 只守住部分门、
   RT 停在 2.3 dB（`readwrite_idempotence_rehearsal_gate.json`）。
3. **设置点代数修正**：增益 1 下设置点写即 `X ← blank + W·proj(mu_p)`，画布被
   精确抵消——往返修得再好也只是更干净的抵消，**带宽中性**。画家假说的累积
   必须用加性增量写。已加 `deslice_write='prior_increment'`（写 `pw·mu_p`，
   不减画布读回），固定先验单元测试钉死两种语义。
4. **画家语义顺序门**（`stroke_sequential_painter_gate.json`）：仍失败
   （digit 0.100 / IoU 0.033），但失效模式改变——每步是弥散小色雾而非重复
   盖章，且步索引首次出现弱使用（abl_t 中途 0.144 vs 0.100）。

**N105 判定**：顺序合成被三个已各自量化的独立屏障共同限制——画布盲先验、
失真读回、每步写入质量。下一步按序：梯度隔离的幂等修复（专用 read/deslice
副本）或步条件先验的处方剂量（如 γ 教训）；写入质量不足可能只是步数预算
（E1 同款悬念）。顺序线暂停至屏障逐一移除。

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
先解决自然 T2I 的全分辨率空间地址基/写入秩，使边缘内容门通过，再使用明确的 B3
capability replay/梯度冲突处理，使自然门与 T2I/current/edit 在同一检查点同时通过。共享语言或视觉写入一旦解冻，
必须按每个门保存最佳候选；禁止默认 joint、无回放 `language_rgb`，也禁止把单项
自然图过拟合候选升级为冠军。最终仍要求 held-out、多 seed 和全能力矩阵。

### U1：64×64 统一候选合并协议（2026-09-04，进行中）

入口为 `scripts/train_unified_champion.py`。它**不是权重平均**：以已过自然内容
门的 `_natural_t2i_g8b_candidate.pt`（64×64、64 Slice、固定 γ=8）保留视觉
读写底座；只从 `omni_d64_pythia_token_best.pt` 复制不会回流改变 (X) 的末层
token 似然读出和共享分割头。随后在同一张图中轮换真实两图 T2I、静态 T2I、当前
重建/分割、官方 next-color 编辑以及 T2T/I2T/IT2T token NLL。stem、视觉 K/V/FFN/local
仍冻结；Pythia 仍冻结；不调用 `lm.generate()`；γ=8 不可训练。

候选只在**同一保存状态**同时通过自然内容门、三项静态视觉门、三项 token 门和
逐 token 图上 decode 后才可晋级。首次 CPU 单边界冒烟已分别跑通自然 T2I、静态
T2I、当前、编辑和 I2T token 的前向/反传；这只证明训练路径可执行，**不证明
能力已合并**。64×64 是当前具有真实两图内容门的注册尺度；架构的点场/Slice 算子
可接受变长 (HW)：同一 γ=8 候选以固定 64 Slice 在 CPU 上已完成 256×256 的
无梯度缺失图像前向（输出 `1×3×256×256`、全有限、零形状跳过）。但目前权重没有
经过跨尺度训练，因此这只验证计算图，不得宣称 256×256 生成质量已验证。
下一尺度协议应固定 (M)（或登记 M/点数曲线）、在归一化坐标上混合 64/128/256
网格，并以面积归一化 RGB/边缘似然和每尺度独立门报告结果。

**首轮 U1 结果（GPU 400 步，已拒绝）**：自然门全程保持，最终为 PSNR **24.76**、
2/2 retrieval、shuffle gap **0.207**、edge relative MSE/correlation **0.600/0.626**；
但静态 T2I/current/edit 三门均失败（digit 分别 **0.043/0.578/0.144**），token
T2T/I2T/IT2T 准确率为 **0.350/0.111/0.233**，图上 decode 为 **0.533/0.100/0.233**。
`_unified_u1_candidate.pt` 和画廊仅为拒绝的 candidate。结论不是自然 γ=8 写入回归，
而是 16×16/16-Slice 静态冠军的表征无法在 64×64/64-Slice 图表上由少量交替 replay
恢复；禁止权重平均或据此宣称统一冠军。下一实验应先在**同一 64×64、64-Slice**图表
上用 ground-truth 静态银行闭合视觉边界，再接 token likelihood；随后才训练可审计的
多步“读持久场—增量写入—重读”循环。

**U1 排错审计**（`results/published/u1_chart_diagnosis.json`）：这不是精度接线、
冻结或无梯度 bug。三种边界分别正确给出 T2I `(πx,πh)=(0,1)`、当前 `(1,0)`、
编辑 `(1,1)`；T2I 对 `text_in`、语言→视觉 K/V、F2 prior、Slice/Deslice、RGB/seg
似然均有非零梯度，当前与编辑也对可用读写路径有非零梯度。失败在训练前已存在：自然
64×64 底座的静态 T2I digit 为 **0.026**、current 为 **0.578**、edit 为 **0.156**；
400 步后为 **0.043/0.578/0.144**。每个端口只得到 57 次更新，无法从自然图图表中
重建已在 16×16/16-Slice 才训练成熟的数字图表。另有 5 个 `mot_stack.readout` 张量因
16→64 Slice 维度不匹配无法安全迁移；它们虽不写 (X)，但其 Slice 身份不能按行复制，
需在 64 Slice 上重新拟合。这是**跨图表初始化与预算不足**，不是循环缺失或已发现的
实现错误。

**U2 对照（静态图表闭合，已拒绝）**：U2-A 从同一 γ=8 自然底座只训练 64×64
静态 T2I 1000 步；U2-B 再额外打开语言侧 MoT 专家。两者的静态 T2I digit 都仅
**0.034**、paired IoU 约 **0.016**，但训练 loss 降至 0.3–0.5：它们在稀疏图的
背景/颜色统计上降损，没有获得数字词→局部笔画绑定。同时自然门分别退至
PSNR/edge-corr **10.76/0.058** 与 **9.73/0.022**。补做的 toy→Pythia lexical
ridge 对齐拟合误差仅 `5.2e-6`，却仍得 digit **0.034**，并把自然 edge-corr 降至
**0.320**。因此词向量初始化不是充分修复；也不能把静态 loss 下降当能力。

**根因排序**：第一，两个 T2I 数据都使用 `(πx,πh,τ)=(0,1,0)`，必须由语言内容
而非私有端口区分自然图和稀疏数字，而当前 64 Slice 图表没有学会数字语义绑定；第二，
稀疏数字的逐点似然可由背景下降，现有十数字对比从未在这个随机图表上获胜；第三，
自然/静态梯度在 `text_in`、F2 prior、Slice/Deslice 上已测得负或近零 cosine，RGB
似然的静态梯度量级约为自然的 100×，导致持续覆盖；第四，16→64 不仅增网格还改变
数字尺度和 Slice 身份。**下一步是 64×64 静态 T2I 的前景/配对观测能量课程与自然
rehearsal 的逐参数冲突处理，并每阶段双门保存；不是先加循环。**

**U3 前景课程 + 一阶自然约束（2026-09-05，已拒绝）**：这条后续直接检验了上段的
最后一个建议，而没有改模型图。每个 64×64 静态 T2I 十数字组同时使用同一 RGB
观测的 figure/ground 均衡 BCE（系数 4）和 10×10 配对观测能量（系数 4）；静态梯度
若逐参数与两张真实自然图 T2I 梯度相冲突，就投影掉冲突分量，再合并。800 GPU 步中，
自然门始终通过：最终 PSNR **24.63**、edge correlation **0.622**。这不是“自然门
保护失败”。但静态 T2I 仍只有 digit **0.077**、color **0.403**、paired IoU **0.008**、
flood **0.201**；当前帧保留 digit/IoU/seg **0.589/0.585/0.865**，官方编辑 digit/IoU
仅 **0.122/0.073**。静态 digit 在每 100 步评估为 0.051、0.051、0.051、0.043、
0.026、0.026、0.034、0.077，未形成上升趋势。

这条阴性结果很重要：foreground 权重、配对几何对比与“保护自然下降”的 PCGrad 式
投影都**不足以**让 64-Slice 自然图表写出数字；它不支持“只要梯度冲突处理就可合并”，
也不支持现在就把失败归咎于缺少循环。训练中投影后的静态梯度范数仍为自然的约
70–300 倍，说明它在同一参数坐标里主要是大幅、非共享的更新，而非可累积的共同下降。
`_u3_static_natural_candidate.pt`、JSON 与画廊都仅为拒绝候选。

**下一步（重新排序）**：先登记并完成一个**64×64/64-Slice 静态图表课程**，从大笔画
低频布局到当前 16px 字形，要求它在没有自然图 rehearsal 时通过 T2I 几何门；然后以
冻结的 64-Slice 静态写图表为起点，逐阶段引入两张自然图，并只在每阶段双门都保持时
保存候选。这个课程必须把“同一 `(πx,πh,τ)` 的两种 T2I 语义”用文本内容与共享 F2
prior 分开；语义任务 TToken 只能进入共享 H 工作区，不能选择私有 decoder、主干或
输出头。若静态 64-Slice 单图表本身仍
无法过拟合，才将其定为单次写入/坐标表表达问题，并用可审计的持久画布循环来检验；
在这之前不要引入循环作为猜测性修补。

**16×16 历史配方的直接尺度复验（2026-09-05，已拒绝）**：复查后确认，B2 当年不是
从自然图底座硬学字形，而是从 toy 静态能力冠军开始：Pythia `text_in` 做 ridge 对齐，
100 步 `language_rgb` T2I 启动，再冻结 RGB 头并以 T2I/每四步一次无文本当前帧交替；
语言侧 MoT+F2 prior、四色 bank、digit-shuffle 和 6/8 加权均已启用。为了检验遗漏的
是否仅是这条顺序，入口现已支持显式 `--resolution` / `--n-slices`，并从同一个
`northstar_slice_capability_best.pt` 把这条完整课程直接迁到 64×64/64 Slice（100+200 步，
Pythia 冻结、无自然 rehearsal、无循环）。

结果仍未过：T2I digit/paired IoU 从 **0.077/0.015** 到 **0.077/0.077**，全程无
语言→字形的增长；反之，当前帧 digit/IoU/seg 从 **0.511/0.571/0.872** 增至
**0.611/0.666/0.971**。16×16 的配方在大网格中保持了读取已有笔画，却没有把空场
F2 prior 的局部写地址一起迁上来。故真正缺口是**跨尺度的先验地址图表**，而不是
Pythia 接口、静态 RGB 损失、通用 Slice 表达或“尚未循环”。下一步应采用 16→32→64
的课程：归一化相同位置、先保持字形相对占比和 Slice/点比，再逐步放开 64 的实际字形
尺度；每一尺度都先通过 T2I 的 digit/IoU/shuffle 门，才进入下一尺度或自然图合并。
本次 `static64_b2_curriculum` 仍为拒绝候选。

**最小跨尺度课程筛选（2026-09-05，部分通过，未晋级）**：直接把 M 从 16 改到 64
会同时换掉 Slice 身份；把字形框放大却保持 1px 笔画又会改变相对几何。现已给
`grid_digit_mask`/`capability_sample` 增加默认关闭的 `glyph_box`、`glyph_stroke_px` 与
归一化地址参数，旧 16×16 bank 位级不变。用已过门的 16-Slice Pythia 冠军，在
32×32 仍固定 **M=16**，并保持 6/16→12/32 字形框、1/16→2/32 笔画宽度与九宫格
相对地址：无需训练时 T2I 为 digit/IoU **0.188/0.178**（读端也保持可用）。只训
语言 MoT/F2 的 1000 步空场 T2I 后，IoU 到 **0.608**、digit 到 **0.179**、color
约 0.89、flood 约 0.05；它能写出正确地址的字形，但十个数字仍未充分分离。

随后从该候选以较低学习率把既有十数字配对观测能量系数从 0.1 提至 10，100/200 步
digit 提至 **0.248/0.265**，但 IoU 降至 **0.489/0.463**，呈现清晰的语义–几何交换；
试验在此停止，没有产出可晋级权重。结论：**固定 Slice 身份 + 全几何比例不变量是
有效、简洁的尺度桥**，但下一次应使用有双门保存的两阶段/交替调度（先几何、后低
学习率语义、以 digit 与 IoU 同时改善为保存条件），而非单一大系数。只有 32×32 先
闭合后才继续 64×64；自然图合并和循环都仍后置。

### U4：32×32 从零训练、任务 TToken 与写入秩审计（2026-09-05）

为区分“跨尺度搬运失败”和“目标分辨率本身学不会”，在 **32×32、M=16** 上随机
初始化并完整训练 2000 步（前 800 步生成课程、余下生成/当前/编辑联合、余弦退火）。
同时修复一项数据边界错误：静态 current/edit 原先把重复源图以
`history_precision=1` 暗中送入 `history_stem`；现在只有 future 能观察历史。修复后
从零模型仍达到 current paired-IoU/seg **0.946/0.998**、edit **0.539/0.994**，证明
目标分辨率的读图、重建、分割和编辑路径可训练；T2I 只有 digit **0.089**、IoU
**0.257**，余弦退火后约 300 步即平台。该结果只说明已试配方未收敛，不能排除
学习率/训练覆盖率问题；旧 digit 指标另有渲染错配，见 U5。

十数字差异对比又发现量纲错误：正确项用全图平衡能量，错误项只在差异像素计能量，
导致相同平均字形也能轻易满足交叉熵。改用同一 RGB/seg 观测的组中心残差后，短程
最佳仅到 digit **0.122**、IoU **0.304**。逐层秩审计定位得更直接：十个提示在语言
与 F2 路径的有效秩约 **5–8**，视觉场在 L1 已达 **5.26**（目标 **5.23**），随后
L2/L3 降到 **2.64/1.88**。这提示谱能量集中，不能单凭参与率秩证明信息被擦除或
写入容量充足（U5 测得差异幅度同时增长）。把共享 RGB 与身份残差同时施加到各层也只使快速 T2I 峰值到
0.570，全量 digit 仍约 0.11，故“仅缺逐层监督”同样被否决。

按用户提出的模式显式化方案，现有图加入四个 masked **任务 TToken**：generation、
current、edit、future。token 与语言/动作共同进入 H，被所有 Slice 交互层读取；它不
选择头或私有模型。旧置零消融显示统一候选去掉 mode 后，T2I 分数 **0.516→0.294**、
current **0.673→0.568**；edit mode 消融反而 **0.653→0.688**，说明生成/重建已使用
模式。随后对固定的 90 样本/模式做了更严格的 **4×4 TToken 替换审计**：图像、
文字、精度、动作和时距全部不变，只替换任务 token。generation/current/edit 的正确
token 均取得本行最高分，正确项相对最佳错误 token 的间隔分别为
**+0.007/+0.003/+0.041**；future 为 **-0.020**。编辑使用正确 token 时 paired-IoU
为 **0.664**，换成 future/generation/current 后为 **0.620/0.252/0.210**。这修正了
旧的“edit token 置零后分数上升即表示冗余”解释：合法 token 互换显示 edit 已形成
模式依赖，但置零后改善的原因尚未定位，不能直接称为偶然。已识别的前三
类中，generation/current 间隔仍很小；future 又未参与本阶段训练，因此不能宣称四
模式闭合。
审计产物为 `results/published/northstar_task_token_audit.json`。低学习率宏平均收尾得到
当前 32px 统一候选
`northstar_static32_unified_task_best.pt`：T2I/current/edit 分数
**0.516/0.673/0.653**，current IoU/seg **0.937/0.999**，edit IoU/seg
**0.664/0.993**。它是最均衡的 TToken 候选，**不是冠军**：T2I 数字身份门未过，
future 未在本阶段训练，自然图与冻结 Pythia token 门也未合并。

后续方案已由 U5 的具体代码与评估缺陷修正：先修高斯逐头排列、控制可见性和评估，
再进行匹配重训。任务 token 互换作为诊断，不强制所有不相容/等价边界都出现固定
间隔；不通过人为损坏错误 token 输出制造模式分离，也不强制逐层有效秩单调。

### U5：Attention sink 核查与代码修复（2026-09-05）

详见 `docs/ATTENTION_SINK_REPAIR.md`。确认三项问题：

- TToken 追加在因果序列尾部，普通语言查询对它的直接注意力为 0。新增可选
  `control_prefix_attention` 后两模态都可读取观测控制，同时阻断答案经控制回流。
- F2 `prior_head/post_head` 按 `[head, mean/logvar, channel]` 输出，却在 flatten
  后全局二分；四头中的一半因此不能直接影响 mean。`gaussian_head_layout=per_head`
  修复拆分，逐头梯度测试通过。旧权重保持 `legacy` 解释，修复布局需显式重训。
- 32px 真值 digit 的旧 evaluator 得分也只有 **0.411**。新增精确渲染的配对身份
  指标，其 16/32/64px 完美目标上限均为 **1.000**。旧指标仍保留，不能静默改历史。

同权重 600 步匹配续训（原结构/前缀修复/前缀+sink）T2I IoU 都约 **0.274**，
故本轮 sink 接入不是生成充分解。配对身份重算表明原候选生成/重建/编辑为
**0.189/0.956/0.744**；sink 续训候选为 **0.156/0.989/0.933**，编辑 IoU **0.737**。
这仍是静态候选，future、自然图、Pythia 没有合并。完整产物在
`attention_write_audit.json`、`registered_digit_audit.json` 与
`attention_repair_{control,prefix,sink}.json`。

高斯布局匹配从零训练 1200 步后，按宏分选出的旧/新布局权重生成 IoU 为
**0.736/0.863**（身份均 0.900）。再给修复臂独立的 800+400 步低学习率联合
收尾，独立重载 `gaussian_layout_per_head_finish.pt` 后，生成/重建/编辑身份及
文本答案均 **1.000**，RGB IoU **0.979/0.993/0.996**，分割均 **1.000**。
轮换数字提示或对应源图后，目标数字身份均降至 **0.000**。验证范围只有
**32px、中心、红色源图、十数字 × 三任务共 30 样本**；不含完整四颜色规则、
九位置、自然图、Pythia 或 future。该权重是固定集容量证明，原冠军未覆盖。
产物：`gaussian_layout_probe.json`、`gaussian_layout_closure.json`、
`gaussian_layout_finish.json`、`gaussian_layout_finish_causal.json`。
代码验证：**376 passed**，Deslice 独立检查通过。

---

### U6：原生 256px 四颜色九地址扩展（2026-09-05）

用户选择直接做 256×256。入口 `scripts/train_static_resolution.py` 保留 U5 修复，
从随机权重训练 d64/M16/L4 同图；只参考旧配置，不搬运旧权重。训练集扩至
十数字 × 四源颜色 × 九位置 × 三模式共 1,080 样本，原生绘制 88px 字形框和
7px 笔画，不做输出放大，也不替代独立的 1px OCR 测试。

GTX 1650 4GB 单样本原损失反向已跑通（约 1,128MiB 峰值已分配显存）。
采用 microbatch=2、梯度累积 3 次、按需绘制，避免高分辨率全银行同时驻留。
首轮 240 步、80 步生成预热后联合训练已完成。独立重载末尾权重，全量
1,080 样本生成/重建/编辑 RGB IoU 为 **0.016/0.484/0.030**，身份为
**0.075/0.756/0.372**，分割为 **0.019/0.989/0.942**。生成背景涂色 0.999，
编辑颜色准确率 0.036；**尚未证明 256px 能力闭合，不能提升为冠军**。
当前和编辑各仅实际呈现 320 个场景，少于完整 360 个；短预算不排除训练不足。

选择器已修复为排除纯生成预热权重。`static256_scratch_pilot_best.pt` 是首次
错误选择的 80 步历史产物，不要使用；实际候选为 `static256_scratch_pilot_last.pt`，
完整复测在 `results/published/static256_joint_audit.json`。详见
`docs/NATIVE_256_CAPACITY.md`。原 32px 成功权重及保护检查点哈希未变。

同场景/样本预算的 32px 对照，完整生成身份也是 **0.075**、RGB IoU **0.019**，
说明短程失败并非只在 256px 出现。主线保持 256px；先建立充分的固定集训练
覆盖，再扩大组合，不因本轮失败增加循环或专家。该对照使用近似比例的栅格，
候选选择步也不同，不是纯分辨率因果分解。验证：**381 tests passed**。

### U7：真实 256px 单图联合训练（2026-09-05）

按用户要求，主线由数字扩展切回真实数据；数字仅作回归。保留可见 TToken
和修复高斯布局，不启用额外 sink/MoE 或循环。详见 `docs/REAL_256_JOINT.md`。

`real256_joint.pt` 从随机多模态权重与冻结 Pythia-70m 开始，在同一个 d64/M16/L4
图内训练 14 个真实/明确 identity 派生样本：T2I、重建、官方编辑、完整句子
I2T、DAVIS 官方分割、真实连续帧预测。GDN-2 仅 tau>0；视频没有动作标签，
动作精度为 0。新增 token 版本同图未来后验包装，目标不进入预测输入。

800 步重载：T2I/重建/编辑 PSNR **18.29/26.25/21.10 dB**，T2I 边缘相关
**0.024**；I2T NLL **2.314**，完整回答 **0/2**；分割 IoU **0.186**；未来
PSNR **21.10 dB**，仍比复制当前帧差。**真实任务已联合训练，不是统一冠军**。
实际曝光每张 T2I 126 次、其他 45–46 次。384 tests passed，峰值约 2.82GB。

展示 `present/real256_joint.html`；原始指标和错误回答 `results/published/real256_joint.json`。
权重 SHA256 `995a81e62c258d5d5f561fa0a1caea00c86b65cc9cc7966c0fbad1ff20c0acfe`。
旧保护权重未覆盖，纯文本问答/真正 IT2T/1px 保持未在本轮合并验收。

### U8：同权重原生 64px / 256px 混训（2026-09-05）

当前目标是两个尺寸共用一份完整能力冠军，不能降为分辨率兼容演示。
已修复外围固定 `res` 的编码检查、局部卷积、RGB/分割还原和时序坐标。
原生方形网格从输入推断，无新位置表/分辨率参数、不修改模型全局尺寸。
两个尺寸计算图共同反向、动态与固定尺寸同权重前向一致等检查通过；387 tests passed。

`scripts/train_mixed_native.py` 已从 U7 真实任务权重启动 3,200 步混训，
每步同样本 64px/256px 原生损失各半后共同更新；LR 3e-4 余弦降至 3e-5。
保留 TToken，不额外启用 sink/MoE。增加同历史、同任务 token 的 τ=0 当前帧
对照，防止 future 标签单独识别时间目标。旧检查点不覆盖。

具体检查与产物见 `MIXED_NATIVE_CHAMPION.md`；运行指标由
`results/published/mixed_native_64_256.json` 更新。当前仍是训练候选；本轮真实
固定银行通过后，还要把纯文本、真正图文问答、1px 保持合入同一权重才能收尾。
800 步中途：64/256 重建 PSNR 28.11/29.11dB、当前分割 IoU 0.694/0.737；
生成边缘相关仅 0.136/0.039，完整图像描述都为 1/2 且两图输出相同句子；
正时距预测仍劣于复制当前帧。训练仍在运行，不能把上述当最终验收。

U8 最终 3,200 步已独立重载：两个原生尺寸完整图像描述均 **2/2 exact+EOS**；
重建 PSNR 30.91/31.85dB，当前分割 IoU 0.799/0.812。生成边缘相关仍为
0.179/0.059，未来时距对照仍负，两个尺寸各有 19 项检查失败。候选 SHA256
`fa923cd0427d7bdf33068851cf88bede75c891094e42f764eb60cea164052b2a`。
不能把教师强制低损失等同于完整解码；本轮完整解码是最终真实测出的改善。

### U9：完整端口数据与解码补齐（2026-09-05）

`fine_grain/unified_capacity.py` 与混训 `--full-bank` 已准备每个原生尺寸 42 条
监督，保留 U8 全部真实任务，补入 2 条文本 QA、4 条人工核对的真实图片 QA、
10 条原生 noisy 1px OCR、10 条原生 1px 生成及掩码。图像问答明确为派生监督，
不是冒称 OpenGV 原数据有 QA。当前 U8 进程不更换数据，此入口用于后续阶段。

修复展示解码丢弃观测问题的限制；现在从 collator 原前缀开始，答案不送入
预测前向，生成 token 不作为视觉证据。添加缺项检查、QA 双模态对照、
1px 真笔画 IoU，避免只凭全黑图高 PSNR 通过。393 tests passed。
U8 完成后，`mixed_native_full_64_256` 已从其最终权重启动完整 42 条混训，
6,400 步、新优化器、LR 1e-4 余弦到 1e-5，每 800 步评估；原架构/似然不变。
800 步首次评估：两种尺寸真实描述/纯文本均 2/2，图文 QA 均 3/4；1px OCR
为 1/10、2/10，RGB 笔画 IoU 全为 0，尚未合入成功检查点。

约 1050 步后系统虚拟内存耗尽（Windows Event 2004）；并行 CPU 梯度诊断
保留主干反向图占 7GB，与训练占用叠加导致 CUDA OOM。已将诊断改为轻量
无梯度主干，正式训练以 `mixed_native_full_64_256_resume` 从 800 步恢复优化器、
原退火和样本位置。约 250 步未保存更新重跑；旧失败报告保留。旧检查点无 RNG，
不声称该 GPU 轨迹逐位复现；新保存格式及确定性恢复单测已覆盖 RNG。
详见混训文档末节。当前仍未获得统一冠军。

### U10：S1 空间监督对照已实现并排队（2026-09-05）

用户批准七方案独立选择后，已实现 Gaussian RGB 似然的前景/邻域/远背景
分层测度，默认均匀路径不变；不改图、分割平衡、KL 或 Pythia。
`run_spatial_measure_ab.py` 将等待 U9 恢复训练正常结束，再顺序执行同起点
800 步的 uniform/stratified 两臂，保留原 6400 步退火与相同优化器状态。
起点为保护的完整银行 800 步权重，目标全部 42 例与两个原生尺寸。
完整测试 398 项通过，新增一项集成测试及 Deslice 检查也通过；A/B 结果尚未
产生。具体权重定义、启动条件及失败判据见 `SEVEN_UNIFIED_SOLUTIONS.md`。

## 9. 环境

- Windows，PowerShell，Python 3.10+，包在仓库根 `pip install -e .`
- CUDA：GTX 1650 4GB 上 70m + d64/16² 可训
- 测试：`pytest tests/test_pythia_omni.py tests/test_pythia_generation.py -q`
- 缓存：`ML_CACHE_ROOT=D:\ml_cache`
