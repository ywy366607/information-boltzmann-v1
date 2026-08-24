# Pythia 接入 Slice–MoT 主动推理架构：实施计划

**进度（2026-08-24）：** 阶段 A/B 完成；B3 静态冠军保持。阶段 C/D 已实现统一 collator、token NLL、答案隔离和每-token-回跑图的 greedy，但只部分通过：T2T token/greedy 0.983/0.950，next-color IT2T 1.000/1.000，I2T 0.444/0.400（图像 shuffle 差 1.289 nat）。因此只有 `omni_d64_pythia_token_candidate.pt`，没有 token 冠军。操作细节与否证路径见 [`STATUS.md`](STATUS.md)。

## 1. 目标与边界

Pythia 只提供三样东西：预训练 tokenizer、token embedding 和冻结的因果语言分布。它不是第二个 VLM，也不能直接生成图像。项目主状态仍是全分辨率视觉场 `X` 与语言场 `H`；每层临时读取 Slice `S`，执行 `SliceRead → MoT(S,H) → Deslice → X`。RGB、分割和文本必须从同一次联合演化的终态读出。

首选 `EleutherAI/pythia-70m` 做低成本联调，结构通过后用 `pythia-160m` 复验。一次实验只绑定一个明确模型 ID/版本；请求 Pythia 时加载失败应直接报错，不能静默回退 GPT-2 或随机 TinyLM。

## 2. 锁定的数据流

```text
input_ids ── Pythia embedding ──┐
                               v
image ── full-resolution X ── [Slice–MoT: S ↔ H] ── X* ── RGB / segmentation
                               │
                               └── H* + terminal Slice interface
                                      └── frozen Pythia decoder ── next-token logits
```

具体接口新增在 `DualStreamOmni.forward_tokens(...)`，不另建视觉主干：

1. 用 Pythia 的输入 embedding 得到 `[B,T,d_llm]`，通过现有 `text_in` 映射到 MoT 宽度。
2. `NativeMoTStack.forward_native` 同时演化 `X` 与 `H`，返回 `X*`、`H*_llm` 和终端 Slice 接口 token。
3. 将 `[terminal_slice_tokens, H*_llm]` 作为 `inputs_embeds` 送入**冻结** Pythia，计算 shifted causal token NLL。
4. 同一个 `X*` 同时进入现有高斯 RGB 头、分割头和 belief 头。
5. Pythia 参数不进优化器；checkpoint 保存模型 ID、revision、tokenizer 配置以及非 Pythia 参数。

终端 Slice token 只是文本似然的临时接口，不取代持久 `X`。禁止调用 `model.lm.generate()`，因为它会绕过 Slice–MoT；自回归解码必须每生成一个 token 就把增长后的前缀重新送入统一图。KV cache 只能在证明不改变 `X/H` 联合演化后再优化。

## 3. 文本空白、因果掩码与精度

必须分清三种“空白”：

- 输入文本缺失：保留合法 BOS/null 位置，令 `text_precision=0`，不能用空字符串冒充缺失。
- 下一个待生成 token：它在因果序列中尚不可见，由 Pythia next-token likelihood 预测。
- 文本不作为本样本目标：令 `target_text_precision=0`，整个 token NLL 严格消失。

训练可使用 teacher forcing，但必须建立两个不同 mask：

- `text_mask`：所有有效 token；文本 query 只能读到自身及更早 token。
- `prompt_mask = (labels == -100) & text_mask`：视觉/Slice query 只能读取已观测提示，绝不能读取答案 token。

先前答案 token 可以作为后续答案 token 的因果前缀，但不能反向写入视觉场。改变尚未可见的答案后缀时，`X*` 与此前位置 logits 必须完全不变。

## 4. 主动推理目标

Pythia token NLL 是统一生成模型中的文本观测项，不是新的独立任务损失：

\[
\mathcal F =
\lambda_h\,\pi_h^{tgt}\,\mathrm{CE}_{causal}(y)
+\lambda_x\,\pi_x^{tgt}\,[-\log p(x\mid X^*)]
+\lambda_s\,\pi_s^{tgt}\,\mathrm{CE}(seg)
+\beta_{F2}\mathcal F_{F2}
+\mathbf 1_{\tau>0}\beta_{dyn}D_{KL}(q_{t+1}\Vert p_{t+1}).
\]

Pythia 给出语言先验和词法/句法坐标；观测后的 `q(X,H)` 仍由 Slice–MoT 摊销推理产生。T2I 与编辑继续通过语言条件的 F2 prior action 写回 `X`，而不是让 Pythia 画图。GDN-2 仍只处理 `tau>0` 的物理时间先验，不能为了文本生成而启用动作或未来分支。

## 5. 文件级实施顺序

### 现有代码应直接复用

- `fine_grain/llm_backend.py` 已能从 `D:\ml_cache` 加载并冻结 Pythia。
- `DualStreamOmni(language="pythia")` 已能把 Pythia hidden state 接入 MoT，但终端仍是小类别 `head`；这不是自回归文本能力。
- `scripts/run_slicemot_mini_gate.py::SliceMoTMiniVLM` 已有 `inputs_embeds + causal labels + prompt_mask` 的 token loss 原型，但没有统一 RGB/分割/belief 读出。应把这段语义迁入 `DualStreamOmni`，不要保留第二套模型。
- `scripts/train_apple_pythia.py` 只证明冻结 Pythia 表示可以条件化 T2I；它没有使用 Pythia causal LM head，不能作为文本闭环证据。

实现时应以 `DualStreamOmni` 为唯一产品入口：从 mini gate 搬 token-likelihood 逻辑，从 Omni 保留同一 `X*` 的全部视觉读出。

### 阶段 A：最小桥接

- `fine_grain/llm_backend.py`：支持精确 Pythia ID；禁止静默换模型；记录版本与隐藏维度。
- `fine_grain/omni_model.py`：增加 `forward_tokens`，复用现有 dense readout；保留旧类别头仅作 checkpoint 兼容，不再作为真实语言主线。
- `fine_grain/native_mot.py`：复核并测试 causal `text_mask` 与 `prompt_mask`，不要改变 Slice/Deslice 主路。
- `tests/test_pythia_omni.py`：先用 TinyCausalLM 测形状、梯度、泄露和冻结，不依赖下载。

### 阶段 B：先保住生成

从统一能力冠军加载所有形状兼容的视觉、Slice–MoT、RGB 和分割参数；排除旧 toy embedding、类别头及不兼容的 `text_in/text_out`。先以闭式 ridge 把 Pythia token embedding 对齐到能力冠军的 post-`text_in` 语言坐标，再训练现有语言侧 MoT/F2 接口。T2I、重建、分割和编辑必须共享 `capability_sample/grid_digit_mask` 场景生成器；禁止拿另一套 `one_sample` 笔画门约束能力冠军。重建边界是图像存在、文本缺失、`tau=0`，不能把 `Reconstruct current frame` 当 `pi_text=1` 的内容证据。T2I 数字以同色同位置十数字的配对观测能量识别；随后仅在 T2I/当前帧门已过时训练官方 next-color。实现入口为 `scripts/train_pythia_capabilities.py`。

### 阶段 C：加入真实文本似然

构建统一 collator，输出 `input_ids`、`attention_mask`、`labels`、输入模态精度和终端似然精度。先混合 T2T + I2T/OCR，确认语言稳定且视觉确实改善 token NLL；再加入 IT2T、T2I、IT2I、重建和分割。端口只决定边界与监督，不能选择不同 backbone。

当前实现位于 `fine_grain/token_tasks.py` 与 `scripts/train_pythia_tokens.py`。T2T/IT2T 已过；I2T 已通过 matched/shuffled 因果门但未过准确门，故阶段 C 状态为 partial。固定 Eulerian terminal atlas 与 prototype/ridge 对齐均只保留为默认关闭的实验开关，不能作为已验证主线。

### 阶段 D：自回归推理与规模化

实现统一图 greedy decode，先验证短答案/OCR，再做自然描述。基础非循环能力矩阵全部通过后，才评估不确定性循环；不得用循环掩盖错误的 token 因果性或失败的生成。

graph greedy 已实现且不调用 `lm.generate()`；T2T/IT2T 通过，I2T 仍未通过，因此阶段 D 也是 partial。

## 6. 必须通过的验收

### 结构测试

- Pythia 全部 `requires_grad=False`，但 token NLL 对 `text_in/text_out`、MoT 和 I2T 的视觉 stem 有非零梯度。
- RGB、分割、token loss 来自同一 `X*/H*` 调用；每次 forward 只编码一次图像、终端解码一次。
- 答案后缀扰动不改变视觉终态或更早 token logits。
- `text_precision=0` 删除词汇证据；`target_text_precision=0` 删除文本准确率项。

### 反旁路测试

- I2T/IT2T 上 `NLL(matched image) < NLL(shuffled image)`；正式 OCR 门沿用已登记的至少 `0.5 nat/target-token` 中位差。
- 关闭图像精度显著恶化 I2T，关闭文本精度显著恶化 T2I/编辑。
- 文本-only 验证损失不能发散；T2T 不应被强迫依赖随机图像。
- 直接 Pythia-only、旧类别头和独立图像生成器都只能作为消融，不能进入最终 checkpoint 的能力声明。

### 北极星能力门

同一个 checkpoint 同时报告 T2T、I2T、IT2T、T2I、IT2I、重建、分割的 token NLL/准确率、RGB 指标、数字/颜色/位置及分割 IoU，并附 matched/shuffled、模态精度、F2 prior action 和 Deslice 消融。任何文本提升若破坏当前生成冠军，均不算接入成功。

## 7. 明确禁止的捷径

- 不把 Pythia hidden state 当静态 prompt 后继续只做小类别分类。
- 不让 Pythia 或外挂 diffusion 直接承担图像生成。
- 不把答案 token 暴露给视觉 query，不用 `need_text/need_pix/need_seg` 改变主干。
- 不把生成包装成 action，也不为 `tau=0` 启用 GDN-2。
- 不先做 LoRA、循环或大数据训练；冻结 Pythia 的小闭环尚未通过时，这些只会增加归因混乱。

实施完成的定义不是“Pythia 能输出句子”，而是：预训练语言分布已经成为同一 Slice 全分辨率信念图中的文本观测模型，且视觉证据、语言条件和各终端似然都通过受控消融证明参与了同一次联合推理。
