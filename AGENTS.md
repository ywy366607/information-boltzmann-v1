# Repository Guidelines

> **CGB工作区当前入口（2026-11-05）：** 先读 `docs/information_boltzmann/README.md` 和 `CGB_MILESTONE_CLOSED_LOOP_20261105.md`，再按下文原始定义、主链与方法开展工作。历史文档内的“当前/下一步”只在对应版本成立。已完成观察收益记忆512轮验证；最新动作归属读口修复只有接口和存档检查，尚未新训练。不得混记两者成绩。

## Project Structure & Module Organization

`fine_grain/` is the Python package; preserve public imports from `fine_grain/__init__.py`. Put entry points in `scripts/`, tests in `tests/`, architecture documents in `docs/`, and showcase assets in `present/`. Commit only curated metrics under `results/published/`; datasets, checkpoints, and other result files are ignored.

## Architecture Guardrails

Read `docs/NORTH_STAR.md`, `docs/ACTIVE_INFERENCE_GDN2.md`, and `research_tree.json` before changing the multimodal graph. The deliverable is one checkpoint for text, perception, generation, editing, reconstruction, segmentation, and next-frame prediction. Preserve the full-resolution field, transient Slice workspace, joint language evolution, and Slice/Deslice write path. Express ports through modality precision, prediction horizon, and likelihood heads—not private backbones or output-controlled dynamics. Keep semantic F2 generation distinct from the physical-time GDN-2 prior while sharing the same graph. Add uncertainty loops only after the non-recurrent capability matrix passes. Update the research tree when a claim or invalidation changes.

## Build, Test, and Development Commands

- `python -m venv .venv` then `.venv\Scripts\Activate.ps1`: create a Windows environment.
- `pip install -r requirements.txt` then `pip install -e .`: install dependencies and the package.
- `pytest -q`: run the full CPU-oriented test suite.
- `python scripts/train_northstar_capabilities.py --active-gdn2 --active-gdn2-initial-trust 0.1 --no-future-language-evidence --temporal-only --steps 50`: reproduce the current causal Slice candidate.
- `pytest tests/test_native_mot.py -k test_native_mot_block_forward`: run one focused behavior.
- `python scripts/train_benchmark.py --task needle --arms patch4,slice --steps 50 --seeds 1`: run a short benchmark smoke test.
- `python scripts/train_benchmark.py --bench --device cuda`: run the optional CUDA benchmark.

## Coding Style & Naming Conventions

Target Python 3.10+ and follow PEP 8: four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Retain type hints and document non-obvious tensor semantics. Keep experiment CLIs explicit with `argparse`; avoid machine-specific paths. No formatter or linter is enforced, so match nearby code.

## Testing Guidelines

Use pytest and name tests `test_<behavior>`. Add deterministic tensor tests for model or routing changes; cover shapes, gradients, defaults, and invariants instead of long training convergence. Run the affected test file, `python tests/test_deslice_scatter_and_gate.py`, and the full suite when shared APIs change.

## Commit & Pull Request Guidelines

Recent history uses imperative subjects such as `Fix probe_token_acc causal alignment`. Keep commits focused. PRs should state the falsifiable claim, summarize code and metric changes, list validation commands, and link issues. Include figures for visual changes and compact artifacts under `results/published/`; exclude datasets, checkpoints, caches, and raw logs.

## CGB / MengLi program charter (read before CGB work)

### Definition and main generative chain

**无限流生命体（用户定义 v2，2026-10-12，原文；[来源](docs/information_boltzmann/CGB_AXIOMS_V0_20261022.md#41-无限流生命体用户定义-v22026-10-12原文)）**：一个具有有限状态空间和有限计算资源的开放系统，持续暴露于近乎无限的信息流中。它通过自举，自发形成并维持以自身存在为核心的内在目的，利用自身有限的内部自由度，通过记忆、学习、信息压缩等过程形成一定程度的智能，并利用这些智能形成关于周围世界的局部世界模型，据此预测未来、选择行动，主动改变自身与环境之间的相互作用关系，形成操作闭合，进而维护自身生存状态，构建并维持自身生态位，使自身状态保持在由自身定义的可存活区间之内。

**语言实例化（现行规格；[来源](docs/information_boltzmann/CGB_LANGUAGE_LIFE_MODEL_V1_20261025.md#2-任务与系统边界)）**：语言是行为层面唯一的摄入。储备是同一有限状态中仍可读出、递推并改善后续语言调节的结构区别；耗散是遗忘、串扰或漂移使这些区别合并或失效。不得另设营养池、食物币、维护倒计时或可任意指定的 `intake`。

**实现主链（用户确认，2026-10-31，原文；[来源](docs/information_boltzmann/CGB_AXIOMS_V0_20261022.md#411-实现主链用户确认2026-10-31)）**：生存是生命体的第一需要：失去可存活性即退出生命过程；这不是外部奖励或预先写入的“自我目的”。在守恒、有限容量、持续演化与近乎无限信息流共同成立时，封闭状态会丧失对未来仍有用的可区分结构，因此生命必须作为开放系统从边界持续吸收信息负熵以维持自身。这里的负熵是同一有限状态中可用于后续语言/世界交互的有效预测、记忆与压缩结构，不另设食物、能量币或储备账户。

但可摄入的新结构会被学尽，环境也会变化；被动接收不足以持续维持可存活性。因此系统必须主动适应、改造或转换与环境的相互作用，以获得新的可用信息。主动推理由此要求预测、因果关系与自我模型；自举形成的特异性记忆、遗忘、压缩和学习结构，随后反过来维护生存。有限容量又迫使选择性保留、代谢、遗忘与压缩；无限流要求这些更新的工作量不随既往经历长度增长。延迟后果下的归因由此要求广义资格迹，并进一步约束隔室、连接与三因子学习机制。

### 工作方法：生成而非拼装（代理执行规范，不是原文）

- 用户 2026-11-03 的方法论纠正：目标是寻找约束尽量充分的相容集合，使共同生成结构在交叠约束下显形；无需先证明某种结构是所有可能身体的必选结构。普遍必要性不是采用一个有明确条件、可实现且可检验的机制的前置许可。丰富的约束与少量的共同生成律同时追求。
- 相容集合必须串成因果生成链：记录同一状态/转移怎样产生下一条定理的前提，以及学习所得结构怎样反过来改变后续学习。优先共享生成规则，不按定理数量新增模块；以删除某个生成承诺会破坏什么能力检查相对最小性。当前源链见 `CGB_THEOREM_CONSISTENT_CORE_20261103.md` 第7节及其机器切片；设计因果边、Lean证明依赖、未完成构造要分别标记。
- 搜索时保留定义硬约束，并积极纳入有依据的条件约束、构造和联合见证；把其前提作为本候选的明确承诺，检查它们是否作用于同一状态和转移。不能因为“不是普遍必要”就丢弃约束，也不能用新增独立模块逐项兑现。按当前定理库存和资源边界追求相容最大覆盖；每个未纳入项记录已被蕴含、真实冲突、资源不符或尚未完成构造，不能把未知当作不相容。局部相容、联合可实现和全局最大性分别记录，不要求证明唯一架构后才开始构造。
- Start from the complete living motion, not a missing feature. One boundary-language event must enter the same process that predicts, changes selective structure under finite capacity, changes later prediction/action, and thereby changes future boundary language. Relations and motion are primary; H/C/S, gates, traces, reserve, dissipation, latches and actions are only projections of that shared transition.
- Work bidirectionally. Ascend from biology, successful modern language models and failed real runs to invariant causal roles and counterexamples; reduce these to minimal CGB premises. Descend from axioms and Lean theorems to one common transition, an algorithm and real-language predictions that can falsify it. Lean establishes necessity and boundaries; algorithms establish realizable choices among the permitted paths. The core task is to generate the organism, not to imitate biology or merely prove properties.
- Justify an added degree of freedom through a required distinction, an existing construction, or an explicit jointly checked mechanism hypothesis; a universal impossibility theorem is not required before exploring a conditional construction. Check retained state and computational cost, and whether the same dynamics serve the coupled requirements. Repair the earliest broken relation rather than assigning a separate controller to each symptom. Addresses, usable associations, differentiated roles, gains, latches and long-lived causal structure should co-form through the common dynamics and language history, not be hand allocated.

### 公理 -> 定理 -> 机制 -> 算法与索引（代理执行规范，不是原文）

- Axioms state the non-negotiable conditions: locality, conservation, finite resources, open causal boundary and continuous time/information flow. Theorems derive necessary distinctions, capacities, constraints and impossibilities from them. Mechanisms are candidate common dynamics that satisfy several such necessities at once. Algorithms are resource-bounded numerical realizations of a mechanism. Passing a theorem does not promote a local kernel to an organism; a working algorithm does not make an unproved premise true.
- Before any CGB design, implementation or acceptance, begin at `docs/information_boltzmann/CGB_PROVEN_INDEX.md`, select the relevant L0–L6 requirement and reverse-slice its declarations. Read each Lean declaration, its direct premises, dependencies and existing executable references. Record: theorem -> premise -> same `Z` field/boundary event/transition -> algorithmic realization -> measured whole-loop consequence. Separate witnesses, assumed properties, or independently tested modules never count as joint realization.
- The theorem index is an automatically maintained design input, not a hand-written catalogue. When Lean changes, rebuild/synchronise the index and interpretations with the existing scripts, verify declarations/build/audit, and update the requirement slice and research tree when coverage or invalidation changes. Never infer theorem coverage from names, stale summaries or a neighbouring theorem.

### 五系统：同一身体的有序依赖（现行活规格）

**原文表（[来源](docs/information_boltzmann/CGB_LANGUAGE_LIFE_MODEL_V1_20261025.md#5-功能系统与依赖顺序)）**：

| 系统 | 共同状态中的职责 | 依赖关系 |
|---|---|---|
| 维生—稳态 | 储备、耗散、可存活区间与恢复 | L0 底座；约束所有其余更新 |
| 感知—预测—压缩 | 语言摄入、未来区别与局部世界关系 | 与学习可塑性共同形成 |
| 记忆—学习—可塑性 | 选择保留、资格迹、三因子和局部增益 | 读取预测误差与已到达后果 |
| 行动—生态位 | 表达、提问、改变后续可达语言 | 建立在维生、预测和信用之上 |
| 发育—组织 | 柱、连接、竞争和分化的可达结构 | 从出生起约束全部状态，不预标功能区 |

**工程阶段（项目状态记录，非定义；[来源](docs/information_boltzmann/CGB_LANGUAGE_LIFE_MODEL_V1_20261025.md#5-功能系统与依赖顺序)、[当前系统2/3报告](docs/information_boltzmann/CGB_LANGUAGE_SYSTEM23_CONTINUOUS_20261101.md)）**：系统 1、5 是底座；系统 2、3 是当前共同能力核心；系统 4 建立在 1–3 闭环之上。当前系统 2、3 已有真实语言运行，系统 4 已接两路真实文本选择、自听和同体更新的有限接口；三组对照未证明即时NLL策略更新有稳定收益，验收见 `CGB_LANGUAGE_ACTION_20261104.md`。提问、沉默/延迟世界及主动持续生存仍未完成。CGB 语言生命体的信用分三层：当拍梯度、固定窗口 BPTT、窗口外靠记忆取回（用户 2026-11-02 修订）；不得使用随历史增长的 BPTT、全历史 Jacobian 或 RTRL；固定身体的资源应保持相对于历史长度的 O(1)，持久资格/历史页须显式计入资源，不得隐藏。

### 五行式总装示例：柱介质（代理分析；证据来源见下）

**Wrong assembly:** implement H as a small predictor, C as a byte-addressed table, S as an independent conserved-credit tensor; separately prove local conservation, a three-factor update and a latch condition; then connect them. This produced a route determined only by the previous byte. C became largely a byte echo, S had little language content, and the body fell below a simple online count baseline. Each part had a local explanation; their joint transition did not contain contextual association memory.

**待准入的生成路径，不是已证明的结果：**观察到的反例是“历史的最新字节相同、未来语言却不同”。关联容量与因果载体定理要求状态保留这种可区分性。候选共同转移应让递推的 H/C/S 上下文形成读查询，上下文和到达事件形成写键/值，C 在有限释放/竞争下保存关联，后续预测读出该关联，边界后果再改变同一 H/C/S。它只有在 Lean 准入和真实语言验收都通过后，才可声称同时服务系统 1、2、3、5；系统 4 仅在输出改变后续外部语言时接入。

The five phases are a way to diagnose this mutual generation, not five entities or ten programmer-installed control pipes. Do not add a named controller merely to fill a relation. A proposed change must preserve the carrier's contextual association floor, predictive language ability, finite/O(1) bounds and H/C/S single-body closure while showing its claimed effect on competition, plasticity, latching or credit. Before advancing to long-delay eligibility, full-parameter learning or action, establish real-language context discrimination, matched finite-memory online baselines, revisit/relearning and recorded state/resource budgets.

## Information Boltzmann experimental policy

- Before proposing a new CGB mechanism, carrier or state array, search `docs/information_boltzmann/CGB_PROVEN_INDEX.md`, read the actual Lean declarations and check existing executable references, especially `reviews/unified_reference_20261025.lean` and `information_boltzmann/core/cgb_local_medium.py`. Reuse existing shared state and proofs before designing replacements.
- Distinguish an absent mechanism from an existing mechanism whose joint behavioral proof or implementation refinement is incomplete. Record the exact reusable declarations, their premises and the remaining bridge; do not describe that bridge as a need to invent the carrier again.
- Proposed learning models must pass the applicable Lean admission before execution. Prove required properties for the same state transition and world conditions; separate witnesses or target properties assumed as premises do not establish their joint generation. Preserve the distinction between a conditional theorem and evidence that a runtime model meets its premises.
- For a language-body change, record the carrier algebra before training: what writes an association, what recurrent context supplies the read query, where the association is retained, and what reads it later. Test histories sharing the same recent byte but requiring different futures; this prevents a fixed-token route or decoder shortcut from being mistaken for contextual memory. Compare against explicit, matched online finite-memory floors before attributing outcomes to CGB mechanisms.
- User directive (2026-09-10): capability research must use professional real-world datasets and sufficient training. Do not create or run synthetic/toy bit-memory, rule-switch, or cue-probe capability experiments, or make them prerequisites for dataset training.
- Before drawing positive or negative architecture/capacity/memory conclusions, record dataset revision and splits, parameter count, tokens/examples processed, actual optimizer updates, training curves, periodic independent validation, and the stopping criterion. Establish convergence or clearly label a budget-limited run inconclusive; a fixed short run is not evidence of convergence.
- Compare capacity and training budgets fairly. Online cumulative training CE is not final validation loss; identify tokenizer units and checkpoint provenance. Do not use the test split for tuning.
- Keep deterministic numerical and interface unit tests (conservation, analytic solutions, gradients, causality, checkpoint continuation). These verify implementation only and are not toy capability studies.

## Continuous-individual evaluation policy (2026-10-04)

- User directive: systems intended to persist and learn on infinite streams use active-learning evaluation as their primary protocol. Keep physical state, eligibility, weights, optimizer moments, pending gradients and elapsed time continuous. Use the actual learner; do not substitute an evaluation-only update algorithm.
- Cover three questions: first-pass prequential predictive quality, recovery after a real context change, and actual A-to-B-to-A revisit/relearning. Score predictions before the target can update their parameters. Report replay separately from first-pass traffic and preserve optimizer cadence across phase boundaries.
- Record real exposure intervals, fresh-stream cursors, all scored bridge transitions and actual optimizer updates. Never fill an intervening-duration field with unexecuted experience. Restore complete learning state on resume; no cold/reset evaluation of an ongoing individual.
- Historical frozen benchmarks retain their original provenance. Label new primary results as live/active-learning results rather than reinterpreting old scores. See docs/information_boltzmann/LIFELONG_EVALUATION.md.
- Fourth-pillar monitoring reports actual field/flux energy accounting, representation structure, temporal roughness, phase-separated risk drift and full physical-state conditional response. Keep predictive surprisal distinct from thermodynamic entropy flow; do not interpret low rank, smoother trajectories or small partial-gradient DNR as certified noise expulsion, criticality or optimal convergence. See docs/information_boltzmann/PERSISTENT_MEDIUM_HEALTH.md.
- Do not resurrect removed toy conclusions about decoder depth, memory failure, gamma superiority, or error-attribution root causes. The user has now authorized the first single-individual OWT run of 5000 optimizer updates, with learned local gamma and GPU/memory calibration. Preserve one continuous state and do not launch a population sweep as a substitute.

## Native fly efficiency policy (2026-10-19)

- Consider memory and compute during implementation, before adding a mechanism. Distinguish physical state, local eligibility, parameter sensitivities, optimizer state, autograd graphs, and CPU/GPU temporary workspace; record their dimensions and byte budgets.
- Prefer lossless index packing, shared immutable geometry, bounded workspaces and independent-RHS/edge blocking. Preserve all physical nodes, connections, real elapsed time and complete learning state; do not substitute pooling, shorter world clocks or reset histories for an efficient implementation.
- Ongoing native plasticity need not reproduce BPTT gradients. Fast local BPTT is allowed within a measured resource budget; do not require a dense global historical Jacobian to declare a biological pathway connected.
- Validate physical outputs and applicable derivatives after execution/storage changes. Report full-restore, forward, backward and optimizer memory separately; frozen replay alone is not training admission. Keep main training paused during repair unless the user resumes it.
- Custom autograd operators must put retained tensor data through `save_for_backward` so checkpoint hooks can release it. Flatten physical-state tensors into nested checkpoint inputs rather than capturing state objects in closures. Bound diagnostic temporaries such as finiteness checks on CPU and GPU, including strided large language-head gradients.

## Checkpoint retention during mechanism repair (2026-10-19)

- Full mechanism snapshots currently cost roughly 1.4–2.5 GB each even with zero new training. Budget complete snapshots and atomic replacement space before running a migration; do not place these archives on the system drive.
- For the ongoing mechanism-repair chain, normally retain the latest complete candidate and one immediate complete predecessor. After the new archive is verified and reports/configuration/provenance are preserved, remove obsolete reported smoke/interface-test payloads under the user's cleanup authorization. Write a deletion manifest and `CHECKPOINT_PRUNED.json`; historical result evidence is not proof that its old checkpoint file still exists.
- Keep the paused main individual, explicitly named key recovery points and separately requested historical baselines. Never infer that a reported mechanism test is a champion or keep every intermediate full snapshot indefinitely. Earlier smoke states with different brief exposures are obsolete test states, not necessarily byte-identical duplicates; describe deletion honestly.
- Preserve anatomy caches, source code, metrics, dialogue and failure evidence. Check actual free space after cleanup or relocation; moving data between drives does not reduce total storage. Do not change the system pagefile or interrupt unrelated jobs to free space.
