# 2026-10-18 检查点精简

## 2026-10-27：果蝇权重全部删除（以本节为准）

用户确认这些果蝇检查点都已不可用：果蝇耦合线 2026-10-19 已关闭，当前主线（嵌套 CGB 框架里生成的生物）不使用其中任何权重。

- **删除范围**：E:/ib_checkpoints、E:/fly_stream_overflow、F:/fly_checkpoints、F:/information-boltzmann-pt-review-20260928 等目录下的全部权重载荷，共 165 个文件，26.35 GB。下文提到的主线 32384、6120、各恢复点、m_main、CTM 基线、APL/KC 候选都已不存在。
- **清单**：`results/published/checkpoint_cleanup_fly_all_weights_20261027.json`；各目录另有 `CHECKPOINT_PRUNED.json`。
- **保留**：报告、配置、日志、对话、恢复元数据、连接组数据 `whole_brain_contacts_buckets` 和几何缓存。
- **结论去处**：都在文档和记忆里，例如冠军发现、机制索引、耦合线关闭说明。
- 同日更早一次为 Lean 构建腾空间删了 2.0 GB，见 `checkpoint_cleanup_lean_enospc_20261027.json`。

## 当前保留规则与2026-10-19清理

以下“最新”段落均为历史交接。本次按用户再次授权，删除16份已有报告的过期机制/接口测试权重，合计30,081,934,040字节（约30.08GB），保留全部报告、配置、对话和解剖缓存；目录写入`CHECKPOINT_PRUNED.json`。这些短测试状态可能有不同暴露，不把它们称为逐字节重复文件。完整清单及删除前SHA256见`results/published/checkpoint_cleanup_superseded_20261019.json`。

当前机制链只保留最新KC轴突调制候选`F:/fly_stream_overflow/native_kc_lateral_20261019/last.pt`，以及直接前驱APL候选`E:/fly_stream_overflow/native_apl_local_release_20261019/last.pt`。两者连同主线`E:/ib_checkpoints/newborn_reading/last.pt`在本次删除前均重新核对SHA256。主线32384、6120边界及6632/6656/16000/24448/31360/32384关键恢复点、m_main及历史CTM基线未动。此前DAN信用/接收信用、APL被动电缆、MB末梢、amines、motor、body/AL/visual和context机制测试的旧payload已删除，历史路径不能再用于恢复。

以后机制链通常只留“最新完整副本＋一个完整直接前驱”，命名关键恢复点另算；不要每次接线永久累积2GB级完整快照。C盘两份FlashAttention安装缓存经逐文件哈希核对迁到`F:/rebuildable_cache/uv_archive_20261019`，原路径保留junction，未删除已安装环境。迁移报告见`results/published/system_cache_relocation_20261019.json`。分页文件设置未改，其他正在运行的任务未中断。

## 以下为历史记录

最新：保留 `native_dan_receiver_credit_20261019/last.pt`（33000）、完整父状态 `native_dan_credit_20261019/last.pt`、APL来源和暂停主线。零暴露 `native_dan_release_20261019/last.pt` 已在逐项证明被完整父状态覆盖后精简，以保留新完整副本及原子保存余量；报告/配置/源码快照保留。见 `results/published/checkpoint_cleanup_dan_receiver_credit_20261019.json`。以下为历史记录，旧路径不保证仍有权重。


最新：保留 `native_dan_credit_20261019/last.pt`（33000）、直接父状态 `native_dan_release_20261019/last.pt`、APL来源及暂停主线。为保留完整新副本和原子保存余量，旧零暴露KC空间多巴胺中间权重在逐项验证模型/物理/学习器/优化器及世界RNG被直接父状态完整覆盖后精简；报告配置及源码快照仍在。见 `results/published/checkpoint_cleanup_dan_credit_20261019.json`。以下为历史记录，历史路径不保证仍有权重。


用户授权清理当前没有完整冠军、仅用于机制 smoke 或已淘汰架构的权重。本轮删除 29 个权重/压缩恢复包，合计 28.367 GiB；逐文件清单见 `results/published/checkpoint_cleanup2_20261018.json`。日志、对话、配置、报告和归档源材料均保留。没有恢复训练。

## 最新连续机制候选（2026-10-19更新）

当前候选 `E:/fly_stream_overflow/native_dan_release_20261019/last.pt`（33000），
直接父状态 `E:/fly_stream_overflow/native_kc_dopamine_20261019/last.pt`、APL来源及暂停主线保留。
新增正式输入/更新均为0，完整恢复及来源对照通过，不称冠军。
旧零暴露KC终端、STP、接触记忆中间权重已逐项验证完整覆盖后精简；
报告/配置/来源快照保留。删除清单分别见 `results/published/` 下
`checkpoint_cleanup_contact_20261019.json`、`checkpoint_cleanup_dopamine_20261019.json`、
`checkpoint_cleanup_dan_release_20261019.json`，原目录保留 `CHECKPOINT_PRUNED.json`。
以下是历史记录，历史路径不等于权重仍在。

截至APL本轮验收，保留暂停主线 `E:/ib_checkpoints/newborn_reading/last.pt`（32384）、
完整来源 `E:/fly_stream_overflow/native_mb_terminal_20261018/last.pt`（32968）、
当前完整候选 `E:/fly_stream_overflow/native_apl_cable_20261018/last.pt`（33000）。
后两者均有实际完整训练器恢复检查，不称为冠军。33000在显存保护退出前已经完整保存，
主线仍暂停；最新大小、时间和摘要见候选目录的 `checkpoint_metadata.json`。
APL失败保存留下的临时碎片已清理，失败日志保留。旧四份列表只是下方清理完成时的快照，
不能据此覆盖最新候选或把日志进度当存档进度。

## 清理完成时可恢复状态（历史快照）

| 用途 | 路径 | 已存 token |
|---|---|---:|
| 暂停主线 | `E:/ib_checkpoints/newborn_reading/last.pt` | 32384 |
| 修复基线 | `E:/fly_stream_overflow/native_context_20261018/continue128/candidate.pt` | 32576 |
| 身体机制测试副本 | `F:/fly_stream_overflow/native_body_20261018/last.pt` | 32768 |
| AL 机制测试副本，部分进度存档 | `F:/fly_stream_overflow/native_al_20261018/last.pt` | 32772 |

以上四份均已用 CPU 检查可读取，包含模型、learner、CPU RNG 和妈妈世界状态；这不是重新执行恢复等价测试，也不把候选称为冠军。AL 的 32832 仅存在日志，最终保存失败，不能从该点续训。

## 回退与历史

保留 `before_consequence_6120.pt`，以及 reading 谱系的 6632、6656、16000、24448、31360、32384 压缩恢复点。保留 `m_main/last.pt` 和旧 CTM 对照 `q8_fly_ctm_settle14_100k/last.pt` 等历史主状态。本轮没有根据“当前无冠军”抹掉所有学习历史。

删除 CX、MB 中间测试权重（后续身体/AL 副本已继承），旧 live64/live256 smoke，HX1、LEWM 两分支及 pipeline14 的旧权重，普通中间恢复包。删去 payload 的目录写入 `CHECKPOINT_PRUNED.json`，目录中的历史报告路径只能作来源引用，不再代表权重仍在。

清理后 E 盘约 12.47 GiB、F 盘约 19.25 GiB 可用。下一次运行前重新核实空间，为完整 checkpoint 和原子替换临时文件同时留足余量。

## 2026-10-19机制候选补充

保留`C:/Users/85750/.codex/tmp/fly_apl_local_release_20261019/last.pt`（33000，SHA256 `cac15cd68d3d05accd3655e82fa68a81b202e9652fa219b3b0b0e443a586374f`）和其E盘DAN接收信用来源。这是零暴露机制候选，不是完成报告即可清理的纯smoke，也不是冠军。新增APL局部钙/释放资源；168项检查和CPU完整状态恢复通过，全脑因果回放未获资源准入，主线32384仍暂停。详情及元数据在该目录和`FLY_APL_LOCAL_RELEASE_20261019.md`。本条不重新确认上表历史文件的现存状态或旧磁盘余量。

随后APL候选经逐文件SHA256核对迁至`E:/fly_stream_overflow/native_apl_local_release_20261019`，原C目录为junction，未删学习状态。最新KC轴突间调制候选在`F:/fly_stream_overflow/native_kc_lateral_20261019/last.pt`，33000，SHA256 `9b1a2142c297f5669ad2282afdb9363c2a2d3a858670f21b0f59f8304276bf6b`。该目录有完整归档及迁移/核对元数据；旧权重、脑状态、优化器、信用、队列及RNG保留一致。暂不批准训练采用，缺完整原子替换空间及新候选全脑恢复/运行验收；不能按纯smoke清理。主线32384及原有边界恢复点保留。
