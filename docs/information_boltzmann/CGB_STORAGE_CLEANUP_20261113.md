# 2026-11-13 研究目录空间清理

用户明确允许删除项目中无用文件。本次按“保留当前身体、每条历史运行的最新完整恢复点、原始数据和日志”的范围清理；较早中间检查点是不同时间的状态，**不是按内容相同判定的重复文件**。删除后不能回到这些旧时间点，但仍可从列出的后续完整检查点恢复。

## 删除的旧中间检查点

| 被删除文件（`results/`下） | 字节 | 删除前SHA-256 | 保留的同运行后续恢复点 |
|---|---:|---|---|
| `cgb_delayed_squad_20261109/continuous_sft_study/raw/checkpoint_0008.pt` | 801052298 | `acb71baffb0b7238661c4ae94e3e3617db12a55a86f5fc78583180d5ec10b34c` | `checkpoint_0016.pt`，STOPPED.md记录为最新完整恢复点 |
| `cgb_native_s_credit_20261108/measured_resume64/cut/checkpoint_000064.pt` | 804763586 | `a00634cc9d108f2e84b9c0a3e809427fb5c813710fd34643fa457777d58978e6` | `checkpoint_000096.pt`，SHA `d17d16e82230b79df2ad313cdd58dd5d70d3e0a3b6277f9d6ed4e6389bf2776c` |
| `cgb_native_s_credit_20261108/measured_resume64/mismatch/checkpoint_000032.pt` | 803425780 | `6bf6373fe3d65f753f5f20848e559938c7f599fb226fc7dc68f6dfb2bfda12ec` | `checkpoint_000064.pt`，SHA `7a6cde6f4f9c7f0c200d674a752a6d1b28da36481ce07c5df5418e31e1341d82` |
| `cgb_ruler_adaptation_20261109/common_v2/checkpoint_0064.pt` | 796980944 | `e5265d55998bf8893f8f2e4e1d993a5fced4fe3619fb84163dc82be1ee99a548` | `checkpoint_0128.pt`，SHA `3755988c4545f6d97f2ae8800f83e1f50ae31fe4a3bb7d180a0b03e1ee99a548` |

合计3,206,222,608字节，约2.99 GiB。删除前复核后三个保留恢复点的实际SHA与各自`latest.json`一致。个别历史`latest.json`和`checkpoints.json`仍提到已删的`previous`或中间路径；它们保留当时运行记录，本清单记录其当前文件已删除。

## 删除的运行残留

- 中断写入的 `cgb_mother_continuation_20261110/longrun_512/real_01/last.tmp.gz`（23,622,446字节）和 `pair_real_1800_to_1856/last.tmp.gz`（11,273,650字节）；各自正式`last.pt.gz`保留。
- `cgb_navigation_v4q_20261105/last.tmp.gz`（304字节）及 `medium_v8_joint_bptt32_96k/progress.7896.tmp`（2013字节）。
- 本轮检查生成的两个快表临时工作区 `F:/cgb_joint_temp_20261113_01/` 与 `results/cgb_joint_world_cycle_20261113/table_runtime_20261113/`，各约230.56 MiB；只删除了核对过的6个临时数组及空目录。后者所属只读GPU审计进程在用户中断后仍存活，已明确匹配命令行并停止；审计记录保留。
- `scratch/compiler_cache/torchinductor/fxgraph/` 下45个旧编译图缓存文件，合计676.2 MiB，最后修改于10月10日；目录内容可由PyTorch再生成。保留该目录以外的编译产物，再运行使用这些旧图的任务时可能重新编译。

## 当前空间与边界

清理后D盘空闲约3.80 GiB，C盘约1.50 GiB；`results/`尚约14.04 GiB，其中有完整恢复点、评估和研究历史，不能仅凭文件大就称无用。`data/`里的果蝇连接组、GPT-2权重和OWT语料属于原始数据/模型，也未因当前路线未使用而删除。当前持续身体 `results/cgb_mother_dsh_20261113/body_resumed/last.pt.gz` 保留，SHA-256仍为`851f47bb4a72b46c98682b6b3d2188a94239bc4923e3cab65b26b0b082f9b9f1`。

**结果更正：**完整身体的GPU只读审计在用户中断后、检查残留进程期间完成了4份草稿的规划与逐份信用计算。记录`results/cgb_joint_world_cycle_20261113/cuda_credit_memory_project.json`最终为`completed=true`，策略回放误差0，GPU峰值分配146,006,016字节；没有妈妈调用或optimizer更新。临时工作区在进程退出后已清理，真实训练仍未完成。
