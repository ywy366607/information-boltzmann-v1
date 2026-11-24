# 2026-11-19 退役运行清理

本轮按用户明确指令清理D盘本项目无用运行材料。已删除740个文件，逻辑大小37.167 GiB，实际磁盘可用空间增加21.572 GiB；结束时D盘空闲24.145 GiB。稀疏表文件的逻辑大小大于其实际占用。

删除对象为111个实验目录下已退役的检查点、重复表、SQLite工作库、恢复副本与激活暂存。逐文件清单与原始大小/时间/理由见 `results/published/cgb_retired_runtime_cleanup_20261119.files.jsonl`；汇总见同目录 `cgb_retired_runtime_cleanup_20261119.json`。相关目录已有RUNTIME_PAYLOADS_RETIRED标记。旧result/status作为原始研究记录保留，恢复能力以该退役标记及当前源为准。

完整保留并在删除前后校验哈希：

- 当前carry2493：`scratch/cgb_momentum_value_2489_20261119/carry/`整个目录，含权重、Adam、S/P、表、档案、待结算、世界与RNG。
- 原主2401：`C:/Users/85750/.codex/cgb_native_topology_20261116/body/last.pt.gz`及其因果档案，原目录未操作。

代码、Lean、数据集、文档、配置、发布结果和事件记录保留。Lean的.lake依赖与构建保持，以便继续原证明工作。

schema3四轮配对仍待重启。控制臂此前完成1轮/4更新，第二轮因D盘临时激活写入空间不足退出；实验臂尚未启动。部分轨迹保留为失败证据，下一次两臂都从完整2493共同源开始。当前pipeline_status已改为awaiting_restart_after_disk_cleanup；原失败状态单独保留。
