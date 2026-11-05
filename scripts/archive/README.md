# 历史工具归档

这里保留已经完成用途的脚本原文，避免它们继续出现在当前执行入口里。`.txt` 后缀表示历史文本，不作为可运行模块。文件字节与清理清单的SHA256对应。

- `theorem_interpretations_20261105/`：一次性释义种子和整批重建脚本。当前维护入口是 `scripts/incremental_sync_theorems.py` 及 `build_proven_index.py`、`build_theorem_basis.py`、`build_theorem_catalog.py`；不要用历史种子覆盖当前审核结果。
- `navigation_v2_20261105/`：绑定旧进程PID与固定目录的v2排队、收尾工具。当前使用显式模块CLI，旧结果留在本地原目录。
- `observed_gain_run_20261105/`：已完成512轮运行登记的11份实际执行源，保留原始字节与SHA256；其中收益读口仍为修复前版本。当前代码已经修复动作归属，历史快照只用于复核旧结果，不作为当前运行入口。

当前项目入口：[CGB / 梦漓](../../docs/information_boltzmann/README.md)。
