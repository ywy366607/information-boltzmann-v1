# KC三路输出进入统一执行程序：33000词存档复核

本轮把已实现的三类KC原生输出（KC→全脑快边、KC→APL局部、KC→DAN局部）从各自的手写函数降为同一份数据驱动执行程序（`fly_native_templates.compile_existing_kc_routes` → `fly_native_runtime`）。这是**无损降阶**：没有新生物机制、新参数或新物理历史。

入口：`scripts/ib/fly_interface/unify_native_runtime.py --migrate`；字段复核：`verify_migrated_archive_fields.py`；完整恢复：`verify_native_full_restore.py`。

## 真实存档复核（源：`F:/fly_stream_overflow/native_kc_lateral_20261019/last.pt`，33000词）

| 检查 | 结果 |
|---|---|
| 同一段真实时钟（保存的待办队列，19拍，词467→832，母亲呼唤） | legacy / sham / compiled / restored 四臂全部物理场、肌肉、logits逐位一致 |
| 源存档、参数、CPU/CUDA RNG | 未改变 |
| 编译程序 | 18条指令，接管 kc_fast / kc_apl / kc_dan，新增参数0、新增物理状态0 |
| 新存档逐字段（逐项mmap哈希） | 旧字段改变0；仅新增 `native_program_digest`、空 `native_state`、`native_runtime_document`、`native_runtime_program` |
| 完整训练器恢复（CUDA） | 模型、learner、优化器、RNG、MotherWorld及其RNG逐项一致：`exact_restore_passed` |

CPU单线程每臂约71秒（19拍）。新候选：`F:/fly_stream_overflow/native_unified_runtime_20261019/last.pt`（sha256 `398b232e…`，2485326705字节）；报告 `report.json`、`field_check.json`、`full_restore/report.json` 同目录。按保留策略，当前完整候选为此存档，直接前驱为 `native_kc_lateral_20261019`。

## 边界

- 这只证明执行层改写不改变行为；不代表探索、学习或行为收益。
- 本存档的48.6万对感觉回流仍未运行；之后已在下一个候选中接入，见 `FLY_SENSORY_TERMINALS_20261019.md`。
- 主训练仍暂停。
