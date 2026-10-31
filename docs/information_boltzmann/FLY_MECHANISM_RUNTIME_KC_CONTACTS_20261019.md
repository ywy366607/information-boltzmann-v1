# 当前接触记忆候选：运行与验收清单

候选：`E:/fly_stream_overflow/native_kc_contacts_20261019/last.pt`，33000。
原完整图165122个神经元、25321395条有向边；本轮0真实输入/优化器更新。
主线32384继续暂停。开关启用和本轮因果验收分开记录。

| 系统 | 当前启用实现 | 本轮因果证据 |
|---|---|---|
| sensory_surfaces | early_visual_graded | 本存档未重测专用机制 |
| olfactory_relay | local_relay | 本存档未重测专用机制 |
| mushroom_body_memory | kc_mbon_memory, kc_dan_feedback, dpm_apl_axonal_projection, apl_local_cable, kc_local_terminals, kc_site_resource_dynamics, kc_contact_memory | 本轮接触读/写控制通过；其他机制未重测 |
| neuromodulatory_state | dopamine, oa_5ht_extension | 本存档未重测专用机制 |
| central_complex_state_action | cx_extension | 本存档未重测专用机制 |
| association_and_context | 无启用项 | 本存档未重测专用机制 |
| ascending_body_feedback | historical_body_state_retained | 本存档未重测专用机制 |
| descending_and_vnc_motor | historical_body_state_retained, native_motor_language_port | 本存档未重测专用机制 |
| visual_and_long_range_projection | early_visual_graded | 本存档未重测专用机制 |
| endocrine_and_ens | lk_receptor_exchange | 本存档未重测专用机制 |

## 本轮交付

- 463628个实际KC→MBON接触（391849个物理释放位置）具有独立有效φ、资格迹、生化状态及条件信用；同一释放位置的资源事件仍共享。
- 初始递送与旧连接参考最大差1.86e-9；所有旧状态及完整训练器恢复检查通过。
- 同状态/RNG、实际队列19拍的5支检查包含写入关闭、读出关闭、sham、恢复；补充4支检查核对实际MBON输入通道。
- 423条旧连接内部出现局部φ差异，最大范围0.00125885；当前MBON输入与旧参考相差3.9080e-5。最终预测/选词差在这段重放中为0，不能称学习或行为收益。
- 238项相关检查通过；后续46项接触/上下文/后果特征/摘要检查通过，两批有重叠。独立Deslice检查通过。全量收集仍被5个既有缺失诊断脚本阻断，不能称全套通过。
- 新增浮点持久状态44.22MiB，另有接触身份索引约10.6MiB。冻结重放峰值2744.33MiB，19拍约23秒；未来BPTT窗口1，未进行真实训练显存准入。
- 当前多巴胺教师仍沿MBON隔室映射，空间DAN→KC接触/受体递送是明确下一缺口；新慢参数完整跨窗导数、长期选择性熟悉化与持续学习收益另列。

详见 `results/published/fly_native_kc_contacts_20261019.json` 与
`results/published/fly_mechanism_runtime_kc_contacts_20261019.json`。
