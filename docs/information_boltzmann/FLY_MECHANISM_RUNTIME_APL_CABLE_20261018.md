# 当前候选的统一接线与验收清单：APL局部形态

> 本文为APL阶段历史快照。最新零暴露KC末梢候选见 `FLY_MECHANISM_RUNTIME_KC_TERMINALS_20261018.md`。

候选：`E:/fly_stream_overflow/native_apl_cable_20261018/last.pt`，33000词、4154次累计更新；主线32384保持暂停。

| 系统 | 注释细胞 | 专用开关 | 本存档急性作用（预测/选词最大log概率差） |
|---|---:|---|---|
| sensory_surfaces | 15896 | early_visual_graded=开, sensory_precision=关 | visual: 0.9769/4.241 |
| olfactory_relay | 3783 | local_relay=开 | al_relay: 0.9413/3.219 |
| mushroom_body_memory | 4501 | kc_mbon_memory=开, kc_dan_feedback=开, dpm_apl_axonal_projection=开, apl_local_cable=开 | mb_regulators: 0/0；mb_terminal: 1.465/4.172；mb_terminal_gaba: 0.9482/5.106；dpm_5ht: 1.716/3.985；apl_release: 1.163/4.023；apl_axial: 0.8915/3.736 |
| neuromodulatory_state | 389 | dopamine=开, oa_5ht_extension=开 | amines: 2.021/5.708 |
| central_complex_state_action | 2950 | cx_extension=开 | cx: 1.191/2.823 |
| association_and_context | 121550 | 通用图动力学 | 未独立验收 |
| ascending_body_feedback | 21362 | historical_body_state_retained=开, historical_proxy_diagnostic_only=关, native_peripheral_exchange=关 | 未独立验收 |
| descending_and_vnc_motor | 2227 | historical_body_state_retained=开, historical_proxy_diagnostic_only=关, native_peripheral_exchange=关, native_motor_language_port=开 | motor_terminal: 0/9.586 |
| visual_and_long_range_projection | 99154 | early_visual_graded=开 | visual: 0.9769/4.241 |
| endocrine_and_ens | 141 | lk_receptor_exchange=开 | endocrine: 0/0 |

注释分组可重叠，不能相加。表内是干预作用强度，不是NLL改善或学习收益。

15支配对使用当前存档后4个真实输入间隔 `[19,17,17,10]`，共63拍；同状态/感觉采样、sham、恢复、来源及RNG控制通过。旧双侧均值代理无作用。APL释放与轴向传导都影响KC、MBON、DAN和选词。两个APL关断支在这4词上的CE分别低0.01514和0.03179，不能声称接入就有收益。

APL采用全部88905个提供形态节点、88900段及实际突触点。新材料参数Adam更新8次，局部电位已经分化；KC各bouton仍未重建。

32个已保存妈妈主动输入，8次更新、483实际拍，更新前NLL **4.44436**，同批固定词频 **6.20404**。来源是mother_call；本轮没有新母女话轮，不能冒充正常妈妈回复或互动能力改善。

209项相关检查与独立Deslice检查通过；完整模型、脑、优化器、信用、世界和RNG恢复一致。全量入口仍受原先5项缺失诊断脚本阻塞，8个导入即训练入口另行排除。

**资源未通过长训准入**：完整33000已保存，但最后重算分段2的峰值3847MiB超过3800MiB，保存后按保护规则退出。旧分段4已跑完32输入但最终保存失败；失败重试引用与缓冲清理已修，真实重载及两项故障保存检查通过。当前进程不运行，不能把完整保存写成资源风险已消失。

先前两份对照失败报告保留：原有多巴胺Triton原子加法的舍入扰动了跨窗敏感度；冻结审计改用已有等方程的有序归约，未放宽容差，sham与恢复支现在逐字段零差。

机制说明：`FLY_NATIVE_APL_CABLE_20261018.md`；剩余缺口：`FLY_NATIVE_COMPLETENESS_20261018.md`；报告：`results/published/fly_native_apl_cable_20261018.json`。
