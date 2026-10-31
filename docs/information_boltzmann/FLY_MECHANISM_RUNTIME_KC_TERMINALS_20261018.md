# 当前候选的统一接线与验收清单：KC局部末梢

候选33000为零新暴露扩展，本轮0真实BPTT更新；主线32384仍暂停。

| 系统 | 注释细胞 | 当前专用开关 | 短程急性作用：预测log概率差/动作读出logit差 |
|---|---:|---|---|
| sensory_surfaces | 15896 | early_visual_graded=开, sensory_precision=关 | visual: 0.1698/2.174 |
| olfactory_relay | 3783 | local_relay=开 | al_relay: 0.2489/2.894 |
| mushroom_body_memory | 4501 | kc_mbon_memory=开, kc_dan_feedback=开, dpm_apl_axonal_projection=开, apl_local_cable=开, kc_local_terminals=开 | mb_regulators: 0/0；mb_terminal: 0.05342/1.201；mb_terminal_gaba: 0.0519/0.2245；dpm_5ht: 0.06928/0.3985；apl_release: 0.3002/4.962；apl_axial: 0.0004826/0；kc_sites: 0.05342/1.201；kc_site_transport: 0.01985/0.03876 |
| neuromodulatory_state | 389 | dopamine=开, oa_5ht_extension=开 | amines: 1.249/8.258 |
| central_complex_state_action | 2950 | cx_extension=开 | cx: 0.06895/0.09161 |
| association_and_context | 121550 | 通用图动力学 | 未独立验收 |
| ascending_body_feedback | 21362 | historical_body_state_retained=开, historical_proxy_diagnostic_only=关, native_peripheral_exchange=关 | 未独立验收 |
| descending_and_vnc_motor | 2227 | historical_body_state_retained=开, historical_proxy_diagnostic_only=关, native_peripheral_exchange=关, native_motor_language_port=开 | motor_terminal: 0/0 |
| visual_and_long_range_projection | 99154 | early_visual_graded=开 | visual: 0.1698/2.174 |
| endocrine_and_ens | 141 | lk_receptor_exchange=开 | endocrine: 0/0 |

分组可重叠。动作读出logit变化不等于实际选词变化；本次妈妈发言，没有提交动作。
17支使用同状态、RNG及存档后一个真实间隔19拍；sham与恢复支逐字段零差。
普通输出、可塑MBON、DAN反馈、局部APL四路的初始扩展总量均对齐。
同一KC各位点释放已分化，局部调节及运输干预均影响MBON/DAN和φ。

新增7项数值检查包含无优化器φ写入、条件迹导数与单窗口梯度重算。218项相关检查与独立Deslice通过。
全量入口仍受5项既有缺失诊断脚本阻塞，不能称全套通过。

本轮无新训练NLL；一个冻结目标CE4.77154只用于配对。此前APL的32词NLL4.44436/词频6.20404为历史母亲主动输入，不能替代本轮或主线正常回复。

KC使用4184720个提供形态节点与709034个真实释放位置的有效受体后信号；
胞体放电仍沿用原细胞模型，轴突电位波/钙通道、分子定位、位点φ/STP和电耦合仍缺。
原生前向可塑性已接入；持续探索、行为熟悉化与学习收益留待真实经历验收。

通用稀疏LU曾因主存工作区失败，已改为全部原始树节点上的精确消元，数值与伴随一致。
长BPTT与统一内存/计算优化按用户要求后置，当前未获得长训资源准入。

详见 `FLY_NATIVE_KC_TERMINALS_20261018.md`、`FLY_NATIVE_COMPLETENESS_20261018.md` 及当前JSON报告。
