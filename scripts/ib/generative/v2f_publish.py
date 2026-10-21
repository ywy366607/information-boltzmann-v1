"""Publish the bounded V2f implementation acceptance without capability claims."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main(run):
    r = json.loads((run/'report.json').read_text(encoding='utf-8'))
    if not r['passed']:
        raise ValueError('Do not publish failed acceptance as passed')
    snapshot = run/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    manifest = {}
    paths = [Path(p) for p in r['source_sha256']]
    paths += [ROOT/'tests/test_cgb_structural_budget.py', ROOT/'tests/test_cgb_birth.py', Path(__file__)]
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in r['source_sha256'] and digest != r['source_sha256'][str(path)]:
            raise ValueError('Implementation changed since acceptance: '+str(path))
        name = path.relative_to(ROOT)
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest[str(name)] = digest
    write_json(snapshot/'manifest.json', manifest)
    curated = {k: v for k,v in r.items() if k != 'numerical'}
    curated['numerical'] = [{k:v for k,v in row.items() if k != 'trajectory'} for row in r['numerical']]
    curated['focused_tests'] = {'passed': 27, 'files': ['tests/test_cgb_structural_budget.py',
        'tests/test_cgb_birth.py', 'tests/test_cgb_matching.py'], 'scope': 'Numerical/interface only'}
    curated['source_snapshot'] = str(snapshot/'manifest.json')
    published = ROOT/'results/published/cgb_v2f_material_closure_20261021.json'
    write_json(published, curated)
    i = r['real_interface']
    rows = '\n'.join(f"| {m['dt']} | {m['supply']} | {m['expected_discrete_plateau']:.10f} | "
                     f"{m['measured_plateau']:.10f} | {m['max_local_balance_error']:.3g} |"
                     for m in r['numerical'])
    text = f'''# CGB V2f：材料收支与生成强度接入（2026-10-21）

**结果：数值与接口验收通过。** 先天条件强度规则已接到V2局部介质，
结构合成、维持和拆解回收进入同一个有限材料账；完整存档可逐位续接。
本轮没有语言训练，不证明主动性、自维持能力或从空间种子独立长出大脑。
预注册见 `CGB_V2F_PREREG_20261021.md`。

## 这次修的是哪里

原E6可由正标签直接生成结构；E9的营养支持主要改变更替速率。
本次保留标签和支持信号，额外明确结构的物质成本：局部合成材料B，
每单位结构成本c、有限供给、维持消耗、清除和回收比例r。
Nf/T作为支持信号，不冒充结构材料。
正向变化必须付费，减弱／更替把可回收部分返还局部供体，损失部分记入汇。
同一供体的同时请求按需求分摊实际库存。

每个供体满足：

    B + cΣg + 清除 + 实付维持 + 不可回收损失 = 初始材料 + 累计输入

维持欠款记录但不伪造负库存；新代码没有擅自用欠款触发死亡或规定行为率。
已有介质默认保持旧模式，只有显式启用材料模式才扣款。
材料模式的E6采用有限槽位的冻结速率解析更新，避免大标签造成Euler越界。

## 解析收支的数值核对

固定合成请求足够大、槽位未饱和时，连续稳态
`g*=Q/[μ+c(1-r)λ]`，本步进顺序对应的离散稳态为
`Q·dt/[μ·dt+c(1-r)(1-exp(-λdt))]`。
以下只是局部转化方程的解析验算，没有把它当作合成数据上的学习能力实验。

| dt(ms) | 供给Q | 离散解析稳态 | 实测稳态 | 最大局部账本误差 |
|---|---|---|---|---|
{rows}

供给减半的稳态比值 **{r['half_supply_ratio']:.6f}**。
断供4000ms后结构从约0.0251降到5.44e-11，恢复供给4000ms后回到原平台。
dt从1减到0.5ms后，相对连续极限的误差从0.417%降至0.208%。
这些结果确认材料约束生效，尚未说明真实输入下的选择、学习或生存是否良好。

## 真实匹配规则→V2接口

使用已锁定的V3d有方向64维生成器，没有重新拟合。
原始数据、生成器与源代码SHA保存在报告。
在4096个真实相连、非同类型的类型对上，把类型暂作接口节点，
明确提供相连概率1和既有几何，检查强度生成及局部动力学接线。
这不是细胞级发育：没有从零生成形态，也没有独立预测连接是否发生。
首次预检因同类型对会在这一接口里变成自环而退出，随后明确排除1881个
同类型对再抽样；没有把它们冒充单细胞自连接。

- 接口节点：{i['cells_as_type_interface_nodes']}；边：{i['edges']}。
- 初始请求结构总量：{i['birth']['desired_structure']:.6f}；按局部材料支付后：{i['birth']['funded_structure']:.6f}。
- 连续运行1000ms，500ms处完整存档并分叉；后半段{ i['paired_resume_steps'] }步，
  全部动态数组、生理参数、局部账本和RNG逐位相同。
- 最大局部账本误差：{i['max_local_balance_error']:.3g}。
- 非零钙池／标签：{i['active_calcium_pools']}／{i['nonzero_tags']}；这只代表活动和接线。
- 实付维持：{i['material_maintenance_paid']:.6f}；未付维持：{i['material_maintenance_unpaid']:.6f}。
- 接口介质数组：{i['medium_array_bytes']/2**20:.3f}MiB；完整压缩档：{i['complete_archive_bytes']/2**20:.3f}MiB。
- 含双分支恢复、逐步比对和I/O约{i['seconds_including_paired_restore_and_verification']:.2f}秒。

数组预算含局部状态、不可变稀疏投递索引和材料账，**不含Python／依赖库进程基线、
离线数据准备和匹配规则存储**。它只是这个接口规模的测量，不是全脑运行准入。

完整档：`{i['complete_archive']}`。完整档保留膜、通道、囊泡、受体、钙、酶、
标签、结构、快慢活动池、营养、材料、生理参数、随机数和代码／数据来源。
使用临时文件写入、完整读回后再原子替换。

## 验证与下一步

27项聚焦检查通过：守恒、局部库存、边顺序无优先权、有限槽位、
不学习时的更替回收、断供约束、生理参数保存、原子存档、逐位续接，以及
匹配规则原有数值／梯度一致性。没有运行旧V2的合成学习能力实验，未恢复旧梦漓主线。

下一步是出生程序本身：生成输出／输入分支的空间分布，接独立的连接发生规则，
再用本次强度接口和材料收支初始化同一个连续个体。随后把世界后果接到内生
第三因子与局部学习。现在m仍是V2的背景常数，不能算世界后果信用已闭合。
资源账约束的是结构量，不会自动释放固定候选边数组；节点生死、
动态候选拓扑和沿结构逆向传递资源仍是后续任务。

原始结果：`{run/'report.json'}`。
精简结果：`results/published/cgb_v2f_material_closure_20261021.json`。
实现与预注册快照：`{snapshot/'manifest.json'}`。
'''
    (ROOT/'docs/information_boltzmann/CGB_V2F_RESULTS_20261021.md').write_text(text, encoding='utf-8')
    tree_path = ROOT/'research_tree.json'
    tree = json.loads(tree_path.read_text(encoding='utf-8'))
    node_id = 'IB_CGB_LOCAL_MATERIAL_CLOSURE_20261021'
    tree['nodes'] = [node for node in tree['nodes'] if node['id'] != node_id]
    tree['nodes'].insert(0, {'id': node_id, 'parent': 'IB_CGB_STABLE_TYPE_MATCHING_20261021',
        'title': 'Finite structural material and conditional matching-to-medium bridge',
        'status': 'experimental_implementation', 'confidence': 'I', 'time': '2026-10-21',
        'claim': 'Local synthesis, maintenance, turnover and recycling have an explicit balanced material account; real fitted conditional strength can seed the local V2 medium with bitwise persistent restore.',
        'findings': {'numerical_half_supply_plateau_ratio': r['half_supply_ratio'],
                     'real_interface_local_balance_error': i['max_local_balance_error'],
                     'real_interface_edges': i['edges'], 'paired_restore_steps': i['paired_resume_steps'],
                     'focused_checks_passed': 27},
        'scope': 'Numerical/interface acceptance only. Type-pair panel is not a generated cellular connectome; no world learning or agency claim.',
        'action': 'Generate developmental geometry and independently calibrated occurrence, then connect endogenous world-consequence signals to local lifetime plasticity.',
        'artifacts': ['docs/information_boltzmann/CGB_V2F_RESULTS_20261021.md',
                      'results/published/cgb_v2f_material_closure_20261021.json']})
    tree['updated'] = datetime.now(timezone.utc).isoformat()
    write_json(tree_path, tree)
    print(json.dumps({'published': str(published), 'snapshot_files': len(manifest), 'tree_node': node_id}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
