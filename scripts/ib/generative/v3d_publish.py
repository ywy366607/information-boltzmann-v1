"""Publish the completed real-data matching study and its generator checks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.ib.generative.v3d_stable_matching import prepare,linear_fit,write_json,digest


def main(args):
    path=args.run/'refined_report.json'
    raw=json.loads(path.read_text())
    deployment=json.loads((args.run/'generator_verification.json').read_text())
    if raw['status']!='completed' or not all(v['passed'] for v in deployment.values()):
        raise ValueError('Publish only completed fits with verified deployment')
    data=prepare(args.work)
    if data['files']!=raw['provenance']['data_sha256']:
        raise ValueError('Publication data provenance mismatch')
    _,null,_=linear_fit(data['n'],data['pre'],data['post'],data['features'][:,:0],data['target'],data['splits']['train'])
    test=data['splits']['test']
    legacy=np.load(args.audit/'legacy_null_prediction.npy')
    new_mse=float(np.mean((null[test]-data['target'][test])**2))
    old_mse=float(np.mean((legacy[test]-data['target'][test])**2))
    families={}
    for name,row in raw['selected'].items():
        fits=row['fits']
        families[name]={
            'rank':fits[0]['rank'],'parameters':fits[0]['parameters'],'lambda':row['lambda'],
            'test_mean':row['test_mean'],'test_std':row['test_std'],
            'gain_over_geometry':row['test_mean']-raw['baselines_test_explained']['geometry'],
            'gain_pair_bootstrap_95':row['gain_pair_bootstrap_95'],
            'test_by_seed':row['test_by_seed'],'numerically_settled':row['numerically_settled'],
            'edge_of_searched_grid':row['edge_of_searched_grid'],
            'largest_extra_update_relative_improvement':max(abs(v['relative_objective_improvement']) for v in row['extension_check']),
            'largest_selected_gradient_l2':max(v['gradient_l2'] for v in fits),
            'cold_start_test':row['cold_test'],'generator':deployment[name]}
    records=[]
    for directory in (args.run/'fits',args.run/'rank_extension'):
        for p in directory.glob('*.json'):
            record=json.loads(p.read_text())
            if 'adam_updates' in record:records.append(record)
    report={'status':'completed','date':'2026-10-21','study':'CGB V3d stable type matching',
        'source_report':str(path),'source_sha256':digest(path),
        'target':'Right-source type-pair log synapses per possible cell pair, among observed connected type pairs',
        'metric':'1 - held-out squared error / degree-only held-out squared error',
        'provenance':raw['provenance'],'baselines':raw['baselines_test_explained'],
        'families':families,'predictions':raw['predictions'],
        'baseline_denominator_audit':{'old_1500_test_mse':old_mse,'converged_test_mse':new_mse,
            'relative_difference':(old_mse-new_mse)/new_mse},
        'execution':{'fits':len(records),'adam_updates':sum(r['adam_updates'] for r in records),
            'lbfgs_iterations':sum(r['lbfgs_iterations'] for r in records),
            'sum_fit_seconds':sum(r['seconds'] for r in records),
            'peak_allocated_mib':max(r['peak_allocated_mib'] for r in records)},
        'rank_regularization_scaling':{'same_interaction_penalty':'k^(-4/3)',
            'equivalent_lambda':'k^(4/3)','factor_256_vs_64':4**(4/3),'scope':'factorization gauge, not physical energy'},
        'checks':{'numerical_unit_tests':6,'data_audit':str(args.audit/'data_audit.json'),
            'deployment':str(args.run/'generator_verification.json')},
        'next':'Use the stable directed matching rule together with spatial encounter in a developmental generator; retain local medium plasticity for lifetime adaptation'}
    artifact=ROOT/'results/published/cgb_v3d_stable_matching_20261021.json'
    write_json(artifact,report)
    doc=ROOT/'docs/information_boltzmann/CGB_V3D_RESULTS_20261021.md'
    text=doc.read_text(encoding='utf-8')
    text=text.replace('状态：多种子拟合运行中，最终表格待全部验证选择锁定后填写。','状态：已完成真实连接组多种子拟合、秩尺度细网格、冷启动/额外更新检查与 CPU 生成器重建。')
    text=text.replace('数值接口 5 项检查通过','数值接口 6 项检查通过')
    table=['全部模型按平均验证误差选正则后，统一在原 E1 历史测试对上评分。',
        '', '| 模型 | 测试解释比例（种子均值 ± 标准差） | 超过几何 | 正则 |',
        '|---|---:|---:|---:|',
        f"| 几何 4 + 2 µm | {100*report['baselines']['geometry']:.2f}% | — | — |"]
    labels={'directed64':'有方向码 64 + 几何','symmetric64':'对称码 64 + 几何','directed256':'有方向码 256 + 几何'}
    for name in ('symmetric64','directed64','directed256'):
        r=families[name]
        table.append(f"| {labels[name]} | {100*r['test_mean']:.2f}% ± {100*r['test_std']:.2f} 点 | {100*r['gain_over_geometry']:+.2f} 点 | {r['lambda']:g} |")
    table.append(f"| 左侧复本参照 | {100*report['baselines']['replicate']:.2f}% | — | — |")
    table.extend(['', '### 稳定性与预言', ''])
    for name in ('directed64','symmetric64','directed256'):
        r=families[name]
        table.append(f"- {labels[name]}：额外更新的最大相对目标改善 {r['largest_extra_update_relative_improvement']:.3g}；"
            f"收敛阈值检查{'通过' if r['numerically_settled'] else '未通过'}；选中正则{'仍在边缘' if r['edge_of_searched_grid'] else '位于网格内部'}。"
            f"冷启动测试为 {100*r['cold_start_test']['test_explained']:.2f}%。")
    table.append('')
    for prediction,passed in report['predictions'].items():
        table.append(f"- `{prediction}`：{'成立' if passed else '未成立'}。")
    table.extend(['', '### 可执行产物', ''])
    for name,r in families.items():
        g=r['generator']
        table.append(f"- {labels[name]}：`{g['generator']}`，{g['bytes']/2**20:.2f} MiB；"
            f"在 {g['verified_real_pairs']} 个实际类型对上重建，最大误差 {g['max_cpu_reconstruction_error']:.3g}。"
            f"相互作用的谱参与秩为 {g['interaction_power_effective_rank']:.2f}。")
    e=report['execution']
    table.extend(['', f"总计 {e['fits']} 次拟合、{e['adam_updates']} 个 Adam 更新；拟合用时合计 {e['sum_fit_seconds']/60:.1f} 分钟，"
        f"峰值已分配显存 {e['peak_allocated_mib']:.1f} MiB。测试分母相对旧实现只差 {(old_mse-new_mse)/new_mse:.3g}。",
        '', f"汇总结果：`{artifact}`。历史测试集与细胞类型共享依赖按原口径报告；bootstrap 是类型对的条件重采样。"])
    text=text.replace('待运行完成。所有超参数锁定前不计算测试分数。','\n'.join(table))
    doc.write_text(text,encoding='utf-8')
    # Add the new evidence to the existing research ledger without changing
    # earlier results or interpreting this as language/lifetime capability.
    tree_path=ROOT/'research_tree.json'
    tree=json.loads(tree_path.read_text(encoding='utf-8'))
    node_id='IB_CGB_STABLE_TYPE_MATCHING_20261021'
    node={'id':node_id,'parent':None,'title':'Stable directed matching codes for a developmental brain generator',
        'status':'experimental_result','confidence':'D','time':'2026-10-21',
        'claim':'Spatial encounter and directed type matching jointly predict static connection strength; multi-seed convergence and CPU generator reconstruction tested on the unchanged real E1 pair split.',
        'findings':{name:{key:r[key] for key in ('test_mean','test_std','lambda','numerically_settled')} for name,r in families.items()},
        'action':report['next'],'artifacts':[str(artifact.relative_to(ROOT)),str(doc.relative_to(ROOT))],
        'scope':'Static conditional strength generation on known types; developmental geometry and lifetime active learning are separate next tests'}
    tree['nodes']=[n for n in tree['nodes'] if n['id']!=node_id]
    tree['nodes'].insert(0,node);tree['updated']=datetime.now(timezone.utc).isoformat()
    write_json(tree_path,tree)
    write_json(args.run/'progress.json',{'status':'published_and_verified','artifact':str(artifact),
        'report':str(path),'generators':{name:r['generator']['generator'] for name,r in families.items()}})
    print(json.dumps({'artifact':str(artifact),'families':{n:{k:r[k] for k in ('test_mean','test_std','lambda')} for n,r in families.items()},'execution':e},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--work',type=Path,default=Path('F:/fly_stream_overflow/cgb_v1'))
    parser.add_argument('--audit',type=Path,required=True)
    main(parser.parse_args())
