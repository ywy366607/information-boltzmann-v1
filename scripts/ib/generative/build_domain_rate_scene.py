"""Read-only display of three actual complete archives per continued arm."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from scripts.ib.generative.native_learning_stream import atomic_json


def main(run):
    report=json.loads((run/'report.json').read_text(encoding='utf-8'))
    shared_path=run/'cohort_analysis.json'
    shared=json.loads(shared_path.read_text(encoding='utf-8')) if shared_path.exists() else None
    primary=next(label for label,item in report['arms'].items() if item.get('primary'))
    arms=[]
    origin_count=None
    for label in report['arms']:
        frames=[]
        source_work=source_heat=None
        history_path=run/(label+'_windows.json')
        history=json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
        for stage in ('start','mid','complete'):
            model,meta=load_exchange(run/(label+'_'+stage+'.npz'))
            if stage=='start':source_work,source_heat=model.work_in,model.heat_out
            origin_count=meta['growth_observer']['initial_n']
            degree=model.degrees()
            ids=meta['growth_observer']['ids']
            seen={ids.index(0)} if 0 in ids else set()
            pending=list(seen)
            adjacency=[[] for _ in range(model.n)]
            for i,j in zip(model.pre,model.post):
                adjacency[i].append(int(j));adjacency[j].append(int(i))
            while pending:
                for j in adjacency[pending.pop()]:
                    if j not in seen:seen.add(j);pending.append(j)
            frames.append(dict(ms=model.t,coordinates=model.coordinates.tolist(),
                node_ids=ids,pre=model.pre.tolist(),post=model.post.tolist(),
                trace=model.trace.tolist(),bound=model.bound.tolist(),material=int(model.material.sum()),
                source_work=model.work_in-source_work,heat_out=model.heat_out-source_heat,port_nodes=model.port_nodes.tolist(),
                morphology=dict(bridges=len(model.pre),branch_nodes=int(np.sum(degree>=3)),seed_component_nodes=len(seen)),
                boundary_contacts=int(model.port_installed.sum()),archive_stage=stage))
            if 'nll_by_phase' in report['arms'][label]:
                cohort=meta['stage_reference']['cohort']
                current=next((row for row in history if row['time_ms']==model.t),None)
                phase=current['phase'] if current else None
                frames[-1]['memory']=dict(total_carriers=len(cohort),
                    source_trace_carriers=sum(x['trace'] for x in cohort.values()),
                    retained=current['original_carriers_retained'] if current else len(cohort),
                    retained_trace=current['original_trace_carriers_retained'] if current else sum(x['trace'] for x in cohort.values()),
                    retained_with_trace_now=current['original_trace_carriers_current_trace'] if current else sum(x['trace'] for x in cohort.values()),
                    rebuilt=current['rebuilt_old_positions'] if current else 0,
                    phase=phase,assessment=report['arms'][label]['nll_by_phase'].get(phase),
                    samples=report['arms'][label].get('phase_sample_counts',{}).get(phase,128) if phase else 0)
                if report.get('protocol_kind')=='readout_observation' and stage=='complete':
                    frames[-1]['memory']['frozen_readout_validation']=report['arms'][label]['frozen_external_observer_validation']
                if shared:
                    fixed=next(row['fate'] for row in shared['arms'][label]['windows'] if row['time_ms']==model.t)
                    frames[-1]['memory'].update(total_carriers=fixed['source_carriers'],
                        source_trace_carriers=fixed['groups']['traced']['initial'],retained=fixed['retained'],
                        retained_trace=fixed['groups']['traced']['retained'],
                        retained_with_trace_now=fixed['groups']['traced']['current_trace'],rebuilt=fixed['rebuilt_old_positions'])
        arms.append(dict(id=label,frames=frames))
    selected=next(arm for arm in arms if arm['id']==primary)
    first,last=selected['frames'][0],selected['frames'][-1]
    notes='仅显示实际起点、中点和末态；原学习、支持与空间过程保持活动。图中为内部桥，边界接触另计；结构数量不代表预测学习已通过。'
    title='同一材料过程的持续生长'
    if 'source_disconnected' in report['arms']:
        off=report['arms']['source_disconnected']
        title='持续驱动与材料库存对照'
        notes=('只有真实文本支接收512个新OWT字节；恒源与断源均0新文字，只有物理时间继续。'
               '断源支末态'+str(off['nodes'])+'域、源功'+format(off['source_work'],'.2f')+
               '、净热流出'+format(off['heat_out'],'.2f')+'（负值为热浴输入）。'
               '三支实际源功不同，当前不能据域数判断自维持或语义收益。页面显示内部桥，边界另计。')
    if 'symmetric_dt2' in report['arms']:
        title='同一物理过程 · 整核交错推进'
        notes=('三支均经历同一256新OWT字节、1024参考毫秒；分别为旧顺序2ms、新对称顺序2ms与1ms。'
               '主线预先指定为新顺序2ms，未按域数或NLL挑选。原痕迹、支持、身份和空间反应全部继续。'
               '每支中点后的128字节已逐位恢复；这是数值执行检查，不是单轨迹收敛或语言能力结论。'
               '页面显示实际内部桥，边界另计；图中只有实际存档，无插值造型。')
    if 'nll_by_phase' in report['arms'][primary]:
        title='痕迹、支持与真实载体周转'
        notes=('自然支先读128新OWT字节，再回访128个确实见过的字节，学习持续，4ms/字节。'
               '保留痕迹/条件化去痕迹副本用原允许反应匹配q/P/D；另有付费关闭捕获耦合的副本。'
               '它们是诊断对照，未经修改的自然支始终是主线，不挑域数或NLL的赢家。'
               '原载体存活、痕迹当前值和旧位置重建分别计数；当前未证明语义记忆或预测收益。')
        notes+=' 各条件按同一未干预父档的载体群计数；当前r值也可能经历重写，不等于原内容连续留存。'
    if report.get('protocol_kind')=='readout_observation':
        title='原生状态、观测概率与真实生成'
        notes=('自然支先接收1024新训练字节，仅用其输入前状态拟合8种外部观察器；'
               '随后冻结观察器，接收独立split的1024新验证字节，身体原生学习一直继续。'
               'raw端口NLL保留；校准分数只属于外部仪器，不是已安装的脑内预测回路。'
               '本轮未交付超过源/频率参照的端口预测收益；下一步从原电荷连续性推导体内反向流。'
               '只回放真实三份材料状态，无插值造型或按分数挑选的个体。')
    atomic_json(run/'mobile_scene.json',dict(box_size=model.params.box_size,default=primary,
        default_frame=2,birth_origin_count=origin_count,arms=arms,read_only=True,source=str(run),
        title=title,work_label='本段外界源功',heat_label='本段净热流出 Q',
        intro='由实际起点、中点、末态三个完整存档回放；中间没有插值生成的结构。新域沿用原支持、痕迹、驱动、运动和身份匹配；青色编号表示这条连续历史中出生的域。',
        reference=str(len(first['coordinates']))+' → '+str(len(last['coordinates']))+'个域 · '+str(int(last['ms']-first['ms']))+'参考毫秒 · 总材料'+str(int(model.initial_material)),
        notes=notes))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,default=Path('F:/fly_stream_overflow/cgb_domain_rate_r1_20261022'))
    main(p.parse_args().run)
