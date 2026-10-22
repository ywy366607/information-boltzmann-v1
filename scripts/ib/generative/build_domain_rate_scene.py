"""Read-only display of three actual complete archives per continued arm."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from scripts.ib.generative.native_learning_stream import atomic_json


def main(run):
    arms=[]
    for label in ('marginal','resolved_reference','no_capture'):
        frames=[]
        for stage in ('start','mid','complete'):
            model,meta=load_exchange(run/(label+'_'+stage+'.npz'))
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
                source_work=model.work_in,heat_out=model.heat_out,port_nodes=model.port_nodes.tolist(),
                morphology=dict(bridges=len(model.pre),branch_nodes=int(np.sum(degree>=3)),seed_component_nodes=len(seen)),
                boundary_contacts=int(model.port_installed.sum()),archive_stage=stage))
        arms.append(dict(id=label,frames=frames))
    first,last=arms[0]['frames'][0],arms[0]['frames'][-1]
    atomic_json(run/'mobile_scene.json',dict(box_size=model.params.box_size,default='marginal',
        default_frame=2,birth_origin_count=24,arms=arms,read_only=True,source=str(run),
        title='同一材料过程的持续生长',
        intro='由实际起点、中点、末态三个完整存档回放；中间没有插值生成的结构。新域沿用原支持、痕迹、驱动、运动和身份匹配；青色编号24起属于这条连续历史中出生的域。',
        reference=str(len(first['coordinates']))+' → '+str(len(last['coordinates']))+'个域 · '+str(int(last['ms']-first['ms']))+'参考毫秒 · 总材料122',
        notes='本段512个新OWT字节，原自然个体从28长到43域，材料账25B+43域+53内部桥+1边界=122。原逐项采样与解析采样具有相同物理生成元，单条随机轨迹不同；关闭捕获副本显式支付12单位参数工作。当前预测仍未胜过固定频率，自然域回收和有用记忆继续检验。页面只显示内部桥，边界接触另计。'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,default=Path('F:/fly_stream_overflow/cgb_domain_rate_r1_20261022'))
    main(p.parse_args().run)
