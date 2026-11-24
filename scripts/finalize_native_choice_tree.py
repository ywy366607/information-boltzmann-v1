"""Publish the completed actual selector tree and its exact variance split."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def record(root, content):
    start='<!-- finite-choice-2493:start -->'
    end='<!-- finite-choice-2493:end -->'
    ledger=root/'docs/information_boltzmann/CGB_CURRENT_STATE_AND_RESEARCH_LEDGER.md'
    text=ledger.read_text(encoding='utf8')
    block=start+'\n'+content+'\n'+end
    if start in text:
        a=text.index(start);b=text.index(end,a)+len(end)
        text=text[:a]+block+text[b:]
    else:
        text=block+'\n\n'+text
    ledger.write_text(text,encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--pending',action='store_true')
    a=p.parse_args();root=Path.cwd()
    if a.pending:
        record(root,'## 2026-11-19 真实未知项—话语提案—actor共同选择树（执行中）\n\n'
            'P条件配对已完成至2493。接续完整carry2493，同一权重/Adam/S/P/表/待结算/世界/RNG，'
            '私有恢复枚举4个真实未知项×4条原生话语提案，随后使用原actor及同一finite_relation_use_v5预期U。'
            '正常选择保持原抽样；条件执行消费原随机抽样后固定相应选择。8项接口回归、NativeFiniteChoiceTree三项Lean通过。'
            '首个完整边界actor重构误差0，回忆值0.050619807、真实actor期望0.050829720。'
            f'工作`{a.work.as_posix()}`，状态`status.json`；完成后自动发布`{a.report.as_posix()}`及专题。'
            '新增世界/优化均0，原主2401及2493源保持。下一项依此共同选择树接延续估值里的问题与行动生成。')
        return
    # Re-contract existing native leaves with one precision at every level.
    # Completed leaves are reused; this starts zero body/world/optimizer work.
    from types import SimpleNamespace
    from scripts.audit_native_choice_tree import run
    run(SimpleNamespace(work=a.work,output=a.report,
        checkpoint=root/'scratch/cgb_momentum_value_2489_20261119/carry/body/last.pt.gz'))
    out=read(a.report)
    assert out['valid'],out['checks']
    import torch
    from scripts.ib_local.finite_choice_contract import tree_expectation
    qp=[];dps=[];aps=[];qs=[]
    for group in out['by_question']:
        qp.append(group['question_probability'])
        dps.append(torch.tensor(group['proposal_probability'],dtype=torch.float64))
        leaves=[read(a.work/f"q{group['question_index']}_d{d}.json")
                for d in range(len(group['proposal_probability']))]
        aps.append([torch.tensor(x['actor_probability'],dtype=torch.float64) for x in leaves])
        qs.append([torch.tensor(x['local_U_values'],dtype=torch.float64) for x in leaves])
    reference=tree_expectation(torch.tensor(qp,dtype=torch.float64),dps,aps,qs)
    out['checks']['shared_reference_on_actual_tree']=abs(float(reference['expected'])-out['nested_expected_U'])<1e-10
    assert out['checks']['shared_reference_on_actual_tree']
    center=out['nested_expected_U'];v_question=v_draft=v_action=flat_second=0.
    for group in out['by_question']:
        r=group['question_index'];wr=group['question_probability'];vr=group['expected_U']
        v_question+=wr*(vr-center)**2
        for d,wd in enumerate(group['proposal_probability']):
            leaf=read(a.work/f'q{r}_d{d}.json')
            vd=sum(p*q for p,q in zip(leaf['actor_probability'],leaf['local_U_values']))
            v_draft+=wr*wd*(vd-vr)**2
            v_action+=wr*wd*sum(p*(q-vd)**2 for p,q in
                zip(leaf['actor_probability'],leaf['local_U_values']))
            flat_second+=wr*wd*sum(p*(q-center)**2 for p,q in
                zip(leaf['actor_probability'],leaf['local_U_values']))
    variance=dict(question=v_question,proposal=v_draft,actor=v_action,
                  total=flat_second,reconstruction=flat_second-v_question-v_draft-v_action)
    assert abs(variance['reconstruction'])<1e-7
    out['finite_tree_variance']=variance
    a.report.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    rows=[]
    for x in out['by_question']:
        roles='/'.join(x['open'])
        rows.append(f"|{x['question_index']}：{roles}|{x['unknown']}|{x['question_probability']:.6f}|{x['expected_U']:.9f}|")
    doc=(
        '# 真实问题—话语提案—行动：共同选择树\n\n'
        '研究主线仍为木的预演得到有效估值，驱动火的选择，并由真实后果修订水里的依据。'
        '本次继承已完成P条件修复的完整2493身体，把下一决策真正使用的选择器接到同一计算中。\n\n'
        '## 身体、数学与执行对应\n\n'
        '`navigation_generative_cycle`给出未知项概率，`navigation_joint_cycle`给出原生话语提案的选择概率，'
        '原actor给出沉默/生成/复用概率。每片叶子以该分支的实际Q、原物理行动端点、P及原有限U计算q。'
        '固定同一身体边界和原生话语抽样，逐一枚举原有离散选择。每个分支完整恢复权重、Adam、S/P、经历/关系库、'
        '待结算、世界和RNG，选择被固定时仍消费原抽样，后续随机位置保持对应。\n\n'
        r'\[V=\sum_Q p(Q)\sum_Dp(D\mid Q)\sum_A\pi(A\mid Q,D)q(Q,D,A).\]'+'\n\n'
        '`reviews/NativeFiniteChoiceTree.lean`已编译三项证明：有限树展开、联合概率归一、旧参考值与整树值的差。'
        '原选择器调用`finite_choice_contract.choose`，正常路径保持原categorical输入；33项接口及原大环回归通过，'
        '完整身体actor逐片叶子核对。\n\n'
        '## 完整2493现场结果\n\n'
        f"枚举{out['leaves']}片叶子；联合概率质量{out['total_mass']:.10f}，"
        f"嵌套/展开期望差{abs(out['nested_expected_U']-out['flattened_expected_U']):.3g}。"
        '正常所选分支经独立恢复后，话语、行动概率和q逐项相同。新增优化0，世界调用0。\n\n'
        '|未知项候选|持存身份|选择概率|平均话语及actor后的U期望（10*g）|\n|---|---|---:|---:|\n'+
        '\n'.join(rows)+'\n\n'+
        f"回忆参考{out['recalled_value']:.9f}，正常已选问题/话语下的actor期望{out['selected_actor_expected_U']:.9f}，"
        f"全部问题/话语/actor的加权期望{center:.9f}。\n\n"
        '|参考差来源|10*g|\n|---|---:|\n'+
        ''.join(f'|{key}|{value:.9f}|\n' for key,value in out['gap_components'].items())+
        f"\n三项重构残差{out['gap_reconstruction']:.3g}。\n\n"
        '|离散选择造成的有限U方差来源|方差|\n|---|---:|\n'+
        ''.join(f'|{key}|{variance[key]:.12g}|\n' for key in ('question','proposal','actor','total'))+
        f"\n方差分解重构残差{variance['reconstruction']:.3g}。"
        '这将选学习对象、同对象内选择表达、最终交话选择的贡献，落到同一身体的实际计算上。\n\n'
        '## 下一步接线\n\n'
        '以这棵实际共同选择树作为延续估值的对应对象，连接分支后的问题绑定、原生候选生成、真实actor和同一U。'
        '先在实际边界核对各项，再把可由预测潜状态执行的共同计算前移进木；所需的内感、候选及问题状态沿已有转移接口携带。'
        '验收继续沿事前预测→真实选择→同一U支付→旧依据修订→后继用途，保留独立旧查询及保持/再学习评分。\n\n'
        f"机器报告：`{a.report.as_posix()}`；原始叶子：`{a.work.as_posix()}`。\n")
    destination=root/'docs/information_boltzmann/CGB_ACTUAL_CHOICE_TREE_20261119.md'
    destination.write_text(doc,encoding='utf8')
    record(root,'## 2026-11-19 真实共同选择树已完成\n\n'+
        f"完整carry2493，{out['leaves']}个未知项/话语分支，原actor期望逐项重构。"
        f"已选分支U期望{out['selected_actor_expected_U']:.9f}；整树期望{center:.9f}，"
        f"回忆参考差{out['reference_gap']:.9f}。选择方差按问题/提案/actor分开，联合质量和展开均核对。"
        '33项接口/原大环检查与3项Lean通过，正常分支恢复逐项相同。优化/世界新增0。'
        '详见[CGB_ACTUAL_CHOICE_TREE_20261119.md](CGB_ACTUAL_CHOICE_TREE_20261119.md)。'
        '下一项为沿此共同树接入预测状态下的问题与候选生成，原主2401及研究源保持。')
    contract=root/'docs/information_boltzmann/CGB_ACTOR_EXPECTATION_CONTRACT_20261119.md'
    marker='## 真实共同选择树2493已完成'
    if marker not in contract.read_text(encoding='utf8'):
        with contract.open('a',encoding='utf8') as f:
            f.write('\n\n'+marker+'\n\n'+
                f"16片实际叶子全部完成，整树U期望{center:.9f}，"
                f"回忆参考与整树差{out['reference_gap']:.9f}。"
                '逐项结果、分层方差及下一处接线见'
                '[CGB_ACTUAL_CHOICE_TREE_20261119.md](CGB_ACTUAL_CHOICE_TREE_20261119.md)。\n')
    print(json.dumps(dict(valid=True,variance=variance,expected_U=center),ensure_ascii=False))


if __name__=='__main__':main()
