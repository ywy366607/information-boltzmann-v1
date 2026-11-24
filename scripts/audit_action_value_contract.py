"""Reconstruct settled U/D and the logged world/native/wood/actor contract."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cycle', type=int, default=2575)
    a = p.parse_args()
    out = dict(equation='S_s(Q,R_after)-S_0(Q,R_before) = U + D',
               D='existing finite_U.representation_shift; scoring-port change', arms={})
    for name in ('off', 'relation'):
        path = a.work/name/'events.jsonl'
        rows = [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
        rows = [r for r in rows if r['autonomous_event']['kind'] == 'external'
                and r['causal_use']['finite_U']['committed']]
        payments = []
        for r in rows:
            finite = r['causal_use']['finite_U']
            computed = mean(c['after']-c['before'] for c in r['causal_use']['comparisons'])
            assert abs(computed-r['metal_g']) < 1e-7
            u, d = r['metal_g'], finite['representation_shift']
            payments.append(dict(cycle=r['cycle'], U=u, D=d, total=u+d,
                                 changes_sign=u*(u+d) < 0))
        out['arms'][name] = dict(events=str(path), sha256=digest(path), payments=payments,
            count=len(payments), mean_U=mean(r['U'] for r in payments),
            mean_D=mean(r['D'] for r in payments), mean_total=mean(r['total'] for r in payments),
            sign_changes=sum(r['changes_sign'] for r in payments))
        if name != 'off':
            continue
        r = next(r for r in rows if r['cycle'] == a.cycle)
        e = r['causal_use']['epistemic_value']
        option = ('silence', 'generate', 'reuse').index(r['choice'])
        logits = [h+v for h,v in zip(r['habit_logits'], r['imagination_offset'])]
        exp = [math.exp(v-max(logits)) for v in logits]
        pi = [v/sum(exp) for v in exp]
        assert max(abs(x-y) for x,y in zip(pi, r['choice_probability'])) < 1e-6
        assert max(abs(10*q-v) for q,v in zip(e['contextual_total'], r['imagined_value'])) < 2e-6
        assert max(abs(q+e['gamma']*v-total) for q,v,total in zip(
            e['local_expected'], e['contextual_future'], e['contextual_total'])) < 1e-6
        native = e['root_native_correction']
        out['event_contract'] = dict(cycle=r['cycle'], selected=r['choice'], option=option,
            action=dict(utterance_hex=r['utterance_hex'], bytes=len(bytes.fromhex(r['utterance_hex'])),
                        unknown=r['shared_unknown']['identity'], relation=r['causal_use']['relation'],
                        target_event=r['causal_program']['committed_episode_identity'],
                        queries=[c['query'] for c in r['causal_use']['comparisons']]),
            timing={k:r['causal_use'][k] for k in ('prediction_event','prediction_update','settlement_event','settlement_update')},
            world_label=dict(target=10*r['metal_g'], units='10*U at this executed event',
                             kind='arrived-world local payment', future_bootstrap=False),
            native_teacher=dict(kind='conditional next-state question/proposal/actor expectation of local 10*U',
                                actor_selection_uses='local q plus gamma*terminal; returned value averages local q',
                                same_as_current_world_label=False, rows=native['contributions']),
            wood=dict(local=e['local_expected'], future=e['contextual_future'], gamma=e['gamma'],
                      total=e['contextual_total'], units='10*U + gamma*conditional-next-local-10*U',
                      native_current_mode=native['mode']),
            fire=dict(critic=r['imagined_value'], units='wood total multiplied by 10',
                      habit=r['habit_logits'], offset=r['imagination_offset'], precision=r['precision'],
                      probability=r['choice_probability'], reconstructed_probability=pi),
            observed_world=r['world'],
            pair_status='one executed action; candidate counterfactual learning gains require paired continuations')
    out['valid'] = all(r['count'] == 16 for r in out['arms'].values())
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({n:{k:r[k] for k in ('count','mean_U','mean_D','mean_total','sign_changes')}
                      for n,r in out['arms'].items()}))


if __name__ == '__main__':
    main()
