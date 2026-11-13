"""Read-only numerical summary of the registered final water diagnostics."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    run = root/'results/cgb_water_last_attempt_20261109'
    local = json.loads((run/'probe_v2/report.json').read_text())
    coupled = json.loads((run/'coupled_probe/report.json').read_text())
    if local['status'] != 'complete' or coupled['status'] != 'complete':
        raise ValueError('both fixed diagnostics must be complete')
    summary = dict(status='bounded_attempt_complete_candidate_stopped',
        decision='Stop the current local-S/shared-K2 candidate; do not train or sweep',
        optimizer_updates=0, main_life_updates=0,
        source_unchanged=local['source_unchanged'] and coupled['source_unchanged'],
        pulse_parameters_restored=coupled['parameters_restored'],
        source_sha256=local['source_sha256'],
        primary_doc='docs/information_boltzmann/CGB_WATER_FINAL_DIAGNOSIS_20261109.md',
        local=[], coupled=[], numerical_warnings=[], candidate_promoted=False,
        scope='Two factual first-window A_log[0] directions, two OWT segments, present precision; not all heads/owners',
        hypotheses=dict(H1='selected local paths become extremely weak',
            H2='selected responsibilities already absent from shared K2 by128',
            H3='finite whole-body effects survive, most central differences unstable',
            H4='not newly tested: future learning benefit',
            H5='not newly tested: strict-old actual Adam displacement',
            H6='extra retrieval disabled in both diagnostics'),
        review=dict(hypothesis='/root/water_hypothesis_review',
            plan='/root/water_plan_review', conclusion='/root/water_history_review',
            conclusion_status='accepted_with_scope_limits'),
        timings_seconds=dict(local=local['seconds'],coupled=coupled['seconds']),
        new_lean_declarations=0, new_body_coverage=0)
    for record in local['records']:
        first = record['snapshots'][0]
        last = record['snapshots'][-1]
        summary['local'].append(dict(raw_start=record['raw_start'],
            at128=first['directions'], at256=record['snapshots'][1],
            at512=record['snapshots'][3],
            log_alpha_128_to640={k:v-first['cumulative_log_alpha'][k]
                                for k,v in last['cumulative_log_alpha'].items()}))
        for snap in record['snapshots']:
            for direction in snap['directions']:
                if direction['exact_norm'] == 0 and snap.get('teaching'):
                    summary['numerical_warnings'].append(dict(raw_start=record['raw_start'],
                        layer=direction['layer'], tokens=snap['tokens'],
                        warning='Naive squared norm underflow; do not interpret reported norm/Cauchy zero as mathematical zero'))
    stable = total = 0
    for record in coupled['records']:
        agreement=[]
        for layer in (2,13):
            cases=[x for x in record['conditions'] if x['layer']==layer]
            for horizon in ('256','512','640'):
                a,b=[x['finite_difference'][horizon] for x in cases]
                accepted=a*b>0 and abs(a-b)<=.5*max(abs(a),abs(b))
                stable+=accepted;total+=1
                agreement.append(dict(layer=layer,tokens=int(horizon),epsilon01=a,
                    epsilon005=b,passes_registered_agreement=accepted))
        summary['coupled'].append(dict(raw_start=record['raw_start'],
            baseline_repeat_effect=record['baseline_repeat_effect'],finite_difference_agreement=agreement))
    summary['central_difference_agreement']=dict(passed=stable,total=total)
    # Recover registration texts only when their exact pre-result SHA matches.
    doc=(root/summary['primary_doc']).read_text(encoding='utf-8')
    p02=doc.split('\n\n## 实测结果与最后决定')[0]
    start=p02.index('### P-20261109-02')
    end=p02.index('使用安装的树校验',start)
    p01=p02[:start]+p02[end:]
    versions={}
    for name,text,expected in [('p01',p01,local['protocol_sha256']),
                                ('p02',p02,coupled['registration_sha256'])]:
        actual=hashlib.sha256(text.encode('utf-8')).hexdigest()
        versions[name]=dict(expected=expected,reconstructed=actual,matched=actual==expected)
        if actual==expected:
            (run/f'registration_{name}.md').write_text(text,encoding='utf-8')
    summary['registration_snapshots']=versions
    (root/'results/published/cgb_water_last_attempt_20261109.json').write_text(
        json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(dict(decision=summary['decision'],agreement=summary['central_difference_agreement'],
                         registration=versions,source_unchanged=summary['source_unchanged']),indent=2))


if __name__=='__main__':
    main()
