"""Summarize completed continuous lives, without rerunning or selecting cases."""
import argparse
import json
from pathlib import Path


def summarize(arm):
    cases = arm['cases']
    seen, fresh, revisited, old_answer_intrusions = set(), [], [], 0
    for case in cases:
        refs = set(case['references'])
        (revisited if refs & seen else fresh).append(case)
        number = case.get('first_number')
        old_answer_intrusions += int(number in seen and number not in refs)
        seen.update(refs)
    groups = {'all': cases, 'first_half': cases[:len(cases)//2],
              'second_half': cases[len(cases)//2:],
              'first_reference_appearance_in_life': fresh,
              'repeated_reference_in_life': revisited}
    groups.update({f'length_{length}': [c for c in cases if c['length'] == length]
                   for length in sorted({c['length'] for c in cases})})
    scores = {name: dict(correct=sum(c['first_number_exact'] for c in group), cases=len(group))
              for name, group in groups.items()}
    checks = []
    previous = arm['start_seen']
    for case in cases:
        checks.append(case['incoming_seen'] == previous)
        checks.append(case['outgoing_seen']-case['incoming_seen'] ==
                      case['outgoing_events']-case['incoming_events'])
        previous = case['outgoing_seen']
    return dict(scores=scores, events=arm['events'], phase_counts=arm['phase_counts'],
                exact_wrong_previous_answer_intrusions=old_answer_intrusions,
                optimizer_updates=arm['end_updates']-arm['start_updates'],
                contiguous_life=all(checks),
                all_lived_caches_preserved=all(w['live_cache_preserved'] for w in arm['windows']),
                maximum_reexecution_ce_difference=max((abs(w['reexecution_ce_difference'])
                    for w in arm['windows'] if w['reexecution_ce_difference'] is not None), default=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.report.read_text())
    if raw['status'] != 'complete' or set(raw['arms']) != {'held', 'full'}:
        raise ValueError('both continuous arms must finish before paired reporting')
    held, full = (raw['arms'][name]['cases'] for name in ('held', 'full'))
    if len(held) != len(full) or len(held) != len(raw['lengths'])*raw['cases_per_length']:
        raise ValueError('incomplete matched sequence')
    for a, b in zip(held, full):
        if (a['source_index'], a['length'], a['references']) != (b['source_index'], b['length'], b['references']):
            raise ValueError('world cases differ across arms')
    result = dict(status='complete', raw_report=str(args.report),
        protocol='Two irreversible learner lives; held cuts only age>128 native replay derivatives.',
        arms={name: summarize(arm) for name, arm in raw['arms'].items()},
        paired=dict(full_only=sum(b['first_number_exact'] and not a['first_number_exact'] for a,b in zip(held,full)),
                    held_only=sum(a['first_number_exact'] and not b['first_number_exact'] for a,b in zip(held,full)),
                    both_correct=sum(a['first_number_exact'] and b['first_number_exact'] for a,b in zip(held,full)),
                    both_wrong=sum(not a['first_number_exact'] and not b['first_number_exact'] for a,b in zip(held,full))),
        source_sha256=raw['source_sha256'], source_files=raw['source_files'],
        source_unchanged=raw['source_unchanged'], source_files_unchanged=raw['source_files_unchanged'],
        task_seed=raw['task_seed'], official_revision=raw['official_revision'],
        seconds=raw['seconds'], new_lean_coverage=0,
        interpretation='Finite-budget route validation on previously examined tasks; not convergence or a new blind test.')
    if not all(a['contiguous_life'] and a['all_lived_caches_preserved'] for a in result['arms'].values()):
        raise AssertionError('life continuity failed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
