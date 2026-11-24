"""Pre-register a whole-loop ablation of the measured largest relation group.

Reuses the admitted full learner worker and the already completed exact normal
branch. The target removes all actual shared-variable incidences of the largest
component; the control removes an equal number of incidences, stratified by port,
from the SAME component. Native fallback and re-registration stay enabled.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

from scripts.measure_native_relation_percolation import build, export_lean, read_frames
from scripts.run_native_relation_mode_use import ROOT, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--baseline-work', type=Path, required=True)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    report = ROOT/'results/published/cgb_native_relation_percolation_2401_20261117.json'
    measured = json.loads(report.read_text(encoding='utf8'))
    frame = read_frames(args.task/'work'/measured['frames'][-1]['segment'])[-1]
    graph = build(frame)
    group = graph['largest']
    cut = sorted((v, u) for v, u in graph['dependencies']
                 if v in group and len(graph['supports'][u]) > 1)
    # Fixed randomization is specified before inspecting intervention outcomes.
    generator = random.Random(2401)
    control = []
    for port, count in sorted(Counter(u[1] for _, u in cut).items()):
        candidates = sorted(k for k in graph['dependencies']
                            if k[0] in group and k[1][1] == port)
        control.extend(generator.sample(candidates, count))
    assert Counter(u[1] for _, u in cut) == Counter(u[1] for _, u in control)
    graphs, arms = [], []
    for name, removed in [('baseline', []), ('bridge_cut', cut), ('matched_cut', control)]:
        removed = set(removed)
        changed = dict(frame, rows=[r for r in frame['rows']
                                   if ((r[0], r[3]), (r[1], r[2])) not in removed])
        g = build(changed)
        graphs.append(g)
        arms.append(dict(arm=name, removed=sorted(removed),
            R=len(g['largest'])/g['archive_count'], R_numerator=len(g['largest']),
            archive_count=g['archive_count'], pair_edges=len(g['witnesses']),
            edges_removed=len(graph['witnesses'])-len(g['witnesses']),
            target_members_still_registered=sum(v in group for v in g['vertices']),
            target_members_without_registry=sum(v not in g['vertices'] for v in group),
            components_meeting_group=[c for c in g['components'] if group.intersection(c)]))
        if name == 'bridge_cut':
            assert all(not (set(edge) <= group) for edge in g['witnesses'])
    proof = ROOT/'lean/CGBAxioms/reviews/NativeLargestGroupUseActual2401.lean'
    export_lean(proof, graphs)
    checked = subprocess.run(['lake', 'env', 'lean', str(proof.relative_to(ROOT/'lean/CGBAxioms'))],
        cwd=ROOT/'lean/CGBAxioms', capture_output=True, text=True, encoding='utf8', errors='replace')
    (args.work/'measurement_mapping.lean.log').write_text(checked.stdout+checked.stderr, encoding='utf8')
    if checked.returncode:
        raise RuntimeError('actual largest-group mapping certificate failed')
    old = json.loads((args.baseline_work/'plan.json').read_text(encoding='utf8'))
    baseline = json.loads((args.baseline_work/'baseline/result.json').read_text(encoding='utf8'))
    assert sha(args.task/'body/last.pt.gz') == old['checkpoint']['sha256']
    assert baseline['status'] == 'completed' and baseline['private_final_cycle'] == 2409
    baseline_directory = args.work/'baseline'
    baseline_directory.mkdir()
    shutil.copy2(args.baseline_work/'baseline/result.json', baseline_directory/'result.json')
    # Reuse read-only observation files without another 230-MiB table or model copy.
    write(baseline_directory/'reused_source.json', dict(
        result=str(args.baseline_work/'baseline/result.json'),
        sha256=sha(args.baseline_work/'baseline/result.json'),
        observations=str(args.baseline_work/'baseline/observations')))
    os.link(args.baseline_work/'anchor.pt.gz', args.work/'anchor.pt.gz')
    specification = dict(status='mapped_and_lean_checked', main_cycle=2401,
        identified_group=sorted(group), arms=arms, matches=[],
        matching='same largest component; equal withdrawal count and exact port histogram; random seed 2401',
        intervention='once-only SQL reference withdrawal; every shared incidence in the largest group; native fallback and recovery enabled',
        pending_policy='original saved pending scopes remain; their historical facts are not deleted',
        source_report=dict(path=str(report), sha256=sha(report)),
        proof=dict(path=str(proof), sha256=sha(proof), exit_code=0),
        checkpoint=old['checkpoint'], archive=old['archive'],
        external_events_per_arm=8, maximum_native_polls_per_arm=512,
        baseline_reused_from=str(args.baseline_work/'baseline'),
        baseline_actual_read_coverage=dict(total=len(baseline['read_calls']),
            in_target=sum(tuple(r['relation']) in group for r in baseline['read_calls'])),
        training=old['training'], main_body_will_be_changed=False,
        interpretation='Measure actual delivered facts, not just R; zero registered uses invokes native own-evidence fallback.')
    write(args.work/'plan.json', specification)
    print(json.dumps(dict(status=specification['status'], group=len(group),
        removed=len(cut), read_coverage=specification['baseline_actual_read_coverage'],
        arms=[{k: a[k] for k in ('arm', 'R_numerator', 'pair_edges', 'edges_removed',
                              'target_members_without_registry')} for a in arms])))


if __name__ == '__main__':
    main()
