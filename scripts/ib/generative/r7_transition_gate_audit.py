"""Read-only R7 transition-evidence audit and conditional exposure-law checks.

No fitting, simulation of capability, biological parameter assignment, or
execution of the author's Matlab code. Proposed cross-table links are unverified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import itertools
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.generative.r7_identifiability_audit import require, sha256


def worksheet(path: Path) -> list:
    """Read the documented first sheet without importing training dependencies."""
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(t.text or '' for t in item.findall('.//s:t', ns))
                       for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        sheet = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        rows = []
        for item in sheet.findall('s:sheetData/s:row', ns):
            row = {}
            for cell in item:
                key = re.match('[A-Z]+', cell.attrib['r']).group()
                value = cell.find('s:v', ns)
                if cell.attrib.get('t') == 'inlineStr':
                    row[key] = ''.join(t.text or '' for t in cell.findall('.//s:t', ns))
                elif value is not None:
                    row[key] = (strings[int(value.text)] if cell.attrib.get('t') == 's'
                                else float(value.text))
            if row:
                rows.append((int(item.attrib['r']), row))
        return rows


def evidence_audit(base: Path) -> dict:
    source_root = base / 'cgb_inverse_sources_20261022/Filo2'
    source = json.loads((source_root / 'download_manifest.json').read_text())
    obs_path = base / 'cgb_r7_observation_audit_20261022/observations.json'
    observations = json.loads(obs_path.read_text())
    for item in source['files']:
        require(sha256(source_root / item['path']) == item['sha256'],
                'Published source changed: ' + item['path'])
    require(source['commit'] == observations['source']['commit'],
            'Observation source revision differs')
    books = observations['workbooks']
    groups = {}
    control_links = []
    for book in books:
        if book['kind'] != 'filopodium':
            continue
        genotype = book['genotype']
        group = groups.setdefault(genotype, dict(
            workbooks=0, admitted_workbooks=0, rows_with_marker_schema=0,
            marker_positive=0, partial_marker=0, incident_marker_positive=0,
            partial_rows=[], missing_marker_schema_files=[],
            percentage_max_error=0.0, same_numeric_id_link_failures=0))
        group['workbooks'] += 1
        group['admitted_workbooks'] += int(book['fit_admitted'])
        raw_path = source_root / book['file']
        require(sha256(raw_path) == book['sha256'], 'Workbook/cache mismatch')
        sheet = worksheet(raw_path)
        header = sheet[0][1]
        if header.get('C') != 'Bulbous' or header.get('D') != 'Bulbous (%)':
            group['missing_marker_schema_files'].append(book['file'])
            continue
        raw = {str(row['B']): row for _, row in sheet[1:] if 'B' in row}
        marked = []
        for record in book['records']:
            row = raw[record['id']]
            count = row['C']
            length = record['last_frame'] - record['first_frame'] + 1
            require(count == int(count) and 0 <= count <= length,
                    'Marker count outside parent interval')
            require(count == record['bulbous_frame_count'],
                    'Raw marker differs from cached marker')
            error = abs(row['D'] - count / length)
            require(error < 1e-6, 'Marker percentage does not match duration')
            group['percentage_max_error'] = max(group['percentage_max_error'], error)
            group['rows_with_marker_schema'] += 1
            group['marker_positive'] += int(count > 0)
            group['incident_marker_positive'] += int(count > 0 and record['first_frame'] > 0)
            if 0 < count < length:
                group['partial_marker'] += 1
                group['partial_rows'].append(dict(
                    file=book['file'], parent_id=record['id'],
                    first=record['first_frame'], last=record['last_frame'],
                    marked_frames=int(count), total_frames=length,
                    minimum_binary_marker_changes=1))
            if count > 0:
                marked.append(record)
        if genotype != 'Control':
            continue
        number = int(re.search(r'GC(\d+)', book['file']).group(1))
        bulb_book = next(w for w in books if w['genotype'] == genotype
                         and w['kind'] == 'bulb'
                         and re.search(r'CG(\d+)', w['file']).group(1) == str(number))
        require(bulb_book['fit_admitted'], 'Control bulb observation conflict')
        for bulb in bulb_book['records']:
            duration = bulb['last_frame'] - bulb['first_frame'] + 1
            # Same file number is only a provisional unit correspondence.
            candidates = [r for r in marked
                          if r['bulbous_frame_count'] == duration
                          and r['first_frame'] <= bulb['first_frame']
                          and bulb['last_frame'] <= r['last_frame']]
            same_id = next((r for r in book['records'] if r['id'] == bulb['id']), None)
            valid_same_id = same_id is not None and same_id in candidates
            group['same_numeric_id_link_failures'] += int(not valid_same_id)
            control_links.append(dict(
                filopodium_file=book['file'], bulb_file=bulb_book['file'],
                bulb_id=bulb['id'], bulb_first=bulb['first_frame'],
                bulb_last=bulb['last_frame'], candidate_parent_ids=[r['id'] for r in candidates],
                unique_candidate=len(candidates) == 1,
                same_numeric_id_is_candidate=valid_same_id,
                conditional_suffix_violations=[r['id'] for r in candidates
                                              if bulb['last_frame'] != r['last_frame']],
                linkage_confirmed=False))
    for genotype, group in groups.items():
        if not group['rows_with_marker_schema']:
            for name in ('marker_positive', 'partial_marker', 'incident_marker_positive',
                         'percentage_max_error'):
                group[name] = None
        if genotype != 'Control':
            group['same_numeric_id_link_failures'] = None
    return dict(
        source_commit=source['commit'], source_files_hash_verified=len(source['files']),
        observations_sha256=sha256(obs_path), groups=groups,
        control_candidate_links=control_links,
        control_candidate_links_confirmed=False,
        candidate_link_scope='Same-numbered cones plus duration/containment only; '
            'no source crosswalk confirms a shared unit or parent ID namespace',
        raw_framewise_marker_sequences_present_in_workbook_schema=False,
        source_hashes={item['path']: item['sha256'] for item in source['files']
                       if item['path'] in ('Description.docx', 'Data/DataImport.m',
                                         'Data/DataImportBulbous.m',
                                         'MechanisticModel/MechanisticModel.m')})


def exposure_law_checks() -> dict:
    """Check Q(r)=E[1-(1-r)^T] via an independent first-hit calculation."""
    checks = 0
    finite_horizon_checks = 0
    for length in range(1, 10):
        for numerator in range(9):
            r = Fraction(numerator, 8)
            no_hit, hits = Fraction(1), Fraction(0)
            for _ in range(length):
                hits += no_hit * r
                no_hit *= 1 - r
            require(hits == 1 - (1-r)**length, 'First-hit identity failed')
            checks += 1
    # Arbitrary bounded integer lifetime laws, no geometric assumption needed.
    for weights in itertools.product(range(3), repeat=4):
        if sum(weights) == 0:
            continue
        law = [Fraction(w, sum(weights)) for w in weights]
        survival = [sum(law[n:]) for n in range(4)]
        for numerator in range(9):
            r = Fraction(numerator, 8)
            by_lifetime = sum(mass * (1-(1-r)**(t+1))
                              for t, mass in enumerate(law))
            by_survival = r * sum((1-r)**n * s for n, s in enumerate(survival))
            require(by_lifetime == by_survival, 'Survival transfer identity failed')
            checks += 1
            for horizon in range(1, 5):
                partial = r * sum((1-r)**n * survival[n] for n in range(horizon))
                remaining = (1-r)**horizon * (survival[horizon] if horizon < 4 else 0)
                require(partial <= by_lifetime <= partial + remaining,
                        'Finite-horizon gate bound failed')
                finite_horizon_checks += 1
    # For geometric persistence, verify weak/strong relative response to extension.
    sensitivity_cases = 0
    for survival1, survival2 in itertools.combinations(range(8), 2):
        s1, s2 = Fraction(survival1, 8), Fraction(survival2, 8)
        for weak, strong in itertools.combinations(range(1, 9), 2):
            rw, rs = Fraction(weak, 8), Fraction(strong, 8)
            q = lambda r, s: r / (1 - (1-r)*s)
            require(q(rw, s2)/q(rw, s1) >= q(rs, s2)/q(rs, s1),
                    'Geometric relative-gain inequality failed')
            sensitivity_cases += 1
    # A partial binary duration forces a change, but does not locate its time.
    binary_cases = 0
    ambiguous_timing = 0
    for length in range(2, 9):
        by_count = Counter()
        for sequence in itertools.product((0, 1), repeat=length):
            count = sum(sequence)
            if 0 < count < length:
                require(any(a != b for a, b in zip(sequence, sequence[1:])),
                        'Partial occupancy without a transition')
                by_count[count] += 1
                binary_cases += 1
        ambiguous_timing += sum(value > 1 for value in by_count.values())
    return dict(first_hit_and_survival_transfer_checks=checks,
                finite_horizon_gate_bound_checks=finite_horizon_checks,
                geometric_relative_gain_checks=sensitivity_cases,
                partial_binary_duration_checks=binary_cases,
                ambiguous_duration_classes=ambiguous_timing,
                status='Exact arithmetic checks, paper results not Lean formalization')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Refusing to overwrite a completed audit')
    result = dict(schema_version=1, stage='inverse_development_transition_and_gate_audit',
                  evidence=evidence_audit(args.base), algebra=exposure_law_checks(),
                  script_sha256=sha256(Path(__file__)),
                  dependency_sha256={str(path.relative_to(ROOT)): sha256(path) for path in (
                      ROOT/'scripts/ib/generative/r7_identifiability_audit.py',)},
                  conclusions=dict(new_fit_or_training=False,
                                   author_code_executed=False,
                                   single_parameter_gene_intervention_identified=False,
                                   gate_law_is_conditional_H_not_core_axiom=True,
                                   independent_mechanism_review_completed=False,
                                   biological_validation_complete=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                        encoding='utf-8')
    summary = {k: {name: value for name, value in v.items()
                   if name not in ('partial_rows', 'missing_marker_schema_files')}
               for k, v in result['evidence']['groups'].items()}
    print(json.dumps(dict(groups=summary, algebra=result['algebra'],
                          conditional_control_links=result['evidence']['control_candidate_links']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
