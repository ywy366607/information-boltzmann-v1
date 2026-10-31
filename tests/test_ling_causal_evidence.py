"""Current physiology must not inherit causal certification from a different save."""
import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'ling_audit', Path(__file__).resolve().parents[1] / 'scripts/ib/fly_interface/ling_harness.py')
ling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ling)


def test_causal_report_tracks_checkpoint_and_requires_sham(tmp_path):
    report = {'trained_tokens': 32, 'checkpoint_sha256': 'source', 'tokens': 8,
              'physical_ticks': 100, 'summary': {'phi_zero': {'mean_ce_delta': -.1}},
              'windows': [{'sham_numerical_floor': {'ce_delta_mean': 0.,
                  'hazard_delta_abs_mean': 0., 'muscle_maxabs': 0.}}]}
    for k in ('model_tensors_unchanged', 'checkpoint_unchanged',
              'caregiver_pause_unchanged', 'runtime_unchanged'):
        report[k] = True
    target = tmp_path / 'report.json'
    target.write_text(json.dumps(report), encoding='utf-8')
    (tmp_path / 'native_causal_latest.json').write_text(json.dumps({'report': str(target)}), encoding='utf-8')
    metadata = {'trained': 32, 'checkpoint_sha256': 'source'}
    assert ling._native_causal_evidence(tmp_path, metadata)['status'] == 'matched_saved_checkpoint_acute'
    (tmp_path / 'native_causal_latest.json').write_text(json.dumps({
        'report': str(target), 'decision_conditioned_audit': {'status': 'controls_incomplete'}}), encoding='utf-8')
    evidence = ling._native_causal_evidence(tmp_path, metadata)
    assert evidence['hazard_scope'] == 'all_physical_ticks'
    assert evidence['decision_conditioned_audit']['status'] == 'controls_incomplete'
    assert ling._native_causal_evidence(tmp_path, {'trained': 64})['status'] == 'historical_checkpoint_acute'
    report['windows'][0]['sham_numerical_floor']['hazard_delta_abs_mean'] = .01
    target.write_text(json.dumps(report), encoding='utf-8')
    assert ling._native_causal_evidence(tmp_path, metadata)['status'] == 'audit_controls_incomplete'


def test_missing_report_is_unmeasured(tmp_path):
    assert ling._native_causal_evidence(tmp_path, {})['status'] == 'unmeasured'
