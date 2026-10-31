"""Ling aunt's read-only audit and optional parenting-plan update.

This is the polling entry point for the mother-world harness.  It keeps the
environmental lesson and the fly's internal health checks separate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# The package __init__ imports Torch and the whole brain. Caregiver polling only
# needs these standard-library tools and must stay cheap beside a 4GB learner.
_spec = importlib.util.spec_from_file_location(
    '_ling_parenting_tools', ROOT / 'information_boltzmann/core/fly_parenting_harness.py')
_parenting = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parenting)
ParentingHarness = _parenting.ParentingHarness
_atomic_json = _parenting._atomic_json
conversation_health = _parenting.conversation_health
stage_readiness = _parenting.stage_readiness


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _native_causal_evidence(run, metadata):
    """Display only completed controlled audits, with saved-checkpoint provenance."""
    pointer = _read_json(Path(run) / 'native_causal_latest.json') or {}
    report = _read_json(pointer.get('report', '')) or {}
    if not report:
        return {'status': 'unmeasured'}
    guards = ('model_tensors_unchanged', 'checkpoint_unchanged',
              'caregiver_pause_unchanged', 'runtime_unchanged')
    windows = report.get('windows') or []
    sham_ok = bool(windows) and all(
        all(w.get('sham_numerical_floor', {}).get(k) == 0 for k in
            ('ce_delta_mean', 'hazard_delta_abs_mean', 'muscle_maxabs')) for w in windows)
    if not all(report.get(k) is True for k in guards) or not sham_ok:
        return {'status': 'audit_controls_incomplete', 'report': pointer.get('report')}
    matched = (metadata.get('checkpoint_sha256') == report.get('checkpoint_sha256')
               and metadata.get('trained') == report.get('trained_tokens'))
    return {'status': 'matched_saved_checkpoint_acute' if matched else 'historical_checkpoint_acute',
            'trained': report.get('trained_tokens'), 'tokens': report.get('tokens'),
            'physical_ticks': report.get('physical_ticks'), 'summary': report.get('summary'),
            'source_counts': report.get('source_counts'), 'report': pointer.get('report'),
            'hazard_scope': 'all_physical_ticks',
            'decision_conditioned_audit': pointer.get('decision_conditioned_audit', {'status': 'unmeasured'}),
            'interpretation': 'Acute frozen replay effects, not selective familiarity devaluation or learned exploration benefit.'}


def _reply_scores(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return None
    for line in p.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        src = row.get('sources') or []
        pairs = [(a, b) for a, b, s in zip(row.get('scores') or [], row.get('unigram') or [], src)
                 if s == 'mother_reply']
        if pairs:
            rows.append({'trained': row.get('trained'), 'nll': sum(a for a, _ in pairs) / len(pairs),
                         'unigram': sum(b for _, b in pairs) / len(pairs), 'n': len(pairs)})
    return rows


def _act_counts(path):
    p = Path(path)
    decisions = said = 0
    rows = []
    if not p.exists():
        return {'decisions': 0, 'said': 0, 'said_rate': None}
    for line in p.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        rows.append(row)
    rows = rows[-64:]
    for row in rows:
        for item in row.get('speech_decisions') or []:
            decisions += 1
            said += int(len(item) >= 4 and int(item[3]) == 1)
    return {'decisions': decisions, 'said': said, 'said_rate': said / decisions if decisions else None,
            'scope': 'last 64 recorded windows', 'last_trained': rows[-1].get('bptt_train_tokens') if rows else None}


def _consequence_stats(path):
    """Report the new protocol separately; a falling gain is not a capability pass."""
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line).get('consequence') or {}
            except ValueError:
                continue
            if row.get('source') == 'native_action_consequence_parameter_ig':
                rows.append(row)
    rows = rows[-64:]
    if not rows:
        return {'status': 'not_recorded', 'behavior_verified': False}
    n = sum(r.get('consequence_observations', 0) for r in rows)
    mean = lambda key: sum(r.get(key, 0) * r.get('consequence_observations', 0) for r in rows) / max(1, n)
    return {'status': 'interface_observed_behavior_unverified', 'behavior_verified': False,
            'scope': 'last 64 native consequence windows', 'observed_responses': n,
            'posterior_updates': rows[-1].get('consequence_updates'),
            'gain_mean': mean('consequence_gain_mean'), 'prediction_mse': mean('consequence_prediction_mse'),
            'noise': rows[-1].get('consequence_noise'),
            'timing_decisions': sum(r.get('timing_decisions', 0) for r in rows),
            'content_decisions': sum(r.get('content_decisions', 0) for r in rows)}


def audit(run_dir, checkpoint_dir, apply=False):
    run = Path(run_dir)
    transcript = []
    conversation_path = run / 'mother.jsonl'
    if conversation_path.exists():
        for line in conversation_path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
                transcript.append((row['who'], row['text']))
            except (ValueError, KeyError):
                continue
    monitor = {'monitor': conversation_health(transcript)}
    raising = _read_json('data/harness/raising.json') or {}
    progress = _read_json(run / 'progress.json') or {}
    cfg = _read_json(run / 'config.json') or {}
    score_files = sorted(run.glob('scores_c*.jsonl'), key=lambda p: int(p.stem.split('_c')[-1]))
    if (run / 'scores.jsonl').exists():
        score_files.append(run / 'scores.jsonl')
    scores = []
    for path in score_files:
        scores.extend(_reply_scores(path) or [])
    recent = sorted(scores, key=lambda r: r['trained'])[-32:]
    total_n = sum(x['n'] for x in recent)
    gain = sum((x['unigram']-x['nll'])*x['n'] for x in recent)/total_n if total_n else None
    acts = _act_counts(run / 'ai_trace.jsonl')
    probe_names = ('alpha3_paused.json', 'da_info_paused.json', 'mengli_probe_paused.json')
    probes = {name.removesuffix('.json'): bool((run / name).exists()) for name in probe_names}
    consequence = _read_json(run / 'speech_consequence_paused.json') or _read_json(run / 'speech_consequence.json')
    brain = {
        'firing_rate': progress.get('firing_rate'),
        'field_energy': progress.get('field_energy'),
        'read_norm_mean': progress.get('read_norm_mean'),
        'novelty_mean': progress.get('novelty_mean'),
        'hazard_entropy': progress.get('hazard_entropy'),
        'dopamine_cells_firing': progress.get('dopamine_cells_firing'),
        'dopamine_phasic_sd': progress.get('dopamine_phasic_sd'),
        'dopamine_concentration': progress.get('dopamine_concentration'),
        'dopamine_sensitivity_absmax': progress.get('dopamine_sensitivity_absmax'),
        'intrinsic_firing': progress.get('intrinsic_firing'),
        'intrinsic_kappa_mean': progress.get('intrinsic_kappa_mean'),
        'intrinsic_kappa_min': progress.get('intrinsic_kappa_min'),
        'intrinsic_kappa_max': progress.get('intrinsic_kappa_max'),
        'dopamine_plasticity_spread': progress.get('dopamine_plasticity_spread'),
        'native_terminal_feedback': {
            'enabled': cfg.get('kc_dan_release', False),
            'native_credit_enabled': cfg.get('mother_native_credit', False),
            'gain_mean': progress.get('kc_dan_release_gain_mean'),
            'gain_min': progress.get('kc_dan_release_gain_min'),
            'gain_max': progress.get('kc_dan_release_gain_max'),
            'status': 'configuration_and_last_logged_only_function_unverified'},
        'probes': probes,
    }
    # Existing probes are historical diagnostics.  File presence alone proves no function or freshness.
    brain['checks'] = {}
    for name in ('kc', 'mbon', 'dan', 'plasticity', 'homeostasis', 'familiarity', 'read_think_speak'):
        brain['checks'][name] = {'status': 'needs_current_paired_evidence'}
    parsed_probes = {name: _read_json(run / name) for name in probe_names if (run / name).exists()}
    alpha = parsed_probes.get('alpha3_paused.json') or {}
    info = parsed_probes.get('da_info_paused.json') or {}
    paired = parsed_probes.get('mengli_probe_paused.json') or {}
    for name, group in (('mbon', 'MBON_all'), ('dan', 'PPL1_all')):
        rate = (alpha.get('rate_per_tick') or {}).get(group)
        if rate is not None:
            brain['checks'][name] = {'status': 'historical_activity_observed_function_unverified', 'rate_per_tick': rate,
                                     'source': 'alpha3_paused.json', 'protocol': '512 validation tokens, 8 ticks/token; differs from variable training pace'}
    if info:
        brain['checks']['plasticity'] = {'status': 'historical_write_observed',
            'edges_written_share_min_med_mean_max': info.get('edges written share'), 'source': 'da_info_paused.json'}
        brain['checks']['familiarity'] = {'status': 'historical_correlations_not_causal_proof',
            'phi_repeat_rho': info.get('familiarity: phi over repeats (spearman)'),
            'mbon_repeat_rho': info.get('familiarity: MBON response over repeats (spearman)'),
            'source': 'da_info_paused.json'}
    if paired:
        brain['checks']['kc'] = {'status': 'historical_perturbation_inconclusive',
            'acute_ce_delta': paired.get('K_kc_acute_minus_intact'), 'source': 'mengli_probe_paused.json',
            'protocol': '256 tokens, 6 ticks/token; residual suppressed release nonzero; differs from training pace'}
    brain['checks']['homeostasis'] = {'status': 'last_logged_only_no_cell_health_pass',
        'kappa_mean': progress.get('intrinsic_kappa_mean'), 'firing_rate': progress.get('firing_rate'),
        'source': 'progress.json'}
    brain['historical_probe_results'] = parsed_probes
    brain['interpretation'] = 'Activity is distinct from causal learning; absent evidence is unknown.'
    metadata = _read_json(run / 'checkpoint_metadata.json') or {}
    cp = Path(checkpoint_dir) / 'last.pt'
    if cp.exists() and metadata.get('mtime_ns') != cp.stat().st_mtime_ns:
        metadata = {'status': 'metadata_stale', 'path': str(cp)}
    curriculum_path = ROOT / (raising.get('parenting', {}).get('curriculum_path') or 'docs/information_boltzmann/MENGLI_EDUCATION.json')
    curriculum = _read_json(curriculum_path) or {}
    report = {
        'run': str(run), 'checkpoint': str(checkpoint_dir), 'trained': progress.get('bptt_train_tokens'),
        'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'logged_progress_status': progress.get('status'),
        'current_neural_function': 'unproven; match checkpoint, intervals and causal controls before passing',
        'normal_reply': {'windows': len(recent), 'tokens': total_n, 'mean_gain_over_unigram': gain},
        'active_loop': acts,
        'action_consequence': _consequence_stats(run / 'ai_trace.jsonl'),
        'native_causal_evidence': _native_causal_evidence(run, metadata),
        'active_inference_contract': {
            'perception': 'continuous input; independently learned sensory precision',
            'thinking': 'ongoing physical dynamics; waiting affects the actual commit read; independent think-action benefit unverified',
            'speech': 'existing decoder content and motor first-commit timing receive consequence credit',
            'native_path': 'KC-to-MBON topology, MBON membranes, native phi and DAN form the consequence context; outcome is KC response',
            'uncertainty': 'additional persistent linear-Gaussian parameter belief; native phi is not equated to Bayesian uncertainty',
            'familiarity': 'posterior covariance shrinks with repeated matched evidence; causal native familiarity/behavioral redirection unverified',
            'functional_status': 'implementation connected; unified useful exploration and native causal benefit still need current evidence'},
        'speech_consequence': {'status': 'not_validated', 'reason': 'Fixed-reply self-hearing comparison cannot establish environmental information gain.'} if consequence else None,
        'checkpoint_provenance': metadata,
        'curriculum': curriculum,
        'stage_readiness': stage_readiness(curriculum, raising.get('parenting', {}).get('education_stage', 'connection')),
        'world': monitor.get('monitor', monitor),
        'brain': brain,
        'plan': {'topic': (raising.get('parenting') or {}).get('current_topic'),
                 'intent': (raising.get('parenting') or {}).get('intention'),
                 'stage': (raising.get('parenting') or {}).get('education_stage'),
                 'care_mode': (raising.get('parenting') or {}).get('care_mode'),
                 'activity': (raising.get('parenting') or {}).get('activity')},
        'config': {k: cfg.get(k) for k in ('mother_world', 'mother_credit', 'mother_reward', 'ai', 'motor_acts')},
    }
    _atomic_json(run / 'ling_audit.json', report)
    if monitor.get('monitor'):
        latest = recent[-1] if recent else {}
        decision = ParentingHarness(ROOT / 'data/harness/raising.json', run / 'parenting_state.json', ROOT / 'data/harness/mother_notes.md').update(
            trained=int(progress.get('bptt_train_tokens') or 0), monitor=monitor['monitor'],
            nll=latest.get('nll'), unigram=latest.get('unigram'),
            reply_nll=latest.get('nll'), reply_unigram=latest.get('unigram'),
            brain=brain, active_loop=acts, speech_consequence=consequence, apply=apply)
        report['parenting_action'] = decision
        _atomic_json(run / 'ling_audit.json', report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    print(json.dumps(audit(args.run, args.checkpoint_dir, args.apply), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
