"""Summarize previously executed numerical credit-path interventions."""

import json
from pathlib import Path


def main():
    root = Path('results/published')
    names = {
        'all': 'fly_w32_state_adjoints_20261008.json',
        'active_only': 'fly_w32_pulse_active_credit_20261008.json',
        'silent_only': 'fly_w32_pulse_silent_credit_20261008.json',
    }
    reports = {key: json.loads((root / value).read_text(encoding='utf-8'))
               for key, value in names.items()}
    baseline = reports['all']
    arms = {}
    for key, report in reports.items():
        assert report['status'] == 'passed'
        assert report['optimizer_updates'] == 0
        assert report['physical_ticks'] == 480
        assert report['source_checkpoint'] == baseline['source_checkpoint']
        assert report['target_sha256'] == baseline['target_sha256']
        assert all(row['delta_max_abs'] == 0 for row in report['parameter_updates'].values())
        arms[key] = {
            'gradient_norm': report['stable_total_fp64_norm'],
            'pre_update_training_loss': report['pre_update_training_loss'],
            'max_score_difference_from_all': max(abs(a - b) for a, b in zip(
                report['pre_update_scores'], baseline['pre_update_scores'])),
            'direct_read_raw_norms': {
                name: row['fp64_norm'] for name, row in report['gradients_before_clip'].items()
                if name.startswith(('output_read.', 'decoder.', 'read_norm.'))},
            'current_read_norm_after_clip': report['parameter_norms_after_clip']['output_read.weight'],
            'board_peak_stage_snapshot_mib': max(report['board_memory_snapshots_mib'].values()),
        }
        if key != 'all':
            audit = report['pulse_credit_intervention']
            assert audit['original_ticks'] == 480
            assert audit['ticks_with_hook'] == 479
            assert audit['ticks_without_hook'] == 1
            assert audit['spike_pulse_mask_disagreements'] == 0
            assert audit['ticks'][-1]['hook_calls'] == 0
            arms[key].update(
                covered_physical_ticks=audit['original_ticks'],
                pulse_hooks_fired=audit['ticks_with_hook'],
                last_unused_pulse_hook_fired=False,
                spike_pulse_mask_disagreements=0,
                first_h_adjoint_norm=audit['ticks'][0]['h_adjoint_norm'],
                last_h_adjoint_norm=audit['ticks'][-1]['h_adjoint_norm'],
            )
    result = {
        'scope': 'One restored real OWT window; diagnostic derivatives only, no training benefit',
        'source_reports': names,
        'target_sha256': baseline['target_sha256'],
        'optimizer_updates': 0, 'physical_ticks_per_arm': 480,
        'arms': arms,
        'norm_reduction_all_to_active_only': arms['all']['gradient_norm'] / arms['active_only']['gradient_norm'],
        'conclusion': 'Extreme surrogate amplification in this window depends on silent transmitted-pulse feedback; active-only is a diagnostic, not an approved learner.',
        'limitations': [
            'Interventions remove mixed paths as well; their gradients do not add to the baseline.',
            'GPU floating scores differ within existing baseline reduction variation; CPU forward/state/direct-read contracts are exact.',
            'Silent neurons need first-firing credit; no production active-only mask is deployed.',
            'No NLL improvement or biological learning claim; augmented small-gain calibration remains to be implemented and task-validated.',
        ],
        'reviewer': '/root/rtc_contract_review',
    }
    output = root / 'fly_w32_pulse_credit_comparison_20261008.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
