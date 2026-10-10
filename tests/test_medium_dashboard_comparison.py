"""The live dashboard must compare matching streams and follow appended evals."""
import json

from scripts.ib.serve_medium_cosmos import DashboardDataStore


def fixture_store(tmp_path):
    results = tmp_path / 'results'
    medium, fly = results / 'medium', results / 'fly'
    medium.mkdir(parents=True)
    fly.mkdir()
    report = results / 'published' / 'comparison.json'
    report.parent.mkdir()
    report.write_text(json.dumps({'medium_run': 'medium', 'prior_hash_equal': True,
                                 'prior_hash': 'prior',
                                 'fly_same_individual_50k_stage': 'fly'}))
    dataset = {'train.npy': 'train', 'validation.npy': 'val'}
    (medium / 'config.json').write_text(json.dumps(
        {'prior_counts_sha256': 'prior', 'data_sha256': dataset}))
    (fly / 'config.json').write_text(json.dumps({'dataset_manifest': {'files': dataset}}))
    return DashboardDataStore(medium, comparison_report=report), medium, fly


def write_eval(run, data, append=False):
    with (run / 'lifelong_evaluation.jsonl').open('a' if append else 'w') as handle:
        handle.write(json.dumps(data) + '\n')


def test_comparison_updates_with_new_matching_intervals(tmp_path):
    store, medium, fly = fixture_store(tmp_path)
    row = {'fresh_training_tokens': 50016, 'fresh_validation_cursor': 256,
           'B_curve': [6., 6.2], 'B_prior_curve': [7., 7.2], 'optimizer_updates': 1600}
    reference = {'bptt_train_tokens': 50000, 'fresh_validation_cursor': 256,
                 'B_curve': [7., 7.2], 'bptt_optimizer_updates': 4900}
    write_eval(medium, row)
    write_eval(fly, reference)
    first = store.comparison(medium)
    assert first['matched_evaluations'] == first['medium_wins'] == 1
    assert abs(first['latest']['advantage'] - 1.) < 1e-8
    assert first['latest']['target_start'] == 254
    assert first['latest']['fly_stage'] == 'S4'
    row.update(fresh_training_tokens=55008, fresh_validation_cursor=258)
    reference.update(bptt_train_tokens=55000, fresh_validation_cursor=258)
    write_eval(medium, row, append=True)
    write_eval(fly, reference, append=True)
    assert store.comparison(medium)['matched_evaluations'] == 2
    assert store.comparison(medium)['latest']['fresh_tokens'] == 55008


def test_comparison_rejects_different_exposure_cursor_dataset_or_prior(tmp_path):
    store, medium, fly = fixture_store(tmp_path)
    row = {'fresh_training_tokens': 50016, 'fresh_validation_cursor': 256,
           'B_curve': [6.], 'B_prior_curve': [7.]}
    reference = {'bptt_train_tokens': 60000, 'fresh_validation_cursor': 256,
                 'B_curve': [7.]}
    write_eval(medium, row)
    write_eval(fly, reference)
    assert store.comparison(medium)['pairs'] == []
    reference.update(bptt_train_tokens=50000, fresh_validation_cursor=257)
    write_eval(fly, reference)
    assert store.comparison(medium)['pairs'] == []
    reference['fresh_validation_cursor'] = 256
    write_eval(fly, reference)
    (fly / 'config.json').write_text('{}')
    assert store.comparison(medium)['pairs'] == []
    (medium / 'config.json').write_text('{"prior_counts_sha256":"different"}')
    assert store.comparison(medium) is None
    other = tmp_path / 'other'
    other.mkdir()
    assert store.comparison(other) is None
