"""Small stdlib-only JSON fixtures; no model, torch, or training imports."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/summarize_native_preview_restore_comparison.py'
spec = importlib.util.spec_from_file_location('preview_summary', SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture(root, label, increment):
    folder = root/label
    folder.mkdir()
    evaluations = []
    for n in (0, 8, 16, 24, 32):
        queries = [dict(relation=i//8+1, query=f'episode:{i}',
                        score=-i/64+increment*n/32, mse=i/64-increment*n/32)
                   for i in range(64)]
        if label == 'fixed':
            queries.reverse()  # Pair by identities, not list indices.
        evaluations.append(dict(external=n, cycle=2437+n, queries=queries))
    result = dict(status='completed', origin_cycle=2437, cycle=2469, cycles=32,
                  updates=128, external_bytes=30720, initial_source_sha256='same-source',
                  runtime_revision=0 if label == 'legacy' else 1,
                  mode='legacy_restore_simulation' if label == 'legacy' else 'normal_repaired_restore',
                  evaluations=evaluations)
    (folder/'result.json').write_text(json.dumps(result), encoding='utf-8')
    events = [dict(autonomous_event=dict(kind='release'), updates=0, external_bytes=0)]
    for n in range(1, 33):
        events.append(dict(autonomous_event=dict(kind='external'), updates=4, external_bytes=960,
            cycle=2437+n, choice='silence', metal_g=.2,
            pre_outcome_gain=dict(base=1, raw=3, expected=2, target=2,
                                 units='10*finite_U_nats_per_latent_coordinate'),
            causal_use=dict(observed_use=.2, expected_use=.2,
                            finite_U=dict(committed=True, verified=True, committed_score_max_error=0)),
            use_transport=dict(mean=dict(append_gain=.3, release_effect=-.1, actual_U_gain=.2,
                                          representation_shift=0), decomposition_residual=0,
                               queries=[dict(actual_U_gain=.2, append_gain=.3, release_effect=-.1)]),
            wood_confidence_used=[0, 0, 1, .25],
            internal_rollout=dict(delivered_offset=[.1, -.1, 0], choice_probability_tv=.05),
            read_use=[dict(target_fact_id='episode:0', support_ids=['episode:1'],
                           relation=dict(identity=n % 2+1, generation=1, version=n),
                           comparison_update_clock=4*n, log_score=-.5)],
            read_credit={phase: dict(records=2, participations=3, stale=0, credit_sum=.25)
                         for phase in ('response', 'heldout')}))
    (folder/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events), encoding='utf-8')
    return folder


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.legacy = fixture(self.root, 'legacy', 0)
        self.fixed = fixture(self.root, 'fixed', .25)

    def change_result(self, folder, mutate):
        path = folder/'result.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        mutate(data)
        path.write_text(json.dumps(data), encoding='utf-8')

    def test_cli_and_paired_distributions(self):
        output = self.root/'summary.json'
        run = subprocess.run([sys.executable, '-S', str(SCRIPT), '--legacy', str(self.legacy),
                              '--fixed', str(self.fixed/'result.json'), '--output', str(output)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        data = json.loads(output.read_text(encoding='utf-8'))
        self.assertTrue(data['valid'])
        distribution = data['legacy']['fixed_queries']['0']['score']
        self.assertEqual(distribution, dict(n=64, missing=0, min=-63/64,
                                           median=-31.5/64, mean=-31.5/64, max=0))
        for n in (0, 8, 16, 24, 32):
            paired = data['paired_fixed_minus_legacy'][str(n)]
            self.assertEqual(paired['score']['fixed_minus_legacy_change']['mean'], .25*n/32)
            self.assertEqual(paired['mse']['fixed_minus_legacy_change']['mean'], -.25*n/32)
        self.assertEqual(len(data['paired_fixed_minus_legacy']['32']['queries']), 64)

    def test_logged_gain_finite_u_confidence_and_reads(self):
        data = summary.compare(self.legacy, self.fixed)['fixed']
        gain = data['pre_outcome_gain']
        self.assertEqual(gain['base']['signed_error']['mean'], -1)
        self.assertEqual(gain['raw']['squared_error']['mean'], 1)
        self.assertEqual(gain['expected']['squared_error']['mean'], 0)
        self.assertEqual(data['finite_U']['per_event']['causal_use.observed_use']['mean'], .2)
        self.assertEqual(data['finite_U']['flags']['causal_use.finite_U.verified']['true'], 32)
        self.assertEqual(data['confidence']['wood_confidence_used']['3']['mean'], .25)
        self.assertEqual(data['continuation_recorded']['internal_rollout.delivered_offset']['0']['mean'], .1)
        self.assertEqual(data['read_use']['cross_relation_facts'], 2)
        self.assertEqual(data['read_use']['later_update_facts'], 2)
        self.assertEqual(data['read_credit']['response']['totals']['records'], 64)
        self.assertEqual(data['read_credit']['response']['totals']['credit_sum'], 8)

    def test_missing_is_not_zero(self):
        for folder in (self.legacy, self.fixed):
            path = folder/'events.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            for row in rows:
                for key in ('read_use', 'read_credit', 'wood_confidence_used', 'internal_rollout',
                            'pre_outcome_gain', 'causal_use', 'use_transport'):
                    row.pop(key, None)
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        data = summary.compare(self.legacy, self.fixed)
        self.assertTrue(data['valid'])  # Optional observation absence does not falsify run budgets.
        fixed = data['fixed']
        self.assertEqual(fixed['pre_outcome_gain']['raw']['squared_error']['missing'], 32)
        self.assertEqual(fixed['confidence']['wood_confidence_used']['mean'], 'missing')
        self.assertEqual(fixed['read_use']['missing_events'], 32)
        self.assertEqual(fixed['read_credit']['heldout']['totals']['credit_sum'], 'missing')
        self.assertEqual(fixed['continuation_isolated_ranking_effect'], 'missing')

    def test_budget_origin_and_initial_scores_fail(self):
        def corrupt(data):
            data.update(cycles=31, updates=124, external_bytes=29760, origin_cycle=2436)
            data['evaluations'][0]['queries'][0]['score'] += 1
        self.change_result(self.fixed, corrupt)
        data = summary.compare(self.legacy, self.fixed)
        self.assertFalse(data['valid'])
        for key in ('fixed.cycles', 'fixed.updates', 'fixed.external_bytes', 'fixed.origin_cycle',
                    'paired.initial_score_exact'):
            self.assertFalse(data['checks'][key]['ok'])

    def test_query_identity_and_event_totals_fail(self):
        self.change_result(self.fixed, lambda d: d['evaluations'][0]['queries'][0].update(query='wrong'))
        path = self.legacy/'events.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        rows[-1]['updates'] = 3
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        data = summary.compare(self.legacy, self.fixed)
        self.assertFalse(data['valid'])
        self.assertFalse(data['checks']['paired.initial_query_identities']['ok'])
        self.assertFalse(data['checks']['legacy.event_updates']['ok'])
        self.assertEqual(data['paired_fixed_minus_legacy']['32']['status'], 'missing')

    def test_nonfinite_and_absent_statistics(self):
        self.assertEqual(summary.stats([None, float('nan'), float('inf')])['mean'], 'missing')
        self.assertEqual(summary.stats([None, 0])['mean'], 0)
        self.assertEqual(summary.stats([None, 0])['missing'], 1)


if __name__ == '__main__':
    unittest.main()
