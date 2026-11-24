"""Disposable whole-event comparison from one actual checkpoint and RNG.

Only private working stores change. No save/export to the individual's body.
The baseline below reconstructs the measured sequential runtime, not a new
learning algorithm. Run arms in separate processes to bound host/GPU memory.
"""
import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path

import torch

from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation


def original_runtime():
    from scripts.ib_local import causal_relation_program as causal
    from scripts.ib_local import generative_relations as relations
    from scripts.ib_local import navigation_joint_cycle as joint
    from scripts.ib_local.scoped_decision_graph import ScopedDecisionGraph
    original = ScopedDecisionGraph.__init__
    def init(self, model, capture=True, **kw):
        original(self, model, capture=capture, optimize=False)
    ScopedDecisionGraph.__init__ = init

    def partitions(x, y, order, ridge, *, noise_prior=None):
        return torch.stack([relations.marginal(x[order[:i]], y[order[:i]], ridge, noise_prior=noise_prior)+
            relations.marginal(x[order[i:]], y[order[i:]], ridge, noise_prior=noise_prior)
            for i in range(2, len(x)-1)])
    relations.partition_evidence = partitions

    def design(model, facts, kind, transport, legacy):
        xs, ys = [], []
        enc = model.encoder
        for e in facts:
            s, a = enc(e.source), enc(e.action)
            y = e.outcome if e.encoded else enc(e.outcome)
            if transport:
                mother = e.mother if e.encoded else enc(e.mother)
                x, y = causal.relation_features(model, s, a, mother, legacy=legacy), y-mother
            else:
                x = causal.relation_features(model, s, a, legacy=legacy)
                prior = model.long_predictor(torch.cat((s, a), -1))
                if kind == causal.Port.MOTHER:
                    prior = prior+model.mother_from_boundary(prior)
                y = y-prior
            xs.append(x); ys.append(y)
        return torch.cat(xs), torch.cat(ys)
    causal._bound_evidence_design = design

    def choice(self, targets, own, record):
        features, logits = joint.causal_prefix(self, targets, own, record)
        if getattr(self, 'variable_expression_schema', None) == 1 and own:
            features, _ = joint.causal_prefix(self, targets, 0, record)
        habit, _, _ = self.plan(features[:1])
        combined = habit[0]+habit.new_tensor(record['offset'])
        result = combined.log_softmax(-1)[record['choice']]
        replay = dict(choice=combined.detach().tolist(), stops=[])
        if record['choice'] == 1 and own:
            replay['byte_logp'] = logits[0].log_softmax(-1).gather(1, targets[0, :own, None]).flatten().detach()
        return result, replay
    joint.JointCycleMixin._choice_policy_term = choice


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=('original', 'before', 'reference', 'optimized', 'current'), required=True)
    p.add_argument('--warmup-events', type=int, default=0,
                   help='private complete events before profiling; advances only this disposable body')
    p.add_argument('--reference-arm', choices=('original', 'before', 'reference'), default='original')
    p.add_argument('--save-checkpoint', action='store_true')
    p.add_argument('--runtime-baseline', choices=('v2', 'v3_static', 'v4_before'), default='v2')
    p.add_argument('--optimizer-policy', choices=('adamw','budgeted'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.arm == 'original':
        original_runtime()
    if a.arm == 'before':
        previous_runtime()
    if a.arm == 'reference':
        if a.runtime_baseline == 'v4_before':
            from scripts.ib_local.navigation_generative_cycle import GenerativeCycleMixin
            GenerativeCycleMixin._utter = GenerativeCycleMixin._utter_with_payload_cache
        elif a.runtime_baseline == 'v3_static':
            from scripts.ib_local.benchmark_dynamic_lengths import length_specialized
            from scripts.ib_local import causal_workspace_scan, bfss_momentum_fused
            causal_workspace_scan._forward = length_specialized(causal_workspace_scan._forward)
            causal_workspace_scan._backward = length_specialized(causal_workspace_scan._backward)
            bfss_momentum_fused._capture = length_specialized(bfss_momentum_fused._capture)
        else:
            reference_runtime()
    runner_options = [
        '--checkpoint', str(a.body/'last.pt.gz'), '--output', str(a.body),
        '--cycles', '1', '--resume-output', '--world-source', 'text',
        '--table-storage', str(a.output/a.arm/'table'),
        '--causal-storage', str(a.output/a.arm/'causal')]
    if a.optimizer_policy: runner_options.extend(['--optimizer-policy', a.optimizer_policy])
    args = continuation_parser().parse_args(runner_options)
    loaded = prepare_continuation(args, resume_output=True)
    e = loaded.engine
    if a.warmup_events < 0:
        raise ValueError('warmup events must be nonnegative')
    for _ in range(a.warmup_events):
        e.step(torch.randn(e.model.encoder.out_features, 32, device=e.device))
    if a.arm == 'original':
        e.policy_replay_backend = 'one_byte'
    count = [0]
    old_step = e.optimizer.step
    def step(*args, **kw):
        result = old_step(*args, **kw)
        count[0] += 1
        return result
    e.optimizer.step = step
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    row = e.step(torch.randn(e.model.encoder.out_features, 32, device=e.device))
    torch.cuda.synchronize()
    profiler.disable()
    seconds = time.perf_counter()-start
    with (a.output/f'{a.arm}_profile.txt').open('w', encoding='utf-8') as f:
        pstats.Stats(profiler, stream=f).sort_stats('cumulative').print_stats(250)
    profiler.dump_stats(str(a.output/f'{a.arm}_profile.prof'))
    result = dict(arm=a.arm, warmup_events=a.warmup_events,
        runtime_baseline=a.runtime_baseline, source_cycle=loaded.cycle, source_sha256=loaded.source_sha,
        seconds=seconds, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        optimizer_updates=count[0], choice=row['choice'], self_bytes=row['self_bytes'],
        external_bytes=row['external_bytes'], metal_g=row['metal_g'],
        candidates=row['joint_language_search']['candidates'],
        replay=row.get('runtime_replay'), policy_error=row.get('policy_replay_max_error'),
        proposal_error=row['joint_language_search'].get('replay_max_error'),
        main_checkpoint_written=False)
    from scripts.ib_local.optimizer_policy import diagnostics
    result['optimizer_plasticity'] = diagnostics(e)
    if a.save_checkpoint:
        from scripts.ib_local.mother_checkpoint import save
        if loaded.config.get('native_body') is not None:
            from scripts.ib_local.native_navigation_body import record_birth_event
            row['updates'] = count[0]
            record_birth_event(e, row)
        directory = a.output/f'{a.arm}_checkpoint'
        directory.mkdir(exist_ok=True)
        e.optimizer.step = old_step
        checkpoint_start = time.perf_counter()
        save(e, directory, loaded.cycle+1, loaded.config.copy())
        result['checkpoint_seconds'] = time.perf_counter()-checkpoint_start
    parameters = {n: v.detach().cpu() for n, v in e.model.named_parameters()}
    snapshot = a.output/f'{a.arm}_parameters.pt'
    torch.save(dict(parameters=parameters, recurrent=e.recurrent.cpu()), snapshot)
    if a.arm == 'optimized':
        baseline = torch.load(a.output/f'{a.reference_arm}_parameters.pt', weights_only=True)
        maximum, squared = 0., 0.
        for n, v in baseline['parameters'].items():
            error = (parameters[n]-v).double()
            maximum = max(maximum, float(error.abs().max()))
            squared += float(error.square().sum())
        result['parameter_max_error'] = maximum
        result['parameter_error_l2'] = squared**.5
        result['recurrent_max_error'] = float((e.recurrent.cpu()-baseline['recurrent']).abs().max())
        ref = json.loads((a.output/f'{a.reference_arm}.json').read_text())
        result['same_realized_path'] = all(result[k] == ref[k] for k in
            ('source_sha256', 'choice', 'self_bytes', 'candidates', 'external_bytes', 'optimizer_updates'))
        result['event_speedup'] = ref['seconds']/seconds
    report_path = a.output/f'{a.arm}.json'
    if report_path.exists():
        with (a.output/'prior_measurements.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(json.loads(report_path.read_text()))+'\n')
    report_path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
    e.model._causal_archive.db.close()


def previous_runtime():
    """Runtime immediately after the first acceleration, for repeated A/B."""
    from scripts.ib_local import quadratic_revision, mother_checkpoint
    from scripts.ib_local.scoped_decision_graph import ScopedDecisionGraph
    init = ScopedDecisionGraph.__init__
    def graph(self, model, capture=True, **kw):
        init(self, model, capture=capture, **dict(kw, pack_inputs=False))
    ScopedDecisionGraph.__init__ = graph
    correction = quadratic_revision.evidence_correction
    def solve(x, y, q, ridge, *, return_factor=False):
        result = correction(x, y, q, ridge)
        return (*result, None) if return_factor else result
    quadratic_revision.evidence_correction = solve
    save = mother_checkpoint.save
    def legacy_save(*args, **kw):
        return save(*args, **dict(kw, streamed_hash=False))
    mother_checkpoint.save = legacy_save


def reference_runtime():
    """The shipped v2 runtime; no preview reuse or captured stopping head."""
    from contextlib import nullcontext
    from scripts.ib_local import preview_runtime, generative_relations as relations
    from scripts.ib_local.scoped_decision_graph import ScopedDecisionGraph
    from scripts.ib_local.causal_archive import CausalArchive
    sync = CausalArchive.sync
    def legacy_sync(self):
        return sync(self, bulk=False)
    CausalArchive.sync = legacy_sync
    preview_runtime.preview_scope = lambda model: nullcontext()
    def scores(model, state, x, y, raw, ridge, noise):
        encoded = model.encoder(raw)
        centered = encoded-encoded.mean(0)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        order = (centered@vh[0]).argsort()
        return (relations.marginal(x, y, ridge, noise_prior=noise),
                relations.partition_evidence(x, y, order, ridge, noise_prior=noise), vh, order)
    relations.organization_scores = scores
    def heldout(x, y, indices, ridge, *, noise_prior=None):
        return torch.stack([relations.log_score(relations.posterior(
            x[[j for j in range(len(x)) if j != i]],
            y[[j for j in range(len(x)) if j != i]], x[i:i+1], ridge, noise_prior=noise_prior),
            y[i:i+1]).sum() for i in indices]) if indices else y.new_empty(0)
    relations.leave_one_out_scores = heldout
    init = ScopedDecisionGraph.__init__
    def graph(self, model, capture=True, **kw):
        init(self, model, capture=capture, **dict(kw, capture_handover=False))
    ScopedDecisionGraph.__init__ = graph


if __name__ == '__main__':
    main()
