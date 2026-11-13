"""Archive the actual complete body, fault hot addresses, and verify disk recall.

No optimizer, mother request or live event. Faults affect only the disposable
loaded workset; the exported archive is captured before the fault injection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.ib_local.causal_archive import attach_archive, sha256
from scripts.ib_local.causal_relation_program import Port, mechanism, real_evidence
from scripts.ib_local.run_causal_water_reply import build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    before_sha = sha256(args.checkpoint)
    engine, _, _, source_meta = build(args.checkpoint, device=args.device)
    model, m = engine.model, engine.model.memory
    archive = attach_archive(engine, args.output/'work')
    try:
        before_stats = archive.stats()
        manifest = archive.checkpoint(args.output/'saved')
        pinned, _ = archive._protected()
        bank = model.causal_relations
        index = next((i for i in range(bank.capacity)
                      if bool(bank.valid[i].any()) and not pinned.intersection(
                          bank.slots[i][bank.valid[i]].cpu().tolist())), None)
        if index is None:
            raise RuntimeError('no unpinned relation available for the disposable cache-fault check')
        state = bank.state(index)
        slots = real_evidence(model, state, Port.MOTHER)
        source = bank.condition[index:index+1].detach().clone()
        action = m.action[slots[:1]].detach().clone()
        rng = torch.get_rng_state().clone()
        params = [p.detach().clone() for p in model.parameters()]
        with torch.no_grad():
            expected = mechanism(model, state, source, action)['mean'].clone()
            # Simulate physical workset reuse. The immutable pre-fault archive
            # must supply the OLD facts, not the replacement cache values.
            m.last_write[slots] += 10000000
            m.read_credit_version[slots] += 1
            m.source[slots] += 5
            m.action[slots] -= 3
        old_rejected = len(real_evidence(model, state, Port.MOTHER)) == 0
        with torch.no_grad():
            bank.identity[index] = 0
            bank.valid[index].zero_()
        assert archive.warm_relation(state.identity)
        restored = bank.state(index)
        actual = mechanism(model, restored, source, action)['mean']
        max_error = float((actual.detach()-expected).abs().max())
        actual.square().mean().backward()
        gradient = float(model.encoder.weight.grad.norm())
        stats = archive.stats()
        report = dict(source=str(args.checkpoint.resolve()), source_sha256=before_sha,
            source_cycle=source_meta['cycle'], device=args.device, initial=before_stats,
            after_disposable_cache_recall=stats, manifest=manifest,
            relation_identity=state.identity, original_evidence=len(slots),
            cold_relation_identity_reloaded=restored.identity == state.identity,
            reloaded_evidence=len(real_evidence(model, restored, Port.MOTHER)),
            stale_hot_reference_rejected=old_rejected, prediction_max_error=max_error,
            current_encoder_gradient_norm=gradient,
            parameters_unchanged=all(torch.equal(p, v) for p, v in zip(model.parameters(), params)),
            torch_rng_unchanged=torch.equal(rng, torch.get_rng_state()),
            source_unchanged=before_sha == sha256(args.checkpoint),
            disk_snapshot_unchanged=manifest['sha256'] == sha256(manifest['snapshot']),
            foreign_key_errors=archive.db.execute('PRAGMA foreign_key_check').fetchall(),
            optimizer_updates=0, mother_calls=0, retained_training_events=0,
            scope='real-body storage/gradient interface audit, no capability improvement measurement')
        report['passed'] = (max_error < 1e-5 and gradient > 0 and
            report['original_evidence'] == report['reloaded_evidence'] and
            all(report[k] for k in ('stale_hot_reference_rejected', 'parameters_unchanged',
                'torch_rng_unchanged', 'source_unchanged', 'disk_snapshot_unchanged',
                'cold_relation_identity_reloaded')) and
            not report['foreign_key_errors'])
        (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
        if not report['passed']:
            raise RuntimeError('real archive recall admission failed')
    finally:
        archive.close()


if __name__ == '__main__':
    main()
