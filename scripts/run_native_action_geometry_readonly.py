"""One frozen native planning call, with the complete trained body/archive.

Uses the restored current source, its legal silence endpoint, and the mean
successor from the original relation program. All additional evaluation is
inside the original native teacher; no world or optimizer call is made.
"""
import argparse
import json
from pathlib import Path
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ports', type=Path, default=Path('scratch/cgb_math_contracts_20261122/source_ports_2537.pt'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--option-calibration', action='store_true')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)

    def status(phase, **extra):
        record = dict(phase=phase, **extra)
        (args.work/'status.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(record), flush=True)

    from scripts.ib_local.run_causal_water_reply import build, sha256_file
    from scripts.ib_local.causal_archive import attach_archive
    from scripts.ib_local import native_amortized_decision as amort, gdn_fused
    from scripts.ib_local.prospective_decision import preview_bind, value
    from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
    from scripts.ib_local.causal_relation_program import compose
    from scripts.native_action_contrast_geometry import observe

    source_sha = sha256_file(args.checkpoint)
    metadata = torch.load(args.ports,
                          weights_only=False, map_location='cpu')
    if Path(metadata['source']).resolve() != args.checkpoint.resolve():
        raise ValueError('frozen ports and checkpoint must identify the same source body')
    cfg = metadata['config']
    del metadata
    status('restoring', source_sha256=source_sha)
    gdn_fused.enable()
    engine, _, _, source = build(args.checkpoint, 'cuda',
        table_storage=args.work/'table', allow_external_archive=True)
    budgets = cfg['causal_archive']['budgets']
    archive = attach_archive(engine, args.work/'causal', manifest=cfg['causal_archive'],
        episode_capacity=budgets['episodes'], relation_capacity=budgets['relations'],
        max_bytes=budgets['max_bytes'])
    try:
        if cfg.get('shared_unknowns') is not None:
            from scripts.ib_local.shared_unknown_bindings import configure_shared_unknowns
            configure_shared_unknowns(engine)
        # Same runtime reattachment as prepare_continuation, lines 781--800:
        # these are engine-owned objects, not model state_dict tensors.
        from scripts.ib_local.causal_use_value import configure as configure_use
        configure_use(engine, cfg['causal_use_value']['arm'])
        from scripts.ib_local import epistemic_value_reference as ev
        ev.configure(engine, schema=cfg['epistemic_value']['schema'],
                     word_execution=cfg['epistemic_value']['word_execution'])
        engine.native_word_runtime = cfg.get('native_word_runtime', {}).get('mode', 'serial')
        engine.model._preview_batch_enabled = (
            cfg.get('native_word_runtime', {}).get('schema', 1) >= 2 or
            engine.epistemic_value_schema >= 6)
        amort.configure(engine, cfg['native_amortization'])
        def forbidden(*a, **kw):
            raise RuntimeError('frozen native measurement attempted a world or optimizer step')
        engine.world.step = forbidden
        engine.optimizer.step = forbidden
        parameter_keys = [(id(p), p._version) for p in engine.model.parameters()]
        state = engine.recurrent.clone()
        momenta = [p.clone() for p in engine.bfss_momenta]
        started = time.perf_counter()
        with torch.no_grad():
            pre = engine._read_source()
            root, address = preview_bind(engine.model, pre)
            momentum = decision_momentum_read(engine.model, engine.bfss_momenta, pre)
            path = compose(engine.model, root, pre, pre, ridge=engine.relation_ridge,
                reward_valid=engine.reward_label_valid(wood=True), momentum=momentum)
            status('native_teacher', cycle=source['cycle'], address=address)
            with observe(engine, max_cases=2, option_calibration=args.option_calibration,
                         capsule_dir=args.work/'capsules' if args.option_calibration else None) as records:
                target = value(engine, root, pre, pre, path['mother']['mean'],
                               path['decision']['mean'], momentum)
        assert records
        assert parameter_keys == [(id(p), p._version) for p in engine.model.parameters()]
        assert torch.equal(state, engine.recurrent)
        assert all(torch.equal(a, b) for a, b in zip(momenta, engine.bfss_momenta))
        result = dict(schema=1, cycle=source['cycle'], source_sha256=source_sha,
            boundary='restored current source; legal silence; original mean successor',
            native_value=float(target[0]), cases=records, elapsed_seconds=time.perf_counter()-started,
            world_calls=0, optimizer_updates=0, source_unchanged=sha256_file(args.checkpoint)==source_sha)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        status('complete', cases=len(records), elapsed_seconds=result['elapsed_seconds'])
    finally:
        archive.close()


if __name__ == '__main__':
    main()
