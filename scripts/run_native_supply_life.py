"""Run the original private life worker with an explicit supply objective."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--supply-coefficient', type=float, required=True)
    p.add_argument('--defer-initial-checkpoint', action='store_true')
    p.add_argument('--activation-host-mib', type=int, default=128)
    p.add_argument('--activation-directory', type=Path)
    own, worker_args = p.parse_known_args()
    work = Path(worker_args[worker_args.index('--work')+1])
    from scripts.ib_local import run_mother_continuation as continuation
    # A frozen script-only package keeps the original data root explicit.
    continuation.ROOT = Path.cwd()
    original = continuation.prepare_continuation
    specification = dict(schema=1, coefficient=own.supply_coefficient, gamma=9.,
        ridge=1e-3, epsilon=1e-8, port='raw body features -> shared encoder',
        history='stored raw features detached; current body graph live',
        demand='original live SIGReg; shape/count-matched zero reference')

    def prepare(*args, **kwargs):
        saved = continuation.save
        export = continuation.export_live
        deferred = []

        def initial_copy(engine, output, cycle, config, **save_kwargs):
            # This worker restores an immutable complete source and commits a
            # complete final snapshot. The intermediate pre-event copy has no
            # consumer; keep the source as its recoverable anchor instead.
            opts = args[0]
            assert len(deferred) == 0 and not kwargs.get('resume_output', False)
            assert Path(output).resolve() == Path(opts.output).resolve()
            assert Path(output).resolve().is_relative_to(work.resolve())
            metadata = json.loads(Path(opts.checkpoint).with_name(
                'last.metadata.json').read_text(encoding='utf8'))
            assert cycle == metadata['cycle']
            deferred.append(dict(source=str(opts.checkpoint), cycle=cycle,
                source_sha256=metadata['checkpoint_sha256'],
                output=str(output), final_checkpoint='required'))
            return metadata['checkpoint_sha256']

        if own.defer_initial_checkpoint:
            continuation.save = initial_copy
            # A private branch has no pre-event checkpoint to publish. Its
            # final checkpoint retains the original life-card configuration.
            continuation.export_live = lambda *a, **k: None
        try:
            loaded = original(*args, **kwargs)
        finally:
            continuation.save = saved
            continuation.export_live = export
        if own.defer_initial_checkpoint:
            assert len(deferred) == 1
            (work/'deferred_initial_checkpoint.json').write_text(
                json.dumps(deferred[0], indent=2)+'\n', encoding='utf8')
        loaded.config['native_supply_regularizer'] = specification
        return loaded

    from scripts.ib_local.native_supply_regularizer import objective_supply
    from scripts import run_wood_policy_use_worker as worker
    from scripts import run_native_preview_restore_comparison as runner
    original_runner = runner.main

    def run_with_storage():
        sys.argv += ['--activation-host-mib', str(own.activation_host_mib)]
        if own.activation_directory:
            sys.argv += ['--world-activations', str(own.activation_directory/'world'),
                         '--policy-activations', str(own.activation_directory/'policy')]
        return original_runner()

    runner.main = run_with_storage
    continuation.prepare_continuation = prepare
    argv = sys.argv
    receipts = []
    try:
        sys.argv = [argv[0], *worker_args]
        with objective_supply(coefficient=own.supply_coefficient) as receipts:
            worker.main()
    finally:
        continuation.prepare_continuation = original
        runner.main = original_runner
        sys.argv = argv
        work.mkdir(parents=True, exist_ok=True)
        (work/'supply_runtime.json').write_text(json.dumps(dict(
            specification=specification, objectives=receipts),
            ensure_ascii=False, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
