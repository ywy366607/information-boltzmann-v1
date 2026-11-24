"""Profile and continue the complete native body with bounded transient storage.

The immutable source is the initial recovery point. Only the completed final
checkpoint is copied; no intermediate duplicate of the source is necessary.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--profile-core', action='store_true')
    parser.add_argument('--python-profile', action='store_true')
    parser.add_argument('--direct-typed-question', action='store_true')
    parser.add_argument('--factorized-endpoints', action='store_true')
    parser.add_argument('--shared-utter-preview', action='store_true')
    parser.add_argument('--fused-decode-step', action='store_true')
    parser.add_argument('--native-replay-batch-size', type=int, choices=(1,2,4,8))
    parser.add_argument('--policy-proposal-batch-size', type=int, choices=(1,2,4))
    parser.add_argument('--reuse-objective-logits', action='store_true')
    parser.add_argument('--warm-runtime', action='store_true')
    parser.add_argument('--checkpoint-compressor', choices=('gzip', 'isal'))
    parser.add_argument('--checkpoint-compression-workers', type=int, choices=(1,2,4))
    own, worker_args = parser.parse_known_args()
    work = Path(worker_args[worker_args.index('--work') + 1]).resolve()
    from scripts.ib_local import run_mother_continuation as continuation
    from scripts.ib_local import preview_profile
    from scripts import run_native_preview_restore_comparison as runner
    continuation.ROOT = Path.cwd()
    prepare = continuation.prepare_continuation
    instrument = preview_profile.instrument

    def prepared(*args, **kwargs):
        opts = args[0]
        if kwargs.get('resume_output', False):
            return prepare(*args, **kwargs)
        save, export = continuation.save, continuation.export_live
        copies = []

        def anchor(engine, output, cycle, config, **save_kwargs):
            metadata = json.loads(Path(opts.checkpoint).with_name(
                'last.metadata.json').read_text(encoding='utf8'))
            assert not copies and cycle == metadata['cycle']
            assert Path(output).resolve().is_relative_to(work)
            copies.append(dict(source=str(opts.checkpoint), cycle=cycle,
                source_sha256=metadata['checkpoint_sha256'], final_checkpoint='required'))
            return metadata['checkpoint_sha256']

        continuation.save = anchor
        continuation.export_live = lambda *a, **kw: None
        try:
            loaded = prepare(*args, **kwargs)
            if own.checkpoint_compressor is not None or own.checkpoint_compression_workers is not None:
                from scripts.ib_local.mother_checkpoint import configure_compression
                configure_compression(loaded.config, backend=own.checkpoint_compressor,
                                      workers=own.checkpoint_compression_workers)
            loaded.engine.native_direct_question = own.direct_typed_question
            loaded.engine.native_factorized_endpoints = own.factorized_endpoints
            loaded.engine.native_shared_utter_preview = own.shared_utter_preview
            if own.native_replay_batch_size is not None or own.reuse_objective_logits:
                from scripts.ib_local.native_replay_batch import configure as configure_replay
                configure_replay(loaded.engine, loaded.config, batch_size=own.native_replay_batch_size,
                                 reuse_actual=True if own.reuse_objective_logits else None)
            if own.policy_proposal_batch_size is not None:
                loaded.engine.policy_proposal_batch_size = own.policy_proposal_batch_size
            if own.fused_decode_step:
                from scripts.ib_local.scoped_decision_graph import configure_fused_step
                configure_fused_step(loaded.engine, loaded.config)
            if own.warm_runtime:
                import numpy as np
                import time
                from scripts.ib_local.table_branch import table_branch
                from scripts.ib_local.language_action import peek
                from scripts.ib_local.navigation_live_witness import frozen_table_logp
                began = time.perf_counter()
                engine = loaded.engine
                readonly = np.frombuffer(bytes((int(engine.pending), int(engine.pending))), dtype=np.uint8)
                for raw in (readonly, readonly.copy()):
                    with table_branch(engine.table) as branch:
                        branch.run_credit(raw, max_window=1)
                        peek(branch, int(raw[-1]))
                    frozen_table_logp(engine.table, raw)
                (work/'runtime_warmup.json').write_text(json.dumps(dict(
                    seconds=time.perf_counter()-began, world_calls=0, optimizer_updates=0,
                    scope='read-only table branches rolled back; both readonly/writable input signatures'))+'\n', encoding='utf8')
        finally:
            continuation.save, continuation.export_live = save, export
        assert len(copies) == 1
        (work / 'source_anchor.json').write_text(json.dumps(copies[0], indent=2) + '\n', encoding='utf8')
        return loaded

    @contextmanager
    def detailed(engine, *, enabled):
        with instrument(engine, enabled=enabled) as profile:
            if profile is None or not own.profile_core:
                yield profile
                return
            import torch
            targets = [(gc, 'collect'), (torch.cuda, 'empty_cache'),
                       (engine, '_utter'), (engine.model, 'objective'),
                       (engine.model, 'predict'), (engine.model, 'predict_observed'),
                       (engine.decision, 'step')]
            prior = [(owner, name, getattr(owner, name), name in vars(owner))
                     for owner, name in targets if hasattr(owner, name)]
            import cProfile
            python_profile = cProfile.Profile() if own.python_profile else None
            try:
                for owner, name, function, present in prior:
                    label = ('host' if owner in (gc, torch.cuda) else
                             'decision' if owner is engine.decision else
                             'body' if owner is engine.model else 'engine')
                    setattr(owner, name, profile.wrap(label + '.' + name, function))
                if python_profile is not None:
                    python_profile.enable()
                yield profile
            finally:
                if python_profile is not None:
                    python_profile.disable()
                    python_profile.dump_stats(str(work / 'python_profile.pstats'))
                for owner, name, function, present in reversed(prior):
                    if present:
                        setattr(owner, name, function)
                    else:
                        delattr(owner, name)

    continuation.prepare_continuation = prepared
    preview_profile.instrument = detailed
    argv = sys.argv
    try:
        sys.argv = [argv[0], *worker_args]
        runner.main()
    finally:
        sys.argv = argv
        continuation.prepare_continuation = prepare
        preview_profile.instrument = instrument


if __name__ == '__main__':
    main()
