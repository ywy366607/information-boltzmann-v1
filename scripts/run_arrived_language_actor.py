"""Private H=2 actor experiment; use this entry for first run and resume.

Pass --arrived-mode observe|apply plus the standard private comparison arguments.
"""
import argparse
from unittest.mock import patch


def main():
    import sys
    from scripts.ib_local import run_mother_continuation as continuation
    from scripts.ib_local.arrived_language_actor import attach
    from scripts import run_native_preview_restore_comparison as runner
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--arrived-mode', required=True, choices=('observe','apply'))
    p.add_argument('--probe-seed', type=int, default=20261120)
    args, rest = p.parse_known_args()
    prepare = continuation.prepare_continuation
    def prepared(*a, **kw):
        loaded = prepare(*a, **kw)
        attach(loaded.engine, args.arrived_mode, args.probe_seed)
        loaded.config['arrived_language_actor'] = dict(schema=1, mode=args.arrived_mode,
            horizon=2, probe_seed=args.probe_seed, units='nats/byte', runner=__name__ if __name__!='__main__' else 'scripts.run_arrived_language_actor')
        return loaded
    from scripts import run_native_causal_use_value as use
    from scripts.ib_local.wood_policy_measurement import score, segments
    evaluate = use.evaluate
    def evaluated(engine, material):
        result = evaluate(engine, material)
        source = engine.world.source_manifest
        panel = dict(path=source['validation_path'], sha256=source['validation_sha256'],
                     starts=[8192, 81920, 163840, 262144], bytes_per_segment=2048)
        result['independent_language'] = dict(material=panel, **score(engine,segments(panel)))
        return result
    with patch.object(continuation, 'prepare_continuation', prepared), patch.object(use,'evaluate',evaluated), patch.object(sys,'argv',[sys.argv[0],*rest]):
        runner.main()


if __name__=='__main__':
    main()
