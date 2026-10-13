"""The mainline profile: a birth takes every unnamed switch from it, a resume from the checkpoint's own config, and a
switch the checkpoint never had takes the parser default (no mechanism is dropped or added by omission)."""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _trainer():
    spec = importlib.util.spec_from_file_location('trainer', ROOT / 'scripts/ib/train_fly_bptt_stream.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}
    original = argparse.ArgumentParser.parse_known_args

    def capture(self, *a, **k):
        captured['parser'] = self
        raise SystemExit

    argparse.ArgumentParser.parse_known_args = capture
    argv, sys.argv = sys.argv, ['trainer']
    try:
        module.main()
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_known_args = original
        sys.argv = argv
    return module, captured['parser']


def test_birth_takes_profile_and_command_line_wins():
    m, p = _trainer()
    args = m.apply_profile(p, ['--from-scratch', '--output', 'x', '--no-predictive-writer'], 'mainline')
    assert args.predictive_writer is False and 'predictive_writer' in args.explicit_dests
    assert args.eg_plasticity is True and args.writer == 'receptor_text' and args.E_E == 0.55
    assert args.lr_physiology == 0.0 and args.dopamine_adapted is True


def test_resume_inherits_checkpoint_not_profile():
    m, p = _trainer()
    args = m.apply_profile(p, ['--extend-budget', '--output', 'x'], 'mainline')
    old = {'eg_plasticity': False, 'predictive_writer': False, 'writer': 'receptor_all', 'lr_physiology': 0.005,
           'graph': 'data/malecns_v1/fly_reservoir_coba_r2.npz'}
    m.inherit_profile_switches(p, args, old)
    assert args.eg_plasticity is False and args.predictive_writer is False and args.lr_physiology == 0.005
    assert isinstance(args.graph, Path)
    assert args.E_E is None                     # an old checkpoint without the switch gets the parser default
