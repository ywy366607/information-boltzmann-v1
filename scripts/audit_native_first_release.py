"""Enumerate the native no-input continuation up to its first release.

Each pass records the actual menu, then takes its internal child in a private
body. Release is absorbing at this boundary: no world call is dispatched.
"""
import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.clock_credit_jet import first_arrival_prefix, mark
from scripts.ib_local import autonomous_cycle
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.run_persistent_value_growth_comparison import digest


def audit(source, work, steps):
    opts = continuation_parser().parse_args([
        '--checkpoint', str(source), '--output', str(work/'body'), '--world-source', 'text',
        '--autonomous-events', '--optimizer-policy', 'budgeted', '--cycles', '1',
        '--table-storage', str(work/'table'), '--causal-storage', str(work/'causal')])
    torch.set_num_threads(1)
    loaded = prepare_continuation(opts)
    engine = loaded.engine
    try:
        if engine.autonomous_ready:
            raise ValueError('checkpoint has already released; there is no pre-world selector to enumerate')
        menus, survival = [], 1.
        original, draw = engine._sample_internal_choice, torch.multinomial

        def select_internal(prob, count, *args, **kwargs):
            assert prob.shape == (2,) and count == 1
            return torch.ones_like(draw(prob, count, *args, **kwargs))

        def conditional_child():
            with patch.object(torch, 'multinomial', select_internal):
                return original()

        with patch.object(engine, '_sample_internal_choice', conditional_child):
            for k in range(steps):
                row = autonomous_cycle.dispatch(engine)
                event = row['autonomous_event']
                assert event['kind'] == 'internal' and event['boundary_clocks_unchanged']
                assert row['updates'] == row['external_bytes'] == 0
                release, stay = event['admission']['probability']
                release, stay = release/(release+stay), stay/(release+stay)
                menus.append(dict(quiet_steps_before_release=k, event=event['event'],
                    p_reach=survival, p_release_given_reached=release, p_first_release=survival*release,
                    p_continue_given_reached=stay, quiet_state_change=event['internal']['state_change']))
                survival *= stay

        def block(key, elapsed):
            return [mark(torch.tensor([[m[key]]], dtype=torch.double), elapsed) for m in menus]

        hit, tail = first_arrival_prefix(block('p_continue_given_reached', 1.),
                                         block('p_release_given_reached', 0.))
        assert math.isclose(float(tail.mass), survival, abs_tol=1e-12)
        assert math.isclose(float(hit.mass)+survival, 1., abs_tol=1e-12)
        frozen = menus[0]['p_continue_given_reached']
        return dict(cycle=loaded.cycle, source_sha256=digest(source), menus=menus,
            tail_probability=survival, releases_within_budget=float(hit.mass),
            expected_quiet_steps_capped=float(hit.clock+tail.clock),
            frozen_initial_hazard_tail=frozen**steps,
            frozen_initial_hazard_expected_quiet_steps_capped=sum(frozen**k for k in range(1, steps+1)),
            world_calls=0, optimizer_updates=0)
    finally:
        engine.model._causal_archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--steps', type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    result = audit(args.checkpoint, args.work, args.steps)
    text = json.dumps(result, indent=2, allow_nan=False)
    (args.work/'first_release.json').write_text(text+'\n', encoding='utf8')
    print(text)


if __name__ == '__main__':
    main()
