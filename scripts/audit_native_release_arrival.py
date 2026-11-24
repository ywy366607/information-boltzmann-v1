"""One actual native arrival at a specified first-release depth, without learning."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.ib_local import autonomous_cycle
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.native_value_residual_filter import refined_root_values


def audit(checkpoint, work, depth):
    opts = continuation_parser().parse_args([
        '--checkpoint', str(checkpoint), '--output', str(work/'body'), '--world-source', 'text',
        '--autonomous-events', '--optimizer-policy', 'budgeted', '--cycles', '1',
        '--table-storage', str(work/'table'), '--causal-storage', str(work/'causal')])
    torch.set_num_threads(1)
    loaded = prepare_continuation(opts)
    e = loaded.engine
    try:
        assert not e.autonomous_ready and loaded.config['persistent_value_refinement']['steps'] == 2
        original, draw = e._sample_internal_choice, torch.multinomial
        selected = 1

        def conditional_draw(probs, n, *args, **kwargs):
            assert probs.shape == (2,) and n == 1
            return torch.full_like(draw(probs, n, *args, **kwargs), selected)

        def conditional_select():
            with patch.object(torch, 'multinomial', conditional_draw):
                return original()

        torch.cuda.synchronize()
        start = time.perf_counter()
        with patch.object(e, '_sample_internal_choice', conditional_select):
            for _ in range(depth):
                assert autonomous_cycle.dispatch(e)['autonomous_event']['kind'] == 'internal'
            torch.cuda.synchronize()
            quiet_seconds = time.perf_counter()-start
            selected = 0
            release = autonomous_cycle.dispatch(e)['autonomous_event']
        torch.cuda.synchronize()
        released_at = time.perf_counter()
        assert release['kind'] == 'release'
        utter, take = e._utter, e.world.take
        receipt = {}

        def action():
            words, record = utter()
            receipt.update(action=('silence', 'generate', 'reuse')[record['choice']],
                action_probability=torch.tensor(record['choice_logits']).softmax(0).tolist(),
                self_bytes=len(words), utterance_sha256=hashlib.sha256(words).hexdigest())
            return words, record

        def receive(*args, **kwargs):
            started = time.perf_counter()
            reply, heldout, info = take(*args, **kwargs)
            receipt.update(world_take_seconds=time.perf_counter()-started, world=info,
                external_bytes=len(reply)+len(heldout),
                observation_sha256=hashlib.sha256(reply.tobytes()+heldout.tobytes()).hexdigest())
            return reply, heldout, info

        with refined_root_values(e), patch.object(e, '_utter', action), \
                patch.object(e.world, 'take', receive), \
                patch.object(e.optimizer, 'step', side_effect=AssertionError('learning is out of scope')):
            row = autonomous_cycle.dispatch(e)
        event = row['autonomous_event']
        assert event['phase'] == 'world_arrived' and row['updates'] == 0
        elapsed = time.perf_counter()-released_at
        return dict(cycle=loaded.cycle, quiet_steps=depth, quiet_seconds=quiet_seconds,
            release_selector_seconds=released_at-start-quiet_seconds,
            p_release_given_reached=release['admission']['probability'][0],
            release_event=release['event'], arrival_event=event['event'],
            release_to_arrival_seconds=elapsed, action_and_handoff_seconds=elapsed-receipt['world_take_seconds'],
            optimizer_updates=0, **receipt)
    finally:
        engine_archive = e.model._causal_archive
        engine_archive.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--depth', type=int, choices=range(8), required=True)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    result = audit(args.checkpoint, args.work, args.depth)
    (args.work/'arrival.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
