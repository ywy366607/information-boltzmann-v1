"""Continue the three right-censored drafts from the recorded 2610 decision, offline."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.ib_local import variable_expression as expression
from scripts.ib_local.autonomous_cycle import dispatch
from scripts.ib_local.language_action import peek
from scripts.ib_local.run_causal_water_reply import sha256_file
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.ib_local.table_branch import table_branch


@torch.no_grad()
def continue_tail(engine, table, byte, pair, generator, limit):
    logits, state = pair
    tail = bytearray()
    first_stop_probability = None
    with expression.budget_scope(engine.model, 0.), table_branch(table) as branch:
        while len(tail) < limit:
            stop = getattr(engine.decision, 'handover_logit', None)
            if stop is None:
                stop = expression.handover_logits(engine.model, state, 0.).reshape(())
            probability = stop.sigmoid()
            if first_stop_probability is None:
                first_stop_probability = float(probability)
            if bool(torch.rand((), device=engine.device, generator=generator) < probability):
                break
            chosen = int(torch.multinomial(logits.log_softmax(-1).exp(), 1, generator=generator))
            branch.run_credit(np.array([byte, chosen], np.uint8), max_window=1)
            tail.append(chosen)
            byte = chosen
            logits, state = engine.decision.step(byte, peek(branch, byte))
    return dict(extra=len(tail), suffix_hex=tail.hex(), first_stop_probability=first_stop_probability,
                ending='guard' if len(tail) == limit else 'handover')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('result', type=Path)
    args = p.parse_args()
    source = Path('G:/cgb_arrived_actor_2601_20261120/observe/body/last.pt.gz')
    metrics = Path('G:/cgb_mother_return_2609_20261120/body/metrics.jsonl')
    with metrics.open(encoding='utf8') as f:
        recorded = json.loads(next(f))
    originals = [bytes.fromhex(h) for h in recorded['joint_language_search']['candidates']]
    assert recorded['cycle'] == 2610 and list(map(len, originals)) == [128, 16, 128, 128]
    digest = sha256_file(source)
    assert digest == '908b1ecaf43b201569e8d541acb7a0dfaded50d8f2c101ba95a0d46780468b9a'
    meta = json.loads((source.parent/'last.metadata.json').read_text(encoding='utf8'))
    assert meta['checkpoint_sha256'] == digest
    args.work.mkdir(parents=True, exist_ok=False)
    opts = continuation_parser().parse_args(['--checkpoint', str(source), '--output', str(source.parent),
        '--cycles', '1', '--world-source', 'text', '--table-storage', str(args.work/'table'),
        '--causal-storage', str(args.work/'causal')])
    print('Restoring 2609 for private draft replay', flush=True)
    loaded = prepare_continuation(opts, resume_output=True)
    e = loaded.engine
    rows = []
    guard = 512
    seed = 261000
    original_film = expression.film

    class DraftsComplete(Exception):
        pass

    def probe(engine, words, table):
        if words is not None:
            return original_film(engine, words, table)
        last = None
        step = engine.decision.step
        def capture(byte, probability):
            nonlocal last
            last = step(byte, probability)
            return last
        with patch.object(engine.decision, 'step', capture):
            result = original_film(engine, None, table)
        prefix = result[0]
        i = len(rows)
        if prefix != originals[i]:
            raise RuntimeError(f'2610 draft {i} differs from recorded bytes (length {len(prefix)})')
        data = dict(index=i, prefix_bytes=len(prefix), prefix_sha256=sha256(prefix).hexdigest(),
                    originally_selected=(i == recorded['joint_language_search']['selected']))
        if len(prefix) == engine.max_utterance:
            rng = torch.Generator(device=engine.device).manual_seed(seed+i)
            data.update(continue_tail(engine, table, prefix[-1], last, rng, guard-len(prefix)))
        else:
            data.update(extra=0, ending='original_handover')
        rows.append(data)
        print(f"draft {i}: {len(prefix)} + {data['extra']} ({data['ending']})", flush=True)
        if len(rows) == len(originals):
            raise DraftsComplete
        return result

    try:
        with patch.object(expression, 'film', probe), patch.object(e.world, 'take', side_effect=AssertionError('world access forbidden')):
            try:
                for _ in range(64):
                    dispatch(e)
            except DraftsComplete:
                pass
            else:
                raise RuntimeError('did not reach recorded draft boundary')
    finally:
        e.model._causal_archive.close()
    report = dict(source_sha256=digest, source_cycle=2609, replay_cycle=2610, guard=guard,
                  suffix_budget_input=0, suffix_seed_base=seed, world_calls=0, updates=0, drafts=rows)
    with args.result.open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
