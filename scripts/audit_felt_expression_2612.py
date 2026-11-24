"""Private 2612 next-contact audit; stop before any world call or update."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.ib_local import expression_recall
from scripts.ib_local.autonomous_cycle import dispatch
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation

SOURCE = Path('G:/cgb_mother_return_2609_20261120/body/last.pt.gz')


class SpeechReady(Exception):
    pass


def audit(mode, work):
    args = continuation_parser().parse_args(['--checkpoint', str(SOURCE), '--output', str(work/'body'),
        '--table-storage', str(work/'table'), '--causal-storage', str(work/'causal'),
        '--world-source', 'text', '--mode', 'real', '--cycles', '1'])
    loaded = prepare_continuation(args)
    engine = loaded.engine
    row = dict(mode=mode, world_calls=0, updates=0)
    utter, plan, value = engine._utter, engine.plan, engine._value

    def no_replay(h):
        habit, packet, reusable = plan(h)
        habit = habit.clone()
        habit[..., 2] = -torch.inf
        if mode == 'anchor':
            # Positive control only: offer the immediately preceding own turn.
            # No response text, phrase, or desired output is supplied.
            m, table = engine.model.memory, engine.model.retrieval
            slot = int(engine.prev['slot'])
            count = int(table.word_count[slot])
            assert slot == 2356 and count == 128
            words = bytes(table.words[slot,:count].cpu().tolist())
            distance = float((m.feeling[slot]-engine.intero()[0]).square().mean().sqrt())
            engine._draft_recall = (expression_recall.Recall(slot, int(m.read_credit_version[slot]),
                                                             words, 1., distance),)
        elif mode != 'none':
            engine._draft_recall = expression_recall.retrieve(engine.model, h, engine.intero(),
                                                             felt=mode == 'felt')
        return habit, packet, torch.zeros_like(reusable)

    def capture_value(h, films):
        row['source'] = h.detach().flatten().float().cpu().tolist()
        return value(h, films)

    def capture():
        words, record = utter()
        assert record['choice'] in (0, 1)
        proposals = engine._joint_proposals
        selected = engine._joint_selection['selected']
        row.update(choice=('speak' if record['choice'] else 'silence'),
                   probability=torch.tensor(record['choice_logits']).softmax(0).tolist(),
                   proposal=selected, utterance_hex=words.hex(),
                   candidates=[dict(utf8=p['words'].decode('utf8', 'replace'), hex=p['words'].hex(),
                                    length=len(p['words']), ending=p['handover']['reason']) for p in proposals],
                   recalls=[dict(slot=r.slot, version=r.version, words=r.words.hex(),
                                 feeling_distance=r.feeling_distance, weight=r.weight)
                            for r in getattr(engine, '_draft_recall', ())])
        raise SpeechReady

    try:
        with patch.object(engine, 'plan', no_replay), patch.object(engine, '_utter', capture), \
             patch.object(engine, '_value', capture_value), \
             patch.object(engine.world, 'take', side_effect=AssertionError('world access forbidden')):
            try:
                for _ in range(64):
                    dispatch(engine)
            except SpeechReady:
                return row
            raise RuntimeError('no speech boundary reached')
    finally:
        engine.model._causal_archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('none','content','felt','anchor'))
    parser.add_argument('work', type=Path)
    p = parser.parse_args()
    result = audit(p.mode, p.work)
    with (p.work/'result.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(p.mode, result['choice'], result['probability'], result['proposal'], flush=True)
