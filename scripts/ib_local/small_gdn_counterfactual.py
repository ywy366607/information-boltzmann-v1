"""Prompt curriculum and frozen routing interventions for the existing body.

The source is the authorized official-generator short RULER layout. This is
explicitly a changed prompt layout, not a standard RULER benchmark. Model
updates remain ordinary answer NLL; researcher-selected reads are probe-only.
"""
from contextlib import contextmanager
import re

import numpy as np
import torch


CLUE = re.compile(r'One of the special magic numbers for (.+) is: (\d{7})\.')


def counterpart(number):
    """Change every digit, preserving seven digits and a nonzero first digit."""
    if len(number) != 7 or not number.isascii() or not number.isdigit() or number[0] == '0':
        raise ValueError('expected the official seven-digit answer support')
    return str((int(number[0])-1+4) % 9+1)+''.join(str((int(c)+5) % 10) for c in number[1:])


def near_pair(row):
    """Keep the actual clue/background, move the clue last and shorten query."""
    matches = list(CLUE.finditer(row['input']))
    if len(matches) != 1 or row['outputs'] != [matches[0].group(2)]:
        raise ValueError('one named clue and one matching official answer required')
    match = matches[0]
    lines = row['input'].splitlines()
    question = f'What is the special magic number for {match.group(1)} mentioned in the provided text?'
    if lines[-1] != question:
        raise ValueError('unexpected official question; do not silently rewrite it')
    background = row['input'][:match.start()]+row['input'][match.end():]
    background = background[:background.rfind(question)].strip()
    prompt = background+'\n'+match.group(0)+'\n'+match.group(1)+'?'
    number = match.group(2)
    changed = counterpart(number)
    paired = []
    for variant, value in enumerate((number, changed)):
        text = prompt.replace('is: '+number+'.', 'is: '+value+'.', 1)
        paired.append(dict(index=2*row['index']+variant, source_index=row['index'],
            pair_variant=variant, input=text, answer_prefix='', outputs=[value],
            curriculum='official_short_layout_near_named_query_v1'))
    return paired


def answer_windows(prefix, gold, pending, window=128):
    """Each real transition once, with a final answer-anchored gradient window.

    Earlier prompt transitions are observed without gradients. Every answer
    byte and the newline are supervised in one final <=window graph.
    """
    if window < len(gold)+1:
        raise ValueError('answer does not fit the declared gradient window')
    raw = np.frombuffer(bytes([pending])+prefix+gold+b'\n', dtype=np.uint8)
    stop = len(raw)-1
    tail = max(0, stop-window)
    bounds = [(start, min(start+window, tail)) for start in range(0, tail, window)]
    bounds.append((tail, stop))
    answer_start = 1+len(prefix)
    return raw, [(start, end, np.arange(start+1, end+1) >= answer_start)
                 for start, end in bounds]


def top_one_value(weights, values, null_weight):
    """Hard read including the existing empty option, only for frozen probes."""
    if weights.shape[-1] != values.shape[0]:
        raise ValueError('routing/value alignment mismatch')
    joint = torch.cat((weights, null_weight[..., None]), dim=-1)
    selected = joint.argmax(-1)
    padded = torch.cat((values, values.new_zeros(1, values.shape[-1])))
    return padded[selected.to(padded.device)]


@contextmanager
def frozen_read(bank, mode, metadata, starts, needle, layer=None):
    """Override final-query reads only; never install an oracle into training.

    'first' delivers the observed first-digit native response, 'wrong' an
    observed adjacent position, and 'top1' the learned highest-weight address.
    These are interventions through the learned inlet/downstream, not bounds
    on what a different decoder could learn.
    """
    originals = []
    traces = []
    for i, memory in enumerate(bank.memories):
        if layer is not None and layer != i:
            continue
        owned = 'retrieve_context' in memory.__dict__
        previous = memory.retrieve_context
        originals.append((memory, owned, previous))

        def read(query, block, i=i, memory=memory, previous=previous):
            actual = previous(query, block)
            pr = memory.position_read
            if pr is None:
                raise RuntimeError('probe requires retained observed positions')
            if mode == 'top1':
                joint = torch.cat((pr['weights'], pr['null_weight'][:, None]), dim=-1)
                selected = joint.argmax(-1).tolist()
                # Read only selected vectors; do not launch a device gather
                # for every unselected position in all retained records.
                chosen = []
                for index in selected:
                    if index == pr['weights'].shape[-1]:
                        chosen.append(actual.new_zeros(actual.shape[-1]))
                    else:
                        owner = int(pr['record_ids'][index])
                        slot = int(pr['slots'][owner])
                        pos = int(pr['value_positions'][index])
                        chosen.append(memory.replayed_by_slot[slot][0, pos])
                changed = torch.stack(chosen)
                selected_position = None
            elif mode in ('first', 'wrong'):
                selected_position = needle+(mode == 'wrong')
                found = []
                for slot, meta in metadata[i].items():
                    lo, hi = meta['start']-starts[i], meta['end']-starts[i]
                    capsule = memory.capsules[slot]
                    if (lo <= selected_position < hi and capsule is not None and
                            capsule.version == meta['version']):
                        found.append((slot, selected_position-lo))
                if len(found) != 1:
                    raise RuntimeError(('observed position missing/ambiguous', i+1, found))
                slot, pos = found[0]
                changed = memory.replayed_by_slot[slot][:, pos]
            else:
                raise ValueError(mode)
            traces.append(dict(layer=i+1, mode=mode, observed_position=selected_position,
                original_rms=float(actual.square().mean().sqrt()),
                changed_rms=float(changed.square().mean().sqrt())))
            return changed.to(actual)

        memory.retrieve_context = read
    try:
        yield traces
    finally:
        for memory, owned, previous in originals:
            if owned:
                memory.retrieve_context = previous
            else:
                del memory.retrieve_context
