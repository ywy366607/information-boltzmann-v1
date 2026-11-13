"""Measure a trusted complete checkpoint without constructing or advancing a body."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(values):
    return sum(v.numel() * v.element_size() for v in values if torch.is_tensor(v))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    before = file_digest(args.checkpoint)
    with gzip.open(args.checkpoint, 'rb') as stream:
        state = torch.load(stream, map_location='cpu', weights_only=False)
    model, engine = state['model'], state['engine']
    groups = {}
    for prefix in ('memory.', 'retrieval.', 'causal_relations.'):
        fields = {k[len(prefix):]: v for k, v in model.items() if k.startswith(prefix)}
        groups[prefix[:-1]] = dict(bytes=tensor_bytes(fields.values()), fields={
            k: dict(shape=list(v.shape), dtype=str(v.dtype), bytes=v.numel()*v.element_size())
            for k, v in fields.items() if torch.is_tensor(v)})
    occupied = model['memory.occupied']
    clock = int(model['memory.clock'])
    ages = (clock-model['memory.birth'][occupied]).double()
    memory = groups['memory']
    memory.update(capacity=len(occupied), occupied=int(occupied.sum()), clock=clock,
                  config=model['memory._extra_state'],
                  counters={k: int(model['memory.'+k]) for k in
                            ('episodes', 'admitted_total', 'merged_total', 'evicted_total')},
                  birth_age_input_steps={str(q): float(torch.quantile(ages, q))
                                         for q in (0., .25, .5, .75, 1.)},
                  valid_counts={k: int(model['memory.'+k].sum()) for k in
                                ('mother_valid', 'internal_valid', 'question_valid', 'progress_valid')})
    slot_fields = [v for k, v in model.items() if k.startswith('memory.') and
                   torch.is_tensor(v) and v.ndim and v.shape[0] == len(occupied)]
    memory['bytes_per_slot'] = tensor_bytes(slot_fields)//len(occupied)
    memory['fixed_buffer_bytes'] = memory['bytes']-tensor_bytes(slot_fields)
    words = model['retrieval.word_count']
    groups['retrieval'].update(capacity=len(words), occupied=int((words > 0).sum()),
                               payload_bytes_used=int(words.sum()), max_bytes_per_utterance=32)
    relations = groups['causal_relations']
    valid = model['causal_relations.valid']
    slots = model['causal_relations.slots']
    # The live code verifies both version and unmerged mother evidence.
    current = valid & occupied[slots] & model['memory.mother_valid'][slots]
    current &= model['memory.count'][slots] == 1
    current &= model['causal_relations.stamps'] == model['memory.last_write'][slots]
    current &= model['causal_relations.stamps'] == model['memory.mother_stamp'][slots]
    decision = current & model['memory.internal_valid'][slots]
    decision &= model['causal_relations.stamps'] == model['memory.internal_stamp'][slots]
    relations.update(capacity=len(valid), occupied=int((model['causal_relations.identity'] > 0).sum()),
                     evidence_per_relation=valid.shape[1], flagged_references=int(valid.sum()),
                     current_references=int(current.sum()),
                     current_unique_slots=int(slots[current].unique().numel()),
                     decision_current_references=int(decision.sum()))
    relation_fields = [v for k, v in model.items() if k.startswith('causal_relations.') and
                       torch.is_tensor(v) and v.ndim and v.shape[0] == len(valid)]
    relations['bytes_per_relation'] = tensor_bytes(relation_fields)//len(valid)
    relations['shared_geometry_bytes'] = relations['bytes']-tensor_bytes(relation_fields)
    arrays = state['table']['arrays']
    table = dict(config=state['table']['config'], bytes=tensor_bytes(arrays.values()),
                 rows=arrays['rows'].shape[0], next_byte_bins=arrays['rows'].shape[1],
                 occupied=int((arrays['count'] > 0).sum()),
                 fields={k: dict(shape=list(v.shape), dtype=str(v.dtype),
                                 bytes=v.numel()*v.element_size()) for k, v in arrays.items()})
    report = dict(checkpoint=str(args.checkpoint.resolve()), cycle=state['cycle'],
                  checkpoint_sha256=before, groups=groups, byte_ngram=table,
                  recurrent_bytes=tensor_bytes([engine['recurrent']]),
                  momentum_bytes=tensor_bytes(engine.get('bfss_momenta', [])),
                  optimizer_updates=0, mother_calls=0,
                  note='Tensor payload bytes; allocator, autograd, file metadata and OS cache excluded.')
    # The public retrieval object is a payload column family of the same
    # episode slots, not an independently addressed history or a second copy.
    report['logical_stores'] = dict(
        causal_memory=dict(episode_bytes=memory['bytes'],
                           utterance_attachment_bytes=groups['retrieval']['bytes'],
                           relation_catalog_bytes=relations['bytes'],
                           total_bytes=memory['bytes']+groups['retrieval']['bytes']+relations['bytes'],
                           episode_and_payload_share_slots=True,
                           relation_evidence_is_versioned_episode_reference=True),
        byte_ngram=dict(total_bytes=table['bytes'], role='online next-byte statistics'))
    report['checkpoint_unchanged'] = before == file_digest(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
