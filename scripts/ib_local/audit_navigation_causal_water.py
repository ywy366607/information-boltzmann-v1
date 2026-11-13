"""Read a trained complete individual; audit water wiring without training it."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.train_gdn_fast_table import TableGDN
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.language_navigation_kanli_v4b import attach_imagination
from scripts.ib_local.language_navigation_kanli_v4m import WoodObservationBody
from scripts.ib_local.run_learning_rooms import RoomAuditEngine
from scripts.ib_local.learning_rooms_world import LearningRoomsWorld
from scripts.ib_local.navigation_checkpoint import restore


def audit(checkpoint):
    """One CPU forward/backward on saved evidence; zero optimizer/world steps."""
    torch.set_num_threads(1)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with gzip.open(checkpoint, 'rb') as stream:
        saved = torch.load(stream, map_location='cpu', weights_only=False)
    base = TableGDN(GatedDeltaNetLM(256, 128, 4, 4, 'decay_then_delta', 256, 4, 32.,
                                  decay_init='log_timescale'), True)
    optimizer = torch.optim.AdamW(base.parameters(), lr=3e-4, foreach=True)
    model = attach_imagination(WoodObservationBody(base))
    optimizer.add_param_group(dict(params=[p for n, p in model.named_parameters()
                                          if not n.startswith('body.')], lr=3e-4))
    # Tiny construction table is replaced by the entire saved table on restore.
    table = RetrievedCreditTable(orders=(0,), heads=1, slots_log2=1)
    engine = RoomAuditEngine(model, optimizer, saved['engine']['recurrent'].clone(), table,
        LearningRoomsWorld(11, switch_turn=256, routing='direct_bits'),
        saved['engine']['pending'], np.random.default_rng(11), 'active',
        32, 257, 257, capture=False)
    restore(engine, saved)
    # The archive's execution-device label is CUDA; all tensors were explicitly
    # mapped to CPU for this isolated audit, including the pending actor.
    engine.device = next(model.parameters()).device
    migrated = model.state_dict()
    tensor_errors = []
    for name, value in saved['model'].items():
        if torch.is_tensor(value):
            mapped = name.replace('memory.words', 'retrieval.words').replace(
                'memory.word_count', 'retrieval.word_count')
            assert torch.equal(value, migrated[mapped]), name
            tensor_errors.append(0.)
    memory = model.memory
    assert not any('words' in k or 'word_count' in k for k in memory.state_dict())
    assert engine.prev is not None and engine.prev['actor'] is not None
    assert len(optimizer.state) == len(saved['optimizer']['state'])
    ids, targets, logp = engine.w_ids[0], engine.w_tgt[0], engine.w_logp[0]
    with torch.no_grad():
        memory.read_credit_enabled = False
        reference = model.predict(ids, engine.recurrent, logp)[0].clone()
        memory.read_credit_enabled = True
        neutral = model.predict(ids, engine.recurrent, logp)[0]
    assert torch.equal(reference, neutral)
    old_gain = memory.progress.clone()
    old_words = model.retrieval.words.clone()
    parameters = [p.detach().clone() for p in model.parameters()]
    loss, _, features, _ = model.objective(ids, targets, engine.recurrent, logp,
                                          torch.randn(64, 32))
    with torch.no_grad():
        decision_before = engine.plan(features[:1].detach())[0].clone()
    loss.backward()
    receipt = memory.settle_read_credit()
    credited = memory.read_credit_count > 0
    age = (memory.clock-memory.last_write)[credited]
    with torch.no_grad():
        after = model.predict(ids, engine.recurrent, logp)[0]
        decision_after = engine.plan(features[:1].detach())[0]
    unchanged = all(torch.equal(p, old) for p, old in zip(model.parameters(), parameters))
    assert unchanged and torch.equal(old_gain, memory.progress)
    assert torch.equal(old_words, model.retrieval.words)
    assert receipt['participations'] > 0
    assert digest == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return dict(source_checkpoint=str(checkpoint), checkpoint_sha256=digest,
        scope='CPU stored-witness forward/backward on a migrated copy; no world or optimizer steps',
        cycle=saved['cycle'], migrated_tensor_max_error=max(tensor_errors),
        restored_optimizer_states=len(optimizer.state), pending_actor_preserved=True,
        causal_state_has_byte_payload=False, neutral_prediction_max_error=float((neutral-reference).abs().max()),
        actual_loss=float(loss.detach()), read_credit=receipt,
        credited_records_since_last_write=dict(min=int(age.min()), max=int(age.max()),
            over_128_bytes=int((age > 128).sum()), over_512_bytes=int((age > 512).sum())),
        next_prediction_max_change=float((after-neutral).abs().max()),
        next_choice_logit_max_change=float((decision_after-decision_before).abs().max()),
        parameters_unchanged=unchanged, measured_gains_unchanged=True,
        retrieved_words_unchanged=True, checkpoint_file_unchanged=True,
        parameter_count=sum(p.numel() for p in model.parameters()),
        occupied=int(memory.occupied.sum()), capacity=memory.capacity,
        persistent_credit_bytes=sum(getattr(memory, k).numel()*getattr(memory, k).element_size()
            for k in ('read_credit_mean', 'read_credit_count', 'read_credit_version')),
        exact_payload_bytes=sum(x.numel()*x.element_size() for x in model.retrieval.buffers()),
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path('scripts/ib_local/navigation_read_credit.py'),
            Path('scripts/ib_local/navigation_retrieval.py'),
            Path('scripts/ib_local/latent_navigation_v3.py'),
            Path('scripts/ib_local/language_navigation_kanli_v4.py'),
            Path('scripts/ib_local/language_navigation_kanli_v4l.py'),
            Path('scripts/ib_local/language_navigation_v2.py'))},
        world_steps=0, optimizer_steps=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
