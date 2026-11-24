"""Read-only natural-life action probabilities and held-out learning readouts."""
from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.action_value_pair_audit import tensor_digest
from scripts.ib_local.navigation_live_witness import frozen_table_logp


def action_receipt(engine, words, record, trace):
    actor = next(x for x in trace.events if x['stage']=='actor')
    raw = torch.tensor(record['choice_logits'],dtype=torch.float64)
    offset = raw.new_tensor(record['offset'])
    habit = raw.new_tensor(record['habit_logits'])
    legal = habit > -1e3
    production = (raw.masked_fill(~legal,-torch.inf)).softmax(-1)
    blind = ((raw-offset).masked_fill(~legal,-torch.inf)).softmax(-1)
    torch.testing.assert_close(production,actor['probability'].double().cpu(),atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(raw-offset,habit,atol=2e-6,rtol=1e-6)
    values = raw.new_tensor(record['values'])
    expected = engine.precision()*(values-values[legal].mean())
    torch.testing.assert_close(offset[legal],expected[legal],atol=2e-6,rtol=2e-5)
    e=engine.epistemic_value_last
    torch.testing.assert_close(values,raw.new_tensor(e['contextual_total'])*10,atol=2e-5,rtol=2e-5)
    return dict(option=int(actor['selected']),sampled_option=int(actor['sampled']),
        utterance_hex=words.hex(),legal=torch.nonzero(legal).flatten().tolist(),
        production=production.tolist(),without_final_wood_bias=blind.tolist(),
        logits=raw.tolist(),offset=offset.tolist(),habit=habit.tolist(),
        total=copy.deepcopy(e['contextual_total']),local=copy.deepcopy(e['local_expected']),
        future=copy.deepcopy(e['contextual_future']),gamma=e['gamma'],
        source_sha256=tensor_digest(engine._internal_source),
        endpoint_sha256=[tensor_digest(x) for x in engine._internal_action],
        recurrent_sha256=tensor_digest(engine.recurrent),
        unknown=engine._program_question_state.binding.identity,
        questions=copy.deepcopy(engine._generative_query['alternatives']),
        proposals=[p['words'].hex() for p in engine._joint_proposals],
        selected_proposal=engine._joint_selection['selected'],
        learning='original nominal policy',
        intervention='only final actor sample; all candidates and subsequent learning retained')


def segments(manifest):
    data=Path(manifest['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==manifest['sha256']
    size=manifest['bytes_per_segment']
    return [np.frombuffer(data[s:s+size+1],np.uint8).copy() for s in manifest['starts']]


@torch.no_grad()
def score(engine, material):
    versions=tuple(p._version for p in engine.model.parameters())
    recurrent=tensor_digest(engine.recurrent)
    before=(int(engine.model.memory.clock),int(engine.table.history),engine.model._causal_archive.db.total_changes)
    rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
    q=getattr(engine.model,'_question_context',None)
    engine.model._question_context=None
    means=[]
    try:
        for raw in material:
            logp=torch.as_tensor(frozen_table_logp(engine.table,raw[:-1]),device=engine.device)[None]
            ids=torch.as_tensor(raw[:-1].astype(np.int64),device=engine.device)[None]
            targets=torch.as_tensor(raw[1:].astype(np.int64),device=engine.device)[None]
            state=engine.recurrent.detach().clone()
            values=[]
            chunk=min(128,engine.model.max_chunk)
            for start in range(0,len(raw)-1,chunk):
                end=min(start+chunk,len(raw)-1)
                logits,state,_=engine.model.predict(ids[:,start:end],state,logp[:,start:end])
                values.append(float(F.cross_entropy(logits[0],targets[0,start:end],reduction='sum')))
            means.append(sum(values)/(len(raw)-1))
    finally:
        engine.model._question_context=q
    assert versions==tuple(p._version for p in engine.model.parameters())
    assert recurrent==tensor_digest(engine.recurrent)
    assert before==(int(engine.model.memory.clock),int(engine.table.history),engine.model._causal_archive.db.total_changes)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
    return dict(mean_nll=float(np.mean(means)),segment_nll=means)


@contextmanager
def serializable_engine(engine):
    """Remove only observation/runner instance methods during a real checkpoint."""
    saved={k:vars(engine)[k] for k in ('_utter','_policy_grad','_advance','_choice_policy_term')
           if k in vars(engine)}
    for key in saved:delattr(engine,key)
    try:yield
    finally:
        for key,value in saved.items():setattr(engine,key,value)


def contribution(probabilities, blind, outcomes):
    p,q,g=map(lambda x:np.asarray(x,dtype=np.float64),(probabilities,blind,outcomes))
    assert p.shape==q.shape==g.shape and abs(p.sum()-1)<1e-6 and abs(q.sum()-1)<1e-6
    delta=p-q
    mass=float(np.abs(delta).sum()/2)
    assert abs(float(delta@g)) <= mass*float(np.ptp(g))+1e-12
    return dict(C=float(delta@g),production_expected_G=float(p@g),
        blind_expected_G=float(q@g),probability_moved=mass,
        outcome_span=float(np.ptp(g)),delta_probability=delta.tolist(),
        receiver_minus_donor_G=(float(delta@g)/mass if mass>0 else 0.))
