"""Teacher fork, pending labels and identical128-event protocol checks."""
import copy
from types import SimpleNamespace

import torch
from torch import nn

from scripts.ib_local import continuous_frozen_case as module
from scripts.ib_local.irreversible_credit_stream import IrreversibleCreditStream


def test_frozen_case_preserves_pending_cadence_and_discards_gold_branch(monkeypatch):
    class Memory:
        def __init__(self): self.commits = 0
        def state_dict(self): return {'commits': self.commits}
        def load_state_dict(self, state): self.commits = state['commits']
    learner = SimpleNamespace(model=nn.Linear(1,1),memories={0:Memory()},tokens=0,updates=524,
                              cache=SimpleNamespace(_seen_tokens=0,state=0))
    class Backend:
        device = torch.device('cpu')
        def __init__(self, owner): self.learner = owner
        def boundary(self): return {'state': self.learner.cache.state}
        def advance(self,tokens,clock_offset):
            result = []
            for token in tokens:
                self.learner.cache._seen_tokens += 1
                self.learner.cache.state = (self.learner.cache.state*7+token)%101
                row = torch.zeros(11); row[2] = 3
                result.append(row)
            return torch.stack(result)
        def commit(self,inputs,targets,mask,boundary):
            assert len(inputs)==128 and not any(mask)
            self.learner.memories[0].commits += 1
            self.learner.tokens += 128
            return {'observed_only':True}
        def close(self): pass
    monkeypatch.setattr(module,'PretrainedStreamBackend',Backend)
    monkeypatch.setattr(module,'clone_cache',copy.deepcopy)
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda:torch.zeros(1,dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda state:None)
    from scripts.ib_local import evaluate_pretrained_gdn_ruler as evaluation
    monkeypatch.setattr(evaluation,'encode_case',lambda tokenizer,row:([4]*180,[7]*10))
    stream = IrreversibleCreditStream(Backend(learner))
    stream.receive([3]*68,phase='existing')
    state = stream.state_dict()
    state['learn_mask'][0] = True
    tokenizer = SimpleNamespace(eos_token_id=10,decode=lambda ids,**kwargs:','.join(map(str,ids)))
    result = module.run_continuous_frozen_case(learner,tokenizer,dict(index=0,outputs=['answer']),state,5)
    assert result['frozen_pending_labels'] == 1 and state['learn_mask'][0]
    assert result['source_pending_inputs'] == 68 and result['stream_window'] == 128
    assert result['generated_ids'] == [2]*5
    assert learner.cache._seen_tokens == 68+180+5
    # The gold fork crossed256; the actual generated fork did not.
    assert learner.memories[0].commits == 1 and learner.tokens == 128
    assert learner.updates == 524
