"""Causal byte interface checks, not task capability experiments."""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.small_gdn_ruler import consume_prompt, encode_case, fingerprint, run_case
from scripts.ib_local.train_small_gdn_ruler import training_window


class TinyBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.eye(256)*.1)
        self.memory=nn.Module()
        self.memory.register_buffer('clock',torch.tensor(7))

    def initial_state(self,batch,device):
        return torch.zeros(batch,device=device)

    def predict(self,ids,state,table_logp):
        return self.weight[ids]+table_logp,state+ids.shape[1],{}

    def predict_observed(self,ids,state,table_logp):
        logits,next_state,diag=self.predict(ids,state,table_logp)
        return logits,next_state,diag,ids[0,:,None].float(),None

    def commit_observed(self,features,read):
        self.memory.clock.add_(len(features))


def table():
    return RetrievedCreditTable(orders=(0,1),heads=1,slots_log2=3)


def test_byte_encoding_has_no_sentencepiece_prefix_or_prompt_change():
    row=dict(input='prompt',answer_prefix=' Answer:',outputs=['123'])
    p,g=encode_case(row)
    q,h=encode_case(dict(row,outputs=['999']))
    assert p==q==b'prompt Answer:'
    assert p+g==b'prompt Answer: 123'
    assert h==b' 999'


def test_short_training_supervises_answer_and_newline_without_padding():
    row=dict(input='Question:',outputs=['123'])
    raw,mask=training_window(row,10)
    assert bytes(raw)==b'\nQuestion: 123\n'
    assert bytes(raw[1:][mask])==b' 123\n'
    assert bytes(raw[:-1])==b'\nQuestion: 123'


def test_prompt_executes_each_byte_once_and_never_invents_a_last_target():
    body=TinyBody();bank=table()
    prompt=b'abcde'
    _,state,pending=consume_prompt(body,bank,prompt,chunk=2)
    assert state.item()==len(prompt)
    assert int(body.memory.clock)==7+len(prompt)
    assert bank.events==len(prompt)-1
    assert int(bank.history)==int.from_bytes(prompt[:-1],'big')
    assert pending==prompt[-1]


def test_teacher_scoring_cannot_change_free_generation_or_live_memories():
    engine=SimpleNamespace(model=TinyBody(),table=table(),device=torch.device('cpu'))
    before=fingerprint(engine.model,engine.table)
    row=dict(index=0,input='context 123 and other text. Question:',outputs=['123'])
    a=run_case(engine,row,6)
    b=run_case(engine,dict(row,outputs=['999']),6)
    assert a['prediction']==b['prediction']
    assert a['generated_bytes']==b['generated_bytes']==6
    assert fingerprint(engine.model,engine.table)==before
    assert not a['has_eos']
