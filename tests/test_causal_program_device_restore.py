"""Restored structured receipts must follow the continuing body's device."""
import pytest
import torch

from scripts.ib_local.causal_relation_program import Evidence, Port, RelationState
from scripts.ib_local.run_causal_water_reply import to_device


def receipt():
    evidence=Evidence(torch.ones(1,3),torch.zeros(1,3),torch.ones(1,2),Port.MOTHER,
                      observed=False,encoded=True)
    return RelationState(8,2,17,0,torch.tensor([1,3]),torch.tensor([7,9]),
                         torch.tensor([True,False]),(evidence,),release=0)


def test_frozen_nested_receipt_moves_all_tensors_and_preserves_identity():
    original=receipt()
    moved=to_device({'pending':[original]},'meta')['pending'][0]
    assert (moved.identity,moved.generation,moved.version)==(8,2,17)
    assert moved.release==0 and moved.extra[0].kind==Port.MOTHER
    assert moved.slots.device.type==moved.valid.device.type=='meta'
    assert moved.extra[0].source.device.type==moved.extra[0].outcome.device.type=='meta'
    assert original.slots.device.type=='cpu' and original.extra[0].source.device.type=='cpu'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA restore contract')
def test_cpu_cuda_cpu_receipt_roundtrip_preserves_tensor_values():
    original=receipt()
    gpu=to_device(original,'cuda')
    assert gpu.slots.is_cuda and gpu.valid.is_cuda and gpu.extra[0].source.is_cuda
    restored=to_device(gpu,'cpu')
    for a,b in ((original.slots,restored.slots),(original.stamps,restored.stamps),
                (original.valid,restored.valid),(original.extra[0].outcome,restored.extra[0].outcome)):
        assert torch.equal(a,b)
