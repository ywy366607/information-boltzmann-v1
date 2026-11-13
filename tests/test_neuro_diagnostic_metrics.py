"""Mathematical/measurement equivalence, not synthetic capability studies."""
import numpy as np
import pytest
import torch

from scripts.ib_local.neuro_diagnostic_metrics import (
    mismatched_information,phi_star_gaussian,factorial_interactions,graph_summary)
from scripts.ib_local.audit_neuro_diagnostics import reset_each_token
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM


def test_matched_gaussian_decoding_equals_mutual_information():
    sx=np.array([[1.,.2],[.2,1.]])
    A=np.array([[.7,.3],[.1,.6]])
    noise=np.array([[1.,.1],[.1,1.]])
    sy=noise+A@sx@A.T
    value=mismatched_information(1.,sx,sy,A@sx,A,noise)
    expected=.5*(np.linalg.slogdet(sy)[1]-np.linalg.slogdet(noise)[1])
    assert value==pytest.approx(expected,abs=1e-10)


def test_independent_gaussian_parts_phi_star_zero():
    result=phi_star_gaussian(np.diag([.8,.6]),np.eye(2),np.eye(2),[[0],[1]])
    assert abs(result['phi_star_nats'])<1e-8
    assert result['bounds_ok']


def test_crossed_copy_has_no_mismatched_decode_information():
    result=phi_star_gaussian(np.array([[0.,.8],[.8,0.]]),np.eye(2),np.eye(2),[[0],[1]])
    assert result['mismatched_information_nats']==pytest.approx(0,abs=1e-9)
    assert result['phi_star_nats']==pytest.approx(result['mutual_information_nats'])


def test_joint_lesion_damage_and_joint_use_benefit_have_opposite_signs():
    losses={f'M{m}T{t}R{r}':np.full(8,1. if not(m or t) else 0.)
            for m in (0,1) for t in (0,1) for r in (0,1)}
    out=factorial_interactions(losses)['R1']
    assert out['joint_lesion_superadditive_damage']['mean']==1
    assert out['joint_use_superadditive_benefit']['mean']==-1


def test_disconnected_cliques_not_given_short_global_path():
    import networkx as nx
    g=nx.disjoint_union(nx.complete_graph(4),nx.complete_graph(4))
    result=graph_summary(g)
    assert not result['connected']
    assert result['path_length'] is None
    assert result['transitivity']==1
    assert result['global_efficiency']<1


def test_vectorized_state_reset_matches_literal_token_resets():
    torch.manual_seed(4)
    model=GatedDeltaNetLM(16,16,2,2,'decay_then_delta',32,4,32.)
    ids=torch.tensor([[1,2,3,4,5]])
    initial=model.initial_state(1)
    with torch.no_grad():
        slow=torch.cat([model.predict_logits(ids[:,i:i+1],initial)[0] for i in range(5)],1)
        with reset_each_token(model.layers):
            fast=model.predict_logits(ids,initial)[0]
    torch.testing.assert_close(fast,slow,atol=1e-5,rtol=1e-5)


def test_full_table_read_preserves_context_while_single_byte_calls_erase_it():
    from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
    from scripts.ib_local.navigation_live_witness import frozen_table_logp
    table=RetrievedCreditTable(orders=(1,2),heads=1,slots_log2=5)
    data=np.tile(np.array([3,4,5,3,4,6],np.uint8),10)
    table.run_credit(data,max_window=128)
    query=np.array([3,4,5,3,4,6],np.uint8)
    before=table.state_dict()
    full=frozen_table_logp(table,query)
    reset=np.concatenate([frozen_table_logp(table,query[i:i+1]) for i in range(len(query))])
    assert np.max(np.abs(full-reset))>1e-3
    assert all(np.array_equal(v,before['arrays'][k]) for k,v in table.state_dict()['arrays'].items())


def test_unestimated_input_means_do_not_masquerade_as_internal_variation():
    from scripts.ib_local.analyze_neuro_diagnostics import byte_conditioned
    ids=np.array([[1,2,1,2],[1,2,1,2],[1,3,1,2]])
    features=np.broadcast_to(ids[:,:,None,None],(3,4,4,8)).astype(float).copy()
    adjusted=byte_conditioned(features,ids,2)
    counts=np.bincount(ids[:2].flatten(),minlength=256)
    known=counts[ids]>0
    assert np.max(np.abs(adjusted[known]))==0
    assert not known[2,1]


def test_invalid_input_pairs_are_excluded_from_information_holdout():
    from scripts.ib_local.analyze_neuro_diagnostics import fit_information
    z=np.random.default_rng(71).normal(size=(4,64,8))
    known=np.ones((4,64),dtype=bool);known[2,10]=False
    first=fit_information(z,2,known=known)
    z[2,10]*=10000
    second=fit_information(z,2,known=known)
    assert first['heldout_cut_penalty']==second['heldout_cut_penalty']
    assert first['heldout_valid_transition_pairs']==124


def test_partial_start_cascades_not_joined_to_complete_cascades():
    from scripts.ib_local.neuro_diagnostic_metrics import avalanche_report
    values=np.array([[[0.],[10.],[0.],[0.],[0.],[0.]],
                     [[0.],[0.],[10.],[0.],[0.],[0.]]])
    report=avalanche_report(values,threshold=.5)
    assert report['completed']==1
    assert report['sizes']==[1]


def test_declared_mmi_decomposition_sums_to_joint_information():
    from scripts.ib_local.neuro_diagnostic_metrics import gaussian_mmi_pid
    cov=np.array([[1.,0.,1.],[0.,1.,1.],[1.,1.,3.]])
    result=gaussian_mmi_pid(cov,[0],[1],[2])
    assert result['mi_joint']==pytest.approx(.5*np.log(3))
    assert result['redundancy']+result['unique_a']+result['unique_b']+result['synergy']==pytest.approx(result['mi_joint'])
    assert 'MMI' in result['definition']


def test_conditional_powerlaw_sampler_matches_tail_cdf_without_rejection():
    from scipy.special import zeta
    from scripts.ib_local.neuro_diagnostic_metrics import sample_powerlaw_tail
    alpha,xmin=4.2,50
    draws=sample_powerlaw_tail(alpha,xmin,10000,np.random.default_rng(52))
    assert np.min(draws)>=xmin
    expected=1-zeta(alpha,xmin+1)/zeta(alpha,xmin)
    assert abs(np.mean(draws==xmin)-expected)<.01
