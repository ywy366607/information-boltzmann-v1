"""Real body contracts for conditional action planning and private correction."""
import copy

import pytest
import torch

from test_relation_retention_use import configured
from scripts.ib_local import epistemic_value_reference as ev
from scripts.ib_local import native_amortized_decision as amort
from scripts.ib_local import latent_action_planner as planner


def test_full_support_importance_correction_preserves_finite_native_mean():
    p = torch.tensor([.6, .3, .1], dtype=torch.float64)
    learned = torch.tensor([.1, .2, .7], dtype=torch.float64)
    proposal = (p+learned)/2
    mu = torch.tensor([-.2, .1, 2.], dtype=torch.float64)
    native = torch.tensor([1., -.5, .7], dtype=torch.float64)
    estimates = (p*mu).sum()+(p/proposal)*(native-mu)
    torch.testing.assert_close((proposal*estimates).sum(), (p*native).sum())
    assert bool(((p/proposal) <= 2).all())
    anchor = int(p.argmax())
    tail = torch.tensor([i for i in range(len(p)) if i!=anchor])
    q = proposal[tail]/proposal[tail].sum()
    estimates = (p*mu).sum()+p[anchor]*(native[anchor]-mu[anchor])+(p[tail]/q)*(native[tail]-mu[tail])
    torch.testing.assert_close((q*estimates).sum(), (p*native).sum())


@pytest.mark.parametrize('selection', ['novelty_diversity', 'competing_actions'])
def test_action_plans_learn_from_native_and_real_settlement_and_freeze_reads(tmp_path, monkeypatch, selection):
    e, archive = configured(tmp_path)
    try:
        e.gain_units_schema = 1
        ev.configure(e, schema=5, word_execution='native_anchor')
        e.native_word_runtime = 'batch'
        e.model._preview_batch_enabled = True
        specification = dict(schema=1, seed=201120, planner='action_batch',correction='decision_budget')
        if selection=='competing_actions':
            specification['teacher_selection'] = selection
        amort.configure(e, specification)
        # The input packet is captured before the world and is reused verbatim,
        # even though real observation has since changed S/P and parameters.
        captured = []
        original_capture = planner.capture_world
        def capture(engine, record):
            original_capture(engine, record)
            captured.append(copy.deepcopy(engine._latent_world_receipt['packet']))
        monkeypatch.setattr(planner, 'capture_world', capture)
        first = e.step(torch.randn(8, 4))
        bank = e.latent_action_bank
        receipt = first['causal_use']['native_amortization']['action_planner']
        assert receipt['new_world_label'] and receipt['world_labels'] == 1
        assert receipt['native_labels'] > 0 and receipt['records'] <= receipt['capacity']
        assert 0 < receipt['teacher_count'] <= 2
        if selection=='competing_actions':
            picking = receipt['teacher_selection']
            assert picking['covered_roots']==picking['available_roots']
            assert len(set(picking['selected_roots']))==receipt['teacher_count']
            assert {r['root_key'] for r in receipt['teacher_checks']}==set(picking['selected_roots'])
            assert e._competing_action_roots is None
        assert bank['revision'] == 1 and not bank['pending']
        world = next(r for r in bank['records'] if r['kind']=='world')
        assert torch.equal(world['packet']['raw'],captured[0]['raw'])
        assert world['packet']['scalars']==captured[0]['scalars']
        packets = [r['packet'] for r in bank['records']]
        from scripts.ib_local.causal_value_learning import prediction as raw_gain
        evidence = e.model._causal_use_value_evidence
        pre = torch.cat([p['raw'][:1] for p in packets]).to(e.model.encoder.weight)
        act = torch.cat([p['raw'][1:2] for p in packets]).to(pre)
        options = torch.tensor([p['option'] for p in packets],device=pre.device)
        with torch.no_grad():
            z, az = e.model.encoder(pre),e.model.encoder(act)
            expected = evidence.bases()[options]+evidence.confidence()[options]*(raw_gain(e.model,z,az-z,options)-evidence.bases()[options])
            torch.testing.assert_close(planner.prior(e,packets)[:,-1],expected[:,3])
        # An unsupported residual must leave the established action baseline
        # intact, whatever the raw training head happens to predict.
        confidence = evidence.confidence
        monkeypatch.setattr(evidence,'confidence',lambda: torch.zeros_like(confidence()))
        with torch.no_grad():
            torch.testing.assert_close(planner.gain_readout(e.model,z,az-z,options),evidence.bases()[options,3])
        monkeypatch.setattr(evidence,'confidence',confidence)
        predicted, novelty = planner.predict(e, packets)
        with torch.no_grad():
            baseline = planner.prior(e, packets)
        target = torch.stack([r['target'] for r in bank['records']]).to(predicted)
        counts = {kind: sum(r['kind']==kind for r in bank['records']) for kind in ('native','world')}
        weight = predicted.new_tensor([1/counts[r['kind']] for r in bank['records']])[:, None]
        assert float((weight*(predicted-target).square()).sum()) <= float((weight*(baseline-target).square()).sum())+1e-5
        assert predicted.shape[-1] == 2*e.model.encoder.in_features+1
        assert bool(torch.isfinite(predicted).all()) and bool((novelty >= 0).all())
        saved = copy.deepcopy(bank)
        rng = torch.get_rng_state().clone()
        p = copy.deepcopy(packets[0]); changed = copy.deepcopy(p)
        changed['raw'][1] += .8
        before, _ = planner.predict(e, [p]); after, _ = planner.predict(e, [changed])
        assert not torch.equal(before, after)
        assert torch.equal(rng, torch.get_rng_state()) and bank['revision']==saved['revision']
        assert not bank['pending']
        second = e.step(torch.randn(8, 4))
        assert second['causal_use']['finite_U']['verified']
        assert second['causal_use']['reward_version'] == 'finite_relation_use_v5'
        routes = [r for group in second['causal_use']['epistemic_value']['successor_routes'] for r in group]
        assert routes and all(r['native_count']<=2 for r in routes)
        assert 0 < second['causal_use']['native_amortization']['action_planner']['teacher_count'] <= 2
        assert all(r['planned_candidates'] > 0 for r in routes)
        assert bank['revision'] == 2 and not bank['pending']
        # Compare whole latent plans at the SAME actual learned body boundary.
        # Reference splits the post-query batch and uses the original per-layer
        # P path; the candidate packs both axes without changing program IDs.
        from scripts.ib_local import navigation_momentum_wood as momentum
        from scripts.ib_local.preview_runtime import preview_scope
        state = e._program_question_state
        source = e._wood_source.detach().reshape(1, -1)
        with preview_scope(e.model):
            fast, fast_trace, fast_count = planner.plan(e, [state], [source])
        original_read = momentum.decision_momentum_read
        def serial_read(model, momenta, query, *, capture=False):
            model._packed_momentum_enabled = False
            try:
                return torch.cat([original_read(model, momenta, row[None]) for row in query])
            finally:
                model._packed_momentum_enabled = True
        monkeypatch.setattr(momentum, 'decision_momentum_read', serial_read)
        with preview_scope(e.model):
            reference, reference_trace, reference_count = planner.plan(e, [state], [source])
        torch.testing.assert_close(fast, reference, rtol=2e-5, atol=3e-6)
        assert fast_count == reference_count
        for got, want in zip(fast_trace[0], reference_trace[0]):
            assert (got['role'], got['unknown'], got['programs']) == (want['role'], want['unknown'], want['programs'])
            for key in ('proposal_probability', 'predicted_local_use', 'predicted_total', 'novelty'):
                torch.testing.assert_close(torch.tensor(got[key]), torch.tensor(want[key]), rtol=2e-5, atol=3e-6)
        if torch.cuda.is_available():
            # The real generic CUDA restore moves old nested records to CUDA.
            # After one new settlement the bank would then contain CPU AND
            # CUDA rows. Exercise both bank adapters and a subsequent fit.
            saved_bank = copy.deepcopy(e.latent_action_bank)
            for bank in (e.latent_action_bank, e.native_value_bank):
                for record in bank['records'][::2]:
                    record['packet']['raw'] = record['packet']['raw'].cuda()
                    if isinstance(record['target'], torch.Tensor):
                        record['target'] = record['target'].cuda()
            amort.configure(e, e.native_amortization)
            for bank in (e.latent_action_bank, e.native_value_bank):
                assert all(record['packet']['raw'].device.type=='cpu' for record in bank['records'])
            assert e.latent_action_bank['revision']==saved_bank['revision']
            for old, new in zip(saved_bank['records'], e.latent_action_bank['records']):
                torch.testing.assert_close(old['packet']['raw'], new['packet']['raw'], rtol=0, atol=0)
                torch.testing.assert_close(old['target'], new['target'], rtol=0, atol=0)
            planner.predict(e, [r['packet'] for r in e.latent_action_bank['records']])
    finally:
        archive.close()
