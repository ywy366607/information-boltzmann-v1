"""Regression tests for self-evolution beat hygiene, thermostat NaN guard, H-Net B>1 & unscoped 1B, and Gaze NaN guard."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts.ib_local import gaze_pointer as gp_mod
from scripts.ib_local import hnet_chunking as hnet_mod
from scripts.ib_local import phase_space_boltzmann as psb_mod
from scripts.ib_local.world_learning_cycle import _complete_self_evolution_beat


def test_self_evolution_beat_clears_q_skips_fake_memory_refreshes_phase_and_preserves_replay_rng():
    """Directly exercises `_complete_self_evolution_beat` for both silent & spoken beats (#1, #2, #13, #14)."""
    torch.manual_seed(17)
    d = 16
    d_lat = 8

    class _FakeMem:
        def __init__(self):
            self.long = 512
            self.clock = torch.tensor(1000)
            self.occupied = torch.ones(32, dtype=torch.bool)
            self.internal_valid = torch.ones(32, dtype=torch.bool)
            self.count = torch.ones(32, dtype=torch.long)
            self.internal_stamp = torch.ones(32, dtype=torch.long)
            self.last_write = torch.ones(32, dtype=torch.long)
            self.source = torch.randn(32, d)
            self.action = torch.randn(32, d)
            self.internal_next = torch.randn(32, d)
            self.commit_called = False
            self.commit_exp_called = False

        def commit(self, *args, **kwargs):
            self.commit_called = True
            raise AssertionError('memory.commit must NOT be called on zero-observation self-evolution beat')

        def commit_experience(self, *args, **kwargs):
            self.commit_exp_called = True
            raise AssertionError('memory.commit_experience must NOT fabricate zero-observation episodes')

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(d, d_lat)
            self.long_predictor = nn.Linear(2 * d_lat, d_lat)
            self.memory = _FakeMem()
            self._question_context = torch.ones(d_lat)
            self._question_input_mask = torch.ones(1, 2)
            self._question_output_mask = torch.ones(1, 2)
            self.phase_refreshed = 0
            pb_real = psb_mod.PhaseSpaceBoltzmann(latent=d_lat)
            orig_refresh = pb_real.refresh_live_phase_state

            def _tracked_refresh(eng, query_raw=None, update_ema=False):
                self.phase_refreshed += 1
                return orig_refresh(eng, query_raw=query_raw, update_ema=update_ema)

            pb_real.refresh_live_phase_state = _tracked_refresh
            self.phase_boltzmann = pb_real
            self.body = SimpleNamespace(base=SimpleNamespace(d=d))

        def predict_observed(self, ids, state, logp, decode=True):
            T = ids.shape[1]
            logits = torch.zeros(1, T, 256)
            h = torch.randn(T, d)
            return logits, state + 0.01, {}, h, None

    model = _FakeModel()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    witness_q_seen = []

    def _witness_nll():
        witness_q_seen.append(model._question_context)
        return 1.5

    class _FakeInteroStats:
        def update(self, x):
            pass

    rng = np.random.default_rng(12345)
    rng_before = rng.bit_generator.state
    eng = SimpleNamespace(
        model=model,
        device=torch.device('cpu'),
        mode='silence',
        pending=ord('a'),
        recurrent=torch.zeros(2, 1, d),
        bfss_enabled=False,
        bfss_momenta=[],
        phase_space_boltzmann_schema=1,
        prev=None,
        settlement_kind='witness',
        w_ids=[],
        w_admitted=0,
        delayed_applied=0,
        critic_error=None,
        outcome_weight=0.1,
        causal_use_arm='C2',
        causal_use_pending=dict(stale=True),
        _finite_use_commit_pending=dict(stale=True),
        _mother_boundary_target=torch.ones(d),
        _internal_label=(torch.zeros(d), torch.zeros(d), torch.zeros(d)),
        _internal_source=torch.randn(1, d),
        replay_rng=rng,
        optimizer=opt,
        params=list(model.parameters()),
        groups=(list(model.parameters()),),
        intero_stats=_FakeInteroStats(),
        precision=lambda: 1.0,
        _inputs=lambda raw, probs: (
            torch.as_tensor(raw[:-1].astype(np.int64))[None],
            torch.as_tensor(raw[1:].astype(np.int64))[None],
            torch.log(torch.as_tensor(probs).clamp_min(1e-6))[None],
        ),
        _outcome_loss=lambda s, a, l: (model.encoder(s).square().mean(), dict(outcome_error=0.05)),
        witness_nll=_witness_nll,
        table=SimpleNamespace(orders=(1,), probs=lambda h: np.full(256, 1.0 / 256.0, dtype=np.float32)),
    )

    rec = dict(
        choice=0,
        reused_slot=None,
        truncated=False,
        probs=np.zeros((0, 256), dtype=np.float32),
        habit_logits=[0.0, 0.0, -10000.0],
        offset=[0.0, 0.0, 0.0],
        values=[0.1, 0.0, 0.0],
        critic_inputs=torch.zeros(3, 4),
        feeling=torch.zeros(3),
    )
    info = dict(self_evolution_beat=True, send_target_id='none', observe_target_id='none', interaction_mode='quiet_self_evolution')

    row = _complete_self_evolution_beat(
        engine=eng,
        utterance=b'',
        record=rec,
        probabilities=torch.tensor([0.6, 0.4, 0.0]),
        value_now=0.06,
        witness_pre=1.5,
        source_time=1000,
        info=info,
        option=None,
    )
    # 1) Question context was cleared BEFORE witness_nll() and remains None on exit!
    assert witness_q_seen == [None]
    assert model._question_context is None
    assert model._question_input_mask is None
    assert model._question_output_mask is None
    # 2) Zero fabricated memory commits or clock jumps!
    assert not model.memory.commit_called
    assert not model.memory.commit_exp_called
    assert int(model.memory.clock.item()) == 1000
    assert row['episode_slot'] is None
    assert row['self_evolution']['memory_committed'] is False
    assert eng.causal_use_pending is None
    assert eng._finite_use_commit_pending is None
    # 3) Phase state refreshed & replay_rng untouched!
    assert model.phase_refreshed == 1
    assert rng.bit_generator.state == rng_before


def test_thermostat_nan_guard_and_gaze_nan_guard():
    """Verify #5 (NaN entropy never poisons thermostat or pumps P) and #12 (NaN gaze loss/grad never corrupts null logits)."""
    pb = psb_mod.PhaseSpaceBoltzmann(latent=16)
    p0 = torch.ones(1, 16)
    eng = SimpleNamespace(bfss_momenta=[p0.clone()], model=SimpleNamespace(_momentum_current=None))
    # Normal step initializes entropy_ema
    pb.apply_free_energy_thermostat(eng, torch.randn(1, 4, 16))
    ema_good = float(pb.entropy_ema.item())
    p_before = eng.bfss_momenta[0].clone()
    # NaN logits MUST early-return without touching entropy_ema or scaling momenta!
    bad = torch.full((1, 4, 16), float('nan'))
    res = pb.apply_free_energy_thermostat(eng, bad)
    assert res['applied'] is False and res.get('non_finite') is True
    assert float(pb.entropy_ema.item()) == ema_good
    assert torch.equal(eng.bfss_momenta[0], p_before)

    # #12: GazeObjectPointer finite check preserves finite null logits.
    from tests.test_gaze_pointer import _make_gaze_engine
    g_eng = _make_gaze_engine(d_model=16)
    ptr = gp_mod.configure(g_eng, {}, d_gaze=8)
    ptr.select_interaction(g_eng, candidates=[('book', b'book')], utterance_bytes=b'hi')
    ptr.settle_credit(float('nan'), evidence_available=True)
    assert bool(torch.isfinite(ptr._dual().in_null_logit).all())
    assert bool(torch.isfinite(ptr._dual().out_null_logit).all())


def test_hnet_b_gt_1_exact_core_state_unscoped_1byte_bypass_and_experience_layer_assert():
    """Verify #8 (B>1 padded core state matches B=1 per row), #9 (unscoped 1B bypasses H-Net), #11 (experience_layer assert)."""
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
    from scripts.ib_local.train_gdn_fast_table import TableGDN

    torch.manual_seed(9)
    base = GatedDeltaNetLM(vocab_size=256, d=16, layers=6, heads=2)
    body = TableGDN(base, use_table=True, inject_layer=1)
    body.eval()
    chunker = hnet_mod.HNetDynamicChunker(16, enc_layers=2, dec_layers=2, target_ratio=3.0, min_seq_len=1, router_thresh_init=0.35)
    chunker.calibrated.fill_(True)
    body.hnet_chunker = chunker

    # #8: B=2 with very different sequences -> different boundary counts; batched state must equal per-row B=1 state!
    ids0 = torch.full((1, 16), 65, dtype=torch.long)  # repetitive -> fewer boundaries
    ids1 = torch.arange(16, dtype=torch.long)[None] * 13 % 256  # varied -> more boundaries
    ids_b = torch.cat((ids0, ids1), dim=0)
    logp_b = torch.log_softmax(torch.randn(2, 16, 256), dim=-1)
    s0 = body.initial_state(1)
    s_b = body.initial_state(2)

    with torch.no_grad():
        logits_b, st_b, _ = body.predict(ids_b, s_b, logp_b)
        logits_0, st_0, _ = body.predict(ids0, s0, logp_b[0:1])
        logits_1, st_1, _ = body.predict(ids1, s0, logp_b[1:2])

    assert torch.allclose(logits_b[0:1], logits_0, atol=1e-5)
    assert torch.allclose(logits_b[1:2], logits_1, atol=1e-5)
    assert torch.allclose(st_b[:, 0:1], st_0, atol=1e-5)
    assert torch.allclose(st_b[:, 1:2], st_1, atol=1e-5)

    # #9 invariant: 1-byte eager forward (used by `_policy_grad` when own==0) MUST match
    # step 0 of `hnet_step_scope` bit-for-bit (`b_0=1, p_0=1` + skip residual `W_res x_0`).
    id1 = torch.tensor([[70]], dtype=torch.long)
    lp1 = torch.log_softmax(torch.randn(1, 1, 256), dim=-1)
    with torch.no_grad():
        l_eager, st_eager, _ = body.predict(id1, s0, lp1)
        with hnet_mod.hnet_step_scope(body):
            l_scoped, st_scoped, _ = body.predict(id1, s0, lp1)
    assert torch.allclose(l_eager, l_scoped, atol=1e-6)
    assert torch.allclose(st_eager, st_scoped, atol=1e-6)

    # #11: configure() rejects experience_layer < num_layers - dec_layers
    fake_eng = SimpleNamespace(
        model=SimpleNamespace(body=body, experience_layer=3),
        optimizer=torch.optim.SGD(body.parameters(), lr=1e-3),
        params=[],
        groups=(),
    )
    with pytest.raises(ValueError, match='experience_layer'):
        hnet_mod.configure(fake_eng, {}, enc_layers=2, dec_layers=2)
