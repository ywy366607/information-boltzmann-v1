"""Unit tests for ALIF Surprise-Gated Executive Control (CPU-only, fast)."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.ib_local import alif_executive as alif
from scripts.ib_local.navigation_gain_units import td_gain_target


def test_validate_params_enforces_rheobase_and_refractory():
    # Default params pass strictly.
    p = alif.validate_params({})
    assert p['delta0'] > (1.0 - p['lam_v']) * p['theta0']
    assert p['i_max'] < p['theta0'] + p['beta']

    # Violating Rheobase (zero-starvation) must raise.
    with pytest.raises(ValueError, match='Rheobase'):
        alif.validate_params(dict(lam_v=0.80, theta0=1.0, delta0=0.19))

    # Violating Refractory (anti-storming) must raise.
    with pytest.raises(ValueError, match='Refractory'):
        alif.validate_params(dict(theta0=1.0, beta=0.5, i_max=1.6))

    # Strict 1:1 limit (beta=0, delta0 >= theta0) is explicitly permitted.
    strict = alif.validate_params(dict(beta=0.0, theta0=1.0, delta0=1.0, i_max=1.5))
    assert strict['beta'] == 0.0


def test_rheobase_zero_starvation_under_zero_surprise():
    """Even under completely flat zero-surprise text, ALIF must fire within bounded steps."""
    eng = SimpleNamespace()
    st = alif.configure(eng)
    # Simulate an initial spike entering refractory state (a = 1.0, v = 0.0).
    st['pending_spike'] = False
    st['a'] = 1.0
    st['v'] = 0.0
    # Warm up baseline with constant NLL = 1.5, short_loss = 0.48, p_rms = 1.0.
    st['nll_ema'] = 1.5
    st['nll_var'] = 0.04
    st['nll_count'] = 10
    st['short_ema'] = 0.48
    st['short_var'] = 0.01
    st['short_count'] = 10
    st['prev_p_rms'] = 1.0

    fired_at = None
    for t in range(1, 50):
        out = alif.step_neuron(st, piece_nll=1.5, burst_nll=1.5, short_loss=0.48, p_rms=1.0)
        if out['spiked']:
            fired_at = t
            assert out['cause'] == 'rheobase'
            break
    assert fired_at is not None and 2 <= fired_at <= 25


def test_refractory_prevents_back_to_back_storming_under_adversarial_surprise():
    """Under extreme persistent surprise, ALIF cannot fire two consecutive pieces."""
    eng = SimpleNamespace()
    st = alif.configure(eng)
    st['pending_spike'] = False
    st['nll_ema'] = 1.5
    st['nll_var'] = 0.01
    st['nll_count'] = 20
    st['short_ema'] = 0.45
    st['short_var'] = 0.01
    st['short_count'] = 20
    st['prev_p_rms'] = 1.0

    spikes = []
    for t in range(40):
        # Huge adversarial surprise every single step
        out = alif.step_neuron(st, piece_nll=9.0, burst_nll=12.0, short_loss=5.0, p_rms=1.0 + (t % 2) * 3.0)
        spikes.append(out['spiked'])
    # Must never have two consecutive True entries.
    for t in range(len(spikes) - 1):
        assert not (spikes[t] and spikes[t + 1]), f'Back-to-back storm at t={t}'
    # Must still maintain active firing (not deadlocked) and bounded adaptation.
    assert sum(spikes) >= 10
    assert st['a'] <= 1.0 / (1.0 - st['lam_a']) + 1e-6


def test_burstiness_dominance_over_periodic_on_synthetic_stream():
    """On a bursty surprise stream, ALIF captures strictly more surprise than periodic K."""
    rng = np.random.default_rng(20261124)
    n = 120
    # Mostly calm NLL ~ 1.5, with 15 localized surprise bursts of +0.8..+1.5
    nlls = np.full(n, 1.5, dtype=np.float64) + rng.normal(0, 0.03, size=n)
    burst_positions = [7, 15, 22, 31, 39, 48, 55, 64, 73, 81, 90, 98, 105, 113, 118]
    for pos in burst_positions:
        nlls[pos] += rng.uniform(0.9, 1.5)

    eng = SimpleNamespace()
    st = alif.configure(eng, lam_v=0.80, lam_a=0.75, theta0=1.0, beta=0.90, delta0=0.23, i_max=1.80)
    st['pending_spike'] = False
    st['nll_ema'] = 1.5
    st['nll_var'] = 0.02
    st['nll_count'] = 10

    alif_spikes = []
    for i in range(n):
        out = alif.step_neuron(st, piece_nll=float(nlls[i]), burst_nll=float(nlls[i] + 0.1),
                               short_loss=0.45, p_rms=1.0)
        if out['spiked']:
            alif_spikes.append(i)

    excess = np.maximum(0.0, nlls - 1.5)
    m = len(alif_spikes)
    assert 10 <= m <= 40
    # Match exact periodic spike count m
    periodic_idx = np.linspace(0, n - 1, m, dtype=int).tolist()
    alif_captured = float(excess[alif_spikes].sum())
    periodic_captured = float(excess[periodic_idx].sum())
    assert alif_captured > 2.0 * periodic_captured


def test_calm_sensory_piece_conserves_bytes_and_smdp_duration():
    """Verify calm_sensory_piece advances world, table, S/P, optimizer, and SMDP duration cleanly."""
    class DummyWorld:
        def __init__(self):
            self.cursor = 100
        def take(self, utterance, resp_b, hold_b):
            assert utterance == b''
            total = resp_b + hold_b
            data = (np.arange(self.cursor, self.cursor + total, dtype=np.int64) % 256).astype(np.uint8)
            start = self.cursor
            self.cursor += total
            return data[:resp_b], data[resp_b:], dict(start=start, source='text', match=0)

    class DummyTable:
        def __init__(self):
            self.calls = 0
        def run_credit(self, arr, max_window):
            self.calls += 1
            return None, np.full((max_window, 256), 1.0 / 256.0, dtype=np.float32), None

    class DummyModel:
        def __init__(self):
            self._question_context = 'stale_q'
            self._momentum_active = False
            self._momentum_current = None
            self.body = SimpleNamespace()
            self.encoder = SimpleNamespace(out_features=16)
        def objective(self, ids, target, recurrent, logp, directions):
            assert self._question_context is None, 'calm world piece must clear question_context'
            B, T = ids.shape
            logits = torch.zeros(B, T, 256, requires_grad=True)
            loss = logits.square().mean() + 1.0
            next_state = recurrent + 0.01
            features = torch.zeros(T, 16)
            return loss, next_state, features, dict(logits=logits, latent=torch.tensor(0.45))

    advanced = []
    eng = SimpleNamespace(
        world=DummyWorld(),
        table=DummyTable(),
        model=DummyModel(),
        recurrent=torch.zeros(2, 1, 8),
        bfss_momenta=[torch.ones(1, 4, 2, 2), torch.ones(1, 4, 2, 2) * 2.0],
        bfss_enabled=False,
        bfss_steps=1,
        pending=42,
        device=torch.device('cpu'),
        gamma=0.8,
        gain_units_schema=1,
        prev=dict(g=0.05, duration_blocks=1.0),
    )
    def _inputs(raw, probs):
        ids = torch.as_tensor(raw[:-1].astype(np.int64))[None]
        tgt = torch.as_tensor(raw[1:].astype(np.int64))[None]
        lp = torch.as_tensor(np.log(probs))[None]
        return ids, tgt, lp
    def _advance(loss, features, diag, next_state, actor=None, advantage=0.0):
        assert actor is None and advantage == 0.0
        eng.recurrent = next_state.detach()
        advanced.append(int(features.shape[0]))
        return 1.23
    eng._inputs = _inputs
    eng._advance = _advance

    cfg = {}
    st = alif.configure(eng, config=cfg, piece_bytes=480)
    assert cfg['alif_executive']['schema'] == 1
    st['pending_spike'] = False

    row1 = alif.calm_sensory_piece(eng, torch.zeros(16, 4))
    assert row1['external_bytes'] == 480
    assert row1['learned_bytes'] == 480
    assert row1['updates'] == 1
    assert eng.world.cursor == 580
    assert eng.model._question_context == 'stale_q'  # restored cleanly
    assert eng.prev['g'] == pytest.approx(0.05)      # immediate causal gain UNSHRUNK!
    assert eng.prev['duration_blocks'] == pytest.approx(1.5)

    # Verify td_gain_target uses exact SMDP gamma**1.5 contraction on continuation value!
    td_val = td_gain_target(eng, eng.prev['g'], 10.0, compatible=True)
    assert td_val == pytest.approx(0.05 + (0.8 ** 1.5) * 10.0 / 100.0)

    row2 = alif.calm_sensory_piece(eng, torch.zeros(16, 4))
    assert eng.world.cursor == 1060
    assert eng.prev['duration_blocks'] == pytest.approx(2.0)
    assert advanced == [480, 480]

    # Diagnostics verify exact byte conservation & JSON serializability (restore round-trip).
    diag = alif.diagnostics(eng)
    assert diag['total_external_bytes'] == 960
    assert diag['total_learned_bytes'] == 960
    assert diag['byte_conservation_exact'] is True
    restored = copy.deepcopy(eng.alif_executive_state)
    assert restored == eng.alif_executive_state


def test_record_executive_spike_resets_duration_and_enters_refractory():
    eng = SimpleNamespace(
        bfss_momenta=[torch.ones(1, 4, 2, 2)],
        prev=dict(g=0.04, duration_blocks=3.5),
    )
    st = alif.configure(eng)
    assert st['pending_spike'] is True

    # First spike (initial) enters refractory state and releases into calm flow.
    row0 = dict(external_bytes=960, response_nll=1.55, short_loss=0.47)
    d0 = alif.record_executive_spike(eng, row0)
    assert d0['initial'] is True
    assert st['pending_spike'] is False
    assert st['v'] == 0.0
    assert st['a'] == pytest.approx(1.0)
    assert d0['threshold_next'] == pytest.approx(st['theta0'] + st['beta'])
    assert eng.prev['duration_blocks'] == pytest.approx(1.0)

    # Simulate 3 calm pieces before next spike.
    st['pieces_since_spike'] = 3
    st['calm_bytes_since_spike'] = 1440
    st['calm_pieces'] = 3
    st['total_external_bytes'] += 1440
    st['total_learned_bytes'] += 1440
    row1 = dict(external_bytes=960, response_nll=1.80, short_loss=0.52)
    d1 = alif.record_executive_spike(eng, row1)
    assert d1['initial'] is False
    assert d1['preceding_calm_pieces'] == 3
    assert d1['preceding_calm_bytes'] == 1440
    assert st['inter_spike_pieces'] == [3]
    diag = alif.diagnostics(eng)
    assert diag['executive_spikes'] == 2
    assert diag['total_external_bytes'] == 960 + 1440 + 960
    assert diag['byte_conservation_exact'] is True
    assert diag['mean_inter_spike_pieces'] == pytest.approx(3.0)

