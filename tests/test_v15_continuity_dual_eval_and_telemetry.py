"""Deterministic verification for v15 bundle: #17 input continuity, #18 dual-chunk eval, #4/#7 telemetry."""
from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ib_local.phase_space_boltzmann import PhaseSpaceBoltzmann
from scripts.ib_local.unified_world_port import UnifiedWorldPort
from scripts.ib_local.wood_policy_measurement import score as measure_score


class _ArrayBaseWorld:
    def __init__(self, raw: bytes):
        self.data = np.frombuffer(raw, dtype=np.uint8).copy()
        self.seen = np.zeros(len(self.data), dtype=np.bool_)
        self.cursor = 0
        self.anchor = 0
        self.offset = 0
        self.window = 4096

    def take(self, utterance: bytes, response_bytes: int, holdout_bytes: int):
        req = int(response_bytes) + int(holdout_bytes)
        if response_bytes <= 0 or holdout_bytes <= 0:
            raise ValueError('positive lengths required')
        start = self.cursor
        if start + req > len(self.data):
            start = 0
        end = start + req
        self.seen[start:end] = True
        self.cursor = end
        self.anchor = start
        self.offset = req % self.window
        return (
            self.data[start:start + response_bytes].copy(),
            self.data[start + response_bytes:end].copy(),
            dict(start=start, anchor=start, match=0, source='topic', holdout_position_fresh=True),
        )

    def state_dict(self):
        return dict(cursor=int(self.cursor), seen=self.seen.copy())

    def checkpoint_state_dict(self):
        return self.state_dict()

    def load_state_dict(self, state):
        self.cursor = int(state['cursor'])
        self.seen[:] = state['seen']


class _FixedCaregiver:
    def __init__(self, reply_text: str):
        self.reply_text = reply_text
        self.last_telemetry = dict(source='stub', api_usage=dict(prompt_tokens=10, cache_hit_tokens=5, cache_miss_tokens=5))
        self.usage_totals = {}
        self.checkpoint_ref = {}

    def begin_turn(self, **kwargs):
        pass

    def reply(self, turn):
        return self.reply_text

    def commit_turn(self):
        pass

    def state_dict(self):
        return dict(reply_text=self.reply_text)

    def load_state_dict(self, state):
        self.reply_text = state['reply_text']


def _make_cn_corpus(tmp_path: Path) -> tuple[Path, bytes]:
    lines = [f'第{i:03d}句：清晨的小鸟停在河边石头上唱歌。\n' for i in range(40)]
    raw = ''.join(lines).encode('utf-8')
    p = tmp_path / 'cn_story.bin'
    p.write_bytes(raw)
    return p, raw


def test_mother_reply_consumes_only_remaining_book_bytes_and_stays_contiguous(tmp_path: Path):
    cn_path, raw = _make_cn_corpus(tmp_path)
    base = _ArrayBaseWorld(b'English fallback text ' * 400)
    cg = _FixedCaregiver('妈妈听见啦，慢慢读。')
    port = UnifiedWorldPort(
        base, caregiver=cg, sidecar_corpus=cn_path, sidecar_mix_ratio=1.0,
        mother_cooldown=0, mother_max_gap=1,
    )
    # Prime sidecar as active book source.
    r0, c0, _ = port.take_ambient(64, 32)
    assert len(r0) == 64 and len(c0) == 32
    cur_before = port.sidecar.cursor

    # Mother responds to an explicit call; response_bytes=128, holdout_bytes=64.
    resp, cont, info = port.take('妈妈'.encode('utf-8'), 128, 64)
    assert len(resp) == 128 and len(cont) == 64
    assert info['channel'] == 'dsh_mother'
    cur_after = port.sidecar.cursor

    # Extract delivered book tail from response after `said` + `书页：`.
    spans = info['envelope_mask_spans']
    assert len(spans) == 2  # `妈妈：` and `书页：`
    book_start_in_resp = spans[1][1]
    book_tail = bytes(resp[book_start_in_resp:])
    assert len(book_tail) == info['book_body_bytes']
    delivered_book = book_tail + bytes(cont)

    # Every single consumed byte from cur_before..cur_after must match delivered_book (minus any trailing UTF-8 '\n' pad).
    raw_slice = raw[cur_before:cur_after]
    assert delivered_book.startswith(raw_slice), 'Book bytes were skipped or disjoint between ambient and holdout!'
    assert len(raw_slice) <= len(delivered_book) <= len(raw_slice) + 3

    # Next ambient read continues seamlessly from cur_after with zero gap.
    r_next, _, _ = port.take_ambient(32, 16)
    assert bytes(r_next).startswith(raw[cur_after:cur_after + 20])


def test_long_mother_reply_utf8_safe_truncation_and_zero_book_body_consumption(tmp_path: Path):
    cn_path, raw = _make_cn_corpus(tmp_path)
    base = _ArrayBaseWorld(b'English fallback text ' * 400)
    long_reply = '这是一段很长的妈妈回应，包含很多完整的中文汉字，用来测试截断后不裂字符并且书页游标不跳过正文。' * 4
    cg = _FixedCaregiver(long_reply)
    port = UnifiedWorldPort(
        base, caregiver=cg, sidecar_corpus=cn_path, sidecar_mix_ratio=1.0,
        mother_cooldown=0, mother_max_gap=1,
    )
    port.take_ambient(32, 16)
    cur_before = port.sidecar.cursor

    resp, cont, info = port.take('妈妈'.encode('utf-8'), 128, 64)
    assert len(resp) == 128 and len(cont) == 64
    # Must decode cleanly with zero replacement characters (no split UTF-8 byte!).
    decoded_resp = bytes(resp).decode('utf-8', errors='strict')
    assert decoded_resp.startswith('妈妈：')
    # Holdout begins right at cur_before (zero skipped bytes!).
    cur_after = port.sidecar.cursor
    raw_slice = raw[cur_before:cur_after]
    assert bytes(cont).startswith(raw_slice[:max(0, len(raw_slice) - 3)])
    assert cur_after - cur_before <= 64 + info['book_body_bytes']


def test_roster_notice_on_book_turn_preserves_zero_skip_and_restore_roundtrip(tmp_path: Path):
    cn_path, raw = _make_cn_corpus(tmp_path)
    base = _ArrayBaseWorld(b'English fallback text ' * 400)
    cg = _FixedCaregiver('好的。')
    port = UnifiedWorldPort(base, caregiver=cg, sidecar_corpus=cn_path, sidecar_mix_ratio=1.0)
    port.take_ambient(32, 16)
    # Trigger a roster dirty notice (`[在场] 书页(...)\n`) by Mother stepping away.
    port.leave_scene('mother')
    cur_before = port.sidecar.cursor

    resp, cont, info = port.take(b'', 128, 64)
    assert len(resp) == 128 and len(cont) == 64
    pref = int(info['envelope_prefix_bytes'])
    assert pref > 0
    assert info['envelope_mask_spans'] == [(0, pref)]
    assert info['book_body_bytes'] == 128 - pref
    delivered_book = bytes(resp[pref:]) + bytes(cont)
    cur_after = port.sidecar.cursor
    assert delivered_book.startswith(raw[cur_before:cur_after])

    # State-dict roundtrip preserves exact sidecar cursor.
    saved = port.state_dict()
    port2 = UnifiedWorldPort(
        _ArrayBaseWorld(b'English fallback text ' * 400),
        caregiver=None, sidecar_corpus=cn_path, sidecar_mix_ratio=1.0,
    )
    port2.load_state_dict(saved)
    assert port2.sidecar.cursor == cur_after
    r2, _, _ = port2.take_ambient(32, 16)
    assert bytes(r2).startswith(raw[cur_after:cur_after + 20])


def test_parent_inbox_and_tool_preserve_book_continuity_on_base_world():
    base_text = ''.join(f'Sentence {i:03d} about birds and rivers.\n' for i in range(60)).encode('utf-8')
    base = _ArrayBaseWorld(base_text)
    port = UnifiedWorldPort(base, caregiver=None)
    port.register_tool('calc', lambda arg: '42 is the answer')

    # 1. Tool turn on base world (wide enough for roster notice + tool reply + book tail).
    cur0 = base.cursor
    resp_t, cont_t, info_t = port.take(b'@\xe5\xb7\xa5\xe5\x85\xb7:calc 6*7', 160, 32)
    assert len(resp_t) == 160 and len(cont_t) == 32
    spans_t = info_t['envelope_mask_spans']
    book_tail_t = bytes(resp_t[spans_t[-1][1]:])
    assert len(book_tail_t) == info_t['book_body_bytes'] > 0
    assert book_tail_t + bytes(cont_t) == base_text[cur0:base.cursor]

    # 2. Long parent inbox message leaving 0 book_body_bytes on base world.
    cur1 = base.cursor
    port.enqueue_participant('P' * 150, speaker='爸爸', role='parent')
    resp_p, cont_p, info_p = port.take(b'', 96, 32)
    assert len(resp_p) == 96 and len(cont_p) == 32
    assert info_p['book_body_bytes'] == 0
    assert bytes(cont_p) == base_text[cur1:cur1 + 32]
    assert base.cursor == cur1 + 32


def test_phase_space_energy_three_ledger_and_shift_invariant_free_energy():
    psb1 = PhaseSpaceBoltzmann(latent=8)
    psb2 = PhaseSpaceBoltzmann(latent=8)
    # Legacy strict state_dict compatibility check: zero unexpected keys added.
    sd = psb1.state_dict()
    psb2.load_state_dict(sd, strict=True)

    p1 = [torch.full((4, 4), 0.5)]
    p2 = [torch.full((4, 4), 0.5)]
    eng1 = SimpleNamespace(bfss_momenta=p1, model=SimpleNamespace(_momentum_current=p1))
    eng2 = SimpleNamespace(bfss_momenta=p2, model=SimpleNamespace(_momentum_current=p2))

    # Prime step 0.
    base_logits = torch.randn(1, 16, 256)
    shift = torch.linspace(-5.0, 12.0, 16).view(1, 16, 1)
    r1_0 = psb1.apply_free_energy_thermostat(eng1, base_logits)
    r2_0 = psb2.apply_free_energy_thermostat(eng2, base_logits + shift)
    # Raw free_energy shifts, while centered_free_energy, kl_to_uniform, entropy, pressure, scale are identical!
    assert abs(r1_0['free_energy'] - r2_0['free_energy']) > 1.0
    assert abs(r1_0['centered_free_energy'] - r2_0['centered_free_energy']) < 1e-5
    assert abs(r1_0['kl_to_uniform'] - r2_0['kl_to_uniform']) < 1e-5
    assert abs(r1_0['entropy'] - r2_0['entropy']) < 1e-5
    assert abs(r1_0['thermostat_scale'] - r2_0['thermostat_scale']) < 1e-6

    # Simulate backbone momentum injection (x1.3) between step 0 and step 1.
    p1[0].mul_(1.3)
    p2[0].mul_(1.3)
    logits_hot = base_logits * 0.5  # higher entropy -> positive pressure
    r1_1 = psb1.apply_free_energy_thermostat(eng1, logits_hot)
    r2_1 = psb2.apply_free_energy_thermostat(eng2, logits_hot + 9.25)
    assert abs(r1_1['centered_free_energy'] - r2_1['centered_free_energy']) < 1e-5
    assert abs(r1_1['thermostat_scale'] - r2_1['thermostat_scale']) < 1e-6

    # Verify exact three-ledger accounting: Δlog E_total == Δlog E_body + Δlog E_thermostat
    # and Δu == Δlog E_total - Δlog E_ema.
    led = r1_1['energy_ledger']
    assert led['source'] == 'thermostat'
    assert abs(led['delta_log_e_body'] - 2.0 * math.log(1.3)) < 1e-4
    assert abs((led['delta_log_e_body'] + led['delta_log_e_thermostat']) - led['delta_log_e_total']) < 1e-4
    assert abs((led['delta_log_e_total'] - led['delta_log_e_ema']) - led['delta_u']) < 1e-6

    # Simulate an executive boundary refresh (no thermostat scaling).
    p1[0].mul_(0.8)
    psb1.refresh_live_phase_state(eng1, update_ema=True)
    diag = psb1.diagnostics()
    led2 = diag['energy_ledger']
    assert led2['source'] == 'boundary_refresh'
    assert led2['delta_log_e_thermostat'] == 0.0
    assert abs(led2['delta_log_e_body'] - 2.0 * math.log(0.8)) < 1e-4
    assert abs((led2['delta_log_e_total'] - led2['delta_log_e_ema']) - led2['delta_u']) < 1e-6
    assert diag['last_centered_free_energy'] is not None
    assert diag['last_kl_to_uniform'] is not None


def test_wood_policy_measurement_score_supports_paired_chunk_sizes(monkeypatch):
    seen_chunks: list[int] = []

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(4))
            self.max_chunk = 1024
            self.memory = SimpleNamespace(clock=7)
            self._causal_archive = SimpleNamespace(db=SimpleNamespace(total_changes=3))
            self._question_context = None

        def predict(self, ids, state, logp):
            seen_chunks.append(int(ids.shape[1]))
            # Deterministic logits equal to logp so NLL is finite and fast.
            return logp.clone(), state, None

    dummy_table = SimpleNamespace(history=11)
    monkeypatch.setattr(
        'scripts.ib_local.wood_policy_measurement.frozen_table_logp',
        lambda table, arr: np.full((len(arr), 256), -math.log(256.0), dtype=np.float32),
    )
    engine = SimpleNamespace(
        model=_DummyModel(),
        table=dummy_table,
        recurrent=torch.zeros(1, 4),
        device=torch.device('cpu'),
    )
    material = [(np.arange(0, 701, dtype=np.int64) % 256).astype(np.uint8)]

    res128 = measure_score(engine, material, chunk_size=128)
    chunks_128 = list(seen_chunks)
    seen_chunks.clear()

    res640 = measure_score(engine, material, chunk_size=640)
    chunks_640 = list(seen_chunks)

    assert res128['chunk_size'] == 128
    assert res640['chunk_size'] == 640
    assert chunks_128 == [128, 128, 128, 128, 128, 60]
    assert chunks_640 == [640, 60]
    assert abs(res128['mean_nll'] - math.log(256.0)) < 1e-5
    assert abs(res640['mean_nll'] - math.log(256.0)) < 1e-5
