"""Tests for orthogonal `(u, out, in)` dual-channel interaction & symmetric Mother directionality."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ib_local import gaze_pointer as gp_mod
from scripts.ib_local.gaze_pointer import GazeObjectPointer, classify_interaction_mode
from scripts.ib_local.unified_world_port import UnifiedWorldPort, parse_mother_direction


class _FakeBaseWorld:
    """Minimal deterministic storybook world for CPU unit tests."""

    def __init__(self, pages: list[str] | None = None):
        if pages is None:
            pages = [
                '第一章：清晨的阳光照在安静的小院里，风轻轻吹过树梢，露珠在叶子上闪闪发亮。' * 20,
                '第二章：公园里有一只金色羽毛的小鸟，它在绿色的云朵树下快乐地唱歌和拍翅膀。' * 20,
                '第三章：夜空中的月亮像弯弯的小船，小兔子坐在河边数着亮晶晶的星星。' * 20,
            ]
        self._pages = [p.encode('utf-8') for p in pages]
        self._raw = b'\n'.join(self._pages)
        self.data = np.frombuffer(self._raw, dtype=np.uint8)
        self.seen = np.zeros(len(self.data), dtype=bool)
        self.cursor = 0
        self.navigate_calls: list[bytes] = []
        self.passive_calls = 0

    def take(self, utterance: bytes, response_bytes: int, holdout_bytes: int):
        req = int(response_bytes + holdout_bytes)
        if utterance:
            self.navigate_calls.append(bytes(utterance))
            q = utterance.decode('utf-8', errors='ignore').strip()
            # Simple keyword jump across pages
            hit_pos = None
            for idx, page in enumerate(self._pages):
                if q and q in page.decode('utf-8', errors='ignore'):
                    hit_pos = sum(len(p) + 1 for p in self._pages[:idx])
                    break
            if hit_pos is not None:
                self.cursor = hit_pos
                matched = 1
            else:
                matched = 0
        else:
            self.passive_calls += 1
            matched = 0

        start = int(self.cursor) % len(self.data)
        buf = np.concatenate((self.data[start:], self.data))[:req].copy()
        self.cursor = (start + req) % len(self.data)
        resp = buf[:response_bytes]
        cont = buf[response_bytes:req]
        return resp, cont, dict(
            source='text',
            start=start,
            match=matched,
            text_reply=bytes(resp[:160]).decode('utf-8', errors='ignore'),
            response_previously_seen_bytes=0,
            holdout_previously_seen_bytes=0,
            holdout_position_fresh=True,
        )

    def state_dict(self):
        return dict(cursor=int(self.cursor))

    def load_state_dict(self, st):
        self.cursor = int(st.get('cursor', 0))


class _FakeCaregiver:
    """Scripted Mother mock that records every received packet and returns planned replies."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[list[tuple[str, str]]] = []
        self.committed = 0
        self.last_telemetry = dict(source='mock', api_usage=dict(prompt_tokens=10, cache_hit_tokens=8, cache_miss_tokens=2))

    def begin_turn(self, **kwargs):
        pass

    def reply(self, turn):
        self.calls.append(copy.deepcopy(turn))
        if not self._replies:
            return '嗯，妈妈在这里。'
        return self._replies.pop(0)

    def commit_turn(self):
        self.committed += 1


class _TinyBody(nn.Module):
    def __init__(self, dim: int = 16, latent: int = 8):
        super().__init__()
        self.encoder = nn.Linear(dim, latent)
        self.body = SimpleNamespace(embed=nn.Embedding(256, dim))
        self.long_predictor = nn.Linear(latent * 2, latent)

    def mother_from_boundary(self, pred):
        return pred


def _make_fake_engine(dim: int = 16, latent: int = 8):
    torch.manual_seed(7)
    model = _TinyBody(dim=dim, latent=latent)
    engine = SimpleNamespace(
        model=model,
        device=torch.device('cpu'),
        recurrent=torch.randn(1, dim),
        _internal_source=torch.randn(1, dim),
        _internal_action=torch.randn(2, dim),
        alif_executive_state=dict(nll_ema=2.0, last_piece_nll=2.3, last_burst_nll=2.5),
    )
    gp_mod.configure(engine, {}, d_gaze=8)
    return engine


def test_parse_mother_direction_variants():
    assert parse_mother_direction('@安静陪伴') == ('none', '', '')
    assert parse_mother_direction('  @安静陪伴。 ') == ('none', '', '')
    assert parse_mother_direction('@quiet') == ('none', '', '')

    out, q, spoken = parse_mother_direction('@翻书:公园里小鸟 来看看公园里的小鸟！')
    assert out == 'book'
    assert q == '公园里小鸟'
    assert spoken == '来看看公园里的小鸟！'
    assert '@翻书' not in spoken

    out2, q2, spoken2 = parse_mother_direction('@翻书：月亮')
    assert out2 == 'book'
    assert q2 == '月亮'
    assert spoken2 == ''

    out3, q3, spoken3 = parse_mother_direction('@对爸爸 她今天读得很开心。')
    assert out3 == 'parent'
    assert q3 == ''
    assert spoken3 == '她今天读得很开心。'

    out4, q4, spoken4 = parse_mother_direction('@对梦漓 我们在聊天呢！')
    assert out4 == 'mengli'
    assert q4 == ''
    assert spoken4 == '我们在聊天呢！'

    out5, q5, spoken5 = parse_mother_direction('今天天气真好呀！')
    assert out5 == 'mengli'
    assert q5 == ''
    assert spoken5 == '今天天气真好呀！'


def test_classify_interaction_mode_covers_full_grid():
    assert classify_interaction_mode(True, 'mother', 'mother') == 'face_to_face_chat'
    assert classify_interaction_mode(True, 'mother', 'book') == 'co_reading_talk'
    assert classify_interaction_mode(True, 'book', 'book') == 'book_search'
    assert classify_interaction_mode(True, 'none', 'none') == 'inner_monologue'
    assert classify_interaction_mode(True, 'none', 'book') == 'read_aloud_solo'
    assert classify_interaction_mode(True, 'none', 'mother') == 'self_talk_observing_mother'
    assert classify_interaction_mode(False, 'none', 'none') == 'quiet_self_evolution'
    assert classify_interaction_mode(False, 'none', 'mother') == 'look_at_mother'
    assert classify_interaction_mode(False, 'none', 'book') == 'quiet_reading'
    assert classify_interaction_mode(True, 'mother', 'none') == 'call_mother_and_pause'
    assert classify_interaction_mode(True, 'book', 'mother') == 'read_aloud_to_mother'


def test_pure_self_talk_and_quiet_self_evolution_zero_external_pull():
    base = _FakeBaseWorld()
    cg = _FakeCaregiver(['不应该被调用'])
    port = UnifiedWorldPort(base, caregiver=cg, auto_commit=True)

    start_cursor = base.cursor
    # 1. Pure self-talk: u != b'', out='none', in='none'
    resp, cont, info = port.take(
        '小鸟飞呀飞'.encode('utf-8'),
        256,
        256,
        planned_target=dict(send_target='none', observe_target='none', target_id='none', sampled=True),
    )
    assert len(resp) == 0 and len(cont) == 0
    assert info['self_evolution_beat'] is True
    assert info['interaction_mode'] == 'inner_monologue'
    assert info['send_target_id'] == 'none'
    assert info['observe_target_id'] == 'none'
    assert info['channel'] == 'inner_monologue'
    assert len(cg.calls) == 0
    assert base.cursor == start_cursor
    assert len(base.navigate_calls) == 0
    assert port.self_evolution_turns == 1
    assert port.inner_monologue_turns == 1

    # 2. Pure quiet self-evolution: u == b'', out='none', in='none'
    resp2, cont2, info2 = port.take(
        b'',
        256,
        256,
        planned_target=dict(send_target='none', observe_target='none', target_id='none', sampled=True),
    )
    assert len(resp2) == 0 and len(cont2) == 0
    assert info2['self_evolution_beat'] is True
    assert info2['interaction_mode'] == 'quiet_self_evolution'
    assert info2['channel'] == 'self_evolution'
    assert len(cg.calls) == 0
    assert base.cursor == start_cursor
    assert port.self_evolution_turns == 2
    assert port.inner_monologue_turns == 1


def test_face_to_face_chat_vs_co_reading_packet_hygiene():
    base = _FakeBaseWorld()
    cg = _FakeCaregiver(['我们聊聊今天开心的事吧！', '这页上的小院真安静呢。'])
    port = UnifiedWorldPort(base, caregiver=cg, auto_commit=True)
    # Seed shared_page_bytes so [共读书页] WOULD appear unless explicitly omitted.
    port.take_ambient(256, 256)
    assert len(port.shared_page_bytes) > 0

    # 1. Pure face-to-face chat: (spoke, out='mother', in='mother') -> NO [共读书页]!
    resp1, cont1, info1 = port.take(
        '我想听你说说话'.encode('utf-8'),
        256,
        256,
        planned_target=dict(send_target='mother', observe_target='mother', target_id='mother', sampled=True),
    )
    assert info1['interaction_mode'] == 'face_to_face_chat'
    assert info1['include_book_excerpt'] is False
    assert info1['mother_out'] == 'mengli'
    packet1 = cg.calls[0][-1][1]
    assert '[共读书页]' not in packet1
    assert '面对面聊天' in packet1
    assert '[行动指向]' in packet1

    # 2. Genuine co-reading: (spoke, out='mother', in='book') -> [共读书页] INCLUDED!
    resp2, cont2, info2 = port.take(
        '树梢上的露珠'.encode('utf-8'),
        256,
        256,
        planned_target=dict(send_target='mother', observe_target='book', target_id='mother', sampled=True),
    )
    assert info2['interaction_mode'] == 'co_reading_talk'
    assert info2['include_book_excerpt'] is True
    packet2 = cg.calls[1][-1][1]
    assert '[共读书页]' in packet2


def test_read_aloud_solo_does_not_jump_book_while_book_search_does():
    base = _FakeBaseWorld()
    cg = _FakeCaregiver([])
    port = UnifiedWorldPort(base, caregiver=cg, auto_commit=True)

    # 1. (spoke, out='none', in='book') -> read_aloud_solo: ZERO book search jump!
    _, _, info_solo = port.take(
        '月亮'.encode('utf-8'),
        256,
        256,
        planned_target=dict(send_target='none', observe_target='book', target_id='book', sampled=True),
    )
    assert info_solo['interaction_mode'] == 'read_aloud_solo'
    assert info_solo['channel'] == 'read_aloud_solo'
    assert bool(info_solo.get('text_reply'))
    assert bool(info_solo.get('book_excerpt'))
    assert len(base.navigate_calls) == 0
    assert base.passive_calls == 1
    assert len(cg.calls) == 0

    # 2. (spoke, out='book', in='book') -> book_search: DOES search & jump to 月亮 page!
    resp_s, _, info_s = port.take(
        '月亮'.encode('utf-8'),
        256,
        256,
        planned_target=dict(send_target='book', observe_target='book', target_id='book', sampled=True),
    )
    assert info_s['interaction_mode'] == 'book_search'
    assert info_s['channel'] == 'book_navigation'
    assert '月亮' in (info_s.get('text_reply') or '')
    assert len(base.navigate_calls) == 1
    assert '月亮' in bytes(resp_s).decode('utf-8', errors='ignore')


def test_mother_symmetric_directives_flip_book_quiet_and_parent():
    base = _FakeBaseWorld()
    cg = _FakeCaregiver([
        '@翻书:公园里 我们一起看看公园里唱歌的小鸟！',
        '@安静陪伴',
        '@对爸爸 她刚刚自己找到月亮那页了。',
    ])
    port = UnifiedWorldPort(base, caregiver=cg, auto_commit=True)
    port.presence_roster['parent'] = dict(kind='parent', label='爸爸', description='爸爸')

    # 1. Mother flips book via `@翻书:公园里` and tag is STRIPPED before MengLi's Membrane!
    resp1, _, info1 = port.take(
        b'',
        320,
        256,
        planned_target=dict(send_target='none', observe_target='mother', target_id='mother', sampled=True),
    )
    decoded1 = bytes(resp1).decode('utf-8', errors='ignore')
    assert info1['mother_out'] == 'book'
    assert info1['mother_book_query'] == '公园里'
    assert '@翻书' not in decoded1
    assert '我们一起看看公园里唱歌的小鸟！' in decoded1
    assert '公园里有一只金色羽毛的小鸟' in decoded1
    assert port.mother_book_turns == 1

    # 2. Mother chooses `@安静陪伴`: ZERO spoken Mother bytes injected!
    resp2, _, info2 = port.take(
        '嗯嗯'.encode('utf-8'),
        256,
        256,
        planned_target=dict(send_target='none', observe_target='mother', target_id='mother', sampled=True),
    )
    assert info2['mother_out'] == 'none'
    assert info2['channel'] == 'mother_quiet_presence'
    assert info2['mother_bytes'] == 0
    assert info2['mother_reply'] == ''
    assert '@安静陪伴' not in bytes(resp2).decode('utf-8', errors='ignore')
    assert port.mother_quiet_turns == 1

    # 3. Mother addresses Dad via `@对爸爸`: addressee_id='parent', is_action_consequence=False!
    resp3, _, info3 = port.take(
        b'',
        256,
        256,
        planned_target=dict(send_target='none', observe_target='mother', target_id='mother', sampled=True),
    )
    decoded3 = bytes(resp3).decode('utf-8', errors='ignore')
    assert info3['mother_out'] == 'parent'
    assert info3['speaker_id'] == 'mother'
    assert info3['addressee_id'] == 'parent'
    assert info3['is_action_consequence'] is False
    assert '@对爸爸' not in decoded3
    assert '她刚刚自己找到月亮那页了。' in decoded3


def test_gaze_pointer_dual_channel_selection_training_and_checkpoint_roundtrip():
    engine = _make_fake_engine()
    gp: GazeObjectPointer = engine.model.gaze_pointer
    candidates = [('book', '书页故事'.encode('utf-8')), ('mother', '妈妈陪读'.encode('utf-8'))]

    # Select dual interaction `(send_target, observe_target)` over `(O ∪ {'none'})²`
    dec = gp.select_interaction(
        engine,
        candidates=candidates,
        utterance_bytes='小鸟'.encode('utf-8'),
        causal_values=dict(book=0.2, mother=0.8),
        stumble_ratio=1.2,
        nll_norm=0.1,
    )
    assert dec['send_target'] in ('book', 'mother', 'none')
    assert dec['observe_target'] in ('book', 'mother', 'none')
    assert abs(sum(dec['send_probs'].values()) - 1.0) < 1e-3
    assert abs(sum(dec['observe_probs'].values()) - 1.0) < 1e-3
    assert set(dec['send_probs'].keys()) == {'book', 'mother', 'none'}
    assert set(dec['observe_probs'].keys()) == {'book', 'mother', 'none'}

    # Settle credit with boundary context -> trains BOTH send and observe channels + target/observe_to_action!
    b_ctx = dict(
        model=engine.model,
        source=engine._internal_source[0],
        raw_action=engine._internal_action[0],
        actual_target_id=dec['send_target'],
        actual_observe_id=dec['observe_target'],
        boundary_target=torch.randn(16),
        evidence_available=True,
    )
    receipt = gp.settle_credit(0.05, evidence_available=True, boundary_context=b_ctx)
    assert receipt is not None
    assert receipt['step_count'] == 1
    assert receipt['send_target'] == dec['send_target']
    assert receipt['observe_target'] == dec['observe_target']
    assert engine.model.encoder.weight.grad is None

    # Verify strict=True PyTorch state_dict compatibility (legacy models load with zero missing/unexpected keys!)
    legacy_keys = set(gp.state_dict().keys())
    assert not any('_dual_channel' in k for k in legacy_keys)

    # Verify bit-exact export/load of dual channel weights + Adam state!
    snap = gp.export_dynamic_state()
    assert snap['schema'] == 2
    assert snap['dual_channel_v1'] is True
    assert 'dual_weights_b64' in snap
    assert 'dual_optimizer_state_b64' in snap

    engine2 = _make_fake_engine()
    engine2.model.gaze_pointer.enable_other_world = True
    engine2.model.load_state_dict(engine.model.state_dict(), strict=True)
    engine2.model.gaze_pointer.load_dynamic_state(snap)

    # Legacy schema-2 checkpoint migration (without dual_channel_v1) must also succeed cleanly!
    legacy_snap = copy.deepcopy(snap)
    legacy_snap.pop('dual_channel_v1', None)
    legacy_snap.pop('dual_weights_b64', None)
    legacy_snap.pop('dual_optimizer_state_b64', None)
    engine3 = _make_fake_engine()
    engine3.model.gaze_pointer.load_dynamic_state(legacy_snap)

    # Next-step forward equivalence between original and reloaded dual channel (< 1e-6)
    torch.manual_seed(99)
    d1 = gp.select_interaction(engine, candidates=candidates, utterance_bytes=b'hello')
    torch.manual_seed(99)
    d2 = engine2.model.gaze_pointer.select_interaction(engine2, candidates=candidates, utterance_bytes=b'hello')
    assert d1['send_target'] == d2['send_target']
    assert d1['observe_target'] == d2['observe_target']
    for k in ('book', 'mother', 'none'):
        assert abs(d1['send_probs'][k] - d2['send_probs'][k]) < 1e-6
        assert abs(d1['observe_probs'][k] - d2['observe_probs'][k]) < 1e-6

    # Verify bit-exact next-step Adam update equivalence across BOTH _opt and _dual_opt!
    b2 = dict(
        model=engine.model,
        source=engine._internal_source[0],
        raw_action=engine._internal_action[0],
        actual_target_id=d1['send_target'],
        actual_observe_id=d1['observe_target'],
        boundary_target=torch.ones(16) * 0.25,
        evidence_available=True,
    )
    b2_copy = dict(b2, model=engine2.model)
    gp.settle_credit(0.08, evidence_available=True, boundary_context=b2)
    engine2.model.gaze_pointer.settle_credit(0.08, evidence_available=True, boundary_context=b2_copy)
    for (n1, p1), (n2, p2) in zip(gp._dual().named_parameters(), engine2.model.gaze_pointer._dual().named_parameters()):
        assert n1 == n2
        assert float((p1 - p2).abs().max().item()) < 1e-6

    # Verify closed-loop boundary supervision STILL trains `observe_to_action` when `out == 'none'` & `in == 'mother'`!
    before_obs_w = gp._dual().observe_to_action.weight.detach().clone()
    d_silent = gp.select_interaction(
        engine, candidates=candidates, utterance_bytes=b'',
        causal_values=dict(book=0.1, mother=0.9),
        observe_causal_values=dict(book=0.2, mother=0.8),
    )
    assert d_silent['send_target'] == 'none'
    b_obs_only = dict(
        model=engine.model,
        source=engine._internal_source[0],
        raw_action=engine._internal_action[0],
        actual_target_id='none',
        actual_observe_id='mother',
        boundary_target=torch.randn(16),
        evidence_available=True,
    )
    r_obs = gp.settle_credit(0.06, evidence_available=True, boundary_context=b_obs_only)
    assert r_obs is not None
    assert r_obs['boundary_mse'] is not None
    assert float((gp._dual().observe_to_action.weight.detach() - before_obs_w).abs().max().item()) > 0.0


def test_other_world_model_persistent_state_bptt_and_roundtrip():
    torch.manual_seed(7)
    engine = _make_fake_engine()
    gp: GazeObjectPointer = engine.model.gaze_pointer
    assert gp.enable_other_world is False
    candidates = [('book', 'Once upon a time'.encode('utf-8')), ('mother', '妈妈在这里陪你'.encode('utf-8'))]
    for cid, intro in candidates:
        gp.ensure_object(engine, cid, intro)

    src = torch.randn(16)
    act = src + 0.5 * torch.randn(16)
    tgt_m = torch.ones(16) * 0.6

    # Verify default OFF (`enable_other_world=False`) NEVER instantiates, updates, or affects selection!
    r_off = gp.settle_credit(
        0.02,
        evidence_available=True,
        boundary_context=dict(
            model=engine.model, source=src, raw_action=act,
            actual_target_id='mother', actual_observe_id='mother',
            boundary_target=tgt_m, evidence_available=True,
        ),
    )
    assert r_off is not None and r_off['other_model'] is None
    assert gp.__dict__.get('_other_world_model') is None
    assert gp.other_states == {}
    assert gp.other_model_ema == {}
    assert 'other_world_v1' not in gp.export_dynamic_state()

    # Now enable opt-in candidate (`enable_other_world=True`): step 0 must be exact zero-shock!
    gp.enable_other_world = True
    r0 = gp.settle_credit(
        0.02,
        evidence_available=True,
        boundary_context=dict(
            model=engine.model, source=src, raw_action=act,
            actual_target_id='mother', actual_observe_id='mother',
            boundary_target=tgt_m, evidence_available=True,
        ),
    )
    assert r0 is not None and r0['other_model'] is not None
    assert r0['other_model']['object_id'] == 'mother'
    assert abs(r0['other_model']['gain']) < 1e-6
    assert abs(r0['other_model']['gate_mean'] - 0.5) < 0.05
    assert r0['other_model']['action_shift'] > 0.0

    # Train 25 steps on a smoothly evolving persistent Mother trajectory vs a distinct Book target.
    # Verify `_OtherWorldModel` learns positive prequential MSE gain and updates `gate_net` via 1-step BPTT.
    owm = gp.__dict__['_other_world_model']
    assert owm is not None
    gate_w0 = owm.gate_net.weight.detach().clone()
    last_m_rec = None
    last_b_rec = None
    for step in range(25):
        m_tgt = torch.ones(16) * (0.6 + 0.01 * (step % 3))
        b_tgt = torch.ones(16) * (-0.5)
        gp.select_interaction(engine, candidates=candidates, utterance_bytes=b'hi')
        last_m_rec = gp.settle_credit(
            0.03,
            evidence_available=True,
            boundary_context=dict(
                model=engine.model, source=src, raw_action=act,
                actual_target_id='mother', actual_observe_id='mother',
                boundary_target=m_tgt, evidence_available=True,
            ),
        )
        gp.select_interaction(engine, candidates=candidates, utterance_bytes=b'hi')
        last_b_rec = gp.settle_credit(
            0.01,
            evidence_available=True,
            boundary_context=dict(
                model=engine.model, source=src, raw_action=act,
                actual_target_id='book', actual_observe_id='book',
                boundary_target=b_tgt, evidence_available=True,
            ),
        )
    assert last_m_rec['other_model']['gain'] > 0.0
    assert last_b_rec['other_model']['gain'] > 0.0
    assert float((owm.gate_net.weight.detach() - gate_w0).abs().max().item()) > 0.0
    assert engine.model.encoder.weight.grad is None

    # Verify strict=True PyTorch state_dict compatibility & bit-exact export/reload!
    assert not any('_other_world_model' in k for k in gp.state_dict().keys())
    snap = gp.export_dynamic_state()
    assert snap['other_world_v1'] is True
    assert 'mother' in snap['other_states_b64'] and 'book' in snap['other_states_b64']

    engine2 = _make_fake_engine()
    engine2.model.gaze_pointer.enable_other_world = True
    engine2.model.load_state_dict(engine.model.state_dict(), strict=True)
    engine2.model.gaze_pointer.load_dynamic_state(snap)

    # Next-step Adam update equivalence across `_other_world_model` (< 1e-6)!
    b_next = dict(
        model=engine.model, source=src, raw_action=act,
        actual_target_id='mother', actual_observe_id='mother',
        boundary_target=torch.ones(16) * 0.62, evidence_available=True,
    )
    r_a = gp.settle_credit(0.04, evidence_available=True, boundary_context=b_next)
    r_b = engine2.model.gaze_pointer.settle_credit(0.04, evidence_available=True, boundary_context=dict(b_next, model=engine2.model))
    assert abs(r_a['other_model']['pred_mse'] - r_b['other_model']['pred_mse']) < 1e-6
    for (n1, p1), (n2, p2) in zip(
        gp.__dict__['_other_world_model'].named_parameters(),
        engine2.model.gaze_pointer.__dict__['_other_world_model'].named_parameters(),
    ):
        assert n1 == n2
        assert float((p1 - p2).abs().max().item()) < 1e-6


def test_clean_english_sidecar_routing_and_cursor_roundtrip(tmp_path):
    en_bin = tmp_path / 'clean_en.bin'
    en_bin.write_bytes(('Once upon a time a little bird sang in the sunny garden.\n' * 80).encode('utf-8'))
    base = _FakeBaseWorld()
    port = UnifiedWorldPort(base, caregiver=None, english_corpus=en_bin)
    resp, cont, info = port.take_ambient(128, 128)
    assert info['source'] == 'text'
    assert b'little bird' in bytes(resp) or b'sunny garden' in bytes(resp)
    st = port.state_dict()
    assert st['mother_unified_port']['english_sidecar_cursor'] == port.english_sidecar.cursor
    port2 = UnifiedWorldPort(_FakeBaseWorld(), caregiver=None, english_corpus=en_bin)
    port2.load_state_dict(st)
    assert port2.english_sidecar.cursor == port.english_sidecar.cursor


def test_self_cons_holdout_exclusion_and_scale_invariance():
    from scripts.ib_local.alif_executive import hippocampal_anchor_loss
    from scripts.ib_local.navigation_internal_rollout import InternalRolloutEngine

    # 1. Verify normalized consolidation MSE is 100% invariant to encoder scale shrinkage (`w *= 0.5`).
    torch.manual_seed(11)
    pred = torch.randn(16, 8)
    tgt = torch.randn(16, 8)
    raw_pre = float(torch.nn.functional.mse_loss(pred, tgt).item())
    raw_shrunk = float(torch.nn.functional.mse_loss(0.5 * pred, 0.5 * tgt).item())
    assert raw_pre - raw_shrunk > 0.1  # raw MSE fakes a gain under 0.5x scale shrinkage!
    norm_pre = float(torch.nn.functional.mse_loss(
        torch.nn.functional.normalize(pred, dim=-1), torch.nn.functional.normalize(tgt, dim=-1),
    ).item())
    norm_shrunk = float(torch.nn.functional.mse_loss(
        torch.nn.functional.normalize(0.5 * pred, dim=-1), torch.nn.functional.normalize(0.5 * tgt, dim=-1),
    ).item())
    assert abs(norm_pre - norm_shrunk) < 1e-6  # scale-invariant normalized MSE yields exact 0.0 fake gain!

    # 2. Verify `exclude_internal_slots` strictly excludes holdout slots from `hippocampal_anchor_loss`.
    n_slots, d_in, d_lat = 20, 16, 8
    mem = SimpleNamespace(
        occupied=torch.ones(n_slots, dtype=torch.bool),
        internal_valid=torch.ones(n_slots, dtype=torch.bool),
        count=torch.ones(n_slots, dtype=torch.long),
        internal_stamp=torch.ones(n_slots, dtype=torch.long),
        last_write=torch.ones(n_slots, dtype=torch.long),
        source=torch.randn(n_slots, d_in),
        action=torch.randn(n_slots, d_in),
        internal_next=torch.randn(n_slots, d_in),
    )
    model = SimpleNamespace(
        encoder=nn.Linear(d_in, d_lat, bias=False),
        long_predictor=nn.Linear(d_lat * 2, d_lat, bias=False),
        memory=mem,
    )
    # Exclude ALL slots except 2 -> `internal_transition_samples` MUST equal 2!
    excl = torch.arange(2, n_slots, dtype=torch.long)
    eng = SimpleNamespace(
        model=model,
        replay_batch=32,
        replay_rng=np.random.default_rng(0),
        alif_executive_state=dict(piece_bytes=640),
    )
    loss, info = hippocampal_anchor_loss(eng, torch.randn(1, d_in), exclude_internal_slots=excl)
    assert info['internal_transition_samples'] == 2
    assert float(loss.item()) >= 0.0

    # 3. Verify `InternalRolloutEngine._outcome_loss` also strictly respects `_exclude_internal_slots`.
    from unittest.mock import patch
    parent_cls = InternalRolloutEngine.__mro__[1]
    stub = InternalRolloutEngine.__new__(InternalRolloutEngine)
    stub.model = model
    stub.device = torch.device('cpu')
    stub.replay_batch = 32
    stub.replay_rng = np.random.default_rng(0)
    stub._internal_label = None
    stub.internal_pending = None
    stub._exclude_internal_slots = excl
    with patch.object(parent_cls, '_outcome_loss', return_value=(torch.zeros(()), dict(outcome_error=0.0))):
        _, diag = stub._outcome_loss(torch.randn(d_in), torch.randn(d_in), torch.randn(d_in))
    assert diag['internal_transition_samples'] == 2
