"""Tests for pre-linguistic O(1) GazeObjectPointer (Drosophila/infant steering over Umwelt objects)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scripts.ib_local import gaze_pointer as gp_mod
from scripts.ib_local.mengli_harness.family_provider import FamilyCaregiver
from scripts.ib_local.mengli_harness.mengli_world_client import DshMother
from scripts.ib_local.unified_world_port import UnifiedWorldPort
from tests.test_unified_world_port import FakeBaseWorld, FakeDshTransport


def _make_gaze_engine(d_model: int = 32):
    class _FakeSource(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(256, d_model)

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = SimpleNamespace(base=SimpleNamespace(d=d_model, source=_FakeSource()))
            self.phase_boltzmann = SimpleNamespace(diagnostics=lambda: dict(live_z_kin=0.3))

    model = _FakeModel()
    state_vec = torch.randn(1, d_model)
    eng = SimpleNamespace(
        model=model,
        device=torch.device('cpu'),
        alif_executive_state=dict(
            last_burst_bytes_hex='看到'.encode('utf-8').hex(),
            last_burst_nll=2.8,
            nll_ema=1.8,
            last_piece_nll=2.1,
            last_trigger_cause='burst',
        ),
        _read_source=lambda: state_vec,
    )
    return eng


def test_o1_parameters_and_uniform_birth_across_dynamic_objects():
    eng = _make_gaze_engine(d_model=32)
    cfg = {}
    ptr = gp_mod.configure(eng, cfg, d_gaze=16)
    p0 = sum(p.numel() for p in ptr.parameters())
    assert cfg['gaze_pointer']['schema'] == 2

    # Hot-plug M = 2, then M = 5 arbitrary objects -> ZERO parameter growth, exact 1/M uniform birth!
    cands2 = [('book', b'storybook reading'), ('mother', b'mama gentle voice')]
    dec2 = ptr.select_target(eng, candidates=cands2, utterance_bytes=b'apple', rng=np.random.default_rng(1))
    assert abs(dec2['probs']['book'] - 0.5) < 1e-3
    assert abs(dec2['probs']['mother'] - 0.5) < 1e-3

    cands5 = cands2 + [
        ('parent', b'dad voice'),
        ('tool:bash', b'shell terminal'),
        ('weather', b'forecast api'),
    ]
    dec5 = ptr.select_target(eng, candidates=cands5, utterance_bytes=b'apple', rng=np.random.default_rng(2))
    assert len(dec5['probs']) == 5
    for prob in dec5['probs'].values():
        assert abs(prob - 0.2) < 1e-3
    assert sum(p.numel() for p in ptr.parameters()) == p0


def test_gaze_pointer_learns_from_pure_metal_g_without_synthetic_reward():
    torch.manual_seed(7)
    eng = _make_gaze_engine(d_model=32)
    ptr = gp_mod.configure(eng, {}, d_gaze=16, lr=0.04, entropy_coef=0.005)
    rng = np.random.default_rng(123)

    # Give book & mother clearly distinct linguistic fingerprints.
    ptr.observe_object(eng, 'book', b'The quick brown fox jumps over the lazy dog in the meadow.' * 4, fuse_body_state=False)
    ptr.observe_object(eng, 'mother', '慢慢来，不着急，妈妈陪着你一起看故事书里的萤火虫呀。'.encode('utf-8') * 4, fuse_body_state=False)

    cands = [('book', b'book'), ('mother', b'mother')]
    # High-stumble state: turning to Mother yields higher predictive gain (+0.08) than Book (-0.04).
    for _ in range(90):
        dec = ptr.select_target(eng, candidates=cands, utterance_bytes='陌生词'.encode('utf-8'),
                                stumble_ratio=2.2, nll_norm=1.5, rng=rng)
        g = 0.08 if dec['target_id'] == 'mother' else -0.04
        ptr.settle_credit(g)

    final_dec = ptr.select_target(eng, candidates=cands, utterance_bytes='陌生词'.encode('utf-8'),
                                  stumble_ratio=2.2, nll_norm=1.5, rng=rng)
    assert final_dec['probs']['mother'] > 0.75
    assert ptr.target_g_ema['mother'] > ptr.target_g_ema['book']


def test_gaze_pointer_end_to_end_book_search_vs_mother_and_roundtrip(tmp_path: Path):
    torch.manual_seed(42)
    eng = _make_gaze_engine(d_model=32)
    ptr = gp_mod.configure(eng, {}, d_gaze=16)

    transport = FakeDshTransport()
    caregiver = FamilyCaregiver(DshMother(branch_id='test-gaze-e2e', transport=transport))
    cn_file = tmp_path / 'cn_stories.txt'
    cn_file.write_text(
        ('小兔在草地上跳来跳去。太阳暖洋洋的。\n' * 60)
        + '莉莉在院子里看到一只会发光的小萤火虫，高兴地拍手。\n'
        + ('小熊在河边钓到一条亮晶晶的小鱼。\n' * 60),
        encoding='utf-8',
    )
    base = FakeBaseWorld('Apples fall to Earth because of gravity. ' * 200)
    port = UnifiedWorldPort(
        base, caregiver=caregiver, mother_cooldown=1, mother_max_gap=3,
        mirror_ratio=1.0, auto_commit=False,
        sidecar_corpus=cn_file, sidecar_mix_ratio=0.5,
    )
    port.attach_engine(eng)
    port.take_ambient(256, 256)
    port.take_ambient(256, 256)

    # Run 12 spikes with utterance '看到' & silence -> MengLi's GazePointer explores BOTH
    # Book-as-Search-Engine (`gaze_book_search`) AND Mother (`gaze_mother_speak` / `gaze_mother_look`)!
    reasons = []
    for idx in range(12):
        port.set_cycle_hint(300 + idx)
        utt = '看到'.encode('utf-8') if idx % 2 == 0 else b''
        resp, _, info = port.take(utt, 256, 256)
        reasons.append(info['routing_reason'])
        assert info['gaze_decision'] is not None
        if info['routing_reason'] == 'gaze_book_search':
            assert info['channel'] == 'book_navigation'
            assert '萤火虫' in bytes(resp).decode('utf-8', 'ignore')
        port.commit_pending_turn(metal_g=0.02 if info['speaker'] == 'mother' else 0.01)

    assert any(r in ('gaze_book_search', 'gaze_book_read') for r in reasons)
    assert any(r in ('gaze_mother_speak', 'gaze_mother_look') for r in reasons)
    # Every multi-candidate gaze step (spoken turns + post-cooldown silent turns) updates step_count.
    assert int(ptr.step_count.item()) >= 6
    assert int(ptr.step_count.item()) <= 12

    # Exact checkpoint roundtrip of dynamic fingerprints & counts.
    saved = port.state_dict()
    eng2 = _make_gaze_engine(d_model=32)
    ptr2 = gp_mod.configure(eng2, {}, d_gaze=16)
    ptr2.load_state_dict(ptr.state_dict())
    port2 = UnifiedWorldPort(
        FakeBaseWorld('Apples fall to Earth because of gravity. ' * 200),
        caregiver=FamilyCaregiver(DshMother(branch_id='test-gaze-e2e', transport=transport)),
        auto_commit=False, sidecar_corpus=cn_file, sidecar_mix_ratio=0.5,
    )
    port2.load_state_dict(saved)
    port2.attach_engine(eng2)
    assert ptr2.target_counts == ptr.target_counts
    assert set(ptr2.fingerprints.keys()) == set(ptr.fingerprints.keys())
    for k in ptr.fingerprints:
        assert torch.allclose(ptr2.fingerprints[k], ptr.fingerprints[k], atol=1e-5)


def test_unified_response_boundary_all_sources_and_utf8_alignment(tmp_path: Path):
    """Gate 2: ALL sources (chinese_storybook, text, mother, parent, tool, object) get valid boundaries."""
    from scripts.ib_local.navigation_mother_boundary import mother_boundary_feature
    from scripts.ib_local.unified_world_port import SidecarCorpusStream

    feats = torch.randn(1000, 32)
    # Foreground interactive sources: mother, parent, tool, object
    for src in ('mother', 'parent', 'tool', 'object'):
        b = mother_boundary_feature(feats, own=10, response_len=960, info=dict(source=src, mother_bytes=120))
        assert b is not None and torch.allclose(b, feats[130])
    # Book / text sources: text, chinese_storybook, matched, topic, parent_reply
    for src in ('text', 'chinese_storybook', 'matched', 'topic', 'parent_reply'):
        b = mother_boundary_feature(feats, own=10, response_len=960, info=dict(source=src, navigation_response_bytes=480))
        assert b is not None and torch.allclose(b, feats[490])

    # UTF-8 start alignment on SidecarCorpusStream never slices mid-character.
    cn = tmp_path / 'utf8_align.bin'
    cn.write_text('从前有一只快乐的小萤火虫在夏夜的草地上飞来飞去。' * 150, encoding='utf-8')
    stream = SidecarCorpusStream(cn, cursor=1)  # Deliberately start on a UTF-8 continuation byte!
    resp, cont, info = stream.take_ambient(256, 256)
    assert info['navigation_response_bytes'] == 256
    assert info['source'] == 'chinese_storybook'
    # Leading bytes must decode cleanly without replacement char at index 0.
    head_str = bytes(resp[:30]).decode('utf-8', 'replace')
    assert not head_str.startswith('\ufffd')


def test_target_conditioned_action_differentiation_and_exact_optimizer_reload():
    """Gates 1 & 3: Same utterance aimed at book vs mother separates in action space, AND
    save + reload reproduces the exact next learning update (< 1e-6) including Adam moments!
    """
    torch.manual_seed(11)
    eng = _make_gaze_engine(d_model=32)
    eng.model.encoder = nn.Linear(32, 16)
    ptr = gp_mod.configure(eng, {}, d_gaze=16, lr=0.02, boundary_loss_weight=0.5)
    rng = np.random.default_rng(77)

    ptr.observe_object(eng, 'book', b'The quick brown fox jumps over the lazy dog.' * 4, fuse_body_state=False)
    ptr.observe_object(eng, 'mother', '慢慢来，妈妈陪你一起读绘本里的小故事呀。'.encode('utf-8') * 4, fuse_body_state=False)

    same_src = torch.randn(32)
    same_raw_act = torch.randn(32)
    book_target_feat = torch.ones(32) * 0.8
    mother_target_feat = torch.ones(32) * -0.8
    cands = [('book', b'book'), ('mother', b'mother')]

    # Run 8 mixed steps -> populates Adam m/v/step AND separates target_to_action(book) vs (mother).
    for i in range(8):
        dec = ptr.select_target(eng, candidates=cands, utterance_bytes='看到'.encode('utf-8'), rng=rng)
        tid = dec['target_id']
        tgt = book_target_feat if tid == 'book' else mother_target_feat
        b_ctx = dict(model=eng.model, source=same_src, raw_action=same_raw_act,
                     actual_target_id=tid, boundary_target=tgt, evidence_available=True)
        ptr.settle_credit(0.03 if tid == 'mother' else 0.01, evidence_available=True, boundary_context=b_ctx)

    # Gate 1: Same raw action produces distinct target-conditioned actions for book vs mother!
    act_book = ptr.condition_action(same_raw_act, 'book')
    act_mother = ptr.condition_action(same_raw_act, 'mother')
    assert float((act_book - act_mother).norm().item()) > 1e-3

    # Missing-evidence turn must NOT increment gaze policy step_count.
    steps_before = int(ptr.step_count.item())
    ptr.select_target(eng, candidates=cands, utterance_bytes=b'x', rng=rng)
    ptr.settle_credit(0.0, evidence_available=False, boundary_context=None)
    assert int(ptr.step_count.item()) == steps_before

    # Gate 3: Save & reload BOTH model weights AND dynamic/Adam state -> exact next-step equivalence!
    w_state = {k: v.detach().clone() for k, v in ptr.state_dict().items()}
    dyn_state = ptr.export_dynamic_state()

    eng2 = _make_gaze_engine(d_model=32)
    eng2.model.body.base.source.embedding.load_state_dict(eng.model.body.base.source.embedding.state_dict())
    eng2._read_source = eng._read_source
    eng2.model.encoder = eng.model.encoder
    ptr2 = gp_mod.configure(eng2, {}, d_gaze=16, lr=0.02, boundary_loss_weight=0.5)
    ptr2.load_state_dict(w_state)
    ptr2.load_dynamic_state(dyn_state)

    # Execute the EXACT same 9th learning step on BOTH ptr (live) and ptr2 (reloaded).
    rng_a = np.random.default_rng(999)
    rng_b = np.random.default_rng(999)
    dec_a = ptr.select_target(eng, candidates=cands, utterance_bytes='看到'.encode('utf-8'), rng=rng_a)
    dec_b = ptr2.select_target(eng2, candidates=cands, utterance_bytes='看到'.encode('utf-8'), rng=rng_b)
    assert dec_a['target_id'] == dec_b['target_id']
    assert dec_a['probs'] == dec_b['probs']

    tid = dec_a['target_id']
    tgt = book_target_feat if tid == 'book' else mother_target_feat
    b_ctx = dict(model=eng.model, source=same_src, raw_action=same_raw_act,
                 actual_target_id=tid, boundary_target=tgt, evidence_available=True)
    ptr.settle_credit(0.05, evidence_available=True, boundary_context=b_ctx)
    ptr2.settle_credit(0.05, evidence_available=True, boundary_context=b_ctx)

    for (n1, p1), (n2, p2) in zip(ptr.named_parameters(), ptr2.named_parameters()):
        assert n1 == n2
        max_diff = float((p1.detach() - p2.detach()).abs().max().item())
        assert max_diff < 1e-6, f'Next-step mismatch on {n1}: max_diff={max_diff}'


def test_external_interrupt_does_not_corrupt_alif_rhythm_phi():
    """Gate 4: External human/social interrupts reset ALIF refractory state without mutating phi."""
    from scripts.ib_local import alif_executive as alif
    from scripts.ib_local import learnable_alif_rhythm as lar

    eng = _make_gaze_engine(d_model=32)
    eng.alif_executive_state = None
    eng.model.max_chunk = 1600
    eng.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    eng.params = []
    eng.groups = ()
    eng.prev = None
    alif.configure(eng, {})
    rhythm = lar.configure(eng, {})
    rhythm.eligibility.fill_(0.5)
    rhythm.pending_eligibility.fill_(0.5)
    phi_before = rhythm.phi.detach().clone()

    eng.alif_executive_state['external_interrupt'] = True
    diag = alif.record_executive_spike(eng, dict(external_bytes=960, response_nll=1.5, short_loss=0.1, metal_g=0.08, choice='generate'))
    assert diag['learnable_rhythm'] == dict(skipped='external_interrupt')
    assert torch.equal(rhythm.phi.detach(), phi_before)
    assert float(rhythm.eligibility.abs().max().item()) == 0.0
    assert float(rhythm.pending_eligibility.abs().max().item()) == 0.0


def test_centered_relative_fingerprints_break_anisotropy_cone_and_match_wood_scale():
    """Verify global centering expands cos=0.995 cone to -1.0, preserves solo continuity,
    trains cleanly through frozen long_predictor + mother_from_boundary with zero model grad
    leakage, scales delta with ||a||, and migrates legacy schema-2 checkpoints safely.
    """
    torch.manual_seed(19)
    d = 32
    d_lat = 16
    eng = _make_gaze_engine(d_model=d)
    eng.model.encoder = nn.Linear(d, d_lat)
    eng.model.long_predictor = nn.Sequential(nn.Linear(2 * d_lat, d_lat), nn.GELU(), nn.Linear(d_lat, d_lat))
    eng.model.mother_from_boundary = nn.Linear(d_lat, d_lat, bias=False)
    ptr = gp_mod.configure(eng, {}, d_gaze=16, lr=0.015, boundary_loss_weight=0.5)

    # Construct an extreme anisotropic cone: cos(book, mother) = 0.995 (matching real cycle 3860!).
    shared = torch.nn.functional.normalize(torch.ones(d), dim=-1)
    orth = torch.nn.functional.normalize(
        torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(d)]), dim=-1
    )
    k_book = torch.nn.functional.normalize(shared + 0.05 * orth, dim=-1)
    k_mother = torch.nn.functional.normalize(shared - 0.05 * orth, dim=-1)
    raw_cos = float(torch.nn.functional.cosine_similarity(k_book[None], k_mother[None]).item())
    assert raw_cos > 0.994 and raw_cos < 0.996

    ptr.fingerprints['book'] = k_book.clone()
    ptr.fingerprints['mother'] = k_mother.clone()
    ptr._refresh_fp_center(snap=True)

    rel_b = ptr.relative_fingerprint('book')
    rel_m = ptr.relative_fingerprint('mother')
    rel_cos = float(torch.nn.functional.cosine_similarity(rel_b[None], rel_m[None]).item())
    assert rel_cos < -0.999

    # Solo continuity: even when only `book` is passed as candidate, `k_rel(book)` stays identical!
    rel_b_before = rel_b.clone()
    ptr.select_target(eng, candidates=[('book', b'book')], utterance_bytes=b'hi')
    assert torch.allclose(ptr.relative_fingerprint('book'), rel_b_before, atol=1e-6)

    # Real body-scale action ||a|| ~ 58.0: train 20 steps through frozen Wood prior.
    src = torch.randn(d) * (58.0 / (d ** 0.5))
    raw_act = torch.randn(d) * (58.0 / (d ** 0.5))
    act_norm = float(raw_act.norm().item())
    tgt_book = torch.randn(d) * 10.0 + 4.0
    tgt_mother = torch.randn(d) * 10.0 - 4.0
    for step in range(20):
        tid = 'book' if step % 2 == 0 else 'mother'
        b_ctx = dict(
            model=eng.model, source=src, raw_action=raw_act,
            actual_target_id=tid, boundary_target=(tgt_book if tid == 'book' else tgt_mother),
        )
        ptr.settle_credit(None, evidence_available=False, boundary_context=b_ctx)

    # Zero gradient leakage onto frozen body modules!
    for mod in (eng.model.encoder, eng.model.long_predictor, eng.model.mother_from_boundary):
        for p in mod.parameters():
            assert p.grad is None

    # Action separation is real body-scale (> 5% of ||a||) yet smoothly bounded (< 0.9 * ||a||).
    a_book = ptr.condition_action(raw_act, 'book')
    a_mother = ptr.condition_action(raw_act, 'mother')
    sep = float((a_book - a_mother).norm().item())
    assert sep > 0.05 * act_norm
    assert float((a_book - raw_act).norm().item()) < 0.45 * act_norm + 1e-3
    assert float((a_mother - raw_act).norm().item()) < 0.45 * act_norm + 1e-3

    diag = ptr.diagnostics()
    assert diag['raw_fp_max_cos'] > 0.994
    assert diag['rel_fp_max_cos'] < -0.999
    assert diag['delta_unit_sep'] > 0.05

    # Legacy schema-2 checkpoint migration: zero-resets ONLY target_to_action, keeps q_state & initializes fp_center.
    legacy_dyn = ptr.export_dynamic_state()
    legacy_dyn.pop('centered_target_v1', None)
    legacy_dyn.pop('fp_center', None)
    with torch.no_grad():
        ptr.q_state.weight.fill_(0.123)
    ptr.load_dynamic_state(legacy_dyn)
    assert ptr.fp_center is not None
    assert float(ptr.target_to_action.weight.abs().max().item()) == 0.0
    assert abs(float(ptr.q_state.weight.mean().item()) - 0.123) < 1e-6


def test_single_object_wood_valuation_reaches_selector_end_to_end(monkeypatch):
    """Verify the real caller `_select_and_condition_target` evaluates Wood consequence values
    even with ONE external candidate (`book` in solo reading), compares `book` against the
    genuinely evaluated null action (`null_causal_value`), and records explicit `prior_status`/`prior_error`.
    """
    from scripts.ib_local import causal_relation_program as crp
    from scripts.ib_local import causal_use_value as cuv
    from scripts.ib_local import finite_use_value as fuv
    from scripts.ib_local import learnable_precision_scales as lps
    from scripts.ib_local.world_learning_cycle import _select_and_condition_target

    torch.manual_seed(23)
    d = 32
    eng = _make_gaze_engine(d_model=d)
    ptr = gp_mod.configure(eng, {}, d_gaze=16)
    ptr.observe_object(eng, 'book', b'The little rabbit hopped across the sunny meadow.' * 3, fuse_body_state=False)
    with torch.no_grad():
        ptr._dual().observe_to_action.weight.fill_(0.25)
        ptr.target_to_action.weight.fill_(0.2)

    eng.world = SimpleNamespace(
        prepare_gaze_candidates=lambda utt: ([('book', b'book')], dict(stumble_ratio=1.2, nll_norm=0.3)),
    )
    eng.relation_ridge = 0.1
    eng.causal_use_arm = 'C2'
    eng._program_question_state = SimpleNamespace(tag='solo_state')
    eng._internal_source = torch.randn(1, d)
    eng._internal_action = torch.randn(3, d)
    eng._program_forecasts = dict(path={})

    # Real batched contract: 3 rows = [out(book), in(book), null(raw_act)].
    monkeypatch.setattr(lps, 'effective_ridge', lambda model, r: float(r))
    monkeypatch.setattr(cuv, 'queries', lambda engine, st: [('fact_a',)])
    captured_rows = {}

    def fake_mechanism(model, state, src_rows, cond_acts, ridge=0.1):
        captured_rows['n_rows'] = int(cond_acts.shape[0])
        return dict(cond_acts=cond_acts.detach().clone())

    def fake_finite_expected(engine, state, src_rows, cond_acts, dist, facts):
        # Distinct values even when all negative: book (-0.02) clearly beats null (-0.08)!
        # Using real null_causal_value instead of hardcoded 0.0 must give book > 0 and none < 0 prior!
        return torch.tensor([-0.03, -0.02, -0.08], dtype=torch.float32)

    monkeypatch.setattr(crp, 'mechanism', fake_mechanism)
    monkeypatch.setattr(fuv, 'expected', fake_finite_expected)

    dec = _select_and_condition_target(eng, b'rabbit', dict(choice=1))
    assert dec is not None
    assert captured_rows['n_rows'] == 3
    assert dec['prior_status'] == 'evaluated'
    assert 'prior_error' not in dec
    assert dec['causal_values'] == {'book': -0.3}
    assert dec['observe_causal_values'] == {'book': -0.2}
    assert dec['null_causal_value'] == -0.8
    assert dec['observe_causal_prior']['book'] > 0.0
    assert dec['observe_causal_prior']['none'] < 0.0
    assert dec['observe_probs']['book'] > dec['observe_probs']['none']

    # Error path must be recorded explicitly instead of silently leaving null.
    def boom(*args, **kwargs):
        raise RuntimeError('simulated valuation error')

    monkeypatch.setattr(fuv, 'expected', boom)
    dec_err = _select_and_condition_target(eng, b'rabbit', dict(choice=1))
    assert dec_err['prior_status'] == 'error'
    assert 'RuntimeError: simulated valuation error' in dec_err['prior_error']
    assert dec_err['causal_values'] is None



