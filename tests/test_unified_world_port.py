"""Tests for UnifiedWorldPort, BodyMirrorSnapshot, AdaptiveCarePlanTracker, and DSH cache invariance."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ib_local.alif_executive import _peak_window_slice
from scripts.ib_local.body_mirror import (
    AdaptiveCarePlanTracker,
    STAGE_EMERGING,
    STAGE_INFANT,
    STAGE_REFLECTIVE,
    capture_body_mirror,
    classify_utterance,
    format_caregiver_turn,
)
from scripts.ib_local.mengli_harness.family_provider import FamilyCaregiver
from scripts.ib_local.mengli_harness.mengli_world_client import DshMother
from scripts.ib_local.unified_world_port import UnifiedWorldPort


class FakeBaseWorld:
    def __init__(self, text: str):
        raw = text.encode('utf-8')
        self.data = np.frombuffer(raw, dtype=np.uint8).copy()
        self.window = min(4096, len(self.data))
        self.cursor = 0
        self.anchor = 0
        self.offset = 0
        self.moves = 0
        self.seed = 42
        self.seen = np.zeros(len(self.data), dtype=bool)
        self.retained = {}

    def take(self, utterance, response_bytes, holdout_bytes):
        total = response_bytes + holdout_bytes
        if self.cursor + total > len(self.data):
            self.cursor = 0
        start = self.cursor
        buf = self.data[start:start + total].copy()
        self.cursor = start + total
        self.seen[start:start + total] = True
        return buf[:response_bytes], buf[response_bytes:], dict(
            start=start, anchor=start, match=0, source='topic', holdout_position_fresh=True,
        )

    def state_dict(self, *, copy_arrays=True):
        return dict(
            cursor=int(self.cursor),
            anchor=int(self.anchor),
            offset=int(self.offset),
            moves=int(self.moves),
            seed=int(self.seed),
            seen=(self.seen.copy() if copy_arrays else self.seen),
        )

    def checkpoint_state_dict(self):
        return self.state_dict(copy_arrays=False)

    def load_state_dict(self, state):
        self.cursor = int(state['cursor'])
        self.anchor = int(state['anchor'])
        self.offset = int(state['offset'])
        self.moves = int(state['moves'])
        self.seen = np.array(state['seen'], dtype=bool, copy=True)


class FakeDshTransport:
    """In-memory mock of `/api/mengli/*` recording prefix stability & cache hits."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.seq = 0
        self.system_prompt = None
        self.plan_versions: list[int] = []
        self.parent_observations: list[dict] = []
        self.prepared_packets: list[str] = []
        self.pending = None
        self.total_hit = 0
        self.total_miss = 0
        self.prefix_tokens = 320

    def _ref(self):
        return dict(
            session_id='mengli-test-session',
            turn_end_seq=self.seq,
            assistant_seq=self.seq,
            result_seq=self.seq,
            summary_version=0,
        )

    def __call__(self, operation: str, payload: dict) -> dict:
        self.calls.append((operation, copy.deepcopy(payload)))
        if operation == 'restore':
            if payload.get('migration'):
                assert self.system_prompt is None, 'system_prompt must be seeded once'
                self.system_prompt = payload['migration']['system_prompt']
            return dict(checkpoint_ref=self._ref(), usage_totals=self._usage_totals())
        if operation == 'context':
            if 'plan' in payload:
                self.plan_versions.append(int(payload['plan']['version']))
            for obs in payload.get('observations', []):
                self.parent_observations.append(obs)
            self.seq += 1
            return dict(checkpoint_ref=self._ref(), plan_version=(self.plan_versions[-1] if self.plan_versions else 0))
        if operation == 'prepare':
            packet = payload['transcript']
            self.prepared_packets.append(packet)
            is_mirror = '[状态镜映]' in packet
            reply = (
                '梦漓，刚才在苹果落地那儿停了一下，那是地球的引力把它拉下来的，你觉得神奇吗？'
                if (is_mirror and '[起句节奏]' not in packet) else
                '刚才在苹果落地那儿停了一下，那是地球的引力把它拉下来的。'
                if is_mirror else
                '我们一起看看这一页讲了什么吧，地球会轻轻拉住苹果往下落。'
            )
            hit = self.prefix_tokens if len(self.prepared_packets) > 1 else 0
            miss = 120 if len(self.prepared_packets) > 1 else (self.prefix_tokens + 120)
            self.prefix_tokens += 140
            self.total_hit += hit
            self.total_miss += miss
            self.seq += 2
            self.pending = dict(step_id=payload['step_id'], ref=self._ref())
            return dict(
                raw=reply,
                delivered=reply,
                result_ref=self._ref(),
                telemetry=dict(
                    source='api',
                    api_usage=dict(
                        prompt_tokens=hit + miss,
                        completion_tokens=28,
                        cache_hit_tokens=hit,
                        cache_miss_tokens=miss,
                        total_tokens=hit + miss + 28,
                    ),
                ),
            )
        if operation == 'commit':
            assert self.pending is not None and self.pending['step_id'] == payload['step_id']
            ref = self.pending['ref']
            self.pending = None
            return dict(checkpoint_ref=ref, usage_totals=self._usage_totals())
        if operation == 'reflect':
            assert '语言预测与动态分块' in payload['dossier'] and '因果关系库' in payload['dossier']
            raw = json.dumps(dict(
                mengli_model='梦漓表达活跃，能抓住中文高频词，但遇长英文术语时节奏偏急。',
                progress_note='本阶段中文流畅度提升明显，可见视界下对卡顿提醒响应更稳。',
                update_plan=True,
                new_care_plan='接下来16轮继续以中文短句陪伴，遇长英文词先拆开解释再讲因果，偶发一个温和开放小问题。',
            ), ensure_ascii=False)
            return dict(raw=raw, telemetry=dict(source='api'), usage_totals=self._usage_totals())
        raise AssertionError(f'unexpected operation: {operation}')

    def _usage_totals(self):
        return dict(
            reply_usage=dict(
                prompt_tokens=self.total_hit + self.total_miss,
                completion_tokens=28 * len(self.prepared_packets),
                cache_hit_tokens=self.total_hit,
                cache_miss_tokens=self.total_miss,
                total_tokens=self.total_hit + self.total_miss + 28 * len(self.prepared_packets),
            ),
            summary_usage=None,
            source_counts=dict(api=len(self.prepared_packets), local=0, shared=0),
            api_calls=len(self.prepared_packets),
            failures=0,
        )


def _make_fake_engine(stumble_text: str = 'gravity pulls the apple downward'):
    stumble_hex = stumble_text.encode('utf-8').hex()
    pb = SimpleNamespace(diagnostics=lambda: dict(live_z_kin=0.42))
    hnet = SimpleNamespace(last_compression_ratio=3.92)
    body = SimpleNamespace(hnet_chunker=hnet)
    model = SimpleNamespace(phase_boltzmann=pb, body=body)
    return SimpleNamespace(
        alif_executive_state=dict(
            last_burst_bytes_hex=stumble_hex,
            last_burst_nll=3.4,
            nll_ema=1.8,
            last_trigger_cause='burst',
        ),
        model=model,
    )


def test_ambient_never_calls_mother_or_pollutes_transcript():
    transport = FakeDshTransport()
    caregiver = FamilyCaregiver(DshMother(branch_id='test-ambient', transport=transport))
    base = FakeBaseWorld('An apple falls from the tall green tree because Earth pulls it downward. ' * 200)
    port = UnifiedWorldPort(base, caregiver=caregiver, auto_commit=True)

    for _ in range(12):
        resp, cont, info = port.take_ambient(800, 800)
        assert len(resp) == 800 and len(cont) == 800
        assert info['channel'] == 'storybook_ambient'
        assert info['mother_bytes'] == 0

    assert len(transport.calls) == 0
    assert port.mother_transcript == []
    assert len(port.shared_page_bytes) == port.shared_page_cap


def test_priority_preemption_and_parent_observation_sync(tmp_path: Path):
    transport = FakeDshTransport()
    caregiver = FamilyCaregiver(DshMother(branch_id='test-priority', transport=transport))
    base = FakeBaseWorld('The sun warms the river and water rises into white clouds. ' * 200)
    inbox = tmp_path / 'inbox.jsonl'
    port = UnifiedWorldPort(
        base, caregiver=caregiver, mother_cooldown=0, mother_max_gap=1,
        auto_commit=False, inbox_path=inbox,
    )
    port.attach_engine(_make_fake_engine())
    port.register_tool('calc', lambda arg: f'{arg}=5')

    # 1. Live human parent via inbox preempts Mother immediately.
    inbox.write_text(json.dumps(dict(speaker='爸爸', role='parent', text='梦漓，你看苹果为什么会掉下来？'), ensure_ascii=False) + '\n', encoding='utf-8')
    port.set_cycle_hint(10)
    resp, cont, info = port.take(b'apple', 256, 256)
    assert info['channel'] == 'live_participant'
    assert info['speaker'] == '爸爸'
    assert '爸爸：梦漓，你看苹果为什么会掉下来？' in bytes(resp[:info['mother_bytes']]).decode('utf-8')
    assert '[在场]' in bytes(resp[:info['envelope_prefix_bytes']]).decode('utf-8')
    assert info['envelope_prefix_bytes'] > len('爸爸：'.encode('utf-8'))
    port.commit_pending_turn(metal_g=0.01)
    assert len(transport.parent_observations) == 1
    assert transport.parent_observations[0]['caregiver'] == 'parent'

    # 2. Registered tool fires on the exact same UTF-8 stream (zero extra body heads).
    port.set_cycle_hint(11)
    resp2, cont2, info2 = port.take(b'please [tool:calc 2+3]', 256, 256)
    assert info2['channel'] == 'tool'
    assert '[工具:calc] 2+3=5' in bytes(resp2[:info2['mother_bytes']]).decode('utf-8')
    assert info2['envelope_prefix_bytes'] == len('[工具:calc] '.encode('utf-8'))
    port.commit_pending_turn(metal_g=0.02)

    # 3. Next spike with vocal call to Mother routes to existing DSH Mother, carrying the tool note!
    port.set_cycle_hint(12)
    resp3, cont3, info3 = port.take('妈妈 why'.encode('utf-8'), 256, 256)
    assert info3['channel'] == 'dsh_mother'
    assert info3['routing_reason'] == 'vocal_call_mother'
    assert 'calc' in transport.prepared_packets[-1]
    assert '妈妈：' in bytes(resp3[:info3['envelope_prefix_bytes']]).decode('utf-8')
    assert info3['envelope_prefix_bytes'] == len('妈妈：'.encode('utf-8'))
    assert '[来自:书页]' not in bytes(resp3).decode('utf-8', 'ignore')
    port.commit_pending_turn(metal_g=0.03)
    assert port.mother_turns == 1


def test_deepseek_kv_cache_prefix_invariance_and_stage_hysteresis():
    transport = FakeDshTransport()
    caregiver = FamilyCaregiver(DshMother(branch_id='test-cache', transport=transport))
    base = FakeBaseWorld('Gravity pulls apples toward the ground whenever a ripe fruit lets go. ' * 200)
    port = UnifiedWorldPort(
        base, caregiver=caregiver, mother_cooldown=1, mother_max_gap=2,
        mirror_ratio=0.7, auto_commit=False,
    )
    port.attach_engine(_make_fake_engine())
    port.take_ambient(400, 400)

    # Run 10 Mother turns: first 4 babble/silent, next 6 recognizable fragments -> triggers stage transition ONCE!
    utterances = [
        b'', b'\xff\xfe', b'a', b'',
        b'apple falls', b'earth gravity', '苹果落地'.encode('utf-8'), '地球引力'.encode('utf-8'),
        b'apple earth', b'gravity pulls',
    ]
    for idx, utt in enumerate(utterances, start=1):
        port.force_next_mother()
        port.set_cycle_hint(100 + idx)
        _, _, info = port.take(utt, 256, 256)
        assert info['channel'] == 'dsh_mother'
        port.commit_pending_turn(metal_g=0.01)

    # Exact deterministic 7/3 mirror_visible vs book_only split across 10 turns.
    assert port.mirror_visible_turns == 7
    assert port.book_only_turns == 3
    mirror_packets = [p for p in transport.prepared_packets if '[状态镜映]' in p]
    book_only_packets = [p for p in transport.prepared_packets if '仅见共读书页' in p]
    assert len(mirror_packets) == 7
    assert len(book_only_packets) == 3
    for p in book_only_packets:
        assert '[状态镜映]' not in p

    # Prefix invariance: system_prompt seeded ONCE; care_plan synced ONCE at v1,
    # transitioned cleanly to v2 after turn 8 (4/8 fragments >= 50% & dwell >= 6),
    # and synced ONCE at v2 on turn 9 — zero per-turn thrashing!
    restore_calls = [p for op, p in transport.calls if op == 'restore']
    assert len(restore_calls) == 1
    assert transport.plan_versions == [1, 2]
    assert port.care_plan_tracker.stage == STAGE_EMERGING
    assert port.care_plan_tracker.version == 2

    # Turn 11 stays on v2 with ZERO extra context() call!
    port.force_next_mother()
    port.set_cycle_hint(111)
    _, _, info11 = port.take(b'why apple falls', 256, 256)
    port.commit_pending_turn(metal_g=0.02)
    assert transport.plan_versions == [1, 2]
    assert info11['care_stage'] == STAGE_EMERGING
    assert port.telemetry_summary()['provider_hit_rate'] > 0.70


def test_body_mirror_factual_stumble_and_zero_jargon():
    text = b'aaaabbbbccccddddeeeeffffgggghhhhSTUMBLE_HERE_64_BYTES_XYZZY1234567890abcdefghijklmnop'
    chunk = np.frombuffer(text, dtype=np.uint8)
    ce = np.full(len(chunk), 1.5, dtype=np.float32)
    ce[32:48] = 6.0
    peak_nll, peak_bytes = _peak_window_slice(ce, chunk, 16)
    assert peak_nll > 5.5
    assert b'STUMBLE_HERE_64_' == peak_bytes

    snap = capture_body_mirror(
        _make_fake_engine('gravity pulls the apple'),
        utterance_bytes='地球引力让苹果落地'.encode('utf-8'),
        recent_book_bytes=b'Newton watched the apple fall.',
        recent_metal_g=[0.01, 0.02, 0.015],
    )
    packet = format_caregiver_turn(snap, perspective_mode='mirror_visible')
    assert 'gravity pulls the apple' in packet
    for forbidden in ('alif_executive_state', 'live_z_kin', 'hnet_chunker', 'tensor', 'bfss_momenta'):
        assert forbidden not in packet
    assert classify_utterance(b'')['kind'] == 'silent'
    assert classify_utterance(b'\xff\xfe\x80')['kind'] == 'babble'
    assert classify_utterance(b'apple')['kind'] == 'fragments'
    assert classify_utterance('地球引力为什么拉苹果'.encode('utf-8'))['kind'] == 'coherent'


def test_sidecar_chinese_corpus_and_exact_checkpoint_roundtrip(tmp_path: Path):
    cn_file = tmp_path / 'cn_story.txt'
    cn_file.write_text('从前有一棵苹果树，熟透的红苹果从枝头掉到草地上，因为地球有引力。' * 120, encoding='utf-8')
    base = FakeBaseWorld('Once upon a time an apple fell onto the soft grass. ' * 200)
    transport = FakeDshTransport()
    caregiver = FamilyCaregiver(DshMother(branch_id='test-roundtrip', transport=transport))
    port = UnifiedWorldPort(
        base, caregiver=caregiver, mother_cooldown=0, mother_max_gap=2,
        mirror_ratio=0.7, auto_commit=True,
        sidecar_corpus=cn_file, sidecar_mix_ratio=0.5,
    )
    port.attach_engine(_make_fake_engine())

    sources = []
    for _ in range(6):
        _, _, info = port.take_ambient(256, 256)
        sources.append(info['source'])
    assert sources.count('chinese_storybook') == 3
    assert port.sidecar_reads == 3

    port.set_cycle_hint(1)
    port.take(b'apple', 256, 256)
    saved = port.state_dict()
    assert 'mother_unified_port' in saved

    # Round-trip restore into a brand new UnifiedWorldPort instance.
    base2 = FakeBaseWorld('Once upon a time an apple fell onto the soft grass. ' * 200)
    caregiver2 = FamilyCaregiver(DshMother(branch_id='test-roundtrip', transport=transport))
    port2 = UnifiedWorldPort(
        base2, caregiver=caregiver2, mother_cooldown=0, mother_max_gap=2,
        mirror_ratio=0.7, auto_commit=True,
        sidecar_corpus=cn_file, sidecar_mix_ratio=0.5,
    )
    port2.load_state_dict(saved)
    assert port2.ambient_reads == port.ambient_reads
    assert port2.sidecar_reads == port.sidecar_reads
    assert port2.sidecar.cursor == port.sidecar.cursor
    assert port2.mother_turns == port.mother_turns
    assert port2.mother_transcript == port.mother_transcript
    assert bytes(port2.shared_page_bytes) == bytes(port.shared_page_bytes)
    assert port2.care_plan_tracker.state_dict() == port.care_plan_tracker.state_dict()


def test_umwelt_membrane_horizon_canonical_exports():
    from scripts.ib_local import umwelt
    assert umwelt.MembranePort is UnifiedWorldPort
    assert umwelt.HorizonSnapshot.__name__ == 'BodyMirrorSnapshot'
    assert callable(umwelt.capture_horizon_snapshot)
    assert umwelt.UmweltBridge is DshMother
    assert umwelt.UmweltCaregiver is FamilyCaregiver


def test_yoneda_bidirectional_mother_proactive_and_dynamic_objects(tmp_path: Path):
    """Verify strict Yoneda separation (`Hom(MengLi, Book)` vs `Hom(MengLi, Mother)`) + anti-name-lock."""
    transport = FakeDshTransport()
    caregiver = FamilyCaregiver(DshMother(branch_id='test-yoneda', transport=transport))
    cn_file = tmp_path / 'cn_stories.txt'
    cn_file.write_text(
        ('小兔在草地上跳来跳去。太阳暖洋洋的。\n' * 60)
        + '莉莉在院子里看到一只会发光的小萤火虫，高兴地拍手。\n'
        + ('小熊在河边钓到一条亮晶晶的小鱼。\n' * 60),
        encoding='utf-8',
    )
    base = FakeBaseWorld('Apples fall to Earth because of gravity. ' * 200)
    port = UnifiedWorldPort(
        base, caregiver=caregiver, mother_cooldown=5, mother_max_gap=2,
        mirror_ratio=1.0, auto_commit=True,
        sidecar_corpus=cn_file, sidecar_mix_ratio=0.5,
    )
    port.attach_engine(_make_fake_engine())

    # 1. Dynamically mount a brand-new black-box object (no body changes!).
    port.register_object('weather', lambda q: f'今天晴，22度 ({q})', label='天气')

    # 2. Policy A (Presence-based Yoneda default):
    #    - When Mother leaves scene (`[在场] 书页`), unaddressed Chinese utterance `看到` navigates
    #      the Chinese storybook (`Hom(MengLi, Book)`) and delivers `[在场] 书页` notice!
    port.leave_scene('mother')
    resp_nav, _, info_nav = port.take('看到'.encode('utf-8'), 256, 256)
    assert info_nav['channel'] == 'book_navigation'
    assert info_nav['routing_reason'] == 'book_floor_navigation'
    assert info_nav['match'] == 1
    nav_text = bytes(resp_nav).decode('utf-8', 'ignore')
    assert '[在场] 书页' in nav_text
    assert '萤火虫' in nav_text
    assert port.mother_turns == 0

    #    - When Mother joins co-reading (`[在场] 书页, 妈妈`), infant unaddressed speech routes
    #      directly to Mother (`co_reading_child_speech`), zero hidden book jump!
    port.join_mother()
    resp_co, _, info_co = port.take('看到萤火虫'.encode('utf-8'), 256, 256)
    assert info_co['channel'] == 'dsh_mother'
    assert info_co['trigger_kind'] == 'child_spoke'
    assert info_co['routing_reason'] == 'co_reading_child_speech'
    assert port.turns_since_name_call == 0  # Mock reply started with '梦漓，...'

    # 3. On Social Floor (`active_floor == 'mother'`), unaddressed reply routes to Mother (`child_replied`)
    #    AND receives the KV-cache-safe `[起句节奏]` anti-name-lock cue!
    _, _, info2 = port.take('为什么掉下来'.encode('utf-8'), 256, 256)
    assert info2['channel'] == 'dsh_mother'
    assert info2['trigger_kind'] == 'child_replied'
    assert info2['routing_reason'] == 'social_floor_reply'
    assert '[起句节奏]' in transport.prepared_packets[-1]
    assert port.turns_since_name_call >= 1

    # 4. Vocal call (`妈妈你看`) always routes to Mother with loss-masked `妈妈：` identity header.
    resp1, _, info1 = port.take('妈妈你看'.encode('utf-8'), 256, 256)
    assert info1['channel'] == 'dsh_mother'
    assert info1['routing_reason'] == 'vocal_call_mother'
    assert bytes(resp1[:info1['envelope_prefix_bytes']]).decode('utf-8').endswith('妈妈：')
    assert info1['envelope_prefix_bytes'] >= len('妈妈：'.encode('utf-8'))

    # 5. Explicit addressing `@书页` navigates book directly (Mother stays out of the way, zero markup tag).
    resp3, _, info3 = port.take('@书页 gravity'.encode('utf-8'), 256, 256)
    assert info3['channel'] == 'book_addressed'
    assert '[来自:书页]' not in bytes(resp3).decode('utf-8', 'ignore')

    # 6. Explicit addressing `@天气 北京` invokes dynamic black-box object through same Membrane.
    resp4, _, info4 = port.take('@天气 北京'.encode('utf-8'), 256, 256)
    assert info4['channel'] == 'dynamic_object'
    assert '[天气] 今天晴，22度 (北京)' in bytes(resp4[:info4['mother_bytes']]).decode('utf-8')

    # 7. Quiet pauses on Book Floor hit mother_max_gap=2 -> Mother initiates co-reading (`mother_proactive`),
    #    opening Social Floor -> then MengLi replies `看到` -> routed as `child_replied`!
    port.spikes_since_mother = 0
    port.quiet_spikes_since_mother = 0
    _, _, q1 = port.take(b'', 256, 256)
    assert q1['channel'] == 'storybook_spike_fallback'
    _, _, q2 = port.take(b'', 256, 256)
    assert q2['channel'] == 'storybook_spike_fallback'
    _, _, pro = port.take(b'', 256, 256)
    assert pro['channel'] == 'dsh_mother'
    assert pro['trigger_kind'] == 'mother_proactive'
    assert port.active_floor == 'mother'
    assert port.awaiting_reply_from == 'mother'
    assert '[情境: 孩子安静读书时停顿下来' in transport.prepared_packets[-1]

    _, _, ans = port.take('看到'.encode('utf-8'), 256, 256)
    assert ans['channel'] == 'dsh_mother'
    assert ans['trigger_kind'] == 'child_replied'
    assert '[情境: 孩子回应了你上一轮的话' in transport.prepared_packets[-1]

    # 8. Solo reading when Mother leaves scene -> active speech navigates book instead.
    port.leave_scene('mother')
    _, _, solo = port.take('苹果'.encode('utf-8'), 256, 256)
    assert solo['channel'] == 'book_navigation'
    port.join_mother()
    _, _, back = port.take('妈妈回来了'.encode('utf-8'), 256, 256)
    assert back['channel'] == 'dsh_mother'


def test_five_element_mirror_and_autonomous_mother_reflection(tmp_path: Path):
    import torch
    from scripts.ib_local.mengli_harness.mengli_world_client import SETTINGS_64K

    transport = FakeDshTransport()
    dsh = DshMother(branch_id='test-64k-reflect', settings=SETTINGS_64K, transport=transport)
    caregiver = FamilyCaregiver(dsh)
    base = FakeBaseWorld('Apples fall to Earth because of gravity. ' * 200)
    port = UnifiedWorldPort(base, caregiver=caregiver, mother_max_gap=2, mirror_ratio=1.0, auto_commit=False)

    eng = _make_fake_engine('gravity pulls apples')
    eng.model.root_latent_planner = SimpleNamespace(diagnostics=lambda: dict(decisions=12, latent_single_rate=0.75))
    eng.model.memory = SimpleNamespace(occupied=torch.ones(64, dtype=torch.bool), episodes=torch.tensor(128))
    eng.model.causal_relations = SimpleNamespace(occupied=torch.ones(16, dtype=torch.bool), evidence_count=torch.tensor(42))
    port.attach_engine(eng)

    for idx in range(2):
        port.set_cycle_hint(200 + idx)
        port.force_next_mother()
        port.take('苹果落地'.encode('utf-8'), 256, 256)
        port.commit_pending_turn(metal_g=0.015)

    # Verify per-turn mirror includes concrete internal module & memory bank snapshot.
    assert '[内部运转与记忆库]' in transport.prepared_packets[-1]
    assert '因果关系库16条(42条证据)' in transport.prepared_packets[-1]
    assert '情景经历库64/64' in transport.prepared_packets[-1]

    # Trigger periodic autonomous Mother reflection -> writes mother_notes.md and updates plan ONCE!
    notes = tmp_path / 'mother_notes.md'
    refl = port.maybe_reflect_and_adapt(reflect_every_turns=2, cycle=201, notes_path=notes)
    assert refl is not None
    assert refl['parsed_json'] is True
    assert refl['plan_updated'] is True
    assert port.care_plan_tracker.version == 2
    assert '梦漓模型' in notes.read_text(encoding='utf-8')
    assert '长英文词' in port.care_plan_tracker.current_plan()['text']

    # Parent override takes precedence over Mother's autonomous plan updates.
    port.set_user_plan('家长锁定：只聊太空与行星故事。')
    v_locked = port.care_plan_tracker.version
    for idx in range(2):
        port.set_cycle_hint(210 + idx)
        port.force_next_mother()
        port.take('星星'.encode('utf-8'), 256, 256)
        port.commit_pending_turn(metal_g=0.02)
    refl2 = port.maybe_reflect_and_adapt(reflect_every_turns=2, cycle=211, notes_path=notes)
    assert refl2 is not None and refl2['plan_updated'] is False
    assert port.care_plan_tracker.version == v_locked


def test_lossless_parent_tool_credit_and_turn_taking_physics(tmp_path: Path):
    """Verify lossless credit symmetry (`observed_pieces`) for Dad/Tools + immediate interrupt & Parent floor."""
    from scripts.ib_local.world_learning_cycle import observed_pieces

    transport = FakeDshTransport()
    caregiver = FamilyCaregiver(DshMother(branch_id='test-physics', transport=transport))
    cn_file = tmp_path / 'cn_stories.txt'
    cn_file.write_text('小雨滴答滴答落下，小鸭子在池塘里快乐地玩水。\n' * 120, encoding='utf-8')
    base = FakeBaseWorld('Raindrops fall gently onto the pond while the little duck swims. ' * 200)
    inbox = tmp_path / 'inbox.jsonl'
    port = UnifiedWorldPort(
        base, caregiver=caregiver, mother_cooldown=2, mother_max_gap=3,
        mirror_ratio=1.0, auto_commit=True, inbox_path=inbox,
        sidecar_corpus=cn_file, sidecar_mix_ratio=0.5,
    )
    port.attach_engine(_make_fake_engine())
    port.register_tool('bash', lambda cmd: f'out:{cmd}')

    # 1. Sidecar Chinese storybook exposes full World contract (`holdout_position_fresh`).
    port.take_ambient(256, 256)
    r_amb, c_amb, info_amb = port.take_ambient(256, 256)
    assert info_amb['source'] == 'chinese_storybook'
    assert info_amb['holdout_position_fresh'] is True
    assert port.has_pending_interrupt() is False

    # 2. Live Dad message via inbox immediately triggers `has_pending_interrupt()`.
    inbox.write_text(json.dumps(dict(speaker='爸爸', role='parent', text='外面下雨啦，你喜欢吗？'), ensure_ascii=False) + '\n', encoding='utf-8')
    assert port.has_pending_interrupt() is True
    resp_dad, cont_dad, info_dad = port.take(b'', 256, 256)
    assert info_dad['channel'] == 'live_participant'
    assert info_dad['source'] == 'parent'
    assert port.active_floor == 'parent'
    assert port.awaiting_reply_from == 'parent'
    assert port.has_pending_interrupt() is True  # Immediate reply window open for MengLi!

    # Lossless credit symmetry: `observed_pieces` splits `'parent'` foreground cleanly like `'mother'`!
    _, pieces_dad = observed_pieces(resp_dad, cont_dad, info_dad)
    assert [p.source for p in pieces_dad][:2] == ['parent', 'parent']
    assert any(p.source == 'ambient' for p in pieces_dad)

    # 3. MengLi immediately replies on Dad's Social Floor -> routed to `reply_to_parent`, ZERO book jump,
    #    Mother notified in room context, and floor smoothly relaxes to Book afterwards!
    _, _, info_ans = port.take('喜欢下雨'.encode('utf-8'), 256, 256)
    assert info_ans['channel'] == 'reply_to_parent'
    assert info_ans['routing_reason'] == 'social_floor_reply'
    assert port.active_floor == 'book'
    assert port.awaiting_reply_from is None
    assert port.has_pending_interrupt() is False
    assert any('喜欢下雨' in note for note in port._pending_tool_notes)

    # 4. Tool object also receives 100% identical lossless interactive credit splitting!
    resp_tool, cont_tool, info_tool = port.take('@工具:bash ls'.encode('utf-8'), 256, 256)
    assert info_tool['source'] == 'tool'
    _, pieces_tool = observed_pieces(resp_tool, cont_tool, info_tool)
    assert [p.source for p in pieces_tool][:2] == ['tool', 'tool']


def test_interaction_logic_stumble_alignment_babble_and_silent_cooldown():
    """Verify live-audit fixes:
    1. `book_excerpt` centers on `stumble_excerpt` across a 1280B calm piece (zero story mismatch).
    2. Non-empty babble bytes (`b'\\xff\\xfe'`) classify as vocal (`gaze_mother_speak`, `[咿呀声]`), not silent look.
    3. Silent gaze respects `mother_cooldown`, while vocal speech keeps Mother immediately responsive.
    """
    story_a = '海里有一只笨手笨脚的小虾，它在礁石旁边慢慢学游泳。' * 10
    story_b = '莉莉在院子里发现了一把会下雨的魔法小伞，大家都很开心。' * 15
    full_page = (story_a + '\n' + story_b).encode('utf-8')
    assert len(full_page) > 1200

    eng = _make_fake_engine('笨手笨脚的小虾，它在礁石旁边')
    snap = capture_body_mirror(eng, utterance_bytes=b'', recent_book_bytes=full_page)
    assert '笨手笨脚的小虾' in snap.stumble_excerpt
    assert snap.stumble_excerpt in snap.book_excerpt

    transport = FakeDshTransport()
    caregiver = FamilyCaregiver(DshMother(branch_id='test-logic', transport=transport))
    base = FakeBaseWorld(story_a + '\n' + story_b)
    port = UnifiedWorldPort(
        base, caregiver=caregiver, mother_cooldown=3, mother_max_gap=8,
        mirror_ratio=1.0, auto_commit=True,
    )
    port.attach_engine(eng)
    assert port.shared_page_cap >= 1600

    # Right after a Mother turn (`spikes_since_mother == 0`), silent gaze excludes Mother during cooldown,
    # whereas spoken utterances keep Mother immediately available!
    port.spikes_since_mother = 0
    silent_cands, _ = port.prepare_gaze_candidates(b'')
    assert [cid for cid, _ in silent_cands] == ['book']
    vocal_cands, _ = port.prepare_gaze_candidates('妈妈'.encode('utf-8'))
    assert 'mother' in [cid for cid, _ in vocal_cands]
    babble_cands, _ = port.prepare_gaze_candidates(b'\xff\xfe')
    assert 'mother' in [cid for cid, _ in babble_cands]

    # Once `spikes_since_mother >= mother_cooldown`, silent gaze includes Mother again.
    port.spikes_since_mother = 3
    silent_cands_ready, _ = port.prepare_gaze_candidates(b'')
    assert 'mother' in [cid for cid, _ in silent_cands_ready]

    # Vocal babble with `planned_target='mother'` must route as `gaze_mother_speak`, not silent `gaze_mother_look`.
    import types
    import torch
    from scripts.ib_local import gaze_pointer as gp_mod
    eng.model = types.SimpleNamespace(
        body=types.SimpleNamespace(embed=torch.nn.Embedding(256, 32)),
        encoder=torch.nn.Linear(32, 32),
    )
    eng.device = torch.device('cpu')
    gp_mod.configure(eng, {}, d_gaze=16)
    _, _, info = port.take(b'\xff\xfe', 256, 256, planned_target=dict(target_id='mother'))
    assert info['channel'] == 'dsh_mother'
    assert info['routing_reason'] == 'gaze_mother_speak'
    assert info['child_text'] == '[咿呀声]'


def test_utf8_tail_alignment_bilingual_page_reset_and_strict_coherent(tmp_path: Path):
    """Verify:
    1. SidecarCorpusStream never slices a 3-byte Chinese character in half at chunk end (engine.pending is valid UTF-8).
    2. Switching between English and Chinese resets `shared_page_bytes` (zero cross-language gluing).
    3. `classify_utterance` does not mark short 4-char garbled CJK or bracket markup as `coherent`.
    """
    from scripts.ib_local.body_mirror import classify_utterance
    from scripts.ib_local.unified_world_port import SidecarCorpusStream

    cn_file = tmp_path / 'cn_tail.txt'
    # Pure 3-byte Chinese chars; 640+640 = 1280 bytes is NOT divisible by 3 (1280 % 3 == 2 -> would slice char!).
    cn_file.write_text('小鸭子在池塘里快乐地游来游去。' * 200, encoding='utf-8')
    stream = SidecarCorpusStream(cn_file)
    for _ in range(5):
        resp, cont, _ = stream.take_ambient(640, 640)
        full = bytes(np.concatenate((resp, cont)))
        # Must decode strictly with ZERO UnicodeDecodeError!
        decoded = full.decode('utf-8')
        assert '\ufffd' not in decoded

    base = FakeBaseWorld('English story about castles and rolling boulders. ' * 200)
    port = UnifiedWorldPort(base, sidecar_corpus=cn_file, sidecar_mix_ratio=0.5)
    port.take_ambient(256, 256)  # English
    assert 'English' in bytes(port.shared_page_bytes).decode('utf-8', 'ignore')
    port.take_ambient(256, 256)  # Chinese -> must clear old English bytes!
    page_txt = bytes(port.shared_page_bytes).decode('utf-8')
    assert '小鸭子' in page_txt and 'English' not in page_txt

    assert classify_utterance('莉莉讶偼'.encode('utf-8'))['kind'] == 'fragments'
    assert classify_utterance('[来自:书页] 汤姆'.encode('utf-8'))['kind'] == 'fragments'
    assert classify_utterance('自行车非常兴奋'.encode('utf-8'))['kind'] == 'coherent'


def test_loss_masked_identity_card_separates_identical_hello_without_table_echo():
    """Verify:
    1. Mom saying '你好' and Dad saying '你好' have distinct loss-masked identity headers (`妈妈：` vs `爸爸：`)
       and identical message content, with `[在场] 角色(描述)` identity cards on entry.
    2. Masked prefix bytes are READ-ONLY for the Fast Table (`frozen_table_logp`), so repeated `妈妈：`
       headers never pollute n-gram Fast Table counts!
    """
    from scripts.ib_local.navigation_live_witness import frozen_table_logp
    from scripts.ib_local.ngram_fast_table import ByteNgramFastTable

    transport = FakeDshTransport()
    caregiver = FamilyCaregiver(DshMother(branch_id='test-idcard', transport=transport))
    base = FakeBaseWorld('Apples fall to Earth. ' * 200)
    port = UnifiedWorldPort(base, caregiver=caregiver, mother_cooldown=0, auto_commit=True)
    port.attach_engine(_make_fake_engine())

    # Dad says '你好' (first entry -> includes `[在场] ...；爸爸(爸爸实时陪伴与对话)\n爸爸：`).
    port.enqueue_participant('你好', speaker='爸爸', role='parent')
    resp_dad, _, info_dad = port.take(b'', 256, 256)
    prefix_dad = bytes(resp_dad[:info_dad['envelope_prefix_bytes']]).decode('utf-8')
    body_dad = bytes(resp_dad[info_dad['envelope_prefix_bytes']:info_dad['mother_bytes']]).decode('utf-8')
    assert '[在场]' in prefix_dad and '爸爸(爸爸实时陪伴与对话)' in prefix_dad and prefix_dad.endswith('爸爸：')
    assert body_dad == '你好\n'

    # Mother replies (starts with loss-masked `妈妈：`, AND marks `书页：` right where the book resumes!).
    port.force_next_mother()
    resp_mom, _, info_mom = port.take(b'', 256, 256)
    prefix_mom = bytes(resp_mom[:info_mom['envelope_prefix_bytes']]).decode('utf-8')
    assert prefix_mom == '妈妈：'
    spans = info_mom['envelope_mask_spans']
    assert len(spans) == 2
    assert spans[0] == (0, len('妈妈：'.encode('utf-8')))
    s_b, e_b = spans[1]
    assert s_b == info_mom['mother_bytes']
    assert bytes(resp_mom[s_b:e_b]).decode('utf-8') == '书页：'

    # Verify Fast Table read-only masking: feeding `妈妈：你好` with masked prefix leaves `妈妈：` unseen!
    table = ByteNgramFastTable()
    seq = np.frombuffer('妈妈：你好'.encode('utf-8'), dtype=np.uint8)
    mask_len = len('妈妈：'.encode('utf-8'))
    before_p = float(np.exp(frozen_table_logp(table, np.concatenate((np.array([10], np.uint8), seq[:mask_len])))[-1, seq[mask_len - 1]]))
    # Simulate world_learning_cycle read-only head + credit-only tail (exact shape == (mask_len, 256)!)
    head_probs = np.exp(frozen_table_logp(table, np.concatenate((np.array([10], np.uint8), seq[:mask_len - 1]))))
    assert head_probs.shape == (mask_len, 256)
    table.run(np.concatenate((seq[mask_len - 1:mask_len], seq[mask_len:])))
    after_p = float(np.exp(frozen_table_logp(table, np.concatenate((np.array([10], np.uint8), seq[:mask_len])))[-1, seq[mask_len - 1]]))
    # Masked prefix receives ZERO Fast-Table counts (prob does not rise; compare unmasked table which memorizes it!)
    assert after_p <= before_p + 1e-6
    unmasked = ByteNgramFastTable()
    for _ in range(3):
        unmasked.run(np.concatenate((np.array([10], np.uint8), seq)))
    unmasked_p = float(np.exp(frozen_table_logp(unmasked, np.concatenate((np.array([10], np.uint8), seq[:mask_len])))[-1, seq[mask_len - 1]]))
    assert unmasked_p > 3.0 * after_p

    # Verify theatrical stage directions are cleanly stripped before delivery.
    from scripts.ib_local.unified_world_port import _strip_stage_directions
    assert _strip_stage_directions('（轻声笑）ed di，你像在敲小鼓点呢。') == 'ed di，你像在敲小鼓点呢。'
    assert _strip_stage_directions('*smiles warmly* Look at that little boat!') == 'Look at that little boat!'
    assert _strip_stage_directions('慢一点没关系（摸摸头），咱们一起看。') == '慢一点没关系，咱们一起看。'








