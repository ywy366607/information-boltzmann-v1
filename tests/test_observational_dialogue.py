"""Tests for the unified Membrane Event Stream contract & arbitrary multi-span masking.

Verifies all 5 contract invariants across ambient reading, offline multi-source streams,
live Mother dialogue, live human interrupts, and addressed book navigation:
  1. `speaker_id` (who spoke / produced bytes)
  2. `addressee_id` (who it was addressed to)
  3. `is_action_consequence` (strict separation of MengLi's own action consequences vs observations)
  4. `source_transitions` + `envelope_mask_spans` (exact byte offsets where source switches)
  5. Full save/reload persistence (`event_ledger`, `dialogue_cursor`, `dialogue_reads`)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.ib_local import alif_executive as alif
from scripts.ib_local import gaze_pointer as gp_mod
from scripts.ib_local.mengli_harness.family_provider import FamilyCaregiver
from scripts.ib_local.mengli_harness.mengli_world_client import DshMother
from scripts.ib_local.unified_world_port import MultiSourceEventStream, UnifiedWorldPort
from tests.test_gaze_pointer import _make_gaze_engine
from tests.test_unified_world_port import FakeBaseWorld, FakeDshTransport


def _write_sample_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / 'dialogues.jsonl'
    scenes = [
        dict(
            scene='joint_attention_zh',
            roster='[在场] 妈妈(陪读与讲解)、爸爸(陪伴与聊天)',
            turns=[
                dict(speaker='妈妈', id='mother', text='你看窗外那只橘色的小猫，正趴在石阶上晒太阳呢。'),
                dict(speaker='爸爸', id='parent', text='它把尾巴卷在脚边，像是睡着了。'),
                dict(speaker='妈妈', id='mother', text='刚才风吹过树叶，它耳朵还轻轻抖了一下。'),
            ],
        ),
        dict(
            scene='concise_question_zh',
            turns=[
                dict(speaker='爸爸', id='parent', text='这页里说的“潮汐”是什么意思？'),
                dict(speaker='妈妈', id='mother', text='就是海水每天有规律地涨上来、又退下去。'),
            ],
        ),
        dict(
            scene='joint_attention_en',
            turns=[
                dict(speaker='Mom', id='mother', text='Look at the red kite in the picture—its tail is longer than the tree.'),
                dict(speaker='Dad', id='parent', text='The wind must be strong today to lift it that high.'),
            ],
        ),
    ]
    corpus.write_text('\n'.join(json.dumps(s, ensure_ascii=False) for s in scenes) + '\n', encoding='utf-8')
    return corpus


class _RecordingTable:
    """Minimal Fast Table stub recording every byte sequence passed to `run_credit`."""

    def __init__(self):
        self.credit_segments: list[bytes] = []

    def run_credit(self, seq: np.ndarray, *, max_window: int):
        seg = bytes(seq[1:].tolist())
        self.credit_segments.append(seg)
        return None, np.full((len(seg), 256), 1.0 / 256.0, dtype=np.float32), None


def test_multi_source_stream_contract_spans_and_utf8_safety(tmp_path: Path):
    corpus = _write_sample_corpus(tmp_path)
    stream = MultiSourceEventStream(corpus)
    resp, cont, info = stream.take_ambient(320, 320)
    assert len(resp) == 320 and len(cont) == 320
    full = np.concatenate((resp, cont)).tobytes()
    assert '\ufffd' not in full.decode('utf-8')

    # Strict contract fields: observational stream is NEVER MengLi's action consequence.
    assert info['is_action_consequence'] is False
    assert info['speaker_id'] in ('mother', 'parent')
    assert info['addressee_id'] in ('mother', 'parent', 'all')

    # Exact byte-offset source transitions match the encoded headers.
    trans = info['source_transitions']
    assert len(trans) >= 4
    assert trans[0]['speaker_id'] == 'umwelt'
    assert trans[0]['offset'] == 0
    for tr in trans:
        if tr['mask_end'] > tr['mask_start']:
            hdr = full[tr['mask_start']:tr['mask_end']].decode('utf-8')
            assert hdr.endswith('：') or hdr.startswith('[在场]')

    # Unmasked bytes contain ONLY spoken dialogue content, zero raw speaker headers.
    mask = np.ones(len(full), dtype=bool)
    for s, e in info['envelope_mask_spans']:
        mask[s:e] = False
    unmasked_text = bytes(b for b, keep in zip(full, mask) if keep).decode('utf-8')
    assert '妈妈：' not in unmasked_text
    assert '爸爸：' not in unmasked_text
    assert '[在场]' not in unmasked_text
    assert '窗外' in unmasked_text


def test_span_masked_table_never_writes_headers_to_fast_table(tmp_path: Path, monkeypatch):
    def _fake_frozen(table, seq):
        return np.full((len(seq), 256), -np.log(256.0), dtype=np.float32)

    monkeypatch.setattr('scripts.ib_local.navigation_live_witness.frozen_table_logp', _fake_frozen)

    corpus = _write_sample_corpus(tmp_path)
    stream = MultiSourceEventStream(corpus)
    resp, cont, info = stream.take_ambient(320, 320)
    chunk = np.concatenate((resp, cont))
    spans = alif._merge_mask_spans(info['envelope_mask_spans'], len(chunk))
    table = _RecordingTable()
    probs = alif._span_masked_table_probs(table, ord('\n'), chunk, spans)
    assert probs.shape == (len(chunk), 256)

    written = b''.join(table.credit_segments).decode('utf-8')
    assert '妈妈：' not in written
    assert '爸爸：' not in written
    assert '[在场]' not in written
    assert len(written) > 100


def test_unified_membrane_event_contract_and_roundtrip_across_all_paths(tmp_path: Path):
    corpus = _write_sample_corpus(tmp_path)
    eng = _make_gaze_engine(d_model=32)
    ptr = gp_mod.configure(eng, {}, d_gaze=16)

    transport = FakeDshTransport()
    caregiver = FamilyCaregiver(DshMother(branch_id='test-event-contract', transport=transport))
    base = FakeBaseWorld('Apples fall to Earth because of gravity. ' * 200)
    port = UnifiedWorldPort(
        base,
        caregiver=caregiver,
        auto_commit=True,
        dialogue_demo_corpus=corpus,
        dialogue_demo_ratio=0.25,
    )
    port.attach_engine(eng)

    # 1) Ambient storybook + offline multi-source dialogue -> strictly `is_action_consequence == False`.
    channels = []
    for _ in range(4):
        _, _, info = port.take_ambient(256, 256)
        channels.append(info['channel'])
        assert info['event_schema'] == 1
        assert info['is_action_consequence'] is False
        assert info['source_transitions']
    assert channels.count('observational_dialogue') == 1
    assert 'mother' in ptr.fingerprints and 'parent' in ptr.fingerprints and 'book' in ptr.fingerprints
    # Shared storybook page stays strictly on storybook text.
    assert '妈妈：' not in bytes(port.shared_page_bytes).decode('utf-8', 'ignore')

    # 2) Live Mother turn triggered by MengLi speaking -> `is_action_consequence == True` + 2 transitions (`mother` -> `book`)!
    port.set_cycle_hint(101)
    resp_m, _, info_m = port.take('@妈妈 小熊回来了'.encode('utf-8'), 256, 256)
    assert info_m['channel'] == 'dsh_mother'
    assert info_m['speaker_id'] == 'mother'
    assert info_m['addressee_id'] == 'mengli'
    assert info_m['is_action_consequence'] is True
    trans_m = info_m['source_transitions']
    assert len(trans_m) == 2
    assert trans_m[0]['speaker_id'] == 'mother' and trans_m[0]['offset'] == 0
    assert trans_m[1]['speaker_id'] == 'book' and trans_m[1]['offset'] == info_m['mother_bytes']
    assert bytes(resp_m[trans_m[1]['mask_start']:trans_m[1]['mask_end']]).decode('utf-8') == '书页：'

    # 3) Live unsolicited human/Dad interrupt -> `is_action_consequence == False` + exact `parent` -> `book` transition!
    port.set_cycle_hint(102)
    port.enqueue_participant('爸爸来看看你们读到哪啦。', speaker='爸爸', role='parent')
    _, _, info_p = port.take(b'', 256, 256)
    assert info_p['channel'] == 'live_participant'
    assert info_p['speaker_id'] == 'parent'
    assert info_p['addressee_id'] == 'mengli'
    assert info_p['is_action_consequence'] is False
    assert info_p['source_transitions'][0]['speaker_id'] == 'parent'
    assert info_p['source_transitions'][1]['speaker_id'] == 'book'

    # 4) Addressed book navigation -> `is_action_consequence == True`, `speaker_id == 'book'`.
    port.set_cycle_hint(103)
    _, _, info_b = port.take('@书页 gravity'.encode('utf-8'), 256, 256)
    assert info_b['speaker_id'] == 'book'
    assert info_b['addressee_id'] == 'mengli'
    assert info_b['is_action_consequence'] is True

    # 5) Full checkpoint save + reload preserves `event_ledger`, `dialogue_cursor`, `dialogue_reads` bit-identically!
    assert len(port.event_ledger) >= 4
    saved = port.state_dict()
    port2 = UnifiedWorldPort(
        FakeBaseWorld('Apples fall to Earth because of gravity. ' * 200),
        caregiver=FamilyCaregiver(DshMother(branch_id='test-event-contract', transport=transport)),
        auto_commit=True,
        dialogue_demo_corpus=corpus,
        dialogue_demo_ratio=0.25,
    )
    port2.load_state_dict(saved)
    assert port2.event_ledger == port.event_ledger
    assert port2.dialogue_reads == port.dialogue_reads
    assert port2.dialogue_demo.cursor == port.dialogue_demo.cursor
