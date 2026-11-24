"""Tests for ObservationalDialogueStream & arbitrary multi-span masking in calm sensory learning."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.ib_local import alif_executive as alif
from scripts.ib_local import gaze_pointer as gp_mod
from scripts.ib_local.unified_world_port import ObservationalDialogueStream, UnifiedWorldPort
from tests.test_gaze_pointer import _make_gaze_engine
from tests.test_unified_world_port import FakeBaseWorld


class _RecordingTable:
    """Minimal Fast Table stub recording every byte sequence passed to `run_credit` vs `frozen_table_logp`."""

    def __init__(self):
        self.credit_segments: list[bytes] = []

    def run_credit(self, seq: np.ndarray, *, max_window: int):
        # seq is [prev_byte, *seg]
        seg = bytes(seq[1:].tolist())
        self.credit_segments.append(seg)
        n = len(seg)
        probs = np.full((n, 256), 1.0 / 256.0, dtype=np.float32)
        return None, probs, None


def test_observational_dialogue_stream_spans_and_utf8_safety():
    corpus = Path('data/cgb_observational_dialogues_20261124/scenes.jsonl')
    assert corpus.exists()
    stream = ObservationalDialogueStream(corpus)
    resp, cont, info = stream.take_ambient(320, 320)
    assert len(resp) == 320 and len(cont) == 320
    full = np.concatenate((resp, cont)).tobytes()
    # Clean UTF-8 decode (no sliced mid-character bytes anywhere!).
    decoded = full.decode('utf-8')
    assert '\ufffd' not in decoded

    spans = info['envelope_mask_spans']
    assert len(spans) >= 4
    masked_slices = [full[s:e].decode('utf-8') for s, e in spans]
    # Every speaker/roster header is strictly inside masked spans.
    assert any(h in ('妈妈：', '爸爸：') for h in masked_slices)
    assert any(h.startswith('[在场]') for h in masked_slices)

    # Unmasked bytes contain actual dialogue content, NOT raw speaker headers.
    mask = np.ones(len(full), dtype=bool)
    for s, e in spans:
        mask[s:e] = False
    unmasked_text = bytes(b for b, keep in zip(full, mask) if keep).decode('utf-8')
    assert '妈妈：' not in unmasked_text
    assert '爸爸：' not in unmasked_text
    assert '[在场]' not in unmasked_text
    assert '窗外' in unmasked_text or '潮汐' in unmasked_text


def test_span_masked_table_never_writes_headers_to_fast_table(monkeypatch):
    # Patch `frozen_table_logp` so our lightweight table stub doesn't need full C++ table.
    def _fake_frozen(table, seq):
        return np.full((len(seq), 256), -np.log(256.0), dtype=np.float32)

    monkeypatch.setattr('scripts.ib_local.navigation_live_witness.frozen_table_logp', _fake_frozen)

    corpus = Path('data/cgb_observational_dialogues_20261124/scenes.jsonl')
    stream = ObservationalDialogueStream(corpus)
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
    assert '[工具:' not in written
    assert len(written) > 100


def test_unified_world_port_dialogue_mix_preserves_shared_page_and_updates_gaze():
    corpus = Path('data/cgb_observational_dialogues_20261124/scenes.jsonl')
    eng = _make_gaze_engine(d_model=32)
    ptr = gp_mod.configure(eng, {}, d_gaze=16)

    base = FakeBaseWorld('Apples fall to Earth because of gravity. ' * 200)
    port = UnifiedWorldPort(
        base,
        caregiver=None,
        dialogue_demo_corpus=corpus,
        dialogue_demo_ratio=0.25,
    )
    port.attach_engine(eng)

    channels = []
    for _ in range(8):
        resp, cont, info = port.take_ambient(256, 256)
        channels.append(info['channel'])
        if info['channel'] == 'observational_dialogue':
            assert info['envelope_mask_spans']

    # Exact Bresenham 0.25 ratio over 8 reads -> 2 dialogue reads, 6 storybook reads.
    assert channels.count('observational_dialogue') == 2
    assert channels.count('storybook_ambient') == 6
    assert port.dialogue_reads == 2

    # Shared storybook page stays strictly on storybook text (never polluted by dialogue headers!).
    shared_text = bytes(port.shared_page_bytes).decode('utf-8', 'ignore')
    assert '妈妈：' not in shared_text
    assert 'Apples fall' in shared_text

    # GazeObjectPointer organically acquired sensory fingerprints for observed Mom & Dad!
    assert 'mother' in ptr.fingerprints
    assert 'parent' in ptr.fingerprints
    assert 'book' in ptr.fingerprints

    # Checkpoint roundtrip preserves `dialogue_cursor` and `dialogue_reads`.
    saved = port.state_dict()
    port2 = UnifiedWorldPort(
        FakeBaseWorld('Apples fall to Earth because of gravity. ' * 200),
        caregiver=None,
        dialogue_demo_corpus=corpus,
        dialogue_demo_ratio=0.25,
    )
    port2.load_state_dict(saved)
    assert port2.dialogue_reads == port.dialogue_reads
    assert port2.dialogue_demo.cursor == port.dialogue_demo.cursor
