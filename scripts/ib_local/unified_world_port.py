"""Single unified UTF-8 world interaction port for MengLi.

Keeps MengLi's body strictly single-ported (1 UTF-8 input/output byte stream,
zero per-participant neural heads or extra gradient ports) while multiplexing
multiple world-side channels cleanly:

  1. ``LiveParticipantChannel`` — live human parent or guest input (immediate
     preemption via in-process queue or append-only JSONL inbox file). Recorded
     into the existing DSH Mother session as ``actual_parent_care`` observations
     so Mother remembers who spoke without ever fabricating it as her own words.
  2. ``ToolChannel`` — pluggable tool demonstrations or tool invocations
     returning ``[工具:名称]`` observation bytes through the same UTF-8 stream.
  3. ``DshMirrorCaregiverChannel`` — wraps the EXISTING ``FamilyCaregiver`` /
     ``DshMother`` session (never a second parallel Mother), firing on
     contingent ALIF spikes with cooldown, alternating ``mirror_visible`` and
     ``book_only`` perspectives, and keeping ``system_prompt`` + prior prefix
     100% immutable for high DeepSeek server-side KV prompt-cache hit rates.
  4. ``StorybookChannel`` — high-throughput local reading (``take_ambient`` at
     0 API calls for 1600B H-Net calm flow, plus fast local fallback on
     cooldown spikes), with optional sidecar Chinese corpus interleaving that
     never disturbs inherited English ``NaturalTextWorld`` addresses.
"""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable

import numpy as np

from scripts.ib_local.body_mirror import (
    AdaptiveCarePlanTracker,
    BodyMirrorSnapshot,
    STAGE_INFANT,
    build_five_element_reflection_dossier,
    capture_body_mirror,
    classify_utterance,
    clean_utf8_excerpt,
    format_caregiver_turn,
    parse_mother_reflection,
)


_TERM_RE = re.compile(r'[a-z]{3,}|[\u4e00-\u9fff]{2,}')
_CJK_RUN_RE = re.compile(r'[\u4e00-\u9fff]+')
_VOCAL_MOTHER_RE = re.compile(r'(妈妈|妈咪|^\s*(?:mama|mommy|mom)\b)', re.IGNORECASE)
_VOCAL_PARENT_RE = re.compile(r'(爸爸|爹地|^\s*(?:papa|daddy|dad)\b)', re.IGNORECASE)
_NAME_OPENING_RE = re.compile(r'^\s*(?:小?梦漓|meng\s*li)\b|^\s*小?梦漓', re.IGNORECASE)
_LEADING_STAGE_RE = re.compile(r'^\s*(?:[（(][^（）()]{1,24}[）)]|\*[^*\n]{1,30}\*)\s*')
_INLINE_STAGE_RE = re.compile(
    r'[（(][^（）()]{0,12}(?:笑|微笑|轻声|柔声|点头|眨眼|摸|抱|拍|叹|顿|看|指|摇|低声|温声|smile|laugh|chuckl|nod|gently|softly|whisper|hug|pat|warm)[^（）()]{0,12}[）)]|\*[^*\n]{1,30}\*',
    re.IGNORECASE,
)


_MOTHER_QUIET_RE = re.compile(
    r'^\s*@?(?:安静陪伴|静静陪着|保持安静|quiet|silent)\s*[。！!,.，]*\s*$',
    re.IGNORECASE,
)
_MOTHER_BOOK_RE = re.compile(
    r'^\s*@(?:翻书|找书|书页|book)\s*[:：]\s*([^\s\n，。！？!?]{1,48})(?:[\s，,]+(.*))?$',
    re.IGNORECASE | re.DOTALL,
)
_MOTHER_PARENT_RE = re.compile(
    r'^\s*@(?:对爸爸|爸爸|dad|parent)\s*[:：]?\s*(.*)$',
    re.IGNORECASE | re.DOTALL,
)
_MOTHER_MENGLI_RE = re.compile(
    r'^\s*@(?:对梦漓|梦漓|mengli)\s*[:：]?\s*(.*)$',
    re.IGNORECASE | re.DOTALL,
)


def _strip_stage_directions(text: str) -> str:
    """Remove theatrical stage directions like `（轻声笑）`, `(smiles)`, `*chuckles*` before delivery."""
    cleaned = _LEADING_STAGE_RE.sub('', text)
    cleaned = _INLINE_STAGE_RE.sub('', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned or text.strip()


def parse_mother_direction(raw_reply: str) -> tuple[str, str, str]:
    """Parse & strip Mother's optional leading action directive before MengLi's UTF-8 Membrane sees it.

    Returns ``(mother_out, book_query, clean_spoken_text)`` where ``mother_out`` is one of:
      - ``'none'``:   ``@安静陪伴`` — Mother chooses non-interrupting silent presence (0 spoken bytes).
      - ``'book'``:   ``@翻书:<关键词> [口语]`` — Mother flips the active storybook before optional speech.
      - ``'parent'``: ``@对爸爸 [口语]`` — Mother addresses Dad (when present); MengLi overhears.
      - ``'mengli'``: default or ``@对梦漓 [口语]`` — Mother speaks directly to MengLi.
    """
    text = (raw_reply or '').strip()
    if not text:
        return 'mengli', '', ''
    if _MOTHER_QUIET_RE.match(text):
        return 'none', '', ''
    m_book = _MOTHER_BOOK_RE.match(text)
    if m_book:
        query = m_book.group(1).strip()
        rest = (m_book.group(2) or '').strip()
        clean = _strip_stage_directions(rest) if rest else ''
        return 'book', query, clean
    m_parent = _MOTHER_PARENT_RE.match(text)
    if m_parent:
        rest = (m_parent.group(1) or '').strip()
        clean = _strip_stage_directions(rest) if rest else ''
        return 'parent', '', clean
    m_mengli = _MOTHER_MENGLI_RE.match(text)
    if m_mengli:
        rest = (m_mengli.group(1) or '').strip()
        clean = _strip_stage_directions(rest) if rest else ''
        return 'mengli', '', clean
    return 'mengli', '', _strip_stage_directions(text)


def _terms(text: str) -> set[str]:
    return set(_TERM_RE.findall(text.lower()))


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RUN_RE.search(text or ''))


def _is_vocal_mother_call(text: str) -> bool:
    return bool(text and _VOCAL_MOTHER_RE.search(text))


def _is_vocal_parent_call(text: str) -> bool:
    return bool(text and _VOCAL_PARENT_RE.search(text))


class SidecarCorpusStream:
    """Contiguous UTF-8 sidecar storybook (e.g., Chinese corpus) with exact cursor, seen mask & keyword navigation."""

    def __init__(self, path: str | Path, *, cursor: int = 0):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        if len(raw) < 2048:
            raise ValueError(f'sidecar corpus too small ({len(raw)} bytes): {self.path}')
        self._raw_bytes = raw
        self.data = np.frombuffer(raw, dtype=np.uint8)
        self.seen = np.zeros(len(self.data), dtype=bool)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.cursor = int(cursor) % len(self.data)
        self.hybrid_index = None
        idx_prefix = self.path.parent / 'hybrid_gemma2_index'
        if idx_prefix.with_suffix('.npz').exists() and idx_prefix.with_suffix('.json').exists():
            try:
                from scripts.ib_local.book_semantic_index import HybridBookIndex
                cand_idx = HybridBookIndex.load(idx_prefix, raw)
                if not cand_idx.corpus_sha256 or cand_idx.corpus_sha256 == self.sha256:
                    self.hybrid_index = cand_idx
            except Exception:
                self.hybrid_index = None

    def _align_utf8_start(self, pos: int) -> int:
        """Advance past at most 3 UTF-8 continuation bytes (0x80..0xBF) so slices never split a character."""
        n = len(self.data)
        p = int(pos) % n
        for _ in range(3):
            if 0x80 <= int(self.data[p]) < 0xC0:
                p = (p + 1) % n
            else:
                break
        return p

    @staticmethod
    def _incomplete_utf8_tail_len(arr: np.ndarray) -> int:
        """Return number of trailing bytes (0..3) forming an incomplete UTF-8 sequence."""
        n = len(arr)
        if n == 0:
            return 0
        k = 0
        while k < 3 and k < n and 0x80 <= int(arr[n - 1 - k]) < 0xC0:
            k += 1
        if k == n:
            return k
        lead = int(arr[n - 1 - k])
        if lead < 0x80:
            return 0 if k == 0 else k
        if (lead & 0xE0) == 0xC0:
            expected = 2
        elif (lead & 0xF0) == 0xE0:
            expected = 3
        elif (lead & 0xF8) == 0xF0:
            expected = 4
        else:
            return k + 1
        actual = k + 1
        return actual if actual < expected else 0

    def _slice_from(self, start: int, response_bytes: int, holdout_bytes: int) -> tuple[np.ndarray, np.ndarray, int, int, int]:
        total = int(response_bytes) + int(holdout_bytes)
        n = len(self.data)
        start = self._align_utf8_start(start)
        if start + total <= n:
            buf = self.data[start:start + total].copy()
            prior_resp = int(self.seen[start:start + response_bytes].sum())
            prior_hold = int(self.seen[start + response_bytes:start + total].sum())
            self.seen[start:start + total] = True
            raw_end = start + total
        else:
            part1 = self.data[start:]
            rem = total - len(part1)
            part2 = self.data[:rem]
            buf = np.concatenate((part1, part2)).copy()
            seen_wrap = np.concatenate((self.seen[start:], self.seen[:rem]))
            prior_resp = int(seen_wrap[:response_bytes].sum())
            prior_hold = int(seen_wrap[response_bytes:].sum())
            self.seen[start:] = True
            self.seen[:rem] = True
            raw_end = rem
        # Never end a Chinese chunk mid-UTF-8 character (which forces engine.pending to be a cut lead byte
        # and forces MengLi's next utterance to begin with broken continuation bytes ``).
        tail = self._incomplete_utf8_tail_len(buf)
        if 0 < tail < holdout_bytes:
            buf[-tail:] = ord('\n')
            next_cursor = (raw_end - tail) % n
        else:
            next_cursor = self._align_utf8_start(raw_end)
        self.cursor = next_cursor
        return buf[:response_bytes], buf[response_bytes:], start, prior_resp, prior_hold

    def take_ambient(self, response_bytes: int, holdout_bytes: int) -> tuple[np.ndarray, np.ndarray, dict]:
        if response_bytes < 1 or holdout_bytes < 1:
            raise ValueError('positive response and holdout lengths required')
        resp, cont, start, prior_resp, prior_hold = self._slice_from(self.cursor, response_bytes, holdout_bytes)
        info = dict(
            start=int(start),
            anchor=int(start),
            match=0,
            navigation_response_bytes=int(len(resp)),
            response_previously_seen_bytes=prior_resp,
            holdout_previously_seen_bytes=prior_hold,
            holdout_position_fresh=(prior_hold == 0),
            source='chinese_storybook',
            sidecar_corpus=self.path.name,
            sidecar_cursor=int(self.cursor),
        )
        return resp, cont, info

    def take(self, utterance: bytes, response_bytes: int, holdout_bytes: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """Navigate the Chinese/sidecar storybook via Hybrid EmbeddingGemma-2 + e-means + IDF search."""
        if response_bytes < 1 or holdout_bytes < 1:
            raise ValueError('positive response and holdout lengths required')
        query_text = clean_utf8_excerpt(utterance, max_chars=96) if utterance else ''

        if query_text and self.hybrid_index is not None:
            hit = self.hybrid_index.search(query_text, cursor=self.cursor)
            if hit is not None:
                resp, cont, actual_start, prior_resp, prior_hold = self._slice_from(
                    int(hit['start']), response_bytes, holdout_bytes,
                )
                return resp, cont, dict(
                    start=int(actual_start),
                    anchor=int(hit['start']),
                    match=1,
                    matched_term=str(hit.get('matched_term') or ''),
                    retrieval_method=str(hit.get('method') or 'hybrid_gemma2_emeans_idf'),
                    retrieval_score=float(hit.get('score', 0.0)),
                    retrieval_semantic_cos=float(hit.get('semantic_cos', 0.0)),
                    navigation_response_bytes=int(len(resp)),
                    response_previously_seen_bytes=prior_resp,
                    holdout_previously_seen_bytes=prior_hold,
                    holdout_position_fresh=(prior_hold == 0),
                    source='chinese_storybook',
                    sidecar_corpus=self.path.name,
                    sidecar_cursor=int(self.cursor),
                )
            return self.take_ambient(response_bytes, holdout_bytes)

        # Fallback lexical search (when no prebuilt index exists): slide over ALL 2-4 char
        # CJK n-grams & 3+ char English words — NEVER search noisy single characters!
        from scripts.ib_local.book_semantic_index import extract_lexical_terms
        candidates = extract_lexical_terms(query_text) if query_text else []

        hit_pos = -1
        matched_term = ''
        n = len(self._raw_bytes)
        search_from = (self.cursor + 1) % n
        for term in candidates[:12]:
            needle = term.encode('utf-8')
            pos = self._raw_bytes.find(needle, search_from)
            if pos < 0:
                pos = self._raw_bytes.find(needle, 0, search_from)
            if pos >= 0:
                hit_pos = pos
                matched_term = term
                break

        if hit_pos >= 0:
            win_start = max(0, hit_pos - 96)
            prefix = self._raw_bytes[win_start:hit_pos]
            nl = prefix.rfind(b'\n')
            dot = prefix.rfind('。'.encode('utf-8'))
            if nl >= 0:
                start = win_start + nl + 1
            elif dot >= 0:
                start = win_start + dot + 3
            else:
                start = hit_pos
            resp, cont, actual_start, prior_resp, prior_hold = self._slice_from(start, response_bytes, holdout_bytes)
            return resp, cont, dict(
                start=int(actual_start),
                anchor=int(hit_pos),
                match=1,
                matched_term=matched_term,
                retrieval_method='sliding_ngram_fallback',
                navigation_response_bytes=int(len(resp)),
                response_previously_seen_bytes=prior_resp,
                holdout_previously_seen_bytes=prior_hold,
                holdout_position_fresh=(prior_hold == 0),
                source='chinese_storybook',
                sidecar_corpus=self.path.name,
                sidecar_cursor=int(self.cursor),
            )
        return self.take_ambient(response_bytes, holdout_bytes)


class MultiSourceEventStream:
    """Unified multi-source event stream over the single UTF-8 Membrane.

    Reads structured JSONL multi-speaker records (e.g., large-scale offline bilingual
    natural dialogues) and emits exact byte-level `source_transitions` and
    `envelope_mask_spans`:
      - Every speaker/source header (`妈妈：`, `爸爸：`, `[在场] ...\\n`, etc.) enters
        the 32-layer GDN state in full while being masked out of BOTH neural LM loss
        and Fast Table writes.
      - Every spoken body byte and turn-ending newline is learned normally (`score_mask=1`).
      - Strictly marked `is_action_consequence=False` so observational streams are NEVER
        conflated with the consequences of MengLi's own actions.
    """

    def __init__(self, path: str | Path, *, cursor: int = 0):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f'multi-source event corpus not found: {self.path}')
        # Store only lightweight byte offsets (~128 KB for 16k scenes instead of ~300 MB of Python dicts).
        offsets: list[int] = []
        with self.path.open('rb') as f:
            while True:
                pos = f.tell()
                raw_line = f.readline()
                if not raw_line:
                    break
                if raw_line.strip():
                    offsets.append(pos)
        if not offsets:
            raise ValueError(f'multi-source event corpus has no valid scenes: {self.path}')
        self._offsets = offsets
        self.scene_count = len(offsets)
        self.cursor = int(cursor) % self.scene_count

    @property
    def scenes(self) -> list[int]:
        """Lightweight length-compatible view (`len(stream.scenes)` without holding parsed dicts in RAM)."""
        return self._offsets

    def _read_scene(self, index: int) -> dict | None:
        pos = self._offsets[int(index) % self.scene_count]
        with self.path.open('rb') as f:
            f.seek(pos)
            raw_line = f.readline()
        line = raw_line.decode('utf-8', errors='replace').strip()
        if not line:
            return None
        obj = json.loads(line)
        turns = obj.get('turns')
        return obj if isinstance(turns, list) and turns else None

    @staticmethod
    def _format_header(speaker: str) -> str:
        s = str(speaker).strip()
        if not s:
            return ''
        if s.startswith('工具:') or s.startswith('tool:'):
            return f'[{s}] '
        return f'{s}：'

    @staticmethod
    def _speaker_object_id(turn: dict) -> str:
        oid = turn.get('id') or turn.get('speaker_id')
        if isinstance(oid, str) and oid.strip():
            return oid.strip()
        sp = str(turn.get('speaker', '')).strip()
        if sp in ('妈妈', 'Mom', 'Mother', 'mother'):
            return 'mother'
        if sp in ('爸爸', 'Dad', 'Father', 'parent'):
            return 'parent'
        if sp.startswith('工具:'):
            return 'tool:' + sp.split(':', 1)[1].strip()
        if sp.startswith('tool:'):
            return sp
        return sp or 'unknown'

    @staticmethod
    def _infer_addressee_id(turn: dict, participants: list[str], speaker_id: str) -> str:
        explicit = turn.get('addressee_id') or turn.get('addressee')
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        others = [p for p in participants if p != speaker_id]
        if len(others) == 1:
            return others[0]
        return 'all' if others else 'observer'

    def _encode_scene(self, scene: dict) -> tuple[bytes, list[tuple[int, int]], list[dict], list[tuple[str, bytes]]]:
        buf = bytearray()
        spans: list[tuple[int, int]] = []
        transitions: list[dict] = []
        observed: list[tuple[str, bytes]] = []
        turns = scene.get('turns', [])
        participants: list[str] = []
        for t in turns:
            oid = self._speaker_object_id(t)
            if oid not in participants:
                participants.append(oid)

        roster = scene.get('roster')
        if isinstance(roster, str) and roster.strip():
            start = len(buf)
            buf.extend((roster.strip() + '\n').encode('utf-8'))
            spans.append((start, len(buf)))
            transitions.append(dict(
                offset=start,
                speaker_id='umwelt',
                addressee_id='mengli',
                mask_start=start,
                mask_end=len(buf),
            ))
        for turn in turns:
            text = str(turn.get('text', '')).strip()
            if not text:
                continue
            oid = self._speaker_object_id(turn)
            addr = self._infer_addressee_id(turn, participants, oid)
            header = self._format_header(turn.get('speaker', ''))
            hs = len(buf)
            if header:
                buf.extend(header.encode('utf-8'))
                he = len(buf)
                spans.append((hs, he))
            else:
                he = hs
            transitions.append(dict(
                offset=hs,
                speaker_id=oid,
                addressee_id=addr,
                mask_start=hs,
                mask_end=he,
            ))
            body_b = (text + '\n').encode('utf-8')
            buf.extend(body_b)
            if oid != 'unknown':
                observed.append((oid, body_b))
        return bytes(buf), spans, transitions, observed

    @staticmethod
    def _truncate_utf8_safe(data: bytes, max_len: int) -> bytes:
        if len(data) <= max_len:
            return data
        cut = max_len
        while cut > 0 and (data[cut] & 0xC0) == 0x80:
            cut -= 1
        return data[:cut]

    def take_ambient(self, response_bytes: int, holdout_bytes: int) -> tuple[np.ndarray, np.ndarray, dict]:
        if response_bytes < 1 or holdout_bytes < 1:
            raise ValueError('positive response and holdout lengths required')
        total = int(response_bytes + holdout_bytes)
        buf = bytearray()
        spans: list[tuple[int, int]] = []
        transitions: list[dict] = []
        observed: list[tuple[str, bytes]] = []
        scene_names: list[str] = []
        start_cursor = int(self.cursor)

        guard = 0
        while len(buf) < total and guard < self.scene_count * 4 + 4:
            guard += 1
            sc = self._read_scene(self.cursor)
            self.cursor = (self.cursor + 1) % self.scene_count
            if sc is None:
                continue
            sc_bytes, sc_spans, sc_trans, sc_obs = self._encode_scene(sc)
            if not sc_bytes:
                continue
            offset = len(buf)
            scene_names.append(str(sc.get('scene', 'dialogue')))
            for s, e in sc_spans:
                spans.append((offset + s, offset + e))
            for tr in sc_trans:
                transitions.append(dict(
                    offset=offset + int(tr['offset']),
                    speaker_id=tr['speaker_id'],
                    addressee_id=tr['addressee_id'],
                    mask_start=offset + int(tr['mask_start']),
                    mask_end=offset + int(tr['mask_end']),
                ))
            buf.extend(sc_bytes)
            observed.extend(sc_obs)

        raw = self._truncate_utf8_safe(bytes(buf), total)
        if len(raw) < total:
            raw = raw + (b'\n' * (total - len(raw)))
        clipped_spans = [(s, min(e, total)) for s, e in spans if s < total and min(e, total) > s]
        clipped_trans = [
            dict(
                offset=int(tr['offset']),
                speaker_id=str(tr['speaker_id']),
                addressee_id=str(tr['addressee_id']),
                mask_start=int(tr['mask_start']),
                mask_end=min(int(tr['mask_end']), total),
            )
            for tr in transitions
            if int(tr['offset']) < total
        ]
        primary_speaker = next(
            (t['speaker_id'] for t in clipped_trans if t['speaker_id'] != 'umwelt'),
            'multi',
        )
        primary_addressee = next(
            (t['addressee_id'] for t in clipped_trans if t['speaker_id'] != 'umwelt'),
            'observer',
        )
        arr = np.frombuffer(raw, dtype=np.uint8).copy()
        resp = arr[:response_bytes]
        cont = arr[response_bytes:total]
        info = dict(
            source='observational_dialogue',
            dialogue_corpus=self.path.name,
            dialogue_start_cursor=start_cursor,
            dialogue_cursor=int(self.cursor),
            dialogue_scenes=scene_names,
            speaker_id=primary_speaker,
            addressee_id=primary_addressee,
            is_action_consequence=False,
            source_transitions=clipped_trans,
            envelope_mask_spans=clipped_spans,
            observed_speakers=observed,
            navigation_response_bytes=int(len(resp)),
        )
        return resp, cont, info


ObservationalDialogueStream = MultiSourceEventStream


class UnifiedWorldPort:
    """Unified world-side multiplexer over a single UTF-8 body boundary."""

    def __init__(
        self,
        base,
        *,
        caregiver=None,
        history: int = 8,
        ambient_policy: str = 'reply',
        mother_cooldown: int = 3,
        mother_max_gap: int = 8,
        mirror_ratio: float = 0.7,
        auto_commit: bool = False,
        inbox_path: str | Path | None = None,
        sidecar_corpus: str | Path | None = None,
        sidecar_mix_ratio: float = 0.0,
        dialogue_demo_corpus: str | Path | None = None,
        dialogue_demo_ratio: float = 0.0,
        initial_stage: str = STAGE_INFANT,
        user_plan_text: str | None = None,
        respond_on_child_speech: bool = True,
    ):
        if type(history) is not int or not 1 <= history <= 32:
            raise ValueError('mother context history must lie in [1, 32]')
        if ambient_policy not in ('reply', 'quiet'):
            raise ValueError('unknown ambient navigation policy')
        if mother_cooldown < 0 or mother_max_gap < 1:
            raise ValueError('mother_cooldown >= 0 and mother_max_gap >= 1 required')
        if not (0.0 <= float(mirror_ratio) <= 1.0):
            raise ValueError('mirror_ratio must lie in [0, 1]')
        if not (0.0 <= float(sidecar_mix_ratio) <= 1.0):
            raise ValueError('sidecar_mix_ratio must lie in [0, 1]')
        if not (0.0 <= float(dialogue_demo_ratio) <= 1.0):
            raise ValueError('dialogue_demo_ratio must lie in [0, 1]')

        self.base = base
        self.caregiver = caregiver
        self.history = int(history)
        self.ambient_policy = ambient_policy
        self.mother_cooldown = int(mother_cooldown)
        self.mother_max_gap = int(mother_max_gap)
        self.mirror_ratio = float(mirror_ratio)
        self.auto_commit = bool(auto_commit)
        self.inbox_path = None if inbox_path is None else Path(inbox_path)
        self.inbox_offset = 0
        self.sidecar_mix_ratio = float(sidecar_mix_ratio) if sidecar_corpus is not None else 0.0
        self.sidecar = None if sidecar_corpus is None else SidecarCorpusStream(sidecar_corpus)
        self.dialogue_demo_ratio = float(dialogue_demo_ratio) if dialogue_demo_corpus is not None else 0.0
        self.dialogue_demo = (
            None if dialogue_demo_corpus is None else MultiSourceEventStream(dialogue_demo_corpus)
        )
        self.dialogue_reads = 0
        self.event_ledger: list[dict] = []
        self.event_ledger_cap: int = 64
        self._base_hybrid_index = None
        self._base_hybrid_offset = 0
        self._base_hybrid_checked = False

        # Open Yoneda / Presence Roster & Membrane protocol settings:
        #   - Any external object (person, tool, book) can join/leave dynamically.
        #   - Every incoming Membrane slice can carry an explicit source envelope
        #     (`[来自:妈妈]`, `[来自:书页]`, `[来自:工具:...]`) so MengLi learns
        #     each black-box object purely through its interaction morphisms.
        #   - When Mother is present, active speech (`spoke == True`) is NEVER
        #     silently dropped to unrelated book pages by quiet-checkin cooldown.
        self.mother_present = (caregiver is not None)
        self.respond_on_child_speech = bool(respond_on_child_speech)
        self.envelope_tags = True
        self.presence_roster: dict[str, dict] = {'book': dict(kind='book', label='书页', description='阅读与检索书页')}
        if caregiver is not None:
            self.presence_roster['mother'] = dict(kind='caregiver', label='妈妈', description='陪读、回应与讲解')
        self.active_floor: str = 'book'
        self.last_book_source: str = 'text'
        self.active_interlocutor: str | None = ('mother' if caregiver is not None else None)
        self.awaiting_reply_from: str | None = None
        self.consecutive_social_turns: int = 0
        self.max_consecutive_social_turns: int = 3
        self.turns_since_name_call: int = 3
        self.recent_mother_openings: list[str] = []
        self._roster_dirty_notice: str | None = None
        self._dynamic_objects: dict[str, Callable[[str], str]] = {}

        # Clean DSH-aligned caregiver transcript (updated ONLY on actual Mother/Parent turns).
        self.mother_transcript: list[tuple[str, str]] = []
        self.care_plan_tracker = AdaptiveCarePlanTracker(
            initial_stage=initial_stage,
            user_plan_text=user_plan_text,
        )

        # Rolling shared storybook page seen by both MengLi and Mother (2048B covers full 1280-1600B calm pieces
        # so `stumble_excerpt` stays aligned with the exact same story page in `capture_body_mirror`).
        self.shared_page_bytes = bytearray()
        self.shared_page_cap = 2048

        # Scheduling & channel state.
        self.ambient_reads = 0
        self.sidecar_reads = 0
        self.spike_turns = 0
        self.spikes_since_mother = int(mother_max_gap)  # eligible on first quiet pause if desired
        self.quiet_spikes_since_mother = 0
        self.mother_turns = 0
        self.mother_proactive_turns = 0
        self.mother_responsive_turns = 0
        self.book_navigation_turns = 0
        self.explicit_addressed_turns = 0
        self.mirror_visible_turns = 0
        self.book_only_turns = 0
        self.parent_turns = 0
        self.tool_turns = 0
        self.ambient_fallback_turns = 0
        self.self_evolution_turns = 0
        self.inner_monologue_turns = 0
        self.mother_quiet_turns = 0
        self.mother_book_turns = 0
        self.recent_metal_g: list[float] = []
        self.mirror_g: list[float] = []
        self.book_only_g: list[float] = []
        self.recent_stumbles: list[str] = []
        self.reflections_done = 0
        self.turns_since_reflection = 0

        self._participant_queue: deque[dict] = deque()
        self._tool_demo_queue: deque[dict] = deque()
        self._registered_tools: dict[str, Callable[[str], str]] = {}
        self._pending_tool_notes: list[str] = []
        self._pending_commit: dict | None = None
        self._force_next_mother = False
        self._engine_ref = None
        self._cycle_hint = 0

        # Preserve compatibility with code inspecting `world.mother`.
        self.mother = caregiver

    def __getattr__(self, name: str):
        return getattr(self.__dict__['base'], name)

    def attach_engine(self, engine) -> None:
        """Bind a read-only engine reference for third-person bodily mirroring & GazeObjectPointer."""
        self._engine_ref = engine
        if getattr(self, '_deferred_gaze_state', None) is not None:
            from scripts.ib_local import gaze_pointer as _gp
            if _gp.enabled(engine):
                engine.model.gaze_pointer.load_dynamic_state(self._deferred_gaze_state)
                self._deferred_gaze_state = None

    def _gaze_pointer(self):
        from scripts.ib_local import gaze_pointer as _gp
        return self._engine_ref.model.gaze_pointer if _gp.enabled(self._engine_ref) else None

    def _gaze_candidates(self, *, child_vocal: bool = True) -> list[tuple[str, bytes]]:
        # During silent reading, respect `mother_cooldown` so Mother lets MengLi read quietly
        # instead of monologuing every 15s; when MengLi speaks (`child_vocal=True`), Mother is always responsive.
        quiet_mother_cooldown = (
            (not child_vocal)
            and self.mother_cooldown > 0
            and self.spikes_since_mother < self.mother_cooldown
            and self.awaiting_reply_from != 'mother'
        )
        out: list[tuple[str, bytes]] = []
        for oid, meta in self.presence_roster.items():
            if oid == 'mother' and quiet_mother_cooldown:
                continue
            intro = f'[来自:{meta.get("label", oid)}] {meta.get("description", "")}'.encode('utf-8')
            out.append((oid, intro))
        if not out:
            for oid, meta in self.presence_roster.items():
                intro = f'[来自:{meta.get("label", oid)}] {meta.get("description", "")}'.encode('utf-8')
                out.append((oid, intro))
        return out

    def prepare_gaze_candidates(self, utterance: bytes) -> tuple[list[tuple[str, bytes]] | None, dict]:
        """Pre-world hook for `world_cycle_step`: return `(candidates, meta)` if GazePointer should choose target.

        Returns `(None, meta)` if a live external preemption (Dad inbox / tool demo),
        explicit `@target`, legacy `[tool:...]`, or forced Mother turn already determines the target.
        """
        self._poll_inbox()
        utterance_bytes = bytes(utterance) if utterance else b''
        child_clean = clean_utf8_excerpt(utterance_bytes, max_chars=180)
        child_vocal = (classify_utterance(utterance_bytes)['kind'] != 'silent')
        if (
            self._participant_queue
            or self._tool_demo_queue
            or self._force_next_mother
            or self._parse_explicit_target(child_clean) is not None
            or self._match_registered_tool(child_clean) is not None
        ):
            return None, dict(overridden=True)
        alif_st = getattr(self._engine_ref, 'alif_executive_state', None) or {}
        nll_ema = float(alif_st.get('nll_ema', 0.0) or 0.0)
        nll_last = float(alif_st.get('last_piece_nll', 0.0) or 0.0)
        nll_norm = (nll_last - nll_ema) if (nll_ema > 0.0 and nll_last > 0.0) else 0.0
        return self._gaze_candidates(child_vocal=child_vocal), dict(
            stumble_ratio=self._current_stumble_ratio(),
            nll_norm=nll_norm,
            last_speaker=self.active_interlocutor,
            awaiting_from=self.awaiting_reply_from,
        )

    def _observe_gaze_object(self, obj_id: str, raw_bytes: bytes, *, fuse_body_state: bool = True) -> None:
        gp = self._gaze_pointer()
        if gp is not None and raw_bytes:
            gp.observe_object(self._engine_ref, obj_id, raw_bytes, fuse_body_state=fuse_body_state)

    def set_cycle_hint(self, cycle: int) -> None:
        self._cycle_hint = int(cycle)

    def set_user_plan(self, text: str | None) -> dict:
        return self.care_plan_tracker.set_user_plan(text, turn_index=self.mother_turns)

    def force_next_mother(self) -> None:
        self._force_next_mother = True

    def has_pending_interrupt(self) -> bool:
        """Return True if an external human/tool event or open Social Floor awaits MengLi's immediate turn.

        Real-world turn-taking physics:
          - If Dad/guest speaks (inbox/queue) or a tool demo arrives, MengLi hears it immediately
            instead of waiting through 3-6KB of unrelated calm reading.
          - If Mother or Dad just spoke/asked (`awaiting_reply_from is not None`), MengLi gets her
            immediate reply turn BEFORE unrelated storybook pages wash out working memory.
        """
        self._poll_inbox()
        return bool(
            self._participant_queue
            or self._tool_demo_queue
            or self.awaiting_reply_from is not None
        )

    def enqueue_participant(self, text: str, *, speaker: str = '爸爸', role: str = 'parent') -> None:
        """Queue a live human parent or guest utterance for immediate preemption."""
        cleaned = str(text).strip()
        if not cleaned:
            raise ValueError('participant text must be non-empty')
        self._participant_queue.append(dict(speaker=str(speaker).strip() or '爸爸',
                                            role=str(role), text=cleaned))

    def enqueue_tool_demo(self, tool_name: str, output: str) -> None:
        """Queue a caregiver/system tool demonstration delivered on the same byte port."""
        cleaned = str(output).strip()
        if not cleaned:
            raise ValueError('tool output must be non-empty')
        self._tool_demo_queue.append(dict(tool_name=str(tool_name).strip() or '工具', output=cleaned))

    def register_tool(self, name: str, fn: Callable[[str], str], *, description: str = '工具调用') -> None:
        """Register a world-side tool callable via `[tool:<name> <arg>]` or `@工具:<name> <arg>`."""
        key = str(name).strip().lower()
        if not key or not callable(fn):
            raise ValueError('tool requires a non-empty name and callable')
        self._registered_tools[key] = fn
        self.presence_roster[f'tool:{key}'] = dict(kind='tool', label=f'工具:{key}', description=description)
        self._roster_dirty_notice = self._format_roster_notice()

    def register_object(self, obj_id: str, handler: Callable[[str], str], *,
                        label: str | None = None, description: str = '外部交互对象') -> None:
        """Dynamically mount ANY black-box object (person, agent, service, tool) into the Umwelt."""
        key = str(obj_id).strip().lower()
        if not key or not callable(handler):
            raise ValueError('object requires a non-empty id and callable handler')
        self._dynamic_objects[key] = handler
        self.presence_roster[key] = dict(kind='object', label=(label or key), description=description)
        self._roster_dirty_notice = self._format_roster_notice()

    def leave_scene(self, obj_id: str) -> None:
        """Remove an object or caregiver from the active Umwelt presence roster."""
        key = str(obj_id).strip().lower()
        if key == 'mother':
            self.mother_present = False
            self.presence_roster.pop('mother', None)
            if self.active_interlocutor == 'mother':
                self.active_interlocutor = None
            if self.awaiting_reply_from == 'mother':
                self.awaiting_reply_from = None
            if self.active_floor == 'mother':
                self.active_floor = 'book'
        else:
            self._dynamic_objects.pop(key, None)
            self._registered_tools.pop(key.removeprefix('tool:'), None)
            self.presence_roster.pop(key, None)
            if self.active_interlocutor == key:
                self.active_interlocutor = ('mother' if self.mother_present else None)
        self._roster_dirty_notice = self._format_roster_notice()

    def join_mother(self) -> None:
        """Bring Mother back into the active Umwelt presence roster."""
        if self.caregiver is None:
            raise ValueError('no caregiver configured')
        self.mother_present = True
        self.presence_roster['mother'] = dict(kind='caregiver', label='妈妈', description='陪读、回应与讲解')
        if self.active_interlocutor is None:
            self.active_interlocutor = 'mother'
        self._roster_dirty_notice = self._format_roster_notice()

    def _format_roster_notice(self) -> str:
        items = []
        for meta in self.presence_roster.values():
            desc = str(meta.get('description') or '').strip()
            items.append(f'{meta["label"]}({desc})' if desc else str(meta['label']))
        return '[在场] ' + '；'.join(items)

    def _parse_explicit_target(self, child_text: str) -> tuple[str, str] | None:
        """Parse optional explicit addressing (`@妈妈`, `@书页`, `@工具:calc`, `@obj_id`) from MengLi's stream."""
        if not child_text:
            return None
        m_tool = re.match(r'^\s*@(?:工具|tool)[:：]([^\s:：]+)(?:[:：]\s*|\s+)?(.*)$', child_text, re.IGNORECASE)
        if m_tool:
            tname = m_tool.group(1).strip().lower()
            if tname in self._registered_tools:
                return f'tool:{tname}', (m_tool.group(2) or '').strip()
        m = re.match(r'^\s*@([^\s:：]+)(?:[:：]\s*|\s+)?(.*)$', child_text)
        if not m:
            return None
        raw_target = m.group(1).strip()
        payload = (m.group(2) or '').strip()
        low = raw_target.lower()
        if low in ('书页', '书', 'book', 'text'):
            return 'book', payload
        if low in ('妈妈', 'mother', 'mama'):
            return 'mother', payload
        if low in ('爸爸', 'parent', 'dad'):
            return 'parent', payload
        if low in self._registered_tools:
            return f'tool:{low}', payload
        if low in self._dynamic_objects:
            return low, payload
        for oid, meta in self.presence_roster.items():
            if meta.get('label', '').lower() == low:
                return oid, payload
        return None

    def _poll_inbox(self) -> None:
        if self.inbox_path is None or not self.inbox_path.exists():
            return
        size = self.inbox_path.stat().st_size
        if size <= self.inbox_offset:
            return
        with self.inbox_path.open('rb') as f:
            f.seek(self.inbox_offset)
            chunk = f.read()
        self.inbox_offset += len(chunk)
        for line in chunk.decode('utf-8', 'replace').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.enqueue_participant(line, speaker='爸爸', role='parent')
                continue
            kind = msg.get('kind', 'participant')
            if kind == 'care_plan':
                self.set_user_plan(msg.get('text'))
            elif kind == 'tool_demo' and msg.get('output'):
                self.enqueue_tool_demo(msg.get('tool_name', '工具'), msg['output'])
            elif msg.get('text'):
                self.enqueue_participant(
                    msg['text'],
                    speaker=msg.get('speaker', '爸爸'),
                    role=msg.get('role', 'parent'),
                )

    def _record_shared_page(self, data: np.ndarray) -> None:
        raw = data.tobytes()
        if not raw:
            return
        self.shared_page_bytes.extend(raw)
        if len(self.shared_page_bytes) > self.shared_page_cap:
            del self.shared_page_bytes[:-self.shared_page_cap]

    def _use_sidecar_now(self) -> bool:
        if self.sidecar is None or self.sidecar_mix_ratio <= 0.0:
            return False
        k = self.ambient_reads
        r = self.sidecar_mix_ratio
        return math.floor((k + 1) * r + 1e-12) > math.floor(k * r + 1e-12)

    def _use_dialogue_demo_now(self) -> bool:
        if self.dialogue_demo is None or self.dialogue_demo_ratio <= 0.0:
            return False
        k = self.ambient_reads
        r = self.dialogue_demo_ratio
        return math.floor((k + 1) * r + 1e-12) > math.floor(k * r + 1e-12)

    def _next_perspective_mode(self) -> str:
        """Deterministic Bresenham alternation between `mirror_visible` and `book_only`."""
        k = self.mother_turns
        r = self.mirror_ratio
        visible = math.floor((k + 1) * r + 1e-12) > math.floor(k * r + 1e-12)
        return 'mirror_visible' if visible else 'book_only'

    @staticmethod
    def _build_pack_transitions(
        primary_speaker_id: str,
        addressee_id: str,
        said_len: int,
        env_prefix_len: int,
        env_mask_spans: list[tuple[int, int]],
        total_len: int,
    ) -> list[dict]:
        """Construct exact byte-offset source transitions for a packed speaker/tool + book-resume chunk."""
        transitions: list[dict] = [
            dict(
                offset=0,
                speaker_id=str(primary_speaker_id),
                addressee_id=str(addressee_id),
                mask_start=0,
                mask_end=int(env_prefix_len),
            )
        ]
        if int(said_len) < int(total_len):
            book_mask_end = int(said_len)
            for s, e in env_mask_spans:
                if int(s) == int(said_len):
                    book_mask_end = int(e)
                    break
            transitions.append(dict(
                offset=int(said_len),
                speaker_id='book',
                addressee_id='mengli',
                mask_start=int(said_len),
                mask_end=book_mask_end,
            ))
        return transitions

    def _finalize_membrane_event(
        self,
        info: dict,
        *,
        speaker_id: str,
        addressee_id: str,
        is_action_consequence: bool,
        source_transitions: list[dict],
        record_ledger: bool = True,
    ) -> dict:
        """Attach the canonical 5-part Membrane event contract & persist a compact ledger entry."""
        out = dict(info)
        norm_trans = [
            dict(
                offset=int(tr['offset']),
                speaker_id=str(tr['speaker_id']),
                addressee_id=str(tr['addressee_id']),
                mask_start=int(tr.get('mask_start', tr['offset'])),
                mask_end=int(tr.get('mask_end', tr['offset'])),
            )
            for tr in source_transitions
        ]
        out['event_schema'] = 1
        out['speaker_id'] = str(speaker_id)
        out['addressee_id'] = str(addressee_id)
        out['is_action_consequence'] = bool(is_action_consequence)
        out['source_transitions'] = norm_trans
        if not out.get('envelope_mask_spans'):
            derived_spans = [
                (int(tr['mask_start']), int(tr['mask_end']))
                for tr in norm_trans
                if int(tr['mask_end']) > int(tr['mask_start'])
            ]
            if derived_spans:
                out['envelope_mask_spans'] = derived_spans
        if record_ledger:
            entry = dict(
                event_schema=1,
                cycle=int(self._cycle_hint or self.spike_turns),
                channel=str(out.get('channel') or ''),
                speaker_id=str(speaker_id),
                addressee_id=str(addressee_id),
                send_target_id=out.get('send_target_id'),
                observe_target_id=out.get('observe_target_id'),
                interaction_mode=out.get('interaction_mode'),
                mother_out=out.get('mother_out'),
                is_action_consequence=bool(is_action_consequence),
                source_transitions=norm_trans,
                child_text=str(out.get('child_text') or '')[:96],
                reply_excerpt=str(out.get('mother_reply') or '')[:120],
            )
            self.event_ledger.append(entry)
            if len(self.event_ledger) > self.event_ledger_cap:
                del self.event_ledger[:-self.event_ledger_cap]
        return out

    def take_ambient(self, response_bytes: int, holdout_bytes: int):
        """High-throughput calm sensory reading: ZERO caregiver API calls."""
        if self._use_dialogue_demo_now():
            resp, cont, info = self.dialogue_demo.take_ambient(response_bytes, holdout_bytes)
            self.dialogue_reads += 1
            self.ambient_reads += 1
            self.consecutive_social_turns = 0
            observed_speakers = info.pop('observed_speakers', ())
            for oid, body_b in observed_speakers:
                # Only update live GazeObjectPointer fingerprints for actual in-room objects;
                # offline multi-speaker corpora must NEVER bloat or pollute the room gaze roster.
                if oid in self.presence_roster:
                    self._observe_gaze_object(oid, body_b[:256], fuse_body_state=False)
            raw_info = dict(
                info,
                channel='observational_dialogue',
                source='observational_dialogue',
                mother_bytes=0,
                mixed_world_event='observational_dialogue',
            )
            return resp, cont, self._finalize_membrane_event(
                raw_info,
                speaker_id=str(info.get('speaker_id') or 'multi'),
                addressee_id=str(info.get('addressee_id') or 'observer'),
                is_action_consequence=False,
                source_transitions=list(info.get('source_transitions') or []),
            )
        prev_src = self.last_book_source
        if self._use_sidecar_now():
            resp, cont, info = self.sidecar.take_ambient(response_bytes, holdout_bytes)
            self.sidecar_reads += 1
            self.last_book_source = 'chinese_storybook'
        else:
            resp, cont, info = self.base.take(b'', response_bytes, holdout_bytes)
            self.last_book_source = 'text'
        if self.last_book_source != prev_src:
            self.shared_page_bytes.clear()
        self.ambient_reads += 1
        self.consecutive_social_turns = 0
        self._record_shared_page(resp)
        self._record_shared_page(cont)
        self._observe_gaze_object('book', bytes(resp[:256]), fuse_body_state=False)
        raw_info = dict(
            info,
            channel='storybook_ambient',
            source=info.get('source', 'text'),
            mother_bytes=0,
            mixed_world_event='ambient_read',
        )
        return resp, cont, self._finalize_membrane_event(
            raw_info,
            speaker_id='book',
            addressee_id='mengli',
            is_action_consequence=False,
            source_transitions=[dict(offset=0, speaker_id='book', addressee_id='mengli', mask_start=0, mask_end=0)],
            record_ledger=False,
        )

    def _take_passive_ambient(self, response_bytes: int, holdout_bytes: int):
        """Advance passive background storybook WITHOUT navigating on a social/tool utterance."""
        if self.sidecar is not None and self.last_book_source == 'chinese_storybook':
            return self.sidecar.take_ambient(response_bytes, holdout_bytes)
        return self.base.take(b'', response_bytes, holdout_bytes)

    def _ensure_base_hybrid_index(self):
        """Lazily attach a prebuilt HybridBookIndex for the English natural-text book if present."""
        if self._base_hybrid_checked:
            return self._base_hybrid_index
        self._base_hybrid_checked = True
        inner = getattr(self.base, 'base', self.base)
        data = getattr(inner, 'data', None)
        if not isinstance(data, np.ndarray) or len(data) < 65536 or not hasattr(inner, 'seen'):
            return None
        manifest = getattr(self.base, 'source_manifest', None)
        train_path = None
        if isinstance(manifest, dict) and manifest.get('train_path'):
            train_path = Path(manifest['train_path'])
        else:
            cand = Path(__file__).resolve().parents[2] / 'data' / 'cgb_natural_text_20261115' / 'train.bin'
            if cand.exists():
                train_path = cand
        if train_path is None or not train_path.exists():
            return None
        idx_prefix = train_path.parent / 'hybrid_gemma2_index'
        if not (idx_prefix.with_suffix('.npz').exists() and idx_prefix.with_suffix('.json').exists()):
            return None
        try:
            raw_train = train_path.read_bytes()
            if len(data) >= len(raw_train):
                tail = data[-len(raw_train):].tobytes()
                if hashlib.sha256(tail).hexdigest() == hashlib.sha256(raw_train).hexdigest():
                    from scripts.ib_local.book_semantic_index import HybridBookIndex
                    self._base_hybrid_index = HybridBookIndex.load(idx_prefix, raw_train)
                    self._base_hybrid_offset = int(len(data) - len(raw_train))
        except Exception:
            self._base_hybrid_index = None
        return self._base_hybrid_index

    def _navigate_base_hybrid(self, query_clean: str, response_bytes: int, holdout_bytes: int):
        """Navigate the English book via Hybrid EmbeddingGemma-2 + e-means + IDF with noise gating."""
        idx = self._ensure_base_hybrid_index()
        if idx is None:
            return None
        inner = getattr(self.base, 'base', self.base)
        req = int(response_bytes) + int(holdout_bytes)
        cur_rel = max(0, int(getattr(inner, 'cursor', 0)) - self._base_hybrid_offset)
        hit = idx.search(query_clean, cursor=cur_rel)
        if hit is None:
            # Below relevance threshold (garbled/1-char noise) -> calm sequential read, zero random jump.
            resp, cont, info = self.base.take(b'', response_bytes, holdout_bytes)
            info = dict(info, match=0, retrieval_method='hybrid_gated_continue')
            return resp, cont, info
        start = self._base_hybrid_offset + int(hit['start'])
        n = int(len(inner.data))
        if start + req > n:
            start = max(self._base_hybrid_offset, n - req)
        # Keep exact base-world cursor/seen/anchor bookkeeping intact.
        end = start + req
        prior_resp = int(inner.seen[start:start + response_bytes].sum())
        prior_hold = int(inner.seen[start + response_bytes:end].sum())
        inner.seen[start:end] = True
        inner.anchor = int(start)
        inner.offset = req % max(1, int(getattr(inner, 'window', req)))
        inner.cursor = int(end)
        if hasattr(inner, 'moves'):
            inner.moves = int(inner.moves) + 1
        resp = inner.data[start:start + response_bytes].copy()
        cont = inner.data[start + response_bytes:end].copy()
        info = dict(
            start=int(start),
            anchor=int(start),
            match=1,
            matched_term=str(hit.get('matched_term') or ''),
            retrieval_method=str(hit.get('method') or 'hybrid_gemma2_emeans_idf'),
            retrieval_score=float(hit.get('score', 0.0)),
            retrieval_semantic_cos=float(hit.get('semantic_cos', 0.0)),
            response_previously_seen_bytes=prior_resp,
            holdout_previously_seen_bytes=prior_hold,
            holdout_position_fresh=(prior_hold == 0),
            navigation_response_bytes=int(len(resp)),
            source='matched',
        )
        return resp, cont, info

    def _navigate_book(self, query_bytes: bytes, query_clean: str, response_bytes: int, holdout_bytes: int):
        """Navigate the active storybook (Chinese TinyStories or English FineWeb) strictly as `Hom(MengLi, Book)`."""
        prev_src = self.last_book_source
        if self.sidecar is not None:
            has_cjk = _has_cjk(query_clean)
            en_chars = sum(1 for ch in query_clean if 'a' <= ch.lower() <= 'z')
            if has_cjk or (self.last_book_source == 'chinese_storybook' and en_chars < 4):
                resp, cont, info = self.sidecar.take(query_bytes, response_bytes, holdout_bytes)
                self.last_book_source = 'chinese_storybook'
                if self.last_book_source != prev_src:
                    self.shared_page_bytes.clear()
                return resp, cont, info
        hybrid_res = self._navigate_base_hybrid(query_clean, response_bytes, holdout_bytes)
        if hybrid_res is not None:
            resp, cont, info = hybrid_res
        else:
            resp, cont, info = self.base.take(query_bytes, response_bytes, holdout_bytes)
        self.last_book_source = 'text'
        if self.last_book_source != prev_src:
            self.shared_page_bytes.clear()
        return resp, cont, info

    def _match_registered_tool(self, child_text: str) -> tuple[str, str] | None:
        if not self._registered_tools or not child_text:
            return None
        m = re.search(r'\[tool:([a-zA-Z0-9_-]+)(?:\s+([^\]]+))?\]', child_text)
        if not m:
            return None
        name = m.group(1).lower()
        fn = self._registered_tools.get(name)
        if fn is None:
            return None
        arg = (m.group(2) or '').strip()
        try:
            out = str(fn(arg)).strip()
        except Exception as exc:
            out = f'error: {exc}'
        return name, out or 'ok'

    def _pack_prefix_bytes(self, text: str, ambient: np.ndarray, response_bytes: int,
                           *, source_label: str | None = None) -> tuple[np.ndarray, int, int, list[tuple[int, int]]]:
        """Pack speaker/tool identity prefix + utterance + book-resume marker + ambient tail.

        Returns `(response, said_len, prefix_mask_len, mask_spans)` where `mask_spans` lists
        every `[start, end)` identity-card / transition-header span (`[在场] ...\\n` + `妈妈：`/`爸爸：`/
        `[工具:...] ` at turn start, PLUS `书页：` right where the book resumes after a speaker/tool
        finishes). All spans enter GDN hidden state in full while being masked out of BOTH neural LM
        loss and Fast-Table writes.
        """
        body = text.strip()
        prefix_text = ''
        if self._roster_dirty_notice:
            prefix_text += self._roster_dirty_notice + '\n'
            self._roster_dirty_notice = None
        if self.envelope_tags and source_label:
            if source_label in ('妈妈', '爸爸'):
                prefix_text += f'{source_label}：'
            elif source_label != '书页':
                prefix_text += f'[{source_label}] '
        full = prefix_text + body + '\n'
        prefix_mask_len = min(len(prefix_text.encode('utf-8')), response_bytes)
        said = np.frombuffer(full.encode('utf-8')[:response_bytes], dtype=np.uint8)
        said_len = int(len(said))
        mask_spans: list[tuple[int, int]] = []
        if prefix_mask_len > 0:
            mask_spans.append((0, prefix_mask_len))
        rem = response_bytes - said_len
        if rem > 0:
            book_tag = np.frombuffer('书页：'.encode('utf-8'), dtype=np.uint8) if self.envelope_tags else np.empty((0,), dtype=np.uint8)
            if len(book_tag) > 0 and rem > len(book_tag) + 4:
                tail = np.concatenate((book_tag, ambient[:rem - len(book_tag)]))
                mask_spans.append((said_len, said_len + int(len(book_tag))))
            else:
                tail = ambient[:rem]
        else:
            tail = np.empty((0,), dtype=np.uint8)
        response = np.concatenate((said, tail))
        return response, said_len, int(prefix_mask_len), mask_spans

    def _current_stumble_ratio(self) -> float:
        alif_st = getattr(self._engine_ref, 'alif_executive_state', None) if self._engine_ref is not None else None
        if not isinstance(alif_st, dict):
            return 1.0
        burst_nll = float(alif_st.get('last_burst_nll', 0.0) or 0.0)
        nll_ema = float(alif_st.get('nll_ema', 0.0) or 0.0)
        if burst_nll > 0.0 and nll_ema > 0.0:
            return burst_nll / max(nll_ema, 1e-4)
        return 1.0

    def _mother_trigger_kind(
        self,
        child_clean: str,
        explicit_target: str | None = None,
        vocal_mother_call: bool = False,
    ) -> tuple[str, str] | None:
        """Strict Yoneda three-party routing: return `(trigger_kind, routing_reason)` or `None`.

        Morphism separation between `Hom(MengLi, Book)` and `Hom(MengLi, Mother)`:
          1. Explicit `@书页` / `@工具:...` / `@对象` -> NEVER Mother (`None`).
          2. Explicit `@妈妈` or natural vocal call (`妈妈` / `Mama`) -> Mother (`child_spoke` / `child_replied`).
          3. Active Social Floor (`active_floor == 'mother'` / `awaiting_reply_from == 'mother'`):
             - If MengLi speaks -> `child_replied` (`social_floor_reply`), Book does NOT jump.
             - If MengLi stays silent -> `None` (respect silence, floor returns to Book).
          4. Active Book Floor (`active_floor == 'book'`):
             - Unaddressed non-empty speech -> `None`! Routed to **Book navigation** (`Hom(MengLi, Book)`).
             - Quiet pause -> Mother gently initiates co-reading (`mother_proactive` / `co_reading_opening`)
               when `quiet_spikes_since_mother >= mother_max_gap` or on a strong stumble after `mother_cooldown`,
               opening the Social Floor for MengLi's next turn.
        """
        if self.caregiver is None or not self.mother_present:
            return None
        if explicit_target is not None and explicit_target != 'mother':
            return None
        if explicit_target == 'mother':
            trig = 'child_replied' if self.awaiting_reply_from == 'mother' else 'child_spoke'
            return trig, 'explicit_mother'
        if vocal_mother_call:
            trig = 'child_replied' if self.awaiting_reply_from == 'mother' else 'child_spoke'
            return trig, 'vocal_call_mother'
        if self._force_next_mother:
            if child_clean:
                trig = 'child_replied' if self.awaiting_reply_from == 'mother' else 'child_spoke'
                return trig, 'forced_mother'
            return 'mother_proactive', 'forced_mother'
        if self.awaiting_reply_from == 'mother' or self.active_floor == 'mother':
            if child_clean:
                return 'child_replied', 'social_floor_reply'
            return None
        if child_clean and self.respond_on_child_speech:
            return 'child_spoke', 'co_reading_child_speech'
        if not child_clean:
            if self.quiet_spikes_since_mother >= self.mother_max_gap:
                return 'mother_proactive', 'co_reading_opening'
            if (self.spikes_since_mother >= self.mother_cooldown
                    and self._current_stumble_ratio() >= 1.25):
                return 'mother_proactive', 'stumble_co_reading'
        return None

    def take(self, utterance: bytes, response_bytes: int, holdout_bytes: int, *, planned_target: dict | None = None):
        """Route one executive spike through the single UTF-8 Membrane port.

        Implements strict Yoneda morphism routing over `{书页, 妈妈, 爸爸, 工具, 动态对象}`:
          1. Live human parent / guest preemption (inbox/queue) — passive ambient, zero book jump,
          2. Explicit addressing (`@书页`, `@工具:...`, `@任意动态对象`) or tool demo,
          3. Mother (`dsh_mother`): ONLY on explicit `@妈妈` / vocal call (`妈妈`), active
             Social Floor reply (`child_replied`), or quiet co-reading opening (`mother_proactive`).
             NEVER jumps the book (`_take_passive_ambient()`), cleanly separates `[来自:妈妈]` and `[来自:书页]`.
          4. Book Floor (`book_navigation` when MengLi emits words, `storybook_spike_fallback` when quiet):
             unaddressed speech navigates the active Chinese/English storybook (`Hom(MengLi, Book)`)
             without Mother hijacking!
        """
        if response_bytes < 1 or holdout_bytes < 1:
            raise ValueError('positive response and holdout lengths required')
        if self._pending_commit is not None:
            raise RuntimeError('previous unified caregiver turn has not been committed')

        self._poll_inbox()
        self.spike_turns += 1
        utterance_bytes = bytes(utterance) if utterance else b''
        child_clean = clean_utf8_excerpt(utterance_bytes, max_chars=180)
        child_vocal = (classify_utterance(utterance_bytes)['kind'] != 'silent')
        child_display = child_clean or ('[咿呀声]' if child_vocal else '')
        cycle_num = int(self._cycle_hint or self.spike_turns)

        # Priority 1: Live human parent or guest input (immediate preemption, ZERO book jump).
        if self._participant_queue:
            msg = self._participant_queue.popleft()
            speaker = msg['speaker']
            role = msg['role']
            raw_text = msg['text']
            delivered_text = f'[{speaker}] {raw_text}'
            if 'parent' not in self.presence_roster:
                self.presence_roster['parent'] = dict(kind='parent', label=speaker, description='爸爸实时陪伴与对话')
                self._roster_dirty_notice = self._format_roster_notice()
            ambient, heldout, info = self._take_passive_ambient(response_bytes, holdout_bytes)
            response, said_len, env_prefix_len, env_mask_spans = self._pack_prefix_bytes(
                raw_text, ambient, response_bytes, source_label=speaker,
            )
            self._record_shared_page(heldout)
            self._observe_gaze_object('parent', f'{speaker}：{raw_text}'.encode('utf-8'))
            self.active_interlocutor = 'parent'
            self.awaiting_reply_from = 'parent'
            self.active_floor = 'parent'

            if self.caregiver is not None and hasattr(self.caregiver, 'begin_turn'):
                plan = self.care_plan_tracker.current_plan()
                child_entry = child_clean or '[本次未开口]'
                turn = self.mother_transcript[-2 * self.history:] + [('child', child_entry)]
                self.caregiver.begin_turn(
                    caregiver='parent',
                    text=delivered_text,
                    plan=plan,
                    cycle=cycle_num,
                    response_bytes=response_bytes,
                )
                reply = self.caregiver.reply(turn)
                self._pending_commit = dict(
                    kind='parent',
                    turn=turn,
                    reply=reply,
                    utterance_kind=None,
                )
                if self.auto_commit:
                    self.commit_pending_turn()
            self.parent_turns += 1
            part_trans = self._build_pack_transitions(
                'parent', 'mengli', said_len, env_prefix_len, env_mask_spans, response_bytes + holdout_bytes,
            )
            return response, heldout, self._finalize_membrane_event(
                dict(
                    {k: v for k, v in info.items() if k != 'text_reply'},
                    channel='live_participant',
                    source='parent',
                    speaker=speaker,
                    target_object_id='parent',
                    participant_role=role,
                    routing_reason='live_participant',
                    active_floor=self.active_floor,
                    book_excerpt=clean_utf8_excerpt(bytes(self.shared_page_bytes), max_chars=220),
                    mother_bytes=said_len,
                    envelope_prefix_bytes=env_prefix_len,
                    envelope_mask_spans=env_mask_spans,
                    mother_reply=delivered_text,
                    child_text=child_clean,
                    ambient_match=int(info.get('match', 0)),
                    match=len(_terms(child_clean) & _terms(raw_text)) if child_clean else 0,
                ),
                speaker_id='parent',
                addressee_id='mengli',
                is_action_consequence=False,
                source_transitions=part_trans,
            )

        # Parse optional explicit Yoneda target (`@书页`, `@妈妈`, `@工具:calc`, `@dynamic_obj`).
        parsed = self._parse_explicit_target(child_clean)
        explicit_target = parsed[0] if parsed is not None else None
        explicit_payload = parsed[1] if parsed is not None else ''
        if explicit_target is not None:
            self.explicit_addressed_turns += 1

        # Pre-linguistic Biological Dual-Channel Pointer (`GazeObjectPointer`):
        # Orthogonal `(u, out, in)` factorization over `(O ∪ {'none'})²`:
        #   - `send_target` (`out`): who receives MengLi's utterance `u` (`'none'` = pure self-talk).
        #   - `observe_target` (`in`): who MengLi pulls sensory observation from (`'none'` = self-evolution).
        gaze_decision = None
        gaze_target = None
        send_target: str | None = None
        observe_target: str | None = None
        interaction_mode: str | None = None
        gp = self._gaze_pointer()
        valid_dual_ids = set(self.presence_roster.keys()) | {'none'}
        if (
            explicit_target is None
            and not self._tool_demo_queue
            and not self._force_next_mother
            and self._match_registered_tool(child_clean) is None
        ):
            if isinstance(planned_target, dict):
                p_send = planned_target.get('send_target', planned_target.get('target_id'))
                p_obs = planned_target.get('observe_target', planned_target.get('target_id'))
                if p_send in valid_dual_ids and p_obs in valid_dual_ids:
                    gaze_decision = planned_target
                    send_target = str(p_send)
                    observe_target = str(p_obs)
                    gaze_target = str(planned_target.get('target_id') or (observe_target if observe_target != 'none' else send_target))
            elif gp is not None:
                alif_st = getattr(self._engine_ref, 'alif_executive_state', None) or {}
                nll_ema = float(alif_st.get('nll_ema', 0.0) or 0.0)
                nll_last = float(alif_st.get('last_piece_nll', 0.0) or 0.0)
                nll_norm = (nll_last - nll_ema) if (nll_ema > 0.0 and nll_last > 0.0) else 0.0
                gaze_decision = gp.select_interaction(
                    self._engine_ref,
                    candidates=self._gaze_candidates(child_vocal=child_vocal),
                    utterance_bytes=utterance_bytes,
                    stumble_ratio=self._current_stumble_ratio(),
                    nll_norm=nll_norm,
                    last_speaker=self.active_interlocutor,
                    awaiting_from=self.awaiting_reply_from,
                )
                send_target = str(gaze_decision['send_target'])
                observe_target = str(gaze_decision['observe_target'])
                gaze_target = str(gaze_decision['target_id'])

        if send_target is not None and observe_target is not None:
            from scripts.ib_local.gaze_pointer import classify_interaction_mode
            interaction_mode = classify_interaction_mode(child_vocal, send_target, observe_target)

        # Priority 2.0: Pure self-evolution / inner monologue (`in == 'none'`).
        # Zero Mother API call, zero storybook cursor advance; body runs self-hearing + BFSS + hippocampal replay.
        if observe_target == 'none' and explicit_target is None:
            if send_target == 'mother' and child_clean:
                self._pending_tool_notes.append(f'孩子刚刚自顾自朝你这边念了一句：「{child_clean[:96]}」')
                del self._pending_tool_notes[:-3]
            elif send_target == 'parent' and child_clean:
                p_label = str((self.presence_roster.get('parent') or {}).get('label') or '爸爸')
                self._pending_tool_notes.append(f'孩子刚刚自顾自朝{p_label}那边念了一句：「{child_clean[:96]}」')
                del self._pending_tool_notes[:-3]
            self.self_evolution_turns += 1
            if child_vocal:
                self.inner_monologue_turns += 1
            self.spikes_since_mother += 1
            self.awaiting_reply_from = None
            self.consecutive_social_turns = 0
            empty_buf = np.empty(0, dtype=np.uint8)
            return empty_buf, empty_buf.copy(), self._finalize_membrane_event(
                dict(
                    channel=('inner_monologue' if child_vocal else 'self_evolution'),
                    source='self_evolution',
                    speaker='mengli',
                    target_object_id=send_target,
                    send_target_id=send_target,
                    observe_target_id='none',
                    interaction_mode=interaction_mode,
                    self_evolution_beat=True,
                    navigation_response_bytes=0,
                    envelope_prefix_bytes=0,
                    gaze_decision=gaze_decision,
                    routing_reason=(f'dual_{interaction_mode}' if interaction_mode else 'dual_self_evolution'),
                    active_floor=self.active_floor,
                    mother_bytes=0,
                    mother_reply='',
                    child_text=child_display,
                    holdout_position_fresh=True,
                    match=0,
                ),
                speaker_id='mengli',
                addressee_id=str(send_target),
                is_action_consequence=True,
                source_transitions=[],
            )

        # Priority 2A: Explicit book navigation (`@书页 ...`) — MengLi chooses to read/search the book.
        if explicit_target == 'book':
            query_bytes = explicit_payload.encode('utf-8') if explicit_payload else utterance_bytes
            query_clean = explicit_payload or child_clean
            resp, cont, info = self._navigate_book(query_bytes, query_clean, response_bytes, holdout_bytes)
            self._record_shared_page(resp)
            self._record_shared_page(cont)
            self._observe_gaze_object('book', bytes(resp[:256]))
            self.active_floor = 'book'
            self.awaiting_reply_from = None
            self.spikes_since_mother += 1
            self.book_navigation_turns += 1
            self.ambient_fallback_turns += 1
            env_prefix_len = 0
            if self.envelope_tags and self._roster_dirty_notice:
                tag = np.frombuffer((self._roster_dirty_notice + '\n').encode('utf-8'), dtype=np.uint8)
                self._roster_dirty_notice = None
                env_prefix_len = min(int(len(tag)), response_bytes)
                resp = np.concatenate((tag, resp[:max(0, response_bytes - len(tag))]))
            return resp, cont, self._finalize_membrane_event(
                dict(
                    info,
                    channel='book_addressed',
                    source=info.get('source', 'text'),
                    speaker='book',
                    target_object_id='book',
                    navigation_response_bytes=int(len(resp)),
                    envelope_prefix_bytes=env_prefix_len,
                    explicit_target='book',
                    routing_reason='explicit_book',
                    active_floor=self.active_floor,
                    mother_bytes=0,
                    child_text=child_clean,
                ),
                speaker_id='book',
                addressee_id='mengli',
                is_action_consequence=True,
                source_transitions=[dict(
                    offset=0, speaker_id='book', addressee_id='mengli',
                    mask_start=0, mask_end=env_prefix_len,
                )],
            )

        # Priority 2B: Tool demonstration, `@工具:name`, legacy `[tool:name arg]`, dynamic object, OR Gaze -> tool/object.
        tool_hit = None
        tool_obj_id = None
        dual_tool_target = None
        if send_target is not None and (send_target.startswith('tool:') or send_target in self._dynamic_objects):
            dual_tool_target = send_target
        elif observe_target is not None and (observe_target.startswith('tool:') or observe_target in self._dynamic_objects):
            dual_tool_target = observe_target
        eff_tool_target = explicit_target if explicit_target is not None else (dual_tool_target or gaze_target)
        eff_tool_payload = explicit_payload if explicit_target is not None else child_clean
        tool_from_demo = False
        if self._tool_demo_queue:
            demo = self._tool_demo_queue.popleft()
            tool_hit = ('tool', demo['tool_name'], demo['output'])
            tool_obj_id = f'tool:{demo["tool_name"]}'
            tool_from_demo = True
        elif eff_tool_target is not None and eff_tool_target.startswith('tool:'):
            tname = eff_tool_target.split(':', 1)[1]
            fn = self._registered_tools.get(tname)
            if fn is not None:
                try:
                    tout = str(fn(eff_tool_payload)).strip() or 'ok'
                except Exception as exc:
                    tout = f'error: {exc}'
                tool_hit = ('tool', tname, tout)
                tool_obj_id = eff_tool_target
        elif eff_tool_target is not None and eff_tool_target in self._dynamic_objects:
            handler = self._dynamic_objects[eff_tool_target]
            label = self.presence_roster.get(eff_tool_target, {}).get('label', eff_tool_target)
            try:
                dout = str(handler(eff_tool_payload)).strip() or 'ok'
            except Exception as exc:
                dout = f'error: {exc}'
            tool_hit = ('object', label, dout)
            tool_obj_id = eff_tool_target
        else:
            legacy_tool = self._match_registered_tool(child_clean)
            if legacy_tool is not None:
                tool_hit = ('tool', legacy_tool[0], legacy_tool[1])
                tool_obj_id = f'tool:{legacy_tool[0]}'

        if tool_hit is not None:
            obj_kind, obj_name, obj_out = tool_hit
            src_label = f'工具:{obj_name}' if obj_kind == 'tool' else obj_name
            delivered_text = f'[{src_label}] {obj_out}'
            ambient, heldout, info = self._take_passive_ambient(response_bytes, holdout_bytes)
            response, said_len, env_prefix_len, env_mask_spans = self._pack_prefix_bytes(
                obj_out, ambient, response_bytes, source_label=src_label,
            )
            self._record_shared_page(heldout)
            if tool_obj_id is not None:
                self._observe_gaze_object(tool_obj_id, delivered_text.encode('utf-8'))
            self._pending_tool_notes.append(f'刚刚与「{src_label}」交互，返回：{obj_out[:96]}')
            del self._pending_tool_notes[:-3]
            self.tool_turns += 1
            self.spikes_since_mother += 1
            self.active_floor = 'book'
            self.awaiting_reply_from = None
            via_gaze = (explicit_target is None and gaze_target == tool_obj_id)
            t_id = str(tool_obj_id or src_label)
            tool_trans = self._build_pack_transitions(
                t_id, 'mengli', said_len, env_prefix_len, env_mask_spans, response_bytes + holdout_bytes,
            )
            return response, heldout, self._finalize_membrane_event(
                dict(
                    {k: v for k, v in info.items() if k != 'text_reply'},
                    channel=('tool' if obj_kind == 'tool' else 'dynamic_object'),
                    source=('tool' if obj_kind == 'tool' else 'object'),
                    speaker=src_label,
                    target_object_id=t_id,
                    tool_name=obj_name,
                    explicit_target=explicit_target,
                    gaze_decision=gaze_decision,
                    routing_reason=(('gaze_tool' if obj_kind == 'tool' else 'gaze_object') if via_gaze else ('explicit_tool' if obj_kind == 'tool' else 'explicit_object')),
                    active_floor=self.active_floor,
                    mother_bytes=said_len,
                    envelope_prefix_bytes=env_prefix_len,
                    envelope_mask_spans=env_mask_spans,
                    mother_reply=delivered_text,
                    child_text=child_clean,
                    ambient_match=int(info.get('match', 0)),
                    match=len(_terms(child_clean) & _terms(obj_out)) if child_clean else 0,
                ),
                speaker_id=t_id,
                addressee_id='mengli',
                is_action_consequence=(not tool_from_demo),
                source_transitions=tool_trans,
            )

        # Priority 2C: Parent/Guest Social Floor reply, explicit/vocal call to Dad, OR Gaze -> `parent`.
        # Keeps Dad's turn clean (ZERO hidden book jump!) and notifies Mother what MengLi said to Dad.
        vocal_parent_call = (explicit_target is None and _is_vocal_parent_call(child_clean))
        gaze_to_parent = (send_target == 'parent' or observe_target == 'parent' or gaze_target == 'parent')
        on_parent_floor = (
            self.active_floor == 'parent'
            and self.awaiting_reply_from not in (None, 'mother')
            and explicit_target is None
            and gaze_target is None
            and not _is_vocal_mother_call(child_clean)
        )
        if explicit_target == 'parent' or vocal_parent_call or gaze_to_parent or (on_parent_floor and child_clean):
            parent_meta = self.presence_roster.get('parent') or {}
            parent_label = str(parent_meta.get('label') or '爸爸')
            ambient, heldout, info = self._take_passive_ambient(response_bytes, holdout_bytes)
            self._record_shared_page(heldout)
            if child_clean:
                self._pending_tool_notes.append(f'孩子刚刚对{parent_label}说：「{child_clean[:96]}」')
                del self._pending_tool_notes[:-3]
            self.active_floor = 'book'
            self.awaiting_reply_from = None
            self.spikes_since_mother += 1
            env_prefix_len = 0
            if self.envelope_tags and self._roster_dirty_notice:
                tag = np.frombuffer((self._roster_dirty_notice + '\n').encode('utf-8'), dtype=np.uint8)
                self._roster_dirty_notice = None
                env_prefix_len = min(int(len(tag)), response_bytes)
                ambient = np.concatenate((tag, ambient[:max(0, response_bytes - len(tag))]))
            ch_name = 'reply_to_parent' if (on_parent_floor or self.active_interlocutor == 'parent') else 'call_parent'
            r_reason = (
                'gaze_parent' if gaze_to_parent
                else 'social_floor_reply' if on_parent_floor
                else ('explicit_parent' if explicit_target == 'parent' else 'vocal_call_parent')
            )
            return ambient, heldout, self._finalize_membrane_event(
                dict(
                    {k: v for k, v in info.items() if k != 'text_reply'},
                    channel=ch_name,
                    source='parent_reply',
                    speaker=parent_label,
                    target_object_id='parent',
                    navigation_response_bytes=int(len(ambient)),
                    envelope_prefix_bytes=env_prefix_len,
                    explicit_target=explicit_target,
                    gaze_decision=gaze_decision,
                    routing_reason=r_reason,
                    active_floor=self.active_floor,
                    mother_bytes=0,
                    mother_reply='',
                    child_text=child_clean,
                ),
                speaker_id='book',
                addressee_id='mengli',
                is_action_consequence=True,
                source_transitions=[dict(
                    offset=0, speaker_id='book', addressee_id='mengli',
                    mask_start=0, mask_end=env_prefix_len,
                )],
            )

        # Priority 3: Mother routing — via MengLi's orthogonal dual-channel `(out, in)` pointer
        # (`observe_target == 'mother'`, or `send_target == 'mother'` while observing `'book'`),
        # or fallback presence/floor rules when dual pointer is disabled.
        if send_target is not None and observe_target is not None:
            mother_engaged = (
                self.caregiver is not None
                and self.mother_present
                and (
                    observe_target == 'mother'
                    or (send_target == 'mother' and observe_target == 'book')
                )
            )
            if mother_engaged:
                if child_vocal and send_target == 'mother':
                    trig = 'child_replied' if self.awaiting_reply_from == 'mother' else 'child_spoke'
                    trigger_pair = (trig, 'gaze_mother_speak')
                elif child_vocal:
                    trigger_pair = ('child_spoke', f'dual_{interaction_mode or "monologue_watching_mother"}')
                else:
                    trigger_pair = ('mother_proactive', 'gaze_mother_look')
            else:
                trigger_pair = None
        else:
            vocal_call = (explicit_target is None and _is_vocal_mother_call(child_clean))
            trigger_pair = self._mother_trigger_kind(
                child_clean,
                explicit_target=explicit_target,
                vocal_mother_call=vocal_call,
            )
        if trigger_pair is not None:
            trigger_kind, routing_reason = trigger_pair
            self._force_next_mother = False
            perspective_mode = self._next_perspective_mode()
            # Determine honestly whether Mother and MengLi are currently looking at the book together:
            #   - `(spoke/silent, mother/none, mother)` -> pure face-to-face chat (`include_book_excerpt=False`)!
            #   - `(spoke/silent, mother, book)` or legacy mode -> shared co-reading (`include_book_excerpt=True`).
            include_book_excerpt = (
                observe_target != 'mother'
                if observe_target is not None
                else True
            )
            snapshot: BodyMirrorSnapshot = capture_body_mirror(
                self._engine_ref,
                utterance_bytes=utterance_bytes,
                recent_book_bytes=(bytes(self.shared_page_bytes) if include_book_excerpt else b''),
                recent_metal_g=self.recent_metal_g,
            )
            extra_note = '；'.join(self._pending_tool_notes) if self._pending_tool_notes else None
            # Dynamic anti-cliché opening rotator (KV-cache-safe tail cue):
            # matches whether this turn is co-reading or pure face-to-face chat.
            if include_book_excerpt:
                style_rot = (
                    '直接从眼前书页里的画面、声音或动作切入，像指着绘本随口聊起来。',
                    '先用一个轻巧的生活小比方或感叹开场，把刚才的词和书页自然连起来。',
                    '像日常聊天一样轻声抛一个不强求回答的小好奇，再顺口讲一两句。',
                    '按眼前真实情况自然回应：听得懂就接着聊，没听清就如实轻轻问一句，不要替孩子硬补没说过的词。',
                )
            else:
                style_rot = (
                    '顺着孩子此刻的话或神情自然接话，像日常面对面轻声聊天。',
                    '聊一点轻松的日常小事、感受或好奇，不用硬往书页上扯。',
                    '像平时陪伴一样温柔抛一个不强求回答的小问题，或分享一句心里话。',
                    '按眼前真实情况自然回应：听得懂就接着聊，没听清就如实轻轻问一句，不要替孩子硬补没说过的词。',
                )
            cue_parts = []
            if self.mother_turns > 0 and self.turns_since_name_call < 3:
                cue_parts.append('本轮略去称呼')
            recent_stems = list(self.recent_mother_openings[-4:])
            stale = []
            for st_txt in recent_stems:
                low_s = st_txt.lower()
                if (any(k in st_txt for k in ('我听着呢', '我听到了', '我听到你', '嗯，我听'))
                        or any(k in low_s for k in ('i hear you', "i'm listening", "i'm right here", 'good try'))
                        or recent_stems.count(st_txt) >= 2):
                    if st_txt not in stale:
                        stale.append(st_txt)
            if stale:
                cue_parts.append(f'最近几轮刚用过“{"/".join(stale[:2])}”，本轮换全新的自然起句，直接说内容')
            if cue_parts or self.mother_turns > 0:
                cue_parts.append(style_rot[self.mother_turns % len(style_rot)])
                cue_parts.append('只说出口的纯口语，不写任何括号或星号舞台动作描写')
            opening_cue = '；'.join(cue_parts) if cue_parts else None
            packet = format_caregiver_turn(
                snapshot,
                perspective_mode=perspective_mode,
                trigger_kind=trigger_kind,
                extra_note=extra_note,
                opening_cue=opening_cue,
                send_target=send_target,
                observe_target=observe_target,
                include_book_excerpt=include_book_excerpt,
                allow_mother_directives=True,
                has_parent=('parent' in self.presence_roster),
            )
            plan = self.care_plan_tracker.current_plan()
            turn = self.mother_transcript[-2 * self.history:] + [('child', packet)]

            if hasattr(self.caregiver, 'begin_turn'):
                self.caregiver.begin_turn(
                    caregiver='mother',
                    plan=plan,
                    cycle=cycle_num,
                    response_bytes=response_bytes,
                )
            raw_reply = self.caregiver.reply(turn)
            if not isinstance(raw_reply, str) or not raw_reply.strip():
                raise RuntimeError('mother returned no usable reply; world not advanced')

            # Symmetric Mother directionality: parse & strip `@安静陪伴` / `@翻书:<kw>` / `@对爸爸` / `@对梦漓`
            # BEFORE a single byte reaches MengLi's UTF-8 Membrane.
            mother_out, mother_book_query, reply = parse_mother_direction(raw_reply)
            if mother_out == 'parent' and 'parent' not in self.presence_roster:
                mother_out = 'mengli'
            if mother_out in ('mengli', 'parent') and not reply:
                mother_out = 'none'

            eff_send_id = send_target or ('mother' if child_vocal else 'none')
            eff_obs_id = observe_target or 'mother'
            mother_action_caused = (
                trigger_kind in ('child_spoke', 'child_replied')
                or eff_send_id == 'mother'
                or eff_obs_id == 'mother'
            )

            # Case M1: Mother actively chooses `@安静陪伴` (`mother_out == 'none'`, 0 spoken Mother bytes).
            if mother_out == 'none':
                self._pending_commit = dict(
                    kind='mother',
                    turn=turn,
                    reply='@安静陪伴',
                    utterance_kind=snapshot.utterance_kind,
                    perspective_mode=perspective_mode,
                    stumble_excerpt=(snapshot.stumble_excerpt if perspective_mode == 'mirror_visible' else ''),
                )
                self._pending_tool_notes.clear()
                self.spikes_since_mother = 0
                self.quiet_spikes_since_mother = 0
                self.mother_turns += 1
                self.mother_quiet_turns += 1
                self.active_floor = 'book'
                self.awaiting_reply_from = None
                self.consecutive_social_turns = 0
                if perspective_mode == 'mirror_visible':
                    self.mirror_visible_turns += 1
                else:
                    self.book_only_turns += 1
                telemetry = getattr(self.caregiver, 'last_telemetry', None) or {}
                if self.auto_commit:
                    self.commit_pending_turn()
                    telemetry = getattr(self.caregiver, 'last_telemetry', None) or telemetry
                api_usage = telemetry.get('api_usage') or {}
                ambient, heldout, info = self._take_passive_ambient(response_bytes, holdout_bytes)
                self._record_shared_page(ambient)
                self._record_shared_page(heldout)
                return ambient, heldout, self._finalize_membrane_event(
                    dict(
                        {k: v for k, v in info.items() if k != 'text_reply'},
                        channel='mother_quiet_presence',
                        source=info.get('source', 'text'),
                        speaker='book',
                        target_object_id=eff_send_id,
                        send_target_id=eff_send_id,
                        observe_target_id=eff_obs_id,
                        interaction_mode=interaction_mode,
                        mother_out='none',
                        trigger_kind=trigger_kind,
                        routing_reason=f'{routing_reason}:mother_quiet',
                        gaze_decision=gaze_decision,
                        active_floor=self.active_floor,
                        explicit_target=explicit_target,
                        perspective_mode=perspective_mode,
                        care_stage=self.care_plan_tracker.stage,
                        care_plan_version=int(plan['version']),
                        utterance_kind=snapshot.utterance_kind,
                        book_excerpt=snapshot.book_excerpt,
                        mirror_excerpt=snapshot.stumble_excerpt if perspective_mode == 'mirror_visible' else '',
                        mirror_trigger_cause=snapshot.trigger_cause if perspective_mode == 'mirror_visible' else '',
                        mother_bytes=0,
                        envelope_prefix_bytes=0,
                        mother_reply='',
                        child_text=child_display,
                        caregiver_prompt_tokens=api_usage.get('prompt_tokens'),
                        caregiver_cache_hit_tokens=api_usage.get('cache_hit_tokens'),
                        caregiver_cache_miss_tokens=api_usage.get('cache_miss_tokens'),
                        caregiver_source=telemetry.get('source'),
                        ambient_match=int(info.get('match', 0)),
                        match=0,
                    ),
                    speaker_id='book',
                    addressee_id='mengli',
                    is_action_consequence=False,
                    source_transitions=[dict(
                        offset=0, speaker_id='book', addressee_id='mengli', mask_start=0, mask_end=0,
                    )],
                )

            # Track name-opening & cliché-opening cadence when Mother speaks aloud.
            if reply:
                if _NAME_OPENING_RE.search(reply[:14]):
                    self.turns_since_name_call = 0
                elif ('梦漓' in reply) or ('mengli' in reply.lower()):
                    self.turns_since_name_call = min(self.turns_since_name_call, 1)
                else:
                    self.turns_since_name_call += 1
                first_clause = re.split(r'[。！？!?\n—–-]', reply, maxsplit=1)[0].strip()[:14]
                if first_clause:
                    self.recent_mother_openings.append(first_clause)
                    del self.recent_mother_openings[:-6]

            # Case M2: Mother actively flips the storybook (`@翻书:<关键词>` -> HybridBookIndex!).
            if mother_out == 'book' and mother_book_query:
                ambient, heldout, info = self._navigate_book(
                    mother_book_query.encode('utf-8'), mother_book_query, response_bytes, holdout_bytes,
                )
                self.mother_book_turns += 1
                self.book_navigation_turns += 1
            else:
                # Talking to Mother NEVER triggers a hidden book jump unless Mother explicitly chose `@翻书:...`.
                ambient, heldout, info = self._take_passive_ambient(response_bytes, holdout_bytes)

            if reply:
                response, said_len, env_prefix_len, env_mask_spans = self._pack_prefix_bytes(
                    reply, ambient, response_bytes, source_label='妈妈',
                )
                self._observe_gaze_object('mother', f'妈妈：{reply}'.encode('utf-8'))
            else:
                # Pure `@翻书:<kw>` without spoken text: deliver newly turned page cleanly!
                response, said_len, env_prefix_len, env_mask_spans = ambient, 0, 0, []
            self._record_shared_page(heldout)
            if mother_out == 'book':
                self._record_shared_page(ambient)
                self._observe_gaze_object('book', bytes(ambient[:256]))

            transcript_reply = (
                f'@翻书:{mother_book_query} {reply}'.strip() if mother_out == 'book'
                else (f'@对爸爸 {reply}'.strip() if mother_out == 'parent' else reply)
            )
            self._pending_commit = dict(
                kind='mother',
                turn=turn,
                reply=transcript_reply,
                utterance_kind=snapshot.utterance_kind,
                perspective_mode=perspective_mode,
                stumble_excerpt=(snapshot.stumble_excerpt if perspective_mode == 'mirror_visible' else ''),
            )
            self._pending_tool_notes.clear()
            self.spikes_since_mother = 0
            self.quiet_spikes_since_mother = 0
            self.mother_turns += 1
            self.active_interlocutor = 'mother'
            self.consecutive_social_turns += 1
            allow_immediate_followup = (self.consecutive_social_turns < self.max_consecutive_social_turns)
            asked_question = bool(reply and reply.endswith(('?', '？')))
            if trigger_kind == 'mother_proactive':
                self.mother_proactive_turns += 1
                keep_open = allow_immediate_followup and (asked_question or self._gaze_pointer() is None)
                self.active_floor = 'mother' if keep_open else 'book'
                self.awaiting_reply_from = 'mother' if keep_open else None
            else:
                self.mother_responsive_turns += 1
                if asked_question and allow_immediate_followup:
                    self.active_floor = 'mother'
                    self.awaiting_reply_from = 'mother'
                else:
                    self.active_floor = 'book'
                    self.awaiting_reply_from = None
            if perspective_mode == 'mirror_visible':
                self.mirror_visible_turns += 1
            else:
                self.book_only_turns += 1

            telemetry = getattr(self.caregiver, 'last_telemetry', None) or {}
            if self.auto_commit:
                self.commit_pending_turn()
                telemetry = getattr(self.caregiver, 'last_telemetry', None) or telemetry

            api_usage = telemetry.get('api_usage') or {}
            event_addressee = 'parent' if mother_out == 'parent' else 'mengli'
            event_speaker = 'mother' if reply else 'book'
            if reply:
                mother_trans = self._build_pack_transitions(
                    'mother', event_addressee, said_len, env_prefix_len, env_mask_spans, response_bytes + holdout_bytes,
                )
            else:
                mother_trans = [dict(
                    offset=0, speaker_id='book', addressee_id='mengli', mask_start=0, mask_end=0,
                )]
            if mother_out == 'parent':
                mother_action_caused = False
            return response, heldout, self._finalize_membrane_event(
                dict(
                    {k: v for k, v in info.items() if k != 'text_reply'},
                    channel='dsh_mother',
                    source='mother',
                    speaker=event_speaker,
                    target_object_id='mother',
                    send_target_id=eff_send_id,
                    observe_target_id=eff_obs_id,
                    interaction_mode=interaction_mode,
                    mother_out=mother_out,
                    mother_book_query=(mother_book_query if mother_out == 'book' else ''),
                    include_book_excerpt=bool(include_book_excerpt),
                    trigger_kind=trigger_kind,
                    routing_reason=routing_reason,
                    gaze_decision=gaze_decision,
                    active_floor=self.active_floor,
                    explicit_target=explicit_target,
                    perspective_mode=perspective_mode,
                    care_stage=self.care_plan_tracker.stage,
                    care_plan_version=int(plan['version']),
                    utterance_kind=snapshot.utterance_kind,
                    book_excerpt=snapshot.book_excerpt,
                    mirror_excerpt=snapshot.stumble_excerpt if perspective_mode == 'mirror_visible' else '',
                    mirror_trigger_cause=snapshot.trigger_cause if perspective_mode == 'mirror_visible' else '',
                    mother_bytes=said_len,
                    envelope_prefix_bytes=env_prefix_len,
                    envelope_mask_spans=env_mask_spans,
                    mother_reply=reply,
                    child_text=child_display,
                    caregiver_prompt_tokens=api_usage.get('prompt_tokens'),
                    caregiver_cache_hit_tokens=api_usage.get('cache_hit_tokens'),
                    caregiver_cache_miss_tokens=api_usage.get('cache_miss_tokens'),
                    caregiver_source=telemetry.get('source'),
                    ambient_match=int(info.get('match', 0)),
                    match=len(_terms(child_clean) & _terms(reply)) if (child_clean and reply) else 0,
                ),
                speaker_id=event_speaker,
                addressee_id=event_addressee,
                is_action_consequence=mother_action_caused,
                source_transitions=mother_trans,
            )

        # Priority 4: Book Floor (`observe_target == 'book'` or legacy fallback).
        # Under orthogonal `(u, out, in)`:
        #   - `out == 'book'` + speech -> `book_search` (semantic HybridBookIndex lookup & page flip!)
        #   - `out == 'none'` + speech -> `read_aloud_solo` (self-talk while reading calmly; ZERO jump!)
        #   - silent                   -> `quiet_reading` (calm sequential reading).
        is_navigation = bool(child_clean) and (send_target is None or send_target == 'book')
        if is_navigation:
            resp, cont, info = self._navigate_book(utterance_bytes, child_clean, response_bytes, holdout_bytes)
            self.book_navigation_turns += 1
            channel = 'book_navigation'
            routing_reason = 'gaze_book_search' if (send_target == 'book' or gaze_target == 'book') else 'book_floor_navigation'
        else:
            resp, cont, info = self._take_passive_ambient(response_bytes, holdout_bytes)
            self.quiet_spikes_since_mother += 1
            is_read_aloud = bool(child_vocal and send_target == 'none')
            channel = ('read_aloud_solo' if is_read_aloud else 'storybook_spike_fallback')
            routing_reason = (
                'dual_read_aloud_solo' if is_read_aloud
                else ('gaze_book_read' if (observe_target == 'book' or gaze_target == 'book') else 'quiet_book_continue')
            )
        self._record_shared_page(resp)
        self._record_shared_page(cont)
        self._observe_gaze_object('book', bytes(resp[:256]))
        self.spikes_since_mother += 1
        self.active_floor = 'book'
        self.awaiting_reply_from = None
        self.consecutive_social_turns = 0
        self.ambient_fallback_turns += 1
        env_prefix_len = 0
        if self.envelope_tags and self._roster_dirty_notice:
            tag = np.frombuffer((self._roster_dirty_notice + '\n').encode('utf-8'), dtype=np.uint8)
            self._roster_dirty_notice = None
            env_prefix_len = min(int(len(tag)), response_bytes)
            resp = np.concatenate((tag, resp[:max(0, response_bytes - len(tag))]))
        gaze_chose_book = (
            (observe_target == 'book' or gaze_target == 'book')
            and isinstance(gaze_decision, dict)
            and bool(gaze_decision.get('sampled', False))
        )
        book_action_caused = bool(is_navigation or gaze_chose_book)
        return resp, cont, self._finalize_membrane_event(
            dict(
                info,
                channel=channel,
                source=info.get('source', 'text'),
                speaker='book',
                target_object_id='book',
                send_target_id=(send_target if send_target is not None else ('book' if is_navigation else 'none')),
                observe_target_id=(observe_target if observe_target is not None else 'book'),
                interaction_mode=interaction_mode,
                navigation_response_bytes=int(len(resp)),
                envelope_prefix_bytes=env_prefix_len,
                routing_reason=routing_reason,
                gaze_decision=gaze_decision,
                active_floor=self.active_floor,
                mother_bytes=0,
                child_text=child_clean,
            ),
            speaker_id='book',
            addressee_id='mengli',
            is_action_consequence=book_action_caused,
            source_transitions=[dict(
                offset=0, speaker_id='book', addressee_id='mengli',
                mask_start=0, mask_end=env_prefix_len,
            )],
        )

    def commit_pending_turn(
        self,
        *,
        metal_g: float | None = None,
        evidence_available: bool = True,
        boundary_context: dict | None = None,
    ) -> dict | None:
        """Commit an open DSH caregiver turn & settle GazeObjectPointer credit AFTER body learning."""
        if isinstance(metal_g, (int, float)) and math.isfinite(metal_g):
            self.recent_metal_g.append(float(metal_g))
            if len(self.recent_metal_g) > 16:
                del self.recent_metal_g[:-16]

        gaze_receipt = None
        gp = self._gaze_pointer()
        if gp is not None:
            b_ctx = boundary_context
            if b_ctx is None and self._engine_ref is not None:
                b_ctx = getattr(self._engine_ref, '_gaze_boundary_context', None)
                self._engine_ref._gaze_boundary_context = None
            eff_ev = (
                bool(b_ctx['evidence_available'])
                if isinstance(b_ctx, dict) and 'evidence_available' in b_ctx
                else bool(evidence_available)
            )
            gaze_receipt = gp.settle_credit(
                metal_g,
                evidence_available=eff_ev,
                boundary_context=b_ctx,
            )

        if self._pending_commit is None:
            return (dict(committed=None, gaze=gaze_receipt) if gaze_receipt is not None else None)
        pending = self._pending_commit
        if self.caregiver is not None and hasattr(self.caregiver, 'commit_turn'):
            self.caregiver.commit_turn()
        self.mother_transcript = (pending['turn'] + [('mother', pending['reply'])])[-2 * self.history:]
        stage_changed = False
        if pending['kind'] == 'mother':
            self.turns_since_reflection += 1
            if pending.get('stumble_excerpt'):
                self.recent_stumbles.append(str(pending['stumble_excerpt'])[:72])
                del self.recent_stumbles[:-8]
            if isinstance(metal_g, (int, float)) and math.isfinite(metal_g):
                if pending.get('perspective_mode') == 'mirror_visible':
                    self.mirror_g.append(float(metal_g))
                    del self.mirror_g[:-16]
                elif pending.get('perspective_mode') == 'book_only':
                    self.book_only_g.append(float(metal_g))
                    del self.book_only_g[:-16]
            stage_changed = self.care_plan_tracker.observe_turn(
                utterance_kind=pending['utterance_kind'] or 'silent',
                metal_g=metal_g,
                turn_index=self.mother_turns,
            )
        self._pending_commit = None
        return dict(committed=pending['kind'], stage_changed=stage_changed,
                    stage=self.care_plan_tracker.stage,
                    plan_version=self.care_plan_tracker.version)

    def maybe_reflect_and_adapt(
        self,
        *,
        reflect_every_turns: int = 16,
        cycle: int = 0,
        recent_flux: dict | None = None,
        notes_path: str | Path | None = None,
    ) -> dict | None:
        """Periodic Mother Agent reflection using full Five-Element + Causal Bank + Episode dossier."""
        if (self.caregiver is None or reflect_every_turns <= 0
                or self.turns_since_reflection < int(reflect_every_turns)
                or self._pending_commit is not None):
            return None
        auto = getattr(self.caregiver, 'automatic', self.caregiver)
        if not hasattr(auto, 'reflect'):
            return None
        plan = self.care_plan_tracker.current_plan()
        dossier = build_five_element_reflection_dossier(
            self._engine_ref,
            cycle=int(cycle or self._cycle_hint or self.spike_turns),
            mother_turns=int(self.mother_turns),
            care_stage=self.care_plan_tracker.stage,
            plan_version=int(plan['version']),
            current_plan_text=plan['text'],
            prior_mengli_model=self.care_plan_tracker.mengli_model_summary,
            recent_kinds=self.care_plan_tracker.recent_kinds,
            recent_metal_g=self.recent_metal_g,
            mirror_g=self.mirror_g,
            book_only_g=self.book_only_g,
            recent_stumbles=self.recent_stumbles,
            recent_flux=recent_flux,
            presence_labels=[m['label'] for m in self.presence_roster.values()],
        )
        res = auto.reflect(dossier)
        parsed = parse_mother_reflection(res.get('raw', ''))
        plan_updated = self.care_plan_tracker.apply_mother_reflection(
            new_care_plan=(parsed['new_care_plan'] if parsed['update_plan'] else None),
            mengli_model=parsed['mengli_model'],
            turn_index=self.mother_turns,
        )
        self.turns_since_reflection = 0
        self.reflections_done += 1
        if notes_path is not None:
            p = Path(notes_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            import time as _t
            ts = _t.strftime('%Y-%m-%d %H:%M:%S')
            entry = [
                f'\n## [{ts}] 第 {cycle or self._cycle_hint} 轮 · 妈妈自主复盘 #{self.reflections_done}（累计互动 {self.mother_turns} 轮）',
            ]
            if parsed['mengli_model']:
                entry.append(f'- **梦漓模型**：{parsed["mengli_model"]}')
            if parsed['progress_note']:
                entry.append(f'- **阶段手记**：{parsed["progress_note"]}')
            if plan_updated:
                entry.append(f'- **育儿计划更新（v{self.care_plan_tracker.version}）**：{parsed["new_care_plan"]}')
            else:
                entry.append(f'- **育儿计划**：保持 v{self.care_plan_tracker.version} 不变')
            with p.open('a', encoding='utf-8') as f:
                f.write('\n'.join(entry) + '\n')
        return dict(
            reflection_index=int(self.reflections_done),
            plan_updated=bool(plan_updated),
            plan_version=int(self.care_plan_tracker.version),
            **parsed,
        )

    def telemetry_summary(self) -> dict:
        usage_totals = None
        provider_hit_rate = None
        summary_version = None
        if self.caregiver is not None:
            auto = getattr(self.caregiver, 'automatic', self.caregiver)
            usage_totals = copy.deepcopy(getattr(auto, 'usage_totals', None))
            ckpt_ref = getattr(auto, 'checkpoint_ref', None) or {}
            summary_version = ckpt_ref.get('summary_version')
            if isinstance(usage_totals, dict):
                ru = usage_totals.get('reply_usage') or {}
                hit, miss = ru.get('cache_hit_tokens'), ru.get('cache_miss_tokens')
                if isinstance(hit, (int, float)) and isinstance(miss, (int, float)) and (hit + miss) > 0:
                    provider_hit_rate = float(hit) / float(hit + miss)
        gp = self._gaze_pointer()
        return dict(
            ambient_reads=int(self.ambient_reads),
            sidecar_reads=int(self.sidecar_reads),
            spike_turns=int(self.spike_turns),
            mother_turns=int(self.mother_turns),
            mother_responsive_turns=int(self.mother_responsive_turns),
            mother_proactive_turns=int(self.mother_proactive_turns),
            book_navigation_turns=int(self.book_navigation_turns),
            explicit_addressed_turns=int(self.explicit_addressed_turns),
            mirror_visible_turns=int(self.mirror_visible_turns),
            book_only_turns=int(self.book_only_turns),
            parent_turns=int(self.parent_turns),
            tool_turns=int(self.tool_turns),
            ambient_fallback_turns=int(self.ambient_fallback_turns),
            self_evolution_turns=int(self.self_evolution_turns),
            inner_monologue_turns=int(self.inner_monologue_turns),
            mother_quiet_turns=int(self.mother_quiet_turns),
            mother_book_turns=int(self.mother_book_turns),
            mother_present=bool(self.mother_present),
            active_floor=self.active_floor,
            last_book_source=self.last_book_source,
            active_interlocutor=self.active_interlocutor,
            awaiting_reply_from=self.awaiting_reply_from,
            turns_since_name_call=int(self.turns_since_name_call),
            care_stage=self.care_plan_tracker.stage,
            care_plan_version=int(self.care_plan_tracker.version),
            stage_transitions=copy.deepcopy(self.care_plan_tracker.transitions),
            gaze_pointer=(gp.diagnostics() if gp is not None else None),
            summary_version=summary_version,
            provider_hit_rate=provider_hit_rate,
            usage_totals=usage_totals,
        )

    def state_dict(self, *, copy_arrays: bool = True) -> dict:
        if self._pending_commit is not None:
            raise RuntimeError('cannot checkpoint UnifiedWorldPort with an uncommitted caregiver turn')
        base_state = self.base.state_dict() if copy_arrays else self.base.checkpoint_state_dict()
        gp = self._gaze_pointer()
        state = dict(
            base_state,
            mother_transcript=copy.deepcopy(self.mother_transcript),
            mother_history=int(self.history),
            mother_ambient_policy=self.ambient_policy,
            mother_unified_port=dict(
                schema=1,
                mother_cooldown=int(self.mother_cooldown),
                mother_max_gap=int(self.mother_max_gap),
                mirror_ratio=float(self.mirror_ratio),
                shared_page_hex=bytes(self.shared_page_bytes).hex(),
                ambient_reads=int(self.ambient_reads),
                sidecar_reads=int(self.sidecar_reads),
                dialogue_reads=int(self.dialogue_reads),
                spike_turns=int(self.spike_turns),
                spikes_since_mother=int(self.spikes_since_mother),
                quiet_spikes_since_mother=int(self.quiet_spikes_since_mother),
                mother_turns=int(self.mother_turns),
                mother_responsive_turns=int(self.mother_responsive_turns),
                mother_proactive_turns=int(self.mother_proactive_turns),
                book_navigation_turns=int(self.book_navigation_turns),
                explicit_addressed_turns=int(self.explicit_addressed_turns),
                mother_present=bool(self.mother_present),
                active_floor=self.active_floor,
                last_book_source=self.last_book_source,
                active_interlocutor=self.active_interlocutor,
                awaiting_reply_from=self.awaiting_reply_from,
                turns_since_name_call=int(self.turns_since_name_call),
                mirror_visible_turns=int(self.mirror_visible_turns),
                book_only_turns=int(self.book_only_turns),
                parent_turns=int(self.parent_turns),
                tool_turns=int(self.tool_turns),
                ambient_fallback_turns=int(self.ambient_fallback_turns),
                self_evolution_turns=int(self.self_evolution_turns),
                inner_monologue_turns=int(self.inner_monologue_turns),
                mother_quiet_turns=int(self.mother_quiet_turns),
                mother_book_turns=int(self.mother_book_turns),
                recent_metal_g=list(self.recent_metal_g),
                inbox_offset=int(self.inbox_offset),
                sidecar_cursor=(None if self.sidecar is None else int(self.sidecar.cursor)),
                dialogue_cursor=(None if self.dialogue_demo is None else int(self.dialogue_demo.cursor)),
                care_plan_tracker=self.care_plan_tracker.state_dict(),
                pending_tool_notes=list(self._pending_tool_notes),
                event_ledger=copy.deepcopy(self.event_ledger),
                gaze_dynamic_state=(gp.export_dynamic_state() if gp is not None else None),
            ),
        )
        if self.caregiver is not None and hasattr(self.caregiver, 'state_dict'):
            state['mother_provider_state'] = self.caregiver.state_dict()
        return state

    def checkpoint_state_dict(self) -> dict:
        return self.state_dict(copy_arrays=False)

    def load_state_dict(self, state: dict) -> None:
        state = dict(state)
        state.pop('mother_history', None)
        transcript = state.pop('mother_transcript', [])
        state.pop('mother_ambient_policy', None)
        provider = state.pop('mother_provider_state', None)
        uwp = state.pop('mother_unified_port', None) or state.pop('unified_world_port', None)

        self.base.load_state_dict(state)
        self.mother_transcript = [tuple(pair) for pair in copy.deepcopy(transcript)][-2 * self.history:]
        if provider is not None and self.caregiver is not None and hasattr(self.caregiver, 'load_state_dict'):
            self.caregiver.load_state_dict(provider)

        if isinstance(uwp, dict) and uwp.get('schema') == 1:
            self.shared_page_bytes = bytearray(bytes.fromhex(uwp.get('shared_page_hex', '')))
            self.ambient_reads = int(uwp.get('ambient_reads', 0))
            self.sidecar_reads = int(uwp.get('sidecar_reads', 0))
            self.dialogue_reads = int(uwp.get('dialogue_reads', 0))
            self.spike_turns = int(uwp.get('spike_turns', 0))
            self.spikes_since_mother = int(uwp.get('spikes_since_mother', self.mother_max_gap))
            self.quiet_spikes_since_mother = int(uwp.get('quiet_spikes_since_mother', 0))
            self.mother_turns = int(uwp.get('mother_turns', 0))
            self.mother_responsive_turns = int(uwp.get('mother_responsive_turns', 0))
            self.mother_proactive_turns = int(uwp.get('mother_proactive_turns', 0))
            self.book_navigation_turns = int(uwp.get('book_navigation_turns', 0))
            self.explicit_addressed_turns = int(uwp.get('explicit_addressed_turns', 0))
            self.mother_present = bool(uwp.get('mother_present', self.caregiver is not None))
            self.active_floor = str(uwp.get('active_floor', 'book'))
            self.last_book_source = str(uwp.get('last_book_source', 'text'))
            self.active_interlocutor = uwp.get('active_interlocutor', self.active_interlocutor)
            self.awaiting_reply_from = uwp.get('awaiting_reply_from', None)
            self.turns_since_name_call = int(uwp.get('turns_since_name_call', 3))
            self.mirror_visible_turns = int(uwp.get('mirror_visible_turns', 0))
            self.book_only_turns = int(uwp.get('book_only_turns', 0))
            self.parent_turns = int(uwp.get('parent_turns', 0))
            self.tool_turns = int(uwp.get('tool_turns', 0))
            self.ambient_fallback_turns = int(uwp.get('ambient_fallback_turns', 0))
            self.self_evolution_turns = int(uwp.get('self_evolution_turns', 0))
            self.inner_monologue_turns = int(uwp.get('inner_monologue_turns', 0))
            self.mother_quiet_turns = int(uwp.get('mother_quiet_turns', 0))
            self.mother_book_turns = int(uwp.get('mother_book_turns', 0))
            self.recent_metal_g = [float(x) for x in uwp.get('recent_metal_g', [])]
            self.inbox_offset = int(uwp.get('inbox_offset', 0))
            if self.sidecar is not None and uwp.get('sidecar_cursor') is not None:
                self.sidecar.cursor = int(uwp['sidecar_cursor']) % len(self.sidecar.data)
            if self.dialogue_demo is not None and uwp.get('dialogue_cursor') is not None:
                self.dialogue_demo.cursor = int(uwp['dialogue_cursor']) % len(self.dialogue_demo.scenes)
            if isinstance(uwp.get('care_plan_tracker'), dict):
                self.care_plan_tracker.load_state_dict(uwp['care_plan_tracker'])
            self._pending_tool_notes = list(uwp.get('pending_tool_notes', []))
            raw_ledger = uwp.get('event_ledger')
            self.event_ledger = (
                copy.deepcopy(raw_ledger[-self.event_ledger_cap:])
                if isinstance(raw_ledger, list) else []
            )
            gaze_state = uwp.get('gaze_dynamic_state')
            if isinstance(gaze_state, dict):
                gp = self._gaze_pointer()
                if gp is not None:
                    gp.load_dynamic_state(gaze_state)
                else:
                    self._deferred_gaze_state = copy.deepcopy(gaze_state)
        self._pending_commit = None
