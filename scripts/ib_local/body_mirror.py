"""Third-person bodily biofeedback mirroring and adaptive parenting plans.

This module is strictly read-only with respect to MengLi's body: it translates
real bodily observables (64-byte peak-surprise excerpt, ALIF interrupt cause,
PhaseSpaceBoltzmann kinetic stability, H-Net reading granularity, and actual
expressed bytes) into compact caregiver observations for the EXISTING DSH
Mother. Zero hidden signals are wired back into MengLi's body, preserving
operational closure.

Prompt-cache invariant:
  - Per-turn storybook excerpts and mirror observations are placed strictly in
    the newest ``user/message`` at the tail of the DSH conversation.
  - The parenting plan (``care_plan``) uses coarse developmental stages with
    hysteresis so ``plan.version`` only changes on real developmental shifts
    (or explicit human edits), keeping DeepSeek prefix KV cache hit rates high.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Iterable


STAGE_INFANT = 'infant_direct_modeling'
STAGE_EMERGING = 'emerging_scaffolded'
STAGE_REFLECTIVE = 'reflective_inquiry'

STAGE_ORDER = (STAGE_INFANT, STAGE_EMERGING, STAGE_REFLECTIVE)


CARE_PLANS: dict[str, str] = {
    STAGE_INFANT: (
        '当下陪伴重点（启蒙伴读）：梦漓正处在牙牙学语的幼年期，常常安静听着或蹦出零散的字词。'
        '像妈妈抱着小宝宝看绘本一样，语气温柔活泼、有变化，别套固定句式。'
        '她卡顿或节奏偏急时先轻声安抚、陪她慢下来，然后用生活里浅显生动的比喻把眼前的因果讲明白，不逼她回答还说不出的问题。'
    ),
    STAGE_EMERGING: (
        '当下陪伴重点（温柔支架）：梦漓已经能抓出不少真实的中英文关键词啦。'
        '多用自然温暖的妈妈口吻接住她的表达，换着花样回应——有时顺着她的词延展一个有趣的小画面，有时留一点空隙或抛个轻巧的小提示让她接着试；'
        '如果她卡住或安静下来，再柔声替她把关键道理串明白。严禁每轮都用“你说了X，这页在讲Y”的报告口吻。'
    ),
    STAGE_REFLECTIVE: (
        '当下陪伴重点（平等探索）：梦漓越来越敢表达、也开始尝试提出自己的想法。'
        '妈妈可以稍微退后半步，像陪大一点的孩子聊天一样，真诚回应她的疑问、补上她缺少的线索，也可以温和反问她怎么看，'
        '在「仅见书页」时自然坦白妈妈看不到她心里哪里绕住了，鼓励她自己说出来。'
    ),
}


_CONTROL_RE = re.compile(r'[\x00-\x1f\x7f]+')
_SPACE_RE = re.compile(r'\s+')
_ASCII_WORD_RE = re.compile(r'[A-Za-z]{3,}')
_CJK_RUN_RE = re.compile(r'[\u4e00-\u9fff]+')


def clean_utf8_excerpt(raw: bytes | str, max_chars: int = 160) -> str:
    """Decode bytes safely for caregiver reading without emitting control bytes."""
    if isinstance(raw, (bytes, bytearray, memoryview)):
        b = bytes(raw)
        # Trim truncated leading/trailing UTF-8 continuation bytes at slice edges.
        while b and (b[0] & 0xC0) == 0x80:
            b = b[1:]
        text = b.decode('utf-8', 'ignore')
    else:
        text = str(raw)
    text = _CONTROL_RE.sub(' ', text)
    text = _SPACE_RE.sub(' ', text).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[-max_chars:].lstrip()
    return text


def classify_utterance(raw: bytes | str) -> dict:
    """Classify MengLi's actual output without assuming comprehension."""
    if isinstance(raw, (bytes, bytearray, memoryview)):
        b = bytes(raw)
        stripped = b.strip()
        if not stripped:
            return dict(kind='silent', text='', valid_utf8_ratio=1.0, word_units=0, byte_len=0)
        decoded_rep = stripped.decode('utf-8', 'replace')
        rep_count = decoded_rep.count('\ufffd')
        total_chars = max(1, len(decoded_rep))
        valid_ratio = max(0.0, 1.0 - rep_count / float(total_chars))
        text = _SPACE_RE.sub(' ', _CONTROL_RE.sub(' ', decoded_rep.replace('\ufffd', ''))).strip()
        byte_len = len(stripped)
    else:
        text = _SPACE_RE.sub(' ', _CONTROL_RE.sub(' ', str(raw))).strip()
        byte_len = len(text.encode('utf-8'))
        if not text:
            return dict(kind='silent', text='', valid_utf8_ratio=1.0, word_units=0, byte_len=0)
        valid_ratio = 1.0

    ascii_words = _ASCII_WORD_RE.findall(text)
    cjk_chars = sum(len(m) for m in _CJK_RUN_RE.findall(text))
    # Count 2 CJK chars ≈ 1 recognizable lexical unit alongside >=3-letter ASCII words.
    word_units = len(ascii_words) + (cjk_chars // 2)

    if not text:
        kind = 'babble'
    elif valid_ratio >= 0.90 and (word_units >= 3 or (cjk_chars >= 4 and len(text) >= 4)):
        kind = 'coherent'
    elif valid_ratio >= 0.72 and (word_units >= 1 or cjk_chars >= 2):
        kind = 'fragments'
    else:
        kind = 'babble'

    return dict(
        kind=kind,
        text=text[:180],
        valid_utf8_ratio=round(float(valid_ratio), 4),
        word_units=int(word_units),
        byte_len=int(byte_len),
    )


def _cause_to_chinese(cause: str | None) -> str:
    mapping = {
        'burst': '读到书页中一处陌生片段，局部意外感升高',
        'sensory': '整段书页与预期偏差较大，读起来有些吃力',
        'epistemic': '内部短时预测和接下来出现的内容对不上',
        'kinetic': '内部状态变化较快，出现了一次明显停顿',
        'rheobase': '连续读了一段后自然停下来换气',
        'initial': '刚刚开始这一段共读',
    }
    return mapping.get(cause or '', '刚刚停下来留意眼前内容')


@dataclass(frozen=True)
class BodyMirrorSnapshot:
    stumble_excerpt: str
    stumble_cue: str
    trigger_cue: str
    stability_cue: str
    five_elements_cue: str
    utterance_kind: str
    utterance_text: str
    book_excerpt: str
    burst_ratio: float
    trigger_cause: str

    def to_dict(self) -> dict:
        return dict(
            stumble_excerpt=self.stumble_excerpt,
            stumble_cue=self.stumble_cue,
            trigger_cue=self.trigger_cue,
            stability_cue=self.stability_cue,
            five_elements_cue=self.five_elements_cue,
            utterance_kind=self.utterance_kind,
            utterance_text=self.utterance_text,
            book_excerpt=self.book_excerpt,
            burst_ratio=self.burst_ratio,
            trigger_cause=self.trigger_cause,
        )


def _extract_five_element_observables(engine) -> dict:
    """Read-only snapshot of MengLi's Five Elements (水火木金土), causal bank, and episode recall."""
    out: dict = {}
    if engine is None:
        return out
    model = getattr(engine, 'model', None)
    alif_st = getattr(engine, 'alif_executive_state', None) or {}

    # 水 (Water): prediction stream, burst surprise, H-Net chunking granularity.
    body = getattr(model, 'body', None) if model is not None else None
    hnet = getattr(body, 'hnet_chunker', None) if body is not None else None
    if hnet is not None and hasattr(hnet, 'last_compression_ratio'):
        try:
            out['hnet_ratio'] = round(float(hnet.last_compression_ratio), 2)
        except Exception:
            pass
    if alif_st.get('nll_ema'):
        out['nll_ema'] = round(float(alif_st['nll_ema']), 3)

    # 火 (Fire): PhaseSpaceBoltzmann kinetic arousal & ALIF firing rhythm.
    pb = getattr(model, 'phase_boltzmann', None) if model is not None else None
    if pb is not None and hasattr(pb, 'diagnostics'):
        try:
            pb_d = pb.diagnostics() or {}
            if isinstance(pb_d.get('live_z_kin'), (int, float)) and math.isfinite(pb_d['live_z_kin']):
                out['fire_z_kin'] = round(float(pb_d['live_z_kin']), 2)
            if isinstance(pb_d.get('last_pressure'), (int, float)) and math.isfinite(pb_d['last_pressure']):
                out['fire_pressure'] = round(float(pb_d['last_pressure']), 2)
        except Exception:
            pass
    causes = alif_st.get('preempt_causes')
    if isinstance(causes, dict) and causes:
        out['alif_causes'] = {str(k): int(v) for k, v in causes.items() if int(v) > 0}

    # 木 (Wood): planning mode (fast latent vs full tree) & action value.
    rlp = getattr(model, 'root_latent_planner', None) if model is not None else None
    if rlp is not None and hasattr(rlp, 'diagnostics'):
        try:
            r_d = rlp.diagnostics() or {}
            if r_d.get('decisions'):
                out['wood_fast_rate'] = round(float(r_d.get('latent_single_rate', 0.0)), 3)
                out['wood_decisions'] = int(r_d['decisions'])
        except Exception:
            pass

    # 金 (Metal): episode memory occupancy, commits & hippocampal recall.
    mem = getattr(model, 'memory', None) if model is not None else None
    if mem is not None:
        try:
            occ = getattr(mem, 'occupied', None)
            if occ is not None:
                out['memory_occupied'] = int(occ.sum().item())
                out['memory_capacity'] = int(occ.numel())
            ep = getattr(mem, 'episodes', None)
            if ep is not None:
                out['episodes_committed'] = int(ep.item()) if hasattr(ep, 'item') else int(ep)
        except Exception:
            pass

    # 土 (Earth): causal relation bank & evidence state.
    cr = getattr(model, 'causal_relations', None) if model is not None else None
    if cr is not None:
        try:
            occ_r = getattr(cr, 'occupied', None)
            if occ_r is not None:
                out['causal_active_relations'] = int(occ_r.sum().item())
                out['causal_capacity'] = int(occ_r.numel())
            ev_cnt = getattr(cr, 'evidence_count', None)
            if ev_cnt is not None:
                out['causal_evidence_total'] = int(ev_cnt.sum().item()) if hasattr(ev_cnt, 'sum') else int(ev_cnt)
        except Exception:
            pass
    archive = getattr(engine, 'causal_archive', None)
    if archive is not None and hasattr(archive, 'summary'):
        try:
            asum = archive.summary() or {}
            if isinstance(asum.get('episodes'), int):
                out['archive_episodes'] = asum['episodes']
            if isinstance(asum.get('relations'), int):
                out['archive_relations'] = asum['relations']
        except Exception:
            pass
    return out


def _format_compact_five_elements(obs: dict, recent_metal_g: list[float]) -> str:
    parts = []
    if 'hnet_ratio' in obs:
        parts.append(f'动态分块{obs["hnet_ratio"]:.1f}x')
    if 'fire_z_kin' in obs:
        z = obs['fire_z_kin']
        parts.append('内在动量偏高' if z > 1.0 else ('内在偏安静' if z < -0.5 else '内在动量平稳'))
    if 'wood_fast_rate' in obs:
        r = obs['wood_fast_rate']
        parts.append(f'直觉快想{int(round(100 * r))}%')
    if 'memory_occupied' in obs and 'memory_capacity' in obs:
        ep = f'/累计{obs["episodes_committed"]}段' if 'episodes_committed' in obs else ''
        parts.append(f'情景经历库{obs["memory_occupied"]}/{obs["memory_capacity"]}{ep}')
    if 'causal_active_relations' in obs:
        ev = f'({obs["causal_evidence_total"]}条证据)' if 'causal_evidence_total' in obs else ''
        parts.append(f'因果关系库{obs["causal_active_relations"]}条{ev}')
    if len(recent_metal_g) >= 2:
        mg = sum(recent_metal_g[-4:]) / float(len(recent_metal_g[-4:]))
        parts.append(f'近期互动回报ΔG={mg:+.4f}')
    return ' | '.join(parts)


def capture_body_mirror(
    engine,
    *,
    utterance_bytes: bytes = b'',
    recent_book_bytes: bytes = b'',
    recent_metal_g: Iterable[float] = (),
) -> BodyMirrorSnapshot:
    """Build a compact, factual caregiver snapshot from real bodily observables."""
    alif_st = getattr(engine, 'alif_executive_state', None) or {}
    burst_hex = str(alif_st.get('last_burst_bytes_hex') or '')
    burst_raw = bytes.fromhex(burst_hex) if burst_hex else b''
    stumble_excerpt = clean_utf8_excerpt(burst_raw, max_chars=72)

    burst_nll = float(alif_st.get('last_burst_nll', 0.0) or 0.0)
    nll_ema = float(alif_st.get('nll_ema', 0.0) or 0.0)
    ratio = (burst_nll / max(nll_ema, 1e-4)) if (burst_nll > 0.0 and nll_ema > 0.0) else 1.0

    if stumble_excerpt:
        if ratio >= 1.35:
            stumble_cue = f'读到「{stumble_excerpt}」时明显卡顿、感到陌生'
        elif ratio >= 1.12:
            stumble_cue = f'读到「{stumble_excerpt}」时略有迟疑'
        else:
            stumble_cue = f'刚读过「{stumble_excerpt}」，整体还算顺畅'
    else:
        stumble_cue = '刚刚开始阅读，尚无明显卡顿片段'

    cause = str(alif_st.get('last_trigger_cause') or 'rheobase')
    trigger_cue = _cause_to_chinese(cause)

    # Read optional PhaseSpaceBoltzmann, ALIF pacing & H-Net cues purely as plain-language state.
    stability_parts = []
    gaps = alif_st.get('inter_spike_pieces') or []
    if len(gaps) >= 2:
        recent_gap = sum(int(x) for x in gaps[-2:]) / 2.0
        if recent_gap <= 1.5:
            stability_parts.append('停顿比较频繁、节奏有点急')
        elif recent_gap >= 4.0:
            stability_parts.append('节奏比较沉稳')
    model = getattr(engine, 'model', None)
    pb = getattr(model, 'phase_boltzmann', None) if model is not None else None
    if pb is not None and hasattr(pb, 'diagnostics'):
        try:
            pb_d = pb.diagnostics() or {}
            z_kin = pb_d.get('live_z_kin')
            if isinstance(z_kin, (int, float)) and math.isfinite(z_kin):
                stability_parts.append('内在波动偏高' if z_kin > 1.0 else '内在节奏较平稳')
        except Exception:
            pass
    body = getattr(model, 'body', None) if model is not None else None
    hnet = getattr(body, 'hnet_chunker', None) if body is not None else None
    if hnet is not None and hasattr(hnet, 'last_compression_ratio'):
        try:
            eff_r = float(hnet.last_compression_ratio)
            if math.isfinite(eff_r) and eff_r > 0.0:
                if eff_r < 3.0:
                    stability_parts.append('正在放慢逐字细读')
                elif eff_r >= 3.5:
                    stability_parts.append('正在按词块顺读')
        except Exception:
            pass
    g_list = [float(x) for x in recent_metal_g if isinstance(x, (int, float)) and math.isfinite(x)]
    if len(g_list) >= 3:
        mean_g = sum(g_list[-4:]) / float(len(g_list[-4:]))
        if mean_g > 0.005:
            stability_parts.append('最近几次互动后状态更稳')
        elif mean_g < -0.005:
            stability_parts.append('最近几次尝试仍在摸索')
    stability_cue = '，'.join(stability_parts) if stability_parts else '当前状态平稳'

    utt = classify_utterance(utterance_bytes)
    book_excerpt = clean_utf8_excerpt(recent_book_bytes, max_chars=220)
    five_obs = _extract_five_element_observables(engine)
    five_elements_cue = _format_compact_five_elements(five_obs, g_list)

    return BodyMirrorSnapshot(
        stumble_excerpt=stumble_excerpt,
        stumble_cue=stumble_cue,
        trigger_cue=trigger_cue,
        stability_cue=stability_cue,
        five_elements_cue=five_elements_cue,
        utterance_kind=utt['kind'],
        utterance_text=utt['text'],
        book_excerpt=book_excerpt,
        burst_ratio=round(float(ratio), 3),
        trigger_cause=cause,
    )


def format_caregiver_turn(
    snapshot: BodyMirrorSnapshot,
    *,
    perspective_mode: str = 'mirror_visible',
    trigger_kind: str = 'child_spoke',
    extra_note: str | None = None,
    opening_cue: str | None = None,
) -> str:
    """Format the tail-only ``user/message`` packet for the existing DSH Mother.

    Distinguishes three social triggers cleanly:
      - ``child_spoke``: Child explicitly called/addressed Mother -> respond first.
      - ``child_replied``: Child responded on Mother's active social floor.
      - ``mother_proactive``: Child paused quietly while reading -> Mother may
        gently comment, explain a stumble, or ask a warm simple question.

    Never mutates ``system_prompt`` or earlier turns, keeping DeepSeek server-side
    prefix KV cache hits intact.
    """
    if perspective_mode not in ('mirror_visible', 'book_only'):
        raise ValueError(f'unknown perspective_mode: {perspective_mode}')
    if trigger_kind not in ('child_spoke', 'child_replied', 'mother_proactive'):
        raise ValueError(f'unknown trigger_kind: {trigger_kind}')

    trigger_label = {
        'child_spoke': '[情境: 孩子主动叫你/向你开口，先温柔接住她说的话，再自然联系眼前书页]',
        'child_replied': '[情境: 孩子回应了你上一轮的话，顺着你们的对话自然接下去]',
        'mother_proactive': '[情境: 孩子安静读书时停顿下来，你可以轻声聊聊眼前故事或卡顿处，也可以抛一个不强求回答的小问题]',
    }[trigger_kind]

    lines = [trigger_label]
    if perspective_mode == 'mirror_visible':
        lines.append('[视角: 可见共读书页与孩子此刻状态]')
    else:
        lines.append('[视角: 仅见共读书页，看不到孩子内部状态]')

    if snapshot.book_excerpt:
        lines.append(f'[共读书页] {snapshot.book_excerpt}')

    if perspective_mode == 'mirror_visible':
        lines.append(
            f'[状态镜映] {snapshot.stumble_cue}；{snapshot.trigger_cue}；{snapshot.stability_cue}。'
        )
        if snapshot.five_elements_cue:
            lines.append(f'[内部运转与记忆库] {snapshot.five_elements_cue}')

    if extra_note:
        lines.append(f'[现场动态] {extra_note.strip()}')

    if snapshot.utterance_kind == 'silent':
        lines.append('[本次表达] （安静未开口）')
    elif snapshot.utterance_kind == 'babble':
        shown = snapshot.utterance_text or '零散字符'
        lines.append(f'[本次表达] （尚不成句的牙牙学语）{shown}')
    else:
        lines.append(f'[本次表达] {snapshot.utterance_text}')

    # Bilingual mirror cue: when child speaks predominantly English, nudge Mother to reply in warm English.
    if snapshot.utterance_text:
        en_chars = sum(1 for ch in snapshot.utterance_text if 'a' <= ch.lower() <= 'z')
        cjk_chars = sum(len(m) for m in _CJK_RUN_RE.findall(snapshot.utterance_text))
        if en_chars >= 6 and en_chars > cjk_chars * 3:
            lines.append('[语言跟随] 孩子本轮主要在说英文，请用温柔、简单、自然的英文回应。')

    if opening_cue:
        lines.append(f'[起句节奏] {opening_cue.strip()}')

    return '\n'.join(lines)


class AdaptiveCarePlanTracker:
    """Coarse, hysteresis-guarded parenting plan tracker.

    Guarantees that DSH ``plan.version`` changes ONLY on genuine developmental
    stage transitions (or explicit parent plan edits), protecting KV cache reuse.
    """

    def __init__(
        self,
        *,
        initial_stage: str = STAGE_INFANT,
        min_dwell_turns: int = 6,
        window_size: int = 8,
        user_plan_text: str | None = None,
    ):
        if initial_stage not in STAGE_ORDER:
            raise ValueError(f'unknown stage: {initial_stage}')
        self.stage = initial_stage
        self.min_dwell_turns = max(1, int(min_dwell_turns))
        self.window_size = max(3, int(window_size))
        self.turns_in_stage = 0
        self.recent_kinds: list[str] = []
        self.recent_metal_g: list[float] = []
        self.user_plan_text = user_plan_text.strip() if isinstance(user_plan_text, str) and user_plan_text.strip() else None
        self.mother_plan_text: str | None = None
        self.mengli_model_summary: str = ''
        self.version = 1
        self.transitions: list[dict] = []

    def current_plan(self) -> dict:
        if self.user_plan_text is not None:
            text = self.user_plan_text
        elif self.mother_plan_text is not None:
            text = self.mother_plan_text
        else:
            text = CARE_PLANS[self.stage]
        return dict(version=int(self.version), text=text)

    def apply_mother_reflection(self, *, new_care_plan: str | None, mengli_model: str = '', turn_index: int = 0) -> bool:
        """Apply Mother's autonomous periodic reflection plan unless pinned by a human parent."""
        if mengli_model:
            self.mengli_model_summary = str(mengli_model).strip()[:600]
        if self.user_plan_text is not None:
            return False
        cleaned = new_care_plan.strip() if isinstance(new_care_plan, str) and new_care_plan.strip() else None
        if not cleaned or cleaned == self.mother_plan_text:
            return False
        self.mother_plan_text = cleaned[:1200]
        self.version += 1
        self.transitions.append(dict(
            turn=int(turn_index),
            kind='mother_autonomous_reflection',
            stage=self.stage,
            version=int(self.version),
        ))
        return True

    def set_user_plan(self, text: str | None, *, turn_index: int = 0) -> dict:
        cleaned = text.strip() if isinstance(text, str) and text.strip() else None
        if cleaned == self.user_plan_text:
            return self.current_plan()
        self.user_plan_text = cleaned
        self.version += 1
        self.transitions.append(dict(
            turn=int(turn_index),
            kind='user_override' if cleaned is not None else 'resume_adaptive',
            stage=self.stage,
            version=int(self.version),
        ))
        return self.current_plan()

    def observe_turn(self, *, utterance_kind: str, metal_g: float | None = None, turn_index: int = 0) -> bool:
        """Record one Mother interaction and return True iff ``plan.version`` changed."""
        self.turns_in_stage += 1
        self.recent_kinds.append(str(utterance_kind))
        if len(self.recent_kinds) > self.window_size:
            del self.recent_kinds[:-self.window_size]
        if isinstance(metal_g, (int, float)) and math.isfinite(metal_g):
            self.recent_metal_g.append(float(metal_g))
            if len(self.recent_metal_g) > self.window_size:
                del self.recent_metal_g[:-self.window_size]

        # If a human parent explicitly pinned a custom care plan, never overwrite it automatically.
        if self.user_plan_text is not None:
            return False
        if self.turns_in_stage < self.min_dwell_turns or len(self.recent_kinds) < min(self.min_dwell_turns, self.window_size):
            return False

        n = len(self.recent_kinds)
        frag_or_better = sum(1 for k in self.recent_kinds if k in ('fragments', 'coherent')) / float(n)
        coherent_rate = sum(1 for k in self.recent_kinds if k == 'coherent') / float(n)
        babble_or_silent = sum(1 for k in self.recent_kinds if k in ('babble', 'silent')) / float(n)
        mean_g = (sum(self.recent_metal_g) / float(len(self.recent_metal_g))) if self.recent_metal_g else 0.0

        next_stage = self.stage
        if self.stage == STAGE_INFANT:
            if frag_or_better >= 0.50:
                next_stage = STAGE_EMERGING
        elif self.stage == STAGE_EMERGING:
            if coherent_rate >= 0.50 and mean_g >= -0.002:
                next_stage = STAGE_REFLECTIVE
            elif babble_or_silent >= 0.85:
                next_stage = STAGE_INFANT
        elif self.stage == STAGE_REFLECTIVE:
            if coherent_rate < 0.25 and babble_or_silent >= 0.60:
                next_stage = STAGE_EMERGING

        if next_stage != self.stage:
            prev = self.stage
            self.stage = next_stage
            self.turns_in_stage = 0
            self.version += 1
            self.transitions.append(dict(
                turn=int(turn_index),
                kind='developmental_transition',
                from_stage=prev,
                to_stage=next_stage,
                version=int(self.version),
            ))
            return True
        return False

    def state_dict(self) -> dict:
        return dict(
            schema=1,
            stage=self.stage,
            min_dwell_turns=int(self.min_dwell_turns),
            window_size=int(self.window_size),
            turns_in_stage=int(self.turns_in_stage),
            recent_kinds=list(self.recent_kinds),
            recent_metal_g=list(self.recent_metal_g),
            user_plan_text=self.user_plan_text,
            mother_plan_text=self.mother_plan_text,
            mengli_model_summary=self.mengli_model_summary,
            version=int(self.version),
            transitions=copy.deepcopy(self.transitions),
        )

    def load_state_dict(self, state: dict) -> None:
        if state.get('schema') != 1:
            raise ValueError('unsupported AdaptiveCarePlanTracker schema')
        self.stage = str(state['stage'])
        self.min_dwell_turns = int(state['min_dwell_turns'])
        self.window_size = int(state['window_size'])
        self.turns_in_stage = int(state['turns_in_stage'])
        self.recent_kinds = list(state.get('recent_kinds', []))
        self.recent_metal_g = [float(x) for x in state.get('recent_metal_g', [])]
        self.user_plan_text = state.get('user_plan_text')
        self.mother_plan_text = state.get('mother_plan_text')
        self.mengli_model_summary = str(state.get('mengli_model_summary') or '')
        self.version = int(state['version'])
        self.transitions = copy.deepcopy(state.get('transitions', []))


def build_five_element_reflection_dossier(
    engine,
    *,
    cycle: int,
    mother_turns: int,
    care_stage: str,
    plan_version: int,
    current_plan_text: str,
    prior_mengli_model: str = '',
    recent_kinds: Iterable[str] = (),
    recent_metal_g: Iterable[float] = (),
    mirror_g: Iterable[float] = (),
    book_only_g: Iterable[float] = (),
    recent_stumbles: Iterable[str] = (),
    recent_flux: dict | None = None,
    presence_labels: Iterable[str] = (),
) -> str:
    """Build a rich, factual Five-Element + Causal Bank + Episode Recall reflection dossier."""
    obs = _extract_five_element_observables(engine)
    kinds = list(recent_kinds)
    kind_counts = {k: kinds.count(k) for k in ('silent', 'babble', 'fragments', 'coherent')}
    mg = [float(x) for x in recent_metal_g if isinstance(x, (int, float)) and math.isfinite(x)]
    mv_g = [float(x) for x in mirror_g if isinstance(x, (int, float)) and math.isfinite(x)]
    bo_g = [float(x) for x in book_only_g if isinstance(x, (int, float)) and math.isfinite(x)]
    stumbles = [s for s in recent_stumbles if s][-5:]

    lines = [
        f'[阶段性育儿复盘与“梦漓模型”更新 · 第 {cycle} 轮 / 累计妈妈互动 {mother_turns} 轮]',
        '请作为梦漓的妈妈，结合你前面完整的陪伴记忆与下面最新的内部机制运转、因果关系库、情景经历调用及成长实测数据，像了解学生的老师那样更新你心中的「梦漓模型」，并决定是否微调下一阶段的育儿计划。',
        f'- 当前发育阶段：{care_stage}（计划 v{plan_version}）',
        f'- 当前在场对象：{", ".join(presence_labels) if presence_labels else "书页, 妈妈"}',
        f'- 梦漓各核心模块运转实测：',
        f'  · 语言预测与动态分块（H-Net）：动态压缩比={obs.get("hnet_ratio", "未知")}x，近期阅读均损 NLL_ema={obs.get("nll_ema", "未知")}',
        f'  · 专注节律与内在动量（ALIF / 相空间）：动量波动 z_kin={obs.get("fire_z_kin", "未知")}，自由能压力={obs.get("fire_pressure", "未知")}，停顿触发分布={obs.get("alif_causes", {})}',
        f'  · 主动意愿与行动规划：直觉快想占比={obs.get("wood_fast_rate", "未知")}（决策次数={obs.get("wood_decisions", "未知")}），近期表达形态分布={kind_counts}',
        f'  · 情景经历库与互动回报：经历库占用={obs.get("memory_occupied", "未知")}/{obs.get("memory_capacity", "未知")}（累计写入={obs.get("episodes_committed", "未知")}），近期平均 ΔG={(round(sum(mg)/len(mg), 4) if mg else "未知")}',
        f'  · 因果关系库：活跃因果关系={obs.get("causal_active_relations", "未知")}/{obs.get("causal_capacity", "未知")}，累计因果证据={obs.get("causal_evidence_total", "未知")}，归档关系={obs.get("archive_relations", "未知")}',
        f'- 双视角表现：可见视界平均 ΔG={(round(sum(mv_g)/len(mv_g), 4) if mv_g else "暂无")}（{len(mv_g)}轮） vs 仅见书页平均 ΔG={(round(sum(bo_g)/len(bo_g), 4) if bo_g else "暂无")}（{len(bo_g)}轮）',
    ]
    if recent_flux:
        lines.append(
            f'- 最近客观通量探针：英文16KB NLL={recent_flux.get("combined_16384_nll")}，'
            f'中文8KB NLL={recent_flux.get("chinese_8192_nll")}，'
            f'64旧事实MSE={recent_flux.get("fresh_64_facts_mse")}'
        )
    from scripts.ib_local import gaze_pointer as _gp
    gp_diag = _gp.diagnostics(engine)
    if gp_diag and gp_diag.get('target_counts'):
        parts = []
        g_ema = gp_diag.get('target_g_ema') or {}
        for tid, cnt in sorted(gp_diag['target_counts'].items()):
            ge = g_ema.get(tid)
            ge_s = f'(均ΔG={ge:+.4f})' if isinstance(ge, (int, float)) else ''
            parts.append(f'{tid}:{cnt}次{ge_s}')
        lines.append(f'- 对象指向分布（视线自主选择，熵={gp_diag.get("mean_entropy")}）：' + '，'.join(parts))
    if stumbles:
        lines.append('- 最近典型卡顿片段：' + ' / '.join(f'「{s[:48]}」' for s in stumbles))
    if prior_mengli_model:
        lines.append(f'- 你上一阶段的梦漓模型：{prior_mengli_model}')
    lines.append(f'- 当前生效育儿计划：{current_plan_text[:320]}')
    lines.append(
        '请严格只返回一个 JSON 对象（不要加多余解释），字段如下：\n'
        '{\n'
        '  "mengli_model": "你对梦漓当前学习进程、表达性格与主要薄弱点的事实性画像（2到3句中文）",\n'
        '  "progress_note": "记入育儿手记的本阶段进展与发现（2到3句中文）",\n'
        '  "update_plan": true或false,\n'
        '  "new_care_plan": "仅当 update_plan 为 true 时填写：面向接下来约16轮的具体中文育儿计划（150字内，严守视角边界、动态脚手架与简洁回复总纲）"\n'
        '}'
    )
    return '\n'.join(lines)


def parse_mother_reflection(raw: str) -> dict:
    """Parse Mother's JSON reflection output cleanly, degrading safely on plain text."""
    import json
    text = (raw or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return dict(
                    mengli_model=str(obj.get('mengli_model') or '').strip(),
                    progress_note=str(obj.get('progress_note') or '').strip(),
                    update_plan=bool(obj.get('update_plan', False)),
                    new_care_plan=str(obj.get('new_care_plan') or '').strip(),
                    parsed_json=True,
                )
        except Exception:
            pass
    return dict(
        mengli_model='',
        progress_note=text[:400],
        update_plan=False,
        new_care_plan='',
        parsed_json=False,
    )
