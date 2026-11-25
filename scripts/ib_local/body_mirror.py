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
MIRROR_VERSION = 'v2_evidence_compact'

FORM_LABELS: dict[str, str] = {
    'silent': '沉默',
    'babble': '零散字符',
    'fragments': '短片段',
    'coherent': '较长表达',
    'extended': '较长表达',
}

_COMMON_SHORT_EN = frozenset({
    'a', 'i', 'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'hi',
    'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'oh', 'ok', 'on', 'or',
    'so', 'to', 'up', 'us', 'we',
})
_TOKEN_RE = re.compile(r'[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}')


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
    """Classify MengLi's literal surface form (UTF-8 integrity + length) without claiming comprehension."""
    if isinstance(raw, (bytes, bytearray, memoryview)):
        b = bytes(raw)
        stripped = b.strip()
        if not stripped:
            return dict(kind='silent', form_label='沉默', text='', valid_utf8_ratio=1.0, word_units=0, byte_len=0)
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
            return dict(kind='silent', form_label='沉默', text='', valid_utf8_ratio=1.0, word_units=0, byte_len=0)
        valid_ratio = 1.0

    ascii_words = _ASCII_WORD_RE.findall(text)
    cjk_chars = sum(len(m) for m in _CJK_RUN_RE.findall(text))
    # Count 2 CJK chars ≈ 1 lexical unit alongside >=3-letter ASCII runs.
    word_units = len(ascii_words) + (cjk_chars // 2)

    has_markup = ('[' in text) or (']' in text)
    # Surface-form check for ASCII-heavy spans: reject clipped-start or noisy fragments
    # (e.g. "perience. Fe frent ..." or "that meetions for spost, sa") from extended form.
    ascii_heavy = (len(ascii_words) > 0 and cjk_chars < 6)
    ascii_clean = True
    if ascii_heavy:
        raw_tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        orphan_short = [t for t in raw_tokens if len(t) <= 2 and t.lower() not in _COMMON_SHORT_EN]
        starts_clipped = bool(re.match(r'^[^A-Za-z0-9\u4e00-\u9fff]|^[a-z]{2,}\.\s+[A-Z]', text))
        ends_clipped = bool(re.search(r'(?:,\s*[a-z]{1,2}|:\s*[a-z]{1,2})$', text))
        if orphan_short or starts_clipped or ends_clipped:
            ascii_clean = False

    if not text:
        kind = 'babble'
    elif (valid_ratio >= 0.95 and not has_markup
          and ((cjk_chars >= 6 and len(text) >= 6) or (word_units >= 4 and ascii_clean))):
        kind = 'coherent'
    elif valid_ratio >= 0.72 and (word_units >= 1 or cjk_chars >= 2):
        kind = 'fragments'
    else:
        kind = 'babble'

    return dict(
        kind=kind,
        form_label=FORM_LABELS.get(kind, '短片段'),
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
    full_book = clean_utf8_excerpt(recent_book_bytes, max_chars=0)
    if stumble_excerpt and full_book:
        pos = full_book.rfind(stumble_excerpt)
        if pos < 0 and len(stumble_excerpt) >= 12:
            pos = full_book.rfind(stumble_excerpt[:16])
        if pos >= 0:
            half_ctx = max(40, (220 - len(stumble_excerpt)) // 2)
            start = max(0, pos - half_ctx)
            end = min(len(full_book), start + 220)
            start = max(0, end - 220)
            book_excerpt = full_book[start:end].strip()
        else:
            book_excerpt = full_book[-220:].lstrip()
    else:
        book_excerpt = full_book[-220:].lstrip()
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


def _extract_lexical_terms(text: str) -> list[str]:
    """Extract normalized >=3-letter ASCII tokens and 2-char CJK bigrams for literal follow-up tracking."""
    if not text:
        return []
    seen: list[str] = []
    for tok in re.findall(r'[A-Za-z]{3,}', text.lower()):
        if tok not in seen:
            seen.append(tok)
    for run in _CJK_RUN_RE.findall(text):
        if len(run) >= 2:
            for i in range(len(run) - 1):
                bg = run[i:i + 2]
                if bg not in seen:
                    seen.append(bg)
    return seen


def _read_archived_slot_text(engine, slot: int) -> str | None:
    """Return literal stored utterance text for a hot memory slot ONLY if actually recorded in ExactUtteranceTable."""
    if engine is None or slot < 0:
        return None
    model = getattr(engine, 'model', None)
    ret = getattr(model, 'retrieval', None) if model is not None else None
    if ret is None:
        return None
    words = getattr(ret, 'words', None)
    counts = getattr(ret, 'word_count', None)
    if words is None or counts is None:
        return None
    try:
        cap = int(counts.shape[0])
        if slot >= cap:
            return None
        wc = int(counts[slot].item())
        if wc <= 0:
            return None
        raw = bytes(words[slot, :wc].detach().cpu().tolist())
        cleaned = clean_utf8_excerpt(raw, max_chars=48)
        return cleaned if cleaned else None
    except Exception:
        return None


def build_learning_trace(
    engine,
    row: dict,
    *,
    cycle: int = 0,
    prev_trace: dict | None = None,
) -> dict:
    """Build one shared 5-stage factual learning trace for BOTH Mother and the Web UI.

    Strictly read-only and strictly factual:
      1. situation: heard/read context, send/observe targets, perspective boundary
      2. expression_choice: literal utterance, surface form, and pathway push (habit / Wood / episodic / water)
      3. memory_recall: top weighted episodic slots, age in bytes, archive episode id, and literal text ONLY when stored
      4. response_revision: actual reply, ΔG, slot retention feedback (+ / -), causal commit status
      5. followup: literal lexical continuity against the immediately preceding trace
    """
    world = row.get('world') if isinstance(row.get('world'), dict) else {}
    gaze = world.get('gaze_decision') if isinstance(world.get('gaze_decision'), dict) else {}
    send_t = world.get('send_target_id') or row.get('send_target_id') or gaze.get('send_target') or 'none'
    obs_t = world.get('observe_target_id') or row.get('observe_target_id') or gaze.get('observe_target') or 'none'
    mode = world.get('interaction_mode') or row.get('interaction_mode') or gaze.get('interaction_mode') or ''
    persp = world.get('perspective_mode') or None
    channel = world.get('channel') or 'unknown'
    stumble = str(world.get('mirror_excerpt') or '')[:64]
    book_ex = str(world.get('book_excerpt') or '')[:100]

    raw_utt = str(world.get('child_text') or '').strip()
    if raw_utt in ('(silent)', '[咿呀声]'):
        utt_cls = classify_utterance(b'' if raw_utt == '(silent)' else b'\xff')
    else:
        utt_cls = classify_utterance(raw_utt)

    wood_tv = row.get('wood_policy_tv')
    epi_tv = row.get('episodic_policy_tv')
    water_tv = row.get('water_direct_choice_tv')
    push_parts = []
    if isinstance(wood_tv, (int, float)) and math.isfinite(wood_tv) and wood_tv > 1e-4:
        push_parts.append(f'木前瞻TV={wood_tv:.4f}')
    if isinstance(epi_tv, (int, float)) and math.isfinite(epi_tv) and epi_tv > 1e-4:
        push_parts.append(f'情景记忆TV={epi_tv:.4f}')
    if isinstance(water_tv, (int, float)) and math.isfinite(water_tv) and water_tv > 1e-5:
        push_parts.append(f'水直连TV={water_tv:.4f}')
    obs_prior = gaze.get('observe_causal_prior') if isinstance(gaze.get('observe_causal_prior'), dict) else None
    if obs_prior:
        prior_str = '/'.join(f'{k}:{v:+.3f}' for k, v in obs_prior.items())
        push_parts.append(f'视线木先验({prior_str})')

    # Stage 3: Memory recall (only real active weighted slots; text ONLY when stored in ExactUtteranceTable)
    slots = row.get('episodic_read_slots') or []
    weights = row.get('episodic_read_weights') or []
    mem = getattr(getattr(engine, 'model', None), 'memory', None) if engine is not None else None
    arch = getattr(engine, 'causal_archive', None) if engine is not None else None
    hot_map = getattr(arch, 'hot', None) if arch is not None else None
    mem_clock = None
    if mem is not None and getattr(mem, 'clock', None) is not None:
        try:
            mem_clock = int(mem.clock.item()) if hasattr(mem.clock, 'item') else int(mem.clock)
        except Exception:
            mem_clock = None

    top_slots = []
    for s_val, w_val in zip(slots, weights):
        if not isinstance(w_val, (int, float)) or w_val <= 0.01:
            continue
        slot_i = int(s_val)
        age_b = None
        if mem is not None and mem_clock is not None and getattr(mem, 'last_write', None) is not None:
            try:
                lw = int(mem.last_write[slot_i].item())
                if lw > 0:
                    age_b = max(0, mem_clock - lw)
            except Exception:
                age_b = None
        ep_id = None
        if isinstance(hot_map, dict) and str(slot_i) in hot_map:
            ep_id = hot_map[str(slot_i)].get('id')
        stored_txt = _read_archived_slot_text(engine, slot_i)
        top_slots.append(dict(
            slot=slot_i,
            weight=round(float(w_val), 3),
            episode_id=ep_id,
            age_bytes=age_b,
            text=stored_txt,
        ))
        if len(top_slots) >= 3:
            break

    # Stage 4: Response & Revision (actual reply + retention feedback + causal commit)
    reply_txt = str(world.get('mother_reply') or world.get('text_reply') or '').strip()[:140]
    metal_g = row.get('metal_g')
    metal_g_f = round(float(metal_g), 4) if isinstance(metal_g, (int, float)) and math.isfinite(metal_g) else None
    resp_nll = row.get('response_nll')
    resp_nll_f = round(float(resp_nll), 3) if isinstance(resp_nll, (int, float)) and math.isfinite(resp_nll) else None

    rc_obs = (((row.get('read_credit') or {}).get('response') or {}).get('retention_feedback') or {}).get('observations') or []
    pos_obs = [o for o in rc_obs if isinstance(o, dict) and isinstance(o.get('benefit'), (int, float)) and o['benefit'] > 0]
    neg_obs = [o for o in rc_obs if isinstance(o, dict) and isinstance(o.get('benefit'), (int, float)) and o['benefit'] < 0]
    best_fb = None
    if pos_obs:
        b = max(pos_obs, key=lambda o: float(o['benefit']))
        best_fb = dict(slot=int(b['slot']), age_bytes=b.get('age_bytes'), benefit=round(float(b['benefit']), 5))
    worst_fb = None
    if neg_obs:
        w = min(neg_obs, key=lambda o: float(o['benefit']))
        worst_fb = dict(slot=int(w['slot']), age_bytes=w.get('age_bytes'), benefit=round(float(w['benefit']), 5))

    ut = row.get('use_transport') if isinstance(row.get('use_transport'), dict) else {}
    causal_committed = bool(ut.get('committed', False))
    causal_unknown = ut.get('unknown') if isinstance(ut.get('unknown'), str) else None

    # Stage 5: Follow-up lexical continuity vs immediately previous trace
    cur_terms = _extract_lexical_terms(raw_utt if raw_utt not in ('(silent)', '[咿呀声]') else '')
    reused_self: list[str] = []
    reused_reply: list[str] = []
    followup_status = 'quiet' if not cur_terms else 'new_attempt'
    if cur_terms and isinstance(prev_trace, dict):
        prev_u = str((prev_trace.get('expression_choice') or {}).get('utterance_text') or '')
        prev_r = str((prev_trace.get('response_revision') or {}).get('reply_text') or '')
        prev_u_terms = set(_extract_lexical_terms(prev_u))
        prev_r_terms = set(_extract_lexical_terms(prev_r))
        reused_self = [t for t in cur_terms if t in prev_u_terms][:3]
        reused_reply = [t for t in cur_terms if t in prev_r_terms][:3]
        if reused_reply:
            followup_status = 'echoed_reply'
        elif reused_self:
            followup_status = 'continued_self'

    return dict(
        cycle=int(cycle),
        mirror_version=MIRROR_VERSION,
        situation=dict(
            channel=str(channel),
            send_target=str(send_t),
            observe_target=str(obs_t),
            interaction_mode=str(mode),
            perspective_mode=persp,
            stumble_excerpt=stumble,
            book_excerpt=book_ex,
        ),
        expression_choice=dict(
            utterance_text=raw_utt[:100],
            form_label=utt_cls['form_label'],
            prior_status=gaze.get('prior_status'),
            observe_probs=(gaze.get('observe_probs') if isinstance(gaze.get('observe_probs'), dict) else None),
            observe_causal_prior=obs_prior,
            wood_policy_tv=(round(float(wood_tv), 5) if isinstance(wood_tv, (int, float)) and math.isfinite(wood_tv) else None),
            episodic_policy_tv=(round(float(epi_tv), 5) if isinstance(epi_tv, (int, float)) and math.isfinite(epi_tv) else None),
            water_direct_choice_tv=(round(float(water_tv), 5) if isinstance(water_tv, (int, float)) and math.isfinite(water_tv) else None),
            push_summary='，'.join(push_parts) if push_parts else '习惯主导',
        ),
        memory_recall=dict(
            top_slots=top_slots,
            warmed_relations=(row.get('causal_archive') or {}).get('warmed_relations') if isinstance(row.get('causal_archive'), dict) else None,
            loaded_episodes=(row.get('causal_archive') or {}).get('loaded_episodes') if isinstance(row.get('causal_archive'), dict) else None,
        ),
        response_revision=dict(
            reply_text=reply_txt,
            metal_g=metal_g_f,
            response_nll=resp_nll_f,
            slot_feedback=dict(pos=len(pos_obs), neg=len(neg_obs), best=best_fb, worst=worst_fb),
            causal_committed=causal_committed,
            causal_unknown=causal_unknown,
        ),
        followup=dict(
            status=followup_status,
            reused_from_self=reused_self,
            reused_from_reply=reused_reply,
        ),
    )


def format_prev_trace_for_mother(prev_trace: dict | None, current_utterance: str = '') -> str:
    """Format a compact 1-line factual summary of the previous turn's outcome & current carry-over."""
    if not isinstance(prev_trace, dict):
        return ''
    ec = prev_trace.get('expression_choice') or {}
    rr = prev_trace.get('response_revision') or {}
    mr = prev_trace.get('memory_recall') or {}
    parts = []
    prev_u = str(ec.get('utterance_text') or '').strip()
    if prev_u:
        parts.append(f'上轮「{prev_u[:24]}」')
    g = rr.get('metal_g')
    if isinstance(g, (int, float)):
        parts.append(f'ΔG={g:+.4f}')
    fb = rr.get('slot_feedback') or {}
    pos_n, neg_n = int(fb.get('pos') or 0), int(fb.get('neg') or 0)
    if pos_n or neg_n:
        parts.append(f'经历反馈+{pos_n}/-{neg_n}')
    top_slots = mr.get('top_slots') or []
    txt_slots = [f'槽{s["slot"]}「{s["text"][:16]}」' for s in top_slots if s.get('text')]
    if txt_slots:
        parts.append('取用' + ','.join(txt_slots[:2]))
    elif top_slots:
        parts.append(f'取用槽{top_slots[0]["slot"]}(w={top_slots[0]["weight"]:.2f})')
    # Live lexical carry-over between previous turn and THIS current utterance
    cur_terms = _extract_lexical_terms(current_utterance)
    if cur_terms:
        prev_r_terms = set(_extract_lexical_terms(str(rr.get('reply_text') or '')))
        prev_u_terms = set(_extract_lexical_terms(prev_u))
        hit_r = [t for t in cur_terms if t in prev_r_terms][:2]
        hit_u = [t for t in cur_terms if t in prev_u_terms][:2]
        if hit_r:
            parts.append(f'本轮承接回应词「{"/".join(hit_r)}」')
        elif hit_u:
            parts.append(f'本轮延续自词「{"/".join(hit_u)}」')
    return '｜'.join(parts)


def format_trace_for_reflection(trace: dict) -> str:
    """One crisp factual evidence line per turn for Mother's 16-turn reflection dossier."""
    cyc = trace.get('cycle', '?')
    sit = trace.get('situation') or {}
    ec = trace.get('expression_choice') or {}
    mr = trace.get('memory_recall') or {}
    rr = trace.get('response_revision') or {}
    fu = trace.get('followup') or {}

    persp = '可见' if sit.get('perspective_mode') == 'mirror_visible' else ('仅书页' if sit.get('perspective_mode') == 'book_only' else '-')
    tgt = f'{sit.get("send_target", "-")}→{sit.get("observe_target", "-")}'
    utt = ec.get('utterance_text') or '（静）'
    form = ec.get('form_label') or ''
    rep = str(rr.get('reply_text') or '')[:32]
    g = rr.get('metal_g')
    g_s = f'ΔG={g:+.4f}' if isinstance(g, (int, float)) else 'ΔG=-'
    fb = rr.get('slot_feedback') or {}
    fb_s = f'槽+{fb.get("pos", 0)}/-{fb.get("neg", 0)}' if (fb.get('pos') or fb.get('neg')) else ''
    top_slots = mr.get('top_slots') or []
    mem_s = ''
    if top_slots:
        s0 = top_slots[0]
        mem_s = f'忆槽{s0["slot"]}「{s0["text"][:14]}」' if s0.get('text') else f'忆槽{s0["slot"]}'
    fu_map = {'echoed_reply': '承回应', 'continued_self': '续自词', 'new_attempt': '新试', 'quiet': '静'}
    fu_s = fu_map.get(fu.get('status', ''), '')
    reused = (fu.get('reused_from_reply') or fu.get('reused_from_self') or [])
    if reused:
        fu_s += f'({"/".join(reused[:2])})'
    extras = ' '.join(x for x in (g_s, fb_s, mem_s, fu_s) if x)
    return f'  #{cyc} [{persp}|{tgt}|{form}] 梦漓「{utt[:36]}」→ 回「{rep}」｜{extras}'


def format_caregiver_turn(
    snapshot: BodyMirrorSnapshot,
    *,
    perspective_mode: str = 'mirror_visible',
    trigger_kind: str = 'child_spoke',
    extra_note: str | None = None,
    opening_cue: str | None = None,
    send_target: str | None = None,
    observe_target: str | None = None,
    include_book_excerpt: bool = True,
    allow_mother_directives: bool = False,
    has_parent: bool = False,
    prev_trace: dict | None = None,
    current_gaze: dict | None = None,
    turn_number: int = 0,
) -> str:
    """Format an economical, factual tail-only ``user/message`` packet for DSH Mother.

    Never mutates ``system_prompt`` or earlier turns, keeping DeepSeek prefix KV cache hits
    intact while slashing repetitive per-turn instruction tokens by >60%.
    """
    if perspective_mode not in ('mirror_visible', 'book_only'):
        raise ValueError(f'unknown perspective_mode: {perspective_mode}')
    if trigger_kind not in ('child_spoke', 'child_replied', 'mother_proactive'):
        raise ValueError(f'unknown trigger_kind: {trigger_kind}')

    # 1) Compact single-line situation + perspective header (strictly honest perspective boundary).
    vis_tag = '可见视界' if perspective_mode == 'mirror_visible' else '仅见书页，看不到内部状态'
    if send_target == 'mother' and observe_target == 'mother' and not include_book_excerpt:
        sit_txt = '面对面聊天（未看书页），接续刚才对话' if trigger_kind == 'child_replied' else '面对面聊天（未看书页），直接看着你开口'
    elif send_target == 'none' and observe_target == 'mother':
        sit_txt = '安静抬头看你未开口' if snapshot.utterance_kind == 'silent' else '看着你小声自言自语'
    elif send_target == 'mother' and observe_target == 'book':
        sit_txt = '边看书页边向你开口'
    else:
        sit_txt = {
            'child_spoke': '共读中向你开口',
            'child_replied': '共读中回应你上轮的话',
            'mother_proactive': '安静读书停顿',
        }[trigger_kind]
    lines = [f'[{vis_tag}｜{sit_txt}]']

    # 2) Shared storybook excerpt (capped compactly at 150 chars to save tokens).
    if include_book_excerpt and snapshot.book_excerpt:
        lines.append(f'[书页] {snapshot.book_excerpt[:150]}')

    # 3) Factual bodily mirror + previous-turn learning outcome (ONLY in mirror_visible!).
    if perspective_mode == 'mirror_visible':
        mirror_parts = []
        if snapshot.stumble_excerpt:
            mirror_parts.append(f'卡顿「{snapshot.stumble_excerpt[:56]}」({snapshot.burst_ratio:.2f}x)')
        if snapshot.stability_cue:
            mirror_parts.append(snapshot.stability_cue)
        if isinstance(current_gaze, dict):
            op = current_gaze.get('observe_probs') or current_gaze.get('probs')
            if isinstance(op, dict) and op:
                p_str = '/'.join(f'{k}:{int(round(100 * v))}%' for k, v in op.items())
                mirror_parts.append(f'视线({p_str})')
        if mirror_parts:
            lines.append('[视界] ' + '｜'.join(mirror_parts))
        prev_line = format_prev_trace_for_mother(prev_trace, snapshot.utterance_text)
        if prev_line:
            lines.append(f'[学习链] {prev_line}')

    if extra_note:
        lines.append(f'[现场] {extra_note.strip()}')

    # 4) Actual utterance (honest literal form; never overclaim comprehension).
    if snapshot.utterance_kind == 'silent':
        lines.append('[梦漓] （安静未开口）')
    elif snapshot.utterance_kind == 'babble':
        shown = snapshot.utterance_text or '零散字符'
        lines.append(f'[梦漓] （零散字符）{shown}')
    else:
        lines.append(f'[梦漓] {snapshot.utterance_text}')

    # 5) Bilingual mirror cue only when predominantly English.
    if snapshot.utterance_text:
        en_chars = sum(1 for ch in snapshot.utterance_text if 'a' <= ch.lower() <= 'z')
        cjk_chars = sum(len(m) for m in _CJK_RUN_RE.findall(snapshot.utterance_text))
        if en_chars >= 6 and en_chars > cjk_chars * 3:
            lines.append('[语言] 本轮偏英文，请用温柔简短英文回应。')

    # 6) Economical directives & anti-repetition cue (only emit when actually needed!).
    cue_items = []
    if allow_mother_directives and (has_parent or turn_number == 0 or (turn_number % 8 == 0)):
        d_txt = '可用@翻书:词 或 @安静陪伴' + (' 或 @对爸爸' if has_parent else '') + '；纯口语无括号动作'
        cue_items.append(d_txt)
    if opening_cue:
        cue_items.append(opening_cue.strip())
    if cue_items:
        lines.append('[提示] ' + '；'.join(cue_items))

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
    recent_traces: Iterable[dict] = (),
) -> str:
    """Build an evidence-anchored, compact reflection dossier for Mother (zero semantic overclaiming)."""
    obs = _extract_five_element_observables(engine)
    kinds = list(recent_kinds)
    form_counts = {
        '沉默': kinds.count('silent'),
        '零散字符': kinds.count('babble'),
        '短片段': kinds.count('fragments'),
        '较长表达': kinds.count('coherent') + kinds.count('extended'),
    }
    mg = [float(x) for x in recent_metal_g if isinstance(x, (int, float)) and math.isfinite(x)]
    mv_g = [float(x) for x in mirror_g if isinstance(x, (int, float)) and math.isfinite(x)]
    bo_g = [float(x) for x in book_only_g if isinstance(x, (int, float)) and math.isfinite(x)]
    stumbles = [s for s in recent_stumbles if s][-4:]
    traces = [t for t in recent_traces if isinstance(t, dict)][-14:]

    lines = [
        f'[阶段性复盘 · 第 {cycle} 轮 / 妈妈累计 {mother_turns} 轮 / 镜映 {MIRROR_VERSION}]',
        '请严格依据下面最近十余轮的真实交互证据链更新「梦漓模型」：每条判断须标出近期轮次号(#轮次)；旧画像中近16轮缺乏新证据的说法请直接删去，如实记录破碎或未接住的反例。',
        f'- 阶段/在场：{care_stage}(v{plan_version})｜{", ".join(presence_labels) if presence_labels else "书页, 妈妈"}',
        f'- 字面形态计数（仅按字数与UTF-8完整度，不代表成句或理解）：{form_counts}',
        (
            f'- 内部运转：H-Net={obs.get("hnet_ratio", "-")}x, NLL_ema={obs.get("nll_ema", "-")}｜'
            f'火z_kin={obs.get("fire_z_kin", "-")}｜木快想={obs.get("wood_fast_rate", "-")}｜'
            f'经历库={obs.get("memory_occupied", "-")}/{obs.get("memory_capacity", "-")}(写{obs.get("episodes_committed", "-")})｜'
            f'因果库={obs.get("causal_active_relations", "-")}({obs.get("causal_evidence_total", "-")}证)'
        ),
        (
            f'- 互动回报：近期均ΔG={(round(sum(mg)/len(mg), 4) if mg else "-")}｜'
            f'可见视界={(round(sum(mv_g)/len(mv_g), 4) if mv_g else "-")}({len(mv_g)}轮) vs '
            f'仅书页={(round(sum(bo_g)/len(bo_g), 4) if bo_g else "-")}({len(bo_g)}轮)'
        ),
    ]
    if recent_flux:
        lines.append(
            f'- 固定探针：英16KB={recent_flux.get("combined_16384_nll")}｜'
            f'中8KB={recent_flux.get("chinese_8192_nll")}｜64旧事实MSE={recent_flux.get("fresh_64_facts_mse")}'
        )
    from scripts.ib_local import gaze_pointer as _gp
    gp_diag = _gp.diagnostics(engine)
    if gp_diag and gp_diag.get('target_counts'):
        parts = []
        g_ema = gp_diag.get('target_g_ema') or {}
        for tid, cnt in sorted(gp_diag['target_counts'].items()):
            ge = g_ema.get(tid)
            ge_s = f'(ΔG={ge:+.4f})' if isinstance(ge, (int, float)) else ''
            parts.append(f'{tid}:{cnt}{ge_s}')
        lines.append(f'- 对象指向(熵={gp_diag.get("mean_entropy")})：' + '，'.join(parts))
    if traces:
        lines.append('- 最近真实轨迹证据链（原话/回应/回报/经历槽/承接）：')
        for tr in traces:
            lines.append(format_trace_for_reflection(tr))
    if stumbles:
        lines.append('- 近期卡顿：' + ' / '.join(f'「{s[:42]}」' for s in stumbles))
    if prior_mengli_model:
        lines.append(f'- 上期画像（缺乏本期轮次证据的旧结论应淘汰）：{prior_mengli_model[:360]}')
    lines.append(f'- 当前计划(v{plan_version})：{current_plan_text[:280]}')
    lines.append(
        '严格只返回JSON：\n'
        '{"mengli_model":"2-3句中文，每条判断带#轮次依据，含正例与反例，淘汰无近期证据的旧印象",'
        '"progress_note":"2句中文记录本阶段真实变化与未接住处",'
        '"update_plan":true或false,'
        '"new_care_plan":"仅update_plan为true时填，140字内中文计划，保留@翻书:词/@安静陪伴与纯口语要求"}'
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
