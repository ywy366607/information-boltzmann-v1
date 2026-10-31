"""Evidence notebook for Ling's caregiver reviews; never declares a child mature from NLL."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import Any


_CONTENT_STOP_WORDS = frozenset((
    'i you me my your we us our the a an and or is are was were be to for of in on at it that this '
    'with mama papa mengli meng say said hear here there he she him her his they them their its '
    'itself myself yourself ourselves themselves if when where why how what which who whom whose '
    'whether either neither all any some each every both one not no yes so then than as but because '
    'although despite within after before through from into out up down just only still also very '
    'can could would should will shall may might must do does did have has had been being more '
    'most much many other such perhaps re ve ll'
).split())


def _content_terms(text):
    """Retrieval terms, not a filter on words heard or learned by the child."""
    # The nonverbal protocol event is not a child asking about "quiet" or
    # "turn". In particular it must not retrieve the quiet hallway clock on
    # every silent turn. An actually spoken word "quiet" is still searchable.
    text = str(text).replace('<quiet_turn/>', ' ')
    return {term for term in re.findall(r'[a-z]+', text.lower())
            if len(term) > 1 and term not in _CONTENT_STOP_WORDS}


def _atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


class ParentingHarness:
    """Store distinct observations and suggest care adjustments for Ling to review.

    An educational stage is independent of a temporary care adjustment.  By default
    this class writes a notebook only; Ling remains the caregiver making decisions.
    """

    def __init__(self, plan_path='data/harness/raising.json', state_path='data/harness/parenting_state.json',
                 notes_path='data/harness/mother_notes.md'):
        self.plan_path, self.state_path, self.notes_path = map(Path, (plan_path, state_path, notes_path))
        self.history = []
        self.last_review_turn = 0
        if self.state_path.exists():
            saved = json.loads(self.state_path.read_text(encoding='utf-8-sig'))
            if saved.get('version') == 2:
                self.history = saved.get('history', [])[-32:]
                self.last_review_turn = saved.get('last_review_turn', 0)

    def _write_note(self, trained, observation, change, reason):
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        with self.notes_path.open('a', encoding='utf-8') as handle:
            handle.write(f"| {datetime.now():%Y-%m-%d %H:%M:%S} | token {trained}: {observation} | {change} | {reason} |\n")

    def update(self, *, trained, monitor, nll=None, unigram=None, reply_nll=None, reply_unigram=None,
               epistemic=None, brain=None, active_loop=None, speech_consequence=None, apply=False):
        plan = json.loads(self.plan_path.read_text(encoding='utf-8-sig'))
        parenting, settings = plan.get('parenting', {}), plan.get('harness', {})
        stage = parenting.get('education_stage', 'connection')
        # NLL from self-hearing or automatic calls cannot certify normal-reply learning.
        gain = reply_unigram - reply_nll if reply_nll is not None and reply_unigram is not None else None
        turns = int(monitor.get('mother_turns', monitor.get('fly_turns', trained)))
        key = (int(trained), turns)
        if self.history and key <= (self.history[-1]['trained'], self.history[-1]['turns']):
            return {'action': 'already_observed', 'changed': False, 'stage': stage, 'gain': gain}
        spacing = int(settings.get('check_every_new_mother_turns', 12))
        # An overlapping 8-token window of the same conversation is not independent persistence.
        if self.history and turns - self.history[-1]['turns'] < spacing:
            return {'action': 'collect_more_interaction', 'changed': False, 'stage': stage, 'gain': gain}
        speech_n = int(monitor.get('fly_spoken_tokens', 0))
        mother_n = int(monitor.get('mother_word_count', 0))
        child_echo = speech_n >= 24 and monitor.get('fly_repeat_bigrams', 0) >= settings.get('alarm_repeat_bigrams', .5)
        mother_echo = mother_n >= 80 and (monitor.get('mother_repeat_bigrams', 0) >= .4 or
                                          monitor.get('mother_name_prefix_share', 0) >= .65)
        silent = monitor.get('fly_silent_turns', 0) >= .9 and int(monitor.get('recent_fly_turns', 0)) >= 30
        observation = {'trained': int(trained), 'turns': turns, 'child_echo': child_echo,
                       'mother_echo': mother_echo, 'silent': silent, 'gain': gain}
        self.history.append(observation)
        self.history = self.history[-32:]
        persistent = lambda field: len(self.history) >= 3 and all(x[field] for x in self.history[-3:])
        action, reason = 'observe', 'Listen and read the conversation before changing the lesson.'
        if persistent('mother_echo') or persistent('child_echo'):
            action, reason = 'caregiver_review_echo', 'Review mother scaffolding and context; meaningful repetition remains welcome.'
        elif persistent('silent'):
            action, reason = 'caregiver_review_silence', 'Check listening, excitability and invitations; quietness alone is not failure.'
        elif len(self.history) >= 3 and all(x['gain'] is not None and x['gain'] > 0 for x in self.history[-3:]):
            action, reason = 'record_predictive_progress', 'Prediction improved; collect comprehension and transfer evidence before advancing stage.'
        changed = False
        if apply and action.startswith('caregiver_review_'):
            # Only an explicit Ling action changes the environmental teaching mode.
            # Never edit speech policy, curriculum stage, temperature or physiology here.
            plan.setdefault('parenting', {})['care_mode'] = action.removeprefix('caregiver_review_')
            _atomic_json(self.plan_path, plan)
            self._write_note(trained, action, 'care mode only; Ling reviews dialogue', reason)
            changed = True
        self.last_review_turn = turns
        _atomic_json(self.state_path, {'version': 2, 'history': self.history,
                                      'last_review_turn': self.last_review_turn})
        brain = brain or {}
        required = ('kc', 'mbon', 'dan', 'plasticity', 'homeostasis', 'familiarity', 'read_think_speak')
        checks = brain.get('checks', {})
        return {'action': action, 'reason': reason, 'changed': changed, 'stage': stage, 'gain': gain,
                'stage_advance': 'requires Ling evidence review',
                'brain_status': {name: checks.get(name, {'status': 'unmeasured'}) for name in required}}


def conversation_health(transcript, child_name='Mengli', last=50):
    """Comparable lexical metrics: punctuation-normalized, per-turn bigrams.

    Silent turns count as silence, never as a repeated spoken sentence.  Vocabulary
    ratios are descriptive with their sample sizes, not developmental pass marks.
    """
    def words(text):
        return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    turns = {who: [str(t) for w, t in transcript if w == who][-last:] for who in ('fly', 'mother')}
    out = {'health_version': 2, 'sample_turns': last}
    for who, texts in turns.items():
        tokenized = [words(t) for t in texts]
        all_words = [w for ts in tokenized for w in ts]
        bigrams = [pair for ts in tokenized for pair in zip(ts, ts[1:])]
        counts = Counter(all_words)
        top, n = counts.most_common(1)[0] if counts else ('', 0)
        out.update({f'{who}_ttr': len(counts)/len(all_words) if all_words else None,
                    f'{who}_word_entropy_bits': -sum((c / len(all_words)) * math.log2(c / len(all_words)) for c in counts.values()) if all_words else None,
                    f'{who}_effective_vocabulary': 2 ** (-sum((c / len(all_words)) * math.log2(c / len(all_words)) for c in counts.values())) if all_words else None,
                    f'{who}_repeat_bigrams': 1-len(set(bigrams))/len(bigrams) if bigrams else 0.,
                    f'{who}_top_token': top, f'{who}_top_token_share': n/max(1,len(all_words)),
                    f'{who}_word_count': len(all_words), f'recent_{who}_turns': len(texts)})
        if who == 'fly':
            out['fly_spoken_tokens'] = len(all_words)
            out['fly_silent_turns'] = sum(not x for x in tokenized)/max(1,len(texts))
            out['fly_mean_turn_words'] = len(all_words)/max(1,len(texts))
            out['fly_identical_consecutive_turns'] = sum(bool(a) and a == b for a,b in zip(tokenized,tokenized[1:]))
        else:
            out['mother_name_prefix_share'] = sum(bool(ts) and ts[0] == child_name.lower() for ts in tokenized)/max(1,len(texts))
    out['fly_turns'] = sum(w == 'fly' for w,t in transcript)
    out['mother_turns'] = sum(w == 'mother' for w,t in transcript)
    out['caregiver_grounding_review'] = caregiver_grounding_review(transcript, child_name, last)
    out['mother_grounding_flags'] = len(out['caregiver_grounding_review']['flags'])
    return out


def learning_resources_prompt(plan, transcript):
    """Optional cross-subject shelf; relevance and recent content, never a clock.

    These are resources for the caregiver's judgment, not forced topic switches.
    A child can keep following an interest or request a familiar story again.
    """
    curriculum = plan.get('parenting', {}).get('interleaved_learning') or {}
    if not curriculum.get('enabled'):
        return ''
    tracks = curriculum.get('strands') or []
    descriptions = '; '.join(str(t['invitation']) for t in tracks)
    prompt = ('Open learning strands, available together: ' + descriptions +
              '. Follow the child and combine strands naturally. The developmental stage describes support, '
              'not a subject gate. No timed subject blocks or obligation to switch every turn. '
              'Revisit familiar ideas in new contexts; language variety serves meaning, not random word changes. ')
    shelf_path = curriculum.get('shelf_path')
    if not shelf_path or not Path(shelf_path).exists():
        return prompt
    shelf = json.loads(Path(shelf_path).read_text(encoding='utf8'))
    ignored = set('i you me my your we us our the a an and or is are was were be to for of in on at it that this with mama papa mengli meng say said hear here there'.split())
    interest_only = curriculum.get('excerpt_policy') == 'on_interest'
    terms = _content_terms if interest_only else lambda text: set(re.findall(r'[a-z]+', str(text).lower())) - ignored
    child = next((str(text) for who, text in reversed(transcript) if who != 'mother'), '')
    query = terms(child)
    recent = terms(' '.join(str(text) for who, text in list(transcript)[-8:] if who == 'mother'))
    ranked = sorted(shelf.get('entries', []), key=lambda e: (
        -len(query & terms(e['text'])), len(recent & terms(e['text'])) / max(1, len(terms(e['text']))), e['id']))
    if interest_only:
        # Full passages are much stronger demonstrations than a shelf catalogue.
        # Always pasting one from each track made unrelated fragments lead back
        # to the same story openings. Keep the shelf open without reciting it.
        catalogue = '; '.join(str(s['title']) for s in shelf.get('sources', []))
        prompt += ('Free conversation can stand on its own. Reading is available when it helps the exchange. '
                   + ('Available books: ' + catalogue + '. ' if catalogue else ''))
        matched = [e for e in ranked if query & terms(e['text'])]
        requested = bool(re.search(r'\b(read|story|stories|book|rhyme|poem|reading)\b', child, re.I))
        if matched or not requested:
            ranked = matched
        if not ranked:
            return prompt
    selected, used = [], set()
    for entry in ranked:
        if entry['strand'] in used:
            continue
        used.add(entry['strand'])
        selected.append(entry)
        if len(selected) == 3:
            break
    prompt += 'Optional reading material (not a checklist or a claim these events happened here): '
    prompt += ' '.join(f"[{e['id']}; {e['title']}; source PG{e['source']}] {e['text']} {e['use']}" for e in selected)
    return prompt


def caregiver_reply_flags(child_text, reply, child_name='Mengli', mother_name='Mama'):
    """Check literal quotes and explicit name claims, not the child's intentions."""
    normalized = lambda s: ' '.join(re.findall(r"[a-z]+(?:'[a-z]+)?", str(s).lower()))
    actual = normalized(child_text)
    contains = lambda name: f' {normalized(name)} ' in f' {actual} '
    flags = []
    if re.search(r'\bmy name\b\s*(?:is\s+|[—,:-]\s*)?' + re.escape(child_name) + r'\b', reply, re.I):
        flags.append('possible_speaker_identity_confusion')
    for quote in re.findall(r'\byou (?:said|say)\s+["“]([^"”]+)["”]', reply, re.I):
        quoted = normalized(quote)
        if quoted and f' {quoted} ' not in f' {actual} ':
            flags.append('possible_inaccurate_child_quote')
    child_claim = (re.search(r'\byou\b[^.!?]{0,30}\b(?:said|say|saying)\s+your (?:own )?name\b', reply, re.I)
                   or re.search(r'\byou (?:said|say|are saying)\s+' + re.escape(child_name) + r'\b', reply, re.I))
    if child_claim and not contains(child_name):
        flags.append('unsupported_complete_child_name_claim')
    if re.search(r'\byou (?:said|say)\s+my name\b', reply, re.I) and not contains(mother_name):
        flags.append('unsupported_adult_name_claim')
    return list(dict.fromkeys(flags))


def caregiver_grounding_review(transcript, child_name='Mengli', last=50):
    """Conservative text flags for Ling, not a semantic judge or an output filter."""
    flags = []
    latest_child = ''
    recent = list(transcript)[-last:]
    for index, (who, text) in enumerate(recent):
        if who != 'mother':
            latest_child = str(text)
            continue
        text = str(text)
        for kind in caregiver_reply_flags(latest_child, text, child_name):
            flags.append({'turn': index, 'kind': kind, 'child': latest_child, 'mother': text})
    return {'status': 'review_required' if flags else 'no_text_flags_semantics_unverified',
            'flags': flags, 'scope': f'last {last} transcript turns', 'automatic_rewrite': False}


def stage_readiness(curriculum, stage_id):
    """Check Ling's recorded evidence, never infer understanding from fluency or NLL."""
    stage = next((s for s in curriculum.get('stages', []) if s.get('id') == stage_id), None)
    if stage is None:
        return {'ready': False, 'missing': ['unknown stage']}
    missing = []
    for criterion in stage.get('criteria', []):
        evidence = criterion.get('evidence') or []
        verified = [e for e in evidence if e.get('reviewer') == 'Ling' and e.get('session')
                    and e.get('quote') and e.get('context') and e.get('result') == 'pass']
        sessions = {e['session'] for e in verified}
        contexts = {e['context'] for e in verified}
        if criterion.get('status') != 'passed' or len(sessions) < 3 or len(contexts) < 2:
            missing.append(criterion.get('id', criterion.get('text', 'criterion')))
    if not stage.get('criteria'):
        missing.append('no criteria recorded')
    return {'ready': not missing, 'missing': missing,
            'note': 'Recorded observations require Ling review; this check cannot judge meaning.'}
