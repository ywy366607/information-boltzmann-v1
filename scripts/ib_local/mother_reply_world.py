"""Continue an existing language world with contingent caregiver replies.

The ambient world's state remains the individual's actual carried state.  The
caregiver is a boundary process, not a source of synthetic reward.  Failure to
receive a reply is surfaced before the ambient world or transcript advances.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
import types

import numpy as np


RESPONSIVE_SYSTEM = (
    "You are a warm, attentive caregiver having a conversation with a very young child "
    "who is learning language. Respond in plain English, in one to three short sentences. "
    "Notice the child's present words or sounds and follow their apparent interest; "
    "gently recast a recognizable fragment and add a simple related idea or example. "
    "When meaning is unclear, acknowledge it and offer a tentative interpretation, "
    "without pretending to understand or forcing the child into a topic. "
    "If the child asks, try to answer their actual question; it is fine to admit uncertainty. "
    "Take turns and leave room for the child to lead, continue, change subject, or be quiet. "
    "A quiet turn means no words, not necessarily tiredness; offer a brief, gentle opening "
    "or calmly share something, without demanding a response. Do not repeatedly close "
    "the conversation or assume sleep just because earlier turns mentioned it; follow "
    "the child's current signals. Avoid formulas and repeated stock phrases. "
    "No emojis, stage directions, or role labels."
)

RESPONSIVE_SYSTEM_V2 = RESPONSIVE_SYSTEM + (
    " Speak naturally as a person, not in a fixed report format: you need not say "
    "'Mama hears' or quote a word before every response. Vary how you respond according "
    "to what the child actually says. If you have already repeated a phrase, use a "
    "different natural response. Do not keep extending your own previous topic when "
    "the child's new words suggest another."
)

RESPONSIVE_SYSTEM_V3 = RESPONSIVE_SYSTEM + (
    " Let the interaction be natural and responsive to the child's present words."
)
RESPONSIVE_TAIL_V3 = (
    "For this turn, respond directly and naturally to the child's present message. "
    "Avoid copying the wording or conversational formula of your recent replies. "
    "Follow the child's current topic, and allow an unfinished thought or quiet turn."
)


def _responsive_prompt(delegate, transcript=()):
    return delegate.base_system, getattr(delegate, 'responsive_tail', '')


def _terms(text):
    return set(re.findall(r'[a-z]{3,}', text.lower()))


class MotherReplyWorld:
    def __init__(self, base, mother, *, history=6, ambient_policy='reply'):
        if type(history) is not int or not 1 <= history <= 32:
            raise ValueError('mother context must have a finite positive bound')
        self.base, self.mother, self.history = base, mother, history
        if ambient_policy not in ('reply', 'quiet'):
            raise ValueError('unknown ambient navigation policy')
        self.ambient_policy = ambient_policy
        self.transcript = []

    def __getattr__(self, name):
        return getattr(self.__dict__['base'], name)

    def take(self, utterance, response_bytes, holdout_bytes):
        if response_bytes < 1 or holdout_bytes < 1:
            raise ValueError('positive response and holdout lengths required')
        child = re.sub(r'[\x00-\x1f\x7f]', ' ',
                       utterance.decode('utf-8', 'replace')).strip() if utterance else ''
        turn = self.transcript[-2*self.history:] + [('child', child or '<quiet_turn/>')]
        reply = self.mother.reply(turn)
        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError('mother returned no usable reply; world not advanced')
        reply = reply.strip()
        # Keep the same ambient/holdout transition used in the earlier
        # mother-world fork. Only the actually observed prefix is replaced.
        ambient_action = utterance if self.ambient_policy == 'reply' else b''
        ambient, heldout, info = self.base.take(ambient_action, response_bytes, holdout_bytes)
        ambient_match = int(info.get('match', 0))
        said = np.frombuffer((reply+'\n').encode('utf-8')[:response_bytes], dtype=np.uint8)
        response = np.concatenate((said, ambient[:response_bytes-len(said)]))
        self.transcript = (turn + [('mother', reply)])[-2*self.history:]
        return response, heldout, dict(info, source='mother', mother_bytes=len(said),
                                      mother_reply=reply, child_text=child,
                                      ambient_policy=self.ambient_policy, ambient_match=ambient_match,
                                      match=len(_terms(child)&_terms(reply)) if child else 0)

    def state_dict(self, *, copy_arrays=True):
        base_state = self.base.state_dict() if copy_arrays else self.base.checkpoint_state_dict()
        state = dict(base_state, mother_transcript=copy.deepcopy(self.transcript),
                     mother_history=self.history, mother_ambient_policy=self.ambient_policy)
        if hasattr(self.mother, 'state_dict'):
            state['mother_provider_state'] = self.mother.state_dict()
        return state

    def checkpoint_state_dict(self):
        return self.state_dict(copy_arrays=False)

    def load_state_dict(self, state):
        state = dict(state)
        history = state.pop('mother_history', self.history)
        transcript = state.pop('mother_transcript', [])
        policy = state.pop('mother_ambient_policy', 'quiet')
        provider = state.pop('mother_provider_state', None)
        if (history != self.history or policy != self.ambient_policy or
                not isinstance(transcript, list) or len(transcript) > 2*self.history):
            raise ValueError('incompatible mother-world state')
        self.base.load_state_dict(state)
        self.transcript = copy.deepcopy(transcript)
        if provider is not None:
            if not hasattr(self.mother, 'load_state_dict'):
                raise ValueError('mother provider state cannot be restored')
            self.mother.load_state_dict(provider)


class StrictToddlerMother:
    """Reuse the project's established mother with a bounded request cache."""
    def __init__(self, *, max_cache=16, shared_cache=None, shared_limit=4096, policy='legacy'):
        from scratch.claude_mother_world_20261105 import ToddlerMother
        if type(max_cache) is not int or not 1 <= max_cache <= 64:
            raise ValueError('bounded cache required')
        self.delegate = ToddlerMother()
        self.policy = 'legacy'
        self.policy_archives = []
        if policy != 'legacy':
            self.set_policy(policy)
        self.max_cache = max_cache
        self.shared_cache = None if shared_cache is None else str(shared_cache)
        if type(shared_limit) is not int or not 1 <= shared_limit <= 4096:
            raise ValueError('bounded shared cache required')
        self.shared_limit = shared_limit
        if self.shared_cache:
            with sqlite3.connect(self.shared_cache, timeout=30) as db:
                db.execute('CREATE TABLE IF NOT EXISTS replies '
                           '(id INTEGER PRIMARY KEY, key TEXT UNIQUE, reply TEXT NOT NULL)')

    def set_policy(self, policy):
        if policy not in ('legacy', 'responsive', 'responsive_v2', 'responsive_v3'):
            raise ValueError('unknown caregiver policy')
        if policy == self.policy:
            return
        if policy == 'legacy':
            raise ValueError('reverting caregiver policy requires an explicit legacy instance')
        # Delegate's own cache is keyed by transcript alone; archive it so a
        # changed prompt never silently receives a legacy response.
        self.policy_archives.append(dict(policy=self.policy, cache=copy.deepcopy(self.delegate.cache)))
        self.delegate.cache = {}
        self.delegate.base_system = {'responsive': RESPONSIVE_SYSTEM,
            'responsive_v2': RESPONSIVE_SYSTEM_V2, 'responsive_v3': RESPONSIVE_SYSTEM_V3}[policy]
        self.delegate.responsive_tail = RESPONSIVE_TAIL_V3 if policy == 'responsive_v3' else ''
        self.delegate.temperature = 0.7 if policy == 'responsive_v3' else 0.0
        self.delegate.get_prompt_parts = types.MethodType(_responsive_prompt, self.delegate)
        self.policy = policy

    def reply(self, transcript):
        # Include all inputs determining the prompt, not merely the last
        # utterance. The shared observer prevents identical histories in two
        # comparison arms from receiving different API samples.
        descriptor = {'transcript':transcript, 'system':self.delegate.base_system,
                      'model':self.delegate.model, 'temperature':self.delegate.temperature,
                      'max_tokens':self.delegate.max_tokens}
        if getattr(self.delegate, 'responsive_tail', ''):
            descriptor['tail'] = self.delegate.responsive_tail
        key = hashlib.sha256(json.dumps(descriptor,ensure_ascii=False,sort_keys=True,
                                         separators=(',',':')).encode('utf-8')).hexdigest()
        if self.shared_cache:
            with sqlite3.connect(self.shared_cache, timeout=30) as db:
                item = db.execute('SELECT reply FROM replies WHERE key=?',(key,)).fetchone()
            if item is not None:
                return item[0]
        response = self.delegate.reply(transcript)
        if not response:
            # The legacy provider caches an empty failed request. Permit a
            # later explicit retry without inventing a silent response.
            self.delegate.cache.pop(json.dumps(transcript), None)
        elif self.shared_cache:
            if len(response.encode('utf-8')) > 8192:
                raise ValueError('mother reply exceeds fixed cache bound')
            with sqlite3.connect(self.shared_cache, timeout=30) as db:
                db.execute('INSERT OR IGNORE INTO replies (key,reply) VALUES (?,?)',(key,response))
                # FIFO eviction is an audit cache only, not the body's memory.
                db.execute('DELETE FROM replies WHERE id IN '
                           '(SELECT id FROM replies ORDER BY id DESC LIMIT -1 OFFSET ?)',
                           (self.shared_limit,))
                response = db.execute('SELECT reply FROM replies WHERE key=?',(key,)).fetchone()[0]
        while len(self.delegate.cache) > self.max_cache:
            del self.delegate.cache[next(iter(self.delegate.cache))]
        return response

    def state_dict(self):
        return {'cache': copy.deepcopy(self.delegate.cache), 'calls': self.delegate.calls,
                'policy': self.policy, 'policy_archives': copy.deepcopy(self.policy_archives)}

    def load_state_dict(self, state):
        cache = state['cache']
        if not isinstance(cache, dict) or len(cache) > self.max_cache:
            raise ValueError('invalid bounded mother request cache')
        if state.get('policy', 'legacy') != self.policy:
            raise ValueError('caregiver policy differs from saved world')
        self.delegate.cache = copy.deepcopy(cache)
        self.delegate.calls = int(state['calls'])
        self.policy_archives = copy.deepcopy(state.get('policy_archives', []))
