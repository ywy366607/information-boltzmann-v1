"""Mother interaction world (the user's design: docs/information_boltzmann/FLY_MOTHER_WORLD_DESIGN_20261014.md).

The fly's speech goes to a mother (another mind that answers), and her reply comes back through the fly's senses at
the world's pace, so what the fly says changes its own future input. The stream never ends and almost never repeats.

The fly hears every word of the conversation through the same writer:
- her words while she speaks;
- its own words while it speaks (self-hearing, with cross-entropy on them: the predictive writer is the efference
  copy, so its own expected words carry little innovation).

Turn-taking:
- the fly's commit act is "speaking out";
- until that act is learned, a turn ends at an end-of-turn word or after `max_fly_words`;
- then the mother answers the transcript.

The mother is frozen, so she cannot collapse with the fly. A local small model on the CPU is for the first pipeline
tests only; the real mother is a fast API model. API keys are never read here: the SDK takes its standard
environment configuration.

The harness reads `monitor()`: repetition, type/token ratio and self-loops of the fly's speech; the diversity of the
mother's replies; and whether her reply depends on what the fly said (conditionality).
"""
from __future__ import annotations

import collections
import math
import random
import re


class LocalMother:
    """A local causal LM on the CPU continuing a 'Mother: / Child:' transcript; her line ends at the newline."""

    FRAME = 'A loving mother talks with her baby, who is just learning to talk. She answers simply and kindly.\n'

    def __init__(self, path='D:/ml_cache/huggingface/models/google--gemma-3-270m', max_new_tokens=40,
                 temperature=0.8, top_p=0.9, repetition_penalty=1.3, no_repeat_ngram_size=3, seed=0):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
        self.max_new_tokens, self.temperature, self.top_p = max_new_tokens, temperature, top_p
        self.repetition_penalty, self.no_repeat_ngram_size = repetition_penalty, no_repeat_ngram_size
        self.generator_seed = seed

    def reply(self, transcript):
        prompt = self.FRAME + ''.join(f'{"Mother" if who == "mother" else "Child"}: {text}\n' for who, text in transcript) + 'Mother:'
        ids = self.tok(prompt, return_tensors='pt')
        self.torch.manual_seed(self.generator_seed)
        self.generator_seed += 1
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=True,
                                      temperature=self.temperature, top_p=self.top_p,
                                      repetition_penalty=self.repetition_penalty,
                                      no_repeat_ngram_size=self.no_repeat_ngram_size)
        text = self.tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
        return text.split('\n')[0].strip() or '...'


def family_prompt(plan):
    """The family (data/harness/raising.json 'family'): always in the mouth's instructions, whatever the topic -- the
    words for herself and for the others are the first thing a child's world gives her.  Names in Latin letters only:
    her vocabulary is GPT-2, and Chinese characters reach her as broken bytes."""
    fam = plan.get('family') if isinstance(plan, dict) else None
    if not fam:
        return ''
    child = fam.get('child', {}).get('name', 'the child')
    who = [f"you are {fam['mother'].get('full', 'Mama')} (she calls you {fam['mother'].get('name', 'Mama')})"
           if fam.get('mother') else '',
           f"her father is {fam['father'].get('full', 'Papa')} (she calls him {fam['father'].get('name', 'Papa')})"
           if fam.get('father') else '']
    who += [f"{o['name']} is her {o.get('role', 'family')}" for o in fam.get('others', []) if o.get('name')]
    return (f"The child's name is {child}; " + '; '.join(w for w in who if w) + '. ' + fam.get('first_lessons', '') +
            ' ' + fam.get('language', '')).strip()


FAMILY_WORDS = (' Meng', ' Mama', ' Papa', ' Ling', ' Wu')     # GPT-2 first tokens of the family's names


class GeminiApiMother:
    """A fast Google Gemini model as the mother via Google AI Studio API.

    The API key is read from GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment or .env.
    Defaults to 'gemini-3.5-flash'. Uses thinkingBudget=0 for immediate conversational replies.
    """

    SYSTEM = ('You are a warm, patient mother talking with a very young child who is just learning to speak. The child '
              'often babbles. Answer in one or two short, simple sentences. Respond to what the child said when you '
              'can; ask simple questions, name things, sometimes start a new topic.')

    def __init__(self, model='gemini-3.5-flash', api_key=None, max_tokens=60, temperature=0.7, system=None, retries=5,
                 plan_path='data/harness/raising.json'):
        import os
        import requests
        try:
            import dotenv
            dotenv.load_dotenv()
        except ImportError:
            pass
        self.requests = requests
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError('GEMINI_API_KEY or GOOGLE_API_KEY must be set in environment or .env')
        raw_model = model or 'gemini-3.5-flash'
        self.model = raw_model.replace('models/', '')
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.base_system = system or self.SYSTEM
        self.plan_path = plan_path
        self.plan = {}
        self.plan_mtime = 0.0
        self.url = f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent'
        self._refresh_plan()

    def update_plan(self, plan):
        """Update plan directly in-memory from caller."""
        if isinstance(plan, dict):
            self.plan = plan

    def _refresh_plan(self):
        """Reload plan if plan_path exists and has been modified by the background agent."""
        import json, os
        if self.plan_path and os.path.exists(self.plan_path):
            try:
                mt = os.path.getmtime(self.plan_path)
                if mt != self.plan_mtime:
                    with open(self.plan_path, encoding='utf-8') as f:
                        self.plan = json.load(f)
                    self.plan_mtime = mt
            except Exception:
                pass

    def get_system_prompt(self):
        """Construct the prompt combining base persona with the agent's current topic and intention."""
        self._refresh_plan()
        m_cfg = self.plan.get('mother', {}) if isinstance(self.plan, dict) else {}
        p_cfg = self.plan.get('parenting', {}) if isinstance(self.plan, dict) else {}
        persona = m_cfg.get('persona') or self.base_system
        topic = p_cfg.get('current_topic')
        if not topic and p_cfg.get('topics'):
            topic = p_cfg['topics'][0]
        intent = p_cfg.get('intention')
        parts = [persona, family_prompt(self.plan)]
        if topic:
            parts.append(f"Current conversation topic: {topic}.")
        if intent:
            parts.append(f"Parenting intention for this phase: {intent}.")
        return ' '.join(parts)

    def reply(self, transcript):
        import time
        contents = []
        for who, text in transcript:
            role = 'model' if who == 'mother' else 'user'
            if contents and contents[-1]['role'] == role:
                contents[-1]['parts'][0]['text'] += ' ' + text
            else:
                contents.append({'role': role, 'parts': [{'text': text}]})
        if not contents or contents[0]['role'] != 'user':
            contents.insert(0, {'role': 'user', 'parts': [{'text': '...'}]})
        if contents[-1]['role'] != 'user':
            contents.append({'role': 'user', 'parts': [{'text': '...'}]})

        system_instruction = self.get_system_prompt()
        m_cfg = self.plan.get('mother', {}) if isinstance(self.plan, dict) else {}
        temperature = float(m_cfg.get('temperature', self.temperature))
        max_tokens = int(m_cfg.get('max_reply_tokens', self.max_tokens))

        payload = {
            'system_instruction': {'parts': [{'text': system_instruction}]},
            'contents': contents,
            'generationConfig': {
                'thinkingConfig': {'thinkingBudget': 0},
                'maxOutputTokens': max_tokens,
                'temperature': temperature,
            }
        }
        delay = 2.0
        for attempt in range(self.retries):
            try:
                r = self.requests.post(self.url, json=payload,
                                       headers={'x-goog-api-key': self.api_key, 'Connection': 'close'}, timeout=12)
                if r.status_code == 200:
                    data = r.json()
                    parts = data.get('candidates', [{}])[0].get('content', {}).get('parts', [])
                    text = ''.join(p.get('text', '') for p in parts).strip()
                    first_line = text.split('\n')[0].strip()
                    return first_line or '...'
                elif r.status_code in (429, 503):
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
            except Exception:
                time.sleep(1.0)
        return '...'


GeminiMother = GeminiApiMother


class ApiMother:
    """A fast API model as the mother.  The SDK reads its own standard environment configuration (the key is never
    read, printed or stored here).  `model` is the user's choice."""

    SYSTEM = ('You are a warm, patient mother talking with a very young child who is just learning to speak. The child '
              'often babbles. Answer in one or two short, simple sentences. Respond to what the child said when you '
              'can; ask simple questions, name things, sometimes start a new topic.')

    def __init__(self, model=None, max_tokens=60, system=None):
        import os
        try:
            import dotenv
            dotenv.load_dotenv()
        except ImportError:
            pass
        self.model = model or ('gemini-3.5-flash' if (os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')) else 'claude-3-5-haiku-latest')
        self.max_tokens = max_tokens
        self.system = system or self.SYSTEM
        if 'gemini' in self.model.lower():
            self._backend = GeminiApiMother(model=self.model, max_tokens=self.max_tokens, system=self.system)
        else:
            import anthropic
            self.client = anthropic.Anthropic()
            self._backend = None

    def update_plan(self, plan):
        if self._backend is not None and hasattr(self._backend, 'update_plan'):
            self._backend.update_plan(plan)

    def reply(self, transcript):
        if self._backend is not None:
            return self._backend.reply(transcript)
        messages = []
        for who, text in transcript:
            role = 'assistant' if who == 'mother' else 'user'
            if messages and messages[-1]['role'] == role:
                messages[-1]['content'] += ' ' + text
            else:
                messages.append({'role': role, 'content': text})
        if not messages or messages[0]['role'] != 'user':
            messages.insert(0, {'role': 'user', 'content': '...'})
        if messages[-1]['role'] != 'user':
            messages.append({'role': 'user', 'content': '...'})
        out = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=self.system,
                                          messages=messages)
        return ''.join(b.text for b in out.content if getattr(b, 'type', '') == 'text').strip() or '...'


class MailboxMother:
    """The mother as an agent session (another Claude Code / Antigravity window: no API key, the user's subscription).

    The trainer posts each finished fly turn as data/harness/mailbox/outbox/<id>.json (and waits for her):
      {id, time, child_said, transcript, monitor, plan}
    The agent answers by writing data/harness/mailbox/inbox/<id>.json, atomically (write <id>.tmp, then rename):
      {id, text}
    It may also edit data/harness/raising.json (the plan) and keep notes in data/harness/mother_notes.md.
    While she thinks the fly hears nothing new: the wait becomes the interval before her first word (live_window).
    An unanswered request expires after `timeout` seconds: she said nothing, and the floor goes back to the fly.
    Protocol and role: docs/information_boltzmann/MOTHER_AGENT.md."""

    asynchronous = True

    def __init__(self, root='data/harness/mailbox', timeout=900.0, plan='data/harness/raising.json'):
        import os
        self.root, self.timeout, self.plan = root, float(timeout), plan
        for d in ('outbox', 'inbox', 'archive'):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        self.sent = {}

    def request(self, transcript, monitor=None, silent_beats=0, fly_status='yielded', suggested_action='speak'):
        import json, os, time, uuid
        rid = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
        plan = None
        if os.path.exists(self.plan):
            try:
                plan = json.load(open(self.plan, encoding='utf-8'))
            except Exception:
                plan = None
        body = {'id': rid, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'child_said': transcript[-1][1] if transcript and transcript[-1][0] == 'fly' else '',
                'transcript': [{'who': w, 'text': t} for w, t in transcript], 'monitor': monitor, 'plan': plan,
                'silent_beats': int(silent_beats), 'fly_status': str(fly_status),
                'suggested_action': str(suggested_action)}
        tmp = os.path.join(self.root, 'outbox', rid + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(body, f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(self.root, 'outbox', rid + '.json'))
        self.sent[rid] = time.time()
        return rid

    def poll(self, rid):
        """Her answer if it has arrived (dict with action & text), None while she is still thinking."""
        import json, os, time
        reply = os.path.join(self.root, 'inbox', rid + '.json')
        if os.path.exists(reply):
            try:
                with open(reply, encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    action = data.get('action', 'speak')
                    text = str(data.get('text', '')).strip()
                else:
                    action, text = 'speak', str(data).strip()
            except Exception:
                return None                                # half written: look again later
            for d, name in (('inbox', reply), ('outbox', os.path.join(self.root, 'outbox', rid + '.json'))):
                if os.path.exists(name):
                    try:
                        os.replace(name, os.path.join(self.root, 'archive', f'{rid}.{d}.json'))
                    except Exception:
                        pass
            self.sent.pop(rid, None)
            return {'action': action, 'text': text or (text if action == 'wait' else '...')}
        if time.time() - self.sent.get(rid, time.time()) > self.timeout:
            out = os.path.join(self.root, 'outbox', rid + '.json')
            if os.path.exists(out):
                try:
                    os.replace(out, os.path.join(self.root, 'archive', f'{rid}.expired.json'))
                except Exception:
                    pass
            self.sent.pop(rid, None)
            return {'action': 'expired', 'text': ''}
        return None


class MotherWorld:
    """The conversation as the fly's text stream (GPT-2 token ids, the fly's vocabulary).

    next_word(): the id the fly hears next. During the mother's turn, her reply word by word. During the fly's turn,
    the fly's own spoken word: the caller passes it to speak(), and it is heard next. Ids are GPT-2 BPE tokens.
    """

    def __init__(self, mother, gpt2_tokenizer, history_turns=20, max_fly_words=24, end_of_turn=('\n', '.', '?', '!'),
                 opening='Hello, little one.', seed=0, ticks_per_beat=8, pause_beats_before_reply=1,
                 silent_beats_threshold=3, call_prompt=None, max_wait_beats=16, beat_duration_s=1.0,
                 plan_path='data/harness/raising.json'):
        self.mother, self.tok = mother, gpt2_tokenizer
        self.pending = None                        # the request an asynchronous mother is thinking about
        self.history_turns, self.max_fly_words = history_turns, max_fly_words
        self.end_ids = {i for s in end_of_turn for i in gpt2_tokenizer.encode(s)}
        self.family_ids = {gpt2_tokenizer.encode(w)[0] for w in FAMILY_WORDS}
        self.transcript = []                       # [(who, text)]
        self.queue = collections.deque()           # the mother's words still to be heard
        self.fly_words = []                        # the fly's current turn (ids)
        self.turn = 'mother'
        self.rng = random.Random(seed)
        self.stats = collections.Counter()
        self.fly_turns, self.mother_turns = [], []

        # Beat-based rhythm and dark-room defense
        self.ticks_per_beat = int(ticks_per_beat)
        self.pause_beats_before_reply = int(pause_beats_before_reply)
        self.silent_beats_threshold = int(silent_beats_threshold)
        self.silent_beats = 0
        self.call_prompt = call_prompt
        self.max_wait_beats = int(max_wait_beats)
        self.beat_duration_s = float(beat_duration_s)
        self.plan_path = plan_path
        self.mother_turn_fresh = False             # True when mother turn has just started (for pause timing)

        self._say_mother(opening)

    # -- the stream -------------------------------------------------------------------------------------------------
    def _say_mother(self, text):
        clean_text = str(text).strip() or self.get_call_prompt()
        self.transcript.append(('mother', clean_text))
        self.mother_turns.append(clean_text)
        self.queue.extend(self.tok.encode(' ' + clean_text + '\n'))
        self.mother_heard = []                     # her words the child has heard so far this turn
        self.turn = 'mother'
        self.mother_turn_fresh = True
        self.stats['mother_turns'] += 1

    def get_call_prompt(self):
        if self.call_prompt:
            return self.call_prompt
        child = 'Mengli'
        if isinstance(getattr(self.mother, 'plan', None), dict):
            child = self.mother.plan.get('family', {}).get('child', {}).get('name', 'Mengli')
        elif self.plan_path:
            import json, os
            if os.path.exists(self.plan_path):
                try:
                    data = json.load(open(self.plan_path, encoding='utf-8'))
                    child = data.get('family', {}).get('child', {}).get('name', 'Mengli')
                except Exception:
                    pass
        return f"{child}?"

    def next_word(self):
        """The id heard next, and whose it is ('mother', or 'fly' for its own last word)."""
        if self.turn == 'mother':
            word = self.queue.popleft()
            self.mother_heard.append(word)
            self.mother_turn_fresh = False
            if not self.queue:
                self.turn = 'fly'
            return word, 'mother'
        return (self.fly_words[-1] if self.fly_words else None), 'fly'

    def speak(self, word_id):
        """The fly's committed word during its turn; returns True when the turn ended (the mother then answers)."""
        self.fly_words.append(int(word_id))
        self.stats['fly_words'] += 1
        self.silent_beats = 0                      # Fly spoke: reset silent beats counter
        if int(word_id) in self.end_ids or len(self.fly_words) >= self.max_fly_words:
            self.stats['end_word' if int(word_id) in self.end_ids else 'end_breath'] += 1
            self._end_fly_turn()
            return True
        return False

    def interrupt(self):
        """The child spoke during the mother's turn: the rest of her turn is dropped; the transcript keeps what was
        heard, marked as cut off, so her next reply sees that she was interrupted and by what."""
        heard = self.tok.decode(getattr(self, 'mother_heard', [])).strip()
        self.transcript[-1] = ('mother', (heard + ' --') if heard else '--')
        self.mother_turns[-1] = self.transcript[-1][1]
        self.queue.clear()
        self.turn = 'fly'
        self.mother_turn_fresh = False
        self.stats['interruptions'] += 1

    def call(self):
        """She calls the child by name (the floor is hers)."""
        self._say_mother(self.get_call_prompt())
        self.silent_beats = 0
        self.stats['mother_calls'] += 1

    def yield_turn(self):
        """v2: the fly did not commit at an arrival in its turn -- silence yields the floor, and the mother answers
        what was said (nothing at all is a silent turn: the dark-room signal)."""
        self.stats['fly_yields'] += 1
        self.stats['end_yield'] += 1
        if not self.fly_words:
            self.silent_beats += 1
            self.stats['fly_silent_turns'] += 1
        else:
            self.silent_beats = 0
        self._end_fly_turn()

    def _end_fly_turn(self):
        text = self.tok.decode(self.fly_words).strip()
        had_words = bool(self.fly_words)
        self.transcript.append(('fly', text or '...'))
        self.fly_turns.append(list(self.fly_words))
        self.fly_words = []
        self.stats['fly_turns'] += 1
        history = self.transcript[-self.history_turns:]

        # Dark room defense: if synchronous mother and persistently silent, auto-call
        if not getattr(self.mother, 'asynchronous', False) and not had_words and self.silent_beats >= self.silent_beats_threshold:
            call_text = self.get_call_prompt()
            self._say_mother(call_text)
            self.silent_beats = 0
            self.stats['mother_calls'] += 1
            return

        if getattr(self.mother, 'asynchronous', False):
            suggested = 'call' if (not had_words and self.silent_beats >= self.silent_beats_threshold) else 'speak'
            self.pending = self.mother.request(history, self.monitor(), silent_beats=self.silent_beats,
                                               fly_status='silent' if not had_words else 'yielded',
                                               suggested_action=suggested)
            self.turn = 'waiting'
        else:
            self._say_mother(self.mother.reply(history))

    def wait_for_mother(self, poll_interval_s=0.5):
        """An asynchronous mother is deciding.  The world's clock is beats, not her thinking time (network and agent
        latency never reach the child): the simulation simply waits, and the child lives only what she chose.
        Returns the beats of silence before what follows:
        - 'speak' / 'call': her pause (pause_beats_before_reply), then her words;
        - 'wait': one beat of silence, and the floor goes back to the child (who may speak, or stay silent again);
        - expired (she never answered): like 'wait'."""
        import time
        while self.turn == 'waiting':
            resp = self.mother.poll(self.pending)
            if resp is None:
                time.sleep(poll_interval_s)
                continue
            self.pending = None
            act = resp.get('action', 'speak') if isinstance(resp, dict) else 'speak'
            text = (resp.get('text', '') if isinstance(resp, dict) else str(resp)).strip()
            if act in ('wait', 'expired') or (act == 'speak' and not text):
                self.stats['mother_waits' if act == 'wait' else 'mother_silent'] += 1
                self.turn = 'fly'
                return 1
            if act == 'call':
                self._say_mother(text or self.get_call_prompt())
                self.silent_beats = 0
                self.stats['mother_calls'] += 1
            else:
                self._say_mother(text)
            return int(self.pause_beats_before_reply)
        return 0

    # -- harness ----------------------------------------------------------------------------------------------------
    def monitor(self, last=50):
        """Collapse and dark-room signals over the last turns."""
        fly = [w for t in self.fly_turns[-last:] for w in t]
        mom = ' '.join(self.mother_turns[-last:]).lower().split()

        def ttr(xs):
            return len(set(xs)) / max(1, len(xs))

        def repeat_bigrams(xs):
            bg = list(zip(xs, xs[1:]))
            return 1.0 - len(set(bg)) / max(1, len(bg))
        recent = self.fly_turns[-last:]
        loops = sum(1 for a, b in zip(recent, recent[1:]) if a == b)
        return {'fly_turns': len(self.fly_turns), 'fly_ttr': round(ttr(fly), 3),
                'fly_repeat_bigrams': round(repeat_bigrams(fly), 3), 'fly_identical_consecutive_turns': loops,
                'fly_mean_turn_words': round(len(fly) / max(1, len(self.fly_turns[-last:])), 2),
                'fly_silent_turns': round(sum(1 for t in recent if not t) / max(1, len(recent)), 3),
                'mother_ttr': round(ttr(mom), 3), 'mother_repeat_bigrams': round(repeat_bigrams(mom), 3),
                # the words for herself and the others: how often she says them, and how often she hears them
                'fly_family_words': sum(1 for w in fly if w in self.family_ids),
                'mother_family_words': sum(t.count(n.strip().lower()) for t in [' '.join(mom)]
                                           for n in FAMILY_WORDS),
                'silent_beats': self.silent_beats,
                'mother_calls': self.stats['mother_calls'],
                'mother_waits': self.stats['mother_waits'],
                # how her turns ended: breath (the length limit), an end word, or her own yield; learning to stop
                # shows as the yield share rising
                'fly_turn_ends': {k: self.stats[k] for k in ('end_breath', 'end_word', 'end_yield')},
                'interruptions': self.stats['interruptions']}


def live_window(learner, world, n, rng, pace_min, pace_mean, temperature=1.0, ticks_per_second=2.0,
                max_wait_ticks=128, speak_act=False, interrupt=False):
    """The live pass of one window (no gradient, on a copy of the state): the conversation as it happens.

    Each of the n words arrives after its interval (the world's pace: a window mean log-uniform in [pace_min,
    pace_mean], memoryless intervals), advanced exactly as forward_window does (the same window options, segments of
    stream_segment_ticks, held input, the expectation updated per segment).
    - In the mother's turn the next word is hers.
    - In the fly's turn the next word is the fly's own: sampled from its read standing just before the arrival (the
      word it says), heard next (self-hearing); world.speak() ends the turn at an end-of-turn word, and the mother answers.

    Returns (targets, forced, speakers):
    - targets: the n words heard after the learner's previous token;
    - forced: {'pace', 'intervals'}, for learner.forced_world, so the training pass relives this window;
    - speakers: 'mother' or 'fly' per word.
    Timing is beat-based: network jitter is decoupled from biophysics. Wait/pause intervals are discrete multiples of
    world.ticks_per_beat.

    speak_act (v2, motor acts attached): in the fly's turn its muscles decide at each arrival. A commit on any tick of
    the interval (probability 1 - prod(1 - h_t)) says the word sampled from the read, heard next; no commit yields the
    floor and the mother answers (an asynchronous mother's wait or synchronous conversational pause is added to this
    interval, after the decision ticks).
    After a yield the mother answers after her pause, or (an agent) waits a beat: the beat passes in silence and the
    child decides again over it.  interrupt: in the mother's turn the child's commit cuts her off (MotherWorld.interrupt)
    and its word is heard; no commit keeps listening.
    forced then also carries 'decisions': (word index, first tick in the window, ticks, 1 commit / 0 silent) for every
    decision, for the act's credit in the training pass (fly_active_inference.speaking_value).
    """
    import math
    import torch
    from . import fly_bptt_learning as fl
    m = learner.model
    with torch.no_grad():
        s = fl.FlyPhysicalState(**{k: (tuple(t.detach().clone() for t in v) if k == 'ring' else v.detach().clone())
                                   for k, v in learner.state.state_dict().items()})
        opts = dict(base_rates=m.get_decay_rates(), thresholds=m.get_thresholds(),
                    conductance_gains=m.get_conductance_gains(), alif_params=m.get_alif_params(),
                    stp_params=m.get_stp_params())
        if getattr(m, 'dopamine_sensitivity', None) is not None:
            opts['dopamine_s_class'] = fl.dopamine_class_sensitivity(m, s.h)
        pace = (float(math.exp(rng.uniform(math.log(pace_min), math.log(pace_mean)))) if 0 < pace_min < pace_mean
                else float(pace_mean))
        seg = int(getattr(learner, 'stream_segment_ticks', 16) or 16)
        token = int(learner.previous_token)
        acts = bool(speak_act) and getattr(m, 'ai_mode', None) == 'motor'
        ai_g = learner.ai_g.detach().clone() if acts and getattr(learner, 'ai_g', None) is not None else None
        acts = acts and ai_g is not None          # the muscles' state comes from the first trained window
        targets, intervals, speakers, decisions = [], [], [], []
        tpb = int(getattr(world, 'ticks_per_beat', 8))
        pause = int(getattr(world, 'pause_beats_before_reply', 1)) * tpb

        def advance(k, start=0):
            # ticks start .. start + k - 1 of the word's interval, in the training pass's segments (same packets)
            nonlocal s, ai_g
            done, latent, committed, said_read = start, None, False, None
            while done < start + k:
                ticks = min(seg - done % seg, start + k - done)
                extra_kw = (dict(ai_g=ai_g, ai_tape=[], eye_forced=getattr(learner, 'eye_forced', None),
                                 read_every=True) if acts else {})
                out = fl.advance_fly_token_stream(
                    m, s, torch.tensor([token], device=s.h.device), [ticks], ticks=ticks,
                    hold_input=learner.hold_input, read_after=learner.read_lag or None,
                    writer_baseline_clock=learner.writer_baseline_clock, fresh=done == 0,
                    **extra_kw, **opts)
                s, latent = out[0], out[1]
                if acts:
                    ai_g = out[-1]
                    tape = extra_kw['ai_tape']
                    if not committed and any(tape):
                        # the word is said from the read standing at the first commit tick
                        first = tape.index(True)
                        said_read = out[2][first:first + 1]
                    committed = committed or any(tape)
                done += ticks
            return latent, committed, said_read

        def say(read):
            logits = m.decoder(m.read_norm(read)).float()[0] / max(temperature, 1e-6)
            return int(torch.multinomial(torch.softmax(logits, -1), 1))

        tick0 = 0                                  # ticks of the window before this word's interval
        for i in range(n):
            word, who = None, None
            extra = 0
            if world.turn == 'waiting':
                extra = min(int(max_wait_ticks), world.wait_for_mother() * tpb)
            elif world.turn == 'mother' and getattr(world, 'mother_turn_fresh', False):
                extra = min(int(max_wait_ticks), pause)
            k = int(rng.geometric(1.0 / pace)) + extra
            latent, committed, said_read = advance(k)
            if world.turn == 'mother':
                if acts and interrupt:
                    decisions.append((i, tick0, k, int(committed)))
                if acts and interrupt and committed:
                    world.interrupt()                       # she is cut off; the child's word is heard
                    word, who = say(said_read if said_read is not None else latent), 'fly'
                    world.speak(word)
                else:
                    word, who = world.next_word()
            elif acts:
                decisions.append((i, tick0, k, int(committed)))
                tries = 0
                while not committed and word is None:
                    # silence yields the floor: she answers after her pause, or waits a beat and it comes back
                    world.yield_turn()
                    gap = (min(int(max_wait_ticks), world.wait_for_mother() * tpb) if world.turn == 'waiting'
                           else pause if world.turn == 'mother' else 0)
                    if world.turn != 'mother' and tries >= 8:
                        world.call()                        # a long silence: she calls the child by name
                        gap = pause
                    if world.turn == 'mother':
                        if gap > 0:
                            advance(gap, k)
                            k += gap
                        word, who = world.next_word()
                    else:
                        gap = max(gap, tpb)                 # a beat of silence, and the child decides again
                        _, committed, said_read = advance(gap, k)
                        decisions.append((i, tick0 + k, gap, int(committed)))
                        k += gap
                        tries += 1
                if word is None:
                    word, who = say(said_read if said_read is not None else latent), 'fly'
                    world.speak(word)
            else:
                word, who = say(latent), 'fly'
                world.speak(word)
            targets.append(int(word)); intervals.append(k); speakers.append(who)
            token = int(word)
            tick0 += k
    forced = {'pace': pace, 'intervals': intervals}
    if acts:
        forced.update(decisions=decisions, speakers=list(speakers))
    return targets, forced, speakers
