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


def world_prompt(path='data/harness/world.json', limit=40, query=None):
    """The world as it has been established (the mother's memory, data/harness/world.json): facts the heart agent
    consolidated from the conversation -- the cat called Katz who sleeps on the chair, Papa's drum, Aunt Ling's Sunday
    visits -- so the same word keeps pointing at the same thing and the world stops drifting.  Append-mostly and placed
    right after the family, so the cached prompt prefix survives.  Format: {"facts": [{"fact": "...", "since": "..."}]}."""
    import json, os
    if not os.path.exists(path):
        return ''
    try:
        facts = [f['fact'] if isinstance(f, dict) else str(f) for f in json.load(open(path, encoding='utf-8')).get('facts', [])]
    except Exception:
        return ''
    facts = [f.strip() for f in facts if str(f).strip()][:limit]
    if query is not None:
        # Retrieve a relevant memory instead of priming every reply with every home fact.
        from .fly_parenting_harness import _content_terms
        terms = _content_terms(query)
        ranked = [(len(terms & _content_terms(f)), f) for f in facts]
        facts = [f for score, f in sorted(ranked, key=lambda x: -x[0]) if score > 0][:1]
    if not facts:
        return ''
    return ('Private home memory (keep it consistent, but do not recite this list; mention a fact only when the child\'s '
            'latest words or the current activity make it relevant; use at most one fact per reply; do not repeat '
            'the same fact or the child\'s name just because it appeared in the last turn; rotate to a different object, '
            'action, person, or sensation when the child has already heard it): ' + ' '.join(
        f if f.endswith('.') else f + '.' for f in facts))


def activity_prompt(plan):
    """Only the current page reaches the mother; future pages are kept for Ling."""
    activity = (plan.get('parenting') or {}).get('activity') or {}
    pages = activity.get('pages') or []
    page = int(activity.get('page', 0))
    if not pages:
        return ('Current activity: ' + str(activity.get('kind', 'conversation')) + '. ' +
                str(activity.get('instruction', ''))).strip() if activity else ''
    if page < 0 or page >= len(pages):
        return 'The shared reading has ended. Follow the child into conversation or another activity.'
    return ("Shared reading, current page only: " + str(pages[page]) + " " +
            str(activity.get('instruction', 'Let the child listen, interrupt, or request another reading.')) +
            " Continue from the recent conversation; if this page was already shared, respond or explore it instead of restarting it every turn. Story events belong to the story, not to permanent home memory.")


def family_prompt(plan):
    """The family (data/harness/raising.json 'family'): always in the mouth's instructions, whatever the topic -- the
    words for herself and for the others are the first thing a child's world gives her.  Names in Latin letters only:
    her vocabulary is GPT-2, and Chinese characters reach her as broken bytes."""
    fam = plan.get('family') if isinstance(plan, dict) else None
    if not fam:
        return ''
    child = fam.get('child', {}).get('name', 'the child')
    mother = fam.get('mother') or {}
    who = [f"Assistant / adult speaker: {mother.get('full', mother.get('name', 'Mama'))}, called {mother.get('name', 'Mama')}. "
           f"User / child speaker: {child}. These are different people. "
           f"In your reply, I/my refers to the adult and you/your to {child}. "
           f"In the child's speech, I/my refers to {child} and you/your to the listener.",
           f"her father is {fam['father'].get('full', 'Papa')} (she calls him {fam['father'].get('name', 'Papa')})"
           if fam.get('father') else '']
    who += [f"{o['name']} is her {o.get('role', 'family')}" for o in fam.get('others', []) if o.get('name')]
    return (' '.join(w for w in who if w) + '. ' + fam.get('first_lessons', '') +
            ' ' + fam.get('language', '')).strip()


FAMILY_WORDS = (' Meng', ' Mama', ' Papa', ' Ling', ' Wu')     # GPT-2 first tokens of the family's names


def sanitize_mother_reply(text: str, child_name: str = 'Mengli') -> str:
    """Strip repetitive leading vocatives (e.g. 'Mengli, ', 'Mengli! ') from mother replies so she does not prepend
    the child's name to every turn, while preserving intentional calls like 'Mengli?'."""
    if not text:
        return ''
    line = str(text).strip()
    if not line:
        return ''
    if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
        line = line[1:-1].strip()
    import re
    names = [re.escape(child_name)]
    if child_name.lower() != 'mengli':
        names.append('Mengli')
    joined = '|'.join(names)
    pattern = rf'^(?:{joined})[\s,:!\-\.]+\s*'
    m = re.match(pattern, line, flags=re.IGNORECASE)
    if m:
        rest = line[m.end():].strip()
        if len(rest) > 1 and any(c.isalnum() for c in rest):
            line = rest[0].upper() + rest[1:]
    return line


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

    def get_system_prompt(self, transcript=()):
        """Construct the prompt combining base persona with the agent's current topic and intention."""
        stable, volatile = self.get_prompt_parts(transcript)
        return ' '.join([stable, volatile]).strip()

    def get_prompt_parts(self, transcript=()):
        """Separate caregiver instructions from the literal child utterances.

        Both parts belong to system instructions. Never put care notes in a user
        message: that message is the child's actual speech, including partial words.
        """
        self._refresh_plan()
        m_cfg = self.plan.get('mother', {}) if isinstance(self.plan, dict) else {}
        p_cfg = self.plan.get('parenting', {}) if isinstance(self.plan, dict) else {}
        stable = ' '.join([m_cfg.get('persona') or self.base_system, family_prompt(self.plan),
            "The conversation contains literal child utterances and your past replies. "
            "Quote the child's actual words accurately; a fragment or unclear sound may remain unclear. "
            "Acknowledge without assigning an unexpressed intention, action or understanding. "
            "The event <quiet_turn/> means no spoken words; respond to the quiet event, not to its notation. "
            "Past adult replies can be mistaken; follow the family identities above. "
            "Reply with your next words to the child, without caregiver notes or role labels."])
        topic = p_cfg.get('current_topic') or ((p_cfg.get('topics') or [None])[0])
        intent = p_cfg.get('intention')
        latest_child = next((str(text) for who, text in reversed(transcript) if who != 'mother'), '')
        from .fly_parenting_harness import learning_resources_prompt
        volatile = [world_prompt(query=' '.join([latest_child, str(topic or '')])), activity_prompt(self.plan),
                    learning_resources_prompt(self.plan, transcript)]
        if topic:
            volatile.append(f"Current conversation topic: {topic}.")
        if intent:
            volatile.append(f"Parenting intention for this phase: {intent}.")
        if p_cfg.get('context_correction'):
            volatile.append('Caregiver correction of earlier adult statements: ' + str(p_cfg['context_correction']))
        return stable, ' '.join(v for v in volatile if v)

    def get_chat_messages(self, transcript):
        """Provider-neutral chat roles; care notes never overwrite child speech.

        To maximize provider prefix caching hit rate (SiliconFlow / DeepSeek),
        messages[0] contains ONLY the invariant stable persona and family prompt.
        Dynamic, turn-specific volatile guidance (topic, activities, book resources) is placed
        at the tail as a secondary system message.
        """
        stable, volatile = self.get_prompt_parts(transcript)
        child = self.plan.get('family', {}).get('child', {}).get('name', 'Mengli')
        messages = [{'role': 'system', 'content': stable}]
        for who, text in transcript:
            role = 'assistant' if who == 'mother' else 'user'
            cleaned = (sanitize_mother_reply(text, child_name=child)
                       if who == 'mother' and self.plan.get('mother', {}).get('strip_leading_vocative', False)
                       else str(text))
            if len(messages) > 1 and messages[-1]['role'] == role:
                messages[-1]['content'] += '\n' + cleaned
            else:
                messages.append({'role': role, 'content': cleaned})
        if messages[-1]['role'] != 'user':
            messages.append({'role': 'user', 'content': '<quiet_turn/>'})
        if volatile:
            messages.append({'role': 'system', 'content': f"[Caregiver guidance: {volatile}]"})
        return messages

    def reply(self, transcript):
        import time
        self._refresh_plan()
        child = 'Mengli'
        if isinstance(getattr(self, 'plan', None), dict):
            child = self.plan.get('family', {}).get('child', {}).get('name', child)
        contents = []
        for who, text in transcript:
            role = 'model' if who == 'mother' else 'user'
            cleaned_text = (sanitize_mother_reply(text, child_name=child)
                            if who == 'mother' and self.plan.get('mother', {}).get('strip_leading_vocative', False) else text)
            if contents and contents[-1]['role'] == role:
                contents[-1]['parts'][0]['text'] += ' ' + cleaned_text
            else:
                contents.append({'role': role, 'parts': [{'text': cleaned_text}]})
        if not contents or contents[0]['role'] != 'user':
            contents.insert(0, {'role': 'user', 'parts': [{'text': '<quiet_turn/>'}]})
        if contents[-1]['role'] != 'user':
            contents.append({'role': 'user', 'parts': [{'text': '<quiet_turn/>'}]})

        system_instruction = self.get_system_prompt(transcript)
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
                    self.last_raw_reply = text
                    reply = (sanitize_mother_reply(text, child_name=child)
                             if m_cfg.get('strip_leading_vocative', False) else text.strip())
                    return reply or '...'
                elif r.status_code in (429, 503):
                    self.last_status = r.status_code
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                else:
                    self.last_status = r.status_code
            except Exception as error:
                self.last_status = type(error).__name__
                time.sleep(1.0)
        # no answer (quota, rate limit, network): she stays silent -- never a made-up word the child would hear
        self.failures = getattr(self, 'failures', 0) + 1
        print(f'Gemini mouth: no answer after {self.retries} tries (last: {getattr(self, "last_status", None)}), '
              f'{self.failures} so far: she stays silent', flush=True)
        return ''


GeminiMother = GeminiApiMother


class DeepSeekMother(GeminiApiMother):
    """The mouth through DeepSeek's OpenAI-format API (https://api-docs.deepseek.com/): the same family prompt, plan
    reload and silence on failure as the Gemini mouth.  The key is read from DEEPSEEK_API_KEY in the environment or .env
    (the user configures it; it is never printed or stored here).  Default model 'deepseek-flash', thinking off."""

    def __init__(self, model='deepseek-flash', api_key=None, max_tokens=60, temperature=0.7, system=None, retries=5,
                 plan_path='data/harness/raising.json'):
        import os
        import requests
        try:
            import dotenv
            dotenv.load_dotenv()
        except ImportError:
            pass
        self.requests = requests
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY') or self._user_environment('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError('DEEPSEEK_API_KEY must be set in the environment or .env')
        self.model = model or 'deepseek-flash'
        self.max_tokens, self.temperature, self.retries = max_tokens, temperature, retries
        self.base_system = system or self.SYSTEM
        self.plan_path, self.plan, self.plan_mtime = plan_path, {}, 0.0
        self.url = 'https://api.deepseek.com/chat/completions'
        self.thinking_field = True                 # dropped once if the API refuses it
        self.thinking_off = {'thinking': {'type': 'disabled'}}
        self.name = 'DeepSeek'
        self._refresh_plan()

    @staticmethod
    def _user_environment(name):
        """A variable set in the Windows user environment after this process started (HKCU\\Environment)."""
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
                return winreg.QueryValueEx(key, name)[0] or None
        except Exception:
            return None

    def reply(self, transcript):
        import time
        self._refresh_plan()
        child = 'Mengli'
        if isinstance(getattr(self, 'plan', None), dict):
            child = self.plan.get('family', {}).get('child', {}).get('name', child)
        messages = self.get_chat_messages(transcript)
        self.last_prompt_messages = messages  # Inspectable request evidence, without credentials.
        self.last_grounding_rejections = []
        self.last_delivered_reply = ''
        latest_child = next((str(text) for who, text in reversed(transcript) if who != 'mother'), '')
        m_cfg = self.plan.get('mother', {}) if isinstance(self.plan, dict) else {}
        delay = 2.0
        for attempt in range(self.retries):
            payload = {'model': self.model, 'messages': messages, 'stream': False,
                       'max_tokens': max(int(m_cfg.get('max_reply_tokens', self.max_tokens)), getattr(self, 'min_tokens', 0)),
                       'temperature': float(m_cfg.get('temperature', self.temperature))}
            if self.thinking_field:
                payload.update(self.thinking_off)
            try:
                r = self.requests.post(self.url, json=payload, timeout=20,
                                       headers={'Authorization': f'Bearer {self.api_key}',
                                                'Content-Type': 'application/json'})
                if r.status_code == 200:
                    data = r.json()
                    usage = data.get('usage') or {}
                    self.usage = getattr(self, 'usage', {'calls': 0, 'cache_hit': 0, 'cache_miss': 0, 'output': 0})
                    self.usage['calls'] += 1
                    hit = usage.get('prompt_cache_hit_tokens')
                    if hit is None:     # OpenAI format (SiliconFlow): cached tokens inside prompt_tokens_details
                        hit = int((usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0)
                        miss = int(usage.get('prompt_tokens') or 0) - hit
                    else:
                        miss = int(usage.get('prompt_cache_miss_tokens') or 0)
                    self.usage['cache_hit'] += int(hit or 0)
                    self.usage['cache_miss'] += int(miss or 0)
                    self.usage['output'] += int(usage.get('completion_tokens') or 0)
                    text = (data.get('choices', [{}])[0].get('message', {}).get('content') or '').strip()
                    self.last_raw_reply = text
                    cleaned = (sanitize_mother_reply(text, child_name=child)
                               if m_cfg.get('strip_leading_vocative', False) else text.strip())
                    from .fly_parenting_harness import caregiver_reply_flags
                    mother_name = self.plan.get('family', {}).get('mother', {}).get('name', 'Mama')
                    flags = caregiver_reply_flags(latest_child, cleaned, child, mother_name)
                    if flags:
                        self.last_grounding_rejections.append({'reply': cleaned, 'flags': flags})
                        self.usage['grounding_rejections'] = self.usage.get('grounding_rejections', 0) + 1
                        self.last_status = 'caregiver_grounding_rejected'
                        # Retry from the same real dialogue; never fabricate child speech
                        # or quietly rewrite a wrong adult sentence into a new one.
                        import json
                        if len(self.last_grounding_rejections) >= 2:
                            break
                        messages.append({'role': 'system', 'content':
                            'Caregiver grounding check: the draft was rejected for ' + ', '.join(flags) +
                            '. Latest child utterance, verbatim: ' + json.dumps(latest_child) +
                            '. Produce a fresh short reply. Keep partial sounds uncertain; '
                            'a full name claim requires the actual full name. Adult and child identities remain as above.'})
                        continue
                    self.last_delivered_reply = cleaned
                    return cleaned
                self.last_status = r.status_code
                if r.status_code == 400 and self.thinking_field:
                    self.thinking_field = False    # an API that does not know the field: ask again without it
                    continue
                if r.status_code in (429, 500, 502, 503):
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                else:
                    break
            except Exception as error:
                self.last_status = type(error).__name__
                time.sleep(1.0)
        self.failures = getattr(self, 'failures', 0) + 1
        print(f'{getattr(self, "name", "DeepSeek")} mouth: no answer after {self.retries} tries (last: {getattr(self, "last_status", None)}), '
              f'{self.failures} so far: she stays silent', flush=True)
        return ''


class SiliconFlowMother(DeepSeekMother):
    """The mouth through SiliconFlow's OpenAI-format API (https://api-docs.siliconflow.cn/docs/api/chat-completions-post):
    the same family prompt (stable prefix: persona + family; the changing notes ride in the last turn), plan reload and
    silence on failure.  The key is read from the user environment variable 'SiliconFlow' (or SILICONFLOW_API_KEY);
    it is never printed or stored here.  Default model zai-org/GLM-5.3-Flash, thinking off."""

    def __init__(self, model='zai-org/GLM-5.3-Flash', **kwargs):
        import os
        key = (kwargs.pop('api_key', None) or os.environ.get('SiliconFlow') or os.environ.get('SILICONFLOW_API_KEY')
               or self._user_environment('SiliconFlow') or self._user_environment('SILICONFLOW_API_KEY'))
        if not key:
            raise ValueError("The SiliconFlow key must be set (user environment variable 'SiliconFlow')")
        super().__init__(model=model, api_key=key, **kwargs)
        self.url = 'https://api.siliconflow.cn/v1/chat/completions'
        # GLM-5.3-Flash still reasons a little with thinking disabled (enable_thinking is ignored): the token budget
        # leaves room for it, so the reply itself is never cut to ''
        self.thinking_off = {'thinking': {'type': 'disabled'}}
        self.min_tokens = 400
        self.name = 'SiliconFlow'


class FallbackMother:
    """One mother, two mouths (2026-10-15): the primary answers; when it stays silent after its own retries (an outage,
    not a choice), the fallback answers that turn.  DeepSeek answers in ~1.1 s with ~20 tokens; GLM-5.3-Flash on
    SiliconFlow reasons first whatever the thinking switch (50-240 hidden tokens, 3-19 s), so it is the reserve."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback
        self.name = f'{primary.name} (fallback {fallback.name})'
        self.fallback_turns = 0

    def __getattr__(self, name):          # plan, usage, failures, ...: the primary's
        return getattr(self.__dict__['primary'], name)

    def update_plan(self, plan):
        for m in (self.primary, self.fallback):
            if hasattr(m, 'update_plan'):
                m.update_plan(plan)

    def reply(self, transcript):
        text = self.primary.reply(transcript)
        if text:
            return text
        self.fallback_turns += 1
        return self.fallback.reply(transcript)


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
                 plan_path='data/harness/raising.json', contingent=False):
        self.mother, self.tok = mother, gpt2_tokenizer
        # contingent mother: silence is met with silence (one beat; her call by name after silent_beats_threshold
        # silent turns), her answers follow the child's voice -- speaking is what buys a reply (Goldstein et al. 2003)
        self.contingent = bool(contingent)
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
        self.fly_pause_ticks = 0                   # elapsed silence inside an unfinished utterance
        self.call_prompt = call_prompt
        self.max_wait_beats = int(max_wait_beats)
        self.beat_duration_s = float(beat_duration_s)
        self.plan_path = plan_path
        self.mother_turn_fresh = False             # True when mother turn has just started (for pause timing)
        self.caregiver_context_revision = None

        self.mother_heard = []
        self.mother_source = 'mother_opening'
        if opening is not None:
            self._say_mother(opening, source='mother_opening')

    # -- the stream -------------------------------------------------------------------------------------------------
    def _say_mother(self, text, source='mother_reply'):
        child = 'Mengli'
        if isinstance(getattr(self.mother, 'plan', None), dict):
            child = self.mother.plan.get('family', {}).get('child', {}).get('name', child)
        clean_text = str(text).strip()
        plan = self.mother.plan if isinstance(getattr(self.mother, 'plan', None), dict) else {}
        if source == 'mother_reply' and plan.get('mother', {}).get('strip_leading_vocative', False):
            clean_text = sanitize_mother_reply(clean_text, child_name=child)
        clean_text = clean_text or self.get_call_prompt()
        self.transcript.append(('mother', clean_text))
        self.mother_turns.append(clean_text)
        self.queue.extend(self.tok.encode(' ' + clean_text + '\n'))
        self.mother_heard = []                     # her words the child has heard so far this turn
        self.turn = 'mother'
        self.mother_turn_fresh = True
        self.mother_source = source
        self.reply_has_speech = source == 'mother_reply' and bool(getattr(self, '_reply_has_speech', False))
        self.stats['mother_turns'] += 1

    def state_dict(self):
        """Conversation state at the same completed window as the brain checkpoint."""
        return {'format': 'mother-world-v2', 'turn': self.turn, 'queue': list(self.queue),
                'fly_words': list(self.fly_words), 'mother_heard': list(self.mother_heard),
                'transcript': list(self.transcript), 'history_anchor': getattr(self, 'history_anchor', 0),
                'caregiver_context_revision': self.caregiver_context_revision,
                'fly_turns': list(self.fly_turns), 'mother_turns': list(self.mother_turns),
                'stats': dict(self.stats), 'silent_beats': self.silent_beats,
                'fly_pause_ticks': self.fly_pause_ticks,
                'mother_turn_fresh': self.mother_turn_fresh, 'mother_source': self.mother_source,
                'reply_has_speech': getattr(self, 'reply_has_speech', False),
                'pending_reply_has_speech': getattr(self, '_reply_has_speech', False),
                'pending': self.pending, 'rng': self.rng.getstate(),
                'mother_usage': dict(getattr(self.mother, 'usage', {}) or {})}

    def load_state_dict(self, saved):
        if saved.get('format') != 'mother-world-v2':
            raise ValueError('A complete mother-world-v2 state is required')
        if saved['turn'] not in ('mother', 'fly', 'waiting'):
            raise ValueError('Invalid mother-world turn')
        if saved['turn'] == 'mother' and not saved['queue']:
            raise ValueError('A mother turn must retain its unread words')
        self.turn = saved['turn']
        self.queue = collections.deque(saved['queue'])
        for key in ('fly_words', 'mother_heard', 'fly_turns', 'mother_turns'):
            setattr(self, key, list(saved[key]))
        self.transcript = [tuple(x) for x in saved['transcript']]
        self.reply_has_speech = bool(saved.get('reply_has_speech', False))
        self._reply_has_speech = bool(saved.get('pending_reply_has_speech', False))
        self.fly_pause_ticks = int(saved.get('fly_pause_ticks', 0))
        self.stats = collections.Counter(saved['stats'])
        self.caregiver_context_revision = saved.get('caregiver_context_revision')
        for key in ('history_anchor', 'silent_beats', 'mother_turn_fresh', 'mother_source', 'pending'):
            setattr(self, key, saved[key])
        # JSON sidecars turn tuples into lists; random.Random requires nested tuples.
        def as_tuple(value):
            return tuple(as_tuple(x) for x in value) if isinstance(value, (tuple, list)) else value
        self.rng.setstate(as_tuple(saved['rng']))
        if hasattr(self.mother, 'usage'):
            self.mother.usage.update(saved.get('mother_usage', {}))

    def get_call_prompt(self):
        if self.call_prompt:
            return self.call_prompt
        child = 'the little one'
        if isinstance(getattr(self.mother, 'plan', None), dict):
            child = self.mother.plan.get('family', {}).get('child', {}).get('name', child)
        elif self.plan_path:
            import json, os
            if os.path.exists(self.plan_path):
                try:
                    data = json.load(open(self.plan_path, encoding='utf-8'))
                    child = data.get('family', {}).get('child', {}).get('name', child)
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
        if self.fly_pause_ticks:
            self.stats['fly_resumed_pauses'] += 1
        self.fly_pause_ticks = 0
        self.stats['fly_words'] += 1
        if self.tok.decode(self.fly_words).strip():
            self.silent_beats = 0                  # A boundary token alone is not lexical speech.
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
        """Noncontingent presence: keep the world available during quiet listening.

        A generated quiet-time contribution remains a mother_call: neither
        novelty credit nor a speech-caused PAM reward is earned by waiting.
        The fixed reassurance is only a fallback when the mouth is unavailable.
        """
        if hasattr(self.mother, '_refresh_plan'):
            self.mother._refresh_plan()
        plan = getattr(self.mother, 'plan', {}) or {}
        text = None
        if (plan.get('mother', {}).get('spontaneous_when_quiet', False)
                and not getattr(self.mother, 'asynchronous', False)):
            history = self.cached_history()
            if not history or history[-1] != ('fly', '<quiet_turn/>'):
                history = history + [('fly', '<quiet_turn/>')]
            text = self.mother.reply(history)
            if text and str(text).strip():
                self.stats['mother_spontaneous_calls'] += 1
        self._say_mother(text or self.get_call_prompt(), source='mother_call')
        self.silent_beats = 0
        self.stats['mother_calls'] += 1

    def yield_turn(self):
        """Explicitly hand over the floor; a missing motor commit uses pause_speech instead."""
        self.stats['fly_yields'] += 1
        self.stats['end_yield'] += 1
        self._end_fly_turn()

    def pause_speech(self, ticks):
        """Retain partial speech until the listener's existing silence patience expires.

        This is a world turn-taking timeout, not an internal memory constant or
        a minimum utterance length. Before speech starts, keep the existing
        contingent waiting/call protocol. Elapsed ticks have already run through
        the brain and the same no-commit decision is retained for policy credit.
        """
        ticks = int(ticks)
        if ticks <= 0 or self.turn != 'fly':
            raise ValueError('Speech pauses require positive elapsed ticks in the child turn')
        if not self.fly_words:
            self.yield_turn()
            return
        self.fly_pause_ticks += ticks
        self.stats['fly_pause_events'] += 1
        self.stats['fly_pause_ticks'] += ticks
        if self.fly_pause_ticks >= self.speech_pause_limit():
            self.stats['end_pause'] += 1
            self._end_fly_turn()

    def speech_pause_limit(self):
        """Listener patience in physical ticks, from the existing world rhythm."""
        return max(1, self.ticks_per_beat * self.silent_beats_threshold)

    def cached_history(self):
        """The conversation the mouth sees, append-only from an anchor so that consecutive requests share their prefix
        (API prefix caches hit only on an exact prefix; a sliding window shifts it every turn and misses on all of it).
        When it grows past 2 x history_turns, the anchor jumps once to keep the last history_turns // 2 + 1 turns."""
        if hasattr(self.mother, '_refresh_plan'):
            self.mother._refresh_plan()
        m_plan = getattr(self.mother, 'plan', {})
        revision = m_plan.get('mother', {}).get('context_revision') if isinstance(m_plan, dict) else None
        anchor = getattr(self, 'history_anchor', 0)
        if revision and revision != self.caregiver_context_revision:
            # Ling can repair the teacher's few-shot context once. The child's
            # complete experience, queue and physical/learning states stay intact.
            anchor = max(0, len(self.transcript) - 1)
            self.history_anchor = anchor
            self.caregiver_context_revision = revision
            self.stats['caregiver_context_repairs'] += 1
        if len(self.transcript) - anchor > 2 * self.history_turns:
            anchor = len(self.transcript) - (self.history_turns // 2 + 1)
            self.history_anchor = anchor
            self.stats['history_rebases'] += 1
        # Silence is an event, not a lexical child utterance. Keep it in the
        # teacher's chronology so separate adult contributions cannot collapse
        # into one uninterrupted assistant monologue. The actual token tape
        # distinguishes silence from a genuinely spoken ellipsis. Full history
        # and the child's stream remain unchanged.
        recent = self.transcript[anchor:]
        cursor = len(self.fly_turns) - sum(w == 'fly' for w, _ in recent)
        if cursor < 0:
            return self.transcript[anchor:]        # Legacy alignment is unknown; do not invent it.
        history = []
        for who, text in recent:
            if who == 'fly':
                words = self.fly_turns[cursor]
                cursor += 1
                if not self.tok.decode(words).strip():
                    if not history or history[-1] != ('fly', '<quiet_turn/>'):
                        history.append(('fly', '<quiet_turn/>'))
                    continue
            history.append((who, text))
        return history

    def turn_observation(self, index):
        """Disambiguate visible silence placeholders using the saved token tape."""
        who, text = self.transcript[index]
        if who != 'fly':
            return {}
        # New turns are near the tail. Align backwards rather than rescanning
        # the organism's complete lifetime on every mouth request and log row.
        cursor = len(self.fly_turns) - sum(w == 'fly' for w, _ in self.transcript[index:])
        if cursor < 0:
            return {'turn_kind': 'legacy_alignment_unknown'}
        ids = list(self.fly_turns[cursor])
        raw = self.tok.decode(ids)
        return {'turn_kind': ('quiet_yield' if not ids else 'boundary_only' if not raw.strip() else 'utterance'),
                'token_ids': ids, 'spoken_text': raw.strip()}

    def _end_fly_turn(self):
        text = self.tok.decode(self.fly_words).strip()
        had_words = bool(text)
        if had_words:
            self.silent_beats = 0
        else:
            self.silent_beats += 1
            self.stats['fly_silent_turns'] += 1
        self._reply_has_speech = had_words
        self.transcript.append(('fly', text or '...'))
        self.fly_turns.append(list(self.fly_words))
        self.fly_words = []
        self.fly_pause_ticks = 0
        self.stats['fly_turns'] += 1
        history = self.cached_history()

        if self.contingent and not had_words:
            if self.silent_beats >= self.silent_beats_threshold:
                self.call()                        # a long silence: she calls the child by name
            else:
                self.turn = 'fly'                  # she waits with the child: one beat of silence
                self.stats['mother_waits_silence'] += 1
            return

        # Dark room defense: if synchronous mother and persistently silent, auto-call
        if not getattr(self.mother, 'asynchronous', False) and not had_words and self.silent_beats >= self.silent_beats_threshold:
            call_text = self.get_call_prompt()
            self._say_mother(call_text, source='mother_call')
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
            text = self.mother.reply(history)
            if text and str(text).strip():
                self._say_mother(text)
            else:                                  # no answer: silence, and the floor goes back to the child
                self.turn = 'fly'
                self.stats['mother_silent'] += 1

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
                self._say_mother(text or self.get_call_prompt(), source='mother_call')
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
        fly_counts = collections.Counter(fly)
        mother_counts = collections.Counter(mom)
        fly_top = fly_counts.most_common(1)[0] if fly_counts else ('', 0)
        mother_top = mother_counts.most_common(1)[0] if mother_counts else ('', 0)
        from .fly_parenting_harness import conversation_health
        # Preserve cumulative runtime counters while fixing the lexical health definitions.
        summary = conversation_health(self.transcript, last=last)
        return {'fly_turns': len(self.fly_turns), 'fly_ttr': round(ttr(fly), 3),
                'fly_repeat_bigrams': round(repeat_bigrams(fly), 3), 'fly_identical_consecutive_turns': loops,
                'fly_mean_turn_words': round(len(fly) / max(1, len(self.fly_turns[-last:])), 2),
                'fly_silent_turns': round(sum(1 for t in recent if not t) / max(1, len(recent)), 3),
                'fly_top_token': fly_top[0], 'fly_top_token_share': round(fly_top[1] / max(1, len(fly)), 3),
                'mother_ttr': round(ttr(mom), 3), 'mother_repeat_bigrams': round(repeat_bigrams(mom), 3),
                'mother_top_token': mother_top[0], 'mother_top_token_share': round(mother_top[1] / max(1, len(mom)), 3),
                # the words for herself and the others: how often she says them, and how often she hears them
                'fly_family_words': sum(1 for w in fly if w in self.family_ids),
                'mother_family_words': sum(t.count(n.strip().lower()) for t in [' '.join(mom)]
                                           for n in FAMILY_WORDS),
                'silent_beats': self.silent_beats,
                'mother_spontaneous_calls': self.stats['mother_spontaneous_calls'],
                'mother_calls': self.stats['mother_calls'],
                'mother_waits': self.stats['mother_waits'],
                # how her turns ended: breath (the length limit), an end word, or her own yield; learning to stop
                # shows as the yield share rising
                'fly_turn_ends': {k: self.stats[k] for k in ('end_breath', 'end_word', 'end_yield', 'end_pause')},
                'mother_waits_silence': self.stats['mother_waits_silence'],
                'interruptions': self.stats['interruptions'],
                **summary,
                **({'mouth_calls': u['calls'], 'mouth_cache_hit_tokens': u['cache_hit'],
                    'mouth_grounding_rejections': u.get('grounding_rejections', 0),
                    'mouth_cache_miss_tokens': u['cache_miss'], 'mouth_output_tokens': u['output'],
                    'mouth_cache_hit_rate': round(u['cache_hit'] / max(1, u['cache_hit'] + u['cache_miss']), 3)}
                   if (u := getattr(self.mother, 'usage', None)) else {})}


def sensory_segments(ticks, segment_ticks, quiet_ranges=()):
    """Split a real interval at checkpoint and external-silence boundaries.

    A quiet range withdraws text and voice together, while physical time continues.
    Ranges are disjoint half-open intervals in ticks relative to the word's interval.
    """
    if ticks < 1 or segment_ticks < 1:
        raise ValueError('Interval and segment lengths must be positive')
    last = 0
    edges = {0, ticks, *range(segment_ticks, ticks, segment_ticks)}
    for start, end in quiet_ranges:
        if not (last <= start < end <= ticks):
            raise ValueError('Quiet ranges must be ordered, disjoint and inside the interval')
        edges.update((start, end))
        last = end
    edges = sorted(edges)
    q = 0
    for start, end in zip(edges, edges[1:]):
        while q < len(quiet_ranges) and quiet_ranges[q][1] <= start:
            q += 1
        on = not (q < len(quiet_ranges) and quiet_ranges[q][0] <= start < quiet_ranges[q][1])
        yield start, end - start, on


def replay_ce(learner, state0, previous_token, targets, intervals, voices=None, quiet_ranges=None, ai_g0=None,
              rewards=None):
    """Per-word CE of a window's targets, relived (no gradient) from its starting state with the CURRENT weights: the
    same intervals and segments as the training pass, eyes as the learner has them (open when forced).  After the
    update this is CE_after; the training pass's scores are CE_before."""
    import torch
    import torch.nn.functional as F
    from . import fly_bptt_learning as fl
    m = learner.model
    seg = int(getattr(learner, 'stream_segment_ticks', 16) or 16)
    out = []
    with torch.no_grad():
        s = fl.FlyPhysicalState(**{k: (tuple(t.detach().clone() for t in v) if k == 'ring' else v.detach().clone())
                                   for k, v in state0.state_dict().items()})
        opts = dict(base_rates=m.get_decay_rates(), thresholds=m.get_thresholds(),
                    conductance_gains=m.get_conductance_gains(), alif_params=m.get_alif_params(),
                    stp_params=m.get_stp_params())
        if getattr(m, 'dopamine_sensitivity', None) is not None:
            opts['dopamine_s_class'] = fl.dopamine_class_sensitivity(m, s.h)
        token = int(previous_token)
        ai_g = ai_g0.detach().clone() if ai_g0 is not None else None
        for j, (target, k) in enumerate(zip(targets, intervals)):
            done, latent = 0, None
            voice = bool(voices[j]) if voices is not None else False     # the mother's voice, as in the training pass
            quiet = quiet_ranges[j] if quiet_ranges is not None else ()
            token_tensor = torch.tensor([token], device=s.h.device)
            for done, ticks, sensory_on in sensory_segments(k, seg, quiet):
                ai_opts = (dict(ai_g=ai_g, ai_tape=[False] * ticks,
                                eye_forced=getattr(learner, 'eye_forced', None)) if ai_g is not None else {})
                result = fl.advance_fly_token_stream(
                    m, s, token_tensor, [ticks], ticks=ticks, sensory_on=sensory_on,
                    hold_input=learner.hold_input, read_after=learner.read_lag or None,
                    writer_baseline_clock=learner.writer_baseline_clock, fresh=done == 0, voice=voice,
                    reward_ticks=fl.reward_ticks_left(m, rewards is not None and rewards[j], done),
                    **ai_opts, **opts)
                s, latent = result[:2]
                if ai_g is not None:
                    ai_g = result[-1]
            logits = m.decoder(m.read_norm(latent)).float()
            out.append(float(F.cross_entropy(logits, torch.tensor([int(target)], device=logits.device))))
            token = int(target)
    return out


def accumulate_speaking_credit(learner, rec, gains, horizon_ticks=64, beta=0.98, tau_ticks=0.0):
    """Carry decisions across BPTT boundaries until their configured physical-time horizon ends.

    Only normal replies contribute learning progress. Automatic name calls are monitored separately. Stored muscle
    features are detached eligibility for the policy readout; this does not claim delayed gradients into the brain.
    """
    import numpy as np
    import torch
    if horizon_ticks <= 0:
        raise ValueError('Speaking credit horizon must be positive')
    clock = int(getattr(learner, 'speak_credit_clock', 0))
    arrival = clock + np.cumsum(rec['intervals'])
    now = int(arrival[-1]) if len(arrival) else clock
    pending = list(getattr(learner, 'speak_pending_eligibility', []))
    for i, t0, n, d in rec['decisions']:
        if n > 0:
            pending.append({'g': rec['g'][t0:t0 + n].detach().clone(), 'decision': (i, 0, n, d),
                            'end': clock + t0 + n, 'return': 0.0,
                            'after_speech': i > 0 and rec['speakers'][i - 1] == 'fly',
                            'horizon': int(horizon_ticks), 'tau': float(tau_ticks)})
    sources = rec.get('sources')
    eligible = [(sources[j] == 'mother_reply' if sources is not None else who == 'mother')
                for j, who in enumerate(rec['speakers'])]
    ig = [float(g) if on else 0.0 for g, on in zip(gains, eligible)]
    ready, waiting = [], []
    for item in pending:
        for g, t in zip(ig, arrival):
            delay = int(t) - item['end']
            # The word arrives after the decision, even if it shares its terminal physical tick.
            if 0 <= delay <= item['horizon']:
                item['return'] += g * (float(np.exp(-delay / item['tau'])) if item['tau'] > 0 else 1.0)
        (ready if now >= item['end'] + item['horizon'] else waiting).append(item)
    learner.speak_credit_clock = now
    learner.speak_pending_eligibility = waiting
    R = [item['return'] for item in ready]
    baseline = getattr(learner, 'speak_epistemic_baseline', None)
    advantages = [r - baseline for r in R] if baseline is not None else [0.0] * len(R)
    if ready:
        learner.speak_epistemic_baseline = (float(np.mean(R)) if baseline is None else
                                           beta * baseline + (1 - beta) * float(np.mean(R)))
        frames, decisions, offset = [], [], 0
        for item in ready:
            i, _, n, d = item['decision']
            frames.append(item['g'])
            decisions.append((i, offset, n, d))
            offset += n
        learner.pending_speak_credit = (torch.cat(frames, 0), decisions, advantages)
    said = [item['return'] for item in ready if item['decision'][-1]]
    silent = [item['return'] for item in ready if not item['decision'][-1]]
    pure = [item['return'] for item in ready if not item['decision'][-1] and not item['after_speech']]
    yielded = [item['return'] for item in ready if not item['decision'][-1] and item['after_speech']]
    mom = [g for g, on in zip(gains, eligible) if on]
    mean = lambda xs: float(np.mean(xs)) if xs else None
    return {'ig_mother_mean': mean(mom), 'speak_return_ig': mean(R),
            'speak_said_return': mean(said), 'speak_silent_return': mean(silent),
            'pure_silence_return': mean(pure), 'yield_after_speech_return': mean(yielded),
            'pure_silence_n': len(pure), 'yield_after_speech_n': len(yielded),
            'credit_matured_decisions': len(ready), 'credit_pending_decisions': len(waiting),
            'credit_clock_ticks': now, 'credit_horizon_ticks': int(horizon_ticks)}


def accumulate_speaking_trace(learner, rec, gains, tau_ticks, beta=0.98, differential=False, rate_beta=0.98,
                              event_clock=False):
    """The speaking credit as a continuous eligibility trace, O(1) and without a horizon (--mother-credit-trace).

    Each decision adds the gradient of its own log-probability (on the act readout, at the decision) to a trace E that
    decays as exp(-dt / tau) on the physical clock, across windows; each later normal-reply word with learning
    progress g adds g E to the credit, and each decision subtracts b x its gradient (b, the running mean return per
    decision).  This is exactly sum_d (R_d - b) grad log p_d with R_d = sum_words g exp(-delay / tau) over every later
    word, never cut off.  The credit is applied in the next window as a linear surrogate (fly_bptt_learning).
    differential=True (--mother-credit-differential): the average-reward return of an infinite stream (no episodes):
    every tick after a decision also costs r_bar, her own long-run learning progress per tick, so
    R_d = sum_words g e^{-delay/tau} - r_bar int_end^inf e^{-(t-end)/tau} dt.  Deferring an ordinary stream by the
    ticks her own word takes then costs only those ticks (not ~22% of all that follows, as under the plain discount),
    and a stretch that teaches nothing (a dark room of name calls) is below her average: worth leaving.  r_bar is the
    eligible progress per physical tick, a slow average over windows (rate_beta), updated after this window is valued.
    Per-decision returns are also tracked, for the said / silent statistics only, until their weight drops below 1e-3.
    Review (2026-10-15): the differential charge is the same for every decision ending at the same time, so the
    baseline absorbs it: it does NOT remove the bias against speaking.  The bias is the clock itself: her own word
    delays the mother's later words by its ticks, so every later word is discounted more after speaking (22% at 8
    ticks / tau 32, 71% at 32 ticks / tau 26).
    event_clock=True (--mother-credit-event-clock): discount on the conversation's own clock, the count of eligible
    mother words since the decision (tau_ticks is then in words).  Her own word no longer shifts the mother's later
    words; what she says is valued by what the following replies teach, not by when they arrive.
    """
    import numpy as np
    import torch
    from .fly_active_inference import motor_logits
    if tau_ticks <= 0:
        raise ValueError('The continuous speaking trace needs --mother-credit-tau > 0')
    m = learner.model
    params = {name: p for name in ('ai_w', 'ai_b', 'ai_habit')
              if isinstance(p := getattr(m, name, None), torch.nn.Parameter) and p.requires_grad}
    tr = getattr(learner, 'speak_trace', None) or {}
    E = {k: tr['E'][k].to(p) if 'E' in tr and k in tr['E'] else torch.zeros_like(p) for k, p in params.items()}
    G = {k: torch.zeros_like(p) for k, p in params.items()}
    t_trace = float(tr.get('t', 0.0))
    b = tr.get('baseline')
    r_bar = float(tr.get('r_bar', 0.0)) if differential else 0.0
    clock = int(getattr(learner, 'speak_credit_clock', 0))
    arrival = clock + np.cumsum(rec['intervals'])
    now = int(arrival[-1]) if len(arrival) else clock
    sources = rec.get('sources')
    eligible = [(sources[j] == 'mother_reply' if sources is not None else who == 'mother')
                for j, who in enumerate(rec['speakers'])]
    events = []                                   # (time, 0 decision / 1 word, payload): a word at a decision's own
    for i, t0, n, d in rec['decisions']:          # last tick counts for it (delay 0), as in the horizon form
        if n > 0:
            events.append((clock + t0 + n, 0, (i, t0, n, d)))
    for g, t, on in zip(gains, arrival, eligible):
        if on:
            events.append((int(t), 1, float(g)))
    events.sort(key=lambda e: (e[0], e[1]))
    if event_clock:
        # time = eligible mother words so far (carried): a decision takes the index of the next word, so the first
        # word after it has delay 0, the next delay 1, ...
        count = int(tr.get('events', 0))
        remapped = []
        for t, kind, x in events:
            remapped.append((count, kind, x))
            if kind == 1:
                count += 1
        events = remapped
        now = count
    diag = list(tr.get('diag', []))
    said_now = []
    for t, kind, x in events:
        decay = float(np.exp(-(t - t_trace) / tau_ticks))
        if r_bar and t > t_trace:
            # the stretch (t_trace, t] costs r_bar per tick, weighted by the standing trace
            for k in E:
                G[k].sub_(r_bar * tau_ticks * (1.0 - decay) * E[k])
            for item in diag:
                a = max(t_trace, item['end'])
                if t > a:
                    item['return'] -= r_bar * tau_ticks * (float(np.exp(-(a - item['end']) / tau_ticks))
                                                           - float(np.exp(-(t - item['end']) / tau_ticks)))
        for k in E:
            E[k].mul_(decay)
        t_trace = float(t)
        if kind == 0:
            i, t0, n, d = x
            with torch.enable_grad():
                hz = torch.sigmoid(motor_logits(m, rec['g'][t0:t0 + n].detach())[:, 1]).clamp(1e-6, 1 - 1e-6)
                log_quiet = torch.log1p(-hz).sum()
                logp = torch.log((-torch.expm1(log_quiet)).clamp_min(1e-12)) if d else log_quiet
                grads = torch.autograd.grad(logp, list(params.values()), allow_unused=True)
            for k, gr in zip(params, grads):
                if gr is not None:
                    E[k].add_(gr)
                    if b is not None:
                        G[k].sub_(b * gr)
            diag.append({'end': int(t), 'return': 0.0, 'said': bool(d),
                         'after_speech': i > 0 and rec['speakers'][i - 1] == 'fly'})
            said_now.append(bool(d))
        else:
            for k in E:
                G[k].add_(x * E[k])
            for item in diag:
                item['return'] += x * float(np.exp(-(t - item['end']) / tau_ticks))
    life = tau_ticks * np.log(1000.0)
    ready = [it for it in diag if now - it['end'] > life]
    diag = [it for it in diag if now - it['end'] <= life]
    R = [it['return'] for it in ready]
    if R:
        b = float(np.mean(R)) if b is None else beta * b + (1 - beta) * float(np.mean(R))
    if differential:
        ticks = float(sum(rec['intervals']))
        rate = float(sum(g for g, on in zip(gains, eligible) if on)) / max(ticks, 1.0)
        r_bar = rate if 'r_bar' not in tr else rate_beta * float(tr['r_bar']) + (1 - rate_beta) * rate
    learner.speak_trace = {'E': {k: v.detach() for k, v in E.items()}, 't': t_trace, 'baseline': b, 'diag': diag,
                           **({'r_bar': r_bar} if differential else {}), **({'events': now} if event_clock else {})}
    learner.speak_credit_clock = int(arrival[-1]) if len(arrival) else clock
    learner.speak_epistemic_baseline = b
    learner.pending_speak_credit = {'trace': {k: v.detach() for k, v in G.items()}, 'decisions': said_now}
    mean = lambda xs: float(np.mean(xs)) if xs else None
    mom = [g for g, on in zip(gains, eligible) if on]
    return {'ig_mother_mean': mean(mom), 'speak_return_ig': mean(R),
            'speak_said_return': mean([it['return'] for it in ready if it['said']]),
            'speak_silent_return': mean([it['return'] for it in ready if not it['said']]),
            'pure_silence_return': mean([it['return'] for it in ready if not it['said'] and not it['after_speech']]),
            'yield_after_speech_return': mean([it['return'] for it in ready if not it['said'] and it['after_speech']]),
            'pure_silence_n': sum(1 for it in ready if not it['said'] and not it['after_speech']),
            'yield_after_speech_n': sum(1 for it in ready if not it['said'] and it['after_speech']),
            'credit_matured_decisions': len(ready), 'credit_pending_decisions': len(diag),
            'credit_clock_ticks': now, 'credit_trace_tau': float(tau_ticks),
            'credit_trace_norm': float(sum(v.norm() ** 2 for v in E.values()) ** 0.5),
            **({'credit_r_bar': r_bar} if differential else {})}


def epistemic_credit(learner, state0, previous_token, targets, horizon_ticks=64, beta=0.98, tau_ticks=0.0,
                     trace=False, differential=False, event_clock=False):
    """Replay matched real sensory intervals after learning, then advance delayed speaking eligibility (trace=True:
    the continuous O(1) trace, no horizon)."""
    rec = getattr(learner, 'speak_record', None)
    if rec is None:
        return {}
    learner.speak_record = None
    after = replay_ce(learner, state0, previous_token, targets, rec['intervals'], rec.get('voices'),
                      rec.get('quiet_ranges'), rec.get('ai_g0'), rec.get('rewards'))
    before = rec['ce_before'].tolist()
    gains = [b - a for b, a in zip(before, after)]
    if trace:
        return accumulate_speaking_trace(learner, rec, gains, tau_ticks, beta=beta, differential=differential,
                                         event_clock=event_clock)
    return accumulate_speaking_credit(learner, rec, gains,
                                      horizon_ticks=horizon_ticks, beta=beta, tau_ticks=tau_ticks)


def _silence_split(R, decisions, speakers):
    import numpy as np
    after_speech, pure = [], []
    for r, (i, t0, n, d) in zip(R, decisions):
        if d:
            continue
        (after_speech if i > 0 and speakers[i - 1] == 'fly' else pure).append(r)
    return {'yield_after_speech_return': float(np.mean(after_speech)) if after_speech else None,
            'pure_silence_return': float(np.mean(pure)) if pure else None,
            'yield_after_speech_n': len(after_speech), 'pure_silence_n': len(pure)}


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
    the interval (probability 1 - prod(1 - h_t)) says the word sampled from the read, heard next. A missing commit
    retains unfinished speech while the brain evolves in silence. A boundary token or sustained silence ends it;
    the listener's existing patience is measured in actual ticks, independently of the sampled input interval.
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
    consequence = None
    native_generative = bool(getattr(learner, 'native_generative_enabled', False))
    native_context = bool(getattr(learner, 'native_context_enabled', False))
    if getattr(learner, 'speak_credit', None) == 'consequence':
        if not speak_act:
            raise ValueError('Consequence credit needs --mother-speak-act')
        if (native_context or native_generative or getattr(learner, 'consequence_native_enabled', False)
                or getattr(learner, 'native_credit_enabled', False)):
            from .fly_native_policy_credit import prepare_native_policy_backward
            prepare_native_policy_backward(m)
        if native_context:
            if (native_generative or getattr(learner, 'consequence_native_enabled', False)
                    or getattr(learner, 'native_credit_enabled', False)):
                raise ValueError('Native context replaces the other experimental policy modes explicitly')
            from .fly_native_context import NativeContextControl
            consequence = NativeContextControl(learner)
        elif native_generative:
            if getattr(learner, 'consequence_native_enabled', False) or getattr(learner, 'native_credit_enabled', False):
                raise ValueError('Native generative control replaces the old policy-credit objective explicitly')
            from .fly_native_generative_control import NativeGenerativeControl
            consequence = NativeGenerativeControl(learner)
        elif getattr(learner, 'consequence_native_enabled', False):
            if getattr(learner, 'native_credit_enabled', False):
                raise ValueError('Choose native credit alone or the ordered policy candidate, not both')
            from .fly_action_model import OrderedConsequenceCredit
            consequence = OrderedConsequenceCredit(learner)
        elif getattr(learner, 'native_credit_enabled', False):
            from .fly_consequence_belief import NativeCircuitConsequenceCredit
            consequence = NativeCircuitConsequenceCredit(learner)
        else:
            from .fly_consequence_belief import NativeConsequenceCredit
            consequence = NativeConsequenceCredit(learner)
    circuit = getattr(learner, 'circuit_observer', None)      # fly_circuit_observer: counts the live pass only
    if circuit is not None:
        circuit.active = True
    try:
        return _live_window_body(learner, world, n, rng, pace_min, pace_mean, temperature, ticks_per_second,
                                 max_wait_ticks, speak_act, interrupt, consequence, native_generative,
                                 native_context, circuit)
    finally:
        if circuit is not None:
            circuit.active = False


def _live_window_body(learner, world, n, rng, pace_min, pace_mean, temperature, ticks_per_second,
                      max_wait_ticks, speak_act, interrupt, consequence, native_generative, native_context, circuit):
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
        causal_clock = getattr(learner, 'mother_action_clock', 'arrival') == 'commit'
        if causal_clock and not acts:
            raise ValueError('The commit event clock requires native speaking actions')
        observer = None
        if getattr(learner, 'generative_shadow_enabled', False):
            if not causal_clock:
                raise ValueError('The generative event observer requires the actual commit clock')
            from .fly_generative_observer import GenerativeEventObserver
            cached = getattr(learner, '_generative_event_observer', None)
            if cached is not None and cached.state is getattr(learner, 'generative_event_state', None):
                observer = cached
                observer.diagnostics = []
            else:
                observer = GenerativeEventObserver(learner)
                learner._generative_event_observer = observer
        motor_on = getattr(m, 'ai_mode', None) == 'motor'
        motor_port_on = bool(getattr(m, 'native_motor_port_enabled', False))
        if motor_port_on and (not causal_clock or not acts):
            raise ValueError('Native motor port needs actual commit-clock speaking actions')
        if motor_on:
            from .fly_active_inference import motor_rest
            saved_muscles = getattr(learner, 'ai_g', None)
            ai_g = saved_muscles.detach().clone() if saved_muscles is not None else motor_rest(m, s.h)
        else:
            ai_g = None
        targets, intervals, speakers, decisions, quiet_ranges, sources = [], [], [], [], [], []
        arrival_reads = []
        event_tape = []
        input_sources = []
        motor_packet_tape, motor_commands = [], []
        pending_motor_packet = None
        context_endpoints = []
        tpb = int(getattr(world, 'ticks_per_beat', 8))
        pause = int(getattr(world, 'pause_beats_before_reply', 1)) * tpb
        # whose voice is on the senses: the mother's words carry it (Johnston's organ, fly_bptt_learning.attach_mother_voice)
        voice = bool(getattr(learner, 'last_voice', False))
        # her answer's reward pulse rides with its first word as the input (fly_bptt_learning.attach_reply_reward)
        rewarded = bool(getattr(learner, 'last_reward', False))
        reply_onsets = []
        consequence_ratios = []
        prev_who = 'mother' if voice else 'fly'

        def advance(k, start=0, sensory_on=True, can_commit=None):
            # ticks start .. start + k - 1 of the word's interval, in the training pass's segments (same packets)
            nonlocal s, ai_g, first_commit_tick, last_advance_ticks, pending_motor_packet
            done, latent, committed, said_read = start, None, False, None
            if can_commit is None:
                can_commit = world.turn == 'fly' or (world.turn == 'mother' and bool(interrupt))
            while done < start + k:
                ticks = min(seg - done % seg, start + k - done)
                extra_kw = (dict(ai_g=ai_g, ai_tape=[], eye_forced=getattr(learner, 'eye_forced', None),
                                 read_every=True) if motor_on else {})
                if native_context:
                    extra_kw['update_expectation'] = False
                if hasattr(consequence, 'timing_bias'):
                    extra_kw['motor_value_bias'] = consequence.timing_bias
                if consequence is not None:
                    extra_kw['ai_g_tape'] = []
                if causal_clock and can_commit:
                    extra_kw['stop_on_commit'] = True
                out = fl.advance_fly_token_stream(
                    m, s, token_tensor, [ticks], ticks=ticks,
                    hold_input=learner.hold_input, read_after=learner.read_lag or None,
                    writer_baseline_clock=learner.writer_baseline_clock, fresh=done == 0, voice=voice,
                    sensory_on=sensory_on, reward_ticks=fl.reward_ticks_left(m, rewarded, done),
                    executed_action=bool(native_context and held_source == 'self_hearing' and sensory_on),
                    **extra_kw, **opts)
                s, latent = out[0], out[1]
                actual = len(extra_kw['ai_tape']) if motor_on else ticks
                if hasattr(consequence, 'advance_ticks'):
                    consequence.advance_ticks(actual)
                if motor_on:
                    ai_g = out[-1]
                    if motor_port_on:
                        motor_packet_tape.extend(out[4].detach().cpu().tolist())
                    if native_generative or native_context:
                        interval_eyes.extend(out[-4].detach().cpu().tolist())
                    tape = extra_kw['ai_tape']
                    # Muscles keep evolving/sampling during protected external
                    # pauses, but those samples are not world commands. Do not
                    # let a protected pause create a delayed "commit" later.
                    accepted_tape = tape if not causal_clock or can_commit else []
                    if consequence is not None:
                        decision_frames.extend(extra_kw['ai_g_tape'])
                        if sensory_on:
                            sensed_eyes.append(out[-4][out[-3]])
                        if first_commit_tick is None and any(accepted_tape):
                            first_commit_tick = done + accepted_tape.index(True)
                    if not committed and any(accepted_tape):
                        # the word is said from the read standing at the first commit tick
                        first = accepted_tape.index(True)
                        said_read = out[3 if motor_port_on else 2][first:first + 1]
                        if motor_port_on:
                            release = s.ring[0][0, m.native_motor_index].detach()
                            active = release > 0
                            if not bool(active.any()):
                                raise RuntimeError('Native command has no actual motor packet')
                            pending_motor_packet = {
                                'tick': tick0 + done + first,
                                'body_ids': m.native_motor_body_ids[active].cpu().tolist(),
                                'release': release[active].cpu().tolist()}
                    committed = committed or any(accepted_tape)
                done += actual
                if causal_clock and can_commit and committed:
                    break
            last_advance_ticks = done - start
            return latent, committed, said_read

        def say(read):
            nonlocal pending_motor_packet
            from .fly_native_motor_port import action_logits
            if motor_port_on and pending_motor_packet is None:
                raise RuntimeError('Language peripheral cannot execute without a native motor packet')
            logits = action_logits(m, read).float()[0] / max(temperature, 1e-6)
            if getattr(consequence, 'word_bias', None) is not None:
                logits = logits + consequence.word_bias
            word = int(torch.multinomial(torch.softmax(logits, -1), 1))
            if consequence is not None:
                consequence.say(read, word, temperature)
            if motor_port_on:
                motor_commands.append({**pending_motor_packet, 'word': word})
                pending_motor_packet = None
            return word

        def speak_word(word):
            ended = world.speak(word)
            if consequence is not None and ended:
                consequence.state['in_child_turn'] = False

        def choose(context, start, length, mother_turn=False):
            if consequence is not None:
                first = (first_commit_tick - start if first_commit_tick is not None
                         and start <= first_commit_tick < start + length else None)
                frames = torch.cat(decision_frames[start:start + length], 0)
                if hasattr(consequence, 'decision_bounds'):
                    consequence.decision_bounds = (tick0 + start, length)
                consequence.choose(context, frames, first, mother_turn)

        tick0 = 0                                  # ticks of the window before this word's interval
        last_advance_ticks = 0
        for i in range(n):
            word, who = None, None
            decision_frames, sensed_eyes, first_commit_tick = [], [], None
            interval_eyes = []
            if native_context:
                held_source = consequence.state['source']
                input_sources.append(held_source)
                s = fl.prepare_sensory_expectation(m, s, token, held_source)
            if native_generative or native_context:
                from .fly_active_inference import ai_eye
                consequence.initial_eye = float(ai_eye(m, ai_g, getattr(learner, 'eye_forced', None)))
                consequence.begin_interval(s, token, voice, rewarded)
            if observer is not None:
                from .fly_active_inference import ai_eye
                if world.turn == 'fly' and not observer.state.get('in_child_turn', False):
                    observer.begin_child_turn()
                observer.begin_interval(s, float(ai_eye(m, ai_g, getattr(learner, 'eye_forced', None))))
            context = consequence.features(s)[0] if consequence is not None else None
            if (causal_clock and consequence is not None and world.turn == 'fly'
                    and not consequence.state.get('in_child_turn', False)):
                consequence.begin_child_turn()
            if hasattr(consequence, 'prepare'):
                consequence.prepare(context, s, enabled=(world.turn != 'mother' or bool(interrupt)),
                                    mother_turn=world.turn == 'mother', temperature=temperature)
            if consequence is not None:
                source = consequence.state['source']
                consequence_ratios.append(consequence.input_ratio(source))
            quiet = []
            # Reuse the token tensor across checkpoint segments and silent gaps.
            token_tensor = torch.tensor([token], device=s.h.device)
            extra = 0
            if world.turn == 'waiting':
                extra = min(int(max_wait_ticks), world.wait_for_mother() * tpb)
            elif world.turn == 'mother' and getattr(world, 'mother_turn_fresh', False):
                extra = min(int(max_wait_ticks), pause)
            k = int(rng.geometric(1.0 / pace))
            latent, committed, said_read = advance(k)
            k = last_advance_ticks
            if consequence is not None:
                exposure = float(torch.cat(sensed_eyes).mean()) if sensed_eyes else 0.0
                if hasattr(consequence, 'hear_token'):
                    consequence.hear_token(token, source, exposure)
                consequence.observe(s, source, exposure, end=bool(consequence.state.get('reply_end', False)))
            if extra and not (causal_clock and committed):
                latent, extra_commit, extra_read = advance(extra, k, sensory_on=False,
                                                          can_commit=False if causal_clock else None)
                extra = last_advance_ticks
                quiet.append((k, k + extra))
                if not committed and extra_commit:
                    said_read = extra_read
                committed = committed or extra_commit
                k += extra
            if world.turn == 'mother':
                if acts and interrupt:
                    if consequence is not None and committed:
                        consequence.end_response()
                        consequence.begin_child_turn()
                    decisions.append((i, tick0, k, int(committed)))
                    choose(context, 0, k, mother_turn=True)
                if acts and interrupt and committed:
                    world.interrupt()                       # she is cut off; the child's word is heard
                    word, who = say(said_read if said_read is not None else latent), 'fly'
                    speak_word(word)
                else:
                    word, who = world.next_word()
            elif acts:
                if consequence is not None and not consequence.state.get('in_child_turn', False):
                    consequence.begin_child_turn()
                decisions.append((i, tick0, k, int(committed)))
                choose(context, 0, k)
                tries = 0
                silent_ticks = k
                while not committed and word is None:
                    # Keep partial speech and its credit during a short pause.
                    world.pause_speech(silent_ticks)
                    gap = (min(int(max_wait_ticks), world.wait_for_mother() * tpb) if world.turn == 'waiting'
                           else pause if world.turn == 'mother' else 0)
                    if world.turn != 'mother' and tries >= 8 and not world.fly_words:
                        world.call()                        # a long silence: she calls the child by name
                        gap = pause
                    if world.turn == 'mother':
                        if gap > 0:
                            advance(gap, k, sensory_on=False, can_commit=False if causal_clock else None)
                            gap = last_advance_ticks
                            quiet.append((k, k + gap))
                            k += gap
                        word, who = world.next_word()
                    else:
                        gap = max(gap, tpb)                 # a beat of silence, and the child decides again
                        if world.fly_words:
                            gap = min(gap, world.speech_pause_limit() - world.fly_pause_ticks)
                        wait_context = consequence.features(s)[0] if consequence is not None else None
                        if hasattr(consequence, 'prepare'):
                            consequence.prepare(wait_context, s, temperature=temperature)
                        latent, committed, said_read = advance(gap, k, sensory_on=False)
                        gap = last_advance_ticks
                        quiet.append((k, k + gap))
                        decisions.append((i, tick0 + k, gap, int(committed)))
                        choose(wait_context, k, gap)
                        k += gap
                        silent_ticks = gap
                        tries += 1
                if word is None:
                    word, who = say(said_read if said_read is not None else latent), 'fly'
                    speak_word(word)
            else:
                word, who = say(latent), 'fly'
                speak_word(word)
            targets.append(int(word)); intervals.append(k); speakers.append(who)
            quiet_ranges.append(quiet)
            sources.append(world.mother_source if who == 'mother' else 'self_hearing')
            arrival_reads.append(latent[-1:].detach() if latent is not None else None)
            if circuit is not None:
                circuit.end_word(m, token=int(word), who=who, source=sources[-1], read=latent,
                                 extra={'committed': bool(committed) if acts else None,
                                        'reply_onset': bool(who == 'mother' and sources[-1] == 'mother_reply'
                                                            and prev_who != 'mother')})
            if native_generative or native_context:
                consequence.arrive(s, int(word), sources[-1], k, quiet, interval_eyes,
                                   end=who == 'mother' and not world.queue)
            if native_context:
                from .fly_native_context import compartment_chart
                context_endpoints.append({'field': compartment_chart(m, s)[0].detach().cpu(),
                                          'read': latent.detach().cpu()})
                s = fl.expect_next(m, s, latent)
            if observer is not None:
                observer.end_interval(k, int(word), sources[-1])
                if who == 'mother':
                    observer.state['in_child_turn'] = False
            if causal_clock:
                event_tape.append({'tick': tick0 + k, 'interval_ticks': k,
                                   'source': sources[-1], 'token': int(word),
                                   'kind': 'command_echo' if who == 'fly' else 'external_arrival'})
            onset = (who == 'mother' and sources[-1] == 'mother_reply' and prev_who != 'mother'
                     and (consequence is None or bool(getattr(world, 'reply_has_speech', False))))
            reply_onsets.append(onset)
            prev_who = who
            token = int(word)
            voice = who == 'mother'
            rewarded = onset
            if consequence is not None:
                consequence.state['source'] = sources[-1]
                consequence.state['reply_end'] = who == 'mother' and not world.queue
                if who == 'mother':
                    consequence.state['in_child_turn'] = False
            tick0 += k
    forced = {'pace': pace, 'intervals': intervals, 'speakers': list(speakers),
              'quiet_ranges': quiet_ranges, 'sources': sources, 'reply_onsets': reply_onsets}
    if acts:
        forced.update(decisions=decisions)
    if causal_clock:
        forced.update(action_clock='commit', events=event_tape, prediction_objective='external_arrival',
                      exclude_self_prediction=True)
    if native_generative or native_context:
        forced['native_generative_control'] = True
    if native_context:
        forced.update(native_context_control=True, input_sources=input_sources,
                      native_context_teaching=consequence.teaching,
                      native_context_endpoints=context_endpoints)
    if observer is not None:
        forced['generative_shadow'] = observer.diagnostics
        learner.last_generative_event = observer.diagnostics
    if motor_port_on:
        forced.update(native_motor_port=True, native_motor_availability=motor_packet_tape,
                      native_motor_commands=motor_commands)
    if consequence is not None:
        forced['consequence_ratios'] = consequence_ratios
        if hasattr(consequence, 'native_events'):
            forced['consequence_native_events'] = consequence.native_events
            if hasattr(consequence, 'timing_biases'):
                forced['consequence_timing_bias'] = consequence.timing_biases
            forced['exclude_self_prediction'] = causal_clock or bool(
                getattr(learner, 'consequence_native_enabled', False))
        consequence.finish()
    # the lived trajectory itself (brain_learning 'local' commits it instead of reliving the window)
    if getattr(learner, 'brain_learning', 'bptt') == 'local':
        learner.live_result = {'state': s, 'ai_g': ai_g, 'reads': arrival_reads}
    return targets, forced, speakers
