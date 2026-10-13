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


class ApiMother:
    """A fast API model as the mother.  The SDK reads its own standard environment configuration (the key is never
    read, printed or stored here).  `model` is the user's choice."""

    SYSTEM = ('You are a warm, patient mother talking with a very young child who is just learning to speak. The child '
              'often babbles. Answer in one or two short, simple sentences. Respond to what the child said when you '
              'can; ask simple questions, name things, sometimes start a new topic.')

    def __init__(self, model, max_tokens=60, system=None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens
        self.system = system or self.SYSTEM

    def reply(self, transcript):
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


class MotherWorld:
    """The conversation as the fly's text stream (GPT-2 token ids, the fly's vocabulary).

    next_word(): the id the fly hears next. During the mother's turn, her reply word by word. During the fly's turn,
    the fly's own spoken word: the caller passes it to speak(), and it is heard next. Ids are GPT-2 BPE tokens.
    """

    def __init__(self, mother, gpt2_tokenizer, history_turns=20, max_fly_words=24, end_of_turn=('\n', '.', '?', '!'),
                 opening='Hello, little one.', seed=0):
        self.mother, self.tok = mother, gpt2_tokenizer
        self.history_turns, self.max_fly_words = history_turns, max_fly_words
        self.end_ids = {i for s in end_of_turn for i in gpt2_tokenizer.encode(s)}
        self.transcript = []                       # [(who, text)]
        self.queue = collections.deque()           # the mother's words still to be heard
        self.fly_words = []                        # the fly's current turn (ids)
        self.turn = 'mother'
        self.rng = random.Random(seed)
        self.stats = collections.Counter()
        self.fly_turns, self.mother_turns = [], []
        self._say_mother(opening)

    # -- the stream -------------------------------------------------------------------------------------------------
    def _say_mother(self, text):
        self.transcript.append(('mother', text))
        self.mother_turns.append(text)
        self.queue.extend(self.tok.encode(' ' + text.strip() + '\n'))
        self.turn = 'mother'
        self.stats['mother_turns'] += 1

    def next_word(self):
        """The id heard next, and whose it is ('mother' or 'fly' for the fly's own previous word)."""
        if self.turn == 'mother':
            word = self.queue.popleft()
            if not self.queue:
                self.turn = 'fly'
            return word, 'mother'
        return (self.fly_words[-1] if self.fly_words else None), 'fly'

    def speak(self, word_id):
        """The fly's committed word during its turn; returns True when the turn ended (the mother then answers)."""
        self.fly_words.append(int(word_id))
        self.stats['fly_words'] += 1
        if int(word_id) in self.end_ids or len(self.fly_words) >= self.max_fly_words:
            text = self.tok.decode(self.fly_words).strip()
            self.transcript.append(('fly', text or '...'))
            self.fly_turns.append(list(self.fly_words))
            self.fly_words = []
            self.stats['fly_turns'] += 1
            history = self.transcript[-self.history_turns:]
            self._say_mother(self.mother.reply(history))
            return True
        return False

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
                'mother_ttr': round(ttr(mom), 3), 'mother_repeat_bigrams': round(repeat_bigrams(mom), 3)}
