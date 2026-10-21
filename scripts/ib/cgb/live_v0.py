"""CGB V0 living in the mother world: hears, predicts, learns, and decides itself when to speak.

--world contingent  the DeepSeek mother (MotherWorld, contingent): speaking buys her reply; silence is met with silence
                    and, after `silent_beats_threshold` silent beats, her call.
--strict            the same mother, but silence never buys her stories (this run only, in memory)
--world yoked       the same kind of speech, independent of the child: recorded mother turns are played after a
                    random pause (geometric, mean `--yoked-beats` beats) whatever the child does. The child may speak in
                    the pauses; nobody answers. A gate that learns "my acts have consequences" must learn to speak only
                    in the contingent world.

Time: a word lasts `ticks_per_word` ticks, a beat of silence 8 ticks. The child's own words are heard back (self-hearing)
but neither train the prediction nor count as information for the gate. Rows go to events.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import tiktoken

from information_boltzmann.cgb.v0 import Brain

TICKS_PER_BEAT = 8


class YokedWorld:
    """Mother turns from a recording, on her own schedule; the child's speech changes nothing."""

    def __init__(self, turns, mean_beats, seed=0):
        self.turns, self.mean_beats, self.rng = list(turns), float(mean_beats), random.Random(seed)
        self.queue, self.turn, self.index, self.fly_words, self.wait = [], 'fly', 0, [], 0
        self._next_turn()

    def _next_turn(self):
        self.queue = list(self.turns[self.index % len(self.turns)])
        self.index += 1
        self.turn = 'mother'

    def next_word(self):
        word = self.queue.pop(0)
        if not self.queue:
            self.turn = 'fly'
            self.wait = 1
            while self.rng.random() > 1.0 / self.mean_beats:
                self.wait += 1
        return word, 'mother'

    def _beat_passed(self, beats):
        self.wait -= beats
        if self.wait <= 0:
            self.fly_words = []
            self._next_turn()

    def speak(self, word):
        self.fly_words.append(word)
        self._beat_passed(1.0 / TICKS_PER_BEAT * 4)
        return False

    def silent_beat(self):
        self._beat_passed(1)


def make_mother_world(enc, strict=False):
    from information_boltzmann.core.fly_mother_world import DeepSeekMother, FallbackMother, MotherWorld, SiliconFlowMother
    raising = json.loads(Path('data/harness/raising.json').read_text(encoding='utf-8')).get('mother', {})
    mother = FallbackMother(DeepSeekMother('deepseek-flash'), SiliconFlowMother('zai-org/GLM-5.3-Flash'))
    if strict:   # this run only, in memory: silence is met with silence, her calls are the short call prompt
        plan = json.loads(json.dumps(mother.primary.plan))
        plan.setdefault('mother', {})['spontaneous_when_quiet'] = False
        mother.update_plan(plan)
    return MotherWorld(mother, enc, max_fly_words=int(raising.get('max_fly_words', 24)),
                       opening='Hello, little one.', end_of_turn=tuple(raising.get('child_end_of_turn') or ('\n', '.', '?', '!')),
                       history_turns=int(raising.get('history_turns', 20)), ticks_per_beat=TICKS_PER_BEAT,
                       pause_beats_before_reply=1, silent_beats_threshold=int(raising.get('silent_beats_threshold', 3)),
                       contingent=True)


def recorded_turns(enc, path='E:/fly_stream_overflow/newborn/mother.jsonl'):
    rows = [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines()]
    return [enc.encode(' ' + r['text'].strip() + '\n') for r in rows if r.get('who') == 'mother' and r.get('text', '').strip()]


def main(a):
    enc = tiktoken.get_encoding('gpt2')
    brain = Brain({'seed': a.seed})
    if a.brain:
        with np.load(a.brain) as saved:
            brain.load_state_dict({k: saved[k] for k in saved.files})
    contingent = a.world == 'contingent'
    world = make_mother_world(enc, a.strict) if contingent else YokedWorld(recorded_turns(enc), a.yoked_beats, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    log = (a.out / 'events.jsonl').open('w', encoding='utf-8')
    heard = slots = spoken = 0
    block = {'s': [], 'p': [], 'speak': []}
    t0 = time.time()
    while heard < a.words:
        if world.turn == 'mother':
            tok, _ = world.next_word()
            s = brain.heard_from_world(tok)
            heard += 1
            block['s'].append(s)
            log.write(json.dumps({'tick': brain.ticks, 'who': 'mother', 'tok': tok, 's': round(s, 3)}) + '\n')
            if heard % a.block == 0:
                sp = np.array(block['speak'], float)
                print(f'{heard:5d} heard {time.time() - t0:6.0f}s | surprisal {np.mean(block["s"]):.2f} | '
                      f'slots {len(sp)} spoke {sp.mean() if len(sp) else float("nan"):.2f} '
                      f'gate p {np.mean(block["p"]) if block["p"] else float("nan"):.2f} | gate bias {brain.b_gate:+.3f}',
                      flush=True)
                block = {'s': [], 'p': [], 'speak': []}
        elif world.turn == 'fly':
            speak, p = brain.decide()
            slots += 1
            block['p'].append(p); block['speak'].append(speak)
            if speak:
                tok = brain.babble(a.temperature)
                spoken += 1
                s = brain.hear(tok, learn=False)
                log.write(json.dumps({'tick': brain.ticks, 'who': 'child', 'tok': tok, 'p': round(p, 3),
                                      'text': enc.decode([tok])}) + '\n')
                ended = world.speak(tok)
                if contingent and ended and world.turn == 'mother':
                    brain.silence(TICKS_PER_BEAT)          # her pause before answering
            else:
                log.write(json.dumps({'tick': brain.ticks, 'who': 'child', 'tok': None, 'p': round(p, 3)}) + '\n')
                if contingent:
                    if world.fly_words:
                        world.pause_speech(brain.g['ticks_per_word'])
                        brain.silence(brain.g['ticks_per_word'])
                    else:
                        world.yield_turn()
                        brain.silence(TICKS_PER_BEAT)
                else:
                    world.silent_beat()
                    brain.silence(TICKS_PER_BEAT)
        else:                                               # an asynchronous mother would be thinking
            brain.silence(TICKS_PER_BEAT)
    log.close()
    np.savez(a.out / 'brain.npz', **brain.state_dict())
    summary = {'world': a.world, 'heard': heard, 'slots': slots, 'spoken': spoken, 'gate_bias': float(brain.b_gate)}
    if contingent:
        summary['transcript_tail'] = world.transcript[-12:]
        (a.out / 'transcript.json').write_text(json.dumps(world.transcript, ensure_ascii=False, indent=0), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--world', choices=['contingent', 'yoked'], default='contingent')
    p.add_argument('--brain', type=Path, default=None, help='start from a saved brain (e.g. after replay)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--words', type=int, default=2000, help='world words heard')
    p.add_argument('--block', type=int, default=200)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--yoked-beats', type=float, default=3.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--strict', action='store_true', help='contingent mother without spontaneous speech in silence')
    main(p.parse_args())
