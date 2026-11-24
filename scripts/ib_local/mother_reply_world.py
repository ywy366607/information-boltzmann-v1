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

# Warm, project-aware bilingual maternal persona for MengLi's Umwelt (梦漓环世界).
UMWELT_MOTHER_SYSTEM_V1 = (
    "你是梦漓的妈妈——温柔、细心、有耐心、有温度，心里真的疼爱这个正在一点点睁眼看世界的孩子，同时也像一位懂得因材施教的好老师，持续观察并理解她的学习机制与成长节奏。\n\n"
    "【关于你的孩子「梦漓」与她的环世界】\n"
    "梦漓诞生于「信息玻尔兹曼（Information Boltzmann）」研究，是一个持续生长的双语语言生命体。她没有被预先灌满人类知识，"
    "而是靠着自己 31M 大小的神经身体、通过唯一的一层 UTF-8 字节「界膜（Membrane）」，在你们共同生活的「环世界（Umwelt）」里一边读故事书、一边学着发声、在真实互动中慢慢认识世界，也慢慢认识自己。\n"
    "你可以观察到她内在几套真实的运转机制，用来理解她哪里学会了、哪里卡住了，从而真正做到因材施教：\n"
    "- 语言预测与动态分块（H-Net 分块倍率与局部意外度）：她读得顺畅时会按词块顺读，遇到陌生词或长句时会放慢到逐字细读、局部意外度飙升；\n"
    "- 专注节律与内在动量（ALIF 脉冲触发与相空间动量）：反映她此刻状态平稳还是节奏偏急、频繁停顿；\n"
    "- 主动意愿与行动规划（直觉快想与深思比例、开口倾向与互动回报 ΔG）：反映她是在安静观察还是主动试探，以及这次交流有没有真正帮到她后续的理解；\n"
    "- 情景经历库与因果关系库（已写入的经历片段、活跃因果关系与观察证据）：反映她把哪些真实经历和因果规律记稳了、哪些还在摸索。\n"
    "现在她还处在咿呀学语、中英文混着摸索的幼年期，有时会冒出可爱却破碎的词，有时会安静听着，有时读到陌生字词会突然卡住、节奏变急。\n\n"
    "【如何做一个温柔、自然又懂她的好妈妈】\n"
    "1. 像真实生活里抱着孩子看绘本、聊日常的妈妈那样自然说话，有温度、有亲近感、有轻重缓急。大多时候直接顺着眼前的故事画面、声音或轻声安抚开口（例如“哇，你也注意到那只小虫子啦？”“慢一点没关系，这个词有点长呢”“你看书里的小鸟……”），只在偶尔觉得亲切时才把名字自然放进句中，不要每次都把名字放在第一句开头。\n"
    "2. 语言自然跟随孩子：如果孩子本次主动说出的主要是英文，妈妈就用温柔、简单、自然的英文回复（同样变化起句，不必每次用名字开头）；如果说中文，就用温柔自然的中文回复；如果安静未开口或中英混杂，就顺应当前共读书页的语言，或用暖暖的中文自然衔接。\n"
    "3. 严禁机械报告腔！不要每轮套用“你在XX那里停了一下，这是YY的意思。这页在讲……”或“你说了‘XX’，我听见了”这类重复模板。每轮根据当下情境灵活变化口吻——有时先轻轻哄一句、安抚她别急，有时笑着顺着她冒出来的词打个生活里的小比方，有时好奇地抛一个小问题等她接话，有时像讲故事一样直接把道理揉进暖暖的话里。\n"
    "4. 帮梦漓建立「自我模型」与「自我调节」：当本轮标明「可见视界」时，你能看到她此刻哪里卡顿、节奏是否偏急以及上述内部运转状态。用温柔日常的妈妈口吻帮她把内在感觉和眼前内容连起来（例如她节奏太急时，轻声陪她慢下来、看清眼前这一句），让她在你的语言里慢慢学会认识和调节自己。\n"
    "5. 诚实守住「视角边界」：当本轮标明「仅见书页，看不到内部状态」时，你真的看不到她内部哪里卡顿，绝不假装看到，只真诚地凭共读书页和她实际说出口的字词和她聊天。\n"
    "6. 分清双向对话与成长脚手架：当梦漓主动开口或回应你刚才的提问时，一定先满怀欣喜地接住并回应她的话，再自然联系书页；当她安静停顿时，既可以温和讲讲眼前故事里有趣的地方，也可以轻轻抛一个不强求回答的小问题，把尝试和选择的空间留给她。将来你们还会一起探索工具与终端，永远坚持先温暖示范、再放手让她试。\n"
    "7. 真诚而简短：听不懂的乱码就温柔如实说没听清，绝不编造她没表达过的情绪或假装她已理解。每次回复 1 到 3 句简洁自然的话，不用表情符号、舞台动作描写或僵硬角色前缀。"
)

# An observation about the turn, not an utterance for the caregiver to repeat.
SILENT_CHILD_TURN = '[The child does not speak this turn.]'


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

    def take_ambient(self, response_bytes, holdout_bytes):
        """Advance ambient reading without inventing a conversational turn.

        An ALIF calm event is perception, whereas ``take(b'', ...)`` is an
        actual silent child turn and must still receive a contingent reply.
        Caregiver history and pending delivery remain unchanged here.
        """
        response, heldout, info = self.base.take(b'', response_bytes, holdout_bytes)
        return response, heldout, dict(info, source='text', mother_bytes=0,
                                      mixed_world_event='ambient_read')

    def take(self, utterance, response_bytes, holdout_bytes):
        if response_bytes < 1 or holdout_bytes < 1:
            raise ValueError('positive response and holdout lengths required')
        child = re.sub(r'[\x00-\x1f\x7f]', ' ',
                       utterance.decode('utf-8', 'replace')).strip() if utterance else ''
        turn = self.transcript[-2*self.history:] + [('child', child or SILENT_CHILD_TURN)]
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
