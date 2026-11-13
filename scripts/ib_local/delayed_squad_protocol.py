"""Real annotated QA, delayed correction, interference, and future questions.

No model/torch imports. Test labels never enter a prompt before generation.
The same external schedule is used for all continuously learning arms.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import string


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(65536), b''):
            h.update(part)
    return h.hexdigest()


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_scores(prediction, answers):
    prediction = prediction.strip().split('\n')[0]
    p = normalize(prediction).split()
    exact, f1 = 0., 0.
    for answer in answers:
        a = normalize(answer).split()
        exact = max(exact, float(p == a))
        overlap = sum((Counter(p) & Counter(a)).values())
        score = float(p == a) if not p or not a else 2*overlap/(len(p)+len(a))
        f1 = max(f1, score)
    return dict(exact_match=exact, token_f1=f1)


def select_paragraphs(payload, encode, count, seed=11):
    articles = list(payload['data'])
    random.Random(seed).shuffle(articles)
    selected = []
    for article in articles:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            if not 128 <= len(encode(context)) <= 384:
                continue
            qas, seen = [], set()
            for qa in paragraph['qas']:
                answers = qa.get('answers', [])
                if not answers or len(encode(qa['question'])) > 48:
                    continue
                answer = answers[0]
                value, start = answer['text'], answer['answer_start']
                if context[start:start+len(value)] != value or not 1 <= len(encode(' '+value)) <= 12:
                    continue
                if normalize(value) in seen:
                    continue
                seen.add(normalize(value))
                qas.append(dict(id=qa['id'], question=qa['question'], answer=value,
                                answers=list(dict.fromkeys(a['text'] for a in answers)), answer_start=start))
            if len(qas) >= 4:
                selected.append(dict(title=article['title'], context=context, qa=qas[:4]))
                break  # Different article for every selected paragraph.
        if len(selected) == count:
            return selected
    raise ValueError(f'only {len(selected)} paragraphs meet the declared real-data criteria')


def make_episodes(paragraphs, *, gaps=(256, 512)):
    if len(paragraphs) < 4 or len(paragraphs) % 2:
        raise ValueError('even balanced groups, at least two articles per group')
    if not gaps or any(type(n) is not int or n <= 128 or n % 128 for n in gaps):
        raise ValueError('delays must exceed128 and be multiples of the calculation window')
    half = len(paragraphs)//2
    groups = dict(A=paragraphs[:half], B=paragraphs[half:])
    stages = [('A_acquire','A',0,True,True), ('A_before_B','A',0,False,False),
              ('B_interfere','B',0,True,True), ('B_before_A','B',0,False,False),
              ('A_retention','A',0,False,False), ('A_relearn1','A',0,False,True),
              ('A_relearn2','A',0,False,True), ('A_transfer','A',1,False,False),
              ('B_retention','B',0,False,False), ('B_transfer','B',1,False,False)]
    episodes = []
    for stage, group, qindex, present, correct in stages:
        for i, paragraph in enumerate(groups[group]):
            qa = paragraph['qa'][qindex]
            prompt = 'Answer the question using a short phrase.\n'
            if present:
                prompt += 'Passage: '+paragraph['context']+'\n'
            prompt += 'Question: '+qa['question']+'\nAnswer:'
            episodes.append(dict(index=len(episodes), stage=stage, group=group, title=paragraph['title'],
                qa_id=qa['id'], prompt=prompt, answer=qa['answer'], answers=qa['answers'],
                feedback_prefix='\nFeedback for the preceding question. Correct answer:',
                feedback_delay_tokens=gaps[i%len(gaps)], present_context=present,
                new_question=qindex != 0, provide_correction=correct))
    return episodes


def validate_recovery(state, previous, *, task_sha256, arm, episodes):
    driver = state['delayed_squad_driver']
    completed, cursor = driver['episodes_completed'], driver['gap_cursor']
    if (driver['arm'] != arm or previous['arm'] != arm or
            driver['task_sha256'] != task_sha256 or previous['task_sha256'] != task_sha256 or
            previous['requested_episode_count'] != episodes or
            driver.get('frozen_mlp_precision','fp16')!=previous.get('frozen_mlp_precision','fp16') or
            driver.get('frozen_weight_placement','cuda')!=previous.get('frozen_weight_placement','cuda') or
            driver.get('model_source_sha256')!=previous.get('model_source_sha256')):
        raise ValueError('recovery task, arm or budget differs')
    if not 0 < completed < episodes or completed % 8:
        raise ValueError('recovery requires a committed nonterminal episode group')
    cases = previous['cases'][:completed]
    if len(cases) != completed or [c['index'] for c in cases] != list(range(completed)):
        raise ValueError('recovery case prefix missing')
    if cursor != cases[-1]['gap_token_cursor'][1]+128:
        raise ValueError('recovery external cursor differs')
    learner, stream = state['learner'], state['stream']
    if (stream['events'] != cases[-1]['end_event'] or
            stream['chain'] != driver['history_chain'] or learner['updates'] != cases[-1]['updates'] or
            learner['cache']['seen_tokens'] != cases[-1]['seen'] or learner['native_s_sidecar']['mode'] != arm or
            len(stream['delayed_feedback_positions']) != len(stream['targets'])):
        raise ValueError('recovery learner, history or pending feedback differs')
    windows = previous['windows'][:driver['windows_completed']]
    if len(windows) != driver['windows_completed']:
        raise ValueError('recovery window prefix missing')
    return dict(completed=completed, cursor=cursor, cases=cases, windows=windows)


def prepare(task_file, model, output, *, paragraphs=16):
    # The native tokenizer library avoids importing Torch into a data-preparation process.
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(model)/'tokenizer.json'))
    encode = lambda text:tokenizer.encode(text, add_special_tokens=False).ids
    source = json.loads(Path(task_file).read_text(encoding='utf-8'))
    selected = select_paragraphs(source, encode, paragraphs)
    episodes = make_episodes(selected)
    for episode in episodes:
        episode['prompt_tokens'] = encode(episode['prompt'])
        episode['feedback_prefix_tokens'] = encode(episode['feedback_prefix'])
        # Separate actual arrivals are explicitly tokenized, just like an API token stream.
        episode['answer_tokens'] = encode(' '+episode['answer'])
    result = dict(schema=1, seed=11, dataset='SQuAD1.1/train', source_path=str(Path(task_file).resolve()),
        source_sha256=digest(task_file), tokenizer_sha256=digest(Path(model)/'tokenizer.json'),
        paragraphs=selected, episodes=episodes, calculation_window=128,
        gap_source='disjoint real OWT bytes; encoded using this same tokenizer',
        primary=['pre-feedback answer exact match and F1','first-answer-token preupdate NLL',
                 'future unseen questions','return retention and relearning'],
        teaching='SFT: only arrived correction answer targets teach; questions, distraction and self speech remain unlabelled inputs',
        no_reset=True, no_question_repeated_in_feedback=True,
        scope='resource-bounded real-data mechanism study; not official SQuAD score or convergence evidence')
    output = Path(output)
    if output.exists():
        raise FileExistsError('preserve the preregistered task')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
