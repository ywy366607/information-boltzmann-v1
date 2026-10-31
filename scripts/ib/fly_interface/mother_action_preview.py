"""Environmental intervention at one recorded real conversation context.

Different action texts and same-action replicates use the identical saved prompt
except its last child message. This checks the mouth, not the child's competence.
No child state, learning, queue or caregiver file is changed.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_mother_world import DeepSeekMother


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=2)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        parser.error('Keep counterfactual previews outside the actual experience log')
    rows = [json.loads(s) for s in (args.run / 'mother_prompt_audit.jsonl').read_text(encoding='utf8').splitlines()
            if s.strip()]
    saved = rows[-1]
    messages = saved['messages']
    child_index = max(i for i, x in enumerate(messages) if x['role'] == 'user')
    actions = [messages[child_index]['content'], 'more', 'stop']
    mother = DeepSeekMother(retries=1)
    report = {'at': datetime.now(timezone.utc).isoformat(), 'context_trained': saved['trained'],
              'scope': 'same recorded adult context; last child action replaced; teacher guidance held fixed',
              'capability_claim': False, 'learner_updates': 0, 'main_line_writes': 0,
              'base_messages': messages, 'responses': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for action in actions:
        for sample in range(args.samples):
            altered = copy.deepcopy(messages)
            altered[child_index]['content'] = action
            mother.get_chat_messages = lambda _, x=altered: copy.deepcopy(x)
            text = mother.reply([('fly', action)])
            result = {'action': action, 'replicate': sample, 'model': mother.model,
                      'status': getattr(mother, 'last_status', None), 'reply': text,
                      'rejections': getattr(mother, 'last_grounding_rejections', [])}
            report['responses'].append(result)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if not text:
                raise RuntimeError('Mother unavailable; do not replace an environmental counterfactual with invented text')


if __name__ == '__main__':
    main()
