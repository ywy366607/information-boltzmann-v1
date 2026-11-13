"""Prequential replay of held real-body receipts; no body/model mutation."""
import json
from pathlib import Path

from scripts.ib_local.causal_relation_runtime import RelationMemory, admit
from scripts.ib_local.probe_executable_causal_relation import examples

ROOT = Path(__file__).resolve().parents[2]


def main():
    previous = json.loads((ROOT / 'results/cgb_executable_relation_20261109/report.json').read_text(encoding='utf-8'))
    train = examples(0) + examples(1)
    held = examples(2)
    code = previous['program']
    relation = admit(code, 2, [row['x'] for row in train])
    bank = RelationMemory(capacity=1)
    bank.insert(relation)
    train_mean = round(sum(row['y'] for row in train) / len(train))
    history = []
    for row in held:
        before = bank.entries[relation.digest]['count']
        prediction = bank.predict(relation.digest, row['x'])
        error = bank.settle(row['y'])
        history.append({'case': row['case'], 'layer': row['layer'], 'slot': row['slot'],
                        'x': row['x'], 'observed_micro_nats': row['y'],
                        'predicted_micro_nats': prediction, 'absolute_error': error,
                        'settled_before_prediction': before})
    n = len(held)
    report = {'status': 'complete', 'scope': 'offline held receipt replay; not live body',
              'candidate_origin': 'prior researcher fit on cases 0 and 1',
              'lean_scope': 'fixed template execution on 8 training inputs only',
              'train_cases': [0, 1], 'held_case': 2, 'digest': relation.digest,
              'held_count': n, 'mean_absolute_error_nats': sum(x['absolute_error'] for x in history)/n/1e6,
              'train_mean_baseline_mae_nats': sum(abs(row['y']-train_mean) for row in held)/n/1e6,
              'zero_baseline_mae_nats': sum(abs(row['y']) for row in held)/n/1e6,
              'body_updates': 0, 'optimizer_updates': 0, 'records': history}
    out = ROOT / 'results/cgb_causal_relation_runtime_20261109/report.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
