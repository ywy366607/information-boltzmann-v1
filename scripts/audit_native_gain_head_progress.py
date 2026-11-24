"""Compare the repaired gain function on the same real stored experiences."""
import argparse
import gc
import gzip
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def extract(path, slots):
    with gzip.open(path, 'rb') as stream:
        state = torch.load(stream, map_location='cpu', weights_only=False)
    model = state['model']
    weights = {k: v.clone() for k, v in model.items()
               if k.startswith(('encoder.', 'kanli.outcome.'))}
    material = {k: model['memory.' + k][slots].clone()
                for k in ('source', 'action', 'wood_option', 'wood_target')}
    evidence = state['engine']['causal_use_value_evidence']
    result = dict(cycle=state['cycle'], weights=weights, material=material,
                  base=evidence.bases().clone(), confidence=evidence.confidence().tolist(),
                  seen=evidence.seen.tolist(), base_error=evidence.base_error[:, 3].tolist(),
                  raw_error=evidence.raw_error[:, 3].tolist())
    del state, model, evidence
    gc.collect()
    return result


def predict(weights, material, base):
    source = F.linear(material['source'], weights['encoder.weight'])
    action = F.linear(material['action'], weights['encoder.weight'])
    options = material['wood_option']
    x = torch.cat((source, action-source, F.one_hot(options, 3).to(source)), -1)
    hidden = F.silu(F.linear(x, weights['kanli.outcome.0.weight'], weights['kanli.outcome.0.bias']))
    raw = F.linear(hidden, weights['kanli.outcome.2.weight'], weights['kanli.outcome.2.bias'])[:, 3]
    return base[options, 3] + raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    cohort = json.loads(args.cohort.read_text(encoding='utf8'))
    slots = [row['slot'] for row in cohort['rows']]
    before, after = extract(args.before, slots), extract(args.after, slots)
    material = before['material']
    for key, value in material.items():
        assert torch.equal(value, after['material'][key]), f'cohort changed: {key}'
    old = predict(before['weights'], material, before['base'])
    new = predict(after['weights'], material, after['base'])
    fixed_base = predict(after['weights'], material, before['base'])
    torch.testing.assert_close(old, torch.tensor([row['reading_prediction'] for row in cohort['rows']]))
    target = material['wood_target'][:, 3]
    def errors(prediction):
        values = (prediction-target).square()
        return dict(min=float(values.min()), median=float(values.quantile(.5)),
                    mean=float(values.mean()), max=float(values.max()))
    result = dict(start_cycle=before['cycle'], end_cycle=after['cycle'], slots=slots,
        scope='same unmerged real source/action/option/absolute labels; current encoder and plain gain head',
        raw_before=errors(old), raw_after=errors(new), raw_after_fixed_base=errors(fixed_base),
        baseline_before=errors(before['base'][material['wood_option'], 3]),
        baseline_after=errors(after['base'][material['wood_option'], 3]),
        improved_samples=int(((new-target).square() < (old-target).square()).sum()),
        parameter_delta_norm={k: float((after['weights'][k]-v).norm()) for k, v in before['weights'].items()},
        gain_row_delta_norm=float((after['weights']['kanli.outcome.2.weight'][3]-before['weights']['kanli.outcome.2.weight'][3]).norm()),
        gain_bias_delta=float(after['weights']['kanli.outcome.2.bias'][3]-before['weights']['kanli.outcome.2.bias'][3]),
        calibration={side: {k: data[k] for k in ('seen', 'base_error', 'raw_error', 'confidence')}
                     for side, data in (('before', before), ('after', after))},
        optimizer_updates=0, world_calls=0)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
