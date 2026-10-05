"""Read-only prediction decomposition of cached real motor features.

This is a retrospective head diagnostic, not a new prequential capability
score: cached features used the read projection active at each event, whereas
the decoder below is the saved end-of-encounter decoder. No physical evolution,
training, target-conditioned fitting, reset, or CUDA allocation occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference-train', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(args.run / 'last.pt', mmap=True, map_location='cpu', weights_only=False)
    weights, learner = saved['model'], saved['learner']
    targets = torch.tensor(saved['recent_a'], dtype=torch.long)
    cache = learner['latent_window']
    if len(targets) != len(cache):
        raise ValueError('Checkpoint must end after a complete cached A revisit')
    records = [json.loads(line) for line in (args.run / 'lifelong_evaluation.jsonl').read_text().splitlines()]
    matched = next((r for r in reversed(records) if r['bptt_train_tokens'] == saved['bptt_train_tokens']), None)
    if matched is None or matched['replay_tokens'] != targets.tolist():
        raise ValueError('Need a saved evaluation endpoint with exactly matching cached targets')
    if matched['event_interval'][1] != learner['events']:
        raise ValueError('Cached features are not the final replay events')
    split = learner['events'] % len(cache)
    features = torch.cat((cache[split:], cache[:split]))
    train = np.load(args.reference_train, mmap_mode='r')
    counts = np.ones(50257, dtype=np.float64)
    for left in range(0, len(train), 1_000_000):
        counts += np.bincount(np.asarray(train[left:left+1_000_000], dtype=np.int64), minlength=len(counts))
    log_prior = torch.tensor(np.log(counts / counts.sum()), dtype=features.dtype)
    with torch.no_grad():
        normalized = F.rms_norm(features, (features.shape[-1],), weights['read_norm.weight'])
        residual = F.linear(normalized, weights['decoder.weight'])
        bias = weights['decoder.bias']
        mean_residual = residual.mean(0, keepdim=True)
        predictions = {
            'fixed_train_unigram': log_prior[None].expand(len(targets), -1),
            'learned_bias_only': bias[None].expand(len(targets), -1),
            'saved_head_full': residual + bias,
            'common_read_background_only': mean_residual.expand_as(residual) + bias,
            'variation_without_common_background': residual - mean_residual + bias,
            'residual_with_fixed_unigram_bias': residual + log_prior,
        }
        scores = {name: F.cross_entropy(logits, targets, reduction='none')
                  for name, logits in predictions.items()}
        # Exclude the unmatched bridge in the same way as paired revisit reports.
        summary = {name: float(value[1:].mean()) for name, value in scores.items()}
        gaps = {
            'bias_excess_over_unigram': summary['learned_bias_only']-summary['fixed_train_unigram'],
            'read_residual_excess_over_bias': summary['saved_head_full']-summary['learned_bias_only'],
            'content_variation_excess_over_common_background': summary['saved_head_full']-summary['common_read_background_only'],
        }
        centered = features - features.mean(0, keepdim=True)
        residual_centered = residual - mean_residual
        gradient_decomposition = []
        for left in range(0, len(targets), 32):
            q = normalized[left:left+32].double()
            logits = predictions['saved_head_full'][left:left+32]
            # CE dL/dlogits, without constructing a 50257 x 768 gradient.
            errors = logits.softmax(-1).double()
            errors[torch.arange(len(q)), targets[left:left+len(q)]] -= 1
            mean_error, mean_q = errors.mean(0), q.mean(0)
            centered_error = errors-mean_error
            centered_q = q-mean_q
            count = len(q)
            full_square = float(((errors@errors.T)*(q@q.T)).sum()/count**2)
            common_square = float(mean_error.square().sum()*mean_q.square().sum())
            covariance_square = float(((centered_error@centered_error.T)
                *(centered_q@centered_q.T)).sum()/count**2)
            inner = float(((centered_error@mean_error)*(centered_q@mean_q)).sum()/count)
            reconstructed = common_square+covariance_square+2*inner
            gradient_decomposition.append({
                'events': [left,left+count],
                'decoder_gradient_norm': max(full_square,0)**.5,
                'common_background_gradient_norm': max(common_square,0)**.5,
                'conditional_covariance_gradient_norm': max(covariance_square,0)**.5,
                'common_to_covariance_norm_ratio': (max(common_square,0)/max(covariance_square,1e-30))**.5,
                'common_covariance_cosine': inner/max(common_square*covariance_square,1e-30)**.5,
                'relative_squared_norm_identity_error': abs(full_square-reconstructed)/max(full_square,1e-30),
            })
        report = {
            'scope': __doc__, 'checkpoint_tokens': saved['bptt_train_tokens'],
            'cached_events': len(targets), 'scored_matched_targets': len(targets)-1,
            'actual_active_replay_nll_matched': float(np.mean(matched['A2_replay_curve'][1:])),
            'retrospective_head_nll': summary, 'exact_additive_excess_decomposition': gaps,
            'per_32_event_block_full_minus_common': (scores['saved_head_full']-scores['common_read_background_only']).reshape(-1,32).mean(1).tolist(),
            'feature_temporal_variation_energy_fraction': float(centered.square().sum()/features.square().sum()),
            'logit_temporal_variation_energy_fraction': float(residual_centered.square().sum()/residual.square().sum()),
            'read_norm_gain_min_max_mean': [float(weights['read_norm.weight'].min()), float(weights['read_norm.weight'].max()),float(weights['read_norm.weight'].mean())],
            'bias_delta_from_train_prior_centered_rms': float(((bias-log_prior)-(bias-log_prior).mean()).square().mean().sqrt()),
            'saved_head_ce_gradient_decomposition': gradient_decomposition,
            'gradient_scope': 'Exact saved decoder weight-gradient algebra on cached feature/target pairs, before clip/Adam. Not the motor projection gradient or actual optimizer displacement.',
            'decision_rule': 'If bias matches the reference but full head is worse, excess arises in read residuals. If full exceeds common-only, event variation harms this snapshot; this alone does not locate its upstream cause.',
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
