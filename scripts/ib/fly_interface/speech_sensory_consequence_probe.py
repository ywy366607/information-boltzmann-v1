"""Paired speech-vs-silence consequence probe.

Replay recorded mother replies from the same checkpoint.  The two branches differ
only in whether the preceding fly utterance is written into the brain; the mother
reply is then identical.  This is a necessary-condition probe for epistemic speech:
if speaking changes nothing in the later sensory/internal state, no active-inference
credit can be meaningful yet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
exec(open(ROOT / 'scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])


def clone_state(s):
    return fl.FlyPhysicalState(**{
        k: (tuple(x.detach().clone() for x in v) if k == 'ring' else v.detach().clone())
        for k, v in s.state_dict().items()
    })


def load_pairs(run, tok, limit):
    rows = [json.loads(x) for x in (Path(run) / 'mother.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    pairs = []
    for i, row in enumerate(rows[:-1]):
        if row.get('who') != 'fly' or rows[i + 1].get('who') != 'mother':
            continue
        fly = str(row.get('text') or '').strip()
        mother = str(rows[i + 1].get('text') or '').strip()
        if not fly or fly == '...' or not mother or mother == '...':
            continue
        f = tok.encode(' ' + fly)
        m = tok.encode(' ' + mother)
        if f and m:
            pairs.append((f, m, int(row.get('trained') or 0)))
    return pairs[-limit:]


def opts_for(model, state):
    opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(),
                conductance_gains=model.get_conductance_gains(), alif_params=model.get_alif_params(),
                stp_params=model.get_stp_params())
    if getattr(model, 'dopamine_sensitivity', None) is not None:
        opts['dopamine_s_class'] = fl.dopamine_class_sensitivity(model, state.h)
    return opts


def run_branch(model, start, previous, fly_tokens, mother_tokens, word_ticks, say, opts):
    s = clone_state(start)
    prev = int(previous)
    first = True
    mother_ce, mother_spk, mother_da, mother_mbon = [], [], [], []
    n = model.n_neurons
    tok_t = lambda x: torch.tensor([int(x)], device=s.h.device)

    def advance(token, sensory_on, voice):
        nonlocal s, first
        s, latent = fl.advance_fly_token_stream(
            model, s, tok_t(token), [1], ticks=word_ticks, hold_input=True,
            fresh=first, sensory_on=sensory_on, voice=voice, **opts)[:2]
        first = False
        s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
        return latent

    # Keep physical time equal in both branches.  In the silent branch no text or voice
    # is written, but the same number of ticks is allowed to pass.
    if say:
        for w in fly_tokens:
            advance(prev, True, False)
            prev = int(w)
    else:
        for _ in fly_tokens:
            advance(prev, False, False)

    for w in mother_tokens:
        latent = advance(prev, True, True)
        logits = model.decoder(model.read_norm(latent)).float()
        mother_ce.append(float(torch.nn.functional.cross_entropy(logits, tok_t(w))))
        spk = s.ring[0][0] > 0
        mother_spk.append(float(spk.float().mean()))
        if getattr(model, 'dan_indices', None) is not None:
            mother_da.append(float(spk.index_select(0, model.dan_indices).float().mean()))
        if getattr(model, 'read_indices', None) is not None:
            mother_mbon.append(float(spk.index_select(0, model.read_indices).float().mean()))
        prev = int(w)
    return {
        'ce_mean': float(np.mean(mother_ce)) if mother_ce else None,
        'ce_sum': float(np.sum(mother_ce)) if mother_ce else None,
        'spike_mean': float(np.mean(mother_spk)) if mother_spk else None,
        'dopamine_spike_mean': float(np.mean(mother_da)) if mother_da else None,
        'read_surface_spike_mean': float(np.mean(mother_mbon)) if mother_mbon else None,
        'mother_tokens': len(mother_tokens),
    }


def main():
    if len(sys.argv) < 4:
        raise SystemExit('usage: speech_sensory_consequence_probe.py <checkpoint-dir> <run-dir> <out.json> [pairs]')
    fork, run, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    model, cfg, L = load(fork)
    model.eval()
    start = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    opts = opts_for(model, start)
    import tiktoken
    tok = tiktoken.get_encoding('gpt2')
    pairs = load_pairs(run, tok, limit)
    ticks = int(round(float(cfg.get('stream_pace_mean') or cfg.get('stream_ticks') or 8)))
    rows = []
    with torch.no_grad():
        for fly, mother, trained in pairs:
            silent = run_branch(model, start, L.get('previous_token', 0), fly, mother, ticks, False, opts)
            spoken = run_branch(model, start, L.get('previous_token', 0), fly, mother, ticks, True, opts)
            rows.append({'trained': trained, 'fly_tokens': len(fly), 'spoken': spoken, 'silent': silent,
                         'delta_spoken_minus_silent': {
                             k: (spoken[k] - silent[k]) if spoken.get(k) is not None and silent.get(k) is not None else None
                             for k in ('ce_mean', 'ce_sum', 'spike_mean', 'dopamine_spike_mean', 'read_surface_spike_mean')}})
    deltas = [r['delta_spoken_minus_silent'] for r in rows]
    rep = {'pairs': len(rows), 'word_ticks': ticks, 'rows': rows,
           'mean_delta': {k: float(np.mean([d[k] for d in deltas if d[k] is not None])) if any(d[k] is not None for d in deltas) else None
                          for k in ('ce_mean', 'ce_sum', 'spike_mean', 'dopamine_spike_mean', 'read_surface_spike_mean')},
           'scope': 'fixed mother replies; necessary-condition speech consequence, not an environmental counterfactual'}
    out.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps({k: rep['mean_delta'][k] for k in rep['mean_delta']}, indent=1))


if __name__ == '__main__':
    main()
