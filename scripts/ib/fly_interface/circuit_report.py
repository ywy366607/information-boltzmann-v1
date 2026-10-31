"""Which circuits move with language events in the living run (reads fly_circuit_observer output).

Per coarse group: rate distribution [min | median | mean | max], Spearman rho of the per-word rate with the word's
surprisal (the brain's own prediction error), and the contrast of the rate between who spoke (mother vs the fly's
own word heard back) and at her reply onsets vs other mother words, each with a permutation p-value (word labels
shuffled; rows are consecutive words, so p is optimistic under autocorrelation -- a screen, not a causal claim).
Per fine type: the strongest speaker and surprisal effects. Causality needs living ablations of the named groups.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def dist(x):
    x = np.asarray(x, float)
    return {'min': float(x.min()), 'median': float(np.median(x)), 'mean': float(x.mean()), 'max': float(x.max())}


def contrast(values, mask, rng, n=2000):
    values = np.asarray(values, float)
    if mask.sum() < 3 or (~mask).sum() < 3:
        return None
    observed = values[mask].mean() - values[~mask].mean()
    scale = values.std() + 1e-12
    count = 0
    for _ in range(n):
        perm = rng.permutation(mask)
        count += abs(values[perm].mean() - values[~perm].mean()) >= abs(observed)
    return {'difference': float(observed), 'standardized': float(observed / scale), 'p_perm': (count + 1) / (n + 1)}


def main(args):
    rows = [json.loads(line) for line in (args.observer / 'circuits.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = json.loads((args.observer / 'groups.json').read_text(encoding='utf-8'))
    rng = np.random.default_rng(0)
    surprisal = np.array([r['surprisal'] if r['surprisal'] is not None else np.nan for r in rows])
    fly = np.array([r['who'] == 'fly' for r in rows])
    mother = ~fly
    onset = np.array([bool(r.get('reply_onset')) for r in rows])
    ok = np.isfinite(surprisal)
    report = {'words': len(rows), 'fly_words': int(fly.sum()), 'reply_onsets': int(onset.sum()),
              'surprisal': dist(surprisal[ok]) if ok.any() else None,
              'surprisal_mother': dist(surprisal[ok & mother]) if (ok & mother).any() else None,
              'da_mbon_fast': dist([r['da_mbon_fast'] for r in rows]), 'coarse': {}}
    for name, size in zip(groups['coarse'], groups['coarse_size']):
        rate = np.array([r['rates'][name] for r in rows])
        entry = {'cells': size, 'rate': dist(rate)}
        if ok.sum() > 5 and rate[ok].std() > 0:
            rho, p = spearmanr(rate[ok], surprisal[ok])
            entry['surprisal_rho'], entry['surprisal_p'] = float(rho), float(p)
        entry['fly_vs_mother'] = contrast(rate, fly, rng)
        if mother.sum() > 5:
            entry['onset_vs_other_mother'] = contrast(rate[mother], onset[mother], rng)
        report['coarse'][name] = entry
    # the three couplings that failed before the native wiring (HANDOFF_20261016):
    # gate 2 (dopamine carries the word's surprise), familiarity reaching the MBONs, speaking vs silence
    def rho(a, b, mask):
        a, b = np.asarray(a, float)[mask], np.asarray(b, float)[mask]
        if mask.sum() < 6 or a.std() == 0 or b.std() == 0:
            return None
        r, pv = spearmanr(a, b)
        return {'rho': float(r), 'p': float(pv), 'n': int(mask.sum())}
    level = lambda name: np.array([r.get('levels', r['rates']).get(name, np.nan) for r in rows])
    rate = lambda name: np.array([r['rates'].get(name, np.nan) for r in rows])
    heard = ok & mother
    seen, repeats = {}, []
    for r in rows:
        repeats.append(seen.get(r['token'], 0)); seen[r['token']] = seen.get(r['token'], 0) + 1
    repeats = np.array(repeats)
    report['couplings'] = {
        'gate2_surprisal_vs_PAM_rate': rho(surprisal, rate('DAN_PAM'), heard),
        'gate2_surprisal_vs_PPL1_rate': rho(surprisal, rate('DAN_PPL1'), heard),
        'gate2_surprisal_vs_mbon_dopamine': rho(surprisal, [r['da_mbon_fast'] for r in rows], heard),
        'familiarity_repeats_vs_MBON_rate': rho(repeats, rate('MBON'), mother),
        'familiarity_repeats_vs_KC_rate': rho(repeats, rate('KC'), mother),
        'familiarity_repeats_vs_surprisal': rho(repeats, surprisal, heard),
        'surprisal_vs_word_order (learning within the run)': rho(np.arange(len(rows)), surprisal, heard)}
    mother_log = args.observer.parent / 'out' / 'mother.jsonl'
    if mother_log.exists():
        turns = [json.loads(l) for l in mother_log.read_text(encoding='utf-8').splitlines() if l.strip()]
        fly_turns = [t for t in turns if t.get('who') == 'fly']
        report['speaking'] = {'fly_turns': len(fly_turns),
                              'quiet_yields': sum(t.get('turn_kind') == 'quiet_yield' for t in fly_turns),
                              'spoken_turns': sum(bool(t.get('spoken_text')) for t in fly_turns),
                              'mother_turns': sum(t.get('who') == 'mother' for t in turns),
                              'fly_words_heard_back': int(fly.sum())}
    fine_files = sorted(args.observer.glob('fine_*.npz'))
    if fine_files:
        fine = np.concatenate([np.load(f)['rates'].astype(np.float32) for f in fine_files])
        k = len(fine)
        names = np.asarray(groups['fine'])
        sizes = np.asarray(groups['fine_size'])
        active = fine.mean(0) > 0
        z = (fine - fine.mean(0)) / (fine.std(0) + 1e-9)
        fl, ok_k = fly[:k], ok[:k]
        speaker = z[fl].mean(0) - z[~fl].mean(0) if fl.any() and (~fl).any() else np.zeros(len(names))
        s = surprisal[:k][ok_k]
        sz = (s - s.mean()) / (s.std() + 1e-9)
        surpr = (z[ok_k] * sz[:, None]).mean(0) if ok_k.sum() > 5 else np.zeros(len(names))
        pick = lambda score: [(str(names[i]), int(sizes[i]), round(float(score[i]), 3))
                              for i in np.argsort(-np.abs(np.where(active, score, 0)))[:args.top]]
        report['fine'] = {'words': k, 'active_types': int(active.sum()), 'types': len(names),
                          'speaker_effect_top': pick(speaker), 'surprisal_correlation_top': pick(surpr)}
    out = args.output or args.observer / 'circuit_report.json'
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')
    ranked = sorted(((n, e) for n, e in report['coarse'].items() if 'surprisal_rho' in e),
                    key=lambda t: -abs(t[1]['surprisal_rho']))
    print(json.dumps({k: report.get(k) for k in ('words', 'fly_words', 'reply_onsets', 'surprisal', 'couplings', 'speaking')}, indent=1))
    for name, e in ranked[:15]:
        fm = e['fly_vs_mother']
        print(f"{name:32s} cells {e['cells']:6d} rate mean {e['rate']['mean']:.4f} rho {e['surprisal_rho']:+.3f} "
              f"fly-mother {fm['standardized'] if fm else float('nan'):+.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('observer', type=Path)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--top', type=int, default=25)
    main(p.parse_args())
