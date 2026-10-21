"""Match the voice instrument to a reference voice TYPE (not a clone): measure a few acoustic statistics of the
reference recordings, then search the instrument's innate parameters (tract length, pitch, pitch range, open
quotient, return-phase corner, breath) so that its babble lands in the same region.  No model is trained on the
references; only these statistics are used.
Statistics: median pitch and its spread (semitones), formants F1-F4 (LPC) -> vocal tract length by formant
dispersion, harmonics-to-noise ratio, H1-H2 (open-quotient proxy), long-term spectral slope 0.5-8 kHz (dB/oct),
energy share above 4 kHz.
Usage: python voice_match.py <ref dir with sub-dirs per voice of .mp3/.wav> [--out results/voice_demo_v3]"""
import argparse, glob, itertools, json, os, sys
import numpy as np, soundfile as sf, librosa
from scipy.signal import lfilter, welch
import scipy.linalg as sl
sys.path.insert(0, os.path.dirname(__file__))
import voice_demo as vd

C = 350.0


def lpc_formants(frame, sr, order=12):
    x = lfilter([1, -0.97], [1], frame * np.hamming(len(frame)))
    r = np.correlate(x, x, 'full')[len(x) - 1:len(x) + order]
    if r[0] <= 0:
        return []
    a = sl.solve_toeplitz(r[:order], -r[1:order + 1])
    roots = np.roots(np.r_[1, a])
    roots = roots[np.imag(roots) > 0]
    f = np.angle(roots) * sr / (2 * np.pi)
    bw = -sr / np.pi * np.log(np.abs(roots))
    return sorted(v for v, b in zip(f, bw) if 200 < v < 5000 and b < 400)


def analyze(y, sr):
    y = y.mean(1) if y.ndim > 1 else y
    stats = {}
    f, P = welch(y, sr, nperseg=4096)
    stats['hf_share_4k'] = float(P[f > 4000].sum() / P.sum())
    band = (f > 500) & (f < 8000)
    stats['slope_db_oct'] = float(np.polyfit(np.log2(f[band]), 10 * np.log10(P[band] + 1e-20), 1)[0])
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    f0, voiced, _ = librosa.pyin(y16, fmin=80, fmax=900, sr=16000, frame_length=1024, hop_length=160)
    f0v = f0[voiced & np.isfinite(f0)]
    if len(f0v) < 20:
        return None
    stats['f0_median'] = float(np.median(f0v))
    stats['f0_semitone_sd'] = float(np.std(12 * np.log2(f0v / np.median(f0v))))
    y11 = librosa.resample(y16, orig_sr=16000, target_sr=11025)
    fr, hop = 276, 110                                   # 25 ms frames, 10 ms hop at 11.025 kHz
    forms, hnr, h1h2 = [], [], []
    for i, (v, ff) in enumerate(zip(voiced, f0)):
        if not v or not np.isfinite(ff):
            continue
        s = int(i * 0.01 * 11025)
        seg = y11[s:s + fr]
        if len(seg) < fr or np.abs(seg).max() < 1e-3:
            continue
        fm = lpc_formants(seg, 11025)
        if len(fm) >= 4:
            forms.append(fm[:4])
        s16 = int(i * 160)
        seg16 = y16[s16:s16 + 1024]
        if len(seg16) < 1024:
            continue
        lag = int(round(16000 / ff))
        ac = np.correlate(seg16, seg16, 'full')[1023:]
        if ac[0] > 0 and lag < len(ac):
            rr = np.clip(ac[lag - 2:lag + 3].max() / ac[0], 1e-3, 0.999)
            hnr.append(10 * np.log10(rr / (1 - rr)))
        spec = np.abs(np.fft.rfft(seg16 * np.hanning(1024), 4096))
        fbin = 16000 / 4096
        h = lambda fq: 20 * np.log10(spec[max(1, int(fq / fbin) - 2):int(fq / fbin) + 3].max() + 1e-12)
        h1h2.append(h(ff) - h(2 * ff))
    if len(forms) < 10:
        return None
    F = np.median(np.array(forms), 0)
    stats['formants'] = [float(v) for v in F]
    stats['tract_cm'] = float(np.mean([(2 * k + 1) * C / (4 * F[k]) for k in range(4)]) * 100)
    stats['hnr_db'] = float(np.median(hnr))
    stats['h1_h2_db'] = float(np.median(h1h2))
    return stats


def summarize(rows):
    keys = ['f0_median', 'f0_semitone_sd', 'tract_cm', 'hnr_db', 'h1_h2_db', 'slope_db_oct', 'hf_share_4k']
    out = {k: float(np.median([r[k] for r in rows])) for k in keys}
    out['formants'] = [float(v) for v in np.median(np.array([r['formants'] for r in rows]), 0)]
    out['clips'] = len(rows)
    return out


def render_stats(params):
    y, _ = vd.render(params)
    return analyze(y, vd.FS_OUT), y


def fit(target, quick=False):
    base = dict(L=min(.175, max(.10, target['tract_cm'] / 100)), f0=target['f0_median'], jitter=.004, shimmer=.025,
                vib=.008)
    # pitch range: the babble phrase's own spread is ~2.3 semitones at range 1
    base['range'] = float(np.clip(target['f0_semitone_sd'] / 2.3, .5, 3.0))
    grid = itertools.product([.55, .65, .75, .85], [500, 700, 1000, 1500, 2200], [.03, .05, .08, .12])
    best = None
    for oq, fa, br in grid:
        p = dict(base, oq=oq, fa=fa, breath=br)
        st, _ = render_stats(p)
        if st is None:
            continue
        d = ((st['h1_h2_db'] - target['h1_h2_db']) / 3) ** 2 + ((st['hnr_db'] - target['hnr_db']) / 3) ** 2 + \
            ((st['slope_db_oct'] - target['slope_db_oct']) / 2) ** 2 + \
            ((st['hf_share_4k'] - target['hf_share_4k']) / max(.01, target['hf_share_4k'])) ** 2
        if best is None or d < best[0]:
            best = (d, p, st)
    if best is None:
        return best
    d, p, st = best                                      # one correction of the pitch range from the measured spread
    p = dict(p, range=float(np.clip(p['range'] * target['f0_semitone_sd'] / max(.3, st['f0_semitone_sd']), .5, 4)))
    st, _ = render_stats(p)
    return d, p, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('--out', default='results/voice_demo_v3')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = {}
    fitted = {}
    for d in sorted(glob.glob(os.path.join(args.ref, '*'))):
        if not os.path.isdir(d):
            continue
        rows = []
        for f in sorted(glob.glob(os.path.join(d, '*.mp3')) + glob.glob(os.path.join(d, '*.wav'))):
            y, sr = sf.read(f)
            st = analyze(y, sr)
            if st:
                rows.append(st)
        name = os.path.basename(d)
        tgt = summarize(rows)
        print(name, json.dumps(tgt, ensure_ascii=False), flush=True)
        dist, p, st = fit(tgt)
        fitted[name] = p
        report[name] = {'reference': tgt, 'instrument': p, 'instrument_measured': st, 'distance': dist}
        print('  fitted', p, '\n  measured', json.dumps(st), flush=True)
    keys = ['L', 'f0', 'range', 'oq', 'fa', 'breath', 'jitter', 'shimmer', 'vib']
    if len(fitted) >= 2:
        avg = {k: float(np.exp(np.mean([np.log(v[k]) for v in fitted.values()]))) if k in ('f0', 'fa')
               else float(np.mean([v[k] for v in fitted.values()])) for k in keys}
        fitted['average'] = avg
        report['average'] = {'instrument': avg}
    for i, (name, p) in enumerate(fitted.items()):
        y, _ = vd.render(p)
        label = {'瑶': 'yao_type', '西施': 'xishi_type'}.get(name, name if name == 'average' else f'voice{i + 1}')
        fn = os.path.join(args.out, f'{i + 1:02d}_{label}.wav')
        sf.write(fn, y, vd.FS_OUT, subtype='PCM_16')
        report.setdefault(name, {})['file'] = fn
        print('wrote', fn, name)
    json.dump(report, open(os.path.join(args.out, 'match.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
