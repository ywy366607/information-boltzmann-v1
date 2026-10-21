"""Test: does a body with inertia make even random motor commands sound continuous?  (2026-10-19)
Hypothesis: biological speech is smooth because spikes reach the articulators through a population of motor units
(recruited by size) whose twitches low-pass the commands, and the articulators are masses on damped springs; so no
command, however abrupt, can make the tract jump.  Our earlier voices jumped because targets were written straight
into the tube.
Body arm: random drives (Ornstein-Uhlenbeck) -> per muscle 40 motor units, size-ordered thresholds, Poisson spikes,
twitch t/tau*exp(1-t/tau) (tau 30-50 ms, larger units stronger and faster) -> agonist minus antagonist force ->
second-order articulator (6 Hz, damping 0.9) -> the same control space as voice_imitate (tongue body x4, lips, tip,
velum, adduction) plus pitch (cricothyroid) and breath (subglottal pressure, breath groups).
Turbulence is not commanded: it arises at a constriction that is narrow but open while air flows.
No-body arm: the same random drives, sampled every 120 ms and written straight into the tube (10 ms smoothing, as the
old rule score did).
Measures: click proxy (max |dy| / local rms), spectral flux (log-mel change per 10 ms, 99th percentile over loud
frames), against the reference recordings on the same scale.
Usage: python voice_babble.py [--ref <dir of reference clips>] [--out results/voice_babble] [--seconds 10]"""
import argparse, glob, json, os, sys
import numpy as np, soundfile as sf, librosa
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(__file__))
import voice_speak as vs
import voice_imitate as vi
import torch

FS, FS_OUT = vs.FS, vs.FS_OUT
DOFS = ['pharynx', 'place', 'constr', 'oral', 'lips', 'tip', 'velum', 'adduction', 'pitch', 'breath']


def ou(T, tau_ms, rng, mean=.5, sd=.25):
    x = np.zeros(T); x[0] = mean
    a = np.exp(-1 / tau_ms)
    for t in range(1, T):
        x[t] = mean + a * (x[t - 1] - mean) + sd * np.sqrt(1 - a * a) * rng.standard_normal()
    return np.clip(x, 0, 1)


def muscle_force(drive, rng, M=40):
    """population of M motor units (size principle) -> summed twitch force in [0, ~1], 1 ms steps"""
    T = len(drive)
    thr = np.linspace(.03, .9, M)
    size = np.exp(np.linspace(0, np.log(25), M)); size /= size.sum()
    tau = np.linspace(50, 30, M)
    F = np.zeros(T)
    for i in range(M):
        rate = np.where(drive > thr[i], 8 + 27 * (drive - thr[i]) / (1 - thr[i]), 0.)     # Hz
        spikes = rng.random(T) < rate / 1000
        t = np.arange(int(6 * tau[i]))
        k = t / tau[i] * np.exp(1 - t / tau[i])
        F += size[i] * np.convolve(spikes.astype(float), k)[:T] * (tau[i] / 40)
    return F


def articulator(F, f_hz=6., zeta=.9, gain=1.):
    """mass on a damped spring around rest .5, soft tissue limits at 0 and 1"""
    w = 2 * np.pi * f_hz / 1000
    x, v = .5, 0.
    out = np.zeros(len(F))
    for t in range(len(F)):
        lim = 40 * (max(0, x - 1) - max(0, -x))                  # tissue stiffening beyond the range
        acc = w * w * (.5 - x) - 2 * zeta * w * v + gain * w * w * F[t] - w * w * lim
        v += acc; x += v
        out[t] = x
    return np.clip(out, 0, 1)


def breath_drive(T, rng):
    """breath groups: exhalation 1.2-2.5 s (pressure on), inhalation .3-.5 s (off)"""
    d = np.zeros(T); t = 200
    while t < T:
        n = int(rng.uniform(1200, 2500)); d[t:t + n] = .85; t += n + int(rng.uniform(300, 500))
    return d


def to_controls(X):
    """X (T, 10) in [0, 1] -> physical controls (T, 11) of voice_imitate, pitch in semitones, flow"""
    T = len(X)
    P = np.zeros((T, vi.NPAR))
    for i in range(7):
        lo, hi, lg = vi.RANGES[i]
        v = lo + (hi - lo) * X[:, i]
        P[:, i] = 10 ** v if lg else v
    flow = X[:, 9]                                                  # subglottal pressure
    P[:, 7] = 1.1 * flow * np.clip((X[:, 7] - .2) / .6, 0, 1)       # voicing: pressure x adduction
    P[:, 8] = .6 * flow * np.clip((.35 - X[:, 7]) / .35, 0, 1)      # aspiration: pressure through an open glottis
    # turbulence where the tip or the tongue body is narrow but open, scaled by the pressure
    nar = lambda a: np.exp(-((np.log10(np.maximum(a, 1e-4)) - np.log10(.08)) / .45) ** 2)
    P[:, 9] = .5 * flow * nar(P[:, 5])
    P[:, 10] = .5 * flow * nar(P[:, 2])
    st = (X[:, 8] - .5) * 14                                        # +-7 semitones
    return P, st


def render_ms(P, st, seed=0, room=True):
    """1 ms controls -> sound through the voice_speak instrument (no smoothing added)"""
    T = len(P)
    with torch.no_grad():
        A = vi.area_fn(*[torch.as_tensor(P[:, i]) for i in range(6)]).numpy()
    nT = T * vs.HOP
    ts = np.arange(nT) / vs.HOP
    idx = np.arange(T)
    f0 = vs.BODY['f0'] * 2 ** (np.interp(ts, idx, st) / 12)
    f0 *= 1 + vs.BODY['vib'] * np.sin(2 * np.pi * 5.5 * ts / 1000)
    av = np.interp(ts, idx, P[:, 7]); asp = np.interp(ts, idx, P[:, 8])
    B = vs.BODY
    src = vs.source(f0, av, asp, B['oq'], B['breath'], B['jitter'], B['shimmer'], B['fa'], float(FS), seed,
                    vs.ASP_GAIN)
    fpos = np.stack([np.full(T, vi.J_TIP), vi.j_body(P[:, 1])], 1).astype(np.int64)
    out = vs.tract_nasal(np.ascontiguousarray(A), np.ascontiguousarray(P[:, 6]), fpos, np.ascontiguousarray(P[:, 9:11]),
                         float(vs.HOP), src, vi.GLOT, vi.LIP, vi.DAMP, vi.END_LP, vi.JV, vi.NOSE, seed, vs.FRIC_GAIN)
    y = resample_poly(np.diff(out, prepend=0.0), 1, FS // FS_OUT)
    y -= y.mean()
    if room:
        y = vs.room(np.r_[y, np.zeros(int(.3 * FS_OUT))], FS_OUT)
    return (.8 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def measures(y, sr=FS_OUT):
    y = y.astype(float)
    d = np.abs(np.diff(y)); loc = np.sqrt(np.convolve(y ** 2, np.ones(480) / 480, 'same'))[1:] + 1e-4
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=480, n_mels=64, fmin=100, fmax=11000)
    L = 10 * np.log10(S + 1e-10)
    loud = L.max(0) > L.max() - 35
    flux = np.sqrt(np.mean(np.diff(L, axis=1) ** 2, 0))
    lf = flux[loud[1:] & loud[:-1]]
    return {'click_proxy_max': float((d / loc).max()), 'click_proxy_p999': float(np.percentile(d / loc, 99.9)),
            'flux_median_dB': float(np.median(lf)), 'flux_p99_dB': float(np.percentile(lf, 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default=None)
    ap.add_argument('--out', default='results/voice_babble')
    ap.add_argument('--seconds', type=float, default=10)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    T = int(args.seconds * 1000)
    taus = [250, 200, 150, 250, 150, 120, 300, 200, 300]            # how fast each command wanders (ms)
    drives = [(ou(T, tau, rng), ou(T, tau, rng)) for tau in taus]   # agonist, antagonist
    # body arm
    X = np.zeros((T, 10))
    for i, (ag, an) in enumerate(drives):
        f = muscle_force(ag, rng) - muscle_force(an, rng)
        X[:, i] = articulator(f, f_hz=4. if DOFS[i] == 'pitch' else 6., gain=.3)
    X[:, 9] = articulator(muscle_force(breath_drive(T, rng), rng) / 1.6 - .5, f_hz=3., gain=1.)
    P, st = to_controls(X)
    yb = render_ms(P, st, seed=args.seed)
    # no-body arm: the same drives sampled every 120 ms, written straight in (10 ms smoothing)
    Xn = np.zeros((T, 10))
    for i, (ag, an) in enumerate(drives):
        v = np.clip(.5 + 1.2 * (ag - an), 0, 1)
        held = np.repeat(v[::120], 120)[:T]
        Xn[:, i] = np.convolve(np.pad(held, 5, mode='edge'), np.ones(10) / 10, 'same')[5:-5]
    Xn[:, 9] = np.repeat((breath_drive(T, np.random.default_rng(args.seed + 1)) > .5).astype(float) * .9, 1)
    Pn, stn = to_controls(Xn)
    yn = render_ms(Pn, stn, seed=args.seed)
    sf.write(os.path.join(args.out, 'babble_body.wav'), yb, FS_OUT, subtype='PCM_16')
    sf.write(os.path.join(args.out, 'babble_nobody.wav'), yn, FS_OUT, subtype='PCM_16')
    rep = {'body': measures(yb), 'no_body': measures(yn)}
    if args.ref:
        rows = []
        for f in sorted(glob.glob(os.path.join(args.ref, '*', '*.mp3'))):
            y, sr = sf.read(f)
            y = y.mean(1) if y.ndim > 1 else y
            rows.append(measures(librosa.resample(y, orig_sr=sr, target_sr=FS_OUT)))
        rep['reference'] = {k: {'min': float(np.min([r[k] for r in rows])), 'median': float(np.median([r[k] for r in rows])),
                                'mean': float(np.mean([r[k] for r in rows])), 'max': float(np.max([r[k] for r in rows]))}
                            for k in rows[0]}
        rep['reference']['clips'] = len(rows)
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(args.out, 'report.json'), 'w'), indent=1)
    np.savez(os.path.join(args.out, 'trajectories.npz'), body=X, no_body=Xn, names=DOFS)


if __name__ == '__main__':
    main()
