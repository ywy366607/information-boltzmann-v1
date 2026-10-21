"""The life-form's voice instrument, first listening demo (2026-10-19).  No learning, CPU only.
A physical voice: a KLGLOTT88 glottal source (open quotient, breath noise, jitter, shimmer, vibrato, spectral tilt)
drives a Kelly-Lochbaum tube vocal tract (one-dimensional waveguide, segment count set by the tract length, so a
short infant tract has high formants), radiated at the lips.  Articulation = area functions for a few vowels and lip /
tongue closures, smoothed in time; the brain will later drive these same parameters with its motor units.
Writes one babble phrase per voice setting to results/voice_demo/*.wav (48 kHz, 16 bit).
Usage: python voice_demo.py [--out results/voice_demo]"""
import argparse, os, numpy as np, soundfile as sf
from numba import njit
from scipy.signal import resample_poly

FS_OUT, OVER = 48000, 2
FS = FS_OUT * OVER          # the waveguide runs at 96 kHz: one segment per step
C = 350.0                   # speed of sound in warm humid air, m/s
HOP = FS // 1000            # control frames every 1 ms
END_LP = 0.5                # low-pass on the end reflections (frequency-dependent losses)

# vowel area functions (relative cm^2) along the tract, x = 0 glottis .. 1 lips: piecewise-linear knots
VOWELS = {
    'a': ([0, .08, .2, .45, .55, .7, .95, 1], [.5, .6, .5, .6, 2.5, 6.5, 7, 6]),
    'i': ([0, .08, .2, .45, .62, .72, .82, 1], [.5, .6, 7, 7, 3, .4, .5, 2.5]),
    'u': ([0, .08, .2, .45, .55, .65, .75, .9, 1], [.5, .6, 5, 5, 1.2, .8, 4, 4, .6]),
    'e': ([0, .08, .2, .45, .62, .75, .85, 1], [.5, .6, 5, 5, 3, 1.2, 1.5, 3]),
    'o': ([0, .08, .2, .45, .6, .75, .9, 1], [.5, .6, 2.5, 2.5, 1.5, 3.5, 3, 1.2]),
    '@': ([0, .08, .2, 1], [.5, .6, 3, 3]),
}
CLOSURES = {'b': (.94, 1.0), 'd': (.86, .93)}      # region of the tract closed by the lips / tongue tip

# one babble phrase: (kind, target, seconds, pitch multiplier)
PHRASE = [('p', '@', .25, 1), ('v', 'a', .30, 1.05), ('v', 'i', .26, 1.18), ('v', 'a', .42, .95), ('p', '@', .2, 1),
          ('c', 'b', .06, 1), ('v', 'a', .22, 1.12), ('c', 'b', .06, 1), ('v', 'a', .34, .96), ('p', '@', .22, 1),
          ('v', 'u', .28, 1.0), ('v', 'o', .26, 1.04), ('v', 'a', .44, .92), ('p', '@', .2, 1),
          ('c', 'd', .05, 1), ('v', 'a', .2, 1.1), ('c', 'd', .05, 1), ('v', 'i', .26, 1.0), ('p', '@', .22, 1),
          ('v', 'a', .22, .98), ('v', 'e', .24, 1.08), ('v', 'i', .42, 1.3), ('p', '@', .3, 1)]   # a rising "a-e-i?"

VOICES = {   # tract length (m), mean pitch (Hz), open quotient, breath, jitter, shimmer, return-phase corner fa (Hz),
             # vibrato depth
    '01_infant_clean':      dict(L=.095, f0=420, oq=.65, breath=.05, jitter=.006, shimmer=.03, fa=2200, vib=.0),
    '02_infant_breathy':    dict(L=.095, f0=420, oq=.7, breath=.14, jitter=.008, shimmer=.04, fa=1600, vib=.0),
    '03_toddler':           dict(L=.115, f0=310, oq=.62, breath=.07, jitter=.006, shimmer=.03, fa=1800, vib=.0),
    '04_adult_high_warm':   dict(L=.150, f0=215, oq=.6, breath=.08, jitter=.004, shimmer=.025, fa=1200, vib=.012),
    '05_adult_high_bright': dict(L=.145, f0=230, oq=.5, breath=.04, jitter=.004, shimmer=.02, fa=2400, vib=.008),
    '06_adult_mid_neutral': dict(L=.160, f0=165, oq=.55, breath=.06, jitter=.004, shimmer=.025, fa=1500, vib=.008),
    '07_adult_low_warm':    dict(L=.172, f0=115, oq=.5, breath=.06, jitter=.005, shimmer=.03, fa=1000, vib=.01),
}


def area_profile(target, n, kind):
    x = (np.arange(n) + .5) / n
    knots, vals = VOWELS['@' if kind == 'c' else target]
    a = np.interp(x, knots, vals)
    if kind == 'c':                              # a closure on top of the neutral tract
        lo, hi = CLOSURES[target]
        a[(x >= lo) & (x <= hi)] = .02
    return a


def smooth(x, ms):
    k = max(1, int(ms))
    w = np.ones(k) / k
    pad = np.pad(x, [(k, k)] + [(0, 0)] * (x.ndim - 1), mode='edge')
    if x.ndim == 1:
        return np.convolve(pad, w, mode='same')[k:-k]
    return np.stack([np.convolve(pad[:, j], w, mode='same')[k:-k] for j in range(x.shape[1])], 1)


def controls(voice, n_seg, rng):
    frames_a, f0m, amp = [], [], []
    for kind, tgt, dur, mult in PHRASE:
        nf = int(dur * 1000 / voice.get('tempo', 1.0))     # tempo > 1: faster, lighter speech
        frames_a += [area_profile(tgt, n_seg, kind)] * nf
        f0m += [mult] * nf
        amp += [{'v': 1.0, 'c': .15, 'p': 0.0}[kind]] * nf
    A = smooth(np.array(frames_a), 30)
    f0m = smooth(np.array(f0m, float), 80)
    f0m = 1 + voice.get('range', 1.0) * (f0m - 1)        # pitch range (expressiveness)
    amp = smooth(np.array(amp, float), 25)
    t = np.arange(len(f0m)) / 1000.0
    f0 = voice['f0'] * f0m * (1 - .08 * t / t[-1]) * (1 + voice['vib'] * np.sin(2 * np.pi * 5.5 * t))
    f0 *= 1 + .01 * smooth(rng.standard_normal(len(t)), 200)      # slow natural drift
    return A, f0, amp


@njit(cache=True)
def glottal_source(f0, amp, oq, breath, jitter, shimmer, fa, fs, seed):
    """KLGLOTT88 flow; the closure's return phase (LF model's Ta) as a one-pole low-pass at fa Hz, which sets how
    warm (low fa) or bright (high fa) the source is; breath noise band-limited to 0.4-4 kHz (turbulence at the open
    glottis, not white hiss; two poles above 4 kHz because the lips' radiation lifts highs by 6 dB/oct), gated by the
    glottal opening."""
    np.random.seed(seed)
    T = f0.shape[0]
    out = np.zeros(T)
    a_ret = np.exp(-2.0 * np.pi * fa / fs)
    a_nlo = np.exp(-2.0 * np.pi * 4000.0 / fs)        # noise low-pass 4 kHz, two poles
    a_nhi = np.exp(-2.0 * np.pi * 400.0 / fs)         # noise high-pass 400 Hz
    phase, jit, shim, y, nl, nl2, nh = 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0
    for n in range(T):
        phase += f0[n] * jit / fs
        if phase >= 1.0:
            phase -= 1.0
            jit = 1.0 + jitter * np.random.randn()
            shim = 1.0 + shimmer * np.random.randn()
        u = 0.0
        if phase < oq:
            t = phase / oq
            u = 6.75 * t * t * (1.0 - t)                 # KLGLOTT88 flow, peak 1 at t = 2/3
        w = np.random.randn()
        nl = (1.0 - a_nlo) * w + a_nlo * nl
        nl2 = (1.0 - a_nlo) * nl + a_nlo * nl2
        nh = (1.0 - a_nhi) * nl2 + a_nhi * nh
        noise = (nl2 - nh) * 2.0
        x = amp[n] * shim * (u + breath * noise * (0.2 + u))
        y = (1.0 - a_ret) * x + a_ret * y                # return phase: extra -6 dB/oct above fa
        out[n] = y
    return out


@njit(cache=True)
def tract(areas, hop, src, glot_refl, lip_refl, damp, end_lp):
    """end_lp: one-pole low-pass on the waves reflected at the lips and the glottis, so high frequencies lose more
    energy per round trip (wall and radiation losses grow with frequency): wider bandwidths for the high formants,
    no metallic ringing."""
    nF, N = areas.shape
    T = src.shape[0]
    R = np.zeros(N); L = np.zeros(N); Rn = np.zeros(N); Ln = np.zeros(N)
    A = np.zeros(N); k = np.zeros(N)
    out = np.zeros(T)
    lipm, glom = 0.0, 0.0
    for n in range(T):
        f = n / hop
        i0 = int(f)
        fr = f - i0
        if i0 >= nF - 1:
            i0, fr = nF - 2, 1.0
        for j in range(N):
            A[j] = areas[i0, j] * (1.0 - fr) + areas[i0 + 1, j] * fr
        for j in range(1, N):
            k[j] = (A[j - 1] - A[j]) / (A[j - 1] + A[j])
        glom = (1.0 - end_lp) * L[0] + end_lp * glom
        lipm = (1.0 - end_lp) * R[N - 1] + end_lp * lipm
        Rn[0] = glom * glot_refl + src[n]
        Ln[N - 1] = lipm * lip_refl
        for j in range(1, N):
            w = k[j] * (R[j - 1] + L[j])
            Rn[j] = R[j - 1] - w
            Ln[j - 1] = L[j] + w
        for j in range(N):
            R[j] = Rn[j] * damp
            L[j] = Ln[j] * damp
        out[n] = R[N - 1] * (1.0 + lip_refl)
    return out


def render(voice, seed=0):
    rng = np.random.default_rng(seed)
    n_seg = max(8, int(round(voice['L'] * FS / C)))
    A, f0, amp = controls(voice, n_seg, rng)
    T = len(f0) * HOP
    ts = np.arange(T) / HOP
    f0s = np.interp(ts, np.arange(len(f0)), f0)
    amps = np.interp(ts, np.arange(len(amp)), amp)
    src = glottal_source(f0s, amps, voice['oq'], voice['breath'], voice['jitter'], voice['shimmer'], float(voice['fa']),
                         float(FS), seed)
    lips = tract(A, float(HOP), src, 0.75, -0.85, 0.9995, END_LP)
    y = np.diff(lips, prepend=0.0)                       # radiation at the lips
    y = resample_poly(y, 1, OVER)
    y -= y.mean()
    y /= np.abs(y).max() + 1e-9
    fade = int(.01 * FS_OUT)
    y[:fade] *= np.linspace(0, 1, fade); y[-fade:] *= np.linspace(1, 0, fade)
    return (0.8 * y).astype(np.float32), n_seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/voice_demo_v2')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, v in VOICES.items():
        y, n = render(v)
        path = os.path.join(args.out, f'{name}.wav')
        sf.write(path, y, FS_OUT, subtype='PCM_16')
        print(f'{path}: tract {v["L"] * 100:.1f} cm ({n} segments), pitch ~{v["f0"]} Hz, {len(y) / FS_OUT:.1f} s')


if __name__ == '__main__':
    main()
