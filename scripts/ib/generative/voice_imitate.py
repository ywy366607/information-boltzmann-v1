"""Can the body say it naturally?  Fit articulator trajectories to one recorded line (analysis by synthesis), 2026-10-19.
Existence test for the voice instrument, not a model of learning: the body (13 cm tract, nose, glottis) stays fixed;
only the motor controls are searched, every 10 ms: tongue body (pharynx area, constriction place, constriction area,
oral area), lips, tongue tip, velum, voicing, aspiration, frication at the tip and at the tongue body.  Pitch is
copied from the recording (prosody is the brain's, and not under test here).
Forward model for the fit: the SAME digital waveguide as voice_speak.tract_nasal (36 normalized-wave segments, one
sample of delay each, three-port nose junction, low-passed end reflections, damping), solved exactly in the frequency
domain with 2x2 chain matrices per frame (quasi-static), driven by the same KLGLOTT88 source (Fourier series), breath,
aspiration and frication noise filters; predicted STFT power -> log-mel, compared with the recording's.
Initial trajectories: a rule score in the same control space, time-aligned to the recording by DTW.
The fitted controls are then rendered by the real time-domain instrument (voice_speak.source + tract_nasal), which is
what one listens to.
Usage: python voice_imitate.py <ref.wav|mp3> --pinyin "ni3 zai4 kan4 na3 er5" [--out results/voice_imitate]"""
import argparse, json, os, sys, time
import numpy as np, soundfile as sf, librosa, torch
from scipy.signal import get_window, resample_poly
sys.path.insert(0, os.path.dirname(__file__))
import voice_speak as vs

torch.set_default_dtype(torch.float64)
FS, FS_OUT, C, BODY = vs.FS, vs.FS_OUT, vs.C, vs.BODY
N = int(round(BODY['L'] * FS / C))
JV = int(round(.62 * N)); NN = JV
NOSE = np.interp((np.arange(NN) + .5) / NN, [0, .3, .8, 1], [1.2, 1.8, 1.2, .5])
GLOT, LIP, DAMP, END_LP = .75, -.85, .9995, .5
NFFT, HOPS, NB = 2048, 480, 480                    # STFT at 48 kHz, 10 ms frames, bins up to 11.25 kHz
XS = (np.arange(N) + .5) / N
TIP = (XS >= .84) & (XS <= .92)
J_TIP = int(np.nonzero(TIP)[0].max())              # turbulence at the exit of the tip constriction
NPAR = 11
# control ranges: (lo, hi, log10?)  -- physical value = lo + (hi - lo) * sigmoid(theta), or 10**(...) when log
RANGES = [(.3, 9, 0), (.35, .9, 0), (-3, .8, 1), (.3, 9, 0), (-3, .8, 1), (-3, 1, 1), (0, 2, 0), (0, 1.2, 0),
          (0, 1, 0), (0, 1, 0), (0, 1, 0)]
NAMES = ['pharynx', 'place', 'constr', 'oral', 'lips', 'tip', 'velum', 'voice', 'asp', 'fric_tip', 'fric_body']


def physical(th):
    s = torch.sigmoid(th)
    out = []
    for i, (lo, hi, lg) in enumerate(RANGES):
        v = lo + (hi - lo) * s[:, i]
        out.append(10 ** v if lg else v)
    return out


def theta_of(P):
    """inverse of physical for an (T, 11) array of physical values"""
    th = np.zeros_like(P)
    for i, (lo, hi, lg) in enumerate(RANGES):
        v = np.log10(np.maximum(P[:, i], 1e-6)) if lg else P[:, i]
        s = np.clip((v - lo) / (hi - lo), .005, .995)
        th[:, i] = np.log(s / (1 - s))
    return th


def area_fn(ap, xc, ac, ao, al, tip):
    """smooth version of vowel_fit.area_model plus a tongue-tip constriction at the alveolar ridge"""
    x = torch.as_tensor(XS)[None]
    base = ap[:, None] + (ao - ap)[:, None] * torch.sigmoid((x - xc[:, None]) / .02)
    dip = ac[:, None] + (torch.maximum(ap, ao) - ac)[:, None] * (1 - torch.exp(-((x - xc[:, None]) / .08) ** 2))
    a = torch.minimum(base, dip)
    a = torch.where(torch.as_tensor(TIP)[None], torch.minimum(a, tip[:, None]), a)
    a = torch.where(torch.as_tensor(XS < .08)[None], torch.full_like(a, .6), a)
    a = torch.where(torch.as_tensor(XS > .92)[None], al[:, None].expand_as(a), a)
    return a.clamp(min=1e-3)


def j_body(xc):
    return np.clip(np.round((np.asarray(xc) + .05) * N).astype(np.int64) - 1, 9, N - 2)


def onepole(zi, corner):
    a = np.exp(-2 * np.pi * corner / FS)
    return (1 - a) / (1 - a * zi)


def transfer(a, va, freqs, jfs):
    """a (T, N) areas, va (T,) velum, freqs (T, F) Hz, jfs: list of (T,) int arrays (frication exit segments).
    Returns H_glottis (T, F) and [H_fric_q (T, F)]: output (radiated, 48 kHz signal units) per unit source."""
    T, F = freqs.shape
    zi = torch.exp(-2j * np.pi * freqs / FS)
    g = DAMP * zi
    rho = (1 - END_LP) / (1 - END_LP * zi)
    rhoL, rhoG = LIP * rho, GLOT * rho
    sa = torch.sqrt(a).to(torch.complex128)
    vae = (va + 1e-4)
    sv = torch.sqrt(vae)
    # nose reflectance at the velar port (normalized waves), from the nostrils inwards
    A_, B_ = torch.ones_like(zi), rhoL.clone()
    sn = np.sqrt(NOSE)
    for j in range(NN - 1, 0, -1):
        Rn, Ls = A_ / g, g * B_
        if j == 1:
            ap_, sp_ = vae[:, None], sv[:, None]
        else:
            ap_, sp_ = NOSE[j - 1], sn[j - 1]
        k = (ap_ - NOSE[j]) / (ap_ + NOSE[j]); c = 2 * sp_ * sn[j] / (ap_ + NOSE[j])
        A_, B_ = (Rn + k * Ls) / c, (k * Rn + Ls) / c
    alpha_n, beta_n = A_ / g, g * B_
    Gam = beta_n / alpha_n
    # oral tract from the lips inwards; coefficients over the basis [X = R_{N-1}, Z_1 .. Z_K]
    K = len(jfs)
    P1 = [torch.ones_like(zi)] + [torch.zeros_like(zi) for _ in range(K)]
    P2 = [rhoL.clone()] + [torch.zeros_like(zi) for _ in range(K)]
    masks = [torch.as_tensor(jf)[:, None] for jf in jfs]
    coefN = None
    for j in range(N - 1, 0, -1):
        Rn = [p / g for p in P1]; Ls = [g * p for p in P2]
        for q in range(K):                          # sources added to Rn_j (downstream) at junction j = jf + 1
            m = (masks[q] + 1 == j).to(zi.dtype)
            Rn[q + 1] = Rn[q + 1] - .5 * sa[:, j:j + 1] * m
        if j == JV:
            ss = torch.sqrt(a[:, j - 1] + a[:, j] + vae)[:, None]
            u1, u2, u3 = torch.sqrt(a[:, j - 1])[:, None] / ss, torch.sqrt(a[:, j])[:, None] / ss, sv[:, None] / ss
            qd = 1 - 2 * u3 ** 2 * Gam / (1 + Gam)
            m12, m22 = 2 * u1 * u2 / qd, 2 * u2 ** 2 / qd - 1
            m11, m21 = 2 * u1 ** 2 / qd - 1, 2 * u1 * u2 / qd
            R = [(rn - m22 * l) / m12 for rn, l in zip(Rn, Ls)]
            L = [m11 * r + m21 * l for r, l in zip(R, Ls)]
            coefN = [2 * u3 * (u1 * r + u2 * l) / qd / (1 + Gam) for r, l in zip(R, Ls)]
        else:
            s2 = (a[:, j - 1] + a[:, j])[:, None]
            k = (a[:, j - 1, None] - a[:, j, None]) / s2
            c = 2 * torch.sqrt(a[:, j - 1, None] * a[:, j, None]) / s2
            R = [(rn + k * l) / c for rn, l in zip(Rn, Ls)]
            L = [(k * rn + l) / c for rn, l in zip(Rn, Ls)]
        for q in range(K):                          # and -z/2 to Ln_{j-1} (upstream)
            m = (masks[q] + 1 == j).to(zi.dtype)
            L[q + 1] = L[q + 1] - .5 * sa[:, j - 1:j] * m
        P1, P2 = R, L
    Rn0 = [p / g for p in P1]; L0 = [g * p for p in P2]
    e = [r - rhoG * l for r, l in zip(Rn0, L0)]     # glottis: Rn_0 - rhoG L_0 = sa_0 S_g
    D = (1 - zi) * (1 + LIP)
    o = [D * (sa[:, N - 1:N] * (1.0 if b == 0 else 0.0) + sn[-1] * coefN[b] / alpha_n) for b in range(K + 1)]
    Hg = o[0] * sa[:, 0:1] / e[0]
    Hf = [o[q + 1] - o[0] * e[q + 1] / e[0] for q in range(K)]
    return Hg, Hf


class Model:
    def __init__(self, f0, T):
        self.T = T
        w = get_window('hann', NFFT)
        self.sw2 = float(np.sum(w ** 2))
        fb = np.arange(NB) * FS_OUT / NFFT
        self.fbins = fb
        # harmonic kernel |W(delta)|^2 in bins, tabulated
        dl = np.arange(-6, 6.0001, .01)
        n = np.arange(NFFT)
        Wt = np.abs(np.exp(-2j * np.pi * dl[:, None] * n[None] / NFFT) @ w) ** 2
        self.K = int(11250 // max(150, np.min(f0))) + 1
        k = np.arange(1, self.K + 1)
        fh = f0[:, None] * k[None]                                  # (T, K)
        self.hmask = torch.as_tensor((fh < 11250).astype(float))
        delta = fb[None, None, :] / (FS_OUT / NFFT) - fh[:, :, None] / (FS_OUT / NFFT)
        self.Km = torch.as_tensor(np.interp(delta, dl, Wt, left=0, right=0).astype(np.float32))   # (T, K, NB)
        self.fh = torch.as_tensor(fh)
        # noise spectra are smooth: evaluated every 4th bin, linearly interpolated to all bins
        sub = np.arange(0, NB, 4)
        self.fnoise_row = torch.as_tensor(fb[sub])
        self.Mi = torch.as_tensor(np.stack([np.interp(np.arange(NB), sub, np.eye(len(sub))[i])
                                            for i in range(len(sub))], 1))          # (NB, NS)
        # glottal flow Fourier series (body: fixed open quotient), return-phase pole
        ph = (np.arange(8192) + .5) / 8192
        u = np.where(ph < BODY['oq'], 6.75 * (ph / BODY['oq']) ** 2 * (1 - ph / BODY['oq']), 0.)
        ck = np.array([np.mean(u * np.exp(-2j * np.pi * kk * ph)) for kk in k])
        self.ck = torch.as_tensor(ck)[None]
        self.e02 = float(np.mean((.2 + u) ** 2))

    def spec(self, th, t0=0, t1=None):
        """predicted STFT power (t1 - t0, NB) for frames t0..t1 (chunks keep the autograd graph small)"""
        t1 = self.T if t1 is None else t1
        ap, xc, ac, ao, al, tip, va, av, asp, ftip, fbody = physical(th[t0:t1])
        a = area_fn(ap, xc, ac, ao, al, tip)
        jb = j_body(xc.detach().numpy())
        jt = np.full(t1 - t0, J_TIP)
        fh = self.fh[t0:t1]
        freqs = torch.cat([fh, self.fnoise_row[None].expand(t1 - t0, -1)], 1)
        Hg, Hf = transfer(a, va, freqs, [jt, jb])
        zi = torch.exp(-2j * np.pi * freqs / FS)
        ret = onepole(zi, BODY['fa'])
        Kh = fh.shape[1]
        # voiced harmonics
        amp = av[:, None] * self.ck * ret[:, :Kh] * Hg[:, :Kh]
        Ph = torch.einsum('tk,tkb->tb', (amp.abs() ** 2) * self.hmask[t0:t1], self.Km[t0:t1].double())
        # noise: breath (gated by voicing), aspiration, frication; per-sample variance -> STFT power (x .5: 96->48k)
        zn = zi[:, Kh:]
        nbr = 2 * (1 - onepole(zn, 400.)) * onepole(zn, 2500.) ** 2 * ret[:, Kh:]
        glot_var = (av[:, None] * BODY['breath']) ** 2 * self.e02 + (asp[:, None] * vs.ASP_GAIN) ** 2
        nf = vs.FRIC_GAIN * (1 - onepole(zn, 1500.)) * onepole(zn, 7000.) ** 2
        Pn = glot_var * (nbr * Hg[:, Kh:]).abs() ** 2 + (ftip[:, None] * nf * Hf[0][:, Kh:]).abs() ** 2 + \
            (fbody[:, None] * nf * Hf[1][:, Kh:]).abs() ** 2
        Pn = .5 * self.sw2 * Pn
        return Ph + Pn @ self.Mi.T

    def spec_all(self, th, chunk=25):
        with torch.no_grad():
            return torch.cat([self.spec(th, t, min(self.T, t + chunk)) for t in range(0, self.T, chunk)]).numpy()


def track_f0(P, fmin=150., fmax=900., jump_st=1.5):
    """pitch on every frame: subharmonic summation on the log spectrum, then dynamic programming for a continuous
    track (pyin's voicing decision dropped half of this recording's clearly harmonic frames, and an interpolated
    pitch puts the model's harmonics in the wrong places).  Returns f0 (T,) and a harmonicity score (T,)."""
    fb = np.arange(P.shape[0]) * FS_OUT / NFFT
    L = 10 * np.log10(P + P.max() * 1e-8)
    L = L - np.array([np.convolve(c, np.ones(31) / 31, 'same') for c in L.T]).T     # whiten: peaks over local mean
    cand = fmin * 2 ** (np.arange(0, 12 * np.log2(fmax / fmin), .1) / 12)
    S = np.zeros((len(cand), P.shape[1]))
    for k in range(1, 9):
        S += .84 ** k * np.array([np.interp(k * cand, fb, L[:, t]) for t in range(P.shape[1])]).T
    st = 12 * np.log2(cand)
    pen = .6 * np.maximum(np.abs(st[:, None] - st[None]) - jump_st, 0) + .05 * np.abs(st[:, None] - st[None])
    D = S[:, 0].copy(); back = np.zeros(S.shape, int)
    for t in range(1, S.shape[1]):
        M = D[None, :] - pen                     # (to, from)
        back[:, t] = M.argmax(1); D = M.max(1) + S[:, t]
    path = np.zeros(S.shape[1], int); path[-1] = D.argmax()
    for t in range(S.shape[1] - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    return cand[path], S[path, np.arange(S.shape[1])]


def stft_power(y):
    X = librosa.stft(y, n_fft=NFFT, hop_length=HOPS, window='hann', center=True)
    return np.abs(X[:NB]) ** 2


# ------------------------------------------------------------------ rule score in control space (initial guess)
def vowel_ctrl(v):
    p = vs.VP[v]
    return [p[0], p[1], p[2], p[3], p[4], 10., 0., 1., 0., 0., 0.]


def rule_controls(pinyin):
    """per-ms (T, 11) controls from pinyin with the voice_speak rules (simplified to this control space)"""
    rows = []
    def push(c, ms):
        rows.extend([list(c)] * int(ms))
    toks = pinyin.replace(',', ' , ').replace('.', ' . ').replace('?', ' ? ').split()
    for ti, t in enumerate(toks):
        if t in ',.?':
            c = vowel_ctrl('@'); c[7] = 0; push(c, 200); continue
        init, vow, coda, tone = vs.parse_syllable(t)
        first = vowel_ctrl(vow[0])
        if init:
            manner, place, aspd = vs.INIT[init]
            if manner in ('stop', 'affr', 'nas', 'fric', 'lat', 'appr'):
                c = list(first)
                if place in ('alv', 'ret', 'pal'):
                    ci, fi = 5, 9                       # the tongue tip
                elif place == 'vel':
                    ci, fi = 2, 10; c[1] = .6           # the tongue body at the velum
                else:
                    ci, fi = 4, 9                       # the lips
                if manner == 'stop':
                    cc = list(c); cc[ci] = .001; cc[7] = 0; push(cc, 50)
                    cb = list(c); cb[ci] = .1; cb[7] = 0; cb[fi] = .8; push(cb, 15)
                    ca = list(first); ca[7] = 0; ca[8] = .9 if aspd else 0; push(ca, 60 if aspd else 10)
                elif manner == 'affr':
                    cc = list(c); cc[ci] = .001; cc[7] = 0; push(cc, 40)
                    cf = list(c); cf[ci] = .06; cf[7] = 0; cf[fi] = vs.FRIC_STRENGTH[place]; push(cf, 70)
                    if aspd:
                        ca = list(first); ca[7] = 0; ca[8] = .8; push(ca, 45)
                elif manner == 'fric':
                    cf = list(c); cf[ci] = .06 if ci != 2 else .12; cf[7] = 0; cf[fi] = vs.FRIC_STRENGTH[place]
                    cf[8] = .2 if aspd else 0; push(cf, 110)
                elif manner == 'nas':
                    cn = list(c); cn[ci] = .001; cn[6] = 1.6; cn[7] = .8; push(cn, 65)
                elif manner == 'lat':
                    cl = list(c); cl[ci] = .3; cl[7] = .9; push(cl, 55)
                else:
                    cr = list(c); cr[ci] = .25; cr[7] = .9; push(cr, 70)
        for k, v in enumerate(vow):
            c = vowel_ctrl(v)
            if coda and k == len(vow) - 1:
                c[6] = .6
            push(c, 160 / len(vow))
        if coda:
            c = vowel_ctrl(vow[-1]); c[6] = 1.6; c[7] = .85
            if coda == 'n':
                c[5] = .001
            else:
                c[2] = .001; c[1] = .66
            push(c, 70)
    return np.array(rows, float)


# ------------------------------------------------------------------ time-domain rendering of 10 ms controls
def render(P10, f0_10, seed=0, room=True):
    T = len(P10)
    ms = np.arange(T * 10) / 10.0
    Pm = np.stack([np.interp(ms, np.arange(T), P10[:, i]) for i in range(NPAR)], 1)
    for i, (lo, hi, lg) in enumerate(RANGES):                   # log-controls interpolate in log
        if lg:
            Pm[:, i] = 10 ** np.interp(ms, np.arange(T), np.log10(np.maximum(P10[:, i], 1e-6)))
    with torch.no_grad():
        t = [torch.as_tensor(Pm[:, i]) for i in range(6)]
        A = area_fn(*t).numpy()
    f0m = np.interp(ms, np.arange(T), f0_10)
    nT = len(ms) * vs.HOP
    ts = np.arange(nT) / vs.HOP
    idx = np.arange(len(ms))
    f0 = np.interp(ts, idx, f0m)
    av = np.interp(ts, idx, Pm[:, 7]); asp = np.interp(ts, idx, Pm[:, 8])
    src = vs.source(f0, av, asp, BODY['oq'], BODY['breath'], BODY['jitter'], BODY['shimmer'], BODY['fa'], float(FS),
                    seed, vs.ASP_GAIN)
    fpos = np.stack([np.full(len(ms), J_TIP), j_body(Pm[:, 1])], 1).astype(np.int64)
    famp = np.ascontiguousarray(Pm[:, 9:11])
    out = vs.tract_nasal(A, np.ascontiguousarray(Pm[:, 6]), fpos, famp, float(vs.HOP), src, GLOT, LIP, DAMP, END_LP,
                         JV, NOSE, seed, vs.FRIC_GAIN)
    y = np.diff(out, prepend=0.0)
    y = resample_poly(y, 1, FS // FS_OUT)
    dry = y.copy()
    if room:
        y = vs.room(np.r_[y, np.zeros(int(.3 * FS_OUT))], FS_OUT)
    return dry, y


def logmel(P, mel):
    return 10 * np.log10(mel @ P + 1e-30)


def mel_l1(Pa, Pb, mel, floor_db=80):
    """mean |log-mel difference| (dB) after removing the best overall gain; cells below -floor_db are clipped"""
    la, lb = logmel(Pa, mel), logmel(Pb, mel)
    lb = lb - lb.max()
    la = la - la.max()
    act = lb > -floor_db + 20
    la = la - np.median((la - lb)[act])
    return float(np.mean(np.abs(np.maximum(la, -floor_db) - np.maximum(lb, -floor_db))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('--pinyin', required=True)
    ap.add_argument('--out', default='results/voice_imitate')
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--check', action='store_true', help='only compare the frequency model with the time domain')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    y, sr = sf.read(args.ref)
    y = y.mean(1) if y.ndim > 1 else y
    y = librosa.resample(y, orig_sr=sr, target_sr=FS_OUT)
    y = y / (np.abs(y).max() + 1e-9) * .8
    Pref = stft_power(y)
    T = Pref.shape[1]
    f0, harm = track_f0(Pref)
    mel = librosa.filters.mel(sr=FS_OUT, n_fft=NFFT, n_mels=80, fmin=120, fmax=11000)[:, :NB]   # above rumble

    # initial controls: rule score, aligned to the recording by DTW on MFCCs of its time-domain render
    rule_ms = rule_controls(args.pinyin)
    rule10 = rule_ms[::10]
    f0_rule = np.full(len(rule10), np.median(f0))
    _, yr = render(rule10, f0_rule, room=False)
    m_ref = librosa.feature.mfcc(y=y, sr=FS_OUT, n_mfcc=16, n_fft=NFFT, hop_length=HOPS)[1:]
    m_rule = librosa.feature.mfcc(y=yr, sr=FS_OUT, n_mfcc=16, n_fft=NFFT, hop_length=HOPS)[1:]
    z = lambda m: (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9)
    _, wp = librosa.sequence.dtw(X=z(m_ref), Y=z(m_rule), metric='euclidean')
    wp = wp[::-1]
    idx = np.zeros(T)
    for t in range(T):
        idx[t] = np.mean(wp[wp[:, 0] == t, 1]) if np.any(wp[:, 0] == t) else 0
    idx = np.clip(np.round(idx).astype(int), 0, len(rule10) - 1)
    P0 = rule10[idx]
    model = Model(f0, T)
    th = torch.tensor(theta_of(P0), requires_grad=True)
    gain = torch.zeros(1, requires_grad=True)
    melt = torch.as_tensor(mel).double()
    lref = torch.as_tensor(logmel(Pref, mel))
    lref = lref - lref.max()
    floor = -80.

    CH = 25

    def step_grad(th, smooth_w):
        """accumulate gradients chunk by chunk; returns the log-mel L1 (dB)"""
        tot = 0.
        for t0 in range(0, T, CH):
            t1 = min(T, t0 + CH)
            P = model.spec(th, t0, t1).T                        # (NB, chunk)
            lp = 10 * torch.log10(melt @ P + 1e-30) + gain
            d = torch.maximum(lp, torch.tensor(floor)) - torch.maximum(lref[:, t0:t1], torch.tensor(floor))
            fit = d.abs().sum() / d.shape[0] / T
            fit.backward()
            tot += float(fit)
        s = torch.sigmoid(th)
        sm = ((s[2:] - 2 * s[1:-1] + s[:-2]) ** 2).mean() + .1 * ((s[1:] - s[:-1]) ** 2).mean()
        # priors (dB-equivalent weights): turbulence and nasal coupling are brief events, not a way to fill spectra
        ph = physical(th)
        prior = 2.0 * (ph[8] + ph[9] + ph[10]).mean() + .5 * ph[6].mean()
        (smooth_w * sm * 100 + prior).backward()
        return tot

    # check: frequency model vs time-domain instrument on the initial controls
    Pm = model.spec_all(th).T
    dry0, _ = render(P0, f0, room=False)
    Ptd = stft_power(dry0)[:, :T]
    print(f'frequency model vs time-domain render (initial controls): log-mel L1 {mel_l1(Pm, Ptd, mel):.2f} dB; '
          f'rule (DTW-aligned) vs recording {mel_l1(Ptd, Pref, mel):.2f} dB', flush=True)
    if args.check:
        np.savez(os.path.join(args.out, 'check.npz'), Pm=Pm, Ptd=Ptd, Pref=Pref, mel=mel)
        return
    with torch.no_grad():                                       # start the gain at the level match
        gain -= float(logmel(Pm, mel).max())
    opt = torch.optim.Adam([th, gain], lr=.03)
    hist = []
    t0 = time.time()
    for it in range(args.iters):
        smooth_w = 1.0 if it < args.iters // 3 else .3
        opt.zero_grad()
        fit = step_grad(th, smooth_w)
        opt.step()
        hist.append(float(fit))
        if it % 100 == 0 or it == args.iters - 1:
            print(f'it {it:5d}  log-mel L1 {float(fit):.2f} dB  ({time.time() - t0:.0f} s)', flush=True)
    with torch.no_grad():
        P_fit = np.stack([v.numpy() for v in physical(th)], 1)
    Pm_fit = model.spec_all(th).T
    dry, wet = render(P_fit, f0)
    Ptd_fit = stft_power(dry)[:, :T]
    rep = {'ref': args.ref, 'pinyin': args.pinyin,
           'rule_timedomain_vs_ref_dB': mel_l1(Ptd, Pref, mel),
           'fit_freqmodel_vs_ref_dB': mel_l1(Pm_fit, Pref, mel),
           'fit_timedomain_vs_ref_dB': mel_l1(Ptd_fit, Pref, mel),
           'freqmodel_vs_timedomain_fit_dB': mel_l1(Pm_fit, Ptd_fit, mel)}
    print(json.dumps(rep, indent=1))
    pk = lambda v: (.8 * v / (np.abs(v).max() + 1e-9)).astype(np.float32)
    sf.write(os.path.join(args.out, 'fit.wav'), pk(wet), FS_OUT, subtype='PCM_16')
    sf.write(os.path.join(args.out, 'fit_dry.wav'), pk(dry), FS_OUT, subtype='PCM_16')
    sf.write(os.path.join(args.out, 'rule_aligned.wav'), pk(render(P0, f0)[1]), FS_OUT, subtype='PCM_16')
    np.savez(os.path.join(args.out, 'controls.npz'), fit=P_fit, init=P0, f0=f0, names=NAMES, hist=np.array(hist))
    json.dump(rep, open(os.path.join(args.out, 'report.json'), 'w'), indent=1)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(5, 1, figsize=(12, 13))
        for a_, (P, nm) in zip(ax[:4], [(Pref, 'recording'), (Ptd, 'rule score (DTW-aligned), instrument'),
                                         (Pm_fit, 'fitted controls, frequency model'),
                                         (Ptd_fit, 'fitted controls, time-domain instrument')]):
            l = logmel(P, mel); a_.imshow(np.maximum(l - l.max(), -80), origin='lower', aspect='auto', cmap='magma')
            a_.set_title(nm)
        for i in [1, 2, 3, 5, 6, 7]:
            v = P_fit[:, i]; ax[4].plot((v - v.min()) / (np.ptp(v) + 1e-9) + i * 1.1, label=NAMES[i])
        ax[4].legend(fontsize=7, ncol=6); ax[4].set_title('fitted controls (normalized)')
        plt.tight_layout(); plt.savefig(os.path.join(args.out, 'fit.png'), dpi=70)
    except Exception as ex:
        print('plot failed', ex)


if __name__ == '__main__':
    main()
