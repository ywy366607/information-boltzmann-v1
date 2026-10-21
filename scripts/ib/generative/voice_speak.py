"""The voice instrument speaking Mandarin (2026-10-19): a puppeteer's score for the body, not the brain.
The life-form will learn to drive these articulators itself; this script writes the motor score by rule so we can
hear the body's voice saying real words.  Body (fixed): a 13 cm tract, the voice of voice_demo's 01_teen_13cm_light.
Additions to the instrument: a nasal tract coupled at the velum (three-port junction), turbulence noise injected at
a constriction (fricatives, bursts), glottal aspiration independent of voicing (h, aspirated stops).
Score: pinyin syllables with tone numbers (written by hand here) -> phones -> per-ms targets for the oral area
function, velum opening, frication place and strength, aspiration, voicing, and the tone's pitch contour.
Usage: python voice_speak.py [--out results/voice_speak]"""
import argparse, os, re, sys, numpy as np, soundfile as sf
from numba import njit
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(__file__))
import voice_demo as vd

FS, FS_OUT, HOP, C = vd.FS, vd.FS_OUT, vd.HOP, vd.C
FRIC_GAIN, ASP_GAIN = 0.02, 0.10   # turbulence at a constriction, aspiration at the glottis
BODY = dict(L=.130, f0=310.0, oq=.68, breath=.06, fa=1300.0, jitter=.004, shimmer=.02, vib=.006)

# ---------------------------------------------------------------- articulation: vowels (x = 0 glottis .. 1 lips)
V = {
    'a':  ([0, .1, .25, .45, .55, .7, .9, 1], [.6, .8, .6, .8, 3, 6, 7, 6]),
    'A':  ([0, .1, .25, .45, .6, .75, .9, 1], [.6, .8, .4, .6, 2, 5, 6, 5]),          # back a (ang)
    'i':  ([0, .1, .25, .5, .62, .72, .82, .92, 1], [.6, .8, 7, 7, 3, .4, .35, 1.5, 2.5]),
    'v':  ([0, .1, .25, .5, .62, .72, .82, .92, 1], [.6, .8, 7, 7, 3, .45, .4, 1.2, .35]),  # u-umlaut: i + round lips
    'u':  ([0, .1, .25, .5, .58, .64, .72, .88, .96, 1], [.6, .8, 4, 4.5, .6, .35, 3, 4, 1, .15]),
    'o':  ([0, .1, .25, .5, .6, .7, .9, 1], [.6, .8, 2, 2, 1.5, 3, 2.5, .6]),
    'e':  ([0, .1, .25, .5, .6, .7, .9, 1], [.6, .8, 2.5, 2.5, 1.4, 2.8, 3, 2.5]),    # mid back unrounded
    'E':  ([0, .1, .25, .5, .65, .75, .85, 1], [.6, .8, 4.5, 4.5, 2.5, 1.4, 1.8, 3.5]),
    '@':  ([0, .1, .25, 1], [.6, .8, 2.8, 2.8]),
    'R':  ([0, .1, .25, .7, .78, .84, .9, 1], [.6, .8, 2.8, 2.8, 1, .7, 2, 2.5]),    # er, retroflex colouring
    'z':  ([0, .1, .25, .8, .86, .9, .95, 1], [.6, .8, 3, 3, .5, .4, 2, 2.5]),        # apical vowel after z c s
    'r':  ([0, .1, .25, .74, .8, .85, .92, 1], [.6, .8, 3, 3, .6, .5, 2, 2.5]),       # apical vowel after zh ch sh r
}
FRIC_STRENGTH = {'alv': .3, 'ret': .35, 'pal': .55, 'lab': .25, 'vel': .15}   # radiated: s ~ sh > x > h > f
PLACE = {'lab': (.95, 1.0), 'alv': (.86, .92), 'ret': (.79, .85), 'pal': (.74, .82), 'vel': (.58, .66)}

# initials: (manner, place, aspirated)
INIT = {'b': ('stop', 'lab', 0), 'p': ('stop', 'lab', 1), 'm': ('nas', 'lab', 0), 'f': ('fric', 'lab', 0),
        'd': ('stop', 'alv', 0), 't': ('stop', 'alv', 1), 'n': ('nas', 'alv', 0), 'l': ('lat', 'alv', 0),
        'g': ('stop', 'vel', 0), 'k': ('stop', 'vel', 1), 'h': ('fric', 'vel', 1),
        'j': ('affr', 'pal', 0), 'q': ('affr', 'pal', 1), 'x': ('fric', 'pal', 0),
        'zh': ('affr', 'ret', 0), 'ch': ('affr', 'ret', 1), 'sh': ('fric', 'ret', 0), 'r': ('appr', 'ret', 0),
        'z': ('affr', 'alv', 0), 'c': ('affr', 'alv', 1), 's': ('fric', 'alv', 0)}

FINALS = {'a': 'a', 'o': 'o', 'e': 'e', 'ai': 'ai', 'ei': 'Ei', 'ao': 'au', 'ou': 'ou', 'an': 'a+n', 'en': '@+n',
          'ang': 'A+ng', 'eng': '@+ng', 'ong': 'u+ng', 'er': 'R', 'i': 'i', 'ia': 'ia', 'ie': 'iE', 'iao': 'iau',
          'iou': 'iou', 'ian': 'iE+n', 'in': 'i+n', 'iang': 'iA+ng', 'ing': 'i+ng', 'iong': 'iu+ng', 'u': 'u',
          'ua': 'ua', 'uo': 'uo', 'uai': 'uai', 'uei': 'uEi', 'uan': 'ua+n', 'uen': 'u@+n', 'uang': 'uA+ng',
          'ueng': 'u@+ng', 'v': 'v', 've': 'vE', 'van': 'vE+n', 'vn': 'v+n'}
Y_W = {'yi': 'i', 'ya': 'ia', 'ye': 'ie', 'yao': 'iao', 'you': 'iou', 'yan': 'ian', 'yin': 'in', 'yang': 'iang',
       'ying': 'ing', 'yong': 'iong', 'yu': 'v', 'yue': 've', 'yuan': 'van', 'yun': 'vn', 'wu': 'u', 'wa': 'ua',
       'wo': 'uo', 'wai': 'uai', 'wei': 'uei', 'wan': 'uan', 'wen': 'uen', 'wang': 'uang', 'weng': 'ueng'}
# tones (Chao levels) -> contour; neutral tone follows the previous tone's end
TONES = {1: (5, 5), 2: (3, 5), 3: (2, 1, 4), 4: (5, 1)}
LEVEL_ST = {5: 5.0, 4: 2.5, 3: 0.0, 2: -2.5, 1: -5.0}   # semitones; about an octave per sentence, as measured
NEUTRAL_AFTER = {1: 2, 2: 3, 3: 4, 4: 1, 5: 2}


def parse_syllable(s):
    m = re.fullmatch(r'([a-zü]+)([1-5])', s)
    body, tone = m.group(1).replace('ü', 'v'), int(m.group(2))
    init = ''
    for c in ('zh', 'ch', 'sh') + tuple(INIT):
        if body.startswith(c) and c not in ('y', 'w'):
            init = c
            break
    fin = body[len(init):]
    if not init:
        for k in sorted(Y_W, key=len, reverse=True):
            if body == k:
                fin = Y_W[k]
                break
    if init in ('j', 'q', 'x') and fin.startswith('u'):
        fin = 'v' + fin[1:]
    fin = {'iu': 'iou', 'ui': 'uei', 'un': 'vn' if init in ('j', 'q', 'x') else 'uen'}.get(fin, fin)
    if fin == 'o' and init in ('b', 'p', 'm', 'f'):
        fin = 'uo'
    vowels = FINALS[fin]
    if fin == 'i' and init in ('z', 'c', 's'):
        vowels = 'z'
    if fin == 'i' and init in ('zh', 'ch', 'sh', 'r'):
        vowels = 'r'
    coda = ''
    if '+' in vowels:
        vowels, coda = vowels.split('+')
    return init, list(vowels), coda, tone


# vowel shapes fitted to target formants for the 13 cm tract (vowel_fit.py --L 0.13): pharynx area, constriction
# place, constriction area, oral area, lip area
VP = {"a": [0.507, 0.568, 2.457, 5.293, 4.126], "A": [0.315, 0.554, 0.544, 8.965, 3.647],
      "i": [6.007, 0.542, 1.577, 0.778, 2.922], "v": [8.917, 0.637, 0.604, 2.099, 1.101],
      "u": [6.652, 0.402, 0.151, 5.63, 0.522], "o": [0.436, 0.646, 1.18, 8.986, 0.518],
      "e": [1.569, 0.606, 1.424, 4.971, 0.439], "E": [1.696, 0.637, 0.437, 2.891, 5.506],
      "@": [3.091, 0.608, 2.005, 5.19, 1.297], "R": [0.302, 0.82, 1.651, 6.098, 0.41],
      "z": [5.412, 0.587, 1.226, 3.09, 0.581], "r": [2.248, 0.711, 0.194, 2.375, 0.417]}


def area(shape, n):
    import vowel_fit
    return vowel_fit.area_model(VP[shape], n)


def with_constriction(base, place, value, n):
    x = (np.arange(n) + .5) / n
    lo, hi = PLACE[place]
    a = base.copy()
    m = (x >= lo) & (x <= hi)
    a[m] = np.minimum(a[m], value)
    # turbulence forms where the jet leaves the constriction: its exit junction (kept inside the tube for the lips)
    return a, int(min(np.nonzero(m)[0].max(), n - 2))


def score(sentence, n, syl_ms=200.0):
    """sentence: pinyin syllables with tone numbers, ',' '.' '?' as pauses.  Returns per-ms control tracks."""
    toks = sentence.replace(',', ' , ').replace('.', ' . ').replace('?', ' ? ').split()
    sylls = [t for t in toks if t not in ',.?']
    # third-tone sandhi (3 3 -> 2 3) and phrase-final syllables
    tones = [parse_syllable(s)[3] for s in sylls]
    for i in range(len(tones) - 1):
        if tones[i] == 3 and tones[i + 1] == 3:
            tones[i] = 2
    A, vel, fpos, famp, asp, av, f0 = [], [], [], [], [], [], []
    fades = []
    neutral_prev = 5
    si = 0
    question = sentence.strip().endswith('?')
    for ti, t in enumerate(toks):
        if t in ',.?':
            ms = 220 if t == ',' else 360
            A += [area('@', n)] * ms; vel += [0.] * ms; fpos += [0] * ms; famp += [0.] * ms
            asp += [0.] * ms; av += [0.] * ms; f0 += [np.nan] * ms
            continue
        init, vow, coda, _ = parse_syllable(t)
        tone = tones[si]
        final_in_phrase = ti + 1 < len(toks) and toks[ti + 1] in ',.?' or ti + 1 == len(toks)
        dur = syl_ms * (1.35 if final_in_phrase else 1.0) * (0.6 if tone == 5 else 1.0)
        first_v = area(vow[0], n)

        def push(a, ms, v=0., fp=0, fa=0., ap=0., voice=0., env=None):
            # env: shape of the noise envelope over the segment ('ramp' rises and falls, 'decay' a transient)
            ms = int(ms)
            if env == 'ramp':
                e = np.minimum(1, np.minimum(np.arange(ms) + 1, ms - np.arange(ms)) / max(1, min(25, ms // 3)))
            elif env == 'decay':
                e = np.exp(-np.arange(ms) / max(1., ms / 3))
            else:
                e = np.ones(ms)
            A.extend([a] * ms); vel.extend([v] * ms); fpos.extend([fp] * ms); famp.extend(list(fa * e))
            asp.extend(list(ap * e)); av.extend([voice] * ms); f0.extend([np.nan] * ms)

        used = 0
        if init:
            manner, place, aspd = INIT[init]
            if manner == 'stop':
                a, j = with_constriction(first_v, place, .0005, n)
                ar, _ = with_constriction(first_v, place, .1, n)     # the burst: turbulence at the opening gap
                push(a, 50); push(ar, 15, fp=j, fa=1.0, env='decay')
                push(first_v, 65 if aspd else 10, ap=.9 if aspd else 0., env='decay')
                used = 130 if aspd else 70
            elif manner == 'affr':
                a, j = with_constriction(first_v, place, .0005, n)
                af, _ = with_constriction(first_v, place, .06, n)
                push(a, 40); push(af, 70, fp=j, fa=FRIC_STRENGTH[place], env='ramp')
                push(first_v, 45 if aspd else 5, ap=.8 if aspd else 0., env='decay')
                used = 155 if aspd else 115
            elif manner == 'fric':
                a, j = with_constriction(first_v, place, .12 if place == 'vel' else .06, n)
                push(a, 110, fp=j, fa=FRIC_STRENGTH[place], ap=.2 if aspd else 0., env='ramp')
                used = 110
            elif manner == 'nas':
                a, _ = with_constriction(first_v, place, .0005, n)
                push(a, 65, v=1.6, voice=.8)
                used = 65
            elif manner == 'lat':
                a, _ = with_constriction(first_v, place, .3, n)
                push(a, 55, voice=.9)
                used = 55
            elif manner == 'appr':
                a, j = with_constriction(first_v, place, .25, n)
                push(a, 70, fp=j, fa=.08, voice=.9, env='ramp')
                used = 70
        coda_ms = 70 if coda else 0
        vms = max(60, dur - used * .5 - coda_ms * .5)
        start = len(f0)
        for k, sh in enumerate(vow):
            nas = 0.6 if coda and k == len(vow) - 1 else 0.
            push(area(sh, n), vms / len(vow), v=nas, voice=1.0)
        if coda:
            last = area(vow[-1], n)
            a, _ = with_constriction(last, 'alv' if coda == 'n' else 'vel', .0005, n)
            push(a, coda_ms, v=1.6, voice=.85)
        # tone contour over the voiced part of the syllable
        end = len(f0)
        if final_in_phrase:
            fades.append((start + (end - start) // 3, end))
        if tone == 5:
            lv = (NEUTRAL_AFTER[neutral_prev],) * 2
        elif tone == 3 and not final_in_phrase:
            lv = (2, 1, 1)
        else:
            lv = TONES[tone]
        neutral_prev = tone if tone != 5 else neutral_prev
        st = np.interp(np.linspace(0, 1, end - start), np.linspace(0, 1, len(lv)), [LEVEL_ST[l] for l in lv])
        f0[start:end] = list(st)
        si += 1
    A = np.array(A); f0 = np.array(f0, float); av = np.array(av, float)
    for a, b in fades:                       # subglottal pressure runs down at the end of a breath group
        av[a:b] *= np.linspace(1, .25, b - a) ** 1.5
    # pitch: fill unvoiced gaps, declination, question rise; semitones -> Hz
    idx = np.arange(len(f0)); ok = np.isfinite(f0)
    f0 = np.interp(idx, idx[ok], f0[ok])
    f0 -= 1.5 * idx / len(f0)
    if question:
        f0[-400:] += np.linspace(0, 2.0, 400)
    return dict(A=A, vel=np.array(vel), fpos=np.array(fpos), famp=np.array(famp), asp=np.array(asp),
                av=av, st=f0)


@njit(cache=True)
def source(f0, av, asp, oq, breath, jitter, shimmer, fa, fs, seed, asp_gain):
    np.random.seed(seed)
    T = f0.shape[0]
    out = np.zeros(T)
    a_ret = np.exp(-2 * np.pi * fa / fs)
    a_lo = np.exp(-2 * np.pi * 2500.0 / fs); a_hi = np.exp(-2 * np.pi * 400.0 / fs)   # breath turbulence: 0.4-2.5 kHz
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
            u = 6.75 * t * t * (1.0 - t)
        w = np.random.randn()
        nl = (1 - a_lo) * w + a_lo * nl
        nl2 = (1 - a_lo) * nl + a_lo * nl2
        nh = (1 - a_hi) * nl2 + a_hi * nh
        noise = (nl2 - nh) * 2.0
        x = av[n] * shim * (u + breath * noise * (0.2 + u)) + asp[n] * noise * asp_gain
        y = (1 - a_ret) * x + a_ret * y
        out[n] = y
    return out


@njit(cache=True)
def tract_nasal(A, vel, fpos, famp, hop, src, glot, lip, damp, end_lp, jv, nose_area, seed, fric_gain):
    """Oral tract (N segments) with a nasal branch at segment jv (three-port junction whose side area is the velum
    opening), frication noise injected at fpos with strength famp (band-limited turbulence; fpos/famp are (frames,
    sources) so several constrictions can hiss at once, each with its own noise).
    Power-normalized waves (pressure x sqrt(area)): every junction is an orthogonal scattering (a rotation for two
    ports, a Householder reflection for three), so moving articulators never pump energy into the tube.  With plain
    pressure waves the energy a segment holds scales with its area, so a closure opening 0.0005 -> 3 cm^2 multiplied
    it by thousands: the clicks at every nasal and stop boundary."""
    np.random.seed(seed + 7)
    nF, N = A.shape
    Nn = nose_area.shape[0]
    T = src.shape[0]
    R = np.zeros(N); L = np.zeros(N); Rn = np.zeros(N); Ln = np.zeros(N)
    NR = np.zeros(Nn); NL = np.zeros(Nn); NRn = np.zeros(Nn); NLn = np.zeros(Nn)
    a = np.zeros(N); sa = np.zeros(N); sn = np.sqrt(nose_area)
    out = np.zeros(T)
    lipm, glom, nosm = 0.0, 0.0, 0.0
    a_lo = np.exp(-2 * np.pi * 7000.0 / (hop * 1000.0)); a_hi = np.exp(-2 * np.pi * 1500.0 / (hop * 1000.0))
    K = famp.shape[1]
    nl = np.zeros(K); nl2 = np.zeros(K); nh = np.zeros(K)
    for n in range(T):
        f = n / hop
        i0 = int(f)
        fr = f - i0
        if i0 >= nF - 1:
            i0, fr = nF - 2, 1.0
        for j in range(N):
            a[j] = A[i0, j] * (1 - fr) + A[i0 + 1, j] * fr
            sa[j] = np.sqrt(a[j])
        va = vel[i0] * (1 - fr) + vel[i0 + 1] * fr + 1e-4
        sv = np.sqrt(va)
        glom = (1 - end_lp) * L[0] + end_lp * glom
        lipm = (1 - end_lp) * R[N - 1] + end_lp * lipm
        nosm = (1 - end_lp) * NR[Nn - 1] + end_lp * nosm
        Rn[0] = glom * glot + src[n] * sa[0]
        Ln[N - 1] = lipm * lip
        NLn[Nn - 1] = nosm * lip
        for j in range(1, N):
            if j == jv:                                  # three-port junction with the nose
                ss = np.sqrt(a[j - 1] + a[j] + va)
                u1, u2, u3 = sa[j - 1] / ss, sa[j] / ss, sv / ss
                d = u1 * R[j - 1] + u2 * L[j] + u3 * NL[0]
                Ln[j - 1] = 2 * u1 * d - R[j - 1]
                Rn[j] = 2 * u2 * d - L[j]
                NRn[0] = 2 * u3 * d - NL[0]
            else:
                s2 = a[j - 1] + a[j]
                k = (a[j - 1] - a[j]) / s2
                c = 2 * sa[j - 1] * sa[j] / s2
                Rn[j] = c * R[j - 1] - k * L[j]
                Ln[j - 1] = k * R[j - 1] + c * L[j]
        for j in range(1, Nn):
            # the nasal tube's first segment IS the velar port: its area is the velum opening, so a closed velum
            # reflects nearly everything back and the nose stays quiet
            a_prev = va if j == 1 else nose_area[j - 1]
            s_prev = sv if j == 1 else sn[j - 1]
            s2 = a_prev + nose_area[j]
            k = (a_prev - nose_area[j]) / s2
            c = 2 * s_prev * sn[j] / s2
            NRn[j] = c * NR[j - 1] - k * NL[j]
            NLn[j - 1] = k * NR[j - 1] + c * NL[j]
        # turbulence at the constriction exit: a series pressure source (dipole) across junction jf -> jf + 1,
        # +z/2 on the downstream side and -z/2 upstream, carried by the normalized waves of those two segments
        for q in range(K):
            w = np.random.randn()
            nl[q] = (1 - a_lo) * w + a_lo * nl[q]
            nl2[q] = (1 - a_lo) * nl[q] + a_lo * nl2[q]
            nh[q] = (1 - a_hi) * nl2[q] + a_hi * nh[q]
            fa = famp[i0, q]
            if fa > 0:
                jf = fpos[i0, q]
                if jf < N - 1:
                    z = fric_gain * fa * (nl2[q] - nh[q])
                    Rn[jf + 1] += .5 * z * sa[jf + 1]
                    Ln[jf] -= .5 * z * sa[jf]
        for j in range(N):
            R[j] = Rn[j] * damp; L[j] = Ln[j] * damp
        for j in range(Nn):
            NR[j] = NRn[j] * damp; NL[j] = NLn[j] * damp
        # radiated volume velocity ~ area x pressure = sqrt(area) x normalized wave, at the lips and the nostrils
        out[n] = (sa[N - 1] * R[N - 1] + sn[Nn - 1] * NR[Nn - 1]) * (1 + lip)
    return out


def room(y, fs, rt60=.35, wet=.18, seed=1):
    """The world the voice is heard in, not part of the body: a small room's late reverberation (exponentially
    decaying noise, highs absorbed faster by air and walls).  Fully dry synthesis falls to digital silence between
    syllables, which no listener ever hears."""
    from scipy.signal import fftconvolve, lfilter
    rng = np.random.default_rng(seed)
    t = np.arange(int(rt60 * 1.5 * fs)) / fs
    ir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    ir = lfilter([.25], [1, -.75], ir)                    # air absorption: a gentle low-pass
    ir[:int(.008 * fs)] = 0                               # pre-delay: the first reflection arrives after ~8 ms
    ir /= np.sqrt(np.sum(ir ** 2))
    return y + wet * fftconvolve(y, ir)[:len(y)]


def speak(sentence, body=BODY, syl_ms=200.0, seed=0, with_room=True):
    n = max(8, int(round(body['L'] * FS / C)))
    sc = score(sentence, n, syl_ms)
    T = len(sc['st']) * HOP
    ts = np.arange(T) / HOP
    idx = np.arange(len(sc['st']))
    smooth = vd.smooth
    st = smooth(smooth(sc['st'], 50), 30)                 # tones glide into each other
    f0 = body['f0'] * 2 ** (np.interp(ts, idx, st) / 12)
    f0 *= 1 + body['vib'] * np.sin(2 * np.pi * 5.5 * ts / 1000)
    av = np.interp(ts, idx, smooth(smooth(sc['av'], 20), 15))  # voicing onsets/offsets ~30 ms, not a switch
    asp = np.interp(ts, idx, smooth(sc['asp'], 10))
    src = source(f0, av, asp, body['oq'], body['breath'], body['jitter'], body['shimmer'], body['fa'], float(FS), seed, ASP_GAIN)
    # articulators move smoothly; the acoustics follow log area (impedance), so a linear ramp to closure would
    # put the whole 500x impedance change into its last millisecond
    A = np.exp(smooth(np.log(np.maximum(sc['A'], 1e-4)), 12))
    vel = smooth(sc['vel'], 25)
    jv = int(round(.62 * n))
    nose = np.interp((np.arange(int(round(.62 * n))) + .5) / int(round(.62 * n)), [0, .3, .8, 1], [1.2, 1.8, 1.2, .5])
    out = tract_nasal(A, vel, sc['fpos'].astype(np.int64)[:, None], sc['famp'][:, None], float(HOP), src, .75, -.85, .9995, .5, jv,
                      nose, seed, FRIC_GAIN)
    y = np.diff(out, prepend=0.0)
    y = resample_poly(y, 1, FS // FS_OUT)
    y -= y.mean()
    if with_room:
        y = room(np.r_[y, np.zeros(int(.3 * FS_OUT))], FS_OUT)
    y /= np.abs(y).max() + 1e-9
    fade = int(.01 * FS_OUT)
    y[:fade] *= np.linspace(0, 1, fade); y[-fade:] *= np.linspace(1, 0, fade)
    return (.8 * y).astype(np.float32)


SENTENCES = {
    '01_nihao_mama': 'ni3 hao3 , ma1 ma5 . wo3 shi4 ni3 de5 hai2 zi5 .',
    '02_tianqi': 'jin1 tian1 de5 tian1 qi4 zhen1 hao3 , wo3 men5 yi4 qi3 chu1 qu4 wan2 ba5 .',
    '03_xiang_shenme': 'ni3 zai4 xiang3 shen2 me5 ne5 ?',
    # the same words as three reference lines (Yao 09, 14, 11), for side-by-side listening
    '04_ting_shazi': 'ting1 , sha1 zi5 zai4 chang4 ge1 .',
    '05_ni_jiushi': 'wei4 , ni3 jiu4 shi4 wo3 zai4 deng3 de5 ren2 ma5 ?',
    '06_ni_zai_kan': 'ni3 zai4 kan4 na3 er5 ?',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/voice_speak')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, s in SENTENCES.items():
        y = speak(s)
        p = os.path.join(args.out, name + '.wav')
        sf.write(p, y, FS_OUT, subtype='PCM_16')
        print(p, f'{len(y) / FS_OUT:.1f} s :', s)


if __name__ == '__main__':
    main()
