"""Fit vowel area functions for a given tract length to target formants, analytically (2026-10-19).
Transfer function of a tube of cylindrical segments (chain matrices, glottis closed, lips open, small losses) ->
formants as peaks of |U_lips / U_glottis|; a 5-parameter area model per vowel (pharynx area, constriction place,
constriction area, oral area, lip area) fitted with Nelder-Mead.
Usage: python vowel_fit.py [--L 0.13]   (prints the fitted parameters as a Python dict)"""
import argparse, json, numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks

C, RHO = 350.0, 1.14
# targets (F1, F2, F3) for an adult female Mandarin speaker; scaled by tract length below
TARGETS_F = {'a': (1000, 1550, 2900), 'A': (900, 1300, 2900), 'i': (330, 2800, 3500), 'v': (330, 2100, 2700),
             'u': (380, 750, 2800), 'o': (600, 950, 2800), 'e': (520, 1300, 2900), 'E': (650, 2200, 2900),
             '@': (600, 1600, 2900), 'R': (600, 1500, 1900), 'z': (400, 1600, 2900), 'r': (400, 1700, 2300)}
L_FEMALE = 0.145


def area_model(p, n):
    ap, xc, ac, ao, al = p
    x = (np.arange(n) + .5) / n
    a = np.where(x < xc, ap, ao).astype(float)
    a = a + (np.minimum(ap, ao) - a) * 0                       # piecewise base
    a = np.minimum(a, ac + (np.maximum(ap, ao) - ac) * (1 - np.exp(-((x - xc) / .08) ** 2)))
    a[x < .08] = .6                                             # larynx tube
    a[x > .92] = al                                             # lips
    return np.clip(a, .05, 12)


def formants(a, L, fmax=5000):
    n = len(a)
    l = L / n
    f = np.arange(80, fmax, 5.0)
    w = 2 * np.pi * f
    k = w / C * (1 - .004j)                                     # small losses: finite bandwidths
    M11 = np.ones_like(f, complex); M12 = np.zeros_like(f, complex)
    M21 = np.zeros_like(f, complex); M22 = np.ones_like(f, complex)
    for Ai in a:                                                # from glottis to lips
        Z = RHO * C / (Ai * 1e-4)
        c, s = np.cos(k * l), np.sin(k * l)
        A11, A12, A21, A22 = c, 1j * Z * s, 1j * s / Z, c
        M11, M12, M21, M22 = M11 * A11 + M12 * A21, M11 * A12 + M12 * A22, M21 * A11 + M22 * A21, M21 * A12 + M22 * A22
    H = 1 / np.abs(M22)                                         # U_lips / U_glottis with p_lips = 0
    pk, _ = find_peaks(20 * np.log10(H + 1e-12))
    return f[pk][:4]


def fit_vowel(target, L, n, seed=0):
    def loss(p):
        if not (0.3 <= p[0] <= 9 and .3 <= p[1] <= .9 and .15 <= p[2] <= 6 and .3 <= p[3] <= 9 and .1 <= p[4] <= 6):
            return 1e3
        F = formants(area_model(p, n), L)
        if len(F) < 3:
            return 1e3
        return float(sum((np.log(F[i] / target[i])) ** 2 * w for i, w in zip(range(3), (1, 1, .4))))
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(12):
        p0 = [rng.uniform(1, 6), rng.uniform(.4, .85), rng.uniform(.3, 2), rng.uniform(1, 6), rng.uniform(.3, 4)]
        r = minimize(loss, p0, method='Nelder-Mead', options={'maxiter': 600, 'xatol': 1e-3, 'fatol': 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=float, default=.13)
    ap.add_argument('--fs', type=float, default=96000)
    args = ap.parse_args()
    n = int(round(args.L * args.fs / C))
    scale = L_FEMALE / args.L
    out = {}
    for v, t in TARGETS_F.items():
        tgt = tuple(x * scale for x in t)
        r = fit_vowel(tgt, args.L, n)
        F = formants(area_model(r.x, n), args.L)
        out[v] = [round(float(x), 3) for x in r.x]
        print(f"{v}: target {[int(x) for x in tgt]} fitted {[int(x) for x in F[:3]]} params {out[v]}")
    print(json.dumps(out))


if __name__ == '__main__':
    main()
