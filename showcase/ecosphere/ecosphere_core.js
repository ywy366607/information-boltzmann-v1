/* Eco-sphere core: the rain torus world and nested organisms of the Lean table `CGB.NestedWorld`,
   the same table and rates as information_boltzmann/core/cgb_nested_world_triton.py (world leap)
   and cgb_nested_organism_triton.py (exact organism events). Runs in the browser and in Node
   (cross-check: scripts/witness/ecosphere_js_vs_python.py). */
(function (root) {
  'use strict';

  function rng32(seed) {                       // mulberry32
    let s = (seed >>> 0) || 1;
    return function () {
      s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // mass action (Lean poolFactor, mass_action_up/down): pool conductance times exp(-dcellF/2)
  function up(n, V) { return n < V ? V - n : 0; }   // pool gains a quantum
  function dn(n, V) { return n > 0 ? n : 0; }       // pool loses a quantum
  function gauss(r) { return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); }
  // Poisson count: inversion for lam <= 20 (stops when the cumulative passes u), rounded normal beyond
  function pois(lam, u, r) {
    if (lam > 20) return Math.max(0, Math.floor(lam + Math.sqrt(lam) * gauss(r) + 0.5));
    let p = Math.exp(-lam), c = p, k = 0;
    while (u > c && k < 200) { k++; p = p * lam / k; c += p; }
    return k;
  }

  const SHIFT = [[1, 0], [0, 1], [-1, 0], [0, -1]];

  function Sphere(T, seed) {
    this.T = T;
    const W = T.world;
    this.G = W.G; this.V = W.V; this.kin = W.kappa_in; this.kdiff = W.kappa_diff; this.dt = W.dt;
    this.st = W.stimulus;
    const G = this.G, GG = G * G;
    this.kind = new Int8Array(GG);
    this.src = [];
    for (let x = 0; x < G; x++) for (let y = 0; y < G; y++) {
      const k = (x % 3 === 1 && y % 3 === 1) ? 2 : 1;
      this.kind[x * G + y] = k;
      if (k === 1) this.src.push(x * G + y);
    }
    this.nbr = new Int32Array(GG * 4);
    for (let c = 0; c < GG; c++) for (let d = 0; d < 4; d++) this.nbr[c * 4 + d] = this.nb(c, d);
    this.Bbody = W.B; this.kdis = W.kappa_dis; this.foodE = W.foodE || 0;
    this.reset(seed);
  }

  Sphere.prototype.reset = function (seed) {
    const GG = this.G * this.G;
    this.r = rng32(seed == null ? 1 : seed);
    this.f = new Int32Array(GG); this.w = new Int32Array(GG);
    this.fmid = new Int32Array(GG); this.wmid = new Int32Array(GG);
    this.send = new Int32Array(GG * 8);
    this.mu = new Float64Array(GG);
    this.left = new Int32Array(GG);
    this.X = gauss(this.r) * this.st.rate_sigma;
    this.occ = new Int32Array(GG).fill(-1);
    this.orgs = [];
    this.byId = new Map();
    this.nextId = 0;
    this.tick = 0;
    this.dissolved = 0;
  };

  // ---------- the world's outer environment (stimulus) ----------
  Sphere.prototype.rate = function () { return this.st.rain_rate * Math.exp(this.X); };
  Sphere.prototype.rainAt = function (c) { this.left[c] = this.st.rain_dur; };
  Sphere.prototype.nearestSource = function (x, y) {
    const G = this.G; let best = -1, bd = 1e9;
    for (const c of this.src) {
      const cx = (c / G) | 0, cy = c % G;
      let dx = Math.abs(cx - x), dy = Math.abs(cy - y);
      dx = Math.min(dx, G - dx); dy = Math.min(dy, G - dy);
      if (dx + dy < bd) { bd = dx + dy; best = c; }
    }
    return best;
  };
  Sphere.prototype.weather = function () {
    const s = this.st, a = Math.exp(-1 / s.rate_tau);
    this.X = a * this.X + s.rate_sigma * Math.sqrt(1 - a * a) * gauss(this.r);
    // Poisson number of new rains
    const lam = this.rate(); let L = Math.exp(-lam), k = 0, p = 1;
    do { k++; p *= this.r(); } while (p > L);
    for (let i = 0; i < k - 1; i++) this.rainAt(this.src[(this.r() * this.src.length) | 0]);
    const GG = this.G * this.G;
    for (let c = 0; c < GG; c++) {
      this.mu[c] = this.kind[c] === 2 ? s.mu_sink : (this.left[c] > 0 ? s.mu_rain : s.mu_dry);
      if (this.left[c] > 0) this.left[c]--;
    }
  };

  // ---------- world rates and leap (same as the fused Triton leap) ----------
  Sphere.prototype.nb = function (c, d) {
    const G = this.G, x = (c / G) | 0, y = c % G;
    return ((x + SHIFT[d][0] + G) % G) * G + ((y + SHIFT[d][1] + G) % G);
  };
  Sphere.prototype.worldRates = function (c) {
    const V = this.V, f = this.f[c], w = this.w[c], mu = this.mu[c], k = this.kind[c];
    const em = Math.exp(mu / 2), ep = Math.exp(-mu / 2), out = new Float64Array(12);
    const muf = mu - this.foodE;                       // food quanta carry foodE
    if (k === 1) { out[0] = this.kin * up(f, V) * Math.exp(muf / 2); out[1] = this.kin * dn(f, V) * Math.exp(-muf / 2); }
    if (k === 2) { out[2] = this.kin * up(w, V) * em; out[3] = this.kin * dn(w, V) * ep; }
    for (let d = 0; d < 4; d++) {
      const nc = this.nb(c, d);
      out[4 + d] = 2 * this.kdiff * dn(f, V) * up(this.f[nc], V);
      out[8 + d] = 2 * this.kdiff * dn(w, V) * up(this.w[nc], V);
    }
    return out;
  };
  Sphere.prototype.worldLeap = function () {
    const G = this.G, GG = G * G, V = this.V, dt = this.dt, r = this.r, kin = this.kin, k2 = 2 * this.kdiff;
    const F = this.f, Wp = this.w, S = this.send, nb = this.nbr;
    for (let c = 0; c < GG; c++) {
      const f = F[c], w = Wp[c], kd = this.kind[c];
      let fa = f, wa = w;
      if (kd === 1) {
        const mu = this.mu[c] - this.foodE;
        const l0 = kin * up(f, V) * Math.exp(mu / 2) * dt, l1 = kin * dn(f, V) * Math.exp(-mu / 2) * dt;
        fa += Math.min(l0 > 0 ? pois(l0, r(), r) : 0, (V - f) >> 1) - Math.min(l1 > 0 ? pois(l1, r(), r) : 0, f);
      } else if (kd === 2) {
        const mu = this.mu[c];
        const l2 = kin * up(w, V) * Math.exp(mu / 2) * dt, l3 = kin * dn(w, V) * Math.exp(-mu / 2) * dt;
        wa += Math.min(l2 > 0 ? pois(l2, r(), r) : 0, (V - w) >> 1) - Math.min(l3 > 0 ? pois(l3, r(), r) : 0, w);
      }
      const dnf = dn(f, V) * k2 * dt, dnw = dn(w, V) * k2 * dt;
      for (let d = 0; d < 4; d++) {
        const nc = nb[c * 4 + d];
        let sf = 0, sw = 0;
        if (dnf > 0) { const fb = F[nc]; sf = Math.min(pois(dnf * up(fb, V), r(), r), (V - fb) >> 3, fa); fa -= sf; }
        if (dnw > 0) { const wb = Wp[nc]; sw = Math.min(pois(dnw * up(wb, V), r(), r), (V - wb) >> 3, wa); wa -= sw; }
        S[c * 8 + d] = sf; S[c * 8 + 4 + d] = sw;
      }
      this.fmid[c] = fa; this.wmid[c] = wa;
    }
    for (let c = 0; c < GG; c++) {
      const s0 = nb[c * 4 + 2], s1 = nb[c * 4 + 3], s2 = nb[c * 4], s3 = nb[c * 4 + 1];
      F[c] = this.fmid[c] + S[s0 * 8] + S[s1 * 8 + 1] + S[s2 * 8 + 2] + S[s3 * 8 + 3];
      Wp[c] = this.wmid[c] + S[s0 * 8 + 4] + S[s1 * 8 + 5] + S[s2 * 8 + 6] + S[s3 * 8 + 7];
    }
    // The reverse of dissolution (a dead body forming from waste) has rate kappa_dis * exp(-|dE|)
    // with dE >= the cost of gathering B = 16 quanta of waste: below 1e-15 per cell-tick while
    // waste stays far below a half-full pool (formationBound); omitted.
  };
  // Content held by a body: B plus its units' content (state index).
  Sphere.prototype.bodyContent = function (o) {
    const O = this.T.organism, SM = O.SMAX; let c = this.Bbody;
    for (let p = 0; p < 6; p++) for (let s = 1; s < O.STATES[p]; s++) c += s * o.n[p * SM + s];
    return c;
  };
  // Lean dissolution: dE = cellF(w + m) - cellF(w) - bugE(y); Metropolis conductance
  // kappa_dis * exp(-|dE| / 2), so the rate is kappa_dis * min(1, exp(-dE)).
  Sphere.prototype.dissolveRate = function (o) {
    const O = this.T.organism, SM = O.SMAX, V = this.V, w = this.w[o.c], m = this.bodyContent(o);
    if (w + m > V) return 0;
    let dE = 0;
    for (let k = w; k < w + m; k++) dE += Math.log((k + 1) / (V - k));
    for (let p = 0; p < 6; p++) for (let s = 0; s < O.STATES[p]; s++) dE -= o.phi[p * SM + s] * o.n[p * SM + s];
    return this.kdis * Math.min(1, Math.exp(-dE));
  };
  // An upper bound on the formation rate of a dead body in a cell holding w waste quanta, summed
  // over every inner state (per-quantum factor e^{-phi_min} with phi_min = 0, the physics floor).
  Sphere.prototype.formationBound = function (w) {
    const O = this.T.organism, V = this.V, B = this.Bbody;
    if (w < B) return 0;
    let logp = 0;
    for (let k = w; k > w - B; k--) logp += Math.log(k / (V - k + 1));
    let units = 0; for (let p = 0; p < 5; p++) units += O.UNITS[p] * (O.STATES[p] - 1);
    // each further quantum taken from the pool multiplies by at most w / (V - w) (floor energy 0)
    const q = w / (V - w);
    return 4 * this.kdis * Math.exp(logp) * Math.pow(1 + q, units);
  };

  // ---------- organisms (exact events) ----------
  // Genome coordinates in [0, 1] -> numbers (as cgb_nested_population.Coder.decode): a prefactor at
  // coordinate 0 is an absent event (log -inf).
  Sphere.prototype.decode = function (theta) {
    const C = this.T.coder, g = {}, v = [];
    for (let i = 0; i < C.lo.length; i++) {
      const u = Math.min(1, Math.max(0, theta[i]));
      v.push(i >= C.nphi && u <= 0 ? -Infinity : C.lo[i] + u * (C.hi[i] - C.lo[i]));
    }
    let i = 0;
    for (const [k, n] of C.sizes) { if (n === 0) g[k] = v[i++]; else { g[k] = v.slice(i, i + n); i += n; } }
    return g;
  };
  Sphere.prototype.mutate = function (theta) {      // replication fidelity (cgb_nested_population.mutate)
    const t = Float64Array.from(theta), C = this.T.coder;
    if (this.r() < C.pmut) {
      const k = (this.r() * t.length) | 0, s = this.r() < 0.5 ? -1 : 1;
      t[k] = Math.min(1, Math.max(0, Math.round((t[k] + C.delta * s) / C.delta) * C.delta));
    }
    return t;
  };
  Sphere.prototype.place = function (x, y, genome, h, kind, parent) {
    const G = this.G, c = ((x % G) + G) % G * G + ((y % G) + G) % G;
    if (this.occ[c] >= 0) return null;
    const O = this.T.organism, SM = O.SMAX, n = new Int32Array(6 * SM);
    for (let p = 0; p < 6; p++) n[p * SM] = O.UNITS[p];
    n[O.S * SM] = 0; n[O.S * SM + 1] = O.UNITS[O.S];
    if (kind !== 'newborn') {                         // newborn (Lean yb): intact membrane, the rest empty
      const E = O.E, half = O.UNITS[E] >> 1;
      n[E * SM] = O.UNITS[E] - half; n[E * SM + 1] = half;
    }
    const theta = this.T.coder ? Float64Array.from(genome || this.T.theta0) : null;
    const g = theta ? this.decode(theta) : (genome || this.T.genome);
    const phi = new Float64Array(6 * SM);
    O.SELECTED_PHI.forEach((ps, i) => { phi[ps[0] * SM + ps[1]] = g.phi[i]; });
    phi[O.S * SM + 1] = O.PHI_S1;
    const kx = g.logk_x.map(Math.exp);
    for (let i = kx.length; i < O.EXCH.length; i++) kx.push(Math.exp(O.LOGK_LEAK));
    const o = { id: this.nextId++, n, phi, kr: g.logk_rule.map(Math.exp), kx, kflip: Math.exp(g.log_flip),
      kmove: Math.exp(g.log_move), kdiv: g.log_div == null ? 0 : Math.exp(g.log_div), theta,
      c, h: h == null ? (this.r() * 4) | 0 : h, born: this.tick, death: -1,
      intake: 0, moves: 0, gone: false, births: 0,
      gen: parent ? parent.gen + 1 : 0, lineage: parent ? parent.lineage : this.nextId - 1 };
    this.orgs.push(o); this.byId.set(o.id, o); this.occ[c] = o.id;
    return o;
  };
  Sphere.prototype.orgRates = function (o, out) {
    const O = this.T.organism, SM = O.SMAX, n = o.n, phi = o.phi, V = this.V;
    const fc = this.f[o.c], wc = this.w[o.c];
    let k = 0;
    const gate = n[O.GATE[0] * SM + O.GATE[1]];
    const pf = (pool, d) => (d > 0 ? up(pool, V) : (d < 0 ? dn(pool, V) : 1));   // mass action of a pool change
    for (let r = 0; r < O.RULES.length; r++) {
      const [i, j, a1, a2, b1, b2, df = 0, dw = 0, gated = false] = O.RULES[r];
      const dE = phi[i * SM + b1] + phi[j * SM + b2] - phi[i * SM + a1] - phi[j * SM + a2] + this.foodE * df;
      const kk = o.kr[r] * (gated ? gate : 1);
      out[k++] = kk * Math.exp(-dE / 2) * n[i * SM + a1] * (n[j * SM + a2] - (i === j && a1 === a2 ? 1 : 0)) * pf(fc, df) * pf(wc, dw);
      out[k++] = kk * Math.exp(dE / 2) * n[i * SM + b1] * (n[j * SM + b2] - (i === j && b1 === b2 ? 1 : 0)) * pf(fc, -df) * pf(wc, -dw);
    }
    for (let x = 0; x < O.EXCH.length; x++) {
      const [p, a, b, food, gated] = O.EXCH[x];
      const pool = food ? fc : wc, d = a - b;
      const kk = o.kx[x] * (gated ? gate : 1), dphi = phi[p * SM + b] - phi[p * SM + a] + (food ? this.foodE * d : 0);
      out[k++] = kk * Math.exp(-dphi / 2) * (d > 0 ? up(pool, V) : dn(pool, V)) * n[p * SM + a];
      out[k++] = kk * Math.exp(dphi / 2) * (d > 0 ? dn(pool, V) : up(pool, V)) * n[p * SM + b];
    }
    const running = n[O.M * SM + O.RUN] > 0;
    out[k++] = running ? 0 : 3 * o.kflip;
    const [i, j, a1, a2, b1, b2] = O.FUEL;
    const dE = phi[i * SM + b1] + phi[j * SM + b2] - phi[i * SM + a1] - phi[j * SM + a2];
    const ahead = this.nb(o.c, o.h), behind = this.nb(o.c, (o.h + 2) % 4);
    out[k++] = running && this.occ[ahead] === -1 ? o.kmove * Math.exp(-dE / 2) * n[i * SM + a1] * n[j * SM + a2] : 0;
    out[k++] = running && this.occ[behind] === -1 ? o.kmove * Math.exp(dE / 2) * n[i * SM + b1] * n[j * SM + b2] : 0;
    // dissolution of a body without intact membrane, with all it holds (the event of its own heading)
    out[k++] = n[O.GATE[0] * SM + O.GATE[1]] === 0 ? this.dissolveRate(o) : 0;   // dead membrane
    // division (Lean recipes): 4 headings x kdiv x empty neighbours x Metropolis, division page full
    out[k++] = this.divRate(o);
    return k;
  };
  Sphere.prototype.emptyNbrs = function (c) {
    const e = []; for (let d = 0; d < 4; d++) { const nc = this.nb(c, d); if (this.occ[nc] === -1) e.push(nc); } return e;
  };
  Sphere.prototype.divRate = function (o) {
    const O = this.T.organism, SM = O.SMAX, D = O.DIV;
    if (!D || !(o.kdiv > 0) || o.pending) return 0;
    const n = o.n;
    if (n[O.GATE[0] * SM + O.GATE[1]] === 0 || n[O.E * SM + 1] !== D.BQ) return 0;
    const ne = this.emptyNbrs(o.c).length;
    if (ne === 0) return 0;
    const dE = D.NS * O.PHI_S1 - D.BQ * (o.phi[O.E * SM + 1] - o.phi[O.E * SM]);
    return 4 * o.kdiv * ne * Math.min(1, Math.exp(-dE));
  };
  Sphere.prototype.orgLeap = function (o, buf) {
    const O = this.T.organism, SM = O.SMAX, n = o.n, r = this.r, dt = this.dt;
    let t = 0;
    for (let it = 0; it < 100000; it++) {
      const C = this.orgRates(o, buf);
      let tot = 0;
      for (let q = 0; q < C; q++) tot += buf[q];
      if (!(tot > 0)) return;
      t += -Math.log(1 - r()) / tot;
      if (t >= dt) return;
      let u = r() * tot, q = 0;
      while (q < C - 1 && u >= buf[q]) { u -= buf[q]; q++; }
      const frac = u / buf[q];
      let k = 0;
      const pair = (i, s0, j, s1, i2, d0, j2, d1) => { n[i * SM + s0]--; n[j * SM + s1]--; n[i2 * SM + d0]++; n[j2 * SM + d1]++; };
      if (q < 2 * O.RULES.length) {
        const [i, j, a1, a2, b1, b2, df = 0, dw = 0] = O.RULES[q >> 1];
        const sg = (q & 1) ? -1 : 1;
        if (q & 1) pair(i, b1, j, b2, i, a1, j, a2); else pair(i, a1, j, a2, i, b1, j, b2);
        if (df) { this.f[o.c] += sg * df; o.intake -= sg * df; }
        if (dw) this.w[o.c] += sg * dw;
        continue;
      }
      k = 2 * O.RULES.length;
      if (q < k + 2 * O.EXCH.length) {
        const [p, a, b, food, gated] = O.EXCH[(q - k) >> 1];
        const fwd = ((q - k) & 1) === 0, s0 = fwd ? a : b, s1 = fwd ? b : a, d = s0 - s1;
        n[p * SM + s0]--; n[p * SM + s1]++;
        if (food) { this.f[o.c] += d; o.intake -= d; } else this.w[o.c] += d;
        continue;
      }
      k += 2 * O.EXCH.length;
      if (q === k + 4) {                                   // divide into the frac-th empty neighbour
        // a world of SL body slots (cgb_nested_population): with none free the event changes nothing
        if (this.free != null) { if (this.free <= 0) continue; this.free--; }
        const D = O.DIV, e = this.emptyNbrs(o.c), to = e[Math.min(e.length - 1, (frac * e.length) | 0)];
        n[O.E * SM + 1] -= D.BQ; n[O.E * SM] += D.BQ;
        o.pending = true; o.births++;
        this.newborns.push({ c: to, theta: this.mutate(o.theta), h: (this.r() * 4) | 0, parent: o });
        this.occ[to] = -2;                                 // the place is the newborn's from now on
        continue;
      }
      if (q === k + 3) {                                   // dissolve: the body leaves the world
        this.w[o.c] += this.bodyContent(o); this.occ[o.c] = -1; o.gone = true; o.goneAt = this.tick; this.dissolved++;
        return;
      }
      if (q === k) { o.h = (o.h + 1 + Math.min(2, (frac * 3) | 0)) % 4; continue; }
      const [i, j, a1, a2, b1, b2] = O.FUEL;
      const fwd = q === k + 1;
      if (fwd) pair(i, a1, j, a2, i, b1, j, b2); else pair(i, b1, j, b2, i, a1, j, a2);
      const to = this.nb(o.c, fwd ? o.h : (o.h + 2) % 4);
      this.occ[o.c] = -1; this.occ[to] = o.id; o.c = to; o.moves++;
    }
  };

  Sphere.prototype.orgAt = function (c) { const id = this.occ[c]; return id < 0 ? null : this.byId.get(id); };
  Sphere.prototype.placeAt = function (c, genome) { const G = this.G; return this.place((c / G) | 0, c % G, genome); };
  Sphere.prototype.step = function () {
    this.weather();
    const nl = Math.max(1, Math.round(1 / this.dt));
    const O = this.T.organism, SM = O.SMAX, nbuf = 2 * O.RULES.length + 2 * O.EXCH.length + 8;
    const buf = this._buf && this._buf.length >= nbuf ? this._buf : (this._buf = new Float64Array(nbuf));
    this.newborns = [];
    // free slots are counted at the start of the tick; a dissolved body frees its slot at the end
    this.free = this.T.world.SL ? this.T.world.SL - this.orgs.length : null;
    for (let l = 0; l < nl; l++) {
      this.worldLeap();
      for (const o of this.orgs) if (!o.gone) this.orgLeap(o, buf);
    }
    // births join at the end of the tick (as cgb_nested_population.end_tick); one per parent per tick
    for (const b of this.newborns) {
      this.occ[b.c] = -1;
      this.place((b.c / this.G) | 0, b.c % this.G, b.theta, b.h, 'newborn', b.parent);
      this.births = (this.births || 0) + 1;
    }
    for (const o of this.orgs) o.pending = false;
    this.tick++;
    let anyGone = false;
    for (const o of this.orgs) {
      if (o.gone) { anyGone = true; continue; }
      if (o.death < 0 && o.n[O.S * SM + 1] === 0) o.death = this.tick;
    }
    if (anyGone) {                       // dissolved bodies leave the world for good
      this.orgs = this.orgs.filter(o => { if (o.gone) this.byId.delete(o.id); return !o.gone; });
    }
  };

  const api = { Sphere, rng32, up, dn, pois };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Ecosphere = api;
})(typeof window !== 'undefined' ? window : globalThis);
