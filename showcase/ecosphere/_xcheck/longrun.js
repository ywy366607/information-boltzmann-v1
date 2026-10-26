// Long run of the built eco-sphere page's tables: bounds, no NaN, deaths and dissolution.
const E = require('../ecosphere_core.js');
const fs = require('fs');
const html = fs.readFileSync(__dirname + '/../ecosphere.html', 'utf8');
const a0 = html.indexOf('const T = ') + 10, a1 = html.indexOf('(function', a0);
const T = JSON.parse(html.slice(a0, a1).trim().replace(/;$/, ''));
const ticks = +(process.argv[2] || 30000);
const S = new E.Sphere(T, 42);
for (let i = 0; i < 400; i++) S.step();
const O = T.organism, SM = O.SMAX, G = S.G, V = S.V;
function add(k) { let t = 0; while (k > 0 && t++ < 999) { if (S.place((Math.random() * G) | 0, (Math.random() * G) | 0)) k--; } }
add(20);
const t0 = Date.now();
for (let t = 1; t <= ticks; t++) {
  S.step();
  if (t % 5000 === 0) add(10);
  if (t % 2500 === 0) {
    let bad = 0, live = 0, dead = 0, food = 0, waste = 0;
    for (let c = 0; c < G * G; c++) {
      if (!(S.f[c] >= 0 && S.f[c] <= V && S.w[c] >= 0 && S.w[c] <= V)) bad++;
      food += S.f[c]; waste += S.w[c];
    }
    for (const o of S.orgs) { for (const v of o.n) if (!(v >= 0)) bad++; if (o.n[O.S * SM + 1] > 0) live++; else dead++; }
    const ages = S.orgs.filter(o => o.n[O.S * SM + 1] > 0).map(o => S.tick - o.born);
    console.log(JSON.stringify({ t, live, dead, dissolved: S.dissolved, food, waste, bad,
      maxAge: Math.max(0, ...ages), s: ((Date.now() - t0) / 1000).toFixed(0) }));
  }
}
