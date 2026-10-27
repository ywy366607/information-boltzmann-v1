// Per-individual motion audit of the eco-sphere page species (JS core, page tables):
// usage: node scripts/witness/ecosphere_individual_motion.js <ecosphere_core.js abs path> <tables.json>
// (tables = the `const T` object of showcase/ecosphere/ecosphere_v2.html). 6 worlds, 12 founders, <= 4000 ticks.
const E = require(process.argv[2]); const T = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const O = T.organism, SM = O.SMAX;
const rec = [];
for (const seed of [11, 12, 13, 14, 15, 16]) {
  const S = new E.Sphere(T, seed);
  for (let i = 0; i < 400; i++) { S.weather(); for (let l = 0; l < 20; l++) S.worldLeap(); S.tick++; }
  S.tick = 0; let k = 12;
  while (k > 0) { const c = (S.r() * S.G * S.G) | 0; if (S.placeAt(c)) k--; }
  const info = new Map();
  for (let t = 1; t <= 4000 && S.orgs.length < 120; t++) {
    S.step();
    for (const o of S.orgs) {
      let r = info.get(o.id);
      if (!r) { r = { id: o.id, seed, gen: o.gen, born: o.born, food0: S.f[o.c], run: 0, Fb: 0, Y: 0, alive: 0, death: -1, moves: 0, intake: 0, Emax: 0 }; info.set(o.id, r); }
      const alive = o.n[O.S * SM + 1] > 0;
      if (alive) {
        r.alive++; if (o.n[O.M * SM + O.RUN] > 0) r.run++;
        if (o.n[O.F * SM + 1] + o.n[O.F * SM + 2] > 0) r.Fb++;
        if (o.n[O.Y * SM + 1] > 0) r.Y++;
        r.Emax = Math.max(r.Emax, o.n[O.E * SM + 1]);
      } else if (r.death < 0) r.death = S.tick;
      r.moves = o.moves; r.intake = o.intake; r.foodHere = S.f[o.c];
    }
  }
  rec.push(...info.values());
}
const dead = rec.filter(r => r.death >= 0);
const still = dead.filter(r => r.moves === 0), moved = dead.filter(r => r.moves > 0);
const sum = (a, f) => a.length ? +(a.reduce((s, r) => s + f(r), 0) / a.length).toFixed(3) : null;
const desc = a => ({ n: a.length, ever_ran: a.filter(r => r.run > 0).length, ever_bound_food: a.filter(r => r.Fb > 0).length,
  ever_Y: a.filter(r => r.Y > 0).length, mean_life: sum(a, r => r.alive), mean_intake: sum(a, r => r.intake), mean_food_at_birth: sum(a, r => r.food0) });
console.log(JSON.stringify({ bodies: rec.length, dead: dead.length, dead_never_moved: desc(still), dead_moved: desc(moved),
  alive_end: rec.filter(r => r.death < 0).length }, null, 1));
console.log('examples never moved:', JSON.stringify(still.slice(0, 6).map(r => ({ seed: r.seed, id: r.id, gen: r.gen, life: r.alive, run: r.run, Fbound_ticks: r.Fb, Y_ticks: r.Y, intake: r.intake, food_at_birth: r.food0 }))));
