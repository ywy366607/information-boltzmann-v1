// Crowding audit of the eco-sphere page species (JS core, page tables): for dead bodies that never moved,
// were they blocked (both cells along the heading occupied while the motor ran) or out of fuel?
// usage: node scripts/witness/ecosphere_crowding.js <ecosphere_core.js abs path> <tables.json>
const E = require(process.argv[2]); const T = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const O = T.organism, SM = O.SMAX, rec = new Map();
for (const seed of [11, 12, 13, 14, 15, 16]) {
  const S = new E.Sphere(T, seed);
  for (let i = 0; i < 400; i++) { S.weather(); for (let l = 0; l < 20; l++) S.worldLeap(); S.tick++; }
  S.tick = 0; let k = 12;
  while (k > 0) { const c = (S.r() * S.G * S.G) | 0; if (S.placeAt(c)) k--; }
  for (let t = 1; t <= 4000 && S.orgs.length < 120; t++) {
    S.step();
    for (const o of S.orgs) {
      const key = seed + ':' + o.id; let r = rec.get(key);
      if (!r) { r = { run: 0, runBlocked: 0, runNoFuel: 0, nbOcc: 0, alive: 0, death: -1, moves: 0, newborn: o.gen > 0 }; rec.set(key, r); }
      if (o.n[O.S * SM + 1] > 0) {
        r.alive++;
        let occ = 0; for (let d = 0; d < 4; d++) if (S.occ[S.nb(o.c, d)] !== -1) occ++;
        r.nbOcc += occ;
        if (o.n[O.M * SM + O.RUN] > 0) {
          r.run++;
          const ahead = S.nb(o.c, o.h), behind = S.nb(o.c, (o.h + 2) % 4);
          if (S.occ[ahead] !== -1 && S.occ[behind] !== -1) r.runBlocked++;
          const [i, j, a1, a2] = O.FUEL; if (o.n[i * SM + a1] * o.n[j * SM + a2] === 0) r.runNoFuel++;
        }
      } else if (r.death < 0) r.death = S.tick;
      r.moves = o.moves;
    }
  }
}
const all = [...rec.values()], dead = all.filter(r => r.death >= 0);
const g = (a) => {
  const run = a.reduce((s, r) => s + r.run, 0);
  return { n: a.length, newborns: a.filter(r => r.newborn).length,
    mean_occupied_neighbours: +(a.reduce((s, r) => s + r.nbOcc, 0) / Math.max(1, a.reduce((s, r) => s + r.alive, 0))).toFixed(2),
    run_ticks: run, run_both_ways_blocked_share: +(a.reduce((s, r) => s + r.runBlocked, 0) / Math.max(1, run)).toFixed(3),
    run_no_fuel_share: +(a.reduce((s, r) => s + r.runNoFuel, 0) / Math.max(1, run)).toFixed(3) };
};
console.log(JSON.stringify({
  never_moved_never_ran: g(dead.filter(r => r.moves === 0 && r.run === 0)),
  never_moved_but_ran: g(dead.filter(r => r.moves === 0 && r.run > 0)),
  dead_moved: g(dead.filter(r => r.moves > 0)),
  alive_at_end: g(all.filter(r => r.death < 0)) }, null, 1));
