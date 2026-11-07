/**
 * Calibrates the sprinklers' wetted fraction (suppression.js WETTED_FRACTION)
 * and checks it against Evans's density dependence.
 *
 * On the solver tests' 5 x 4 x 2.7 m room with a class A first item, a
 * head over the fire opens at a fixed time. What is compared is the fuel
 * burnt in the cells under it after it opens, and the burning rate's fitted
 * decay r(t) = r0 exp(-(t - t_act) / tau). On this room the free fire's
 * covered fuel already burns down (its first item is consumed), and Evans's
 * cap also stops covered cells not yet burning from igniting (his premise
 * that a sprinklered fire does not grow), so the fitted tau under the cap
 * is far below Evans's own: the cap bounds the fire, the fit is not his tau.
 * The wetted share is therefore found by bisection so that wetting burns
 * the same covered fuel after opening as Evans's cap does at the light
 * hazard design density (85 L/min over 20.9 m², 0.068 mm/s); then the two
 * are compared at the other densities of Evans's crib table (0.026 to
 * 0.133 mm/s), where wetting follows the water and the cap follows Evans's
 * 3.0 w^-1.85.
 *
 *   node tools/wetting-calibrate.mjs [--open 30] [--horizon 330] [--only check]
 */
import { MATERIAL } from "../topologicpy-web-frontend/src/core/voxelize.js";
import { buildScene } from "../topologicpy-web-frontend/src/core/flamescapesRun.js";
import { createSolver, step } from "../topologicpy-web-frontend/src/core/flamescapes.js";
import { createSuppression, evansTau, headDensity, openHeads, sprinklerSetup } from "../topologicpy-web-frontend/src/core/suppression.js";
import { fuelsFor } from "../topologicpy-web-frontend/src/core/fireClasses.js";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const OPEN = Number(flag("--open", 30));
const HORIZON = Number(flag("--horizon", 330));
const ONLY = flag("--only", null);

function box(x0, y0, z0, x1, y1, z1, material = MATERIAL.STONE) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material };
}
/** The solver tests' room (suppression.test.mjs), y up, with a door. */
function room(t = 0.2) {
  const W = 5, D = 4, H = 2.7;
  return {
    walls: [box(-t, 0, -t, W + t, H, 0), box(-t, 0, D, W + t, H, D + t), box(-t, 0, 0, 0, H, D), box(W, 0, 0, W + t, H, 1.5), box(W, 0, 2.4, W + t, H, D), box(W, 2.0, 1.5, W + t, H, 2.4)],
    floors: [box(-t, -t, -t, W + t, 0, D + t)],
    roofs: [box(-t, H, -t, W + t, H + t, D + t)],
    stairs: [],
    obstacles: [],
  };
}
const headAt = (flowLpm) => ({ id: "over", position: [2.5, 2.55, 2], footprint: { centre: [2.5, 2], axis: [1, 0], half: [2.3, 2.3] }, rti: 50, c: 0.4, activationC: 68, flowLpm, coverage: 20.9 });
const { contentsFuel, ignitionFuel, item } = fuelsFor("A");

/** One run; the covered cells' burning rate (kg/s) in 5 s bins after the head opens, and the fitted tau. */
function run({ mode, wet = undefined, flowLpm = 85 }) {
  const scene = buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.35, maxCell: 0.35, contentsFuel, ignitionFuel, item });
  const sim = createSolver(scene, { contentsFuel, ignitionFuel, seed: 1 });
  const head = headAt(flowLpm);
  const setup = sprinklerSetup(scene, [head], { upAxis: "y", fuel: sim.fuelMass });
  const covered = setup.heads[0].covered;
  const sup = mode ? createSuppression(sim.N, { mode, ...(wet !== undefined ? { wet } : {}) }) : null;
  if (sup) sim.suppression = sup;
  const bins = [];
  let opened = false, acc = 0, accT = 0, released = 0, burnt = 0;
  while (sim.time < HORIZON - 1e-9) {
    if (!opened && sim.time >= OPEN) {
      if (sup) openHeads(sup, setup, [{ head: 0, time: sim.time }]);
      opened = true;
    }
    const dt = Math.min(sim.P.dt, HORIZON - sim.time);
    const before = sim.stats.released;
    step(sim, dt);
    if (opened) {
      let loss = 0;
      for (const c of covered) loss += sim.solidLoss[c] || 0;
      acc += loss;
      burnt += loss;
      accT += dt;
      released += sim.stats.released - before;
      if (accT >= 5 - 1e-9) {
        bins.push({ t: sim.time - OPEN, rate: acc / accT });
        acc = 0;
        accT = 0;
      }
    }
  }
  // Least squares on ln(rate) from 10 s after opening, over bins still burning.
  const pts = bins.filter((b) => b.t >= 10 && b.rate > 1e-7);
  let tau = null;
  if (pts.length >= 4) {
    const n = pts.length;
    const mx = pts.reduce((a, b) => a + b.t, 0) / n, my = pts.reduce((a, b) => a + Math.log(b.rate), 0) / n;
    let sxy = 0, sxx = 0;
    for (const p of pts) {
      sxy += (p.t - mx) * (Math.log(p.rate) - my);
      sxx += (p.t - mx) ** 2;
    }
    const slope = sxy / sxx;
    tau = slope < 0 ? -1 / slope : Infinity;
  }
  return { tau, bins, released, burnt, density: headDensity(head), wetted: sup?.stats?.wetted ?? 0, emitted: sup?.stats?.emitted ?? 0 };
}

const fmt = (x) => (x === null ? "n/a" : !Number.isFinite(x) ? "grows" : `${Math.round(x)} s`);
const kg = (x) => `${x.toFixed(3)} kg`;
const t0 = performance.now();
const design = headDensity(headAt(85));
console.log(`Room 5 x 4 x 2.7 m, class A, the head over the fire opens at ${OPEN} s; ${HORIZON} s.`);
console.log(`Design density ${design.toFixed(4)} mm/s: Evans's tau ${Math.round(evansTau(design))} s.`);
const free = run({ mode: null });
console.log(`  free fire     covered fuel burnt after opening ${kg(free.burnt)}, tau ${fmt(free.tau)}`);
const evans = run({ mode: "evans" });
console.log(`  Evans's cap   ${kg(evans.burnt)}, tau ${fmt(evans.tau)}`);
if (ONLY === "check") process.exit(0);

// Bisection on the wetted share (log scale): wetting burns as much covered fuel as the cap.
let lo = 1e-4, hi = 1;
for (let it = 0; it < 10; it += 1) {
  const mid = Math.sqrt(lo * hi);
  const r = run({ mode: "wetting", wet: mid });
  console.log(`  wetting ${mid.toExponential(2)}  ${kg(r.burnt)}, tau ${fmt(r.tau)}; ${r.wetted.toFixed(2)} of ${r.emitted.toFixed(2)} kg of water on the fuel`);
  // More water on the fuel, less burnt.
  if (r.burnt > evans.burnt) lo = mid;
  else hi = mid;
}
const wet = Math.sqrt(lo * hi);
console.log(`
Calibrated wetted share ${wet.toExponential(3)}.`);
console.log("By density (Evans's crib table), covered fuel burnt after opening:");
for (const w of [0.026, 0.05, 0.068, 0.1, 0.133]) {
  const flowLpm = w * 20.9 * 60;
  const a = run({ mode: "wetting", wet, flowLpm });
  const b = run({ mode: "evans", flowLpm });
  console.log(`  ${w.toFixed(3)} mm/s  wetting ${kg(a.burnt)} (tau ${fmt(a.tau)})  Evans's cap ${kg(b.burnt)} (tau ${fmt(b.tau)}, his ${Math.round(evansTau(w))} s)  free ${kg(free.burnt)}`);
}
console.log(`
${((performance.now() - t0) / 1000).toFixed(0)} s`);
