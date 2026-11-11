/**
 * Proposal F (wiki/research/fire-calibration-claim-protocol-2026-11-12.md):
 * the heat-conserving transport. tools/calibration/heat-closure.mjs runs the
 * same checks at the app's three cell sizes.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULTS, NS, SOLID, createSolver, gasDensity, mixtureCp, step } from "./flamescapes.js";
import * as V from "./voxelize.js";

const TA = DEFAULTS.ambient;
const INERT_FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };

function box(x0, y0, z0, x1, y1, z1) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material: V.MATERIAL.STONE };
}

/** A sealed 3 x 2.5 x 3 m stone box, no wall exchange, no radiation; its lowest metre `hot` K above ambient. */
function sealedBox(options, hot = 300) {
  const h = 0.34, W = 3, H = 2.5, D = 3, t = 0.4;
  const boxes = [box(-t, -t, -t, W + t, 0, D + t), box(-t, H, -t, W + t, H + t, D + t), box(-t, 0, -t, W + t, H, 0), box(-t, 0, D, W + t, H, D + t), box(-t, 0, 0, 0, H, D), box(W, 0, 0, W + t, H, D)];
  const lo = [-t - h, -t, -t - h], hi = [W + t + h, H + t + h, D + t + h];
  const n = [0, 1, 2].map((d) => Math.ceil((hi[d] - lo[d]) / h - 1e-6));
  const grid = { nx: n[0], ny: n[1], nz: n[2], h, origin: lo, axes: [0, 1, 2], upAxis: "y", cells: n[0] * n[1] * n[2] };
  const fraction = V.volumeFractions(grid, boxes, { rays: V.raysFor(grid) });
  const solid = V.occupancy(grid, fraction);
  const scene = { grid, fraction, solid, contents: { cells: [], masses: [] }, ignition: null, pilot: null, fuels: { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL } };
  const sim = createSolver(scene, { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL, seed: 1, wallConvection: 0, emitAbove: 1e9, ...options });
  const inside = (c) => {
    const i = c % n[0], j = Math.floor(c / n[0]) % n[1], k = Math.floor(c / (n[0] * n[1]));
    const x = lo[0] + (i + 0.5) * h, y = lo[1] + (j + 0.5) * h, z = lo[2] + (k + 0.5) * h;
    return x > 0 && x < W && y > 0 && y < H && z > 0 && z < D ? y : null;
  };
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] !== SOLID && inside(c) !== null && inside(c) < 1) sim.T[c] = TA + hot;
  const totals = () => {
    let heat = 0, mass = 0;
    for (let c = 0; c < sim.N; c += 1) {
      if (sim.kind[c] === SOLID || inside(c) === null) continue;
      const rho = gasDensity(sim.T[c], sim.Y, c * NS);
      heat += rho * sim.V * mixtureCp(sim.Y, c * NS) * (sim.T[c] - TA);
      mass += rho * sim.V;
    }
    return { heat, mass };
  };
  return { sim, totals, inside };
}

test("F: hot gas mixing in a sealed box keeps its heat and its mass (the paper's transport made heat)", () => {
  const run = (options) => {
    const { sim, totals } = sealedBox(options);
    const start = totals();
    while (sim.time < 20 - 1e-9) step(sim, 0.05);
    const end = totals();
    return { heat: end.heat / start.heat, mass: end.mass / start.mass };
  };
  const now = run({});
  assert.ok(Math.abs(now.heat - 1) < 0.01, `heat kept ${now.heat}`);
  assert.ok(Math.abs(now.mass - 1) < 0.005, `mass kept ${now.mass}`);
  const before = run({ conservative: 0, expansion: 0 });
  assert.ok(before.heat > 1.1, `the back-traced T made heat: ${before.heat}`);
});

test("F: a uniform gas stays uniform whatever flow the projection leaves", () => {
  // The whole box 100 K above ambient and a stirring flow: no hot or cold spots appear.
  const { sim, inside } = sealedBox({}, 0);
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] !== SOLID && inside(c) !== null) sim.T[c] = TA + 100;
  for (let q = 0; q < sim.u.length; q += 1) sim.u[q] = Math.sin(q * 0.37) * 0.8;
  for (let q = 0; q < sim.w.length; q += 1) sim.w[q] = Math.cos(q * 0.29) * 0.8;
  for (let k = 0; k < 20; k += 1) step(sim, 0.05);
  let lo = Infinity, hi = -Infinity;
  for (let c = 0; c < sim.N; c += 1) {
    if (sim.kind[c] === SOLID || inside(c) === null) continue;
    lo = Math.min(lo, sim.T[c]);
    hi = Math.max(hi, sim.T[c]);
  }
  assert.ok(hi - lo < 0.5, `spread ${hi - lo} K`);
});
