#!/usr/bin/env node
/**
 * A fingerprint of the CPU Flamescapes solver's whole state after a run, and
 * its ms a step by stage (2026-11-09). A change meant only to make the solver
 * faster must leave both fingerprints as they were, bit for bit: T, Ts, Y,
 * u, v, w, rho, p, fuel, moisture, the particles' positions and the energy.
 * The CPU draws its random numbers from the GPU's per-cell streams, so a run
 * repeats exactly. Two cases: the GPU check's test room with a 200 kW pilot,
 * and with the app's default class A design fire.
 *
 *   node tools/flamescapes-fingerprint.mjs [steps]   (default 300)
 */
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const core = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../topologicpy-web-frontend/src/core");
const STEPS = Number(process.argv[2] || 300);
const imp = (f) => import(pathToFileURL(path.join(core, f)).href);
const V = await imp("voxelize.js");
const F = await imp("flamescapes.js");
const R = await imp("flamescapesRun.js");
const FC = await imp("fireClasses.js");
const { fingerprint } = await imp("runCache.js");

function box(x0, y0, z0, x1, y1, z1) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material: V.MATERIAL.STONE };
}
const W = 5, D = 4, H = 2.7, t = 0.2;
const geometry = {
  floors: [box(-t, -t, -t, W + t, 0, D + t)],
  roofs: [box(-t, H, -t, W + t, H + t, D + t)],
  walls: [box(-t, 0, -t, W + t, H, 0), box(-t, 0, D, W + t, H, D + t), box(-t, 0, 0, 0, H, D), box(W, 0, 0, W + t, H, 1.5), box(W, 0, 2.4, W + t, H, D), box(W, 2.0, 1.5, W + t, H, 2.4)],
  stairs: [], obstacles: [],
};
const { contentsFuel, ignitionFuel, item } = FC.fuelsFor("A");
const cases = {
  "room 200 kW pilot": () => {
    const s = R.buildScene(geometry, { upAxis: "y", fire: [2.5, 0, 2.0], maxCells: 30000, contentsFuel, ignitionFuel, item, designFire: null, fireLoad: 780e6, seed: 1 });
    s.pilot = { ...s.pilot, power: 200000, temperature: undefined, duration: 60 };
    return s;
  },
  "room design fire": () => R.buildScene(geometry, { upAxis: "y", fire: [2.5, 0, 2.0], maxCells: 30000, contentsFuel, ignitionFuel, item, designFire: FC.designFireFor("A", "class"), fireLoad: 780e6, seed: 1 }),
};
for (const [name, make] of Object.entries(cases)) {
  const scene = make();
  const sim = F.createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, embers: false, seed: 1 });
  sim.profile = {};
  const t0 = performance.now();
  for (let s = 0; s < STEPS; s += 1) F.step(sim, 0.1);
  const ms = performance.now() - t0;
  const fp = fingerprint([sim.T, sim.Ts, sim.Y, sim.u, sim.v, sim.w, sim.rho, sim.p, sim.fuelMass, sim.moisture, sim.parts.x.subarray(0, sim.parts.n * 3), Float64Array.of(sim.stats.released, sim.stats.radiated, sim.parts.n)]);
  const prof = Object.entries(sim.profile).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k, v]) => `${k} ${(v / STEPS).toFixed(2)}`).join(", ");
  console.log(`${name}: ${scene.grid.cells} cells, fingerprint ${fp}, released ${(sim.stats.released / 1e6).toFixed(4)} MJ, ${(ms / STEPS).toFixed(2)} ms a step | ${prof}`);
}
