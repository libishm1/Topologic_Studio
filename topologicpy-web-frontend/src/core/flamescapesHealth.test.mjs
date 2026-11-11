/**
 * The fire calibration's proposals A, B and C (wiki/research/fire-calibration-plan-2026-11-11.md):
 * the run-health guard, the cut steps and the fitted confinement strength.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULTS, HEALTH, SOLID, createSolver, healthReadings, healthVerdict, step, substepsFor } from "./flamescapes.js";
import { MATERIAL } from "./voxelize.js";
import { buildScene, runScene } from "./flamescapesRun.js";
import { designFireFor, fuelsFor } from "./fireClasses.js";

function box(x0, y0, z0, x1, y1, z1) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material: MATERIAL.STONE };
}

/** The solver tests' 5 x 4 x 2.7 m room with a door, y up. */
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

const designScene = (opts = {}) => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  return buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.35, maxCell: 0.35, contentsFuel, ignitionFuel, item, designFire: designFireFor("A", "fast"), ...opts });
};
const nodes = Float32Array.from([2.5, 0.75, 2, 1, 0.75, 1]);

test("C and F: the confinement strength is the fitted 3 on the heat-conserving transport, with the cut steps' limits beside it", () => {
  assert.equal(DEFAULTS.vorticity, 3);
  assert.equal(DEFAULTS.conservative, 1);
  assert.equal(DEFAULTS.expansion, 1);
  assert.equal(DEFAULTS.cfl, 1);
  assert.equal(DEFAULTS.vorticityStep, 0.5);
  assert.equal(DEFAULTS.maxSubsteps, 8);
  assert.deepEqual({ ...HEALTH }, { maxT: 3000, maxSpeed: 50, energyFactor: 1.5, energySlack: 1e6, cp: 1005 });
});

test("B: a frame's steps are cut by the fastest gas (a cell a step) and the confinement (vorticity x step at most 0.5), at most 8", () => {
  // The rules at the confinement of 6 (the fitted 3 x 0.1 = 0.3 never cuts on its own).
  const P = { ...DEFAULTS, vorticity: 6 };
  // Still air: the confinement alone, 6 x 0.1 = 0.6 > 0.5: two steps.
  assert.equal(substepsFor(P, 0.2, 0, 0.1), 2);
  // 5 m/s on 0.2 m cells moves 2.5 cells in 0.1 s: three steps.
  assert.equal(substepsFor(P, 0.2, 5, 0.1), 3);
  // Exactly one cell a step is enough.
  assert.equal(substepsFor(P, 0.25, 2.5, 0.1), 2);
  assert.equal(substepsFor({ ...P, vorticity: 0 }, 0.25, 2.5, 0.1), 1);
  // A runaway is capped: the health guard stops it.
  assert.equal(substepsFor(P, 0.2, 1000, 0.1), 8);
  // Off: one step.
  assert.equal(substepsFor({ ...P, cfl: 0, vorticity: 0 }, 0.2, 50, 0.1), 1);
});

test("A: the verdict names what failed: non-finite, temperature, speed or energy, in that order", () => {
  const healthy = { maxT: 900, maxSpeed: 8, gasHeat: 1e6, nonFinite: false, badCell: -1, hotCell: 3, fastCell: 4 };
  assert.deepEqual(healthVerdict(healthy, 2e6), { ok: true });
  assert.equal(healthVerdict({ ...healthy, nonFinite: true, badCell: 9 }, 2e6).reason, "non-finite");
  assert.equal(healthVerdict({ ...healthy, nonFinite: true, badCell: 9 }, 2e6).cell, 9);
  assert.equal(healthVerdict({ ...healthy, maxT: 3001 }, 2e6).reason, "temperature");
  assert.equal(healthVerdict({ ...healthy, maxSpeed: 51 }, 2e6).reason, "speed");
  assert.equal(healthVerdict({ ...healthy, maxSpeed: 51 }, 2e6).cell, 4);
  // The gas cannot hold more heat than was put in: 1.5 times it plus 1 MJ.
  assert.equal(healthVerdict({ ...healthy, gasHeat: 4.1e6 }, 2e6).reason, "energy");
  assert.ok(healthVerdict({ ...healthy, gasHeat: 3.9e6 }, 2e6).ok);
});

test("A: the readings find a non-finite cell, the hottest and the fastest gas cell, and the gas's heat", () => {
  const scene = designScene();
  const sim = createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, seed: 1 });
  const g = sim.grid;
  const open = [];
  for (let c = 0; c < sim.N && open.length < 3; c += 1) if (sim.kind[c] !== SOLID && c > g.nx * g.ny * 2) open.push(c);
  const quiet = healthReadings(sim, g, sim.P, sim.V);
  assert.equal(quiet.nonFinite, false);
  assert.equal(quiet.maxSpeed, 0);
  assert.equal(quiet.gasHeat, 0);
  sim.T[open[0]] = sim.P.ambient + 100;
  const warm = healthReadings(sim, g, sim.P, sim.V);
  assert.equal(warm.hotCell, open[0]);
  assert.ok(Math.abs(warm.gasHeat - sim.rho[open[0]] * sim.V * HEALTH.cp * 100) < 1e-6 * warm.gasHeat);
  sim.T[open[1]] = Number.NaN;
  const bad = healthReadings(sim, g, sim.P, sim.V);
  assert.ok(bad.nonFinite);
  assert.equal(bad.badCell, open[1]);
  assert.equal(healthVerdict(bad, 0).reason, "non-finite");
});

test("A: a run that fails stops at the frame it failed, keeps the frames before, and says why, when and where", () => {
  // The solver's own gas made non-finite from the start: the first frame fails.
  const out = runScene(designScene(), { nodes, horizon: 10, frameSeconds: 5, particles: 0, solverOptions: { ambient: Number.NaN } });
  assert.equal(out.health.ok, false);
  assert.equal(out.health.reason, "non-finite");
  assert.equal(out.health.time, 5);
  assert.equal(out.K, 0);
  assert.equal(out.T.length, 0);
  assert.equal(out.hrr.length, 0);
  assert.ok(out.health.cell >= 0 && Array.isArray(out.health.position) && out.health.position.length === 3);
});

test("A and B: a healthy design fire runs through, its steps cut, and says so", () => {
  const out = runScene(designScene(), { nodes, horizon: 20, frameSeconds: 5, particles: 0 });
  assert.equal(out.health.ok, true);
  assert.equal(out.K, 4);
  // At least the cut its fastest gas asks for (one cell a step; the fitted
  // confinement, 3 x 0.1, cuts nothing on its own).
  assert.ok(out.health.substeps >= substepsFor(DEFAULTS, 0.35, out.health.maxSpeed, DEFAULTS.dt), `${out.health.substeps} steps a step`);
  assert.ok(out.health.maxSpeed > 0 && out.health.maxSpeed < HEALTH.maxSpeed);
  assert.ok(out.stats.released > 0);
});

test("A: a pilot's heat counts as heat put in, so a pilot-held pool is not stopped as a runaway", () => {
  // The suppression systems test's heptane pool, held burning by its pilot.
  const { contentsFuel, ignitionFuel } = fuelsFor("B");
  const scene = buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.35, maxCell: 0.35, contentsFuel, ignitionFuel, item: { mass: 5, area: 0.1 } });
  scene.contents = { cells: [], masses: [] };
  scene.pilot = { ...scene.pilot, duration: 1e9 };
  const sim = createSolver(scene, { contentsFuel, ignitionFuel, seed: 1 });
  for (let k = 0; k < 50; k += 1) step(sim, 0.1);
  assert.ok(sim.stats.pilotHeat > 0, "the pilot's heat is tallied");
  const out = runScene(scene, { nodes, horizon: 15, frameSeconds: 5, particles: 0 });
  assert.equal(out.health.ok, true, JSON.stringify(out.health));
  assert.ok(out.stats.pilotHeat > 0);
});
