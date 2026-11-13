/**
 * A pressure for each sealed region (calibration plan 13.8, item 5): the two
 * checks it names, with no measured data, fixed before the first run.
 *
 *   Mass: the sealed box of 12.7 mm gypsum boards of flamescapes.wall.test.mjs
 *   (one-dimensional walls, its lower half 300 K above ambient, no radiation),
 *   which gained 31 % of its gas's mass in 60 s at ambient pressure: its gas's
 *   mass (its stored densities by the cell volume) within 1 % of the start
 *   after 60 s.
 *   Pressure: a sealed 3 x 2.5 x 3 m box of air on 0.25 m cells, walls the
 *   domain's outermost cells and adiabatic, no radiation, heated at 100 kW for
 *   20 s, evenly by mass, the way the solver's heat sources add heat (a rise
 *   of T at constant pressure); its pressure rise against the ideal gas at
 *   constant volume, (gamma - 1) Q / V with gamma from air's c_p and R / M,
 *   within 2 %.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULTS, NS, P_AMBIENT, R_GAS, SOLID, SPECIES, createSolver, gasDensity, mixtureCp, step } from "./flamescapes.js";
import { MATERIALS } from "./voxelize.js";

const TA = DEFAULTS.ambient;
const DT = DEFAULTS.dt;
const GYPSUM = { thickness: 0.0127, k: 0.48, rho: 1440, c: 840 };
const FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };

/** A sealed box whose walls are the domain's outermost cells, n cells a side inside it. */
function shellBox([nx, ny, nz], h, inert, options) {
  const N = nx * ny * nz;
  const grid = { nx, ny, nz, h, origin: [0, 0, 0], axes: [0, 1, 2], upAxis: "y", cells: N };
  const fraction = new Float32Array(N * MATERIALS);
  const solid = new Uint8Array(N);
  for (let c = 0; c < N; c += 1) {
    const i = c % nx, j = Math.floor(c / nx) % ny, k = Math.floor(c / (nx * ny));
    if (i === 0 || j === 0 || k === 0 || i === nx - 1 || j === ny - 1 || k === nz - 1) {
      solid[c] = 1;
      fraction[c * MATERIALS] = 1;
    }
  }
  const scene = { grid, fraction, solid, contents: { cells: [], masses: [] } };
  return createSolver(scene, { contentsFuel: FUEL, ignitionFuel: FUEL, seed: 1, emitAbove: 1e9, ...(inert ? { inert } : {}), ...options });
}

const gasMass = (sim) => {
  let m = 0;
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] !== SOLID) m += sim.rho[c] * sim.V;
  return m;
};

test("13.8 item 5: a sealed box cooled by its walls keeps its gas's mass within 1 % over 60 s", () => {
  const h = 0.25, ny = 9;
  const board = { name: "gypsum", rho: (GYPSUM.rho * GYPSUM.thickness) / h, c: GYPSUM.c, k: GYPSUM.k, eps: 0.9, layers: [GYPSUM] };
  const run = (pressureZones) => {
    const sim = shellBox([10, ny, 10], h, [board, board, board], { pressureZones, leakage: [] });
    for (let c = 0; c < sim.N; c += 1) {
      if (sim.kind[c] === SOLID || Math.floor(c / 10) % ny >= ny / 2) continue;
      sim.T[c] = TA + 300;
      sim.rho[c] = gasDensity(sim.T[c], sim.Y, c * NS);
    }
    const start = gasMass(sim);
    let walls0 = 0;
    for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] === SOLID) walls0 += sim.inertC[c] * sim.Ts[c];
    while (sim.time < 60 - 1e-9) step(sim, DT);
    let walls = 0;
    for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] === SOLID) walls += sim.inertC[c] * sim.Ts[c];
    return { kept: gasMass(sim) / start, took: walls - walls0, pi: sim.zones?.pi[0] };
  };
  const zoned = run(1);
  console.log(`sealed box cooled by its walls, 60 s: mass ${zoned.kept.toFixed(5)} of the start with pressure zones, its pressure ${zoned.pi.toFixed(4)} of ambient`);
  assert.ok(Math.abs(zoned.kept - 1) <= 0.01, `mass ${zoned.kept}`);
  assert.ok(zoned.pi < 1, "cooled, its pressure fell");
  assert.ok(zoned.took > 0, "the walls took heat");
  const before = run(0);
  console.log(`  at ambient pressure (pressureZones 0): mass ${before.kept.toFixed(4)} of the start`);
  assert.ok(before.kept > 1.1, `the old solver drew in gas: ${before.kept}`);
});

test("13.8 item 5: a heated sealed box's pressure rises as the ideal gas's at constant volume, within 2 %", () => {
  const h = 0.25, n = [14, 12, 14];
  const sim = shellBox(n, h, null, { pressureZones: 1, wallConvection: 0, leakage: [] });
  const volume = (n[0] - 2) * (n[1] - 2) * (n[2] - 2) * sim.V;
  // Air's gamma: c_p and R / M of the gas the box holds.
  let c0 = 0;
  while (sim.kind[c0] === SOLID) c0 += 1;
  const cp = mixtureCp(sim.Y, c0 * NS);
  let molPerKg = 0, rest = 1;
  for (const [s, M] of [[SPECIES.O2, 0.032]]) {
    molPerKg += sim.Y[c0 * NS + s] / M;
    rest -= sim.Y[c0 * NS + s];
  }
  molPerKg += rest / 0.028;
  const Rm = R_GAS * molPerKg, cv = cp - Rm;
  const power = 100e3;
  let Q = 0;
  step(sim, DT);
  const p0 = sim.zones.pi[0] * P_AMBIENT;
  while (sim.time < 20 + DT - 1e-9) {
    // 100 kW, evenly by mass, at constant pressure (each cell's T rises by its share over its m c_p).
    const M = gasMass(sim);
    for (let c = 0; c < sim.N; c += 1) {
      if (sim.kind[c] === SOLID) continue;
      const m = sim.rho[c] * sim.V;
      const q = (power * DT * m) / M;
      sim.T[c] += q / (m * mixtureCp(sim.Y, c * NS));
      Q += q;
    }
    step(sim, DT);
  }
  const rise = sim.zones.pi[0] * P_AMBIENT - p0;
  const expected = ((Rm / cv) * Q) / volume;
  console.log(`heated sealed box: ${(Q / 1e6).toFixed(3)} MJ, pressure rise ${(rise / 1000).toFixed(3)} kPa against ${(expected / 1000).toFixed(3)} kPa ((gamma - 1) Q / V, gamma ${(cp / cv).toFixed(4)}): ${(100 * (rise / expected - 1)).toFixed(2)} %`);
  assert.ok(Math.abs(rise / expected - 1) <= 0.02, `rise ${rise} Pa against ${expected} Pa`);
  // And the zone's pressure is its gas's by the gas law: p V = sum m R_k T.
  let pv = 0;
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] !== SOLID) pv += (sim.rho[c] * sim.V * P_AMBIENT * sim.T[c]) / (gasDensity(1, sim.Y, c * NS));
  assert.ok(Math.abs(pv / volume / (sim.zones.pi[0] * P_AMBIENT) - 1) < 1e-4);
});

test("13.8 item 5: open regions stay at ambient pressure, and a still zone keeps its pressure", () => {
  // The heat-closure mix box: its gas mixes at constant volume, so its pressure and its heat stay.
  const sim = shellBox([10, 9, 10], 0.25, null, { pressureZones: 1, wallConvection: 0, leakage: [] });
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] !== SOLID && Math.floor(c / 10) % 9 < 4) sim.T[c] = TA + 300;
  const heat = () => {
    let e = 0;
    for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] !== SOLID) e += sim.rho[c] * sim.V * mixtureCp(sim.Y, c * NS) * (sim.T[c] - TA);
    return e;
  };
  step(sim, DT);
  const e0 = heat(), m0 = gasMass(sim);
  while (sim.time < 20 - 1e-9) step(sim, DT);
  assert.ok(Math.abs(sim.zones.pi[0] - 1) < 0.01, `pressure ${sim.zones.pi[0]}`);
  assert.ok(Math.abs(gasMass(sim) / m0 - 1) < 1e-4, `mass ${gasMass(sim) / m0}`);
  assert.ok(Math.abs(heat() / e0 - 1) < 0.02, `heat ${heat() / e0}`);
  // An open domain has no zone.
  const N = 6 * 6 * 6;
  const open = createSolver({ grid: { nx: 6, ny: 6, nz: 6, h: 0.25, origin: [0, 0, 0], axes: [0, 1, 2], upAxis: "y", cells: N }, fraction: new Float32Array(N * MATERIALS), solid: new Uint8Array(N), contents: { cells: [], masses: [] } }, { contentsFuel: FUEL, pressureZones: 1 });
  step(open, DT);
  assert.equal(open.zones, null);
});
