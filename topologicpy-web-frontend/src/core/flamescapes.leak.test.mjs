/**
 * Leakage from sealed regions (calibration plan 13.11, item 7): FDS's leak
 * flow C_d A_L sqrt(2 rho |dp|) from each pressure zone to ambient, C_d 1
 * (FDS master 5f8f194). The checks 13.11 names, with no measured data,
 * their details fixed before the first run:
 *
 *   Steady pressure: item 5's sealed 3 x 2.5 x 3 m box of air on 0.25 m
 *   cells (adiabatic walls, no radiation), heated at 100 kW evenly by mass
 *   at constant pressure as the solver adds heat, with A_L 0.01 m^2: at
 *   60 s its pressure rise against the orifice relation's for that heat,
 *   dp = (rho / 2) (V / (C_d A_L))^2, V = (gamma - 1) Q / (gamma p) the volume
 *   a zone whose energy is steady must let out (rho its mean density, gamma
 *   its gas's); within 5 %.
 *   A_L = 0: item 5's checks (flamescapes.zones.test.mjs, run with
 *   leakage []) give item 5's numbers.
 * The leaks are given as the validation harness passes them (a zone leak
 * { area_m2, referencePressure_Pa, pressureExponent, toZone }).
 *   Closure, the same run: mass (the gas, plus what left) against the start,
 *   and energy (the gas's internal energy's change, plus the enthalpy that
 *   left) against the heat added; each within 1 %.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULTS, NS, P_AMBIENT, R_GAS, SOLID, SPECIES, createSolver, leakFlow, mixtureCp, step } from "./flamescapes.js";
import { MATERIALS } from "./voxelize.js";

const DT = DEFAULTS.dt;
const FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };
const MOLAR = [0.03, 0.032, 0.044, 0.018, 0.012, 0.028];

function shellBox([nx, ny, nz], h, options) {
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
  return createSolver({ grid, fraction, solid, contents: { cells: [], masses: [] } }, { contentsFuel: FUEL, ignitionFuel: FUEL, seed: 1, emitAbove: 1e9, ...options });
}

/** A cell's gas constant, J/(kg K): R times its moles per kg, nitrogen the rest. */
function gasConstant(Y, o) {
  let inv = 0, rest = 1;
  for (let q = 0; q < NS; q += 1) {
    inv += Y[o + q] / MOLAR[q];
    rest -= Y[o + q];
  }
  return R_GAS * (inv + Math.max(0, rest) / 0.028);
}

test("13.11 item 7: a heated leaky sealed box reaches the orifice relation's steady pressure within 5 %, and its mass and energy close within 1 %", () => {
  const h = 0.25, n = [14, 12, 14], AL = 0.01, power = 100e3;
  const sim = shellBox(n, h, { wallConvection: 0, leakage: [{ area_m2: AL, referencePressure_Pa: 4, pressureExponent: 0.5, toZone: 0 }] });
  const totals = () => {
    let m = 0, U = 0, cp = 0, R = 0;
    for (let c = 0; c < sim.N; c += 1) {
      if (sim.kind[c] === SOLID) continue;
      const w = sim.rho[c] * sim.V, Rc = gasConstant(sim.Y, c * NS), cpc = mixtureCp(sim.Y, c * NS);
      m += w;
      U += w * (cpc - Rc) * sim.T[c];
      cp += w * cpc;
      R += w * Rc;
    }
    return { m, U, cp: cp / m, R: R / m };
  };
  step(sim, DT);
  const start = totals(), out0 = { m: sim.stats.leakMass, H: sim.stats.leakEnthalpy };
  let Q = 0;
  while (sim.time < 60 + DT - 1e-9) {
    const M = totals().m;
    for (let c = 0; c < sim.N; c += 1) {
      if (sim.kind[c] === SOLID) continue;
      const m = sim.rho[c] * sim.V, q = (power * DT * m) / M;
      sim.T[c] += q / (m * mixtureCp(sim.Y, c * NS));
      Q += q;
    }
    step(sim, DT);
  }
  const end = totals();
  const volume = (n[0] - 2) * (n[1] - 2) * (n[2] - 2) * sim.V;
  const p = sim.zones.pi[0] * P_AMBIENT, dp = p - P_AMBIENT;
  const gamma = end.cp / (end.cp - end.R);
  const V = ((gamma - 1) * power) / (gamma * p);
  const expected = ((end.m / volume) / 2) * (V / (sim.P.leakageDischarge * AL)) ** 2;
  const massKept = (end.m + (sim.stats.leakMass - out0.m)) / start.m;
  const energyKept = (end.U - start.U + (sim.stats.leakEnthalpy - out0.H)) / Q;
  console.log(`leaky heated box, 60 s: pressure rise ${dp.toFixed(1)} Pa against the orifice relation's ${expected.toFixed(1)} Pa (${(100 * (dp / expected - 1)).toFixed(2)} %); mass ${massKept.toFixed(5)}, energy ${energyKept.toFixed(5)} of what they should be; ${((sim.stats.leakMass - out0.m) / start.m * 100).toFixed(1)} % of the gas let out`);
  assert.ok(Math.abs(dp / expected - 1) <= 0.05, `${dp} Pa against ${expected} Pa`);
  assert.ok(Math.abs(massKept - 1) <= 0.01, `mass ${massKept}`);
  assert.ok(Math.abs(energyKept - 1) <= 0.01, `energy ${energyKept}`);
});

test("13.11 item 7: leak areas by construction, by leak, by vent, and none; FDS's reference pressure and exponent", () => {
  // The default: the region's upright faces against solids times 0.21e-3, its flat ones times 0.52e-4.
  const sim = shellBox([6, 5, 6], 0.25, {});
  step(sim, DT);
  const A = 0.25 * 0.25, walls = 4 * 4 * 3, floors = 2 * 4 * 4;
  assert.ok(Math.abs(sim.zones.termCA[0] - A * (walls * 0.21e-3 + floors * 0.52e-4)) < 1e-12, `${sim.zones.termCA[0]}`);
  // NIST/NRC-like zone leaks: areas at 50 Pa, exponents above 1/2; and a vent leak in the zone joins its terms.
  const own = shellBox([6, 5, 6], 0.25, { leakage: [{ area_m2: 0.06, referencePressure_Pa: 50, pressureExponent: 0.7, toZone: 0, source: "test" }, { area_m2: 0.01, ventXb: [0.3, 0.4, 0.3, 0.4, 0.3, 0.4], to: "AMBIENT" }, { area_m2: 0.5, toZone: 2 }] });
  step(own, DT);
  assert.deepEqual(Array.from(own.zones.termCA), [0.06, 0.01]);
  assert.equal(own.stats.leaksIgnored, 1, "a leak to another zone is counted, not modelled");
  // FDS's relation: the area holds at the reference pressure, and the flow goes as dp^n.
  const at50 = leakFlow(0.06, 0.7, 50, 50, 1.2), at100 = leakFlow(0.06, 0.7, 50, 100, 1.2);
  assert.ok(Math.abs(at50 - 0.06 * Math.sqrt(2 * 1.2 * 50)) < 1e-12);
  assert.ok(Math.abs(at100 / at50 - 2 ** 0.7) < 1e-12);
  const none = shellBox([6, 5, 6], 0.25, { leakage: [] });
  step(none, DT);
  assert.equal(none.zones.termCA.length, 0);
  // A cooled zone draws air in: its pressure, below ambient, comes back towards it.
  const cool = shellBox([6, 5, 6], 0.25, { leakage: [{ area_m2: 0.01, toZone: 0 }], wallConvection: 0 });
  step(cool, DT);
  for (let c = 0; c < cool.N; c += 1) if (cool.kind[c] !== SOLID) cool.T[c] -= 30;
  step(cool, DT);
  const low = cool.zones.pi[0];
  for (let s = 0; s < 50; s += 1) step(cool, DT);
  assert.ok(low < 1 && cool.zones.pi[0] > low && cool.zones.pi[0] <= 1 + 1e-6, `${low} then ${cool.zones.pi[0]}`);
  assert.ok(cool.stats.leakMass < 0, "air came in");
  assert.ok(cool.Y[(1 + 6 * (1 + 5)) * NS + SPECIES.O2] > 0.2, "it is air");
});

test("13.11 item 7: a vent leak in an open room lets the hot layer's gas out at the ceiling", () => {
  // An open room (its door one cell), a burner, and a crack at the ceiling: the hot layer's pressure over ambient pushes gas out.
  const nx = 12, ny = 10, nz = 12, h = 0.25, N = nx * ny * nz;
  const grid = { nx, ny, nz, h, origin: [0, 0, 0], axes: [0, 1, 2], upAxis: "y", cells: N };
  const fraction = new Float32Array(N * MATERIALS), solid = new Uint8Array(N);
  const at = (i, j, k) => i + nx * (j + ny * k);
  for (let k = 0; k < nz; k += 1) for (let j = 0; j < ny; j += 1) for (let i = 0; i < nx; i += 1) {
    const wall = i === 1 || i === nx - 2 || k === 1 || k === nz - 2 || j === ny - 2 || j === 0;
    const inside = i >= 1 && i <= nx - 2 && k >= 1 && k <= nz - 2 && j <= ny - 2;
    const door = i === nx - 2 && k === 5 && j >= 1 && j <= 5;
    if (wall && inside && !door) {
      solid[at(i, j, k)] = 1;
      fraction[at(i, j, k) * MATERIALS] = 1;
    }
  }
  const cells = [at(5, 1, 5), at(6, 1, 5), at(5, 1, 6), at(6, 1, 6)];
  // The crack in FDS's frame (x, y, z up): over the cell under the ceiling at (3, 7, 3), grid (x, z, y).
  const xb = [3 * h, 4 * h, 3 * h, 4 * h, 7 * h, 8 * h];
  const sim = createSolver({ grid, fraction, solid, contents: { cells: [], masses: [] }, burner: { cells, hrr: [[0, 50e3], [1e6, 50e3]], radiantFraction: 0, rays: 0 } }, { contentsFuel: FUEL, ignitionFuel: FUEL, emitAbove: 1e9, leakage: [{ area_m2: 0.01, ventXb: xb, to: "AMBIENT", source: "test" }] });
  for (let s = 0; s < 300; s += 1) step(sim, DT);
  assert.equal(sim.zones, null, "the room is open");
  assert.equal(sim.ventLeaks.length, 1);
  assert.equal(sim.ventLeaks[0].cell, at(3, 7, 3));
  assert.ok(sim.stats.leakMass > 0, `the crack let gas out: ${sim.stats.leakMass} kg`);
  for (let c = 0; c < N; c += 1) if (sim.kind[c] !== SOLID) assert.ok(Number.isFinite(sim.T[c]) && sim.T[c] > 250);
});
