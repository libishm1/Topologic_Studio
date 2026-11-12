/**
 * Soot deposition on surfaces (calibration plan 13.4, item 2): thermophoretic
 * and gravitational, as FDS models them (Floyd, Overholt and Ezekoye 2014,
 * FEMTC, eqs. 1 to 10; Talbot, Cheng, Schefer and Willis 1980). The check it
 * names, with no measured data, fixed before the first run: the solver's
 * defaults (FDS's soot: density 1800 kg/m^3, conductivity 0.26 W/(m K),
 * thermophoretic diameter 0.03 um) and its step, 0.1 s; a sealed box with no
 * gravity (so no buoyant flow), no radiation, its gas at 600 K with a soot
 * mass fraction of 1e-3 and its thick stone walls at ambient; in the gas
 * cells beside one wall and no other, the deposition velocity of each step
 * over 10 s, -(h / dt) ln(1 - dm / m), against Talbot's thermophoretic
 * velocity with FDS's wall gradient h (Tg - Tw) / kg, written out here on its
 * own, at the cell's and the wall's mean temperatures over the step; within 5 %.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULTS, NS, SOLID, SPECIES, createSolver, deposition, gasDensity, mixtureCp, step } from "./flamescapes.js";
import { MATERIALS } from "./voxelize.js";

const DT = DEFAULTS.dt;
const FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };

/** Air's viscosity by Sutherland's law (273.15 K, 1.716e-5 Pa s, S = 110.4 K). */
const viscosity = (T) => (1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4)) / (T + 110.4);

/** Talbot et al.'s thermophoretic velocity towards a wall at Tw, the gradient FDS's h (Tg - Tw) / kg (Floyd et al. eqs. 3 to 6). */
function talbot(Tg, Tw, rhoGas, cp, h, d = 0.03e-6) {
  const mu = viscosity(Tg), kg = (mu * cp) / 0.71;
  const Kn = (0.065e-6 / (d / 2)) * (Tg / 298.15);
  const Cn = 1 + 1.25 * Kn + 0.41 * Kn * Math.exp(-0.88 / Kn);
  const a = kg / 0.26;
  const K = (2 * 1.17 * (a + 2.18 * Kn) * Cn) / ((1 + 3 * 1.14 * Kn) * (1 + 2 * a + 2 * 2.18 * Kn));
  return ((K * mu) / rhoGas / Tg) * ((h * (Tg - Tw)) / kg);
}

/** A sealed box whose walls are the domain's outermost cells (thick stone, lumped); its gas at T K with soot ys. */
function sootBox(options, T = 600, ys = 1e-3) {
  const nx = 8, ny = 8, nz = 8, h = 0.25, N = nx * ny * nz;
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
  const sim = createSolver(scene, { contentsFuel: FUEL, ignitionFuel: FUEL, seed: 1, emitAbove: 1e9, sootDeposition: 1, ...options });
  for (let c = 0; c < N; c += 1) {
    if (sim.kind[c] === SOLID) continue;
    sim.T[c] = T;
    sim.Y[c * NS + SPECIES.SOOT] = ys;
    sim.Y[c * NS + SPECIES.O2] = 0.232 * (1 - ys);
    sim.rho[c] = gasDensity(T, sim.Y, c * NS);
  }
  const gasSoot = () => {
    let m = 0;
    for (let c = 0; c < N; c += 1) if (sim.kind[c] !== SOLID) m += sim.rho[c] * sim.Y[c * NS + SPECIES.SOOT] * sim.V;
    return m;
  };
  return { sim, gasSoot, at: (i, j, k) => i + nx * (j + ny * k) };
}

test("13.4 item 2: on a cold wall in hot gas, soot deposits at Talbot's thermophoretic velocity within 5 %", () => {
  const { sim, at } = sootBox({ gravity: 0 });
  // The gas cells beside the wall at i = 0 and no other solid.
  const cells = [];
  for (let k = 2; k <= 5; k += 1) for (let j = 2; j <= 5; j += 1) cells.push(at(1, j, k));
  const h = sim.grid.h;
  let worst = 0, read = 0, speed = 0;
  while (sim.time < 10 - 1e-9) {
    const before = cells.map((c) => ({ m: sim.rho[c] * sim.Y[c * NS + SPECIES.SOOT] * sim.V, T: sim.T[c], Tw: sim.Ts[c - 1], dep: sim.sootDeposit[c] }));
    step(sim, DT);
    cells.forEach((c, q) => {
      const b = before[q];
      const dm = sim.sootDeposit[c] - b.dep;
      const measured = -(h / DT) * Math.log(1 - dm / b.m);
      // At the step's mean temperatures (the stage runs after the step's heat transfer).
      const Tg = (b.T + sim.T[c]) / 2, Tw = (b.Tw + sim.Ts[c - 1]) / 2;
      const exact = talbot(Tg, Tw, gasDensity(Tg, sim.Y, c * NS), mixtureCp(sim.Y, c * NS), sim.P.wallConvection);
      const err = Math.abs(measured / exact - 1);
      worst = Math.max(worst, err);
      speed = exact;
      read += 1;
      assert.ok(err <= 0.05, `t ${sim.time.toFixed(1)} s: ${measured.toExponential(3)} m/s against ${exact.toExponential(3)} m/s`);
    });
  }
  console.log(`thermophoretic deposition against Talbot: worst ${(100 * worst).toFixed(2)} % over ${read} readings (${(speed * 1000).toFixed(3)} mm/s at 10 s)`);
});

test("13.4 item 2: the deposition stage takes from the gas exactly the soot it tallies", () => {
  const { sim, gasSoot } = sootBox({ sootDiameter: 20e-6 });
  // Cold walls under hot gas, and settling: both mechanisms at work. The gas
  // is float32 (rho and Y each round a cell's soot by about 1e-7 of it), so a
  // stage is held to 1e-6 of the gas's soot and the whole run to 1e-4 of what left it.
  const start = gasSoot();
  let worst = 0;
  for (let s = 0; s < 100; s += 1) {
    const before = gasSoot(), tally = sim.stats.sootDeposited;
    deposition(sim, DT);
    const taken = before - gasSoot(), told = sim.stats.sootDeposited - tally;
    assert.ok(told > 0);
    worst = Math.max(worst, Math.abs(taken - told) / before);
  }
  assert.ok(worst < 1e-6, `a stage's gas lost ${worst} of its soot more or less than the tally`);
  const left = start - gasSoot();
  assert.ok(Math.abs(left - sim.stats.sootDeposited) < 1e-4 * left, `${left} kg left the gas, ${sim.stats.sootDeposited} kg tallied`);
  // Per cell, the tally is the whole.
  const sum = sim.sootDeposit.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - sim.stats.sootDeposited) < 1e-5 * sum);
});

test("13.4 item 2: in an isothermal box soot settles onto the floor only, and gas plus deposited stays what there was", () => {
  // At ambient, walls and gas alike: no thermophoresis; 20 um particles settle at about 2 cm/s.
  const { sim, gasSoot, at } = sootBox({ sootDiameter: 20e-6 }, DEFAULTS.ambient);
  const start = gasSoot();
  while (sim.time < 30 - 1e-9) step(sim, DT);
  const kept = (gasSoot() + sim.stats.sootDeposited) / start;
  assert.ok(Math.abs(kept - 1) < 1e-3, `gas plus deposited ${kept} of the start`);
  assert.ok(sim.stats.sootDeposited > 0.05 * start, `${(sim.stats.sootDeposited / start).toFixed(3)} of the soot settled`);
  for (let c = 0; c < sim.N; c += 1) {
    if (!(sim.sootDeposit[c] > 0)) continue;
    assert.equal(Math.floor(c / 8) % 8, 1, `cell ${c} deposited soot, but is not on the floor`);
  }
  // And no thermophoresis against a wall hotter than its gas.
  const hot = sootBox({ gravity: 0 }, DEFAULTS.ambient).sim;
  for (let c = 0; c < hot.N; c += 1) if (hot.kind[c] === SOLID) hot.Ts[c] = 600;
  deposition(hot, DT);
  assert.equal(hot.stats.sootDeposited, 0);
  void at;
});

test("13.4 item 2: off, all soot stays in the gas, and the budget counts what was released", async () => {
  const off = sootBox({ sootDeposition: 0 }).sim;
  assert.equal(off.sootDeposit, null);
  step(off, DT);
  assert.equal(off.stats.sootDeposited, 0);
  // A burner's soot is tallied as released.
  const { fuelsFor } = await import("./fireClasses.js");
  const fuel = fuelsFor("A").ignitionFuel;
  const nx = 6, N = nx * nx * nx;
  const grid = { nx, ny: nx, nz: nx, h: 0.25, origin: [0, 0, 0], axes: [0, 1, 2], upAxis: "y", cells: N };
  const scene = { grid, fraction: new Float32Array(N * MATERIALS), solid: new Uint8Array(N), contents: { cells: [], masses: [] }, burner: { cells: [2 + nx * (1 + nx * 2)], hrr: [[0, 20e3], [1e6, 20e3]], radiantFraction: 0, fuel } };
  const sim = createSolver(scene, { contentsFuel: FUEL, ignitionFuel: FUEL, emitAbove: 1e9 });
  for (let s = 0; s < 10; s += 1) step(sim, DT);
  const expected = (sim.stats.released / fuel.heatOfCombustion) * fuel.yields.soot;
  assert.ok(Math.abs(sim.stats.sootReleased - expected) < 1e-9 + 1e-6 * expected, `${sim.stats.sootReleased} against ${expected}`);
});
