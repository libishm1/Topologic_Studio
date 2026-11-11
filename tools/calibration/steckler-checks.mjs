/**
 * Stage 1b of the fire calibration (wiki/research/fire-calibration-plan-2026-11-11.md),
 * a first slice: three of Steckler, Quintiere and Rinkinen's (1982) compartment
 * tests (NBSIR 82-2520, doi:10.6028/NBS.IR.82-2520) against the Flamescapes
 * solver as the app runs it, on the app's cells, CPU. The measurements are
 * NBS's, as NIST keeps them in github.com/firemodels/exp/Steckler_Compartment
 * (a US Government work; reduced here to the numbers compared), and the test
 * conditions are those of NIST's FDS validation inputs
 * (github.com/firemodels/fds, Validation/Steckler_Compartment/FDS_Input_Files):
 *
 *   room 2.8 x 2.8 x 2.13 m (13 mm fibre board), a door 0.74 x 1.83 m (6/6)
 *   in the middle of one wall, a 62.9 kW methane burner on the floor at
 *   position A (the room's centre; test 14), B (a back corner; test 410) or
 *   C (the back wall's middle; test 610); steady state after 30 min.
 *
 * Compared, each as the model's time average from 200 to 300 s against the
 * measured steady state: the upper layer's rise over the inflowing air (the
 * room's corner tree at (2.50, 1.10) by Janssens and Tran, measured: the
 * test's two-layer file, over the door's lowest centreline reading); the
 * interface height; the largest outflow speed on the door's centreline; and
 * the neutral plane's height there. Statistics and tiers as transport-checks.mjs.
 *
 * Not like the tests: the solver's walls are its stone a cell thick (the
 * tests' were fibre board), and 300 s is not 30 min; both are stated with
 * the result.
 *
 *   node tools/calibration/steckler-checks.mjs [--cells 0.20,0.27,0.34] [--json out.json]
 */
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { DEFAULTS, createSolver, flameHeight, step, substepsFor } from "../../topologicpy-web-frontend/src/core/flamescapes.js";
import * as V from "../../topologicpy-web-frontend/src/core/voxelize.js";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const cells = flag("--cells", "0.20,0.27,0.34").split(",").map(Number);
const jsonOut = flag("--json", null);
const SOLVER = Object.fromEntries((flag("--solver", "") || "").split(",").filter(Boolean).map((kv) => kv.split("=")).map(([k, v]) => [k, Number(v)]));
const HORIZON = 300, FROM = 200;
// Temperature and velocity: the FDS Validation Guide's Table 4.1; 0.05 for the heights is assumed here (to be checked against its layer depth).
const SIGMA_E = { temperature: 0.07, velocity: 0.08, depth: 0.05 };
const INERT_FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };

/**
 * The measurements (C: the door's centreline; heights m, temperatures C,
 * speeds m/s out of the room positive), and each test's two-layer reduction
 * (upper layer C, interface m), from firemodels/exp.
 */
const TESTS = [
  {
    test: 14, position: "A", burner: [1.4, 0],
    heights: [1.77, 1.66, 1.54, 1.43, 1.31, 1.2, 1.09, 0.97, 0.86, 0.74, 0.63, 0.52, 0.4, 0.29, 0.17, 0.06],
    TC: [128.58, 124.36, 122.18, 120.48, 117.03, 108.55, 75.98, 43.44, 31.61, 30.88, 31.9, 31.75, 30.57, 30.66, 32.84, 33.4],
    VC: [2.15, 1.62, 1.37, 1.15, 0.91, 0.76, 0.39, -0.2, -0.47, -0.6, -0.64, -0.73, -0.74, -0.77, -0.79, -0.82],
    layer: { upper: 115.3, interface: 0.78 },
  },
  {
    test: 410, position: "B", burner: [0.25, -1.2],
    heights: [1.77, 1.66, 1.54, 1.43, 1.31, 1.2, 1.09, 0.97, 0.86, 0.74, 0.63, 0.52, 0.4, 0.29, 0.17, 0.06],
    TC: [176.06, 165.94, 145.9, 73.39, 58.39, 52.39, 42.33, 29.6, 23.67, 23.26, 23.85, 24.03, 22.93, 23.21, 25.72, 26.19],
    VC: [2.23, 1.56, 1.22, 0.76, 0.52, 0.41, 0.24, -0.14, -0.35, -0.44, -0.54, -0.58, -0.61, -0.61, -0.61, -0.61],
    layer: { upper: 139.3, interface: 1.21 },
  },
  {
    test: 610, position: "C", burner: [0.25, 0],
    heights: [1.77, 1.66, 1.54, 1.43, 1.31, 1.2, 1.09, 0.97, 0.86, 0.74, 0.63, 0.52, 0.4, 0.29, 0.17, 0.06],
    TC: [144.69, 142.92, 134.9, 123.38, 65.64, 46.54, 33.51, 23.14, 19.74, 19.37, 20.39, 20.71, 19.29, 19.74, 20.62, 21.25],
    VC: [2.19, 1.65, 1.31, 1.08, 0.67, 0.3, 0.18, -0.24, -0.37, -0.44, -0.53, -0.59, -0.61, -0.65, -0.67, -0.7],
    layer: { upper: 131.9, interface: 1.08 },
  },
];

/** Where a profile (heights falling) crosses zero, by linear interpolation; null if it does not. */
function crossing(z, v) {
  for (let i = 1; i < z.length; i += 1) {
    if (v[i - 1] > 0 && v[i] <= 0) return z[i - 1] + ((z[i] - z[i - 1]) * v[i - 1]) / (v[i - 1] - v[i]);
  }
  return null;
}

function box(x0, y0, z0, x1, y1, z1) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material: V.MATERIAL.STONE };
}

/** Janssens and Tran's (1992) reduction of one tree (heights over the floor, T in K, ceiling H). */
function janssensTran(heights, T, H) {
  const z = [0, ...heights, H];
  const t = [T[0], ...T, T[T.length - 1]];
  let I1 = 0, I2 = 0;
  for (let i = 1; i < z.length; i += 1) {
    const dz = z[i] - z[i - 1];
    I1 += 0.5 * (t[i] + t[i - 1]) * dz;
    I2 += 0.5 * (1 / t[i] + 1 / t[i - 1]) * dz;
  }
  const Tl = t[0];
  const denom = I1 + I2 * Tl * Tl - 2 * Tl * H;
  let zi = Math.abs(denom) > 1e-12 ? (Tl * (I1 * I2 - H * H)) / denom : H;
  zi = Math.min(H, Math.max(0, zi));
  let upper = 0, span = 0;
  for (let i = 1; i < z.length; i += 1) {
    const a = Math.max(z[i - 1], zi), b = z[i];
    if (b <= a) continue;
    const f = (x) => t[i - 1] + ((t[i] - t[i - 1]) * (x - z[i - 1])) / (z[i] - z[i - 1]);
    upper += 0.5 * (f(a) + f(b)) * (b - a);
    span += b - a;
  }
  return { height: zi, upper: span > 0 ? upper / span : Tl };
}

function statistics(pairs, sigmaE) {
  const logs = pairs.filter(([m, e]) => m > 0 && e > 0).map(([m, e]) => Math.log(m / e));
  if (logs.length < 2) return { n: logs.length, delta: logs.length ? +Math.exp(logs[0]).toFixed(3) : null, sigmaM: null };
  const mean = logs.reduce((a, b) => a + b, 0) / logs.length;
  const varLog = logs.reduce((a, b) => a + (b - mean) ** 2, 0) / (logs.length - 1);
  const sigmaM = Math.max(sigmaE, Math.sqrt(Math.max(0, varLog - sigmaE * sigmaE)));
  return { n: logs.length, delta: +Math.exp(mean + (sigmaM * sigmaM) / 2 - (sigmaE * sigmaE) / 2).toFixed(3), sigmaM: +sigmaM.toFixed(3) };
}
const tier = (s) => (!s || s.delta == null || s.sigmaM == null ? "too few points" : s.delta >= 0.8 && s.delta <= 1.25 && s.sigmaM <= 0.35 ? "within the calibrated band" : s.delta >= 0.67 && s.delta <= 1.5 && s.sigmaM <= 0.5 ? "indicative" : "not calibrated");

function runTest(T, h) {
  // The viewer's frame: the tests' x stays x, their y is z, their z is up.
  const L = 2.8, Hr = 2.13, half = 1.4, doorW = 0.74, doorH = 1.83, t = h, out = 0.8;
  const boxes = [
    box(-t, -t, -half - t, L + t, 0, half + t), // floor
    box(-t, Hr, -half - t, L + t, Hr + t, half + t), // ceiling
    box(-t, 0, -half - t, 0, Hr, half + t), // back wall
    box(-t, 0, -half - t, L + t, Hr, -half),
    box(-t, 0, half, L + t, Hr, half + t),
    // The front wall with the door.
    box(L, 0, -half - t, L + t, Hr, -doorW / 2),
    box(L, 0, doorW / 2, L + t, Hr, half + t),
    box(L, doorH, -doorW / 2, L + t, Hr, doorW / 2),
    // Outside: the floor goes on under the open air before the door.
    box(L + t, -t, -half - t, L + t + out, 0, half + t),
  ];
  const lo = [-t, -t, -half - t], hi = [L + t + out, Hr + t + 0.4, half + t];
  const n = [0, 1, 2].map((d) => Math.ceil((hi[d] - lo[d]) / h - 1e-6));
  const grid = { nx: n[0], ny: n[1], nz: n[2], h, origin: lo, axes: [0, 1, 2], upAxis: "y", cells: n[0] * n[1] * n[2] };
  const fraction = V.volumeFractions(grid, boxes, { rays: V.raysFor(grid) });
  const solid = V.occupancy(grid, fraction);
  const at = (i, j, k) => i + grid.nx * (j + grid.ny * k);
  const uAt = (i, j, k) => i + (grid.nx + 1) * (j + grid.ny * k);
  const cellOf = (x, y, z) => [Math.floor((x - lo[0]) / h), Math.floor((y - lo[1]) / h), Math.floor((z - lo[2]) / h)];
  // A 0.3 m square burner on the floor (the tests' 0.3 m porous burner), a Heskestad column.
  const Q = 62.9e3;
  const layers = Math.max(1, Math.min(Math.floor(Hr / h) - 2, Math.round(flameHeight(Q, Math.sqrt((4 * 0.09) / Math.PI)) / h)));
  const burner = [];
  const [bx, bz] = T.burner;
  for (let k = 0; k < grid.nz; k += 1) for (let i = 0; i < grid.nx; i += 1) {
    const x = lo[0] + (i + 0.5) * h, z = lo[2] + (k + 0.5) * h;
    if (Math.abs(x - bx) < 0.15 + 1e-9 && Math.abs(z - bz) < 0.15 + 1e-9) {
      let j = cellOf(x, 1e-6, z)[1];
      while (solid[at(i, j, k)]) j += 1;
      for (let l = 0; l < layers; l += 1) if (!solid[at(i, j + l, k)]) burner.push(at(i, j + l, k));
    }
  }
  if (!burner.length) {
    const [i, , k] = cellOf(bx, 0, bz);
    let j = cellOf(bx, 1e-6, bz)[1];
    while (solid[at(i, j, k)]) j += 1;
    for (let l = 0; l < layers; l += 1) if (!solid[at(i, j + l, k)]) burner.push(at(i, j + l, k));
  }
  const scene = { grid, fraction, solid, contents: { cells: [], masses: [] }, ignition: null, pilot: null, burner: { cells: burner, hrr: [[0, 0], [10, Q], [1e6, Q]], radiantFraction: 0.2 }, fuels: { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL } };
  const sim = createSolver(scene, { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL, seed: 1, ...SOLVER });
  // The corner tree (open cells only) and the door's centreline column (the doorway's cells, their x faces' mean).
  const [ti, , tk] = cellOf(2.5, 0, 1.1);
  const tree = [];
  for (let j = 0; j < grid.ny; j += 1) {
    const y = lo[1] + (j + 0.5) * h;
    if (y > 0 && y < Hr && !solid[at(ti, j, tk)]) tree.push([y, at(ti, j, tk)]);
  }
  const [di, , dk] = cellOf(L + t / 2, 0, 0);
  const door = [];
  for (let j = 0; j < grid.ny; j += 1) {
    const y = lo[1] + (j + 0.5) * h;
    if (y > 0 && y < doorH && !solid[at(di, j, dk)]) door.push([y, j]);
  }
  let sumTree = new Float64Array(tree.length);
  const sumU = new Float64Array(door.length), sumTd = new Float64Array(door.length);
  let samples = 0, next = 1, speed = 0, substeps = substepsFor(sim.P, h, 0, sim.P.dt), worst = 0, connected = false;
  const t0 = performance.now();
  while (sim.time < HORIZON - 1e-9) {
    step(sim, Math.min(sim.P.dt / substeps, next - sim.time));
    if (sim.time >= next - 1e-9) {
      let s = 0;
      for (const a of [sim.u, sim.v, sim.w]) for (const x of a) s = Math.max(s, Math.abs(x));
      speed = s;
      worst = Math.max(worst, s);
      substeps = substepsFor(sim.P, h, speed, sim.P.dt);
      // The tree reads only cells joined to the fire's gas (the solver's
      // regions): on 0.27 m cells the solver seals the row under the
      // ceiling, which straddles two cells, and left an open pocket in the
      // tree's column that the hot layer never reaches (2026-11-12).
      if (!connected && sim.topo) {
        const home = sim.topo.region[burner[0]];
        for (let q = tree.length - 1; q >= 0; q -= 1) if (sim.topo.region[tree[q][1]] !== home) tree.splice(q, 1);
        sumTree = new Float64Array(tree.length);
        connected = true;
      }
      if (sim.time > FROM + 1e-9) {
        tree.forEach(([, c], q) => (sumTree[q] += sim.T[c]));
        door.forEach(([, j], q) => {
          sumU[q] += 0.5 * (sim.u[uAt(di, j, dk)] + sim.u[uAt(di + 1, j, dk)]);
          sumTd[q] += sim.T[at(di, j, dk)];
        });
        samples += 1;
      }
      next += 1;
    }
  }
  const openTop = tree.length ? tree[tree.length - 1][0] + h / 2 : Hr;
  const layer = janssensTran(tree.map(([y]) => y), Array.from(sumTree, (x) => x / samples), openTop);
  const doorZ = door.map(([y]) => y).reverse();
  const doorU = Array.from(sumU, (x) => x / samples).reverse();
  const doorT = Array.from(sumTd, (x) => x / samples).reverse();
  const inflowT = doorT[doorT.length - 1];
  const model = {
    upperRise: layer.upper - inflowT,
    interface: layer.height,
    maxOutflow: Math.max(...doorU),
    neutral: crossing(doorZ, doorU),
  };
  const measured = {
    upperRise: T.layer.upper - T.TC[T.TC.length - 1],
    interface: T.layer.interface,
    maxOutflow: Math.max(...T.VC),
    neutral: crossing(T.heights, T.VC),
  };
  return { test: T.test, position: T.position, cell: h, grid: [grid.nx, grid.ny, grid.nz], wallSeconds: +((performance.now() - t0) / 1000).toFixed(1), maxSpeed: +worst.toFixed(1), diverged: !(worst <= 50), model: Object.fromEntries(Object.entries(model).map(([k, v]) => [k, v == null ? null : +v.toFixed(3)])), measured: Object.fromEntries(Object.entries(measured).map(([k, v]) => [k, v == null ? null : +v.toFixed(3)])), doorProfile: { z: doorZ.map((z) => +z.toFixed(2)), u: doorU.map((u) => +u.toFixed(3)) } };
}

const rows = [];
for (const h of cells) {
  for (const T of TESTS) {
    const r = runTest(T, h);
    rows.push(r);
    console.log(`test ${r.test} (${r.position}), cell ${h} m (${r.grid.join("x")}, ${r.wallSeconds} s${r.diverged ? ", DIVERGED" : ""}): layer +${r.model.upperRise} K vs +${r.measured.upperRise}; interface ${r.model.interface} vs ${r.measured.interface} m; outflow ${r.model.maxOutflow} vs ${r.measured.maxOutflow} m/s; neutral plane ${r.model.neutral} vs ${r.measured.neutral} m`);
  }
}
const stats = {};
for (const h of cells) {
  const at = rows.filter((r) => r.cell === h && !r.diverged);
  stats[h] = {
    upperRise: statistics(at.map((r) => [r.model.upperRise, r.measured.upperRise]), SIGMA_E.temperature),
    interface: statistics(at.map((r) => [r.model.interface, r.measured.interface]), SIGMA_E.depth),
    maxOutflow: statistics(at.map((r) => [r.model.maxOutflow, r.measured.maxOutflow]), SIGMA_E.velocity),
    neutral: statistics(at.map((r) => [r.model.neutral, r.measured.neutral]), SIGMA_E.depth),
  };
  for (const [k, s] of Object.entries(stats[h])) console.log(`cell ${h} m, ${k}: delta ${s.delta}, sigmaM ${s.sigmaM} (n ${s.n}): ${tier(s)}`);
}
if (jsonOut) {
  await writeFile(path.resolve(jsonOut), JSON.stringify({ generated: new Date().toISOString(), tool: "tools/calibration/steckler-checks.mjs", solver: { ...SOLVER, vorticity: SOLVER.vorticity ?? DEFAULTS.vorticity }, horizon: HORIZON, averagedFrom: FROM, notes: "walls: the solver's stone a cell thick, not the tests' fibre board; 300 s, not the tests' 30 min; one dataset (Steckler), so at most indicative under the plan's tiers", rows, stats: Object.fromEntries(Object.entries(stats).map(([h, s]) => [h, Object.fromEntries(Object.entries(s).map(([k, v]) => [k, { ...v, verdict: tier(v) }]))])) }, null, 1));
  console.log(`wrote ${jsonOut}`);
}
