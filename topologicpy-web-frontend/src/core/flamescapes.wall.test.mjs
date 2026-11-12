/**
 * One-dimensional walls (calibration plan 13.4, item 1): the checks it
 * names, with no measured data. Fixed before the first run: the solver's
 * defaults (wallNodes, wallStretch) and its step, 0.1 s; a 12.7 mm gypsum
 * board (k 0.48, rho 1440, c 840) heated by 10 kW/m^2 on one face against
 * Carslaw and Jaeger's series, read at the heated face, the middle and the
 * back at 30 to 600 s, within 2 % of the heated face's exact rise; a 0.2 m
 * concrete wall (k 1.0, rho 2100, c 880) whose face steps by 500 K against
 * erfc, read at every node down to where erfc falls below 0.01, at 60 to
 * 1200 s, within 0.02 of the step.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULTS, NS, SOLID, createSolver, gasDensity, mixtureCp, step, walls, wallStack, wallStep } from "./flamescapes.js";
import { MATERIALS } from "./voxelize.js";

const TA = DEFAULTS.ambient;
const DT = DEFAULTS.dt;
const GYPSUM = { thickness: 0.0127, k: 0.48, rho: 1440, c: 840 };
const CONCRETE = { thickness: 0.2, k: 1.0, rho: 2100, c: 880 };

/** erfc to a fractional error below 1.2e-7 (Numerical Recipes, erfcc). */
function erfc(x) {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

/**
 * A slab 0 <= x <= L, insulated at x = 0, a flux q into x = L from t = 0
 * (Carslaw and Jaeger 1959, section 3.3): the rise at x.
 */
function slabRise(x, t, { thickness: L, k, rho, c }, q) {
  const a = k / (rho * c);
  let sum = 0;
  for (let m = 1; m < 400; m += 1) sum += ((-1) ** m / (m * m)) * Math.exp((-a * m * m * Math.PI ** 2 * t) / (L * L)) * Math.cos((m * Math.PI * x) / L);
  return (q * t) / (rho * c * L) + ((q * L) / k) * ((3 * x * x - L * L) / (6 * L * L) - (2 / Math.PI ** 2) * sum);
}

test("13.4 item 1: a gypsum board heated on one face keeps within 2 % of the series solution", () => {
  const s = wallStack([GYPSUM]);
  const T = new Float32Array(s.n).fill(TA);
  const out = [0, 0];
  const q = 1e4;
  let mid = 0;
  for (let i = 0; i < s.n; i += 1) if (Math.abs(s.x[i] - GYPSUM.thickness / 2) < Math.abs(s.x[mid] - GYPSUM.thickness / 2)) mid = i;
  let t = 0, worst = 0;
  for (const until of [30, 60, 120, 300, 600]) {
    // The flux reaches node 0, as a ray reaching the wall's face does (wallEnd); the back is insulated.
    while (t < until - 1e-9) {
      T[0] += (q * DT) / s.C[0];
      wallStep(T, 0, s.n, s.C, 0, s.G, 0, DT, 0, 0, 0, 0, out);
      t += DT;
    }
    const scale = slabRise(GYPSUM.thickness, t, GYPSUM, q);
    for (const i of [0, mid, s.n - 1]) {
      const err = Math.abs(T[i] - TA - slabRise(GYPSUM.thickness - s.x[i], t, GYPSUM, q)) / scale;
      worst = Math.max(worst, err);
      assert.ok(err <= 0.02, `t ${t.toFixed(0)} s, depth ${(s.x[i] * 1000).toFixed(2)} mm: ${(100 * err).toFixed(2)} % of the face's rise`);
    }
  }
  console.log(`gypsum board against the series: worst ${(100 * worst).toFixed(2)} % of the heated face's rise (${s.n} nodes)`);
});

test("13.4 item 1: a concrete face stepped in temperature keeps within 2 % of erfc", () => {
  const s = wallStack([CONCRETE]);
  const T = new Float32Array(s.n).fill(TA);
  const out = [0, 0];
  const step = 500;
  const alpha = CONCRETE.k / (CONCRETE.rho * CONCRETE.c);
  let t = 0, worst = 0, read = 0;
  for (const until of [60, 300, 600, 1200]) {
    // The face held at the step by a coefficient far above any wall's (a Dirichlet face); the back insulated.
    while (t < until - 1e-9) {
      wallStep(T, 0, s.n, s.C, 0, s.G, 0, DT, 1e5, TA + step, 0, 0, out);
      t += DT;
    }
    for (let i = 0; i < s.n; i += 1) {
      const exact = erfc(s.x[i] / (2 * Math.sqrt(alpha * t)));
      const err = Math.abs((T[i] - TA) / step - exact);
      worst = Math.max(worst, err);
      read += 1;
      assert.ok(err <= 0.02, `t ${t.toFixed(0)} s, depth ${(s.x[i] * 1000).toFixed(1)} mm: ${(T[i] - TA) / step} against erfc ${exact}`);
      if (exact < 0.01) break;
    }
  }
  // The back stayed at ambient: the wall was semi-infinite over the test.
  assert.ok(T[s.n - 1] - TA < 0.01 * step, `the back rose ${(T[s.n - 1] - TA).toFixed(2)} K`);
  console.log(`concrete face against erfc: worst ${(100 * worst).toFixed(2)} % of the step over ${read} readings (${s.n} nodes)`);
});

test("13.4 item 1: a wall step gives its gas exactly the heat it loses, at any step", () => {
  const s = wallStack([{ thickness: 0.003, k: 50, rho: 7850, c: 470 }, GYPSUM, { thickness: 0.1, k: 0.04, rho: 30, c: 1000 }], 16);
  const T = new Float64Array(s.n);
  for (let i = 0; i < s.n; i += 1) T[i] = 300 + 10 * i;
  const out = [0, 0];
  for (const dt of [0.01, 0.1, 10]) {
    const before = s.C.reduce((a, c, i) => a + c * T[i], 0);
    wallStep(T, 0, s.n, s.C, 0, s.G, 0, dt, 25, 900, 10, 280, out);
    const after = s.C.reduce((a, c, i) => a + c * T[i], 0);
    assert.ok(Math.abs(after - before + out[0] + out[1]) < 1e-9 * before, `dt ${dt}: ${after - before} against ${-(out[0] + out[1])}`);
    for (const x of T) assert.ok(Number.isFinite(x) && x > 270 && x < 910, `a node at ${x} K`);
  }
  // Every layer has intervals, and the stack's heat is its layers' rho c L.
  assert.ok(Math.abs(s.heat - (0.003 * 7850 * 470 + 0.0127 * 1440 * 840 + 0.1 * 30 * 1000)) < 1e-6 * s.heat);
  assert.ok(Math.abs(s.thickness - 0.1157) < 1e-12);
});

/**
 * A sealed box whose walls are the domain's outermost cells, so all its gas
 * is inside: its walls one cell thick, of a gypsum board (layers) or lumped,
 * no radiation; the lower half of the gas `hot` K above ambient.
 */
function boardBox(options, hot = 300) {
  const nx = 10, ny = 9, nz = 10, h = 0.25;
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
  const board = { name: "gypsum", rho: (GYPSUM.rho * GYPSUM.thickness) / h, c: GYPSUM.c, k: GYPSUM.k, eps: 0.9, layers: [GYPSUM] };
  const fuel = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };
  const scene = { grid, fraction, solid, contents: { cells: [], masses: [] }, ignition: null, pilot: null };
  const sim = createSolver(scene, { contentsFuel: fuel, ignitionFuel: fuel, seed: 1, emitAbove: 1e9, inert: [board, board, board], ...options });
  for (let c = 0; c < N; c += 1) if (sim.kind[c] !== SOLID && Math.floor(c / nx) % ny < ny / 2) sim.T[c] = TA + hot;
  const heat = () => {
    let gas = 0, walls = 0;
    for (let c = 0; c < N; c += 1) {
      if (sim.kind[c] === SOLID) walls += sim.inertC[c] * (sim.Ts[c] - TA);
      else gas += gasDensity(sim.T[c], sim.Y, c * NS) * sim.V * mixtureCp(sim.Y, c * NS) * (sim.T[c] - TA);
    }
    return { gas, walls };
  };
  return { sim, heat };
}

test("13.4 item 1: the walls stage moves heat between the gas and the boards and makes none", () => {
  const { sim } = boardBox({ wallLayers: 1 });
  // The gas's energy as the stage reads it (its density fixed during the stage) and the boards'.
  const energy = () => {
    let e = 0;
    for (let c = 0; c < sim.N; c += 1) e += sim.kind[c] === SOLID ? sim.inertC[c] * sim.Ts[c] : sim.rho[c] * sim.V * mixtureCp(sim.Y, c * NS) * sim.T[c];
    return e;
  };
  let moved = 0, worst = 0;
  for (let s = 0; s < 600; s += 1) {
    const before = energy(), gas = sim.T.slice();
    walls(sim, DT);
    for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] !== SOLID) moved += sim.rho[c] * sim.V * mixtureCp(sim.Y, c * NS) * (gas[c] - sim.T[c]);
    worst = Math.max(worst, Math.abs(energy() - before));
  }
  assert.ok(moved > 0, "the boards took heat");
  // Float32 fields round each cell's temperature; the stage itself is exact (wallStep).
  assert.ok(worst < 1e-5 * moved, `${worst} J made or lost in a stage, ${moved.toFixed(0)} J moved`);
});

test("13.4 item 1: a sealed box of gypsum boards: its faces heat, and inside first", () => {
  const { sim, heat } = boardBox({ wallLayers: 1 });
  assert.ok(sim.wall && sim.wall.cells.length > 0, "the boards are one-dimensional walls");
  // Every face of the box but its edges and corners: (nx - 2)(ny - 2) cells on each of six faces, and so on.
  assert.equal(sim.wall.cells.length, 2 * (8 * 7 + 8 * 8 + 7 * 8));
  const start = heat();
  while (sim.time < 60 - 1e-9) step(sim, DT);
  const end = heat();
  assert.ok(end.walls > 0.02 * start.gas, `the walls took ${(end.walls / start.gas).toFixed(3)} of it`);
  // Heated from inside only, a board's inner face is hotter than its back (insulated against the domain's side).
  const W = sim.wall;
  let hotter = 0;
  for (let w = 0; w < W.cells.length; w += 1) {
    const axis = W.meta[w] & 3, c = W.cells[w];
    const p = [c % 10, Math.floor(c / 10) % 9, Math.floor(c / 90)][axis];
    const inner = p === 0 ? W.T[w * W.n + W.n - 1] : W.T[w * W.n];
    const back = p === 0 ? W.T[w * W.n] : W.T[w * W.n + W.n - 1];
    if (inner > back) hotter += 1;
  }
  assert.equal(hotter, W.cells.length);
});

test("13.4 item 1: off, or with no material's layers, every wall is lumped as before", () => {
  const off = boardBox({ wallLayers: 0 }).sim;
  assert.equal(off.wall, null);
  const fuel = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };
  const grid = { nx: 4, ny: 4, nz: 4, h: 0.25, origin: [0, 0, 0], axes: [0, 1, 2], upAxis: "y", cells: 64 };
  const plain = createSolver({ grid, fraction: new Float32Array(64 * MATERIALS), solid: new Uint8Array(64), contents: { cells: [], masses: [] } }, { contentsFuel: fuel, wallLayers: 1 });
  assert.equal(plain.wall, null);
  // A lumped box and a box of one-dimensional walls, the same boards: the lumped one holds the board's heat per area too.
  const lumped = boardBox({ wallLayers: 0 }).sim, layered = boardBox({ wallLayers: 1 }).sim;
  const c = layered.wall.cells[0];
  assert.ok(Math.abs(lumped.inertC[c] - layered.inertC[c]) < 1e-3 * lumped.inertC[c], `${lumped.inertC[c]} against ${layered.inertC[c]}`);
});

test("13.4 item 1: rays reach a board's face that they cross, not its back", async () => {
  const { fuelsFor } = await import("./fireClasses.js");
  // A board one cell thick at mid-height across the domain, open air above and below, a radiating burner under it.
  const nx = 12, ny = 12, nz = 12, h = 0.25;
  const N = nx * ny * nz;
  const grid = { nx, ny, nz, h, origin: [0, 0, 0], axes: [0, 1, 2], upAxis: "y", cells: N };
  const fraction = new Float32Array(N * MATERIALS);
  const solid = new Uint8Array(N);
  const at = (i, j, k) => i + nx * (j + ny * k);
  for (let k = 1; k < nz - 1; k += 1) for (let i = 1; i < nx - 1; i += 1) {
    solid[at(i, 7, k)] = 1;
    fraction[at(i, 7, k) * MATERIALS] = 1;
  }
  const board = { name: "gypsum", rho: (GYPSUM.rho * GYPSUM.thickness) / h, c: GYPSUM.c, k: GYPSUM.k, eps: 0.9, layers: [GYPSUM] };
  const cells = [at(5, 1, 5), at(6, 1, 5), at(5, 1, 6), at(6, 1, 6)];
  const fuel = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };
  const scene = { grid, fraction, solid, contents: { cells: [], masses: [] }, burner: { cells, hrr: [[0, 100e3], [1e6, 100e3]], radiantFraction: 0.35, fuel: fuelsFor("A").ignitionFuel } };
  // No convection, so only the rays heat the board.
  const sim = createSolver(scene, { contentsFuel: fuel, ignitionFuel: fuel, seed: 1, wallLayers: 1, wallConvection: 0, inert: [board, board, board] });
  for (let s = 0; s < 20; s += 1) step(sim, DT);
  const W = sim.wall;
  let lower = 0, upper = 0;
  for (let w = 0; w < W.cells.length; w += 1) {
    lower += W.T[w * W.n] - TA;
    upper += W.T[w * W.n + W.n - 1] - TA;
  }
  assert.ok(lower > 1, `the faces towards the burner rose ${lower.toFixed(2)} K in all`);
  assert.ok(upper < 0.1 * lower, `the backs rose ${upper.toFixed(3)} K against ${lower.toFixed(2)} K`);
});
