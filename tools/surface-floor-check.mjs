#!/usr/bin/env node
/**
 * The checks of physics item 11, surfaces emit from just above ambient
 * (wiki/research/fire-calibration-plan-2026-11-11.md, 13.26, "Checks", read
 * with the coordinator's decisions on 13.25 and 13.26; src/core/flamescapes.js
 * surfaceEmitFloor, surfaceFloorOf). Each judged line prints PASS or FAIL with
 * its figure, its reference and its tolerance; lines reported and not judged
 * print REPORT. No measured data enters any of them.
 *
 *   node tools/surface-floor-check.mjs [--only b,c,d,e,f,i,k] [--cells 0.2,0.34]
 *        [--no-gpu] [--json out.json] [--chrome path] [--headed] [--swiftshader]
 *
 * Set up as tools/surface-rays-check.mjs sets up item 10's checks (13.25):
 * every scene is a grid of cells built here by hand, its gas at ambient with
 * no soot (transparent), its surfaces of stone's rho and c with eps 1 (passed
 * as `inert`) unless stated; a box is given by its inner size in cells, with
 * walls two cells thick whose outer layer stays at ambient; the radiation
 * stage alone (flamescapes.js radiationStage) runs the stated steps of 0.05 s,
 * seed 1, sim.steps advanced and every temperature set back after each; each
 * step's deposits are read from sim.work and turned into J by the capacity the
 * stage divided by. "On" is surfaceRayStart 1 with surfaceEmitFloor 1; "off"
 * is surfaceRayStart 1 with surfaceEmitFloor 0 (item 10 alone). theta is
 * flamescapes.js surfaceFloorOf: ambient + 10 K rounded to float32.
 *
 * Tolerances (13.25, 13.26): 4 standard deviations of the ray counts. A share
 * F read from N rays of equal shares: 4 sqrt(F (1 - F) / N). The sum of two
 * binomials whose rays carry s1 and s2 (energies E1, E2): 4 sqrt(F (1 - F)
 * (s1 E1 + s2 E2)). A group of unequal shares: 4 sqrt(s_max E_g), s_max a
 * pillar cell's share and E_g what the group should receive.
 *
 * (b) The floor. A 4 x 4 x 4 box, every surface at ambient, one floor cell
 *     (the y-low wall's inner layer at grid (3, 1, 3)) at ambient + Delta,
 *     Delta in {-5, 0, 5, 9.99, 10.01, 11, 50, 206.84, 206.86} K, one step.
 *     On: the stage emits eps sigma (T^4 - Ta^4) A dt exactly when T > Ta +
 *     10 K, and stats.surfaceFloor holds that value exactly when Ta < T <= Ta
 *     + 10 K, each within 1e-12 relative (else exactly 0); off: it emits only
 *     at 206.86 K. With emitAbove 1e9 nothing emits, on or off; with emitAbove
 *     300 K and the item on, theta is 300 K and the same rule holds about it.
 *     The same for a one-dimensional wall's end face: a partition one cell
 *     thick across the box at inner x = 2 (the 12.7 mm gypsum board of
 *     flamescapes.wall.test.mjs, eps 1), one wall's high end node at each
 *     temperature, its Ts and last at its nodes' mean.
 * (c) Exchange near ambient. The 12 x 10 x 14 box: the x-low face at 323.15
 *     K, the x-high face at 313.15 K, every other surface at Ta + 5 K; 200
 *     steps. The x-high face receives F11(10, 14; 12) E_lo, the x-low face
 *     F11 E_hi; each y wall F14(10, 12; 14) (E_lo + E_hi), each z wall
 *     F14(14, 12; 10) (E_lo + E_hi); the net exchange sigma (T_lo^4 - T_hi^4)
 *     F11 (14.16 W/m^2 of face). Off: nothing is emitted (reported). On the
 *     GPU (joules) every wall's received energy within 1 % of the CPU's.
 * (d) An isothermal enclosure near ambient: tools/surface-rays-check.mjs's (c)
 *     enclosure at 323.15 K, 200 steps. 1. eps 1: each group's net within 4
 *     sqrt(s_max E_g), group iv exactly 0. 2. eps 0.9 everywhere: the same.
 *     3. The block at eps 0.7, all else 0.9: the block's net gain within 4
 *     sqrt(s_max E_g) of (eps_other - eps_block) sigma (T^4 - Ta^4) x its open
 *     area x dt x steps, and the other groups' nets summing to minus the
 *     block's within 1e-9 of the emission.
 * (e) Energy. Every step of (b) to (d): on the CPU the emission equals the J
 *     deposited plus radiatedOut within 1e-9 of the emission; on the GPU
 *     (joules) within 1 %, with no deposit clamped.
 * (f) GPU parity near ambient: (c)'s box and (d)1 on 0.20 m, one stage from
 *     one state on each path: the emission within 1e-5 relative; each wall's
 *     and group's received energy within 1 % of the CPU's; the mean absolute
 *     per-cell difference at most 0.1 of the CPU's own between seeds 1 and 2.
 * (i) Float32 rounding, reported: (c)'s box on 0.27 m, stone, eps 0.9, dt
 *     0.1 s, the x-low face at Ta + 15, 30 and 60 K in turn, every other
 *     surface at Ta + 12 K, the temperatures written and not set back, 200
 *     steps: both sums of stats.radRoundoff against stats.radiated.
 * (k) surfaceWarmRays (the coordinator's decision 9): (c) and (d)1 again with
 *     surfaceWarmRays 1 and 1600 steps (the same number of rays) judged with
 *     the same tolerances; the standard deviation of each wall's (or group's)
 *     received power in a step reported with 8 rays and with 1.
 *
 * (a), (g) and (h) run apart: tools/flamescapes-fingerprint.mjs,
 * tools/flamescapes-gpu-check.mjs, the unit suite and tools/backend-bench.mjs.
 */
import http from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.resolve(HERE, "../topologicpy-web-frontend");
const CORE = path.join(FRONTEND, "src/core");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
};
const imp = (f) => import(pathToFileURL(path.join(CORE, f)).href);
const F = await imp("flamescapes.js");
const V = await imp("voxelize.js");
if (typeof F.surfaceFloorOf !== "function") throw new Error("this solver has no surfaceFloorOf (item 11 is not built)");

const ONLY = new Set(flag("--only", "b,c,d,e,f,i,k").split(","));
const CELLS = flag("--cells", "0.2,0.34").split(",").map(Number);
const GPU_ON = !args.includes("--no-gpu");
const DT = 0.05;
const TA = F.DEFAULTS.ambient;
const M = V.MATERIALS;
const STONE1 = { ...F.INERT[0], eps: 1 };
const GYPSUM = { thickness: 0.0127, k: 0.48, rho: 1440, c: 840 };
const FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };
const ON = { surfaceRayStart: 1, surfaceEmitFloor: 1 };
const OFF = { surfaceRayStart: 1, surfaceEmitFloor: 0 };
const THETA = F.surfaceFloorOf({ ...F.DEFAULTS, ...ON });
const SIGMA = F.SIGMA;

const verdicts = {};
const lines = [];
const report = { tool: "tools/surface-floor-check.mjs", plan: "wiki/research/fire-calibration-plan-2026-11-11.md 13.26", dt: DT, seed: 1, cells: CELLS, ambient: TA, theta: THETA, checks: {} };
function verdict(check, ok, text) {
  verdicts[check] = (verdicts[check] ?? true) && ok;
  const line = `  (${check}) ${ok ? "PASS" : "FAIL"}: ${text}`;
  lines.push(line);
  console.log(line);
}
function reported(check, text) {
  const line = `  (${check}) REPORT: ${text}`;
  lines.push(line);
  console.log(line);
}
const f4 = (x) => (Number.isFinite(x) ? x.toFixed(4) : String(x));
const ex = (x) => (Number.isFinite(x) ? x.toExponential(3) : String(x));
const pct = (x) => `${(100 * x).toFixed(2)} %`;
const relErr = (x, ref) => (ref === 0 ? (x === 0 ? 0 : Infinity) : Math.abs(x - ref) / Math.abs(ref));

// ------------------------------------------------------------ view factors (13.25 (a))

function F11(a, b, c) {
  const X = a / c, Y = b / c;
  const X1 = Math.sqrt(1 + X * X), Y1 = Math.sqrt(1 + Y * Y);
  return (2 / (Math.PI * X * Y)) * (0.5 * Math.log(((1 + X * X) * (1 + Y * Y)) / (1 + X * X + Y * Y)) + X * Y1 * Math.atan(X / Y1) + Y * X1 * Math.atan(Y / X1) - X * Math.atan(X) - Y * Math.atan(Y));
}
function F14(w, hh, l) {
  const W = w / l, H = hh / l, S2 = W * W + H * H, S = Math.sqrt(S2);
  const a = ((1 + W * W) * (1 + H * H)) / (1 + S2);
  const b = (W * W * (1 + S2)) / ((1 + W * W) * S2);
  const c = (H * H * (1 + S2)) / ((1 + H * H) * S2);
  return (1 / (Math.PI * W)) * (W * Math.atan(1 / W) + H * Math.atan(1 / H) - S * Math.atan(1 / S) + 0.25 * Math.log(a * b ** (W * W) * c ** (H * H)));
}
const binomial = (Fr, N) => 4 * Math.sqrt((Fr * (1 - Fr)) / N);

// ------------------------------------------------------------ scenes

/**
 * A solver built from a spec (also run in the page for the GPU, so it holds
 * its own helpers): a grid of cells h m, the listed solid cells of the listed
 * materials, the hot cells' Ts; with wallHigh every one-dimensional wall's
 * high end node at that temperature, with wallOne = [cell, T] only that
 * cell's wall's, its other nodes at ambient and its Ts and last at its nodes'
 * mean.
 */
function buildSim(F, spec) {
  const { nx, ny, nz, h, M } = spec;
  const N = nx * ny * nz;
  const grid = { nx, ny, nz, h, origin: [0, 0, 0], axes: [0, 1, 2], upAxis: "y", cells: N };
  const fraction = new Float32Array(N * M);
  const solid = new Uint8Array(N);
  for (let q = 0; q < spec.solid.length; q += 1) {
    const c = spec.solid[q];
    solid[c] = 1;
    fraction[c * M + spec.material[q]] = 1;
  }
  const scene = { grid, fraction, solid, contents: { cells: [], masses: [] }, ignition: null, pilot: null };
  const sim = F.createSolver(scene, { contentsFuel: spec.fuel, ignitionFuel: spec.fuel, ...spec.options });
  for (const [c, T] of spec.hot) sim.Ts[c] = T;
  const W = sim.wall;
  const setMean = (w) => {
    const n = W.n, o = w * n, s = W.meta[w] >> 2;
    let mean = 0;
    for (let q = 0; q < n; q += 1) mean += W.C[s * n + q] * W.T[o + q];
    mean = (mean * h * h) / (W.heat[s] * h * h);
    sim.Ts[W.cells[w]] = mean;
    W.last[w] = mean;
  };
  if (spec.wallHigh && W) {
    const n = W.n;
    for (let w = 0; w < W.cells.length; w += 1) {
      for (let q = 0; q < n; q += 1) W.T[w * n + q] = q === n - 1 ? spec.wallHigh : sim.P.ambient;
      setMean(w);
    }
  }
  if (spec.wallOne) {
    const w = W ? W.index[spec.wallOne[0]] : -1;
    if (w < 0) throw new Error("wallOne: not a one-dimensional wall");
    W.T[w * W.n + W.n - 1] = spec.wallOne[1];
    setMean(w);
  }
  return sim;
}

/** A box of inner cells n = [nx, ny, nz] in walls two cells thick (tools/surface-rays-check.mjs). */
function box(n) {
  const [ni, nj, nk] = n;
  const nx = ni + 4, ny = nj + 4, nz = nk + 4;
  const at = (i, j, k) => i + nx * (j + ny * k);
  const inner = (i, j, k) => i >= 2 && i < ni + 2 && j >= 2 && j < nj + 2 && k >= 2 && k < nk + 2;
  const wallOf = (i, j, k) => (i < 2 ? 0 : i >= ni + 2 ? 1 : j < 2 ? 2 : j >= nj + 2 ? 3 : k < 2 ? 4 : 5);
  const outer = (i, j, k) => i === 0 || j === 0 || k === 0 || i === nx - 1 || j === ny - 1 || k === nz - 1;
  return { nx, ny, nz, N: nx * ny * nz, at, inner, wallOf, outer, n };
}
const WALLS = ["x-low", "x-high", "y-low", "y-high", "z-low", "z-high"];
const coords = (g, c) => [c % g.nx, Math.floor(c / g.nx) % g.ny, Math.floor(c / (g.nx * g.ny))];

function spec(g, h, { solid, material, hot, options = {}, wallHigh = null, wallOne = null, inert = [STONE1, STONE1, STONE1], dt = DT }) {
  return { nx: g.nx, ny: g.ny, nz: g.nz, h, M, solid, material, hot, fuel: FUEL, wallHigh, wallOne, options: { seed: 1, dt, inert, ...options } };
}

function boxSolid(g, extra = () => false) {
  const solid = [], material = [];
  for (let c = 0; c < g.N; c += 1) {
    const [i, j, k] = coords(g, c);
    if (!g.inner(i, j, k) || extra(i, j, k)) {
      solid.push(c);
      material.push(0);
    }
  }
  return { solid, material };
}

// ------------------------------------------------------------ the stage

const solidCap = (sim, c) => sim.inertC[c] + F.skinMass(sim, c) * (sim.fuels[sim.fuelKind[c]]?.heatCapacity ?? sim.P.fuelHeatCapacity) + sim.moisture[c] * 4186;
const gasCap = (sim, c) => sim.rho[c] * F.mixtureCp(sim.Y, c * F.NS) * sim.V;

/** A lumped cell's emission in a stage, as the stage computes it (item 11: a lumped solid above theta, a porous cell above emitAbove). */
function cellEmission(sim, c, dt = DT) {
  const { P, grid: g } = sim;
  const floor = sim.kind[c] === F.SOLID ? F.surfaceFloorOf(P) : P.emitAbove;
  if (sim.kind[c] === F.GAS || !(sim.Ts[c] > floor)) return 0;
  const [i, j, k] = coords(g, c);
  let open = 0;
  for (const [a, b, d] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
    const ii = i + a, jj = j + b, kk = k + d;
    if (ii < 0 || jj < 0 || kk < 0 || ii >= g.nx || jj >= g.ny || kk >= g.nz || sim.kind[ii + g.nx * (jj + g.ny * kk)] === F.GAS) open += 1;
  }
  if (!open) return 0;
  const A = g.h * g.h;
  const surface = sim.kind[c] === F.POROUS ? Math.max(sim.area[c], A) : A * open;
  let e = sim.eps[c] * SIGMA * (sim.Ts[c] ** 4 - P.ambient ** 4) * surface * dt;
  e = Math.min(e, 0.5 * solidCap(sim, c) * (sim.Ts[c] - P.ambient));
  return e > 0 ? e : 0;
}

/**
 * The radiation stage alone `steps` times from the spec's state (set back
 * after each unless reset is false), sim.steps advanced each time: J received
 * by each group, by the gas and beyond the grid; what was emitted; the worst
 * step's imbalance relative to its emission; item 11's tallies; with perCell
 * each cell's J, with perStep each step's groups.
 */
function stageRun(sp, groupOf, nGroups, steps, { seed = 1, perCell = false, perStep = false, reset = true, dt = DT } = {}) {
  const sim = buildSim(F, { ...sp, options: { ...sp.options, seed } });
  const W = sim.wall;
  const T0 = sim.T.slice(), Ts0 = sim.Ts.slice();
  const WT0 = W ? W.T.slice() : null, WL0 = W ? W.last.slice() : null;
  const caps = new Float64Array(sim.N), gcaps = new Float64Array(sim.N);
  for (let c = 0; c < sim.N; c += 1) {
    caps[c] = Math.max(solidCap(sim, c), 1e-6);
    gcaps[c] = gasCap(sim, c);
  }
  const groups = new Float64Array(nGroups);
  const cells = perCell ? new Float64Array(sim.N) : null;
  const stepGroups = perStep ? [] : null;
  let gas = 0, out = 0, emitted = 0, worst = 0, ends = 0, emittingSteps = 0;
  const st = sim.stats;
  const t0 = { floor: st.surfaceFloor ?? 0, capped: st.surfaceCapped ?? 0, own: st.radRoundoff?.own ?? 0, deposits: st.radRoundoff?.deposits ?? 0, rays: st.surfaceRays ?? 0, crossings: st.surfaceCrossings ?? 0 };
  const now = new Float64Array(nGroups);
  for (let s = 0; s < steps; s += 1) {
    const r0 = st.radiated, o0 = st.radiatedOut ?? 0;
    F.radiationStage(sim, dt);
    const e = st.radiated - r0;
    const o = (st.radiatedOut ?? 0) - o0;
    const w = sim.work;
    now.fill(0);
    let taken = 0;
    for (let c = 0; c < sim.N; c += 1) {
      const dg = w.radGas[c], ds = w.radSolid[c];
      if (dg) {
        const J = dg * gcaps[c];
        gas += J;
        taken += J;
        if (cells) cells[c] += J;
      }
      if (ds) {
        const J = ds * caps[c];
        now[groupOf[c]] += J;
        taken += J;
        if (cells) cells[c] += J;
      }
    }
    if (W) {
      for (let q = 0; q < W.cells.length; q += 1) {
        const d = w.radWallLo[q] + w.radWallHi[q];
        if (!d) continue;
        const c = W.cells[q];
        const J = w.radWallLo[q] * caps[c] + w.radWallHi[q] * caps[c];
        now[groupOf[c]] += J;
        ends += J;
        taken += J;
        if (cells) cells[c] += J;
      }
    }
    for (let q = 0; q < nGroups; q += 1) groups[q] += now[q];
    if (stepGroups) stepGroups.push(Array.from(now));
    emitted += e;
    out += o;
    if (e > 0) {
      emittingSteps += 1;
      worst = Math.max(worst, Math.abs(e - taken - o) / e);
    }
    if (reset) {
      sim.T.set(T0);
      sim.Ts.set(Ts0);
      if (W) {
        W.T.set(WT0);
        W.last.set(WL0);
      }
    }
    sim.steps += 1;
  }
  const has = st.surfaceFloor !== undefined;
  return {
    sim, groups: Array.from(groups), gas, out, emitted, worst, ends, cells, stepGroups, steps, emittingSteps,
    floor: has ? st.surfaceFloor - t0.floor : null, capped: has ? st.surfaceCapped - t0.capped : null,
    own: has ? st.radRoundoff.own - t0.own : null, deposits: has ? st.radRoundoff.deposits - t0.deposits : null,
    rays: (st.surfaceRays ?? 0) - t0.rays, crossings: (st.surfaceCrossings ?? 0) - t0.crossings,
  };
}

/** Each group's emission in one stage (lumped cells), from the spec's state. */
function groupEmission(sp, groupOf, nGroups) {
  const sim = buildSim(F, sp);
  const e = new Float64Array(nGroups);
  let all = 0;
  for (let c = 0; c < sim.N; c += 1) {
    const x = cellEmission(sim, c);
    if (x > 0) {
      e[groupOf[c]] += x;
      all += x;
    }
  }
  return { e: Array.from(e), all, sim };
}

function energyLine(check, label, r) {
  verdict(check, r.worst <= 1e-9, `${label}: worst step's imbalance ${ex(r.worst)} of its emission, against 0 within 1e-9 (${r.emittingSteps} steps that emit)`);
}

// ------------------------------------------------------------ (b)

const DELTAS = [-5, 0, 5, 9.99, 10.01, 11, 50, 206.84, 206.86];

function floorCase(h, kind) {
  const g = box([4, 4, 4]);
  if (kind === "lumped") {
    const { solid, material } = boxSolid(g);
    const cell = g.at(3, 1, 3);
    return { g, cell, sp: (T, options) => spec(g, h, { solid, material, hot: [[cell, T]], options }) };
  }
  // A partition one cell thick across the box at inner x = 2: one-dimensional walls across x, end faces only open.
  const part = (i, j, k) => i === 4 && g.inner(i, j, k);
  const { solid, material } = boxSolid(g, part);
  solid.forEach((c, q) => {
    const [i, j, k] = coords(g, c);
    if (part(i, j, k)) material[q] = 1;
  });
  return { g, cell: g.at(4, 3, 3), solid, material };
}

function boardInert(h) {
  return [STONE1, { name: "gypsum", rho: (GYPSUM.rho * GYPSUM.thickness) / h, c: GYPSUM.c, k: GYPSUM.k, eps: 1, layers: [GYPSUM] }, STONE1];
}

/** The (b) case's spec for one temperature T of the cell (lumped) or the end node (wall), under `options`. */
function floorSpec(h, kind, T, options) {
  const cs = floorCase(h, kind);
  if (kind === "lumped") return { cs, sp: cs.sp(T, options) };
  return { cs, sp: spec(cs.g, h, { solid: cs.solid, material: cs.material, hot: [], wallOne: [cs.cell, T], inert: boardInert(h), options: { wallLayers: 1, ...options } }) };
}

function checkB() {
  console.log(`\n(b) The floor (a 4 x 4 x 4 box, every surface at ambient ${TA} K, one surface at ambient + Delta, one step; theta ${THETA} K)`);
  const out = [];
  for (const h of CELLS) {
    for (const kind of ["lumped", "wall"]) {
      const what = kind === "lumped" ? "lumped floor cell" : "one-dimensional wall's end face";
      // The case must be what it says: one open face for the lumped cell, a wall across x for the board.
      const probe = floorSpec(h, kind, TA, ON);
      const s0 = buildSim(F, probe.sp);
      if (kind === "wall") {
        const w = s0.wall ? s0.wall.index[probe.cs.cell] : -1;
        if (!(w >= 0 && (s0.wall.meta[w] & 3) === 0)) throw new Error(`(b) on ${h} m: the partition's cell is not a one-dimensional wall across x`);
      }
      const runs = [
        ["on", ON, null],
        ["off", OFF, null],
        ["on, emitAbove 1e9", { ...ON, emitAbove: 1e9 }, null],
        ["off, emitAbove 1e9", { ...OFF, emitAbove: 1e9 }, null],
        ["on, emitAbove 300 K", { ...ON, emitAbove: 300 }, null],
      ];
      for (const [mode, options] of runs) {
        const P = { ...F.DEFAULTS, ...options };
        const theta = F.surfaceFloorOf(P);
        const itemOn = P.surfaceEmitFloor > 0 && P.surfaceRayStart > 0;
        if (mode === "on, emitAbove 300 K") verdict("b", theta === 300, `${h} m ${what}, ${mode}: theta ${theta} K against 300 K exactly`);
        for (const delta of DELTAS) {
          const T = Math.fround(TA + delta);
          const { cs, sp } = floorSpec(h, kind, TA + delta, options);
          const groupOf = new Int8Array(cs.g.N).fill(0);
          const r = stageRun(sp, groupOf, 1, 1);
          const eps = r.sim.eps[cs.cell];
          const ref = eps * SIGMA * (T ** 4 - TA ** 4) * (h * h) * DT;
          // Which surface emits: T above the floor this mode sets (emitAbove off, theta on).
          const emits = T > theta;
          const inFloor = itemOn && T > TA && !emits;
          const label = `${h} m ${what}, ${mode}, Delta ${delta} K (T ${T.toFixed(5)} K)`;
          if (emits) verdict("b", relErr(r.emitted, ref) <= 1e-12, `${label}: emitted ${ex(r.emitted)} J against ${ex(ref)} J within 1e-12 relative (${ex(relErr(r.emitted, ref))})`);
          else verdict("b", r.emitted === 0, `${label}: emitted ${r.emitted} J against exactly 0`);
          if (itemOn && P.emitAbove >= 1e4) reported("b", `${label}: surfaceFloor ${ex(r.floor)} J (theta is emitAbove, ${P.emitAbove} K, so every surface above ambient falls below it; the formula gives ${ex(T > TA ? ref : 0)} J)`);
          else if (itemOn && inFloor) verdict("b", relErr(r.floor, ref) <= 1e-12, `${label}: surfaceFloor ${ex(r.floor)} J against ${ex(ref)} J within 1e-12 relative (${ex(relErr(r.floor, ref))})`);
          else if (itemOn) verdict("b", r.floor === 0, `${label}: surfaceFloor ${r.floor} J against exactly 0`);
          if (r.emitted > 0) energyLine("e", `(b) ${label}`, r);
          out.push({ h, kind, mode, delta, T, emitted: r.emitted, floor: r.floor, ref, worst: r.worst });
        }
      }
    }
  }
  report.checks.b = out;
}

// ------------------------------------------------------------ (c) and (k)

const T_LO = 323.15, T_HI = 313.15;

function exchangeCase(h) {
  const n = [12, 10, 14];
  const g = box(n);
  const { solid, material } = boxSolid(g);
  const groupOf = new Int8Array(g.N).fill(-1);
  const hot = [];
  let nLo = 0, nHi = 0;
  for (const c of solid) {
    const [i, j, k] = coords(g, c);
    groupOf[c] = g.wallOf(i, j, k);
    if (g.outer(i, j, k)) continue;
    if (i === 1 && g.inner(2, j, k)) {
      hot.push([c, T_LO]);
      nLo += 1;
    } else if (i === n[0] + 2 && g.inner(n[0] + 1, j, k)) {
      hot.push([c, T_HI]);
      nHi += 1;
    } else hot.push([c, TA + 5]);
  }
  return { g, n, groupOf, nGroups: 6, nLo, nHi, sp: (options) => spec(g, h, { solid, material, hot, options }) };
}

/** (c) judged on one run: `rays` a surface cell a step. */
function judgeC(check, label, cs, r, h, rays) {
  const em = groupEmission(cs.sp(ON), cs.groupOf, 6);
  const steps = r.steps;
  const Elo = em.e[0] * steps, Ehi = em.e[1] * steps;
  const Nlo = cs.nLo * rays * steps, Nhi = cs.nHi * rays * steps;
  const sLo = Elo / Nlo, sHi = Ehi / Nhi;
  const [ni, nj, nk] = cs.n;
  const Fx = F11(nj, nk, ni), Fy = F14(nj, ni, nk), Fz = F14(nk, ni, nj);
  const shares = {};
  const toHi = r.groups[1] / Elo, toLo = r.groups[0] / Ehi;
  verdict(check, Math.abs(toHi - Fx) <= binomial(Fx, Nlo), `${label}: the x-high face receives ${f4(toHi)} of E_lo against F11 ${f4(Fx)} +/- ${f4(binomial(Fx, Nlo))} (${Nlo} rays; ${pct(binomial(Fx, Nlo) / Fx)})`);
  verdict(check, Math.abs(toLo - Fx) <= binomial(Fx, Nhi), `${label}: the x-low face receives ${f4(toLo)} of E_hi against F11 ${f4(Fx)} +/- ${f4(binomial(Fx, Nhi))} (${Nhi} rays)`);
  shares.toHi = toHi;
  shares.toLo = toLo;
  const both = Elo + Ehi;
  for (const [q, Fr] of [[2, Fy], [3, Fy], [4, Fz], [5, Fz]]) {
    const tol = 4 * Math.sqrt(Fr * (1 - Fr) * (sLo * Elo + sHi * Ehi));
    const got = r.groups[q];
    shares[WALLS[q]] = got / both;
    verdict(check, Math.abs(got - Fr * both) <= tol, `${label}: ${WALLS[q]} receives ${f4(got / both)} of E_lo + E_hi against ${f4(Fr)} +/- ${f4(tol / both)} (the sum of two binomials)`);
  }
  const Tl = Math.fround(T_LO), Th = Math.fround(T_HI);
  const area = nj * nk * h * h;
  const per = area * steps * DT;
  const net = (r.groups[1] - r.groups[0]) / per;
  const ref = Fx * SIGMA * (Tl ** 4 - Th ** 4);
  const tol = (4 * Math.sqrt(Fx * (1 - Fx) * (sLo * Elo + sHi * Ehi))) / per;
  verdict(check, Math.abs(net - ref) <= tol, `${label}: net exchange ${net.toFixed(3)} W/m^2 of face against ${ref.toFixed(3)} +/- ${tol.toFixed(3)} W/m^2 (${pct(tol / ref)})`);
  return { Elo, Ehi, stageEmitted: r.emitted, emissionRel: Math.abs(em.all * steps - r.emitted) / r.emitted, shares, net, netRef: ref, netTol: tol, Fx, Fy, Fz };
}

/** The standard deviation over steps of each group's received power (W). */
function stepSD(r, dt = DT) {
  const n = r.stepGroups[0].length;
  return Array.from({ length: n }, (_, q) => {
    const v = r.stepGroups.map((s) => s[q] / dt);
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
  });
}

const cpuC = {};
function checkC() {
  console.log(`\n(c) Exchange near ambient (12 x 10 x 14 box: x-low face ${T_LO} K, x-high face ${T_HI} K, other surfaces at ambient + 5 K; 200 steps)`);
  const out = [];
  for (const h of CELLS) {
    const cs = exchangeCase(h);
    if (cs.nLo !== 140 || cs.nHi !== 140) throw new Error(`(c) on ${h} m: faces of ${cs.nLo} and ${cs.nHi} cells, not 140`);
    const on = stageRun(cs.sp(ON), cs.groupOf, 6, 200, { perStep: true });
    const off = stageRun(cs.sp(OFF), cs.groupOf, 6, 200);
    const label = `on ${h} m`;
    const j = judgeC("c", label, cs, on, h, 8);
    energyLine("e", `(c) ${label}`, on);
    reported("c", `${label}: the stage emitted ${on.emitted.toFixed(4)} J, the two faces' emission ${(j.Elo + j.Ehi).toFixed(4)} J (${ex(j.emissionRel)} relative); gas ${on.gas} J; beyond the grid ${on.out} J; surfaceFloor ${on.floor.toFixed(4)} J (the surfaces at ambient + 5 K); surfaceCapped ${on.capped} J; ${on.rays} surface rays, ${(on.crossings / on.rays).toFixed(2)} crossings a ray`);
    reported("c", `item off (surfaceEmitFloor 0) on ${h} m: emitted ${off.emitted} J over 200 steps (nothing above emitAbove ${F.DEFAULTS.emitAbove} K)`);
    cpuC[h] = on;
    out.push({ h, on: { ...j, groups: on.groups, worst: on.worst, floor: on.floor, capped: on.capped, rays: on.rays, crossings: on.crossings, sd8: stepSD(on) }, off: { emitted: off.emitted } });
  }
  report.checks.c = out;
}

// ------------------------------------------------------------ (d)

const T_ENC = 323.15;
const GROUPS_D = ["i", "ii", "iii", "iv", "v", "vi", "outer"];

function enclosureCase(h, epsOther, epsBlock) {
  const n = [12, 10, 14];
  const g = box(n);
  const block = (i, j, k) => i >= 7 && i <= 8 && j >= 6 && j <= 7 && k >= 8 && k <= 9;
  const pillar = (i, j, k) => i === 4 && k === 5 && j >= 2 && j < 12;
  const { solid, material } = boxSolid(g, (i, j, k) => block(i, j, k) || pillar(i, j, k));
  const isSolid = new Uint8Array(g.N);
  for (const c of solid) isSolid[c] = 1;
  const groupOf = new Int8Array(g.N).fill(-1);
  const hot = [];
  const D = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  solid.forEach((c, q) => {
    const [i, j, k] = coords(g, c);
    if (g.outer(i, j, k)) {
      groupOf[c] = 6;
      return;
    }
    hot.push([c, T_ENC]);
    if (block(i, j, k)) {
      groupOf[c] = 4;
      material[q] = 1;
    } else if (pillar(i, j, k)) groupOf[c] = 5;
    else {
      const open = D.filter(([a, b, d]) => !isSolid[g.at(i + a, j + b, k + d)]);
      if (!open.length) groupOf[c] = 3;
      else {
        const [a, b, d] = open[0];
        const gi = i + a, gj = j + b, gk = k + d;
        let touch = 0;
        for (const [p, qq, r] of D) {
          if (p * a + qq * b + r * d !== 0) continue;
          if (isSolid[g.at(gi + p, gj + qq, gk + r)]) touch += 1;
        }
        groupOf[c] = Math.min(touch, 2);
        if (open.length > 1 || touch > 2) throw new Error(`(d): a wall cell with ${open.length} open faces and ${touch} solid neighbours`);
      }
    }
  });
  const counts = GROUPS_D.map((_, q) => groupOf.reduce((s, x) => s + (x === q ? 1 : 0), 0));
  const other = { ...F.INERT[0], eps: epsOther }, blk = { ...F.INERT[0], eps: epsBlock };
  return { g, groupOf, nGroups: 7, counts, sp: (options) => spec(g, h, { solid, material, hot, options, inert: [other, blk, other] }) };
}

/** (d)1 and (d)2 judged: each group's net within 4 sqrt(s_max E_g), group iv exactly 0. */
function judgeEnclosure(check, label, cs, r, h, rays) {
  const em = groupEmission(cs.sp(ON), cs.groupOf, 7);
  const steps = r.steps;
  const pillarCell = cs.groupOf.findIndex((x) => x === 5);
  const eps = em.sim.eps[pillarCell];
  const T = Math.fround(T_ENC);
  const sMax = (4 * h * h * eps * SIGMA * (T ** 4 - TA ** 4) * DT) / rays;
  const nets = [];
  for (const q of [0, 1, 2, 4, 5]) {
    const E = em.e[q] * steps;
    const net = r.groups[q] - E;
    const tol = 4 * Math.sqrt(sMax * E);
    nets.push({ group: GROUPS_D[q], net: net / E, tol: tol / E });
    verdict(check, Math.abs(net) <= tol, `${label}: group ${GROUPS_D[q]} net ${pct(net / E)} of its emission against 0 +/- ${pct(tol / E)}`);
  }
  verdict(check, r.groups[3] === 0, `${label}: group iv receives ${r.groups[3]} J against exactly 0`);
  return { nets, em: em.e.map((x) => x * steps), sMax };
}

function checkD() {
  console.log(`\n(d) An isothermal enclosure near ambient (the enclosure of tools/surface-rays-check.mjs (c) at ${T_ENC} K, 200 steps)`);
  const out = [];
  for (const h of CELLS) {
    for (const [part, epsOther, epsBlock] of [["1", 1, 1], ["2", 0.9, 0.9], ["3", 0.9, 0.7]]) {
      const cs = enclosureCase(h, epsOther, epsBlock);
      const want = [582, 248, 24, 154, 8, 10];
      if (!want.every((x, q) => cs.counts[q] === x)) throw new Error(`(d) on ${h} m: groups i to vi hold ${cs.counts.slice(0, 6).join(", ")} cells, not 13.25's ${want.join(", ")}`);
      const r = stageRun(cs.sp(ON), cs.groupOf, 7, 200, { perStep: part === "1" });
      const label = `${part} (eps ${epsOther}${epsBlock !== epsOther ? `, block ${epsBlock}` : ""}) on ${h} m`;
      energyLine("e", `(d)${label}`, r);
      if (part !== "3") {
        const j = judgeEnclosure("d", label, cs, r, h, 8);
        if (part === "1") cpuC[`d1 ${h}`] = r;
        out.push({ h, part, nets: j.nets, iv: r.groups[3], worst: r.worst, floor: r.floor, sd8: part === "1" ? stepSD(r) : null });
        continue;
      }
      // 3: the block at 0.7 among surfaces at 0.9.
      const em = groupEmission(cs.sp(ON), cs.groupOf, 7);
      const steps = r.steps;
      const blockCell = cs.groupOf.findIndex((x) => x === 4), pillarCell = cs.groupOf.findIndex((x) => x === 5);
      const eB = em.sim.eps[blockCell], eO = em.sim.eps[pillarCell];
      const T = Math.fround(T_ENC);
      const q4 = SIGMA * (T ** 4 - TA ** 4);
      const area = 24 * h * h;
      const ref = (eO - eB) * q4 * area * DT * steps;
      const Eg = eO * q4 * area * DT * steps;
      const sMax = (4 * h * h * eO * q4 * DT) / 8;
      const tol = 4 * Math.sqrt(sMax * Eg);
      const netB = r.groups[4] - em.e[4] * steps;
      verdict("d", Math.abs(netB - ref) <= tol, `${label}: the block's net gain ${netB.toFixed(4)} J (${f4(netB / (em.e[4] * steps))} of its emission) against (eps_other - eps_block) sigma (T^4 - Ta^4) A dt steps = ${ref.toFixed(4)} J (${f4(ref / (em.e[4] * steps))}) +/- ${tol.toFixed(4)} J`);
      let others = r.gas;
      for (let q = 0; q < 7; q += 1) if (q !== 4) others += r.groups[q] - em.e[q] * steps;
      const total = em.all * steps;
      verdict("d", Math.abs(others + netB) <= 1e-9 * total, `${label}: the other groups' nets sum to ${others.toFixed(6)} J against minus the block's, ${(-netB).toFixed(6)} J, within 1e-9 of the emission (${ex(Math.abs(others + netB) / total)})`);
      const nets = [0, 1, 2, 5].map((q) => `${GROUPS_D[q]} ${pct((r.groups[q] - em.e[q] * steps) / (em.e[q] * steps))}`).join(", ");
      reported("d", `${label}: the other groups' nets ${nets}; group iv ${r.groups[3]} J`);
      out.push({ h, part, blockNet: netB, ref, tol, others, total, worst: r.worst, epsBlock: eB, epsOther: eO });
    }
  }
  report.checks.d = out;
}

// ------------------------------------------------------------ (i)

function checkI() {
  console.log("\n(i) Float32 rounding, reported (0.27 m, stone, eps 0.9, dt 0.1 s; temperatures written and not set back; 200 steps)");
  const out = [];
  const h = 0.27;
  for (const rise of [15, 30, 60]) {
    const g = box([12, 10, 14]);
    const { solid, material } = boxSolid(g);
    const hot = [];
    for (const c of solid) {
      const [i, j, k] = coords(g, c);
      if (g.outer(i, j, k)) continue;
      hot.push([c, i === 1 && g.inner(2, j, k) ? TA + rise : TA + 12]);
    }
    const sp = spec(g, h, { solid, material, hot, options: ON, inert: [F.INERT[0], F.INERT[0], F.INERT[0]], dt: 0.1 });
    const groupOf = new Int8Array(g.N).fill(0);
    const r = stageRun(sp, groupOf, 1, 200, { reset: false, dt: 0.1 });
    reported("i", `x-low face at ambient + ${rise} K: radiated ${r.emitted.toFixed(3)} J; radRoundoff own ${r.own.toFixed(3)} J (${pct(r.own / r.emitted)} of radiated), deposits ${r.deposits.toFixed(3)} J (${pct(r.deposits / r.emitted)}); surfaceFloor ${r.floor.toFixed(3)} J; surfaceCapped ${r.capped} J`);
    out.push({ rise, radiated: r.emitted, own: r.own, deposits: r.deposits, floor: r.floor, capped: r.capped });
  }
  report.checks.i = out;
}

// ------------------------------------------------------------ (k)

function checkK() {
  console.log("\n(k) surfaceWarmRays 1 ((c) and (d)1 again, 1600 steps: the same number of rays)");
  const out = [];
  const WARM = { ...ON, surfaceWarmRays: 1 };
  for (const h of CELLS) {
    const cs = exchangeCase(h);
    const r = stageRun(cs.sp(WARM), cs.groupOf, 6, 1600, { perStep: true });
    if (r.rays !== 280 * 1600) throw new Error(`(k) on ${h} m: ${r.rays} surface rays, not ${280 * 1600}`);
    const j = judgeC("k", `(c) with 1 ray on ${h} m`, cs, r, h, 1);
    energyLine("k", `(c) with 1 ray on ${h} m`, r);
    const sd1 = stepSD(r);
    const sd8 = cpuC[h] ? stepSD(cpuC[h]) : stepSD(stageRun(cs.sp(ON), cs.groupOf, 6, 200, { perStep: true }));
    reported("k", `(c) on ${h} m: each wall's received power in a step, standard deviation with 8 rays / with 1 (W): ${WALLS.map((w, q) => `${w} ${sd8[q].toFixed(4)} / ${sd1[q].toFixed(4)}`).join(", ")}`);
    const ce = enclosureCase(h, 1, 1);
    const re = stageRun(ce.sp(WARM), ce.groupOf, 7, 1600, { perStep: true });
    const je = judgeEnclosure("k", `(d)1 with 1 ray on ${h} m`, ce, re, h, 1);
    energyLine("k", `(d)1 with 1 ray on ${h} m`, re);
    const se1 = stepSD(re);
    const se8 = cpuC[`d1 ${h}`] ? stepSD(cpuC[`d1 ${h}`]) : stepSD(stageRun(ce.sp(ON), ce.groupOf, 7, 200, { perStep: true }));
    reported("k", `(d)1 on ${h} m: each group's received power in a step, standard deviation with 8 rays / with 1 (W): ${[0, 1, 2, 3, 4, 5].map((q) => `${GROUPS_D[q]} ${se8[q].toFixed(4)} / ${se1[q].toFixed(4)}`).join(", ")}`);
    out.push({ h, c: { ...j, sd8, sd1, worst: r.worst }, d1: { nets: je.nets, iv: re.groups[3], sd8: se8, sd1: se1, worst: re.worst } });
  }
  report.checks.k = out;
}

// ------------------------------------------------------------ the GPU: (b), (c), (d) for (e), (c)'s 1 %, and (f)

/** Runs in the page (serialised by Playwright): each case's radiation passes on the GPU. */
async function gpuSuite({ base, cases, builder }) {
  const buildSim = new Function(`return (${builder})`)();
  const F = await import(base + "flamescapes.js");
  const G = await import(base + "flamescapesGpu.js");
  const K = await import(base + "flamescapesGpu/kernels.js");
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return { error: "No WebGPU adapter in this browser." };
  const runner = await G.createGpuRunner(adapter);
  if (!runner) return { error: "createGpuRunner returned null (no device, a limit missing, or the kernels failed to compile)." };
  const info = adapter.info || {};
  const out = { adapter: `${info.vendor || ""} ${info.architecture || ""} ${info.description || ""}`.trim(), cases: [] };
  const solidCap = (sim, c) => sim.inertC[c] + F.skinMass(sim, c) * (sim.fuels[sim.fuelKind[c]]?.heatCapacity ?? sim.P.fuelHeatCapacity) + sim.moisture[c] * 4186;
  for (const cs of cases) {
    const sim = buildSim(F, cs.spec);
    const gpu = runner.createSim(sim, {});
    const W = sim.wall;
    const groups = new Float64Array(cs.nGroups);
    const cells = cs.perCell ? new Float64Array(sim.N) : null;
    let gas = 0, emitted = 0, radOut = 0, worst = 0, clamps = 0, clampJ = 0, joules = null, emitting = 0;
    const t0 = performance.now();
    for (let s = 0; s < cs.steps; s += 1) {
      sim.steps = s;
      const r = await gpu.radiationPass(sim, {});
      joules = r.joules;
      const J = (v, cap) => (r.joules ? v / K.DEP_JOULE : (v / K.DEP_SCALE) * cap);
      let taken = 0;
      for (let c = 0; c < sim.N; c += 1) {
        if (r.gas[c]) {
          const e = J(r.gas[c], sim.rho[c] * F.mixtureCp(sim.Y, c * F.NS) * sim.V);
          gas += e;
          taken += e;
          if (cells) cells[c] += e;
        }
        if (r.solid[c]) {
          const e = J(r.solid[c], Math.max(solidCap(sim, c), 1e-6));
          groups[cs.groupOf[c]] += e;
          taken += e;
          if (cells) cells[c] += e;
        }
      }
      if (W) {
        for (let q = 0; q < W.cells.length; q += 1) {
          if (!(r.wallLo[q] || r.wallHi[q])) continue;
          const c = W.cells[q];
          const e = J(r.wallLo[q], Math.max(solidCap(sim, c), 1e-6)) + J(r.wallHi[q], Math.max(solidCap(sim, c), 1e-6));
          groups[cs.groupOf[c]] += e;
          taken += e;
          if (cells) cells[c] += e;
        }
      }
      emitted += r.radiated;
      radOut += r.radiatedOut ?? 0;
      clamps += r.clamps ?? 0;
      clampJ += r.clampJ ?? 0;
      if (r.radiated > 0) {
        emitting += 1;
        worst = Math.max(worst, Math.abs(r.radiated - taken - (r.radiatedOut ?? 0)) / r.radiated);
      }
    }
    const ms = performance.now() - t0;
    gpu.destroy();
    out.cases.push({ id: cs.id, groups: Array.from(groups), gas, emitted, out: radOut, worst, clamps, clampJ, joules, steps: cs.steps, emitting, ms, cells: cells ? Array.from(cells) : null });
  }
  return out;
}

async function checkGpu() {
  const { chromium } = await import("playwright-core");
  console.log("\n(c), (e), (f) on the GPU (radiation passes alone, deposits in joules with surfaceRayStart 1)");
  const TYPES = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".html": "text/html", ".wasm": "application/wasm" };
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    if (url.pathname === "/" || url.pathname === "/check.html") {
      res.writeHead(200, { "content-type": "text/html" }).end("<!doctype html><meta charset=utf-8><title>Surface floor check</title>");
      return;
    }
    const file = path.resolve(FRONTEND, "." + decodeURIComponent(url.pathname));
    if (!file.startsWith(FRONTEND)) {
      res.writeHead(403).end();
      return;
    }
    try {
      const body = await readFile(file);
      res.writeHead(200, { "content-type": TYPES[path.extname(file)] || "application/octet-stream", "cache-control": "no-store" }).end(body);
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const launchArgs = args.includes("--swiftshader") ? ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader"] : [];
  const browser = await chromium.launch({ executablePath: flag("--chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe"), headless: !args.includes("--headed"), args: launchArgs });
  const page = await browser.newPage();
  page.on("console", (msg) => console.log(`  [page] ${msg.text()}`));
  page.on("pageerror", (error) => console.log(`  [pageerror] ${error.message}`));
  await page.goto(`${origin}/check.html`);
  const run = async (cases) => {
    const r = await page.evaluate(gpuSuite, { base: `${origin}/src/core/`, cases, builder: buildSim.toString() });
    if (r.error) throw new Error(r.error);
    return r;
  };
  const out = { parity: [], runs: [] };
  try {
    // (f): one stage from one state on 0.20 m, (c)'s box and (d)1.
    const h = 0.2;
    const ex02 = exchangeCase(h), en02 = enclosureCase(h, 1, 1);
    const parity = [
      ["(c) box", ex02, ex02.sp(ON), WALLS],
      ["(d)1 enclosure", en02, en02.sp(ON), GROUPS_D],
    ];
    const gp = await run(parity.map(([id, cs, sp]) => ({ id, spec: sp, groupOf: Array.from(cs.groupOf), nGroups: cs.nGroups, steps: 1, perCell: true })));
    out.adapter = gp.adapter;
    console.log(`  adapter: ${gp.adapter}`);
    for (const [id, cs, sp, names] of parity) {
      const cpu = stageRun(sp, cs.groupOf, cs.nGroups, 1, { perCell: true });
      const other = stageRun(sp, cs.groupOf, cs.nGroups, 1, { seed: 2, perCell: true });
      const gpu = gp.cases.find((x) => x.id === id);
      const rel = Math.abs(gpu.emitted - cpu.emitted) / cpu.emitted;
      verdict("f", rel <= 1e-5, `${id}: emission GPU ${gpu.emitted.toFixed(6)} J, CPU ${cpu.emitted.toFixed(6)} J: ${ex(rel)} relative against 0 within 1e-5`);
      let worstGroup = 0;
      for (let q = 0; q < cs.nGroups; q += 1) {
        const d = Math.abs(gpu.groups[q] - cpu.groups[q]);
        if (cpu.groups[q] !== 0) worstGroup = Math.max(worstGroup, d / Math.abs(cpu.groups[q]));
        verdict("f", d <= 0.01 * Math.abs(cpu.groups[q]), `${id}: ${names[q]} receives GPU ${gpu.groups[q].toFixed(6)} J against the CPU's ${cpu.groups[q].toFixed(6)} J within 1 % (${cpu.groups[q] ? ex(d / Math.abs(cpu.groups[q])) : d === 0 ? "both 0" : "CPU 0"})`);
      }
      verdict("f", Math.abs(gpu.out - cpu.out) <= 0.01 * Math.abs(cpu.out) && Math.abs(gpu.gas - cpu.gas) <= 0.01 * Math.abs(cpu.gas), `${id}: radiatedOut GPU ${gpu.out} J against the CPU's ${cpu.out} J, gas GPU ${gpu.gas} J against ${cpu.gas} J, within 1 %`);
      let dg = 0, ds = 0;
      for (let c = 0; c < cpu.cells.length; c += 1) {
        dg += Math.abs(gpu.cells[c] - cpu.cells[c]);
        ds += Math.abs(other.cells[c] - cpu.cells[c]);
      }
      dg /= cpu.cells.length;
      ds /= cpu.cells.length;
      verdict("f", dg <= 0.1 * ds, `${id}: mean |GPU - CPU| a cell ${ex(dg)} J against at most 0.1 of the CPU's between seeds 1 and 2, ${ex(ds)} J (ratio ${ex(dg / ds)})`);
      verdict("e", gpu.worst <= 0.01 && gpu.clamps === 0, `GPU (f) ${id}: the books close within ${ex(gpu.worst)} of the emission against 0 within 1 %; ${gpu.clamps} deposits clamped against 0`);
      reported("f", `${id}: worst group difference ${ex(worstGroup)} relative; deposits in ${gpu.joules ? "joules" : "kelvin"}`);
      out.parity.push({ id, emission: { gpu: gpu.emitted, cpu: cpu.emitted, rel }, groups: { gpu: gpu.groups, cpu: cpu.groups }, out: { gpu: gpu.out, cpu: cpu.out }, meanAbs: { gpu: dg, seeds: ds, ratio: dg / ds }, worstGroup, clamps: gpu.clamps, worst: gpu.worst });
    }
    // (b), (c), (d) on the GPU's passes: (e) in every step; (c)'s walls against the CPU's.
    for (const hg of CELLS) {
      const cases = [];
      for (const kind of ["lumped", "wall"]) {
        for (const delta of DELTAS) {
          const { cs, sp } = floorSpec(hg, kind, TA + delta, ON);
          cases.push({ id: `b ${kind} ${delta} ${hg}`, spec: sp, groupOf: Array.from(new Int8Array(cs.g.N).fill(0)), nGroups: 1, steps: 1, kind, delta });
        }
      }
      const ec = exchangeCase(hg);
      cases.push({ id: `c ${hg}`, spec: ec.sp(ON), groupOf: Array.from(ec.groupOf), nGroups: 6, steps: 200 });
      // (c) with kelvin deposits: only with surfaceRayStart 0, where item 11 is inert.
      cases.push({ id: `c kelvin ${hg}`, spec: ec.sp({ surfaceRayStart: 0, surfaceEmitFloor: 1 }), groupOf: Array.from(ec.groupOf), nGroups: 6, steps: 1 });
      for (const [part, eo, eb] of [["1", 1, 1], ["2", 0.9, 0.9], ["3", 0.9, 0.7]]) {
        const en = enclosureCase(hg, eo, eb);
        cases.push({ id: `d${part} ${hg}`, spec: en.sp(ON), groupOf: Array.from(en.groupOf), nGroups: 7, steps: 200 });
      }
      const gr = await run(cases.map(({ kind, delta, ...x }) => x));
      const get = (id) => gr.cases.find((x) => x.id === id);
      for (const cs of cases) {
        const r = get(cs.id);
        out.runs.push({ id: cs.id, groups: r.groups, emitted: r.emitted, out: r.out, gas: r.gas, worst: r.worst, clamps: r.clamps, clampJ: r.clampJ, ms: r.ms, steps: r.steps, joules: r.joules });
        if (cs.id.startsWith("c kelvin")) {
          reported("c", `GPU with kelvin deposits (surfaceRayStart 0, where item 11 is inert) on ${hg} m, one pass: emitted ${r.emitted} J, walls receive ${r.groups.map((x) => x.toFixed(6)).join(", ")} J`);
          continue;
        }
        if (cs.id.startsWith("b ")) {
          const cpu = stageRun(floorSpec(hg, cs.kind, TA + cs.delta, ON).sp, new Int8Array(cs.groupOf.length).fill(0), 1, 1);
          if (r.emitted > 0 || cpu.emitted > 0) {
            verdict("e", r.worst <= 0.01 && r.clamps === 0, `GPU (b) ${cs.kind} Delta ${cs.delta} K on ${hg} m: the books close within ${ex(r.worst)} of the emission against 0 within 1 %; ${r.clamps} clamped against 0; emitted GPU ${ex(r.emitted)} J, CPU ${ex(cpu.emitted)} J`);
          } else reported("e", `GPU (b) ${cs.kind} Delta ${cs.delta} K on ${hg} m: emitted ${r.emitted} J (the CPU ${cpu.emitted} J)`);
          continue;
        }
        verdict("e", r.worst <= 0.01 && r.clamps === 0, `GPU (${cs.id.split(" ")[0]}) on ${hg} m: the books close within ${ex(r.worst)} of a step's emission in all ${r.emitting} steps, against 0 within 1 %; ${r.clamps} deposits clamped against 0; ${r.ms.toFixed(0)} ms for ${r.steps} passes`);
        if (cs.id.startsWith("c ")) {
          const cpu = cpuC[hg] ?? stageRun(ec.sp(ON), ec.groupOf, 6, 200);
          for (let q = 0; q < 6; q += 1) {
            const d = Math.abs(r.groups[q] - cpu.groups[q]);
            verdict("c", d <= 0.01 * Math.abs(cpu.groups[q]), `GPU on ${hg} m, 200 passes: ${WALLS[q]} receives ${r.groups[q].toFixed(4)} J against the CPU's ${cpu.groups[q].toFixed(4)} J within 1 % (${ex(d / Math.abs(cpu.groups[q]))})`);
          }
        }
        if (cs.id.startsWith("d")) {
          const part = cs.id[1];
          const en = enclosureCase(hg, part === "1" ? 1 : 0.9, part === "3" ? 0.7 : part === "1" ? 1 : 0.9);
          const em = groupEmission(en.sp(ON), en.groupOf, 7);
          const nets = [0, 1, 2, 4, 5].map((q) => `${GROUPS_D[q]} ${pct((r.groups[q] - em.e[q] * 200) / (em.e[q] * 200))}`).join(", ");
          reported("d", `GPU (d)${part} on ${hg} m, 200 passes: nets ${nets}; group iv ${r.groups[3]} J`);
        }
      }
    }
  } finally {
    await browser.close();
    server.close();
  }
  report.checks.gpu = out;
}

// ------------------------------------------------------------ run

const t0 = performance.now();
console.log(`Item 11 checks: ambient ${TA} K, theta ${THETA} K (ambient + ${F.DEFAULTS.surfaceEmitRise} K as float32), emitAbove ${F.DEFAULTS.emitAbove} K, cells ${CELLS.join(", ")} m`);
if (ONLY.has("b")) checkB();
if (ONLY.has("c")) checkC();
if (ONLY.has("d")) checkD();
if (ONLY.has("i")) checkI();
if (ONLY.has("k")) checkK();
if (GPU_ON && (ONLY.has("e") || ONLY.has("f") || ONLY.has("c"))) {
  try {
    await checkGpu();
  } catch (error) {
    verdict("f", false, `the GPU part did not run: ${error.message}`);
  }
}
report.verdicts = verdicts;
report.lines = lines;
report.seconds = (performance.now() - t0) / 1000;
const counts = { pass: lines.filter((l) => l.includes(") PASS:")).length, fail: lines.filter((l) => l.includes(") FAIL:")).length, reported: lines.filter((l) => l.includes(") REPORT:")).length };
report.counts = counts;
console.log(`\nVerdicts (13.26's checks; (a), (g) and (h) run apart; (k) outside what ends the line): ${counts.pass} PASS, ${counts.fail} FAIL, ${counts.reported} reported`);
for (const [k, v] of Object.entries(verdicts)) console.log(`  (${k}) ${v ? "PASS" : "FAIL"}`);
console.log(`${report.seconds.toFixed(1)} s`);
const json = flag("--json", null);
if (json) await writeFile(json, JSON.stringify(report, null, 1));
