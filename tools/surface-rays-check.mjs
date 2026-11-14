#!/usr/bin/env node
/**
 * The checks of physics item 10, a surface's rays start on its emitting face
 * (wiki/research/fire-calibration-plan-2026-11-11.md, 13.25, "Checks", read
 * with the coordinator's decisions on 13.25 and 13.26; src/core/flamescapes.js
 * surfaceRayStart). No measured data enters any of them.
 *
 *   node tools/surface-rays-check.mjs [--only a,b,c,d,e,g] [--cells 0.2,0.34]
 *        [--gpu-cells 0.2] [--json out.json] [--chrome path] [--headed] [--swiftshader]
 *
 * Every scene is a grid of cells built here by hand (as
 * flamescapes.wall.test.mjs builds its mask): its gas at ambient with no soot,
 * so the gas is transparent, and its surfaces of stone's rho and c with eps 1
 * (passed as `inert`). A box is given by its inner size in cells, nx x ny x
 * nz, with walls two cells thick whose outer layer stays at ambient and does
 * not emit. The radiation stage alone (flamescapes.js radiationStage) runs the
 * stated number of steps of 0.05 s, seed 1; after each, 1 is added to
 * sim.steps and every temperature is set back (gas, Ts, wall nodes, last).
 * Each step the stage's deposits are read from sim.work (radGas, radSolid,
 * radWallLo, radWallHi) before anything else adds to them, and turned into J
 * by the capacity the stage divided by (rho cp V; max(solid capacity, 1e-6)).
 * A wall's share is read from the cells of its two layers, an edge column
 * counted with the first of x-low, x-high, y-low, y-high, z-low, z-high that
 * holds it. Cells of 0.20 and 0.34 m. Every case runs with surfaceRayStart 1
 * (judged) and 0 (reported beside, not judged).
 *
 * Tolerances (13.25): where a case's rays carry equal shares, a share F read
 * from N rays passes within 4 sqrt(F (1 - F) / N), F the reference; where
 * shares differ, the energy a group receives passes within 4 sqrt(s_max E_g),
 * s_max the largest share of any ray in the scene and E_g what the group
 * should receive.
 *
 * (a) A flat wall's emission reaches the room: boxes of 8 x 8 x 8, 12 x 10 x
 *     14, 6 x 16 x 10 and 40 x 8 x 8 cells, the x-low wall's inner layer at
 *     1000 K (its cells facing the gas: ny x nz of them), 200 steps. 1. The
 *     x-low wall, the gas and the grid's sides take exactly 0. 2. The opposite
 *     wall takes F11(ny, nz; nx), each y wall F14(ny, nx; nz), each z wall
 *     F14(nz, nx; ny). 3. In the 12 x 10 x 14 box with the floor at 1000 K
 *     instead: A_x F(x-low -> floor) and A_y F(floor -> x-low) agree within 4
 *     standard deviations of their difference, and each lies within its own
 *     4 of A_x F14(10, 12; 14).
 * (b) Outside corners: a 2 x 2 x 2 block at the centre of a 12 x 12 x 12 box
 *     at 1000 K, 1000 steps. The block takes exactly 0; each wall 1/6 within
 *     4 standard deviations.
 * (c) An isothermal enclosure: the 12 x 10 x 14 box with a 2 x 2 x 2 block at
 *     inner cells 5 to 6, 4 to 5, 6 to 7 and a pillar one cell square from
 *     floor to ceiling at inner cell (2, ., 3); every solid cell not on the
 *     grid's outer layer at 1000 K; 200 steps. Groups: wall face cells whose
 *     gas neighbour touches no other solid cell in the face's plane (i), one
 *     (ii), two (iii); inner-layer cells with no open face (iv); the block
 *     (v); the pillar (vi). Each group's net (received less emitted) within 4
 *     sqrt(s_max E_g), s_max a pillar cell's share; group iv receives exactly 0.
 * (d) Walls one cell thick: the 12 x 10 x 14 box divided at inner x = 6 by a
 *     partition across its whole section, 200 steps. 1. Lumped, at 1000 K: the
 *     partition takes exactly 0, the low side 1/2 within 4 standard
 *     deviations and the high side the rest. 2. One-dimensional (the 12.7 mm
 *     gypsum board of flamescapes.wall.test.mjs, eps 1): its high end node at
 *     1000 K, its other nodes at ambient, its Ts and last at its nodes' mean;
 *     the partition's cells, means and ends take exactly 0, the high side the
 *     whole emission and the low side nothing, within 1e-9 of the emission.
 * (e) The grid's side: a gas region of 10 x 10 x 10 cells closed on five
 *     sides by walls two cells thick, on its x-low side a layer one cell thick
 *     on the grid's side whose 10 x 10 cells facing the gas are at 1000 K;
 *     200 steps. The layer takes exactly 0; radiatedOut is 1/2 of the
 *     emission within 4 standard deviations and the walls take the other 1/2.
 * (f) Energy: in every step of (a) to (e) with the item on, the emission
 *     (stats.radiated) equals the J deposited in gas, solids and walls' ends
 *     plus radiatedOut within 1e-9 of the emission.
 * (g) The CPU and the GPU cast the same rays, on 0.20 m cells, in the
 *     installed Chrome through playwright-core: (a)'s 8 x 8 x 8 and 12 x 10 x
 *     14 boxes, (c)'s enclosure and (d)1, one radiation stage from one state
 *     on each path (on the GPU radEmit alone, flamescapesGpu.js
 *     radiationPass, its deposits read before radApply: joules with the item
 *     on). 1. The emission within 1e-5 relative. 2. Each wall's and group's
 *     received energy, and radiatedOut, within 1 % of the CPU's. 3. The mean
 *     absolute difference of the cells' deposits, GPU against CPU, at most
 *     0.1 of the CPU's own between seeds 1 and 2. Then (a), (b) and (c) on
 *     the GPU's passes, judged as on the CPU, with the GPU's books closing
 *     within 1 % in every step. (b) runs its 1000 steps there too; its first
 *     200 are reported beside, with their own 4 standard deviations.
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

const ONLY = new Set(flag("--only", "a,b,c,d,e,g").split(","));
const CELLS = flag("--cells", "0.2,0.34").split(",").map(Number);
const GPU_CELLS = flag("--gpu-cells", "0.2").split(",").map(Number);
const DT = 0.05;
const TA = F.DEFAULTS.ambient;
const HOT = 1000;
const M = V.MATERIALS;
const STONE = { ...F.INERT[0], eps: 1 };
const GYPSUM = { thickness: 0.0127, k: 0.48, rho: 1440, c: 840 };
const FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };

const verdicts = {};
const lines = [];
const report = { tool: "tools/surface-rays-check.mjs", plan: "wiki/research/fire-calibration-plan-2026-11-11.md 13.25", dt: DT, seed: 1, cells: CELLS, checks: {} };
function verdict(check, ok, line) {
  verdicts[check] = (verdicts[check] ?? true) && ok;
  const text = `  (${check}) ${ok ? "pass" : "FAIL"}: ${line}`;
  lines.push(text);
  console.log(text);
}
const note = (line) => console.log(`        ${line}`);
const f4 = (x) => (Number.isFinite(x) ? x.toFixed(4) : String(x));
const ex = (x) => (Number.isFinite(x) ? x.toExponential(2) : String(x));
const pct = (x) => `${(100 * x).toFixed(2)} %`;

// ------------------------------------------------------------ view factors

/** Between directly opposed parallel rectangles a x b at distance c (13.25 (a)). */
function F11(a, b, c) {
  const X = a / c, Y = b / c;
  const X1 = Math.sqrt(1 + X * X), Y1 = Math.sqrt(1 + Y * Y);
  return (2 / (Math.PI * X * Y)) * (0.5 * Math.log(((1 + X * X) * (1 + Y * Y)) / (1 + X * X + Y * Y)) + X * Y1 * Math.atan(X / Y1) + Y * X1 * Math.atan(Y / X1) - X * Math.atan(X) - Y * Math.atan(Y));
}

/** From a w x l rectangle to an h x l rectangle at 90 degrees sharing its edge of length l (13.25 (a)). */
function F14(w, hh, l) {
  const W = w / l, H = hh / l, S2 = W * W + H * H, S = Math.sqrt(S2);
  const a = ((1 + W * W) * (1 + H * H)) / (1 + S2);
  const b = (W * W * (1 + S2)) / ((1 + W * W) * S2);
  const c = (H * H * (1 + S2)) / ((1 + H * H) * S2);
  return (1 / (Math.PI * W)) * (W * Math.atan(1 / W) + H * Math.atan(1 / H) - S * Math.atan(1 / S) + 0.25 * Math.log(a * b ** (W * W) * c ** (H * H)));
}
const binomial = (F, N) => 4 * Math.sqrt((F * (1 - F)) / N);

// ------------------------------------------------------------ scenes

/**
 * A solver built from a spec (also run in the page for the GPU, so it holds
 * its own helpers): a grid nx x ny x nz of cells h m, the listed solid cells
 * of the listed materials, the hot cells' Ts, and with wallHigh each
 * one-dimensional wall's high end node at that temperature, its other nodes
 * at ambient and its Ts and last at its nodes' mean.
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
  if (spec.wallHigh && W) {
    const n = W.n, A = h * h;
    for (let w = 0; w < W.cells.length; w += 1) {
      const o = w * n, s = W.meta[w] >> 2;
      for (let q = 0; q < n; q += 1) W.T[o + q] = q === n - 1 ? spec.wallHigh : sim.P.ambient;
      let mean = 0;
      for (let q = 0; q < n; q += 1) mean += W.C[s * n + q] * W.T[o + q];
      mean = (mean * A) / (W.heat[s] * A);
      sim.Ts[W.cells[w]] = mean;
      W.last[w] = mean;
    }
  }
  return sim;
}

/** A box of inner cells n = [nx, ny, nz] in walls two cells thick: its grid, the walls' cells and their groups. */
function box(n) {
  const [ni, nj, nk] = n;
  const nx = ni + 4, ny = nj + 4, nz = nk + 4;
  const at = (i, j, k) => i + nx * (j + ny * k);
  const inner = (i, j, k) => i >= 2 && i < ni + 2 && j >= 2 && j < nj + 2 && k >= 2 && k < nk + 2;
  // The wall holding a solid cell: x-low, x-high, y-low, y-high, z-low, z-high, the first that holds it.
  const wallOf = (i, j, k) => (i < 2 ? 0 : i >= ni + 2 ? 1 : j < 2 ? 2 : j >= nj + 2 ? 3 : k < 2 ? 4 : 5);
  const outer = (i, j, k) => i === 0 || j === 0 || k === 0 || i === nx - 1 || j === ny - 1 || k === nz - 1;
  return { nx, ny, nz, N: nx * ny * nz, at, inner, wallOf, outer, n };
}
const WALLS = ["x-low", "x-high", "y-low", "y-high", "z-low", "z-high"];
const coords = (g, c) => [c % g.nx, Math.floor(c / g.nx) % g.ny, Math.floor(c / (g.nx * g.ny))];

function spec(g, h, { solid, material, hot, options = {}, wallHigh = null, inert = [STONE, STONE, STONE] }) {
  return { nx: g.nx, ny: g.ny, nz: g.nz, h, M, solid, material, hot, fuel: FUEL, wallHigh, options: { seed: 1, dt: DT, inert, ...options } };
}

/** The box's walls as solid cells of stone (material 0), and anything else `extra` makes solid. */
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

/** A lumped cell's emission in a stage, as the stage computes it (flamescapes.js radiation). */
function cellEmission(sim, c) {
  const { P, grid: g } = sim;
  if (sim.kind[c] === F.GAS || !(sim.Ts[c] > P.emitAbove)) return 0;
  const [i, j, k] = coords(g, c);
  let open = 0;
  for (const [a, b, d] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
    const ii = i + a, jj = j + b, kk = k + d;
    if (ii < 0 || jj < 0 || kk < 0 || ii >= g.nx || jj >= g.ny || kk >= g.nz || sim.kind[ii + g.nx * (jj + g.ny * kk)] === F.GAS) open += 1;
  }
  if (!open) return 0;
  const A = g.h * g.h;
  const surface = sim.kind[c] === F.POROUS ? Math.max(sim.area[c], A) : A * open;
  let e = sim.eps[c] * F.SIGMA * (sim.Ts[c] ** 4 - P.ambient ** 4) * surface * DT;
  e = Math.min(e, 0.5 * solidCap(sim, c) * (sim.Ts[c] - P.ambient));
  return e > 0 ? e : 0;
}

/**
 * The radiation stage alone `steps` times from the spec's state (set back
 * after each), sim.steps advanced each time: J received by each group (the
 * groups' cells' means and walls' ends), by the gas and beyond the grid; what
 * was emitted; the worst step's imbalance relative to its emission; the
 * surface rays' tallies; and with perCell, each cell's J over the steps.
 */
function stageRun(sp, groupOf, nGroups, steps, { seed = 1, perCell = false } = {}) {
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
  let gas = 0, out = 0, emitted = 0, worst = 0, ends = 0;
  const rays0 = sim.stats.surfaceRays ?? 0, cross0 = sim.stats.surfaceCrossings ?? 0;
  for (let s = 0; s < steps; s += 1) {
    const r0 = sim.stats.radiated, o0 = sim.stats.radiatedOut ?? 0;
    F.radiationStage(sim, DT);
    const e = sim.stats.radiated - r0;
    const o = (sim.stats.radiatedOut ?? 0) - o0;
    const w = sim.work;
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
        groups[groupOf[c]] += J;
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
        groups[groupOf[c]] += J;
        ends += J;
        taken += J;
        if (cells) cells[c] += J;
      }
    }
    emitted += e;
    out += o;
    if (e > 0) worst = Math.max(worst, Math.abs(e - taken - o) / e);
    sim.T.set(T0);
    sim.Ts.set(Ts0);
    if (W) {
      W.T.set(WT0);
      W.last.set(WL0);
    }
    sim.steps += 1;
  }
  const rays = (sim.stats.surfaceRays ?? 0) - rays0, crossings = (sim.stats.surfaceCrossings ?? 0) - cross0;
  return { sim, groups: Array.from(groups), gas, out, emitted, worst, ends, cells, rays, crossings, steps };
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
  return { e: Array.from(e), all };
}

const crossLine = (r) => (r.rays ? `${r.rays} surface rays, ${(r.crossings / r.rays).toFixed(2)} crossings a ray` : "no surface tallies (item off)");

// ------------------------------------------------------------ (a)

function aCase(n, h, hotWall = "x-low") {
  const g = box(n);
  const { solid, material } = boxSolid(g);
  const groupOf = new Int8Array(g.N).fill(-1);
  const hot = [];
  for (const c of solid) {
    const [i, j, k] = coords(g, c);
    groupOf[c] = g.wallOf(i, j, k);
    // The hot wall's inner-layer cells facing the gas.
    const face = hotWall === "x-low" ? i === 1 && g.inner(2, j, k) : j === 1 && g.inner(i, 2, k);
    if (face) hot.push([c, HOT]);
  }
  return { g, groupOf, nGroups: 6, sp: (on) => spec(g, h, { solid, material, hot, options: { surfaceRayStart: on } }), hotCells: hot.length };
}

function aRefs(n) {
  const [nx, ny, nz] = n;
  return { opposite: F11(ny, nz, nx), y: F14(ny, nx, nz), z: F14(nz, nx, ny) };
}

/** (a)1 and (a)2 judged on one run's groups. */
function judgeA(label, n, r, N, check = "a") {
  const ref = aRefs(n);
  const E = r.emitted;
  const share = r.groups.map((x) => x / E);
  verdict(check, r.groups[0] === 0 && r.gas === 0 && r.out === 0, `${label}: x-low wall ${r.groups[0]} J, gas ${r.gas} J, grid's sides ${r.out} J (exactly 0 each)`);
  const rows = [["x-high (opposite)", share[1], ref.opposite], ["y-low", share[2], ref.y], ["y-high", share[3], ref.y], ["z-low", share[4], ref.z], ["z-high", share[5], ref.z]];
  for (const [name, F, R] of rows) {
    const tol = binomial(R, N);
    verdict(check, Math.abs(F - R) <= tol, `${label}: ${name} ${f4(F)} against ${f4(R)} +/- ${f4(tol)} (${N} rays)`);
  }
  return share;
}

function checkA() {
  console.log("\n(a) A flat wall's emission reaches the room (x-low wall's inner layer at 1000 K, 200 steps)");
  const boxes = [[8, 8, 8], [12, 10, 14], [6, 16, 10], [40, 8, 8]];
  console.log("  references, F11 and F14 from 13.25's formulas:");
  for (const n of boxes) {
    const ref = aRefs(n);
    const N = n[1] * n[2] * 8 * 200;
    const sum = ref.opposite + 2 * ref.y + 2 * ref.z;
    console.log(`    ${n.join(" x ")}: ${N} rays; opposite ${f4(ref.opposite)} +/- ${f4(binomial(ref.opposite, N))}, each y wall ${f4(ref.y)} +/- ${f4(binomial(ref.y, N))}, each z wall ${f4(ref.z)} +/- ${f4(binomial(ref.z, N))}; five shares sum to 1 - ${ex(1 - sum)}`);
  }
  const out = [];
  for (const h of CELLS) {
    for (const n of boxes) {
      const cs = aCase(n, h);
      const N = cs.hotCells * 8 * 200;
      const on = stageRun(cs.sp(1), cs.groupOf, 6, 200);
      const off = stageRun(cs.sp(0), cs.groupOf, 6, 200);
      const label = `${n.join("x")} on ${h} m`;
      const share = judgeA(label, n, on, N);
      verdict("f", on.worst <= 1e-9, `${label}: worst step's imbalance ${ex(on.worst)} of its emission`);
      const offShare = off.groups.map((x) => x / off.emitted);
      note(`item off: x-low (own wall) ${f4(offShare[0])}, x-high ${f4(offShare[1])}, y ${f4(offShare[2])} / ${f4(offShare[3])}, z ${f4(offShare[4])} / ${f4(offShare[5])}; gas ${f4(off.gas / off.emitted)}; on: ${crossLine(on)}`);
      out.push({ box: n, h, rays: N, on: { share, gas: on.gas, out: on.out, emitted: on.emitted, worst: on.worst, rays: on.rays, crossings: on.crossings }, off: { share: offShare, gas: off.gas / off.emitted, worst: off.worst } });
    }
    // (a)3 reciprocity in the 12 x 10 x 14 box.
    const n = [12, 10, 14];
    const x = aCase(n, h, "x-low"), y = aCase(n, h, "y-low");
    const rx = stageRun(x.sp(1), x.groupOf, 6, 200), ry = stageRun(y.sp(1), y.groupOf, 6, 200);
    const Ax = n[1] * n[2], Ay = n[0] * n[2];
    const Nx = x.hotCells * 1600, Ny = y.hotCells * 1600;
    const ref = Ax * F14(n[1], n[0], n[2]), refY = Ay * F14(n[0], n[1], n[2]);
    const Fx = ref / Ax, Fy = ref / Ay;
    const sx = Ax * Math.sqrt((Fx * (1 - Fx)) / Nx), sy = Ay * Math.sqrt((Fy * (1 - Fy)) / Ny);
    const ax = (Ax * rx.groups[2]) / rx.emitted, ay = (Ay * ry.groups[0]) / ry.emitted;
    verdict("a", Math.abs(ax - ay) <= 4 * Math.hypot(sx, sy), `reciprocity on ${h} m: A_x F(x-low -> floor) ${ax.toFixed(3)} and A_y F(floor -> x-low) ${ay.toFixed(3)} cell^2 differ by ${Math.abs(ax - ay).toFixed(3)} (at most ${(4 * Math.hypot(sx, sy)).toFixed(3)})`);
    verdict("a", Math.abs(ax - ref) <= 4 * sx, `reciprocity on ${h} m: A_x F(x-low -> floor) ${ax.toFixed(3)} against ${ref.toFixed(3)} +/- ${(4 * sx).toFixed(3)} cell^2`);
    verdict("a", Math.abs(ay - ref) <= 4 * sy, `reciprocity on ${h} m: A_y F(floor -> x-low) ${ay.toFixed(3)} against ${refY.toFixed(3)} +/- ${(4 * sy).toFixed(3)} cell^2`);
    verdict("f", ry.worst <= 1e-9, `reciprocity's floor run on ${h} m: worst step's imbalance ${ex(ry.worst)}`);
    // The floor run's own (a)1: its wall and the gas take nothing.
    verdict("a", ry.groups[2] === 0 && ry.gas === 0 && ry.out === 0, `reciprocity's floor run on ${h} m: floor ${ry.groups[2]} J, gas ${ry.gas} J, grid's sides ${ry.out} J (exactly 0 each)`);
    out.push({ reciprocity: true, h, ax, ay, ref, tolDiff: 4 * Math.hypot(sx, sy), tolX: 4 * sx, tolY: 4 * sy });
  }
  report.checks.a = out;
}

// ------------------------------------------------------------ (b)

function bCase(h) {
  const n = [12, 12, 12];
  const g = box(n);
  const block = (i, j, k) => i >= 7 && i <= 8 && j >= 7 && j <= 8 && k >= 7 && k <= 8;
  const { solid, material } = boxSolid(g, block);
  const groupOf = new Int8Array(g.N).fill(-1);
  const hot = [];
  for (const c of solid) {
    const [i, j, k] = coords(g, c);
    if (block(i, j, k)) {
      groupOf[c] = 6;
      hot.push([c, HOT]);
    } else groupOf[c] = g.wallOf(i, j, k);
  }
  return { g, groupOf, nGroups: 7, sp: (on) => spec(g, h, { solid, material, hot, options: { surfaceRayStart: on } }), hotCells: hot.length };
}

function judgeB(label, r, N, check = "b") {
  const E = r.emitted;
  verdict(check, r.groups[6] === 0, `${label}: the block takes ${r.groups[6]} J of its own emission (exactly 0)`);
  const tol = binomial(1 / 6, N);
  const shares = r.groups.slice(0, 6).map((x) => x / E);
  shares.forEach((F, w) => verdict(check, Math.abs(F - 1 / 6) <= tol, `${label}: ${WALLS[w]} ${f4(F)} against 0.1667 +/- ${f4(tol)} (${N} rays)`));
  return shares;
}

function checkB() {
  console.log("\n(b) Outside corners (a 2 x 2 x 2 block at 1000 K in a 12 x 12 x 12 box, 1000 steps)");
  const out = [];
  for (const h of CELLS) {
    const cs = bCase(h);
    const N = cs.hotCells * 8 * 1000;
    const on = stageRun(cs.sp(1), cs.groupOf, 7, 1000);
    const off = stageRun(cs.sp(0), cs.groupOf, 7, 1000);
    const label = `on ${h} m`;
    const shares = judgeB(label, on, N);
    verdict("f", on.worst <= 1e-9, `(b) ${label}: worst step's imbalance ${ex(on.worst)}`);
    const offShares = off.groups.map((x) => x / off.emitted);
    note(`item off: block ${f4(offShares[6])}, walls ${offShares.slice(0, 6).map(f4).join(", ")}; on: ${crossLine(on)}`);
    out.push({ h, rays: N, on: { shares, block: on.groups[6], worst: on.worst, rays: on.rays, crossings: on.crossings }, off: { block: offShares[6], walls: offShares.slice(0, 6) } });
  }
  report.checks.b = out;
}

// ------------------------------------------------------------ (c)

const GROUPS_C = ["i", "ii", "iii", "iv", "v", "vi", "outer"];

function cCase(h) {
  const n = [12, 10, 14];
  const g = box(n);
  // Inner cells 5 to 6, 4 to 5, 6 to 7 (grid + 2), and the pillar at inner (2, ., 3).
  const block = (i, j, k) => i >= 7 && i <= 8 && j >= 6 && j <= 7 && k >= 8 && k <= 9;
  const pillar = (i, j, k) => i === 4 && k === 5 && j >= 2 && j < 12;
  const { solid, material } = boxSolid(g, (i, j, k) => block(i, j, k) || pillar(i, j, k));
  const isSolid = new Uint8Array(g.N);
  for (const c of solid) isSolid[c] = 1;
  const groupOf = new Int8Array(g.N).fill(-1);
  const hot = [];
  const D = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  for (const c of solid) {
    const [i, j, k] = coords(g, c);
    if (g.outer(i, j, k)) {
      groupOf[c] = 6;
      continue;
    }
    hot.push([c, HOT]);
    if (block(i, j, k)) groupOf[c] = 4;
    else if (pillar(i, j, k)) groupOf[c] = 5;
    else {
      // A wall cell: its open face, and how many of its gas neighbour's four neighbours in the face's plane are solid.
      const open = D.filter(([a, b, d]) => !isSolid[g.at(i + a, j + b, k + d)]);
      if (!open.length) groupOf[c] = 3;
      else {
        const [a, b, d] = open[0];
        const gi = i + a, gj = j + b, gk = k + d;
        let touch = 0;
        for (const [p, q, r] of D) {
          if (p * a + q * b + r * d !== 0) continue;
          if (isSolid[g.at(gi + p, gj + q, gk + r)]) touch += 1;
        }
        groupOf[c] = Math.min(touch, 2);
        if (open.length > 1 || touch > 2) throw new Error(`(c): a wall cell with ${open.length} open faces and ${touch} solid neighbours`);
      }
    }
  }
  const counts = GROUPS_C.map((_, q) => groupOf.reduce((s, x) => s + (x === q ? 1 : 0), 0));
  return { g, groupOf, nGroups: 7, counts, sp: (on, eps = STONE) => spec(g, h, { solid, material, hot, options: { surfaceRayStart: on }, inert: [eps, eps, eps] }) };
}

function judgeC(label, r, em, h, check = "c") {
  const steps = r.steps;
  const A = h * h;
  const sMax = (4 * A * F.SIGMA * (HOT ** 4 - TA ** 4) * DT) / 8;
  const nets = [];
  for (const q of [0, 1, 2, 4, 5]) {
    const E = em.e[q] * steps;
    const net = r.groups[q] - E;
    const tol = 4 * Math.sqrt(sMax * E);
    nets.push({ group: GROUPS_C[q], net: net / E, tol: tol / E });
    verdict(check, Math.abs(net) <= tol, `${label}: group ${GROUPS_C[q]} net ${pct(net / E)} of its emission (at most ${pct(tol / E)})`);
  }
  verdict(check, r.groups[3] === 0, `${label}: group iv receives ${r.groups[3]} J (exactly 0)`);
  return nets;
}

function checkC() {
  console.log("\n(c) An isothermal enclosure with transparent gas (every solid cell off the grid's outer layer at 1000 K, 200 steps)");
  const out = [];
  for (const h of CELLS) {
    const cs = cCase(h);
    const want = [582, 248, 24, 154, 8, 10];
    // The scene as 13.25 counts it, or the check cannot run.
    if (!want.every((x, q) => cs.counts[q] === x)) throw new Error(`(c) on ${h} m: groups i to vi hold ${cs.counts.slice(0, 6).join(", ")} cells, not 13.25's ${want.join(", ")}`);
    note(`on ${h} m: groups i to vi hold ${cs.counts.slice(0, 6).join(", ")} cells, as 13.25 counts them`);
    const em = groupEmission(cs.sp(1), cs.groupOf, 7);
    const on = stageRun(cs.sp(1), cs.groupOf, 7, 200);
    const off = stageRun(cs.sp(0), cs.groupOf, 7, 200);
    // The groups' emission as computed here against the stage's own (float64 sums in another order).
    note(`on ${h} m: the groups' emission ${(em.all * 200).toFixed(3)} J, the stage's ${on.emitted.toFixed(3)} J (${ex(Math.abs(em.all * 200 - on.emitted) / on.emitted)} relative)`);
    const label = `on ${h} m`;
    const nets = judgeC(label, on, em, h);
    verdict("f", on.worst <= 1e-9, `(c) ${label}: worst step's imbalance ${ex(on.worst)}`);
    const offNets = [0, 1, 2, 4, 5].map((q) => ({ group: GROUPS_C[q], net: (off.groups[q] - em.e[q] * 200) / (em.e[q] * 200) }));
    note(`item off: nets ${offNets.map((x) => `${x.group} ${pct(x.net)}`).join(", ")}; group iv takes ${pct(off.groups[3] / off.emitted)} of all the emission; outer layer ${pct(off.groups[6] / off.emitted)}`);
    note(`on: outer layer takes ${on.groups[6]} J; ${crossLine(on)}`);
    out.push({ h, counts: cs.counts, on: { nets, iv: on.groups[3], outer: on.groups[6], worst: on.worst, rays: on.rays, crossings: on.crossings }, off: { nets: offNets, iv: off.groups[3] / off.emitted } });
  }
  report.checks.c = out;
}

// ------------------------------------------------------------ (d)

function dCase(h, layered) {
  const n = [12, 10, 14];
  const g = box(n);
  const ip = 2 + 6;
  const part = (i, j, k) => i === ip && g.inner(i, j, k);
  const { solid, material } = boxSolid(g, part);
  const groupOf = new Int8Array(g.N).fill(-1);
  const hot = [];
  for (let q = 0; q < solid.length; q += 1) {
    const c = solid[q];
    const [i, j, k] = coords(g, c);
    if (part(i, j, k)) {
      groupOf[c] = 0;
      if (layered) material[q] = 1;
      else hot.push([c, HOT]);
    } else groupOf[c] = i < ip ? 1 : i > ip ? 2 : 3;
  }
  const board = { name: "gypsum", rho: (GYPSUM.rho * GYPSUM.thickness) / h, c: GYPSUM.c, k: GYPSUM.k, eps: 1, layers: [GYPSUM] };
  const inert = layered ? [STONE, board, STONE] : [STONE, STONE, STONE];
  return { g, groupOf, nGroups: 4, cells: n[1] * n[2], sp: (on) => spec(g, h, { solid, material, hot, inert, wallHigh: layered ? HOT : null, options: { surfaceRayStart: on, wallLayers: layered ? 1 : 0 } }) };
}

function judgeD1(label, r, N, check = "d") {
  const E = r.emitted;
  const tol = binomial(0.5, N);
  verdict(check, r.groups[0] === 0, `${label}: the partition takes ${r.groups[0]} J (exactly 0)`);
  verdict(check, Math.abs(r.groups[1] / E - 0.5) <= tol, `${label}: the low side ${f4(r.groups[1] / E)} against 0.5 +/- ${f4(tol)} (${N} rays)`);
  verdict(check, Math.abs(E - r.groups[1] - r.groups[2]) <= 1e-9 * E && r.groups[3] === 0 && r.gas === 0 && r.out === 0, `${label}: the high side takes the rest, ${f4(r.groups[2] / E)} (the box walls in the partition's column ${r.groups[3]} J, gas ${r.gas} J, out ${r.out} J)`);
}

function checkD() {
  console.log("\n(d) Walls one cell thick with gas on both sides (a partition at inner x = 6, 200 steps)");
  const out = [];
  for (const h of CELLS) {
    // 1. Lumped.
    const cs = dCase(h, false);
    const N = cs.cells * 8 * 200;
    const on = stageRun(cs.sp(1), cs.groupOf, 4, 200);
    const off = stageRun(cs.sp(0), cs.groupOf, 4, 200);
    judgeD1(`1 (lumped) on ${h} m`, on, N);
    verdict("f", on.worst <= 1e-9, `(d)1 on ${h} m: worst step's imbalance ${ex(on.worst)}`);
    note(`item off: the partition takes ${f4(off.groups[0] / off.emitted)}, the low side ${f4(off.groups[1] / off.emitted)}, the high side ${f4(off.groups[2] / off.emitted)}; on: ${crossLine(on)}`);
    // 2. One-dimensional.
    const cl = dCase(h, true);
    const sim0 = buildSim(F, cl.sp(1));
    const walls = sim0.wall ? sim0.wall.cells.length : 0;
    // The partition as one-dimensional walls across x, or (d)2 cannot run.
    if (!(walls === cl.cells && Array.from(sim0.wall.meta).every((m) => (m & 3) === 0))) throw new Error(`(d)2 on ${h} m: the partition is ${walls} one-dimensional walls, not ${cl.cells} across x`);
    note(`2 on ${h} m: the partition is ${walls} one-dimensional walls across x`);
    const lon = stageRun(cl.sp(1), cl.groupOf, 4, 200);
    const loff = stageRun(cl.sp(0), cl.groupOf, 4, 200);
    const E = lon.emitted;
    verdict("d", lon.groups[0] === 0 && lon.ends === 0, `2 (one-dimensional) on ${h} m: the partition's means and ends take ${lon.groups[0]} J (exactly 0)`);
    verdict("d", Math.abs(lon.groups[2] - E) <= 1e-9 * E && Math.abs(lon.groups[1]) <= 1e-9 * E, `2 (one-dimensional) on ${h} m: the high side ${f4(lon.groups[2] / E)} of the emission (1 - ${ex(1 - lon.groups[2] / E)}), the low side ${lon.groups[1]} J`);
    verdict("f", lon.worst <= 1e-9, `(d)2 on ${h} m: worst step's imbalance ${ex(lon.worst)}`);
    // Item off: what one walls stage puts into the low end nodes, from what the radiation stage left in the partition's means.
    const s = buildSim(F, cl.sp(0));
    const r0 = s.stats.radiated;
    F.radiationStage(s, DT);
    const e1 = s.stats.radiated - r0;
    const W = s.wall, n = W.n, A = h * h;
    let low = 0;
    for (let w = 0; w < W.cells.length; w += 1) {
      const added = s.Ts[W.cells[w]] - W.last[w];
      low += added * W.C[(W.meta[w] >> 2) * n] * A;
    }
    note(`item off: the partition takes ${f4(loff.groups[0] / loff.emitted)} of the emission (its ends ${f4(loff.ends / loff.emitted)}), the high side ${f4(loff.groups[2] / loff.emitted)}, the low side ${f4(loff.groups[1] / loff.emitted)}; one walls stage then spreads ${low.toFixed(4)} J into the low end nodes, ${pct(low / e1)} of one stage's emission; on: ${crossLine(lon)}`);
    out.push({ h, lumped: { rays: N, low: on.groups[1] / on.emitted, high: on.groups[2] / on.emitted, partition: on.groups[0], worst: on.worst, off: { partition: off.groups[0] / off.emitted } }, layered: { walls, high: lon.groups[2] / E, low: lon.groups[1], partition: lon.groups[0], ends: lon.ends, worst: lon.worst, off: { partition: loff.groups[0] / loff.emitted, ends: loff.ends / loff.emitted, lowNodeJ: low, lowNodeShare: low / e1 } } });
  }
  report.checks.d = out;
}

// ------------------------------------------------------------ (e)

function checkE() {
  console.log("\n(e) The grid's side (a hot layer one cell thick on the grid's x-low side, 200 steps)");
  const out = [];
  for (const h of CELLS) {
    const nx = 13, ny = 14, nz = 14, N = nx * ny * nz;
    const g = { nx, ny, nz, N };
    const solid = [], material = [], hot = [];
    const groupOf = new Int8Array(N).fill(-1);
    for (let c = 0; c < N; c += 1) {
      const [i, j, k] = coords(g, c);
      const gas = i >= 1 && i <= 10 && j >= 2 && j < 12 && k >= 2 && k < 12;
      if (gas) continue;
      solid.push(c);
      material.push(0);
      groupOf[c] = i === 0 ? 0 : 1;
      if (i === 0 && j >= 2 && j < 12 && k >= 2 && k < 12) hot.push([c, HOT]);
    }
    const sp = (on) => spec(g, h, { solid, material, hot, options: { surfaceRayStart: on } });
    const R = hot.length * 8 * 200;
    const on = stageRun(sp(1), groupOf, 2, 200);
    const off = stageRun(sp(0), groupOf, 2, 200);
    const E = on.emitted, tol = binomial(0.5, R);
    verdict("e", on.groups[0] === 0, `on ${h} m: the layer takes ${on.groups[0]} J (exactly 0)`);
    verdict("e", Math.abs(on.out / E - 0.5) <= tol, `on ${h} m: radiatedOut ${f4(on.out / E)} against 0.5 +/- ${f4(tol)} (${R} rays)`);
    verdict("e", Math.abs(on.groups[1] / E - 0.5) <= tol, `on ${h} m: the walls ${f4(on.groups[1] / E)} against 0.5 +/- ${f4(tol)}`);
    verdict("f", on.worst <= 1e-9, `(e) on ${h} m: worst step's imbalance ${ex(on.worst)}`);
    const lost = off.emitted - off.groups[0] - off.groups[1] - off.gas;
    note(`item off: the layer takes ${f4(off.groups[0] / off.emitted)}, the walls ${f4(off.groups[1] / off.emitted)}, uncounted beyond the grid ${f4(lost / off.emitted)}; on: ${crossLine(on)}`);
    out.push({ h, rays: R, on: { layer: on.groups[0], out: on.out / E, walls: on.groups[1] / E, worst: on.worst }, off: { layer: off.groups[0] / off.emitted, walls: off.groups[1] / off.emitted, lost: lost / off.emitted } });
  }
  report.checks.e = out;
}

// ------------------------------------------------------------ (g) on the GPU

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
    let gas = 0, emitted = 0, radOut = 0, worst = 0, clamps = 0, clampJ = 0, joules = null;
    const t0 = performance.now();
    let applied = null;
    for (let s = 0; s < cs.steps; s += 1) {
      sim.steps = s;
      // One stage from one state also runs radApply, whose fields come back (reported beside (g)).
      const r = await gpu.radiationPass(sim, { apply: Boolean(cs.perCell) });
      if (r.Ts) applied = { T: Array.from(r.T), Ts: Array.from(r.Ts) };
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
          const v = r.wallLo[q] + r.wallHi[q];
          if (!v) continue;
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
      if (r.radiated > 0) worst = Math.max(worst, Math.abs(r.radiated - taken - (r.radiatedOut ?? 0)) / r.radiated);
      if (cs.firstSteps && s + 1 === cs.firstSteps) out.cases.push({ id: `${cs.id} first ${cs.firstSteps}`, groups: Array.from(groups), gas, emitted, out: radOut, worst, clamps, clampJ, joules, steps: s + 1 });
    }
    const ms = performance.now() - t0;
    gpu.destroy();
    out.cases.push({ id: cs.id, groups: Array.from(groups), gas, emitted, out: radOut, worst, clamps, clampJ, joules, steps: cs.steps, ms, cells: cells ? Array.from(cells) : null, applied });
  }
  return out;
}

async function checkG() {
  const { chromium } = await import("playwright-core");
  console.log("\n(g) The CPU and the GPU cast the same rays (on 0.20 m cells: one stage each, then (a), (b) and (c) on the GPU's passes)");
  const TYPES = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".html": "text/html", ".wasm": "application/wasm" };
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    if (url.pathname === "/" || url.pathname === "/check.html") {
      res.writeHead(200, { "content-type": "text/html" }).end("<!doctype html><meta charset=utf-8><title>Surface rays check</title>");
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
  const run = (cases) => page.evaluate(gpuSuite, { base: `${origin}/src/core/`, cases, builder: buildSim.toString() });
  const out = { parity: [], runs: [] };
  try {
    // 1 to 3: one stage from one state on each path.
    const h = 0.2;
    const parity = [
      ["(a) 8x8x8", aCase([8, 8, 8], h)],
      ["(a) 12x10x14", aCase([12, 10, 14], h)],
      ["(c) enclosure", cCase(h)],
      ["(d)1 lumped partition", dCase(h, false)],
    ];
    const gp = await run(parity.map(([id, cs]) => ({ id, spec: cs.sp(1), groupOf: Array.from(cs.groupOf), nGroups: cs.nGroups, steps: 1, perCell: true })));
    if (gp.error) throw new Error(gp.error);
    out.adapter = gp.adapter;
    console.log(`  adapter: ${gp.adapter}`);
    for (const [id, cs] of parity) {
      const cpu = stageRun(cs.sp(1), cs.groupOf, cs.nGroups, 1, { perCell: true });
      const other = stageRun(cs.sp(1), cs.groupOf, cs.nGroups, 1, { seed: 2, perCell: true });
      const gpu = gp.cases.find((x) => x.id === id);
      const rel = Math.abs(gpu.emitted - cpu.emitted) / cpu.emitted;
      verdict("g", rel <= 1e-5, `${id}: emission GPU ${gpu.emitted.toFixed(4)} J, CPU ${cpu.emitted.toFixed(4)} J (${ex(rel)} relative, at most 1e-5)`);
      const names = id.startsWith("(a)") ? WALLS : id.startsWith("(c)") ? GROUPS_C : ["partition", "low side", "high side", "partition's column"];
      let worstGroup = 0;
      const rows = [];
      for (let q = 0; q < cs.nGroups; q += 1) {
        const d = Math.abs(gpu.groups[q] - cpu.groups[q]);
        const ok = d <= 0.01 * Math.abs(cpu.groups[q]);
        if (cpu.groups[q] !== 0) worstGroup = Math.max(worstGroup, d / Math.abs(cpu.groups[q]));
        rows.push(`${names[q]} ${gpu.groups[q].toFixed(3)} / ${cpu.groups[q].toFixed(3)}`);
        verdict("g", ok, `${id}: ${names[q]} GPU ${gpu.groups[q].toFixed(4)} J, CPU ${cpu.groups[q].toFixed(4)} J (within 1 %)`);
      }
      verdict("g", Math.abs(gpu.out - cpu.out) <= 0.01 * Math.abs(cpu.out) && Math.abs(gpu.gas - cpu.gas) <= 0.01 * Math.abs(cpu.gas), `${id}: radiatedOut GPU ${gpu.out} J, CPU ${cpu.out} J; gas GPU ${gpu.gas} J, CPU ${cpu.gas} J (within 1 %)`);
      let dg = 0, ds = 0;
      for (let c = 0; c < cpu.cells.length; c += 1) {
        dg += Math.abs(gpu.cells[c] - cpu.cells[c]);
        ds += Math.abs(other.cells[c] - cpu.cells[c]);
      }
      dg /= cpu.cells.length;
      ds /= cpu.cells.length;
      verdict("g", dg <= 0.1 * ds, `${id}: mean |GPU - CPU| a cell ${ex(dg)} J against the CPU's between seeds 1 and 2 ${ex(ds)} J (ratio ${ex(dg / ds)}, at most 0.1)`);
      note(`${id}: worst group difference ${ex(worstGroup)} relative; GPU clamps ${gpu.clamps} (${gpu.clampJ} J), deposits in ${gpu.joules ? "joules" : "kelvin"}`);
      // Reported, not judged: radApply's fields after the same stage against the CPU's stage.
      let applied = null;
      if (gpu.applied) {
        const s = buildSim(F, cs.sp(1));
        const Ts0 = s.Ts.slice();
        F.radiationStage(s, DT);
        let worst = 0, rise = 0, risen = 0;
        for (let c = 0; c < s.N; c += 1) {
          if (s.kind[c] === F.GAS) continue;
          worst = Math.max(worst, Math.abs(gpu.applied.Ts[c] - s.Ts[c]));
          if (s.Ts[c] !== Ts0[c]) {
            rise += Math.abs(s.Ts[c] - Ts0[c]);
            risen += 1;
          }
        }
        let gasWorst = 0;
        for (let c = 0; c < s.N; c += 1) if (s.kind[c] === F.GAS) gasWorst = Math.max(gasWorst, Math.abs(gpu.applied.T[c] - s.T[c]));
        note(`${id}: after radApply, the solids' Ts differ from the CPU's stage by at most ${ex(worst)} K (the CPU's mean change ${ex(rise / Math.max(1, risen))} K over ${risen} cells); the gas by at most ${ex(gasWorst)} K`);
        applied = { solidMax: worst, meanChange: rise / Math.max(1, risen), gasMax: gasWorst };
      }
      out.parity.push({ id, emission: { gpu: gpu.emitted, cpu: cpu.emitted, rel }, groups: { gpu: gpu.groups, cpu: cpu.groups }, out: { gpu: gpu.out, cpu: cpu.out }, meanAbs: { gpu: dg, seeds: ds, ratio: dg / ds }, worstGroup, clamps: gpu.clamps, joules: gpu.joules, applied });
    }
    // Then (a), (b) and (c) on the GPU's passes.
    for (const hg of GPU_CELLS) {
      const cases = [];
      for (const n of [[8, 8, 8], [12, 10, 14], [6, 16, 10], [40, 8, 8]]) {
        const cs = aCase(n, hg);
        cases.push({ id: `a ${n.join("x")} ${hg}`, spec: cs.sp(1), groupOf: Array.from(cs.groupOf), nGroups: 6, steps: 200, rays: cs.hotCells * 1600, n });
      }
      const fl = aCase([12, 10, 14], hg, "y-low");
      cases.push({ id: `a floor 12x10x14 ${hg}`, spec: fl.sp(1), groupOf: Array.from(fl.groupOf), nGroups: 6, steps: 200, rays: fl.hotCells * 1600 });
      const b = bCase(hg);
      cases.push({ id: `b ${hg}`, spec: b.sp(1), groupOf: Array.from(b.groupOf), nGroups: 7, steps: 1000, firstSteps: 200, rays: b.hotCells * 8000 });
      const c = cCase(hg);
      cases.push({ id: `c ${hg}`, spec: c.sp(1), groupOf: Array.from(c.groupOf), nGroups: 7, steps: 200 });
      const gr = await run(cases.map(({ rays, n, ...x }) => x));
      if (gr.error) throw new Error(gr.error);
      const get = (id) => gr.cases.find((x) => x.id === id);
      for (const cs of cases) {
        const r = get(cs.id);
        const label = `GPU ${cs.id} m`;
        if (cs.id.startsWith("a ") && cs.n) judgeA(label, cs.n, r, cs.rays, "g");
        if (cs.id.startsWith("b ")) {
          judgeB(label, r, cs.rays, "g");
          const first = get(`${cs.id} first 200`);
          const tol = binomial(1 / 6, cs.rays / 5);
          note(`${label}, its first 200 steps (${cs.rays / 5} rays, 4 standard deviations ${f4(tol)}): block ${first.groups[6]} J, walls ${first.groups.slice(0, 6).map((x) => f4(x / first.emitted)).join(", ")}; ${first.groups.slice(0, 6).every((x) => Math.abs(x / first.emitted - 1 / 6) <= tol) ? "all within" : "NOT all within"}`);
        }
        if (cs.id.startsWith("c ")) judgeC(label, r, groupEmission(c.sp(1), c.groupOf, 7), hg, "g");
        verdict("g", r.worst <= 0.01 && r.clamps === 0, `${label}: the GPU's books close within ${ex(r.worst)} of a step's emission (at most 1 %); ${r.clamps} deposits clamped; ${r.ms.toFixed(0)} ms for ${r.steps} passes`);
        out.runs.push({ id: cs.id, groups: r.groups, emitted: r.emitted, out: r.out, gas: r.gas, worst: r.worst, clamps: r.clamps, clampJ: r.clampJ, ms: r.ms, steps: r.steps });
      }
      // (a)3 on the GPU.
      const n = [12, 10, 14];
      const rx = get(`a 12x10x14 ${hg}`), ry = get(`a floor 12x10x14 ${hg}`);
      const Ax = n[1] * n[2], Ay = n[0] * n[2];
      const ref = Ax * F14(n[1], n[0], n[2]);
      const Nx = Ax * 1600, Ny = Ay * 1600;
      const Fx = ref / Ax, Fy = ref / Ay;
      const sx = Ax * Math.sqrt((Fx * (1 - Fx)) / Nx), sy = Ay * Math.sqrt((Fy * (1 - Fy)) / Ny);
      const ax = (Ax * rx.groups[2]) / rx.emitted, ay = (Ay * ry.groups[0]) / ry.emitted;
      verdict("g", Math.abs(ax - ay) <= 4 * Math.hypot(sx, sy) && Math.abs(ax - ref) <= 4 * sx && Math.abs(ay - ref) <= 4 * sy, `GPU reciprocity on ${hg} m: ${ax.toFixed(3)} and ${ay.toFixed(3)} cell^2 against ${ref.toFixed(3)} (difference at most ${(4 * Math.hypot(sx, sy)).toFixed(3)}, each at most ${(4 * sx).toFixed(3)} and ${(4 * sy).toFixed(3)})`);
      verdict("g", ry.groups[2] === 0 && ry.gas === 0 && ry.out === 0, `GPU reciprocity's floor run on ${hg} m: floor ${ry.groups[2]} J, gas ${ry.gas} J, out ${ry.out} J (exactly 0 each)`);
    }
  } finally {
    await browser.close();
    server.close();
  }
  report.checks.g = out;
}

// ------------------------------------------------------------ run

const t0 = performance.now();
if (ONLY.has("a")) checkA();
if (ONLY.has("b")) checkB();
if (ONLY.has("c")) checkC();
if (ONLY.has("d")) checkD();
if (ONLY.has("e")) checkE();
if (ONLY.has("g")) {
  try {
    await checkG();
  } catch (error) {
    verdict("g", false, `the GPU part did not run: ${error.message}`);
  }
}
report.verdicts = verdicts;
report.lines = lines;
report.seconds = (performance.now() - t0) / 1000;
console.log("\nVerdicts (13.25 checks; (h) to (j) run apart):");
for (const [k, v] of Object.entries(verdicts)) console.log(`  (${k}) ${v ? "pass" : "FAIL"}`);
console.log(`${report.seconds.toFixed(1)} s`);
const json = flag("--json", null);
if (json) await writeFile(json, JSON.stringify(report, null, 1));
