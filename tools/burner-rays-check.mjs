#!/usr/bin/env node
/**
 * The checks of physics item 12, the burner's radiant part leaves from its
 * flame (wiki/research/fire-calibration-plan-2026-11-11.md, 16.10, "Checks",
 * read with the coordinator's decisions in 16.12; src/core/flamescapes.js
 * burnerFlameRays, flameRays, burnerMarch, burnerFlameCell). Every reference
 * is a closed form: view factors, symmetry, conservation, bit-identity. No
 * measured datum enters any of them.
 *
 *   node tools/burner-rays-check.mjs [--only a,b,c,d,e,f,g,h,i,k] [--cells 0.2,0.34]
 *        [--base-rev 40f8ea9] [--json out.json]
 *
 * Each scene is a grid of cells built here by hand, as
 * tools/surface-rays-check.mjs builds its own. A box is given by its inner
 * size in cells, n_x x n_y x n_z (y up), with walls two cells thick of stone
 * (INERT[0]); cells are named by their inner coordinates. Every gas cell
 * starts at ambient with no soot, so the gas is transparent.
 *
 * The burner is built by createBurner from `layers` with no `diameter`, so
 * every cell of it takes an equal share of the fuel. Its fuel is METHANE (no
 * soot; (f) gives it a soot yield of 0.1), with radiant fraction 0.3 and 16
 * rays, its table held at 100 kW, dt 0.1 s, so each step releases 10 kJ.
 * Seed 1. The tool calls burnerStep alone 10 000 times (sim.time stays 0;
 * 1 is added to sim.steps after each call), reads the step's deposits from
 * the float64 arrays sim.work.burnerGas, burnerSolid, burnerWallLo,
 * burnerWallHi and burnerOut, turned into J by the capacities they were
 * divided by (rho cp V of the gas after the burn; max(solid capacity, 1e-6)),
 * then sets every gas cell's T, Y and rho, every Ts and every wall node back.
 * Cells of 0.20 m and 0.34 m: 160 000 rays a case.
 *
 * Tolerance (16.10): a share F read from N rays of equal share passes within
 * 4 sqrt(F (1 - F) / N), F the reference. References: the mean over the
 * source cells of the mean over each cell's volume of Omega / (4 pi), Omega
 * the solid angle a wall's rectangle subtends, by Gauss-Legendre quadrature
 * (24 points a side; 16 printed beside); the plan's four-figure values are
 * printed beside each.
 *
 * (a) Symmetry: a 9 x 9 x 9 box, the source its centre cell; each wall 1/6.
 * (b) A one-cell flame on the floor: the 12 x 10 x 14 box, the source cell
 *     (3, 0, 4); the six walls and the floor cell under it (1/6).
 * (c) A flame of 2 x 2 columns and 3 layers: the same box, the cells i in
 *     {5, 6}, j in {0, 1, 2}, k in {6, 7}; each wall, the floor under the
 *     flame and the rest.
 * (d) The reach: a 40 x 8 x 8 box, the source (0, 3, 3); the six walls, and
 *     the gas exactly 0.
 * (e) An open side: (b)'s box with its x-high side the grid's side and no
 *     wall there; burnerOut and the five walls.
 * (f) The flame is clear: (c) with the fuel's soot yield 0.1 kg/kg; every
 *     flame cell's gas exactly 0 J, and each surface its share of (c).
 * (g) The cell drawn by its heat: burnerFlameCell with masses 1 : 2 : 3 : 4
 *     and 10^6 draws of u0 from the salt-4 streams as flameRays takes them
 *     (cell 0, steps 0 to 62 499, 16 rays a step, six draws a ray, seed 1).
 * (h) 1. In every step of (a) to (f) the radiant output equals the J in the
 *     arrays plus burnerOut within 1e-9 of it. 2. In (a), (b), (c) and (e),
 *     one burnerStep on and one off from one state leave every gas cell's T,
 *     Y and rho, and released, convective, radiant and stats.released, bit
 *     for bit the same.
 * (i) Off is off (the part run here): an 8 x 8 x 8 box on 0.2 m cells with a
 *     validation burner (radiant fraction 0.2, 16 rays, one cell on the
 *     floor, METHANE, 20 kW), 300 full steps, against the base's solver
 *     (--base-rev, read from git): every field and every stats entry. With
 *     1, the differences are reported. The GPU's `supported` (its own source,
 *     read from flamescapesGpu.js) on a validation burner and a design fire
 *     with the item on. The fingerprint tool's design-fire room on and off.
 * (k) Speed, reported: burnerStep's ms on and off over (c)'s scene, and the
 *     mean crossings of a burner ray in every scene.
 * Beside, reported with the item off: the shares of (a) to (e) from the
 * present origin and its 16 fixed directions (read from Ts, float32, the
 * solids set to 0 K for the reading; the rest of the radiant output is the
 * gas's, and in (e) what leaves the grid, which the item off does not count).
 * The two cells cast the same rays in cell units (same cells, steps and
 * seed), so their shares agree: they are one sample, not two.
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const FRONTEND = path.join(ROOT, "topologicpy-web-frontend");
const CORE = path.join(FRONTEND, "src/core");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
};
const imp = (f) => import(pathToFileURL(path.join(CORE, f)).href);
const F = await imp("flamescapes.js");
const V = await imp("voxelize.js");
if (typeof F.burnerFlameCell !== "function") throw new Error("this solver has no burnerFlameCell (item 12 is not built)");

const ONLY = new Set(flag("--only", "a,b,c,d,e,f,g,h,i,k").split(","));
const CELLS = flag("--cells", "0.2,0.34").split(",").map(Number);
const BASE_REV = flag("--base-rev", "40f8ea9");
const DT = 0.1;
const STEPS = 10000;
const RAYS = 16;
const CHI = 0.3;
const Q = 1e5;
const M = V.MATERIALS;
const ON = { burnerFlameRays: 1 };
const OFF = { burnerFlameRays: 0 };
const SOOTY = { heatOfCombustion: F.METHANE.heatOfCombustion, yields: { co2: F.METHANE.yields.co2, co: 0, soot: 0.1 } };
const INERT_FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };

const verdicts = {};
const lines = [];
const report = { tool: "tools/burner-rays-check.mjs", plan: "wiki/research/fire-calibration-plan-2026-11-11.md 16.10 and 16.12", dt: DT, seed: 1, steps: STEPS, rays: RAYS, radiantFraction: CHI, hrrW: Q, cells: CELLS, checks: {} };
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
const f6 = (x) => (Number.isFinite(x) ? x.toFixed(6) : String(x));
const ex = (x) => (Number.isFinite(x) ? x.toExponential(2) : String(x));
const tol4 = (F0, N) => 4 * Math.sqrt((F0 * (1 - F0)) / N);

// ------------------------------------------------------------ references

/** Gauss-Legendre nodes and weights on [0, 1], n points (Newton on P_n). */
function gaussLegendre(n) {
  const x = new Float64Array(n), w = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    let dp = 0;
    for (let it = 0; it < 100; it += 1) {
      let p0 = 1, p1 = z;
      for (let k = 2; k <= n; k += 1) [p0, p1] = [p1, ((2 * k - 1) * z * p1 - (k - 1) * p0) / k];
      dp = (n * (z * p1 - p0)) / (z * z - 1);
      const dz = p1 / dp;
      z -= dz;
      if (Math.abs(dz) < 1e-16) break;
    }
    let p0 = 1, p1 = z;
    for (let k = 2; k <= n; k += 1) [p0, p1] = [p1, ((2 * k - 1) * z * p1 - (k - 1) * p0) / k];
    dp = (n * (z * p1 - p0)) / (z * z - 1);
    x[i] = 0.5 * (1 - z);
    w[i] = 1 / ((1 - z * z) * dp * dp);
  }
  return { x, w };
}

/** The solid angle of the rectangle [x1, x2] x [y1, y2] in a plane at distance d, from the foot of the perpendicular. */
function omega(d, x1, x2, y1, y2) {
  const G = (x, y) => Math.atan((x * y) / (d * Math.sqrt(x * x + y * y + d * d)));
  return G(x2, y2) - G(x1, y2) - G(x2, y1) + G(x1, y1);
}

/**
 * Shares of a box's surfaces (inner size n in cells) from a uniform,
 * isotropic source over the cells `src` ([i, j, k] inner), each cell an equal
 * share: x-low, x-high, y-low (floor), y-high (ceiling), z-low, z-high, then
 * each extra floor rectangle [[x1, x2], [z1, z2]].
 */
function shares(n, src, floorRects = [], points = 24) {
  const [X, Y, Z] = n;
  const { x: gx, w: gw } = gaussLegendre(points);
  const out = new Float64Array(6 + floorRects.length);
  for (const [ci, cj, ck] of src) {
    for (let a = 0; a < points; a += 1) {
      const px = ci + gx[a];
      for (let b = 0; b < points; b += 1) {
        const py = cj + gx[b];
        for (let c = 0; c < points; c += 1) {
          const pz = ck + gx[c];
          const wt = (gw[a] * gw[b] * gw[c]) / (4 * Math.PI * src.length);
          out[0] += wt * omega(px, -py, Y - py, -pz, Z - pz);
          out[1] += wt * omega(X - px, -py, Y - py, -pz, Z - pz);
          out[2] += wt * omega(py, -px, X - px, -pz, Z - pz);
          out[3] += wt * omega(Y - py, -px, X - px, -pz, Z - pz);
          out[4] += wt * omega(pz, -px, X - px, -py, Y - py);
          out[5] += wt * omega(Z - pz, -px, X - px, -py, Y - py);
          for (let r = 0; r < floorRects.length; r += 1) {
            const [[x1, x2], [z1, z2]] = floorRects[r];
            out[6 + r] += wt * omega(py, x1 - px, x2 - px, z1 - pz, z2 - pz);
          }
        }
      }
    }
  }
  return Array.from(out);
}

// ------------------------------------------------------------ scenes

const WALLS = ["x-low", "x-high", "floor", "ceiling", "z-low", "z-high"];

/** A box of inner cells n in walls two cells thick; with openHigh, no wall on its x-high side (the grid's side). */
function box(n, openHigh = false) {
  const [ni, nj, nk] = n;
  const nx = ni + (openHigh ? 2 : 4), ny = nj + 4, nz = nk + 4;
  const at = (i, j, k) => i + nx * (j + ny * k);
  const inner = (i, j, k) => i >= 2 && (openHigh || i < ni + 2) && j >= 2 && j < nj + 2 && k >= 2 && k < nk + 2;
  const wallOf = (i, j, k) => (i < 2 ? 0 : i >= ni + 2 ? 1 : j < 2 ? 2 : j >= nj + 2 ? 3 : k < 2 ? 4 : 5);
  const cell = (i, j, k) => at(i + 2, j + 2, k + 2);
  return { nx, ny, nz, N: nx * ny * nz, at, inner, wallOf, cell, n, openHigh };
}
const coords = (g, c) => [c % g.nx, Math.floor(c / g.nx) % g.ny, Math.floor(c / (g.nx * g.ny))];

/**
 * A solver over the box with a burner: `layers` of inner cells, its table
 * held at q W, its radiant fraction chi and `rays` rays; options beside.
 */
function buildSim(Fm, g, h, { layers, fuel = Fm.METHANE, chi = CHI, rays = RAYS, q = Q, options = {}, extra = null }) {
  const N = g.N;
  const grid = { nx: g.nx, ny: g.ny, nz: g.nz, h, origin: [0, 0, 0], axes: [0, 1, 2], upAxis: "y", cells: N };
  const fraction = new Float32Array(N * M);
  const solid = new Uint8Array(N);
  for (let c = 0; c < N; c += 1) {
    const [i, j, k] = coords(g, c);
    // `extra` (inner coordinates) gives a solid's material in the gas, or -1.
    const m = !g.inner(i, j, k) ? 0 : extra ? extra(i - 2, j - 2, k - 2) : -1;
    if (m >= 0) {
      solid[c] = 1;
      fraction[c * M + m] = 1;
    }
  }
  const burner = { layers: layers.map((l) => l.map(([i, j, k]) => g.cell(i, j, k))), hrr: [[0, q]], radiantFraction: chi, rays, fuel };
  const scene = { grid, fraction, solid, contents: { cells: [], masses: [] }, ignition: null, pilot: null, burner };
  return Fm.createSolver(scene, { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL, seed: 1, dt: DT, ...options });
}

const solidCap = (sim, c) => sim.inertC[c] + F.skinMass(sim, c) * (sim.fuels[sim.fuelKind[c]]?.heatCapacity ?? sim.P.fuelHeatCapacity) + sim.moisture[c] * 4186;
const gasCap = (sim, c) => sim.rho[c] * F.mixtureCp(sim.Y, c * F.NS) * sim.V;

/**
 * burnerStep alone `steps` times from the scene's state, set back after each.
 * groupOf maps a solid cell to its groups (a list of indices). With the item
 * on the deposits are read from the float64 arrays; off, from Ts (float32),
 * the gas and what left the grid being the rest of the radiant output.
 */
function burnerRun(sim, groupOf, nGroups, steps, { on, flameCheck = false } = {}) {
  const W = sim.wall;
  const b = sim.burner;
  // Off, the deposits are read from Ts: every solid set to 0 K first, so that float32 holds each to about 1e-7 of it
  // (no part of burnerStep reads a solid's temperature).
  if (!on) for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] !== F.GAS) sim.Ts[c] = 0;
  const T0 = sim.T.slice(), Ts0 = sim.Ts.slice(), Y0 = sim.Y.slice(), rho0 = sim.rho.slice();
  const WT0 = W ? W.T.slice() : null, WL0 = W ? W.last.slice() : null;
  const caps = new Float64Array(sim.N);
  for (let c = 0; c < sim.N; c += 1) caps[c] = Math.max(solidCap(sim, c), 1e-6);
  const groups = new Float64Array(nGroups);
  let gas = 0, out = 0, radiant = 0, worst = 0, rest = 0, ms = 0, flameGas = 0, flameCells = 0, flameAbs = 0;
  let maxSoot = 0;
  for (let s = 0; s < steps; s += 1) {
    const r0 = b.radiant;
    const t0 = performance.now();
    F.burnerStep(sim, DT);
    ms += performance.now() - t0;
    const e = b.radiant - r0;
    radiant += e;
    let taken = 0;
    if (on) {
      const w = sim.work;
      const fl = b.flame;
      for (let c = 0; c < sim.N; c += 1) {
        const dg = w.burnerGas[c], ds = w.burnerSolid[c];
        if (dg) {
          const J = dg * gasCap(sim, c);
          gas += J;
          taken += J;
        }
        if (ds) {
          const J = ds * caps[c];
          for (const q of groupOf[c] || []) groups[q] += J;
          taken += J;
        }
      }
      if (W) {
        for (let q = 0; q < W.cells.length; q += 1) {
          const c = W.cells[q];
          const J = (w.burnerWallLo[q] + w.burnerWallHi[q]) * caps[c];
          if (!J) continue;
          for (const g0 of groupOf[c] || []) groups[g0] += J;
          taken += J;
        }
      }
      const o = w.burnerOut[0];
      out += o;
      if (e > 0) worst = Math.max(worst, Math.abs(e - taken - o) / e);
      if (flameCheck) {
        for (let q = 0; q < fl.n; q += 1) {
          const c = fl.cells[q];
          flameCells += 1;
          flameAbs += Math.abs(w.burnerGas[c]);
          flameGas += w.burnerGas[c] * gasCap(sim, c);
          maxSoot = Math.max(maxSoot, sim.Y[c * F.NS + F.SPECIES.SOOT]);
        }
      }
    } else {
      for (let c = 0; c < sim.N; c += 1) {
        if (sim.kind[c] === F.GAS) continue;
        const d = sim.Ts[c] - Ts0[c];
        if (!d) continue;
        const J = d * caps[c];
        for (const q of groupOf[c] || []) groups[q] += J;
        taken += J;
      }
      rest += e - taken;
    }
    sim.T.set(T0);
    sim.Ts.set(Ts0);
    sim.Y.set(Y0);
    sim.rho.set(rho0);
    if (W) {
      W.T.set(WT0);
      W.last.set(WL0);
    }
    sim.steps += 1;
  }
  const st = sim.stats;
  return { groups: Array.from(groups), gas, out, radiant, worst, rest, ms, rays: st.burnerRays ?? steps * RAYS, crossings: st.burnerCrossings ?? null, flameGas, flameAbs, flameCells, maxSoot };
}

/** The walls' groups of a box's solid cells (by wallOf), and extra groups for given floor cells. */
function wallGroups(g, extra = []) {
  const groupOf = new Array(g.N);
  for (let c = 0; c < g.N; c += 1) {
    const [i, j, k] = coords(g, c);
    if (g.inner(i, j, k)) continue;
    const list = [g.wallOf(i, j, k)];
    extra.forEach((test, q) => {
      if (test(i - 2, j - 2, k - 2)) list.push(6 + q);
    });
    groupOf[c] = list;
  }
  return groupOf;
}

// ------------------------------------------------------------ the cases

const PLAN = {
  a: [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667],
  b: [0.1716, 0.0809, 0.4574, 0.0936, 0.1336, 0.063, 0.1667],
  c: [0.1354, 0.1354, 0.3987, 0.1203, 0.1051, 0.1051, 0.1246, 0.2741],
  d: [0.443, 0.0032, 0.1511, 0.1258, 0.1511, 0.1258],
};

const CASES = {
  a: { n: [9, 9, 9], layers: [[[4, 4, 4]]], groups: WALLS, extra: [] },
  b: { n: [12, 10, 14], layers: [[[3, 0, 4]]], groups: [...WALLS, "floor cell under it"], extra: [(i, j, k) => j === -1 && i === 3 && k === 4] },
  c: {
    n: [12, 10, 14],
    layers: [0, 1, 2].map((j) => [[5, j, 6], [6, j, 6], [5, j, 7], [6, j, 7]]),
    groups: [...WALLS, "floor under the flame", "floor, the rest"],
    extra: [(i, j, k) => j === -1 && i >= 5 && i <= 6 && k >= 6 && k <= 7, (i, j, k) => j === -1 && !(i >= 5 && i <= 6 && k >= 6 && k <= 7)],
  },
  d: { n: [40, 8, 8], layers: [[[0, 3, 3]]], groups: WALLS, extra: [] },
  e: { n: [12, 10, 14], layers: [[[3, 0, 4]]], groups: [...WALLS, "floor cell under it"], extra: [(i, j, k) => j === -1 && i === 3 && k === 4], openHigh: true },
};

/** The quadrature references of every case: the six walls, then the extra floor rectangles. */
function references() {
  const refs = {};
  const t0 = performance.now();
  for (const [id, cs] of Object.entries(CASES)) {
    if (id === "e") continue;
    const src = cs.layers.flat();
    const rects = id === "b" ? [[[3, 4], [4, 5]]] : id === "c" ? [[[5, 7], [6, 8]]] : [];
    const r24 = shares(cs.n, src, rects, 24);
    const r16 = shares(cs.n, src, rects, 16);
    if (id === "c") {
      r24.push(r24[2] - r24[6]);
      r16.push(r16[2] - r16[6]);
    }
    refs[id] = { r24, r16, sum: r24.slice(0, 6).reduce((a, x) => a + x, 0) };
  }
  refs.e = refs.b;
  report.checks.references = { refs, seconds: (performance.now() - t0) / 1000 };
  return refs;
}

function caseLine(check, label, x, F0, plan, N) {
  const t = tol4(F0, N);
  const planText = plan != null ? `; plan ${f4(plan)} +/- ${f4(t)}` : "";
  verdict(check, Math.abs(x - F0) <= t, `${label} ${f4(x)} against ${f6(F0)} +/- ${f4(t)} (4 sd of ${N} rays${planText})`);
}

const crossLine = (r) => `${r.rays} rays, ${r.crossings != null ? (r.crossings / r.rays).toFixed(2) : "-"} crossings a ray`;

function runShares(refs) {
  const out = {};
  const N = STEPS * RAYS;
  for (const id of ["a", "b", "c", "d", "e", "f"]) {
    if (!ONLY.has(id)) continue;
    const base = id === "f" ? "c" : id;
    const cs = CASES[base];
    const ref = refs[base].r24;
    console.log(`\n(${id}) ${{ a: "symmetry", b: "a one-cell flame on the floor", c: "a flame of 2 x 2 columns and 3 layers", d: "the reach", e: "an open side", f: "the flame is clear" }[id]}: ${cs.n.join(" x ")} box${cs.openHigh || id === "e" ? ", its x-high side open" : ""}`);
    const g = box(cs.n, id === "e");
    const groupOf = wallGroups(g, cs.extra);
    const nGroups = 6 + cs.extra.length;
    out[id] = {};
    for (const h of CELLS) {
      const fuel = id === "f" ? SOOTY : F.METHANE;
      const sim = buildSim(F, g, h, { layers: cs.layers, fuel, options: ON });
      const r = burnerRun(sim, groupOf, nGroups, STEPS, { on: true, flameCheck: id === "f" });
      const sh = r.groups.map((J) => J / r.radiant);
      const outShare = r.out / r.radiant;
      const gasShare = r.gas / r.radiant;
      out[id][h] = { shares: sh, out: outShare, gas: gasShare, gasJ: r.gas, radiant: r.radiant, worst: r.worst, ms: r.ms, rays: r.rays, crossings: r.crossings };
      console.log(`  ${h} m: radiant ${r.radiant.toFixed(1)} J over ${STEPS} steps, ${crossLine(r)}`);
      const planRow = id === "f" ? PLAN.c : id === "e" ? PLAN.b : PLAN[id];
      for (let q = 0; q < nGroups; q += 1) {
        if (id === "e" && q === 1) continue; // no x-high wall: its share leaves the grid
        caseLine(id, `${h} m, ${cs.groups[q]}`, sh[q], ref[q], planRow[q], N);
      }
      if (id === "e") {
        caseLine("e", `${h} m, burnerOut (the open x-high side)`, outShare, ref[1], PLAN.b[1], N);
        reported("e", `${h} m: the gas takes ${r.gas} J; the walls' x-high group ${sh[1]} (no wall there)`);
      } else reported(id, `${h} m: burnerOut ${r.out} J (a closed box)`);
      if (id === "d") verdict("d", r.gas === 0, `${h} m, the gas takes ${r.gas} J (exactly 0: transparent gas takes nothing)`);
      else if (id !== "f" && id !== "e") reported(id, `${h} m: the gas takes ${r.gas} J`);
      if (id === "f") {
        verdict("f", r.flameAbs === 0, `${h} m, every flame cell's gas takes ${r.flameGas} J from the burner's rays over ${r.flameCells} flame-cell steps (exactly 0); the gas as a whole ${r.gas} J`);
        const tau = sim.P.sootAbsorption * r.maxSoot * h;
        reported("f", `${h} m: the flame cells' largest soot mass fraction after a step's burn ${ex(r.maxSoot)}, an optical depth of ${ex(tau)} across one cell`);
      }
      verdict("h", r.worst <= 1e-9, `${h} m, (${id}) energy: the worst step's |radiant - deposits - burnerOut| / radiant ${ex(r.worst)} (against 1e-9)`);
      reported("k", `(${id}) ${h} m: burnerStep ${(r.ms / STEPS).toFixed(4)} ms a call on; ${crossLine(r)}`);
    }
  }
  report.checks.shares = out;
}

/** The present origin and its 16 fixed directions, item off: the same scenes, deposits read from Ts. */
function runOffShares() {
  const out = {};
  for (const id of ["a", "b", "c", "d", "e", "f"]) {
    if (!ONLY.has(id)) continue;
    const base = id === "f" ? "c" : id;
    const cs = CASES[base];
    const g = box(cs.n, id === "e");
    const groupOf = wallGroups(g, cs.extra);
    const nGroups = 6 + cs.extra.length;
    out[id] = {};
    for (const h of CELLS) {
      const fuel = id === "f" ? SOOTY : F.METHANE;
      const sim = buildSim(F, g, h, { layers: cs.layers, fuel, options: OFF });
      const r = burnerRun(sim, groupOf, nGroups, STEPS, { on: false });
      const sh = r.groups.map((J) => J / r.radiant);
      out[id][h] = { shares: sh, rest: r.rest / r.radiant, ms: r.ms };
      reported("off", `(${id}) ${h} m, item off: ${cs.groups.map((n0, q) => `${n0} ${f4(sh[q])}`).join(", ")}; the gas${id === "e" ? " and the open side" : ""} ${f4(r.rest / r.radiant)}; burnerStep ${(r.ms / STEPS).toFixed(4)} ms a call`);
    }
  }
  report.checks.off = out;
}

/**
 * Beside (h)1, not pre-registered and judging nothing: (c)'s flame with a
 * partition one cell thick at inner x = 9 across the whole section, of a
 * 12.7 mm gypsum board with its layer (one-dimensional, wallLayers 1), so
 * that the wall-end arrays carry deposits. The partition and the walls are
 * read against the 9 x 10 x 14 box the flame sees.
 */
function wallEnds() {
  const cs = CASES.c;
  const ip = 9;
  console.log(`\n(h) beside, not pre-registered: (c)'s flame and a one-dimensional gypsum partition at inner x = ${ip}`);
  const g = box(cs.n);
  const groupOf = new Array(g.N);
  for (let c = 0; c < g.N; c += 1) {
    const [i, j, k] = coords(g, c);
    if (g.inner(i, j, k)) {
      if (i - 2 === ip) groupOf[c] = [6];
    } else groupOf[c] = [g.wallOf(i, j, k)];
  }
  const ref = shares([ip, 10, 14], cs.layers.flat(), [], 24);
  const GYPSUM = { thickness: 0.0127, k: 0.48, rho: 1440, c: 840 };
  const N = STEPS * RAYS;
  const out = {};
  for (const h of CELLS) {
    const board = { name: "gypsum", rho: (GYPSUM.rho * GYPSUM.thickness) / h, c: GYPSUM.c, k: GYPSUM.k, eps: 0.9, layers: [GYPSUM] };
    const sim = buildSim(F, g, h, { layers: cs.layers, options: { ...ON, wallLayers: 1, inert: [F.INERT[0], board, F.INERT[2]] }, extra: (i) => (i === ip ? 1 : -1) });
    const walls = sim.wall ? sim.wall.cells.length : 0;
    const r = burnerRun(sim, groupOf, 7, STEPS, { on: true });
    const sh = r.groups.map((J) => J / r.radiant);
    const named = [[0, "x-low"], [2, "floor"], [3, "ceiling"], [4, "z-low"], [5, "z-high"], [6, "the partition (wall ends)"]];
    const within = named.map(([q, name]) => {
      const F0 = ref[q === 6 ? 1 : q];
      return `${name} ${f4(sh[q])} against ${f6(F0)} +/- ${f4(tol4(F0, N))}${Math.abs(sh[q] - F0) <= tol4(F0, N) ? "" : " (outside)"}`;
    });
    reported("h", `${h} m, ${walls} one-dimensional wall cells: ${within.join("; ")}; x-high ${sh[1]}; the gas ${r.gas} J; the worst step's imbalance ${ex(r.worst)}`);
    out[h] = { walls, shares: sh, worst: r.worst, ref };
  }
  report.checks.wallEnds = out;
}

/**
 * Beside (h)1, not pre-registered and judging nothing: a step in which no
 * cell burns, yet E - (E / H) H rounds above 0, so the rays carry that
 * residue (the correction of 2026-11-15: they leave the axis cell). (b)'s
 * scene on 0.20 m with no oxygen in any gas cell, its table held at the
 * first whole number of W from 100 kW up whose step leaves such a residue.
 */
function noOxygen() {
  const H = F.METHANE.heatOfCombustion;
  let q = Q;
  while (!(q * DT - ((q * DT) / H) * H > 0) && q < Q + 1e5) q += 1;
  const cs = CASES.b;
  const g = box(cs.n);
  const sim = buildSim(F, g, 0.2, { layers: cs.layers, q, options: ON });
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] === F.GAS) sim.Y[c * F.NS + F.SPECIES.O2] = 0;
  const caps = new Float64Array(sim.N);
  for (let c = 0; c < sim.N; c += 1) caps[c] = Math.max(solidCap(sim, c), 1e-6);
  F.burnerStep(sim, DT);
  const b = sim.burner, w = sim.work;
  let taken = 0;
  for (let c = 0; c < sim.N; c += 1) taken += w.burnerGas[c] * gasCap(sim, c) + w.burnerSolid[c] * caps[c];
  const e = b.radiant;
  reported("h", `no oxygen, ${q} W held on 0.2 m: ${b.flame.n} cells burnt, released ${ex(b.released)} J by rounding, ${sim.stats.burnerRays} rays from the axis cell carrying ${ex(e)} J; deposits plus burnerOut ${ex(taken + w.burnerOut[0])} J (|difference| / radiant ${ex(e > 0 ? Math.abs(e - taken - w.burnerOut[0]) / e : 0)})`);
}

// ------------------------------------------------------------ (g)

function checkG() {
  console.log("\n(g) the cell drawn by its heat");
  const sum = Float64Array.from([1, 3, 6, 10]);
  const counts = [0, 0, 0, 0];
  let stream = 0;
  const draw = () => {
    stream = F.pcgHash(stream);
    return (stream >>> 8) / 16777216;
  };
  const steps = 62500;
  for (let s = 0; s < steps; s += 1) {
    stream = F.cellStream(0, s, 4, 1);
    for (let r = 0; r < RAYS; r += 1) {
      counts[F.burnerFlameCell(sum, 4, draw())] += 1;
      for (let q = 0; q < 5; q += 1) draw();
    }
  }
  const N = steps * RAYS;
  [0.1, 0.2, 0.3, 0.4].forEach((p, q) => {
    const sd = Math.sqrt(N * p * (1 - p));
    verdict("g", Math.abs(counts[q] - N * p) <= 4 * sd, `mass ${q + 1}: ${counts[q]} draws against ${N * p} +/- ${(4 * sd).toFixed(0)} (4 sd of ${N})`);
  });
  // Edges of the search, reported: u = 0 and the largest u, and ties at the boundaries.
  reported("g", `burnerFlameCell at u = 0: ${F.burnerFlameCell(sum, 4, 0)}; at u = 0.1: ${F.burnerFlameCell(sum, 4, 0.1)} (sum[0] = 1 = 0.1 x 10, not above it); at u = 1 - 2^-24: ${F.burnerFlameCell(sum, 4, 1 - 2 ** -24)}; one cell: ${F.burnerFlameCell(Float64Array.of(2), 1, 0.7)}`);
  report.checks.g = { counts, draws: N };
}

// ------------------------------------------------------------ (h)2

function checkH2() {
  console.log("\n(h)2 one burnerStep on and off from one state");
  for (const id of ["a", "b", "c", "e"]) {
    const cs = CASES[id];
    const g = box(cs.n, id === "e");
    for (const h of CELLS) {
      const on = buildSim(F, g, h, { layers: cs.layers, options: ON });
      const off = buildSim(F, g, h, { layers: cs.layers, options: OFF });
      F.burnerStep(on, DT);
      F.burnerStep(off, DT);
      let diff = 0, gasCells = 0;
      for (let c = 0; c < on.N; c += 1) {
        if (on.kind[c] !== F.GAS) continue;
        gasCells += 1;
        if (!Object.is(on.T[c], off.T[c]) || !Object.is(on.rho[c], off.rho[c])) diff += 1;
        else for (let s = 0; s < F.NS; s += 1) if (!Object.is(on.Y[c * F.NS + s], off.Y[c * F.NS + s])) diff += 1;
      }
      const keys = ["released", "convective", "radiant"].filter((k) => !Object.is(on.burner[k], off.burner[k]));
      if (!Object.is(on.stats.released, off.stats.released)) keys.push("stats.released");
      verdict("h", diff === 0 && keys.length === 0, `(h)2 (${id}) ${h} m: ${diff} of ${gasCells} gas cells differ in T, Y or rho; ${keys.length ? `differing: ${keys.join(", ")}` : "released, convective, radiant and stats.released equal"} (radiant ${on.burner.radiant} J)`);
    }
  }
}

// ------------------------------------------------------------ (i)

/** The base's src/core at rev, from git, in a temporary directory (as the suite's snapshotCore). */
function baseCore(rev) {
  const dir = mkdtempSync(path.join(os.tmpdir(), "burner-rays-base-"));
  const core = path.join(dir, "topologicpy-web-frontend", "src", "core");
  mkdirSync(core, { recursive: true });
  const files = execFileSync("git", ["ls-tree", "--name-only", rev, "topologicpy-web-frontend/src/core/"], { cwd: ROOT, encoding: "utf8" }).split("\n").map((p) => p.split("/").pop()).filter((f) => f.endsWith(".js"));
  for (const f of files) writeFileSync(path.join(core, f), execFileSync("git", ["show", `${rev}:topologicpy-web-frontend/src/core/${f}`], { cwd: ROOT, maxBuffer: 1 << 28 }));
  writeFileSync(path.join(dir, "package.json"), JSON.stringify({ type: "module" }));
  const resolved = execFileSync("git", ["rev-parse", rev], { cwd: ROOT, encoding: "utf8" }).trim();
  return { dir, core, rev: resolved };
}

/** Every field of a solver: its typed arrays (bit for bit), the particles', the walls', the burner's and every stats entry. */
function differences(a, b) {
  const diffs = [];
  const sameArray = (x, y) => {
    if (!x || !y) return x === y;
    if (x.constructor !== y.constructor || x.length !== y.length) return false;
    const bx = new Uint8Array(x.buffer, x.byteOffset, x.byteLength), by = new Uint8Array(y.buffer, y.byteOffset, y.byteLength);
    for (let q = 0; q < bx.length; q += 1) if (bx[q] !== by[q]) return false;
    return true;
  };
  const same = (x, y) => {
    if (ArrayBuffer.isView(x) || ArrayBuffer.isView(y)) return sameArray(x, y);
    if (x && y && typeof x === "object" && typeof y === "object") {
      const keys = new Set([...Object.keys(x), ...Object.keys(y)]);
      for (const k of keys) if (!same(x[k], y[k])) return false;
      return true;
    }
    return Object.is(x, y);
  };
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    if (["P", "work", "random", "profile", "grid", "fuels", "stoich", "pilot", "ducts", "suppression", "systems", "zones", "stats", "burner", "parts", "wall"].includes(k)) continue;
    if (!same(a[k], b[k])) diffs.push(k);
  }
  for (const k of ["kind", "x", "vel", "T", "m", "age", "life"]) if (!same(a.parts[k], b.parts[k])) diffs.push(`parts.${k}`);
  if (a.parts.n !== b.parts.n) diffs.push("parts.n");
  if (!same(a.wall, b.wall)) diffs.push("wall");
  if (!same(a.zones, b.zones)) diffs.push("zones");
  for (const k of new Set([...Object.keys(a.burner || {}), ...Object.keys(b.burner || {})])) if (k !== "flame" && !same(a.burner[k], b.burner[k])) diffs.push(`burner.${k}`);
  for (const k of new Set([...Object.keys(a.stats), ...Object.keys(b.stats)])) if (!same(a.stats[k], b.stats[k])) diffs.push(`stats.${k}`);
  return diffs;
}

async function checkI() {
  console.log(`\n(i) off is off: against the base at ${BASE_REV}`);
  const base = baseCore(BASE_REV);
  try {
    const B = await import(pathToFileURL(path.join(base.core, "flamescapes.js")).href);
    report.checks.i = { baseRev: base.rev };
    const g = box([8, 8, 8]);
    const layers = [[[3, 0, 3]]];
    const run = (Fm, fuel, options) => {
      const sim = buildSim(Fm, g, 0.2, { layers, fuel, chi: 0.2, rays: 16, q: 2e4, options });
      for (let s = 0; s < 300; s += 1) Fm.step(sim, 0.1);
      return sim;
    };
    for (const [name, fuel] of [["METHANE", F.METHANE], ["a sooty fuel (yields of 13.B's inert fuel; beside, not pre-registered)", INERT_FUEL]]) {
      const b0 = run(B, fuel, {});
      const off = run(F, fuel, OFF);
      const d0 = differences(b0, off);
      const isPlan = name === "METHANE";
      const text = `8 x 8 x 8 box, 0.2 m, ${name}, 300 steps with burnerFlameRays 0 against the base: ${d0.length ? `differ in ${d0.join(", ")}` : "every field and every stats entry bit for bit the same"} (released ${(off.stats.released / 1e6).toFixed(4)} MJ, maxT ${off.stats.maxT.toFixed(1)} K, ${Object.keys(off.stats).length} stats entries)`;
      if (isPlan) verdict("i", d0.length === 0, text);
      else reported("i", `${d0.length ? "DIFFERS" : "equal"}: ${text}`);
      const on = run(F, fuel, ON);
      const d1 = differences(b0, on);
      reported("i", `${name}, with burnerFlameRays 1: differs from the base in ${d1.length ? d1.join(", ") : "nothing"}; burnerRadiatedOut ${on.stats.burnerRadiatedOut} J, ${on.stats.burnerRays} rays, ${(on.stats.burnerCrossings / on.stats.burnerRays).toFixed(2)} crossings a ray; radiant ${(on.burner.radiant / 1e3).toFixed(1)} kJ`);
    }
    // The GPU's own test, read from its source, on scenes with the item on.
    const gpuSrc = readFileSync(path.join(CORE, "flamescapesGpu.js"), "utf8");
    const m = gpuSrc.match(/function supported\(sim\) \{([\s\S]*?)\n\}/);
    const baseGpu = execFileSync("git", ["show", `${BASE_REV}:topologicpy-web-frontend/src/core/flamescapesGpu.js`], { cwd: ROOT, maxBuffer: 1 << 28 }).toString("utf8");
    reported("i", `flamescapesGpu.js is ${baseGpu.split("\r\n").join("\n") === gpuSrc.split("\r\n").join("\n") ? "the base's, unchanged" : "CHANGED from the base"}`);
    if (!m) verdict("i", false, "supported() not found in flamescapesGpu.js");
    else {
      const supported = new Function("sim", m[1]);
      const val = buildSim(F, g, 0.2, { layers, chi: 0.2, rays: 16, q: 2e4, options: ON });
      verdict("i", supported(val) === false, `supported() with the item on, a validation burner: ${supported(val)} (false)`);
      const fp = await designFire();
      verdict("i", supported(fp.onSim) === true, `supported() with the item on, the fingerprint tool's design fire: ${supported(fp.onSim)} (true)`);
      verdict("i", fp.on === fp.off, `the fingerprint tool's design-fire room, 300 steps on the CPU: ${fp.off} off, ${fp.on} on (equal); the burner's radiant fraction ${fp.onSim.burner.radiantFraction}, rays ${fp.onSim.burner.dirs.length / 3}, item record ${fp.onSim.burner.flame ? "made" : "never made"}`);
      report.checks.i.designFire = { off: fp.off, on: fp.on };
    }
  } finally {
    rmSync(base.dir, { recursive: true, force: true });
  }
}

/** The fingerprint tool's "room design fire" case (tools/flamescapes-fingerprint.mjs), 300 steps, off and on. */
async function designFire() {
  const R = await imp("flamescapesRun.js");
  const FC = await imp("fireClasses.js");
  const { fingerprint } = await imp("runCache.js");
  const boxMesh = (x0, y0, z0, x1, y1, z1) => {
    const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
    const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
    return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material: V.MATERIAL.STONE };
  };
  const Wd = 5, D = 4, H = 2.7, t = 0.2;
  const geometry = {
    floors: [boxMesh(-t, -t, -t, Wd + t, 0, D + t)],
    roofs: [boxMesh(-t, H, -t, Wd + t, H + t, D + t)],
    walls: [boxMesh(-t, 0, -t, Wd + t, H, 0), boxMesh(-t, 0, D, Wd + t, H, D + t), boxMesh(-t, 0, 0, 0, H, D), boxMesh(Wd, 0, 0, Wd + t, H, 1.5), boxMesh(Wd, 0, 2.4, Wd + t, H, D), boxMesh(Wd, 2.0, 1.5, Wd + t, H, 2.4)],
    stairs: [], obstacles: [],
  };
  const { contentsFuel, ignitionFuel, item } = FC.fuelsFor("A");
  const one = (options) => {
    const scene = R.buildScene(geometry, { upAxis: "y", fire: [2.5, 0, 2.0], maxCells: 30000, contentsFuel, ignitionFuel, item, designFire: FC.designFireFor("A", "class"), fireLoad: 780e6, seed: 1 });
    const sim = F.createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, embers: false, seed: 1, ...options });
    for (let s = 0; s < 300; s += 1) F.step(sim, 0.1);
    return { sim, fp: fingerprint([sim.T, sim.Ts, sim.Y, sim.u, sim.v, sim.w, sim.rho, sim.p, sim.fuelMass, sim.moisture, sim.parts.x.subarray(0, sim.parts.n * 3), Float64Array.of(sim.stats.released, sim.stats.radiated, sim.parts.n)]) };
  };
  const off = one({});
  const on = one(ON);
  return { off: off.fp, on: on.fp, onSim: on.sim };
}

// ------------------------------------------------------------ run

const t0 = performance.now();
console.log(`Item 12 checks: ${STEPS} burner steps of ${DT} s a case, ${RAYS} rays, radiant fraction ${CHI}, ${Q / 1e3} kW held, seed 1, cells ${CELLS.join(", ")} m`);
const refs = references();
for (const [id, r] of Object.entries(refs)) {
  if (id === "e") continue;
  reported("ref", `(${id}) quadrature, 24 points a side: ${r.r24.map(f6).join(", ")}; 16 points differ by at most ${ex(Math.max(...r.r24.map((x, q) => Math.abs(x - r.r16[q]))))}; the six walls sum to 1 within ${ex(Math.abs(r.sum - 1))}`);
}
if (["a", "b", "c", "d", "e", "f"].some((x) => ONLY.has(x))) {
  runShares(refs);
  runOffShares();
  if (ONLY.has("c") && ONLY.has("k")) {
    for (const h of CELLS) {
      const on = report.checks.shares.c[h], off = report.checks.off.c[h];
      reported("k", `(c)'s scene on ${h} m, burnerStep alone over ${STEPS} calls: ${(on.ms / STEPS).toFixed(4)} ms a call on, ${(off.ms / STEPS).toFixed(4)} ms off; ${(on.crossings / on.rays).toFixed(2)} crossings a ray on`);
    }
  }
}
if (ONLY.has("g")) checkG();
if (ONLY.has("h")) {
  checkH2();
  wallEnds();
  noOxygen();
}
if (ONLY.has("i")) await checkI();
report.verdicts = verdicts;
report.lines = lines;
report.seconds = (performance.now() - t0) / 1000;
const counts = { pass: lines.filter((l) => l.includes(") PASS:")).length, fail: lines.filter((l) => l.includes(") FAIL:")).length, reported: lines.filter((l) => l.includes(") REPORT:")).length };
report.counts = counts;
console.log(`\nVerdicts (16.10's checks; (i)'s fingerprints, unit suite and GPU check, (j) and (k)'s suite runs run apart): ${counts.pass} PASS, ${counts.fail} FAIL, ${counts.reported} reported`);
for (const [k, v] of Object.entries(verdicts)) console.log(`  (${k}) ${v ? "PASS" : "FAIL"}`);
console.log(`${report.seconds.toFixed(1)} s`);
const json = flag("--json", null);
if (json) await writeFile(json, JSON.stringify(report, null, 1));
process.exitCode = counts.fail ? 1 : 0;
