#!/usr/bin/env node
/**
 * The checks of physics item 14, a flame radiates its radiant fraction once
 * (wiki/research/fire-calibration-plan-2026-11-11.md, 16.14, "Checks", read
 * with the coordinator's decisions in 16.15; src/core/flamescapes.js
 * flameRadiantOnce, flameOnce, burnInStamp, axisRaysClear, burnerMarch,
 * flameOfStep). Every reference is a closed form: an equality bit for bit,
 * one formula in float64, or an energy balance within the float32 rounding
 * this tool computes from the stored state. No measured datum enters any.
 *
 *   node tools/flame-radiant-once-check.mjs [--only a,b,c,d,e,f,g,i]
 *        [--configs once,both] [--cells 0.2,0.34] [--steps 200]
 *        [--base-rev 1dd890a] [--json out.json]
 *
 * Each check runs with item 14 alone (configuration "once": flameRadiantOnce
 * 1, its reference off: nothing set) and with items 12 and 14 together
 * ("both": flameRadiantOnce 1 and burnerFlameRays 1, its reference off:
 * burnerFlameRays 1, that is item 12 alone).
 *
 * The box (16.14): 12 x 10 x 14 inner cells, y up, walls two cells thick of
 * stone (INERT[0], lumped), as tools/burner-rays-check.mjs builds it. Cells
 * of 0.20 m and 0.34 m, dt 0.1 s, seed 1. The flame: 16.10 (c)'s 12 cells,
 * i in {5, 6}, j in {0, 1, 2}, k in {6, 7}, built by createBurner from
 * `layers` with no `diameter` (equal shares); radiant fraction 0.3, 16 rays,
 * the table held at 100 kW (10 kJ a step, 3.0 kJ radiant). Fuels: F1, METHANE
 * with a soot yield of 0.1; F2, class A's fuel (fireClasses.js: 18.0 MJ/kg,
 * soot 0.015); F1_0 and F2_0, the same with soot yield 0. The state before
 * each step, set here and restored after it: every gas cell at T_a, air, no
 * soot, at rest; the 12 flame cells at T_f in {800, 1200} K with Y_s in
 * {1e-3, 1e-2}, O2 0.232 (1 - Y_s), nitrogen the rest, rho by gasDensity;
 * every solid at T_a. A step: burnerStep, radiationStage, the readings, the
 * restore; sim.steps rises by 1. 200 steps a case, 16 cases.
 *
 * Readings (float64 from the stored state): H_f = sum over the flame cells
 * of (rho V) c_p (T - T_a), burnInCell's order of operations; H_g the same
 * over the other gas cells; H_s = sum of C_s (T_s - T_a) over the solids,
 * C_s the solver's solid capacity. b.released, b.radiant, b.rayFate,
 * stats.radiated, flameSilenced, flameCleared and burnerRadiatedOut.
 * u(T) is the float32 spacing at T.
 *
 * (a) The silence: 1. the stamped cells are the 12 flame cells and no other;
 *     2. radiationStage leaves the flame cells' T, Y and rho, every solid's
 *     T_s and stats.radiated as they were after burnerStep; 3. flameSilenced
 *     rises by the sum over the flame cells above 500 K of min(4 kappa_s Y_s
 *     sigma (T^4 - T_a^4) V dt, 1/2 rho c_p V (T - T_a)) within 1e-12 of it,
 *     and from the same state with the item off stats.radiated rises by it
 *     within 1e-12.
 * (b) The flame's radiant output, burnt - (H_f after radiationStage - H_f
 *     before burnerStep), is chi_r burnt within the sum over the flame cells
 *     of 1/2 u(T) rho c_p V plus 1e-12 burnt; b.rayFate holds exactly 0 J in
 *     stamped cells and its other parts sum to chi_r burnt within 1e-9 of it.
 *     Reported beside, with the item off: output - chi_r burnt and its parts.
 * (c) The flame is clear: with F1 or F2 and the item on, every solid's T_s
 *     after burnerStep is bit for bit that of the same case with the item
 *     off, F1_0 or F2_0, and Y_s 0 in the flame cells.
 * (d) Energy closes: 1. every step of every case, on and off, burnt = dH_f +
 *     dH_g + dH_s + burnerRadiatedOut's rise within the sum of k 1/2 u(T) C
 *     over the cells written in the step (k 3 for a gas cell, 17 for a
 *     solid); 2. the box with its x-high side open (16.10 (e)'s), F1, 1200 K,
 *     1e-2, both cells, item on: burnerRadiatedOut above 0 and 1 holds.
 * (e) 1. The flame with plain METHANE at the floor's centre of 13.19 (f)'s
 *     box (3.0 x 2.4 x 3.0 m, a door 0.8 x 2.0 m to open cells beyond),
 *     0.20 m cells, 300 full steps, on and off: every field and every stats
 *     entry bit for bit, the new tallies aside, which read 0. 2. The app's
 *     design fire (radiant fraction 0): the fingerprint tool's design-fire
 *     room, 300 steps on the CPU, on and off; supported() on and off.
 * (f) (e)1's scene with F2, 120 s, item on. In every step: no stamped cell
 *     casts a ray or has its T lowered in the radiation stage (its T after
 *     the stage is its T before plus what other cells' rays left in it, as
 *     the stage adds deposits); flameSilenced rises by the stamped cells'
 *     would-be emission within 1e-12; b.rayFate holds 0 in stamped cells
 *     unless a ray ended in one (counted); its parts sum to chi_r burnt
 *     within 1e-9. Reported: flameSilenced and flameCleared as shares of
 *     chi_r Q over the run, and 13.15's ledger on and off.
 * (g) Off is off (the part run here): 16.10 (i)'s validation burner (radiant
 *     fraction 0.2, 16 rays, one cell on the floor of an 8 x 8 x 8 box on
 *     0.20 m cells), fuel F2, 300 full steps, against the base's solver
 *     (--base-rev, read from git): with flameRadiantOnce 0, and with item 12
 *     on and this item off against item 12 alone. supported() against the
 *     base's on the same scenes.
 * (i) Speed, reported: radiation's and burnerStep's (the pilot stage's) ms a
 *     step on and off over (f)'s run.
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const CORE = path.join(ROOT, "topologicpy-web-frontend", "src", "core");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
};
const imp = (f) => import(pathToFileURL(path.join(CORE, f)).href);
const F = await imp("flamescapes.js");
const V = await imp("voxelize.js");
const FC = await imp("fireClasses.js");
if (!("flameRadiantOnce" in F.DEFAULTS)) throw new Error("this solver has no flameRadiantOnce (item 14 is not built)");

const ONLY = new Set(flag("--only", "a,b,c,d,e,f,g,i").split(","));
const CONFIG_IDS = flag("--configs", "once,both").split(",");
const CELLS = flag("--cells", "0.2,0.34").split(",").map(Number);
const STEPS = Number(flag("--steps", 200));
// (f)'s run, 120 s by 16.14; shorter only to try the tool.
const F_SECONDS = Number(flag("--f-seconds", 120));
const BASE_REV = flag("--base-rev", "1dd890a");
const DT = 0.1;
const RAYS = 16;
const CHI = 0.3;
const Q = 1e5;
const TA = F.DEFAULTS.ambient;
const M = V.MATERIALS;
const NS = F.NS;
const SOOT = F.SPECIES.SOOT, O2 = F.SPECIES.O2;
const CONFIGS = {
  once: { label: "item 14 alone", on: { flameRadiantOnce: 1 }, off: {} },
  both: { label: "items 12 and 14", on: { flameRadiantOnce: 1, burnerFlameRays: 1 }, off: { burnerFlameRays: 1 } },
};
const F1 = { heatOfCombustion: F.METHANE.heatOfCombustion, yields: { co2: F.METHANE.yields.co2, co: 0, soot: 0.1 } };
const F10 = { heatOfCombustion: F.METHANE.heatOfCombustion, yields: { co2: F.METHANE.yields.co2, co: 0, soot: 0 } };
const FA = FC.FIRE_CLASSES.A.fuel;
const F2 = { heatOfCombustion: FA.heatOfCombustion, yields: { ...FA.yields } };
const F20 = { heatOfCombustion: FA.heatOfCombustion, yields: { ...FA.yields, soot: 0 } };
const FUELS = { F1: [F1, F10], F2: [F2, F20] };
const INERT_FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };
const FLAME = [0, 1, 2].map((j) => [[5, j, 6], [6, j, 6], [5, j, 7], [6, j, 7]]);

const verdicts = {};
const lines = [];
const report = { tool: "tools/flame-radiant-once-check.mjs", plan: "wiki/research/fire-calibration-plan-2026-11-11.md 16.14 and 16.15", dt: DT, seed: 1, steps: STEPS, rays: RAYS, radiantFraction: CHI, hrrW: Q, cells: CELLS, configs: CONFIG_IDS, checks: {} };
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
const ex = (x) => (Number.isFinite(x) ? x.toExponential(2) : String(x));
const f1 = (x) => (Number.isFinite(x) ? x.toFixed(1) : String(x));
const f4 = (x) => (Number.isFinite(x) ? x.toFixed(4) : String(x));

/** The float32 spacing at x (an upper bound at a power of two). */
function ulp32(x) {
  const a = Math.abs(Math.fround(x));
  if (!(a > 0)) return 2 ** -149;
  return 2 ** (Math.floor(Math.log2(a)) - 23);
}

// ------------------------------------------------------------ scenes

/** A box of inner cells n in walls two cells thick; with openHigh, no wall on its x-high side (the grid's side). */
function box(n, openHigh = false) {
  const [ni, nj, nk] = n;
  const nx = ni + (openHigh ? 2 : 4), ny = nj + 4, nz = nk + 4;
  const at = (i, j, k) => i + nx * (j + ny * k);
  const open = (i, j, k) => i >= 2 && (openHigh || i < ni + 2) && j >= 2 && j < nj + 2 && k >= 2 && k < nk + 2;
  return { nx, ny, nz, N: nx * ny * nz, at, open, cell: (i, j, k) => at(i + 2, j + 2, k + 2), name: `${ni} x ${nj} x ${nk} box${openHigh ? ", its x-high side open" : ""}` };
}

/**
 * 13.19 (f)'s box: a room of 15 x 12 x 15 inner cells (3.0 x 2.4 x 3.0 m on
 * 0.20 m cells) in walls two cells thick, a door 4 cells wide and 10 high
 * (0.8 x 2.0 m) at the floor in its x-high wall, centred along z as nearly
 * as whole cells allow (inner k 5 to 8), and beyond it 6 columns of open
 * cells over the floor reaching the grid's x-high, z and top sides.
 */
function room() {
  const ni = 15, nj = 12, nk = 15, B = 6;
  const nx = ni + 4 + B, ny = nj + 4, nz = nk + 4;
  const at = (i, j, k) => i + nx * (j + ny * k);
  const open = (i, j, k) => {
    if (j < 2) return false;
    if (i >= ni + 4) return true;
    if (i >= 2 && i < ni + 2 && j < nj + 2 && k >= 2 && k < nk + 2) return true;
    // The door through the x-high wall.
    return i >= ni + 2 && i < ni + 4 && j < 2 + 10 && k >= 2 + 5 && k < 2 + 9;
  };
  return { nx, ny, nz, N: nx * ny * nz, at, open, cell: (i, j, k) => at(i + 2, j + 2, k + 2), name: "13.19 (f)'s room, 15 x 12 x 15 inner cells, a door 4 x 10 cells to 6 open columns" };
}
// The flame at the room's floor centre: its 15 cells have no middle pair, so columns 6 and 7 (centre 7.0 against 7.5).
const ROOM_FLAME = [0, 1, 2].map((j) => [[6, j, 6], [7, j, 6], [6, j, 7], [7, j, 7]]);

const coords = (g, c) => [c % g.nx, Math.floor(c / g.nx) % g.ny, Math.floor(c / (g.nx * g.ny))];

/** A solver over the grid with a burner: `layers` of inner cells, its table held at q W, radiant fraction chi and `rays` rays. */
function buildSim(Fm, g, h, { layers, fuel, chi = CHI, rays = RAYS, q = Q, options = {} }) {
  const N = g.N;
  const grid = { nx: g.nx, ny: g.ny, nz: g.nz, h, origin: [0, 0, 0], axes: [0, 1, 2], upAxis: "y", cells: N };
  const fraction = new Float32Array(N * M);
  const solid = new Uint8Array(N);
  for (let c = 0; c < N; c += 1) {
    const [i, j, k] = coords(g, c);
    if (!g.open(i, j, k)) {
      solid[c] = 1;
      fraction[c * M] = 1;
    }
  }
  const burner = { layers: layers.map((l) => l.map(([i, j, k]) => g.cell(i, j, k))), hrr: [[0, q]], radiantFraction: chi, rays, fuel };
  const scene = { grid, fraction, solid, contents: { cells: [], masses: [] }, ignition: null, pilot: null, burner };
  return Fm.createSolver(scene, { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL, seed: 1, dt: DT, ...options });
}

const solidCap = (sim, c) => sim.inertC[c] + F.skinMass(sim, c) * (sim.fuels[sim.fuelKind[c]]?.heatCapacity ?? sim.P.fuelHeatCapacity) + sim.moisture[c] * 4186;
const cellH = (sim, c) => sim.rho[c] * sim.V * F.mixtureCp(sim.Y, c * NS) * (sim.T[c] - TA);
const gasC = (sim, c) => sim.rho[c] * F.mixtureCp(sim.Y, c * NS) * sim.V;

/** The flame cells at T_f with soot Y_s, O2 0.232 (1 - Y_s), nitrogen the rest, rho by gasDensity; everything else as built (ambient air, solids at T_a). */
function setFlame(sim, cells, Tf, Ys) {
  for (const c of cells) {
    for (let s = 0; s < NS; s += 1) sim.Y[c * NS + s] = 0;
    sim.Y[c * NS + O2] = 0.232 * (1 - Ys);
    sim.Y[c * NS + SOOT] = Ys;
    sim.T[c] = Tf;
    sim.rho[c] = F.gasDensity(sim.T[c], sim.Y, c * NS);
  }
}

const snap = (sim) => ({ T: sim.T.slice(), Ts: sim.Ts.slice(), Y: sim.Y.slice(), rho: sim.rho.slice() });
function restore(sim, s) {
  sim.T.set(s.T);
  sim.Ts.set(s.Ts);
  sim.Y.set(s.Y);
  sim.rho.set(s.rho);
}

/** The would-be Eq. 17 emission of the given gas cells from the stored state, in float64, as the plan writes it. */
function eq17(sim, cells, dt) {
  const P = sim.P;
  let sum = 0;
  for (const c of cells) {
    const T = sim.T[c], ys = sim.Y[c * NS + SOOT];
    if (!(T > P.emitAbove) || !(ys > 0)) continue;
    const e = 4 * P.sootAbsorption * ys * F.SIGMA * (T ** 4 - TA ** 4) * sim.V * dt;
    const cap = 0.5 * sim.rho[c] * F.mixtureCp(sim.Y, c * NS) * sim.V * (T - TA);
    const em = Math.min(e, cap);
    if (em > 0) sum += em;
  }
  return sum;
}

const relDiff = (x, ref) => (ref === 0 ? Math.abs(x) : Math.abs(x - ref) / Math.abs(ref));

// ------------------------------------------------------------ (a) to (d) in the box

/**
 * One step from the set state: burnerStep then radiationStage, with every
 * reading, then the state restored. Returns the readings.
 */
function boxStep(sim, base, flameSet, flameCells, { twin = null } = {}) {
  const b = sim.burner;
  const st = sim.stats;
  const rel0 = b.released, rad0 = b.radiant, conv0 = b.convective;
  const radiated0 = st.radiated, silenced0 = st.flameSilenced ?? 0, cleared0 = st.flameCleared ?? 0, out0 = st.burnerRadiatedOut ?? 0;
  F.burnerStep(sim, DT);
  const afterBurner = snap(sim);
  const burnt = b.released - rel0;
  const radiant = b.radiant - rad0;
  const convective = b.convective - conv0;
  const fate = b.rayFate ? { ...b.rayFate } : null;
  // (a)1: the stamped cells.
  let stampOk = null;
  const rec = b.flame;
  if (rec && rec.step === sim.steps) {
    let stamped = 0, wrong = 0;
    for (let c = 0; c < sim.N; c += 1) {
      if (rec.stamp[c] !== rec.mark) continue;
      stamped += 1;
      if (!flameSet.has(c)) wrong += 1;
    }
    stampOk = { stamped, wrong, ok: stamped === flameCells.length && wrong === 0 };
  }
  const silencedRef = eq17(sim, flameCells, DT);
  // (a)3, the item off from the same state: the twin takes this state and runs the stage alone.
  let twinRadiated = null;
  if (twin) {
    restore(twin, afterBurner);
    const r0 = twin.stats.radiated;
    F.radiationStage(twin, DT);
    twinRadiated = twin.stats.radiated - r0;
    restore(twin, base);
  }
  F.radiationStage(sim, DT);
  const after = snap(sim);
  // (a)2: the flame cells, the solids and stats.radiated through the stage.
  let changed = 0;
  for (const c of flameCells) {
    if (!Object.is(after.T[c], afterBurner.T[c]) || !Object.is(after.rho[c], afterBurner.rho[c])) changed += 1;
    for (let s = 0; s < NS; s += 1) if (!Object.is(after.Y[c * NS + s], afterBurner.Y[c * NS + s])) changed += 1;
  }
  let solidsChanged = 0;
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] !== F.GAS && !Object.is(after.Ts[c], afterBurner.Ts[c])) solidsChanged += 1;
  // Energies, cell by cell, over the cells written in the step.
  let dHf = 0, dHg = 0, dHs = 0, bound = 0, boundFlame = 0;
  const Hof = (s, c) => s.rho[c] * sim.V * F.mixtureCp(s.Y, c * NS) * (s.T[c] - TA);
  for (let c = 0; c < sim.N; c += 1) {
    if (sim.kind[c] === F.GAS) {
      const isFlame = flameSet.has(c);
      const written = isFlame || !Object.is(base.T[c], after.T[c]) || !Object.is(base.T[c], afterBurner.T[c]) || !Object.is(base.rho[c], after.rho[c]);
      if (!written) continue;
      const d = Hof(after, c) - Hof(base, c);
      if (isFlame) dHf += d;
      else dHg += d;
      const C = after.rho[c] * F.mixtureCp(after.Y, c * NS) * sim.V;
      const u = ulp32(Math.max(base.T[c], afterBurner.T[c], after.T[c]));
      bound += 3 * 0.5 * u * C;
      if (isFlame) boundFlame += 0.5 * ulp32(after.T[c]) * C;
    } else {
      if (Object.is(base.Ts[c], after.Ts[c])) continue;
      const C = Math.max(solidCap(sim, c), 1e-6);
      dHs += C * (after.Ts[c] - base.Ts[c]);
      bound += 17 * 0.5 * ulp32(Math.max(base.Ts[c], after.Ts[c])) * C;
    }
  }
  // The flame's H before burnerStep and after the stage (b), each a sum over its cells in burnInCell's order.
  let HfBefore = 0, HfAfter = 0, HfBurner = 0;
  for (const c of flameCells) {
    HfBefore += Hof(base, c);
    HfAfter += Hof(after, c);
    HfBurner += Hof(afterBurner, c);
  }
  const out = (st.burnerRadiatedOut ?? 0) - out0;
  const r = {
    burnt, radiant, convective, fate, stampOk, changed, solidsChanged,
    radiatedRise: st.radiated - radiated0,
    silencedRise: (st.flameSilenced ?? 0) - silenced0,
    clearedRise: (st.flameCleared ?? 0) - cleared0,
    silencedRef, twinRadiated,
    output: burnt - (HfAfter - HfBefore),
    take: HfBurner - HfBefore - convective,
    boundFlame: boundFlame + 1e-12 * burnt,
    dHf, dHg, dHs, out, bound,
    TsAfterBurner: afterBurner.Ts,
  };
  restore(sim, base);
  sim.steps += 1;
  return r;
}

function caseLabel(h, fuelId, Tf, Ys) {
  return `${h} m, ${fuelId}, ${Tf} K, Y_s ${Ys}`;
}

function runBox(cfgId) {
  const cfg = CONFIGS[cfgId];
  const out = [];
  console.log(`\n(a) to (d): the box, ${cfg.label}, ${STEPS} steps a case`);
  for (const h of CELLS) {
    for (const fuelId of ["F1", "F2"]) {
      for (const Tf of [800, 1200]) {
        for (const Ys of [1e-3, 1e-2]) {
          const [fuel, fuel0] = FUELS[fuelId];
          const g = box([12, 10, 14]);
          const flameCells = FLAME.flat().map(([i, j, k]) => g.cell(i, j, k));
          const flameSet = new Set(flameCells);
          const on = buildSim(F, g, h, { layers: FLAME, fuel, options: cfg.on });
          const off = buildSim(F, g, h, { layers: FLAME, fuel, options: cfg.off });
          const ref = buildSim(F, g, h, { layers: FLAME, fuel: fuel0, options: cfg.off });
          const twin = buildSim(F, g, h, { layers: FLAME, fuel, options: cfg.off });
          setFlame(on, flameCells, Tf, Ys);
          setFlame(off, flameCells, Tf, Ys);
          setFlame(ref, flameCells, Tf, 0);
          setFlame(twin, flameCells, Tf, Ys);
          const bOn = snap(on), bOff = snap(off), bRef = snap(ref), bTwin = snap(twin);
          const label = caseLabel(h, fuelId, Tf, Ys);
          const agg = {
            stampBad: 0, stamped: [Infinity, -Infinity], changed: 0, solidsChanged: 0, radiatedMoved: 0,
            worstSil: 0, worstTwin: 0, silRef: 0,
            worstB: 0, worstBdev: 0, boundB: 0, stampedFate: 0, worstFate: 0,
            cDiff: 0, cSteps: 0,
            worstDon: 0, worstDonJ: 0, boundDon: Infinity, worstDoff: 0, worstDoffJ: 0, boundDoff: Infinity,
            offDev: 0, offE17: 0, offTake: 0, onSil: 0, onCleared: 0, burnt: 0, radiant: 0, outOn: 0,
          };
          for (let s = 0; s < STEPS; s += 1) {
            const r = boxStep(on, bOn, flameSet, flameCells, { twin });
            twin.steps += 1;
            // (a)
            if (!r.stampOk || !r.stampOk.ok) agg.stampBad += 1;
            if (r.stampOk) agg.stamped = [Math.min(agg.stamped[0], r.stampOk.stamped), Math.max(agg.stamped[1], r.stampOk.stamped)];
            agg.changed += r.changed;
            agg.solidsChanged += r.solidsChanged;
            if (r.radiatedRise !== 0) agg.radiatedMoved += 1;
            agg.worstSil = Math.max(agg.worstSil, relDiff(r.silencedRise, r.silencedRef));
            agg.worstTwin = Math.max(agg.worstTwin, relDiff(r.twinRadiated, r.silencedRef));
            agg.silRef += r.silencedRef;
            // (b)
            const dev = r.output - CHI * r.burnt;
            if (Math.abs(dev) / r.boundFlame > agg.worstB) {
              agg.worstB = Math.abs(dev) / r.boundFlame;
              agg.worstBdev = dev;
              agg.boundB = r.boundFlame;
            }
            if (agg.boundB === 0) agg.boundB = r.boundFlame;
            agg.stampedFate = Math.max(agg.stampedFate, Math.abs(r.fate.stamped));
            agg.worstFate = Math.max(agg.worstFate, relDiff(r.fate.gas + r.fate.solid + r.fate.out, CHI * r.burnt));
            agg.burnt += r.burnt;
            agg.radiant += r.radiant;
            agg.onSil += r.silencedRise;
            agg.onCleared += r.clearedRise;
            agg.outOn += r.out;
            // (d)1 on
            const imb = r.burnt - (r.dHf + r.dHg + r.dHs + r.out);
            if (Math.abs(imb) / r.bound > agg.worstDon) {
              agg.worstDon = Math.abs(imb) / r.bound;
              agg.worstDonJ = imb;
            }
            agg.boundDon = Math.min(agg.boundDon, r.bound);
            // The item off, the same case (reported (b), judged (d)1).
            const o = boxStep(off, bOff, flameSet, flameCells);
            const devOff = o.output - CHI * o.burnt;
            agg.offDev += devOff;
            agg.offE17 += o.radiatedRise;
            agg.offTake += o.take;
            const imbOff = o.burnt - (o.dHf + o.dHg + o.dHs + o.out);
            if (Math.abs(imbOff) / o.bound > agg.worstDoff) {
              agg.worstDoff = Math.abs(imbOff) / o.bound;
              agg.worstDoffJ = imbOff;
            }
            agg.boundDoff = Math.min(agg.boundDoff, o.bound);
            // (c): the reference, item off, the soot-free fuel and flame.
            const rf = boxStep(ref, bRef, flameSet, flameCells);
            agg.cSteps += 1;
            for (let c = 0; c < on.N; c += 1) if (on.kind[c] !== F.GAS && !Object.is(r.TsAfterBurner[c], rf.TsAfterBurner[c])) agg.cDiff += 1;
          }
          const n = STEPS;
          const tag = `${label}, ${cfg.label}`;
          verdict("a", agg.stampBad === 0, `(a)1 ${tag}: the stamped cells are the 12 flame cells and no other in ${n - agg.stampBad} of ${n} steps (${agg.stamped[0]} to ${agg.stamped[1]} stamped; equality)`);
          verdict("a", agg.changed === 0 && agg.solidsChanged === 0 && agg.radiatedMoved === 0, `(a)2 ${tag}: through radiationStage ${agg.changed} flame-cell values of T, Y and rho, ${agg.solidsChanged} solid T_s and ${agg.radiatedMoved} steps of stats.radiated moved (0, 0 and 0; equality)`);
          verdict("a", agg.worstSil <= 1e-12 && agg.worstTwin <= 1e-12, `(a)3 ${tag}: flameSilenced rises by the flame's Eq. 17 sum, ${f1(agg.silRef / n)} J a step, within ${ex(agg.worstSil)} of it; the item off from the same state, stats.radiated within ${ex(agg.worstTwin)} (each against 1e-12)`);
          verdict("b", agg.worstB <= 1, `(b) ${tag}: the flame's radiant output less chi_r burnt (${f1(CHI * agg.burnt / n)} J a step) at worst ${ex(agg.worstBdev)} J, ${f4(agg.worstB)} of its bound ${ex(agg.boundB)} J (the flame's 1/2 u(T) rho c_p V plus 1e-12 burnt)`);
          verdict("b", agg.stampedFate === 0 && agg.worstFate <= 1e-9, `(b) ${tag}: b.rayFate in stamped cells at most ${agg.stampedFate} J (exactly 0); gas, solid and out sum to chi_r burnt within ${ex(agg.worstFate)} of it (against 1e-9)`);
          reported("b", `${tag}, the item off: the flame's output less chi_r burnt ${f1(agg.offDev / n)} J a step; its Eq. 17 emission ${f1(agg.offE17 / n)} J, its soot's take of the burner's rays ${f1(agg.offTake / n)} J, so its own soot took back ${f1((agg.offE17 - agg.offTake - agg.offDev) / n)} J of its emission; on, flameSilenced ${f1(agg.onSil / n)} J and flameCleared ${f1(agg.onCleared / n)} J a step`);
          verdict("c", agg.cDiff === 0, `(c) ${tag}: every solid's T_s after burnerStep against the item off with ${fuelId}_0 and Y_s 0: ${agg.cDiff} differ over ${agg.cSteps} steps (0; bit for bit)`);
          verdict("d", agg.worstDon <= 1, `(d)1 ${tag}, on: the worst step's imbalance ${ex(agg.worstDonJ)} J, ${f4(agg.worstDon)} of its bound (the smallest bound ${f1(agg.boundDon)} J)`);
          verdict("d", agg.worstDoff <= 1, `(d)1 ${tag.replace(cfg.label, cfg.label + ", its reference off")}: the worst step's imbalance ${ex(agg.worstDoffJ)} J, ${f4(agg.worstDoff)} of its bound (the smallest bound ${f1(agg.boundDoff)} J)`);
          out.push({ h, fuel: fuelId, Tf, Ys, config: cfgId, perStep: { burnt: agg.burnt / n, radiant: agg.radiant / n, silenced: agg.onSil / n, cleared: agg.onCleared / n, offDev: agg.offDev / n, offE17: agg.offE17 / n, offTake: agg.offTake / n }, worst: { a3: agg.worstSil, a3off: agg.worstTwin, b: agg.worstB, bJ: agg.worstBdev, fate: agg.worstFate, dOn: agg.worstDon, dOff: agg.worstDoff }, cDiff: agg.cDiff });
        }
      }
    }
  }
  // (d)2: the open side.
  console.log(`\n(d)2 the open side, ${cfg.label}`);
  for (const h of CELLS) {
    const g = box([12, 10, 14], true);
    const flameCells = FLAME.flat().map(([i, j, k]) => g.cell(i, j, k));
    const flameSet = new Set(flameCells);
    const on = buildSim(F, g, h, { layers: FLAME, fuel: F1, options: cfg.on });
    setFlame(on, flameCells, 1200, 1e-2);
    const bOn = snap(on);
    let worst = 0, worstJ = 0, outSum = 0, radiant = 0;
    for (let s = 0; s < STEPS; s += 1) {
      const r = boxStep(on, bOn, flameSet, flameCells);
      const imb = r.burnt - (r.dHf + r.dHg + r.dHs + r.out);
      if (Math.abs(imb) / r.bound > worst) {
        worst = Math.abs(imb) / r.bound;
        worstJ = imb;
      }
      outSum += r.out;
      radiant += r.radiant;
    }
    verdict("d", outSum > 0 && worst <= 1, `(d)2 ${h} m, F1, 1200 K, Y_s 0.01, ${cfg.label}, the x-high side open: burnerRadiatedOut ${f1(outSum)} J over ${STEPS} steps, ${f4(outSum / radiant)} of the radiant output (above 0); the worst step's imbalance ${ex(worstJ)} J, ${f4(worst)} of its bound`);
  }
  return out;
}

// ------------------------------------------------------------ (e) and (f): the room with the flow

/** Every field of a solver: its typed arrays (bit for bit), the particles', the walls', the burner's and every stats entry, but `skip`. */
function differences(a, b, { skipStats = [], skipBurner = ["flame", "rayFate"] } = {}) {
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
  for (const k of new Set([...Object.keys(a.burner || {}), ...Object.keys(b.burner || {})])) if (!skipBurner.includes(k) && !same(a.burner[k], b.burner[k])) diffs.push(`burner.${k}`);
  for (const k of new Set([...Object.keys(a.stats), ...Object.keys(b.stats)])) if (!skipStats.includes(k) && !same(a.stats[k], b.stats[k])) diffs.push(`stats.${k}`);
  return diffs;
}

const NEW_TALLIES = ["flameSilenced", "flameCleared"];

function checkE1(cfgId) {
  const cfg = CONFIGS[cfgId];
  const g = room();
  console.log(`\n(e)1 ${g.name}, 0.2 m, plain METHANE, 300 full steps, ${cfg.label}`);
  const run = (options) => {
    const sim = buildSim(F, g, 0.2, { layers: ROOM_FLAME, fuel: F.METHANE, options });
    for (let s = 0; s < 300; s += 1) F.step(sim, DT);
    return sim;
  };
  const on = run(cfg.on), off = run(cfg.off);
  // burnerRadiatedOut is item 14's alone with item 12 off; with item 12 on both runs keep it and it must agree.
  const newOnes = cfgId === "once" ? [...NEW_TALLIES, "burnerRadiatedOut"] : NEW_TALLIES;
  const d = differences(on, off, { skipStats: newOnes });
  verdict("e", d.length === 0 && on.stats.flameSilenced === 0 && on.stats.flameCleared === 0, `(e)1 ${cfg.label}: on against off ${d.length ? `differ in ${d.join(", ")}` : "every field and every stats entry bit for bit the same"}, the new tallies aside; flameSilenced ${on.stats.flameSilenced} J and flameCleared ${on.stats.flameCleared} J (0 and 0); released ${(on.stats.released / 1e6).toFixed(4)} MJ, maxT ${f1(on.stats.maxT)} K, ${on.N} cells`);
  reported("e", `(e)1 ${cfg.label}: burnerRadiatedOut on ${f1(on.stats.burnerRadiatedOut)} J${off.stats.burnerRadiatedOut != null ? `, off ${f1(off.stats.burnerRadiatedOut)} J` : " (none off)"}, of ${f1(on.burner.radiant)} J radiant`);
}

/** (f): the room with fuel F2 for 120 s, hooked between the stages through sim.profile, or timed. */
function roomRun(options, { hooked = false, seconds = F_SECONDS } = {}) {
  const g = room();
  const sim = buildSim(F, g, 0.2, { layers: ROOM_FLAME, fuel: F2, options });
  const steps = Math.round(seconds / DT);
  const ledger = [];
  const acc = { steps: 0, raised: 0, deposited: 0, stampedSteps: 0, stampedCells: 0, worstSil: 0, silSum: 0, clearedSum: 0, radiant: 0, fateStamped: 0, stampedEndsSteps: 0, stampedEnds: 0, worstFate: 0, maxStampedNoEnd: 0, noFlameSteps: 0 };
  let pre = null;
  const b = sim.burner;
  let lastRadiant = b.radiant;
  const hook = (name) => {
    // Each name is read just after its stage has run (step's profile).
    if (name === "pilot") {
      pre = { radiantRise: b.radiant - lastRadiant };
      lastRadiant = b.radiant;
    } else if (name === "particles") {
      // The state just before the radiation stage, and the step's stamped cells.
      const rec = b.flame && b.flame.step === sim.steps ? b.flame : null;
      const cells = [];
      if (rec) for (let c = 0; c < sim.N; c += 1) if (rec.stamp[c] === rec.mark) cells.push(c);
      pre.cells = cells;
      pre.T = Float32Array.from(cells, (c) => sim.T[c]);
      pre.sil = sim.stats.flameSilenced;
      pre.silRef = eq17(sim, cells, DT);
      pre.fate = b.rayFate ? { ...b.rayFate } : null;
    } else if (name === "radiation") {
      acc.steps += 1;
      const dep = sim.work.radGas;
      if (!pre.cells.length) acc.noFlameSteps += 1;
      pre.cells.forEach((c, q) => {
        const d = dep ? dep[c] : 0;
        const expect = d ? Math.fround(pre.T[q] + d) : pre.T[q];
        if (!Object.is(sim.T[c], expect)) acc.raised += 1;
        if (d) acc.deposited += 1;
      });
      acc.stampedCells += pre.cells.length;
      if (pre.cells.length) acc.stampedSteps += 1;
      const rise = sim.stats.flameSilenced - pre.sil;
      acc.worstSil = Math.max(acc.worstSil, relDiff(rise, pre.silRef));
      acc.silSum += rise;
      const f = pre.fate;
      if (f && pre.radiantRise > 0) {
        acc.radiant += pre.radiantRise;
        acc.clearedSum += f.cleared;
        if (f.stampedEnds > 0) {
          acc.stampedEndsSteps += 1;
          acc.stampedEnds += f.stampedEnds;
        } else acc.maxStampedNoEnd = Math.max(acc.maxStampedNoEnd, Math.abs(f.stamped));
        acc.fateStamped += f.stamped;
        acc.worstFate = Math.max(acc.worstFate, relDiff(f.stamped + f.gas + f.solid + f.out, pre.radiantRise));
      }
    }
  };
  const target = {};
  sim.profile = hooked
    ? new Proxy(target, {
        get(t, k) {
          if (typeof k === "string") hook(k);
          return t[k];
        },
      })
    : target;
  const t0 = performance.now();
  for (let s = 0; s < steps; s += 1) {
    F.step(sim, DT);
    if ((s + 1) % 300 === 0) {
      let gas = 0, walls = 0;
      for (let c = 0; c < sim.N; c += 1) {
        if (sim.kind[c] === F.GAS) gas += cellH(sim, c);
        if (sim.inertC[c] > 0) walls += sim.inertC[c] * (sim.Ts[c] - TA);
      }
      const st = sim.stats;
      const MJ = (x) => (Number.isFinite(x) ? +(x / 1e6).toFixed(4) : null);
      const MJs = (x) => (Number.isFinite(x) ? +(x / 1e6).toPrecision(6) : null);
      ledger.push({ t: Math.round(sim.time), released: MJ(st.released), radiated: MJ(st.radiated), flameSilenced: MJs(st.flameSilenced), flameCleared: MJs(st.flameCleared), burnerRadiatedOut: MJ(st.burnerRadiatedOut), gas: MJ(gas), walls: MJ(walls), shell: MJ(st.shellHeat) });
    }
  }
  const ms = performance.now() - t0;
  return { sim, acc, ledger, ms, steps, profile: { ...target } };
}

function checkF(cfgId) {
  const cfg = CONFIGS[cfgId];
  console.log(`\n(f) ${room().name}, 0.2 m, fuel F2 (class A), ${F_SECONDS} s, ${cfg.label}`);
  const on = roomRun(cfg.on, { hooked: true });
  const a = on.acc;
  verdict("f", a.raised === 0, `(f) ${cfg.label}: ${a.raised} stamped-cell steps of ${a.stampedCells} had their T lowered or changed other than by other cells' rays in the radiation stage (0; bit for bit: T after = T before + the stage's deposit, float32); ${a.deposited} of them took a deposit from other cells' rays (the declared limit), over ${a.steps} steps, ${a.stampedSteps} with a flame`);
  verdict("f", a.worstSil <= 1e-12, `(f) ${cfg.label}: flameSilenced rises by the stamped cells' would-be Eq. 17 emission within ${ex(a.worstSil)} of it in every step (against 1e-12); ${f1(a.silSum / 1e3)} kJ in all`);
  verdict("f", a.maxStampedNoEnd === 0, `(f) ${cfg.label}: b.rayFate in stamped cells at most ${a.maxStampedNoEnd} J in the steps where no ray ended in one (exactly 0); rays that ended their crossings in a stamped cell: ${a.stampedEnds}, in ${a.stampedEndsSteps} steps, leaving ${f1(a.fateStamped)} J there`);
  verdict("f", a.worstFate <= 1e-9, `(f) ${cfg.label}: b.rayFate's parts sum to the radiant output within ${ex(a.worstFate)} of it in every step (against 1e-9)`);
  reported("f", `${cfg.label}: over the run flameSilenced ${f4(a.silSum / a.radiant)} and flameCleared ${f4(a.clearedSum / a.radiant)} of chi_r Q (${f1(a.radiant / 1e3)} kJ radiant)`);
  const off = roomRun(cfg.off);
  for (const [name, r] of [["on", on], ["off", off]]) reported("f", `${cfg.label}, ledger ${name}: ${r.ledger.map((e) => JSON.stringify(e)).join("; ")}`);
  return { on, off };
}

// ------------------------------------------------------------ (e)2 and (g)

/** The base's src/core at rev, from git, in a temporary directory. */
function baseCore(rev) {
  const dir = mkdtempSync(path.join(os.tmpdir(), "flame-once-base-"));
  const core = path.join(dir, "topologicpy-web-frontend", "src", "core");
  mkdirSync(core, { recursive: true });
  const files = execFileSync("git", ["ls-tree", "--name-only", rev, "topologicpy-web-frontend/src/core/"], { cwd: ROOT, encoding: "utf8" }).split("\n").map((p) => p.split("/").pop()).filter((f) => f.endsWith(".js"));
  for (const f of files) writeFileSync(path.join(core, f), execFileSync("git", ["show", `${rev}:topologicpy-web-frontend/src/core/${f}`], { cwd: ROOT, maxBuffer: 1 << 28 }));
  writeFileSync(path.join(dir, "package.json"), JSON.stringify({ type: "module" }));
  const resolved = execFileSync("git", ["rev-parse", rev], { cwd: ROOT, encoding: "utf8" }).trim();
  return { dir, core, rev: resolved };
}

function supportedOf(src) {
  const m = src.match(/function supported\(sim\) \{([\s\S]*?)\n\}/);
  return m ? new Function("sim", m[1]) : null;
}

/** The fingerprint tool's "room design fire" case (tools/flamescapes-fingerprint.mjs), `steps` steps (300), with the given options. */
async function designFire(options, steps = 300) {
  const R = await imp("flamescapesRun.js");
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
  const scene = R.buildScene(geometry, { upAxis: "y", fire: [2.5, 0, 2.0], maxCells: 30000, contentsFuel, ignitionFuel, item, designFire: FC.designFireFor("A", "class"), fireLoad: 780e6, seed: 1 });
  const sim = F.createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, embers: false, seed: 1, ...options });
  for (let s = 0; s < steps; s += 1) F.step(sim, 0.1);
  return { sim, fp: fingerprint([sim.T, sim.Ts, sim.Y, sim.u, sim.v, sim.w, sim.rho, sim.p, sim.fuelMass, sim.moisture, sim.parts.x.subarray(0, sim.parts.n * 3), Float64Array.of(sim.stats.released, sim.stats.radiated, sim.parts.n)]) };
}

async function checkE2() {
  console.log("\n(e)2 the app's design fire (radiant fraction 0), the fingerprint tool's design-fire room, 300 steps on the CPU");
  const supported = supportedOf(readFileSync(path.join(CORE, "flamescapesGpu.js"), "utf8"));
  const off = await designFire({});
  const runs = { off: off.fp };
  for (const id of CONFIG_IDS) {
    const on = await designFire(CONFIGS[id].on);
    runs[id] = on.fp;
    verdict("e", on.fp === off.fp, `(e)2 ${CONFIGS[id].label}: fingerprint ${on.fp} on, ${off.fp} off (equal); the burner's radiant fraction ${on.sim.burner.radiantFraction}, item record ${on.sim.burner.flame ? "made" : "never made"}, flameSilenced ${on.sim.stats.flameSilenced} J`);
    verdict("e", supported(on.sim) === true && supported(off.sim) === true, `(e)2 ${CONFIGS[id].label}: supported() ${supported(on.sim)} on and ${supported(off.sim)} off (true and true: the GPU runs the design fire either way)`);
  }
  report.checks.e2 = runs;
}

async function checkG() {
  console.log(`\n(g) off is off: against the base at ${BASE_REV}`);
  const base = baseCore(BASE_REV);
  try {
    const B = await import(pathToFileURL(path.join(base.core, "flamescapes.js")).href);
    report.checks.g = { baseRev: base.rev };
    const g = box([8, 8, 8]);
    const layers = [[[3, 0, 3]]];
    const run = (Fm, options) => {
      const sim = buildSim(Fm, g, 0.2, { layers, fuel: F2, chi: 0.2, rays: 16, q: 2e4, options });
      for (let s = 0; s < 300; s += 1) Fm.step(sim, 0.1);
      return sim;
    };
    const b0 = run(B, {});
    const off = run(F, { flameRadiantOnce: 0 });
    const d0 = differences(b0, off, { skipBurner: ["flame"] });
    verdict("g", d0.length === 0, `8 x 8 x 8 box, 0.2 m, F2, 20 kW, 300 steps, flameRadiantOnce 0 against the base: ${d0.length ? `differ in ${d0.join(", ")}` : "every field and every stats entry bit for bit the same"} (released ${(off.stats.released / 1e6).toFixed(4)} MJ, radiated ${(off.stats.radiated / 1e6).toFixed(4)} MJ, maxT ${f1(off.stats.maxT)} K, ${Object.keys(off.stats).length} stats entries)`);
    const b12 = run(B, { burnerFlameRays: 1 });
    const off12 = run(F, { burnerFlameRays: 1, flameRadiantOnce: 0 });
    const d1 = differences(b12, off12, { skipBurner: ["flame"] });
    verdict("g", d1.length === 0, `the same scene with item 12 on and this item off against item 12 alone at the base: ${d1.length ? `differ in ${d1.join(", ")}` : "every field and every stats entry bit for bit the same"} (burnerRadiatedOut ${f1(off12.stats.burnerRadiatedOut)} J, ${Object.keys(off12.stats).length} stats entries)`);
    for (const id of CONFIG_IDS) {
      const on = run(F, CONFIGS[id].on);
      const ref = CONFIGS[id].off.burnerFlameRays ? b12 : b0;
      const d = differences(ref, on, { skipBurner: ["flame", "rayFate"] });
      reported("g", `${CONFIGS[id].label}, with flameRadiantOnce 1: differs from the base's ${id === "both" ? "item 12 alone" : "run"} in ${d.length ? d.join(", ") : "nothing"}; flameSilenced ${f1(on.stats.flameSilenced)} J, flameCleared ${f1(on.stats.flameCleared)} J, burnerRadiatedOut ${f1(on.stats.burnerRadiatedOut)} J, radiant ${f1(on.burner.radiant)} J`);
    }
    // supported(): the branch's and the base's, from their sources.
    const gpuSrc = readFileSync(path.join(CORE, "flamescapesGpu.js"), "utf8");
    const baseGpu = execFileSync("git", ["show", `${BASE_REV}:topologicpy-web-frontend/src/core/flamescapesGpu.js`], { cwd: ROOT, maxBuffer: 1 << 28 }).toString("utf8");
    const sup = supportedOf(gpuSrc), supBase = supportedOf(baseGpu);
    if (!sup || !supBase) verdict("g", false, "supported() not found in flamescapesGpu.js");
    else {
      const val = buildSim(F, g, 0.2, { layers, fuel: F2, chi: 0.2, rays: 16, q: 2e4 });
      const df = await designFire({}, 0);
      const dfChi = await designFire({}, 0);
      dfChi.sim.burner.radiantFraction = 0.2;
      const cases = [["a validation burner", val], ["the design fire", df.sim], ["a design-fire burner given a radiant fraction of 0.2 (no caller makes one)", dfChi.sim]];
      for (const [name, sim] of cases) {
        const now = sup(sim), before = supBase(sim);
        verdict("g", now === before, `supported() with flameRadiantOnce 0, ${name}: ${now} (the base's ${before})`);
      }
      for (const [name, sim] of cases) {
        const P = sim.P;
        sim.P = { ...P, flameRadiantOnce: 1 };
        const now = sup(sim);
        sim.P = P;
        const expect = name === "the design fire";
        verdict("g", now === expect, `supported() with flameRadiantOnce 1, ${name}: ${now} (${expect}: ${expect ? "its radiant fraction is 0, so the GPU runs it" : "a burner with a radiant fraction runs on the CPU"})`);
      }
    }
  } finally {
    rmSync(base.dir, { recursive: true, force: true });
  }
}

// ------------------------------------------------------------ run

const t0 = performance.now();
console.log(`Item 14 checks: ${STEPS} steps a case, dt ${DT} s, ${RAYS} rays, radiant fraction ${CHI}, ${Q / 1e3} kW held, seed 1, cells ${CELLS.join(", ")} m; configurations ${CONFIG_IDS.map((id) => CONFIGS[id].label).join(", ")}`);
if (["a", "b", "c", "d"].some((x) => ONLY.has(x))) {
  report.checks.box = {};
  for (const id of CONFIG_IDS) report.checks.box[id] = runBox(id);
}
if (ONLY.has("e")) {
  for (const id of CONFIG_IDS) checkE1(id);
  await checkE2();
}
if (ONLY.has("f") || ONLY.has("i")) {
  report.checks.f = {};
  for (const id of CONFIG_IDS) {
    const { on, off } = checkF(id);
    const timed = roomRun(CONFIGS[id].on);
    const pr = (r, k) => (r.profile[k] ?? 0) / r.steps;
    reported("i", `(f)'s run, ${CONFIGS[id].label}, ${on.steps} steps (unhooked runs): the radiation stage ${pr(timed, "radiation").toFixed(3)} ms a step on, ${pr(off, "radiation").toFixed(3)} off; burnerStep (the pilot stage) ${pr(timed, "pilot").toFixed(3)} ms on, ${pr(off, "pilot").toFixed(3)} off; the whole step ${(timed.ms / timed.steps).toFixed(2)} ms on, ${(off.ms / off.steps).toFixed(2)} off (a loaded machine)`);
    report.checks.f[id] = { acc: on.acc, ledgerOn: on.ledger, ledgerOff: off.ledger, ms: { on: timed.ms / timed.steps, off: off.ms / off.steps, radiationOn: pr(timed, "radiation"), radiationOff: pr(off, "radiation"), pilotOn: pr(timed, "pilot"), pilotOff: pr(off, "pilot") } };
  }
}
if (ONLY.has("g")) await checkG();
report.verdicts = verdicts;
report.lines = lines;
report.seconds = (performance.now() - t0) / 1000;
const counts = { pass: lines.filter((l) => l.includes(") PASS:")).length, fail: lines.filter((l) => l.includes(") FAIL:")).length, reported: lines.filter((l) => l.includes(") REPORT:")).length };
report.counts = counts;
console.log(`\nVerdicts (16.14's checks; (g)'s fingerprints, unit suite and GPU check, (h) and (i)'s suite runs run apart): ${counts.pass} PASS, ${counts.fail} FAIL, ${counts.reported} reported`);
for (const [k, v] of Object.entries(verdicts)) console.log(`  (${k}) ${v ? "PASS" : "FAIL"}`);
console.log(`${report.seconds.toFixed(1)} s`);
const json = flag("--json", null);
if (json) await writeFile(json, JSON.stringify(report, null, 1));
process.exitCode = counts.fail ? 1 : 0;
