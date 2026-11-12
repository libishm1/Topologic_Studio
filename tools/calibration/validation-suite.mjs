/**
 * The fire calibration's held-out validation suite (wiki/research/fire-calibration-plan-2026-11-11.md,
 * section 12): the Flamescapes solver, as the app runs it, against measured
 * full-scale tests it was never fitted to, on the app's cells, CPU. Rerun it
 * after each change to the solver; the acceptance rule is fixed in the plan
 * and below (VERDICT), and nothing here tunes the solver.
 *
 * Datasets (tools/calibration/data/<name>.json, made by validation-data.mjs
 * from NIST's firemodels/exp and firemodels/fds at fixed commits):
 *   steckler  Steckler, Quintiere and Rinkinen (1982): 55 steady tests in a
 *             2.8 x 2.8 x 2.13 m room, door and window widths and sills,
 *             burner positions and 31.6 to 158 kW varied. Room tree's layer
 *             (Janssens and Tran), door centreline peak outflow and neutral plane.
 *   nbs       NBS multi-room tests 100A, 100O, 100Z (Peacock, Davis and
 *             Babrauskas 1991): 110 kW in a burn room off a 12 m corridor.
 *             Layer temperature and depth at four trees.
 *   nistnrc   NIST/NRC tests (Hamins et al. 2006): 0.4 to 2.3 MW heptane in a
 *             21.7 x 7.04 x 3.82 m compartment, door open or shut. Layer at six
 *             trees, ceiling-jet thermocouples, O2 and CO2.
 *   wtc       NIST WTC phase 1 (NCSTAR 1-5B): 2 to 3.2 MW heptane in a
 *             7.0 x 3.6 x 3.82 m compartment with two openings. Layer at two
 *             trees, two ceiling-jet thermocouples, O2 and CO2.
 *
 * Each test becomes a scene through the FDS input's geometry (lib/fds.mjs:
 * meshes, obstructions less holes, open vents), voxelised by the app's own
 * voxelize.js; the fire is the app's design-fire burner (flamescapesRun.js
 * itemBurner: the open cells over the burner's area a layer at a time up to
 * the ceiling, Heskestad's flame height choosing the layers each step, the
 * class A fuel's soot, radiant fraction 0), its heat release the test's
 * (validation-data.mjs), its CO2 yield set to the test fuel's CO2 per unit
 * heat (CO2 is a passive tracer of the heat here). The solver's ambient is the
 * test's. Each second the run is checked as the app checks it
 * (flamescapes.js healthReadings, healthVerdict) and its steps are cut as the
 * app cuts them (substepsFor); a run that fails stops and its readings are set
 * aside.
 *
 * Compared, as rises above ambient (the FDS Validation Guide's way):
 *   hglTemperature  each tree's upper layer (Janssens and Tran), its largest
 *                   rise over the window (steady: the 200 to 300 s mean);
 *   hglDepth        each tree's layer depth, ceiling less the interface, its
 *                   largest over the window (steady: the mean);
 *   ceilingJet      each ceiling thermocouple's largest rise over the window;
 *   o2, co2         each analyser's largest drop of O2 and rise of CO2 (volume fractions);
 *   doorVelocity    Steckler: the door centreline's peak outflow (steady);
 *   neutralPlane    Steckler: the door centreline's flow reversal height (steady);
 *   doorVelocity    (13.2 items 4, 5) each opening probe's largest speed, either
 *                   way (the guide's maxabs); lib/probes.mjs;
 *   neutralPlane    (13.2 item 5) each probe column's reversal height, mean
 *                   profile over the window; doorMassFlow the door's largest
 *                   inflow and outflow (kg/s); lib/probes.mjs;
 *   headTemperature each tree's thermocouples 1.5 to 2.0 m over its floor, out
 *                   of the burner's plume: the peak rise over the run (steady:
 *                   the 200 to 300 s mean); lib/head.mjs (plan 13.2 item 1).
 * The model is read in the open cells (a tree: its column's open cells, the
 * open top as its ceiling; a point: its cell, or the nearest open one) at
 * the measurement's own times, each a mean of the model's 1 s readings over
 * the measurement's interval.
 *
 *   node tools/calibration/validation-suite.mjs [--datasets steckler,nbs,nistnrc,wtc]
 *        [--cells 0.20,0.27,0.34] [--solver vorticity=6,cfl=1] [--jobs N (default 4)]
 *        [--tests REGEX] [--json out.json] [--summary out.json]
 *
 * --tests filters "dataset/test" ids (e.g. "steckler/(14|410|610)$");
 * --summary file.json recomputes and prints the statistics of an earlier
 * run (--write saves them back into it); --walls stone runs the solver's
 * stone walls instead of each test's (a diagnostic); --all-cells runs the
 * datasets the plan skips on 0.20 m cells; --horizon-cap S shortens every
 * run (a smoke test only).
 *
 * --solver-rev REV (added 2026-11-12, plan 13.5) takes the solver's modules
 * (every src/core/*.js) from git at REV instead of the working tree, so a run
 * of 13.A keeps d3c5751's solver while 13.B's changes are merged:
 * --solver-rev d3c5751. The voxeliser that the sealed-door rule uses comes
 * from the same snapshot. --wall-layers (added 2026-11-12, for 13.B) passes
 * each wall material's layers (thickness, rho, c, k, as lib/fds.mjs
 * surfaceMaterial returns them) to the solver's inert option beside the
 * lumped values; without it (13.A) only the lumped values go.
 */
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "../..");
const CORE = path.join(ROOT, "topologicpy-web-frontend/src/core");

/**
 * Per dataset, fixed before the baseline (plan section 12): the run's
 * length, the cells it is run on by default, and whether it is steady.
 *   - steckler: 300 s, the 200 to 300 s mean against the tests' steady state
 *     (as steckler-checks.mjs; the tests ran 30 min);
 *   - nbs: 900 s, the burner's whole steady burn (FDS's windows 0 to 900 s
 *     for the layer temperature, 300 to 900 s for its depth);
 *   - nistnrc: 600 s (the tests ran 20 to 25 min at full heat release; a
 *     1500 s run on 0.20 m cells would take over an hour), so its windows end
 *     at 600 s; not run on 0.20 m cells by default (about 90,000 cells:
 *     about 50 min a test);
 *   - wtc: 600 s, or the burner's end plus 30 s if sooner; not run on 0.20 m
 *     cells by default (2 to 3.2 MW on about 21,000 cells: about 60 min a test).
 */
export const PLAN = Object.freeze({
  steckler: { horizon: () => 300, steadyFrom: 200 },
  nbs: { horizon: () => 900 },
  nistnrc: { horizon: () => 600, skipCells: [0.2] },
  wtc: { horizon: (t) => Math.min(600, Math.ceil(lastBurning(t.burner.hrrW)) + 30), skipCells: [0.2] },
  // Added 2026-11-12 (plan section 12.9): 600 s; about 78,000 cells on 0.20 m, so not run there by default.
  atf: { horizon: () => 600, skipCells: [0.2] },
  // Added 2026-11-12 (plan 13.2 item 3): each test's own run, ignition to the first of Burner Off, a door or
  // fan event, the file's end and 900 s (validation-data.mjs fsri).
  fsri: { horizon: (t) => t.horizon },
  // Added 2026-11-12 (plan 13.2 item 6): each fire's first interval, to its end (validation-data.mjs ulnist).
  ulnist: { horizon: (t) => t.horizon },
  // Added 2026-11-12 (plan 13.2 item 8): to the end of the guide's first window (validation-data.mjs fse).
  fse: { horizon: (t) => t.horizon },
});

/** The independent test series and what makes each independent (plan section 12). */
export const SERIES = Object.freeze({
  steckler: "NBS 1980 to 1981, Steckler, Quintiere, Rinkinen; small room doorway flows",
  nbs: "NBS 1985, Peacock, Davis, Babrauskas; three-room suite with a corridor",
  nistnrc: "NIST 2003, Hamins et al. for the US NRC; large compartment (shares NIST's large fire laboratory and lead with wtc)",
  wtc: "NIST 2004, Hamins et al. for the WTC investigation; spray-burner compartment (shares the laboratory and lead with nistnrc)",
  atf: "ATF Fire Research Laboratory 2008, Sheppard; two corridors joined by a stairwell (a federal laboratory other than NIST)",
  fsri: "UL FSRI 2018, Willi et al.; gas burners in a 3.66 m room with one door (a laboratory other than NIST and ATF; a box scene, no FDS input)",
  ulnist: "NIST at UL 2012, Opert; gas fires in a 6.1 x 4.3 m room with a door and ceiling vents, first interval (NIST's fire fighting group; UL's building)",
  fse: "NIST 2008, Lock et al.; an ISO 9705 room, the tests with doors of 0.4 m or more, first window",
});

/**
 * The laboratory of each series (plan section 13.1): NBS and NIST count as
 * one. The verdict counts series (12.6); the laboratories of the in-band
 * series are reported beside it.
 */
export const LAB = Object.freeze({ steckler: "NBS/NIST", nbs: "NBS/NIST", nistnrc: "NBS/NIST", wtc: "NBS/NIST", atf: "ATF", fsri: "UL FSRI", ulnist: "NBS/NIST", delco: "NBS/NIST", fse: "NBS/NIST" });

/** Which sigma_E (lib/stats.mjs SIGMA_E) each quantity takes. */
// doorMassFlow takes the velocity's 0.08: the guide gives none for a mass flow (this project's assumption, 2026-11-12).
// soot (a mass fraction) takes the smoke's 0.12: the guide gives none for it (this project's assumption, 2026-11-12).
const SIGMA_KEY = { hglTemperature: "hglTemperature", hglDepth: "hglDepth", ceilingJet: "ceilingJet", o2: "o2", co2: "co2", smoke: "smoke", doorVelocity: "velocity", neutralPlane: "neutralPlane", headTemperature: "headTemperature", doorMassFlow: "velocity", soot: "smoke" };

/**
 * The verdict, fixed before the baseline (plan section 12): per quantity and
 * cell, a dataset is "in band" when its own statistics (its tests and
 * devices) lie within this project's calibrated band (the 2026-11-08 plan's,
 * on the FDS Validation Guide's statistics; the guide defines no band) and none of its runs at that cell
 * stopped; the quantity is "calibrated" on that cell when at least three
 * datasets are in band and all datasets pooled lie within the band too;
 * "indicative" when the pooled statistics lie within the indicative band;
 * otherwise "not calibrated".
 */
export const VERDICT = "calibrated on a cell: at least three independent datasets each within this project's calibrated band (2026-11-08 plan, section 8, on the FDS Validation Guide's statistics: 0.8 <= delta <= 1.25, sigmaM <= 0.35) with none of their runs stopped, and the pooled pairs within that band too; indicative: pooled within 0.67 <= delta <= 1.5, sigmaM <= 0.5; otherwise not calibrated";

/**
 * The claim rule of wiki/research/fire-calibration-claim-protocol-2026-11-12.md,
 * section 4 (written before the held-out results), reported beside VERDICT:
 * calibrated when within the calibrated band in at least three datasets that
 * measure the quantity and outside the indicative band in none; indicative
 * when within the indicative band in at least two; otherwise not calibrated
 * (or when fewer than two datasets measure it). A dataset with a stopped run
 * on that cell counts as outside the indicative band (this suite's reading).
 */
export const CLAIM = "claim protocol 2026-11-12, section 4: calibrated if in the calibrated band in >= 3 datasets and outside the indicative band in none; indicative if in the indicative band (or better) in >= 2; else not calibrated; a dataset with a stopped run counts as outside";

const lastBurning = (table) => {
  let t = 0;
  for (const [s, q] of table) if (q > 0) t = s;
  return Math.min(t, 1e5);
};

// ================================================================ worker

async function runCase(c) {
  // The solver as snapshot at the suite's start (main: snapshotCore), so every run uses the same code.
  const core = c.core || CORE;
  const F = await import(pathToFileURL(path.join(core, "flamescapes.js")).href);
  const R = await import(pathToFileURL(path.join(core, "flamescapesRun.js")).href);
  const FC = await import(pathToFileURL(path.join(core, "fireClasses.js")).href);
  const VX = await import(pathToFileURL(path.join(core, "voxelize.js")).href);
  const L = await import("./lib/fds.mjs");
  const S = await import("./lib/stats.mjs");
  const t0 = performance.now();
  const { test: T, h, horizon, dataset } = c;
  const geo = { ...T.geometry, fire: [] };
  const G = L.gridFor(geo, h, VX);
  const { grid, solid } = G;
  const KELVIN = 273.15;
  const Ta = T.ambientC + KELVIN;

  // The walls (plan section 12): each solid cell takes its slot of the test's own surfaces
  // (validation-data.mjs materialsOf), and each slot the surface's areal heat capacity over
  // the cell (rho = sum rho L / h), its exposed layer's conductivity and emissivity.
  let inert = null, wallCells = null;
  if (c.walls !== "stone" && T.materials) {
    const M = VX.MATERIALS;
    const slots = T.materials.slots;
    wallCells = [0, 0, 0];
    for (let k = 0; k < grid.nz; k += 1) for (let j = 0; j < grid.ny; j += 1) for (let i = 0; i < grid.nx; i += 1) {
      const cc = G.at(i, j, k);
      const f = G.fraction[cc * M];
      if (!(f > 0)) continue;
      const [x, y, z] = G.centre(i, j, k);
      let slot = 0;
      for (const r of T.materials.rules) {
        const inBox = !r.xb || (x >= r.xb[0] - h && x <= r.xb[1] + h && y >= r.xb[2] - h && y <= r.xb[3] + h && z >= r.xb[4] - h && z <= r.xb[5] + h);
        if (inBox && (!r.floor || z < G.bbox[4])) {
          slot = r.slot;
          break;
        }
      }
      if (slot) {
        G.fraction[cc * M] = 0;
        G.fraction[cc * M + slot] += f;
      }
      if (solid[cc]) wallCells[slot] += 1;
    }
    inert = [0, 1, 2].map((q) => {
      const m = slots[q] || slots[0];
      const lumped = { rho: m.arealMass / h, c: m.c, k: m.k, eps: m.eps };
      // --wall-layers (13.B): the wall's real layers too, exposed layer first, for a solver that conducts through them.
      return c.wallLayers && m.layers?.length ? { ...lumped, layers: m.layers.map((l) => ({ thickness: l.thickness, rho: l.rho, c: l.c, k: l.k })) } : lumped;
    });
  }

  // The burner: the app's design-fire burner over the test's burner area.
  // The burner's base column: over its centre, or the nearest open one on its footprint (lib/fds.mjs burnerBase).
  const base = L.burnerBase(G, T.burner);
  if (!base.cells.length) throw new Error(`${c.id}: no open cell over the burner`);
  const cell0 = base.cells[0].c;
  const A = FC.fuelsFor("A").ignitionFuel;
  // The test fuel's heat of combustion and yields (plan 12.9); --fuel app keeps the class A fuel with its
  // CO2 yield set to the test fuel's CO2 per joule (the rule of the runs before 2026-11-12, 12.3).
  const fuel = c.fuelRule !== "app" && T.fuel
    ? { heatOfCombustion: T.fuel.heatOfCombustion, yields: { co2: T.fuel.yields.co2, co: T.fuel.yields.co ?? 0, soot: T.fuel.yields.soot ?? 0 }, embers: false, ignitionTemperature: A.ignitionTemperature }
    : T.co2PerJ ? { ...A, yields: { ...A.yields, co2: T.co2PerJ * A.heatOfCombustion } } : A;
  const spec = R.itemBurner(grid, solid, cell0, { alpha: 0.0117, item: { area: T.burner.area, mass: 1 }, fuel });
  spec.hrr = T.burner.hrrW;
  spec.design = null;
  // The test's radiant fraction along the burner's rays (plan 12.9, amendment 8): the FDS input's, or
  // FDS's default for its fuel. --fuel app keeps the app's design-fire burner (radiant fraction 0).
  if (c.fuelRule !== "app" && T.burner.radiantFraction != null) {
    spec.radiantFraction = T.burner.radiantFraction;
    spec.rays = 16;
  }
  const INERT_FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };
  const scene = { grid, fraction: G.fraction, solid, contents: { cells: [], masses: [] }, ignition: null, pilot: null, burner: spec, fuels: { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL } };
  const sim = F.createSolver(scene, { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL, seed: 1, ambient: Ta, ...(inert ? { inert } : {}), ...c.solver });
  // Whether the solver took the walls' materials (a solver without options.inert keeps its stone).
  let wallCheck = null;
  if (inert) {
    const wc = solid.findIndex((s0) => s0 === 1);
    const f = Array.from({ length: VX.MATERIALS }, (_, m) => G.fraction[wc * VX.MATERIALS + m]);
    const expect = f.reduce((a, x, m) => a + (m < 3 ? x * inert[m].rho * inert[m].c * h ** 3 : 0), 0);
    wallCheck = { honoured: Math.abs(sim.inertC[wc] - expect) <= 1e-6 * expect, cellHeatCapacity: +sim.inertC[wc].toFixed(1), expected: +expect.toFixed(1) };
  }

  // The instruments on the grid; after the first second they keep only cells of the fire's
  // connected gas (the solver's regions): the solver can seal a thin element's straddled row
  // and leave open pockets the hot gas never reaches (steckler-checks.mjs, 2026-11-12).
  // A tree reads every open cell of its column below the test's nominal ceiling, skipping solid
  // ones (lib/fds.mjs treeCells; fixed 2026-11-12 after WTC's west tree on 0.27 m cells stopped at a
  // bracket sealed into two cells at 2.97 m, 0.8 m under the ceiling).
  let trees = (T.trees || []).map((tr) => ({ ...tr, all: L.treeCells(G, tr.x, tr.y, (tr.z0 ?? 0) + tr.H, tr.z0 ?? null).cells }));
  let points = (T.points || []).map((p) => ({ ...p, cell: null }));
  let door = T.door ? { col: L.column(G, T.door.x, T.door.y, null) } : null;
  const uAt = (i, j, k) => i + (grid.nx + 1) * (j + grid.ny * k);
  // Opening probes (13.2 items 4, 5): the connected open cell nearest each; FDS x, y, z are the solver's i, k, j.
  let probes = (T.probes || []).map((p) => ({ ...p, cell: null }));
  const vAt = (i, j, k) => i + grid.nx * (j + (grid.ny + 1) * k);
  const wAt = (i, j, k) => i + grid.nx * (j + grid.ny * k);
  const along = (axis, { i, j, k }) =>
    axis === "x" ? 0.5 * (sim.u[uAt(i, j, k)] + sim.u[uAt(i + 1, j, k)]) : axis === "z" ? 0.5 * (sim.v[vAt(i, j, k)] + sim.v[vAt(i, j + 1, k)]) : 0.5 * (sim.w[wAt(i, j, k)] + sim.w[wAt(i, j, k + 1)]);
  // The door's faces (13.2 item 5): the x faces nearest its plane whose cells' centres lie in the opening.
  let doorFlow = null;
  let region = null;
  const connect = () => {
    const home = cell0;
    const reg = sim.topo?.region;
    let inHome;
    if (reg && reg.length === sim.N) inHome = (cc) => reg[cc] === reg[home];
    else {
      // The open cells joined to the burner through open faces (if the solver keeps no regions).
      const seen = new Uint8Array(sim.N);
      const stack = [home];
      seen[home] = 1;
      while (stack.length) {
        const cc = stack.pop();
        const i = cc % grid.nx, j = Math.floor(cc / grid.nx) % grid.ny, k = Math.floor(cc / (grid.nx * grid.ny));
        for (const [a, b, e] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
          const ii = i + a, jj = j + b, kk = k + e;
          if (ii < 0 || jj < 0 || kk < 0 || ii >= grid.nx || jj >= grid.ny || kk >= grid.nz) continue;
          const n = G.at(ii, jj, kk);
          if (!seen[n] && sim.kind[n] !== F.SOLID) (seen[n] = 1), stack.push(n);
        }
      }
      inHome = (cc) => seen[cc] === 1;
    }
    const ok = (cc) => sim.kind[cc] !== F.SOLID && inHome(cc);
    trees = trees.map((tr) => {
      const kept = tr.all.filter((q) => ok(q.c));
      const floor = kept.length ? kept[0].z - h / 2 : 0;
      const top = kept.length ? kept[kept.length - 1].z + h / 2 : 0;
      return { ...tr, dropped: tr.all.length - kept.length, modelH: top - floor, z: kept.map((q) => q.z - floor), cells: kept.map((q) => q.c) };
    });
    points = points.map((p) => {
      const cell = L.deviceCell(G, p.x, p.y, p.z, ok);
      if (!p.path) return { ...p, cell };
      // A light path: the connected open cells it crosses, sampled at nine points along it.
      const cells = new Set();
      for (let q = 0; q <= 8; q += 1) {
        const [a, b] = p.path;
        const d = L.deviceCell(G, a[0] + ((b[0] - a[0]) * q) / 8, a[1] + ((b[1] - a[1]) * q) / 8, a[2] + ((b[2] - a[2]) * q) / 8, ok);
        if (d) cells.add(d.c);
      }
      return { ...p, cell, pathCells: [...cells] };
    });
    if (door) {
      const d = door.col;
      const kept = d.cells.filter((q) => ok(q.c));
      door.faces = kept.map((q) => [uAt(d.i, q.j, d.k), uAt(d.i + 1, q.j, d.k)]);
      // Heights over the room's floor (a window's column starts at its sill).
      door.z = kept.map((q) => q.z - G.bbox[4]);
      door.sill = kept.length ? kept[0].z - h / 2 - G.bbox[4] : null;
      door.soffit = kept.length ? kept[kept.length - 1].z + h / 2 - G.bbox[4] : null;
      // The opening's modelled width: the connected open cells across it at its middle height.
      const jm = kept.length ? kept[kept.length >> 1].j : 0;
      let w = 0;
      for (let k = 0; k < grid.nz; k += 1) if (ok(G.at(d.i, jm, k))) w += 1;
      door.width = w * h;
    }
    probes = probes.map((p) => ({ ...p, cell: L.deviceCell(G, p.x, p.y, p.z, ok) }));
    if (T.doorFlow) {
      const f = T.doorFlow;
      const iF = Math.round((f.plane - grid.origin[0]) / h);
      const faces = [];
      for (let k = 0; k < grid.nz; k += 1) for (let j = 0; j < grid.ny; j += 1) {
        const [, y, z] = G.centre(iF, j, k);
        if (y < f.y[0] || y > f.y[1] || z < f.z[0] || z > f.z[1] || iF < 1 || iF >= grid.nx) continue;
        const a = G.at(iF - 1, j, k), b = G.at(iF, j, k);
        if (!ok(a) || !ok(b)) continue;
        // inward +1: +x leads into the room, so the cell at iF - 1 is outside.
        faces.push({ u: uAt(iF, j, k), outside: f.inward > 0 ? a : b, inside: f.inward > 0 ? b : a });
      }
      doorFlow = { faces, inward: f.inward, area: faces.length * h * h, plane: grid.origin[0] + iF * h };
    }
    let n = 0;
    for (let cc = 0; cc < sim.N; cc += 1) if (ok(cc)) n += 1;
    region = { source: reg && reg.length === sim.N ? "solver regions" : "flood fill", openCells: n };
  };
  const MOLAR = F.MOLAR, NS = F.NS, SP = F.SPECIES, MN2 = 0.028014;
  const moleFraction = (c0, s) => {
    let inv = 0, rest = 1;
    for (let q = 0; q < NS; q += 1) {
      const y = sim.Y[c0 * NS + q];
      inv += y / MOLAR[q];
      rest -= y;
    }
    inv += Math.max(0, rest) / MN2;
    return sim.Y[c0 * NS + s] / MOLAR[s] / inv;
  };
  const x0O2 = (() => {
    const y = 0.232;
    return y / 0.032 / (y / 0.032 + (1 - y) / MN2);
  })();

  // Each second's readings, from the end of the first second (when the devices are connected).
  const seconds = [];
  let treeT = [], pointV = [], probeV = [];
  const flowIn = [], flowOut = [];
  const doorU = [];
  let speed = 0, mostSubsteps = 1, health = null, peakSpeed = 0;
  const dt = sim.P.dt;
  const built = performance.now();
  for (let k = 0; k < horizon; k += 1) {
    const end = k + 1;
    const substeps = F.substepsFor(sim.P, h, speed, dt);
    mostSubsteps = Math.max(mostSubsteps, substeps);
    while (sim.time < end - 1e-9) F.step(sim, Math.min(dt / substeps, end - sim.time));
    const reading = F.healthReadings(sim, grid, sim.P, sim.V);
    speed = reading.maxSpeed;
    peakSpeed = Math.max(peakSpeed, speed);
    const verdict = F.healthVerdict(reading, sim.stats.released + sim.stats.pilotHeat);
    if (!verdict.ok) {
      health = { ok: false, reason: verdict.reason, time: end, value: Number.isFinite(verdict.value) ? +verdict.value.toFixed(1) : null };
      break;
    }
    if (k === 0) {
      connect();
      treeT = trees.map(() => []);
      pointV = points.map(() => []);
      probeV = probes.map(() => []);
    }
    seconds.push(end);
    trees.forEach((tr, q) => treeT[q].push(Float32Array.from(tr.cells, (cc) => sim.T[cc])));
    points.forEach((p, q) => {
      if (!p.cell) return pointV[q].push(NaN);
      const cc = p.cell.c;
      if (p.quantity === "smoke") {
        // The obscuration per metre (%) from the soot along the path: K = Km rho Y_soot (Mulholland and
        // Croarkin's Km, flamescapesRun.js SOOT_EXTINCTION, as the app's visibility 3 / K uses it).
        const list = p.pathCells?.length ? p.pathCells : [cc];
        let rs = 0;
        for (const c1 of list) rs += sim.rho[c1] * sim.Y[c1 * NS + SP.SOOT];
        const K = R.SOOT_EXTINCTION * (rs / list.length);
        return pointV[q].push(100 * (1 - Math.exp(-K)));
      }
      pointV[q].push(p.quantity === "o2" ? moleFraction(cc, SP.O2) : p.quantity === "co2" ? moleFraction(cc, SP.CO2) : p.quantity === "soot" ? sim.Y[cc * NS + SP.SOOT] : sim.T[cc]);
    });
    if (door) doorU.push(Float32Array.from(door.faces, ([a, b]) => 0.5 * (sim.u[a] + sim.u[b])));
    probes.forEach((p, q) => probeV[q].push(p.cell ? +(p.sign * along(p.axis, p.cell)).toFixed(3) : null));
    if (doorFlow) {
      let fin = 0, fout = 0;
      for (const fc of doorFlow.faces) {
        const un = doorFlow.inward * sim.u[fc.u];
        if (un > 0) fin += un * sim.rho[fc.outside] * h * h;
        else fout -= un * sim.rho[fc.inside] * h * h;
      }
      flowIn.push(+fin.toFixed(4));
      flowOut.push(+fout.toFixed(4));
    }
  }
  const ran = performance.now();
  health ||= { ok: true, maxSpeed: +peakSpeed.toFixed(2), substeps: mostSubsteps };
  // Each tree's open cells around head height (centres 1.5 - h to 2.0 + h over the column's model floor),
  // their 1 s temperatures (K, from 1 s on), kept so that head-height temperature (plan 13.2 item 1) is
  // a reducer on the stored runs (added 2026-11-12: the runs before kept no series).
  const headSeries = {};
  trees.forEach((tr, q) => {
    if (!tr.z) return;
    const pick = tr.z.map((z, i) => (z >= 1.5 - h && z <= 2.0 + h ? i : -1)).filter((i) => i >= 0);
    if (!pick.length) return;
    headSeries[tr.id] = { z: pick.map((i) => +tr.z[i].toFixed(3)), T: pick.map((i) => treeT[q].map((row) => +row[i].toFixed(2))) };
  });

  // ------------------------------------------------ comparisons
  const pairs = [];
  const push = (quantity, device, model, measured, extra = {}) => pairs.push({ dataset, test: T.id, cell: h, quantity, device, model: Number.isFinite(model) ? +model.toFixed(4) : null, measured: Number.isFinite(measured) ? +measured.toFixed(4) : null, ...extra });
  const mean = (rows) => {
    const out = new Float64Array(rows[0].length);
    for (const r of rows) for (let q = 0; q < r.length; q += 1) out[q] += r[q] / rows.length;
    return out;
  };
  if (health.ok) {
    if (T.steady) {
      const from = c.steadyFrom;
      const pick = seconds.map((s, q) => (s > from ? q : -1)).filter((q) => q >= 0);
      if (!pick.length) throw new Error(`${c.id}: no readings after ${from} s (horizon ${horizon} s)`);
      trees.forEach((tr, q) => {
        const prof = mean(pick.map((p) => treeT[q][p]));
        const jt = S.janssensTran(tr.z, Array.from(prof), tr.modelH);
        const m = T.measured.trees[tr.id];
        push("hglTemperature", tr.id, jt.upper - Ta, m.Tu - T.ambientC);
        push("hglDepth", tr.id, tr.modelH - jt.height, m.H - m.zi);
      });
      if (door && T.measured.door.z.length) {
        const u = mean(pick.map((p) => doorU[p]));
        const z = door.z.slice().reverse(), v = Array.from(u).reverse();
        const md = T.measured.door;
        push("doorVelocity", "door", Math.max(...v), Math.max(...md.V));
        push("neutralPlane", "door", S.crossing(z, v), S.crossing(md.z, md.V));
      }
    } else {
      const M = T.measured;
      const dts = M.times.slice(1).map((t, q) => t - M.times[q]).sort((a, b) => a - b);
      const spacing = dts[dts.length >> 1] || 1;
      // The model at a measured time: the mean of its 1 s readings within half the measured interval of it.
      const at = (t, series) => {
        const lo = t - spacing / 2, hi = t + spacing / 2;
        const rows = [];
        for (let q = 0; q < seconds.length; q += 1) if (seconds[q] > lo && seconds[q] <= hi) rows.push(series[q]);
        return rows.length ? rows : null;
      };
      const inWindow = (w) => M.times.map((t, q) => [t, q]).filter(([t]) => t >= Math.max(1, w[0]) && t <= Math.min(w[1], horizon));
      trees.forEach((tr, q) => {
        const m = M.trees[tr.id];
        for (const quantity of ["hglTemperature", "hglDepth"]) {
          const w = T.windows[quantity];
          let best = null, bestM = null;
          for (const [t, iq] of inWindow(w)) {
            const rows = at(t, treeT[q]);
            const meas = quantity === "hglTemperature" ? m.Tu[iq] : m.zi[iq];
            if (!rows || meas == null) continue;
            const jt = S.janssensTran(tr.z, Array.from(mean(rows)), tr.modelH);
            const mv = quantity === "hglTemperature" ? jt.upper - Ta : tr.modelH - jt.height;
            const ev = quantity === "hglTemperature" ? meas - T.ambientC : m.H - meas;
            best = best == null ? mv : Math.max(best, mv);
            bestM = bestM == null ? ev : Math.max(bestM, ev);
          }
          push(quantity, tr.id, best, bestM);
        }
      });
      points.forEach((p, q) => {
        const series = M.points[p.id];
        if (!p.cell || !series) return;
        const pre = M.times.map((t, iq) => (t <= 0 && series[iq] != null && series[iq] > 0.01 ? series[iq] : null)).filter((x) => x != null);
        const x0 = pre.length ? pre.reduce((a, b) => a + b, 0) / pre.length : p.quantity === "o2" ? 0.209 : 0;
        let best = null, bestM = null;
        for (const [t, iq] of inWindow(T.windows[p.quantity])) {
          const rows = at(t, pointV[q].map((x) => [x]));
          const meas = series[iq];
          if (!rows || meas == null || (p.quantity === "o2" && meas < 0.01)) continue;
          const mv = mean(rows)[0];
          const model = p.quantity === "o2" ? x0O2 - mv : p.quantity === "co2" || p.quantity === "smoke" || p.quantity === "soot" ? mv : mv - Ta;
          const measured = p.quantity === "o2" ? x0 - meas : p.quantity === "co2" || p.quantity === "smoke" || p.quantity === "soot" ? meas - x0 : meas - T.ambientC;
          best = best == null ? model : Math.max(best, model);
          bestM = bestM == null ? measured : Math.max(bestM, measured);
        }
        push(p.quantity, p.id, best, bestM, { moved: p.cell.moved });
      });
    }
  }
  // Head-height temperature (plan 13.2 item 1): the same reducer as on stored runs (lib/head.mjs).
  const HD = await import("./lib/head.mjs");
  const head = HD.headPairs({ dataset, cell: h, horizon, health, series: { from: 1, step: 1, head: headSeries } }, T, c.steadyFrom);
  pairs.push(...head.pairs);
  // Opening probes, neutral planes and door mass flow (13.2 items 4, 5): lib/probes.mjs, as on stored runs.
  const probeSeries = Object.fromEntries(probes.map((p, q) => [p.id, probeV[q] || []]));
  const flowSeries = doorFlow ? { in: flowIn, out: flowOut } : null;
  const PR = await import("./lib/probes.mjs");
  const probed = PR.probePairs({ dataset, cell: h, horizon, health, series: { from: 1, step: 1, probes: probeSeries, doorFlow: flowSeries } }, T);
  pairs.push(...probed.pairs);
  const Qpeak = Math.max(...T.burner.hrrW.map((r) => r[1]));
  return {
    id: c.id,
    dataset,
    test: T.id,
    cell: h,
    horizon,
    grid: [grid.nx, grid.ny, grid.nz],
    cells: grid.cells,
    openCells: solid.reduce((a, s) => a + (s ? 0 : 1), 0),
    plumeIndex: +S.plumeIndex(Qpeak, h).toFixed(2),
    peakKW: Math.round(Qpeak / 1000),
    burner: { baseMoved: base.moved, cells: spec.layers.reduce((a, l) => a + l.length, 0), columns: spec.layers[0]?.length ?? 0, layers: spec.layers.length, radiantFraction: spec.radiantFraction, radiantMJ: +((sim.burner?.radiant ?? 0) / 1e6).toFixed(3), released: +(sim.stats.released / 1e6).toFixed(3), expectedMJ: +(burnerIntegral(T.burner.hrrW, health.ok ? horizon : health.time) / 1e6).toFixed(3) },
    trees: trees.map((tr) => ({ id: tr.id, modelCeiling: tr.modelH != null ? +tr.modelH.toFixed(3) : null, nominal: tr.H, cells: tr.cells ? tr.cells.length : null, droppedCells: tr.dropped ?? null })),
    points: points.map((p) => ({ id: p.id, moved: p.cell ? p.cell.moved : null, ...(p.pathCells ? { pathCells: p.pathCells.length } : {}) })),
    fuel: { heatOfCombustion: fuel.heatOfCombustion, yields: fuel.yields, rule: c.fuelRule !== "app" && T.fuel ? "the test fuel's" : "class A, CO2 per joule matched" },
    solverOptions: { ambient: Ta, ...(inert ? { inert } : {}), ...c.solver },
    door: door && door.z ? { cells: door.z.length, width: +door.width.toFixed(3), sill: door.sill != null ? +door.sill.toFixed(3) : null, soffit: door.soffit != null ? +door.soffit.toFixed(3) : null, measured: { width: T.opening?.[0]?.width ?? null, sill: T.door.sill, soffit: T.door.soffit } } : null,
    region,
    walls: inert ? { slots: T.materials.slots.map((m) => m.surface), rhoModel: inert.map((m) => +m.rho.toFixed(2)), solidCells: wallCells, check: wallCheck } : "the solver's stone",
    health,
    series: { from: 1, step: 1, head: headSeries, ...(probes.length ? { probes: probeSeries } : {}), ...(flowSeries ? { doorFlow: flowSeries } : {}) },
    head: { excluded: head.excluded, missing: head.missing },
    ...(probes.length ? { probes: { cells: probes.map((p) => ({ id: p.id, moved: p.cell ? p.cell.moved : null })), notes: probed.notes } } : {}),
    ...(doorFlow ? { doorFlow: { faces: doorFlow.faces.length, area: +doorFlow.area.toFixed(3), plane: +doorFlow.plane.toFixed(3), measuredArea: T.doorFlow.area ?? null } } : {}),
    wallSeconds: +((ran - built) / 1000).toFixed(1),
    buildSeconds: +((built - t0) / 1000).toFixed(1),
    pairs,
  };
}

/** The integral of a [[t, W]] table over [0, t1], J. */
function burnerIntegral(table, t1) {
  let E = 0;
  for (let q = 1; q < table.length; q += 1) {
    const [a, qa] = table[q - 1], [b, qb] = table[q];
    if (a >= t1) break;
    const e = Math.min(b, t1);
    const qe = b > a ? qa + ((qb - qa) * (e - a)) / (b - a) : qb;
    E += 0.5 * (qa + qe) * (e - a);
  }
  // Past its last row the burner holds the last value (flamescapes.js burnerRate), so the record does too
  // (fixed 2026-11-12: before, ATF's tables, ending at 60 s, were integrated to 60 s only; plan 12.10).
  const [tl, ql] = table[table.length - 1];
  if (t1 > tl) E += ql * (t1 - tl);
  return E;
}

// ================================================================== main

/**
 * The statistics both ways (claim protocol addendum, 2026-11-12, 00:28):
 * as pre-registered, every run; and with the rule, the runs whose grid does
 * not represent their test's ventilation left out (lib/represented.mjs,
 * geometry only).
 */
export async function bothWays(report, S, measures, given = null, Rin = null) {
  // The sealed-door rule's module: the snapshot's (its voxeliser pinned with the solver) when a run passes it.
  const R = Rin || (await import("./lib/represented.mjs"));
  // A run passes the tests it read at its start (given); --summary reads the data files.
  const tests = given || {};
  if (!given) for (const d of new Set(report.runs.map((r) => r.dataset))) {
    const data = JSON.parse(readFileSync(path.join(HERE, "data", `${d}.json`), "utf8"));
    for (const T of data.tests) tests[`${d}/${T.id}`] = T;
  }
  const rep = R.representation(report.runs, tests);
  for (const r of report.runs) r.representation = rep[r.id];
  const kept = report.runs.filter((r) => rep[r.id].represented);
  report.stats = summarise(report.runs, S, measures);
  report.statsRule = summarise(kept, S, measures);
  report.notRepresented = report.runs.filter((r) => !rep[r.id].represented).map((r) => r.id).sort();
  report.settings = { ...report.settings, rule: "claim protocol addendum 2026-11-12 00:28: a run whose fire's gas cannot reach the open boundary, though its test had an opening, is left out and reported as not represented on this grid (lib/represented.mjs; sealed by design: " + JSON.stringify(R.SEALED_BY_DESIGN) + ")" };
}

export function summarise(runs, S, measures = {}) {
  const pairs = runs.flatMap((r) => r.pairs);
  const quantities = [...new Set(pairs.map((p) => p.quantity))];
  const cells = [...new Set(runs.map((r) => r.cell))].sort((a, b) => a - b);
  const datasets = [...new Set(runs.map((r) => r.dataset))];
  const byDataset = {}, byQuantity = {};
  for (const q of quantities) {
    const sE = S.SIGMA_E[SIGMA_KEY[q]];
    byQuantity[q] = {};
    for (const h of cells) {
      const per = {};
      for (const d of datasets) {
        const rs = runs.filter((r) => r.dataset === d && r.cell === h);
        if (!rs.length) continue;
        const ps = pairs.filter((p) => p.dataset === d && p.cell === h && p.quantity === q);
        if (!measures[d]?.has(q)) continue;
        const stopped = rs.filter((r) => !r.health.ok).length;
        const s = S.statistics(ps.map((p) => [p.model, p.measured]), sE);
        const t = S.tier(s);
        // The 10-bin weighted statistics (13.3), within the series.
        const sw = S.weightedStatistics(ps.map((p) => [p.model, p.measured]), sE);
        const tw = S.tier(sw);
        per[d] = { ...s, tier: t, weighted: { delta: sw.delta, sigmaM: sw.sigmaM, sigmaEUsed: sw.sigmaEUsed, tier: tw, inBand: tw === "within the calibrated band" && stopped === 0 }, lab: LAB[d] || d, tests: new Set(ps.map((p) => p.test)).size, runs: rs.length, stopped, inBand: t === "within the calibrated band" && stopped === 0 };
        ((byDataset[d] ||= {})[q] ||= {})[h] = per[d];
      }
      if (!Object.keys(per).length) continue;
      const all = pairs.filter((p) => p.cell === h && p.quantity === q && per[p.dataset]);
      const pooled = S.statistics(all.map((p) => [p.model, p.measured]), sE);
      const pooledTier = S.tier(pooled);
      const inBand = Object.entries(per).filter(([, v]) => v.inBand).map(([d]) => d);
      const verdict = inBand.length >= 3 && pooledTier === "within the calibrated band" ? "calibrated" : pooledTier === "within the calibrated band" || pooledTier === "indicative" ? "indicative" : "not calibrated";
      // The claim protocol's rule (CLAIM).
      const okIndicative = (v) => v.stopped === 0 && (v.tier === "within the calibrated band" || v.tier === "indicative");
      const nIndicative = Object.values(per).filter(okIndicative).length;
      const outside = Object.values(per).filter((v) => !okIndicative(v)).length;
      const n = Object.keys(per).length;
      const claim = n >= 3 && inBand.length >= 3 && outside === 0 ? "calibrated" : n >= 2 && nIndicative >= 2 ? "indicative" : "not calibrated";
      // Where the two rules differ, the stricter gives the verdict (plan 12.6; protocol addendum).
      const rank = { calibrated: 2, indicative: 1, "not calibrated": 0 };
      // The same two rules on the weighted statistics (13.3).
      const pooledW = S.weightedStatistics(all.map((p) => [p.model, p.measured]), sE);
      const pooledTierW = S.tier(pooledW);
      const inBandW = Object.entries(per).filter(([, v]) => v.weighted.inBand).map(([d]) => d);
      const verdictW = inBandW.length >= 3 && pooledTierW === "within the calibrated band" ? "calibrated" : pooledTierW === "within the calibrated band" || pooledTierW === "indicative" ? "indicative" : "not calibrated";
      const okIndicativeW = (v) => v.stopped === 0 && (v.weighted.tier === "within the calibrated band" || v.weighted.tier === "indicative");
      const nIndicativeW = Object.values(per).filter(okIndicativeW).length;
      const outsideW = Object.values(per).filter((v) => !okIndicativeW(v)).length;
      const claimW = n >= 3 && inBandW.length >= 3 && outsideW === 0 ? "calibrated" : n >= 2 && nIndicativeW >= 2 ? "indicative" : "not calibrated";
      // The strictest of the four decides (12.6, the protocol's section 4, each unweighted and weighted).
      const final = [verdict, claim, verdictW, claimW].reduce((a, b) => (rank[b] < rank[a] ? b : a));
      const labs = (ds) => [...new Set(ds.map((d) => LAB[d] || d))];
      byQuantity[q][h] = {
        pooled: { ...pooled, tier: pooledTier }, datasets: Object.keys(per), datasetsInBand: inBand, datasetsIndicativeOrBetter: Object.entries(per).filter(([, v]) => okIndicative(v)).map(([d]) => d), verdict, claim,
        weighted: { pooled: { ...pooledW, tier: pooledTierW }, datasetsInBand: inBandW, verdict: verdictW, claim: claimW },
        labs: labs(Object.keys(per)), labsInBand: labs(inBand), labsInBandWeighted: labs(inBandW),
        final, fromOneLab: labs(final === "calibrated" ? inBand : Object.keys(per)).length === 1,
      };
    }
  }
  return { byDataset, byQuantity };
}

export function printTables(report) {
  printStats(report, report.stats, "as pre-registered (every test)");
  if (report.statsRule) {
    console.log(`\nNot represented on their grids (left out under the 2026-11-12 rule): ${report.notRepresented.join(", ") || "none"}`);
    printStats(report, report.statsRule, "with the 2026-11-12 rule (runs that do not represent their test left out)", true);
  }
}

function printStats(report, stats, title, brief = false) {
  console.log(`\n=== ${title}`);
  const f = (x, d = 2) => (x == null || !Number.isFinite(x) ? "-" : x.toFixed(d));
  if (!brief) console.log(`\nRuns: ${report.runs.length} (${report.jobs} at a time; ${report.wallSeconds.toFixed(0)} s wall)`);
  const groups = new Map();
  for (const r of report.runs) {
    const k = `${r.dataset} ${r.cell}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(r);
  }
  if (!brief) console.log("dataset   cell   runs stopped  cells(max)  D*/dx      run s (median, max)");
  for (const [k, rs] of brief ? [] : groups) {
    const ws = rs.map((r) => r.wallSeconds).sort((a, b) => a - b);
    const pi = rs.map((r) => r.plumeIndex);
    console.log(`${k.split(" ")[0].padEnd(9)} ${k.split(" ")[1].padEnd(6)} ${String(rs.length).padStart(4)} ${String(rs.filter((r) => !r.health.ok).length).padStart(7)}  ${String(Math.max(...rs.map((r) => r.cells))).padStart(10)}  ${f(Math.min(...pi), 1)}-${f(Math.max(...pi), 1)}`.padEnd(60) + `${f(ws[ws.length >> 1], 0)}, ${f(ws[ws.length - 1], 0)}`);
  }
  console.log("\nquantity        dataset    cell   n   delta  sigmaM  tier                       | weighted delta sigmaM tier                       | laboratory");
  for (const [d, qs] of Object.entries(stats.byDataset)) {
    for (const [q, hs] of Object.entries(qs)) {
      for (const [h, s] of Object.entries(hs)) {
        const w = s.weighted || {};
        console.log(`${q.padEnd(15)} ${d.padEnd(10)} ${String(h).padEnd(5)} ${String(s.n).padStart(3)}  ${f(s.delta).padStart(6)}  ${f(s.sigmaM).padStart(6)}  ${String(s.tier).padEnd(26)} | ${f(w.delta).padStart(14)} ${f(w.sigmaM).padStart(6)} ${String(w.tier ?? "-").padEnd(26)} | ${s.lab ?? "-"}${s.stopped ? ` (${s.stopped} of ${s.runs} runs stopped)` : ""}`);
      }
    }
  }
  console.log("\nquantity        cell   pooled n  delta  sigmaM | weighted delta sigmaM | series in band (labs)          plan 12.6 / protocol 4 | weighted            final (the strictest)");
  for (const [q, hs] of Object.entries(stats.byQuantity)) {
    for (const [h, v] of Object.entries(hs)) {
      const w = v.weighted || { pooled: {}, verdict: "-", claim: "-", datasetsInBand: [] };
      const labs = v.labsInBand ? ` (${v.labsInBand.length} lab${v.labsInBand.length === 1 ? "" : "s"})` : "";
      console.log(`${q.padEnd(15)} ${String(h).padEnd(5)} ${String(v.pooled.n).padStart(9)}  ${f(v.pooled.delta).padStart(5)}  ${f(v.pooled.sigmaM).padStart(6)} | ${f(w.pooled.delta).padStart(14)} ${f(w.pooled.sigmaM).padStart(6)} | ${`${v.datasetsInBand.length}/${v.datasets.length}${labs}; w ${w.datasetsInBand.length}`.padEnd(30)} ${`${v.verdict} / ${v.claim ?? "-"}`.padEnd(22)} | ${`${w.verdict} / ${w.claim}`.padEnd(19)} ${v.final ?? "-"}${v.fromOneLab ? " (one laboratory)" : ""}`);
    }
  }
}

/**
 * A copy of the solver's modules (src/core/*.js) in a temporary folder, made
 * when the suite starts: every run then uses the same code even if the
 * working tree changes during the suite. Returns the folder and the copies'
 * SHA-256 digests (the first 16 hex digits).
 */
function snapshotCore(rev = null) {
  const dir = mkdtempSync(path.join(os.tmpdir(), "flamescapes-core-"));
  // The repository's layout, so the harness's relative imports (lib/fds.mjs: voxelize.js) reach the copies.
  const core = path.join(dir, "topologicpy-web-frontend", "src", "core");
  mkdirSync(core, { recursive: true });
  const digests = {};
  // From git at rev (--solver-rev), else the working tree. Git's blobs have LF line endings and this working
  // tree CRLF (core.autocrlf), so a file's digest is also given with CRLF turned to LF (lfDigests): equal
  // lfDigests mean the same code.
  const lfDigests = {};
  let resolved = null;
  const files = rev
    ? execFileSync("git", ["ls-tree", "--name-only", rev, "topologicpy-web-frontend/src/core/"], { cwd: ROOT, encoding: "utf8" }).split("\n").map((p) => p.split("/").pop()).filter((f) => f.endsWith(".js"))
    : readdirSync(CORE).filter((f) => f.endsWith(".js"));
  if (rev) resolved = execFileSync("git", ["rev-parse", rev], { cwd: ROOT, encoding: "utf8" }).trim();
  for (const f of files) {
    const buf = rev ? execFileSync("git", ["show", `${rev}:topologicpy-web-frontend/src/core/${f}`], { cwd: ROOT, maxBuffer: 1 << 28 }) : readFileSync(path.join(CORE, f));
    writeFileSync(path.join(core, f), buf);
    digests[f] = createHash("sha256").update(buf).digest("hex").slice(0, 16);
    lfDigests[f] = createHash("sha256").update(Buffer.from(buf.toString("utf8").split("\r\n").join("\n"))).digest("hex").slice(0, 16);
  }
  writeFileSync(path.join(dir, "package.json"), JSON.stringify({ type: "module" }));
  // The harness too (this file and lib/*.mjs), so the workers run the code the suite started with even
  // if the harness is edited during a long run (added 2026-11-12: workers loaded the live file before).
  const harness = path.join(dir, "tools", "calibration");
  mkdirSync(path.join(harness, "lib"), { recursive: true });
  writeFileSync(path.join(harness, "validation-suite.mjs"), readFileSync(fileURLToPath(import.meta.url)));
  for (const f of readdirSync(path.join(HERE, "lib")).filter((f) => f.endsWith(".mjs"))) writeFileSync(path.join(harness, "lib", f), readFileSync(path.join(HERE, "lib", f)));
  return { dir, core, digests, lfDigests, rev: resolved, harness: path.join(harness, "validation-suite.mjs"), harnessDigests: harnessDigests() };
}

/**
 * SHA-256 (first 16 hex digits) of the harness: lib/*.mjs, this file and the
 * reduced data (data/*.json), recorded in every results file; --summary
 * --write refuses to rewrite a file made with a different harness.
 */
function harnessDigests() {
  const out = {};
  const add = (rel) => (out[rel] = createHash("sha256").update(readFileSync(path.join(HERE, rel))).digest("hex").slice(0, 16));
  add("validation-suite.mjs");
  for (const f of readdirSync(path.join(HERE, "lib")).filter((f) => f.endsWith(".mjs")).sort()) add(`lib/${f}`);
  for (const f of readdirSync(path.join(HERE, "data")).filter((f) => f.endsWith(".json")).sort()) add(`data/${f}`);
  return out;
}

function gitState() {
  const git = (...a) => {
    try {
      return execFileSync("git", a, { cwd: ROOT, encoding: "utf8" }).trim();
    } catch {
      return null;
    }
  };
  const files = ["flamescapes.js", "flamescapesRun.js", "flamescapesGpu.js", "voxelize.js", "fireClasses.js"];
  const digests = {};
  for (const f of files) {
    try {
      digests[f] = createHash("sha256").update(readFileSync(path.join(CORE, f))).digest("hex").slice(0, 16);
    } catch {
      digests[f] = null;
    }
  }
  return { head: git("rev-parse", "HEAD"), branch: git("rev-parse", "--abbrev-ref", "HEAD"), status: (git("status", "--porcelain") || "").split("\n").filter(Boolean), solverDiff: (git("diff", "--stat", "HEAD", "--", "topologicpy-web-frontend/src/core") || "").split("\n").filter(Boolean), solverDigests: digests };
}

async function main() {
  const args = process.argv.slice(2);
  const flag = (name, fallback) => {
    const i = args.indexOf(name);
    return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
  };
  const S = await import("./lib/stats.mjs");
  const rev = flag("--solver-rev", null);
  // For --merge and --summary with --solver-rev: the sealed-door rule's voxeliser from that revision.
  const pinnedR = async () => {
    if (!rev) return null;
    const snap = snapshotCore(rev);
    return import(pathToFileURL(path.join(path.dirname(snap.harness), "lib", "represented.mjs")).href);
  };
  if (args.includes("--merge")) {
    // Several results files as one (added 2026-11-12): a run in a later file replaces the same run (dataset/test@cell)
    // in an earlier one; the statistics are recomputed on the union, both ways. Each source's state is kept.
    const files = flag("--merge").split(",");
    const byId = new Map(), measures = {}, sources = [];
    for (const f of files) {
      const r = JSON.parse(readFileSync(f, "utf8"));
      sources.push({ file: f.split("\\").join("/").replace(/^.*benchmark\//, "benchmark/"), generated: r.generated, git: r.git, jobs: r.jobs, wallSeconds: r.wallSeconds, runs: r.runs.length });
      for (const run of r.runs) byId.set(run.id, run);
      for (const [d, qs] of Object.entries(r.settings?.measures || {})) for (const q of qs) (measures[d] ||= new Set()).add(q);
    }
    const report = { generated: new Date().toISOString(), tool: "tools/calibration/validation-suite.mjs --merge", plan: "wiki/research/fire-calibration-plan-2026-11-11.md, sections 12 and 13", sources, solverRev: rev, jobs: "-", wallSeconds: 0, settings: { verdictRule: VERDICT, claimRule: CLAIM, sigmaE: S.SIGMA_E, measures: Object.fromEntries(Object.entries(measures).map(([d, q]) => [d, [...q]])) }, runs: [...byId.values()] };
    await bothWays(report, S, measures, null, await pinnedR());
    printTables(report);
    const out = flag("--json", null);
    if (out) {
      mkdirSync(path.dirname(path.resolve(out)), { recursive: true });
      writeFileSync(path.resolve(out), JSON.stringify(report, null, 1) + "\n");
      console.log(`\nwrote ${out}`);
    }
    return;
  }
  if (args.includes("--summary")) {
    const file = flag("--summary");
    const report = JSON.parse(readFileSync(file, "utf8"));
    const measures = {};
    const stored = report.settings?.measures;
    if (stored) for (const [d, qs] of Object.entries(stored)) measures[d] = new Set(qs);
    else for (const p of report.runs.flatMap((r) => r.pairs)) (measures[p.dataset] ||= new Set()).add(p.quantity);
    report.settings = { ...report.settings, verdictRule: VERDICT, claimRule: CLAIM };
    await bothWays(report, S, measures, null, await pinnedR());
    printTables(report);
    if (args.includes("--write")) {
      const was = report.git?.harnessDigests;
      const now = harnessDigests();
      const differ = was ? Object.keys({ ...was, ...now }).filter((k) => was[k] !== now[k]) : ["(no harness digests recorded)"];
      if (differ.length && !args.includes("--force")) {
        console.log(`\nnot rewritten: the harness differs from the one that made ${file}: ${differ.join(", ")} (--force to rewrite anyway)`);
        return;
      }
      writeFileSync(file, JSON.stringify(report, null, 1) + "\n");
      console.log(`\nrewrote the statistics of ${file}`);
    }
    return;
  }
  const datasets = flag("--datasets", "steckler,nbs,nistnrc,wtc,atf").split(",");
  const cellsArg = flag("--cells", "0.20,0.27,0.34").split(",").map(Number);
  const solver = Object.fromEntries((flag("--solver", "") || "").split(",").filter(Boolean).map((kv) => kv.split("=")).map(([k, v]) => [k, Number(v)]));
  const only = flag("--tests", null);
  const jsonOut = flag("--json", null);
  const snapshot = snapshotCore(rev);
  const git = gitState();
  // The sealed-door rule's module from the snapshot, loaded now (its voxeliser is the snapshot's).
  const Rsnap = await import(pathToFileURL(path.join(path.dirname(snapshot.harness), "lib", "represented.mjs")).href);
  const wallLayers = args.includes("--wall-layers");
  const F = await import(pathToFileURL(path.join(snapshot.core, "flamescapes.js")).href);
  const walls = flag("--walls", "test");
  const fuelRule = flag("--fuel", "test");
  const cases = [];
  const skipped = [];
  for (const d of datasets) {
    const plan = PLAN[d];
    if (!plan) throw new Error(`unknown dataset ${d} (known: ${Object.keys(PLAN).join(", ")})`);
    const data = JSON.parse(readFileSync(path.join(HERE, "data", `${d}.json`), "utf8"));
    for (const h of cellsArg) {
      if ((plan.skipCells || []).some((x) => Math.abs(x - h) < 1e-9) && !args.includes("--all-cells")) {
        skipped.push({ dataset: d, cell: h, why: "too slow on this cell for the suite's time budget (plan section 12); --all-cells runs it" });
        continue;
      }
      for (const T of data.tests) {
        const id = `${d}/${T.id}`;
        if (only && !new RegExp(only).test(id)) continue;
        // --horizon-cap shortens every run (a smoke test only: results are not comparable with the plan's).
        const horizon = Math.min(plan.horizon(T), Number(flag("--horizon-cap", Infinity)));
        // Cost estimate for the queue: the domain's cells (bounding box) x seconds x the fire's speed.
        const bb = T.geometry.meshes.reduce((a, m) => [Math.min(a[0], m[0]), Math.max(a[1], m[1]), Math.min(a[2], m[2]), Math.max(a[3], m[3]), Math.min(a[4], m[4]), Math.max(a[5], m[5])], [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity]);
        const vol = (bb[1] - bb[0] + 2 * h) * (bb[3] - bb[2] + 2 * h) * (bb[5] - bb[4] + 2 * h);
        const cost = (vol / h ** 3) * horizon * Math.max(2, Math.sqrt(T.burner.peakW / 1e5));
        cases.push({ id: `${id}@${h}`, dataset: d, test: T, h, horizon, solver, steadyFrom: plan.steadyFrom, cost, core: snapshot.core, walls, fuelRule, wallLayers });
      }
    }
  }
  cases.sort((a, b) => b.cost - a.cost);
  // What each dataset measures (so a stopped run counts against exactly those quantities).
  const measures = {};
  for (const cs of cases) {
    const m = (measures[cs.dataset] ||= new Set());
    if (cs.test.trees?.length) (m.add("hglTemperature"), m.add("hglDepth"));
    if (cs.test.trees?.some((tr) => tr.head?.length)) m.add("headTemperature");
    if (cs.test.probes?.length) m.add("doorVelocity");
    if (cs.test.profiles?.length) m.add("neutralPlane");
    if (cs.test.doorFlow) m.add("doorMassFlow");
    for (const p of cs.test.points || []) m.add(p.quantity);
    if (cs.test.door && cs.test.measured?.door?.z?.length) (m.add("doorVelocity"), m.add("neutralPlane"));
  }
  // Default 4: a NIST/NRC run on 0.27 m cells holds about 200 MB, and this PC often has 2 to 3 GB free.
  const jobs = Math.max(1, Math.min(Number(flag("--jobs", 4)), cases.length));
  console.log(`${cases.length} runs, ${jobs} at a time${skipped.length ? `; skipped: ${skipped.map((s) => `${s.dataset} at ${s.cell} m`).join(", ")}` : ""}`);
  const t0 = performance.now();
  const runs = new Array(cases.length);
  let next = 0, done = 0;
  await Promise.all(
    Array.from({ length: jobs }, async () => {
      while (next < cases.length) {
        const q = next;
        next += 1;
        runs[q] = await new Promise((resolve) => {
          const w = new Worker(snapshot.harness, { workerData: cases[q] });
          w.once("message", resolve);
          w.once("error", (e) => resolve({ id: cases[q].id, dataset: cases[q].dataset, test: cases[q].test.id, cell: cases[q].h, horizon: cases[q].horizon, health: { ok: false, reason: `error: ${e.message}` }, pairs: [], wallSeconds: 0, cells: 0, plumeIndex: null }));
        });
        done += 1;
        const r = runs[q];
        const brief = r.pairs.filter((p) => ["hglTemperature", "doorVelocity", "ceilingJet"].includes(p.quantity)).slice(0, 3).map((p) => `${p.quantity} ${p.device} ${p.model?.toFixed?.(1)}/${p.measured?.toFixed?.(1)}`).join("; ");
        console.log(`  [${done}/${cases.length}] ${r.id}: ${r.cells} cells, ${r.wallSeconds} s${r.health.ok ? "" : `, STOPPED (${r.health.reason} at ${r.health.time ?? "?"} s)`}; ${brief}`);
      }
    }),
  );
  const report = {
    generated: new Date().toISOString(),
    tool: "tools/calibration/validation-suite.mjs",
    plan: "wiki/research/fire-calibration-plan-2026-11-11.md, section 12 (pre-registered before the baseline)",
    node: process.version,
    cpu: os.cpus()[0]?.model?.trim(),
    jobs,
    wallSeconds: (performance.now() - t0) / 1000,
    git: { ...git, solverRev: snapshot.rev, snapshotDigests: snapshot.digests, snapshotLfDigests: snapshot.lfDigests, harnessDigests: snapshot.harnessDigests },
    solver: { overrides: solver, defaults: { ...F.DEFAULTS }, health: { ...F.HEALTH } },
    settings: { datasets, cells: cellsArg, tests: only, fuel: fuelRule === "app" ? "class A, CO2 per joule matched (diagnostic)" : "each test's own fuel: heat of combustion and yields (plan 12.9)", walls: walls === "stone" ? "the solver's stone (diagnostic)" : `each test's own surfaces from its FDS input, areal heat capacity matched over one cell${wallLayers ? "; and their layers passed to the solver (--wall-layers)" : " (lumped; no layers passed)"}`, plan: Object.fromEntries(Object.entries(PLAN).map(([k, v]) => [k, { horizon: String(v.horizon), steadyFrom: v.steadyFrom ?? null, skipCells: v.skipCells ?? [] }])), skipped, series: SERIES, verdictRule: VERDICT, claimRule: CLAIM, sigmaE: S.SIGMA_E, measures: Object.fromEntries(Object.entries(measures).map(([d, q]) => [d, [...q]])) },
    runs,
  };
  await bothWays(report, S, measures, Object.fromEntries(cases.map((cs) => [`${cs.dataset}/${cs.test.id}`, cs.test])), Rsnap);
  rmSync(snapshot.dir, { recursive: true, force: true });
  printTables(report);
  if (jsonOut) {
    mkdirSync(path.dirname(path.resolve(jsonOut)), { recursive: true });
    writeFileSync(path.resolve(jsonOut), JSON.stringify(report, null, 1) + "\n");
    console.log(`\nwrote ${jsonOut}`);
  }
}

// Run only as a script (head-temperature.mjs imports the summary functions).
if (isMainThread && process.argv[1] && path.resolve(process.argv[1]).toLowerCase() === fileURLToPath(import.meta.url).toLowerCase()) await main();
else if (!isMainThread) parentPort.postMessage(await runCase(workerData));
