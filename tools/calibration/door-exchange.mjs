/**
 * The exchange flow through a door between a hot and a cold sealed box,
 * against the orifice formula, and its neutral plane against the
 * hydrostatic prediction: the check of the calibration plan's 13.4 item 3
 * (openings narrower than a cell), which uses no measured data. Fixed before
 * the first run (2026-11-12):
 *
 *   Two boxes 3.0 x 2.4 x 3.0 m on 0.2 m cells, side by side along x, sealed
 *   (their walls are the domain's outermost cells), a wall one cell thick
 *   between them with a door 2.0 m high from the floor, centred. The hot box
 *   starts at Th (400 K and 600 K; both must pass), the other at ambient;
 *   gravity on, radiation and the walls' exchange off, no soot deposition.
 *   Read every 0.05 s from 2 to 6 s (the exchange set up, before the boxes'
 *   layers come back to the door): the volume flowing out of the hot box
 *   through the door's faces on the cold side, and the door's mean velocity
 *   profile, whose sign change is the neutral plane.
 *
 *   Prediction, with the boxes' mean gas temperatures over the same reads:
 *   with no heat source the gas mixes at constant pressure and keeps its
 *   volume, so the door's volume flows balance and the neutral plane is at
 *   z_n = H / (1 + (Tc / Th)^(1/3)); the hot outflow is (2/3) Cd w
 *   sqrt(2 g (Th - Tc) / Tc) (H - z_n)^(3/2), Cd 0.68. Each within 10 %.
 *
 * --doors gives the door's width in cells. The solver closes an opening
 * narrower than a cell (voxelize.js occupancy seals a cell its wall fills to
 * THIN_FRACTION), so a width that is not a whole number of cells (2.5: two
 * cells and a half) needs a solver with apertures, the face fractions of the
 * local prototype branch flamescapes/openings-prototype; this one refuses it.
 * Beside the orifice formula, the tool fits Cd to the solver's own door
 * dynamics, whose projection divides the pressure gradient by the ambient
 * density: both sides' speeds then go as sqrt(2 g (Th - Tc) / T_ambient), and
 * one Cd should fit every case if that is what sets the flow.
 *
 *   node tools/calibration/door-exchange.mjs [--hot 400,600] [--doors 2,3] [--json out.json]
 */
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { DEFAULTS, NS, SOLID, createSolver, gasDensity, step, substepsFor } from "../../topologicpy-web-frontend/src/core/flamescapes.js";
import { MATERIALS } from "../../topologicpy-web-frontend/src/core/voxelize.js";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const HOT = flag("--hot", "400,600").split(",").map(Number);
const DOORS = flag("--doors", "2,3").split(",").map(Number);
const jsonOut = flag("--json", null);
const FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };
const TA = DEFAULTS.ambient, G = 9.81, CD = 0.68, H = 2.0, h = 0.2;

/**
 * The two boxes with a door `cells` cells wide; the hot box (x below the
 * dividing wall) at Th. A door that is not a whole number of cells has its
 * last column open by the rest (PROTOTYPE, plan 13.4 item 3: scene.apertures).
 */
function boxes(cells, Th) {
  const nx = 33, ny = 14, nz = 17, N = nx * ny * nz, wall = 16;
  const grid = { nx, ny, nz, h, origin: [0, 0, 0], axes: [0, 1, 2], upAxis: "y", cells: N };
  const fraction = new Float32Array(N * MATERIALS);
  const solid = new Uint8Array(N);
  const full = Math.floor(cells + 1e-9), part = cells - full;
  const span = full + (part > 1e-9 ? 1 : 0);
  const k0 = Math.floor((nz - span) / 2);
  const apertures = { cells: [], open: [] };
  for (let k = 0; k < nz; k += 1) for (let j = 0; j < ny; j += 1) for (let i = 0; i < nx; i += 1) {
    const c = i + nx * (j + ny * k);
    const shell = i === 0 || j === 0 || k === 0 || i === nx - 1 || j === ny - 1 || k === nz - 1;
    const door = i === wall && j >= 1 && j <= H / h && k >= k0 && k < k0 + span;
    if (door && k === k0 + full) {
      apertures.cells.push(c);
      apertures.open.push(part);
    }
    if (shell || (i === wall && !door)) {
      solid[c] = 1;
      fraction[c * MATERIALS] = 1;
    }
  }
  const scene = { grid, fraction, solid, contents: { cells: [], masses: [] }, apertures };
  const sim = createSolver(scene, { contentsFuel: FUEL, ignitionFuel: FUEL, emitAbove: 1e9, wallConvection: 0, sootDeposition: 0 });
  for (let c = 0; c < N; c += 1) {
    if (solid[c] || c % nx >= wall) continue;
    sim.T[c] = Th;
    sim.rho[c] = gasDensity(Th, sim.Y, c * NS);
  }
  if (apertures.cells.length && !sim.open) throw new Error("this solver has no apertures: doors must be whole cells");
  return { sim, nx, ny, k0, wall, span };
}

function run(cells, Th) {
  const { sim, nx, ny, k0, wall, span } = boxes(cells, Th);
  const w = cells * h, rows = H / h;
  const uAt = (i, j, k) => i + (nx + 1) * (j + ny * k);
  const profile = new Float64Array(rows);
  let out = 0, inflow = 0, reads = 0, hot = 0, cold = 0, next = 2;
  while (sim.time < 6 - 1e-9) {
    let speed = 0;
    for (const a of [sim.u, sim.v, sim.w]) for (const x of a) speed = Math.max(speed, Math.abs(x));
    step(sim, DEFAULTS.dt / substepsFor(sim.P, h, speed, DEFAULTS.dt));
    if (sim.time < next - 1e-9) continue;
    next += 0.05;
    reads += 1;
    for (let j = 1; j <= rows; j += 1) {
      for (let k = k0; k < k0 + span; k += 1) {
        // The door's face on the cold side: the flow through it, its open fraction of its velocity.
        const f = uAt(wall + 1, j, k);
        const u = sim.u[f] * (sim.open ? sim.open.u[f] : 1);
        profile[j - 1] += u / cells;
        if (u > 0) out += u * h * h;
        else inflow -= u * h * h;
      }
    }
    let th = 0, nh = 0, tc = 0, nc = 0;
    for (let c = 0; c < sim.N; c += 1) {
      if (sim.kind[c] === SOLID) continue;
      if (c % nx < wall) {
        th += sim.T[c];
        nh += 1;
      } else if (c % nx > wall) {
        tc += sim.T[c];
        nc += 1;
      }
    }
    hot += th / nh;
    cold += tc / nc;
  }
  out /= reads;
  inflow /= reads;
  hot /= reads;
  cold /= reads;
  for (let q = 0; q < rows; q += 1) profile[q] /= reads;
  // The neutral plane: where the mean profile turns from inflow (low) to outflow (high), between face centres.
  let zn = NaN;
  for (let q = 0; q + 1 < rows; q += 1) if (profile[q] <= 0 && profile[q + 1] > 0) zn = (q + 0.5 + -profile[q] / (profile[q + 1] - profile[q])) * h;
  const znPredicted = H / (1 + (cold / hot) ** (1 / 3));
  const predicted = (2 / 3) * CD * w * Math.sqrt((2 * G * (hot - cold)) / cold) * (H - znPredicted) ** 1.5;
  // Cd fitted to the solver's own dynamics: speeds sqrt(2 g (Th - Tc) / T_ambient) on both sides, the neutral plane at mid-height.
  const ambientCd = out / ((2 / 3) * w * Math.sqrt((2 * G * (hot - cold)) / TA) * (H / 2) ** 1.5);
  const flowError = out / predicted - 1, planeError = zn / znPredicted - 1;
  return {
    door: +w.toFixed(2), hotStart: Th, boxes: [+hot.toFixed(1), +cold.toFixed(1)],
    outflow: +out.toFixed(4), inflow: +inflow.toFixed(4), predicted: +predicted.toFixed(4), flowError: +flowError.toFixed(4),
    neutralPlane: +zn.toFixed(3), neutralPredicted: +znPredicted.toFixed(3), planeError: +planeError.toFixed(4),
    cdOrifice: +((CD * out) / predicted).toFixed(3), cdAmbient: +ambientCd.toFixed(3),
    pass: Math.abs(flowError) <= 0.1 && Math.abs(planeError) <= 0.1,
  };
}

const results = [];
for (const Th of HOT) {
  for (const cells of DOORS) {
    const r = run(cells, Th);
    results.push(r);
    console.log(`door ${r.door} m, hot box ${Th} K (means ${r.boxes[0]} / ${r.boxes[1]} K): outflow ${r.outflow} m3/s against ${r.predicted} (${(100 * r.flowError).toFixed(1)} %), inflow ${r.inflow}; neutral plane ${r.neutralPlane} m against ${r.neutralPredicted} (${(100 * r.planeError).toFixed(1)} %); Cd fitted to the orifice formula ${r.cdOrifice}, to the ambient-density dynamics ${r.cdAmbient}: ${r.pass ? "passes" : "fails"}`);
  }
}
console.log(`13.4 item 3's check, every case within 10 %: ${results.every((r) => r.pass) ? "passes" : "fails"}`);
if (jsonOut) {
  await writeFile(path.resolve(jsonOut), JSON.stringify({ generated: new Date().toISOString(), results }, null, 1));
  console.log(`wrote ${jsonOut}`);
}
