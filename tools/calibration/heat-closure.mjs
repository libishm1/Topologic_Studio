/**
 * Does the transport conserve heat? (F6 and proposal F of
 * wiki/research/fire-calibration-plan-2026-11-11.md.) Two checks with the
 * walls' exchange and all radiation switched off, so that nothing but the
 * gas holds or carries heat:
 *
 *   mix:    a sealed stone box, no fire; its lowest metre starts at ambient
 *           + 300 K. As it mixes, the gas's heat (rho V cp (T - Ta)) and its
 *           mass should stay as they were.
 *   burner: the Stage 1a room (5 x 4 x 2.7 m, a 0.9 x 2.0 m door) with a
 *           prescribed 150 kW fire of the app's design-fire kind, ramped over
 *           30 s. The heat released should equal the gas's heat plus what
 *           left the domain (stats.shellHeat: the gas the domain's shell held
 *           when it was reset to ambient).
 *   porous: the burner room with the burner's cells porous (contents of no
 *           mass: slowed by porousVelocity, holding no heat of their own), as
 *           burning contents are. The projection weights a porous cell's faces
 *           (2026-11-12; scaled after it, before, the transport made heat
 *           there: closure 1.27, gas at 2,150 K). Accepted: closure 0.9 to 1.0
 *           and the hottest gas's rise within 10 % of the burner run's, on
 *           each cell where both ran.
 *
 *   node tools/calibration/heat-closure.mjs [--cases mix,burner,porous] [--cells 0.20,0.27,0.34]
 *        [--horizon 60] [--solver conservative=0,expansion=0] [--json out.json]
 */
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { DEFAULTS, NS, SOLID, createSolver, gasDensity, mixtureCp, step, substepsFor } from "../../topologicpy-web-frontend/src/core/flamescapes.js";
import { fuelsFor } from "../../topologicpy-web-frontend/src/core/fireClasses.js";
import * as V from "../../topologicpy-web-frontend/src/core/voxelize.js";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const cases = flag("--cases", "mix,burner,porous").split(",");
const cells = flag("--cells", "0.20,0.27,0.34").split(",").map(Number);
const horizon = Number(flag("--horizon", 60));
const jsonOut = flag("--json", null);
const SOLVER = Object.fromEntries((flag("--solver", "") || "").split(",").filter(Boolean).map((kv) => kv.split("=")).map(([k, v]) => [k, Number(v)]));
// No wall exchange, no radiation: the gas alone holds the heat.
const ADIABATIC = { wallConvection: 0, emitAbove: 1e9 };
const TA = DEFAULTS.ambient;
const INERT_FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };

function box(x0, y0, z0, x1, y1, z1) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material: V.MATERIAL.STONE };
}

function gridOf(lo, hi, h) {
  const n = [0, 1, 2].map((d) => Math.ceil((hi[d] - lo[d]) / h - 1e-6));
  return { nx: n[0], ny: n[1], nz: n[2], h, origin: lo, axes: [0, 1, 2], upAxis: "y", cells: n[0] * n[1] * n[2] };
}

/**
 * The gas's heat above ambient and its mass, over the open cells not on the
 * domain's shell, or over those `keep` accepts.
 */
function inventory(sim, keep = null) {
  const { nx, ny, nz } = sim.grid;
  let heat = 0, mass = 0;
  for (let c = 0; c < sim.N; c += 1) {
    if (sim.kind[c] === SOLID || (keep && !keep(c))) continue;
    const i = c % nx, j = Math.floor(c / nx) % ny, k = Math.floor(c / (nx * ny));
    if (i === 0 || k === 0 || i === nx - 1 || k === nz - 1 || j === ny - 1) continue;
    const rho = gasDensity(sim.T[c], sim.Y, c * NS);
    heat += rho * sim.V * mixtureCp(sim.Y, c * NS) * (sim.T[c] - TA);
    mass += rho * sim.V;
  }
  return { heat, mass };
}

function run(sim, h, until, each) {
  let substeps = substepsFor(sim.P, h, 0, sim.P.dt), next = 1;
  while (sim.time < until - 1e-9) {
    step(sim, Math.min(sim.P.dt / substeps, next - sim.time));
    if (sim.time >= next - 1e-9) {
      let s = 0;
      for (const a of [sim.u, sim.v, sim.w]) for (const x of a) s = Math.max(s, Math.abs(x));
      substeps = substepsFor(sim.P, h, s, sim.P.dt);
      each?.(Math.round(sim.time));
      next += 1;
    }
  }
}

function mix(h) {
  const W = 3, H = 2.5, D = 3, t = 0.4;
  const boxes = [box(-t, -t, -t, W + t, 0, D + t), box(-t, H, -t, W + t, H + t, D + t), box(-t, 0, -t, W + t, H, 0), box(-t, 0, D, W + t, H, D + t), box(-t, 0, 0, 0, H, D), box(W, 0, 0, W + t, H, D)];
  const lo = [-t - h, -t, -t - h], hi = [W + t + h, H + t + h, D + t + h];
  const grid = gridOf(lo, hi, h);
  const fraction = V.volumeFractions(grid, boxes, { rays: V.raysFor(grid) });
  const solid = V.occupancy(grid, fraction);
  const scene = { grid, fraction, solid, contents: { cells: [], masses: [] }, ignition: null, pilot: null, fuels: { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL } };
  const sim = createSolver(scene, { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL, seed: 1, ...ADIABATIC, ...SOLVER });
  // The box's inside only: the air round it is open to the domain's sides.
  const inside = (c) => {
    const i = c % grid.nx, j = Math.floor(c / grid.nx) % grid.ny, k = Math.floor(c / (grid.nx * grid.ny));
    const x = lo[0] + (i + 0.5) * h, y = lo[1] + (j + 0.5) * h, z = lo[2] + (k + 0.5) * h;
    return x > 0 && x < W && y > 0 && y < H && z > 0 && z < D;
  };
  for (let c = 0; c < sim.N; c += 1) {
    if (sim.kind[c] === SOLID || !inside(c)) continue;
    const j = Math.floor(c / grid.nx) % grid.ny;
    if (lo[1] + (j + 0.5) * h < 1.0) sim.T[c] = TA + 300;
  }
  const start = inventory(sim, inside);
  const at = {};
  run(sim, h, horizon, (s) => {
    if (s % 30 === 0 || horizon <= 10) {
      const now = inventory(sim, inside);
      at[s] = { heat: +(now.heat / start.heat).toFixed(3), mass: +(now.mass / start.mass).toFixed(3) };
    }
  });
  return { case: "mix", cell: h, startHeatMJ: +(start.heat / 1e6).toFixed(3), ratios: at };
}

/** The burner room; with `porous`, its burner's cells porous (contents of no mass). */
function burner(h, porous = false) {
  const W = 5, D = 4, H = 2.7, t = 0.2, door = [1.5, 2.4], doorH = 2.0, Q = 150e3, RAMP = 30;
  const boxes = [box(-t, -t, -t, W + t, 0, D + t), box(-t, H, -t, W + t, H + t, D + t), box(-t, 0, -t, W + t, H, 0), box(-t, 0, D, W + t, H, D + t), box(-t, 0, 0, 0, H, D), box(W, 0, 0, W + t, H, door[0]), box(W, 0, door[1], W + t, H, D), box(W, doorH, door[0], W + t, H, door[1])];
  const lo = [-t, -t, -t], hi = [W + t + 1.0, H + t + 0.2, D + t];
  const grid = gridOf(lo, hi, h);
  const fraction = V.volumeFractions(grid, boxes, { rays: V.raysFor(grid) });
  const solid = V.occupancy(grid, fraction);
  const at = (i, j, k) => i + grid.nx * (j + grid.ny * k);
  const cellsOf = [];
  for (let k = 0; k < grid.nz; k += 1) for (let i = 0; i < grid.nx; i += 1) {
    const x = lo[0] + (i + 0.5) * h, z = lo[2] + (k + 0.5) * h;
    if (Math.abs(x - 2.5) < 0.3 && Math.abs(z - 2) < 0.3) {
      let j = Math.floor((1e-6 - lo[1]) / h);
      while (solid[at(i, j, k)]) j += 1;
      for (let l = 0; l < 4; l += 1) if (!solid[at(i, j + l, k)]) cellsOf.push(at(i, j + l, k));
    }
  }
  // Contents of no mass make a cell porous with no fuel, water or heat capacity of its own: it burns nothing and holds no heat.
  const contents = porous ? { cells: cellsOf, masses: cellsOf.map(() => 0) } : { cells: [], masses: [] };
  const scene = {
    grid, fraction, solid, contents, ignition: null, pilot: null,
    burner: { cells: cellsOf, hrr: [[0, 0], [RAMP, Q], [1e6, Q]], radiantFraction: 0, fuel: fuelsFor("A").ignitionFuel },
    fuels: { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL },
  };
  const sim = createSolver(scene, { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL, seed: 1, ...ADIABATIC, ...SOLVER });
  const start = inventory(sim);
  const rows = {};
  let peakT = sim.P.ambient;
  run(sim, h, horizon, (s) => {
    peakT = Math.max(peakT, sim.stats.maxT);
    if (s % 30 === 0) {
      const now = inventory(sim);
      const put = sim.stats.released + sim.stats.pilotHeat;
      rows[s] = {
        releasedMJ: +(put / 1e6).toFixed(3),
        gas: +((now.heat - start.heat) / put).toFixed(3),
        left: +(sim.stats.shellHeat / put).toFixed(3),
        closure: +((now.heat - start.heat + sim.stats.shellHeat) / put).toFixed(3),
        maxT: Math.round(sim.stats.maxT),
        // Where a closure short of 1 can go: the mass the transport's floor added, and faces it clamped to a Courant number of 1.
        floorKg: +(sim.stats.massFloor ?? 0).toFixed(4),
        clampedFaces: sim.stats.courantClamped ?? 0,
      };
    }
  });
  const porousCells = porous ? cellsOf.filter((c) => sim.kind[c] !== SOLID).length : 0;
  // The hottest gas over the run, read each second.
  return { case: porous ? "porous" : "burner", cell: h, ...(porous ? { porousCells } : {}), peakT: Math.round(peakT), rows };
}

const results = [];
for (const h of cells) {
  for (const name of cases) {
    const t0 = performance.now();
    const r = name === "mix" ? mix(h) : burner(h, name === "porous");
    r.wallSeconds = +((performance.now() - t0) / 1000).toFixed(1);
    results.push(r);
    console.log(JSON.stringify(r));
  }
}
// The porous burner against the plain one, on each cell where both ran: the
// closure at the horizon within 0.9 to 1.0, and the hottest gas's rise over
// ambient within 10 % of the plain burner's.
for (const h of cells) {
  const plain = results.find((r) => r.case === "burner" && r.cell === h);
  const por = results.find((r) => r.case === "porous" && r.cell === h);
  if (!plain || !por) continue;
  const closure = por.rows[Math.max(...Object.keys(por.rows).map(Number))]?.closure;
  const rise = (por.peakT - TA) / (plain.peakT - TA);
  const ok = closure >= 0.9 && closure <= 1.0 && Math.abs(rise - 1) <= 0.1;
  const verdict = { cell: h, closure, hottestRise: +rise.toFixed(3), porousPeakK: por.peakT, plainPeakK: plain.peakT, pass: ok };
  results.push({ case: "porous verdict", ...verdict });
  console.log(`porous burner, cell ${h} m: closure ${closure} (0.9 to 1.0), hottest gas's rise ${verdict.hottestRise} of the plain burner's (${por.peakT} K against ${plain.peakT} K, within 10 %): ${ok ? "passes" : "fails"}`);
}
if (jsonOut) {
  await writeFile(path.resolve(jsonOut), JSON.stringify({ generated: new Date().toISOString(), solver: SOLVER, adiabatic: ADIABATIC, results }, null, 1));
  console.log(`wrote ${jsonOut}`);
}
