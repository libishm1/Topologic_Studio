/**
 * Proposal D of the fire calibration (wiki/research/fire-calibration-plan-2026-11-11.md):
 * where the heat goes in the test room of Stage 1a (5 x 4 x 2.7 m of stone,
 * a 0.9 x 2.0 m door, a prescribed fire of the app's design-fire kind).
 *
 * Read every second, and summed over the window from 60 s to the horizon:
 *   - the fire's heat (the burner's release);
 *   - the change of the heat stored in the room's gas (rho V cp (T - Ta),
 *     cp 1005, the health guard's) and in the walls (the solids' capacity
 *     times their rise, inertC (Ts - Ta));
 *   - the enthalpy carried out of the door, rho cp (T - Ta) u A over the
 *     faces of the plane just outside the wall, and the mass that leaves;
 *   - the rest (radiation leaving through the door and the domain's sides,
 *     heat outside the room).
 *
 * The decision, fixed before the run: MQH (McCaffrey, Quintiere and
 * Harkleroad 1981) says what the layer's rise should be; at that rise the
 * door carries m cp dT_MQH out (m, the solver's own outflow), and the walls
 * must take the remainder of the fire's heat. The walls are "short of heat"
 * when the solver's walls take less than half that remainder; only then is a
 * wall surface model proposed for building.
 *
 *   node tools/calibration/energy-budget.mjs [--cells 0.20,0.27,0.34] [--fires 150,400]
 *        [--horizon 120] [--solver vorticity=4] [--json out.json]
 */
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { DEFAULTS, INERT, SOLID, createSolver, flameHeight, step, substepsFor } from "../../topologicpy-web-frontend/src/core/flamescapes.js";
import { fuelsFor } from "../../topologicpy-web-frontend/src/core/fireClasses.js";
import * as V from "../../topologicpy-web-frontend/src/core/voxelize.js";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const cells = flag("--cells", "0.20,0.27,0.34").split(",").map(Number);
const fires = flag("--fires", "150,400").split(",").map((q) => Number(q) * 1000);
const horizon = Number(flag("--horizon", 120));
const jsonOut = flag("--json", null);
const SOLVER = Object.fromEntries((flag("--solver", "") || "").split(",").filter(Boolean).map((kv) => kv.split("=")).map(([k, v]) => [k, Number(v)]));
const FROM = 60, RAMP = 30, CP = 1005, TA = DEFAULTS.ambient;
const INERT_FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };

function box(x0, y0, z0, x1, y1, z1) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material: V.MATERIAL.STONE };
}

function budget(h, Q) {
  const W = 5, D = 4, H = 2.7, t = 0.2, door = [1.5, 2.4], doorH = 2.0;
  const boxes = [box(-t, -t, -t, W + t, 0, D + t), box(-t, H, -t, W + t, H + t, D + t), box(-t, 0, -t, W + t, H, 0), box(-t, 0, D, W + t, H, D + t), box(-t, 0, 0, 0, H, D), box(W, 0, 0, W + t, H, door[0]), box(W, 0, door[1], W + t, H, D), box(W, doorH, door[0], W + t, H, door[1])];
  const lo = [-t, -t, -t], hi = [W + t + 1.0, H + t + 0.2, D + t];
  const n = [0, 1, 2].map((d) => Math.ceil((hi[d] - lo[d]) / h - 1e-6));
  const grid = { nx: n[0], ny: n[1], nz: n[2], h, origin: lo, axes: [0, 1, 2], upAxis: "y", cells: n[0] * n[1] * n[2] };
  const fraction = V.volumeFractions(grid, boxes, { rays: V.raysFor(grid) });
  const solid = V.occupancy(grid, fraction);
  const at = (i, j, k) => i + grid.nx * (j + grid.ny * k);
  const uAt = (i, j, k) => i + (grid.nx + 1) * (j + grid.ny * k);
  // The burner: open cells over a 0.6 m square at the room's middle, a Heskestad column.
  const burnerCells = [];
  const layers = Math.max(1, Math.min(Math.floor(H / h) - 2, Math.round(flameHeight(Q, Math.sqrt((4 * 0.36) / Math.PI)) / h)));
  for (let k = 0; k < grid.nz; k += 1) for (let i = 0; i < grid.nx; i += 1) {
    const x = lo[0] + (i + 0.5) * h, z = lo[2] + (k + 0.5) * h;
    if (Math.abs(x - 2.5) < 0.3 && Math.abs(z - 2) < 0.3) {
      let j = Math.floor((1e-6 - lo[1]) / h);
      while (solid[at(i, j, k)]) j += 1;
      for (let l = 0; l < layers; l += 1) if (!solid[at(i, j + l, k)]) burnerCells.push(at(i, j + l, k));
    }
  }
  const scene = {
    grid, fraction, solid, contents: { cells: [], masses: [] }, ignition: null, pilot: null,
    burner: { cells: burnerCells, hrr: [[0, 0], [RAMP, Q], [1e6, Q]], radiantFraction: 0, fuel: fuelsFor("A").ignitionFuel },
    fuels: { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL },
  };
  const sim = createSolver(scene, { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL, seed: 1, ...SOLVER });
  // The room's gas cells, the walls' solid cells, and the door plane's faces just outside the wall.
  const room = [], walls = [], plane = [];
  for (let k = 0; k < grid.nz; k += 1) for (let j = 0; j < grid.ny; j += 1) for (let i = 0; i < grid.nx; i += 1) {
    const c = at(i, j, k);
    const x = lo[0] + (i + 0.5) * h, y = lo[1] + (j + 0.5) * h, z = lo[2] + (k + 0.5) * h;
    if (sim.kind[c] === SOLID) {
      if (sim.inertC[c] > 0) walls.push(c);
    } else if (x > 0 && x < W && y > 0 && y < H && z > 0 && z < D) room.push(c);
  }
  const iPlane = Math.ceil((W + t - lo[0]) / h);
  for (let k = 0; k < grid.nz; k += 1) for (let j = 0; j < grid.ny; j += 1) {
    const y = lo[1] + (j + 0.5) * h, z = lo[2] + (k + 0.5) * h;
    if (y > 0 && y < doorH + h && z > door[0] - h && z < door[1] + h && sim.kind[at(iPlane - 1, j, k)] !== SOLID) plane.push([j, k]);
  }
  const gasHeat = () => room.reduce((s, c) => s + sim.rho[c] * sim.V * CP * (sim.T[c] - TA), 0);
  const wallHeat = () => walls.reduce((s, c) => s + sim.inertC[c] * (sim.Ts[c] - TA), 0);
  const A = h * h;
  // Enthalpy and mass out through the plane this instant (W, kg/s), upwind.
  const doorFlux = () => {
    let H_ = 0, m = 0;
    for (const [j, k] of plane) {
      const u = sim.u[uAt(iPlane, j, k)];
      const c = u > 0 ? at(iPlane - 1, j, k) : at(iPlane, j, k);
      if (u > 0) {
        H_ += sim.rho[c] * CP * (sim.T[c] - TA) * u * A;
        m += sim.rho[c] * u * A;
      }
    }
    return { H: H_, m };
  };
  const t0 = performance.now();
  let next = 1, speed = 0, substeps = substepsFor(sim.P, h, 0, sim.P.dt);
  let at60 = null, outDoor = 0, mass = 0, samples = 0;
  while (sim.time < horizon - 1e-9) {
    step(sim, Math.min(sim.P.dt / substeps, next - sim.time));
    if (sim.time >= next - 1e-9) {
      let s = 0;
      for (const a of [sim.u, sim.v, sim.w]) for (const x of a) s = Math.max(s, Math.abs(x));
      speed = s;
      substeps = substepsFor(sim.P, h, speed, sim.P.dt);
      if (Math.round(sim.time) === FROM) at60 = { released: sim.stats.released, gas: gasHeat(), walls: wallHeat(), radiated: sim.stats.radiated };
      if (sim.time > FROM + 1e-9) {
        const f = doorFlux();
        outDoor += f.H; // one second each
        mass += f.m;
        samples += 1;
      }
      next += 1;
    }
  }
  const end = { released: sim.stats.released, gas: gasHeat(), walls: wallHeat(), radiated: sim.stats.radiated };
  const span = horizon - FROM;
  const fire = end.released - at60.released;
  const dGas = end.gas - at60.gas, dWalls = end.walls - at60.walls;
  const rest = fire - dGas - dWalls - outDoor;
  const mOut = mass / samples; // kg/s
  // MQH's rise at the window's middle, and the walls' share it implies with the solver's own outflow.
  const tMid = (FROM + horizon) / 2 - RAMP / 2;
  const { k, rho, c } = INERT[0];
  const AT = 2 * (W * D) + 2 * (W * H) + 2 * (D * H) - 0.9 * doorH;
  const hk = Math.sqrt((k * rho * c) / tMid) / 1000;
  const dT = 6.85 * ((Q / 1000) ** 2 / (0.9 * doorH * Math.sqrt(doorH) * hk * AT)) ** (1 / 3);
  const mqhWalls = Math.max(0, Q - mOut * CP * dT) / Q;
  const solverWalls = dWalls / fire;
  const share = (x) => +(x / fire).toFixed(3);
  return {
    cell: h, Q_kW: Q / 1000, window: [FROM, horizon], wallSeconds: +((performance.now() - t0) / 1000).toFixed(1),
    fireMJ: +(fire / 1e6).toFixed(2),
    shares: { gas: share(dGas), walls: share(dWalls), door: share(outDoor), rest: share(rest) },
    radiatedMJ: +((end.radiated - at60.radiated) / 1e6).toFixed(2),
    doorOutflowKgPerS: +mOut.toFixed(3),
    meanWallFluxKwPerM2: +(dWalls / span / AT / 1000).toFixed(3),
    mqh: { riseK: +dT.toFixed(1), wallShare: +mqhWalls.toFixed(3) },
    wallsShort: solverWalls < 0.5 * mqhWalls,
  };
}

const rows = [];
for (const h of cells) {
  for (const Q of fires) {
    const r = budget(h, Q);
    rows.push(r);
    console.log(`${Q / 1000} kW, cell ${h} m (${r.wallSeconds} s): fire ${r.fireMJ} MJ over ${FROM} to ${horizon} s; gas ${r.shares.gas}, walls ${r.shares.walls}, door ${r.shares.door}, rest ${r.shares.rest}; door outflow ${r.doorOutflowKgPerS} kg/s; MQH +${r.mqh.riseK} K implies walls ${r.mqh.wallShare}; walls short: ${r.wallsShort}`);
  }
}
if (jsonOut) {
  await writeFile(path.resolve(jsonOut), JSON.stringify({ generated: new Date().toISOString(), solver: SOLVER, rows }, null, 1));
  console.log(`wrote ${jsonOut}`);
}
