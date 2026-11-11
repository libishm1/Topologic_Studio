/**
 * Stage 1a of the fire calibration (wiki/research/fire-calibration-plan-2026-11-11.md):
 * the Flamescapes solver's transport under a prescribed heat release, against
 * three classical correlations, on the app's cells (0.20, 0.27, 0.34 m), CPU.
 *
 * - plume:   a burner in the open, the time-averaged centreline temperature
 *            rise and upward velocity against McCaffrey (1979), his Table 1
 *            and the velocity fits of the same report.
 * - room:    the 5 x 4 x 2.7 m stone test room with a 0.9 x 2.0 m door,
 *            the upper layer's rise (Janssens and Tran 1992 from three
 *            trees) against McCaffrey, Quintiere and Harkleroad (1981).
 * - ceiling: a burner under an unconfined ceiling 3 m above it, the gas
 *            rise in the top cell against Alpert (1972) along a radius.
 *
 * The heat release is prescribed (flamescapes.js createBurner), ramped over
 * RAMP s to a constant value, its heat into a Heskestad flame-height column
 * over the burner. By default the burner is the app's design fire's kind
 * (plan principle P2, the model as used): the class A item's sooty fuel and
 * no radiant fraction, the hot sooty cells radiating as the solver's gas
 * does. `--burner methane` gives the sprinkler validation's kind (soot-free
 * methane, 0.2 radiant along rays) for comparison. Nothing is tuned: DEFAULTS
 * as the app runs them. Every run is small (at most about 30k cells, 120 s).
 * Each run is watched: the largest speed each second (a run past 50 m/s or
 * with a non-finite value is marked diverged and its readings set aside),
 * and for the plume the share of samples whose centreline flow went down
 * (a plume blown off its axis).
 *
 *   node tools/calibration/transport-checks.mjs [--cases plume,room,ceiling]
 *        [--cells 0.20,0.27,0.34] [--horizon 120] [--burner app|methane] [--json out.json]
 */
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { DEFAULTS, INERT, createSolver, flameHeight, plumeRise, step } from "../../topologicpy-web-frontend/src/core/flamescapes.js";
import { fuelsFor } from "../../topologicpy-web-frontend/src/core/fireClasses.js";
import * as V from "../../topologicpy-web-frontend/src/core/voxelize.js";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const cases = flag("--cases", "plume,room,ceiling").split(",");
const cells = flag("--cells", "0.20,0.27,0.34").split(",").map(Number);
const horizon = Number(flag("--horizon", 120));
const jsonOut = flag("--json", null);
const AVERAGE_FROM = Number(flag("--average-from", 60)); // s: the time average starts here
/**
 * s: the burner rises linearly to its heat release over this time, then
 * holds. A step from nothing to 500 kW in one 0.1 s step diverged in the
 * open (velocities past 1e4 m/s within 8 s at 0.34 m; stable at dt 0.05 s,
 * at 150 kW, or with this ramp): recorded as finding F1 in the plan. The
 * app's design fires grow as t^2 and never step.
 */
const RAMP = Number(flag("--ramp", 30));
const BURNER = flag("--burner", "app");
/**
 * Diagnostics only, never the app's values: `--solver vorticity=0,dt=0.05`
 * overrides solver options, and `--diagnostic-into file.json` appends the
 * runs to an earlier output's `diagnostics` instead of writing verdicts.
 */
const SOLVER = Object.fromEntries((flag("--solver", "") || "").split(",").filter(Boolean).map((kv) => kv.split("=")).map(([k, v]) => [k, Number(v)]));
const DIAGNOSTIC_INTO = flag("--diagnostic-into", null);
const TA = DEFAULTS.ambient;
const INERT_FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };
/** A run whose largest gas speed passes this (m/s), or turns non-finite, has diverged. */
const DIVERGED_SPEED = 50;

/** The experimental relative standard deviations of the FDS Validation Guide's Table 4.1. */
const SIGMA_E = { temperature: 0.07, velocity: 0.08 };

function box(x0, y0, z0, x1, y1, z1) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material: V.MATERIAL.STONE };
}

/** A grid over [lo, hi] (x, y up, z) at cell h, its solids from the boxes. */
function gridOver(lo, hi, h, boxes) {
  const n = [0, 1, 2].map((d) => Math.ceil((hi[d] - lo[d]) / h - 1e-6));
  const grid = { nx: n[0], ny: n[1], nz: n[2], h, origin: lo.slice(), axes: [0, 1, 2], upAxis: "y", cells: n[0] * n[1] * n[2] };
  const fraction = V.volumeFractions(grid, boxes, { rays: V.raysFor(grid) });
  const solid = V.occupancy(grid, fraction);
  return { grid, fraction, solid };
}

const at = (g, i, j, k) => i + g.nx * (j + g.ny * k);
const vAt = (g, i, j, k) => i + g.nx * (j + (g.ny + 1) * k);
const cellOf = (g, x, y, z) => [Math.floor((x - g.origin[0]) / g.h), Math.floor((y - g.origin[1]) / g.h), Math.floor((z - g.origin[2]) / g.h)];

/**
 * A square burner of side `side` (m) centred at (cx, cz) on the floor at y0:
 * the open cells over it (centres inside, else the centre's column), a
 * column as tall as Heskestad's flame for Q (W).
 */
function burnerCells(g, solid, { cx, cz, side, y0, Q, ceiling = Infinity }) {
  const cols = [];
  for (let k = 0; k < g.nz; k += 1) {
    for (let i = 0; i < g.nx; i += 1) {
      const x = g.origin[0] + (i + 0.5) * g.h, z = g.origin[2] + (k + 0.5) * g.h;
      if (Math.abs(x - cx) < side / 2 && Math.abs(z - cz) < side / 2) cols.push([i, k]);
    }
  }
  if (!cols.length) {
    const [i, , k] = cellOf(g, cx, y0, cz);
    cols.push([i, k]);
  }
  const D = Math.sqrt((4 * side * side) / Math.PI);
  const maxLayers = Number.isFinite(ceiling) ? Math.max(1, Math.floor((ceiling - y0) / g.h) - 2) : Infinity;
  const layers = Math.min(maxLayers, Math.max(1, Math.round(flameHeight(Q, D) / g.h)));
  const out = [];
  for (const [i, k] of cols) {
    let j = cellOf(g, cx, y0 + 1e-6, cz)[1];
    while (j < g.ny && solid[at(g, i, j, k)]) j += 1;
    for (let l = 0; l < layers && j + l < g.ny; l += 1) if (!solid[at(g, i, j + l, k)]) out.push(at(g, i, j + l, k));
  }
  return { cells: out, layers, columns: cols.length, D };
}

/** The largest gas speed (m/s), Infinity when any face is non-finite. */
function maxSpeed(sim) {
  let m = 0;
  for (const a of [sim.u, sim.v, sim.w]) {
    for (let q = 0; q < a.length; q += 1) {
      const x = a[q];
      if (!Number.isFinite(x)) return Infinity;
      if (x > m) m = x;
      else if (-x > m) m = -x;
    }
  }
  return m;
}

/**
 * Run a scene once a second calling `sample(sim, t)` (every second, from 1 s
 * on; the cases pick their windows), watching the largest speed.
 */
function run(scene, sample) {
  const sim = createSolver(scene, { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL, seed: 1, ...SOLVER });
  const t0 = performance.now();
  let next = 1, peak = 0, divergedAt = null;
  while (sim.time < horizon - 1e-9) {
    step(sim, Math.min(sim.P.dt, horizon - sim.time));
    if (sim.time >= next - 1e-9) {
      const s = maxSpeed(sim);
      peak = Math.max(peak, s);
      if (divergedAt == null && !(s <= DIVERGED_SPEED)) divergedAt = Math.round(sim.time);
      if (divergedAt != null) break;
      sample(sim, Math.round(sim.time));
      next += 1;
    }
  }
  return { sim, seconds: (performance.now() - t0) / 1000, health: { maxSpeed: Number.isFinite(peak) ? +peak.toFixed(1) : "non-finite", diverged: divergedAt != null, divergedAt } };
}

const sceneOf = ({ grid, fraction, solid }, burner, Q) => ({
  grid,
  fraction,
  solid,
  contents: { cells: [], masses: [] },
  ignition: null,
  pilot: null,
  burner: {
    cells: burner.cells,
    hrr: RAMP > 0 ? [[0, 0], [RAMP, Q], [1e6, Q]] : [[0, Q], [1e6, Q]],
    ...(BURNER === "methane" ? { radiantFraction: 0.2 } : { radiantFraction: 0, fuel: fuelsFor("A").ignitionFuel }),
  },
  fuels: { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL },
});

/** D* / dx (FDS User's Guide, mesh resolution): D* = (Q / (rho cp T sqrt g))^(2/5), Q in kW. */
const plumeIndex = (Q, dx) => (Q / 1000 / (1.204 * 1.005 * 293.15 * Math.sqrt(9.81))) ** 0.4 / dx;

/**
 * The FDS Validation Guide's statistics (section 4.4, Eqs. 4.19 to 4.21) on
 * pairs of (model, measured or predicted) rises: sigma_M from the scatter of
 * ln(M/E) less the experiment's, never below it; the bias factor delta.
 */
function statistics(pairs, sigmaE) {
  const logs = pairs.filter(([m, e]) => m > 0 && e > 0).map(([m, e]) => Math.log(m / e));
  if (logs.length < 2) return { n: logs.length, delta: logs.length ? Math.exp(logs[0]) : null, sigmaM: null };
  const mean = logs.reduce((a, b) => a + b, 0) / logs.length;
  const varLog = logs.reduce((a, b) => a + (b - mean) ** 2, 0) / (logs.length - 1);
  const sigmaM = Math.max(sigmaE, Math.sqrt(Math.max(0, varLog - sigmaE * sigmaE)));
  const delta = Math.exp(mean + (sigmaM * sigmaM) / 2 - (sigmaE * sigmaE) / 2);
  return { n: logs.length, delta: +delta.toFixed(3), sigmaM: +sigmaM.toFixed(3), meanRatio: +Math.exp(mean).toFixed(3) };
}

/** The plan's tiers (calibration plan 2026-11-08, section 8); "calibrated" also needs three datasets. */
const tier = (s) => (!s || s.delta == null || s.sigmaM == null ? "too few points" : s.delta >= 0.8 && s.delta <= 1.25 && s.sigmaM <= 0.35 ? "within the calibrated band" : s.delta >= 0.67 && s.delta <= 1.5 && s.sigmaM <= 0.5 ? "indicative" : "not calibrated");

// --------------------------------------------------------------- plume
/** McCaffrey's (1979) centreline velocity (m/s): u / Q^(1/5) = 6.84 z*^(1/2), 1.93, 1.12 z*^(-1/3) by region, Q in kW. */
function mccaffreyVelocity(z, Q) {
  const q = Q / 1000;
  const zs = z / q ** 0.4;
  const k = zs <= 0.08 ? 6.84 * zs ** 0.5 : zs <= 0.2 ? 1.93 : 1.12 * zs ** (-1 / 3);
  return k * q ** 0.2;
}

function plumeCase(h) {
  // 6.4 m wide and 6 m tall: about 32k cells at 0.20 m.
  const Q = 500e3, side = 0.8, W = 6.4, Hd = 6.0;
  const lo = [-W / 2, -h, -W / 2], hi = [W / 2, Hd, W / 2];
  const geo = gridOver(lo, hi, h, [box(-W / 2 - 1, -h, -W / 2 - 1, W / 2 + 1, 0, W / 2 + 1)]);
  const b = burnerCells(geo.grid, geo.solid, { cx: 0, cz: 0, side, y0: 0, Q });
  const g = geo.grid;
  const [ci, , ck] = cellOf(g, 0, 0, 0);
  const firstOpen = cellOf(g, 0, 1e-6, 0)[1];
  const js = [];
  for (let j = firstOpen; j < g.ny - 1; j += 1) js.push(j);
  const sumT = new Float64Array(js.length), sumU = new Float64Array(js.length);
  let samples = 0, downward = 0;
  const mid = js[Math.floor(js.length / 2)];
  const { seconds, health } = run(sceneOf(geo, b, Q), (sim, t) => {
    if (t < AVERAGE_FROM) return;
    js.forEach((j, q) => {
      sumT[q] += sim.T[at(g, ci, j, ck)] - TA;
      sumU[q] += 0.5 * (sim.v[vAt(g, ci, j, ck)] + sim.v[vAt(g, ci, j + 1, ck)]);
    });
    if (0.5 * (sim.v[vAt(g, ci, mid, ck)] + sim.v[vAt(g, ci, mid + 1, ck)]) < 0) downward += 1;
    samples += 1;
  });
  health.offAxisShare = samples ? +(downward / samples).toFixed(2) : null;
  const qkw = Q / 1000;
  const points = js.map((j, q) => {
    const z = (j - firstOpen + 0.5) * h;
    const zs = z / qkw ** 0.4;
    return {
      z: +z.toFixed(3),
      zStar: +zs.toFixed(4),
      region: zs <= 0.08 ? "flame" : zs <= 0.2 ? "intermittent" : "plume",
      riseK: +(sumT[q] / samples).toFixed(1),
      mccaffreyRiseK: +plumeRise(z, Q).toFixed(1),
      velocity: +(sumU[q] / samples).toFixed(3),
      mccaffreyVelocity: +mccaffreyVelocity(z, Q).toFixed(3),
    };
  });
  // Not within a metre of the open top, where the outflow boundary bends the profile.
  const inPlume = points.filter((p) => p.region === "plume" && p.z <= Hd - 1.0);
  const inInt = points.filter((p) => p.region === "intermittent");
  return {
    case: "plume",
    cell: h,
    Q_kW: qkw,
    burnerSide: side,
    grid: [g.nx, g.ny, g.nz],
    cells: g.cells,
    plumeIndex: +plumeIndex(Q, h).toFixed(2),
    burner: { cells: b.cells.length, layers: b.layers, columns: b.columns },
    averagedFrom: AVERAGE_FROM,
    samples,
    health,
    wallSeconds: +seconds.toFixed(1),
    points,
    stats: health.diverged || !samples ? {} : {
      plumeTemperature: statistics(inPlume.map((p) => [p.riseK, p.mccaffreyRiseK]), SIGMA_E.temperature),
      plumeVelocity: statistics(inPlume.map((p) => [p.velocity, p.mccaffreyVelocity]), SIGMA_E.velocity),
      intermittentTemperature: statistics(inInt.map((p) => [p.riseK, p.mccaffreyRiseK]), SIGMA_E.temperature),
      intermittentVelocity: statistics(inInt.map((p) => [p.velocity, p.mccaffreyVelocity]), SIGMA_E.velocity),
    },
  };
}

// ---------------------------------------------------------------- room
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

/**
 * McCaffrey, Quintiere and Harkleroad (1981): the upper layer's rise (K),
 * 6.85 (Q^2 / (A_o sqrt(H_o) h_k A_T))^(1/3), Q in kW, h_k in kW/(m^2 K): for
 * walls thicker than the heat has reached (t < t_p), h_k = sqrt(k rho c / t).
 */
function mqh(Q, { Ao, Ho, AT, k, rho, c, t }) {
  const hk = Math.sqrt((k * rho * c) / t) / 1000;
  return 6.85 * ((Q / 1000) ** 2 / (Ao * Math.sqrt(Ho) * hk * AT)) ** (1 / 3);
}

function roomCase(h, Q) {
  const W = 5, D = 4, H = 2.7, t = 0.2, doorW = 0.9, doorH = 2.0, door = [1.5, 2.4];
  const boxes = [
    box(-t, -t, -t, W + t, 0, D + t), // floor
    box(-t, H, -t, W + t, H + t, D + t), // ceiling
    box(-t, 0, -t, W + t, H, 0),
    box(-t, 0, D, W + t, H, D + t),
    box(-t, 0, 0, 0, H, D),
    box(W, 0, 0, W + t, H, door[0]),
    box(W, 0, door[1], W + t, H, D),
    box(W, doorH, door[0], W + t, H, door[1]),
  ];
  // Outside the door, a metre of open air so the outflow leaves the room.
  const lo = [-t, -t, -t], hi = [W + t + 1.0, H + t + 0.2, D + t];
  const geo = gridOver(lo, hi, h, boxes);
  const g = geo.grid;
  const b = burnerCells(g, geo.solid, { cx: 2.5, cz: 2, side: 0.6, y0: 0, Q, ceiling: H });
  // Each tree reads open cells only: on a coarse grid the 2.7 m ceiling snaps down (to 2.50 m at
  // 0.27 m), and a point in a ceiling cell would read the wall. Its layer reduction takes the
  // column's open height as the ceiling.
  const trees = [[1.5, 2], [3.5, 1], [2.5, 3.3]];
  const openTop = trees.map(([x, z]) => {
    const [i, , k] = cellOf(g, x, 0, z);
    let j = cellOf(g, x, 1e-6, z)[1];
    while (j < g.ny && geo.solid[at(g, i, j, k)]) j += 1;
    while (j + 1 < g.ny && !geo.solid[at(g, i, j + 1, k)] && g.origin[1] + (j + 1.5) * h < H) j += 1;
    return g.origin[1] + (j + 1) * h;
  });
  const treeHeights = openTop.map((top) => {
    const list = [];
    for (let z = 0.3; z <= top - 0.5 * h + 1e-9; z += 0.2) list.push(+z.toFixed(2));
    return list;
  });
  const treeCells = trees.map(([x, z], q) => treeHeights[q].map((y) => { const [i, j, k] = cellOf(g, x, y, z); return at(g, i, j, k); }));
  const times = [60, 90, 120].filter((s) => s <= horizon);
  const window = 5; // s, each reading an 11 s mean about its time (the last ends at the horizon)
  const acc = new Map(times.map((s) => [s, { sum: trees.map((_, q) => new Float64Array(treeHeights[q].length)), n: 0 }]));
  const { seconds, health } = run(sceneOf(geo, b, Q), (sim, t) => {
    for (const s of times) {
      if (Math.abs(t - s) > window) continue;
      const a = acc.get(s);
      treeCells.forEach((list, q) => list.forEach((c, m) => (a.sum[q][m] += sim.T[c])));
      a.n += 1;
    }
  });
  const { k: kWall, rho: rhoWall, c: cWall } = INERT[0]; // "stone", the walls' material in the solver
  const stone = { k: kWall, rho: rhoWall, c: cWall };
  const AT = 2 * (W * D) + 2 * (W * H) + 2 * (D * H) - doorW * doorH;
  const rows = health.diverged ? [] : times.map((s) => {
    const a = acc.get(s);
    const layers = a.sum.map((col, q) => janssensTran(treeHeights[q], Array.from(col, (x) => x / a.n), openTop[q]));
    const rise = layers.reduce((x, l) => x + l.upper - TA, 0) / layers.length;
    const interface_ = layers.reduce((x, l) => x + l.height, 0) / layers.length;
    // The ramp releases in RAMP s what the steady fire releases in RAMP / 2: the walls' exposure is counted from then.
    const predicted = mqh(Q, { Ao: doorW * doorH, Ho: doorH, AT, ...stone, t: s - RAMP / 2 });
    return { time: s, layerRiseK: +rise.toFixed(1), interfaceHeight: +interface_.toFixed(2), mqhRiseK: +predicted.toFixed(1), ratio: +(rise / predicted).toFixed(3) };
  });
  return {
    case: "room",
    cell: h,
    Q_kW: Q / 1000,
    grid: [g.nx, g.ny, g.nz],
    cells: g.cells,
    plumeIndex: +plumeIndex(Q, h).toFixed(2),
    burner: { cells: b.cells.length, layers: b.layers, columns: b.columns },
    walls: { material: "stone (flamescapes.js INERT)", ...stone, wallConvection: DEFAULTS.wallConvection, AT: +AT.toFixed(1), door: [doorW, doorH] },
    openHeight: openTop.map((x) => +x.toFixed(2)),
    health,
    wallSeconds: +seconds.toFixed(1),
    rows,
    stats: { layerTemperature: statistics(rows.map((r) => [r.layerRiseK, r.mqhRiseK]), SIGMA_E.temperature) },
  };
}

// ------------------------------------------------------------- ceiling
/**
 * Alpert (1972): the maximum ceiling-jet rise (K) at radius r (m) under an
 * unconfined ceiling H (m) above a steady fire of Q (kW): 16.9 Q^(2/3) / H^(5/3)
 * for r/H <= 0.18, 5.38 (Q/r)^(2/3) / H beyond.
 */
function alpertRise(Q, H, r) {
  const q = Q / 1000;
  return r / H <= 0.18 ? (16.9 * q ** (2 / 3)) / H ** (5 / 3) : (5.38 * (q / r) ** (2 / 3)) / H;
}

function ceilingCase(h) {
  const Q = 500e3, H = 3.0, W = 7.6, side = 0.6;
  const lo = [-W / 2, -h, -W / 2], hi = [W / 2, H + h, W / 2];
  const geo = gridOver(lo, hi, h, [box(-W, -h, -W, W, 0, W), box(-W, H, -W, W, H + h, W)]);
  const g = geo.grid;
  const b = burnerCells(g, geo.solid, { cx: 0, cz: 0, side, y0: 0, Q, ceiling: H });
  // The top open cell under the ceiling, along +x (and averaged with -x, +z, -z).
  const jTop = (() => {
    let j = cellOf(g, 0, H - 1e-6, 0)[1];
    while (j > 0 && geo.solid[at(g, cellOf(g, 0, 0, 0)[0], j, cellOf(g, 0, 0, 0)[2])]) j -= 1;
    return j;
  })();
  const radii = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
  const probes = radii.map((r) => [[r, 0], [-r, 0], [0, r], [0, -r]].map(([x, z]) => { const [i, , k] = cellOf(g, x, 0, z); return at(g, i, jTop, k); }));
  const sum = new Float64Array(radii.length);
  let n = 0;
  const { seconds, health } = run(sceneOf(geo, b, Q), (sim, t) => {
    if (t < AVERAGE_FROM) return;
    probes.forEach((list, q) => {
      for (const c of list) sum[q] += (sim.T[c] - TA) / list.length;
    });
    n += 1;
  });
  const below = H - (g.origin[1] + (jTop + 0.5) * h);
  const points = radii.map((r, q) => ({ r, rOverH: +(r / H).toFixed(3), riseK: +(sum[q] / n).toFixed(1), alpertRiseK: +alpertRise(Q, H, r).toFixed(1), ratio: +(sum[q] / n / alpertRise(Q, H, r)).toFixed(3) }));
  return {
    case: "ceiling",
    cell: h,
    Q_kW: Q / 1000,
    ceilingHeight: H,
    grid: [g.nx, g.ny, g.nz],
    cells: g.cells,
    plumeIndex: +plumeIndex(Q, h).toFixed(2),
    topCellCentreBelowCeiling: +below.toFixed(3),
    burner: { cells: b.cells.length, layers: b.layers, columns: b.columns },
    averagedFrom: AVERAGE_FROM,
    samples: n,
    health,
    wallSeconds: +seconds.toFixed(1),
    points,
    stats: health.diverged || !n ? {} : { ceilingJetTemperature: statistics(points.filter((p) => p.rOverH > 0.18).map((p) => [p.riseK, p.alpertRiseK]), SIGMA_E.temperature) },
  };
}

// ----------------------------------------------------------------- main
const results = [];
for (const name of cases) {
  for (const h of cells) {
    const runs = name === "plume" ? [plumeCase(h)] : name === "room" ? [roomCase(h, 150e3), roomCase(h, 400e3)] : name === "ceiling" ? [ceilingCase(h)] : [];
    for (const r of runs) {
      results.push(r);
      const s = Object.entries(r.stats).map(([k, v]) => `${k} delta ${v.delta} sigmaM ${v.sigmaM} (n ${v.n}): ${tier(v)}`).join("; ");
      const hl = r.health.diverged ? `DIVERGED at ${r.health.divergedAt} s` : `max speed ${r.health.maxSpeed} m/s${r.health.offAxisShare != null ? `, centreline flowing down in ${Math.round(100 * r.health.offAxisShare)} % of samples` : ""}`;
      console.log(`${r.case} ${r.Q_kW} kW cell ${h} m (${r.grid.join("x")} = ${r.cells} cells, D*/dx ${r.plumeIndex}, ${r.wallSeconds} s; ${hl}): ${s}`);
      if (r.case === "room") for (const row of r.rows) console.log(`   t ${row.time} s: layer +${row.layerRiseK} K (interface ${row.interfaceHeight} m) vs MQH +${row.mqhRiseK} K, ratio ${row.ratio}`);
      if (r.case === "ceiling") console.log(`   top cell ${r.topCellCentreBelowCeiling} m under the ceiling; ` + r.points.map((p) => `r ${p.r}: ${p.riseK}/${p.alpertRiseK} K`).join(", "));
      if (r.case === "plume") console.log("   " + r.points.filter((_, i) => i % 3 === 0).map((p) => `z ${p.z} (${p.region}): ${p.riseK}/${p.mccaffreyRiseK} K, ${p.velocity}/${p.mccaffreyVelocity} m/s`).join("; "));
    }
  }
}

if (DIAGNOSTIC_INTO) {
  const { readFile } = await import("node:fs/promises");
  const earlier = JSON.parse(await readFile(path.resolve(DIAGNOSTIC_INTO), "utf8"));
  earlier.diagnostics = [...(earlier.diagnostics || []), { solver: SOLVER, burner: BURNER, note: "diagnostic only: solver options that are not the app's", results: results.map((r) => ({ ...r, verdicts: Object.fromEntries(Object.entries(r.stats).map(([k, v]) => [k, tier(v)])) })) }];
  await writeFile(path.resolve(DIAGNOSTIC_INTO), JSON.stringify(earlier, null, 1));
  console.log(`appended diagnostics to ${DIAGNOSTIC_INTO}`);
} else if (jsonOut) {
  const out = {
    generated: new Date().toISOString(),
    plan: "wiki/research/fire-calibration-plan-2026-11-11.md, Stage 1a",
    tool: "tools/calibration/transport-checks.mjs",
    solver: "flamescapes.js DEFAULTS as the app runs them (wallConvection 10, sootAbsorption 500, vorticity 10, dt 0.1 s); CPU",
    burner: BURNER === "methane"
      ? "methane, a linear ramp to a constant heat release, radiant fraction 0.2 along 16 rays, convective part into a Heskestad flame-height column"
      : "the app's design-fire kind: the class A item's sooty fuel, radiant fraction 0 (its hot sooty cells radiate as the solver's gas), a linear ramp to a constant heat release into a Heskestad flame-height column",
    divergedAbove: DIVERGED_SPEED,
    rampSeconds: RAMP,
    horizon,
    averagedFrom: AVERAGE_FROM,
    sigmaE: SIGMA_E,
    tiers: "calibration plan 2026-11-08, section 8: within the calibrated band 0.8 <= delta <= 1.25 and sigmaM <= 0.35 (calibrated also needs three independent datasets); indicative 0.67 to 1.5 and sigmaM <= 0.5; else not calibrated",
    results: results.map((r) => ({ ...r, verdicts: Object.fromEntries(Object.entries(r.stats).map(([k, v]) => [k, tier(v)])) })),
  };
  await writeFile(path.resolve(jsonOut), JSON.stringify(out, null, 1));
  console.log(`wrote ${jsonOut}`);
}
