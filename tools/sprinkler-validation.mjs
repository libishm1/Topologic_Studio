/**
 * Checks the Flamescapes sprinkler activation model (src/core/suppression.js,
 * run through flamescapesRun.js runScene on the CPU solver) against measured
 * full-scale tests: Vettori's flat-ceiling (NISTIR 6253) and sloped-set flat
 * ceilings (NISTIR 7079), Bittern's chair fires (via the FDS Validation
 * Guide) and UL/NFPRF Series I (NISTIR 6196-1). The data and their sources
 * are in benchmark/validation/sprinkler-tests.json (NIST; keep its notice);
 * what the data are and which tests to run first is in
 * wiki/research/sprinkler-validation-data-2026-11-07.md.
 *
 * Each test is a box built from its description, without an IFC:
 *   - walls, floor and ceiling one cell thick, doors open or sealed as the
 *     test had them (a closed door's undercut and Bittern's assumed leakage
 *     are far below a cell and are left out: the room is sealed), the
 *     Vettori 1998 stair opening into a closed upper floor of the room's
 *     plan, the 1998 steel I-beam and the 2003 beams (each snaps to whole
 *     cells, so the 0.18 m I-beam is a 0.3 m one), the burner or chair as
 *     an inert block; UL's ceiling over the FDS input's 20.4 x 21.6 m open
 *     domain, without walls, as FDS has it;
 *   - the cell is the nominal size (0.3 m; UL 0.5 m) adjusted so the
 *     ceiling falls on a cell face;
 *   - the fire is the solver's prescribed burner (flamescapes.js
 *     createBurner): the test's heat release table, its convective part
 *     into a column of cells over the burner area as tall as Heskestad's
 *     mean flame height at the heat release of the measured first
 *     activation (at least two cells), the radiant part to the surfaces;
 *   - the heads at their measured positions with the test's RTI, C and
 *     activation temperature; mode "activation" (no water) for the
 *     non-flowing tests, "spray" for UL's flowing heads.
 * Each run reads both of suppression.js's activation times (the "estimate":
 * Alpert's ceiling jet over the grid's layer; the "late": the top cell),
 * and the gas at each head's sensor cell when the head opened in the test.
 *
 *   node tools/sprinkler-validation.mjs [--only REGEX] [--jobs 8]
 *        [--convection 10,200] [--cell 0.3] [--ulcell 0.5] [--horizon-factor 2]
 *        [--solver vorticity=10,cfl=0,vorticityStep=0]  (solver options over the defaults)
 *        [--out benchmark/results/2026-11-08-sprinkler-validation.json]
 *        [--append]  (keep the out file runs not repeated, e.g. a finer --cell)
 *        [--summary file.json]  (print the tables of an earlier run)
 *
 * The convection option sets the solver's surface heat transfer coefficient
 * (W/m2K) for fuel and walls alike (`convection` and `wallConvection`); these
 * scenes hold no fuel, so it is the walls'. 10 is the app's walls' value
 * since 2026-11-08; 200, the fuel's (kept high so a burning item lights its
 * neighbours), was the walls' too before, and opened heads late
 * (wiki/research/sprinkler-activation-fix-2026-11-08.md).
 */
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const CORE = path.join(ROOT, "topologicpy-web-frontend/src/core");
const DATA = path.join(ROOT, "benchmark/validation/sprinkler-tests.json");

// ------------------------------------------------------------------ fuels
// The burner's products only change the gas's species; the heat is the
// table's. No soot: the radiant fraction already takes the flame's
// radiation, and sooty burner gas would radiate it a second time.
const FUELS = {
  methane: null, // flamescapes.js METHANE
  heptane: { heatOfCombustion: 44.6e6, yields: { co2: (44 / 12) * 0.839, co: 0, soot: 0 } },
  // Polyurethane foam: the test's heat of combustion; CO2 from the app's class A carbon fraction.
  polyurethane: (hoc) => ({ heatOfCombustion: hoc * 1e6, yields: { co2: (44 / 12) * 0.444, co: 0, soot: 0 } }),
};
// The solver needs contents fuels; these scenes have none.
const INERT_FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };

const v = (leaf) => (leaf && typeof leaf === "object" && "value" in leaf ? leaf.value : leaf);

/** Heskestad's mean flame height (m) for Q kW over a burner of area A m2. */
const flameHeight = (Q, A) => Math.max(0, 0.235 * Math.max(Q, 0) ** 0.4 - 1.02 * Math.sqrt((4 * A) / Math.PI));

/** Linear interpolation in a [[t, q], ...] table, held at its ends. */
function tableAt(table, t) {
  if (t <= table[0][0]) return table[0][1];
  for (let i = 1; i < table.length; i += 1) {
    if (t <= table[i][0]) {
      const [t0, q0] = table[i - 1], [t1, q1] = table[i];
      return t1 > t0 ? q0 + ((t - t0) / (t1 - t0)) * (q1 - q0) : q1;
    }
  }
  return table.at(-1)[1];
}

const median = (a) => {
  if (!a.length) return null;
  const s = [...a].sort((x, y) => x - y);
  return s.length % 2 ? s[s.length >> 1] : 0.5 * (s[s.length / 2 - 1] + s[s.length / 2]);
};

// ----------------------------------------------------------------- cases

/**
 * Every run: one per test, fire curve and surface coefficient. A case holds
 * what the worker needs to build its scene (in the data set's own frame:
 * x, y in plan, z up) and what was measured.
 */
function makeCases(data, { convections, cell, ulCell, horizonFactor }) {
  const cases = [];
  const D = data.datasets;

  // Vettori 1998, smooth ceiling, detached burner.
  {
    const d = D.vettori_flat_1998;
    const enc = d.enclosure, fire = d.fire, spr = d.sprinkler;
    const H = v(enc.ceiling_height_smooth_m);
    const h = H / Math.round(H / cell);
    const beam = v(enc.steel_I_beam);
    const stair = v(enc.stairwell_opening);
    for (const t of d.tests) {
      const positions = v(spr.positions_xy_m);
      const heads = t.heads.map((hd) => ({
        n: hd.head,
        xy: positions[hd.head],
        z: H - v(spr.link_below_ceiling_m).smooth,
        rti: v(spr.RTI_m05_s05),
        c: v(spr.C_m05_per_s05),
        activationC: v(spr.activation_C),
        measured: v(hd.activation_s),
        measuredGasC: v(hd.gas_T_at_head_at_activation_C) ?? null,
        measuredSpeed: v(hd.ceiling_jet_speed_at_activation_m_per_s) ?? null,
      }));
      const ramps = {
        tabulated: v(fire.tabulated_ramp_kW[t.growth]),
        nominal: null, // alpha t^2, made to the horizon
      };
      const alpha = v(fire.nominal_t_squared_alpha_kW_per_s2)[t.growth];
      const last = Math.max(...heads.map((x) => x.measured ?? 0));
      const horizon = Math.min(600, Math.ceil((horizonFactor * last) / 5) * 5);
      for (const ramp of ["tabulated", "nominal"]) {
        const table = ramp === "tabulated" ? ramps.tabulated : Array.from({ length: horizon + 1 }, (_, s) => [s, alpha * s * s]);
        for (const convection of convections) {
          cases.push({
            id: `vettori_flat/${t.test}/${ramp}/h${convection}/${h.toFixed(3)}`,
            dataset: "vettori_flat_1998",
            test: t.test,
            growth: t.growth,
            ramp,
            convection,
            ambientC: v(t.ambient_C),
            room: { L: v(enc.length_x_m), W: v(enc.width_y_m), H, doors: [], upper: { H: H, hole: { x: stair.x_m, y: stair.y_m } } },
            obstacles: [
              { x: [0, v(enc.length_x_m)], y: beam.y_m, z: [H - beam.depth_m, H], what: "steel I-beam" },
              { x: [4.3, 5.0], y: [1.0, 2.0], z: [0, v(fire.burner_size_m).height], what: "burner" },
            ],
            h,
            margin: 2,
            burner: { x: [4.3, 5.0], y: [1.0, 2.0], z0: v(fire.burner_size_m).height, table, radiantFraction: 0.2, fuel: "methane" },
            heads,
            mode: "activation",
            horizon,
          });
        }
      }
    }
  }

  // Vettori 2003, flat ceiling, detached burner, smooth (1-4) or two beams (13-16).
  {
    const d = D.vettori_sloped_2003;
    const enc = d.enclosure, spr = d.sprinkler;
    const flat = D.vettori_flat_1998.fire; // the same burner and tables (NISTIR 7079; FDS inputs)
    const H = v(enc.ceiling_height_flat_m);
    const h = H / Math.round(H / cell);
    const door = v(enc.door);
    const beams = v(enc.beams);
    for (const t of d.tests) {
      const heads = t.heads.map((hd) => ({
        n: hd.head,
        xy: v(hd.xyz_m).slice(0, 2),
        z: v(hd.xyz_m)[2],
        rti: v(spr.RTI_m05_s05),
        // Not reported for these heads; taken as the 1998 heads' (same RTI and maker class, UL's 0.25).
        c: 0.25,
        activationC: v(spr.activation_C),
        measured: v(hd.activation_s),
        measuredGasC: v(hd.gas_T_at_head_at_activation_C) ?? null,
        measuredSpeed: null,
      }));
      const alpha = v(flat.nominal_t_squared_alpha_kW_per_s2)[t.growth];
      const last = Math.max(...heads.map((x) => x.measured ?? 0));
      const horizon = Math.min(600, Math.ceil((horizonFactor * last) / 5) * 5);
      const obstacles = [{ x: [3.3, 4.0], y: [2.25, 3.25], z: [0, 0.3], what: "burner" }];
      if (t.ceiling !== "smooth") for (const bx of beams.x_m) obstacles.push({ x: bx, y: [0, v(enc.depth_y_m)], z: [H - beams.depth_m, H], what: "beam" });
      for (const ramp of ["tabulated", "nominal"]) {
        const table = ramp === "tabulated" ? v(flat.tabulated_ramp_kW[t.growth]) : Array.from({ length: horizon + 1 }, (_, s) => [s, alpha * s * s]);
        for (const convection of convections) {
          cases.push({
            id: `vettori_sloped/${t.test}/${ramp}/h${convection}/${h.toFixed(3)}`,
            dataset: "vettori_sloped_2003",
            test: t.test,
            growth: t.growth,
            ceiling: t.ceiling,
            ramp,
            convection,
            ambientC: v(t.ambient_C),
            room: { L: v(enc.width_x_m), W: v(enc.depth_y_m), H, doors: [{ wall: "y0", from: door.x_m[0], to: door.x_m[1], height: door.height_m }] },
            obstacles,
            h,
            margin: 2,
            burner: { x: [3.3, 4.0], y: [2.25, 3.25], z0: 0.3, table, radiantFraction: 0.2, fuel: "methane" },
            heads,
            mode: "activation",
            horizon,
          });
        }
      }
    }
  }

  // Bittern 2004, centre chair.
  {
    const d = D.bittern_2004;
    const enc = d.enclosure;
    const H = v(enc.ceiling_height_m);
    const h = H / Math.round(H / cell);
    const door = v(enc.door);
    for (const t of d.tests) {
      const [cx, cy] = v(t.fire_centre_xy_m);
      const top = v(t.fire_top_height_m);
      const heads = t.heads.map((hd) => ({
        n: hd.head,
        type: v(hd.type),
        xy: v(hd.xy_m),
        z: H - v(hd.link_below_ceiling_m),
        rti: v(hd.RTI_m05_s05),
        c: v(hd.C_m05_per_s05),
        activationC: v(hd.activation_C),
        measured: v(hd.activation_s),
        measuredGasC: null,
        measuredSpeed: null,
      }));
      const last = Math.max(...heads.map((x) => x.measured ?? 0));
      const horizon = Math.min(600, Math.ceil((horizonFactor * last) / 5) * 5);
      // The chair's 0.4 x 0.5 m top; its orientation is not given (0.5 m taken along the room).
      const chair = { x: [cx - 0.25, cx + 0.25], y: [cy - 0.2, cy + 0.2] };
      for (const convection of convections) {
        cases.push({
          id: `bittern/${t.test}/measured/h${convection}/${h.toFixed(3)}`,
          dataset: "bittern_2004",
          test: t.test,
          door: t.door,
          ramp: "measured",
          convection,
          ambientC: v(t.ambient_C),
          room: { L: v(enc.length_x_m), W: v(enc.width_y_m), H, doors: t.door === "open" ? [{ wall: "x0", from: door.y_m[0], to: door.y_m[1], height: door.height_m }] : [] },
          obstacles: [{ ...chair, z: [0, top], what: "chair" }],
          h,
          margin: 2,
          burner: { ...chair, z0: top, table: v(t.hrr_curve_kW), radiantFraction: v(d.fire.radiative_fraction), fuel: ["polyurethane", v(t.heat_of_combustion_MJ_per_kg)] },
          heads,
          mode: "activation",
          horizon,
        });
      }
    }
  }

  // UL/NFPRF Series I without draft curtains: flowing heads, heptane spray burner.
  {
    const d = D.ul_nfprf_1998_series_I;
    const enc = d.enclosure, spr = d.sprinkler;
    const ceil = v(enc.ceiling_m);
    const H = ceil.height;
    const h = H / Math.round(H / ulCell);
    const all = v(spr.positions_xyz_m);
    const spacing = v(spr.spacing_m);
    for (const t of d.tests) {
      const [bx, by] = v(t.burner_centre_xy_m);
      // The FDS input's mesh for the test (I-22's burner D sits further south-west).
      const domain = t.burner_position === "D" ? { x: [8.2, 28.6], y: [2.0, 23.6] } : { x: [13.2, 33.6], y: [12.0, 33.6] };
      const measuredBy = new Map(t.first_four_heads.map((f) => [String(f.head), f]));
      const heads = Object.entries(all)
        .filter(([, p]) => p[0] > domain.x[0] && p[0] < domain.x[1] && p[1] > domain.y[0] && p[1] < domain.y[1])
        .map(([id, p]) => {
          const m = measuredBy.get(id);
          return {
            n: Number(id),
            xy: [p[0], p[1]],
            z: H - v(spr.link_below_ceiling_m),
            rti: v(spr.RTI_m05_s05),
            c: v(spr.C_m05_per_s05),
            activationC: v(spr.activation_C),
            measured: m ? v(m.activation_s) : null,
            measuredGasC: m ? v(m.gas_T_near_head_at_activation_C) : null,
            measuredSpeed: null,
            flowLpm: v(spr.flow_per_head_L_per_min),
            coverage: spacing[0] * spacing[1],
            half: [spacing[0] / 2, spacing[1] / 2],
          };
        });
      const Qp = v(t.hrr.plateau_MW) * 1000, tp = v(t.hrr.plateau_reached_s);
      const last = Math.max(...heads.map((x) => x.measured ?? 0));
      const horizon = Math.min(200, Math.ceil((horizonFactor * last) / 5) * 5);
      const table = [];
      for (let s = 0; s <= tp; s += 1) table.push([s, Qp * (s / tp) ** 2]);
      for (const convection of convections) {
        cases.push({
          id: `ul_nfprf/${t.test}/measured/h${convection}/${h.toFixed(3)}`,
          dataset: "ul_nfprf_1998_series_I",
          test: t.test,
          ramp: "measured",
          convection,
          ambientC: v(t.ambient_C),
          open: { x: domain.x, y: domain.y, H, ceilingThickness: 0.4 },
          obstacles: [],
          h,
          margin: 0,
          burner: { x: [bx - 0.5, bx + 0.5], y: [by - 0.5, by + 0.5], z0: 0.6, table, radiantFraction: 1 - v(d.fire.convective_fraction), fuel: "heptane" },
          heads,
          mode: "spray",
          horizon,
          measuredCount: v(t.number_activated),
          measuredCountCsv: v(t.number_activated_exp_csv),
          measuredSequence: v(t.activation_sequence_s),
        });
      }
    }
  }
  return cases;
}

// -------------------------------------------------------------- the runs

async function runCase(c) {
  const V = await import(pathToFileURL(path.join(CORE, "voxelize.js")).href);
  const R = await import(pathToFileURL(path.join(CORE, "flamescapesRun.js")).href);
  const S = await import(pathToFileURL(path.join(CORE, "suppression.js")).href);
  const t0 = performance.now();
  const h = c.h;
  // A box in the data frame (x, y plan; z up) as a mesh in the viewer frame (x, z, y; y up).
  const meshOf = ({ x, y, z }) => {
    const [x0, x1] = x, [y0, y1] = y, [z0, z1] = z;
    const p = [x0, z0, y0, x1, z0, y0, x1, z1, y0, x0, z1, y0, x0, z0, y1, x1, z0, y1, x1, z1, y1, x0, z1, y1];
    const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
    return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material: V.MATERIAL.STONE };
  };
  const boxes = [];
  let lo, hi;
  if (c.room) {
    const { L, W, H, doors, upper } = c.room;
    const t = h, m = c.margin * h;
    const wall = (side, z0, z1, openings) => {
      // A wall along one side, with door openings (from, to along the wall, height above z0).
      const along = side === "x0" || side === "x1" ? [-t, W + t] : [-t, L + t];
      const cuts = openings.filter((d) => d.wall === side).sort((a, b) => a.from - b.from);
      const pieces = [];
      let at = along[0];
      for (const d of cuts) {
        pieces.push([at, d.from, z0, z1]);
        pieces.push([d.from, d.to, z0 + d.height, z1]);
        at = d.to;
      }
      pieces.push([at, along[1], z0, z1]);
      for (const [a, b, za, zb] of pieces) {
        if (!(b > a) || !(zb > za)) continue;
        if (side === "x0") boxes.push({ x: [-t, 0], y: [a, b], z: [za, zb] });
        else if (side === "x1") boxes.push({ x: [L, L + t], y: [a, b], z: [za, zb] });
        else if (side === "y0") boxes.push({ x: [a, b], y: [-t, 0], z: [za, zb] });
        else boxes.push({ x: [a, b], y: [W, W + t], z: [za, zb] });
      }
    };
    boxes.push({ x: [-t, L + t], y: [-t, W + t], z: [-t, 0] });
    for (const side of ["x0", "x1", "y0", "y1"]) wall(side, 0, H, doors);
    let top = H + t;
    if (upper) {
      // The ceiling with the stair opening, a closed upper floor of the same plan and its roof.
      const { x: hx, y: hy } = upper.hole;
      boxes.push({ x: [-t, hx[0]], y: [-t, W + t], z: [H, H + t] });
      boxes.push({ x: [hx[0], L + t], y: [-t, hy[0]], z: [H, H + t] });
      if (hy[1] < W + t) boxes.push({ x: [hx[0], L + t], y: [hy[1], W + t], z: [H, H + t] });
      if (hx[1] < L) boxes.push({ x: [hx[1], L + t], y: [hy[0], hy[1]], z: [H, H + t] });
      const z0 = H + t, z1 = H + t + upper.H;
      for (const side of ["x0", "x1", "y0", "y1"]) wall(side, z0, z1, []);
      boxes.push({ x: [-t, L + t], y: [-t, W + t], z: [z1, z1 + t] });
      top = z1 + t;
    } else boxes.push({ x: [-t, L + t], y: [-t, W + t], z: [H, H + t] });
    lo = [-t - m, -t - m, -t];
    hi = [L + t + m, W + t + m, top + m];
  } else {
    // UL: the ceiling over an open-sided domain, the floor below it.
    const { x, y, H, ceilingThickness } = c.open;
    boxes.push({ x: [x[0] - 1, x[1] + 1], y: [y[0] - 1, y[1] + 1], z: [-h, 0] });
    boxes.push({ x: [x[0] - 1, x[1] + 1], y: [y[0] - 1, y[1] + 1], z: [H, H + ceilingThickness] });
    lo = [x[0], y[0], -h];
    hi = [x[1], y[1], H + h];
  }
  for (const o of c.obstacles) boxes.push(o);
  const meshes = boxes.map(meshOf);
  const n = [0, 1, 2].map((d) => Math.ceil((hi[d] - lo[d]) / h - 1e-6));
  // The grid's axes are the viewer's (x, y up, z): data (x, z, y).
  const grid = { nx: n[0], ny: n[2], nz: n[1], h, origin: [lo[0], lo[2], lo[1]], axes: [0, 1, 2], upAxis: "y", cells: n[0] * n[1] * n[2] };
  const fraction = V.volumeFractions(grid, meshes, { rays: V.raysFor(grid) });
  const solid = V.occupancy(grid, fraction);
  const at = (i, j, k) => i + grid.nx * (j + grid.ny * k);
  const cellOf = (x, y, z) => [Math.floor((x - grid.origin[0]) / h), Math.floor((z - grid.origin[1]) / h), Math.floor((y - grid.origin[2]) / h)];

  // The burner: the open cells over its area (centres inside it, else the one at its centre), a flame-height column.
  const b = c.burner;
  const area = (b.x[1] - b.x[0]) * (b.y[1] - b.y[0]);
  const measuredFirst = Math.min(...c.heads.map((x) => x.measured ?? Infinity));
  const qAtFirst = Number.isFinite(measuredFirst) ? tableAt(b.table, measuredFirst) : Math.max(...b.table.map((r) => r[1]));
  const ceilingZ = c.room ? c.room.H : c.open.H;
  const maxLayers = Math.max(2, Math.floor((ceilingZ - b.z0) / h) - 2);
  const layers = Math.min(maxLayers, Math.max(2, Math.round(flameHeight(qAtFirst, area) / h)));
  const columns = [];
  for (let k = 0; k < grid.nz; k += 1) {
    for (let i = 0; i < grid.nx; i += 1) {
      const x = grid.origin[0] + (i + 0.5) * h, y = grid.origin[2] + (k + 0.5) * h;
      if (x > b.x[0] && x < b.x[1] && y > b.y[0] && y < b.y[1]) columns.push([i, k]);
    }
  }
  if (!columns.length) {
    const [i, , k] = cellOf((b.x[0] + b.x[1]) / 2, (b.y[0] + b.y[1]) / 2, 0);
    columns.push([i, k]);
  }
  const cells = [];
  for (const [i, k] of columns) {
    let j = cellOf(0, 0, b.z0 + 1e-6)[1];
    while (j < grid.ny && solid[at(i, j, k)]) j += 1;
    for (let l = 0; l < layers && j + l < grid.ny; l += 1) if (!solid[at(i, j + l, k)]) cells.push(at(i, j + l, k));
  }
  const fuel = b.fuel === "methane" ? undefined : Array.isArray(b.fuel) ? FUELS[b.fuel[0]](b.fuel[1]) : FUELS[b.fuel];
  const scene = {
    grid,
    fraction,
    solid,
    contents: { cells: [], masses: [] },
    ignition: null,
    pilot: null,
    burner: { cells, hrr: b.table.map(([t, q]) => [t, q * 1000]), radiantFraction: b.radiantFraction, fuel },
    fuels: { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL },
  };
  const heads = c.heads.map((x) => ({
    id: String(x.n),
    position: [x.xy[0], x.z, x.xy[1]],
    // The measured link's own distance under the ceiling (suppression.js LINK_UNDER_CEILING otherwise).
    ceilingAbove: Math.max(0, ceilingZ - x.z),
    footprint: { centre: [x.xy[0], x.xy[1]], axis: [1, 0], half: x.half || [2.3, 2.3] },
    rti: x.rti,
    c: x.c,
    activationC: x.activationC,
    flowLpm: x.flowLpm || 0,
    coverage: x.coverage || 1,
  }));
  const frames = [];
  const built = performance.now();
  const result = R.runScene(scene, {
    nodes: Float32Array.from([b.x[0] - 1, 0.75, b.y[0]]),
    upAxis: "y",
    horizon: c.horizon,
    frameSeconds: 1,
    particles: 0,
    keep: false,
    sensorGas: true,
    sprinklers: { heads, mode: c.mode, openOn: "estimate" },
    solverOptions: { ambient: c.ambientC + 273.15, convection: c.convection, wallConvection: c.convection, ...c.solver },
    onFrame: (f) => frames.push({ time: f.time, hrr: f.hrr, T: Array.from(f.sprinklers.gas.T), u: Array.from(f.sprinklers.gas.u), open: f.sprinklers.open.length }),
  });
  const ran = performance.now();
  const spr = result.sprinklers;
  const H = spr.plumeHeight;
  // The gas each reading saw at a time: the sensor cell's, and the estimate's (Alpert over the layer, as activationFrame).
  const gasAt = (q, time) => {
    const k = Math.max(0, Math.min(frames.length - 1, Math.round(time) - 1));
    const f = frames[k];
    let sum = 0, count = 0;
    spr.heads.forEach((hd, j) => {
      if (hd.onGrid && hd.inSight && hd.r >= 0.2 * (hd.H ?? 0)) {
        sum += f.T[j];
        count += 1;
      }
    });
    const layer = count ? sum / count : c.ambientC + 273.15;
    const hd = spr.heads[q];
    const jet = hd.inSight && hd.H ? S.alpert(f.hrr, hd.H, Math.max(hd.r, 0.01)) : { dT: 0, U: 0 };
    return {
      cellC: f.T[q] - 273.15,
      cellSpeed: f.u[q],
      estimateC: hd.inSight ? Math.max(f.T[q], layer + jet.dT) - 273.15 : f.T[q] - 273.15,
      estimateSpeed: hd.inSight ? Math.max(f.u[q], jet.U) : f.u[q],
      hrr: f.hrr,
    };
  };
  const out = c.heads.map((x, q) => {
    const hd = spr.heads[q];
    return {
      head: x.n,
      type: x.type,
      r: hd.r,
      inSight: hd.inSight,
      onGrid: hd.onGrid,
      measured: x.measured,
      estimate: hd.estimate,
      late: hd.late,
      hrrEstimate: hd.hrrEstimate,
      hrrLate: hd.hrrLate,
      measuredGasC: x.measuredGasC,
      measuredSpeed: x.measuredSpeed,
      atMeasured: x.measured != null && x.measured <= c.horizon ? gasAt(q, x.measured) : null,
    };
  });
  const grid3 = [grid.nx, grid.ny, grid.nz];
  return {
    id: c.id,
    dataset: c.dataset,
    test: c.test,
    growth: c.growth,
    ceiling: c.ceiling,
    door: c.door,
    ramp: c.ramp,
    convection: c.convection,
    mode: c.mode,
    horizon: c.horizon,
    grid: { n: grid3, cells: grid.cells, h },
    burner: { cells: cells.length, columns: columns.length, layers, columnHeight: layers * h, radiantFraction: b.radiantFraction, released: result.burner.released, qAtMeasuredFirst: qAtFirst },
    plumeHeight: H,
    // The run's health (flamescapes.js healthVerdict, proposal A of the 2026-11-11 calibration plan).
    health: result.health ?? null,
    heads: out,
    openAtEnd: spr.open.length,
    openedSequence: c.mode === "spray" ? frames.filter((f, k) => k === 0 || f.open !== frames[k - 1].open).map((f) => [f.time, f.open]) : undefined,
    measuredCount: c.measuredCount,
    measuredCountCsv: c.measuredCountCsv,
    measuredSequence: c.measuredSequence,
    water: spr.stats,
    ms: { build: built - t0, run: ran - built, perStep: (ran - built) / Math.round(c.horizon / 0.1) },
  };
}

// --------------------------------------------------------------- summary

/** Per data set, fire curve, coefficient and reading: predicted over measured, the bracket, and the share opened. */
function summarise(results) {
  const groups = new Map();
  for (const r of results) {
    const key = `${r.dataset}|${r.ramp}|${r.convection}|${r.grid.h.toFixed(3)}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  const rows = [];
  for (const [key, runs] of groups) {
    const [dataset, ramp, convection, cell] = key.split("|");
    for (const reading of ["estimate", "late"]) {
      const ratios = [], firsts = [];
      let measured = 0, opened = 0, bracketed = 0, early = 0;
      for (const r of runs) {
        const withTimes = r.heads.filter((x) => x.measured != null);
        for (const x of withTimes) {
          measured += 1;
          const p = x[reading];
          if (p != null) {
            opened += 1;
            ratios.push(p / x.measured);
            if (p < x.measured) early += 1;
          }
          const est = x.estimate ?? Infinity, late = x.late ?? Infinity;
          if (est <= x.measured && x.measured <= late) bracketed += reading === "estimate" ? 1 : 0;
        }
        // The first head to open, against the first measured.
        const m = Math.min(...withTimes.map((x) => x.measured));
        const p = Math.min(...r.heads.map((x) => x[reading] ?? Infinity));
        if (Number.isFinite(m)) firsts.push(Number.isFinite(p) ? p / m : null);
      }
      const f = firsts.filter((x) => x !== null);
      rows.push({
        dataset,
        ramp,
        convection: Number(convection),
        cell: Number(cell),
        reading,
        tests: runs.length,
        heads: measured,
        opened,
        openedShare: measured ? opened / measured : null,
        median: median(ratios),
        min: ratios.length ? Math.min(...ratios) : null,
        max: ratios.length ? Math.max(...ratios) : null,
        early,
        bracketedShare: reading === "estimate" && measured ? bracketed / measured : null,
        firstMedian: median(f),
        firstMin: f.length ? Math.min(...f) : null,
        firstMax: f.length ? Math.max(...f) : null,
        firstOpened: f.length,
      });
    }
  }
  return rows;
}

function printTables(report) {
  const f = (x, d = 0) => (x == null || !Number.isFinite(x) ? "-" : x.toFixed(d));
  console.log(`\nRuns (${report.results.length}; ${report.jobs} at a time on ${report.cpu}):`);
  console.log("case                                   grid (cells, h)          run s | head  r m  measured  estimate  late   | gas at measured time: cell / estimate / measured C; cell speed / measured m/s");
  for (const r of report.results) {
    const g = `${r.grid.n.join("x")} (${r.grid.cells}, ${r.grid.h.toFixed(3)})`;
    let first = true;
    for (const x of r.heads) {
      if (x.measured == null && x.estimate == null && x.late == null) continue;
      const a = x.atMeasured;
      const gas = a ? `${f(a.cellC)} / ${f(a.estimateC)} / ${f(x.measuredGasC)}; ${f(a.cellSpeed, 2)} / ${f(x.measuredSpeed, 2)}` : "";
      console.log(`${(first ? r.id : "").padEnd(38)} ${(first ? g : "").padEnd(24)} ${(first ? f(r.ms.run / 1000, 0) : "").padStart(5)} | ${String(x.head).padStart(4)} ${f(x.r, 1).padStart(4)} ${f(x.measured, 1).padStart(8)} ${f(x.estimate, 1).padStart(9)} ${f(x.late, 1).padStart(6)}   | ${gas}`);
      first = false;
    }
    if (r.mode === "spray") console.log(`${"".padEnd(64)}   opened by ${r.horizon} s: ${r.openAtEnd}; measured ${r.measuredCount} (Table 1), ${r.measuredCountCsv} (exp CSV); sequence ${JSON.stringify(r.openedSequence?.slice(1, 12))}`);
  }
  console.log("\nSummary: predicted / measured activation per head (and for the first head per test); bracketed: estimate <= measured <= late (a late reading not open by the horizon counts as later)");
  console.log("data set                ramp       h W/m2K cell m reading   tests heads opened  median  range         early  bracketed | first: median  range        opened");
  for (const s of report.summary) {
    console.log(`${s.dataset.padEnd(23)} ${s.ramp.padEnd(10)} ${String(s.convection).padStart(6)} ${s.cell.toFixed(3).padStart(6)} ${s.reading.padEnd(9)} ${String(s.tests).padStart(5)} ${String(s.heads).padStart(5)} ${String(s.opened).padStart(6)}  ${f(s.median, 2).padStart(6)}  ${`${f(s.min, 2)}-${f(s.max, 2)}`.padEnd(12)} ${String(s.early).padStart(5)}  ${s.bracketedShare == null ? "        -" : `${f(100 * s.bracketedShare)} %`.padStart(9)} |        ${f(s.firstMedian, 2).padStart(6)}  ${`${f(s.firstMin, 2)}-${f(s.firstMax, 2)}`.padEnd(12)} ${s.firstOpened}/${s.tests}`);
  }
}

// ------------------------------------------------------------------ main

async function main() {
  const args = process.argv.slice(2);
  const flag = (name, fallback) => {
    const i = args.indexOf(name);
    return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
  };
  if (args.includes("--summary")) {
    printTables(JSON.parse(await readFile(flag("--summary"), "utf8")));
    return;
  }
  const data = JSON.parse(await readFile(DATA, "utf8"));
  const settings = {
    convections: flag("--convection", "10,200").split(",").map(Number),
    cell: Number(flag("--cell", 0.3)),
    ulCell: Number(flag("--ulcell", 0.5)),
    horizonFactor: Number(flag("--horizon-factor", 2)),
    // --solver vorticity=10,cfl=0,vorticityStep=0: the solver as it was before the 2026-11-11 calibration.
    solver: Object.fromEntries((flag("--solver", "") || "").split(",").filter(Boolean).map((kv) => kv.split("=")).map(([k, v]) => [k, Number(v)])),
  };
  const only = flag("--only", null);
  let cases = makeCases(data, settings).map((c) => ({ ...c, solver: settings.solver }));
  if (only) cases = cases.filter((c) => new RegExp(only).test(c.id));
  const jobs = Math.max(1, Math.min(Number(flag("--jobs", Math.max(1, Math.floor(os.cpus().length / 2)))), cases.length));
  const out = flag("--out", path.join(ROOT, "benchmark/results/2026-11-08-sprinkler-validation.json"));
  console.log(`${cases.length} runs, ${jobs} at a time`);
  const t0 = performance.now();
  const results = new Array(cases.length);
  let next = 0, done = 0;
  await Promise.all(
    Array.from({ length: jobs }, async () => {
      while (next < cases.length) {
        const q = next;
        next += 1;
        results[q] = await new Promise((resolve, reject) => {
          const w = new Worker(fileURLToPath(import.meta.url), { workerData: cases[q] });
          w.once("message", resolve);
          w.once("error", reject);
        });
        done += 1;
        const r = results[q];
        const firstHead = r.heads.filter((x) => x.measured != null).sort((a, b) => a.measured - b.measured)[0];
        console.log(`  [${done}/${cases.length}] ${r.id}: ${r.grid.cells} cells, ${(r.ms.run / 1000).toFixed(1)} s; first measured head ${firstHead?.head}: ${firstHead?.measured} s, estimate ${firstHead?.estimate?.toFixed(1) ?? "-"} s, late ${firstHead?.late?.toFixed(1) ?? "-"} s`);
      }
    }),
  );
  // With --append, the out file's other runs (another cell size, say) are kept.
  let earlier = [], earlierSettings = [];
  if (args.includes("--append")) {
    try {
      const old = JSON.parse(await readFile(out, "utf8"));
      earlier = old.results.filter((r) => !results.some((x) => x.id === r.id));
      earlierSettings = [old.settings, ...(old.earlierSettings || [])];
    } catch {
      earlier = [];
    }
  }
  const all = [...earlier, ...results];
  const report = {
    generated: new Date().toISOString(),
    tool: "tools/sprinkler-validation.mjs",
    data: "benchmark/validation/sprinkler-tests.json",
    node: process.version,
    cpu: os.cpus()[0]?.model?.trim(),
    cores: os.cpus().length,
    jobs,
    settings,
    earlierSettings,
    wallSeconds: (performance.now() - t0) / 1000,
    results: all,
    summary: summarise(all),
  };
  await mkdir(path.dirname(out), { recursive: true });
  await writeFile(out, JSON.stringify(report, null, 1) + "\n");
  printTables(report);
  console.log(`\nWrote ${path.relative(ROOT, out)} (${report.wallSeconds.toFixed(0)} s wall)`);
}

if (isMainThread) await main();
else parentPort.postMessage(await runCase(workerData));
