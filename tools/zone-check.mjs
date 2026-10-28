/**
 * Cross-model check of the two-zone fire model (gap LG8; src/core/rooms.js,
 * src/core/zone.js; wiki/research/zone-model-spec-2026-10-28.md).
 *
 *   node tools/zone-check.mjs [model.ifc ...] [--json out.json]
 *
 * For each IFC model: build the walkable graph as the site does, find the
 * exits (tools/exits.mjs) and the room graph (IfcSpace rooms, lattice nodes
 * assigned to them, doors joining rooms or venting outside), then start a
 * medium t-squared fire in the largest room that has lattice nodes and a door
 * (else the largest with nodes) and run the model for 600 s. Recorded per
 * model:
 *   - rooms found, the share of lattice nodes in a room footprint and the
 *     share assigned to a room after the search (kept within 1 m of each
 *     room's floor, and, for comparison, unbounded as the specification has
 *     it), doors joining rooms, vents;
 *   - the fire room, its vent, the ventilation-limited peak, the flashover
 *     time, and how many rooms the smoke has reached by 300 s and by 600 s;
 *   - the routers' checks on the zone forecast, as tools/forecast-check.mjs
 *     makes them: from sampled starts the earliest safe arrival must equal the
 *     first point of the time-dose front (not truncated), and at sampled nodes
 *     the dose-limited deadline map (FED 0.3) must work in the exact model and
 *     never be later than the exact deadline (bisection on doseFront).
 *
 * The fire is illustrative (medium growth, 5 MW asked for, ventilation
 * limited; 1.2 m/s walking): this checks that the model runs on each model,
 * finds rooms and spreads smoke plausibly, and that the routers stay exact on
 * it; it is not a fire assessment of any building. Model files are read in
 * place and never copied.
 */
import { readdirSync, existsSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

import { buildLatticeGraph } from "../topologicpy-web-frontend/src/core/lattice.js";
import { assignNodes, buildRoomGraph, footprintIndex } from "../topologicpy-web-frontend/src/core/rooms.js";
import { buildZoneForecast } from "../topologicpy-web-frontend/src/core/zone.js";
import { doseFront, planWithForecast } from "../topologicpy-web-frontend/src/core/forecast.js";
import { doseDeadline, doseDeadlineMap, escapeDeadlines } from "../topologicpy-web-frontend/src/core/deadline.js";
import { findExits } from "./exits.mjs";
import { extract } from "./ifc-extract.mjs";

const args = process.argv.slice(2);
const jsonAt = args.indexOf("--json");
const jsonOut = jsonAt >= 0 ? args[jsonAt + 1] : null;
let files = args.filter((a, i) => a.toLowerCase().endsWith(".ifc") && i !== jsonAt + 1);

/** Every .ifc under a folder, at any depth. */
function ifcsIn(dir) {
  const out = [];
  if (!existsSync(dir)) return out;
  for (const f of readdirSync(dir)) {
    const p = path.join(dir, f);
    if (statSync(p).isDirectory()) out.push(...ifcsIn(p));
    else if (f.toLowerCase().endsWith(".ifc")) out.push(p);
  }
  return out;
}

if (!files.length) {
  for (const d of ["C:/Users/lmurugesan/ifc-samples", "topologicpy-web-frontend/public/examples"]) files.push(...ifcsIn(d));
}

const WALK = 1.2, LIMIT = 0.3, GROWTH = "medium", PEAK = 5000, HORIZON = 600, FRAME = 1, UNTENABLE_C = 60;
const rows = [];
const pct = (x) => +(100 * x).toFixed(1);

for (const file of files) {
  const row = { model: path.basename(file), folder: path.basename(path.dirname(file)) };
  let geo, built, g, n;
  try {
    const t0 = performance.now();
    geo = await extract(path.resolve(file), { withDoorFlags: true });
    built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
    row.graphMs = Math.round(performance.now() - t0);
    g = { ...built, nodes: built.points };
    n = g.nodes.length / 3;
  } catch (error) {
    rows.push({ ...row, status: `error: ${error.message}` });
    console.log(`${row.model.slice(0, 34).padEnd(34)} error: ${error.message}`);
    continue;
  }
  row.nodes = n;
  row.spaces = new Set((geo.spaces || []).map((m) => m.itemId)).size;
  if (!n) {
    rows.push({ ...row, status: "no walkable graph" });
    console.log(`${row.model.slice(0, 34).padEnd(34)} no walkable graph`);
    continue;
  }

  // The room graph (spec section 1).
  const doors = built.doorReport.map((d, i) => ({ ...d, ux: built.doorPoints[i][3], uy: built.doorPoints[i][4] }));
  let t = performance.now();
  const rg = buildRoomGraph(g, { spaces: geo.spaces, doors, declaredDoors: geo.semantics?.doors || [], upAxis: "y", agentHeight: 0.75 });
  row.roomGraphMs = +(performance.now() - t).toFixed(1);
  const s = rg.stats;
  // The specification's unbounded search, for comparison (rooms.js header).
  const unbounded = s.rooms ? assignNodes(g, rg.rooms, footprintIndex(rg.rooms), { upAxis: "y", agentHeight: 0.75, band: Infinity }).assigned : 0;
  Object.assign(row, {
    rooms: s.rooms,
    inFootprintPct: pct(s.inFootprint / n),
    assignedPct: pct(s.assigned / n),
    assignedUnboundedPct: pct(unbounded / n),
    graphDoors: built.doorReport.filter((d) => d.node >= 0).length,
    joins: s.joins,
    vents: s.vents,
    insideOneRoom: s.inside,
    noRoom: s.unplaced,
    doorsSizedFromIfc: s.sizedFromIfc,
  });
  if (!s.rooms) {
    rows.push({ ...row, status: "no IfcSpace rooms (graph models only)" });
    console.log(`${row.model.slice(0, 34).padEnd(34)} n ${String(n).padStart(6)}  no IfcSpace rooms`);
    continue;
  }

  // The fire: in the largest room with lattice nodes in its footprint and a
  // door in the graph (the largest space is often a gross-area or roof space
  // with no door, which would burn on the nominal leak), else the largest
  // with nodes; at the node nearest the middle of that room's nodes.
  const members = Array.from({ length: s.rooms }, () => []);
  for (let v = 0; v < n; v += 1) if (rg.direct[v] >= 0) members[rg.direct[v]].push(v);
  const largest = (ok) => {
    let best = -1;
    rg.rooms.forEach((room, k) => {
      if (members[k].length && ok(room) && (best < 0 || room.floorArea > rg.rooms[best].floorArea)) best = k;
    });
    return best;
  };
  const anyRoom = largest(() => true);
  let fireRoom = largest((room) => room.doors.length > 0);
  row.fireRoomRule = fireRoom >= 0 ? "largest room with a door" : "largest room (none has a door)";
  if (fireRoom < 0) fireRoom = anyRoom;
  if (fireRoom < 0) {
    rows.push({ ...row, status: "no room holds a lattice node" });
    console.log(`${row.model.slice(0, 34).padEnd(34)} no room holds a lattice node`);
    continue;
  }
  row.largestRoomM2 = +rg.rooms[anyRoom].floorArea.toFixed(1);
  row.largestRoomDoors = rg.rooms[anyRoom].doors.length;
  const list = members[fireRoom];
  let cx = 0, cz = 0;
  for (const v of list) {
    cx += g.nodes[v * 3];
    cz += g.nodes[v * 3 + 2];
  }
  cx /= list.length;
  cz /= list.length;
  let seed = list[0], best = Infinity;
  for (const v of list) {
    const d = (g.nodes[v * 3] - cx) ** 2 + (g.nodes[v * 3 + 2] - cz) ** 2;
    if (d < best) [seed, best] = [v, d];
  }

  t = performance.now();
  const forecast = buildZoneForecast(g, rg, { seed, growth: GROWTH, peak: PEAK, untenableC: UNTENABLE_C, horizon: HORIZON, frameSeconds: FRAME });
  row.forecastMs = +(performance.now() - t).toFixed(1);
  const z = forecast.zone;
  Object.assign(row, {
    fireRoomAreaM2: +z.fireRoomArea.toFixed(1),
    fireRoomHeightM: +z.fireRoomHeight.toFixed(2),
    fireRoomNodes: list.length,
    ventAreaM2: +z.ventArea.toFixed(2),
    ventHeightM: +z.ventHeight.toFixed(2),
    leakageOnly: z.leakage,
    peakKw: Math.round(z.Qmax),
    ventLimited: z.ventLimited,
    flashoverS: z.flashover,
    fireRoomUntenableS: Number.isFinite(z.fireRoomUntenable) ? z.fireRoomUntenable : null,
    smokeRooms300: z.roomsReached300,
    smokeRooms600: z.roomsReached,
    untenableRooms600: z.roomsUntenable,
    hottestC: Math.round(z.hottestC),
    stride: forecast.stride,
  });

  // The routers on the zone forecast.
  let exits = [];
  try {
    ({ exits } = findExits(geo, built));
  } catch {
    exits = [];
  }
  row.exits = exits.length;
  if (!exits.length) {
    row.status = "ok (no exits: routers not checked)";
    rows.push(row);
    console.log(`${row.model.slice(0, 30).padEnd(30)} rooms ${row.rooms} ${row.assignedPct}% flashover ${row.flashoverS ?? "-"} smoke@300 ${row.smokeRooms300}  no exits`);
    continue;
  }
  try {
    const opts = { exits, forecast, speed: WALK, blocked: g.blocked };
    const hard = escapeDeadlines(g, { exits, untenable: forecast.closes, speed: WALK, blocked: g.blocked });
    const can = [];
    for (let v = 0; v < n; v += 1) if (hard[v] > 0) can.push(v);
    row.canLeavePct = pct(can.length / n);
    const starts = can.length ? Array.from({ length: 5 }, (_, i) => can[Math.floor(((i + 0.5) * can.length) / 5)]) : [];
    let agree = 0, planned = 0, truncated = 0, planMs = 0;
    for (const start of starts) {
      t = performance.now();
      const plan = planWithForecast(g, { start, ...opts });
      planMs = Math.max(planMs, performance.now() - t);
      planned += 1;
      if (plan.truncated) truncated += 1;
      if (plan.earliest && plan.fastest && Math.abs(plan.earliest.arrival - plan.fastest.arrival) <= 1e-9 * Math.max(1, plan.earliest.arrival)) agree += 1;
      else if (!plan.earliest && !plan.fastest) agree += 1;
    }
    row.routes = `${agree}/${planned}`;
    row.planMs = +planMs.toFixed(1);
    row.truncated = truncated;
    t = performance.now();
    const map = doseDeadlineMap(g, { ...opts, limit: LIMIT });
    row.mapMs = Math.round(performance.now() - t);
    let checked = 0, works = 0, notLater = 0, gap = 0;
    const sample = can.length ? Array.from({ length: 8 }, (_, i) => can[Math.floor(((i + 0.37) * can.length) / 8) % can.length]) : [];
    for (const v of sample) {
      if (map.deadline[v] === -Infinity) continue;
      const exact = doseDeadline(g, v, { ...opts, limit: LIMIT, tolerance: 0.02 });
      checked += 1;
      const { front } = doseFront(g, v, { ...opts, depart: map.deadline[v] });
      if (front.length && front[front.length - 1].dose <= LIMIT) works += 1;
      if (exact && map.deadline[v] <= exact.bad + 1e-9) {
        notLater += 1;
        gap = Math.max(gap, exact.good - map.deadline[v]);
      }
    }
    row.mapCheck = `${works}/${checked} work, ${notLater}/${checked} not later`;
    row.mapGapS = +gap.toFixed(2);
    row.status = agree === planned && works === checked && notLater === checked && !truncated ? "ok" : "MISMATCH";
  } catch (error) {
    row.status = `error: ${error.message}`;
  }
  rows.push(row);
  console.log(
    `${row.model.slice(0, 30).padEnd(30)} ${String(row.status).padEnd(8)} n ${String(n).padStart(6)} rooms ${String(row.rooms).padStart(4)} ` +
      `in ${String(row.assignedPct).padStart(5)}% joins ${String(row.joins).padStart(3)} vents ${String(row.vents).padStart(3)} | ` +
      `fire ${row.fireRoomAreaM2} m2 peak ${row.peakKw} kW flashover ${row.flashoverS ?? "none"} smoke@300 ${row.smokeRooms300}/${row.rooms} | ` +
      `rooms ${row.roomGraphMs} ms zone ${row.forecastMs} ms routes ${row.routes} map ${row.mapMs} ms ${row.mapCheck} gap ${row.mapGapS} s`,
  );
}

if (jsonOut) {
  writeFileSync(
    jsonOut,
    JSON.stringify(
      {
        date: new Date().toISOString().slice(0, 10),
        spec: "wiki/research/zone-model-spec-2026-10-28.md",
        fire: { growth: GROWTH, alpha: 0.0117, peakAskedKw: PEAK, horizonS: HORIZON, frameSeconds: FRAME, eulerStepS: 0.5, untenableC: UNTENABLE_C, smokeCriterion: true },
        walkSpeed: WALK,
        doseLimit: LIMIT,
        rows,
      },
      null,
      1,
    ),
  );
}
