/**
 * Cross-model check of planning with the fire forecast (src/core/forecast.js)
 * and of the escape-deadline maps (src/core/deadline.js).
 *
 *   node tools/forecast-check.mjs [model.ifc ...] [--json out.json]
 *
 * For each IFC model: build the walkable graph as the site does, find the
 * exits (tools/exits.mjs), start a fire in the middle of the largest part,
 * forecast 60 steps of 5 s with each fire model, then
 *   - from sampled starts: the earliest safe arrival (time-dependent Dijkstra)
 *     must equal the first point of the time-dose front, the front must not be
 *     truncated, and a route planned on the fire at departure is walked under
 *     the forecast to see whether it stays safe;
 *   - the dose-limited deadline map (FED 0.3) is compared at sampled nodes
 *     with the exact deadline (bisection on the exact least-dose search): the
 *     map's departure must work in the exact model and never be later.
 *
 * Speeds and the step length are illustrative (1.2 m/s walking, 5 s per fire
 * step, untenable from 80 model degrees C): this checks that the methods run,
 * agree and are fast on each model, not what to conclude about a building.
 */
import { readdirSync, existsSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

import { buildLatticeGraph } from "../topologicpy-web-frontend/src/core/lattice.js";
import { componentLabels } from "../topologicpy-web-frontend/src/core/route.js";
import { buildForecast, doseFront, planWithForecast } from "../topologicpy-web-frontend/src/core/forecast.js";
import { doseDeadline, doseDeadlineMap, escapeDeadlines } from "../topologicpy-web-frontend/src/core/deadline.js";
import { findExits } from "./exits.mjs";
import { extract } from "./ifc-extract.mjs";

const args = process.argv.slice(2);
const jsonAt = args.indexOf("--json");
const jsonOut = jsonAt >= 0 ? args[jsonAt + 1] : null;
let files = args.filter((a, i) => a.toLowerCase().endsWith(".ifc") && i !== jsonAt + 1);

/** Every .ifc under a folder, one level of sub-folders deep. */
function ifcsIn(dir) {
  const out = [];
  if (!existsSync(dir)) return out;
  for (const f of readdirSync(dir)) {
    const p = path.join(dir, f);
    if (f.toLowerCase().endsWith(".ifc")) out.push(p);
    else if (statSync(p).isDirectory()) for (const g of readdirSync(p)) if (g.toLowerCase().endsWith(".ifc")) out.push(path.join(p, g));
  }
  return out;
}

if (!files.length) {
  for (const d of ["C:/Users/lmurugesan/ifc-samples", "topologicpy-web-frontend/public/examples", "C:/Users/lmurugesan/Downloads"]) files.push(...ifcsIn(d));
  // The Downloads copy of the Duplex is the same file as the shipped example.
  files = files.filter((f) => !/Ifc2x3_Duplex_Architecture\.ifc$/i.test(f));
}

const WALK = 1.2, DT = 5, STEPS = 60, THRESHOLD = 80, LIMIT = 0.3;
const rows = [];

for (const file of files) {
  const base = { model: path.basename(file) };
  let geo, built, g, n, exits, source;
  try {
    const t0 = performance.now();
    geo = await extract(path.resolve(file), { withDoorFlags: true });
    built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
    base.graphMs = Math.round(performance.now() - t0);
    g = { ...built, nodes: built.points };
    n = g.nodes.length / 3;
    ({ exits, source } = findExits(geo, built));
  } catch (error) {
    rows.push({ ...base, status: `error: ${error.message}` });
    console.log(`${base.model.slice(0, 34).padEnd(34)} error: ${error.message}`);
    continue;
  }
  base.nodes = n;
  base.exits = exits.length;
  base.exitSource = source;
  if (!n || !exits.length) {
    rows.push({ ...base, status: !n ? "no walkable graph" : "no exits" });
    console.log(`${base.model.slice(0, 34).padEnd(34)} ${!n ? "no walkable graph" : "no exits"}`);
    continue;
  }
  const { labels, sizes } = componentLabels(g);
  const big = sizes.indexOf(Math.max(...sizes));
  const members = [];
  for (let v = 0; v < n; v += 1) if (labels[v] === big) members.push(v);
  const seed = members[Math.floor(members.length / 2)];

  for (const model of ["temperature", "flood"]) {
    const row = { ...base, fire: model };
    try {
      let t = performance.now();
      const forecast = buildForecast(g, { model, seed, steps: STEPS, dt: DT, threshold: THRESHOLD, blocked: g.blocked });
      row.forecastMs = +(performance.now() - t).toFixed(1);
      const opts = { exits, forecast, speed: WALK, blocked: g.blocked };
      const hard = escapeDeadlines(g, { exits, untenable: forecast.closes, speed: WALK, blocked: g.blocked });

      // Starts: nodes that can still leave at time 0, spread over the graph.
      const can = [];
      for (let v = 0; v < n; v += 1) if (hard[v] > 0) can.push(v);
      row.canLeave = +(can.length / n).toFixed(3);
      const starts = can.length ? Array.from({ length: 5 }, (_, i) => can[Math.floor(((i + 0.5) * can.length) / 5)]) : [];
      let agree = 0, planned = 0, unsafeSnapshots = 0, tradeoffs = 0, truncated = 0, planMs = 0, maxLabels = 0;
      for (const s of starts) {
        t = performance.now();
        const plan = planWithForecast(g, { start: s, ...opts });
        planMs = Math.max(planMs, performance.now() - t);
        planned += 1;
        maxLabels = Math.max(maxLabels, plan.labels);
        if (plan.truncated) truncated += 1;
        if (plan.earliest && plan.fastest && Math.abs(plan.earliest.arrival - plan.fastest.arrival) <= 1e-9 * Math.max(1, plan.earliest.arrival)) agree += 1;
        if (plan.front.length > 1) tradeoffs += 1;
        if (plan.snapshot && !plan.snapshot.safe && !plan.snapshot.beyondForecast) unsafeSnapshots += 1;
      }
      row.routes = `${agree}/${planned}`;
      row.planMs = +planMs.toFixed(1);
      row.maxLabels = maxLabels;
      row.truncated = truncated;
      row.tradeoffs = tradeoffs;
      row.snapshotUnsafe = unsafeSnapshots;

      t = performance.now();
      const map = doseDeadlineMap(g, { ...opts, limit: LIMIT });
      row.mapMs = Math.round(performance.now() - t);
      row.mapStep = +map.step.toFixed(3);
      row.mapLayers = map.layers;
      let checked = 0, works = 0, notLater = 0, gap = 0;
      const sample = can.length ? Array.from({ length: 12 }, (_, i) => can[Math.floor(((i + 0.37) * can.length) / 12) % can.length]) : [];
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
      `${row.model.slice(0, 30).padEnd(30)} ${model.padEnd(11)} ${String(row.status).padEnd(8)} n ${String(n).padStart(6)} ` +
        `routes ${row.routes ?? "-"} plan ${row.planMs ?? "-"} ms (labels ${row.maxLabels ?? "-"}, trade-offs ${row.tradeoffs ?? "-"}, snapshot unsafe ${row.snapshotUnsafe ?? "-"}) ` +
        `map ${row.mapMs ?? "-"} ms step ${row.mapStep ?? "-"} s, ${row.mapCheck ?? "-"}, gap ${row.mapGapS ?? "-"} s`,
    );
  }
}

if (jsonOut) writeFileSync(jsonOut, JSON.stringify({ walkSpeed: WALK, secondsPerStep: DT, steps: STEPS, threshold: THRESHOLD, doseLimit: LIMIT, rows }, null, 1));
