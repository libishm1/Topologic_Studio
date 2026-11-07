/**
 * Speed of the features added on 2026-11-07, which tools/benchmark.mjs and
 * tools/fire-bench.mjs do not time: the plan with its deadline maps (and the
 * second map with the exit nearest the fire blocked), the performance check
 * with each of its options (measured pre-movement, no response, door
 * queues), a report's cases and its HTML, and the sprinkler layout with its
 * rule and requirement reports.
 *
 *   node tools/feature-bench.mjs [model.ifc ...] [--runs 5] [--out file]
 *
 * Without models, the site examples. Each timing is the median of --runs
 * repeats after one warm-up. Results go to benchmark/results/<date>-features.json.
 */
import { mkdir, readdir, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extract } from "./ifc-extract.mjs";
import { buildLatticeGraph, floorLevels } from "../topologicpy-web-frontend/src/core/lattice.js";
import { exitDoors } from "../topologicpy-web-frontend/src/core/ifcSemantics.js";
import { buildForecast, planWithForecast } from "../topologicpy-web-frontend/src/core/forecast.js";
import { doseDeadlineMap } from "../topologicpy-web-frontend/src/core/deadline.js";
import { performanceCheck } from "../topologicpy-web-frontend/src/core/performanceCheck.js";
import { presetOf, criterionOf, measuredOf } from "../topologicpy-web-frontend/src/core/rsetPresets.js";
import { planCases, reportHtml } from "../topologicpy-web-frontend/src/core/perfReport.js";
import { layoutRooms, roomsFromSpaces, requirementReport } from "../topologicpy-web-frontend/src/core/sprinklers.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const RUNS = Number(flag("--runs", 5));
let models = args.filter((a) => a.toLowerCase().endsWith(".ifc"));
if (!models.length) {
  const dir = path.join(HERE, "../topologicpy-web-frontend/public/examples");
  models = (await readdir(dir)).filter((f) => f.endsWith(".ifc")).map((f) => path.join(dir, f));
}

/** Median of RUNS timed calls after one warm-up; the last result is kept. */
function time(fn) {
  let out = fn();
  const t = [];
  for (let r = 0; r < RUNS; r += 1) {
    const t0 = performance.now();
    out = fn();
    t.push(performance.now() - t0);
  }
  t.sort((a, b) => a - b);
  return { ms: Math.round(t[Math.floor(t.length / 2)] * 100) / 100, out };
}

/** The walkable node nearest the middle of the lowest storey. */
function seedNode(nodes) {
  const n = nodes.length / 3;
  let lo = Infinity;
  for (let v = 0; v < n; v += 1) lo = Math.min(lo, nodes[v * 3 + 1]);
  let sx = 0, sz = 0, c = 0;
  for (let v = 0; v < n; v += 1) if (nodes[v * 3 + 1] < lo + 0.5) { sx += nodes[v * 3]; sz += nodes[v * 3 + 2]; c += 1; }
  let best = 0, bd = Infinity;
  for (let v = 0; v < n; v += 1) {
    const d = (nodes[v * 3] - sx / c) ** 2 + (nodes[v * 3 + 2] - sz / c) ** 2 + 100 * Math.abs(nodes[v * 3 + 1] - lo);
    if (d < bd) { bd = d; best = v; }
  }
  return best;
}

const office = presetOf("office");
const criterion = criterionOf("base");
const sensitivity = criterionOf("sensitivity");
const results = [];
for (const file of models) {
  const name = path.basename(file);
  const geo = await extract(file, { withDoorFlags: true });
  const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
  const graph = { nodes: built.points, edges: built.edges, blocked: built.blocked, kinds: built.kinds, doors: built.doorReport, spacing: 0.5 };
  const n = graph.nodes.length / 3;
  const exits = exitDoors(built.doorReport, geo.semantics, graph);
  const exitNodes = exits.final?.length ? exits.final : exits.nodes;
  const row = { model: name, nodes: n, exits: exitNodes.length, spaces: geo.spaces.length, ms: {} };
  console.log(`\n${name}: ${n} nodes, ${exitNodes.length} exits, ${geo.spaces.length} spaces`);
  if (exitNodes.length && n) {
    const seed = seedNode(graph.nodes);
    const levels = floorLevels(geo.floors, "y");
    const blocked = graph.blocked;
    const forecast = time(() => buildForecast(graph, { model: "temperature", seed, steps: 60, dt: 5, threshold: 80, blocked }));
    row.ms.forecast = forecast.ms;
    row.ms.plan = time(() => planWithForecast(graph, { start: seed, exits: exitNodes, forecast: forecast.out, depart: 0, speed: 1.2, blocked })).ms;
    const map = time(() => doseDeadlineMap(graph, { exits: exitNodes, forecast: forecast.out, limit: 0.3, speed: 1.2, blocked }));
    row.ms.deadlineMap = map.ms;
    const common = {
      levels,
      exits: exitNodes,
      horizon: forecast.out.horizon,
      step: map.out.step,
      speed: 1.2,
      detection: 60,
      alarm: office.values.perfAlarm,
      pre1: office.values.perfPre1,
      pre99: office.values.perfPre99,
      factor: criterion.factor,
      margin: criterion.margin,
    };
    const deadlines = map.out.deadline;
    row.ms.checkDesign = time(() => performanceCheck(graph, deadlines, common)).ms;
    const measured = measuredOf(office.values.perfMeasured);
    row.ms.checkMeasured = time(() => performanceCheck(graph, deadlines, { ...common, measured, noResponse: 0.05 })).ms;
    const queues = { occupantArea: office.values.perfOccupantArea, cellArea: 0.25, widths: null, defaultWidth: 0.813 };
    const withQueues = time(() => performanceCheck(graph, deadlines, { ...common, measured, noResponse: 0.05, queues }));
    row.ms.checkQueues = withQueues.ms;
    row.status = withQueues.out.status;
    const plan = { deadlines, horizon: forecast.out.horizon, mapStep: map.out.step };
    const cases = time(() => planCases(performanceCheck, graph, plan, { ...common, criterion, sensitivity, queues }));
    row.ms.planCases = cases.ms;
    const entries = [1, 2, 3, 4].map((k) => ({ plan, label: `Fire ${k}`, model: "temperature", detectionFrom: "set", cases: cases.out }));
    row.ms.reportHtml4 = time(() => reportHtml(entries, { title: "Bench", model: name, date: "", inputs: [["Preset", office.label, ""]], limits: ["bench"] })).ms;
    console.log(`  forecast ${row.ms.forecast} ms, plan ${row.ms.plan}, deadline map ${row.ms.deadlineMap}; check ${row.ms.checkDesign} / measured ${row.ms.checkMeasured} / queues ${row.ms.checkQueues}; cases ${row.ms.planCases}; report of 4 fires ${row.ms.reportHtml4} ms (${row.status})`);
  }
  if (geo.spaces.length) {
    const names = new Map((geo.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]));
    const layout = time(() => layoutRooms(roomsFromSpaces(geo.spaces, { upAxis: "y", names })));
    row.ms.sprinklerLayout = layout.ms;
    row.sprinklerHeads = layout.out.heads.length;
    row.ms.requirement = time(() => requirementReport({ occupancy: ["B"], storeyElevations: floorLevels(geo.floors, "y") })).ms;
    console.log(`  sprinkler layout ${row.ms.sprinklerLayout} ms (${row.sprinklerHeads} heads), requirement ${row.ms.requirement} ms`);
  }
  results.push(row);
}

const out = flag("--out", path.join(HERE, `../benchmark/results/${new Date().toISOString().slice(0, 10)}-features.json`));
await mkdir(path.dirname(out), { recursive: true });
await writeFile(out, JSON.stringify({ date: new Date().toISOString(), machine: { cpu: os.cpus()[0]?.model, cores: os.cpus().length }, runs: RUNS, results }, null, 2));
console.log(`\nWrote ${path.relative(process.cwd(), out)}`);
