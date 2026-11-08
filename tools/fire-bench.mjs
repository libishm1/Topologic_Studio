/**
 * Speed and sanity of every fire model the Simulate step offers, on the
 * same models from the same fire origin:
 *
 *   radial, flood, temperature   the graph models (src/core/fire.js), 60 steps
 *   zone                         the two-zone room model (src/core/zone.js),
 *                                600 s, on models with IfcSpace rooms
 *   flamescapes, -embers         the Flamescapes solver on the CPU
 *                                (src/core/flamescapesRun.js), `--horizon` s
 *
 * For each: the time to run the model, the time to build the planners'
 * forecast from it, and what it gives (nodes reached, nodes untenable; for
 * Flamescapes the grid, ms per step and per stage, heat release, embers and
 * any non-finite value). Results go to benchmark/results/<date>-fire.json.
 *
 *   node tools/fire-bench.mjs [model.ifc ...] [--horizon 60] [--cells 30000] [--out file] [--sprinklers]
 *
 * With --sprinklers, each model with IfcSpace rooms also runs Flamescapes
 * with sprinklers placed per room by NFPA 13 (src/core/sprinklers.js), open
 * heads capping the fuel and spraying (src/core/suppression.js): its ms per
 * step against the run without, and when the first head opens.
 *
 * Without models it runs the four examples shipped with the site.
 */
import { mkdir, readdir, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extract } from "./ifc-extract.mjs";
import { buildLatticeGraph } from "../topologicpy-web-frontend/src/core/lattice.js";
import { floodTimeline, radialTimeline, temperatureSteps } from "../topologicpy-web-frontend/src/core/fire.js";
import { buildForecast } from "../topologicpy-web-frontend/src/core/forecast.js";
import { buildRoomGraph } from "../topologicpy-web-frontend/src/core/rooms.js";
import { buildZoneForecast } from "../topologicpy-web-frontend/src/core/zone.js";
import { buildScene, forecastFromFrames, runScene } from "../topologicpy-web-frontend/src/core/flamescapesRun.js";
import { layoutRooms, roomsFromSpaces } from "../topologicpy-web-frontend/src/core/sprinklers.js";
import { fuelsFor } from "../topologicpy-web-frontend/src/core/fireClasses.js";
import { floorsFromSpaces, slabsMissing } from "../topologicpy-web-frontend/src/core/spaceFloors.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const horizon = Number(flag("--horizon", 60));
const maxCells = Number(flag("--cells", 30000));
const withSprinklers = args.includes("--sprinklers");
let models = args.filter((a) => a.toLowerCase().endsWith(".ifc"));
if (!models.length) {
  const dir = path.join(HERE, "../topologicpy-web-frontend/public/examples");
  models = (await readdir(dir)).filter((f) => f.endsWith(".ifc")).map((f) => path.join(dir, f));
}

const ms = (t0) => performance.now() - t0;
const rss = () => Math.round(process.memoryUsage().rss / 2 ** 20);

/** The node nearest the middle of the lowest storey's floor. */
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

const results = [];
for (const file of models) {
  const name = path.basename(file);
  const t0 = performance.now();
  const geo = await extract(file, { withDoorFlags: true });
  const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
  const graph = { nodes: built.points, edges: built.edges, blocked: built.blocked };
  const n = graph.nodes.length / 3;
  const seed = seedNode(graph.nodes);
  const row = { model: name, nodes: n, edges: graph.edges.length / 2, spaces: geo.spaces.length, prepareMs: ms(t0), methods: {} };
  console.log(`\n${name}: ${n} nodes, ${geo.spaces.length} spaces, seed ${seed}`);

  // The graph models.
  for (const model of ["radial", "flood", "temperature"]) {
    const t1 = performance.now();
    let reached = 0;
    if (model === "temperature") {
      let last = null;
      for (const frame of temperatureSteps(graph, seed, { maxSteps: 60, blocked: graph.blocked })) last = frame;
      for (let v = 0; v < n; v += 1) if (last[v] > 20.5) reached += 1;
    } else {
      const timeline = model === "flood" ? floodTimeline(graph, seed, 60, graph.blocked) : radialTimeline(graph, seed, 60);
      for (const nodes of timeline) reached += nodes.length;
    }
    const runMs = ms(t1);
    const t2 = performance.now();
    const forecast = buildForecast(graph, { model, seed, steps: 60, dt: 5, threshold: 80, blocked: graph.blocked });
    const forecastMs = ms(t2);
    let untenable = 0;
    for (let v = 0; v < n; v += 1) if (forecast.untenable[v] < forecast.horizon) untenable += 1;
    row.methods[model] = { runMs, forecastMs, reached, untenable, rssMB: rss() };
    console.log(`  ${model.padEnd(19)} run ${runMs.toFixed(1).padStart(8)} ms  forecast ${forecastMs.toFixed(1).padStart(8)} ms  reached ${reached}  untenable ${untenable}`);
  }

  // The two-zone room model, where the model has rooms.
  if (geo.spaces.length) {
    const t1 = performance.now();
    const doors = built.doorReport.map((d, i) => ({ ...d, ux: built.doorPoints[i][3], uy: built.doorPoints[i][4] }));
    const rooms = buildRoomGraph(graph, { spaces: geo.spaces, doors, declaredDoors: geo.semantics?.doors || [], upAxis: "y", agentHeight: 0.75 });
    const roomsMs = ms(t1);
    if (rooms.rooms.length) {
      const t2 = performance.now();
      const forecast = buildZoneForecast(graph, rooms, { seed, growth: "medium", peak: 5000, untenableC: 60, smoke: true, horizon: 600 });
      const runMs = ms(t2);
      let untenable = 0;
      for (let v = 0; v < n; v += 1) if (forecast.untenable[v] < forecast.horizon) untenable += 1;
      row.methods.zone = { roomsMs, runMs, rooms: rooms.rooms.length, untenable, rssMB: rss() };
      console.log(`  ${"zone".padEnd(19)} rooms ${roomsMs.toFixed(1)} ms, run and forecast ${runMs.toFixed(1)} ms  rooms ${rooms.rooms.length}  untenable ${untenable}`);
    } else {
      row.methods.zone = { skipped: "the spaces give no rooms" };
    }
  } else {
    row.methods.zone = { skipped: "no IfcSpace rooms" };
  }

  // Flamescapes, without and with embers, on the CPU.
  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  const fire = [graph.nodes[seed * 3], graph.nodes[seed * 3 + 1], graph.nodes[seed * 3 + 2]];
  // With --sprinklers, a third run: heads placed per room, capping and spraying.
  let heads = null;
  if (withSprinklers && geo.spaces.length) {
    const t1 = performance.now();
    const names = new Map((geo.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]));
    heads = layoutRooms(roomsFromSpaces(geo.spaces, { upAxis: "y", names, overhead: [...geo.floors, ...(geo.roofs || [])], ceilings: geo.ceilings, walls: geo.walls })).heads;
    row.sprinklerLayout = { heads: heads.length, ms: ms(t1) };
    console.log(`  ${"sprinkler layout".padEnd(19)} ${heads.length} heads in ${ms(t1).toFixed(0)} ms`);
  }
  for (const [embers, sprinklers] of [[false, null], [true, null], ...(heads?.length ? [[false, { heads, mode: "spray" }]] : [])]) {
    const key = sprinklers ? "flamescapes-sprinklers" : embers ? "flamescapes-embers" : "flamescapes";
    try {
      const t1 = performance.now();
      const scene = buildScene(slabsMissing(geo.floors, geo.spaces) ? { ...geo, floors: [...geo.floors, ...floorsFromSpaces(geo.spaces)] } : geo, { upAxis: "y", fire, maxCells, contentsFuel, ignitionFuel, item });
      const setupMs = ms(t1);
      if (!scene.ignition) {
        row.methods[key] = { skipped: "no place for the first item at the fire origin" };
        console.log(`  ${key.padEnd(19)} skipped: no place for the first item`);
        continue;
      }
      const profile = {};
      let peak = 0;
      const t2 = performance.now();
      const frames = runScene(scene, {
        nodes: graph.nodes,
        horizon,
        frameSeconds: 5,
        embers,
        particles: 0,
        solverOptions: {},
        sprinklers,
        onFrame: (f) => { if (f.hrr > peak) peak = f.hrr; },
        profileInto: profile,
      });
      const runMs = ms(t2);
      const steps = Math.round(horizon / 0.1);
      let nonFinite = 0, hot = 0, smoky = 0;
      const last = (frames.K - 1) * n;
      for (const a of [frames.T, frames.vis, frames.co]) for (const x of a) if (!Number.isFinite(x)) nonFinite += 1;
      for (let v = 0; v < n; v += 1) {
        if (frames.T[last + v] >= 60) hot += 1;
        if (frames.vis[last + v] < 10) smoky += 1;
      }
      const t3 = performance.now();
      const forecast = forecastFromFrames(frames, { threshold: 60, smoke: true, visibilityLimit: 10 });
      const forecastMs = ms(t3);
      let untenable = 0;
      for (let v = 0; v < n; v += 1) if (forecast.untenable[v] < forecast.horizon) untenable += 1;
      const g = scene.grid;
      const stages = Object.fromEntries(Object.entries(profile).map(([k, v]) => [k, v / steps]));
      row.methods[key] = {
        grid: [g.nx, g.ny, g.nz],
        cell: g.h,
        cells: g.cells,
        clipped: scene.clipped,
        setupMs,
        runMs,
        msPerStep: runMs / steps,
        realTime: horizon / (runMs / 1000),
        stagesMsPerStep: stages,
        peakHrrKw: peak,
        embers: frames.stats.embersSpawned,
        spots: frames.stats.spotIgnitions,
        nonFinite,
        hotAtEnd: hot,
        smokyAtEnd: smoky,
        untenable,
        forecastMs,
        nodesInGrid: frames.inGrid,
        rssMB: rss(),
        sprinklers: frames.sprinklers
          ? {
              heads: frames.sprinklers.heads.length,
              onGrid: frames.sprinklers.heads.filter((h) => h.onGrid).length,
              firstEstimate: Math.min(...frames.sprinklers.heads.map((h) => h.estimate ?? Infinity)),
              firstLate: Math.min(...frames.sprinklers.heads.map((h) => h.late ?? Infinity)),
              open: frames.sprinklers.open.length,
              water: frames.sprinklers.stats,
            }
          : null,
      };
      const top = Object.entries(stages).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(", ");
      console.log(`  ${key.padEnd(19)} grid ${g.nx}x${g.ny}x${g.nz} (${g.h.toFixed(2)} m)  ${(runMs / steps).toFixed(1)} ms/step  x${(horizon / (runMs / 1000)).toFixed(2)} real time  peak ${peak.toFixed(0)} kW  embers ${frames.stats.embersSpawned}/${frames.stats.spotIgnitions}  untenable ${untenable}  non-finite ${nonFinite}`);
      console.log(`  ${"".padEnd(19)} slowest stages (ms/step): ${top}`);
    } catch (error) {
      row.methods[key] = { error: error.message };
      console.log(`  ${key.padEnd(19)} error: ${error.message}`);
    }
  }
  results.push(row);
}

const out = flag("--out", path.join(HERE, `../benchmark/results/${new Date().toISOString().slice(0, 10)}-fire.json`));
await mkdir(path.dirname(out), { recursive: true });
await writeFile(out, JSON.stringify({ machine: { cpu: os.cpus()[0]?.model, cores: os.cpus().length, node: process.version }, horizon, maxCells, results }, null, 1));
console.log(`\nwritten ${path.relative(process.cwd(), out)}`);
