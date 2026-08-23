/**
 * End-to-end smoke test of the Next fast path, without a browser.
 *
 * Runs the real sampler worker source against real IFC geometry, then drives
 * the live backend exactly as the app does:
 *
 *   triangles -> sampler.worker -> POST /api/ifc/graph -> POST /api/ifc/path
 *
 * Usage:
 *   node tools/smoke-pipeline.mjs <payload.json> [apiBase]
 *
 * The payload is the cache file written by tools/bench_egress.py.
 */
import { readFile } from "node:fs/promises";

import { performance } from "node:perf_hooks";

const [, , payloadPath, apiBaseArg] = process.argv;
const API = (apiBaseArg || "http://127.0.0.1:8000").replace(/\/$/, "");

if (!payloadPath) {
  console.error("usage: node tools/smoke-pipeline.mjs <payload.json> [apiBase]");
  process.exit(2);
}

let failures = 0;
function check(label, condition, detail = "") {
  const ok = Boolean(condition);
  if (!ok) failures += 1;
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  ${detail}` : ""}`);
  return ok;
}

// --- 1. Load geometry and shape it the way ViewerManager does ---------------

console.log(`\n[1] Loading geometry from ${payloadPath}`);
const raw = JSON.parse(await readFile(payloadPath, "utf8"));

const toMeshes = (items) =>
  items.map((g) => ({
    positions: new Float32Array(g.vertices),
    indices: new Uint32Array(g.indices),
  }));

const geometry = {
  floors: toMeshes(raw.floors || []),
  stairs: toMeshes(raw.stairs || []),
  doors: toMeshes(raw.doors || []),
  walls: toMeshes(raw.walls || []),
};

const triangleCount = Object.values(geometry).reduce(
  (sum, meshes) => sum + meshes.reduce((n, m) => n + m.indices.length / 3, 0),
  0,
);
const classicPayloadBytes = Buffer.byteLength(JSON.stringify(raw));
console.log(
  `    ${triangleCount.toLocaleString()} triangles; ` +
    `Classic would upload ${(classicPayloadBytes / 1e6).toFixed(1)} MB of JSON`,
);

// --- 2. Run the real worker source -----------------------------------------

console.log("\n[2] Running the sampler worker");

let workerOnMessage = null;
let resolveResult;
const resultPromise = new Promise((resolve) => {
  resolveResult = resolve;
});

// Minimal WorkerGlobalScope so the worker module can be imported as-is.
globalThis.self = {
  set onmessage(fn) {
    workerOnMessage = fn;
  },
  get onmessage() {
    return workerOnMessage;
  },
  postMessage(message) {
    resolveResult(message);
  },
};
globalThis.performance = performance;

await import(
  new URL("../topologicpy-web-frontend/src/workers/sampler.worker.js", import.meta.url).href
);

check("worker registered an onmessage handler", typeof workerOnMessage === "function");

const t0 = performance.now();
workerOnMessage({
  data: {
    id: "smoke",
    payload: {
      ...geometry,
      upAxis: "z",
      floorSpacing: 0.5,
      maxPoints: 40000,
    },
  },
});
const message = await resultPromise;
const sampleMs = performance.now() - t0;

check("worker returned ok", message.ok, message.error || "");
const sampled = message.result;
check("floor points produced", sampled.floorPoints.length > 0,
  `${sampled.floorPoints.length / 3} pts`);
check("stair points produced", sampled.stairPoints.length > 0,
  `${sampled.stairPoints.length / 3} pts`);
check("door waypoints produced", sampled.doorPoints.length > 0,
  `${sampled.doorPoints.length}`);
check("wall segments produced", sampled.walls.length > 0, `${sampled.walls.length}`);
console.log(`    sampling took ${sampleMs.toFixed(0)} ms`);

// --- 3. Build the graph over HTTP ------------------------------------------

console.log("\n[3] POST /api/ifc/graph");

const toTriples = (flat) => {
  const out = [];
  for (let i = 0; i < flat.length; i += 3) out.push([flat[i], flat[i + 1], flat[i + 2]]);
  return out;
};

const body = {
  floor_points: toTriples(sampled.floorPoints),
  stair_points: toTriples(sampled.stairPoints),
  door_points: sampled.doorPoints,
  walls: sampled.walls,
  options: {
    up_axis: "z",
    agent_height: 0.75,
    max_edge_floor: 2.25,
    max_edge_stair: 0.4,
    max_degree: 12,
    use_walls: true,
  },
};
const uploadBytes = Buffer.byteLength(JSON.stringify(body));
console.log(
  `    upload ${(uploadBytes / 1e6).toFixed(2)} MB ` +
    `(${(classicPayloadBytes / uploadBytes).toFixed(1)}x smaller than Classic)`,
);

let response;
try {
  const started = performance.now();
  response = await fetch(`${API}/api/ifc/graph`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const ms = performance.now() - started;
  check("graph endpoint responded 200", response.ok, `${response.status} in ${ms.toFixed(0)} ms`);
} catch (error) {
  console.error(`  FAIL  cannot reach ${API}: ${error.message}`);
  process.exit(1);
}

const graph = await response.json();
if (!response.ok) {
  console.error("  detail:", graph.detail);
  process.exit(1);
}

check("graph has nodes", graph.stats.nodes > 0, `${graph.stats.nodes}`);
check("graph has edges", graph.stats.edges > 0, `${graph.stats.edges}`);
check("stair nodes present", graph.stats.stair_nodes > 0, `${graph.stats.stair_nodes}`);
check("door nodes present", graph.stats.door_nodes > 0, `${graph.stats.door_nodes}`);
check(
  "binary node payload decodes to the node count",
  Buffer.from(graph.nodes_b64, "base64").length === graph.stats.nodes * 12,
);
check(
  "binary edge payload decodes to the edge count",
  Buffer.from(graph.edges_b64, "base64").length === graph.stats.edges * 8,
);
console.log(
  `    largest connected component: ${graph.stats.largest_component} / ${graph.stats.nodes} ` +
    `(${graph.stats.components} component(s))`,
);
console.log(`    server phases: ${JSON.stringify(graph.timings)}`);

// --- 4. Route across the model ----------------------------------------------

console.log("\n[4] POST /api/ifc/path");

const nodes = new Float32Array(
  Buffer.from(graph.nodes_b64, "base64").buffer.slice(
    Buffer.from(graph.nodes_b64, "base64").byteOffset,
  ),
);
const at = (i) => [nodes[i * 3], nodes[i * 3 + 1], nodes[i * 3 + 2]];

// Pick two far-apart nodes so the route has to cross the building.
let a = 0;
let b = 0;
let best = -1;
for (let i = 0; i < graph.stats.nodes; i += Math.max(1, Math.floor(graph.stats.nodes / 60))) {
  for (let j = 0; j < graph.stats.nodes; j += Math.max(1, Math.floor(graph.stats.nodes / 60))) {
    const p = at(i);
    const q = at(j);
    const d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
    if (d > best) {
      best = d;
      a = i;
      b = j;
    }
  }
}

async function route(engine) {
  const started = performance.now();
  const res = await fetch(`${API}/api/ifc/path`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      graph_id: graph.graph_id,
      start_point: at(a),
      end_point: at(b),
      engine,
    }),
  });
  const ms = performance.now() - started;
  const payload = await res.json();
  return { payload, ms, ok: res.ok };
}

const fast = await route("fast");
check("fast engine found a route", fast.ok && fast.payload.found,
  `${fast.payload.length?.toFixed(1)} m in ${fast.ms.toFixed(0)} ms`);
check("route spans more than two waypoints", (fast.payload.points || []).length > 2,
  `${fast.payload.points?.length} pts`);

// First topologicpy call also pays the one-off Graph.ByMeshData construction,
// which is cached on the graph; the second call is the steady-state query cost.
const topoCold = await route("topologicpy");
const topo = await route("topologicpy");
check("topologicpy engine found a route", topo.ok && topo.payload.found,
  `${topo.payload.length?.toFixed(1)} m via ${topo.payload.engine}`);

if (fast.payload.found && topo.payload.found && topo.payload.engine === "topologicpy") {
  const delta = Math.abs(fast.payload.cost - topo.payload.cost);
  check("both engines agree on cost", delta < 0.01, `delta ${delta.toFixed(5)}`);
  console.log(
    `    fast ${fast.ms.toFixed(0)} ms | ` +
      `topologicpy cold ${(topoCold.ms / 1000).toFixed(1)} s (includes graph build), ` +
      `warm ${(topo.ms / 1000).toFixed(2)} s ` +
      `-> ${(topo.ms / Math.max(fast.ms, 0.01)).toFixed(0)}x slower per query`,
  );
}

// --- 5. Fire simulation ------------------------------------------------------

console.log("\n[5] Fire simulation + live rerouting");

const timelineRes = await fetch(`${API}/api/fire/timeline`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    graph_id: graph.graph_id,
    model: "flood",
    start_point: at(a),
    max_steps: 400,
  }),
});
const timeline = await timelineRes.json();
check("flood timeline produced steps", timeline.steps > 0, `${timeline.steps} steps`);
const reached = (timeline.timeline || []).reduce((n, step) => n + step.length, 0);
check(
  "fire reaches most of the graph",
  reached >= graph.stats.largest_component * 0.9,
  `${reached} / ${graph.stats.largest_component}`,
);

// Fire at `a`, evacuee at `b`, exit at a third node so the reroute has a real
// route to solve rather than a degenerate start-equals-exit.
const exitNode = Math.floor(graph.stats.nodes / 2);
const streamUrl =
  `${API}/api/fire/stream?graph_id=${graph.graph_id}&model=temperature` +
  `&max_steps=14&delay_ms=0&stream_path=true&path_recompute_interval=4&path_alpha=2` +
  `&start_x=${at(a)[0]}&start_y=${at(a)[1]}&start_z=${at(a)[2]}` +
  `&end_x=${at(exitNode)[0]}&end_y=${at(exitNode)[1]}&end_z=${at(exitNode)[2]}` +
  `&path_start_x=${at(b)[0]}&path_start_y=${at(b)[1]}&path_start_z=${at(b)[2]}`;

const streamRes = await fetch(streamUrl);
const text = await streamRes.text();
const events = text
  .split("\n\n")
  .filter((chunk) => chunk.startsWith("data: "))
  .map((chunk) => JSON.parse(chunk.slice(6)));

const kinds = events.reduce((acc, e) => {
  acc[e.type] = (acc[e.type] || 0) + 1;
  return acc;
}, {});
check("stream emitted meta", kinds.meta > 0);
check("stream emitted temperature steps", kinds.temperature_step > 0, `${kinds.temperature_step}`);
check("stream emitted path updates", kinds.path_update > 0, `${kinds.path_update}`);
check("stream terminated cleanly", kinds.done > 0);

const temps = events.filter((e) => e.type === "temperature_step");
if (temps.length > 1) {
  const first = Object.keys(temps[0].temperatures).length;
  const last = Object.keys(temps[temps.length - 1].temperatures).length;
  check("the heated region grows over time", last > first, `${first} -> ${last} hot nodes`);
}

const updates = events.filter((e) => e.type === "path_update");
check(
  "rerouting returned a usable path",
  updates.some((u) => u.found && u.path.length > 1),
  updates.length ? `${updates[0].path?.length} waypoints, cost ${updates[0].cost}` : "",
);
check(
  "hazard cost rises as the fire grows",
  updates.length > 1 && updates[updates.length - 1].cost >= updates[0].cost,
  updates.length > 1 ? `${updates[0].cost} -> ${updates[updates.length - 1].cost}` : "",
);

// --- summary -----------------------------------------------------------------

console.log(`\n${failures === 0 ? "ALL CHECKS PASSED" : `${failures} CHECK(S) FAILED`}\n`);
process.exit(failures === 0 ? 0 : 1);
