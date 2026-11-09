#!/usr/bin/env node
/**
 * The two Flamescapes backends side by side, as the app runs them
 * (2026-11-09): for each case, the first item a class A design fire (the
 * app's defaults: growth "class", 780 MJ/m2 of contents) or burnt by the
 * solver, three sets of runs, each `--members` runs with the room k mK
 * warmer:
 *
 *   cpu        the CPU on the grid the app gives it here (lib/hardware.js cpuBudget)
 *   gpu-same   the GPU on that same grid: does the CPU reproduce the GPU?
 *   gpu-app    the GPU on the grid the app's worker gives it here (its
 *              memory's budget, capped by the level its measured speed keeps
 *              to 33 ms a step; flamescapes.worker.js gpuCells)
 *
 * Recorded per run: wall time, ms a step and the multiple of real time; the
 * energy released, the peak heat release and the fuel burnt; from the
 * forecast (60 C, visibility 10 m) how many nodes become untenable and when;
 * the mean and hottest node temperature at the end. The cases: the GPU
 * check's 5 x 4 m test room (fire at its middle), and the site's examples
 * (Duplex, the NIBS Office and Clinic, Vienna) from tools/fire-bench.mjs's
 * fire origin. Results go to benchmark/results/<date>-backends.json.
 *
 *   node tools/backend-bench.mjs [--cases room,room-solver,duplex,office,clinic,vienna]
 *        [--members 4] [--horizon 180] [--sets cpu,gpu-same,gpu-app] [--out file] [--headed]
 *
 * Uses playwright-core with the installed Chrome (WebGPU in headless Chrome
 * on this machine) and serves topologicpy-web-frontend/ itself.
 */
import { chromium } from "playwright-core";
import http from "node:http";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extract } from "./ifc-extract.mjs";
import { buildLatticeGraph } from "../topologicpy-web-frontend/src/core/lattice.js";
import { floorsFromSpaces, slabsMissing } from "../topologicpy-web-frontend/src/core/spaceFloors.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const FRONTEND = path.join(ROOT, "topologicpy-web-frontend");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const CHROME = flag("--chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe");
const MEMBERS = Number(flag("--members", 4));
const HORIZON = Number(flag("--horizon", 180));
const CASES = flag("--cases", "room,room-solver,duplex,office,clinic,vienna").split(",");
const SETS = flag("--sets", "cpu,gpu-same,gpu-app").split(",");
const today = new Date().toISOString().slice(0, 10);
const OUT = flag("--out", path.join(ROOT, `benchmark/results/${today}-backends.json`));
const EXAMPLES = { duplex: "duplex-apartment.ifc", office: "nibs-office.ifc", clinic: "nibs-clinic.ifc", vienna: "vienna-escape-route-model.ifc" };
const GROUPS = ["floors", "roofs", "walls", "stairs", "obstacles", "ceilings", "doors"];

/** tools/fire-bench.mjs's fire origin: the lowest floor's node nearest its centre. */
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

/** Geometry and nodes as one binary file and a JSON index: typed arrays by offset, the rest as is. */
function pack(geometry, nodes) {
  const chunks = [];
  let offset = 0;
  const put = (a) => {
    const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    const pad = (4 - (offset % 4)) % 4;
    if (pad) { chunks.push(new Uint8Array(pad)); offset += pad; }
    chunks.push(bytes);
    const at = { type: a.constructor.name, offset, length: a.length };
    offset += bytes.byteLength;
    return at;
  };
  const index = { groups: {}, nodes: put(Float32Array.from(nodes)) };
  for (const group of GROUPS) {
    index.groups[group] = (geometry[group] || []).map((mesh) => {
      const out = {};
      for (const [key, value] of Object.entries(mesh)) out[key] = ArrayBuffer.isView(value) ? { $typed: put(value) } : value;
      return out;
    });
  }
  return { index, bin: Buffer.concat(chunks.map((c) => Buffer.from(c.buffer, c.byteOffset, c.byteLength))) };
}

// ---- the cases' geometry, extracted here and served to the page
const caseFiles = new Map();
const cases = [];
for (const name of CASES) {
  if (name === "room" || name === "room-solver") {
    cases.push({ name, synthetic: true, solver: name === "room-solver" });
    continue;
  }
  const file = EXAMPLES[name];
  if (!file) throw new Error(`Unknown case ${name}`);
  const t0 = performance.now();
  const geo = await extract(path.join(FRONTEND, "public/examples", file), { withDoorFlags: true });
  const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
  const seed = seedNode(built.points);
  const fire = [built.points[seed * 3], built.points[seed * 3 + 1], built.points[seed * 3 + 2]];
  const geometry = slabsMissing(geo.floors, geo.spaces) ? { ...geo, floors: [...geo.floors, ...floorsFromSpaces(geo.spaces)] } : geo;
  const { index, bin } = pack(geometry, built.points);
  caseFiles.set(`/cases/${name}.json`, Buffer.from(JSON.stringify(index)));
  caseFiles.set(`/cases/${name}.bin`, bin);
  cases.push({ name, file, fire, nodes: built.points.length / 3 });
  console.log(`${file}: ${built.points.length / 3} nodes, fire at ${fire.map((x) => x.toFixed(2)).join(", ")}, ${(bin.length / 2 ** 20).toFixed(1)} MB of geometry (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
}

// ---- a static server: the frontend's source files and the cases
const TYPES = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".html": "text/html", ".wasm": "application/wasm" };
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  if (url.pathname === "/" || url.pathname === "/bench.html") {
    res.writeHead(200, { "content-type": "text/html" }).end("<!doctype html><meta charset=utf-8><title>Flamescapes backends</title>");
    return;
  }
  if (caseFiles.has(url.pathname)) {
    res.writeHead(200, { "content-type": url.pathname.endsWith(".json") ? "application/json" : "application/octet-stream" }).end(caseFiles.get(url.pathname));
    return;
  }
  const file = path.resolve(FRONTEND, "." + decodeURIComponent(url.pathname));
  if (!file.startsWith(FRONTEND)) {
    res.writeHead(403).end();
    return;
  }
  try {
    const body = await readFile(file);
    res.writeHead(200, { "content-type": TYPES[path.extname(file)] || "application/octet-stream", "cache-control": "no-store" }).end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const origin = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({ executablePath: CHROME, headless: !args.includes("--headed") });
const page = await browser.newPage();
page.on("console", (msg) => console.log(`  [page] ${msg.text()}`));
page.on("pageerror", (error) => console.log(`  [pageerror] ${error.message}`));
await page.goto(`${origin}/bench.html`);

/** One case in the page: the grids, then each set of runs. */
async function benchCase({ base, info, members, horizon, sets }) {
  const R = await import(base + "core/flamescapesRun.js");
  const F = await import(base + "core/flamescapes.js");
  const G = await import(base + "core/flamescapesGpu.js");
  const FC = await import(base + "core/fireClasses.js");
  const HW = await import(base + "lib/hardware.js");
  const log = (...a) => console.log(...a);

  // The geometry: the GPU check's test room, or an example from the server.
  let geometry, fire, nodes;
  if (info.synthetic) {
    const box = (x0, y0, z0, x1, y1, z1) => ({
      positions: Float32Array.from([x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1]),
      indices: Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3]),
    });
    const W = 5, D = 4, H = 2.7, t = 0.2;
    geometry = {
      floors: [box(-t, -t, -t, W + t, 0, D + t)],
      roofs: [box(-t, H, -t, W + t, H + t, D + t)],
      walls: [box(-t, 0, -t, W + t, H, 0), box(-t, 0, D, W + t, H, D + t), box(-t, 0, 0, 0, H, D), box(W, 0, 0, W + t, H, 1.5), box(W, 0, 2.4, W + t, H, D), box(W, 2.0, 1.5, W + t, H, 2.4)],
      stairs: [], obstacles: [],
    };
    fire = [2.5, 0, 2.0];
    const list = [];
    for (let x = 0.25; x < W; x += 0.5) for (let z = 0.25; z < D; z += 0.5) list.push(x, 0.75, z);
    for (let x = 5.5; x <= 7.0; x += 0.5) for (let z = 1.5; z <= 2.5; z += 0.5) list.push(x, 0.75, z);
    nodes = Float32Array.from(list);
  } else {
    const index = await (await fetch(`/cases/${info.name}.json`)).json();
    const bin = await (await fetch(`/cases/${info.name}.bin`)).arrayBuffer();
    const typed = ({ type, offset, length }) => new globalThis[type](bin, offset, length);
    geometry = {};
    for (const [group, meshes] of Object.entries(index.groups)) {
      geometry[group] = meshes.map((m) => Object.fromEntries(Object.entries(m).map(([k, v]) => [k, v && v.$typed ? typed(v.$typed) : v])));
    }
    nodes = typed(index.nodes);
    fire = info.fire;
  }
  const { contentsFuel, ignitionFuel, item } = FC.fuelsFor("A");
  const designFire = info.solver ? null : FC.designFireFor("A", "class");
  const build = (maxCells) => R.buildScene(geometry, { upAxis: "y", fire, maxCells, contentsFuel, ignitionFuel, item, designFire, fireLoad: 780e6, seed: 1 });

  // The runner, kept across cases.
  if (!window.__runner) {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    window.__runner = adapter ? await G.createGpuRunner(adapter) : null;
    window.__adapter = adapter?.info ? `${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description || ""}`.trim() : null;
  }
  const runner = window.__runner;
  if (!runner) return { error: "no WebGPU" };
  const gpuBackend = await HW.chooseBackend("gpu");
  const cpuCells = HW.cpuBudget(navigator);

  // The worker's GPU grid (flamescapes.worker.js gpuCells): the memory's budget,
  // capped by the level the measured speed keeps to 33 ms a step.
  const LEVELS = [400000, 300000, 220000, 160000, 120000, 90000, 60000];
  const probe = build(gpuBackend.cells);
  const sim = runner.createSim(F.createSolver(probe, { contentsFuel: probe.fuels.contentsFuel, ignitionFuel: probe.fuels.ignitionFuel, embers: false, seed: 1 }), {});
  for (let s = 0; s < 3; s += 1) sim.step(0.1);
  await sim.finish();
  const t0 = performance.now();
  for (let s = 0; s < 20; s += 1) sim.step(0.1);
  await sim.finish();
  const msProbe = (performance.now() - t0) / 20;
  sim.destroy();
  const rate = probe.grid.cells / msProbe;
  const appCells = Math.min(gpuBackend.cells, LEVELS.find((n) => n <= rate * 33) ?? LEVELS.at(-1));
  log(`${info.name}: CPU budget ${cpuCells} cells; GPU memory budget ${gpuBackend.cells}, ${rate.toFixed(0)} cells a ms on ${probe.grid.cells} -> ${appCells}`);

  const out = { name: info.name, solver: Boolean(info.solver), adapter: window.__adapter, cpuCells, gpuMemoryCells: gpuBackend.cells, gpuRate: rate, gpuAppCells: appCells, sets: {} };
  for (const [label, backend, maxCells] of [["cpu", "cpu", cpuCells], ["gpu-same", "gpu", cpuCells], ["gpu-app", "gpu", appCells]].filter(([label]) => sets.includes(label))) {
    const scene = build(maxCells);
    if (!scene.ignition) return { ...out, error: "no place for the first item at the fire origin" };
    const g = scene.grid;
    const set = { grid: [g.nx, g.ny, g.nz], cells: g.cells, cell: g.h, clipped: Boolean(scene.clipped), runs: [] };
    for (let k = 0; k < members; k += 1) {
      const opts = { nodes, upAxis: "y", horizon, frameSeconds: 5, particles: 3000, solverOptions: { ambient: 293.15 + k * 1e-3 } };
      const t1 = performance.now();
      const frames = backend === "cpu" ? R.runScene(scene, opts) : await runner.runScene(scene, opts);
      const wall = performance.now() - t1;
      const forecast = R.forecastFromFrames(frames, { threshold: 60, smoke: true, visibilityLimit: 10 });
      const n = frames.n, K = frames.K;
      const times = [];
      for (let v = 0; v < n; v += 1) if (forecast.untenable[v] < horizon) times.push(forecast.untenable[v]);
      times.sort((a, b) => a - b);
      let meanT = 0, maxT = 0;
      for (let v = 0; v < n; v += 1) {
        const T = frames.T[(K - 1) * n + v];
        meanT += T / n;
        if (T > maxT) maxT = T;
      }
      set.runs.push({
        wallMs: wall, msPerStep: wall / (horizon / 0.1), realTime: (horizon * 1000) / wall,
        releasedMJ: frames.stats.released / 1e6, burntKg: frames.stats.burnt, peakKw: Math.max(...frames.hrr),
        untenable: times.length, firstUntenable: times[0] ?? null, medianUntenable: times.length ? times[times.length >> 1] : null,
        meanNodeT: meanT, maxNodeT: maxT, hrr: Array.from(frames.hrr),
      });
      log(`  ${info.name} ${label} ${k}: ${(wall / 1000).toFixed(1)} s, ${set.runs.at(-1).releasedMJ.toFixed(1)} MJ, ${times.length} untenable`);
    }
    out.sets[label] = set;
  }
  return out;
}

const report = { generated: new Date().toISOString(), tool: "tools/backend-bench.mjs", members: MEMBERS, horizon: HORIZON, cases: [] };
const t0 = Date.now();
for (const info of cases) {
  console.log(`\n== ${info.name}`);
  const res = await page.evaluate(benchCase, { base: `${origin}/src/`, info, members: MEMBERS, horizon: HORIZON, sets: SETS });
  if (res.error) console.log(`  ${res.error}`);
  report.cases.push(res);
  report.adapter ??= res.adapter;
}
report.wallSeconds = (Date.now() - t0) / 1000;
await browser.close();
server.close();
await mkdir(path.dirname(OUT), { recursive: true });
await writeFile(OUT, JSON.stringify(report, null, 1));

// ---- the report
const f = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : "-");
const med = (v) => {
  const s = v.filter(Number.isFinite).sort((a, b) => a - b);
  return s.length ? s[s.length >> 1] : NaN;
};
const span = (runs, key, d = 1) => {
  const v = runs.map((r) => r[key]).filter(Number.isFinite);
  return v.length ? `${f(med(v), d)} (${f(Math.min(...v), d)}-${f(Math.max(...v), d)})` : "-";
};
console.log(`\nFlamescapes backends, ${MEMBERS} runs each, ${HORIZON} s, adapter ${report.adapter}; medians (range)`);
for (const c of report.cases) {
  if (c.error) {
    console.log(`\n${c.name}: ${c.error}`);
    continue;
  }
  console.log(`\n${c.name}${c.solver ? " (the item burnt by the solver)" : " (class A design fire)"}: GPU ${f(c.gpuRate, 0)} cells a ms`);
  console.log("  set       | grid                 | wall s        | x real time | released MJ        | peak kW            | untenable nodes | first untenable s | mean node T end");
  for (const [label, s] of Object.entries(c.sets)) {
    console.log(`  ${label.padEnd(9)} | ${s.grid.join("x")} ${f(s.cell, 3)} m`.padEnd(34) + ` | ${span(s.runs.map((r) => ({ ...r, wallS: r.wallMs / 1000 })), "wallS")} | ${span(s.runs, "realTime", 2)} | ${span(s.runs, "releasedMJ")} | ${span(s.runs, "peakKw", 0)} | ${span(s.runs, "untenable", 0)} | ${span(s.runs, "firstUntenable", 0)} | ${span(s.runs, "meanNodeT")}`);
  }
}
console.log(`\nWrote ${path.relative(process.cwd(), OUT)} (${f(report.wallSeconds / 60)} min)`);
