/**
 * Checks the WebGPU Flamescapes solver (src/core/flamescapesGpu.js) against
 * the CPU solver (src/core/flamescapes.js) in real Chrome, on the synthetic
 * room of the solver's smoke test: a 5 x 4 x 2.7 m room with a 0.9 x 2.0 m
 * door, contents on the floor, a 30 kW burner under a 2 kg first item.
 *
 *   1. Agreement: radiation and particles off on both paths (solverOptions
 *      { emitAbove: 1e9, maxParticles: 0 }: no cell is hot enough to cast a
 *      ray and no particle can be spawned; `rays: 0` would not do, as the CPU
 *      still takes the emitted heat out), 200 steps from the same state; max,
 *      99th percentile and median differences of gas temperature, oxygen and
 *      soot, beside the same differences between the CPU and a CPU twin
 *      started one ulp warmer (what round-off alone grows to). Then again
 *      with flame-front particles on: they draw no random numbers, so they
 *      too should agree, count for count.
 *   1b. Radiation: one step from a developed CPU state on the GPU and on two
 *      CPU copies with different seeds; the GPU should differ from the CPU no
 *      more than the CPU differs from itself.
 *   2. The full model (radiation, flame-front particles) through runScene on
 *      both paths: heat release and node readings compared statistically,
 *      then with embers on.
 *   1c. One step from identical states: the CPU's own state of the app's
 *      scene (buildScene's first item, a fuel-rich plume) at several times,
 *      uploaded to the GPU, one step on each; differences beside the CPU's
 *      one-ulp twin. The sharpest test that the port matches the CPU.
 *   1d. Ensembles: the same fire with tiny perturbations on each path; the
 *      two distributions of heat released should match.
 *   2b. The app's worker (src/workers/flamescapes.worker.js) on backend
 *      "auto" (the GPU here) and "cpu": the same result shape.
 *   3. Milliseconds per step, CPU and GPU, at about 30k and 200k cells.
 *
 * Uses playwright-core with the installed Chrome (no browser download) and
 * serves topologicpy-web-frontend/ itself, so the dev server is not needed.
 * On this machine (Chrome 156, Intel UHD 770, D3D12) headless Chrome exposes
 * the GPU's WebGPU adapter with no extra flags; --swiftshader forces the
 * software adapter (--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader).
 *
 *   node tools/flamescapes-gpu-check.mjs [--headed] [--swiftshader] [--quick]
 *        [--phase agreement|onestep|ensemble|radiation|full|worker|timing] [--horizon 120]
 *        [--json out.json] [--url http://localhost:5173] (a running Vite server)
 */
import { chromium } from "playwright-core";
import http from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.resolve(HERE, "../topologicpy-web-frontend");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const CHROME = flag("--chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe");
const HEADED = args.includes("--headed");
const SWIFTSHADER = args.includes("--swiftshader");
const QUICK = args.includes("--quick");
const PHASE = flag("--phase", "all");
// --class B runs the checks with that fire class's fuels (fireClasses.js); default: a class A fuel.
const FIRE_CLASS = flag("--class", null);
// --ducts: the app scene with two ceiling air terminals and the ventilation running.
const DUCTS = args.includes("--ducts");
const HORIZON = Number(flag("--horizon", QUICK ? 60 : 120));

// ---- a static server for the frontend's source files
const TYPES = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".html": "text/html", ".wasm": "application/wasm" };
let server = null;
let origin = flag("--url", null);
if (!origin) {
  server = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    if (url.pathname === "/favicon.ico") {
      res.writeHead(204).end();
      return;
    }
    if (url.pathname === "/" || url.pathname === "/check.html") {
      res.writeHead(200, { "content-type": "text/html" });
      res.end("<!doctype html><meta charset=utf-8><title>Flamescapes GPU check</title>");
      return;
    }
    const file = path.resolve(FRONTEND, "." + decodeURIComponent(url.pathname));
    if (!file.startsWith(FRONTEND)) {
      res.writeHead(403).end();
      return;
    }
    try {
      const body = await readFile(file);
      res.writeHead(200, { "content-type": TYPES[path.extname(file)] || "application/octet-stream", "cache-control": "no-store" });
      res.end(body);
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
}

const launchArgs = SWIFTSHADER ? ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader"] : [];
const browser = await chromium.launch({ executablePath: CHROME, headless: !HEADED, args: launchArgs });
const page = await browser.newPage();
page.on("console", (msg) => {
  if (msg.type() === "log" || msg.type() === "info") console.log(`  [page] ${msg.text()}`);
  else if (msg.type() === "error" || msg.type() === "warning") console.log(`  [page ${msg.type()}] ${msg.text()}`);
});
page.on("pageerror", (error) => console.log(`  [pageerror] ${error.message}`));
// A page on the server's origin (a secure context, so navigator.gpu exists).
await page.goto(server ? `${origin}/check.html` : `${origin}/src/core/flamescapesGpu.js`);

/**
 * Everything below runs in the page. It is serialised by Playwright, so it
 * holds its own helpers.
 */
async function suite({ base, phase, horizon, quick, fireClass, ducts }) {
  const V = await import(base + "voxelize.js");
  const F = await import(base + "flamescapes.js");
  const R = await import(base + "flamescapesRun.js");
  const G = await import(base + "flamescapesGpu.js");
  const log = (...a) => console.log(...a);

  // ---- the room (tools' fs-room smoke test)
  function box(x0, y0, z0, x1, y1, z1, material) {
    const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
    const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
    return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material };
  }
  const S = V.MATERIAL.STONE;
  const W = 5, D = 4, H = 2.7, t = 0.2;
  const meshes = [
    box(-t, -t, -t, W + t, 0, D + t, S),
    box(-t, H, -t, W + t, H + t, D + t, S),
    box(-t, 0, -t, W + t, H, 0, S),
    box(-t, 0, D, W + t, H, D + t, S),
    box(-t, 0, 0, 0, H, D, S),
    box(W, 0, 0, W + t, H, 1.5, S),
    box(W, 0, 2.4, W + t, H, D, S),
    box(W, 2.0, 1.5, W + t, H, 2.4, S),
  ];
  const classA = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: true, ignitionTemperature: 300 };
  const chosen = fireClass ? (await import(base + "fireClasses.js")).fuelsFor(fireClass) : null;
  const contentsFuel = chosen?.contentsFuel ?? classA;
  const ignitionFuel = chosen?.ignitionFuel ?? classA;
  function room(cell, maxCells) {
    const grid = V.gridFor(meshes, { upAxis: "y", cell, margin: 3, headroom: 3, maxCells });
    const fraction = V.volumeFractions(grid, meshes, { rays: V.raysFor(grid) });
    const solid = V.occupancy(grid, fraction);
    const cells = V.interiorCells(grid, fraction, { ceiling: 5, layers: 2 });
    const fuel = V.interiorFuel(grid, cells, { fireLoad: 780e6, heatOfCombustion: 17.5e6, spread: 0.3, layers: 2, seed: 3 });
    const at = (a, b, c) => Math.floor((a - grid.origin[0]) / grid.h) + grid.nx * (Math.floor((b - grid.origin[1]) / grid.h) + grid.ny * Math.floor((c - grid.origin[2]) / grid.h));
    const ign = at(2.5, 0.1, 2.0);
    return {
      grid, fraction, solid,
      contents: { cells, masses: fuel.masses },
      ignition: { cell: ign, mass: 2 },
      pilot: { cells: [ign], power: 30000, duration: 60 },
      fuels: { contentsFuel, ignitionFuel },
    };
  }
  const solverOf = (scene, options) => F.createSolver(scene, { contentsFuel, ignitionFuel, ...options });
  // Graph nodes 0.75 m above the floor: a 0.5 m lattice in the room and a few outside the door.
  const nodes = [];
  for (let x = 0.25; x < W; x += 0.5) for (let z = 0.25; z < D; z += 0.5) nodes.push(x, 0.75, z);
  for (let x = 5.5; x <= 7.0; x += 0.5) for (let z = 1.5; z <= 2.5; z += 0.5) nodes.push(x, 0.75, z);
  const nodeArray = Float32Array.from(nodes);

  // ---- the runner, kept across phases
  if (!window.__fsRunner) {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return { error: "No WebGPU adapter in this browser." };
    const info = adapter.info || {};
    const t0 = performance.now();
    const runner = await G.createGpuRunner(adapter);
    window.__fsCompileMs = performance.now() - t0;
    if (!runner) return { error: "createGpuRunner returned null (no device, a limit missing, or the kernels failed to compile)." };
    window.__fsRunner = runner;
    window.__fsAdapter = { vendor: info.vendor, architecture: info.architecture, description: info.description, fallback: Boolean(adapter.isFallbackAdapter || info.isFallbackAdapter) };
  }
  const runner = window.__fsRunner;
  const out = { adapter: window.__fsAdapter, compileMs: window.__fsCompileMs };

  const absDiffs = (a, b, pick) => {
    const d = [];
    for (let c = 0; c < a.length; c += 1) if (pick(c)) d.push(Math.abs(a[c] - b[c]));
    d.sort((x, y) => x - y);
    return { max: d.length ? d[d.length - 1] : 0, median: d.length ? d[d.length >> 1] : 0, p99: d.length ? d[Math.floor(d.length * 0.99)] : 0, n: d.length };
  };
  const species = (Y, s) => {
    const n = Y.length / 6;
    const o = new Float32Array(n);
    for (let c = 0; c < n; c += 1) o[c] = Y[c * 6 + s];
    return o;
  };
  const compare = (A, B) => {
    const open = (c) => A.kind[c] !== 2;
    const solidish = (c) => A.kind[c] !== 0;
    return {
      T: absDiffs(A.T, B.T, open),
      O2: absDiffs(species(A.Y, 1), species(B.Y, 1), open),
      soot: absDiffs(species(A.Y, 4), species(B.Y, 4), open),
      Ts: absDiffs(A.Ts, B.Ts, solidish),
      rho: absDiffs(A.rho, B.rho, open),
      v: absDiffs(A.v, B.v, () => true),
      kindMismatch: A.kind.reduce((s, k, c) => s + (k !== B.kind[c] ? 1 : 0), 0),
    };
  };
  const range = (sim) => {
    let Tmax = 0, sootMax = 0, o2Min = 1;
    for (let c = 0; c < sim.N; c += 1) {
      if (sim.kind[c] === 2) continue;
      Tmax = Math.max(Tmax, sim.T[c]);
      sootMax = Math.max(sootMax, sim.Y[c * 6 + 4]);
      o2Min = Math.min(o2Min, sim.Y[c * 6 + 1]);
    }
    return { TmaxC: Tmax - 273.15, sootMax, o2Min };
  };

  /** A deep copy of a CPU solver state with its own random stream. */
  const cloneSim = (sim, seed) => {
    const c = { ...sim, topo: null, random: V.rng(seed), stats: { ...sim.stats }, parts: { ...sim.parts } };
    for (const key of Object.keys(sim)) if (ArrayBuffer.isView(sim[key])) c[key] = sim[key].slice();
    for (const key of Object.keys(sim.parts)) if (ArrayBuffer.isView(sim.parts[key])) c.parts[key] = sim.parts[key].slice();
    return c;
  };

  // ---- 1. agreement: deterministic stages, 200 steps from the same state
  if (phase === "all" || phase === "agreement") {
    const scene = room(0.25, 200000);
    out.agreement = { grid: [scene.grid.nx, scene.grid.ny, scene.grid.nz], cells: scene.grid.cells, h: scene.grid.h, runs: [] };
    // Radiation off (no cell reaches emitAbove) and particles off (no room for one); then
    // particles on: flame-front particles draw no random numbers, so they too should agree.
    for (const [label, extra] of [["radiation and particles off", { maxParticles: 0 }], ["flame-front particles on, radiation off", {}]]) {
      const opts = { emitAbove: 1e9, embers: false, dt: 0.1, ...extra };
      const cpu = solverOf(scene, opts);
      const twin = solverOf(scene, opts);
      // One ulp up on every open cell's temperature: what float round-off alone becomes.
      for (let c = 0; c < twin.N; c += 1) if (twin.kind[c] !== 2) twin.T[c] = Math.fround(twin.T[c] * (1 + 2 ** -23));
      const gpu = runner.createSim(solverOf(scene, opts), {});
      const checkpoints = new Set([1, 10, 50, 100, 200]);
      const rows = [];
      for (let s = 1; s <= 200; s += 1) {
        F.step(cpu, 0.1);
        F.step(twin, 0.1);
        gpu.step(0.1);
        if (!checkpoints.has(s)) continue;
        const g = await gpu.readFields();
        rows.push({ step: s, ...range(cpu), particles: { cpu: cpu.parts.n, gpu: g.particles }, sweeps: { cpu: cpu.stats.sweeps, gpu: g.sweeps }, gpu: compare(cpu, g), twin: compare(cpu, twin) });
        log(`agreement (${label}) step ${s}: max |dT| GPU ${rows.at(-1).gpu.T.max.toExponential(2)} K, twin ${rows.at(-1).twin.T.max.toExponential(2)} K; particles CPU ${cpu.parts.n} GPU ${g.particles}`);
      }
      gpu.destroy();
      out.agreement.runs.push({ label, rows });
    }
  }

  // ---- 1b. radiation: one step from a developed fire, against a CPU with another random seed
  if (phase === "all" || phase === "radiation") {
    out.radiation = [];
    // The default model at 90 s, and a radiation-heavy variant (everything above 330 K casts rays).
    for (const [label, options, until] of [["default", {}, quick ? 60 : 90], ["emitAbove 330 K", { emitAbove: 330 }, 60]]) {
    const scene = room(0.25, 200000);
    const base = solverOf(scene, { dt: 0.1, ...options });
    while (base.time < until - 1e-9) F.step(base, 0.1);
    const a = cloneSim(base, 11);
    const b = cloneSim(base, 12);
    const gpu = runner.createSim(base, {});
    const r0 = base.stats.radiated, q0 = base.stats.released;
    F.step(a, 0.1);
    F.step(b, 0.1);
    gpu.step(0.1);
    const frame = await gpu.readFrame({ particles: false });
    const g = await gpu.readFields();
    gpu.destroy();
    let hot = 0;
    for (let c = 0; c < base.N; c += 1) if ((base.kind[c] === 2 ? base.Ts[c] : Math.max(base.T[c], base.Ts[c])) > base.P.emitAbove) hot += 1;
    const meanAbs = (A, Bf, pick) => {
      let s = 0, n = 0;
      for (let c = 0; c < A.length; c += 1) if (pick(c)) { s += Math.abs(A[c] - Bf[c]); n += 1; }
      return s / Math.max(1, n);
    };
    const open = (c) => a.kind[c] !== 2;
    const solidish = (c) => a.kind[c] !== 0;
    out.radiation.push({
      label, time: base.time, hotCells: hot, maxTC: range(base).TmaxC,
      radiatedJ: { cpu: a.stats.radiated - r0, cpuOtherSeed: b.stats.radiated - r0, gpu: frame.radiated },
      releasedJ: { cpu: a.stats.released - q0, cpuOtherSeed: b.stats.released - q0, gpu: frame.released },
      particles: { cpu: a.parts.n, cpuOtherSeed: b.parts.n, gpu: g.particles },
      gasT: { gpuMeanAbs: meanAbs(a.T, g.T, open), seedMeanAbs: meanAbs(a.T, b.T, open), gpuMax: compare(a, g).T.max, seedMax: compare(a, b).T.max },
      solidTs: { gpuMeanAbs: meanAbs(a.Ts, g.Ts, solidish), seedMeanAbs: meanAbs(a.Ts, b.Ts, solidish), gpuMax: compare(a, g).Ts.max, seedMax: compare(a, b).Ts.max },
    });
    log(`radiation (${label}) step at ${base.time.toFixed(1)} s: radiated CPU ${(a.stats.radiated - r0).toFixed(0)} J, GPU ${frame.radiated.toFixed(0)} J`);
    }
  }

  // ---- 2. the full model through runScene on both
  if (phase === "all" || phase === "full") {
    const scene = room(0.25, 200000);
    const run = async (fn, embers) => {
      const hrr = [];
      const t0 = performance.now();
      const frames = await fn(scene, { nodes: nodeArray, upAxis: "y", horizon, frameSeconds: 5, embers, particles: 3000, onFrame: (f) => hrr.push({ time: f.time, hrr: f.hrr, maxT: f.maxT, parts: f.particles?.temps.length ?? 0 }) });
      const ms = performance.now() - t0;
      const forecast = R.forecastFromFrames(frames, { threshold: 60, smoke: true, visibilityLimit: 10 });
      return { frames, forecast, ms, perFrame: hrr };
    };
    const cpu = await run(R.runScene, false);
    log(`full model CPU done in ${(cpu.ms / 1000).toFixed(1)} s`);
    const gpu = await run(runner.runScene, false);
    log(`full model GPU done in ${(gpu.ms / 1000).toFixed(1)} s`);
    const { n, K } = cpu.frames;
    // Node readings: per frame, the mean and max over nodes; overall RMS and correlation.
    const series = [];
    let se = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, m = 0, smokyAgree = 0;
    for (let k = 0; k < K; k += 1) {
      let mc = 0, mg = 0, xc = -Infinity, xg = -Infinity, vc = 0, vg = 0;
      for (let v = 0; v < n; v += 1) {
        const a = cpu.frames.T[k * n + v], b = gpu.frames.T[k * n + v];
        mc += a; mg += b; xc = Math.max(xc, a); xg = Math.max(xg, b);
        se += (a - b) ** 2; sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; m += 1;
        const sc = cpu.frames.vis[k * n + v] < 10, sg = gpu.frames.vis[k * n + v] < 10;
        vc += sc; vg += sg; smokyAgree += sc === sg;
      }
      series.push({ time: (k + 1) * 5, hrrCpu: cpu.frames.hrr[k], hrrGpu: gpu.frames.hrr[k], meanTCpu: mc / n, meanTGpu: mg / n, maxTCpu: xc, maxTGpu: xg, smokyCpu: vc, smokyGpu: vg, partsCpu: cpu.perFrame[k]?.parts, partsGpu: gpu.perFrame[k]?.parts });
    }
    const corr = (m * sxy - sx * sy) / Math.sqrt((m * sxx - sx * sx) * (m * syy - sy * sy));
    // The extra readings (2026-11-06): soot (mg/m^3), CO2 and O2 (%), the same statistics.
    const agree = (field) => {
      const A = cpu.frames[field], B = gpu.frames[field];
      if (!A || !B) return null;
      let e2 = 0, a1 = 0, b1 = 0, aa = 0, bb = 0, ab = 0;
      for (let i = 0; i < A.length; i += 1) {
        const a = A[i], b = B[i];
        e2 += (a - b) ** 2; a1 += a; b1 += b; aa += a * a; bb += b * b; ab += a * b;
      }
      const N = A.length;
      const den = Math.sqrt((N * aa - a1 * a1) * (N * bb - b1 * b1));
      return { rms: Math.sqrt(e2 / N), correlation: den > 0 ? (N * ab - a1 * b1) / den : 1, meanCpu: a1 / N, meanGpu: b1 / N };
    };
    const extras = { soot: agree("soot"), co2: agree("co2"), o2: agree("o2"), co: agree("co") };
    for (const [k, x] of Object.entries(extras)) if (x) log(`node ${k}: CPU mean ${x.meanCpu.toFixed(3)}, GPU mean ${x.meanGpu.toFixed(3)}, RMS ${x.rms.toFixed(3)}, correlation ${x.correlation.toFixed(3)}`);
    const untenable = [];
    for (let v = 0; v < n; v += 1) untenable.push(Math.abs(Math.min(cpu.forecast.untenable[v], horizon) - Math.min(gpu.forecast.untenable[v], horizon)));
    untenable.sort((a, b) => a - b);
    const closes = (f) => f.untenable.reduce((s, x) => s + (x < horizon ? 1 : 0), 0);
    out.full = {
      cells: scene.grid.cells, nodes: n, frames: K, horizon,
      released: { cpu: cpu.frames.stats.released, gpu: gpu.frames.stats.released },
      burnt: { cpu: cpu.frames.stats.burnt, gpu: gpu.frames.stats.burnt },
      radiated: { cpu: cpu.frames.stats.radiated, gpu: gpu.frames.stats.radiated },
      peakHrr: { cpu: Math.max(...cpu.frames.hrr), gpu: Math.max(...gpu.frames.hrr) },
      nodeT: { rms: Math.sqrt(se / m), correlation: corr },
      extras,
      smokyAgreement: smokyAgree / m,
      untenable: { cpu: closes(cpu.forecast), gpu: closes(gpu.forecast), medianAbsDiff: untenable[untenable.length >> 1], maxAbsDiff: untenable.at(-1) },
      ms: { cpu: cpu.ms, gpu: gpu.ms },
      series,
      inGrid: { cpu: cpu.frames.inGrid, gpu: gpu.frames.inGrid },
    };
    // Embers on, GPU and CPU: the ember path runs and spawns, deposits and spot-ignites.
    if (!quick) {
      const ec = await run(R.runScene, true);
      const eg = await run(runner.runScene, true);
      out.embers = {
        cpu: { spawned: ec.frames.stats.embersSpawned, spots: ec.frames.stats.spotIgnitions, released: ec.frames.stats.released, peakHrr: Math.max(...ec.frames.hrr) },
        gpu: { spawned: eg.frames.stats.embersSpawned, spots: eg.frames.stats.spotIgnitions, released: eg.frames.stats.released, peakHrr: Math.max(...eg.frames.hrr) },
      };
    }
  }

  // The room as the app builds it (buildScene: a 5 kg first item held at 900 K
  // for 30 s), whose fuel-rich, oxygen-starved plume is the most sensitive case.
  const appScene = () => R.buildScene({ floors: [meshes[0]], roofs: [meshes[1]], walls: meshes.slice(2) }, { upAxis: "y", fire: [2.5, 0, 2.0], maxCells: 400000, contentsFuel, ignitionFuel, ventilation: ducts ? { terminals: [[1, 2.3, 1], [4, 2.3, 3]], flow: 0.1 } : null });
  const bare = { maxParticles: 0, emitAbove: 1e9 };

  // ---- 1c. one step from identical states: the CPU's state at several times, uploaded
  if (phase === "all" || phase === "onestep") {
    const scene = appScene();
    const opts = { dt: 0.1, ...bare };
    const sim0 = solverOf(scene, opts);
    const rows = [];
    let s = 0;
    for (const target of quick ? [100, 400] : [100, 300, 450, 500, 550, 590]) {
      while (s < target) { F.step(sim0, 0.1); s += 1; }
      const a = cloneSim(sim0, 1);
      const tw = cloneSim(sim0, 1);
      for (let c = 0; c < tw.N; c += 1) if (tw.kind[c] !== 2) tw.T[c] = Math.fround(tw.T[c] * (1 + 2 ** -23));
      const gpu = runner.createSim(sim0, {});
      const q0 = sim0.stats.released;
      F.step(a, 0.1);
      F.step(tw, 0.1);
      gpu.step(0.1);
      const fr = await gpu.readFrame({ particles: false });
      const g = await gpu.readFields();
      gpu.destroy();
      const rel = (A, B, pick) => {
        let worst = 0;
        for (let c = 0; c < a.N; c += 1) if (pick(c) && Number.isFinite(A[c])) worst = Math.max(worst, Math.abs(A[c] - B[c]) / Math.max(Math.abs(A[c]), 1e-30));
        return worst;
      };
      const open = (c) => a.kind[c] !== 2;
      const solidish = (c) => a.kind[c] !== 0;
      rows.push({
        step: target, releasedJ: { cpu: a.stats.released - q0, gpu: fr.released }, sweeps: { cpu: a.stats.sweeps, gpu: fr.sweeps },
        gpu: { T: compare(a, g).T.max, Trel: rel(a.T, g.T, open), Ts: compare(a, g).Ts.max, Tsrel: rel(a.Ts, g.Ts, solidish), rho: rel(a.rho, g.rho, open), O2: compare(a, g).O2.max, soot: compare(a, g).soot.max },
        twin: { T: compare(a, tw).T.max, Trel: rel(a.T, tw.T, open) },
      });
    }
    out.onestep = { cells: scene.grid.cells, rows, ducts: scene.ducts?.cells.length ?? 0 };
  }

  // ---- 1d. ensembles: eight tiny perturbations of the same fire on each path
  if (phase === "all" || phase === "ensemble") {
    const scene = appScene();
    const members = quick ? 4 : 8;
    const runs = { cpu: [], gpu: [] };
    for (let k = 0; k < members; k += 1) {
      const o = { nodes: nodeArray, horizon: 60, frameSeconds: 5, particles: 0, keep: false, solverOptions: { ...bare, ambient: 293.15 + k * 1e-3 } };
      const c = R.runScene(scene, o);
      const g = await runner.runScene(scene, o);
      runs.cpu.push({ released: c.stats.released, peak: Math.max(...c.hrr) });
      runs.gpu.push({ released: g.stats.released, peak: Math.max(...g.hrr) });
    }
    out.ensemble = { cells: scene.grid.cells, members, runs };
  }

  // ---- 2b. the app's worker (src/workers/flamescapes.worker.js), as App.jsx starts it
  if (phase === "all" || phase === "worker") {
    const job = (backend) => new Promise((resolve) => {
      const worker = new Worker(base.replace(/core\/$/, "workers/") + "flamescapes.worker.js", { type: "module" });
      let frames = 0;
      let info = null;
      const t0 = performance.now();
      worker.onerror = (event) => resolve({ backend, error: event.message || "worker error" });
      worker.onmessage = ({ data }) => {
        if (data.progress?.type === "scene") info = data.progress.info;
        else if (data.progress?.type === "frame") frames += 1;
        if ("ok" in data) {
          worker.terminate();
          const r = data.result;
          resolve(data.ok
            ? { backend: info?.backend, grid: info?.grid, frames, keys: Object.keys(r).sort().join(","), statKeys: Object.keys(r.stats).join(","), released: r.stats.released, peakHrr: Math.max(...r.hrr), inGrid: r.inGrid, n: r.n, K: r.K, ms: performance.now() - t0 }
            : { backend, error: data.error });
        }
      };
      worker.postMessage({ id: 1, payload: {
        backend, geometry: { floors: [meshes[0]], roofs: [meshes[1]], walls: meshes.slice(2) }, upAxis: "y", fire: [2.5, 0, 2.0],
        contentsFuel, ignitionFuel, nodes: nodeArray, agentHeight: 0.75, headHeight: 1.8,
        horizon: quick ? 30 : 60, frameSeconds: 5, embers: false, particles: 500,
      } });
    });
    out.worker = { gpu: await job("auto"), cpu: await job("cpu") };
    log(`worker: auto ran on ${out.worker.gpu.backend?.kind}, cpu on ${out.worker.cpu.backend?.kind}`);
  }

  // ---- 3. ms per step
  if (phase === "all" || phase === "timing") {
    out.timing = [];
    for (const [label, cell, cap] of [["30k", 0.15, 30000], ["200k", 0.05, 200000]]) {
      const scene = room(cell, cap);
      const opts = { dt: 0.1 };
      // CPU: the first steps of the fire (burner on), full model.
      const cpu = solverOf(scene, opts);
      const cpuSteps = label === "30k" ? (quick ? 10 : 40) : (quick ? 3 : 8);
      F.step(cpu, 0.1);
      let t0 = performance.now();
      for (let s = 0; s < cpuSteps; s += 1) F.step(cpu, 0.1);
      const cpuMs = (performance.now() - t0) / cpuSteps;
      // GPU: the same start, then the same steps; queue them all and wait.
      const gpu = runner.createSim(solverOf(scene, opts), {});
      for (let s = 0; s < 5; s += 1) gpu.step(0.1);
      await gpu.finish();
      const gpuSteps = quick ? 50 : 200;
      t0 = performance.now();
      let encode = 0;
      for (let s = 0; s < gpuSteps; s += 1) {
        const e0 = performance.now();
        gpu.step(0.1);
        encode += performance.now() - e0;
      }
      await gpu.finish();
      const gpuMs = (performance.now() - t0) / gpuSteps;
      // And the GPU with radiation and particles off (the grid stages alone).
      const bare = runner.createSim(solverOf(scene, { ...opts, emitAbove: 1e9, maxParticles: 0 }), {});
      for (let s = 0; s < 5; s += 1) bare.step(0.1);
      await bare.finish();
      t0 = performance.now();
      for (let s = 0; s < gpuSteps; s += 1) bare.step(0.1);
      await bare.finish();
      const bareMs = (performance.now() - t0) / gpuSteps;
      const fields = await gpu.readFields();
      gpu.destroy();
      bare.destroy();
      out.timing.push({ label, grid: [scene.grid.nx, scene.grid.ny, scene.grid.nz], cells: scene.grid.cells, h: scene.grid.h, cpuMs, cpuSteps, gpuMs, gpuSteps, encodeMsPerStep: encode / gpuSteps, gpuGridOnlyMs: bareMs, gpuParticlesAtEnd: fields.particles });
      log(`timing ${label}: ${scene.grid.cells} cells, CPU ${cpuMs.toFixed(1)} ms/step, GPU ${gpuMs.toFixed(2)} ms/step`);
    }
  }
  return out;
}

const phases = PHASE === "all" ? ["agreement", "onestep", "ensemble", "radiation", "full", "worker", "timing"] : [PHASE];
const report = {};
const t0 = Date.now();
for (const phase of phases) {
  console.log(`\n== ${phase}`);
  const res = await page.evaluate(suite, { base: `${origin}/src/core/`, phase, horizon: HORIZON, quick: QUICK, fireClass: FIRE_CLASS, ducts: DUCTS });
  if (res.error) {
    console.log(`  ${res.error}`);
    report.error = res.error;
    break;
  }
  Object.assign(report, res);
}
await browser.close();
server?.close();

// ---- the report
const f = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : String(x));
const e = (x) => (Number.isFinite(x) ? x.toExponential(2) : String(x));
console.log(`\nChrome ${CHROME}${HEADED ? " (headed)" : " (headless)"}${launchArgs.length ? ` ${launchArgs.join(" ")}` : ", no extra flags"}`);
if (report.adapter) console.log(`Adapter: ${[report.adapter.vendor, report.adapter.architecture, report.adapter.description].filter(Boolean).join(" ")}${report.adapter.fallback ? " (fallback)" : ""}; kernels compiled in ${f(report.compileMs, 0)} ms`);
if (report.agreement) {
  const a = report.agreement;
  console.log(`\nAgreement over 200 steps of 0.1 s, ${a.grid.join("x")} = ${a.cells} cells of ${f(a.h, 3)} m. |CPU - GPU| over open cells (solid cells for Ts),`);
  console.log("beside |CPU - CPU twin| for a twin started one ulp warmer (round-off grown by the flow).");
  for (const run of a.runs) {
    console.log(`\n  ${run.label}`);
    console.log("  step | gas T (K) max / p99 / median     | O2 max / p99 / median          | soot max / p99 / median         | Ts max   | twin: T max  O2 max   soot max  Ts max   | particles CPU/GPU | sweeps | hottest gas C");
    for (const r of run.rows) {
      const G = r.gpu, Tw = r.twin;
      console.log(`  ${String(r.step).padStart(4)} | ${e(G.T.max)} / ${e(G.T.p99)} / ${e(G.T.median)} | ${e(G.O2.max)} / ${e(G.O2.p99)} / ${e(G.O2.median)} | ${e(G.soot.max)} / ${e(G.soot.p99)} / ${e(G.soot.median)} | ${e(G.Ts.max)} | ${e(Tw.T.max)} ${e(Tw.O2.max)} ${e(Tw.soot.max)} ${e(Tw.Ts.max)} | ${r.particles.cpu}/${r.particles.gpu} | ${r.sweeps.cpu}/${r.sweeps.gpu} | ${f(r.TmaxC, 0)}${G.kindMismatch ? `  (${G.kindMismatch} cells differ in kind)` : ""}`);
    }
  }
}
for (const r of report.radiation || []) {
  console.log(`\nRadiation (${r.label}), one step from the CPU state at ${f(r.time, 1)} s (${r.hotCells} cells above emitAbove, hottest gas ${f(r.maxTC, 0)} C); CPU seed 11 vs GPU, and CPU seed 11 vs CPU seed 12`);
  console.log(`  radiated J   CPU ${f(r.radiatedJ.cpu, 0)}  CPU other seed ${f(r.radiatedJ.cpuOtherSeed, 0)}  GPU ${f(r.radiatedJ.gpu, 0)}`);
  console.log(`  released J   CPU ${f(r.releasedJ.cpu, 0)}  CPU other seed ${f(r.releasedJ.cpuOtherSeed, 0)}  GPU ${f(r.releasedJ.gpu, 0)}`);
  console.log(`  particles    CPU ${r.particles.cpu}  CPU other seed ${r.particles.cpuOtherSeed}  GPU ${r.particles.gpu}`);
  console.log(`  gas T  mean |diff| GPU ${e(r.gasT.gpuMeanAbs)} K vs other seed ${e(r.gasT.seedMeanAbs)} K; max ${e(r.gasT.gpuMax)} vs ${e(r.gasT.seedMax)} K`);
  console.log(`  solid Ts mean |diff| GPU ${e(r.solidTs.gpuMeanAbs)} K vs other seed ${e(r.solidTs.seedMeanAbs)} K; max ${e(r.solidTs.gpuMax)} vs ${e(r.solidTs.seedMax)} K`);
}
if (report.full) {
  const r = report.full;
  console.log(`\nFull model (radiation, flame-front particles), ${r.horizon} s, ${r.cells} cells, ${r.nodes} nodes`);
  console.log(`  released MJ   CPU ${f(r.released.cpu / 1e6)}  GPU ${f(r.released.gpu / 1e6)}`);
  console.log(`  burnt kg      CPU ${f(r.burnt.cpu, 3)}  GPU ${f(r.burnt.gpu, 3)}`);
  console.log(`  radiated MJ   CPU ${f(r.radiated.cpu / 1e6)}  GPU ${f(r.radiated.gpu / 1e6)}`);
  console.log(`  peak HRR kW   CPU ${f(r.peakHrr.cpu, 1)}  GPU ${f(r.peakHrr.gpu, 1)}`);
  console.log(`  node T: RMS difference ${f(r.nodeT.rms)} C, correlation ${f(r.nodeT.correlation, 4)}; vis < 10 m agreement ${f(100 * r.smokyAgreement, 1)} %`);
  console.log(`  untenable nodes CPU ${r.untenable.cpu}  GPU ${r.untenable.gpu}; |untenable time difference| median ${f(r.untenable.medianAbsDiff, 0)} s, max ${f(r.untenable.maxAbsDiff, 0)} s`);
  console.log(`  run time CPU ${f(r.ms.cpu / 1000, 1)} s  GPU ${f(r.ms.gpu / 1000, 1)} s`);
  console.log("  time s | HRR kW CPU / GPU | mean node T C CPU / GPU | max node T C CPU / GPU | nodes vis<10 m CPU / GPU | particles CPU / GPU");
  for (const s of r.series) console.log(`  ${String(s.time).padStart(5)}  | ${f(s.hrrCpu, 1).padStart(7)} / ${f(s.hrrGpu, 1).padStart(7)} | ${f(s.meanTCpu, 1).padStart(6)} / ${f(s.meanTGpu, 1).padStart(6)} | ${f(s.maxTCpu, 1).padStart(6)} / ${f(s.maxTGpu, 1).padStart(6)} | ${String(s.smokyCpu).padStart(3)} / ${String(s.smokyGpu).padStart(3)} | ${s.partsCpu} / ${s.partsGpu}`);
}
if (report.onestep) {
  console.log(`
One step from the CPU's own state (the app's scene, ${report.onestep.cells} cells, radiation and particles off${report.onestep.ducts ? `, ventilation running through ${report.onestep.ducts} terminals` : ""}): largest |CPU - GPU|, and |CPU - CPU twin one ulp warmer|`);
  console.log("  step | released J CPU / GPU | sweeps | gas T K (rel)        | Ts K (rel)           | rho rel  | O2       | soot     | twin T K (rel)");
  for (const r of report.onestep.rows) console.log(`  ${String(r.step).padStart(4)} | ${f(r.releasedJ.cpu, 1).padStart(8)} / ${f(r.releasedJ.gpu, 1).padStart(8)} | ${r.sweeps.cpu}/${r.sweeps.gpu} | ${e(r.gpu.T)} (${e(r.gpu.Trel)}) | ${e(r.gpu.Ts)} (${e(r.gpu.Tsrel)}) | ${e(r.gpu.rho)} | ${e(r.gpu.O2)} | ${e(r.gpu.soot)} | ${e(r.twin.T)} (${e(r.twin.Trel)})`);
}
if (report.ensemble) {
  const q = (list, key) => {
    const v = list.map((x) => x[key]).sort((a, b) => a - b);
    return { mean: v.reduce((a, b) => a + b, 0) / v.length, min: v[0], max: v.at(-1) };
  };
  console.log(`
Ensembles, the app's scene (${report.ensemble.cells} cells), 60 s, radiation and particles off, ${report.ensemble.members} members (ambient + k mK):`);
  for (const side of ["cpu", "gpu"]) {
    const r = q(report.ensemble.runs[side], "released"), p = q(report.ensemble.runs[side], "peak");
    console.log(`  ${side.toUpperCase()}: released MJ mean ${f(r.mean / 1e6, 3)} (${f(r.min / 1e6, 3)} to ${f(r.max / 1e6, 3)}); peak HRR kW mean ${f(p.mean, 2)} (${f(p.min, 2)} to ${f(p.max, 2)})`);
  }
}
if (report.worker) {
  console.log("\nThe app's worker (flamescapes.worker.js), the room's meshes through buildScene:");
  for (const [label, w] of Object.entries(report.worker)) {
    if (w.error) console.log(`  ${label}: error ${w.error}`);
    else console.log(`  ${label === "gpu" ? "backend auto" : "backend cpu "}: ran on ${w.backend?.kind} (${w.backend?.name}), grid ${w.grid?.join("x")}, ${w.frames} frames, ${w.n} nodes (${w.inGrid} in grid), released ${f(w.released / 1e6, 3)} MJ, peak ${f(w.peakHrr, 1)} kW, ${f(w.ms / 1000, 1)} s; result keys ${w.keys}; stats ${w.statKeys}`);
  }
}
if (report.embers) console.log(`\nEmbers on: CPU spawned ${report.embers.cpu.spawned}, spots ${report.embers.cpu.spots}, peak ${f(report.embers.cpu.peakHrr, 1)} kW; GPU spawned ${report.embers.gpu.spawned}, spots ${report.embers.gpu.spots}, peak ${f(report.embers.gpu.peakHrr, 1)} kW`);
if (report.timing) {
  console.log("\nms per step (full model at the start of the fire; GPU steps queued, then waited for)");
  for (const r of report.timing) console.log(`  ${r.label.padEnd(5)} ${r.grid.join("x")} = ${r.cells} cells, h ${f(r.h, 3)} m: CPU ${f(r.cpuMs, 1)} ms (${r.cpuSteps} steps), GPU ${f(r.gpuMs, 2)} ms (${r.gpuSteps} steps; encode ${f(r.encodeMsPerStep, 2)} ms; grid stages only ${f(r.gpuGridOnlyMs, 2)} ms), speed-up ${f(r.cpuMs / r.gpuMs, 1)}x`);
}
console.log(`\ntotal ${((Date.now() - t0) / 1000).toFixed(0)} s`);
const json = flag("--json", null);
if (json) await writeFile(json, JSON.stringify(report, null, 1));
if (report.error) process.exit(1);
