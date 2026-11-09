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
 *   2c. Sprinklers (src/core/suppression.js): a head over the fire, runScene on
 *      both paths with no sprinklers, activation only, Evans and the spray;
 *      activation times, heat, fuel and water, and each run's time.
 *   2d. Systems (src/core/suppressionSystems.js): the room flooded with
 *      clean agent or foamed, 200 deterministic steps on both paths beside
 *      the one-ulp twin; then runScene on both paths: no system, the agent
 *      on detection, the agent with a 30 s hold, and a heptane spill with
 *      and without foam.
 *   3. Milliseconds per step, CPU and GPU, at about 30k and 200k cells.
 *
 * Uses playwright-core with the installed Chrome (no browser download) and
 * serves topologicpy-web-frontend/ itself, so the dev server is not needed.
 * On this machine (Chrome 156, Intel UHD 770, D3D12) headless Chrome exposes
 * the GPU's WebGPU adapter with no extra flags; --swiftshader forces the
 * software adapter (--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader).
 *
 *   node tools/flamescapes-gpu-check.mjs [--headed] [--swiftshader] [--quick]
 *        [--phase agreement|onestep|ensemble|radiation|full|worker|sprinklers|spread|design|leak|systems|timing] [--horizon 120] [--members 6]
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
// --members N: the spread phase's ensemble on each path (0 skips it; default 6, 3 with --quick).
const MEMBERS = Number(flag("--members", QUICK ? 3 : 6));

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
async function suite({ base, phase, horizon, quick, fireClass, ducts, members: spreadMembers }) {
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

  /** A deep copy of a CPU solver state with its own random streams (the rays follow P.seed since 2026-11-09). */
  const cloneSim = (sim, seed) => {
    const c = { ...sim, P: { ...sim.P, seed }, topo: null, random: V.rng(seed), stats: { ...sim.stats }, parts: { ...sim.parts } };
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
    // The ceiling detectors (2026-11-07), read after the nodes on the GPU: their
    // visibility, and when the first reaches C/VM2's 0.097 per metre (13.4 m).
    const firstAlarm = (f) => {
      const D = f.detectorVis ? f.detectorVis.length / K : 0;
      for (let k = 0; k < K; k += 1) for (let d = 0; d < D; d += 1) if (f.detectorVis[k * D + d] <= 3 / (0.097 * Math.LN10)) return (k + 1) * 5;
      return null;
    };
    const detectors = { count: cpu.frames.detectorVis ? cpu.frames.detectorVis.length / K : 0, vis: agree("detectorVis"), firstCpu: firstAlarm(cpu.frames), firstGpu: firstAlarm(gpu.frames) };
    log(`ceiling detectors: ${detectors.count}, first alarm CPU ${detectors.firstCpu} s, GPU ${detectors.firstGpu} s`);
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
      detectors,
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

  // ---- 2c. sprinklers (src/core/suppression.js, 2026-11-07): one quick-response
  // head over the fire, through runScene on both paths, with no sprinklers,
  // activation only, Evans, and Evans with the spray. Activation times, heat
  // released, fuel burnt and the water, and each run's time.
  if (phase === "all" || phase === "sprinklers") {
    const head = { id: "over", position: [2.5, 2.55, 2.0], footprint: { centre: [2.5, 2.0], axis: [1, 0], half: [2.3, 2.3] }, rti: 50, c: 0.4, activationC: 68, flowLpm: 85, coverage: 20.9 };
    out.sprinklers = [];
    for (const mode of [null, "activation", "evans", "spray", "wetting"]) {
      const o = { nodes: nodeArray, upAxis: "y", horizon, frameSeconds: 5, particles: 3000, sprinklers: mode ? { heads: [head], mode } : null };
      const row = { mode: mode || "none" };
      // A 200 kW burner under the first item, so the head opens within the run.
      const scene = () => {
        const sc = room(0.25, 200000);
        sc.pilot = { ...sc.pilot, power: 200000 };
        return sc;
      };
      for (const [name, fn] of [["cpu", (sc) => R.runScene(sc, o)], ["gpu", (sc) => runner.runScene(sc, o)]]) {
        const t0 = performance.now();
        const r = await fn(scene());
        const s = r.sprinklers;
        row[name] = {
          ms: performance.now() - t0,
          released: r.stats.released,
          burnt: r.stats.burnt,
          peakHrr: Math.max(...r.hrr),
          estimate: s?.heads[0].estimate ?? null,
          late: s?.heads[0].late ?? null,
          covered: s?.covered ?? null,
          columns: s?.columns ?? null,
          water: s?.stats ?? null,
        };
      }
      out.sprinklers.push(row);
      log(`sprinklers ${row.mode}: CPU opens ${row.cpu.estimate} s, GPU ${row.gpu.estimate} s; released CPU ${(row.cpu.released / 1e6).toFixed(2)} MJ, GPU ${(row.gpu.released / 1e6).toFixed(2)} MJ`);
    }
  }

  // ---- 2d. spread (2026-11-09): the sprinkler phase's fire with no heads,
  // members on each path with the room k mK warmer (member 0 is that phase's
  // run). Whether the 200 kW burner lights the contents around it is a
  // threshold, so one run on each path says little; the members show each
  // path's spread. Only with --phase spread.
  if (phase === "spread") {
    const members = spreadMembers;
    out.spread = { members, horizon, runs: { cpu: [], gpu: [] } };
    for (let k = 0; k < members; k += 1) {
      const o = { nodes: nodeArray, upAxis: "y", horizon, frameSeconds: 5, particles: 3000, solverOptions: { ambient: 293.15 + k * 1e-3 } };
      for (const [name, fn] of [["cpu", (sc) => R.runScene(sc, o)], ["gpu", (sc) => runner.runScene(sc, o)]]) {
        const sc = room(0.25, 200000);
        sc.pilot = { ...sc.pilot, power: 200000 };
        const r = await fn(sc);
        out.spread.runs[name].push({ released: r.stats.released, burnt: r.stats.burnt, hrr: Array.from(r.hrr) });
      }
      log(`spread member ${k}: released CPU ${(out.spread.runs.cpu[k].released / 1e6).toFixed(2)} MJ, GPU ${(out.spread.runs.gpu[k].released / 1e6).toFixed(2)} MJ`);
    }
    // From identical states: the CPU's run at t0 handed to the GPU and to two
    // CPU copies with other random streams (radiation's rays), 10 s on each,
    // the heat released each second (kW); with radiation off too, which draws
    // no random numbers.
    out.spread.steps = [];
    for (const [label, extra] of [["default", {}], ["radiation off", { emitAbove: 1e9 }]]) {
      const sc = room(0.25, 200000);
      sc.pilot = { ...sc.pilot, power: 200000 };
      const base = solverOf(sc, { dt: 0.1, embers: false, ...extra });
      for (const t0 of quick ? [25] : [20, 25, 30, 35]) {
        while (base.time < t0 - 1e-9) F.step(base, 0.1);
        const a = cloneSim(base, base.P.seed), b = cloneSim(base, 12);
        const gpu = runner.createSim(base, {});
        const kw = { cpu: [], cpuOther: [], gpu: [] };
        let qa = a.stats.released, qb = b.stats.released;
        for (let s = 0; s < 10; s += 1) {
          for (let q = 0; q < 10; q += 1) {
            F.step(a, 0.1);
            F.step(b, 0.1);
            gpu.step(0.1);
          }
          const fr = await gpu.readFrame({ particles: false });
          kw.cpu.push((a.stats.released - qa) / 1000);
          kw.cpuOther.push((b.stats.released - qb) / 1000);
          kw.gpu.push(fr.released / 1000);
          qa = a.stats.released;
          qb = b.stats.released;
        }
        gpu.destroy();
        out.spread.steps.push({ label, t0, kw });
        const sum = (v) => v.reduce((x, y) => x + y, 0) / 1000;
        log(`spread from ${label} at ${t0} s, next 10 s: released CPU ${sum(kw.cpu).toFixed(2)} MJ, CPU other rays ${sum(kw.cpuOther).toFixed(2)} MJ, GPU ${sum(kw.gpu).toFixed(2)} MJ`);
      }
    }
    // One step after the contents flare: the heat radiated, and the heat the
    // step left in the fuel (porous cells), the walls and the gas (J; the
    // gas at c_p 1100), on each path from the same state.
    out.spread.budget = [];
    {
      const sc = room(0.25, 200000);
      sc.pilot = { ...sc.pilot, power: 200000 };
      const base = solverOf(sc, { dt: 0.1, embers: false });
      const P = base.P;
      const cap = (s, c) => base.inertC[c] + Math.min(s.fuelMass[c], P.skinDensity * base.area[c]) * (base.fuels[base.fuelKind[c]].heatCapacity ?? P.fuelHeatCapacity) + s.moisture[c] * 4186;
      const budget = (s0, s) => {
        const e = { fuel: 0, walls: 0, gas: 0 };
        for (let c = 0; c < base.N; c += 1) {
          const k = base.kind[c];
          if (k !== 0) e[k === 1 ? "fuel" : "walls"] += cap(s, c) * (s.Ts[c] - s0.Ts[c]);
          if (k !== 2) e.gas += s.rho[c] * base.V * 1100 * (s.T[c] - s0.T[c]);
        }
        return e;
      };
      for (const t0 of quick ? [32] : [31, 32, 34]) {
        while (base.time < t0 - 1e-9) F.step(base, 0.1);
        const s0 = { T: base.T.slice(), Ts: base.Ts.slice(), rho: base.rho.slice(), fuelMass: base.fuelMass.slice(), moisture: base.moisture.slice() };
        const a = cloneSim(base, base.P.seed), b = cloneSim(base, 12);
        const gpu = runner.createSim(base, {});
        F.step(a, 0.1);
        F.step(b, 0.1);
        gpu.step(0.1);
        const fr = await gpu.readFrame({ particles: false });
        const g = await gpu.readFields();
        gpu.destroy();
        const row = { t0 };
        for (const [name, s, rad, rel] of [["cpu", a, a.stats.radiated - base.stats.radiated, a.stats.released - base.stats.released], ["cpuOther", b, b.stats.radiated - base.stats.radiated, b.stats.released - base.stats.released], ["gpu", g, fr.radiated, fr.released]]) {
          row[name] = { radiated: rad, released: rel, ...budget(s0, s) };
        }
        out.spread.budget.push(row);
        log(`spread one step at ${t0} s: radiated CPU ${row.cpu.radiated.toFixed(0)} J, other rays ${row.cpuOther.radiated.toFixed(0)} J, GPU ${row.gpu.radiated.toFixed(0)} J`);
      }
    }
  }

  // ---- 2e. A design-fire first item (flamescapesRun.js itemBurner,
  // 2026-11-08): a fast t-squared 5 kg item, no pilot. Its heat is the
  // host's (flamescapes.js burnerHeat), the same on both paths, so the
  // frames' heat release should match while it burns alone; then with a
  // head over it (wetting): when the head opens and the item's decay.
  if (phase === "all" || phase === "design") {
    const head = { id: "over", position: [2.5, 2.55, 2.0], footprint: { centre: [2.5, 2.0], axis: [1, 0], half: [2.3, 2.3] }, rti: 50, c: 0.4, activationC: 68, flowLpm: 85, coverage: 20.9 };
    out.design = [];
    for (const mode of [null, "wetting"]) {
      const scene = () => {
        const sc = room(0.25, 200000);
        const fire = R.itemBurner(sc.grid, sc.solid, sc.ignition.cell, { alpha: 0.0469, growth: "fast", item: { mass: 5, area: 1 }, fuel: ignitionFuel });
        const under = new Set(fire.layers.flat());
        const keep = [];
        for (let q = 0; q < sc.contents.cells.length; q += 1) if (!under.has(sc.contents.cells[q])) keep.push(q);
        return { ...sc, contents: { cells: keep.map((q) => sc.contents.cells[q]), masses: keep.map((q) => sc.contents.masses[q]) }, ignition: { ...sc.ignition, designFire: fire.design }, pilot: null, burner: fire };
      };
      const o = { nodes: nodeArray, upAxis: "y", horizon: quick ? 90 : 150, frameSeconds: 5, particles: 3000, sprinklers: mode ? { heads: [head], mode } : null };
      const row = { mode: mode || "none" };
      for (const [name, fn] of [["cpu", (sc, opts) => R.runScene(sc, opts)], ["gpu", (sc, opts) => runner.runScene(sc, opts)]]) {
        // The last frame's flame points' temperatures (the item's drawn flame
        // takes its cells', read back with the nodes on the GPU).
        let flame = [];
        const onFrame = (f) => {
          flame = [];
          for (let q = 0; q < (f.particles?.kinds.length ?? 0); q += 1) if (f.particles.kinds[q] === 0) flame.push(f.particles.temps[q]);
        };
        const t0 = performance.now();
        const r = await fn(scene(), { ...o, onFrame });
        flame.sort((a, b) => a - b);
        const at = (f) => (flame.length ? +flame[Math.floor(f * (flame.length - 1))].toFixed(0) : null);
        row[name] = { ms: performance.now() - t0, hrr: Array.from(r.hrr), released: r.stats.released, item: r.designFire ?? null, estimate: r.sprinklers?.heads[0].estimate ?? null, flame: { n: flame.length, p10: at(0.1), p50: at(0.5), p90: at(0.9) } };
      }
      out.design.push(row);
      log(`design ${row.mode}: item released CPU ${(row.cpu.item?.released / 1e6).toFixed(2)} MJ, GPU ${(row.gpu.item?.released / 1e6).toFixed(2)} MJ; head CPU ${row.cpu.estimate} s, GPU ${row.gpu.estimate} s; flame points CPU ${JSON.stringify(row.cpu.flame)}, GPU ${JSON.stringify(row.gpu.flame)} K`);
    }
  }

  // ---- 2e. walls (tools/wall-leak-check.mjs on the GPU, 2026-11-08): a
  // closed 5 x 4 x 2.7 m room, its walls 0.1 and 0.2 m thick, square to the
  // grid and turned 30 degrees, on 0.27 and 0.34 m cells, a fast design fire
  // in the middle and no contents, 60 s on each path; the heat and soot
  // outside the walls against the soot inside.
  if (phase === "all" || phase === "leak") {
    const turned = (x0, y0, z0, x1, y1, z1, deg) => {
      const b = box(x0, y0, z0, x1, y1, z1, S);
      const a = (deg * Math.PI) / 180, cos = Math.cos(a), sin = Math.sin(a);
      for (let q = 0; q < b.positions.length; q += 3) {
        const x = b.positions[q] - 2.5, z = b.positions[q + 2] - 2;
        b.positions[q] = 2.5 + cos * x - sin * z;
        b.positions[q + 2] = 2 + sin * x + cos * z;
      }
      return b;
    };
    const zoneOf = (x, y, z, th, deg) => {
      const a = (-deg * Math.PI) / 180, cos = Math.cos(a), sin = Math.sin(a);
      const u = 2.5 + cos * (x - 2.5) - sin * (z - 2), v = 2 + sin * (x - 2.5) + cos * (z - 2);
      if (u > 0 && u < W && v > 0 && v < D && y > 0 && y < H) return 0;
      if (u > -th && u < W + th && v > -th && v < D + th && y > -th && y < H + th) return 1;
      return 2;
    };
    out.leak = [];
    for (const h of [0.27, 0.34]) {
      for (const th of [0.1, 0.2]) {
        for (const deg of [0, 30]) {
          const geometry = {
            walls: [turned(-th, 0, -th, W + th, H, 0, deg), turned(-th, 0, D, W + th, H, D + th, deg), turned(-th, 0, 0, 0, H, D, deg), turned(W, 0, 0, W + th, H, D, deg)],
            floors: [turned(-th, -th, -th, W + th, 0, D + th, deg)],
            roofs: [turned(-th, H, -th, W + th, H + th, D + th, deg)],
            stairs: [],
            obstacles: [],
          };
          const make = () => R.buildScene(geometry, { upAxis: "y", fire: [2.5, 0.75, 2], minCell: h, maxCell: h - 1e-3, reach: 9, maxCells: 5e6, contentsFuel, ignitionFuel, item: { mass: 5, area: 1 }, designFire: { alpha: 0.0469, growth: "fast" }, fireLoad: 0 });
          const measure = (scene, T, Y, kind) => {
            const g = scene.grid;
            let insideSoot = 0, outsideSoot = 0, outsideRise = 0;
            for (let c = 0; c < g.cells; c += 1) {
              if (kind[c] === 2) continue;
              const x = g.origin[0] + ((c % g.nx) + 0.5) * g.h, y = g.origin[1] + ((Math.floor(c / g.nx) % g.ny) + 0.5) * g.h, z = g.origin[2] + (Math.floor(c / (g.nx * g.ny)) + 0.5) * g.h;
              const zone = zoneOf(x, y, z, th, deg);
              if (zone === 0) insideSoot += Y[c * 6 + 4];
              else if (zone === 2) {
                outsideSoot += Y[c * 6 + 4];
                outsideRise = Math.max(outsideRise, T[c] - 293.15);
              }
            }
            return { outsideShare: insideSoot > 0 ? outsideSoot / insideSoot : 0, outsideRise };
          };
          const cpuScene = make();
          const cpu = solverOf(cpuScene, {});
          for (let s = 0; s < 600; s += 1) F.step(cpu, 0.1);
          const gpuScene = make();
          const gsim = runner.createSim(solverOf(gpuScene, {}), {});
          for (let s = 0; s < 600; s += 1) gsim.step(0.1);
          await gsim.finish();
          const fields = await gsim.readFields();
          gsim.destroy();
          const row = { cell: h, thick: th, angle: deg, grid: [cpuScene.grid.nx, cpuScene.grid.ny, cpuScene.grid.nz], cpu: measure(cpuScene, cpu.T, cpu.Y, cpu.kind), gpu: measure(gpuScene, fields.T, fields.Y, fields.kind) };
          out.leak.push(row);
          log(`leak ${h} m, wall ${th} m at ${deg} deg: soot outside CPU ${(100 * row.cpu.outsideShare).toFixed(3)} %, GPU ${(100 * row.gpu.outsideShare).toFixed(3)} % of inside; hottest outside CPU +${row.cpu.outsideRise.toFixed(2)} K, GPU +${row.gpu.outsideRise.toFixed(2)} K`);
        }
      }
    }
  }

  // ---- 2d. flooded and foamed rooms (src/core/suppressionSystems.js,
  // 2026-11-07). First the deterministic stages (radiation and particles
  // off) stepped 200 times on both paths from one state, with the room
  // flooded with clean agent from 5 s (manual release) and, in a second run,
  // foamed from 0 s: field differences beside the CPU's one-ulp twin. Then
  // the full model through runScene on both paths: no system, the clean
  // agent on detection (and with a 30 s hold, which lets the fire back),
  // and foam on a heptane spill kept alight by the burner.
  if (phase === "all" || phase === "systems") {
    const SS = await import(base + "suppressionSystems.js");
    const FC = await import(base + "fireClasses.js");
    const roomRec = SS.rectangleRoom({ id: "room", name: "SERVER RM", x0: 0, z0: 0, x1: W, z1: D, floor: 0, ceiling: H });
    out.systems = { agreement: [], runs: [] };
    for (const [label, system, params] of [["clean agent released at 5 s", "clean-agent", { releaseAt: 5 }], ["foam released at 0 s", "foam", { releaseAt: 0 }]]) {
      const scene = room(0.25, 200000);
      const opts = { emitAbove: 1e9, embers: false, dt: 0.1, maxParticles: 0 };
      const spec = { rooms: [{ ...roomRec, system, params }] };
      const cpu = solverOf(scene, opts);
      cpu.systems = R.systemsRun(scene, cpu, spec);
      const twin = solverOf(scene, opts);
      twin.systems = R.systemsRun(scene, twin, spec);
      for (let c = 0; c < twin.N; c += 1) if (twin.kind[c] !== 2) twin.T[c] = Math.fround(twin.T[c] * (1 + 2 ** -23));
      const simG = solverOf(scene, opts);
      const gpu = runner.createSim(simG, { systems: R.systemsRun(scene, simG, spec) });
      const rows = [];
      for (let s = 1; s <= 200; s += 1) {
        F.step(cpu, 0.1);
        F.step(twin, 0.1);
        gpu.step(0.1);
        if (![50, 100, 150, 200].includes(s)) continue;
        const g = await gpu.readFields();
        const moist = absDiffs(cpu.moisture, g.moisture, (c) => cpu.kind[c] !== 0);
        const moistTwin = absDiffs(cpu.moisture, twin.moisture, (c) => cpu.kind[c] !== 0);
        rows.push({ step: s, ...range(cpu), released: { cpu: cpu.stats.released, twin: twin.stats.released }, gpu: compare(cpu, g), twin: compare(cpu, twin), moist, moistTwin, stats: { ...cpu.systems.rooms[0].stats } });
        log(`systems agreement (${label}) step ${s}: max |dT| GPU ${rows.at(-1).gpu.T.max.toExponential(2)} K, twin ${rows.at(-1).twin.T.max.toExponential(2)} K; |d moisture| GPU ${moist.max.toExponential(2)} twin ${moistTwin.max.toExponential(2)} kg`);
      }
      gpu.destroy();
      out.systems.agreement.push({ label, rows });
    }
    const heptane = FC.fuelsFor("B").ignitionFuel;
    for (const [label, systemsOf, pool] of [
      ["no system", () => null, false],
      ["clean agent on detection", () => ({ rooms: [{ ...roomRec, system: "clean-agent" }] }), false],
      ["clean agent, 30 s hold", () => ({ rooms: [{ ...roomRec, system: "clean-agent", params: { hold: 30 } }] }), false],
      ["heptane spill, no system", () => null, true],
      ["heptane spill, foam on detection", () => ({ rooms: [{ ...roomRec, system: "foam" }] }), true],
    ]) {
      const scene = () => {
        const sc = room(0.25, 200000);
        if (pool) {
          // A 5 kg, 0.1 m2 heptane spill as the first item, kept alight by the burner (its own flame's feedback is below the grid).
          sc.fuels = { contentsFuel, ignitionFuel: heptane };
          sc.ignition = { ...sc.ignition, mass: 5, area: 0.1 };
          sc.pilot = { ...sc.pilot, duration: 1e9 };
        }
        return sc;
      };
      const o = { nodes: nodeArray, upAxis: "y", horizon, frameSeconds: 5, particles: 3000, systems: systemsOf() };
      const row = { label };
      for (const [name, fn] of [["cpu", (sc) => R.runScene(sc, o)], ["gpu", (sc) => runner.runScene(sc, o)]]) {
        const t0 = performance.now();
        const r = await fn(scene());
        const s = r.systems?.rooms[0] ?? null;
        const after = s?.dischargeAt != null ? Array.from(r.hrr).filter((_, k) => (k + 1) * 5 > s.dischargeAt + (s.floodedAt != null ? s.floodedAt - s.dischargeAt : s.outAt - s.dischargeAt)) : [];
        row[name] = {
          ms: performance.now() - t0,
          released: r.stats.released,
          burnt: r.stats.burnt,
          peakHrr: Math.max(...r.hrr),
          peakAfter: after.length ? Math.max(...after) : null,
          endHrr: r.hrr[r.hrr.length - 1],
          detectedAt: s?.detectedAt ?? null,
          dischargeAt: s?.dischargeAt ?? null,
          hrr: Array.from(r.hrr),
        };
      }
      out.systems.runs.push(row);
      log(`systems ${label}: detected CPU ${row.cpu.detectedAt} s, GPU ${row.gpu.detectedAt} s; released CPU ${(row.cpu.released / 1e6).toFixed(2)} MJ, GPU ${(row.gpu.released / 1e6).toFixed(2)} MJ`);
    }
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

const phases = PHASE === "all" ? ["agreement", "onestep", "ensemble", "radiation", "full", "worker", "sprinklers", "design", "systems", "timing"] : [PHASE];
const report = {};
const t0 = Date.now();
for (const phase of phases) {
  console.log(`\n== ${phase}`);
  const res = await page.evaluate(suite, { base: `${origin}/src/core/`, phase, horizon: HORIZON, quick: QUICK, fireClass: FIRE_CLASS, ducts: DUCTS, members: MEMBERS });
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
if (report.sprinklers) {
  console.log(`
Sprinklers: one quick-response head (RTI 50, C 0.4, 68 C) over a 200 kW burner; runScene on both paths`);
  console.log("  mode        | opens (estimate / late) CPU   GPU         | released MJ CPU / GPU | burnt kg CPU / GPU | covered, columns | water kg emitted (evaporated) | run s CPU / GPU");
  for (const r of report.sprinklers) {
    const t = (x) => (x === null ? "-" : f(x, 1));
    const w = r.cpu.water ? `${f(r.cpu.water.emitted, 1)} (${f(r.cpu.water.evaporated, 2)}) / ${f(r.gpu.water?.emitted, 1)}` : "-";
    console.log(`  ${r.mode.padEnd(11)} | ${t(r.cpu.estimate)} / ${t(r.cpu.late)}   ${t(r.gpu.estimate)} / ${t(r.gpu.late)} | ${f(r.cpu.released / 1e6)} / ${f(r.gpu.released / 1e6)} | ${f(r.cpu.burnt, 3)} / ${f(r.gpu.burnt, 3)} | ${r.cpu.covered ?? "-"}, ${r.cpu.columns ?? "-"} | ${w} | ${f(r.cpu.ms / 1000, 1)} / ${f(r.gpu.ms / 1000, 1)}`);
  }
}
if (report.spread) {
  const s = report.spread;
  console.log(`\nSpread: the sprinklers phase's fire, no heads, ${s.horizon} s, ${s.members} members on each path (ambient + k mK); frame heat release kW every 10 s`);
  for (const side of ["cpu", "gpu"]) {
    s.runs[side].forEach((r, k) => {
      console.log(`  ${side.toUpperCase()} ${k}: released ${f(r.released / 1e6)} MJ, burnt ${f(r.burnt, 3)} kg, peak ${f(Math.max(...r.hrr), 0)} kW | ${r.hrr.filter((_, q) => (q + 1) % 2 === 0).map((q) => f(q, 0)).join(" ")}`);
    });
  }
  console.log("  From identical states, 10 s on each: heat release kW each second, CPU with the GPU's rays / CPU other rays / GPU");
  for (const r of s.steps || []) {
    console.log(`  ${r.label} at ${r.t0} s: ${r.kw.cpu.map((q, k) => `${f(q, 0)}/${f(r.kw.cpuOther[k], 0)}/${f(r.kw.gpu[k], 0)}`).join("  ")}`);
  }
  console.log("  One step after the flare, kJ: radiated, released, and the heat the step left in the fuel, the walls and the gas; CPU with the GPU's rays / CPU other rays / GPU");
  for (const r of s.budget || []) {
    const k = (key) => `${f(r.cpu[key] / 1000, 1)}/${f(r.cpuOther[key] / 1000, 1)}/${f(r.gpu[key] / 1000, 1)}`;
    console.log(`  at ${r.t0} s: radiated ${k("radiated")}  released ${k("released")}  fuel ${k("fuel")}  walls ${k("walls")}  gas ${k("gas")}`);
  }
}
if (report.design) {
  console.log("\nDesign-fire first item (a fast t-squared 5 kg item, no pilot): frame heat release kW every 10 s, CPU / GPU; the item's own release and suppression");
  for (const r of report.design) {
    const it = (x) => (x ? `${f(x.released / 1e6)} MJ${x.suppressed ? `, ${x.suppressed} from ${f(x.suppressedAt, 1)} s` : ""}` : "-");
    console.log(`  ${r.mode}: head opens CPU ${r.cpu.estimate === null ? "-" : f(r.cpu.estimate, 1)} s, GPU ${r.gpu.estimate === null ? "-" : f(r.gpu.estimate, 1)} s; item CPU ${it(r.cpu.item)}, GPU ${it(r.gpu.item)}; run s CPU ${f(r.cpu.ms / 1000, 1)} / GPU ${f(r.gpu.ms / 1000, 1)}`);
    console.log(`    ${r.cpu.hrr.map((q, k) => [q, k]).filter(([, k]) => (k + 1) % 2 === 0).map(([q, k]) => `${(k + 1) * 5} s ${f(q, 0)}/${f(r.gpu.hrr[k], 0)}`).join(", ")}`);
  }
}
if (report.systems) {
  console.log("\nSystems (suppressionSystems.js): the room flooded with HFC-227ea or foamed, deterministic stages (radiation, particles off), 200 steps of 0.1 s");
  console.log("  |CPU - GPU| beside |CPU - CPU twin| (one ulp warmer); moisture over fuel cells");
  for (const run of report.systems.agreement) {
    console.log(`\n  ${run.label}`);
    console.log("  step | gas T (K) max / p99      | O2 max            | Ts max   | moisture max (kg) | twin: T max  O2 max   Ts max   moisture | released J CPU / twin | hottest gas C");
    for (const r of run.rows) {
      const G = r.gpu, Tw = r.twin;
      console.log(`  ${String(r.step).padStart(4)} | ${e(G.T.max)} / ${e(G.T.p99)} | ${e(G.O2.max)} | ${e(G.Ts.max)} | ${e(r.moist.max)} | ${e(Tw.T.max)} ${e(Tw.O2.max)} ${e(Tw.Ts.max)} ${e(r.moistTwin.max)} | ${f(r.released.cpu, 0)} / ${f(r.released.twin, 0)} | ${f(r.TmaxC, 0)}${G.kindMismatch ? `  (${G.kindMismatch} cells differ in kind)` : ""}`);
    }
  }
  console.log("\n  Full model through runScene (radiation, flame-front particles), both paths");
  console.log("  case                              | detected s CPU / GPU | discharge s CPU / GPU | released MJ CPU / GPU | burnt kg CPU / GPU | peak after kW CPU / GPU | end kW CPU / GPU | run s CPU / GPU");
  const t = (x) => (x === null || x === undefined ? "-" : f(x, 1));
  for (const r of report.systems.runs) {
    console.log(`  ${r.label.padEnd(33)} | ${t(r.cpu.detectedAt)} / ${t(r.gpu.detectedAt)} | ${t(r.cpu.dischargeAt)} / ${t(r.gpu.dischargeAt)} | ${f(r.cpu.released / 1e6)} / ${f(r.gpu.released / 1e6)} | ${f(r.cpu.burnt, 3)} / ${f(r.gpu.burnt, 3)} | ${t(r.cpu.peakAfter)} / ${t(r.gpu.peakAfter)} | ${t(r.cpu.endHrr)} / ${t(r.gpu.endHrr)} | ${f(r.cpu.ms / 1000, 1)} / ${f(r.gpu.ms / 1000, 1)}`);
  }
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
  if (r.detectors?.count) console.log(`  ceiling detectors ${r.detectors.count}: visibility RMS difference ${f(r.detectors.vis.rms)} m, correlation ${f(r.detectors.vis.correlation, 4)}; first alarm CPU ${r.detectors.firstCpu ?? "none"} s, GPU ${r.detectors.firstGpu ?? "none"} s`);
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
