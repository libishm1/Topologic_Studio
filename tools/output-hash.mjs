/**
 * Fingerprints of everything the browser core computes on the benchmark
 * corpus, to show that a speed-up changed no result (as tools/graph-hash.mjs
 * does for the graph alone):
 *
 *   node tools/output-hash.mjs > before.txt          # every model in benchmark/manifest.json
 *   (change route.js, deadline.js, forecast.js, rl.js, ...)
 *   node tools/output-hash.mjs > after.txt
 *   node tools/output-hash.mjs --diff before.txt after.txt
 *
 * Options as tools/benchmark.mjs: --only REGEX, --samples DIR. --save DIR also
 * writes every output in full (JSON per model), so outputs whose hashes differ
 * can be compared with a tolerance: --diff A B --values DIRA DIRB --tol 1e-12.
 *
 * One line per model and output: the graph; A* routes; Yen and distinct
 * alternatives (k = 3, no time budget, so the result cannot depend on speed);
 * straightened routes; both fire forecasts (rates, untenable and closing
 * times) and the hop-front surrogate; hard deadlines; planWithForecast from
 * five starts; doseDeadline (tolerance 0.1, as the worker) from the same
 * starts; the dose deadline map; RL training with fixed seeds. Floats are
 * hashed by their bytes, or by their shortest round-trip text, so equal
 * hashes mean byte-identical results.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const argv = process.argv.slice(2);
const option = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : fallback;
};

// ------------------------------------------------------------------ diff

if (argv.includes("--diff")) {
  const i = argv.indexOf("--diff");
  const read = (f) => new Map(readFileSync(f, "utf8").replace(/^﻿/, "").split(/\r?\n/).filter((l) => /^[0-9a-f]{16} /.test(l)).map((l) => {
    const [hash, ...rest] = l.trim().split(/\s+/);
    return [rest.join(" "), hash];
  }));
  const a = read(argv[i + 1]), b = read(argv[i + 2]);
  let same = 0;
  const differ = [];
  for (const [k, h] of a) {
    if (!b.has(k)) differ.push(`${k}: missing after`);
    else if (b.get(k) !== h) differ.push(k);
    else same += 1;
  }
  for (const k of b.keys()) if (!a.has(k)) differ.push(`${k}: missing before`);
  console.log(`${same} identical, ${differ.length} different`);
  for (const d of differ) console.log(`  DIFFERENT ${d}`);
  const values = argv.indexOf("--values");
  if (values >= 0 && differ.length) {
    const tol = Number(option("--tol", "1e-12"));
    const [da, db] = [argv[values + 1], argv[values + 2]];
    for (const d of differ) {
      const [model, what] = d.split(" ");
      const fa = path.join(da, `${model}.json`), fb = path.join(db, `${model}.json`);
      if (!existsSync(fa) || !existsSync(fb)) continue;
      const va = JSON.parse(readFileSync(fa, "utf8"))[what], vb = JSON.parse(readFileSync(fb, "utf8"))[what];
      let worst = 0, structural = false;
      const walk = (x, y) => {
        if (typeof x === "number" && typeof y === "number") {
          const e = Math.abs(x - y) / Math.max(1, Math.abs(x));
          if (e > worst) worst = e;
        } else if (Array.isArray(x) && Array.isArray(y) && x.length === y.length) x.forEach((v, k) => walk(v, y[k]));
        else if (x && y && typeof x === "object" && typeof y === "object") for (const k of new Set([...Object.keys(x), ...Object.keys(y)])) walk(x[k], y[k]);
        else if (x !== y) structural = true;
      };
      walk(va, vb);
      console.log(`  ${model} ${what}: ${structural ? "STRUCTURAL difference" : `largest relative difference ${worst.toExponential(2)} (${worst <= tol ? "within" : "OUTSIDE"} ${tol})`}`);
    }
  }
  process.exit(differ.length ? 1 : 0);
}

// ------------------------------------------------------------- compute

const CORE = "../topologicpy-web-frontend/src/core/";
const { buildLatticeGraph } = await import(CORE + "lattice.js");
const { astar, componentLabels } = await import(CORE + "route.js");
const { alternativeRoutes } = await import(CORE + "alternatives.js");
const { straightenRoute } = await import(CORE + "straighten.js");
const { buildForecast, planWithForecast } = await import(CORE + "forecast.js");
const { doseDeadline, doseDeadlineMap, escapeDeadlines } = await import(CORE + "deadline.js");
const { hopSurrogate } = await import(CORE + "surrogate.js");
const { trainRl } = await import(CORE + "rl.js");
const { extract } = await import("./ifc-extract.mjs");
const { findExits } = await import("./exits.mjs");

for (const name of ["log", "warn", "error", "info"]) {
  const orig = console[name].bind(console);
  console[name] = (...a) => {
    if (!(typeof a[0] === "string" && a[0].startsWith("[WEB-IFC]"))) orig(...a);
  };
}

/** JSON with non-finite numbers and typed arrays kept apart. */
const plain = (x) =>
  JSON.parse(JSON.stringify(x, (k, v) => {
    if (typeof v === "number" && !Number.isFinite(v)) return String(v);
    if (ArrayBuffer.isView(v)) return Array.from(v, (y) => (Number.isFinite(y) ? y : String(y)));
    return v;
  }));
const hash = (x) => {
  const h = createHash("sha256");
  if (ArrayBuffer.isView(x)) h.update(Buffer.from(x.buffer, x.byteOffset, x.byteLength));
  else h.update(JSON.stringify(plain(x)));
  return h.digest("hex").slice(0, 16);
};

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const manifest = JSON.parse(readFileSync(path.join(ROOT, "benchmark", "manifest.json"), "utf8"));
const only = option("--only", null);
const save = option("--save", null);
const samples = option("--samples", process.env.BENCHMARK_SAMPLES || path.join(os.homedir(), "ifc-samples"));
if (save) mkdirSync(save, { recursive: true });

for (const model of manifest.models.filter((m) => !m.stressOnly && (!only || new RegExp(only, "i").test(m.name)))) {
  const file = model.root === "examples" ? path.join(ROOT, manifest.roots.examples, model.path) : path.join(samples, ...model.path.split("/"));
  if (!existsSync(file)) {
    console.log(`# ${model.name}: file not found`);
    continue;
  }
  const tag = model.name.replace(/\s+/g, "_");
  const outputs = {};
  const geo = await extract(file, { withDoorFlags: true });
  const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
  const g = { ...built, nodes: built.points, upAxis: "y", agentHeight: 0.75 };
  const n = g.nodes.length / 3;
  outputs.graph = [built.points, built.edges, built.kinds, built.blocked, ...(built.sections || []).map((s) => s.segs)].map(hash);
  if (n) {
    const { labels, sizes } = componentLabels(g);
    const { exits } = findExits(geo, built);
    const random = rng(parseInt(model.sha256.slice(0, 8), 16) ^ 0x5eed);
    const pick = (list) => list[Math.floor(random() * list.length)];
    const big = [];
    for (let v = 0; v < n; v += 1) if (sizes[labels[v]] >= Math.max(20, Math.floor(0.01 * n))) big.push(v);
    const pairs = [];
    for (let k = 0; k < 400 && pairs.length < 6 && big.length; k += 1) {
      const s = pick(big), t = exits.length ? pick(exits) : pick(big);
      if (s !== t && labels[s] === labels[t]) pairs.push([s, t]);
    }
    const P = g.nodes;
    const point = (v) => [P[v * 3], P[v * 3 + 1], P[v * 3 + 2]];
    const routes = pairs.map(([s, t]) => astar(g, s, t));
    outputs.astar = routes.map((r) => r && { nodes: r.nodes, cost: r.cost });
    for (const method of ["yen", "distinct"]) {
      outputs[method] = pairs.slice(0, 3).map(([s, t]) => {
        const r = alternativeRoutes(g, point(s), point(t), { k: 3, method, timeBudgetMs: 1e12 });
        return { routes: r.routes.map((x) => ({ nodes: x.node_ids, cost: x.cost, length: x.length, stretch: x.stretch, overlap: x.overlap })), note: r.note };
      });
    }
    outputs.straighten = routes.map((r) => r && straightenRoute(g, r.nodes)).map((s) => s && { ids: s.node_ids, length: s.length });

    const members = [];
    const largest = sizes.indexOf(Math.max(...sizes));
    for (let v = 0; v < n; v += 1) if (labels[v] === largest) members.push(v);
    const seed = members[Math.floor(members.length / 2)];
    const fOpts = { seed, steps: 60, dt: 5, threshold: 80, blocked: g.blocked };
    const sur = hopSurrogate(g, fOpts);
    outputs.surrogate = [sur.rate, sur.untenable, sur.closes].map(hash);
    for (const fire of ["temperature", "flood"]) {
      const f = buildForecast(g, { model: fire, ...fOpts });
      outputs[`forecast_${fire}`] = [f.rate, f.untenable, f.closes, f.K, f.stride].map((x) => (typeof x === "number" ? x : hash(x)));
      if (!exits.length) continue;
      const hard = escapeDeadlines(g, { exits, untenable: f.closes, speed: 1.2, blocked: g.blocked });
      outputs[`hard_${fire}`] = hard;
      const can = [];
      for (let v = 0; v < n; v += 1) if (hard[v] > 0) can.push(v);
      const starts = can.length ? Array.from({ length: 5 }, (_, i) => can[Math.floor(((i + 0.5) * can.length) / 5)]) : [];
      const opts = { exits, forecast: f, speed: 1.2, blocked: g.blocked };
      outputs[`plan_${fire}`] = starts.map((s) => {
        const p = planWithForecast(g, { start: s, ...opts });
        delete p.ms;
        return p;
      });
      outputs[`doseDeadline_${fire}`] = starts.map((s) => doseDeadline(g, s, { ...opts, limit: 0.3, tolerance: 0.1 }));
      const map = doseDeadlineMap(g, { ...opts, limit: 0.3 });
      outputs[`map_${fire}`] = { deadline: map.deadline, step: map.step, layers: map.layers };
    }
    outputs.rl = pairs.slice(0, 2).map(([s, t], i) => {
      const r = trainRl(g, { startPoint: point(s), exitPoint: point(t), firePoint: point(seed), episodes: 400, maxSteps: 200, useFire: true, fireSpeed: 1, useWalls: true, seed: 11 + i });
      delete r.train_ms;
      return r;
    });
  }
  for (const [what, value] of Object.entries(outputs)) console.log(`${hash(value)} ${tag} ${what}`);
  if (save) writeFileSync(path.join(save, `${tag}.json`), JSON.stringify(plain(outputs)));
}
