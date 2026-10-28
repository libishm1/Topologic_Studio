/**
 * Speed and accuracy benchmark of Topologic Studio Next's browser core over a
 * fixed set of public IFC models (gap LG5 of the 2026-10-27 gap analysis).
 * What is measured, and why, is in benchmark/README.md.
 *
 *   node tools/benchmark.mjs                     # every model in benchmark/manifest.json
 *   node tools/benchmark.mjs --only "FZK|Duplex" # models whose name matches
 *   node tools/benchmark.mjs --quick             # one run per timing, fewer samples
 *   node tools/benchmark.mjs --verify            # only check the files' SHA-256
 *   node tools/benchmark.mjs --summary benchmark/results/2026-10-28.json
 *
 * Options: --runs N (timed repeats, default 5), --budget S (seconds per
 * computation and model after which repeats stop, default 20), --samples DIR
 * (folder holding the downloaded models; default $BENCHMARK_SAMPLES or
 * ~/ifc-samples), --out FILE (default benchmark/results/<date>.json).
 *
 * Each model runs in its own Node process (one at a time, so timings do not
 * compete), with a large heap for the 70 MB model and --expose-gc so a
 * collection runs between computations. The models themselves are never
 * copied into the repository: the manifest names each one's public source and
 * SHA-256, and a file that does not match is skipped.
 */
import { createHash } from "node:crypto";
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const MANIFEST = path.join(ROOT, "benchmark", "manifest.json");
const CORE = "../topologicpy-web-frontend/src/core/";

// The app's defaults (src/state/useStudio.js DEFAULT_SETTINGS).
const APP = {
  spacing: 0.5,
  agentHeight: 0.75,
  maxEdgeRise: 0.35,
  columnGap: 0.9,
  maxPoints: 40000,
  fireSteps: 60,
  secondsPerStep: 5,
  untenableAbove: 80,
  walkSpeed: 1.2,
  doseLimit: 0.3,
  rlEpisodes: 400,
  rlMaxSteps: 200,
  rlFireSpeed: 1,
  altCount: 3,
};
const OCTILE = 1.0824; // Lattice.lean, K_bounds: sqrt(4 - 2 sqrt 2) < 1.0824

// ------------------------------------------------------------------ args

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const option = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : fallback;
};

// ----------------------------------------------------------------- utils

/** mulberry32: a seeded generator, so the samples are the same every run. */
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

const median = (xs) => {
  const a = xs.filter((x) => Number.isFinite(x)).sort((p, q) => p - q);
  if (!a.length) return null;
  const k = a.length >> 1;
  return a.length % 2 ? a[k] : (a[k - 1] + a[k]) / 2;
};
const quantile = (xs, q) => {
  const a = xs.filter((x) => Number.isFinite(x)).sort((p, q2) => p - q2);
  if (!a.length) return null;
  return a[Math.min(a.length - 1, Math.max(0, Math.ceil(q * a.length) - 1))];
};
const max = (xs) => {
  const a = xs.filter((x) => Number.isFinite(x));
  return a.length ? Math.max(...a) : null;
};
const round = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(x) ? x ?? null : +x.toFixed(d));
const gc = () => globalThis.gc?.();

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

/** SHA-256 of a file, also tried with CRLF line ends folded to LF (a Windows checkout of the shipped examples). */
function fileDigest(file) {
  const bytes = readFileSync(file);
  const raw = sha256(bytes);
  let lf = null;
  if (bytes.includes(13)) {
    const out = Buffer.allocUnsafe(bytes.length);
    let k = 0;
    for (let i = 0; i < bytes.length; i += 1) if (!(bytes[i] === 13 && bytes[i + 1] === 10)) out[k++] = bytes[i];
    lf = sha256(out.subarray(0, k));
  }
  return { raw, lf, bytes: bytes.length };
}

function schemaOf(file) {
  const head = readFileSync(file).subarray(0, 4096).toString("latin1");
  return /FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'/i.exec(head)?.[1] ?? null;
}

function localPath(model, manifest) {
  const samples = option("--samples", process.env.BENCHMARK_SAMPLES || path.join(os.homedir(), "ifc-samples"));
  const roots = { samples, examples: path.join(ROOT, manifest.roots.examples) };
  return path.join(roots[model.root], ...model.path.split("/"));
}

/** Time fn over up to `runs` repeats, stopping once `budget` ms are spent. */
async function timed(fn, { runs, budget }) {
  const times = [];
  let out;
  let spent = 0;
  for (let r = 0; r < Math.max(1, runs); r += 1) {
    gc();
    const t = performance.now();
    out = await fn();
    const dt = performance.now() - t;
    times.push(dt);
    spent += dt;
    if (spent > budget) break;
  }
  return { out, ms: median(times), min: Math.min(...times), max: Math.max(...times), runs: times.length };
}

function machine() {
  const cpus = os.cpus();
  return {
    cpu: cpus[0]?.model?.trim() ?? "unknown",
    logicalCores: cpus.length,
    memoryGB: round(os.totalmem() / 2 ** 30, 1),
    os: `${os.type()} ${os.release()} (${os.arch()})`,
    node: process.version,
  };
}

// =============================================================== child

async function runModel(model, file, opts) {
  const { buildLatticeGraph, floorLevels, KIND_DOOR, KIND_FLOOR, KIND_STAIR } = await import(CORE + "lattice.js");
  const { astar, componentLabels, csr, edgeLengths, findRoute } = await import(CORE + "route.js");
  const { alternativeRoutes } = await import(CORE + "alternatives.js");
  const { straightenRoute } = await import(CORE + "straighten.js");
  const { buildForecast, doseFront, planWithForecast } = await import(CORE + "forecast.js");
  const { doseDeadline, doseDeadlineMap, escapeDeadlines } = await import(CORE + "deadline.js");
  const { compareForecasts, hopSurrogate } = await import(CORE + "surrogate.js");
  const { trainRl } = await import(CORE + "rl.js");
  const { extract } = await import("./ifc-extract.mjs");
  const { findExits, flaggedDoors, oneSidedDoors, roomOneSideDoors } = await import("./exits.mjs");
  const { FreePlane } = await import("./geodesic.mjs");
  const { levelSpread } = await import("./stress.mjs");

  // web-ifc reports geometry it cannot build on the console: count, do not print.
  let webIfcMessages = 0;
  for (const name of ["log", "warn", "error", "info"]) {
    const orig = console[name].bind(console);
    console[name] = (...a) => {
      if (typeof a[0] === "string" && a[0].startsWith("[WEB-IFC]")) webIfcMessages += 1;
      else orig(...a);
    };
  }

  const R = { runs: opts.runs, budget: opts.budget * 1000 };
  const out = { name: model.name, schema: model.schema, bytes: model.bytes, speed: {}, accuracy: {}, quality: {}, stairs: {}, robustness: {}, memory: {} };
  const memo = (label) => {
    out.memory[label] = { maxRssMB: Math.round(process.resourceUsage().maxRSS / 1024), heapMB: Math.round(process.memoryUsage().heapUsed / 2 ** 20) };
  };
  const seed = parseInt(model.sha256.slice(0, 8), 16);
  const random = rng(seed);
  const pick = (list) => list[Math.floor(random() * list.length)];
  memo("start");

  // ------------------------------------------------------------ extraction
  const geometryOnly = await timed(() => extract(file, { withDoorFlags: false }), R);
  out.speed.extractMs = round(geometryOnly.ms, 1);
  out.speed.extractRuns = geometryOnly.runs;
  const withFlags = await timed(() => extract(file, { withDoorFlags: true }), { runs: Math.min(3, R.runs), budget: R.budget });
  out.speed.extractWithFlagsMs = round(withFlags.ms, 1);
  const geo = withFlags.out;
  out.robustness.webIfcMessages = webIfcMessages;
  memo("extract");

  // --------------------------------------------------------------- lattice
  const build = () => buildLatticeGraph({ ...geo, upAxis: "y", spacing: APP.spacing, agentHeight: APP.agentHeight, maxEdgeRise: APP.maxEdgeRise, columnGap: APP.columnGap, maxPoints: APP.maxPoints });
  const lat = await timed(build, R);
  const built = lat.out;
  out.speed.latticeMs = round(lat.ms, 1);
  out.speed.latticeRuns = lat.runs;
  {
    const h = createHash("sha256");
    const bytes = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength);
    for (const a of [built.points, built.edges, built.kinds, built.blocked]) h.update(bytes(a));
    for (const s of built.sections || []) h.update(bytes(s.segs));
    out.graphHash = h.digest("hex").slice(0, 20); // as tools/graph-hash.mjs
  }
  memo("lattice");

  const g = { ...built, nodes: built.points, upAxis: "y", agentHeight: APP.agentHeight };
  const n = g.nodes.length / 3;
  const m = g.edges.length / 2;
  const P = g.nodes;
  let t0 = performance.now();
  csr(g);
  edgeLengths(g);
  const { labels, sizes } = componentLabels(g);
  out.speed.prepMs = round(performance.now() - t0, 2);
  const st = built.stats;
  out.graph = {
    nodes: n,
    edges: m,
    floorNodes: 0,
    stairNodes: st.stairNodes,
    doorNodes: st.doorNodes,
    wallFaceNodes: st.wallFaceNodes,
    blockedEdges: st.blockedEdges,
    prunedNodes: st.prunedNodes,
    spacing: round(st.spacing, 3),
    coarsened: st.coarsened,
    windingFallback: st.windingFallback,
    sections: (built.sections || []).length,
    sectionSegments: (built.sections || []).reduce((s, x) => s + x.segs.length / 4, 0),
  };
  for (let v = 0; v < n; v += 1) if (g.kinds[v] === KIND_FLOOR) out.graph.floorNodes += 1;

  // Inputs and what the file lacks.
  const tri = (list) => (list || []).reduce((s, x) => s + (x.indices?.length || 0) / 3, 0);
  const flags = flaggedDoors(geo);
  out.input = {
    floorMeshes: geo.floors.length,
    floorTriangles: tri(geo.floors),
    stairMeshes: geo.stairs.length,
    doorMeshes: geo.doors.length,
    doors: built.doorPoints.length,
    wallMeshes: geo.walls.length,
    spaceMeshes: geo.spaces.length,
    obstacleMeshes: geo.obstacles.length,
    flaggedDoors: flags.ids.size,
    flagSource: flags.source,
    storeys: geo.semantics?.storeys?.length ?? null,
    levels: floorLevels(geo.floors, "y").map((x) => round(x, 2)),
  };

  // ---------------------------------------------------------------- stairs
  {
    const levels = floorLevels(geo.floors, "y");
    const level = (y) => {
      let best = -1, gap = Infinity;
      levels.forEach((l, i) => {
        const d = Math.abs(y - APP.agentHeight - l);
        if (d < gap) {
          gap = d;
          best = i;
        }
      });
      return gap <= 0.6 ? best : -1;
    };
    const { indptr, neighbours, edgeIds } = csr(g);
    const sc = new Int32Array(n).fill(-1);
    const cases = [];
    for (let s = 0; s < n; s += 1) {
      if (g.kinds[s] !== KIND_STAIR || sc[s] >= 0) continue;
      const members = [s];
      sc[s] = cases.length;
      for (let i = 0; i < members.length; i += 1) {
        for (let k = indptr[members[i]]; k < indptr[members[i] + 1]; k += 1) {
          const w = neighbours[k];
          if (g.blocked[edgeIds[k]] || g.kinds[w] !== KIND_STAIR || sc[w] >= 0) continue;
          sc[w] = cases.length;
          members.push(w);
        }
      }
      const touched = new Set(), heights = [];
      for (const v of members) {
        for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
          const w = neighbours[k];
          if (g.blocked[edgeIds[k]] || g.kinds[w] === KIND_STAIR) continue;
          touched.add(level(P[w * 3 + 1]));
          heights.push(P[w * 3 + 1]);
        }
      }
      touched.delete(-1);
      cases.push({ nodes: members.length, floorLinks: heights.length, levels: touched.size, rise: heights.length ? Math.max(...heights) - Math.min(...heights) : 0 });
    }
    out.stairs = {
      staircases: cases.length,
      ends: st.stairEnds ?? null,
      endsLinked: st.stairEndsLinked ?? null,
      joinTwoLevels: cases.filter((c) => c.levels >= 2).length,
      // Without slab levels (floors only as rooms) the levels test cannot
      // apply; a staircase whose two floor links differ by 2 m joins storeys.
      joinTwoByRise: cases.filter((c) => c.floorLinks >= 2 && c.rise >= 2).length,
      unlinked: cases.filter((c) => c.floorLinks === 0).length,
      // Vertical cores (src/core/cores.js): stairs assumed from stair rooms
      // or parts, and lifts (off, as in the app).
      cores: st.cores ?? null,
      coreStairsAssumed: st.coreStairsAssumed ?? null,
    };
    // Storeys joined, as tools/core-check.mjs reports it: the levels the
    // best-joined part spans, and the share of floor in multi-level parts.
    const spread = levelSpread(built, levels);
    out.stairs.levels = levels.length;
    out.stairs.levelsJoined = spread.levels;
    out.stairs.multiLevelFloorPct = spread.multiPct;
  }

  // ----------------------------------------------------------------- exits
  const found = findExits(geo, built);
  const exits = found.exits;
  out.exits = { count: exits.length, source: found.source, flagged: found.flagged };
  {
    // Rules 2 and 3 scored against rule 1 at the level of graph doors.
    const doors = built.doorReport.map((d, i) => ({ ...d, i })).filter((d) => d.node >= 0);
    const truth = new Set(doors.filter((d) => flags.ids.has(d.itemId)).map((d) => d.i));
    const score = (chosen) => {
      const ids = new Set(chosen.map((d) => built.doorReport.indexOf(d)));
      let tp = 0;
      for (const i of ids) if (truth.has(i)) tp += 1;
      return { chosen: ids.size, truePositives: tp, precision: ids.size ? round(tp / ids.size, 3) : null, recall: truth.size ? round(tp / truth.size, 3) : null };
    };
    out.accuracy.exits = {
      graphDoors: doors.length,
      flagged: truth.size,
      oneSided: score(oneSidedDoors(built)),
      roomOneSide: score(roomOneSideDoors(geo, built)),
    };
  }

  // ------------------------------------------------------- graph quality
  const minPart = Math.max(20, Math.floor(0.01 * n));
  {
    const exitParts = new Set(exits.map((x) => labels[x]));
    let reach = 0;
    if (exits.length) {
      const free = escapeDeadlines(g, { exits, speed: APP.walkSpeed, blocked: g.blocked });
      for (const x of free) if (x !== -Infinity) reach += 1;
    }
    const area = walkableArea(geo, { KIND_FLOOR });
    out.quality = {
      parts: sizes.length,
      partsOver20: sizes.filter((s) => s >= 20).length,
      largestShare: n ? round(Math.max(...sizes) / n, 3) : null,
      reachableFromExits: n && exits.length ? round(reach / n, 3) : null,
      partsWithExit: exitParts.size,
      floorAreaM2: round(area, 1),
      floorNodesPerM2: area > 0 ? round(out.graph.floorNodes / area, 3) : null,
      // Share of the walkable plan area the floor nodes stand for, one
      // lattice cell (spacing squared) each, wall-face nodes excluded.
      latticeCoverage: area > 0 ? round(((out.graph.floorNodes - st.wallFaceNodes) * st.spacing ** 2) / area, 3) : null,
    };
  }
  memo("quality");

  // -------------------------------------------------------------- routes
  const exitSet = new Set(exits);
  const pairs = [];
  if (n) {
    const exitByPart = new Map();
    for (const x of exits) if (sizes[labels[x]] >= minPart) (exitByPart.get(labels[x]) || exitByPart.set(labels[x], []).get(labels[x])).push(x);
    const starts = [];
    for (let v = 0; v < n; v += 1) {
      if (g.kinds[v] !== KIND_FLOOR || sizes[labels[v]] < minPart) continue;
      if (exitByPart.size && !exitByPart.has(labels[v])) continue;
      starts.push(v);
    }
    for (let attempt = 0; attempt < 50 * opts.pairs && pairs.length < opts.pairs && starts.length; attempt += 1) {
      const s = pick(starts);
      let t;
      if (exitByPart.size) t = pick(exitByPart.get(labels[s]));
      else {
        t = pick(starts);
        if (labels[t] !== labels[s]) continue;
      }
      if (s !== t && !exitSet.has(s)) pairs.push([s, t]);
    }
  }
  out.pairs = { routes: pairs.length, kind: exits.length ? "start to exit" : "random pairs (no exits)" };
  const point = (v) => [P[v * 3], P[v * 3 + 1], P[v * 3 + 2]];
  const routes = [];
  if (pairs.length) {
    const astarMs = [], routeMs = [];
    const per = { runs: R.runs, budget: R.budget / pairs.length };
    for (const [s, t] of pairs) {
      const a = await timed(() => astar(g, s, t), per);
      astarMs.push(a.ms);
      routes.push(a.out);
      routeMs.push((await timed(() => findRoute(g, point(s), point(t)), per)).ms);
    }
    out.speed.astarMs = round(median(astarMs), 3);
    out.speed.astarMaxMs = round(max(astarMs), 3);
    out.speed.routeMs = round(median(routeMs), 3);
    out.routeLengths = { medianM: round(median(routes.map((r) => r?.cost)), 1), maxM: round(max(routes.map((r) => r?.cost)), 1) };

    // Alternatives on the first few pairs: Yen (k = 3) and the app's default, distinct.
    for (const method of ["yen", "distinct"]) {
      const ms = [], found2 = [];
      let budgetHits = 0;
      for (const [s, t] of pairs.slice(0, opts.altPairs)) {
        const r = await timed(() => alternativeRoutes(g, point(s), point(t), { k: APP.altCount, method }), { runs: R.runs, budget: R.budget / opts.altPairs });
        ms.push(r.ms);
        found2.push(r.out.routes.length);
        if (/time budget/.test(r.out.note || "")) budgetHits += 1;
      }
      out.speed[`${method}Ms`] = round(median(ms), 2);
      out.speed[`${method}MaxMs`] = round(max(ms), 2);
      out[`${method}Routes`] = { found: found2, budgetHits };
    }

    // Straightening: the per-graph lookups once (cold), then each route warm.
    const first = routes.find((r) => r && r.nodes.length > 2);
    if (first) {
      delete g._straight;
      t0 = performance.now();
      straightenRoute(g, first.nodes);
      const cold = performance.now() - t0;
      const sms = [];
      for (const r of routes) if (r && r.nodes.length > 2) sms.push((await timed(() => straightenRoute(g, r.nodes), { runs: R.runs, budget: R.budget / routes.length })).ms);
      out.speed.straightenMs = round(median(sms), 3);
      out.speed.straightenMaxMs = round(max(sms), 3);
      out.speed.straightenColdMs = round(cold, 2);
    }
  }
  memo("routes");

  // ---------------------------------------------- forecasts and deadlines
  if (n) {
    const big = sizes.indexOf(Math.max(...sizes));
    const members = [];
    for (let v = 0; v < n; v += 1) if (labels[v] === big) members.push(v);
    const fireSeed = members[Math.floor(members.length / 2)];
    out.fireSeed = fireSeed;
    const fOpts = { seed: fireSeed, steps: APP.fireSteps, dt: APP.secondsPerStep, threshold: APP.untenableAbove, blocked: g.blocked };
    out.forecast = {};
    let temperature = null;
    for (const model2 of ["temperature", "flood"]) {
      const row = {};
      out.forecast[model2] = row;
      const f = await timed(() => buildForecast(g, { model: model2, ...fOpts }), R);
      const forecast = f.out;
      if (model2 === "temperature") temperature = forecast;
      row.forecastMs = round(f.ms, 2);
      row.coarsened = forecast.stride > 1 ? forecast.frameSeconds : null;
      let closing = 0;
      for (let v = 0; v < n; v += 1) if (forecast.untenable[v] < Infinity) closing += 1;
      row.closingShare = round(closing / n, 3);
      if (!exits.length) {
        row.note = "no exits";
        continue;
      }
      const opts2 = { exits, forecast, speed: APP.walkSpeed, blocked: g.blocked };
      const ed = await timed(() => escapeDeadlines(g, { exits, untenable: forecast.closes, speed: APP.walkSpeed, blocked: g.blocked }), R);
      row.escapeDeadlinesMs = round(ed.ms, 2);
      const hard = ed.out;
      const can = [];
      for (let v = 0; v < n; v += 1) if (hard[v] > 0) can.push(v);
      row.canLeaveShare = round(can.length / n, 3);
      const starts = can.length ? Array.from({ length: opts.starts }, (_, i) => can[Math.floor(((i + 0.5) * can.length) / opts.starts)]) : [];
      const planMs = [], deadlineMs = [];
      let agree = 0, truncated = 0, maxLabels = 0, tradeoffs = 0;
      for (const s of starts) {
        const p = await timed(() => planWithForecast(g, { start: s, ...opts2 }), { runs: R.runs, budget: R.budget / starts.length });
        planMs.push(p.ms);
        const plan = p.out;
        maxLabels = Math.max(maxLabels, plan.labels);
        if (plan.truncated) truncated += 1;
        if (plan.front.length > 1) tradeoffs += 1;
        // Earliest arrival (time-dependent Dijkstra) against the dose front's first point.
        if (plan.earliest && plan.fastest && Math.abs(plan.earliest.arrival - plan.fastest.arrival) <= 1e-9 * Math.max(1, plan.earliest.arrival)) agree += 1;
        else if (!plan.earliest && !plan.fastest) agree += 1;
        deadlineMs.push((await timed(() => doseDeadline(g, s, { ...opts2, limit: APP.doseLimit, tolerance: 0.1 }), { runs: R.runs, budget: R.budget / starts.length })).ms);
      }
      row.planMs = round(median(planMs), 2);
      row.planMaxMs = round(max(planMs), 2);
      row.doseDeadlineMs = round(median(deadlineMs), 2);
      row.doseDeadlineMaxMs = round(max(deadlineMs), 2);
      out.accuracy[`front_${model2}`] = { starts: starts.length, earliestEqualsFrontHead: agree, truncated, maxLabels, tradeoffs };

      const mp = await timed(() => doseDeadlineMap(g, { ...opts2, limit: APP.doseLimit }), R);
      const map = mp.out;
      let slots = 0;
      const { neighbours, edgeIds } = csr(g);
      for (let k = 0; k < neighbours.length; k += 1) if (!g.blocked[edgeIds[k]]) slots += 1;
      row.doseDeadlineMapMs = round(mp.ms, 1);
      row.mapStepS = round(map.step, 4);
      row.mapLayers = map.layers;
      row.mapWork = Math.round((forecast.horizon / map.step) * (slots + n));
      memo(`map_${model2}`);

      // The map against the exact deadline (bisection on the exact least-dose search).
      const sample = can.length ? Array.from({ length: opts.mapSamples }, (_, i) => can[Math.floor(((i + 0.37) * can.length) / opts.mapSamples) % can.length]) : [];
      const gaps = [];
      let checked = 0, works = 0, notLater = 0, noMapDeparture = 0;
      for (const v of sample) {
        if (map.deadline[v] === -Infinity) {
          noMapDeparture += 1;
          continue;
        }
        const exact = doseDeadline(g, v, { ...opts2, limit: APP.doseLimit, tolerance: 0.02 });
        checked += 1;
        const { front } = doseFront(g, v, { ...opts2, depart: map.deadline[v] });
        if (front.length && front[front.length - 1].dose <= APP.doseLimit) works += 1;
        if (exact && map.deadline[v] <= exact.bad + 1e-9) {
          notLater += 1;
          gaps.push(exact.good - map.deadline[v]);
        }
      }
      out.accuracy[`map_${model2}`] = {
        checked,
        works,
        notLater,
        noMapDeparture,
        gapMedianS: round(median(gaps), 2),
        gapMaxS: round(max(gaps), 2),
        gapMedianSteps: round(median(gaps) / map.step, 2),
      };
    }

    // The hop-front surrogate against the temperature forecast.
    if (temperature) {
      const sg = await timed(() => hopSurrogate(g, fOpts), R);
      const sur = sg.out;
      const cmp = compareForecasts(temperature, sur);
      const earlier = [];
      for (let v = 0; v < n; v += 1) if (temperature.untenable[v] < Infinity) earlier.push(temperature.closes[v] - sur.closes[v]);
      const row = {
        surrogateMs: round(sg.ms, 2),
        fullMs: out.forecast.temperature.forecastMs,
        conservative: cmp.conservative,
        lateNodes: cmp.lateNodes,
        lowRates: cmp.lowRates,
        closingNodes: earlier.length,
        earlierMedianS: round(median(earlier), 1),
        earlierMaxS: round(max(earlier), 1),
        surrogateOnlyClosing: 0,
      };
      for (let v = 0; v < n; v += 1) if (temperature.untenable[v] === Infinity && sur.untenable[v] < Infinity) row.surrogateOnlyClosing += 1;
      if (exits.length) {
        const full = escapeDeadlines(g, { exits, untenable: temperature.closes, speed: APP.walkSpeed, blocked: g.blocked });
        const cheap = escapeDeadlines(g, { exits, untenable: sur.closes, speed: APP.walkSpeed, blocked: g.blocked });
        const shift = [];
        let later = 0;
        for (let v = 0; v < n; v += 1) {
          if (!Number.isFinite(full[v]) || !Number.isFinite(cheap[v])) continue;
          shift.push(full[v] - cheap[v]);
          if (cheap[v] > full[v] + 1e-9) later += 1;
        }
        row.deadlineEarlierMedianS = round(median(shift), 1);
        row.deadlineEarlierMaxS = round(max(shift), 1);
        row.deadlineLater = later;
      }
      out.accuracy.surrogate = row;
    }
    memo("forecast");

    // -------------------------------------------------------------- RL
    const rlPairs = pairs.slice(0, opts.rlPairs);
    if (rlPairs.length) {
      const ms = [], fixed = [], rows = [];
      for (let i = 0; i < rlPairs.length; i += 1) {
        const [s, t] = rlPairs[i];
        const args = { startPoint: point(s), exitPoint: point(t), firePoint: point(fireSeed), episodes: APP.rlEpisodes, maxSteps: APP.rlMaxSteps, useFire: true, fireSpeed: APP.rlFireSpeed, useWalls: true, seed: 1 + i };
        const r = await timed(() => trainRl(g, args), { runs: R.runs, budget: R.budget / rlPairs.length });
        ms.push(r.ms);
        // One episode: the fixed cost (fire timeline, shaping, the exact
        // optimum's backward pass, the escape search).
        fixed.push((await timed(() => trainRl(g, { ...args, episodes: 1 }), { runs: R.runs, budget: R.budget / rlPairs.length })).ms);
        rows.push({ reached: r.out.reached_exit, successRate: round(r.out.success_rate, 3), policyReturn: round(r.out.policy_return, 3), optimalReturn: round(r.out.optimal_return, 3), escapeSteps: r.out.escape_steps });
      }
      out.speed.rlMs = round(median(ms), 1);
      out.speed.rlMaxMs = round(max(ms), 1);
      out.speed.rlOneEpisodeMs = round(median(fixed), 1);
      out.rl = rows;
    }
    memo("rl");
  }

  // ------------------------------------------- route length vs geodesic
  if (n && (built.sections || []).length) {
    const sectionKeys = new Map((built.sections || []).map((s) => [s.key, s.segs]));
    const planes = new Map();
    const planeFor = (key) => {
      if (!planes.has(key)) planes.set(key, sectionKeys.has(key) ? new FreePlane(sectionKeys.get(key)) : null);
      return planes.get(key);
    };
    // The section a floor node's links were blocked with (lattice.js sectionFor).
    const keyOf = (v) => {
      const k = Math.round((P[v * 3 + 1] - APP.agentHeight + built.sectionHeight) * 10);
      for (const c of [k, k - 1, k + 1]) if (sectionKeys.has(c)) return c;
      return null;
    };
    const byKey = new Map();
    for (let v = 0; v < n; v += 1) {
      if (g.kinds[v] !== KIND_FLOOR || sizes[labels[v]] < minPart) continue;
      const k = keyOf(v);
      if (k === null) continue;
      if (!byKey.has(k)) byKey.set(k, []);
      byKey.get(k).push(v);
    }
    const keys = [...byKey.keys()];
    const plan = (v) => [P[v * 3], P[v * 3 + 2]];
    const planLength = (nodes) => {
      let s = 0;
      for (let i = 1; i < nodes.length; i += 1) s += Math.hypot(P[nodes[i] * 3] - P[nodes[i - 1] * 3], P[nodes[i] * 3 + 2] - P[nodes[i - 1] * 3 + 2]);
      return s;
    };
    const rows = [];
    let leftFloor = 0, attempts = 0;
    t0 = performance.now();
    while (rows.length < opts.geoPairs && attempts < 60 * opts.geoPairs && keys.length && performance.now() - t0 < opts.geoBudget * 1000) {
      attempts += 1;
      const key = pick(keys);
      const list = byKey.get(key);
      const a = pick(list), b = pick(list);
      if (a === b || labels[a] !== labels[b]) continue;
      const [ax, ay] = plan(a), [bx, by] = plan(b);
      if (Math.hypot(bx - ax, by - ay) < 2) continue;
      const route = astar(g, a, b);
      if (!route) continue;
      if (route.nodes.some((v) => g.kinds[v] === KIND_STAIR || Math.abs((keyOf(v) ?? -1e9) - key) > 1)) {
        leftFloor += 1;
        continue;
      }
      const plane = planeFor(key);
      if (!plane) continue;
      const gridLen = planLength(route.nodes);
      const straight = straightenRoute(g, route.nodes);
      const straightLen = planLength(straight.node_ids);
      const tg = performance.now();
      let geo2 = plane.geodesic(ax, ay, bx, by, { bound: gridLen });
      let violation = false;
      if (geo2.status === "beyond bound") {
        // The grid route is shorter than every free path: find the true
        // geodesic (if any) to measure by how much.
        violation = true;
        geo2 = plane.geodesic(ax, ay, bx, by, { bound: Infinity });
      }
      const row = {
        floorKey: key,
        euclid: round(Math.hypot(bx - ax, by - ay), 3),
        geodesic: Number.isFinite(geo2.length) ? round(geo2.length, 4) : geo2.length === Infinity ? "none" : null,
        status: geo2.status,
        grid: round(gridLen, 4),
        straight: round(straightLen, 4),
        geodesicMs: round(performance.now() - tg, 1),
        checks: geo2.checks,
      };
      if (Number.isFinite(geo2.length) && geo2.length > 0) {
        row.stretchGrid = round(gridLen / geo2.length, 5);
        row.stretchStraight = round(straightLen / geo2.length, 5);
      }
      // Where a route is shorter than the geodesic, which of its links are not free.
      const diagnose = (nodes) => {
        const causes = { doorLinks: 0, latticeLinks: 0 };
        for (let i = 1; i < nodes.length; i += 1) {
          const [px, py] = plan(nodes[i - 1]), [qx, qy] = plan(nodes[i]);
          if (plane.visible(px, py, qx, qy)) continue;
          if (g.kinds[nodes[i - 1]] === KIND_DOOR || g.kinds[nodes[i]] === KIND_DOOR) causes.doorLinks += 1;
          else causes.latticeLinks += 1;
        }
        return causes;
      };
      const short = (len) => Number.isFinite(geo2.length) ? geo2.length - len : geo2.length === Infinity ? Infinity : NaN;
      if (violation) row.gridViolation = { shortBy: round(short(gridLen), 6), ...diagnose(route.nodes) };
      if (Number.isFinite(geo2.length) ? straightLen < geo2.length * (1 - 1e-9) - 1e-9 : geo2.length === Infinity) {
        row.straightViolation = { shortBy: round(short(straightLen), 6), ...diagnose(straight.node_ids) };
      }
      rows.push(row);
    }
    out.accuracy.geodesic = { pairs: rows, leftFloor, attempts, floors: keys.length, ms: Math.round(performance.now() - t0) };
  }
  memo("geodesic");

  // ------------------------------------------------------------ robustness
  {
    const lacks = [];
    const does = [];
    if (!geo.floors.length) lacks.push("floor slabs");
    if (!geo.spaces.length) lacks.push("spaces");
    if (!geo.walls.length) lacks.push("walls");
    if (!geo.doors.length) lacks.push("doors");
    if (!geo.stairs.length) lacks.push("stairs");
    if (geo.doors.length && !flags.ids.size) lacks.push("exit flags");
    if (!n) does.push("no walkable graph: every routing feature is unavailable");
    else {
      if (!geo.floors.length && geo.spaces.length) does.push("floors taken from IfcSpace bottoms");
      if (st.windingFallback) does.push("winding untrusted: faces of either orientation sampled");
      if (st.coarsened) does.push(`grid coarsened to ${round(st.spacing, 2)} m to fit 40,000 points`);
      if (!geo.doors.length) does.push("no door nodes: rooms join only through openings in the wall geometry");
      if (exits.length) does.push(`exits from ${found.source}`);
      else does.push("no exits found: forecast planning, deadlines and the exit metrics are skipped");
      if (out.quality.partsOver20 > 1) does.push(`${out.quality.partsOver20} parts of 20+ nodes, largest ${Math.round(out.quality.largestShare * 100)}%`);
      if (out.stairs.staircases && out.stairs.endsLinked < out.stairs.ends) does.push(`${out.stairs.ends - out.stairs.endsLinked} of ${out.stairs.ends} stair ends not linked to a floor`);
      if (geo.stairs.length && !out.stairs.staircases) does.push("stairs in the file but no staircase in the graph");
    }
    out.robustness = { ...out.robustness, lacks, does };
  }
  out.memory.peakRssMB = Math.round(process.resourceUsage().maxRSS / 1024);
  return out;
}

/**
 * Walkable plan area in m2, rasterised at 0.25 m: upward floor faces (either
 * orientation if the model has none facing up, as lattice.js does), room
 * bottoms, roofs excluded by the lattice's rule (within a metre of the model
 * top when floor lies at least 2 m lower); one cell per column and storey,
 * storeys split where heights in a column differ by more than 0.9 m.
 */
function walkableArea(geo) {
  const cell = 0.25;
  const up = 1;
  let floorLow = Infinity, modelTop = -Infinity;
  for (const group of [geo.floors, geo.stairs, geo.spaces]) for (const m of group) for (let i = up; i < m.positions.length; i += 3) floorLow = Math.min(floorLow, m.positions[i]);
  for (const group of [geo.floors, geo.stairs, geo.walls, geo.doors]) for (const m of group) for (let i = up; i < m.positions.length; i += 3) modelTop = Math.max(modelTop, m.positions[i]);
  const excludeAbove = Number.isFinite(modelTop) && modelTop - floorLow >= 2 ? modelTop - 1 : null;
  const columns = new Map();
  const minUp = Math.cos((10 * Math.PI) / 180);
  const raster = (meshes, accept) => {
    let added = 0;
    for (const { positions: p, indices: ix } of meshes) {
      let bottom = Infinity;
      for (let o = up; o < p.length; o += 3) bottom = Math.min(bottom, p[o]);
      for (let t = 0; t + 2 < ix.length; t += 3) {
        const a = ix[t] * 3, b = ix[t + 1] * 3, c = ix[t + 2] * 3;
        const ah = p[a + 1], bh = p[b + 1], ch = p[c + 1];
        if (excludeAbove !== null && ah >= excludeAbove && bh >= excludeAbove && ch >= excludeAbove) continue;
        const e1x = p[b] - p[a], e1y = p[b + 1] - p[a + 1], e1z = p[b + 2] - p[a + 2];
        const e2x = p[c] - p[a], e2y = p[c + 1] - p[a + 1], e2z = p[c + 2] - p[a + 2];
        const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
        const mag = Math.hypot(nx, ny, nz);
        if (mag < 1e-12 || !accept(ny / mag, Math.max(ah, bh, ch), bottom)) continue;
        const ax = p[a] / cell, az = p[a + 2] / cell, bx = p[b] / cell, bz = p[b + 2] / cell, cx = p[c] / cell, cz = p[c + 2] / cell;
        const det = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz);
        if (Math.abs(det) < 1e-12) continue;
        for (let i = Math.floor(Math.min(ax, bx, cx)); i <= Math.ceil(Math.max(ax, bx, cx)); i += 1) {
          for (let j = Math.floor(Math.min(az, bz, cz)); j <= Math.ceil(Math.max(az, bz, cz)); j += 1) {
            const x = i + 0.5, z = j + 0.5;
            const w0 = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / det;
            const w1 = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / det;
            const w2 = 1 - w0 - w1;
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            const k = (i + 2 ** 20) * 2 ** 21 + (j + 2 ** 20);
            const h = w0 * ah + w1 * bh + w2 * ch;
            const list = columns.get(k);
            if (list) list.push(h);
            else columns.set(k, [h]);
            added += 1;
          }
        }
      }
    }
    return added;
  };
  raster(geo.spaces, (nUp, top, bottom) => Math.abs(nUp) >= minUp && top <= bottom + 0.05);
  if (!raster(geo.floors, (nUp) => nUp >= minUp) && geo.floors.length) raster(geo.floors, (nUp) => Math.abs(nUp) >= minUp);
  let cells = 0;
  for (const list of columns.values()) {
    list.sort((x, y) => x - y);
    cells += 1;
    for (let i = 1; i < list.length; i += 1) if (list[i] - list[i - 1] > 0.9) cells += 1;
  }
  return cells * cell * cell;
}

// ============================================================== summary

const fmt = (x, d = 0) => (x === null || x === undefined ? "-" : typeof x === "number" ? (Number.isFinite(x) ? x.toFixed(d) : String(x)) : String(x));

/** Markdown tables from a results file. */
export function summarize(results) {
  const ok = results.models.filter((r) => !r.error);
  const lines = [];
  const table = (head, rows) => {
    lines.push(`| ${head.join(" | ")} |`, `|${head.map(() => "---").join("|")}|`);
    for (const r of rows) lines.push(`| ${r.join(" | ")} |`);
    lines.push("");
  };
  const speedOf = {
    "IFC extraction (geometry)": (r) => r.speed.extractMs,
    "IFC extraction with door flags": (r) => r.speed.extractWithFlagsMs,
    "Lattice build": (r) => r.speed.latticeMs,
    "A* route": (r) => r.speed.astarMs,
    "Route (snap and A*)": (r) => r.speed.routeMs,
    "Yen alternatives (k = 3)": (r) => r.speed.yenMs,
    "Distinct alternatives (k = 3)": (r) => r.speed.distinctMs,
    "Straightening (warm)": (r) => r.speed.straightenMs,
    "Straightening lookups (once per graph)": (r) => r.speed.straightenColdMs,
    "Forecast, temperature": (r) => r.forecast?.temperature?.forecastMs,
    "Forecast, flood": (r) => r.forecast?.flood?.forecastMs,
    "Hop-front surrogate": (r) => r.accuracy.surrogate?.surrogateMs,
    "planWithForecast (temperature)": (r) => r.forecast?.temperature?.planMs,
    "planWithForecast (flood)": (r) => r.forecast?.flood?.planMs,
    "escapeDeadlines (temperature)": (r) => r.forecast?.temperature?.escapeDeadlinesMs,
    "doseDeadline (temperature)": (r) => r.forecast?.temperature?.doseDeadlineMs,
    "doseDeadline (flood)": (r) => r.forecast?.flood?.doseDeadlineMs,
    "doseDeadlineMap (temperature)": (r) => r.forecast?.temperature?.doseDeadlineMapMs,
    "doseDeadlineMap (flood)": (r) => r.forecast?.flood?.doseDeadlineMapMs,
    "RL training, app defaults": (r) => r.speed.rlMs,
    "RL, one episode (fixed cost)": (r) => r.speed.rlOneEpisodeMs,
  };
  lines.push("### Speed per computation (ms; median of each model's median, and the slowest model)", "");
  table(
    ["Computation", "Models", "Median", "Worst", "Worst model (nodes)"],
    Object.entries(speedOf).map(([label, f]) => {
      const vals = ok.map((r) => [f(r), r]).filter(([v]) => Number.isFinite(v));
      if (!vals.length) return [label, "0", "-", "-", "-"];
      const worst = vals.reduce((a, b) => (b[0] > a[0] ? b : a));
      return [label, String(vals.length), fmt(median(vals.map((v) => v[0])), vals.length && median(vals.map((v) => v[0])) < 10 ? 2 : 0), fmt(worst[0], worst[0] < 10 ? 2 : 0), `${worst[1].name} (${worst[1].graph?.nodes ?? "-"})`];
    }),
  );

  lines.push("### Speed per model (ms)", "");
  table(
    ["Model", "Nodes", "Extract", "Lattice", "A*", "Yen", "Straighten", "Forecast T", "Plan T", "Deadline T", "Map T (step s)", "RL", "Peak RSS MB"],
    ok.map((r) => [
      r.name,
      fmt(r.graph?.nodes),
      fmt(r.speed.extractMs),
      fmt(r.speed.latticeMs),
      fmt(r.speed.astarMs, 2),
      fmt(r.speed.yenMs),
      fmt(r.speed.straightenMs, 2),
      fmt(r.forecast?.temperature?.forecastMs),
      fmt(r.forecast?.temperature?.planMs, 1),
      fmt(r.forecast?.temperature?.doseDeadlineMs),
      r.forecast?.temperature?.doseDeadlineMapMs !== undefined ? `${fmt(r.forecast.temperature.doseDeadlineMapMs)} (${fmt(r.forecast.temperature.mapStepS, 3)})` : "-",
      fmt(r.speed.rlMs),
      fmt(r.memory?.peakRssMB),
    ]),
  );

  // Route length against the geodesic.
  const geoRows = ok.flatMap((r) => (r.accuracy.geodesic?.pairs || []).map((p) => ({ ...p, model: r.name })));
  const measured = geoRows.filter((p) => Number.isFinite(p.stretchGrid));
  const dist = (xs) => `${fmt(quantile(xs, 0.5), 4)} / ${fmt(quantile(xs, 0.9), 4)} / ${fmt(max(xs), 4)}`;
  lines.push("### Route length against the geodesic (plan lengths, same floor and part)", "");
  table(
    ["Measure", "Value"],
    [
      ["Pairs measured (models)", `${measured.length} (${new Set(measured.map((p) => p.model)).size})`],
      ["Pairs in sight of each other (geodesic straight)", String(measured.filter((p) => p.status === "direct").length)],
      ["Pairs skipped: route leaves the floor", String(ok.reduce((s, r) => s + (r.accuracy.geodesic?.leftFloor || 0), 0))],
      ["Pairs skipped: search budget", String(geoRows.filter((p) => p.status === "capped").length)],
      ["Grid route stretch, median / p90 / max", dist(measured.map((p) => p.stretchGrid))],
      ["Straightened stretch, median / p90 / max", dist(measured.map((p) => p.stretchStraight))],
      [`Grid stretch above ${OCTILE}, all pairs`, String(measured.filter((p) => p.stretchGrid > OCTILE).length)],
      [`Grid stretch above ${OCTILE}, pairs in sight`, String(measured.filter((p) => p.status === "direct" && p.stretchGrid > OCTILE).length)],
      ["Grid routes shorter than the geodesic", String(geoRows.filter((p) => p.gridViolation).length)],
      ["  of which through a door link that is not free", String(geoRows.filter((p) => p.gridViolation?.doorLinks).length)],
      ["  of which through a lattice link that clips solid", String(geoRows.filter((p) => p.gridViolation?.latticeLinks).length)],
      ["  of which by under 1e-5 m with every link free (rounding)", String(geoRows.filter((p) => p.gridViolation && !p.gridViolation.doorLinks && !p.gridViolation.latticeLinks && p.gridViolation.shortBy < 1e-5).length)],
      ["Straightened routes shorter than the geodesic", String(geoRows.filter((p) => p.straightViolation).length)],
    ],
  );
  lines.push("Per model (pairs; grid stretch median and max; straightened median; violations):", "");
  table(
    ["Model", "Pairs", "Grid median", "Grid max", "Straight median", "Grid shorter", "Straight shorter"],
    ok.filter((r) => r.accuracy.geodesic).map((r) => {
      const ps = r.accuracy.geodesic.pairs.filter((p) => Number.isFinite(p.stretchGrid));
      return [r.name, String(ps.length), fmt(median(ps.map((p) => p.stretchGrid)), 3), fmt(max(ps.map((p) => p.stretchGrid)), 3), fmt(median(ps.map((p) => p.stretchStraight)), 3), String(r.accuracy.geodesic.pairs.filter((p) => p.gridViolation).length), String(r.accuracy.geodesic.pairs.filter((p) => p.straightViolation).length)];
    }),
  );

  // Exits.
  lines.push("### Exit rules against the IFC's own exit flags (graph doors)", "");
  const withFlags = ok.filter((r) => r.accuracy.exits?.flagged);
  const pool = (key) => {
    let tp = 0, chosen = 0, truth = 0;
    for (const r of withFlags) {
      tp += r.accuracy.exits[key].truePositives;
      chosen += r.accuracy.exits[key].chosen;
      truth += r.accuracy.exits.flagged;
    }
    return [chosen ? tp / chosen : null, truth ? tp / truth : null];
  };
  table(
    ["Model", "Graph doors", "Flagged", "One-sided P", "One-sided R", "Room-one-side P", "Room-one-side R"],
    [
      ...withFlags.map((r) => {
        const e = r.accuracy.exits;
        return [r.name, String(e.graphDoors), String(e.flagged), fmt(e.oneSided.precision, 2), fmt(e.oneSided.recall, 2), fmt(e.roomOneSide.precision, 2), fmt(e.roomOneSide.recall, 2)];
      }),
      ["Pooled", String(withFlags.reduce((s, r) => s + r.accuracy.exits.graphDoors, 0)), String(withFlags.reduce((s, r) => s + r.accuracy.exits.flagged, 0)), fmt(pool("oneSided")[0], 2), fmt(pool("oneSided")[1], 2), fmt(pool("roomOneSide")[0], 2), fmt(pool("roomOneSide")[1], 2)],
    ],
  );

  lines.push("### Graph quality and stairs", "");
  table(
    ["Model", "Nodes", "Parts (20+)", "Largest", "Reach exit", "Floor m2", "Floor nodes/m2", "Coverage", "Staircases", "Ends linked", "Join 2 levels", "Levels joined", "Multi-level floor"],
    ok.map((r) => [
      r.name,
      fmt(r.graph?.nodes),
      `${fmt(r.quality?.parts)} (${fmt(r.quality?.partsOver20)})`,
      fmt(r.quality?.largestShare, 2),
      fmt(r.quality?.reachableFromExits, 2),
      fmt(r.quality?.floorAreaM2),
      fmt(r.quality?.floorNodesPerM2, 2),
      fmt(r.quality?.latticeCoverage, 2),
      fmt(r.stairs?.staircases),
      r.stairs?.ends ? `${r.stairs.endsLinked}/${r.stairs.ends}` : "-",
      r.stairs?.staircases ? `${r.stairs.joinTwoLevels} (rise: ${r.stairs.joinTwoByRise})` : "-",
      r.stairs?.levels ? `${r.stairs.levelsJoined} of ${r.stairs.levels}` : "-",
      r.stairs?.levels ? `${fmt(r.stairs.multiLevelFloorPct, 1)}%` : "-",
    ]),
  );

  lines.push("### Forecast, deadlines and the surrogate", "");
  table(
    ["Model", "Exits (source)", "Earliest = front head T/F", "Map works T/F", "Map gap median / max s (T)", "Map gap median / max s (F)", "Surrogate conservative", "Closes earlier median / max s", "Deadline earlier median / max s"],
    ok.filter((r) => r.forecast).map((r) => {
      const a = r.accuracy;
      const fh = (k) => (a[`front_${k}`] ? `${a[`front_${k}`].earliestEqualsFrontHead}/${a[`front_${k}`].starts}` : "-");
      const mw = (k) => (a[`map_${k}`] ? `${a[`map_${k}`].works}/${a[`map_${k}`].checked}` : "-");
      const gap = (k) => (a[`map_${k}`] ? `${fmt(a[`map_${k}`].gapMedianS, 2)} / ${fmt(a[`map_${k}`].gapMaxS, 2)}` : "-");
      const s = a.surrogate;
      return [r.name, `${r.exits?.count ?? 0} (${r.exits?.source ?? "none"})`, `${fh("temperature")}, ${fh("flood")}`, `${mw("temperature")}, ${mw("flood")}`, gap("temperature"), gap("flood"), s ? (s.conservative ? "yes" : `NO (${s.lateNodes} late, ${s.lowRates} low)`) : "-", s ? `${fmt(s.earlierMedianS, 1)} / ${fmt(s.earlierMaxS, 1)}` : "-", s?.deadlineEarlierMedianS !== undefined ? `${fmt(s.deadlineEarlierMedianS, 1)} / ${fmt(s.deadlineEarlierMaxS, 1)}` : "-"];
    }),
  );

  lines.push("### Robustness: what each model lacks and what the pipeline then does", "");
  table(
    ["Model", "Schema", "Lacks", "Pipeline", "web-ifc messages"],
    results.models.map((r) => (r.error ? [r.name, r.schema ?? "-", "-", `error: ${r.error}`, "-"] : [r.name, r.schema ?? "-", r.robustness.lacks.join(", ") || "nothing checked for", r.robustness.does.join("; "), fmt(r.robustness.webIfcMessages)])),
  );
  return lines.join("\n");
}

// ================================================================ stress

/** The stress and regression suite (tools/stress.mjs), one process per case. */
async function stress(manifest) {
  const { coreAt, listCases, STRESS_TMP } = await import("./stress.mjs");
  const samples = option("--samples", process.env.BENCHMARK_SAMPLES || path.join(os.homedir(), "ifc-samples"));
  const tmp = path.join(STRESS_TMP(), String(process.pid));
  mkdirSync(tmp, { recursive: true });
  const env = { ...process.env };
  const rev = option("--core-rev", null);
  if (rev) {
    const git = (args) => execFileSync("git", args, { cwd: ROOT, encoding: "utf8", maxBuffer: 1 << 26 });
    env.STRESS_CORE_DIR = coreAt(rev, path.join(tmp, "core"), git);
  }
  const only = option("--only", null);
  const cases = listCases({ samples, manifest, tmp }).filter((c) => !only || new RegExp(only, "i").test(`${c.id} ${c.title}`));
  const date = new Date().toISOString().slice(0, 10);
  const outFile = path.resolve(option("--out", path.join(ROOT, "benchmark", "results", `${date}-stress.json`)));
  let commit = null;
  try {
    commit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" }).trim();
  } catch {
    // Not a git checkout.
  }
  const results = { suite: "stress", date, machine: machine(), commit, coreRev: rev, cases: [] };
  for (const [i, c] of cases.entries()) {
    if (c.skip) {
      results.cases.push({ ...c, status: "skipped", reason: c.skip });
      console.log(`${"SKIP".padEnd(6)} ${c.id}: ${c.skip}`);
      continue;
    }
    const jobFile = path.join(tmp, `case-${i}.json`), outJson = path.join(tmp, `case-${i}-out.json`);
    writeFileSync(jobFile, JSON.stringify({ case: c, out: outJson }));
    const t0 = performance.now();
    const child = spawnSync(process.execPath, ["--max-old-space-size=12288", fileURLToPath(import.meta.url), "--stress-case", jobFile], { env, stdio: ["ignore", "ignore", "pipe"], encoding: "utf8", timeout: c.timeoutMs, maxBuffer: 64 * 1024 * 1024 });
    const wall = Math.round(performance.now() - t0);
    let r;
    try {
      r = JSON.parse(readFileSync(outJson, "utf8"));
    } catch {
      const why = child.error?.code === "ETIMEDOUT" || child.signal ? `no result within ${c.timeoutMs / 1000} s (killed)` : `process exited ${child.status}: ${String(child.stderr || "").split("\n").filter(Boolean).slice(-2).join(" | ").slice(0, 300)}`;
      r = { pass: false, checks: [{ name: "case finished", ok: false, detail: why }] };
    }
    // xfail: the fix is on a branch not yet merged; open: a weakness this suite
    // found, outside this work's scope, documented in benchmark/README.md.
    const failedChecks = (r.checks || []).filter((k) => !k.ok);
    const openOnly = failedChecks.length > 0 && failedChecks.every((k) => k.open);
    const status = r.pass ? (c.expectFail || c.open ? "fixed?" : "pass") : c.expectFail ? "xfail" : c.open || openOnly ? "open" : "FAIL";
    results.cases.push({ id: c.id, group: c.group, title: c.title, cite: c.cite, expectFail: c.expectFail, open: c.open ?? (openOnly ? failedChecks[0].open : null), status, ms: r.ms ?? null, wallMs: wall, checks: r.checks, metrics: r.metrics });
    const failed = r.checks.filter((k) => !k.ok).map((k) => `${k.name}${k.detail ? ` (${k.detail})` : ""}`);
    console.log(`${status.padEnd(6)} ${c.id.slice(0, 70).padEnd(70)} ${String(wall).padStart(6)} ms${failed.length ? `  ${failed.join("; ").slice(0, 220)}` : ""}`);
  }
  const count = (s) => results.cases.filter((x) => x.status === s).length;
  results.totals = { cases: results.cases.length, pass: count("pass"), fail: count("FAIL"), open: count("open"), expectedFail: count("xfail"), unexpectedPass: count("fixed?"), skipped: count("skipped") };
  mkdirSync(path.dirname(outFile), { recursive: true });
  writeFileSync(outFile, JSON.stringify(results, null, 1));
  console.log(`\n${JSON.stringify(results.totals)}\nWrote ${path.relative(ROOT, outFile)}\n`);
  console.log(summarizeStress(results));
}

/** Markdown tables of a stress results file. */
export function summarizeStress(results) {
  const lines = [];
  const esc = (s) => String(s ?? "").replace(/\|/g, "/");
  for (const [group, head] of [["geometry", "Synthetic geometry"], ["regression", "Known-bug regressions"], ["ifc", "IFC files: fixtures, broken files, corpus"]]) {
    const rows = results.cases.filter((c) => c.group === group);
    if (!rows.length) continue;
    lines.push(`### ${head} (${rows.filter((c) => c.status === "pass").length} of ${rows.length} pass)`, "");
    lines.push(group === "ifc" ? "| Case | Status | Outcome | Extract ms | Graph ms | Nodes | Note |" : "| Case | Status | ms | Detail |", group === "ifc" ? "|---|---|---|---|---|---|---|" : "|---|---|---|---|");
    for (const c of rows) {
      const failed = (c.checks || []).filter((k) => !k.ok).map((k) => `${k.name}${k.detail ? `: ${k.detail}` : ""}`).join("; ");
      if (group === "ifc") {
        const m = c.metrics || {};
        const err = (c.checks || []).find((k) => /^error:/.test(k.detail || ""))?.detail;
        lines.push(`| ${esc(c.title)} | ${c.status} | ${esc(m.outcome ?? "-")} | ${m.extractMs ?? "-"} | ${m.latticeMs ?? "-"} | ${m.nodes ?? "-"} | ${esc((failed || err || "").slice(0, 160))} |`);
      } else {
        const detail = failed || (c.checks || []).map((k) => k.detail).filter(Boolean).join("; ");
        lines.push(`| ${esc(c.title)}${c.cite ? ` (${esc(c.cite)})` : ""} | ${c.status}${c.expectFail ? ` (${esc(c.expectFail)})` : c.open ? ` (${esc(c.open)})` : ""} | ${c.ms ?? c.wallMs ?? "-"} | ${esc(detail.slice(0, 200))} |`);
      }
    }
    lines.push("");
  }
  return lines.join("\n");
}

// ================================================================= main

async function main() {
  if (flag("--child")) {
    const job = JSON.parse(readFileSync(option("--child"), "utf8"));
    let result;
    try {
      result = await runModel(job.model, job.file, job.opts);
    } catch (error) {
      result = { name: job.model.name, error: String(error?.stack || error).slice(0, 2000) };
    }
    writeFileSync(job.out, JSON.stringify(result));
    return;
  }

  if (flag("--stress-case")) {
    const job = JSON.parse(readFileSync(option("--stress-case"), "utf8"));
    const { runCase } = await import("./stress.mjs");
    let result;
    try {
      result = await runCase(job.case);
    } catch (error) {
      result = { pass: false, checks: [{ name: "case ran", ok: false, detail: `threw: ${String(error?.stack || error).slice(0, 600)}` }] };
    }
    writeFileSync(job.out, JSON.stringify(result));
    return;
  }

  if (flag("--summary")) {
    const results = JSON.parse(readFileSync(option("--summary"), "utf8"));
    console.log(results.suite === "stress" ? summarizeStress(results) : summarize(results));
    return;
  }

  const manifest = JSON.parse(readFileSync(MANIFEST, "utf8"));
  if (flag("--stress")) return stress(manifest);
  const only = option("--only", null);
  // Stress-only models (large coordinates, broken placements) run in --stress.
  const models = manifest.models.filter((m) => !m.stressOnly && (!only || new RegExp(only, "i").test(m.name)));
  const quick = flag("--quick");
  const opts = {
    runs: Number(option("--runs", quick ? 1 : 5)),
    budget: Number(option("--budget", quick ? 5 : 20)),
    pairs: quick ? 8 : 20,
    altPairs: quick ? 2 : 5,
    rlPairs: quick ? 1 : 3,
    starts: 5,
    mapSamples: quick ? 6 : 12,
    geoPairs: quick ? 6 : 20,
    geoBudget: quick ? 60 : 240,
  };

  // Files: present and matching the manifest.
  const jobs = [];
  for (const model of models) {
    const file = localPath(model, manifest);
    if (!existsSync(file)) {
      console.log(`${model.name.padEnd(36)} missing (${file}); download it from ${model.download || model.source}`);
      jobs.push({ model, file, missing: true });
      continue;
    }
    const d = fileDigest(file);
    const match = d.raw === model.sha256 ? "exact" : d.lf === model.sha256 ? "CRLF folded" : null;
    if (!match) console.log(`${model.name.padEnd(36)} SHA-256 MISMATCH (${d.raw.slice(0, 12)}, expected ${model.sha256.slice(0, 12)}): skipped`);
    else if (flag("--verify")) console.log(`${model.name.padEnd(36)} ok (${match})`);
    jobs.push({ model, file, mismatch: !match });
  }
  if (flag("--verify")) return;

  const date = new Date().toISOString().slice(0, 10);
  const outFile = path.resolve(option("--out", path.join(ROOT, "benchmark", "results", `${date}.json`)));
  let commit = null, dirty = null;
  try {
    commit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" }).trim();
    dirty = execFileSync("git", ["status", "--porcelain", "--", "topologicpy-web-frontend/src", "tools"], { cwd: ROOT, encoding: "utf8" }).trim().length > 0;
  } catch {
    // Not a git checkout.
  }
  const results = { date, machine: machine(), commit, dirty, settings: { app: APP, ...opts }, models: [] };
  const tmp = path.join(os.tmpdir(), `studio-benchmark-${process.pid}`);
  mkdirSync(tmp, { recursive: true });
  const t0 = performance.now();
  for (const [i, job] of jobs.entries()) {
    if (job.missing || job.mismatch) {
      results.models.push({ name: job.model.name, schema: job.model.schema, error: job.missing ? "file not found" : "SHA-256 mismatch" });
      continue;
    }
    const jobFile = path.join(tmp, `job-${i}.json`);
    const outJson = path.join(tmp, `out-${i}.json`);
    writeFileSync(jobFile, JSON.stringify({ model: job.model, file: job.file, opts, out: outJson }));
    const ts = performance.now();
    const child = spawnSync(process.execPath, ["--max-old-space-size=12288", "--expose-gc", fileURLToPath(import.meta.url), "--child", jobFile], { stdio: ["ignore", "ignore", "pipe"], encoding: "utf8", maxBuffer: 256 * 1024 * 1024 });
    let r;
    try {
      r = JSON.parse(readFileSync(outJson, "utf8"));
    } catch {
      r = { name: job.model.name, error: `child exited ${child.status}: ${String(child.stderr || "").split("\n").filter(Boolean).slice(-3).join(" | ").slice(0, 600)}` };
    }
    r.wallSeconds = round((performance.now() - ts) / 1000, 1);
    results.models.push(r);
    const s = r.speed || {};
    console.log(
      `${job.model.name.slice(0, 36).padEnd(36)} ${r.error ? `ERROR ${r.error.slice(0, 160)}` : `n ${String(r.graph.nodes).padStart(6)}  extract ${fmt(s.extractMs)}  lattice ${fmt(s.latticeMs)}  A* ${fmt(s.astarMs, 2)}  yen ${fmt(s.yenMs)}  map ${fmt(r.forecast?.temperature?.doseDeadlineMapMs)}  RL ${fmt(s.rlMs)} ms  (${r.wallSeconds} s)`}`,
    );
  }
  results.totalMinutes = round((performance.now() - t0) / 60000, 1);
  mkdirSync(path.dirname(outFile), { recursive: true });
  writeFileSync(outFile, JSON.stringify(results, null, 1));
  console.log(`\nWrote ${path.relative(ROOT, outFile)} (${results.totalMinutes} min)\n`);
  console.log(summarize(results));
}

await main();
