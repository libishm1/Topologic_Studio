/**
 * The learned escape against a real fire's timeline: the shielded learner
 * (rlSafe.js) and the shipped Q-learning (rl.js) trained on the deadline map
 * of a Flamescapes run instead of the graph fire's hop front. Companion to
 * tools/rl-bench.mjs and wiki/research/rl-egress-study-2026-11-05.md.
 *
 *   node tools/rl-flamescapes.mjs [--labels corpus-1,duplex-apartment] [--starts 2]
 *        [--horizon 120] [--departs 0,30,60] [--growths ultra-fast] [--bins 0,3-5]
 *        [--pair class] [--pair-bins 0] [--pair-starts 1]
 *        [--cache DIR] [--fires DIR] [--samples DIR] [--out FILE]
 *   node tools/rl-flamescapes.mjs --summary FILE [FILE ...]
 *
 * Graphs. The cache tools/rl-bench.mjs --prepare writes (--cache, default
 * $RL_BENCH_CACHE or <os temp>/topologic-rl-bench): the same lattice and
 * exits. The geometry for the fire is extracted again from the IFC.
 *
 * Scenarios. rl-bench's draw, copied below with its seeds, so a start, its
 * exit and a fire origin here are the study's; only the first `starts` starts
 * of each model are used (default 2 of the study's 4), each with its six fire
 * origins (hop distance 0, 1-2, 3-5, 6-10, 11-20 and 21+ from the route).
 *
 * The fire. Flamescapes on the CPU, as the page sets it up by default
 * (App.jsx combustionInputs, flamescapes.worker.js): class A contents, the
 * class's design fire (medium t-squared growth) or another growth class,
 * 780 MJ/m2, doors open, windows closed, no ventilation, no sprinklers, the
 * standard grid (30,000 cells), the validated solver, the page's frame count
 * (min(120, 60, horizon) frames), and a short horizon (default 120 s, not the
 * page's 180 s). The deadline map is the forecast the page plans with
 * (forecast.js forecastFromFrames: untenable from the first frame with gas at
 * 60 C, visibility under 10 m, or an FED of 0.3; every node closes at the
 * horizon at the latest), the input of the page's escape-deadline map.
 *
 * The clock. rl.js and rlSafe.js count moves: one link a step. A step here
 * lasts unit / 1.2 s, with unit the median link length (rl.js's cost unit)
 * and 1.2 m/s the planner's walking speed. An evacuee leaving at `depart` s
 * makes the move at step t safely iff it arrives before the node closes,
 * depart + (t + 1) step < c(v), so the ignition step handed to both learners
 * is ig(v) = (c(v) - depart) / step - 1 (both test t < ig(v)). Links longer
 * than the median (diagonals, stairs) take longer than a step in fact, so
 * every learned route is also walked in seconds at 1.2 m/s against c(v)
 * (forecast.js walkRoute), as the page checks its planned routes.
 *
 * Methods, on the same scenario, fire and departure, with the step limit 200:
 *   exact      rlSafe.earliestSafeEscape, to the scenario's exit and to any exit.
 *   shipped    rl.js qLearningPath at the app's defaults (400 episodes), to the
 *              scenario's exit; "reached" is the drawn path the page shows.
 *   safe1      rlSafe.learnSafeEscape to the scenario's exit, 400 trials (the
 *              study's old-against-new setting).
 *   safeAny    rlSafe.learnSafeEscape to any exit, 20,000 trials (the page's
 *              shielded default, trainSafeRl).
 * When the fire departs from the forecast:
 *   hop        the page as shipped: both learners on the hop front at fire
 *              speed 1 (trainSafeRl and trainRl build their fire so, whatever
 *              fire model the page runs), their routes checked against the
 *              Flamescapes timeline.
 *   growth     the shielded route learned on the class's own (medium) fire,
 *              walked through the same origin's ultra-fast fire.
 *   scaled     the shielded route walked through the same forecast with every
 *              closing time (from ignition) multiplied by 0.9, 0.75 and 0.5.
 *
 * Flamescapes runs are kept in --fires (default <os temp>/topologic-rl-fires),
 * so the learners can be rerun without the solver.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const CORE = pathToFileURL(path.join(ROOT, "topologicpy-web-frontend", "src", "core")).href + "/";
const { astar, componentLabels, distancesFrom, edgeLengths } = await import(CORE + "route.js");
const { qLearningPath } = await import(CORE + "rl.js");
const safe = await import(CORE + "rlSafe.js");
const { forecastFromFrames } = await import(CORE + "forecast.js");
const { escapeDeadlines } = await import(CORE + "deadline.js");

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const option = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : fallback;
};
const CACHE = option("--cache", process.env.RL_BENCH_CACHE || path.join(os.tmpdir(), "topologic-rl-bench"));
const FIRES = option("--fires", path.join(os.tmpdir(), "topologic-rl-fires"));
const SAMPLES = option("--samples", process.env.BENCHMARK_SAMPLES || path.join(os.homedir(), "ifc-samples"));
const STARTS = Number(option("--starts", 2));
const HORIZON = Number(option("--horizon", 120));
const DEPARTS = option("--departs", "0,30,60").split(",").map(Number);
// The growth every chosen fire runs with, and a second growth run for the fires in PAIR_BINS only
// (the forecast with one growth, the fire with the other).
const GROWTHS = option("--growths", "ultra-fast").split(",");
const PAIR = option("--pair", "class") || null;
const PAIR_BINS = option("--pair-bins", "0").split(",");
const PAIR_STARTS = Number(option("--pair-starts", 1));
const FIRE_BINS = option("--bins", "0,3-5").split(",");
const SCALES = [0.9, 0.75, 0.5];
const H = 200; // the page's step limit (useStudio rlMaxSteps)
const EPISODES = 400; // the page's episodes (useStudio rlEpisodes)
const ANY_TRIALS = 20000; // trainSafeRl's default
const WALK = 1.2; // m/s, the planner's walking speed (useStudio walkSpeed)
const HOP_SPEED = 1; // links a move, the page's RL fire speed (useStudio rlFireSpeed)

// The same model list as rl-bench.mjs.
const DEFAULT_CORPUS = [0, 9, 19, 1, 10, 2];
const EXAMPLES = ["duplex-apartment", "office-two-storey", "courtyard-ring"];
const BINS = [[0, 0], [1, 2], [3, 5], [6, 10], [11, 20], [21, Infinity]];
const binLabel = ([a, b]) => (a === b ? `${a}` : Number.isFinite(b) ? `${a}-${b}` : `${a}+`);

const round = (x, d = 3) => (x == null || !Number.isFinite(x) ? x ?? null : Math.round(x * 10 ** d) / 10 ** d);
const median = (a) => {
  const s = a.filter((x) => x != null && Number.isFinite(x)).sort((x, y) => x - y);
  return s.length ? (s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : null;
};
const quantile = (a, q) => {
  const s = a.filter((x) => x != null && Number.isFinite(x)).sort((x, y) => x - y);
  return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))] : null;
};

/** mulberry32, as rl-bench.mjs. */
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

const b64 = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString("base64");
const from64 = (s, T) => {
  const buf = Buffer.from(s, "base64");
  return new T(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
};

function modelList() {
  const manifest = JSON.parse(readFileSync(path.join(ROOT, "benchmark", "manifest.json"), "utf8"));
  const list = DEFAULT_CORPUS.map((i, k) => {
    const m = manifest.models[i];
    return { label: `corpus-${k + 1}`, file: path.join(SAMPLES, ...m.path.split("/")), sha256: m.sha256 };
  });
  for (const name of EXAMPLES) {
    const m = manifest.models.find((x) => x.name === name);
    list.push({ label: name, file: path.join(ROOT, manifest.roots.examples, ...m.path.split("/")), sha256: m.sha256 });
  }
  const only = option("--labels", null);
  return only ? list.filter((m) => only.split(",").includes(m.label)) : list;
}

function loadGraph(model) {
  const file = path.join(CACHE, `${model.sha256.slice(0, 16)}.json`);
  if (!existsSync(file)) throw new Error(`No cached graph for ${model.label}: run node tools/rl-bench.mjs --prepare --cache ${CACHE}`);
  const j = JSON.parse(readFileSync(file, "utf8"));
  return { graph: { nodes: from64(j.nodes, Float32Array), edges: from64(j.edges, Uint32Array), blocked: from64(j.blocked, Uint8Array), kinds: from64(j.kinds, Uint8Array) }, exits: j.exits };
}

/** rl-bench.mjs's scenario draw, unchanged (same seeds, same order). */
function scenarios(graph, exits, sha, starts) {
  const n = graph.nodes.length / 3;
  const { labels } = componentLabels(graph);
  const blocked = graph.blocked;
  const hopsToExit = safe.exitHops(graph, { exits, blocked });
  const exitParts = new Set(exits.map((x) => labels[x]));
  const candidates = [];
  for (let v = 0; v < n; v += 1) if (graph.kinds[v] === 0 && exitParts.has(labels[v]) && hopsToExit[v] >= 10 && hopsToExit[v] < safe.NEVER) candidates.push(v);
  const random = rng(parseInt(sha.slice(0, 8), 16));
  const out = [];
  const { first, to } = safe.openLinks(graph, blocked);
  for (let s = 0; s < starts && candidates.length; s += 1) {
    const start = candidates[Math.floor(random() * candidates.length)];
    const metres = distancesFrom(graph, start, { blocked });
    let exit = -1;
    for (const x of exits) if (exit < 0 || metres[x] < metres[exit]) exit = x;
    const route = astar(graph, start, exit, { blocked })?.nodes;
    if (!route) continue;
    const delta = new Int32Array(n).fill(-1);
    let frontier = [...route];
    for (const v of route) delta[v] = 0;
    for (let d = 1; frontier.length; d += 1) {
      const next = [];
      for (const u of frontier) for (let j = first[u]; j < first[u + 1]; j += 1) {
        const v = to[j];
        if (delta[v] >= 0) continue;
        delta[v] = d;
        next.push(v);
      }
      frontier = next;
    }
    for (const bin of BINS) {
      const pool = [];
      for (let v = 0; v < n; v += 1) if (v !== start && v !== exit && delta[v] >= bin[0] && delta[v] <= bin[1]) pool.push(v);
      if (!pool.length) continue;
      const fire = pool[Math.floor(random() * pool.length)];
      out.push({ startIndex: s, start, exit, route, fire, delta: delta[fire], bin: binLabel(bin), routeHops: route.length - 1 });
    }
  }
  return out;
}

const point = (graph, v) => [graph.nodes[v * 3], graph.nodes[v * 3 + 1], graph.nodes[v * 3 + 2]];
const timeIt = (fn) => {
  const t = performance.now();
  const out = fn();
  return [out, performance.now() - t];
};

// ============================================================ the fire

let fireTools = null;
async function tools() {
  if (fireTools) return fireTools;
  const { extract } = await import("./ifc-extract.mjs");
  const run = await import(CORE + "flamescapesRun.js");
  const classes = await import(CORE + "fireClasses.js");
  const floors = await import(CORE + "spaceFloors.js");
  const versions = await import(CORE + "solverVersions.js");
  fireTools = { extract, ...run, ...classes, ...floors, ...versions };
  return fireTools;
}

/** The geometry the page hands the solver (App.jsx combustionInputs) at its defaults: doors open, windows closed, no overhead services. */
function sceneGeometry(geo, T) {
  const names = new Map((geo.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]));
  const isVoid = (m) => T.VOID_NAME.test(names.get(m.itemId) || "");
  const spaces = geo.spaces || [];
  return {
    walls: geo.walls || [],
    floors: [...(geo.floors || []), ...(spaces.length && T.slabsMissing(geo.floors, spaces, { upAxis: "y" }) ? T.floorsFromSpaces(spaces.filter((m) => !isVoid(m)), { upAxis: "y" }) : [])],
    voids: spaces.filter(isVoid),
    roofs: geo.roofs || [],
    ceilings: geo.ceilings || [],
    stairs: geo.stairs || [],
    obstacles: geo.obstacles || [],
  };
}

/** One Flamescapes run from a node, kept on disk: each node's closing time c(v) and what the run was. */
async function flamescapes(model, geoRef, graph, fire, growth) {
  mkdirSync(FIRES, { recursive: true });
  const key = `${model.sha256.slice(0, 16)}-${fire}-${growth}-${HORIZON}`;
  const file = path.join(FIRES, `${key}.json`);
  if (existsSync(file)) {
    const j = JSON.parse(readFileSync(file, "utf8"));
    return { ...j, closes: j.closes ? from64(j.closes, Float64Array) : null };
  }
  const T = await tools();
  if (!geoRef.geo) {
    const quiet = console.log;
    console.log = (...a) => (typeof a[0] === "string" && a[0].startsWith("[WEB-IFC]") ? null : quiet(...a));
    geoRef.geo = await T.extract(model.file, { withDoorFlags: true });
    console.log = quiet;
    geoRef.scene = sceneGeometry(geoRef.geo, T);
  }
  const { contentsFuel, ignitionFuel, item } = T.fuelsFor("A");
  const designFire = T.designFireFor("A", growth);
  const frames = Math.max(1, Math.min(120, 60, Math.floor(HORIZON)));
  const info = { key, growth, alpha: designFire?.alpha ?? null, horizon: HORIZON, frameSeconds: HORIZON / frames };
  let out;
  try {
    const t0 = performance.now();
    const scene = T.buildScene(geoRef.scene, { upAxis: "y", fire: point(graph, fire), maxCells: 30000, fireLoad: 780e6, contentsFuel, ignitionFuel, item, designFire, ventilation: null, seed: 1 });
    const setupMs = performance.now() - t0;
    if (!scene.ignition) {
      out = { ...info, skipped: "no place for the first item at the fire origin", closes: null };
    } else {
      let peak = 0;
      const t1 = performance.now();
      const run = T.runScene(scene, {
        nodes: graph.nodes,
        upAxis: "y",
        agentHeight: 0.75,
        headHeight: 1.8,
        horizon: HORIZON,
        frameSeconds: HORIZON / frames,
        embers: false,
        particles: 0,
        smoke: false,
        solverOptions: T.solverOptionsFor("validated"),
        keep: true,
        onFrame: (f) => {
          if (f.hrr > peak) peak = f.hrr;
        },
      });
      const runMs = performance.now() - t1;
      const g = scene.grid;
      const meta = { ...info, grid: [g.nx, g.ny, g.nz], cell: round(g.h, 3), cells: g.cells, clipped: Boolean(scene.clipped), inGrid: run.inGrid, nodes: run.n, setupMs: round(setupMs, 0), runMs: round(runMs, 0), peakHrrKw: round(peak, 1), health: run.health };
      if (run.health?.ok === false) {
        // The page does not plan with a run its health guard stopped.
        out = { ...meta, skipped: "the solver's health guard stopped the run", closes: null };
      } else {
        const forecast = forecastFromFrames(run, { threshold: 60, smoke: true, visibilityLimit: 10 });
        let untenable = 0, first = Infinity;
        for (let v = 0; v < forecast.n; v += 1) if (forecast.untenable[v] < forecast.horizon) {
          untenable += 1;
          first = Math.min(first, forecast.untenable[v]);
        }
        out = { ...meta, forecastHorizon: forecast.horizon, untenableNodes: untenable, firstUntenable: Number.isFinite(first) ? first : null, closes: forecast.closes };
      }
    }
  } catch (error) {
    out = { ...info, skipped: `error: ${error?.message || error}`, closes: null };
  }
  writeFileSync(file, JSON.stringify({ ...out, closes: out.closes ? b64(out.closes) : null }));
  return out;
}

// ======================================================== the learners

/** Ignition steps on the hop clock for an evacuee leaving at `depart` s (see the header). */
function hopIgnition(closes, depart, stepSeconds) {
  const ig = new Float64Array(closes.length);
  for (let v = 0; v < closes.length; v += 1) ig[v] = (closes[v] - depart) / stepSeconds - 1;
  return ig;
}

const asMap = (ig) => {
  const m = new Map();
  ig.forEach((s, v) => Number.isFinite(s) && m.set(v, s));
  return m;
};

/** The first move of `path` that enters a node at or after its ignition step (rl.js's test), or -1. */
function hopUnsafe(path, ig) {
  for (let i = 1; i < path.length; i += 1) if (!(i - 1 < ig[path[i]])) return i;
  return -1;
}

/**
 * Walked in seconds at 1.2 m/s from `depart` against the closing times: the
 * first node reached at or after it closes, or -1. The test of forecast.js
 * walkRoute's firstUnsafe, without its dose (which needs the frames).
 */
function walkUnsafe(graph, path, closes, depart) {
  const P = graph.nodes;
  if (!(depart < closes[path[0]])) return 0;
  let t = depart;
  for (let i = 1; i < path.length; i += 1) {
    const a = path[i - 1] * 3, b = path[i] * 3;
    t += Math.hypot(P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]) / WALK;
    if (!(t < closes[path[i]])) return i;
  }
  return -1;
}

const scaled = (closes, s) => closes.map((c) => c * s);

function runCase(graph, exits, sc, fireRun, depart, seed, stepSeconds, extra) {
  const blocked = graph.blocked;
  const closes = fireRun.closes;
  const horizon = fireRun.forecastHorizon;
  const ig = hopIgnition(closes, depart, stepSeconds);
  const row = { start: sc.start, startIndex: sc.startIndex, exit: sc.exit, fire: sc.fire, bin: sc.bin, delta: sc.delta, routeHops: sc.routeHops, growth: fireRun.growth, depart, seed };
  row.startUntenable = !(depart < closes[sc.start]);
  // Route nodes the fire grid covers: outside it the run reads clean air.
  let routeClosing = 0;
  for (const v of sc.route) if (closes[v] < horizon) routeClosing += 1;
  row.routeClosing = routeClosing;

  const [D, dMs] = timeIt(() => safe.hopDeadlines(graph, { exits, ignite: ig, horizon: H, blocked }));
  row.deadlineMs = round(dMs, 2);
  row.deadlineStart = D[sc.start];
  // The page's own hard deadline at the start, in seconds (deadline.js escapeDeadlines on the same closing times).
  row.metricDeadline = round(escapeDeadlines(graph, { exits, untenable: closes, speed: WALK, blocked })[sc.start], 2);
  row.hopDeadlineSeconds = D[sc.start] >= 0 ? round(depart + D[sc.start] * stepSeconds, 2) : null;

  const e1 = row.startUntenable ? null : safe.earliestSafeEscape(graph, sc.start, { exits: [sc.exit], ignite: ig, horizon: H, blocked });
  const eAny = row.startUntenable ? null : safe.earliestSafeEscape(graph, sc.start, { exits, ignite: ig, horizon: H, blocked });
  row.escape = e1 ? e1.moves : null;
  row.escapeAny = eAny ? eAny.moves : null;
  row.exactWalkUnsafe = e1 ? walkUnsafe(graph, e1.path, closes, depart) >= 0 : null;
  row.exactAnyWalkUnsafe = eAny ? walkUnsafe(graph, eAny.path, closes, depart) >= 0 : null;

  // The shipped Q-learning, on this deadline map, to the scenario's exit.
  const [q, qMs] = timeIt(() => qLearningPath(graph, sc.start, sc.exit, { ignite: asMap(ig), episodes: EPISODES, maxSteps: H, seed, blocked }));
  row.shipped = { reached: q.reachedExit, burned: q.burned, moves: q.reachedExit ? q.path.length - 1 : null, ms: round(qMs, 1), hopUnsafe: q.reachedExit ? hopUnsafe(q.path, ig) >= 0 : null, walkUnsafe: q.reachedExit ? walkUnsafe(graph, q.path, closes, depart) >= 0 : null, escapeCheck: q.escapeSteps };

  // The shielded learner, to the scenario's exit (400 trials) and to any exit
  // (20,000). Both times include the deadline map, as rl-bench.mjs's do.
  const [s1, s1Ms] = timeIt(() => safe.learnSafeEscape(graph, sc.start, { exits: [sc.exit], ignite: ig, horizon: H, blocked, trials: EPISODES }));
  row.safe1 = { reached: s1.reachedExit, burned: s1.burned, moves: s1.moves, trials: s1.trials, certified: s1.certified, ms: round(s1Ms, 2), hopUnsafe: s1.reachedExit ? hopUnsafe(s1.path, ig) >= 0 : null, walkUnsafe: s1.reachedExit ? walkUnsafe(graph, s1.path, closes, depart) >= 0 : null };
  const [sA, sAMs] = timeIt(() => safe.learnSafeEscape(graph, sc.start, { exits, ignite: ig, horizon: H, blocked, trials: ANY_TRIALS, deadline: D }));
  row.safeAny = { reached: sA.reachedExit, burned: sA.burned, moves: sA.moves, trials: sA.trials, certified: sA.certified, ms: round(sAMs + dMs, 2), hopUnsafe: sA.reachedExit ? hopUnsafe(sA.path, ig) >= 0 : null, walkUnsafe: sA.reachedExit ? walkUnsafe(graph, sA.path, closes, depart) >= 0 : null };

  // The fire departs from the forecast: the same shielded route through a fire that closes every node sooner.
  row.scaled = {};
  for (const s of SCALES) {
    const c2 = scaled(closes, s);
    const ig2 = hopIgnition(c2, depart, stepSeconds);
    const exists = !(depart < c2[sc.start]) ? false : Boolean(safe.earliestSafeEscape(graph, sc.start, { exits, ignite: ig2, horizon: H, blocked }));
    row.scaled[s] = { escapeExists: exists, hopUnsafe: sA.reachedExit ? hopUnsafe(sA.path, ig2) >= 0 : null, walkUnsafe: sA.reachedExit ? walkUnsafe(graph, sA.path, c2, depart) >= 0 : null };
  }
  if (extra?.other) {
    // The route learned on this forecast, walked through the other growth's fire from the same origin.
    const c2 = extra.other.closes;
    const ig2 = hopIgnition(c2, depart, stepSeconds);
    const exists = !(depart < c2[sc.start]) ? false : Boolean(safe.earliestSafeEscape(graph, sc.start, { exits, ignite: ig2, horizon: H, blocked }));
    row.otherGrowth = { growth: extra.other.growth, escapeExists: exists, hopUnsafe: sA.reachedExit ? hopUnsafe(sA.path, ig2) >= 0 : null, walkUnsafe: sA.reachedExit ? walkUnsafe(graph, sA.path, c2, depart) >= 0 : null };
  }
  if (extra?.hop) {
    // The page as shipped: the hop front at fire speed 1, whatever fire the page ran.
    const igH = extra.hop;
    const hExact = safe.earliestSafeEscape(graph, sc.start, { exits, ignite: igH, horizon: H, blocked });
    const hs = safe.learnSafeEscape(graph, sc.start, { exits, ignite: igH, horizon: H, blocked, trials: ANY_TRIALS });
    const hq = qLearningPath(graph, sc.start, sc.exit, { ignite: asMap(igH), episodes: EPISODES, maxSteps: H, seed, blocked });
    row.hop = {
      escapeAny: hExact ? hExact.moves : null,
      safeAny: { reached: hs.reachedExit, certified: hs.certified, moves: hs.moves, hopUnsafe: hs.reachedExit ? hopUnsafe(hs.path, ig) >= 0 : null, walkUnsafe: hs.reachedExit ? walkUnsafe(graph, hs.path, closes, depart) >= 0 : null },
      shipped: { reached: hq.reachedExit, moves: hq.reachedExit ? hq.path.length - 1 : null, hopUnsafe: hq.reachedExit ? hopUnsafe(hq.path, ig) >= 0 : null, walkUnsafe: hq.reachedExit ? walkUnsafe(graph, hq.path, closes, depart) >= 0 : null },
    };
  }
  return row;
}

// ============================================================= summary

function summarize(files) {
  const results = files.map((f) => JSON.parse(readFileSync(f, "utf8")));
  const models = results.flatMap((r) => r.models);
  const rows = models.flatMap((m) => m.rows.map((r) => ({ ...r, model: m.label })));
  const runs = models.flatMap((m) => (m.fires || []).map((f) => ({ ...f, model: m.label })));
  const lines = [];
  const pct = (a, b) => (b ? `${((100 * a) / b).toFixed(1)}%` : "-");
  const frac = (a, b) => `${a} of ${b} (${pct(a, b)})`;
  const table = (title, head, body) => lines.push("", `### ${title}`, "", `| ${head.join(" | ")} |`, `|${head.map(() => "---").join("|")}|`, ...body.map((r) => `| ${r.join(" | ")} |`));
  const s0 = results[0].settings;
  // The pooled tables below read the main growth only (the paired growth is in its own rows above).
  const main = rows.filter((r) => r.growth === s0.growths[0]);
  lines.push(`Settings: ${JSON.stringify(s0)}`);
  lines.push(`Files: ${files.map((f) => path.basename(f)).join(", ")}`);

  table("Flamescapes runs (CPU, standard grid)", ["Model", "Nodes", "Fires", "Growth", "Runs used", "Skipped", "Median run (s)", "Median peak HRR (kW)", "Median nodes in grid", "Median nodes untenable by the horizon", "Median first untenable (s)"],
    models.flatMap((m) => [...new Set((m.fires || []).map((f) => f.growth))].map((g) => {
      const l = m.fires.filter((f) => f.growth === g);
      const ok = l.filter((f) => !f.skipped);
      return [m.label, m.nodes, new Set(l.map((f) => f.fire)).size, g, ok.length, l.length - ok.length, round(median(ok.map((f) => f.runMs / 1000)), 1), round(median(ok.map((f) => f.peakHrrKw)), 0), median(ok.map((f) => f.inGrid)), median(ok.map((f) => f.untenableNodes)), median(ok.map((f) => f.firstUntenable))];
    })));
  {
    const skipped = runs.filter((f) => f.skipped);
    const reasons = {};
    for (const f of skipped) reasons[f.skipped] = (reasons[f.skipped] || 0) + 1;
    lines.push("", `Runs: ${runs.length}; used ${runs.length - skipped.length}; skipped ${skipped.length} (${Object.entries(reasons).map(([k, v]) => `${v}: ${k}`).join("; ") || "none"}). Grid clipped to the 30,000-cell budget in ${runs.filter((f) => f.clipped).length} runs.`);
  }

  const groups = [...new Set(rows.map((r) => r.growth))].flatMap((g) => [
    [`${g}, all departures`, rows.filter((r) => r.growth === g)],
    ...[...new Set(rows.filter((r) => r.growth === g).map((r) => r.depart))].map((d) => [`${g}, leave at ${d} s`, rows.filter((r) => r.growth === g && r.depart === d)]),
  ]);
  table("Scenarios", ["Group", "Runs", "Start untenable at departure", "Fire closes a route node by the horizon", "Escape to the scenario's exit", "Escape to some exit"], groups.map(([k, l]) => [k, l.length, l.filter((r) => r.startUntenable).length, l.filter((r) => r.routeClosing > 0).length, l.filter((r) => r.escape != null).length, l.filter((r) => r.escapeAny != null).length]));

  const gap = (l, pick, ref) => median(l.map((r) => (pick(r) != null && ref(r) != null ? pick(r) - ref(r) : null)));
  table("To the scenario's exit, where an escape to it exists (the study's old against new)", ["Group", "Runs", "Shipped escapes", "Shipped earliest", "Shipped median extra moves", "Shipped median ms", "Shielded escapes", "Shielded earliest", "Shielded certified (400 trials)", "Shielded median ms"], groups.map(([k, l0]) => {
    const l = l0.filter((r) => r.escape != null);
    return [k, l.length, pct(l.filter((r) => r.shipped.reached).length, l.length), pct(l.filter((r) => r.shipped.moves === r.escape).length, l.length), gap(l, (r) => r.shipped.moves, (r) => r.escape), round(median(l.map((r) => r.shipped.ms)), 1), pct(l.filter((r) => r.safe1.reached).length, l.length), pct(l.filter((r) => r.safe1.moves === r.escape).length, l.length), pct(l.filter((r) => r.safe1.certified).length, l.length), round(median(l.map((r) => r.safe1.ms)), 2)];
  }));
  table("To any exit, where an escape to some exit exists (the page's shielded default, 20,000 trials)", ["Group", "Runs", "Escapes", "Earliest", "Certified within 400 trials", "Certified within 20,000", "Trials median / 90th / largest", "Median / largest ms (with the deadline map)"], groups.map(([k, l0]) => {
    const l = l0.filter((r) => r.escapeAny != null);
    return [k, l.length, pct(l.filter((r) => r.safeAny.reached).length, l.length), pct(l.filter((r) => r.safeAny.moves === r.escapeAny).length, l.length), pct(l.filter((r) => r.safeAny.certified && r.safeAny.trials <= 400).length, l.length), pct(l.filter((r) => r.safeAny.certified).length, l.length), `${median(l.map((r) => r.safeAny.trials))} / ${quantile(l.map((r) => r.safeAny.trials), 0.9)} / ${Math.max(0, ...l.map((r) => r.safeAny.trials))}`, `${round(median(l.map((r) => r.safeAny.ms)), 2)} / ${round(Math.max(0, ...l.map((r) => r.safeAny.ms)), 1)}`];
  }));
  {
    const all = main;
    const noEsc = all.filter((r) => r.escape == null && !r.startUntenable);
    table("Where no escape exists (main growth): does any learner claim one?", ["Measure", "Value"], [
      ["runs with no escape to the scenario's exit (start tenable)", noEsc.length],
      ["shipped drawn path reaches the exit anyway", noEsc.filter((r) => r.shipped.reached).length],
      ["shielded (one exit) reaches the exit anyway", noEsc.filter((r) => r.safe1.reached).length],
      ["runs with no escape to any exit (start tenable)", all.filter((r) => r.escapeAny == null && !r.startUntenable).length],
      ["shielded (any exit) reaches an exit anyway", all.filter((r) => r.escapeAny == null && !r.startUntenable && r.safeAny.reached).length],
    ]);
  }
  table("Routes walked in seconds at 1.2 m/s against the same forecast (forecast.js walkRoute): reached too late somewhere", ["Group", "Shipped (of its escapes)", "Shielded, one exit", "Shielded, any exit", "Exact earliest, one exit", "Exact earliest, any exit", "Shipped routes the hop clock calls unsafe"], groups.map(([k, l]) => {
    const f = (pick, test) => {
      const m = l.filter((r) => pick(r));
      return frac(m.filter(test).length, m.length);
    };
    return [k, f((r) => r.shipped.reached, (r) => r.shipped.walkUnsafe), f((r) => r.safe1.reached, (r) => r.safe1.walkUnsafe), f((r) => r.safeAny.reached, (r) => r.safeAny.walkUnsafe), f((r) => r.escape != null, (r) => r.exactWalkUnsafe), f((r) => r.escapeAny != null, (r) => r.exactAnyWalkUnsafe), f((r) => r.shipped.reached, (r) => r.shipped.hopUnsafe)];
  }));
  {
    const l = rows.filter((r) => r.metricDeadline != null && Number.isFinite(r.metricDeadline) && r.hopDeadlineSeconds != null);
    const diffs = l.map((r) => r.hopDeadlineSeconds - r.metricDeadline);
    lines.push("", `Deadline at the start: the hop clock's (depart + D(start) x step) against the page's escapeDeadlines in seconds, ${l.length} runs with both finite: median difference ${round(median(diffs), 2)} s, 10th to 90th percentile ${round(quantile(diffs, 0.1), 2)} to ${round(quantile(diffs, 0.9), 2)} s; hop later than metric in ${diffs.filter((d) => d > 0.5).length}.`);
  }

  // The fire departs from the forecast.
  const hopRows = rows.filter((r) => r.hop);
  if (hopRows.length) {
    const both = hopRows.filter((r) => !r.startUntenable);
    table("The page as shipped: learners on the hop front (fire speed 1), checked against the Flamescapes timeline", ["Measure", "Value"], [
      ["runs (start tenable)", both.length],
      ["escape exists on Flamescapes / on the hop front", `${both.filter((r) => r.escapeAny != null).length} / ${both.filter((r) => r.hop.escapeAny != null).length}`],
      ["Flamescapes has an escape the hop front says does not exist", both.filter((r) => r.escapeAny != null && r.hop.escapeAny == null).length],
      ["hop front has an escape, Flamescapes has none", both.filter((r) => r.escapeAny == null && r.hop.escapeAny != null).length],
      ["shielded (hop) routes that reach an exit: certified", frac(both.filter((r) => r.hop.safeAny.reached && r.hop.safeAny.certified).length, both.filter((r) => r.hop.safeAny.reached).length)],
      ["shielded (hop) routes unsafe on the Flamescapes timeline, hop clock", frac(both.filter((r) => r.hop.safeAny.hopUnsafe).length, both.filter((r) => r.hop.safeAny.reached).length)],
      ["the same, walked in seconds", frac(both.filter((r) => r.hop.safeAny.walkUnsafe).length, both.filter((r) => r.hop.safeAny.reached).length)],
      ["  of them, where Flamescapes had an escape", frac(both.filter((r) => r.hop.safeAny.hopUnsafe && r.escapeAny != null).length, both.filter((r) => r.hop.safeAny.reached && r.escapeAny != null).length)],
      ["shipped (hop) drawn paths that reach the exit: unsafe on Flamescapes, hop clock", frac(both.filter((r) => r.hop.shipped.hopUnsafe).length, both.filter((r) => r.hop.shipped.reached).length)],
      ["the same, walked in seconds", frac(both.filter((r) => r.hop.shipped.walkUnsafe).length, both.filter((r) => r.hop.shipped.reached).length)],
    ]);
  }
  const og = rows.filter((r) => r.otherGrowth && !r.startUntenable && r.safeAny.reached);
  if (og.length) {
    table("Forecast with one growth, the fire grows at another: the shielded route (any exit) walked through the other fire", ["Forecast / fire", "Routes", "Unsafe, hop clock", "Unsafe, in seconds", "Unsafe where the other fire still had an escape", "Other fire had no escape at all"], [...new Set(og.map((r) => `${r.growth} / ${r.otherGrowth.growth}`))].map((k) => {
      const l = og.filter((r) => `${r.growth} / ${r.otherGrowth.growth}` === k);
      return [k, l.length, frac(l.filter((r) => r.otherGrowth.hopUnsafe).length, l.length), frac(l.filter((r) => r.otherGrowth.walkUnsafe).length, l.length), frac(l.filter((r) => r.otherGrowth.hopUnsafe && r.otherGrowth.escapeExists).length, l.filter((r) => r.otherGrowth.escapeExists).length), l.filter((r) => !r.otherGrowth.escapeExists).length];
    }));
  }
  {
    const l = main.filter((r) => !r.startUntenable && r.safeAny.reached);
    table("The same forecast, every closing time multiplied by s (the fire faster than forecast): the shielded route (any exit)", ["s", "Routes", "Unsafe, hop clock", "Unsafe, in seconds", "Unsafe where the faster fire still had an escape", "Faster fire had no escape"], SCALES.map((s) => [s, l.length, frac(l.filter((r) => r.scaled[s].hopUnsafe).length, l.length), frac(l.filter((r) => r.scaled[s].walkUnsafe).length, l.length), frac(l.filter((r) => r.scaled[s].hopUnsafe && r.scaled[s].escapeExists).length, l.filter((r) => r.scaled[s].escapeExists).length), l.filter((r) => !r.scaled[s].escapeExists).length]));
  }
  table("By model (main growth, all departures)", ["Model", "Runs", "Escape (one exit)", "Shipped escapes", "Shielded escapes", "Escape (any exit)", "Shielded any: escapes / earliest / certified", "Shipped median ms", "Shielded any median ms"], models.map((m) => {
    const l = main.filter((r) => r.model === m.label);
    const e = l.filter((r) => r.escape != null), a = l.filter((r) => r.escapeAny != null);
    return [m.label, l.length, e.length, pct(e.filter((r) => r.shipped.reached).length, e.length), pct(e.filter((r) => r.safe1.reached).length, e.length), a.length, `${pct(a.filter((r) => r.safeAny.reached).length, a.length)} / ${pct(a.filter((r) => r.safeAny.moves === r.escapeAny).length, a.length)} / ${pct(a.filter((r) => r.safeAny.certified).length, a.length)}`, round(median(l.map((r) => r.shipped.ms)), 1), round(median(l.map((r) => r.safeAny.ms)), 2)];
  }));
  {
    const bad = rows.filter((r) => (r.escape != null && !r.safe1.reached) || (r.escapeAny != null && !r.safeAny.reached) || r.safe1.hopUnsafe || r.safeAny.hopUnsafe || (r.safe1.reached && r.safe1.certified && r.safe1.moves !== r.escape) || (r.safeAny.reached && r.safeAny.certified && r.safeAny.moves !== r.escapeAny));
    lines.push("", `Checks: shielded runs that miss an existing escape, enter a closed node on the hop clock, or are certified but not earliest: ${bad.length} of ${rows.length}.${bad.length ? " " + JSON.stringify(bad.slice(0, 5).map((r) => ({ model: r.model, start: r.start, fire: r.fire, growth: r.growth, depart: r.depart }))) : ""} Shipped escape count matches rlSafe's (rl.js earliestEscape on the same map): ${rows.filter((r) => !r.startUntenable && r.shipped.escapeCheck === r.escape).length} of ${rows.filter((r) => !r.startUntenable).length}.`);
  }
  return lines.join("\n");
}

// ================================================================ main

async function main() {
  if (flag("--summary")) {
    const i = argv.indexOf("--summary");
    console.log(summarize(argv.slice(i + 1).filter((a) => !a.startsWith("--"))));
    return;
  }
  const results = { date: new Date().toISOString().slice(0, 10), machine: { cpu: os.cpus()[0]?.model?.trim(), logicalCores: os.cpus().length, node: process.version }, settings: { horizon: HORIZON, departs: DEPARTS, growths: GROWTHS, pair: PAIR, pairBins: PAIR_BINS, pairStarts: PAIR_STARTS, bins: FIRE_BINS, starts: STARTS, stepLimit: H, episodes: EPISODES, anyTrials: ANY_TRIALS, walk: WALK, hopSpeed: HOP_SPEED, scales: SCALES, untenableC: 60, visibilityM: 10, grid: "standard, 30000 cells, CPU", fireClass: "A" }, models: [] };
  const out = option("--out", path.join(ROOT, "benchmark", "results", `${results.date}-rl-flamescapes.json`));
  for (const model of modelList()) {
    const t0 = performance.now();
    const { graph, exits } = loadGraph(model);
    const n = graph.nodes.length / 3;
    const lengths = Array.from(edgeLengths(graph)).sort((a, b) => a - b);
    const unit = lengths[Math.floor(lengths.length / 2)] || 1;
    const stepSeconds = unit / WALK;
    const entry = { label: model.label, nodes: n, exits: exits.length, unit: round(unit, 4), stepSeconds: round(stepSeconds, 4), fires: [], rows: [] };
    results.models.push(entry);
    const list = scenarios(graph, exits, model.sha256, STARTS).filter((sc) => FIRE_BINS.includes(sc.bin));
    const geoRef = {};
    let seed = 0;
    for (const sc of list) {
      const byGrowth = {};
      const growths = PAIR && PAIR_BINS.includes(sc.bin) && sc.startIndex < PAIR_STARTS ? [...GROWTHS, PAIR] : GROWTHS;
      for (const growth of growths) {
        const run = await flamescapes(model, geoRef, graph, sc.fire, growth);
        const { closes, ...meta } = run;
        entry.fires.push({ fire: sc.fire, bin: sc.bin, startIndex: sc.startIndex, ...meta });
        if (closes) byGrowth[growth] = run;
      }
      const igHop = safe.ignitionSteps(graph, sc.fire, { fireSpeed: HOP_SPEED, maxSteps: H, blocked: graph.blocked });
      for (const growth of growths) {
        const run = byGrowth[growth];
        if (!run) continue;
        const other = growths.find((g) => g !== growth && byGrowth[g]);
        for (const depart of DEPARTS) {
          seed += 1;
          entry.rows.push(runCase(graph, exits, sc, run, depart, seed, stepSeconds, { other: other ? byGrowth[other] : null, hop: growth === GROWTHS[0] ? igHop : null }));
        }
      }
      writeFileSync(out, JSON.stringify(results));
    }
    console.log(`${model.label.padEnd(20)} ${n} nodes, ${list.length} fires x ${GROWTHS.length} growths, ${Math.round((performance.now() - t0) / 1000)} s`);
    writeFileSync(out, JSON.stringify(results));
  }
  console.log(`\nWrote ${out}`);
  console.log(summarize([out]));
}

await main();
