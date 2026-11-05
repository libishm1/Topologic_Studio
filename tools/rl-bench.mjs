/**
 * Why the learned escape fails when the fire is close, and what fixes it:
 * the diagnosis and the old-against-new comparison behind
 * wiki/research/rl-egress-study-2026-11-05.md.
 *
 *   node tools/rl-bench.mjs --prepare        # build and cache the graphs
 *   node tools/rl-bench.mjs                  # run, write benchmark/results/<date>-rl.json
 *   node tools/rl-bench.mjs --summary FILE   # print the tables from a results file
 *
 * Options: --models 0,2,9 (indices into benchmark/manifest.json's models; the
 * default is six corpus models spanning about 1,000 to 34,000 nodes, plus the
 * three shipped examples with exits), --samples DIR (default
 * $BENCHMARK_SAMPLES or ~/ifc-samples), --cache DIR (default
 * <os temp>/topologic-rl-bench, outside any synced folder), --starts N
 * (starts per model, default 4), --quick (two starts, fewer knobs),
 * --out FILE.
 *
 * Graphs are built as tools/benchmark.mjs builds them (web-ifc extraction,
 * the lattice at the app's defaults, exits by tools/exits.mjs). Corpus models
 * are labelled corpus-1, corpus-2, ... in the results: their files and names
 * stay out of the repository.
 *
 * Scenarios. Per model, `starts` floor nodes drawn (seeded by the model's
 * SHA-256) from parts that hold an exit, at least 10 links from it; the exit
 * is the nearest by walking distance, as the app's "nearest exit" picks it;
 * the fire-free route is A*. Fire origins are drawn at hop distances 0, 1-2,
 * 3-5, 6-10, 11-20 and 21+ from that route (not the start or the exit), and
 * each runs at fire speeds 0.25, 0.5, 1 and 2 links per move.
 *
 * Methods, on the same scenario:
 *   shipped    rl.js qLearningPath at the app's defaults (400 episodes, 200
 *              steps); "reached" is the drawn path the app shows.
 *   legacy*    a copy of its training loop with one knob changed (episodes,
 *              epsilon or its schedule, alpha, the step limit, a node-only
 *              state as in Jabi et al. 2019, or the deadline shield). The copy
 *              is checked to reproduce rl.js exactly at the defaults.
 *   optimum    the exact optimal policy of rl.js's own reward (backward
 *              induction), walked: whether the reward's optimum escapes at all
 *              separates a model defect from a training defect.
 *   safe       rlSafe.js learnSafeEscape (deadline shield, learning
 *              real-time search on (node, step)), to the one exit, and to any
 *              exit; "arrival" masks only nodes untenable on arrival.
 *   exact      rlSafe.js earliestSafeEscape (the earliest escape, any exit).
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const CORE = pathToFileURL(path.join(ROOT, "topologicpy-web-frontend", "src", "core")).href + "/";
const route = await import(CORE + "route.js");
const { astar, componentLabels, csr, distancesFrom, edgeLengths } = route;
const { qLearningPath, earliestEscape, BURN_REWARD, EXIT_REWARD, STEP_COST } = await import(CORE + "rl.js");
const safe = await import(CORE + "rlSafe.js");

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const option = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : fallback;
};
const QUICK = flag("--quick");
const CACHE = option("--cache", process.env.RL_BENCH_CACHE || path.join(os.tmpdir(), "topologic-rl-bench"));
const SAMPLES = option("--samples", process.env.BENCHMARK_SAMPLES || path.join(os.homedir(), "ifc-samples"));
/** Six corpus models (manifest indices) spanning about 1,000 to 34,000 nodes. */
const DEFAULT_CORPUS = [0, 9, 19, 1, 10, 2];
const EXAMPLES = ["duplex-apartment", "office-two-storey", "courtyard-ring"];
const APP = { episodes: 400, maxSteps: 200, fireSpeed: 1 };
const SPEEDS = [0.25, 0.5, 1, 2];
const BINS = [[0, 0], [1, 2], [3, 5], [6, 10], [11, 20], [21, Infinity]];
const binLabel = ([a, b]) => (a === b ? `${a}` : Number.isFinite(b) ? `${a}-${b}` : `${a}+`);

const round = (x, d = 3) => (x == null || !Number.isFinite(x) ? x ?? null : Math.round(x * 10 ** d) / 10 ** d);
const quantile = (a, q) => {
  const s = a.filter((x) => x != null && Number.isFinite(x)).sort((x, y) => x - y);
  return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))] : null;
};
const median = (a) => {
  const s = a.filter((x) => x != null && Number.isFinite(x)).sort((x, y) => x - y);
  return s.length ? (s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : null;
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

// ============================================================== graphs

const b64 = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString("base64");
const from64 = (s, T) => {
  const buf = Buffer.from(s, "base64");
  return new T(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
};

function modelList() {
  const manifest = JSON.parse(readFileSync(path.join(ROOT, "benchmark", "manifest.json"), "utf8"));
  const picks = option("--models", null);
  const indices = picks ? picks.split(",").map(Number) : DEFAULT_CORPUS;
  const list = [];
  indices.forEach((i, k) => {
    const m = manifest.models[i];
    if (!m) throw new Error(`No manifest model ${i}`);
    const file = m.root === "examples" ? path.join(ROOT, manifest.roots.examples, ...m.path.split("/")) : path.join(SAMPLES, ...m.path.split("/"));
    list.push({ label: m.root === "examples" ? m.name : `corpus-${k + 1}`, file, sha256: m.sha256 });
  });
  if (!picks) {
    for (const name of EXAMPLES) {
      const m = manifest.models.find((x) => x.name === name);
      list.push({ label: name, file: path.join(ROOT, manifest.roots.examples, ...m.path.split("/")), sha256: m.sha256 });
    }
  }
  return list;
}

async function prepare(model) {
  mkdirSync(CACHE, { recursive: true });
  const out = path.join(CACHE, `${model.sha256.slice(0, 16)}.json`);
  if (existsSync(out)) return out;
  const { extract } = await import("./ifc-extract.mjs");
  const { buildLatticeGraph } = await import(CORE + "lattice.js");
  const { findExits } = await import("./exits.mjs");
  const quiet = console.log;
  console.log = (...a) => (typeof a[0] === "string" && a[0].startsWith("[WEB-IFC]") ? null : quiet(...a));
  const geo = await extract(model.file, { withDoorFlags: true });
  console.log = quiet;
  const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75, maxEdgeRise: 0.35, columnGap: 0.9, maxPoints: 40000 });
  const { exits } = findExits(geo, built);
  writeFileSync(out, JSON.stringify({ nodes: b64(built.points), edges: b64(built.edges), blocked: b64(built.blocked), kinds: b64(built.kinds), exits }));
  return out;
}

function load(file) {
  const j = JSON.parse(readFileSync(file, "utf8"));
  return { nodes: from64(j.nodes, Float32Array), edges: from64(j.edges, Uint32Array), blocked: from64(j.blocked, Uint8Array), kinds: from64(j.kinds, Uint8Array), exits: j.exits };
}

// ===================================================== the legacy learner

/**
 * rl.js's qLearningPath training loop and rollouts, copied with knobs:
 * epsilonAt(ep) (default 0.2), alpha, episodes, maxSteps, timed (false: the
 * node-only state of Jabi et al. 2019 and Classic), and allow(v, t) (the
 * deadline shield), and priorShift (added to every untried value: +10 makes
 * the prior optimistic). At the defaults it reproduces rl.js (checked per run).
 */
function legacyQ(graph, start, exit, { ignite, episodes = 400, maxSteps = 200, epsilonAt = () => 0.2, alpha = 0.5, gamma = 0.99, seed = 1, blocked, timed: forceTimed = null, allow = null, priorShift = 0 }) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const distance = distancesFrom(graph, exit, { blocked });
  const r = rng(seed);
  const lengths = edgeLengths(graph);
  const sorted = Array.from(lengths).sort((a, b) => a - b);
  const unit = sorted[Math.floor(sorted.length / 2)] || 1;
  const slotCost = new Float64Array(neighbours.length);
  const source = new Int32Array(neighbours.length);
  for (let u = 0; u < n; u += 1) for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
    slotCost[k] = (lengths[edgeIds[k]] / unit) * STEP_COST;
    source[k] = u;
  }
  const slots = Array.from({ length: n }, (_, u) => {
    const out = [];
    for (let k = indptr[u]; k < indptr[u + 1]; k += 1) if (!blocked?.[edgeIds[k]]) out.push(k);
    return out;
  });
  let worst = 0;
  for (const d of distance) if (Number.isFinite(d) && d > worst) worst = d;
  const potential = new Float64Array(n);
  for (let u = 0; u < n; u += 1) potential[u] = (-(Number.isFinite(distance[u]) ? distance[u] : worst) / unit) * STEP_COST;
  const burnsAt = (v) => (ignite ? ignite.get(v) ?? Infinity : Infinity);
  const timed = forceTimed ?? Boolean(ignite && ignite.size);
  const horizon = maxSteps + 1;
  const prior = new Float64Array(neighbours.length);
  for (let k = 0; k < neighbours.length; k += 1) {
    const v = neighbours[k];
    prior[k] = -slotCost[k] + gamma * (v === exit ? 0 : potential[v]) - potential[source[k]] + priorShift;
  }
  const q = new Map();
  const key = (k, t) => (timed ? k * horizon + t : k);
  const value = (k, t) => q.get(key(k, t)) ?? prior[k];
  const optionsAt = (u, t) => (allow ? slots[u].filter((k) => allow(neighbours[k], t)) : slots[u]);
  const greedy = (u, t, avoid) => {
    let options = optionsAt(u, t);
    if (avoid) options = options.filter((k) => !avoid.has(neighbours[k]));
    if (!options.length) return -1;
    let top = -Infinity;
    let winners = [];
    for (const k of options) {
      const v = value(k, t);
      if (v > top) {
        top = v;
        winners = [k];
      } else if (v === top) winners.push(k);
    }
    return winners[Math.floor(r() * winners.length)];
  };
  const successes = [];
  let updates = 0;
  for (let ep = 0; ep < Math.max(1, episodes); ep += 1) {
    const epsilon = epsilonAt(ep, episodes);
    let state = start;
    let reached = 0;
    for (let t = 0; t < maxSteps; t += 1) {
      const options = optionsAt(state, t);
      if (!options.length) break;
      const k = r() < epsilon ? options[Math.floor(r() * options.length)] : greedy(state, t);
      const next = neighbours[k];
      let reward = -slotCost[k];
      let done = false;
      if (t >= burnsAt(next)) {
        reward = BURN_REWARD;
        done = true;
      } else if (next === exit) {
        reward = EXIT_REWARD;
        done = true;
        reached = 1;
      }
      const last = timed && t === maxSteps - 1;
      const phiNext = done || last ? 0 : potential[next];
      const shaped = reward + gamma * phiNext - potential[state];
      let target = shaped;
      if (!done && !last) {
        let bestNext = -Infinity;
        for (const j of optionsAt(next, t + 1)) bestNext = Math.max(bestNext, value(j, t + 1));
        target += gamma * (Number.isFinite(bestNext) ? bestNext : 0);
      }
      const current = value(k, t);
      q.set(key(k, t), current + alpha * (target - current));
      updates += 1;
      state = next;
      if (done) break;
    }
    successes.push(reached);
  }
  // The drawn path (no revisits, random ties), as rl.js.
  const path = [start];
  const visited = new Set([start]);
  let state = start, reachedExit = false, burned = false, drawnEnd = "steps", skipped = 0;
  for (let t = 0; t < maxSteps; t += 1) {
    const k = greedy(state, t, visited);
    if (k < 0) {
      drawnEnd = "stuck";
      break;
    }
    const next = neighbours[k];
    path.push(next);
    visited.add(next);
    if (t >= burnsAt(next)) {
      burned = true;
      drawnEnd = "burn";
      break;
    }
    if (next === exit) {
      reachedExit = true;
      drawnEnd = "exit";
      break;
    }
    state = next;
  }
  // The policy walk (revisits allowed, ties to the first), as rl.js scores it.
  // It runs after the drawing, so it draws nothing from the generator.
  let walkReached = false, walkBurned = false, walkMoves = null, policyReturn = 0;
  const walkSeen = new Set([start]);
  let walkRevisits = 0, walkTies = 0;
  for (let t = 0, u = start, discount = 1; t < maxSteps; t += 1) {
    let k = -1, top = -Infinity;
    let ties = 0;
    for (const j of optionsAt(u, t)) {
      const x = value(j, t);
      if (x > top) {
        top = x;
        k = j;
        ties = 0;
      } else if (x === top) ties += 1;
    }
    if (k < 0) break;
    if (ties) walkTies += 1;
    const v = neighbours[k];
    if (walkSeen.has(v)) walkRevisits += 1;
    walkSeen.add(v);
    // Where the drawing went another way than the policy at the same step.
    if (t < path.length - 1 && path[t + 1] !== v && path[t] === u) skipped += 1;
    if (t >= burnsAt(v)) {
      policyReturn += discount * BURN_REWARD;
      walkBurned = true;
      break;
    }
    if (v === exit) {
      policyReturn += discount * EXIT_REWARD;
      walkReached = true;
      walkMoves = t + 1;
      break;
    }
    policyReturn -= discount * slotCost[k];
    discount *= gamma;
    u = v;
  }
  const tail = successes.slice(-Math.max(1, Math.floor(successes.length / 5)));
  // The first step at which the drawing leaves the policy walk.
  return {
    path,
    reachedExit,
    burned,
    drawnEnd,
    skipped,
    walkRevisits,
    walkTies,
    moves: reachedExit ? path.length - 1 : null,
    walkReached,
    walkBurned,
    walkMoves,
    policyReturn,
    successRate: tail.reduce((s, x) => s + x, 0) / tail.length,
    tableSize: q.size,
    updates,
    trained: (u, v, t) => {
      for (const k of slots[u]) if (neighbours[k] === v) return q.has(key(k, t));
      return false;
    },
  };
}

/**
 * The exact optimum of rl.js's reward (backward induction over the step
 * limit, as rl.js computes optimalReturn), and the walk of an optimal policy
 * (ties to the first move): does the best return escape?
 */
function legacyOptimum(graph, start, exit, { ignite, maxSteps, gamma = 0.99, blocked }) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const lengths = edgeLengths(graph);
  const sorted = Array.from(lengths).sort((a, b) => a - b);
  const unit = sorted[Math.floor(sorted.length / 2)] || 1;
  const burn = new Float64Array(n).fill(Infinity);
  if (ignite) for (const [v, s] of ignite) burn[v] = s;
  const first = new Int32Array(n + 1);
  for (let u = 0; u < n; u += 1) {
    let c = 0;
    for (let k = indptr[u]; k < indptr[u + 1]; k += 1) if (!blocked?.[edgeIds[k]]) c += 1;
    first[u + 1] = first[u] + c;
  }
  const head = new Int32Array(first[n]);
  const cost = new Float64Array(first[n]);
  for (let u = 0, i = 0; u < n; u += 1) for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
    if (blocked?.[edgeIds[k]]) continue;
    head[i] = neighbours[k];
    cost[i] = (lengths[edgeIds[k]] / unit) * STEP_COST;
    i += 1;
  }
  // Nodes within t links of the start, as rl.js.
  const order = new Int32Array(n);
  const hops = new Int32Array(n).fill(-1);
  let reached = 1;
  order[0] = start;
  hops[start] = 0;
  for (let i = 0; i < reached; i += 1) {
    const u = order[i];
    if (hops[u] + 1 >= maxSteps) continue;
    for (let j = first[u]; j < first[u + 1]; j += 1) {
      const v = head[j];
      if (hops[v] >= 0) continue;
      hops[v] = hops[u] + 1;
      order[reached++] = v;
    }
  }
  const within = new Int32Array(Math.max(1, maxSteps));
  for (let i = 0; i < reached; i += 1) within[hops[order[i]]] = i + 1;
  for (let t = 1; t < within.length; t += 1) within[t] = Math.max(within[t], within[t - 1]);
  const layers = new Array(maxSteps + 1);
  layers[maxSteps] = new Float64Array(n);
  for (let t = maxSteps - 1; t >= 0; t -= 1) {
    const ahead = layers[t + 1];
    const here = new Float64Array(n);
    for (let q = 0; q < within[t]; q += 1) {
      const u = order[q];
      let best = first[u] < first[u + 1] ? -Infinity : 0;
      for (let i = first[u]; i < first[u + 1]; i += 1) {
        const v = head[i];
        const value = t >= burn[v] ? BURN_REWARD : v === exit ? EXIT_REWARD : -cost[i] + gamma * ahead[v];
        if (value > best) best = value;
      }
      here[u] = best;
    }
    layers[t] = here;
  }
  let outcome = "horizon", moves = null;
  const walk = [start];
  for (let t = 0, u = start; t < maxSteps; t += 1) {
    let best = -Infinity, choice = -1, c = 0;
    for (let i = first[u]; i < first[u + 1]; i += 1) {
      const v = head[i];
      const value = t >= burn[v] ? BURN_REWARD : v === exit ? EXIT_REWARD : -cost[i] + gamma * layers[t + 1][v];
      if (value > best) {
        best = value;
        choice = v;
        c = i;
      }
    }
    if (choice < 0) {
      outcome = "stuck";
      break;
    }
    walk.push(choice);
    if (t >= burn[choice]) {
      outcome = "burn";
      break;
    }
    if (choice === exit) {
      outcome = "exit";
      moves = t + 1;
      break;
    }
    u = choice;
    void c;
  }
  return { value: layers[0][start], outcome, moves, walk };
}

// ============================================================= scenarios

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
    // Hop distance of every node from the route, along open links.
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
      out.push({ start, exit, route, fire, delta: delta[fire], bin: binLabel(bin), routeHops: route.length - 1 });
    }
  }
  return out;
}

// ================================================================= run

const timeIt = (fn) => {
  const t = performance.now();
  const out = fn();
  return [out, performance.now() - t];
};

const EPS = {
  "eps 0.05": () => 0.05,
  "eps 0 (greedy)": () => 0,
  "eps decay 1 to 0.05": (ep, total) => Math.max(0.05, 1 - (0.95 * ep) / Math.max(1, total - 1)),
};

function runScenario(graph, exits, sc, speed, seed, knobs) {
  const blocked = graph.blocked;
  const H = APP.maxSteps;
  const ig = safe.ignitionSteps(graph, sc.fire, { fireSpeed: speed, maxSteps: H, blocked });
  const ignite = new Map();
  ig.forEach((s, v) => Number.isFinite(s) && ignite.set(v, s));
  const row = { start: sc.start, fire: sc.fire, delta: sc.delta, bin: sc.bin, speed, routeHops: sc.routeHops, seed };
  // Is the fire-free route cut, and by how much does it beat or lose the race?
  let slack = Infinity;
  for (let i = 1; i < sc.route.length; i += 1) slack = Math.min(slack, Math.ceil(ig[sc.route[i]]) - i);
  row.routeSlack = Number.isFinite(slack) ? slack : null;
  row.routeCut = slack < 0;

  const [esc1, escMs] = timeIt(() => earliestEscape(graph, sc.start, sc.exit, ignite, H, blocked));
  row.escape = esc1;
  const anyExit = safe.earliestSafeEscape(graph, sc.start, { exits, ignite: ig, horizon: H, blocked });
  row.escapeAny = anyExit ? anyExit.moves : null;
  row.escapeMs = round(escMs, 3);

  // The shipped method, exactly as the app runs it.
  const [shipped, shippedMs] = timeIt(() => qLearningPath(graph, sc.start, sc.exit, { ignite, episodes: APP.episodes, maxSteps: H, seed, blocked }));
  row.shipped = { reached: shipped.reachedExit, burned: shipped.burned, moves: shipped.reachedExit ? shipped.path.length - 1 : null, length: shipped.reachedExit ? round(shipped.length, 2) : null, policyReturn: round(shipped.policyReturn, 4), optimalReturn: round(shipped.optimalReturn, 4), successRate: round(shipped.successRate, 3), ms: round(shippedMs, 1) };

  // The copy at the defaults must reproduce it.
  const base = legacyQ(graph, sc.start, sc.exit, { ignite, episodes: APP.episodes, maxSteps: H, seed, blocked });
  row.copyMatches = base.path.join(",") === shipped.path.join(",") && Math.abs(base.policyReturn - shipped.policyReturn) < 1e-9;
  row.legacy = { walkReached: base.walkReached, walkBurned: base.walkBurned, walkMoves: base.walkMoves, tableSize: base.tableSize, updates: base.updates, drawnEnd: base.drawnEnd, departs: base.skipped, walkRevisits: base.walkRevisits, walkTies: base.walkTies };

  // The exact optimum of the shipped reward: does it escape?
  const opt = legacyOptimum(graph, sc.start, sc.exit, { ignite, maxSteps: H, blocked });
  row.optimum = { outcome: opt.outcome, moves: opt.moves, value: round(opt.value, 4) };
  // How much of the optimal walk's (move, step) pairs training ever updated.
  if (opt.walk.length > 1) {
    let hit = 0;
    for (let t = 0; t + 1 < opt.walk.length; t += 1) if (base.trained(opt.walk[t], opt.walk[t + 1], t)) hit += 1;
    row.optimum.trainedShare = round(hit / (opt.walk.length - 1), 3);
  }
  row.legacy.walkOutcome = base.walkReached ? "exit" : base.walkBurned ? "burn" : "horizon";
  if (Math.abs(opt.value - shipped.optimalReturn) > 1e-9) row.optimumMismatch = true;

  // The new method, to the one exit and to any exit, and the arrival mask.
  const D1 = safe.hopDeadlines(graph, { exits: [sc.exit], ignite: ig, horizon: H, blocked });
  const [s1, s1Ms] = timeIt(() => safe.learnSafeEscape(graph, sc.start, { exits: [sc.exit], ignite: ig, horizon: H, blocked, trials: APP.episodes }));
  row.safe = { reached: s1.reachedExit, moves: s1.moves, length: s1.reachedExit ? round(pathLength(graph, s1.path), 2) : null, trials: s1.trials, certified: s1.certified, values: s1.values.size, ms: round(s1Ms, 2) };
  const exact1 = safe.earliestSafeEscape(graph, sc.start, { exits: [sc.exit], ignite: ig, horizon: H, blocked });
  row.exactLength = exact1 ? round(pathLength(graph, exact1.path), 2) : null;
  const fireFree = distancesFrom(graph, sc.start, { blocked })[sc.exit];
  row.fireFreeLength = round(fireFree, 2);
  const nb = safe.learnSafeEscape(graph, sc.start, { exits: [sc.exit], ignite: ig, horizon: H, blocked, trials: APP.episodes, backward: false });
  row.safeNoBackward = { trials: nb.trials, certified: nb.certified, moves: nb.moves };
  const nm = safe.learnSafeEscape(graph, sc.start, { exits, ignite: ig, horizon: H, blocked, trials: APP.episodes, monotone: false });
  const [long, longMs] = timeIt(() => safe.learnSafeEscape(graph, sc.start, { exits, ignite: ig, horizon: H, blocked, trials: 20000 }));
  row.safeAnyLong = { trials: long.trials, certified: long.certified, moves: long.moves, ms: round(longMs, 1) };
  const nmLong = safe.learnSafeEscape(graph, sc.start, { exits, ignite: ig, horizon: H, blocked, trials: 20000, monotone: false });
  row.safeAnyLongNoMonotone = { trials: nmLong.trials, certified: nmLong.certified };
  row.safeAnyNoMonotone = { trials: nm.trials, certified: nm.certified, moves: nm.moves };
  const [sAll, sAllMs] = timeIt(() => safe.learnSafeEscape(graph, sc.start, { exits, ignite: ig, horizon: H, blocked, trials: APP.episodes }));
  row.safeAny = { reached: sAll.reachedExit, moves: sAll.moves, trials: sAll.trials, certified: sAll.certified, ms: round(sAllMs, 2) };
  const arr = safe.learnSafeEscape(graph, sc.start, { exits: [sc.exit], ignite: ig, horizon: H, blocked, trials: APP.episodes, shield: "arrival" });
  row.arrival = { reached: arr.reachedExit, moves: arr.moves, trials: arr.trials, certified: arr.certified, firstTrial: arr.curve[0] != null };
  const [full, fullMs] = timeIt(() => safe.trainSafeRl(graph, { startPoint: point(graph, sc.start), exitPoint: point(graph, sc.exit), firePoint: point(graph, sc.fire), trials: APP.episodes, maxSteps: H, fireSpeed: speed }));
  row.safeFullMs = round(fullMs, 2);
  row.snapSame = route.snap(graph, point(graph, sc.start)) === sc.start && route.snap(graph, point(graph, sc.fire)) === sc.fire && route.snap(graph, point(graph, sc.exit)) === sc.exit;
  row.safeFullMatches = full.moves === s1.moves;

  if (!knobs) return row;
  // One knob at a time, the copy of the shipped learner.
  const variants = {
    "episodes 100": { episodes: 100 },
    "episodes 1000": { episodes: 1000 },
    "episodes 4000": { episodes: 4000 },
    "eps 0.05": { epsilonAt: EPS["eps 0.05"] },
    "eps 0 (greedy)": { epsilonAt: EPS["eps 0 (greedy)"] },
    "eps decay 1 to 0.05": { epsilonAt: EPS["eps decay 1 to 0.05"] },
    "alpha 0.1": { alpha: 0.1 },
    "alpha 1": { alpha: 1 },
    "optimistic prior (+10)": { priorShift: EXIT_REWARD },
    "node-only state (Jabi 2019)": { timed: false },
    "deadline shield": { allow: (w, t) => t + 1 <= D1[w] },
    "deadline shield, eps decay, 4000 episodes": { allow: (w, t) => t + 1 <= D1[w], epsilonAt: EPS["eps decay 1 to 0.05"], episodes: 4000 },
  };
  if (QUICK) for (const k of ["episodes 100", "episodes 1000", "alpha 0.1", "eps 0.05", "deadline shield, eps decay, 4000 episodes"]) delete variants[k];
  row.knobs = {};
  for (const [name, opts] of Object.entries(variants)) {
    const [r, ms] = timeIt(() => legacyQ(graph, sc.start, sc.exit, { ignite, episodes: APP.episodes, maxSteps: H, seed, blocked, ...opts }));
    row.knobs[name] = { reached: r.reachedExit, burned: r.burned, moves: r.moves, walkReached: r.walkReached, ms: round(ms, 1) };
  }
  // The step limit changes the problem, so its escape is recomputed.
  for (const steps of [100, 400]) {
    const ig2 = safe.ignitionSteps(graph, sc.fire, { fireSpeed: speed, maxSteps: steps, blocked });
    const ignite2 = new Map();
    ig2.forEach((s, v) => Number.isFinite(s) && ignite2.set(v, s));
    const r = legacyQ(graph, sc.start, sc.exit, { ignite: ignite2, episodes: APP.episodes, maxSteps: steps, seed, blocked });
    row.knobs[`steps ${steps}`] = { reached: r.reachedExit, burned: r.burned, moves: r.moves, walkReached: r.walkReached, escape: earliestEscape(graph, sc.start, sc.exit, ignite2, steps, blocked) };
  }
  return row;
}

function pathLength(graph, path) {
  const P = graph.nodes;
  let length = 0;
  for (let i = 1; i < path.length; i += 1) {
    const a = path[i - 1] * 3, b = path[i] * 3;
    length += Math.hypot(P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]);
  }
  return length;
}

function point(graph, v) {
  return [graph.nodes[v * 3], graph.nodes[v * 3 + 1], graph.nodes[v * 3 + 2]];
}

/** Every-start costs on one fire: the deadline map and the exact escape field. */
function fieldCosts(graph, exits, fire) {
  const ig = safe.ignitionSteps(graph, fire, { fireSpeed: 1, maxSteps: APP.maxSteps, blocked: graph.blocked });
  const [, dMs] = timeIt(() => safe.hopDeadlines(graph, { exits, ignite: ig, horizon: APP.maxSteps, blocked: graph.blocked }));
  const [field, fMs] = timeIt(() => safe.escapeField(graph, { exits, ignite: ig, horizon: APP.maxSteps, blocked: graph.blocked }));
  let escapable = 0;
  for (const s of field.steps) if (s >= 0) escapable += 1;
  return { deadlinesMs: round(dMs, 2), escapeFieldMs: round(fMs, 1), escapableShare: round(escapable / field.steps.length, 3) };
}

// ============================================================= summary

function summarize(results) {
  const rows = results.models.flatMap((m) => m.rows.map((r) => ({ ...r, model: m.label })));
  const lines = [];
  const pct = (a, b) => (b ? `${((100 * a) / b).toFixed(1)}%` : "-");
  const table = (title, head, body) => {
    lines.push("", `### ${title}`, "", `| ${head.join(" | ")} |`, `|${head.map(() => "---").join("|")}|`, ...body.map((r) => `| ${r.join(" | ")} |`));
  };
  const esc = rows.filter((r) => r.escape != null);
  const failures = (list, pick) => list.filter((r) => !pick(r)).length;

  table("Models", ["Model", "Nodes", "Links", "Exits", "Scenarios", "With an escape (one exit)"], results.models.map((m) => [m.label, m.nodes, m.edges, m.exits, m.rows.length, m.rows.filter((r) => r.escape != null).length]));

  // How often an escape exists at all, by distance and speed.
  const byBin = [...new Set(rows.map((r) => r.bin))];
  table("Scenarios where an escape to the chosen exit exists at all (earliest-escape search)", ["Fire to route (links)", ...SPEEDS.map((sp) => `speed ${sp}`), "all"], byBin.map((bin) => {
    const cells = SPEEDS.map((sp) => {
      const l = rows.filter((r) => r.bin === bin && r.speed === sp);
      return `${pct(l.filter((r) => r.escape != null).length, l.length)} (${l.length})`;
    });
    const l = rows.filter((r) => r.bin === bin);
    return [bin, ...cells, `${pct(l.filter((r) => r.escape != null).length, l.length)} (${l.length})`];
  }));
  {
    const all = rows.length, none = rows.filter((r) => r.escape == null).length;
    const shownFail = rows.filter((r) => !r.shipped.reached).length;
    lines.push("", `Of ${all} runs the app would show a path that does not reach the exit in ${shownFail} (${pct(shownFail, all)}); in ${none} of them (${pct(none, shownFail)} of those) no escape exists, and the note says so.`);
  }
  // Failure of the shipped method by distance and speed (escape exists).
  table("Shipped Q-learning: share of scenarios with an escape where the drawn path does not reach the exit", ["Fire to route (links)", ...SPEEDS.map((s) => `speed ${s}`), "all"], byBin.map((bin) => {
    const cells = SPEEDS.map((s) => {
      const l = esc.filter((r) => r.bin === bin && r.speed === s);
      return `${pct(failures(l, (r) => r.shipped.reached), l.length)} (${l.length})`;
    });
    const l = esc.filter((r) => r.bin === bin);
    return [bin, ...cells, `${pct(failures(l, (r) => r.shipped.reached), l.length)} (${l.length})`];
  }));
  // The main rates with 95% Wilson intervals, and proximity pooled.
  {
    const wilson = (k, n) => {
      if (!n) return "-";
      const z = 1.96, p = k / n, d = 1 + (z * z) / n, c = p + (z * z) / (2 * n);
      const e = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
      return `${((100 * (c - e)) / d).toFixed(1)} to ${((100 * (c + e)) / d).toFixed(1)}%`;
    };
    const near = ["0", "1-2", "3-5"];
    const rate = (label, l, bad) => {
      const k = l.filter(bad).length;
      return [label, `${k} of ${l.length}`, pct(k, l.length), wilson(k, l.length)];
    };
    const closeFast = rows.filter((r) => (r.bin === "0" || r.bin === "1-2") && r.speed >= 1);
    table("Rates with 95% Wilson intervals", ["Rate", "Count", "Share", "95% interval"], [
      rate("shipped drawn path fails, escape exists", esc, (r) => !r.shipped.reached),
      rate("  fire within 5 links of the route", esc.filter((r) => near.includes(r.bin)), (r) => !r.shipped.reached),
      rate("  fire more than 5 links from the route", esc.filter((r) => !near.includes(r.bin)), (r) => !r.shipped.reached),
      rate("shipped policy walk fails, escape exists", esc, (r) => !r.legacy.walkReached),
      rate("drawing fails although the policy walk escapes", esc, (r) => r.legacy.walkReached && !r.shipped.reached),
      rate("an escape exists, fire within 2 links and speed 1 or 2", closeFast, (r) => r.escape != null),
      rate("no escape to the chosen exit, but one to another exit", rows, (r) => r.escape == null && r.escapeAny != null),
      rate("reward optimum escapes, but later than the earliest escape", esc.filter((r) => r.optimum.outcome === "exit"), (r) => r.optimum.moves !== r.escape),
    ]);
  }
  const cut = esc.filter((r) => r.routeCut), clear = esc.filter((r) => !r.routeCut);
  table("Shipped Q-learning by whether the fire cuts the fire-free shortest route", ["Route", "Scenarios", "Drawn path fails", "Policy walk fails", "Reward optimum does not escape"], [
    ["cut by the fire", cut.length, pct(failures(cut, (r) => r.shipped.reached), cut.length), pct(failures(cut, (r) => r.legacy.walkReached), cut.length), pct(failures(cut, (r) => r.optimum.outcome === "exit"), cut.length)],
    ["not cut", clear.length, pct(failures(clear, (r) => r.shipped.reached), clear.length), pct(failures(clear, (r) => r.legacy.walkReached), clear.length), pct(failures(clear, (r) => r.optimum.outcome === "exit"), clear.length)],
  ]);

  // Decomposition.
  const model = esc.filter((r) => r.optimum.outcome !== "exit");
  const train = esc.filter((r) => r.optimum.outcome === "exit" && !r.legacy.walkReached);
  const rollout = esc.filter((r) => r.optimum.outcome === "exit" && r.legacy.walkReached && !r.shipped.reached);
  const ok = esc.filter((r) => r.shipped.reached);
  const failed = esc.length - ok.length;
  const outcomes = (l) => ["burn", "horizon", "stuck"].map((o) => `${o} ${l.filter((r) => r.optimum.outcome === o).length}`).join(", ");
  table("Where the shipped method's failures come from (scenarios with an escape)", ["Class", "Scenarios", "Share of failures", "Test"], [
    ["model: the reward's exact optimum does not escape", model.length, pct(model.filter((r) => !r.shipped.reached).length, failed), `optimum walk ends: ${outcomes(model)}`],
    ["training: the optimum escapes, the learned policy does not", train.length, pct(train.filter((r) => !r.shipped.reached).length, failed), "policy walk (revisits allowed) fails"],
    ["rollout: the policy escapes, the drawn path does not", rollout.length, pct(rollout.length, failed), "no-revisit drawing fails"],
    ["success", ok.length, "-", "drawn path reaches the exit"],
  ]);

  // How the drawn path fails.
  {
    const rolloutFails = esc.filter((r) => r.legacy.walkReached && !r.shipped.reached);
    const ends = (l) => ["burn", "stuck", "steps"].map((o) => `${o} ${l.filter((r) => r.legacy.drawnEnd === o).length}`).join(", ");
    table("How the drawn path fails where the policy itself escapes", ["Measure", "Value"], [
      ["scenarios", rolloutFails.length],
      ["drawn path ends", ends(rolloutFails)],
      ["the policy walk revisits a node (median revisits)", `${rolloutFails.filter((r) => r.legacy.walkRevisits > 0).length} (${median(rolloutFails.map((r) => r.legacy.walkRevisits))})`],
      ["steps where the drawing and the policy walk part (median)", median(rolloutFails.map((r) => r.legacy.departs))],
      ["successful drawn paths whose policy walk revisits", `${ok.filter((r) => r.legacy.walkRevisits > 0).length} of ${ok.length}`],
      ["policy walks with tied best moves: failing drawings / successful ones", `${rolloutFails.filter((r) => r.legacy.walkTies > 0).length} / ${ok.filter((r) => r.legacy.walkTies > 0).length}`],
    ]);
  }
  // Knobs on the training failures.
  const knobRows = esc.filter((r) => r.knobs);
  if (knobRows.length) {
    const names = Object.keys(knobRows[0].knobs);
    const baseFail = failures(knobRows, (r) => r.shipped.reached);
    table(`One knob at a time (${knobRows.length} scenarios with an escape where the knobs ran; shipped fails ${baseFail})`, ["Variant", "Drawn path fails", "Policy walk fails", "Fixed (of shipped failures)", "Broken (of shipped successes)"], [
      ["shipped (400 episodes, eps 0.2, alpha 0.5, 200 steps)", pct(baseFail, knobRows.length), pct(failures(knobRows, (r) => r.legacy.walkReached), knobRows.length), "-", "-"],
      ...names.map((k) => {
        const l = knobRows.filter((r) => (k.startsWith("steps") ? r.knobs[k].escape != null : true));
        return [k, pct(failures(l, (r) => r.knobs[k].reached), l.length), pct(failures(l, (r) => r.knobs[k].walkReached), l.length), l.filter((r) => !r.shipped.reached && r.knobs[k].reached).length, l.filter((r) => r.shipped.reached && !r.knobs[k].reached).length];
      }),
    ]);
  }

  // Old against new.
  const gap = (l, pick) => median(l.map((r) => (pick(r) != null && r.escape != null ? pick(r) - r.escape : null)));
  const optimalShare = (l, pick) => pct(l.filter((r) => pick(r) != null && pick(r) === r.escape).length, l.length);
  table("Old against new, to the same one exit (scenarios with an escape)", ["Method", "Escapes", "Earliest escape (moves = exact)", "Median extra moves when escaping", "Median run time (ms)"], [
    ["shipped Q-learning (400 episodes)", pct(ok.length, esc.length), optimalShare(esc, (r) => r.shipped.moves), gap(esc, (r) => r.shipped.moves), round(median(esc.map((r) => r.shipped.ms)), 1)],
    ["arrival mask + learning real-time search", pct(esc.filter((r) => r.arrival.reached).length, esc.length), optimalShare(esc, (r) => r.arrival.moves), gap(esc, (r) => r.arrival.moves), "-"],
    ["deadline shield + learning real-time search (new)", pct(esc.filter((r) => r.safe.reached).length, esc.length), optimalShare(esc, (r) => r.safe.moves), gap(esc, (r) => r.safe.moves), round(median(esc.map((r) => r.safe.ms)), 2)],
    ["exact earliest escape (breadth first)", "100.0%", "100.0%", 0, round(median(esc.map((r) => r.escapeMs)), 3)],
  ]);
  const lengthRatio = (l, pick) => round(median(l.map((r) => (pick(r) != null && r.fireFreeLength ? pick(r) / r.fireFreeLength : null))), 3);
  table("Objectives and lengths (scenarios with an escape, one exit)", ["Measure", "Value"], [
    ["reward optimum escapes in the earliest number of moves", pct(esc.filter((r) => r.optimum.moves === r.escape).length, esc.length)],
    ["median extra moves of the reward optimum, when it escapes", gap(esc, (r) => r.optimum.moves)],
    ["median length over fire-free shortest: shipped drawn path", lengthRatio(esc, (r) => r.shipped.length)],
    ["median length over fire-free shortest: new method", lengthRatio(esc, (r) => r.safe.length)],
    ["median length over fire-free shortest: exact earliest route", lengthRatio(esc, (r) => r.exactLength)],
    ["share of the optimal walk's (move, step) pairs ever trained, shipped, when it fails", round(median(esc.filter((r) => !r.shipped.reached && r.optimum.trainedShare != null).map((r) => r.optimum.trainedShare)), 3)],
    ["the same when it succeeds", round(median(esc.filter((r) => r.shipped.reached && r.optimum.trainedShare != null).map((r) => r.optimum.trainedShare)), 3)],
    ["policy walk outcome when it fails: burn / horizon", `${esc.filter((r) => r.legacy.walkOutcome === "burn").length} / ${esc.filter((r) => r.legacy.walkOutcome === "horizon").length}`],
  ]);
  const anyEsc = rows.filter((r) => r.escapeAny != null);
  table("To any exit (the new method's default)", ["Measure", "Value"], [
    ["scenarios with an escape to the one exit", esc.length],
    ["scenarios with an escape to some exit", anyEsc.length],
    ["new method escapes (any exit)", pct(anyEsc.filter((r) => r.safeAny.reached).length, anyEsc.length)],
    ["new method certified optimal", pct(anyEsc.filter((r) => r.safeAny.certified).length, anyEsc.length)],
    ["new method moves = exact earliest (any exit)", pct(anyEsc.filter((r) => r.safeAny.moves === r.escapeAny).length, anyEsc.length)],
    ["median trials to the certificate", median(anyEsc.map((r) => r.safeAny.trials))],
    ["one exit: trials = 1 / at most 10", `${esc.filter((r) => r.safe.trials === 1).length} / ${esc.filter((r) => r.safe.trials <= 10).length} of ${esc.length}`],
    ["one exit: values stored by the learner, median / largest (shipped table entries, median)", `${median(esc.map((r) => r.safe.values))} / ${Math.max(...esc.map((r) => r.safe.values))} (${median(esc.map((r) => r.legacy.tableSize))})`],
    ["one exit: median / largest trials, with the backward pass", `${median(esc.map((r) => r.safe.trials))} / ${Math.max(...esc.map((r) => r.safe.trials))}`],
    ["one exit: median / largest trials, without it", `${median(esc.map((r) => r.safeNoBackward.trials))} / ${Math.max(...esc.map((r) => r.safeNoBackward.trials))}`],
    ["one exit: certified without the backward pass (400 trials)", pct(esc.filter((r) => r.safeNoBackward.certified).length, esc.length)],
    ["any exit: certified without sharing bounds across steps (400 trials)", pct(anyEsc.filter((r) => r.safeAnyNoMonotone.certified).length, anyEsc.length)],
    ["any exit: mean trials with / without sharing bounds across steps", `${round(anyEsc.reduce((a, r) => a + r.safeAny.trials, 0) / anyEsc.length, 2)} / ${round(anyEsc.reduce((a, r) => a + r.safeAnyNoMonotone.trials, 0) / anyEsc.length, 2)}`],
    ["any exit: certified within 400 trials", pct(anyEsc.filter((r) => r.safeAny.certified).length, anyEsc.length)],
    ["largest number of trials", Math.max(...anyEsc.map((r) => r.safeAny.trials))],
    ["first trial of the arrival mask trapped (one exit)", pct(esc.filter((r) => !r.arrival.firstTrial).length, esc.length)],
    ["any exit, up to 20,000 trials: certified", pct(anyEsc.filter((r) => r.safeAnyLong.certified).length, anyEsc.length)],
    ["any exit, up to 20,000 trials: median / 90th percentile / largest trials", `${median(anyEsc.map((r) => r.safeAnyLong.trials))} / ${quantile(anyEsc.map((r) => r.safeAnyLong.trials), 0.9)} / ${Math.max(...anyEsc.map((r) => r.safeAnyLong.trials))}`],
    ["any exit, up to 20,000 trials: median / largest time (ms)", `${median(anyEsc.map((r) => r.safeAnyLong.ms))} / ${Math.max(...anyEsc.map((r) => r.safeAnyLong.ms))}`],
    ["any exit, up to 20,000 trials, bounds not shared across steps: certified / mean trials", `${pct(anyEsc.filter((r) => r.safeAnyLongNoMonotone.certified).length, anyEsc.length)} / ${round(anyEsc.reduce((a, r) => a + r.safeAnyLongNoMonotone.trials, 0) / anyEsc.length, 1)}`],
    ["any exit, up to 20,000 trials, bounds shared: mean trials", round(anyEsc.reduce((a, r) => a + r.safeAnyLong.trials, 0) / anyEsc.length, 1)],
  ]);
  table("Run time per model (median over scenarios, ms)", ["Model", "Nodes", "Shipped (400 episodes, with its optimum)", "New, learning", "New, whole request", "Exact escape", "Deadline map", "Escape field (every start)"], results.models.map((m) => [
    m.label,
    m.nodes,
    round(median(m.rows.map((r) => r.shipped.ms)), 1),
    round(median(m.rows.map((r) => r.safe.ms)), 2),
    round(median(m.rows.map((r) => r.safeFullMs)), 1),
    round(median(m.rows.map((r) => r.escapeMs)), 3),
    m.field?.deadlinesMs ?? "-",
    m.field?.escapeFieldMs ?? "-",
  ]));
  const mism = rows.filter((r) => !r.copyMatches).length;
  const optMism = rows.filter((r) => r.optimumMismatch).length;
  const fullMism = rows.filter((r) => !r.safeFullMatches).length;
  const snapDiff = rows.filter((r) => !r.safeFullMatches && !r.snapSame).length;
  lines.push("", `Checks: the copied learner reproduced rl.js in ${rows.length - mism} of ${rows.length} runs; the optimum walk's value matched rl.js's optimalReturn in ${rows.length - optMism} of ${rows.length}; trainSafeRl matched learnSafeEscape in ${rows.length - fullMism} of ${rows.length} (${snapDiff} of the others snap a node's own position to another node).`);
  return lines.join("\n");
}

// ================================================================ main

async function main() {
  const summaryFile = option("--summary", null);
  if (summaryFile) {
    console.log(summarize(JSON.parse(readFileSync(summaryFile, "utf8"))));
    return;
  }
  const models = modelList();
  if (flag("--prepare")) {
    for (const m of models) {
      const t = performance.now();
      const f = await prepare(m);
      const g = load(f);
      console.log(`${m.label.padEnd(20)} ${g.nodes.length / 3} nodes, ${g.exits.length} exits (${Math.round(performance.now() - t)} ms)`);
    }
    return;
  }
  const starts = Number(option("--starts", QUICK ? 2 : 4));
  const results = { date: new Date().toISOString().slice(0, 10), machine: { cpu: os.cpus()[0]?.model?.trim(), logicalCores: os.cpus().length, node: process.version }, settings: { ...APP, speeds: SPEEDS, bins: BINS.map(binLabel), starts, quick: QUICK, knobsAtSpeed: 1 }, models: [] };
  for (const m of models) {
    const f = await prepare(m);
    const g = load(f);
    const graph = { nodes: g.nodes, edges: g.edges, blocked: g.blocked, kinds: g.kinds };
    const exits = g.exits;
    const entry = { label: m.label, nodes: g.nodes.length / 3, edges: g.edges.length / 2, exits: exits.length, rows: [] };
    results.models.push(entry);
    if (!exits.length) continue;
    const list = scenarios(graph, exits, m.sha256, starts);
    const t0 = performance.now();
    let i = 0;
    for (const sc of list) {
      for (const speed of SPEEDS) {
        i += 1;
        entry.rows.push(runScenario(graph, exits, sc, speed, i, speed === 1));
      }
    }
    if (list.length) entry.field = fieldCosts(graph, exits, list[Math.floor(list.length / 2)].fire);
    console.log(`${m.label.padEnd(20)} ${entry.nodes} nodes, ${list.length} fires x ${SPEEDS.length} speeds, ${Math.round((performance.now() - t0) / 1000)} s`);
  }
  const out = option("--out", path.join(ROOT, "benchmark", "results", `${results.date}-rl.json`));
  writeFileSync(out, JSON.stringify(results));
  const digest = createHash("sha256").update(readFileSync(out)).digest("hex").slice(0, 12);
  console.log(`\nWrote ${path.relative(ROOT, out)} (sha256 ${digest})`);
  console.log(summarize(results));
}

await main();
