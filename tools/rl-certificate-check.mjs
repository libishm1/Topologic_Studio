/**
 * Does rlSafe.js's `certified` hold for the route it returns? Random small
 * graphs with random closing steps, the shielded learner against the exact
 * earliest escape. A result with `certified: true` must reach an exit in the
 * earliest number of moves (SafeEscape.certificate_optimal certifies a walk
 * whose length is at most an admissible value at the start).
 *
 *   node tools/rl-certificate-check.mjs [--cases 200000] [--seed 1] [--hard]
 *
 * Prints the counts and the first failing case in full (graph, closing
 * steps, start, exits, step limit, the learner's result and the exact one).
 */
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CORE = pathToFileURL(path.join(HERE, "..", "topologicpy-web-frontend", "src", "core")).href + "/";
const safe = await import(CORE + "rlSafe.js");

const argv = process.argv.slice(2);
const option = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : fallback;
};
const CASES = Number(option("--cases", 200000));
const SEED = Number(option("--seed", 1));
// --hard: up to 24 nodes and four exits, most exits sealed within a few steps.
const HARD = argv.includes("--hard");

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

/** A connected random graph on a jittered grid, with extra chords. */
function randomCase(r) {
  const n = 5 + Math.floor(r() * (HARD ? 20 : 10));
  const nodes = new Float32Array(n * 3);
  const jitter = HARD ? 1 : 0.2;
  for (let v = 0; v < n; v += 1) {
    nodes[v * 3] = (v % 4) * 0.5 + r() * jitter;
    nodes[v * 3 + 1] = 0;
    nodes[v * 3 + 2] = Math.floor(v / 4) * 0.5 + r() * jitter;
  }
  const edges = [];
  const has = new Set();
  const add = (a, b) => {
    if (a === b) return;
    const k = a < b ? `${a},${b}` : `${b},${a}`;
    if (has.has(k)) return;
    has.add(k);
    edges.push(a, b);
  };
  for (let v = 1; v < n; v += 1) add(v, Math.floor(r() * v)); // a random tree
  const chords = Math.floor(r() * n);
  for (let i = 0; i < chords; i += 1) add(Math.floor(r() * n), Math.floor(r() * n));
  const H = 3 + Math.floor(r() * 14);
  const ig = new Float64Array(n);
  for (let v = 0; v < n; v += 1) ig[v] = r() < 0.4 ? Infinity : Math.floor(r() * (H + 2)) - 0.5 * (r() < 0.5);
  const exitCount = 1 + Math.floor(r() * (HARD ? 4 : 2));
  const exits = [];
  while (exits.length < exitCount) {
    const x = Math.floor(r() * n);
    if (!exits.includes(x)) exits.push(x);
  }
  // Exits open at the first steps and sealed soon after: the learner's
  // potential counts an exit while it can be entered at step 1.
  if (HARD) for (const x of exits) if (r() < 0.6) ig[x] = 1 + Math.floor(r() * 4);
  let start = Math.floor(r() * n);
  while (exits.includes(start)) start = Math.floor(r() * n);
  return { graph: { nodes, edges: Uint32Array.from(edges), blocked: null }, ig, exits, start, H };
}

const variants = {
  "defaults (200 trials)": { trials: 200 },
  "1 trial": { trials: 1 },
  "3 trials": { trials: 3 },
  "no sharing across steps": { trials: 200, monotone: false },
  "no backward pass": { trials: 200, backward: false },
};
const counts = Object.fromEntries(Object.keys(variants).map((k) => [k, { cases: 0, escapable: 0, reached: 0, certified: 0, certifiedNotEarliest: 0, missed: 0, unsafe: 0 }]));
let first = null;
const r = rng(SEED);
for (let c = 0; c < CASES; c += 1) {
  const kase = randomCase(r);
  const { graph, ig, exits, start, H } = kase;
  const exact = safe.earliestSafeEscape(graph, start, { exits, ignite: ig, horizon: H });
  for (const [name, opts] of Object.entries(variants)) {
    const res = safe.learnSafeEscape(graph, start, { exits, ignite: ig, horizon: H, ...opts });
    const k = counts[name];
    k.cases += 1;
    if (exact) k.escapable += 1;
    if (res.reachedExit) k.reached += 1;
    if (exact && !res.reachedExit) k.missed += 1;
    if (res.certified) k.certified += 1;
    // Safe on the hop clock: each move at step t into w with t < ig(w).
    for (let i = 1; i < res.path.length; i += 1) if (!(i - 1 < ig[res.path[i]])) { k.unsafe += 1; break; }
    if (res.certified && (!exact || !res.reachedExit || res.moves !== exact.moves)) {
      k.certifiedNotEarliest += 1;
      if (!first) first = { variant: name, case: c, n: graph.nodes.length / 3, edges: Array.from(graph.edges), nodes: Array.from(graph.nodes).map((x) => Math.round(x * 1000) / 1000), ig: Array.from(ig), exits, start, H, learner: { moves: res.moves, path: res.path, certified: res.certified, trials: res.trials, curve: res.curve }, exact };
    }
  }
}
console.log(`cases ${CASES}, seed ${SEED}`);
console.table(counts);
if (first) console.log("first certified result that is not an earliest escape:\n" + JSON.stringify(first));
else console.log("no certified result that is not an earliest escape");
