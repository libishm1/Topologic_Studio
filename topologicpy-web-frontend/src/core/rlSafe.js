/**
 * Shielded learning on the time-expanded fire graph: the learned escape that
 * fixes what made rl.js's tabular Q-learning (after Jabi et al. 2019) fail
 * when the fire is close. The study, with its numbers, is
 * wiki/research/rl-egress-study-2026-11-05.md; the proofs are
 * formal/fireegress/FireEgress/SafeEscape.lean.
 *
 * The problem is rl.js's, on its clock. The evacuee moves along one open link
 * per step. The fire's hop front reaches node v at step ig(v) (fire.js
 * floodTimeline, divided by the fire speed); a move made at step t into v is
 * safe when t < ig(v), that is when it arrives at t + 1 <= open(v), with
 * open(v) = ceil(ig(v)) capped at the step limit H. The goal is
 * lexicographic: reach any exit safely within H steps, and as early as
 * possible. rl.js instead maximises a discounted sum of -0.1 per median link,
 * +10 at the exit and -10 in the fire, whose optimum need not escape at all
 * (the study's model-defect count).
 *
 * Three parts.
 * 1. The deadline map D (hopDeadlines): D(v) is the latest step at which an
 *    evacuee can stand at v and still reach an exit safely by step H, -1 if
 *    never. D(exit) = open(exit) and, for other nodes,
 *    D(v) = min(open(v), max over open links (v, w) of D(w) - 1): the
 *    max-min recursion of deadline.js on the hop clock, set from the exits
 *    outward, largest label first, in O(nodes + links + H).
 * 2. The shield: at (v, t) only moves into w with t + 1 <= D(w) are allowed.
 *    From any (v, t) with t <= D(v) every allowed move is safe on arrival,
 *    some move is allowed until an exit is reached, and every shielded walk
 *    reaches an exit by step H (SafeEscape.shield_safe, shield_nonempty,
 *    shield_reaches_exit). A move is masked only when no safe escape goes
 *    through it (shield_complete), so the shield removes no escape. With it
 *    even a random policy never fails when an escape exists; masking only
 *    the nodes untenable on arrival (mode "arrival") does not have this
 *    property, since a node can be tenable on arrival and sealed later.
 * 3. The learner (learnSafeEscape): learning real-time search on the
 *    time-expanded states (v, t) (LRTA*, Korf 1990; trial-based real-time
 *    dynamic programming, Barto, Bradtke & Singh 1995). A move costs 1 (one
 *    step) and an exit is worth 0. Unvisited values start at the hop
 *    distance to the nearest exit through nodes the deadline map keeps
 *    (D >= 0), which never exceeds the true number of steps left, so it is
 *    an admissible potential (SafeEscape.consistent_admissible) and, as
 *    Wiewiora (2003) shows for Q-values, the same as potential-based shaping
 *    with it (potential_init_eq_shaping): shaping from the deadline map.
 *    Trials are greedy inside the shield and update
 *    V(v, t) <- max(V(v, t), min over allowed w of 1 + V(w, t + 1)), with a
 *    backward pass over each finished trial. Values never exceed the exact
 *    ones (lrta_admissible, raise_admissible); a trial that takes no more
 *    steps than V(start, 0) had at its start is an earliest escape
 *    (certificate_optimal); a trial that changes no value passes that test
 *    (trial_no_change), and values only rise to a cap, so learning stops with
 *    the certificate after finitely many trials (bounded_raises). The trial
 *    budget can end it sooner; the result then says it is not certified.
 *
 * The exact reference (escapeValues, earliestSafeEscape) is finite-horizon
 * value iteration on the same time-expanded graph, which equals the earliest
 * safe arrival (SafeEscape.vi_eq_earliest); the breadth-first form is exact
 * because the fire only spreads, so arriving earlier never hurts.
 *
 * The values are kept per (node, step), not per node: under a spreading fire
 * one node can need different moves at different steps, and no table indexed
 * by the node alone is right for every start (SafeEscape.node_policy_fails).
 * One table serves every start: values learned from one start are lower
 * bounds for any other, so learning from several starts shares its work.
 */
import { Heap, csr, distancesFrom, edgeLengths, snap } from "./route.js";
import { floodTimeline } from "./fire.js";

/** A number of steps larger than any step limit: "never" or "no escape". */
export const NEVER = 0x3fffffff;
/** The step limit the page offers, as rl.js. */
export const SAFE_MAX_STEPS = 1000;
const CURVE_POINTS = 50;

/**
 * Ignition step of every node, as rl.js's trainRl builds it: the hop front of
 * fire.js's floodTimeline from `fireStart`, divided by `fireSpeed`; Infinity
 * where the fire does not arrive within the timeline.
 */
export function ignitionSteps(graph, fireStart, { fireSpeed = 1, maxSteps = 200, blocked = null } = {}) {
  const n = graph.nodes.length / 3;
  const ig = new Float64Array(n).fill(Infinity);
  if (fireStart == null || fireStart < 0) return ig;
  const timeline = floodTimeline(graph, fireStart, Math.ceil(maxSteps * fireSpeed) + 1, blocked);
  timeline.forEach((nodes, step) => nodes.forEach((v) => (ig[v] = step / fireSpeed)));
  return ig;
}

/** rl.js's ignition Map (or null), or an array, as a Float64Array. */
export function toIgnition(ignite, n) {
  if (ignite instanceof Float64Array && ignite.length === n) return ignite;
  const ig = new Float64Array(n).fill(Infinity);
  if (!ignite) return ig;
  if (ignite instanceof Map) for (const [v, s] of ignite) ig[v] = s;
  else for (let v = 0; v < n; v += 1) ig[v] = ignite[v];
  return ig;
}

/**
 * The last arrival step at which each node is tenable, capped at H: a move at
 * step t into v is safe iff t < ig(v), iff t + 1 <= ceil(ig(v)).
 */
export function openUntil(ig, horizon) {
  const out = new Int32Array(ig.length);
  for (let v = 0; v < ig.length; v += 1) {
    const s = ig[v];
    out[v] = Number.isFinite(s) ? Math.min(horizon, Math.max(0, Math.ceil(s))) : horizon;
  }
  return out;
}

function exitFlags(n, exits) {
  const isExit = new Uint8Array(n);
  for (const x of exits) if (x >= 0 && x < n) isExit[x] = 1;
  return isExit;
}

/** The open links, flat: from node v, to[first[v] .. first[v + 1]). */
export function openLinks(graph, blocked = null) {
  if (!blocked && graph._openLinks) return graph._openLinks;
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const first = new Int32Array(n + 1);
  for (let v = 0; v < n; v += 1) {
    let c = 0;
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) if (!blocked?.[edgeIds[k]]) c += 1;
    first[v + 1] = first[v] + c;
  }
  const to = new Int32Array(first[n]);
  const edge = new Int32Array(first[n]);
  for (let v = 0, i = 0; v < n; v += 1) {
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      if (blocked?.[edgeIds[k]]) continue;
      to[i] = neighbours[k];
      edge[i] = edgeIds[k];
      i += 1;
    }
  }
  const out = { n, first, to, edge };
  if (!blocked) graph._openLinks = out;
  return out;
}

/**
 * The deadline map on the hop clock (see the header): D(v), the latest step
 * at which an evacuee standing at v can still reach an exit safely by step
 * `horizon`, or -1. Labels are whole steps in [0, horizon], so a bucket per
 * step replaces the heap of deadline.js.
 */
export function hopDeadlines(graph, { exits, ignite = null, horizon, blocked = null }) {
  const { n, first, to } = openLinks(graph, blocked);
  const H = Math.max(0, Math.floor(horizon));
  const open = openUntil(toIgnition(ignite, n), H);
  const isExit = exitFlags(n, exits);
  const D = new Int32Array(n).fill(-1);
  const buckets = Array.from({ length: H + 1 }, () => []);
  for (let v = 0; v < n; v += 1) {
    if (!isExit[v]) continue;
    D[v] = open[v];
    buckets[open[v]].push(v);
  }
  const done = new Uint8Array(n);
  for (let L = H; L >= 1; L -= 1) {
    const bucket = buckets[L];
    for (let i = 0; i < bucket.length; i += 1) {
      const w = bucket[i];
      if (done[w] || D[w] !== L) continue;
      done[w] = 1;
      for (let j = first[w]; j < first[w + 1]; j += 1) {
        const v = to[j];
        if (done[v] || isExit[v]) continue;
        const candidate = Math.min(open[v], L - 1);
        if (candidate > D[v]) {
          D[v] = candidate;
          buckets[candidate].push(v);
        }
      }
    }
    buckets[L] = null;
  }
  return D;
}

/**
 * Hop distance from every node to the nearest exit over open links (NEVER if
 * none). With `deadline` (a hopDeadlines map) only nodes with D >= 0 are
 * walked and only exits with D >= 1 count: every safe escape stays on such
 * nodes and ends at such an exit, so the distance is still a lower bound on
 * the steps of any escape, and a sharper one (the fire-free distance ignores
 * an exit the fire seals). Either distance changes by at most one along a
 * shielded move, which is what SafeEscape.consistent_admissible needs.
 */
export function exitHops(graph, { exits, blocked = null, deadline = null }) {
  const { n, first, to } = openLinks(graph, blocked);
  const hops = new Int32Array(n).fill(NEVER);
  let frontier = [];
  for (const x of exits) {
    if (x < 0 || x >= n || hops[x] === 0 || (deadline && deadline[x] < 1)) continue;
    hops[x] = 0;
    frontier.push(x);
  }
  for (let d = 1; frontier.length; d += 1) {
    const next = [];
    for (const u of frontier) {
      for (let j = first[u]; j < first[u + 1]; j += 1) {
        const v = to[j];
        if (hops[v] !== NEVER || (deadline && deadline[v] < 0)) continue;
        hops[v] = d;
        next.push(v);
      }
    }
    frontier = next;
  }
  return hops;
}

/** Walking distance (model units) from every node to the nearest exit; with `deadline`, as exitHops. */
export function exitMetres(graph, { exits, blocked = null, deadline = null }) {
  const { n, first, to, edge } = openLinks(graph, blocked);
  const len = edgeLengths(graph);
  const dist = new Float64Array(n).fill(Infinity);
  const heap = new Heap();
  for (const x of exits) {
    if (x < 0 || x >= n || (deadline && deadline[x] < 1)) continue;
    dist[x] = 0;
    heap.push(0, x);
  }
  const done = new Uint8Array(n);
  while (heap.size) {
    const u = heap.pop();
    if (done[u]) continue;
    done[u] = 1;
    for (let j = first[u]; j < first[u + 1]; j += 1) {
      const v = to[j];
      if (deadline && deadline[v] < 0) continue;
      const d = dist[u] + len[edge[j]];
      if (d < dist[v]) {
        dist[v] = d;
        heap.push(d, v);
      }
    }
  }
  return dist;
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

/**
 * The earliest safe escape from `start` to any exit (fewest steps), exact:
 * breadth first over the steps, keeping each node's first arrival, which
 * dominates any later one because the fire only spreads (rl.js's
 * earliestEscape, for any of several exits, with the route). Among routes of
 * the fewest steps it returns the first found.
 *
 * @returns {{moves: number, path: number[], exit: number} | null}
 */
export function earliestSafeEscape(graph, start, { exits, ignite = null, horizon, blocked = null }) {
  const { n, first, to } = openLinks(graph, blocked);
  const H = Math.max(0, Math.floor(horizon));
  const isExit = exitFlags(n, exits);
  if (isExit[start]) return { moves: 0, path: [start], exit: start };
  const open = openUntil(toIgnition(ignite, n), H);
  const parent = new Int32Array(n).fill(-1);
  const seen = new Uint8Array(n);
  seen[start] = 1;
  let frontier = [start];
  for (let t = 0; t < H && frontier.length; t += 1) {
    const next = [];
    for (const u of frontier) {
      for (let j = first[u]; j < first[u + 1]; j += 1) {
        const v = to[j];
        if (seen[v] || t + 1 > open[v]) continue;
        seen[v] = 1;
        parent[v] = u;
        if (isExit[v]) {
          const path = [v];
          for (let c = parent[v]; c >= 0; c = parent[c]) path.push(c);
          return { moves: t + 1, path: path.reverse(), exit: v };
        }
        next.push(v);
      }
    }
    frontier = next;
  }
  return null;
}

/**
 * Finite-horizon value iteration (backward induction) on the time-expanded
 * graph: V_t(v) = min over open links (v, w) safe on arrival at t + 1 of
 * 1 (w an exit) or 1 + V_{t+1}(w), V_H = NEVER off the exits. Returns every
 * node's earliest-escape steps when leaving at step 0 (NEVER: none) and its
 * first move then (-1: none): the exact policy for every start at once.
 *
 * With `prune` (the default) a node is skipped at the steps after its
 * deadline. A move reads a node's value only where it may be entered, and
 * there a value is finite exactly at the steps up to the deadline
 * (SafeEscape.deadline_iff), so pruning changes no value that is read and none
 * at step 0; the tests compare it with the unpruned table. With `rows`, every
 * layer is kept (H + 1 rows of n values; for tests and small graphs).
 */
export function escapeValues(graph, { exits, ignite = null, horizon, blocked = null, prune = true, rows = false, deadline = null }) {
  const { n, first, to } = openLinks(graph, blocked);
  const H = Math.max(0, Math.floor(horizon));
  const isExit = exitFlags(n, exits);
  const open = openUntil(toIgnition(ignite, n), H);
  const D = prune ? deadline || hopDeadlines(graph, { exits, ignite, horizon: H, blocked }) : null;
  let ahead = new Int32Array(n).fill(NEVER);
  let here = new Int32Array(n);
  const kept = rows ? [] : null;
  const nextMove = new Int32Array(n).fill(-1);
  for (let v = 0; v < n; v += 1) if (isExit[v]) ahead[v] = 0;
  if (rows) kept[H] = Int32Array.from(ahead);
  for (let t = H - 1; t >= 0; t -= 1) {
    for (let v = 0; v < n; v += 1) {
      if (isExit[v]) {
        here[v] = 0;
        continue;
      }
      let best = NEVER, move = -1;
      if (!D || D[v] >= t) {
        for (let j = first[v]; j < first[v + 1]; j += 1) {
          const w = to[j];
          if (t + 1 > open[w]) continue;
          const value = isExit[w] ? 1 : ahead[w] === NEVER ? NEVER : 1 + ahead[w];
          if (value < best) {
            best = value;
            move = w;
          }
        }
      }
      here[v] = best;
      if (t === 0) nextMove[v] = move;
    }
    if (rows) kept[t] = Int32Array.from(here);
    [here, ahead] = [ahead, here];
  }
  return { moves: ahead, next: nextMove, rows: kept };
}

/** mulberry32, as rl.js, so a seed reproduces a run. */
function rngFrom(seed) {
  let a = (seed ?? 1) >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The learned values: per node, lower bounds on the steps left to escape as a
 * function of the step t. The exact number never falls with a later step
 * (leaving later never escapes sooner, since the fire only spreads:
 * SafeEscape.val_mono_time), so a bound learned at (v, t) holds at (v, t')
 * for every t' >= t. With `monotone` the table keeps each node's bounds as a
 * non-decreasing step function of t and reads at t the last breakpoint at or
 * before it: one update then serves every later step, the generalisation
 * across time that a table indexed by (node, step) otherwise lacks. Without
 * it a value holds at its own step only.
 */
export class StepTable {
  constructor({ monotone = true } = {}) {
    this.monotone = monotone;
    this.nodes = new Map();
    this.size = 0;
  }

  /** The bound at (v, t), or undefined when none has been learned. */
  get(v, t) {
    const e = this.nodes.get(v);
    if (!e) return undefined;
    let lo = 0, hi = e.ts.length - 1, at = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (e.ts[mid] <= t) {
        at = mid;
        lo = mid + 1;
      } else hi = mid - 1;
    }
    if (at < 0 || (!this.monotone && e.ts[at] !== t)) return undefined;
    return e.vs[at];
  }

  /** Raise the bound at (v, t), and with `monotone` at every later step, to c. */
  raise(v, t, c) {
    let e = this.nodes.get(v);
    if (!e) this.nodes.set(v, (e = { ts: [], vs: [] }));
    let p = 0;
    while (p < e.ts.length && e.ts[p] <= t) p += 1;
    if (p > 0 && e.ts[p - 1] === t) e.vs[p - 1] = Math.max(e.vs[p - 1], c);
    else {
      e.ts.splice(p, 0, t);
      e.vs.splice(p, 0, c);
      this.size += 1;
      p += 1;
    }
    if (this.monotone) {
      // Later breakpoints at or below c say nothing more.
      let q = p;
      while (q < e.ts.length && e.vs[q] <= c) q += 1;
      if (q > p) {
        e.ts.splice(p, q - p);
        e.vs.splice(p, q - p);
        this.size -= q - p;
      }
    }
  }

  /** Every breakpoint as [node, step, bound]. */
  *entries() {
    for (const [v, e] of this.nodes) for (let i = 0; i < e.ts.length; i += 1) yield [v, e.ts[i], e.vs[i]];
  }
}

/**
 * Learn an escape from `start` by shielded learning real-time search on the
 * time-expanded graph (see the header).
 *
 * @param {object} opts  exits: node ids; ignite: Map or Float64Array of
 *   ignition steps (null: no fire); horizon: the step limit H; blocked: per-edge
 *   flags; trials: the most trials to run; shield: "deadline" (default),
 *   "arrival" (mask only nodes untenable on arrival) or "none" (an unsafe move
 *   burns and ends the trial); values: a StepTable from an earlier call on the
 *   same graph, exits, fire and horizon, to keep learning from (one table
 *   serves every start); monotone: share a bound with every later step
 *   (default true; see StepTable); explore: probability of a random allowed move during
 *   training (default 0: optimism explores); seed: for `explore`; backward:
 *   repeat the update from each trial's end back to its start (default true).
 * @returns the drawn path (the greedy walk after training), whether it
 *   reached an exit or burned, its steps, the trials run, whether the last
 *   trial carries the optimality certificate, the steps of each trial
 *   (`curve`), and the value table.
 */
export function learnSafeEscape(graph, start, {
  exits,
  ignite = null,
  horizon,
  blocked = null,
  trials = 200,
  shield = "deadline",
  values = null,
  explore = 0,
  seed = 1,
  deadline = null,
  backward = true,
  monotone = true,
} = {}) {
  const { n, first, to } = openLinks(graph, blocked);
  const H = Math.max(0, Math.floor(horizon));
  const isExit = exitFlags(n, exits);
  const ig = toIgnition(ignite, n);
  const open = openUntil(ig, H);
  const D = shield === "deadline" ? deadline || hopDeadlines(graph, { exits, ignite: ig, horizon: H, blocked }) : null;
  // The initial values (the potential): hop distance to a usable exit inside
  // the deadline map's viable nodes, or the fire-free one without the shield.
  const h = exitHops(graph, { exits, blocked, deadline: D });
  const metres = exitMetres(graph, { exits, blocked, deadline: D });
  const FAIL = H + 1; // more than any feasible number of steps left
  const V = values || new StepTable({ monotone });
  const get = (v, t) => (isExit[v] ? 0 : t >= H ? FAIL : Math.max(V.get(v, t) ?? 0, Math.min(FAIL, h[v])));
  const rng = rngFrom(seed);
  const allowed = (w, t) => (D ? t + 1 <= D[w] : shield === "arrival" ? t + 1 <= open[w] : true);

  const empty = { path: [start], moves: null, reachedExit: false, burned: false, trials: 0, certified: false, curve: [], values: V, deadlineStart: D ? D[start] : null };
  if (isExit[start]) return { ...empty, moves: 0, reachedExit: true, certified: true };
  if (D && D[start] < 0) return empty;

  // One walk from (start, 0): greedy on 1 + V(w, t + 1) among allowed moves,
  // ties to the shorter fire-free walk to an exit, then the first link.
  // With `learn`, the LRTA* update is applied at each state before moving.
  const walk = (learn, eps) => {
    const path = [start];
    let v = start, t = 0, changed = false, burned = false, reached = false, masked = 0;
    while (t < H) {
      let best = Infinity, bestMetres = Infinity, choice = -1, count = 0;
      for (let j = first[v]; j < first[v + 1]; j += 1) {
        const w = to[j];
        if (!allowed(w, t)) {
          masked += 1;
          continue;
        }
        count += 1;
        const f = 1 + (t + 1 > open[w] ? FAIL : get(w, t + 1));
        if (f < best || (f === best && metres[w] < bestMetres)) {
          best = f;
          bestMetres = metres[w];
          choice = w;
        }
      }
      if (learn) {
        const target = Math.min(FAIL, choice < 0 ? FAIL : best);
        if (target > get(v, t)) {
          V.raise(v, t, target);
          changed = true;
        }
      }
      if (choice < 0) break; // only without the deadline shield
      if (eps > 0 && count > 1 && rng() < eps) {
        let pick = Math.floor(rng() * count);
        for (let j = first[v]; j < first[v + 1]; j += 1) {
          if (!allowed(to[j], t)) continue;
          if (pick-- === 0) {
            choice = to[j];
            break;
          }
        }
      }
      path.push(choice);
      t += 1;
      if (t > open[choice]) {
        burned = true;
        break;
      }
      v = choice;
      if (isExit[v]) {
        reached = true;
        break;
      }
    }
    return { path, moves: reached ? t : null, reached, burned, changed, masked };
  };

  // The same update, at every state of a finished trial from its end back to
  // the start, so what the trial learned near its end reaches the start in
  // one trial rather than one step per trial (the backward pass of trial-based
  // RTDP; each update is the one lrta_admissible covers).
  const backup = (path) => {
    for (let i = path.length - 2; i >= 0; i -= 1) {
      const v = path[i];
      if (isExit[v]) continue;
      let best = FAIL;
      for (let j = first[v]; j < first[v + 1]; j += 1) {
        const w = to[j];
        if (!allowed(w, i)) continue;
        const f = 1 + (i + 1 > open[w] ? FAIL : get(w, i + 1));
        if (f < best) best = f;
      }
      if (Math.min(FAIL, best) > get(v, i)) V.raise(v, i, Math.min(FAIL, best));
    }
  };

  const curve = [];
  let certified = false, used = 0, masked = 0;
  for (let k = 0; k < Math.max(1, trials); k += 1) {
    const before = get(start, 0);
    const trial = walk(true, explore);
    used += 1;
    masked += trial.masked;
    curve.push(trial.reached ? trial.moves : null);
    // The certificate: a safe escape no longer than an admissible value is an
    // earliest escape (SafeEscape.certificate_optimal).
    if (trial.reached && explore === 0 && trial.moves <= before) {
      certified = true;
      break;
    }
    if (backward) backup(trial.path);
  }
  const final = walk(false, 0);
  // The greedy walk after training: with no update since the last trial it
  // is that trial again; it is certified when no longer than V(start, 0).
  if (final.reached && final.moves <= get(start, 0)) certified = true;
  return {
    path: final.path,
    moves: final.moves,
    reachedExit: final.reached,
    burned: final.burned,
    trials: used,
    certified,
    curve,
    masked,
    values: V,
    deadlineStart: D ? D[start] : null,
  };
}

/**
 * The whole learned-escape request, in the shape of rl.js's trainRl (the
 * server's RLResponse) with the fields this method adds. `trials` is the most
 * trials (default 20,000: a trial costs about one walk, and on the study's
 * corpus every escape to any exit was certified within 12,501); rl.js's
 * `episodes` is not read. `exitPoints` (or the
 * one `exitPoint`) are snapped to nodes; the escape may use any of them.
 */
export function trainSafeRl(graph, {
  startPoint,
  exitPoint = null,
  exitPoints = null,
  firePoint = null,
  trials = 20000,
  maxSteps = 200,
  useFire = true,
  fireSpeed = 1,
  useWalls = true,
  shield = "deadline",
} = {}) {
  const t0 = performance.now();
  const blocked = useWalls ? graph.blocked : null;
  const H = Math.max(1, Math.min(SAFE_MAX_STEPS, Math.floor(maxSteps)));
  const start = snap(graph, startPoint);
  const exits = [...new Set((exitPoints?.length ? exitPoints : [exitPoint]).filter(Boolean).map((p) => snap(graph, p)))];
  const notes = [];
  let fireStart = null;
  let ig = null;
  if (useFire) {
    if (!firePoint) notes.push("No fire origin was given, so the policy was trained without fire.");
    else {
      fireStart = snap(graph, firePoint);
      ig = ignitionSteps(graph, fireStart, { fireSpeed, maxSteps: H, blocked });
    }
  }
  const D = hopDeadlines(graph, { exits, ignite: ig, horizon: H, blocked });
  const r = learnSafeEscape(graph, start, { exits, ignite: ig, horizon: H, blocked, trials, shield, deadline: D });
  const exact = earliestSafeEscape(graph, start, { exits, ignite: ig, horizon: H, blocked });
  const fireFree = distancesFrom(graph, start, { blocked });
  let shortest = Infinity;
  for (const x of exits) shortest = Math.min(shortest, fireFree[x]);
  if (!exact) {
    const reachable = exitHops(graph, { exits, blocked })[start] < NEVER;
    if (!reachable) notes.push("The exit cannot be reached from the start without crossing a wall.");
    else if (!ig) notes.push(`The exit is more than ${H} moves away; raise the step limit.`);
    else notes.push("No route escapes this fire within the step limit: it reaches or cuts off every exit before any path can.");
  } else if (!r.reachedExit) {
    notes.push(`An escape exists (${exact.moves} moves) but the policy did not find it; train for more episodes.`);
  } else if (!r.certified) {
    notes.push(`The policy escapes in ${r.moves} moves; the earliest escape takes ${exact.moves}. Train for more episodes to close the gap.`);
  }
  const reached = r.curve.map((m) => (m == null ? 0 : 1));
  const buckets = Math.min(CURVE_POINTS, reached.length);
  const curve = Array.from({ length: buckets }, (_, b) => {
    const chunk = reached.slice(Math.floor((b * reached.length) / buckets), Math.floor(((b + 1) * reached.length) / buckets));
    return chunk.length ? Math.round((chunk.reduce((s, x) => s + x, 0) / chunk.length) * 1000) / 1000 : 0;
  });
  const tail = reached.slice(-Math.max(1, Math.floor(reached.length / 5)));
  return {
    graph_id: graph.graphId ?? null,
    method: "shielded",
    path: r.path,
    points: r.path.map((i) => [graph.nodes[i * 3], graph.nodes[i * 3 + 1], graph.nodes[i * 3 + 2]]),
    reached_exit: r.reachedExit,
    episodes: r.trials,
    burned: r.burned,
    reward: null,
    length: pathLength(graph, r.path),
    optimal_length: Number.isFinite(shortest) ? shortest : null,
    success_rate: tail.length ? tail.reduce((s, x) => s + x, 0) / tail.length : 0,
    curve,
    moves_curve: r.curve,
    escape_steps: exact ? exact.moves : null,
    escape_length: exact ? pathLength(graph, exact.path) : null,
    moves: r.moves,
    certified: r.certified,
    deadline_start: r.deadlineStart,
    exit_id: r.reachedExit ? r.path[r.path.length - 1] : null,
    policy_return: null,
    optimal_return: null,
    fire_start_id: fireStart,
    train_ms: Math.round(performance.now() - t0),
    note: notes.join(" ") || null,
  };
}

/**
 * The exact escape field for every start at once (finite-horizon value
 * iteration from step 0), with the deadline map: per node, the earliest
 * escape in steps when leaving at step 0 (-1: none), its first move, and its
 * deadline. O(H x links), restricted to nodes before their deadline.
 */
export function escapeField(graph, { exits, ignite = null, horizon, blocked = null }) {
  const D = hopDeadlines(graph, { exits, ignite, horizon, blocked });
  const { moves, next } = escapeValues(graph, { exits, ignite, horizon, blocked, deadline: D });
  const steps = new Int32Array(moves.length);
  for (let v = 0; v < moves.length; v += 1) steps[v] = moves[v] >= NEVER ? -1 : moves[v];
  return { deadline: D, steps, next };
}

