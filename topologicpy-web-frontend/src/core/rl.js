/**
 * Tabular Q-learning egress in the browser: a port of the server's
 * app/graphs/rl.py with the literature review's two corrections (finding W5,
 * wiki/research/fire-egress-methods-2026-10-27.md):
 *
 * 1. Terminal states have zero potential. Potential-based shaping preserves
 *    optimal policies (Ng, Harada & Russell 1999) only when the potential is 0
 *    at terminal states; the server used the burn node's own potential, which
 *    adds gamma^T * Phi(s_T) to the return and can change which failure is
 *    "least bad" - or, when no escape exists, the optimum itself.
 * 2. With fire, the clock is part of the state and the last allowed step is
 *    terminal (value 0, potential 0), instead of bootstrapping from untrained
 *    prior values beyond the horizon. Without fire the state has no clock, so
 *    the step limit only cuts an episode short and the update still
 *    bootstraps (Pardo et al. 2018): otherwise one state would get two
 *    different targets and Q* would not be a fixed point (QLearning.lean).
 *
 * Everything else matches the server: time-indexed state when there is fire,
 * shaping by minus the remaining distance, untried actions initialised to their
 * one-step shaped value, a rollout that never revisits a node, and an exact
 * earliest-escape BFS so "the agent failed" and "no escape exists" are told apart.
 */
import { csr, distancesFrom, edgeLengths, snap } from "./route.js";
import { floodTimeline } from "./fire.js";

export const EXIT_REWARD = 10;
export const BURN_REWARD = -10;
export const STEP_COST = 0.1;
const CURVE_POINTS = 50;

/** A small seeded generator (mulberry32), so a seed reproduces a run. */
function rngFrom(seed) {
  let a = (seed ?? Math.floor(Math.random() * 2 ** 32)) >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fewest moves from start to exit that never enter a burning node. Exact,
 *  because the fire only spreads. */
export function earliestEscape(graph, start, exit, ignite, maxSteps, blocked) {
  if (start === exit) return 0;
  const { indptr, neighbours, edgeIds } = csr(graph);
  const seen = new Set([start]);
  let frontier = [start];
  for (let t = 0; t < maxSteps; t += 1) {
    const next = [];
    for (const u of frontier) {
      for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
        const v = neighbours[k];
        if (seen.has(v) || blocked?.[edgeIds[k]]) continue;
        if (ignite && t >= (ignite.get(v) ?? Infinity)) continue;
        if (v === exit) return t + 1;
        seen.add(v);
        next.push(v);
      }
    }
    if (!next.length) return null;
    frontier = next;
  }
  return null;
}

export function qLearningPath(graph, start, exit, {
  ignite = null,
  episodes = 200,
  maxSteps = 200,
  epsilon = 0.2,
  alpha = 0.5,
  gamma = 0.99,
  seed = null,
  blocked = null,
  shaping = true,
} = {}) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  if (start === exit) return { path: [start], reachedExit: true, episodes: 0, reward: 0, burned: false, length: 0, optimalLength: 0, successRate: 1, curve: [], escapeSteps: 0, note: null };
  const distance = distancesFrom(graph, exit, { blocked });
  if (!Number.isFinite(distance[start])) {
    return { path: [start], reachedExit: false, episodes: 0, reward: 0, burned: false, length: 0, optimalLength: null, successRate: 0, curve: [], escapeSteps: null, note: "The exit cannot be reached from the start without crossing a wall." };
  }
  const rng = rngFrom(seed);
  const lengths = edgeLengths(graph);
  const sorted = Array.from(lengths).sort((a, b) => a - b);
  const unit = sorted[Math.floor(sorted.length / 2)] || 1;

  const slotCost = new Float64Array(neighbours.length);
  const source = new Int32Array(neighbours.length);
  for (let u = 0; u < n; u += 1) {
    for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
      slotCost[k] = (lengths[edgeIds[k]] / unit) * STEP_COST;
      source[k] = u;
    }
  }
  const slots = Array.from({ length: n }, (_, u) => {
    const out = [];
    for (let k = indptr[u]; k < indptr[u + 1]; k += 1) if (!blocked?.[edgeIds[k]]) out.push(k);
    return out;
  });
  let worst = 0;
  for (const d of distance) if (Number.isFinite(d) && d > worst) worst = d;
  const potential = new Float64Array(n);
  if (shaping) for (let u = 0; u < n; u += 1) potential[u] = (-(Number.isFinite(distance[u]) ? distance[u] : worst) / unit) * STEP_COST;

  const burnsAt = (v) => (ignite ? ignite.get(v) ?? Infinity : Infinity);
  const timed = Boolean(ignite && ignite.size);
  const horizon = maxSteps + 1;
  const prior = new Float64Array(neighbours.length);
  for (let k = 0; k < neighbours.length; k += 1) {
    const v = neighbours[k];
    prior[k] = -slotCost[k] + gamma * (v === exit ? 0 : potential[v]) - potential[source[k]];
  }
  const q = new Map();
  const key = (k, t) => (timed ? k * horizon + t : k);
  const value = (k, t) => q.get(key(k, t)) ?? prior[k];
  const greedy = (u, t, avoid) => {
    let options = slots[u];
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
    return winners[Math.floor(rng() * winners.length)];
  };

  const successes = [];
  for (let ep = 0; ep < Math.max(1, episodes); ep += 1) {
    let state = start;
    let reached = 0;
    for (let t = 0; t < maxSteps; t += 1) {
      const options = slots[state];
      if (!options.length) break;
      const k = rng() < epsilon ? options[Math.floor(rng() * options.length)] : greedy(state, t);
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
      // A transition into a terminal state (burning, the exit, or with fire the
      // horizon) is shaped as if the terminal had zero potential (W5); with a
      // non-zero potential at the horizon the shaped optimum can prefer burning
      // on the last step (QLearning.horizon_potential_prefers_fire).
      const last = timed && t === maxSteps - 1;
      const phiNext = done || last ? 0 : potential[next];
      const shaped = reward + gamma * phiNext - potential[state];
      let target = shaped;
      if (!done && !last) {
        let bestNext = -Infinity;
        for (const j of slots[next]) bestNext = Math.max(bestNext, value(j, t + 1));
        target += gamma * (Number.isFinite(bestNext) ? bestNext : 0);
      }
      const current = value(k, t);
      q.set(key(k, t), current + alpha * (target - current));
      state = next;
      if (done) break;
    }
    successes.push(reached);
  }

  const path = [start];
  const visited = new Set([start]);
  let state = start;
  let reachedExit = false, burned = false, total = 0;
  // The rollout's discounted return, to compare with the exact optimum below.
  let policyReturn = 0, discount = 1;
  for (let t = 0; t < maxSteps; t += 1) {
    const k = greedy(state, t, visited);
    if (k < 0) break;
    const next = neighbours[k];
    path.push(next);
    visited.add(next);
    if (t >= burnsAt(next)) {
      burned = true;
      total += BURN_REWARD;
      policyReturn += discount * BURN_REWARD;
      break;
    }
    if (next === exit) {
      reachedExit = true;
      total += EXIT_REWARD;
      policyReturn += discount * EXIT_REWARD;
      break;
    }
    total -= slotCost[k];
    policyReturn -= discount * slotCost[k];
    discount *= gamma;
    state = next;
  }

  // The exact optimum of the same problem, unshaped: backward induction over
  // the step limit, V_t(u) = max over moves of the reward plus gamma V_{t+1}
  // after a non-terminal move, V at the limit 0 (QLearning.lean,
  // dag_value_iteration_exact). RL evacuation papers report success rates, not
  // this gap (gap analysis 2026-10-27, LG6).
  const burnTime = new Float64Array(n);
  for (let u = 0; u < n; u += 1) burnTime[u] = burnsAt(u);
  let ahead = new Float64Array(n), here = new Float64Array(n);
  for (let t = maxSteps - 1; t >= 0; t -= 1) {
    for (let u = 0; u < n; u += 1) {
      let best = slots[u].length ? -Infinity : 0;
      for (const k of slots[u]) {
        const v = neighbours[k];
        const value = t >= burnTime[v] ? BURN_REWARD : v === exit ? EXIT_REWARD : -slotCost[k] + gamma * ahead[v];
        if (value > best) best = value;
      }
      here[u] = best;
    }
    [here, ahead] = [ahead, here];
  }
  const optimalReturn = ahead[start];

  const P = graph.nodes;
  let length = 0;
  for (let i = 1; i < path.length; i += 1) {
    const a = path[i - 1] * 3, b = path[i] * 3;
    length += Math.hypot(P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]);
  }
  const escapeSteps = earliestEscape(graph, start, exit, ignite, maxSteps, blocked);
  let note = null;
  if (escapeSteps === null && !timed) note = `The exit is more than ${maxSteps} moves away; raise the step limit.`;
  else if (escapeSteps === null) note = "No route escapes this fire within the step limit: it reaches or cuts off the exit before any path can.";
  else if (!reachedExit) note = `An escape exists (${escapeSteps} moves) but the policy did not find it; train for more episodes.`;

  const tail = successes.slice(-Math.max(1, Math.floor(successes.length / 5)));
  const buckets = Math.min(CURVE_POINTS, successes.length);
  const curve = Array.from({ length: buckets }, (_, b) => {
    const chunk = successes.slice(Math.floor((b * successes.length) / buckets), Math.floor(((b + 1) * successes.length) / buckets));
    return chunk.length ? Math.round((chunk.reduce((s, x) => s + x, 0) / chunk.length) * 1000) / 1000 : 0;
  });
  return {
    path,
    reachedExit,
    episodes,
    reward: Math.round(total * 1e4) / 1e4,
    burned,
    length,
    optimalLength: distance[start],
    successRate: tail.reduce((s, x) => s + x, 0) / tail.length,
    curve,
    escapeSteps,
    note,
    policyReturn,
    optimalReturn,
  };
}

/** The whole RL request in the browser, returning the server's RLResponse shape. */
export function trainRl(graph, { startPoint, exitPoint, firePoint = null, episodes = 200, maxSteps = 200, useFire = true, fireSpeed = 1, useWalls = true, seed = null } = {}) {
  const t0 = performance.now();
  const blocked = useWalls ? graph.blocked : null;
  const start = snap(graph, startPoint);
  const exit = snap(graph, exitPoint);
  const notes = [];
  let ignite = null;
  let fireStart = null;
  if (useFire) {
    if (!firePoint) {
      notes.push("No fire origin was given, so the policy was trained without fire.");
    } else {
      fireStart = snap(graph, firePoint);
      const timeline = floodTimeline(graph, fireStart, Math.ceil(maxSteps * fireSpeed) + 1, blocked);
      ignite = new Map();
      timeline.forEach((nodes, step) => nodes.forEach((node) => ignite.set(node, step / fireSpeed)));
    }
  }
  const r = qLearningPath(graph, start, exit, { ignite, episodes, maxSteps, seed, blocked });
  if (r.note) notes.push(r.note);
  return {
    graph_id: graph.graphId ?? null,
    path: r.path,
    points: r.path.map((i) => [graph.nodes[i * 3], graph.nodes[i * 3 + 1], graph.nodes[i * 3 + 2]]),
    reached_exit: r.reachedExit,
    episodes: r.episodes,
    burned: r.burned,
    reward: r.reward,
    length: r.length,
    optimal_length: Number.isFinite(r.optimalLength) ? r.optimalLength : null,
    success_rate: r.successRate,
    curve: r.curve,
    escape_steps: r.escapeSteps,
    // Discounted, unshaped: the policy's rollout and the exact optimum.
    policy_return: r.policyReturn ?? null,
    optimal_return: r.optimalReturn ?? null,
    fire_start_id: fireStart,
    train_ms: Math.round(performance.now() - t0),
    note: notes.join(" ") || null,
  };
}
