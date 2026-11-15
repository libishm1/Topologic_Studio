/**
 * The learners against the simulated fire: the shielded learner (rlSafe.js)
 * and the shipped Q-learning (rl.js) trained on the closing times of a fire
 * forecast in seconds (the Flamescapes run the page has made, read by
 * forecast.js forecastFromFrames as the forecast plan reads it), instead of
 * the graph's hop front. The study behind it is
 * wiki/research/rl-egress-study-2026-11-05.md; the clock lemma is
 * formal/fireegress/FireEgress/SafeClock.lean.
 *
 * Closing times. The forecast closes node v at c(v) = min(u(v), horizon),
 * u(v) the start of the first frame at which v is untenable (gas at the
 * threshold at head height, visibility under the limit, or an FED at 0.3);
 * past the horizon nothing is known, so every node counts as closed. A margin
 * s in (0, 1] scales every closing time to s c(v) (the fire faster than
 * forecast); s = 1 is the forecast itself.
 *
 * The clock. Both learners count moves along links. A move lasts at most
 * the step tau seconds: tau = L / speed, with
 *   "split" (the default): L the longest open link no longer than
 *     SPLIT_FACTOR (sqrt 2, rounded up) times the median one: on Studio's
 *     lattice the diagonal, 0.707 m at 0.5 m spacing. A link longer than L (a
 *     stair flight, a long link of a coarse graph) is walked as
 *     ceil(len / L) moves through virtual nodes placed along it, each of which
 *     closes when the sooner of the link's two ends closes. The drawn route
 *     keeps the real nodes only. On the RL study's nine buildings, with a
 *     Flamescapes fire, this found as many escapes as splitting at the median
 *     link, at 200 moves and at 400, with about 2% more nodes instead of
 *     about 145%; the longest-link clock found 31% fewer
 *     (tools/rl-flamescapes.mjs --clock).
 *   "longest": L the longest open link, no splitting.
 * tau is raised by a relative 1e-9 (STEP_SLACK) so that rounding in the
 * conversion or in summing link times never turns a move that is safe on the
 * clock into a late one in seconds.
 *
 * The conversion. An evacuee leaving at d seconds makes the move at step t
 * (arriving at step t + 1) safely iff it arrives before the node closes,
 * d + (t + 1) tau < s c(v). Both learners test t < ig(v), so the ignition step
 * handed to them is
 *     ig(v) = (s c(v) - d) / tau - 1,
 * which rlSafe.js turns into open(v) = min(H, max(0, ceil(ig(v)))). If every
 * move takes at most tau seconds, a route that is safe on this clock reaches
 * every node before s c(v) <= c(v) in seconds (SafeClock.clock_safe). The
 * returned route is walked again in seconds along its real links
 * (forecast.js walkRoute); a late arrival there would be a bug, and it makes
 * the result uncertified.
 *
 * What is certified is relative to the forecast: the route is safe if the
 * real fire closes no node sooner than s times its forecast time. Not here:
 * a shield built from an ensemble of runs (the earliest closing over them),
 * and a shield recomputed as the fire is observed (closed loop); both would
 * change what the certificate is relative to, not how it is computed.
 */
import { distancesFrom, edgeLengths, snap } from "./route.js";
import { walkRoute } from "./forecast.js";
import { RL_MAX_STEPS, qLearningPath } from "./rl.js";
import { NEVER, SAFE_MAX_STEPS, earliestSafeEscape, exitHops, hopDeadlines, learnSafeEscape } from "./rlSafe.js";

/** Relative slack on the step: tau = (L / speed) (1 + STEP_SLACK). */
export const STEP_SLACK = 1e-9;
/** The clocks a forecast can be put on (see the header). */
export const RL_CLOCKS = ["split", "longest"];
const CURVE_POINTS = 50;

/**
 * How much longer than the median a link may be and still be one move on the
 * "split" clock: sqrt 2 rounded up, so the lattice's diagonals are one move.
 */
export const SPLIT_FACTOR = 1.415;

/**
 * Median and longest open link (model units, metres), and the split clock's
 * default step length: the longest open link no longer than SPLIT_FACTOR
 * times the median.
 */
export function linkStats(graph, blocked = null) {
  const len = edgeLengths(graph);
  const open = [];
  for (let e = 0; e < len.length; e += 1) if (!blocked?.[e]) open.push(len[e]);
  if (!open.length) return { median: 0, longest: 0, links: 0, splitAt: 0 };
  open.sort((a, b) => a - b);
  const median = open[Math.floor(open.length / 2)];
  let splitAt = median;
  for (const l of open) if (l <= SPLIT_FACTOR * median) splitAt = l;
  return { median, longest: open[open.length - 1], links: open.length, splitAt };
}

/**
 * The step clock (see the header): the step `tau` in seconds and the graph
 * the learners walk. Real nodes keep their ids; virtual nodes follow them,
 * and `ends` holds each virtual node's link ends (real nodes).
 *
 * @param {object} opts  speed: walking speed (m/s); blocked: per-edge flags of
 *   links never walked (walls); clock: "split" or "longest"; stepMetres: the
 *   most a move covers with "split" (default linkStats' splitAt).
 */
export function stepClock(graph, { speed = 1.2, blocked = null, clock = "split", stepMetres = null } = {}) {
  if (!(speed > 0)) throw new Error("The walking speed must be above 0 m/s.");
  const n = graph.nodes.length / 3;
  const stats = linkStats(graph, blocked);
  const base = { clock, speed, n, median: stats.median, longest: stats.longest };
  if (clock === "longest" || !(stats.longest > 0)) {
    const L = stats.longest > 0 ? stats.longest : 1;
    return { ...base, clock: "longest", graph, blocked, stepMetres: L, tau: (L / speed) * (1 + STEP_SLACK), virtual: 0, split: 0, ends: new Int32Array(0) };
  }
  if (clock !== "split") throw new Error(`Unknown clock "${clock}".`);
  const L = stepMetres > 0 ? stepMetres : stats.splitAt > 0 ? stats.splitAt : stats.longest;
  const len = edgeLengths(graph);
  const m = graph.edges.length / 2;
  // How many moves each link takes: one, or ceil(len / L) for an open link longer than L.
  const parts = new Int32Array(m);
  let virtual = 0, split = 0, links = 0;
  for (let e = 0; e < m; e += 1) {
    const k = !blocked?.[e] && len[e] > L ? Math.ceil(len[e] / L) : 1;
    parts[e] = k;
    links += k;
    if (k > 1) {
      virtual += k - 1;
      split += 1;
    }
  }
  if (!split) return { ...base, graph, blocked, stepMetres: L, tau: (L / speed) * (1 + STEP_SLACK), virtual: 0, split: 0, ends: new Int32Array(0) };
  const P = graph.nodes;
  const nodes = new Float32Array((n + virtual) * 3);
  nodes.set(P.subarray ? P.subarray(0, n * 3) : P.slice(0, n * 3));
  const edges = new Uint32Array(links * 2);
  const flags = new Uint8Array(links);
  const ends = new Int32Array(virtual * 2);
  let next = n, out = 0;
  for (let e = 0; e < m; e += 1) {
    const a = graph.edges[e * 2], b = graph.edges[e * 2 + 1], k = parts[e];
    if (k === 1) {
      edges[out * 2] = a;
      edges[out * 2 + 1] = b;
      flags[out] = blocked?.[e] ? 1 : 0;
      out += 1;
      continue;
    }
    let prev = a;
    for (let j = 1; j < k; j += 1) {
      const v = next++;
      for (let c = 0; c < 3; c += 1) nodes[v * 3 + c] = P[a * 3 + c] + ((P[b * 3 + c] - P[a * 3 + c]) * j) / k;
      ends[(v - n) * 2] = a;
      ends[(v - n) * 2 + 1] = b;
      edges[out * 2] = prev;
      edges[out * 2 + 1] = v;
      out += 1;
      prev = v;
    }
    edges[out * 2] = prev;
    edges[out * 2 + 1] = b;
    out += 1;
  }
  const g = { nodes, edges, blocked: flags };
  return { ...base, graph: g, blocked: blocked ? flags : null, stepMetres: L, tau: (L / speed) * (1 + STEP_SLACK), virtual, split, ends };
}

/** Closing times on the clock's graph: real nodes their own, a virtual node the sooner of its link's two ends. */
export function clockClosings(clk, closes) {
  const out = new Float64Array(clk.n + clk.virtual);
  out.set(closes.subarray ? closes.subarray(0, clk.n) : closes.slice(0, clk.n));
  for (let i = 0; i < clk.virtual; i += 1) out[clk.n + i] = Math.min(closes[clk.ends[i * 2]], closes[clk.ends[i * 2 + 1]]);
  return out;
}

/**
 * The conversion (see the header): ig(v) = (s c(v) - d) / tau - 1, so that a
 * move at step t into v passes both learners' test t < ig(v) exactly when
 * d + (t + 1) tau < s c(v).
 */
export function forecastIgnition(closes, { depart = 0, tau, margin = 1 }) {
  const ig = new Float64Array(closes.length);
  for (let v = 0; v < closes.length; v += 1) ig[v] = (margin * closes[v] - depart) / tau - 1;
  return ig;
}

/** The most moves that can still arrive before the forecast's horizon: d + k tau < s horizon. */
export function stepsToHorizon(horizon, { depart = 0, tau, margin = 1 }) {
  return Math.max(0, Math.ceil((margin * horizon - depart) / tau));
}

/** A route on the clock's graph as real nodes: virtual nodes dropped, repeats merged. */
export function realPath(clk, path) {
  const out = [];
  for (const v of path) if (v < clk.n && out[out.length - 1] !== v) out.push(v);
  return out;
}

/**
 * The route walked in seconds against the closing times the learner used
 * (forecast.js walkRoute), and its tightest margin against the forecast's
 * own closing times: the least c(v) minus the arrival at v, and where.
 */
export function walkInSeconds(graph, nodes, { forecast, closes, depart = 0, speed = 1.2 }) {
  const walked = walkRoute(graph, nodes, { forecast: { ...forecast, closes }, depart, speed });
  let slack = Infinity, slackNode = -1;
  nodes.forEach((v, i) => {
    const s = forecast.closes[v] - walked.times[i];
    if (s < slack) {
      slack = s;
      slackNode = v;
    }
  });
  return { times: walked.times, arrival: walked.times[walked.times.length - 1], dose: walked.dose, firstUnsafe: walked.firstUnsafe, slack, slackNode };
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

/** The success curve as trainSafeRl and rl.js give it: per bucket, the share of trials that reached an exit. */
function curveOf(reached) {
  const buckets = Math.min(CURVE_POINTS, reached.length);
  return Array.from({ length: buckets }, (_, b) => {
    const chunk = reached.slice(Math.floor((b * reached.length) / buckets), Math.floor(((b + 1) * reached.length) / buckets));
    return chunk.length ? Math.round((chunk.reduce((s, x) => s + x, 0) / chunk.length) * 1000) / 1000 : 0;
  });
}

const r1 = (x) => Math.round(x * 10) / 10;

/**
 * The whole RL request against a forecast, in the shape of trainSafeRl and
 * trainRl (the server's RLResponse) with the fields this adds: fire_source
 * "forecast", `forecast` (what the result is certified against and on which
 * clock) and `seconds` (the returned route walked in seconds).
 *
 * @param {object} opts  method: "shielded" (default) or "qlearning";
 *   forecast: a forecast.js forecast (closes, horizon, rate, ...); depart:
 *   seconds after ignition the evacuee leaves; walkSpeed: m/s; margin: s in
 *   (0, 1]; clock, stepMetres: see stepClock; the rest as trainSafeRl and
 *   trainRl (startPoint, exitPoint, exitPoints, firePoint, trials, episodes,
 *   maxSteps, useWalls, seed).
 */
export function trainForecastRl(graph, {
  method = "shielded",
  forecast,
  depart = 0,
  walkSpeed = 1.2,
  margin = 1,
  clock = "split",
  stepMetres = null,
  startPoint,
  exitPoint = null,
  exitPoints = null,
  firePoint = null,
  trials = 20000,
  episodes = 200,
  maxSteps = 200,
  useWalls = true,
  seed = null,
} = {}) {
  const t0 = performance.now();
  if (!forecast?.closes) throw new Error("No forecast to train against.");
  if (!(margin > 0 && margin <= 1)) throw new Error("The forecast margin must be above 0 and at most 1.");
  if (!(depart >= 0)) throw new Error("The departure must be at least 0 s.");
  const shielded = method !== "qlearning";
  const blocked = useWalls ? graph.blocked : null;
  const n = graph.nodes.length / 3;
  const start = snap(graph, startPoint);
  const exits = shielded
    ? [...new Set((exitPoints?.length ? exitPoints : [exitPoint]).filter(Boolean).map((p) => snap(graph, p)))]
    : [snap(graph, exitPoint)];
  const fireStart = firePoint ? snap(graph, firePoint) : null;

  const clk = stepClock(graph, { speed: walkSpeed, blocked, clock, stepMetres });
  const closes = new Float64Array(n);
  for (let v = 0; v < n; v += 1) closes[v] = margin * forecast.closes[v];
  const g = clk.graph;
  const gBlocked = clk.blocked;
  // ig over the clock's graph, from the scaled closing times (margin 1 here: already applied).
  const ig = forecastIgnition(clockClosings(clk, closes), { depart, tau: clk.tau, margin: 1 });
  const limit = Math.max(1, Math.min(shielded ? SAFE_MAX_STEPS : RL_MAX_STEPS, Math.floor(maxSteps)));
  const toHorizon = stepsToHorizon(forecast.horizon, { depart, tau: clk.tau, margin });
  // Past the horizon every node is closed, so no move can arrive after
  // toHorizon steps: a longer limit changes nothing but the work.
  const H = Math.max(1, Math.min(limit, toHorizon));
  const info = {
    horizon: forecast.horizon,
    depart,
    margin,
    speed: walkSpeed,
    clock: clk.clock,
    stepMetres: clk.stepMetres,
    step: clk.tau,
    longestLink: clk.longest,
    split: clk.split,
    virtual: clk.virtual,
    stepLimit: H,
    stepLimitSeconds: depart + H * clk.tau,
    limitBinds: limit < toHorizon,
    startCloses: forecast.closes[start],
  };
  const notes = [];
  const startClosed = !(depart < closes[start]);
  const shortest = (() => {
    const free = distancesFrom(graph, start, { blocked });
    let best = Infinity;
    for (const x of exits) best = Math.min(best, free[x]);
    return best;
  })();

  let path, moves, reached, burned, trialsUsed, certifiedOnClock, curve, movesCurve = null, escapeSteps, escapePath = null, deadlineStart = null, successRate, policyReturn = null, optimalReturn = null;
  if (startClosed) {
    notes.push(`The start is untenable when the evacuee leaves (${r1(depart)} s): the forecast closes it at ${r1(closes[start])} s.`);
    path = [start];
    moves = null;
    reached = false;
    burned = false;
    trialsUsed = 0;
    certifiedOnClock = false;
    curve = [];
    escapeSteps = null;
    successRate = 0;
  } else if (shielded) {
    const D = hopDeadlines(g, { exits, ignite: ig, horizon: H, blocked: gBlocked });
    const r = learnSafeEscape(g, start, { exits, ignite: ig, horizon: H, blocked: gBlocked, trials, deadline: D });
    const exact = earliestSafeEscape(g, start, { exits, ignite: ig, horizon: H, blocked: gBlocked });
    path = r.path;
    moves = r.moves;
    reached = r.reachedExit;
    burned = r.burned;
    trialsUsed = r.trials;
    certifiedOnClock = r.certified;
    movesCurve = r.curve;
    deadlineStart = r.deadlineStart;
    escapeSteps = exact ? exact.moves : null;
    escapePath = exact ? exact.path : null;
    const hits = r.curve.map((x) => (x == null ? 0 : 1));
    curve = curveOf(hits);
    const tail = hits.slice(-Math.max(1, Math.floor(hits.length / 5)));
    successRate = tail.length ? tail.reduce((s, x) => s + x, 0) / tail.length : 0;
    if (!exact) {
      if (exitHops(g, { exits, blocked: gBlocked })[start] >= NEVER) notes.push("The exit cannot be reached from the start without crossing a wall.");
      else notes.push(noEscapeNote(info));
    } else if (!r.reachedExit) notes.push(`An escape exists (${exact.moves} moves) but the policy did not find it; train for more episodes.`);
    else if (!r.certified) notes.push(`The policy escapes in ${r.moves} moves; the earliest escape takes ${exact.moves}. Train for more episodes to close the gap.`);
  } else {
    const ignite = new Map();
    ig.forEach((s, v) => ignite.set(v, s));
    const r = qLearningPath(g, start, exits[0], { ignite, episodes, maxSteps: H, seed, blocked: gBlocked });
    path = r.path;
    reached = r.reachedExit;
    burned = r.burned;
    moves = reached ? r.path.length - 1 : null;
    trialsUsed = r.episodes;
    certifiedOnClock = false;
    curve = r.curve;
    escapeSteps = r.escapeSteps;
    successRate = r.successRate;
    policyReturn = r.policyReturn ?? null;
    optimalReturn = r.optimalReturn ?? null;
    if (escapeSteps === null) notes.push(Number.isFinite(r.optimalLength) ? noEscapeNote(info) : "The exit cannot be reached from the start without crossing a wall.");
    else if (!reached) notes.push(`An escape exists (${escapeSteps} moves) but the policy did not find it; train for more episodes.`);
  }

  const real = realPath(clk, path);
  // The route walked in seconds along its real links: never late, by the
  // clock lemma; a late arrival is a bug and withdraws the certificate.
  const walked = reached ? walkInSeconds(graph, real, { forecast, closes, depart, speed: walkSpeed }) : null;
  const late = Boolean(walked && walked.firstUnsafe >= 0);
  if (late) notes.push(`Walked in seconds, the route reaches a node at ${r1(walked.times[walked.firstUnsafe])} s, after the forecast closes it: the clock should not allow that, so the result is not certified. Please report it.`);
  const realEscape = escapePath ? realPath(clk, escapePath) : null;
  return {
    graph_id: graph.graphId ?? null,
    method: shielded ? "shielded" : "qlearning",
    fire_source: "forecast",
    path: real,
    points: real.map((i) => [graph.nodes[i * 3], graph.nodes[i * 3 + 1], graph.nodes[i * 3 + 2]]),
    reached_exit: reached,
    episodes: trialsUsed,
    burned,
    reward: null,
    length: pathLength(graph, real),
    optimal_length: Number.isFinite(shortest) ? shortest : null,
    success_rate: successRate,
    curve,
    moves_curve: movesCurve,
    escape_steps: escapeSteps,
    escape_length: realEscape ? pathLength(graph, realEscape) : null,
    moves,
    certified: shielded ? certifiedOnClock && reached && !late : null,
    deadline_start: deadlineStart,
    exit_id: reached ? real[real.length - 1] : null,
    policy_return: policyReturn,
    optimal_return: optimalReturn,
    fire_start_id: fireStart,
    start_untenable: startClosed,
    forecast: info,
    seconds: walked
      ? { arrival: walked.arrival, late, firstUnsafe: walked.firstUnsafe, slack: walked.slack, slackNode: walked.slackNode, dose: walked.dose }
      : null,
    train_ms: Math.round(performance.now() - t0),
    note: notes.join(" ") || null,
  };
}

function noEscapeNote(info) {
  const within = info.limitBinds
    ? `within the step limit (${info.stepLimit} moves, to ${r1(info.stepLimitSeconds)} s on this clock); raise Max steps to look further`
    : `within the forecast (to ${r1(info.horizon)} s)`;
  return `No route escapes the forecast fire ${within}: it closes the way out before any path can pass.`;
}
