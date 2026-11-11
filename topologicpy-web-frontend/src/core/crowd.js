/**
 * People leaving the building, for the Route step's optional egress
 * animation: a floor-field model on the walkable graph, timed by the SFPE
 * Handbook's hydraulic values.
 *
 * Where everyone goes. By default (`choice` "quickest") each person weighs
 * every exit and takes the one that gets them out soonest: the walk there at
 * their own speed, or the people ahead of them for it (nearer it by the walk)
 * passing at the flow seen through it, whichever is longer. Each looks again
 * every `replan` seconds and turns to another exit only where the two ways
 * part and when it saves SWITCH_GAIN seconds and SWITCH_SHARE of their time,
 * so a queue's back leaves for a free door while its front stays, and no one
 * changes their mind over and over in a jam all the ways out share. The flow
 * through an exit is its door's capacity until someone heading for it is held
 * up, then what passes: a stair that feeds it slower than its door passes
 * makes it look as slow as it is. Pathfinder's "locally quickest" door choice
 * weighs a door the same way, by the walk and the person's place in its queue
 * at the door's observed flow, with a preference for the door already chosen
 * (Thunderhead Engineering, Pathfinder Technical Reference, Path Planning).
 * On the site's examples with no fire, the last one was out sooner where
 * queues formed at some exits while others stood free (the NIBS clinic 459
 * to 333 s, the NIBS office 433 to 350 s), and within 3 s on the others
 * (2026-11-10). With "nearest"
 * each person heads for the nearest exit along the shortest walk, the tree
 * the performance check walks (performanceCheck.js walkTree), as the
 * "nearest exit" policy that gave the shortest exit times in Carrera et al.
 * (2019, doi:10.1007/978-3-319-99608-0_16) and that people without a known
 * exit take (Park et al. 2019, doi:10.1109/iemcon.2019.8936208). The
 * quickest choice keeps a walk tree to each exit; on a graph where those
 * would take more than QUICKEST_CELLS, everyone takes the nearest.
 * A node holds one person at a time, as a cell of the static floor-field
 * automaton holds one (Burstedde et al. 2001, 0.4 m cells; the
 * graph's are about 0.5 m, a jam of about 4 persons/m^2). When the next node
 * on the tree is taken, a person steps to another free neighbour that is
 * also closer to an exit, so the crowd spreads over a corridor's width and
 * queues form behind doors rather than in single file.
 *
 * How fast. Unimpeded speeds spread about SFPE's maximum, 0.85 k = 1.19 m/s
 * with k = 1.40 m/s, and slow with the density D round the node ahead as
 * k (1 - a D), a = 0.266; on a stair k is 1.08 m/s along the slope (7 in
 * risers, 11 in treads), 0.92 m/s at most. A door passes at
 * most the SFPE specific flow of 1.3 persons/s per metre of effective width,
 * the door's less 0.15 m a side (performanceCheck.js doorCapacity): a door
 * of 0.9 m lets about 0.8 persons a second through.
 *
 * With a fire (`closes`, each node's untenable time by the planners' rule,
 * core/forecast.js), no one steps on to a place the fire has made
 * untenable: every `replan` seconds each way out is worked out again with a
 * closed place costing FIRE_PENALTY metres, so people turn back and take
 * another exit when the fire cuts theirs, and anyone the fire has reached
 * walks out of it the shortest way. Someone whose only way out is through
 * the fire stays where they are (refuge in place) and is counted as cut off.
 * With the quickest choice a person weighs only the exits whose walk the
 * fire has not cut; once it has cut every one, or reached them, they take
 * the way out worked out with the fire, as with the nearest.
 *
 * On a stair a person steps on only when no one else stands within
 * `stairGap` of the next node on the flight (0.9 m, about a person's
 * length), so a stream down a stair keeps its gaps.
 *
 * Randomness in how people move (off by default; the plan is
 * wiki/roadmap/crowd-motion-randomness-plan-2026-11-11.md). With `fluctuate`
 * each person's speed wobbles about their own as Corbetta et al. (2017)
 * measured walkers' do (WALK_FLUCTUATION). With `kS` each step is drawn as the
 * floor-field model draws it (Burstedde et al. 2001; Kirchner and
 * Schadschneider 2002): staying, or a free neighbour, with weight exp(kS x
 * the cells it brings nearer the exit), so people sometimes step aside or
 * hesitate, the more so the smaller kS. Both draw from a stream of their own,
 * so with both off a run is exactly as before; with either on, one run is one
 * sample, and the panel runs several.
 *
 * Out. Past an exit each person walks on to a gathering place a few metres
 * outside, at the level of the lowest exit (an upper exit's people come
 * down to the ground), and stands there, the later arrivals round the
 * earlier ones, so each exit's crowd shows how many chose it. With a
 * `shelterArea`, a gathering place holds what that area takes at the
 * spacing they stand at (GATHER_AREA a person); once full, each newcomer
 * takes the place of the earliest there, who leaves (`goneAt`) and fades
 * from the view.
 *
 * The result is each person's walk as straight segments (time, from, to),
 * which the view interpolates on the GPU from the playback time alone.
 */
import { Heap, csr, edgeLengths } from "./route.js";
import { KIND_DOOR, KIND_LIFT, KIND_STAIR } from "./lattice.js";
import { doorCapacity, walkTree } from "./performanceCheck.js";

/** The SFPE Handbook's hydraulic model (5th ed., 2016, ch. 59, Gwynne and Rosenbaum; doi:10.1007/978-1-4939-2565-0_59). */
export const SFPE = {
  // Speed k (1 - a D), m/s, D persons/m^2: corridors, ramps and doorways.
  k: 1.4,
  a: 0.266,
  // Along a stair's slope, 7 in risers and 11 in treads.
  kStair: 1.08,
  // The unimpeded speed, a share of k (the Handbook's table: 1.19 m/s for k = 1.40).
  free: 0.85,
};
/** The unimpeded walking speed, SFPE's 0.85 k: 1.19 m/s. */
export const FREE_SPEED = SFPE.free * SFPE.k;

/** Metres a place the fire has made untenable adds to a way out: only taken by those already in it. */
export const FIRE_PENALTY = 1000;

/** m^2 a person standing at a gathering place: the sunflower's spacing, pi 0.45^2. */
export const GATHER_AREA = Math.PI * 0.45 ** 2;

/** How each person picks an exit: the one that gets them out soonest, or the nearest. */
export const EXIT_CHOICES = ["quickest", "nearest"];

/** Seconds, and the share of their time out, another exit must save before someone heading for one turns to it. */
export const SWITCH_GAIN = 5;
export const SWITCH_SHARE = 0.2;

/** At most this many exits x nodes for the quickest choice (a walk tree an exit, 12 bytes a node); above it, the nearest. */
export const QUICKEST_CELLS = 2e6;

/**
 * Walking-speed fluctuations as Corbetta, Lee, Benzi, Muntean and Toschi
 * measured them on 72,376 walkers (2017, doi:10.1103/PhysRevE.95.032316;
 * arXiv:1610.07429, Table 1): the speed u along the walk follows
 * du = -4 alpha u (u^2 - up^2) dt + sigma dW, alpha 0.0625 m^-2 s,
 * sigma 0.16 m s^-3/2, up 1.0 m/s; about up it relaxes at 8 alpha up^2 =
 * 0.5 per second (about 2 s) with a spread of 0.16 m/s. Here as a factor
 * f = u / up on each person's own unimpeded speed, worked in steps of at most
 * `step` seconds and kept above `floor`, the unstable point (about 0.2 m/s):
 * the one walker in about 450 to 1,000 who turns back is left out.
 */
export const WALK_FLUCTUATION = { alpha: 0.0625, sigma: 0.16, up: 1.0, step: 0.1, floor: 0.2 };

/** The drift of the speed factor f: -4 alpha up^2 f (f^2 - 1), zero at f = 1. */
export const fluctuationDrift = (f) => -4 * WALK_FLUCTUATION.alpha * WALK_FLUCTUATION.up ** 2 * f * (f * f - 1);

/**
 * The floor field's sensitivity kS for the step choice (Kirchner and
 * Schadschneider 2002, doi:10.1016/S0378-4371(02)00857-9, who ran 0.1, 0.4
 * and 10): null steps as before (the limit of a large kS); 10 their "strong",
 * near-ordered walking; 0.4 their intermediate, people far less sure of the
 * way; 2 between.
 */
export const STEP_RANDOMNESS = { exact: null, direct: 10, wandering: 2, disoriented: 0.4 };

/** Seconds the flow seen through an exit is averaged over. */
export const FLOW_SECONDS = 20;

/** The least share of its door's capacity an exit's flow is taken at, so one no one has reached yet is not thought shut. */
export const FLOW_FLOOR = 0.1;

/** A seeded uniform generator (mulberry32), so a crowd is the same each time. */
function random(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The crowd's walk out.
 *
 * @param {object} graph  the walkable graph (nodes, edges, kinds, blocked, spacing, agentHeight, upAxis)
 * @param {object} opts
 *   exits         the exit nodes (the plan's: exits out of the building)
 *   occupantArea  m^2 of walkable floor per person (the performance check's)
 *   maxPeople     at most this many, spread alike over the floors
 *   speed         the mean unimpeded speed, m/s (FREE_SPEED)
 *   spread        each person's speed is speed x (1 +- spread), uniform
 *   start         [earliest, latest] start, s: a short stagger, not a pre-movement time
 *   doorWidth     m, every door's (the IFC's widths are not read)
 *   dt            s, the step
 *   horizon       s, stop then; anyone still inside is reported
 *   seed          for where people are and how fast they walk
 *   blocked       edges no one may cross (walls)
 *   stairGap      m kept clear round the next node on a stair (0: none)
 *   shelterArea   m^2 each gathering place holds (Infinity: everyone)
 *   closes        each node's untenable time, s (a fire's; null: none)
 *   replan        s between ways out worked out again with a fire, and exits weighed again
 *   choice        "quickest" (default) or "nearest" (EXIT_CHOICES)
 *   fluctuate     each person's speed wobbles in time (WALK_FLUCTUATION; default off)
 *   kS            the floor field's step choice (STEP_RANDOMNESS; null, the default: as before)
 * @returns {{
 *   people: number, points: Float32Array, segments: Float32Array, first: Int32Array, count: Int32Array,
 *   exitOf: Int32Array, outAt: Float32Array, exits: Array<{node: number, people: number, last: number}>,
 *   unreached: number, inside: number, lastOut: number, longestWait: number, capped: boolean,
 *   goneAt: Float32Array, shelter: number, cutOff: number, choice: string, tooLarge: boolean,
 *   switches: Int32Array
 * }}
 *   choice: the one taken, "nearest" when the quickest was asked with a
 *   single exit or on a graph too large for it (tooLarge, QUICKEST_CELLS);
 *   switches: how many times each person turned to another exit after the
 *   start (0 with the nearest);
 *   points: xyz of the graph's nodes at foot level, then the gathering places;
 *   segments: 4 numbers each, start s, end s, point from, point to, each person's
 *   from first[i], count[i] of them, in time order (a gap between two is a wait).
 */
export function simulateCrowd(graph, opts = {}) {
  const {
    exits = [],
    occupantArea = 9,
    maxPeople = 4000,
    speed = FREE_SPEED,
    spread = 0.25,
    start = [0, 10],
    doorWidth = 0.9,
    dt = 0.1,
    horizon = 900,
    seed = 1,
    blocked = null,
    stairGap = 0.9,
    shelterArea = Infinity,
    closes = null,
    replan = 2,
    choice = "quickest",
    fluctuate = false,
    kS = null,
  } = opts;
  const n = graph.nodes.length / 3;
  const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
  const kinds = graph.kinds || new Uint8Array(n);
  const cell = (graph.spacing || 0.5) ** 2;
  const { indptr, neighbours, edgeIds } = csr(graph);
  const length = edgeLengths(graph);
  // Metres to the nearest exit, and the next node on the way.
  let { time: dist, next } = walkTree(graph, exits, { speed: 1, blocked });
  // The same with the places the fire has closed by t costing FIRE_PENALTY each.
  const closedAt = (v, t) => closes !== null && closes[v] <= t;
  const openTree = (t) => {
    const time = new Float64Array(n).fill(Infinity);
    const via = new Int32Array(n).fill(-1);
    const heap = new Heap();
    for (const e of exits) {
      if (e < 0) continue;
      const start = closedAt(e, t) ? FIRE_PENALTY : 0;
      if (start >= time[e]) continue;
      time[e] = start;
      heap.push(start, e);
    }
    const done = new Uint8Array(n);
    while (heap.size) {
      const a = heap.pop();
      if (done[a]) continue;
      done[a] = 1;
      for (let k = indptr[a]; k < indptr[a + 1]; k += 1) {
        if (blocked?.[edgeIds[k]]) continue;
        const b = neighbours[k];
        const d = time[a] + length[edgeIds[k]] + (closedAt(b, t) ? FIRE_PENALTY : 0);
        if (d < time[b]) {
          time[b] = d;
          via[b] = a;
          heap.push(d, b);
        }
      }
    }
    return { time, next: via };
  };
  const isExit = new Uint8Array(n);
  for (const e of exits) if (e >= 0) isExit[e] = 1;

  // Where people are: walkable nodes that reach an exit (not doors, stairs
  // or lifts), as many as the floor's area holds at the occupant area.
  const rand = random(seed);
  const places = [];
  let unreached = 0;
  for (let v = 0; v < n; v += 1) {
    if (kinds[v] === KIND_DOOR || kinds[v] === KIND_STAIR || kinds[v] === KIND_LIFT || isExit[v]) continue;
    if (Number.isFinite(dist[v])) places.push(v);
    else unreached += 1;
  }
  const wanted = Math.ceil((places.length * cell) / Math.max(0.1, occupantArea));
  const people = Math.min(wanted, maxPeople, places.length);
  // A partial Fisher-Yates shuffle picks them.
  for (let i = 0; i < people; i += 1) {
    const j = i + Math.floor(rand() * (places.length - i));
    [places[i], places[j]] = [places[j], places[i]];
  }
  const at = Int32Array.from(places.slice(0, people));
  const vmax = new Float64Array(people);
  const ready = new Float64Array(people);
  for (let i = 0; i < people; i += 1) {
    vmax[i] = speed * (1 + spread * (2 * rand() - 1));
    ready[i] = start[0] + (start[1] - start[0]) * rand();
  }
  const holder = new Int32Array(n).fill(-1);
  for (let i = 0; i < people; i += 1) holder[at[i]] = i;

  // Foot-level points: the nodes stand agentHeight above their floor.
  const lift = graph.agentHeight || 0;
  const extra = [];
  const pointOf = (v) => [graph.nodes[v * 3] - (up === 0 ? lift : 0), graph.nodes[v * 3 + 1] - (up === 1 ? lift : 0), graph.nodes[v * 3 + 2] - (up === 2 ? lift : 0)];
  // Each exit's gathering place: a few metres out, away from the nodes inside, on the ground.
  const gather = new Map();
  const ground = Math.min(...exits.filter((e) => e >= 0).map((e) => pointOf(e)[up]));
  const outward = (e) => {
    if (gather.has(e)) return gather.get(e);
    const p = pointOf(e);
    const d = [0, 0, 0];
    for (let k = indptr[e]; k < indptr[e + 1]; k += 1) {
      const q = pointOf(neighbours[k]);
      for (let c = 0; c < 3; c += 1) if (c !== up) d[c] += p[c] - q[c];
    }
    let norm = Math.hypot(d[0], d[1], d[2]);
    if (norm < 1e-9) {
      d[(up + 1) % 3] = 1;
      norm = 1;
    }
    const dir = d.map((x) => x / norm);
    const side = [0, 0, 0];
    // The horizontal perpendicular: dir rotated a quarter turn about up.
    const [h1, h2] = [0, 1, 2].filter((c) => c !== up);
    side[h1] = -dir[h2];
    side[h2] = dir[h1];
    const g = { centre: p.map((x, c) => (c === up ? ground : x + dir[c] * 3.5)), dir, side, people: 0, last: 0, spots: [], here: [] };
    gather.set(e, g);
    return g;
  };
  // How many a gathering place holds, and when each person left a full one.
  const shelter = Number.isFinite(shelterArea) ? Math.max(1, Math.floor(shelterArea / GATHER_AREA)) : Infinity;
  const goneAt = new Float32Array(people).fill(Infinity);
  const spot = (g, i, t) => {
    // A sunflower: the k-th place 0.45 sqrt(k) m out at the golden angle; in
    // a full one the newcomer takes the place of the earliest still there.
    const j = g.people;
    g.people += 1;
    g.here.push(i);
    if (j >= shelter) goneAt[g.here[j - shelter]] = t;
    const k = j % shelter;
    if (g.spots[k] === undefined) {
      const r = 0.45 * Math.sqrt(k + 0.5), a = k * 2.399963;
      extra.push(g.centre.map((x, c) => x + r * (Math.cos(a) * g.dir[c] + Math.sin(a) * g.side[c])));
      g.spots[k] = n + extra.length - 1;
    }
    return g.spots[k];
  };

  const capacity = doorCapacity(doorWidth);
  const doorFree = new Float64Array(n);
  const segs = Array.from({ length: people }, () => []);
  const exitOf = new Int32Array(people).fill(-1);
  const outAt = new Float32Array(people).fill(Infinity);
  const waitSince = new Float64Array(people).fill(-1);
  let longestWait = 0;
  let left = people;

  // The quickest choice: a walk tree to each exit, and the one each person
  // heads for (goal, an index into them; -1: the way out worked out with the
  // fire, as everyone's with the nearest choice).
  const exitList = [...new Set(exits.filter((e) => e >= 0))];
  const trees =
    choice === "quickest" && exitList.length > 1 && exitList.length * n <= QUICKEST_CELLS
      ? exitList.map((e) => walkTree(graph, [e], { speed: 1, blocked }))
      : null;
  const goal = new Int32Array(people).fill(-1);
  // How many times each turned to another exit after the start.
  const switches = new Int32Array(people);
  // Each person's metres to their exit, and the next node on the way.
  const distTo = (i) => (goal[i] >= 0 ? trees[goal[i]].time : dist);
  const nextTo = (i) => (goal[i] >= 0 ? trees[goal[i]].next : next);
  // Who heads for each exit, by metres from it: a Fenwick tree, a slot every
  // 0.1 m (a slot a metre wide left out a dense queue's band round each person).
  const slot = (d) => Math.floor(d * 10);
  let slots = 2;
  if (trees) for (const { time } of trees) for (let v = 0; v < n; v += 1) if (time[v] < Infinity) slots = Math.max(slots, slot(time[v]) + 2);
  const heading = trees ? trees.map(() => new Int32Array(slots + 1)) : [];
  const tally = (g, d, k) => {
    for (let x = Math.min(slots, slot(d) + 1); x <= slots; x += x & -x) heading[g][x] += k;
  };
  // How many heading for exit g are nearer it than d (by a slot or more).
  const ahead = (g, d) => {
    let s = 0;
    for (let x = Math.min(slots, slot(d)); x > 0; x -= x & -x) s += heading[g][x];
    return s;
  };
  // The flow seen out through each exit, persons/s: its door's capacity at
  // first, then an exponential average over FLOW_SECONDS of those who passed
  // while someone heading for it was held up, so an exit a stair feeds slower
  // than its door passes looks as slow as it is. With no one held up for it,
  // what passes says how many want it, not how many it takes: its flow goes
  // back towards the door's capacity.
  const flow = new Float64Array(exitList.length).fill(capacity);
  const passed = new Int32Array(exitList.length);
  const held = new Uint8Array(exitList.length);
  const exitIndex = new Map(exitList.map((e, g) => [e, g]));
  let lastWeigh = 0;
  // Seconds till person i at v is out by exit g: the walk there, or those ahead through it at its flow, whichever is longer.
  const outBy = (i, g, v) => {
    const before = ahead(g, trees[g].time[v]);
    return Math.max(trees[g].time[v] / vmax[i], before > 0 ? before / Math.max(flow[g], FLOW_FLOOR * capacity) : 0);
  };
  // Whether the walk from v to exit g is clear of the fire at t.
  const clear = (g, v, t) => {
    if (closes === null) return true;
    for (let u = v; u >= 0; u = trees[g].next[u]) if (closedAt(u, t)) return false;
    return true;
  };
  // Everyone still inside weighs the exits again, nearer first, each counted
  // for the exit they take before the next weighs theirs. At the start each
  // takes the quickest clear one; later they keep theirs unless another saves
  // SWITCH_GAIN or the fire cuts it, and once the fire has cut every one they
  // keep to the way out worked out with it.
  const weigh = (t) => {
    if (t > lastWeigh) {
      const span = t - lastWeigh, a = 1 - Math.exp(-span / FLOW_SECONDS);
      for (let g = 0; g < flow.length; g += 1) flow[g] += ((held[g] ? passed[g] / span : capacity) - flow[g]) * a;
      passed.fill(0);
      held.fill(0);
      lastWeigh = t;
    }
    for (const h of heading) h.fill(0);
    for (const i of order) if (goal[i] >= 0) tally(goal[i], trees[goal[i]].time[at[i]], 1);
    for (const i of order) {
      const v = at[i], own = goal[i];
      if (own < 0 && t > 0) continue;
      let keep = Infinity;
      if (own >= 0) {
        tally(own, trees[own].time[v], -1);
        if (clear(own, v, t)) keep = outBy(i, own, v);
      }
      let pick = keep < Infinity ? own : -1;
      let bar = keep < Infinity ? Math.min(keep - SWITCH_GAIN, keep * (1 - SWITCH_SHARE)) : Infinity;
      for (let g = 0; g < trees.length; g += 1) {
        if (g === own || !(trees[g].time[v] < Infinity)) continue;
        // While the two ways go on together there is nothing to choose yet.
        if (keep < Infinity && trees[g].next[v] === trees[own].next[v]) continue;
        const out = outBy(i, g, v);
        if (out < bar && clear(g, v, t)) {
          pick = g;
          bar = out;
        }
      }
      if (t > 0 && pick !== own && pick >= 0) switches[i] += 1;
      goal[i] = pick;
      if (pick >= 0) tally(pick, trees[pick].time[v], 1);
    }
  };

  // Nearer the exits first, so a queue moves up from its front in one step.
  let order = Array.from({ length: people }, (_, i) => i);
  const sortOrder = () => {
    order = order.filter((i) => exitOf[i] < 0);
    order.sort((p, q) => distTo(p)[at[p]] - distTo(q)[at[q]]);
  };
  sortOrder();
  if (trees) {
    weigh(0);
    sortOrder();
  }
  // The stair nodes within stairGap of each stair node, by the walk, found when first needed.
  const nearStair = new Map();
  const stairNear = (u) => {
    if (nearStair.has(u)) return nearStair.get(u);
    const found = [];
    const seen = new Map([[u, 0]]);
    const queue = [u];
    while (queue.length) {
      const a = queue.shift();
      for (let k = indptr[a]; k < indptr[a + 1]; k += 1) {
        const b = neighbours[k], d = seen.get(a) + length[edgeIds[k]];
        if (d > stairGap || (seen.has(b) && seen.get(b) <= d) || kinds[b] !== KIND_STAIR) continue;
        seen.set(b, d);
        found.push(b);
        queue.push(b);
      }
    }
    nearStair.set(u, found);
    return found;
  };
  // Whether a person at v may step on to u: no one holds it, and on a stair no one else stands close by.
  const free = (u, v) => {
    if (holder[u] >= 0) return false;
    if (stairGap > 0 && kinds[u] === KIND_STAIR) for (const w of stairNear(u)) if (w !== v && holder[w] >= 0) return false;
    return true;
  };
  // The density round a node: those holding it and its neighbours, over their floor.
  const densityAt = (u) => {
    let held = holder[u] >= 0 ? 1 : 0;
    const deg = indptr[u + 1] - indptr[u];
    for (let k = indptr[u]; k < indptr[u + 1]; k += 1) if (holder[neighbours[k]] >= 0) held += 1;
    return held / ((deg + 1) * cell);
  };

  // Randomness in how people move (wiki/roadmap/crowd-motion-randomness-plan-2026-11-11.md,
  // stage 1), drawn from a stream of its own so that with it off a run is
  // exactly as it was. Speed: each person's factor on their unimpeded speed
  // follows Corbetta et al.'s Langevin equation, worked to the time asked.
  const wobble = random((seed ^ 0x9e3779b9) >>> 0);
  const gauss = () => {
    const a = Math.max(1e-12, wobble()), b = wobble();
    return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
  };
  const factor = new Float64Array(people).fill(1);
  const factorAt = new Float64Array(people);
  const speedFactor = (i, t) => {
    if (!fluctuate) return 1;
    let f = factor[i];
    for (let h = t - factorAt[i]; h > 1e-9; h -= WALK_FLUCTUATION.step) {
      const s = Math.min(WALK_FLUCTUATION.step, h);
      f += fluctuationDrift(f) * s + (WALK_FLUCTUATION.sigma / WALK_FLUCTUATION.up) * Math.sqrt(s) * gauss();
      f = Math.max(WALK_FLUCTUATION.floor, f);
    }
    factor[i] = f;
    factorAt[i] = t;
    return f;
  };
  // Steps: with kS set, staying or each free open neighbour, weighed as the
  // floor field weighs a cell (Kirchner and Schadschneider 2002, Eq. 1, with
  // no dynamic field): exp(kS x the cells it brings nearer the exit).
  const spacing = graph.spacing || 0.5;
  const pickStep = (i, v, D, inFire, t) => {
    const options = [v], logw = [0];
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      if (blocked?.[edgeIds[k]]) continue;
      const w = neighbours[k];
      if (kinds[w] === KIND_LIFT || (closedAt(w, t) && !inFire) || !(D[w] < Infinity)) continue;
      if (!isExit[w] && !free(w, v)) continue;
      options.push(w);
      logw.push((-kS * (D[w] - D[v])) / spacing);
    }
    let top = -Infinity;
    for (const l of logw) if (l > top) top = l;
    let total = 0;
    const weights = logw.map((l) => {
      const e = Math.exp(l - top);
      total += e;
      return e;
    });
    let r = wobble() * total;
    for (let j = 0; j < options.length; j += 1) {
      r -= weights[j];
      if (r <= 0) return options[j];
    }
    return options[options.length - 1];
  };

  let t = 0;
  const every = Math.max(1, Math.round(replan / dt));
  for (let step = 0; left > 0 && t < horizon; step += 1, t = step * dt) {
    // With a fire, the ways out over the places still open, now and then.
    if (closes !== null && step > 0 && step % every === 0) ({ time: dist, next } = openTree(t));
    if (trees && step > 0 && step % every === 0) weigh(t);
    if (step % 10 === 0) sortOrder();
    for (const i of order) {
      if (exitOf[i] >= 0 || ready[i] > t + 1e-9) continue;
      const v = at[i];
      const D = distTo(i);
      // The next node on the tree, or the best free neighbour as near the
      // exit (any open one nearer, from a place the fire has closed).
      let u = nextTo(i)[v];
      // From an open place no one steps into the fire; from inside it, anywhere out.
      const inFire = closedAt(v, t);
      if (kS !== null) {
        // The floor field's step, or staying a step where they are.
        u = pickStep(i, v, D, inFire, t);
        if (u === v) {
          ready[i] = t + dt;
          continue;
        }
      } else if (u >= 0 && closedAt(u, t) && !inFire) u = -1;
      if (kS === null && (u < 0 || (!isExit[u] && !free(u, v)))) {
        const limit = u >= 0 ? D[u] + 1.0 : Infinity;
        let best = -1;
        for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
          if (blocked?.[edgeIds[k]]) continue;
          const w = neighbours[k];
          if (!free(w, v) || (closedAt(w, t) && !inFire) || !(D[w] < D[v]) || D[w] > limit || kinds[w] === KIND_LIFT) continue;
          if (best < 0 || D[w] < D[best]) best = w;
        }
        u = best;
      }
      const doorShut = u >= 0 && (kinds[u] === KIND_DOOR || isExit[u]) && doorFree[u] > t + 1e-9;
      if (u < 0 || doorShut) {
        if (waitSince[i] < 0) waitSince[i] = t;
        if (goal[i] >= 0) held[goal[i]] = 1;
        // Cut off by the fire: wait for the next look at the ways out.
        const cut = u < 0 && goal[i] < 0 && dist[v] >= FIRE_PENALTY;
        ready[i] = doorShut ? Math.max(t + dt, Math.ceil((doorFree[u] - 1e-9) / dt) * dt) : cut ? (Math.floor(step / every) + 1) * every * dt : t + dt;
        continue;
      }
      if (waitSince[i] >= 0) {
        longestWait = Math.max(longestWait, t - waitSince[i]);
        waitSince[i] = -1;
      }
      let len = 0;
      for (let k = indptr[v]; k < indptr[v + 1]; k += 1) if (neighbours[k] === u) len = length[edgeIds[k]];
      const stair = kinds[u] === KIND_STAIR || kinds[v] === KIND_STAIR;
      const k0 = stair ? SFPE.kStair : SFPE.k;
      const own = (stair ? vmax[i] * (SFPE.kStair / SFPE.k) : vmax[i]) * speedFactor(i, t);
      const crowded = k0 * (1 - SFPE.a * densityAt(u));
      const vel = Math.max(0.15, Math.min(own, crowded));
      const arrive = t + len / vel;
      segs[i].push(t, arrive, v, u);
      if (kinds[u] === KIND_DOOR || isExit[u]) doorFree[u] = Math.max(doorFree[u], t) + 1 / capacity;
      holder[v] = -1;
      if (isExit[u]) {
        // Out: on to the gathering place, and the exit is free at once.
        const g = outward(u);
        const p = spot(g, i, arrive);
        const q = extra[p - n], e = pointOf(u);
        const walk = Math.hypot(q[0] - e[0], q[1] - e[1], q[2] - e[2]);
        segs[i].push(arrive, arrive + walk / vmax[i], u, p);
        exitOf[i] = u;
        outAt[i] = arrive;
        if (trees) passed[exitIndex.get(u)] += 1;
        g.last = Math.max(g.last, arrive);
        left -= 1;
        continue;
      }
      holder[u] = i;
      at[i] = u;
      ready[i] = arrive;
    }
  }

  // Anyone who never moved (boxed in till the horizon) stands where they are.
  for (let i = 0; i < people; i += 1) if (!segs[i].length) segs[i].push(t, t, at[i], at[i]);
  // Packed for the view.
  const first = new Int32Array(people), count = new Int32Array(people);
  let total = 0;
  for (let i = 0; i < people; i += 1) {
    first[i] = total;
    count[i] = segs[i].length / 4;
    total += count[i];
  }
  const segments = new Float32Array(total * 4);
  for (let i = 0; i < people; i += 1) segments.set(segs[i], first[i] * 4);
  const points = new Float32Array((n + extra.length) * 3);
  for (let v = 0; v < n; v += 1) points.set(pointOf(v), v * 3);
  extra.forEach((p, j) => points.set(p, (n + j) * 3));
  let lastOut = 0;
  for (let i = 0; i < people; i += 1) if (Number.isFinite(outAt[i])) lastOut = Math.max(lastOut, outAt[i]);
  // Still inside with no open way out at the end: cut off by the fire.
  let cutOff = 0;
  if (closes !== null) for (let i = 0; i < people; i += 1) if (exitOf[i] < 0 && dist[at[i]] >= FIRE_PENALTY) cutOff += 1;
  return {
    people,
    points,
    segments,
    first,
    count,
    exitOf,
    outAt,
    exits: [...gather.entries()].map(([node, g]) => ({ node, people: g.people, last: g.last })).sort((a, b) => b.people - a.people),
    unreached,
    inside: left,
    lastOut,
    longestWait,
    capped: wanted > people,
    doorCapacity: capacity,
    goneAt,
    shelter,
    cutOff,
    choice: trees ? "quickest" : "nearest",
    tooLarge: choice === "quickest" && exitList.length > 1 && !trees,
    switches,
  };
}

/**
 * One person running down a stair on a loop, to show the stair: of the
 * crowd's walks, the one that goes down the most stair nodes, from a few
 * steps before its first stair node to a few after its last, at that walk's
 * own speeds with its waits left out, as a one-person crowd from time 0.
 * Null when no walk takes a stair down at least `drop` metres.
 *
 * @returns {null | {people: 1, points: Float32Array, segments: Float32Array, first: Int32Array, count: Int32Array, outAt: Float32Array, duration: number}}
 */
export function stairRunner(graph, crowd, { margin = 4, drop = 1.5 } = {}) {
  if (!crowd?.people) return null;
  const n = graph.nodes.length / 3;
  const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
  const kinds = graph.kinds || new Uint8Array(n);
  let best = null;
  for (let i = 0; i < crowd.people; i += 1) {
    // The walk's moves between graph nodes (not the last, to the gathering place).
    const moves = [];
    for (let j = 0; j < crowd.count[i]; j += 1) {
      const q = (crowd.first[i] + j) * 4;
      const a = crowd.segments[q + 2], b = crowd.segments[q + 3];
      if (a < n && b < n && a !== b) moves.push({ a, b, time: crowd.segments[q + 1] - crowd.segments[q] });
    }
    const onStair = moves.map((m) => kinds[m.a] === KIND_STAIR || kinds[m.b] === KIND_STAIR);
    const firstStair = onStair.indexOf(true), lastStair = onStair.lastIndexOf(true);
    if (firstStair < 0) continue;
    const from = Math.max(0, firstStair - margin), to = Math.min(moves.length - 1, lastStair + margin);
    const height = crowd.points[moves[from].a * 3 + up] - crowd.points[moves[to].b * 3 + up];
    const steps = onStair.slice(firstStair, lastStair + 1).filter(Boolean).length;
    if (height >= drop && (!best || steps > best.steps)) best = { steps, moves: moves.slice(from, to + 1) };
  }
  if (!best) return null;
  // Its own points, so the view uploads a few rather than the whole graph.
  const nodes = [best.moves[0].a, ...best.moves.map((m) => m.b)];
  const points = new Float32Array(nodes.length * 3);
  nodes.forEach((v, k) => points.set(crowd.points.subarray(v * 3, v * 3 + 3), k * 3));
  const segments = new Float32Array(best.moves.length * 4);
  let t = 0;
  best.moves.forEach((m, k) => {
    segments.set([t, t + m.time, k, k + 1], k * 4);
    t += m.time;
  });
  return { people: 1, points, segments, first: Int32Array.of(0), count: Int32Array.of(best.moves.length), outAt: Float32Array.of(Infinity), duration: t };
}

/**
 * One person on the forecast's way out (core/forecast.js planWithForecast:
 * its earliest safe route, a node list and the time each is reached, in
 * seconds from ignition), as a one-person crowd on that timeline: standing
 * at the start until the plan sets off, out at its arrival. Null without a
 * route.
 */
export function forecastRunner(graph, route) {
  const nodes = route?.nodes, times = route?.times;
  if (!nodes?.length || nodes.length < 2 || times?.length !== nodes.length) return null;
  const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
  const lift = graph.agentHeight || 0;
  const points = new Float32Array(nodes.length * 3);
  nodes.forEach((v, k) => {
    for (let c = 0; c < 3; c += 1) points[k * 3 + c] = graph.nodes[v * 3 + c] - (c === up ? lift : 0);
  });
  const segments = new Float32Array((nodes.length - 1) * 4);
  for (let k = 0; k + 1 < nodes.length; k += 1) segments.set([times[k], Math.max(times[k], times[k + 1]), k, k + 1], k * 4);
  const arrival = times[times.length - 1];
  return { people: 1, points, segments, first: Int32Array.of(0), count: Int32Array.of(nodes.length - 1), outAt: Float32Array.of(arrival), duration: arrival };
}
