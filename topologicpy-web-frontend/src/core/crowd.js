/**
 * People leaving the building, for the Route step's optional egress
 * animation: a floor-field model on the walkable graph, timed by the SFPE
 * Handbook's hydraulic values.
 *
 * Where everyone goes. Each person heads for the nearest exit along the
 * shortest walk, the tree the performance check walks (performanceCheck.js
 * walkTree), as the "nearest exit" policy that gave the shortest exit times
 * in Carrera et al. (2019, doi:10.1007/978-3-319-99608-0_16) and that people
 * without a known exit take (Park et al. 2019, doi:10.1109/iemcon.2019.8936208).
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
 * Out. Past an exit each person walks on to a gathering place a few metres
 * outside, at the level of the lowest exit (an upper exit's people come
 * down to the ground), and stands there, the later arrivals round the
 * earlier ones, so each exit's crowd shows how many chose it.
 *
 * The result is each person's walk as straight segments (time, from, to),
 * which the view interpolates on the GPU from the playback time alone.
 */
import { csr, edgeLengths } from "./route.js";
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
 * @returns {{
 *   people: number, points: Float32Array, segments: Float32Array, first: Int32Array, count: Int32Array,
 *   exitOf: Int32Array, outAt: Float32Array, exits: Array<{node: number, people: number, last: number}>,
 *   unreached: number, inside: number, lastOut: number, longestWait: number, capped: boolean
 * }}
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
  } = opts;
  const n = graph.nodes.length / 3;
  const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
  const kinds = graph.kinds || new Uint8Array(n);
  const cell = (graph.spacing || 0.5) ** 2;
  const { indptr, neighbours, edgeIds } = csr(graph);
  const length = edgeLengths(graph);
  // Metres to the nearest exit, and the next node on the way.
  const { time: dist, next } = walkTree(graph, exits, { speed: 1, blocked });
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
    const g = { centre: p.map((x, c) => (c === up ? ground : x + dir[c] * 3.5)), dir, side, people: 0, last: 0 };
    gather.set(e, g);
    return g;
  };
  const spot = (g) => {
    // A sunflower: the j-th arrival 0.45 sqrt(j) m out at the golden angle.
    const j = g.people;
    g.people += 1;
    const r = 0.45 * Math.sqrt(j + 0.5), a = j * 2.399963;
    extra.push(g.centre.map((x, c) => x + r * (Math.cos(a) * g.dir[c] + Math.sin(a) * g.side[c])));
    return n + extra.length - 1;
  };

  const capacity = doorCapacity(doorWidth);
  const doorFree = new Float64Array(n);
  const segs = Array.from({ length: people }, () => []);
  const exitOf = new Int32Array(people).fill(-1);
  const outAt = new Float32Array(people).fill(Infinity);
  const waitSince = new Float64Array(people).fill(-1);
  let longestWait = 0;
  let left = people;
  // Nearer the exits first, so a queue moves up from its front in one step.
  let order = Array.from({ length: people }, (_, i) => i);
  const sortOrder = () => {
    order = order.filter((i) => exitOf[i] < 0);
    order.sort((p, q) => dist[at[p]] - dist[at[q]]);
  };
  sortOrder();
  // The density round a node: those holding it and its neighbours, over their floor.
  const densityAt = (u) => {
    let held = holder[u] >= 0 ? 1 : 0;
    const deg = indptr[u + 1] - indptr[u];
    for (let k = indptr[u]; k < indptr[u + 1]; k += 1) if (holder[neighbours[k]] >= 0) held += 1;
    return held / ((deg + 1) * cell);
  };

  let t = 0;
  for (let step = 0; left > 0 && t < horizon; step += 1, t = step * dt) {
    if (step % 10 === 0) sortOrder();
    for (const i of order) {
      if (exitOf[i] >= 0 || ready[i] > t + 1e-9) continue;
      const v = at[i];
      // The next node on the tree, or the best free neighbour as near the exit.
      let u = next[v];
      if (u >= 0 && holder[u] >= 0 && !isExit[u]) {
        const limit = dist[u] + 1.0;
        let best = -1;
        for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
          if (blocked?.[edgeIds[k]]) continue;
          const w = neighbours[k];
          if (holder[w] >= 0 || dist[w] >= dist[v] || dist[w] > limit || kinds[w] === KIND_LIFT) continue;
          if (best < 0 || dist[w] < dist[best]) best = w;
        }
        u = best;
      }
      const doorShut = u >= 0 && (kinds[u] === KIND_DOOR || isExit[u]) && doorFree[u] > t + 1e-9;
      if (u < 0 || doorShut) {
        if (waitSince[i] < 0) waitSince[i] = t;
        ready[i] = doorShut ? Math.max(t + dt, Math.ceil((doorFree[u] - 1e-9) / dt) * dt) : t + dt;
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
      const own = stair ? vmax[i] * (SFPE.kStair / SFPE.k) : vmax[i];
      const crowded = k0 * (1 - SFPE.a * densityAt(u));
      const vel = Math.max(0.15, Math.min(own, crowded));
      const arrive = t + len / vel;
      segs[i].push(t, arrive, v, u);
      if (kinds[u] === KIND_DOOR || isExit[u]) doorFree[u] = Math.max(doorFree[u], t) + 1 / capacity;
      holder[v] = -1;
      if (isExit[u]) {
        // Out: on to the gathering place, and the exit is free at once.
        const g = outward(u);
        const p = spot(g);
        const q = extra[p - n], e = pointOf(u);
        const walk = Math.hypot(q[0] - e[0], q[1] - e[1], q[2] - e[2]);
        segs[i].push(arrive, arrive + walk / vmax[i], u, p);
        exitOf[i] = u;
        outAt[i] = arrive;
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
