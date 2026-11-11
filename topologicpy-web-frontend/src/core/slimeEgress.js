/**
 * A slime mould's way out: Jones's Physarum agents on the building's
 * walkable graph, drawn to the exits and kept from the fire, whose trail
 * weighs the ways out (wiki/research/physarum-jones-models-2026-11-11.md,
 * section 4).
 *
 * The agents are Jones's (2010, doi:10.1162/artl.2010.16.2.16202), with the
 * rules of his example code: each stands on a node with a heading, senses
 * ahead and SA to either side at SO links out, and turns RA (keeping its
 * heading when ahead is strictly the strongest; when both sides beat ahead,
 * towards the larger, left on a tie; else towards the larger side); then,
 * in a second pass, steps forward if the node there is free (one agent a
 * node) and lays `depT` of trail, a blocked step taking a new heading at
 * random and a share pCD of steps a random turn instead. SA and RA are 60
 * degrees as in his map example. The exits are his food: each step each
 * feeds `food` to itself and its neighbours, `foodCovered` while an agent
 * stands there. The trail then becomes (1 - damp) times the mean over the
 * node and its eight lattice neighbours, a missing neighbour counting as
 * none, so walls absorb trail as his grid's do (a stair, a chain of single
 * nodes, takes the mean over the neighbours it has). On the graph a sensor
 * is the node reached by marching SO links, each along the link best
 * aligned with the sensor's direction (within SENSE_CONE); a step goes
 * along the link best aligned with the heading. Where his grid differs: a
 * sensor stopped by a wall or the fire reads -Infinity (his walls read an
 * empty trail), and an agent turns RA when all three are stopped (his, when
 * any sensor leaves the grid). Stairs are links like any other, so the
 * mould climbs between storeys. The turn and the one-agent-a-node movement
 * are proved in formal/fireegress/FireEgress/Physarum.lean.
 *
 * What the agents sense. Jones's food is a diffusing source, flat far from
 * it (the plateau Jones notes); here the exits draw as the walking distance
 * to them, the static floor field of Burstedde et al. (2001,
 * doi:10.1016/S0378-4371(01)00141-8), so every place keeps a gradient. The
 * trail adds to it, as the dynamic field of Kirchner and Schadschneider's
 * evacuation model does (2002, doi:10.1016/S0378-4371(02)00857-9):
 *
 *   S(v) = -dExit(v) + trailWeight T(v) / (T(v) + trailHalf) - repel exp(-dFire(v) / repelLength)
 *
 * in metres of walk: a full tube is worth `trailWeight` metres, and the
 * fire's edge costs `repel` metres, falling off over `repelLength`. A place
 * the fire has made untenable (`closes` at or before `time`) is a wall: no
 * agent enters it, no trail spreads into it, and dExit goes round it, as the
 * maze's walls do in Jones and Saeed's chemotactic paths (2007,
 * doi:10.3233/mgs-2007-3108).
 *
 * A flux, not a crowd. Agents start on the occupied floor; one that steps
 * through an exit starts again at a random free place (Jones's teleport), so
 * the trail measures the escape traffic each place carries. The trail
 * averaged over the second half of the run, over its 99th percentile and
 * capped at 1, is each node's weight. The route from `start` is the cheapest walk to an exit where a
 * link costs its length times 1 + tube (1 - its weight): it keeps to the
 * tubes, and is reported against the shortest walk.
 *
 * Seeded (mulberry32): one seed, one run. An illustration of where the
 * escape concentrates, not an evacuation model to quote (forecast.js and
 * crowd.js time people).
 */
import { Heap, csr, edgeLengths } from "./route.js";
import { KIND_DOOR, KIND_LIFT, KIND_STAIR } from "./lattice.js";

/** The run's settings: Jones's agent (SO in links, SA and RA in degrees), and the field's weights in metres of walk. */
export const SLIME_DEFAULTS = {
  agents: 2000,
  steps: 400,
  so: 3,
  sa: 60,
  ra: 60,
  depT: 5,
  damp: 0.1,
  pCD: 0.02,
  food: 10,
  foodCovered: 2.5,
  trailWeight: 3,
  trailHalf: 20,
  repel: 10,
  repelLength: 3,
  tube: 2,
};

/** Cosine of the widest angle off a direction a step or a sensor may take a link: 67.5 degrees, so one of the lattice's eight links always qualifies in open floor. */
export const SENSE_CONE = Math.cos((67.5 * Math.PI) / 180);

/** A seeded uniform generator (mulberry32), so a run is the same each time. */
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

/** Metres from the sources over the open links (Dijkstra). */
function walkFrom(sources, n, adj, open) {
  const dist = new Float64Array(n).fill(Infinity);
  const heap = new Heap();
  for (const s of sources) {
    if (s < 0 || !open[s] || dist[s] === 0) continue;
    dist[s] = 0;
    heap.push(0, s);
  }
  const done = new Uint8Array(n);
  while (heap.size) {
    const a = heap.pop();
    if (done[a]) continue;
    done[a] = 1;
    for (let k = adj.start[a]; k < adj.start[a + 1]; k += 1) {
      const b = adj.to[k];
      const d = dist[a] + adj.len[k];
      if (d < dist[b]) {
        dist[b] = d;
        heap.push(d, b);
      }
    }
  }
  return dist;
}

/**
 * The slime mould's run.
 *
 * @param {object} graph  the walkable graph (nodes, edges, kinds, blocked, upAxis)
 * @param {object} opts
 *   exits    the exit nodes
 *   start    a node to route from (optional)
 *   closes   each node's untenable time, s (a fire's; null: none)
 *   time     s: places untenable by then are walls (Infinity: every place the fire ever closes)
 *   places   nodes agents start from (default: all the open floor)
 *   blocked  links no one may cross (walls)
 *   seed, and SLIME_DEFAULTS' settings
 * @returns {{weights: Float32Array, visits: Uint32Array, exits: Array<{node: number, flux: number}>,
 *            route: null | {nodes: number[], length: number, shortest: number}, agents: number, steps: number,
 *            closed: number, unreached: number}}
 */
export function slimeEgress(graph, opts = {}) {
  const { exits = [], start = -1, closes = null, time = Infinity, blocked = null, seed = 1 } = opts;
  const p = { ...SLIME_DEFAULTS, ...opts };
  const n = graph.nodes.length / 3;
  const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
  const [h0, h1] = [0, 1, 2].filter((c) => c !== up);
  const kinds = graph.kinds || new Uint8Array(n);
  const P = graph.nodes;

  // Open nodes: not made untenable by the fire by `time`, and no lift (a fire evacuation does not use them).
  const open = new Uint8Array(n);
  let closed = 0;
  for (let v = 0; v < n; v += 1) {
    const shut = closes !== null && closes[v] <= time;
    if (shut) closed += 1;
    open[v] = shut || kinds[v] === KIND_LIFT ? 0 : 1;
  }
  // The open links in compressed rows, with each link's length and plan direction.
  const { indptr, neighbours, edgeIds } = csr(graph);
  const length = edgeLengths(graph);
  const start0 = new Int32Array(n + 1);
  const to = [], len = [], dx = [], dy = [];
  for (let a = 0; a < n; a += 1) {
    start0[a] = to.length;
    if (!open[a]) continue;
    for (let k = indptr[a]; k < indptr[a + 1]; k += 1) {
      if (blocked?.[edgeIds[k]]) continue;
      const b = neighbours[k];
      if (!open[b]) continue;
      const ex = P[b * 3 + h0] - P[a * 3 + h0], ey = P[b * 3 + h1] - P[a * 3 + h1];
      const plan = Math.hypot(ex, ey);
      to.push(b);
      len.push(length[edgeIds[k]]);
      dx.push(plan > 1e-6 ? ex / plan : 0);
      dy.push(plan > 1e-6 ? ey / plan : 0);
    }
  }
  start0[n] = to.length;
  const adj = { start: start0, to: Int32Array.from(to), len: Float64Array.from(len), dx: Float32Array.from(dx), dy: Float32Array.from(dy) };

  const exitList = [...new Set(exits.filter((e) => e >= 0 && open[e]))];
  const isExit = new Uint8Array(n);
  for (const e of exitList) isExit[e] = 1;
  const dExit = walkFrom(exitList, n, adj, open);
  // The fire's repellent: metres from the nearest untenable place, over the open links from its open edge.
  let dFire = null;
  if (closed) {
    const edge = [];
    for (let a = 0; a < n; a += 1) {
      if (!open[a]) continue;
      for (let k = indptr[a]; k < indptr[a + 1]; k += 1) {
        if (!blocked?.[edgeIds[k]] && !open[neighbours[k]]) {
          edge.push(a);
          break;
        }
      }
    }
    dFire = walkFrom(edge, n, adj, open);
  }
  const base = new Float64Array(n);
  for (let v = 0; v < n; v += 1) {
    base[v] = open[v] && dExit[v] < Infinity ? -dExit[v] - (dFire && dFire[v] < Infinity ? p.repel * Math.exp(-dFire[v] / p.repelLength) : 0) : -Infinity;
  }

  // Where agents start: open floor that reaches an exit (not doors, stairs or
  // exits), or the given places that do.
  const rand = random(seed);
  const places = [];
  let unreached = 0;
  const from = opts.places ? opts.places : Array.from({ length: n }, (_, v) => v);
  for (const v of from) {
    if (!open[v] || isExit[v] || kinds[v] === KIND_DOOR || kinds[v] === KIND_STAIR) continue;
    if (dExit[v] < Infinity) places.push(v);
    else unreached += 1;
  }
  const count = Math.min(p.agents, places.length);
  const holder = new Int32Array(n).fill(-1);
  const at = new Int32Array(count), heading = new Float64Array(count);
  const placeFree = () => {
    // A random free place (a few tries; the population is a share of the places).
    for (let tries = 0; tries < 64; tries += 1) {
      const v = places[Math.floor(rand() * places.length)];
      if (holder[v] < 0) return v;
    }
    return -1;
  };
  for (let i = 0; i < count; i += 1) {
    const j = i + Math.floor(rand() * (places.length - i));
    [places[i], places[j]] = [places[j], places[i]];
    at[i] = places[i];
    holder[places[i]] = i;
    heading[i] = rand() * 2 * Math.PI;
  }

  const trail = new Float64Array(n), next = new Float64Array(n), sum = new Float64Array(n);
  const visits = new Uint32Array(n);
  const flux = new Map();
  const field = (v) => (base[v] === -Infinity ? -Infinity : base[v] + (p.trailWeight * trail[v]) / (trail[v] + p.trailHalf));
  // The link from v best aligned with direction (c, s) within the cone, or -1.
  const along = (v, c, s) => {
    let best = -1, bestCos = SENSE_CONE;
    for (let k = adj.start[v]; k < adj.start[v + 1]; k += 1) {
      const cos = adj.dx[k] * c + adj.dy[k] * s;
      if (cos > bestCos) {
        bestCos = cos;
        best = adj.to[k];
      }
    }
    return best;
  };
  const sense = (v, angle) => {
    const c = Math.cos(angle), s = Math.sin(angle);
    let u = v;
    for (let step = 0; step < p.so; step += 1) {
      u = along(u, c, s);
      if (u < 0) return -Infinity;
    }
    return field(u);
  };
  const sa = (p.sa * Math.PI) / 180, ra = (p.ra * Math.PI) / 180;
  const order = Array.from({ length: count }, (_, i) => i);
  const shuffle = () => {
    for (let i = order.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rand() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
  };
  const burnIn = Math.floor(p.steps / 2);

  for (let step = 0; step < p.steps; step += 1) {
    // The exits are food: each feeds trail to itself and its neighbours, less while an agent stands there.
    for (const e of exitList) {
      let covered = holder[e] >= 0;
      for (let k = adj.start[e]; k < adj.start[e + 1] && !covered; k += 1) covered = holder[adj.to[k]] >= 0;
      const fed = covered ? p.foodCovered : p.food;
      trail[e] += fed;
      for (let k = adj.start[e]; k < adj.start[e + 1]; k += 1) trail[adj.to[k]] += fed;
    }
    // Motor: a step forward if the node ahead is free; through an exit, out and back in at a free place.
    shuffle();
    for (const i of order) {
      if (at[i] < 0) continue;
      if (rand() < p.pCD) {
        heading[i] = rand() * 2 * Math.PI;
        continue;
      }
      const u = along(at[i], Math.cos(heading[i]), Math.sin(heading[i]));
      if (u < 0 || holder[u] >= 0) {
        heading[i] = rand() * 2 * Math.PI;
        continue;
      }
      holder[at[i]] = -1;
      trail[u] += p.depT;
      visits[u] += 1;
      if (isExit[u]) {
        flux.set(u, (flux.get(u) || 0) + 1);
        const v = placeFree();
        at[i] = v;
        if (v >= 0) holder[v] = i;
        heading[i] = rand() * 2 * Math.PI;
        continue;
      }
      at[i] = u;
      holder[u] = i;
    }
    // Sensory: turn towards the strongest of ahead, left and right.
    shuffle();
    for (const i of order) {
      const v = at[i];
      if (v < 0) {
        // Every place was taken when it left: try again.
        const w = placeFree();
        if (w >= 0) {
          at[i] = w;
          holder[w] = i;
        }
        continue;
      }
      const h = heading[i];
      const F = sense(v, h), L = sense(v, h - sa), R = sense(v, h + sa);
      if (F === -Infinity && L === -Infinity && R === -Infinity) heading[i] = h + ra;
      else if (F > L && F > R) continue;
      else if (F < L && F < R) heading[i] = h + (L < R ? ra : -ra);
      else if (L < R) heading[i] = h + ra;
      else if (R < L) heading[i] = h - ra;
    }
    // The trail spreads and fades: (1 - damp) times the mean over the node and
    // its eight lattice neighbours, a missing one (a wall, the fire) counting
    // as none, so walls absorb trail as in Jones's grid; a stair, a chain of
    // single nodes, takes the mean over the neighbours it has, or no trail
    // would last on a flight.
    for (let v = 0; v < n; v += 1) {
      if (!open[v]) continue;
      let total = trail[v];
      const deg = adj.start[v + 1] - adj.start[v];
      for (let k = adj.start[v]; k < adj.start[v + 1]; k += 1) total += trail[adj.to[k]];
      next[v] = ((1 - p.damp) * total) / (kinds[v] === KIND_STAIR ? 1 + deg : Math.max(9, 1 + deg));
    }
    trail.set(next);
    if (step >= burnIn) for (let v = 0; v < n; v += 1) sum[v] += trail[v];
  }

  // Weights over the 99th percentile of the trail, capped at 1: the food at
  // the exits would otherwise set the scale and leave the building's tubes faint.
  const laid = Array.from(sum).filter((s) => s > 0).sort((a, b) => a - b);
  const scale = laid.length ? laid[Math.min(laid.length - 1, Math.floor(0.99 * laid.length))] : 0;
  const weights = new Float32Array(n);
  if (scale > 0) for (let v = 0; v < n; v += 1) weights[v] = Math.min(1, sum[v] / scale);

  // The route along the tubes, against the shortest walk.
  let route = null;
  if (start >= 0 && open[start] && dExit[start] < Infinity) {
    const cost = new Float64Array(n).fill(Infinity);
    const via = new Int32Array(n).fill(-1);
    const heap = new Heap();
    cost[start] = 0;
    heap.push(0, start);
    const done = new Uint8Array(n);
    let reached = -1;
    while (heap.size) {
      const a = heap.pop();
      if (done[a]) continue;
      done[a] = 1;
      if (isExit[a]) {
        reached = a;
        break;
      }
      for (let k = adj.start[a]; k < adj.start[a + 1]; k += 1) {
        const b = adj.to[k];
        const w = (weights[a] + weights[b]) / 2;
        const c = cost[a] + adj.len[k] * (1 + p.tube * (1 - w));
        if (c < cost[b]) {
          cost[b] = c;
          via[b] = a;
          heap.push(c, b);
        }
      }
    }
    if (reached >= 0) {
      const nodes = [];
      for (let v = reached; v >= 0; v = via[v]) nodes.push(v);
      nodes.reverse();
      let walked = 0;
      for (let k = 1; k < nodes.length; k += 1) {
        const a = nodes[k - 1], b = nodes[k];
        for (let j = adj.start[a]; j < adj.start[a + 1]; j += 1) {
          if (adj.to[j] === b) {
            walked += adj.len[j];
            break;
          }
        }
      }
      route = { nodes, length: walked, shortest: dExit[start] };
    }
  }

  return {
    weights,
    visits,
    exits: [...flux.entries()].map(([node, f]) => ({ node, flux: f })).sort((a, b) => b.flux - a.flux),
    route,
    agents: count,
    steps: p.steps,
    closed,
    unreached,
  };
}
