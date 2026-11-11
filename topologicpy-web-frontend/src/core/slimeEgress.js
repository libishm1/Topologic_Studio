/**
 * A slime mould's way out: Jones's Physarum agents on the building's
 * walkable graph, drawn to the exits and kept from the fire, whose trail
 * weighs the ways out (wiki/research/physarum-jones-models-2026-11-11.md,
 * section 4; wiki/roadmap/egress-desire-lines-plan-2026-11-11.md, D2).
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
 * sensor stopped by a wall reads -Infinity (his walls read an empty trail),
 * and an agent turns RA when all three are stopped (his, when any sensor
 * leaves the grid). Stairs are links like any other, so the mould climbs
 * between storeys. The turn and the one-agent-a-node movement are proved in
 * formal/fireegress/FireEgress/Physarum.lean.
 *
 * What the agents sense. Jones's food is a diffusing source, flat far from
 * it (the plateau Jones notes); here the exits draw as the walk to them, the
 * static floor field of Burstedde et al. (2001,
 * doi:10.1016/S0378-4371(01)00141-8), so every place keeps a gradient. The
 * trail adds to it, as the dynamic field of Kirchner and Schadschneider's
 * evacuation model does (2002, doi:10.1016/S0378-4371(02)00857-9), in
 * metres of walk: a full tube is worth `trailWeight` metres.
 *
 * The fire, soft (the default). The run keeps the fire's clock, a step the
 * time to walk a link (`stepSeconds`, 0.5 m at 1.19 m/s). A place's smoke
 * begins to retard `lead` seconds before the fire makes it untenable
 * (`closes`) and is at full strength from then on: r rises from 0 to 1. The
 * walk to the exits charges a link its length times 1 + fireCost r, worked
 * out again every `replanSteps`, so the slime is drawn round the fire as it
 * grows but never walled off: where the only way out is through smoke it
 * still goes, at a cost. The place itself repels by `repel` r metres, and
 * its trail fades faster, by a further share `retard` r a step: the fire and
 * its smoke retard the mould. No one starts or starts again where r is 1.
 *
 * The fire, hard. A place the fire has made untenable by `time` is a wall:
 * no agent enters it, no trail spreads into it, and the walk to the exits
 * goes round it, as the maze's walls do in Jones and Saeed's chemotactic
 * paths (2007, doi:10.3233/mgs-2007-3108); near it a repellent of `repel`
 * metres falls off over `repelLength`. With a large fire nothing may be
 * left: the soft fire is for that.
 *
 * A flux, not a crowd. Agents start on the occupied floor; one that steps
 * through an exit starts again at a random free place (Jones's teleport), so
 * the trail measures the escape traffic each place carries. The trail
 * averaged over the second half of the run, over its 99th percentile and
 * capped at 1, is each node's weight. The route from `start` is the cheapest
 * walk to an exit where a link costs its length times 1 + tube (1 - its
 * weight), and with the soft fire times 1 + fireCost r at the run's end: it
 * keeps to the tubes, and is reported against the shortest walk.
 *
 * Seeded (mulberry32): one seed, one run. An illustration of where the
 * escape concentrates, not an evacuation model to quote (forecast.js and
 * crowd.js time people).
 */
import { Heap } from "./route.js";
import { KIND_DOOR, KIND_STAIR } from "./lattice.js";
import { openAdjacency } from "./skeleton.js";

/** The run's settings: Jones's agent (SO in links, SA and RA in degrees), the field's weights in metres of walk, and the fire's. */
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
  // The fire: "soft", a retardant on its clock, or "hard", walls from `time`.
  fire: "soft",
  stepSeconds: 0.42,
  lead: 30,
  fireCost: 20,
  retard: 0.5,
  replanSteps: 10,
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

/** The soft fire's strength at a place untenable from `close` s, at `t` s: 0, rising over `lead`, 1 from `close` on. */
export function smokeAt(close, t, lead) {
  if (!(close < Infinity)) return 0;
  if (t >= close) return 1;
  return t > close - lead ? (t - (close - lead)) / lead : 0;
}

/** Metres (or cost) from the sources over the links, each link's cost given (Dijkstra). */
function walkFrom(sources, adj, cost = null) {
  const { n, start, to, len } = adj;
  const dist = new Float64Array(n).fill(Infinity);
  const heap = new Heap();
  for (const s of sources) {
    if (s < 0 || dist[s] === 0) continue;
    dist[s] = 0;
    heap.push(0, s);
  }
  const done = new Uint8Array(n);
  while (heap.size) {
    const a = heap.pop();
    if (done[a]) continue;
    done[a] = 1;
    for (let k = start[a]; k < start[a + 1]; k += 1) {
      const b = to[k];
      const d = dist[a] + (cost ? cost(a, b, k) : len[k]);
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
 *   fire     "soft" (default): a retardant on the fire's clock from `from` s; "hard": walls from `time`
 *   from     s on the fire's clock the soft run starts at (0)
 *   time     s: with the hard fire, places untenable by then are walls (Infinity: every place it ever closes)
 *   places   nodes agents start from (default: all the open floor)
 *   blocked  links no one may cross (walls)
 *   seed, and SLIME_DEFAULTS' settings
 * @returns {{weights: Float32Array, visits: Uint32Array, dExit: Float64Array, exits: Array<{node: number, flux: number}>,
 *            route: null | {nodes: number[], length: number, shortest: number}, agents: number, steps: number,
 *            closed: number, unreached: number, fire: string|null, seconds: number, fireShare: number}}
 *   closed: places untenable by the run's end (soft) or walled (hard); fireShare: the share of the
 *   trail laid in places at least half retarded (soft).
 */
export function slimeEgress(graph, opts = {}) {
  const { exits = [], start = -1, closes = null, time = Infinity, from = 0, blocked = null, seed = 1 } = opts;
  const p = { ...SLIME_DEFAULTS, ...opts };
  const soft = closes !== null && p.fire !== "hard";
  const hard = closes !== null && p.fire === "hard";
  const n = graph.nodes.length / 3;
  const kinds = graph.kinds || new Uint8Array(n);
  const seconds = p.steps * p.stepSeconds;

  // Open nodes: with the hard fire, not untenable by `time`.
  const open = new Uint8Array(n).fill(1);
  let closed = 0;
  if (closes !== null) {
    for (let v = 0; v < n; v += 1) {
      const shut = closes[v] <= (hard ? time : from + seconds);
      if (shut) closed += 1;
      if (hard && shut) open[v] = 0;
    }
  }
  const adj = openAdjacency(graph, { blocked, open });

  const exitList = [...new Set(exits.filter((e) => e >= 0 && open[e]))];
  const isExit = new Uint8Array(n);
  for (const e of exitList) isExit[e] = 1;
  // The soft fire's strength now, and the walk to the exits it weighs.
  const smoke = new Float32Array(n);
  const smokeNow = (t) => {
    if (soft) for (let v = 0; v < n; v += 1) smoke[v] = smokeAt(closes[v], t, p.lead);
  };
  const fireWalk = () => walkFrom(exitList, adj, (a, b, k) => adj.len[k] * (1 + p.fireCost * Math.max(smoke[a], smoke[b])));
  smokeNow(from);
  let dExit = soft ? fireWalk() : walkFrom(exitList, adj);
  const plain = soft ? walkFrom(exitList, adj) : dExit;
  // With the hard fire, a repellent from the walk to the nearest untenable place.
  let dFire = null;
  if (hard && closed) {
    const edge = [];
    for (let a = 0; a < n; a += 1) {
      if (!open[a]) continue;
      for (let k = adj.start[a]; k < adj.start[a + 1]; k += 1) if (!open[adj.to[k]]) edge.push(a);
    }
    // The open nodes beside walled-off ones: links to closed nodes are not in the adjacency, so look at the graph's.
    const near = new Set();
    for (let k = 0; k < graph.edges.length; k += 2) {
      const a = graph.edges[k], b = graph.edges[k + 1];
      if (blocked?.[k / 2]) continue;
      if (open[a] && !open[b]) near.add(a);
      if (open[b] && !open[a]) near.add(b);
    }
    dFire = walkFrom([...near, ...edge], adj);
  }
  const base = new Float64Array(n);
  const setBase = () => {
    for (let v = 0; v < n; v += 1) {
      if (!open[v] || !(dExit[v] < Infinity)) {
        base[v] = -Infinity;
        continue;
      }
      let b = -dExit[v];
      if (soft) b -= p.repel * smoke[v];
      else if (dFire && dFire[v] < Infinity) b -= p.repel * Math.exp(-dFire[v] / p.repelLength);
      base[v] = b;
    }
  };
  setBase();

  // Where agents start: open floor that reaches an exit (not doors, stairs or
  // exits), or the given places that do; with the soft fire, not where it is full.
  const rand = random(seed);
  const places = [];
  let unreached = 0;
  const fromList = opts.places ? opts.places : Array.from({ length: n }, (_, v) => v);
  for (const v of fromList) {
    if (!open[v] || isExit[v] || kinds[v] === KIND_DOOR || kinds[v] === KIND_STAIR) continue;
    if (plain[v] < Infinity) places.push(v);
    else unreached += 1;
  }
  const count = Math.min(p.agents, places.length);
  const holder = new Int32Array(n).fill(-1);
  const at = new Int32Array(count), heading = new Float64Array(count);
  const placeFree = () => {
    // A random free place out of the fire (a few tries; the population is a share of the places).
    for (let tries = 0; tries < 64; tries += 1) {
      const v = places[Math.floor(rand() * places.length)];
      if (holder[v] < 0 && smoke[v] < 1) return v;
    }
    return -1;
  };
  let placed = 0;
  for (let i = 0; i < places.length && placed < count; i += 1) {
    const j = i + Math.floor(rand() * (places.length - i));
    [places[i], places[j]] = [places[j], places[i]];
    if (smoke[places[i]] >= 1) continue;
    at[placed] = places[i];
    holder[places[i]] = placed;
    heading[placed] = rand() * 2 * Math.PI;
    placed += 1;
  }
  for (let i = placed; i < count; i += 1) at[i] = -1;

  const trail = new Float64Array(n), next = new Float64Array(n), sum = new Float64Array(n);
  const visits = new Uint32Array(n);
  const flux = new Map();
  let laidAll = 0, laidInFire = 0;
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
    // The soft fire on its clock: its strength, and the walk round it, now and then.
    if (soft && step > 0 && step % p.replanSteps === 0) {
      smokeNow(from + step * p.stepSeconds);
      dExit = fireWalk();
      setBase();
    }
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
      if (step >= burnIn) {
        laidAll += p.depT;
        if (smoke[u] >= 0.5) laidInFire += p.depT;
      }
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
    // its eight lattice neighbours, a missing neighbour (a wall, the hard
    // fire) counting as none, so walls absorb trail as in Jones's grid; a
    // stair, a chain of single nodes, takes the mean over the neighbours it
    // has, or no trail would last on a flight. The soft fire's smoke fades it
    // faster, by a further share retard x its strength.
    for (let v = 0; v < n; v += 1) {
      if (!open[v]) continue;
      let total = trail[v];
      const deg = adj.start[v + 1] - adj.start[v];
      for (let k = adj.start[v]; k < adj.start[v + 1]; k += 1) total += trail[adj.to[k]];
      const keep = (1 - p.damp) * (1 - p.retard * smoke[v]);
      next[v] = (keep * total) / (kinds[v] === KIND_STAIR ? 1 + deg : Math.max(9, 1 + deg));
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

  // The route along the tubes (and round the soft fire as it ends), against the shortest walk.
  let route = null;
  if (start >= 0 && open[start] && plain[start] < Infinity) {
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
        const c = cost[a] + adj.len[k] * (1 + p.tube * (1 - w)) * (1 + p.fireCost * Math.max(smoke[a], smoke[b]));
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
      route = { nodes, length: walked, shortest: plain[start] };
    }
  }

  return {
    weights,
    visits,
    dExit,
    exits: [...flux.entries()].map(([node, f]) => ({ node, flux: f })).sort((a, b) => b.flux - a.flux),
    route,
    agents: count,
    steps: p.steps,
    closed,
    unreached,
    fire: soft ? "soft" : hard ? "hard" : null,
    seconds,
    fireShare: laidAll ? laidInFire / laidAll : 0,
    // The soft fire's strength at the run's end, for the desire lines (skeleton.js).
    smoke: soft ? smoke : null,
  };
}
