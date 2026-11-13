/**
 * The egress map on the walkable graph
 * (wiki/roadmap/evacuation-map-and-accessibility-audit-plan-2026-11-11.md):
 * the walk to the nearest exit from everywhere, as the code check's
 * (codeCheck.js exitEntrances: exit doors on the storeys that have them,
 * stair entrances elsewhere), read room by room and exit by exit.
 *
 * - The walk from each room's worst place, and its route (one multi-source
 *   Dijkstra from all exits: Dijkstra 1959; Ahuja, Magnanti and Orlin 1993).
 * - Each exit's catchment: the floor it is nearest for.
 * - Contours of the walk every few metres.
 * - The blocked-exit sweep: each exit closed in turn and the floor solved
 *   again; then the worst-first cascade, the exit whose loss lengthens the
 *   farthest walk most closed next, round after round. Removing the most
 *   damaging element first is the targeted attack of network robustness
 *   (Albert, Jeong and Barabasi 2000); as a building manager's test of a
 *   floor's exits, "redundant alone, critical together", it is Poux's (2026).
 * - The common path (the walk before a second, separate way out opens: the
 *   Gromov product of the walks to the two nearest exits), with dead-end
 *   corridors told from rooms by their IfcSpace names, against IBC 2021
 *   1020.5 for corridors and Table 1006.2.1 for rooms.
 * - Doors no way out can do without: the bridges (Tarjan 1974) of the graph of
 *   rooms joined by doors, the exits joined to the outside.
 * - Values within NEAR_LIMIT of a limit listed for measuring on site (after
 *   Poux 2026: a tape check, not a pass).
 */
import { exitEntrances } from "./codeCheck.js";
import { edgeLengths } from "./route.js";

const KIND_STAIR = 1;
const KIND_DOOR = 2;
/** Exit doors closer than this (m) on one storey are one exit: double doors. */
const MERGE_M = 2;

/** A value this close below its limit (a share of it) is listed for measuring on site. */
export const NEAR_LIMIT = 0.05;
/** IBC 2021 1020.5, Group B: dead-end corridors 20 ft, 50 ft sprinklered (from summaries; to check against the text). */
export const DEAD_END_LIMIT = { plain: 6.096, sprinklered: 15.24 };
/** IBC 2021 1020.5's exception: no limit where the dead end is shorter than this many times its least width. */
export const DEAD_END_WIDTHS = 2.5;
/** IBC 2021 Table 1006.2.1, Group B common path: 100 ft sprinklered, or with 30 occupants or fewer; else 75 ft (secondary sources; to check). */
export const COMMON_PATH_LIMIT = { sprinklered: 30.48, few: 30.48, many: 22.86, fewOccupants: 30 };
/** One-way floor: a common path longer than this (m); shorter is a lattice's corner, not a dead end. */
export const ONE_WAY_M = 1;
/** A dead-end corridor shorter than this (m) is not listed (Poux 2026 lists those over 2 m). */
export const MIN_DEAD_END_M = 2;
/** Spaces whose names say they are circulation: their one-way floor is a dead-end corridor, not a room's common path. */
export const CIRCULATION_NAME = /\b(corridor|corr|hallway|circulation|passage|passageway|lobby|foyer|vestibule|gallery|concourse|walkway|landing|egress|flur|korridor|gang|couloir|pasillo)\b/i;

/** A binary min-heap of (key, value) pairs in typed arrays, growing as needed. */
class MinHeap {
  constructor(capacity = 1024) {
    this.keys = new Float64Array(capacity);
    this.vals = new Int32Array(capacity);
    this.size = 0;
  }
  push(key, val) {
    if (this.size === this.keys.length) {
      const keys = new Float64Array(this.size * 2);
      keys.set(this.keys);
      const vals = new Int32Array(this.size * 2);
      vals.set(this.vals);
      this.keys = keys;
      this.vals = vals;
    }
    let i = this.size++;
    const K = this.keys, V = this.vals;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (K[p] <= key) break;
      K[i] = K[p];
      V[i] = V[p];
      i = p;
    }
    K[i] = key;
    V[i] = val;
  }
  /** The smallest key's value; its key is left in `top`. */
  pop() {
    const K = this.keys, V = this.vals;
    const val = V[0];
    this.top = K[0];
    const n = --this.size;
    if (n > 0) {
      const key = K[n], v = V[n];
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        if (l >= n) break;
        const r = l + 1;
        const m = r < n && K[r] < K[l] ? r : l;
        if (K[m] >= key) break;
        K[i] = K[m];
        V[i] = V[m];
        i = m;
      }
      K[i] = key;
      V[i] = v;
    }
    return val;
  }
}
export { MinHeap };

/** The node's plan coordinates (the axes other than up). */
const planOf = (graph) => {
  const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
  const [h0, h1] = [0, 1, 2].filter((c) => c !== up);
  return { up, h0, h1 };
};

/**
 * Each node's room: the smallest room (roomsFromSpaces, sprinklers.js) whose
 * footprint holds the node in plan with its floor within 0.6 m of the node's;
 * -1 for none.
 */
export function roomsOfNodes(graph, rooms) {
  const n = graph.nodes.length / 3;
  const out = new Int32Array(n).fill(-1);
  if (!rooms?.length) return out;
  const { up, h0, h1 } = planOf(graph);
  const lift = graph.agentHeight || 0;
  const P = graph.nodes;
  const CELL = 2;
  const grid = new Map();
  const boxes = rooms.map((r, i) => {
    if (!r?.footprint) return null;
    const T = r.footprint;
    let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
    for (let k = 0; k < T.length; k += 2) {
      a0 = Math.min(a0, T[k]);
      a1 = Math.max(a1, T[k]);
      b0 = Math.min(b0, T[k + 1]);
      b1 = Math.max(b1, T[k + 1]);
    }
    for (let a = Math.floor(a0 / CELL); a <= Math.floor(a1 / CELL); a += 1) {
      for (let b = Math.floor(b0 / CELL); b <= Math.floor(b1 / CELL); b += 1) {
        const key = `${a},${b}`;
        if (!grid.has(key)) grid.set(key, []);
        grid.get(key).push(i);
      }
    }
    return { a0, a1, b0, b1 };
  });
  const inside = (T, x, y) => {
    for (let k = 0; k < T.length; k += 6) {
      const d1 = (x - T[k + 2]) * (T[k + 1] - T[k + 3]) - (T[k] - T[k + 2]) * (y - T[k + 3]);
      const d2 = (x - T[k + 4]) * (T[k + 3] - T[k + 5]) - (T[k + 2] - T[k + 4]) * (y - T[k + 5]);
      const d3 = (x - T[k]) * (T[k + 5] - T[k + 1]) - (T[k + 4] - T[k]) * (y - T[k + 1]);
      const neg = d1 < -1e-9 || d2 < -1e-9 || d3 < -1e-9;
      const pos = d1 > 1e-9 || d2 > 1e-9 || d3 > 1e-9;
      if (!(neg && pos)) return true;
    }
    return false;
  };
  for (let v = 0; v < n; v += 1) {
    const x = P[v * 3 + h0], y = P[v * 3 + h1], h = P[v * 3 + up] - lift;
    const list = grid.get(`${Math.floor(x / CELL)},${Math.floor(y / CELL)}`);
    if (!list) continue;
    let best = -1;
    for (const i of list) {
      const r = rooms[i], b = boxes[i];
      if (x < b.a0 || x > b.a1 || y < b.b0 || y > b.b1 || Math.abs(h - r.floor) > 0.6) continue;
      if ((best < 0 || r.area < rooms[best].area) && inside(r.footprint, x, y)) best = i;
    }
    out[v] = best;
  }
  return out;
}

/** A name that says what a room is (a word of three letters or more), as against a code such as "A101". */
const SAYS_WHAT = /[a-z]{3,}/i;

/**
 * Whether a room is circulation: by its name; or, where its name says
 * nothing, long and narrow (at least 3 to 1, at most 3 m wide), or by Poux's
 * (2026) rule, the storey's largest space, one an exit opens from, or one
 * with four doors or more (`facts`: {doors, exit, largest}).
 */
export function isCirculation(room, facts = null) {
  if (!room) return false;
  if (room.name && CIRCULATION_NAME.test(room.name)) return true;
  const half = room.rect?.half;
  if (half && half[1] > 0 && half[0] / half[1] >= 3 && 2 * half[1] <= 3) return true;
  if (room.name && SAYS_WHAT.test(room.name.replace(/\d+/g, " "))) return false;
  return Boolean(facts && (facts.largest || facts.exit || facts.doors >= 4));
}

/**
 * The exits, grouped: exit doors within MERGE_M on one storey are one exit,
 * each stair one. Also the base every solve shares.
 */
export function prepareEgress(graph, { exits = [], levels = [], useWalls = true, rooms = null, doorNames = null } = {}) {
  const n = graph.nodes.length / 3;
  const kinds = graph.kinds || new Uint8Array(n);
  const base = exitEntrances(graph, { exits, levels, useWalls });
  const { up, h0, h1 } = planOf(graph);
  const P = graph.nodes;
  const lengths = edgeLengths(graph);
  const groups = [];
  const stairGroup = new Map();
  const groupOf = base.floors.map(() => new Map()); // per storey: node -> group
  const stairOf = (piece) => {
    if (!stairGroup.has(piece)) {
      stairGroup.set(piece, groups.length);
      groups.push({ kind: "stair", piece, nodes: [], storeys: new Set(), label: `Stair ${stairGroup.size}` });
    }
    return stairGroup.get(piece);
  };
  for (let v = 0; v < n; v += 1) if (kinds[v] === KIND_STAIR && base.stairPiece[v] >= 0) stairOf(base.stairPiece[v]);
  let doors = 0;
  base.entrances.forEach((map, s) => {
    for (const [v, id] of map) {
      let g;
      if (id.startsWith("stair ")) g = stairOf(Number(id.slice(6)));
      else {
        const x = P[v * 3 + h0], y = P[v * 3 + h1];
        g = groups.findIndex((q) => q.kind === "door" && q.storeys.has(s) && Math.hypot(q.x - x, q.y - y) < MERGE_M);
        if (g < 0) {
          doors += 1;
          g = groups.length;
          groups.push({ kind: "door", nodes: [], storeys: new Set([s]), x, y, label: null });
        }
      }
      groups[g].nodes.push(v);
      groups[g].storeys.add(s);
      groupOf[s].set(v, g);
    }
  });
  // Door exits by name where the model gives one, else numbered.
  let k = 0;
  for (const g of groups) {
    if (g.kind !== "door") continue;
    k += 1;
    const named = g.nodes.map((v) => doorNames?.get?.(v)).find(Boolean);
    g.label = named ? `Exit ${k} (${named})` : `Exit ${k}`;
    const at = g.nodes[0];
    g.position = [P[at * 3], P[at * 3 + 1], P[at * 3 + 2]];
  }
  for (const g of groups) if (g.kind === "stair" && !g.position && g.nodes.length) g.position = [P[g.nodes[0] * 3], P[g.nodes[0] * 3 + 1], P[g.nodes[0] * 3 + 2]];
  const roomOf = rooms ? roomsOfNodes(graph, rooms) : new Int32Array(n).fill(-1);
  // Each room's doors, whether an exit opens from it, and each storey's largest room.
  const R = (rooms || []).length;
  const doorCount = new Int32Array(R), exitRoom = new Uint8Array(R), storeyOfRoom = new Int32Array(R).fill(-1);
  for (let v = 0; v < n; v += 1) {
    if (roomOf[v] >= 0 && base.storeyOf[v] >= 0) storeyOfRoom[roomOf[v]] = base.storeyOf[v];
    if (kinds[v] !== KIND_DOOR) continue;
    const sides = new Set();
    for (let k = base.indptr[v]; k < base.indptr[v + 1]; k += 1) if (roomOf[base.neighbours[k]] >= 0) sides.add(roomOf[base.neighbours[k]]);
    for (const r of sides) doorCount[r] += 1;
  }
  for (const g of groups) if (g.kind === "door") for (const v of g.nodes) {
    if (roomOf[v] >= 0) exitRoom[roomOf[v]] = 1;
    for (let k = base.indptr[v]; k < base.indptr[v + 1]; k += 1) if (roomOf[base.neighbours[k]] >= 0) exitRoom[roomOf[base.neighbours[k]]] = 1;
  }
  const largest = new Map();
  for (let r = 0; r < R; r += 1) {
    const s = storeyOfRoom[r];
    if (s >= 0 && (!largest.has(s) || rooms[r].area > rooms[largest.get(s)].area)) largest.set(s, r);
  }
  const circulation = Uint8Array.from((rooms || []).map((room, r) => (isCirculation(room, { doors: doorCount[r], exit: exitRoom[r], largest: largest.get(storeyOfRoom[r]) === r }) ? 1 : 0)));
  return { ...base, graph, n, kinds, lengths, groups, groupOf, stairGroup, roomOf, rooms: rooms || [], circulation, doorCount, doors, up, h0, h1 };
}

/**
 * The walk to the nearest open exit from every node, on its own storey, with
 * the exit it reaches and the node before it on that walk. `closed`: exit
 * groups out of use. A part of a storey with exit doors that reaches none of
 * them leaves by a stair (an exit stair is an exit, IBC 202), as the code check.
 */
export function exitField(base, closed = null) {
  const { n, storeyOf, entrances, discharge, indptr, neighbours, edgeIds, blocked, kinds, lengths, groupOf, stairGroup, stairPiece } = base;
  const dist = new Float64Array(n).fill(Infinity);
  const exit = new Int32Array(n).fill(-1);
  const parent = new Int32Array(n).fill(-1);
  const heap = new MinHeap(1024);
  const seed = (v, g) => {
    if (closed?.has(g) || dist[v] === 0) return;
    dist[v] = 0;
    exit[v] = g;
    heap.push(0, v);
  };
  const spread = () => {
    while (heap.size) {
      const v = heap.pop();
      const d = heap.top;
      if (d > dist[v]) continue;
      for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
        const w = neighbours[k];
        if (storeyOf[w] !== storeyOf[v] || (blocked && blocked[edgeIds[k]])) continue;
        const nd = d + lengths[edgeIds[k]];
        if (nd < dist[w]) {
          dist[w] = nd;
          exit[w] = exit[v];
          parent[w] = v;
          heap.push(nd, w);
        }
      }
    }
  };
  entrances.forEach((map, s) => {
    for (const v of map.keys()) seed(v, groupOf[s].get(v));
  });
  spread();
  for (const s of discharge) {
    for (let v = 0; v < n; v += 1) {
      if (storeyOf[v] !== s || Number.isFinite(dist[v])) continue;
      for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
        const w = neighbours[k];
        if (kinds[w] === KIND_STAIR && !(blocked && blocked[edgeIds[k]])) seed(v, stairGroup.get(stairPiece[w]));
      }
    }
  }
  spread();
  return { dist, exit, parent };
}

/** The floor nodes the readings count: on a storey, not a door. */
const floorNodes = (base) => {
  const out = [];
  for (let v = 0; v < base.n; v += 1) if (base.storeyOf[v] >= 0 && base.kinds[v] !== KIND_DOOR) out.push(v);
  return out;
};

/** Farthest and mean walk, the share over the limit and the floor reaching no exit. */
function summary(base, field, nodes, limit, cell) {
  let far = 0, sum = 0, count = 0, over = 0, none = 0, at = -1;
  for (const v of nodes) {
    const d = field.dist[v];
    if (!Number.isFinite(d)) {
      none += 1;
      continue;
    }
    if (d > far) {
      far = d;
      at = v;
    }
    sum += d;
    count += 1;
    if (d > limit) over += 1;
  }
  return { farthest: far, farthestNode: at, mean: count ? sum / count : 0, overShare: count ? over / count : 0, unreachedArea: none * cell };
}

/** Each room's worst place, its walk, exit and route (node list, from the place to the exit). */
function roomWorst(base, field) {
  const { roomOf, rooms } = base;
  const worst = new Map();
  for (let v = 0; v < base.n; v += 1) {
    const r = roomOf[v];
    if (r < 0 || base.kinds[v] === KIND_DOOR) continue;
    const d = field.dist[v];
    const w = worst.get(r);
    if (!w) worst.set(r, { node: v, dist: d, nodes: 1 });
    else {
      w.nodes += 1;
      if (d > w.dist || (!Number.isFinite(w.dist) && Number.isFinite(d))) {
        w.node = v;
        w.dist = d;
      }
    }
  }
  return [...worst].map(([r, w]) => ({ room: r, name: rooms[r]?.name || null, node: w.node, dist: w.dist, exit: field.exit[w.node], nodes: w.nodes }));
}

/** The route from a node down its field to the exit: node indices. */
export function routeOf(field, v) {
  const out = [];
  for (let guard = 0; v >= 0 && guard < 1e6; guard += 1) {
    out.push(v);
    v = field.parent[v];
  }
  return out;
}

/**
 * Contour dashes where the walk crosses each multiple of `step` (m): a short
 * segment across each link that crosses, in the viewer frame.
 */
export function contours(base, dist, { step = 5 } = {}) {
  const { graph, h0, h1 } = base;
  const P = graph.nodes;
  const E = graph.edges;
  const half = (graph.spacing || 0.5) * 0.45;
  const out = [];
  for (let e = 0; e < E.length / 2; e += 1) {
    const a = E[e * 2], b = E[e * 2 + 1];
    const da = dist[a], db = dist[b];
    if (!Number.isFinite(da) || !Number.isFinite(db) || Math.floor(da / step) === Math.floor(db / step)) continue;
    if (base.storeyOf[a] < 0 || base.storeyOf[a] !== base.storeyOf[b]) continue;
    const level = Math.max(Math.floor(da / step), Math.floor(db / step)) * step;
    const t = (level - da) / (db - da);
    const p = [0, 1, 2].map((c) => P[a * 3 + c] + t * (P[b * 3 + c] - P[a * 3 + c]));
    const ux = P[b * 3 + h0] - P[a * 3 + h0], uy = P[b * 3 + h1] - P[a * 3 + h1];
    const len = Math.hypot(ux, uy) || 1;
    const q0 = p.slice(), q1 = p.slice();
    q0[h0] -= (-uy / len) * half;
    q0[h1] -= (ux / len) * half;
    q1[h0] += (-uy / len) * half;
    q1[h1] += (ux / len) * half;
    out.push(...q0, ...q1);
  }
  return Float32Array.from(out);
}

/**
 * Each exit closed in turn, then the worst-first cascade: the exit whose loss
 * lengthens the farthest walk most (floor reaching no exit first) closed
 * next, for at most `rounds` rounds or until one exit is left.
 */
export function exitSweep(base, { limit = Infinity, rounds = 6, onProgress = null } = {}) {
  const nodes = floorNodes(base);
  const cell = (base.graph.spacing || 0.5) ** 2;
  const all = exitField(base);
  const baseRooms = roomWorst(base, all);
  const open = base.groups.map((_, g) => g).filter((g) => base.groups[g].nodes.length);
  const solves = { done: 0, total: open.length + Math.max(0, Math.min(rounds, open.length - 1)) * open.length };
  const score = (s) => [s.unreachedArea, s.farthest, s.mean];
  const worse = (a, b) => {
    const x = score(a), y = score(b);
    for (let i = 0; i < 3; i += 1) if (Math.abs(x[i] - y[i]) > 1e-9) return x[i] > y[i];
    return false;
  };
  const solve = (closed) => {
    const field = exitField(base, closed);
    solves.done += 1;
    onProgress?.(solves.done, solves.total);
    return { field, ...summary(base, field, nodes, limit, cell) };
  };
  const rerouted = (field) => baseRooms.filter((r) => field.exit[r.node] !== r.exit).length;
  const single = open.map((g) => {
    const s = solve(new Set([g]));
    return { exit: g, label: base.groups[g].label, farthest: s.farthest, mean: s.mean, overShare: s.overShare, unreachedArea: s.unreachedArea, roomsRerouted: rerouted(s.field) };
  });
  const cascade = [];
  const closed = new Set();
  for (let r = 0; r < rounds && closed.size < open.length - 1; r += 1) {
    let pick = null;
    for (const g of open) {
      if (closed.has(g)) continue;
      const s = solve(new Set([...closed, g]));
      if (!pick || worse(s, pick)) pick = { ...s, exit: g };
    }
    if (!pick) break;
    closed.add(pick.exit);
    cascade.push({ exit: pick.exit, label: base.groups[pick.exit].label, closed: [...closed], farthest: pick.farthest, mean: pick.mean, overShare: pick.overShare, unreachedArea: pick.unreachedArea });
    if (pick.unreachedArea > 0 && pick.farthest === 0) break;
  }
  return { single, cascade, rounds: Math.min(rounds, Math.max(0, open.length - 1)), exits: open.length };
}

/**
 * The walk before a second, separate way out opens, at every node: with dA
 * and dB the walks to the two nearest exits and dAB the walk between them,
 * (dA + dB - dAB) / 2 (the Gromov product). A node that reaches one exit only
 * is all common path. One field per exit; null past `maxExits`.
 */
export function commonPaths(base, { maxExits = 40 } = {}) {
  const live = base.groups.map((_, g) => g).filter((g) => base.groups[g].nodes.length);
  if (!live.length || live.length > maxExits) return null;
  const fields = live.map((g) => exitField(base, new Set(live.filter((h) => h !== g))).dist);
  const n = base.n;
  const out = new Float64Array(n).fill(Infinity);
  for (let v = 0; v < n; v += 1) {
    let a = -1, b = -1;
    for (let i = 0; i < live.length; i += 1) {
      const d = fields[i][v];
      if (!Number.isFinite(d)) continue;
      if (a < 0 || d < fields[a][v]) {
        b = a;
        a = i;
      } else if (b < 0 || d < fields[b][v]) b = i;
    }
    if (a < 0) continue;
    if (b < 0) {
      out[v] = fields[a][v];
      continue;
    }
    let between = Infinity;
    for (const u of base.groups[live[b]].nodes) between = Math.min(between, fields[a][u]);
    out[v] = Number.isFinite(between) ? Math.max(0, (fields[a][v] + fields[b][v] - between) / 2) : fields[a][v];
  }
  return out;
}

/**
 * Dead-end corridors and rooms' common paths. One-way floor (a common path
 * over ONE_WAY_M) in circulation spaces, as connected pieces: each a dead-end
 * corridor, its length the longest common path in it (the walk back to where
 * two ways open), its tip there, its mouth where it is shortest, and its
 * least width (`widthAt`, m a node). Rooms take their longest common path,
 * against Table 1006.2.1 by their occupant load.
 */
export function deadEnds(base, common, { sprinklered = false, widthAt = null, occupantArea = 9 } = {}) {
  const { n, roomOf, rooms, indptr, neighbours, edgeIds, blocked, storeyOf, kinds, graph, lengths } = base;
  const P = graph.nodes;
  const circulation = base.circulation || Uint8Array.from(rooms.map((r) => (isCirculation(r) ? 1 : 0)));
  const limit = sprinklered ? DEAD_END_LIMIT.sprinklered : DEAD_END_LIMIT.plain;
  // One-way floor by the Gromov product; its depth by a search back to the
  // two-way floor, since a difference of distances reads short round corners
  // (Poux 2026: a 5.0 m stub read 4.6 m). The search starts from each two-way
  // node at its own common path, so it carries on from where two ways open,
  // not from the edge of the threshold.
  const oneWay = new Uint8Array(n);
  const depth = new Float64Array(n).fill(Infinity);
  const heap = new MinHeap(1024);
  for (let v = 0; v < n; v += 1) {
    if (!Number.isFinite(common[v])) continue;
    if (common[v] > ONE_WAY_M) oneWay[v] = 1;
    else {
      depth[v] = common[v];
      heap.push(common[v], v);
    }
  }
  while (heap.size) {
    const v = heap.pop();
    const d = heap.top;
    if (d > depth[v]) continue;
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      const w = neighbours[k];
      if (!oneWay[w] || storeyOf[w] !== storeyOf[v] || (blocked && blocked[edgeIds[k]])) continue;
      const nd = d + lengths[edgeIds[k]];
      if (nd < depth[w]) {
        depth[w] = nd;
        heap.push(nd, w);
      }
    }
  }
  // A one-way piece that never meets two-way floor (a storey with one exit) keeps its Gromov value: all of it is common path.
  for (let v = 0; v < n; v += 1) if (oneWay[v] && !Number.isFinite(depth[v])) depth[v] = common[v];
  const seen = new Uint8Array(n);
  const corridors = [];
  for (let v = 0; v < n; v += 1) {
    if (seen[v] || !oneWay[v] || roomOf[v] < 0 || !circulation[roomOf[v]]) continue;
    const piece = [];
    const stack = [v];
    seen[v] = 1;
    while (stack.length) {
      const u = stack.pop();
      piece.push(u);
      for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
        const w = neighbours[k];
        if (seen[w] || storeyOf[w] !== storeyOf[u] || (blocked && blocked[edgeIds[k]])) continue;
        if (!oneWay[w] || roomOf[w] < 0 || !circulation[roomOf[w]]) continue;
        seen[w] = 1;
        stack.push(w);
      }
    }
    let tip = piece[0], mouth = piece[0];
    for (const u of piece) {
      if (depth[u] > depth[tip]) tip = u;
      if (depth[u] < depth[mouth]) mouth = u;
    }
    const length = depth[tip];
    if (length < MIN_DEAD_END_M) continue;
    let width = Infinity;
    if (widthAt) for (const u of piece) if (kinds[u] !== KIND_DOOR && widthAt[u] > 0) width = Math.min(width, widthAt[u]);
    const exempt = Number.isFinite(width) && length < DEAD_END_WIDTHS * width;
    corridors.push({
      length,
      width: Number.isFinite(width) ? width : null,
      tip: [P[tip * 3], P[tip * 3 + 1], P[tip * 3 + 2]],
      mouth: [P[mouth * 3], P[mouth * 3 + 1], P[mouth * 3 + 2]],
      room: roomOf[tip],
      name: rooms[roomOf[tip]]?.name || null,
      nodes: piece,
      over: !exempt && length > limit,
      near: !exempt && length <= limit && length >= limit * (1 - NEAR_LIMIT),
      exempt,
    });
  }
  corridors.sort((a, b) => b.length - a.length);
  // Rooms: their longest common path, against Table 1006.2.1 by occupant load.
  const roomPaths = new Map();
  for (let v = 0; v < n; v += 1) {
    const r = roomOf[v];
    if (r < 0 || circulation[r] || !Number.isFinite(depth[v])) continue;
    const at = roomPaths.get(r);
    if (at === undefined || depth[v] > depth[at]) roomPaths.set(r, v);
  }
  const roomList = [...roomPaths].map(([r, v]) => {
    const load = Math.ceil((rooms[r]?.area || 0) / occupantArea);
    const lim = sprinklered ? COMMON_PATH_LIMIT.sprinklered : load <= COMMON_PATH_LIMIT.fewOccupants ? COMMON_PATH_LIMIT.few : COMMON_PATH_LIMIT.many;
    const length = depth[v];
    return { room: r, name: rooms[r]?.name || null, length, limit: lim, occupants: load, at: [P[v * 3], P[v * 3 + 1], P[v * 3 + 2]], over: length > lim, near: length <= lim && length >= lim * (1 - NEAR_LIMIT) };
  });
  roomList.sort((a, b) => b.length - a.length);
  let ways = 0, floor = 0;
  for (let v = 0; v < n; v += 1) {
    if (storeyOf[v] < 0 || kinds[v] === KIND_DOOR || !Number.isFinite(common[v])) continue;
    floor += 1;
    if (oneWay[v]) ways += 1;
  }
  return { corridors, rooms: roomList, limit, oneWayShare: floor ? ways / floor : 0, depth };
}

/**
 * Doors no way out can do without: the bridges (Tarjan 1974) of the graph whose
 * vertices are the rooms, the stairs and the outside, and whose edges are the
 * doors between rooms, the exit doors to the outside and the stair entrances
 * (each stair an exit, joined to the outside). A door that is a bridge is the
 * only way out of the rooms behind it.
 */
export function criticalDoors(base) {
  const { n, roomOf, rooms, indptr, neighbours, kinds, graph, groups, groupOf, stairGroup } = base;
  const R = rooms.length;
  const OUT = R;
  const stairVertex = new Map();
  let V = R + 1;
  for (const piece of stairGroup.keys()) stairVertex.set(piece, V++);
  const edges = []; // [a, b, door node or -1]
  const exitNodes = new Set();
  for (const g of groups) if (g.kind === "door") for (const v of g.nodes) exitNodes.add(v);
  for (let d = 0; d < n; d += 1) {
    if (kinds[d] !== KIND_DOOR) continue;
    const sides = new Set();
    for (let k = indptr[d]; k < indptr[d + 1]; k += 1) {
      const r = roomOf[neighbours[k]];
      if (r >= 0) sides.add(r);
    }
    const list = [...sides];
    if (list.length >= 2) for (let i = 1; i < list.length; i += 1) edges.push([list[0], list[i], d]);
    else if (list.length === 1 && exitNodes.has(d)) edges.push([list[0], OUT, d]);
  }
  // Exit doors that are not door nodes (the exit is a floor node at the door).
  for (const v of exitNodes) if (kinds[v] !== KIND_DOOR && roomOf[v] >= 0) edges.push([roomOf[v], OUT, v]);
  // Stair entrances: the room of the entrance to its stair; each stair to the outside.
  const seenStair = new Set();
  groupOf.forEach((map, s) => {
    for (const [v, g] of map) {
      const piece = groups[g].piece;
      if (piece === undefined || roomOf[v] < 0) continue;
      const key = `${roomOf[v]}:${piece}:${s}`;
      if (seenStair.has(key)) continue;
      seenStair.add(key);
      edges.push([roomOf[v], stairVertex.get(piece), -1]);
    }
  });
  for (const vtx of stairVertex.values()) edges.push([vtx, OUT, -1]);
  // Tarjan's bridges, iteratively, with edge ids so that parallel edges are not bridges.
  const adj = Array.from({ length: V }, () => []);
  edges.forEach(([a, b], e) => {
    adj[a].push([b, e]);
    adj[b].push([a, e]);
  });
  const disc = new Int32Array(V).fill(-1), low = new Int32Array(V), parentEdge = new Int32Array(V).fill(-1);
  const bridges = [];
  let time = 0;
  for (let root = 0; root < V; root += 1) {
    if (disc[root] >= 0) continue;
    const stack = [[root, 0]];
    disc[root] = low[root] = time++;
    while (stack.length) {
      const top = stack[stack.length - 1];
      const [u, i] = top;
      if (i < adj[u].length) {
        top[1] += 1;
        const [w, e] = adj[u][i];
        if (e === parentEdge[u]) continue;
        if (disc[w] < 0) {
          parentEdge[w] = e;
          disc[w] = low[w] = time++;
          stack.push([w, 0]);
        } else low[u] = Math.min(low[u], disc[w]);
      } else {
        stack.pop();
        if (stack.length) {
          const p = stack[stack.length - 1][0];
          low[p] = Math.min(low[p], low[u]);
          if (low[u] > disc[p]) bridges.push(parentEdge[u]);
        }
      }
    }
  }
  // For each bridge door, the rooms on the side away from the outside.
  const P = graph.nodes;
  const out = [];
  for (const e of bridges) {
    const [a, b, door] = edges[e];
    if (door < 0) continue;
    const reach = new Uint8Array(V);
    const stack = [OUT];
    reach[OUT] = 1;
    while (stack.length) {
      const u = stack.pop();
      for (const [w, f] of adj[u]) {
        if (f === e || reach[w]) continue;
        reach[w] = 1;
        stack.push(w);
      }
    }
    const behind = [];
    for (let r = 0; r < R; r += 1) if (!reach[r]) behind.push(r);
    out.push({ door, between: [a, b], position: [P[door * 3], P[door * 3 + 1], P[door * 3 + 2]], rooms: behind });
  }
  // Rooms with no way out at all (no path to the outside with every door open).
  const reachAll = new Uint8Array(V);
  const st = [OUT];
  reachAll[OUT] = 1;
  while (st.length) {
    const u = st.pop();
    for (const [w] of adj[u]) if (!reachAll[w]) {
      reachAll[w] = 1;
      st.push(w);
    }
  }
  for (const c of out) c.rooms = c.rooms.filter((r) => reachAll[r]);
  const area = (list) => list.reduce((s, r) => s + (rooms[r]?.area || 0), 0);
  const doors = out
    .filter((c) => c.rooms.length)
    .map((c) => ({ ...c, area: area(c.rooms), names: c.rooms.map((r) => rooms[r]?.name).filter(Boolean) }))
    .sort((x, y) => y.rooms.length - x.rooms.length || y.area - x.area);
  return { doors, suites: doors.filter((d) => d.rooms.length >= 2), rooms: R };
}

/**
 * The egress map: the field with every exit open, each room's worst walk and
 * route, the catchments, the contours, the near-limit list, the sweep and
 * cascade, the common paths and dead ends, and the critical doors.
 */
export function egressMap(graph, { exits = [], levels = [], useWalls = true, rooms = null, doorNames = null, limit = Infinity, sprinklered = false, occupantArea = 9, widthAt = null, rounds = 6, contourStep = 5, onProgress = null } = {}) {
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const base = prepareEgress(graph, { exits, levels, useWalls, rooms, doorNames });
  const nodes = floorNodes(base);
  const cell = (graph.spacing || 0.5) ** 2;
  const field = exitField(base);
  const P = graph.nodes;
  const where = (v) => [P[v * 3], P[v * 3 + 1], P[v * 3 + 2]];
  const roomList = roomWorst(base, field).map((r) => ({
    ...r,
    exitLabel: r.exit >= 0 ? base.groups[r.exit].label : null,
    at: where(r.node),
    route: Number.isFinite(r.dist) ? routeOf(field, r.node) : [],
    near: Number.isFinite(limit) && r.dist <= limit && r.dist >= limit * (1 - NEAR_LIMIT),
    over: r.dist > limit,
  }));
  roomList.sort((a, b) => (Number.isFinite(b.dist) ? b.dist : 1e9) - (Number.isFinite(a.dist) ? a.dist : 1e9));
  const catchment = base.groups.map(() => ({ area: 0, rooms: 0 }));
  for (const v of nodes) if (field.exit[v] >= 0) catchment[field.exit[v]].area += cell;
  for (const r of roomList) if (r.exit >= 0) catchment[r.exit].rooms += 1;
  const sweep = exitSweep(base, { limit, rounds, onProgress });
  const common = commonPaths(base);
  const dead = common ? deadEnds(base, common, { sprinklered, widthAt, occupantArea }) : null;
  const critical = criticalDoors(base);
  const total = summary(base, field, nodes, limit, cell);
  return {
    exits: base.groups.map((g, i) => ({ label: g.label, kind: g.kind, position: g.position || null, catchment: catchment[i], live: g.nodes.length > 0 })),
    field: { dist: Float32Array.from(field.dist), exit: field.exit },
    summary: total,
    rooms: roomList,
    contours: contours(base, field.dist, { step: contourStep }),
    sweep,
    common: common ? Float32Array.from(common) : null,
    deadEnds: dead,
    critical,
    limit,
    ms: (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0,
  };
}
