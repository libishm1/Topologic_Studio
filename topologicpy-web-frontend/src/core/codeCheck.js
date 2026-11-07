/**
 * A first code check of the egress on the walkable graph, for an office
 * (occupancy group B): walking distance to an exit, exits per storey against
 * the occupant load, exit separation, exit door widths, and whether the
 * building is high-rise.
 *
 * Each limit is the stricter of two codes, so a pass meets both (the user's
 * rule, 2026-11-07: it has to pass SBC 2018, with IBC 2021 where it is
 * stricter):
 * - the Saudi Fire Code SBC 801-CR-2018, Chapter 10 (means of egress), read
 *   in a licensed copy and cited by clause and value only;
 * - the International Building Code 2021, which the current SBC 2024 edition
 *   appears to follow (its table of contents keeps IBC 2021's sections).
 * SBC 801-CR-2024 replaces the 2018 values when its text is available. The
 * high-rise provisions are SBC 201's, which was not read, so they are IBC
 * 2021's (wiki/research/fire-suppression-sprinklers-2026-11-07.md). Each check
 * reports pass, fail or cannot tell, with the clauses of both codes.
 *
 * What the graph measures, and what it assumes:
 * - An exit is entered at an exit door on the storey that has them (the
 *   level of exit discharge) and at the door into a stair elsewhere: the
 *   code measures "to the entrance to an exit", which on an upper storey is
 *   the exit stair's enclosure. Whether a stair is enclosed is not read; every
 *   stair counts as an exit stair.
 * - Distances are walked on the storey's own graph, never through walls.
 * - The occupant load uses the storey's walkable area, which is less than the
 *   gross area the code divides, so it is a lower bound.
 */

import { csr, edgeLengths } from "./route.js";

const KIND_STAIR = 1;
const KIND_DOOR = 2;
/** Exit entrances closer than this (m) are one exit: double doors, two doors into one lobby. */
const MERGE_M = 2;

/** Group B limits of SBC 801-CR-2018, Chapter 10, with their clauses. */
export const SBC2018_B = Object.freeze({
  occupantArea: { value: 9, clause: "SBC 801-CR-2018 Table 1004.1.2: business areas, 9 m² gross per occupant" },
  travel: { unsprinklered: 60, sprinklered: 90, clause: "SBC 801-CR-2018 Table 1017.2: Group B, 60 m, 90 m with sprinklers" },
  exitsPerStorey: { rows: [[500, 2], [1000, 3], [Infinity, 4]], clause: "SBC 801-CR-2018 Table 1006.3.1: 1 to 500 occupants 2 exits, 501 to 1,000 3, more 4" },
  singleExit: {
    first: { occupants: 49, travel: 23 },
    second: { occupants: 29, travel: 23 },
    sprinkleredTravel: 30,
    clause: "SBC 801-CR-2018 Table 1006.3.2(2): one exit for Group B only on the first storey (49 occupants) or second (29), within 23 m (30 m with sprinklers)",
  },
  separation: { unsprinklered: 1 / 2, sprinklered: 1 / 3, clause: "SBC 801-CR-2018 1007.1.1: 1/2 of the diagonal, 1/3 with sprinklers" },
  doorWidth: { value: 0.8, clause: "SBC 801-CR-2018 1010.1.1: 800 mm clear" },
  highRise: { value: 23, clause: "SBC 801-CR-2018 Chapter 2: an occupied floor more than 23 m above the lowest Civil Defense vehicle access" },
});

/** Group B limits of IBC 2021, with their clauses. */
export const IBC2021_B = Object.freeze({
  occupantArea: { value: 13.94, clause: "IBC 2021 Table 1004.5: 150 sq ft (13.94 m²) gross" },
  travel: { unsprinklered: 60.96, sprinklered: 91.44, clause: "IBC 2021 Table 1017.2: 200 ft (60.96 m), 300 ft (91.44 m) with sprinklers" },
  exitsPerStorey: { rows: [[500, 2], [1000, 3], [Infinity, 4]], clause: "IBC 2021 Table 1006.3.3" },
  singleExit: {
    first: { occupants: 49, travel: 22.86 },
    second: { occupants: 29, travel: 22.86 },
    sprinkleredTravel: 30.48,
    clause: "IBC 2021 Table 1006.3.4(2): within 75 ft (22.86 m), 100 ft (30.48 m) with sprinklers; not from the third storey up",
  },
  separation: { unsprinklered: 1 / 2, sprinklered: 1 / 3, clause: "IBC 2021 1007.1.1" },
  doorWidth: { value: 0.813, clause: "IBC 2021 1010.1.1: 32 in (813 mm) clear" },
  highRise: { value: 22.86, clause: "IBC 2021 Section 202: more than 75 ft (22.86 m) above the lowest fire department vehicle access" },
  highRiseProvisions: [
    { above: 22.86, text: "smokeproof exit enclosures (IBC 2021 403.5.4) and luminous egress path markings (403.5.5, 1025); SBC 201 not read" },
    { above: 36.58, text: "at least two fire service access elevators (IBC 2021 403.6.1, above 120 ft); SBC 201 not read" },
    { above: 128.0, text: "an additional exit stair, unless occupant evacuation elevators are provided (IBC 2021 403.5.2, above 420 ft); SBC 201 not read" },
  ],
});

/**
 * The stricter of two codes' limits, so a pass meets both: a smaller area per
 * occupant, travel distance or high-rise height, a larger door width,
 * separation or number of exits. Each clause names both codes.
 */
export function stricter(a, b, source) {
  const lo = Math.min;
  const hi = Math.max;
  const both = (x, y) => `${x.clause}; ${y.clause}`;
  const highRise = lo(a.highRise.value, b.highRise.value);
  return Object.freeze({
    source,
    occupantArea: { value: lo(a.occupantArea.value, b.occupantArea.value), clause: both(a.occupantArea, b.occupantArea) },
    travel: {
      unsprinklered: lo(a.travel.unsprinklered, b.travel.unsprinklered),
      sprinklered: lo(a.travel.sprinklered, b.travel.sprinklered),
      clause: both(a.travel, b.travel),
    },
    exitsPerStorey: {
      rows: a.exitsPerStorey.rows.map(([upTo, n], k) => [upTo, hi(n, b.exitsPerStorey.rows[k][1])]),
      clause: both(a.exitsPerStorey, b.exitsPerStorey),
    },
    singleExit: {
      first: { occupants: lo(a.singleExit.first.occupants, b.singleExit.first.occupants), travel: lo(a.singleExit.first.travel, b.singleExit.first.travel) },
      second: { occupants: lo(a.singleExit.second.occupants, b.singleExit.second.occupants), travel: lo(a.singleExit.second.travel, b.singleExit.second.travel) },
      sprinkleredTravel: lo(a.singleExit.sprinkleredTravel, b.singleExit.sprinkleredTravel),
      clause: both(a.singleExit, b.singleExit),
    },
    separation: {
      unsprinklered: hi(a.separation.unsprinklered, b.separation.unsprinklered),
      sprinklered: hi(a.separation.sprinklered, b.separation.sprinklered),
      clause: both(a.separation, b.separation),
    },
    doorWidth: { value: hi(a.doorWidth.value, b.doorWidth.value), clause: both(a.doorWidth, b.doorWidth) },
    highRise: { value: highRise, clause: both(a.highRise, b.highRise) },
    highRiseProvisions: b.highRiseProvisions.map((p, k) => (k === 0 ? { ...p, above: highRise } : p)),
  });
}

/** The limits the check applies: SBC 801-CR-2018 and IBC 2021, the stricter of each. */
export const EGRESS_B = stricter(SBC2018_B, IBC2021_B, "the stricter of SBC 801-CR-2018 and IBC 2021; SBC 801-CR-2024 replaces the 2018 values when available");

/** Indices of the axis that is up. */
const upIndex = (upAxis) => (upAxis === "z" ? 2 : upAxis === "x" ? 0 : 1);

/** The farthest two points of a set, in the horizontal plane (over its convex hull). */
function diagonal(points) {
  if (points.length < 2) return 0;
  const sorted = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  for (let i = sorted.length - 1; i >= 0; i -= 1) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  let best = 0;
  for (let i = 0; i < hull.length; i += 1) {
    for (let j = i + 1; j < hull.length; j += 1) best = Math.max(best, Math.hypot(hull[i][0] - hull[j][0], hull[i][1] - hull[j][1]));
  }
  return best;
}

/**
 * Check the egress on the graph.
 *
 * @param {object} graph   nodes (viewer frame, lifted by agentHeight), edges, kinds, blocked, upAxis, agentHeight, spacing
 * @param {object} opts
 *   exits        exit door nodes (ifcSemantics.exitDoors(...).nodes)
 *   levels       storey floor heights in the viewer frame (lattice.floorLevels), ascending
 *   storeyNames  names for the levels, optional
 *   exitDoors    [{widthM}] of the flagged exit doors, optional
 *   sprinklered  the building has an automatic sprinkler system
 *   useWalls     respect links blocked by walls (default true)
 * @returns {{limits, building, storeys, doors, travel: Float64Array, limit: number}}
 */
export function codeCheck(graph, { exits = [], levels = [], storeyNames = [], exitDoors = [], sprinklered = false, useWalls = true } = {}) {
  const L = EGRESS_B;
  const n = graph.nodes.length / 3;
  const up = upIndex(graph.upAxis || "y");
  const lift = graph.agentHeight || 0;
  const kinds = graph.kinds || new Uint8Array(n);
  const blocked = useWalls ? graph.blocked : null;
  const { indptr, neighbours, edgeIds } = csr(graph);
  const lengths = edgeLengths(graph);
  const limit = sprinklered ? L.travel.sprinklered : L.travel.unsprinklered;
  const floors = levels.length ? levels.slice().sort((a, b) => a - b) : [0];

  // Storey of every node off the stairs: the highest floor at or below it.
  const storeyOf = new Int32Array(n).fill(-1);
  for (let v = 0; v < n; v += 1) {
    if (kinds[v] === KIND_STAIR) continue;
    const h = graph.nodes[v * 3 + up] - lift;
    let s = 0;
    for (let i = 0; i < floors.length; i += 1) if (floors[i] <= h + 0.5) s = i;
    storeyOf[v] = s;
  }

  // Stairs as exits: the stair nodes' connected pieces, each one exit.
  const stairPiece = new Int32Array(n).fill(-1);
  let pieces = 0;
  for (let v = 0; v < n; v += 1) {
    if (kinds[v] !== KIND_STAIR || stairPiece[v] >= 0) continue;
    const stack = [v];
    stairPiece[v] = pieces;
    while (stack.length) {
      const u = stack.pop();
      for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
        const w = neighbours[k];
        if (kinds[w] === KIND_STAIR && stairPiece[w] < 0) {
          stairPiece[w] = pieces;
          stack.push(w);
        }
      }
    }
    pieces += 1;
  }

  // Exit entrances per storey: exit doors on the storeys that have them,
  // the doors into the stairs on the others.
  const exitSet = new Set(exits);
  const discharge = new Set([...exitSet].map((v) => storeyOf[v]).filter((s) => s >= 0));
  const entrances = floors.map(() => new Map()); // node -> exit id
  for (const v of exitSet) if (storeyOf[v] >= 0) entrances[storeyOf[v]].set(v, `door ${v}`);
  for (let v = 0; v < n; v += 1) {
    const s = storeyOf[v];
    if (s < 0 || discharge.has(s)) continue;
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      const w = neighbours[k];
      if (kinds[w] === KIND_STAIR && !(blocked && blocked[edgeIds[k]])) entrances[s].set(v, `stair ${stairPiece[w]}`);
    }
  }

  // Walking distance on each storey to its nearest exit entrance (Dijkstra from all of them).
  const travel = new Float64Array(n).fill(Infinity);
  const nearest = new Array(n).fill(null);
  const heap = [];
  const push = (d, v) => {
    heap.push([d, v]);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heap[p][0] <= heap[i][0]) break;
      [heap[p], heap[i]] = [heap[i], heap[p]];
      i = p;
    }
  };
  const pop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
        if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
        if (m === i) break;
        [heap[m], heap[i]] = [heap[i], heap[m]];
        i = m;
      }
    }
    return top;
  };
  const seed = (v, id) => {
    travel[v] = 0;
    nearest[v] = id;
    push(0, v);
  };
  const spread = () => {
    while (heap.length) {
      const [d, v] = pop();
      if (d > travel[v]) continue;
      for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
        const w = neighbours[k];
        if (storeyOf[w] !== storeyOf[v] || (blocked && blocked[edgeIds[k]])) continue;
        const nd = d + lengths[edgeIds[k]];
        if (nd < travel[w]) {
          travel[w] = nd;
          nearest[w] = nearest[v];
          push(nd, w);
        }
      }
    }
  };
  for (const map of entrances) for (const [v, id] of map) seed(v, id);
  spread();
  // On a storey with exit doors, a part that reaches none of them on the
  // storey (a lower level the storey also holds, reached by a stair) leaves
  // by the stair: an exit stair is an exit too (IBC 202).
  for (const s of discharge) {
    for (let v = 0; v < n; v += 1) {
      if (storeyOf[v] !== s || Number.isFinite(travel[v])) continue;
      for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
        const w = neighbours[k];
        if (kinds[w] === KIND_STAIR && !(blocked && blocked[edgeIds[k]])) {
          entrances[s].set(v, `stair ${stairPiece[w]}`);
          seed(v, `stair ${stairPiece[w]}`);
        }
      }
    }
  }
  spread();

  // Per storey.
  const cell = (graph.spacing || 0.5) ** 2;
  const groundIndex = discharge.size ? Math.min(...discharge) : 0;
  const storeys = floors.map((elevation, s) => {
    const nodes = [];
    for (let v = 0; v < n; v += 1) if (storeyOf[v] === s && kinds[v] !== KIND_DOOR) nodes.push(v);
    if (!nodes.length) return null;
    const area = nodes.length * cell;
    const occupants = Math.ceil(area / L.occupantArea.value);
    // Exit entrances, one per exit: a door's nodes, or a stair's, and any
    // within MERGE_M of each other (double doors, two doors into one lobby).
    const where = new Map();
    for (const [v, id] of entrances[s]) {
      const p = where.get(id) || { x: 0, z: 0, k: 0 };
      p.x += graph.nodes[v * 3];
      p.z += graph.nodes[v * 3 + (up === 1 ? 2 : 1)];
      p.k += 1;
      where.set(id, p);
    }
    const groups = [];
    for (const p of where.values()) {
      const at = [p.x / p.k, p.z / p.k];
      const g = groups.find((q) => q.some((r) => Math.hypot(r[0] - at[0], r[1] - at[1]) < MERGE_M));
      if (g) g.push(at);
      else groups.push([at]);
    }
    const found = groups.length;
    const required = L.exitsPerStorey.rows.find(([upTo]) => occupants <= upTo)[1];
    let maxTravel = 0, over = 0, unreached = 0;
    for (const v of nodes) {
      if (!Number.isFinite(travel[v])) unreached += 1;
      else {
        maxTravel = Math.max(maxTravel, travel[v]);
        if (travel[v] > limit) over += 1;
      }
    }
    // One exit is allowed only low down, for few people, close to it.
    const rank = s - groundIndex; // 0: the level of exit discharge (first storey); 1: the second
    const single = rank <= 0 ? L.singleExit.first : rank === 1 ? L.singleExit.second : null;
    const oneAllowed = Boolean(single && occupants <= single.occupants && maxTravel <= (sprinklered ? L.singleExit.sprinkleredTravel : single.travel) && !unreached);
    const needed = oneAllowed ? 1 : required;
    // Exit separation: the two exits farthest apart, against the storey's diagonal.
    const spots = groups.map((g) => [g.reduce((a, p) => a + p[0], 0) / g.length, g.reduce((a, p) => a + p[1], 0) / g.length]);
    const plan = nodes.map((v) => [graph.nodes[v * 3], graph.nodes[v * 3 + (up === 1 ? 2 : 1)]]);
    const diag = diagonal(plan);
    const apart = diagonal(spots);
    const fraction = sprinklered ? L.separation.sprinklered : L.separation.unsprinklered;
    const checks = [
      {
        id: "travel",
        status: over ? "fail" : unreached === nodes.length ? "cannot tell" : "pass",
        text:
          unreached === nodes.length
            ? "no place on this storey reaches an exit on the graph"
            : `longest walk to an exit ${maxTravel.toFixed(1)} m, limit ${limit.toFixed(2)} m${over ? `; ${over} of ${nodes.length} places over it` : ""}`,
        clause: L.travel.clause,
      },
      // A place with no exit on the graph is a question about the model first
      // (a door or stair it lacks, a pocket of floor), not a code failure.
      ...(unreached && unreached < nodes.length
        ? [
            {
              id: "unreached",
              status: "cannot tell",
              text: `${unreached} of ${nodes.length} places reach no exit on the graph: often a door or a stair the model lacks, or a pocket of floor; see the graph's parts`,
              clause: L.travel.clause,
            },
          ]
        : []),
      {
        id: "exits",
        status: found >= needed ? "pass" : "fail",
        text: `${found} exit${found === 1 ? "" : "s"} for at least ${occupants} occupants; ${needed} needed${oneAllowed ? " (one is allowed here)" : ""}`,
        clause: oneAllowed || found < 2 ? `${L.exitsPerStorey.clause}; ${L.singleExit.clause}` : L.exitsPerStorey.clause,
      },
      found >= 2
        ? {
            id: "separation",
            status: apart >= fraction * diag ? "pass" : "fail",
            text: `exits ${apart.toFixed(1)} m apart; ${(fraction * diag).toFixed(1)} m needed (${sprinklered ? "1/3" : "1/2"} of the ${diag.toFixed(1)} m diagonal)`,
            clause: L.separation.clause,
          }
        : { id: "separation", status: "n/a", text: "fewer than two exits", clause: L.separation.clause },
    ];
    return {
      index: s,
      name: storeyNames[s] || `Storey ${s + 1}`,
      elevation,
      discharge: discharge.has(s),
      places: nodes.length,
      area,
      occupants,
      exits: found,
      maxTravel,
      checks,
    };
  }).filter(Boolean);

  // Exit door widths, where the IFC gives them.
  const widths = exitDoors.map((d) => d.widthM).filter((w) => w > 0);
  const narrow = widths.filter((w) => w < L.doorWidth.value);
  const doors = {
    status: !widths.length ? "cannot tell" : narrow.length ? "fail" : "pass",
    text: !widths.length
      ? "the exit doors' widths are not in the file"
      : `${widths.length} exit doors with a width; ${narrow.length ? `${narrow.length} narrower than` : "all at least"} ${L.doorWidth.value} m` +
        (widths.length < exitDoors.length ? `; ${exitDoors.length - widths.length} without one` : ""),
    clause: L.doorWidth.clause,
  };

  // High-rise: the highest occupied storey above the level of exit discharge.
  const occupied = storeys.map((s) => s.elevation);
  const base = floors[groundIndex] ?? 0;
  const height = occupied.length ? Math.max(...occupied) - base : 0;
  const highRise = height > L.highRise.value;
  const building = {
    height,
    highRise,
    text: `the highest occupied floor is ${height.toFixed(1)} m above the level of exit discharge: ${highRise ? "a high-rise building" : "not a high-rise building"}`,
    clause: L.highRise.clause,
    provisions: highRise ? L.highRiseProvisions.filter((p) => height > p.above).map((p) => p.text) : [],
    sprinklered,
  };

  return { limits: L, building, storeys, doors, travel, limit };
}
