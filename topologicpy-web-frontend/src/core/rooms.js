/**
 * The room graph of the two-zone fire model (gap LG8), section 1 of the
 * specification (wiki/research/zone-model-spec-2026-10-28.md). The dynamics
 * on it are in src/core/zone.js.
 *
 * Rooms are IfcSpace volumes, one per IFC item. For room k the model needs
 * its floor area A_f, floor level z_f and height H (from the space mesh), and
 * its internal surface area A_T = 2 A_f + P H minus the area of its doors, P
 * the footprint perimeter.
 *   - The floor is the mesh's lowest horizontal faces, whichever way the
 *     exporter wound them (as lattice.js samples spaces). A_f is their plan
 *     area, and P the length of their outer edges (edges used by one floor
 *     triangle only), or 4 sqrt(A_f) when the faces give none. A space with no
 *     floor faces is not a room.
 *   - A lattice node belongs to room k when it lies in k's footprint within
 *     0.5 m of k's floor level (a node stands agentHeight above its floor).
 *     Where footprints overlap (a gross-area space drawn over its rooms) the
 *     smallest room wins. Nodes in no footprint take the room of the nearest
 *     assigned node along the graph: a multi-source breadth-first search over
 *     the open links, so a node in a door opening joins the room it is reached
 *     from first. Nodes in a part with no room get none, and no fire exposure.
 *     Departure from the specification: the search does not take a room more
 *     than `band` (1 m) above or below that room's floor. Unbounded, it carried
 *     the rooms of the one storey with spaces up and down the stairs to every
 *     storey without (ISSUE_159: half the building took the room at the stair
 *     head, and would close with its smoke). band = Infinity is the
 *     specification's rule.
 *   - A graph door (lattice.js doorReport) joins the rooms on its two sides.
 *     A side's room is the room most of the door node's neighbours on that
 *     side stand in (by footprint, not by the search above, which would hand
 *     the outside the room behind the door); failing that, the room under a
 *     probe point 0.25 to 1 m across the opening, within 1 m of the sill
 *     height. A side in no room is outside. A door with one side outside, or flagged as an exit in the IFC
 *     (ifcSemantics.js), vents its room to the outside; one with the same room
 *     on both sides is inside that room and takes no part.
 *   - Door width W_d and height H_d: OverallWidth and OverallHeight in metres
 *     when the IFC gives them, else the width measured from the door mesh and
 *     2.0 m. Doors are taken as open.
 */
import { axisIndex, horizontalAxes } from "./lattice.js";
import { csr } from "./route.js";

/** A node is at a room's floor level within this many metres (spec section 1). */
export const FLOOR_TOLERANCE = 0.5;
/** The search and the door probes reach this far above or below a room's floor (m). */
export const SEARCH_BAND = 1.0;
/** H_d when the IFC gives no OverallHeight (m). */
export const DEFAULT_DOOR_HEIGHT = 2.0;
/** W_d when neither the IFC nor the mesh gives a width (m). */
const FALLBACK_DOOR_WIDTH = 0.9;
/** Horizontal faces within 10 degrees, at most 5 cm above the lowest point, as lattice.js. */
const MIN_UP = Math.cos((10 * Math.PI) / 180);
const BOTTOM_BAND = 0.05;
/** Probe distances across a door opening when no neighbour settles a side (m). */
const PROBES = [0.25, 0.5, 0.75, 1.0];

/**
 * Rooms from IfcSpace meshes, one per item (several meshes of one space are
 * pooled).
 *
 * @returns {Array<{itemId, floorArea: number, floorLevel: number, height: number,
 *           perimeter: number, perimeterFromMesh: boolean, tris: Float64Array,
 *           box: number[]}>}  tris: the floor triangles in plan, six numbers each.
 */
export function roomsFromSpaces(spaces, upAxis = "y") {
  const up = axisIndex(upAxis);
  const [h0, h1] = horizontalAxes(upAxis);
  const groups = new Map();
  (spaces || []).forEach((mesh, i) => {
    if (!mesh?.positions?.length || !mesh?.indices?.length) return;
    const k = mesh.itemId ?? `mesh:${i}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(mesh);
  });
  const rooms = [];
  for (const [k, meshes] of groups) {
    let lo = Infinity, hi = -Infinity;
    for (const { positions: p } of meshes) {
      for (let o = up; o < p.length; o += 3) {
        if (p[o] < lo) lo = p[o];
        if (p[o] > hi) hi = p[o];
      }
    }
    const tris = [];
    const edges = new Map();
    let area = 0;
    const vkey = (x, y) => `${Math.round(x * 1000)},${Math.round(y * 1000)}`;
    for (const { positions: p, indices: ix } of meshes) {
      for (let t = 0; t + 2 < ix.length; t += 3) {
        const a = ix[t] * 3, b = ix[t + 1] * 3, c = ix[t + 2] * 3;
        if (a + 2 >= p.length || b + 2 >= p.length || c + 2 >= p.length) continue;
        if (Math.max(p[a + up], p[b + up], p[c + up]) > lo + BOTTOM_BAND) continue;
        const e1 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
        const e2 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
        const nrm = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const mag = Math.hypot(nrm[0], nrm[1], nrm[2]);
        if (mag < 1e-9 || Math.abs(nrm[up] / mag) < MIN_UP) continue;
        const ax = p[a + h0], ay = p[a + h1], bx = p[b + h0], by = p[b + h1], cx = p[c + h0], cy = p[c + h1];
        const twice = Math.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay));
        if (twice < 1e-9) continue;
        area += twice / 2;
        tris.push(ax, ay, bx, by, cx, cy);
        for (const [x0, y0, x1, y1] of [[ax, ay, bx, by], [bx, by, cx, cy], [cx, cy, ax, ay]]) {
          const ka = vkey(x0, y0), kb = vkey(x1, y1);
          const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
          const entry = edges.get(key);
          if (entry) entry.count += 1;
          else edges.set(key, { count: 1, length: Math.hypot(x1 - x0, y1 - y0) });
        }
      }
    }
    if (!(area > 1e-6) || !(hi > lo)) continue;
    let perimeter = 0;
    for (const { count, length } of edges.values()) if (count % 2 === 1) perimeter += length;
    const perimeterFromMesh = perimeter > 0;
    if (!perimeterFromMesh) perimeter = 4 * Math.sqrt(area);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < tris.length; i += 2) {
      x0 = Math.min(x0, tris[i]);
      x1 = Math.max(x1, tris[i]);
      y0 = Math.min(y0, tris[i + 1]);
      y1 = Math.max(y1, tris[i + 1]);
    }
    rooms.push({
      itemId: typeof k === "string" && k.startsWith("mesh:") ? null : k,
      floorArea: area,
      floorLevel: lo,
      height: hi - lo,
      perimeter,
      perimeterFromMesh,
      tris: Float64Array.from(tris),
      box: [x0, y0, x1, y1],
    });
  }
  return rooms;
}

/**
 * A plan grid over the rooms' floor triangles: roomAt(x, y, floor, tolerance)
 * is the smallest room whose footprint holds (x, y) and whose floor level is
 * within `tolerance` (default FLOOR_TOLERANCE) of `floor`, or -1.
 */
export function footprintIndex(rooms, cell = 2) {
  const cells = new Map();
  const ckey = (i, j) => i * 1048576 + j;
  rooms.forEach((room, r) => {
    const T = room.tris;
    for (let t = 0; t < T.length; t += 6) {
      const i0 = Math.floor(Math.min(T[t], T[t + 2], T[t + 4]) / cell);
      const i1 = Math.floor(Math.max(T[t], T[t + 2], T[t + 4]) / cell);
      const j0 = Math.floor(Math.min(T[t + 1], T[t + 3], T[t + 5]) / cell);
      const j1 = Math.floor(Math.max(T[t + 1], T[t + 3], T[t + 5]) / cell);
      for (let i = i0; i <= i1; i += 1) {
        for (let j = j0; j <= j1; j += 1) {
          const k = ckey(i, j);
          if (!cells.has(k)) cells.set(k, []);
          cells.get(k).push(r, t);
        }
      }
    }
  });
  const inside = (T, t, x, y) => {
    const ax = T[t], ay = T[t + 1], bx = T[t + 2], by = T[t + 3], cx = T[t + 4], cy = T[t + 5];
    const d1 = (x - bx) * (ay - by) - (ax - bx) * (y - by);
    const d2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy);
    const d3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay);
    const eps = 1e-9;
    const neg = d1 < -eps || d2 < -eps || d3 < -eps;
    const pos = d1 > eps || d2 > eps || d3 > eps;
    return !(neg && pos);
  };
  return {
    roomAt(x, y, floor, tolerance = FLOOR_TOLERANCE) {
      const list = cells.get(ckey(Math.floor(x / cell), Math.floor(y / cell)));
      if (!list) return -1;
      let best = -1;
      for (let q = 0; q < list.length; q += 2) {
        const r = list[q];
        const room = rooms[r];
        if (best >= 0 && room.floorArea >= rooms[best].floorArea) continue;
        if (Math.abs(floor - room.floorLevel) > tolerance) continue;
        if (inside(room.tris, list[q + 1], x, y)) best = r;
      }
      return best;
    },
  };
}

/**
 * Lattice nodes to rooms (spec section 1, second item, with the search kept
 * within `band` of each room's floor; see the header).
 *
 * @returns {{room: Int32Array, direct: Int32Array, directCount: number, assigned: number}}
 *          room: each node's room after the search (-1: none); direct: by
 *          footprint alone.
 */
export function assignNodes(graph, rooms, index, { upAxis = "y", agentHeight = 0.75, blocked = graph.blocked, band = SEARCH_BAND } = {}) {
  const up = axisIndex(upAxis);
  const [h0, h1] = horizontalAxes(upAxis);
  const P = graph.nodes;
  const n = P.length / 3;
  const direct = new Int32Array(n).fill(-1);
  const room = new Int32Array(n).fill(-1);
  const queue = new Int32Array(n);
  let head = 0, tail = 0, directCount = 0;
  for (let v = 0; v < n; v += 1) {
    const r = index.roomAt(P[v * 3 + h0], P[v * 3 + h1], P[v * 3 + up] - agentHeight);
    direct[v] = r;
    if (r >= 0) {
      room[v] = r;
      queue[tail++] = v;
      directCount += 1;
    }
  }
  const { indptr, neighbours, edgeIds } = csr(graph);
  while (head < tail) {
    const v = queue[head++];
    const level = rooms[room[v]].floorLevel;
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      if (blocked?.[edgeIds[k]]) continue;
      const w = neighbours[k];
      if (room[w] >= 0 || Math.abs(P[w * 3 + up] - agentHeight - level) > band) continue;
      room[w] = room[v];
      queue[tail++] = w;
    }
  }
  return { room, direct, directCount, assigned: tail };
}

/**
 * The rooms on each side of a graph door, its size, and its part in the model
 * (spec section 1, third item).
 *
 * @param {object[]} doors  graph doors: {itemId, node, sides, x, y, sill, width,
 *                          ux, uy} (sampler.worker.js; the last six from
 *                          lattice.js doorReport and doorPoints)
 * @param {Map} declared    itemId -> {widthM, heightM, exit} (ifcSemantics.js)
 * @returns {Array<{itemId, node: number, a: number, b: number, width: number,
 *           height: number, exit: boolean, kind: string, sized: boolean}>}
 *          kind "join" (rooms a and b), "vent" (room a to the outside), or
 *          "inside" / "none" (no part in the model). An exit door with rooms on
 *          both sides vents each of them, so it appears twice.
 */
export function roomDoors(graph, doors, direct, index, { declared = new Map(), upAxis = "y", agentHeight = 0.75, band = SEARCH_BAND } = {}) {
  const up = axisIndex(upAxis);
  const [h0, h1] = horizontalAxes(upAxis);
  const P = graph.nodes;
  const { indptr, neighbours } = csr(graph);
  const out = [];
  for (const d of doors || []) {
    if (!(d?.node >= 0)) continue;
    const info = declared.get(d.itemId) || {};
    const width = info.widthM > 0 ? info.widthM : d.width > 0 ? d.width : FALLBACK_DOOR_WIDTH;
    const height = info.heightM > 0 ? info.heightM : DEFAULT_DOOR_HEIGHT;
    const base = { itemId: d.itemId ?? null, node: d.node, width, height, exit: Boolean(info.exit), sized: info.widthM > 0 || info.heightM > 0 };
    const v = d.node;
    const px = Number.isFinite(d.x) ? d.x : P[v * 3 + h0];
    const py = Number.isFinite(d.y) ? d.y : P[v * 3 + h1];
    const floor = Number.isFinite(d.sill) ? d.sill : P[v * 3 + up] - agentHeight;
    let sideRooms;
    if (Number.isFinite(d.ux) && Number.isFinite(d.uy)) {
      // Across the opening: the normal to the door's long axis.
      const nx = -d.uy, ny = d.ux;
      const votes = [new Map(), new Map()];
      for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
        const w = neighbours[k];
        const across = (P[w * 3 + h0] - px) * nx + (P[w * 3 + h1] - py) * ny;
        if (Math.abs(across) <= 1e-6 || direct[w] < 0) continue;
        const side = votes[across > 0 ? 0 : 1];
        side.set(direct[w], (side.get(direct[w]) || 0) + 1);
      }
      sideRooms = [1, -1].map((sign, s) => {
        let best = -1, most = 0;
        for (const [r, c] of votes[s]) if (c > most) [best, most] = [r, c];
        // Within the search band: a door's sill is its mesh's lowest point,
        // which some exporters sink into the slab (HouseZ: 0.51 m below the
        // spaces' floor).
        for (let i = 0; best < 0 && i < PROBES.length; i += 1) {
          best = index.roomAt(px + sign * PROBES[i] * nx, py + sign * PROBES[i] * ny, floor, Math.min(band, SEARCH_BAND));
        }
        return best;
      });
    } else {
      // No direction (an older graph): the two most common rooms among the
      // neighbours; with only one, outside if a side has no floor at all or a
      // neighbour stands in no room.
      const votes = new Map();
      let roomless = false;
      for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
        const r = direct[neighbours[k]];
        if (r >= 0) votes.set(r, (votes.get(r) || 0) + 1);
        else roomless = true;
      }
      const ranked = [...votes].sort((p, q) => q[1] - p[1]).map(([r]) => r);
      const outside = roomless || d.sides?.[0] === 0 || d.sides?.[1] === 0;
      sideRooms = [ranked[0] ?? -1, ranked[1] ?? (outside ? -1 : (ranked[0] ?? -1))];
    }
    const [a, b] = sideRooms;
    if (base.exit) {
      const vented = [...new Set([a, b].filter((r) => r >= 0))];
      if (!vented.length) out.push({ ...base, a: -1, b: -1, kind: "none" });
      for (const r of vented) out.push({ ...base, a: r, b: -1, kind: "vent" });
    } else if (a >= 0 && b >= 0) {
      out.push({ ...base, a, b, kind: a === b ? "inside" : "join" });
    } else if (a >= 0 || b >= 0) {
      out.push({ ...base, a: a >= 0 ? a : b, b: -1, kind: "vent" });
    } else {
      out.push({ ...base, a: -1, b: -1, kind: "none" });
    }
  }
  return out;
}

/**
 * The whole room graph for zone.js: rooms with their A_T and doors, the node
 * assignment, and counts for the record.
 *
 * @param {object} graph  {nodes, edges, blocked}
 * @param {object} opts   spaces: IfcSpace meshes {itemId, positions, indices};
 *                        doors: graph doors (see roomDoors); declaredDoors:
 *                        ifcSemantics doors ({id, widthM, heightM, exit});
 *                        upAxis; agentHeight; band: how far the search
 *                        reaches above or below a room's floor (Infinity:
 *                        the specification's unbounded search).
 */
export function buildRoomGraph(graph, { spaces = [], doors = [], declaredDoors = [], upAxis = "y", agentHeight = 0.75, band = SEARCH_BAND } = {}) {
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const rooms = roomsFromSpaces(spaces, upAxis);
  const index = footprintIndex(rooms);
  const { room: nodeRoom, direct, directCount, assigned } = assignNodes(graph, rooms, index, { upAxis, agentHeight, band });
  const declared = new Map((declaredDoors || []).map((d) => [d.id, d]));
  const all = roomDoors(graph, doors, direct, index, { declared, upAxis, agentHeight, band });
  const used = all.filter((d) => d.kind === "join" || d.kind === "vent");
  rooms.forEach((room) => {
    room.doors = [];
    room.doorArea = 0;
  });
  used.forEach((d, i) => {
    for (const r of d.kind === "join" ? [d.a, d.b] : [d.a]) {
      rooms[r].doors.push(i);
      rooms[r].doorArea += d.width * d.height;
    }
  });
  for (const room of rooms) {
    // A_T = 2 A_f + P H less the doors; kept above the floor area, so a room
    // whose doors are larger than its walls (a mis-sized door, a tiny space)
    // still has walls to lose heat to.
    room.AT = Math.max(room.floorArea, 2 * room.floorArea + room.perimeter * room.height - room.doorArea);
  }
  const t1 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const n = graph.nodes.length / 3;
  return {
    rooms,
    doors: used,
    nodeRoom,
    direct,
    stats: {
      spaces: new Set((spaces || []).map((m, i) => m?.itemId ?? `mesh:${i}`)).size,
      rooms: rooms.length,
      nodes: n,
      inFootprint: directCount,
      assigned,
      joins: used.filter((d) => d.kind === "join").length,
      vents: used.filter((d) => d.kind === "vent").length,
      inside: all.filter((d) => d.kind === "inside").length,
      unplaced: all.filter((d) => d.kind === "none").length,
      sizedFromIfc: used.filter((d) => d.sized).length,
      ms: t1 - t0,
    },
  };
}
