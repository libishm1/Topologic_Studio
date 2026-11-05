/**
 * LOD1 building models: one vertical prism per room, extruded per storey,
 * and the egress geometry the lattice (lattice.js) builds its graph from.
 *
 * Every source becomes one model, and one converter turns the model into the
 * shape ViewerManager.extractEgressGeometry returns, so the graph, routes,
 * fire models and Flamescapes run on it unchanged. Nothing is sliced into
 * plans: a room's footprint is a polygon, its floor and walls are extruded
 * from it.
 *
 * Sources:
 *   - parseTopologicLOD1: a TopologicPy CellComplex written by
 *     Topology.ExportToJSON (one Cell per room, a dictionary on each), with
 *     its circulation graph (doors, stairs, lifts) and adjacency graph;
 *   - parseGeoJSONPlan: Studio's GeoJSON plan schema (below), for drawings
 *     traced in other tools;
 *   - lod1FromIfcSpaces: IfcSpace meshes, door meshes and storeys of an IFC,
 *     a fast path for large files that needs nothing else;
 *   - lod1FromJsonFiles: whichever of the first two is among files dropped
 *     together (lod1JsonKind says what each file is).
 * Out: lod1ToEgressGeometry (the egress geometry), lod1ToGeoJSON (export).
 * The plan's schema is also in wiki/concepts/lod1-models.md.
 *
 * The model, in IFC axes (x, y in plan, z up) and metres:
 *
 *   {
 *     levels: [{id, name, ffl, height}],                  ascending ffl
 *     rooms:  [{id, name, level, ffl, height, class, area,
 *               footprint: [[x, y], ...],                 outer ring, anticlockwise
 *               holes: [[[x, y], ...]],                   clockwise
 *               walkable}],                               false: shafts, voids, tanks
 *     doors:  [{id, position: [x, y, z], level, kind,     z: the sill (floor level); kind below
 *               direction: [ux, uy],                      along its wall; may be null
 *               rooms: [roomA, roomB],                    room ids; null is outside
 *               width, assumed}],                         width null when unknown
 *     links:  [{id, kind, rooms: [lower, upper],          stair, lift or ramp
 *               position, width, footprint,               position and footprint may be null
 *               line?}],                                  walking line [[x, y], ...], foot to head
 *     meta:   {source, name, north, boundary, counts, warnings}
 *   }
 *
 * Door kinds: "door" (a leaf, between two rooms or a room and the outside),
 * "exit" (a final exit to the street), "opening" (no leaf: an arch, or the
 * drawn line between two parts of one space). meta.boundary says where the
 * walls stand: "shared" (rooms meet on a line, the wall's centre line:
 * TopologicPy and GeoJSON) or "inner" (rooms end at the walls' inner faces,
 * as IfcSpace volumes do, so walls stand outside them).
 */

import { CORE_LIFT, CORE_STAIR, coreKind, walkingLine } from "./cores.js";
import { doorWaypoints } from "./lattice.js";

/** Room classes nobody walks in: their floors are left out of the graph. */
const NOT_WALKABLE = /^(shaft|void|tank|duct|riser)$/i;
/** Plan tolerance for "the same point" and "on the same line" (m). */
const EPS = 1e-3;

// ------------------------------------------------------------ plan geometry

/** Signed area of a ring (positive: anticlockwise). */
export function ringArea(ring) {
  let a = 0;
  for (let i = 0, n = ring.length; i < n; i += 1) {
    const [x0, y0] = ring[i], [x1, y1] = ring[(i + 1) % n];
    a += x0 * y1 - x1 * y0;
  }
  return a / 2;
}

/** Whether (x, y) is inside a ring (crossing parity). */
export function pointInRing(x, y, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if (yi > y !== yj > y && x < xj + ((y - yj) * (xi - xj)) / (yi - yj)) inside = !inside;
  }
  return inside;
}

/** Whether (x, y) is on a room's floor: in its footprint and in none of its holes. */
export function pointInRoom(x, y, room) {
  if (!pointInRing(x, y, room.footprint)) return false;
  for (const hole of room.holes || []) if (pointInRing(x, y, hole)) return false;
  return true;
}

/** Distance from (x, y) to the segment a-b, and the parameter of the nearest point. */
function toSegment(x, y, [ax, ay], [bx, by]) {
  const vx = bx - ax, vy = by - ay;
  const len2 = vx * vx + vy * vy;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * vx + (y - ay) * vy) / len2)) : 0;
  return { d: Math.hypot(x - (ax + t * vx), y - (ay + t * vy)), t };
}

/** A ring without repeated points, closing duplicates and straight-through vertices. */
export function simplifyRing(ring, tol = EPS) {
  let pts = ring.filter((p, i) => {
    const q = ring[(i + 1) % ring.length];
    return Math.hypot(p[0] - q[0], p[1] - q[1]) > tol;
  });
  for (let changed = true; changed && pts.length > 3; ) {
    changed = false;
    for (let i = 0; i < pts.length && pts.length > 3; i += 1) {
      const a = pts[(i + pts.length - 1) % pts.length], b = pts[i], c = pts[(i + 1) % pts.length];
      if (toSegment(b[0], b[1], a, c).d <= tol) {
        pts.splice(i, 1);
        changed = true;
        i -= 1;
      }
    }
  }
  return pts.map(([x, y]) => [x, y]);
}

/** The ring anticlockwise (outer) or clockwise (hole). */
function oriented(ring, anticlockwise) {
  return ringArea(ring) > 0 === anticlockwise ? ring : ring.slice().reverse();
}

/**
 * Closed outlines from plan segments [ax, ay, bx, by]: the segments used an
 * odd number of times (an edge shared by two faces of one floor cancels),
 * chained into loops. The largest loop is the outer ring; loops inside it are
 * holes. Returns {footprint, holes} or null.
 */
export function outlineFromSegments(segments) {
  const vkey = (x, y) => `${Math.round(x * 1000)},${Math.round(y * 1000)}`;
  // Split each segment at the corners that lie on it (a T-junction: one
  // triangle's corner on another's edge), so shared stretches cancel.
  const corners = new Map();
  for (const s of segments) for (const [x, y] of [[s[0], s[1]], [s[2], s[3]]]) corners.set(vkey(x, y), [x, y]);
  const points = [...corners.values()];
  const split = [];
  for (const s of segments) {
    const dx = s[2] - s[0], dy = s[3] - s[1], len2 = dx * dx + dy * dy;
    if (len2 < 1e-12) continue;
    const cuts = [];
    for (const [x, y] of points) {
      const t = ((x - s[0]) * dx + (y - s[1]) * dy) / len2;
      if (t <= 1e-6 || t >= 1 - 1e-6) continue;
      if (Math.abs((x - s[0]) * dy - (y - s[1]) * dx) / Math.sqrt(len2) < 5e-4) cuts.push(t);
    }
    if (!cuts.length) {
      split.push(s);
      continue;
    }
    cuts.sort((a, b) => a - b);
    let t0 = 0;
    for (const t of [...cuts, 1]) {
      split.push([s[0] + dx * t0, s[1] + dy * t0, s[0] + dx * t, s[1] + dy * t]);
      t0 = t;
    }
  }
  const count = new Map();
  for (const s of split) {
    const ka = vkey(s[0], s[1]), kb = vkey(s[2], s[3]);
    if (ka === kb) continue;
    const k = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
    const entry = count.get(k);
    if (entry) entry.n += 1;
    else count.set(k, { n: 1, a: [s[0], s[1]], b: [s[2], s[3]], ka, kb });
  }
  const edges = [...count.values()].filter((e) => e.n % 2 === 1);
  const at = new Map();
  edges.forEach((e, i) => {
    for (const k of [e.ka, e.kb]) {
      if (!at.has(k)) at.set(k, []);
      at.get(k).push(i);
    }
  });
  const used = new Uint8Array(edges.length);
  const loops = [];
  for (let s = 0; s < edges.length; s += 1) {
    if (used[s]) continue;
    used[s] = 1;
    const start = edges[s].ka;
    const loop = [edges[s].a];
    let key = edges[s].kb, point = edges[s].b;
    for (let guard = 0; key !== start && guard <= edges.length; guard += 1) {
      loop.push(point);
      const next = (at.get(key) || []).find((i) => !used[i]);
      if (next === undefined) break;
      used[next] = 1;
      const e = edges[next];
      [key, point] = e.ka === key ? [e.kb, e.b] : [e.ka, e.a];
    }
    if (key === start && loop.length >= 3) loops.push(simplifyRing(loop));
  }
  const rings = loops.filter((r) => r.length >= 3 && Math.abs(ringArea(r)) > 1e-6);
  if (!rings.length) return null;
  rings.sort((p, q) => Math.abs(ringArea(q)) - Math.abs(ringArea(p)));
  const footprint = oriented(rings[0], true);
  const holes = rings.slice(1).filter((r) => pointInRing(r[0][0], r[0][1], footprint)).map((r) => oriented(r, false));
  return { footprint, holes };
}

/** Plan area of a footprint less its holes. */
const roomArea = (footprint, holes = []) => Math.abs(ringArea(footprint)) - holes.reduce((s, h) => s + Math.abs(ringArea(h)), 0);

/** A point inside the room: its centroid if that is on the floor, else a probe near it. */
export function roomPoint(room) {
  const ring = room.footprint;
  let cx = 0, cy = 0, a = 0;
  for (let i = 0; i < ring.length; i += 1) {
    const [x0, y0] = ring[i], [x1, y1] = ring[(i + 1) % ring.length];
    const f = x0 * y1 - x1 * y0;
    cx += (x0 + x1) * f;
    cy += (y0 + y1) * f;
    a += f;
  }
  if (Math.abs(a) > 1e-9) {
    cx /= 3 * a;
    cy /= 3 * a;
    if (pointInRoom(cx, cy, room)) return [cx, cy];
  }
  // An L or U: the midpoint of the longest chord across a vertical line through each vertex.
  let best = null, bestLen = -1;
  for (const [vx] of ring) {
    for (const dx of [0.05, -0.05]) {
      const x = vx + dx;
      const ys = [];
      for (const r of [ring, ...(room.holes || [])]) {
        for (let i = 0; i < r.length; i += 1) {
          const [x0, y0] = r[i], [x1, y1] = r[(i + 1) % r.length];
          if (x0 > x !== x1 > x) ys.push(y0 + ((x - x0) * (y1 - y0)) / (x1 - x0));
        }
      }
      ys.sort((p, q) => p - q);
      for (let k = 0; k + 1 < ys.length; k += 2) {
        if (ys[k + 1] - ys[k] > bestLen) {
          bestLen = ys[k + 1] - ys[k];
          best = [x, (ys[k] + ys[k + 1]) / 2];
        }
      }
    }
  }
  return best || ring[0];
}

/**
 * Triangles of a polygon with holes, by ear clipping after bridging each hole
 * into the outer ring at a vertex it can see. Returns indices into the flat
 * list of the outer ring's points followed by each hole's, anticlockwise.
 * LOD1 rooms have tens of vertices, so the quadratic search is no cost.
 */
export function triangulate(outer, holes = []) {
  const pts = [...oriented(outer, true), ...holes.flatMap((h) => oriented(h, false))];
  let ring = oriented(outer, true).map((_, i) => i);
  let offset = outer.length;
  const segsCross = (a, b, c, d) => {
    const o = (p, q, r) => Math.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]));
    return o(a, b, c) * o(a, b, d) < 0 && o(c, d, a) * o(c, d, b) < 0;
  };
  const holeIdx = holes.map((h) => {
    const idx = h.map((_, i) => offset + i);
    offset += h.length;
    return idx;
  });
  // Holes with the rightmost vertex first, as usual for bridging.
  holeIdx.sort((p, q) => Math.max(...q.map((i) => pts[i][0])) - Math.max(...p.map((i) => pts[i][0])));
  for (const h of holeIdx) {
    let m = h[0];
    for (const i of h) if (pts[i][0] > pts[m][0]) m = i;
    const others = [ring, ...holeIdx.filter((g) => g !== h)];
    let best = -1, bestD = Infinity;
    for (const p of ring) {
      const d = Math.hypot(pts[p][0] - pts[m][0], pts[p][1] - pts[m][1]);
      if (d >= bestD) continue;
      let blocked = false;
      for (const r of [...others, h]) {
        for (let k = 0; k < r.length && !blocked; k += 1) {
          const a = r[k], b = r[(k + 1) % r.length];
          if (a === p || b === p || a === m || b === m) continue;
          if (segsCross(pts[m], pts[p], pts[a], pts[b])) blocked = true;
        }
      }
      if (!blocked) [best, bestD] = [p, d];
    }
    if (best < 0) continue;
    const at = ring.indexOf(best);
    const start = h.indexOf(m);
    const loop = [...h.slice(start), ...h.slice(0, start), m];
    ring = [...ring.slice(0, at + 1), ...loop, best, ...ring.slice(at + 1)];
  }
  const tris = [];
  const cross = (a, b, c) => (pts[b][0] - pts[a][0]) * (pts[c][1] - pts[a][1]) - (pts[b][1] - pts[a][1]) * (pts[c][0] - pts[a][0]);
  const inTri = (p, a, b, c) => {
    const P = pts[p];
    if ([a, b, c].some((q) => pts[q][0] === P[0] && pts[q][1] === P[1])) return false;
    return cross(a, b, p) >= 0 && cross(b, c, p) >= 0 && cross(c, a, p) >= 0;
  };
  let guard = ring.length * ring.length + 16;
  while (ring.length > 3 && guard-- > 0) {
    let clipped = false;
    for (let i = 0; i < ring.length; i += 1) {
      const a = ring[(i + ring.length - 1) % ring.length], b = ring[i], c = ring[(i + 1) % ring.length];
      const turn = cross(a, b, c);
      if (Math.abs(turn) < 1e-12) {
        // A straight-through or doubled-back vertex: drop it, no triangle.
        ring.splice(i, 1);
        clipped = true;
        break;
      }
      if (turn < 0) continue;
      let ear = true;
      for (const p of ring) {
        if (p !== a && p !== b && p !== c && inTri(p, a, b, c)) {
          ear = false;
          break;
        }
      }
      if (!ear) continue;
      tris.push(a, b, c);
      ring.splice(i, 1);
      clipped = true;
      break;
    }
    if (!clipped) {
      // Rounding left no clean ear: clip the most convex corner.
      let pick = 0, most = -Infinity;
      for (let i = 0; i < ring.length; i += 1) {
        const t = cross(ring[(i + ring.length - 1) % ring.length], ring[i], ring[(i + 1) % ring.length]);
        if (t > most) [pick, most] = [i, t];
      }
      tris.push(ring[(pick + ring.length - 1) % ring.length], ring[pick], ring[(pick + 1) % ring.length]);
      ring.splice(pick, 1);
    }
  }
  if (ring.length === 3 && Math.abs(cross(ring[0], ring[1], ring[2])) > 1e-12) tris.push(...ring);
  return { points: pts, indices: tris };
}

// ------------------------------------------------------------- the model

/** A room class from a name, in the corpus's languages (cores.js for stairs and lifts). */
export function classFromName(name = "") {
  const kind = coreKind({ type: "IFCSPACE", name });
  if (kind === CORE_STAIR) return "Stair";
  if (kind === CORE_LIFT || /\belev\b/i.test(name)) return "Shaft";
  if (/(corridor|hall|lobby|foyer|flur|gang|diele|couloir|pasillo|circulation|landing|vestibule|entrance)/i.test(name)) return "Corridor";
  if (/(\bwc\b|toilet|bath|shower|lavatory|restroom|\bbad\b|dusche)/i.test(name)) return "Toilet";
  if (/(shaft|schacht|\bduct|\bvoid\b|luftraum|riser)/i.test(name)) return "Shaft";
  return "Unit";
}

/** Whether people walk in a room of this class. */
export const isWalkableClass = (cls) => !NOT_WALKABLE.test(String(cls || "").trim());

/** Levels from rooms when a source lists none: one per level id, lowest floor first. */
function levelsFromRooms(rooms) {
  const byId = new Map();
  for (const r of rooms) {
    const l = byId.get(r.level);
    if (!l) byId.set(r.level, { id: r.level, name: r.levelName ?? String(r.level), ffl: r.ffl, height: r.height });
    else {
      l.ffl = Math.min(l.ffl, r.ffl);
      l.height = Math.max(l.height, r.height);
    }
  }
  return [...byId.values()].sort((a, b) => a.ffl - b.ffl);
}

/** Counts for meta, the same for every source. */
function countsOf(model) {
  const kinds = {};
  for (const d of model.doors) kinds[d.kind] = (kinds[d.kind] || 0) + 1;
  for (const l of model.links) kinds[l.kind] = (kinds[l.kind] || 0) + 1;
  return { levels: model.levels.length, rooms: model.rooms.length, doors: model.doors.length, links: model.links.length, kinds };
}

/**
 * The shared stretches of two rooms' outlines: collinear, overlapping edges
 * within `tol` of each other. Each as {a, b, length} in plan.
 */
export function sharedBoundary(roomA, roomB, tol = 0.02) {
  const out = [];
  const ringsA = [roomA.footprint, ...(roomA.holes || [])], ringsB = [roomB.footprint, ...(roomB.holes || [])];
  for (const ra of ringsA) {
    for (let i = 0; i < ra.length; i += 1) {
      const a0 = ra[i], a1 = ra[(i + 1) % ra.length];
      const len = Math.hypot(a1[0] - a0[0], a1[1] - a0[1]);
      if (len < EPS) continue;
      const ux = (a1[0] - a0[0]) / len, uy = (a1[1] - a0[1]) / len;
      for (const rb of ringsB) {
        for (let j = 0; j < rb.length; j += 1) {
          const b0 = rb[j], b1 = rb[(j + 1) % rb.length];
          // Both ends of b on a's line, then the overlap along it.
          const off0 = -(b0[0] - a0[0]) * uy + (b0[1] - a0[1]) * ux;
          const off1 = -(b1[0] - a0[0]) * uy + (b1[1] - a0[1]) * ux;
          if (Math.abs(off0) > tol || Math.abs(off1) > tol) continue;
          const t0 = (b0[0] - a0[0]) * ux + (b0[1] - a0[1]) * uy;
          const t1 = (b1[0] - a0[0]) * ux + (b1[1] - a0[1]) * uy;
          const lo = Math.max(0, Math.min(t0, t1)), hi = Math.min(len, Math.max(t0, t1));
          if (hi - lo < 0.05) continue;
          out.push({ a: [a0[0] + ux * lo, a0[1] + uy * lo], b: [a0[0] + ux * hi, a0[1] + uy * hi], length: hi - lo });
        }
      }
    }
  }
  return out;
}

/**
 * Where a door between two rooms (or a room and the outside) stands: the
 * point on their shared boundary nearest `near` (or the middle of the longest
 * stretch, without `near`), kept half its width and 10 cm from the stretch's
 * ends so the opening clears the walls across them; else (rooms that do not
 * touch, or one room) on the nearer room's outline. Returns {at: [x, y],
 * direction, touching} or null; direction is the stretch's, touching false
 * when two rooms were given that share no boundary.
 */
export function doorOnBoundary(roomA, roomB, near = null, width = 0.9, levelRooms = null) {
  const pick = (segments) => {
    let best = null, bestD = Infinity;
    for (const { a, b } of segments) {
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (len < 1e-9) continue;
      const direction = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
      const margin = Math.min((width > 0 ? width : 0.9) / 2 + 0.1, len / 2) / len;
      let t = 0.5, d = -len;
      if (near) ({ d, t } = toSegment(near[0], near[1], a, b));
      t = Math.max(margin, Math.min(1 - margin, t));
      if (d < bestD) [best, bestD] = [{ at: [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])], direction }, d];
    }
    return best;
  };
  if (roomA && roomB) {
    const shared = sharedBoundary(roomA, roomB);
    if (shared.length) return { ...pick(shared), touching: true };
  }
  if (!roomA && !roomB) return near ? { at: near, direction: null, touching: false } : null;
  // A door to the outside stands on the room's outside wall: the stretches of
  // its outline no other room of its floor shares (a door drawn at a corner
  // would otherwise as soon land on the wall to the next room).
  if (!(roomA && roomB) && levelRooms) {
    const outside = outsideStretches(roomA || roomB, levelRooms);
    if (outside.length) return { ...pick(outside), touching: true };
  }
  const edges = [];
  for (const room of [roomA, roomB]) {
    if (!room) continue;
    for (const r of [room.footprint, ...(room.holes || [])]) for (let i = 0; i < r.length; i += 1) edges.push({ a: r[i], b: r[(i + 1) % r.length] });
  }
  const placed = pick(edges);
  return placed && { ...placed, touching: !(roomA && roomB) };
}

/** The stretches {a, b} of a room's outline that no other of `rooms` at its floor level runs along. */
function outsideStretches(room, rooms) {
  const out = [];
  const others = rooms.filter((r) => r !== room && Math.abs(r.ffl - room.ffl) < 0.05);
  for (const ring of [room.footprint, ...(room.holes || [])]) {
    for (let i = 0; i < ring.length; i += 1) {
      const a = ring[i], b = ring[(i + 1) % ring.length];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (len < 0.05) continue;
      let parts = [[0, len]];
      for (const s of others.flatMap((o) => sharedBoundary({ footprint: [a, b], holes: [] }, o))) {
        const t0 = Math.hypot(s.a[0] - a[0], s.a[1] - a[1]), t1 = Math.hypot(s.b[0] - a[0], s.b[1] - a[1]);
        const [lo, hi] = t0 < t1 ? [t0, t1] : [t1, t0];
        parts = parts.flatMap(([p, q]) => [[p, Math.min(q, lo)], [Math.max(p, hi), q]]).filter(([p, q]) => q - p > 0.05);
      }
      const ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len;
      for (const [p, q] of parts) out.push({ a: [a[0] + ux * p, a[1] + uy * p], b: [a[0] + ux * q, a[1] + uy * q] });
    }
  }
  return out;
}

// ------------------------------------------------------- TopologicPy JSON

/**
 * The model from a TopologicPy CellComplex (Topology.ExportToJSON: Vertex,
 * Edge, Wire, Face, Shell and Cell entries linked by uuid, one CellComplex)
 * and its graphs (Graph.ExportToJSON: {properties, vertices, edges}).
 *
 * Rooms are the Cells. A Cell's footprint is the outline of its lowest
 * horizontal faces (a floor shared with several rooms below is split into
 * several faces, whose common edges cancel); its floor level and height are
 * the Cell's lowest and highest z; name, class and level come from its
 * dictionary (id, name, level, level_name, ffl, height, space_class).
 *
 * Doors come from the circulation graph: each opening vertex (door,
 * exterior door, open, open boundary, assumed access, stair, lift, ramp ...)
 * joins the two vertices its edges reach. Two rooms at one level make a door
 * (an "opening" for open and open boundary), set on their shared boundary at
 * the point nearest the drawn one; a room and a street vertex make an exit; a
 * room and another outdoor vertex make a door to the outside. Rooms at
 * different levels make a link of the opening's kind (stair, lift, ramp).
 * Edges marked walkable 0 (service lifts, openings to voids) are left out.
 * The adjacency graph is a cross-check only: how many of the rooms it says
 * share a wall share a stretch of outline here.
 *
 * @param {object[]|string} cellComplexJson  the exported array (or its text)
 * @param {{circulation?: object, adjacency?: object}} graphs
 */
export function parseTopologicLOD1(cellComplexJson, { circulation = null, adjacency = null } = {}) {
  const records = typeof cellComplexJson === "string" ? JSON.parse(cellComplexJson) : cellComplexJson;
  if (!Array.isArray(records)) throw new Error("A TopologicPy JSON export is an array of topologies.");
  const byId = new Map(records.map((r) => [r.uuid, r]));
  const warnings = [];
  const xyz = (uuid) => byId.get(uuid)?.coordinates;
  // A wire's points in order: its edges chained end to end.
  const wirePoints = (uuid) => {
    const edges = (byId.get(uuid)?.edges || []).map((e) => byId.get(e)?.vertices).filter((v) => v?.length === 2);
    if (!edges.length) return [];
    const used = new Uint8Array(edges.length);
    used[0] = 1;
    const seq = [edges[0][0], edges[0][1]];
    for (let k = 1; k < edges.length; k += 1) {
      const last = seq[seq.length - 1];
      const i = edges.findIndex((e, j) => !used[j] && (e[0] === last || e[1] === last));
      if (i < 0) break;
      used[i] = 1;
      seq.push(edges[i][0] === last ? edges[i][1] : edges[i][0]);
    }
    if (seq.length > 1 && seq[seq.length - 1] === seq[0]) seq.pop();
    return seq.map(xyz).filter(Boolean);
  };

  const complex = records.find((r) => r.type === "CellComplex");
  const cells = complex ? complex.cells.map((u) => byId.get(u)).filter(Boolean) : records.filter((r) => r.type === "Cell");
  const rooms = [];
  cells.forEach((cell, i) => {
    const d = cell.dictionary || {};
    const faces = (cell.shells || []).flatMap((s) => byId.get(s)?.faces || []).map((f) => byId.get(f)).filter(Boolean);
    const rings = faces.map((f) => (f.wires || []).map(wirePoints));
    let zmin = Infinity, zmax = -Infinity;
    for (const face of rings) for (const ring of face) for (const p of ring) {
      zmin = Math.min(zmin, p[2]);
      zmax = Math.max(zmax, p[2]);
    }
    const segments = [];
    for (const face of rings) {
      if (!face.length || !face[0].every((p) => Math.abs(p[2] - zmin) < EPS)) continue;
      for (const ring of face) for (let k = 0; k < ring.length; k += 1) {
        const a = ring[k], b = ring[(k + 1) % ring.length];
        segments.push([a[0], a[1], b[0], b[1]]);
      }
    }
    const outline = outlineFromSegments(segments);
    const id = String(d.id ?? d.TOPOLOGIC_name ?? `cell-${i + 1}`);
    if (!outline || !Number.isFinite(zmin)) {
      warnings.push(`Cell ${id}: no floor outline; left out.`);
      return;
    }
    const area = roomArea(outline.footprint, outline.holes);
    if (Number.isFinite(d.area) && Math.abs(area - d.area) > Math.max(0.05 * d.area, 0.1)) {
      warnings.push(`Cell ${id}: floor outline ${area.toFixed(2)} m2, dictionary ${d.area} m2.`);
    }
    const cls = d.space_class ?? classFromName(d.name);
    rooms.push({
      id,
      name: d.name ?? id,
      level: d.level ?? null,
      levelName: d.level_name ?? null,
      ffl: zmin,
      height: zmax - zmin,
      class: cls,
      area,
      footprint: outline.footprint,
      holes: outline.holes,
      walkable: isWalkableClass(cls),
    });
  });
  // Rooms without a level id: by floor level.
  for (const r of rooms) if (r.level === null) r.level = `L${r.ffl.toFixed(2)}`;
  const levels = levelsFromRooms(rooms);
  for (const r of rooms) delete r.levelName;

  const model = { levels, rooms, doors: [], links: [], meta: {} };
  if (circulation) readCirculation(model, circulation, warnings);
  model.meta = {
    source: "topologicpy",
    name: complex?.dictionary?.name ?? null,
    north: complex?.dictionary?.north ?? null,
    boundary: "shared",
    counts: countsOf(model),
    warnings,
  };
  if (adjacency) model.meta.adjacency = checkAdjacency(model, adjacency);
  return model;
}

/** Vertices and edges of TopologicPy graph JSON, keyed or listed. */
function graphParts(graph) {
  const vertices = Array.isArray(graph?.vertices) ? graph.vertices : Object.values(graph?.vertices || {});
  const edges = Array.isArray(graph?.edges) ? graph.edges : Object.values(graph?.edges || {});
  const keys = Array.isArray(graph?.vertices) ? null : Object.keys(graph?.vertices || {});
  const index = new Map(keys ? keys.map((k, i) => [k, i]) : []);
  const ends = (e) => [e.src ?? index.get(e.source), e.dst ?? index.get(e.target)];
  return { vertices, edges, ends };
}

/** Room index of each graph vertex: by id, else by position (in the footprint, between floor and ceiling). */
function vertexRooms(model, vertices) {
  const byId = new Map(model.rooms.map((r, i) => [r.id, i]));
  return vertices.map((v) => {
    if (v.node_type && v.node_type !== "space") return -1;
    const id = v.id ?? v.TOPOLOGIC_name;
    if (id !== undefined && byId.has(String(id))) return byId.get(String(id));
    if (!Number.isFinite(v.x) || !Number.isFinite(v.y)) return -1;
    const z = Number.isFinite(v.z) ? v.z : null;
    return model.rooms.findIndex((r) => pointInRoom(v.x, v.y, r) && (z === null || (z >= r.ffl - 0.1 && z <= r.ffl + r.height + 0.1)));
  });
}

/** Doors and links from a circulation graph (see parseTopologicLOD1). */
function readCirculation(model, graph, warnings) {
  const { vertices, edges, ends } = graphParts(graph);
  const roomOf = vertexRooms(model, vertices);
  const outside = (v) => v && (v.space_class === "Exterior" || v.node_type === "street" || v.node_type === "outdoor");
  const isStreet = (v) => v && (v.node_type === "street" || /street/i.test(v.name || ""));
  const incident = vertices.map(() => []);
  edges.forEach((e, k) => {
    const [a, b] = ends(e);
    if (a === undefined || b === undefined || !vertices[a] || !vertices[b]) return;
    incident[a].push([b, k]);
    incident[b].push([a, k]);
  });
  const walkable = (e) => e.walkable === undefined || Number(e.walkable) !== 0;
  let n = 0;
  const add = (type, a, b, v, e, width) => {
    const ra = a >= 0 ? model.rooms[a] : null, rb = b >= 0 ? model.rooms[b] : null;
    const t = String(type || "door").toLowerCase();
    if (ra && rb && Math.abs(ra.ffl - rb.ffl) >= 1.0) {
      const [lo, hi] = ra.ffl <= rb.ffl ? [ra, rb] : [rb, ra];
      const kind = /lift|elevator/.test(t) ? "lift" : /ramp/.test(t) ? "ramp" : "stair";
      model.links.push({
        id: v?.id ?? `link-${model.links.length + 1}`,
        kind,
        rooms: [lo.id, hi.id],
        position: v && Number.isFinite(v.x) ? [v.x, v.y] : null,
        width: Number.isFinite(width) ? width : null,
        footprint: null,
      });
      return;
    }
    if (ra && rb && Math.abs(ra.ffl - rb.ffl) > 0.3) warnings.push(`Opening ${v?.id ?? "?"}: rooms ${(rb.ffl - ra.ffl).toFixed(2)} m apart in height.`);
    const street = [a, b].some((x, s) => x < 0 && isStreet(s ? e.bv : e.av));
    const open = /^open/.test(t);
    const kind = street ? "exit" : open ? "opening" : "door";
    const room = ra || rb;
    if (!room) return;
    const drawn = v && Number.isFinite(v.x) ? [v.x, v.y] : null;
    const placed = doorOnBoundary(ra, rb, drawn, width, model.rooms);
    if (!placed) return;
    if (!placed.touching) warnings.push(`Opening ${v?.id ?? "?"}: its rooms do not touch; placed on the nearer one's wall.`);
    model.doors.push({
      id: v?.id ?? `door-${(n += 1)}`,
      position: [placed.at[0], placed.at[1], Math.min(ra?.ffl ?? Infinity, rb?.ffl ?? Infinity)],
      direction: placed.direction,
      level: room.level,
      kind,
      rooms: [ra?.id ?? null, rb?.id ?? null],
      width: Number.isFinite(width) && width > 0 ? width : null,
      assumed: /assumed/.test(t) || Number(v?.verify) === 1,
      type: t,
    });
  };
  vertices.forEach((v, i) => {
    if (roomOf[i] >= 0 || outside(v) || !v) return;
    // An opening: its walkable edges' far ends, rooms first.
    const far = incident[i].filter(([, k]) => walkable(edges[k]));
    if (far.length < 2) return;
    const type = v.node_type && v.node_type !== "space" ? v.node_type : edges[far[0][1]].type;
    const width = Number(v.width);
    const sides = far.map(([j]) => j).sort((p, q) => (roomOf[q] >= 0) - (roomOf[p] >= 0));
    for (let s = 1; s < sides.length; s += 1) {
      const a = sides[0], b = sides[s];
      add(type, roomOf[a], roomOf[b], v, { av: vertices[a], bv: vertices[b] }, width);
    }
  });
  // Rooms joined directly, with no opening vertex between them.
  edges.forEach((e) => {
    if (!walkable(e)) return;
    const [a, b] = ends(e);
    if (roomOf[a] === undefined || roomOf[b] === undefined) return;
    if (roomOf[a] >= 0 && roomOf[b] >= 0) add(e.type, roomOf[a], roomOf[b], null, { av: vertices[a], bv: vertices[b] }, Number(e.width));
    else if ((roomOf[a] >= 0 && outside(vertices[b])) || (roomOf[b] >= 0 && outside(vertices[a]))) {
      add(e.type, roomOf[a], roomOf[b], null, { av: vertices[a], bv: vertices[b] }, Number(e.width));
    }
  });
}

/** How many of the adjacency graph's wall pairs share a stretch of outline in the model. */
function checkAdjacency(model, graph) {
  const { vertices, edges, ends } = graphParts(graph);
  const roomOf = vertexRooms(model, vertices);
  let walls = 0, found = 0, slabs = 0;
  for (const e of edges) {
    const [a, b] = ends(e);
    if (e.shared === "slab") slabs += 1;
    if (e.shared !== "wall" || !(roomOf[a] >= 0) || !(roomOf[b] >= 0)) continue;
    walls += 1;
    if (sharedBoundary(model.rooms[roomOf[a]], model.rooms[roomOf[b]], 0.02).length) found += 1;
  }
  return { wallPairs: walls, wallPairsFound: found, slabPairs: slabs };
}

// ---------------------------------------------------------- dropped files

/**
 * What a JSON file is to a LOD1 import: "cellcomplex" (a TopologicPy export
 * with Cells), "circulation" (a graph with doors, stairs or lifts among its
 * edge types or vertex types), "adjacency" (a graph whose edges say what the
 * rooms share), "geojson" (a FeatureCollection), or null (an energy graph,
 * whose edges are faces and apertures, is none of these).
 */
export function lod1JsonKind(json) {
  if (Array.isArray(json)) return json.some((r) => r?.type === "CellComplex" || r?.type === "Cell") ? "cellcomplex" : null;
  if (json?.type === "FeatureCollection") return "geojson";
  if (json && typeof json === "object" && json.vertices && json.edges) {
    const { vertices, edges } = graphParts(json);
    if (edges.some((e) => e?.shared !== undefined)) return "adjacency";
    const walked = /^(door|exterior door|stair|stair door|lift|lift door|open|open boundary|assumed access|ramp|street|outdoor)$/i;
    if (edges.some((e) => walked.test(String(e?.type ?? ""))) || vertices.some((v) => walked.test(String(v?.node_type ?? "")))) return "circulation";
  }
  return null;
}

/**
 * The model from the JSON files dropped together (parsed): a TopologicPy
 * CellComplex with its circulation and adjacency graphs if they are among
 * them, else a GeoJSON plan.
 */
export function lod1FromJsonFiles(jsons) {
  const kinds = jsons.map(lod1JsonKind);
  const of = (kind) => jsons[kinds.indexOf(kind)] ?? null;
  if (of("cellcomplex")) return parseTopologicLOD1(of("cellcomplex"), { circulation: of("circulation"), adjacency: of("adjacency") });
  if (of("geojson")) return parseGeoJSONPlan(of("geojson"));
  throw new Error("No LOD1 model among these files: drop a TopologicPy CellComplex export (with its graphs) or a GeoJSON plan.");
}

// ------------------------------------------------------- GeoJSON plans

/**
 * Studio's GeoJSON plan: a neutral format for plans traced in other tools
 * (QGIS, a CAD export, a script). One FeatureCollection; coordinates are plan
 * metres in the building's own frame (x and y as drawn, the same axes as the
 * IFC's), not longitude and latitude.
 *
 *   FeatureCollection.properties (all optional):
 *     name, north ([x, y] or degrees), units ("m"),
 *     levels: [{id, name, ffl (m), height (m)}],
 *     boundary: "shared" (default: rooms meet on the walls' centre lines)
 *               or "inner" (rooms end at the walls' faces, as IfcSpaces do)
 *
 *   Rooms: Polygon features (a MultiPolygon is one room per part).
 *     properties: {id, name, level, ffl (m), height (m), class}
 *     level is required unless the plan has one level; ffl and height
 *     default to the level's; class defaults from the name (Stair, Corridor,
 *     Toilet, Shaft, Unit). Shaft, Void and Tank rooms are not walked in.
 *
 *   Doors and openings: Point or LineString features.
 *     properties: {level, kind: "door" | "exit" | "opening", width (m), rooms?}
 *     A LineString runs jamb to jamb: its middle is the door and its length
 *     the width when none is given. rooms: [idA, idB], null for the outside;
 *     without it, the rooms whose outlines pass within 0.3 m of the door.
 *     "exit" is a final exit to the street.
 *
 *   Stairs and lifts: Point or LineString features.
 *     properties: {level, kind: "stair" | "lift" | "ramp", toLevel?, levels?, width?, rooms?}
 *     They join the room under the point on `level` to the room under it on
 *     `toLevel` (default: the next level up); `levels` lists every level a
 *     stair or lift serves, bottom to top. A LineString is the walking line
 *     in plan, foot to head; its box (0.6 m round it) is where the stair is
 *     walked. rooms: [lower, upper] for a single pair.
 *
 * @param {object|string} fc  the FeatureCollection (or its text)
 */
export function parseGeoJSONPlan(fc) {
  const data = typeof fc === "string" ? JSON.parse(fc) : fc;
  if (data?.type !== "FeatureCollection" || !Array.isArray(data.features)) throw new Error("A GeoJSON plan is a FeatureCollection.");
  const props = data.properties || {};
  const inner = props.boundary === "inner";
  const warnings = [];
  const declared = new Map();
  for (const l of props.levels || []) {
    if (l?.id === undefined) continue;
    declared.set(String(l.id), { id: String(l.id), name: l.name ?? String(l.id), ffl: Number(l.ffl) || 0, height: Number(l.height) || 3 });
  }
  const only = declared.size === 1 ? [...declared.keys()][0] : null;
  const ring = (coords) => {
    const pts = (coords || []).map((p) => [Number(p[0]), Number(p[1])]).filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
    if (pts.length > 1 && pts[0][0] === pts[pts.length - 1][0] && pts[0][1] === pts[pts.length - 1][1]) pts.pop();
    return simplifyRing(pts);
  };
  const rooms = [];
  const openings = [];
  data.features.forEach((f, i) => {
    const g = f?.geometry, p = f?.properties || {};
    if (!g) return;
    if (g.type === "Polygon" || g.type === "MultiPolygon") {
      const parts = g.type === "Polygon" ? [g.coordinates] : g.coordinates;
      parts.forEach((poly, k) => {
        const outer = ring(poly?.[0]);
        if (outer.length < 3 || Math.abs(ringArea(outer)) < 1e-6) {
          warnings.push(`Feature ${i}: a room with no area; left out.`);
          return;
        }
        const level = p.level !== undefined && p.level !== null ? String(p.level) : only;
        if (level === null) {
          warnings.push(`Feature ${i}: a room with no level; left out.`);
          return;
        }
        const lv = declared.get(level);
        const ffl = p.ffl !== undefined && p.ffl !== null && Number.isFinite(Number(p.ffl)) ? Number(p.ffl) : lv?.ffl;
        const height = Number(p.height) > 0 ? Number(p.height) : lv?.height;
        if (!Number.isFinite(ffl) || !(height > 0)) {
          warnings.push(`Feature ${i}: no floor level or height for level ${level}; left out.`);
          return;
        }
        const footprint = oriented(outer, true);
        const holes = (poly || []).slice(1).map(ring).filter((h) => h.length >= 3).map((h) => oriented(h, false));
        const id = String(p.id ?? `room-${i + 1}`) + (parts.length > 1 ? `#${k + 1}` : "");
        const cls = p.class ?? classFromName(p.name ?? "");
        rooms.push({ id, name: p.name ?? id, level, ffl, height, class: cls, area: roomArea(footprint, holes), footprint, holes, walkable: p.walkable === false ? false : isWalkableClass(cls) });
      });
    } else if (g.type === "Point" || g.type === "LineString") {
      openings.push({ i, g, p });
    }
  });
  const ids = new Set();
  for (const r of rooms) {
    if (ids.has(r.id)) warnings.push(`Room id ${r.id} is used twice.`);
    ids.add(r.id);
  }
  const levels = [...declared.values()];
  for (const l of levelsFromRooms(rooms)) if (!declared.has(l.id)) levels.push(l);
  levels.sort((a, b) => a.ffl - b.ffl);
  const model = { levels, rooms, doors: [], links: [], meta: {} };
  const byId = new Map(rooms.map((r) => [r.id, r]));
  const levelIndex = new Map(levels.map((l, k) => [l.id, k]));
  const roomUnder = (x, y, level) => {
    let best = null;
    for (const r of rooms) if (r.level === level && pointInRoom(x, y, r) && (!best || r.area < best.area)) best = r;
    return best;
  };
  for (const { i, g, p } of openings) {
    const kind = String(p.kind || "door").toLowerCase();
    const pts = g.type === "Point" ? [g.coordinates] : g.coordinates;
    if (!pts?.length || !pts.every((q) => Number.isFinite(Number(q?.[0])) && Number.isFinite(Number(q?.[1])))) {
      warnings.push(`Feature ${i}: no coordinates; left out.`);
      continue;
    }
    const first = [Number(pts[0][0]), Number(pts[0][1])];
    const last = [Number(pts[pts.length - 1][0]), Number(pts[pts.length - 1][1])];
    const mid = [(first[0] + last[0]) / 2, (first[1] + last[1]) / 2];
    const level = p.level !== undefined && p.level !== null ? String(p.level) : only;
    if (level === null || !levelIndex.has(level)) {
      warnings.push(`Feature ${i}: a ${kind} on an unknown level; left out.`);
      continue;
    }
    if (kind === "stair" || kind === "lift" || kind === "ramp") {
      let chain = Array.isArray(p.levels) ? p.levels.map(String) : null;
      if (!chain) {
        const up = p.toLevel !== undefined ? String(p.toLevel) : levels[levelIndex.get(level) + 1]?.id;
        chain = up === undefined ? [level] : [level, up];
      }
      chain = chain.filter((l) => levelIndex.has(l));
      let footprint = null;
      if (g.type === "LineString") {
        const xs = pts.map((q) => Number(q[0])), ys = pts.map((q) => Number(q[1]));
        const x0 = Math.min(...xs) - 0.6, x1 = Math.max(...xs) + 0.6, y0 = Math.min(...ys) - 0.6, y1 = Math.max(...ys) + 0.6;
        footprint = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
      }
      for (let k = 0; k + 1 < chain.length; k += 1) {
        const given = chain.length === 2 && Array.isArray(p.rooms) ? p.rooms.map((r) => byId.get(String(r))) : null;
        const lower = given?.[0] || roomUnder(...first, chain[k]) || roomUnder(...mid, chain[k]);
        const upper = given?.[1] || roomUnder(...last, chain[k + 1]) || roomUnder(...mid, chain[k + 1]);
        if (!lower || !upper) {
          warnings.push(`Feature ${i}: no room under the ${kind} on ${lower ? chain[k + 1] : chain[k]}; that flight is left out.`);
          continue;
        }
        model.links.push({
          id: String(p.id ?? `${kind}-${i + 1}`) + (chain.length > 2 ? `#${k + 1}` : ""),
          kind,
          rooms: [lower.id, upper.id],
          position: mid,
          width: Number(p.width) > 0 ? Number(p.width) : null,
          footprint,
          // One flight drawn as a line is walked along it.
          ...(g.type === "LineString" && chain.length === 2 && kind !== "lift" ? { line: pts.map((q) => [Number(q[0]), Number(q[1])]) } : {}),
        });
      }
      continue;
    }
    // A door or opening: its rooms given, else those whose outline passes near it.
    let pair;
    if (Array.isArray(p.rooms)) pair = [p.rooms[0], p.rooms[1]].map((r) => (r === null || r === undefined ? null : byId.get(String(r)) || null));
    else {
      const near = [];
      for (const r of rooms) {
        if (r.level !== level) continue;
        let d = Infinity;
        for (const rr of [r.footprint, ...r.holes]) for (let k = 0; k < rr.length; k += 1) d = Math.min(d, toSegment(mid[0], mid[1], rr[k], rr[(k + 1) % rr.length]).d);
        if (d <= 0.3) near.push([d, r]);
      }
      near.sort((a, b) => a[0] - b[0]);
      pair = [near[0]?.[1] || null, near[1]?.[1] || null];
    }
    if (!pair[0] && !pair[1]) {
      warnings.push(`Feature ${i}: a ${kind} with no room beside it; left out.`);
      continue;
    }
    const length = g.type === "LineString" ? Math.hypot(last[0] - first[0], last[1] - first[1]) : 0;
    // Set on the rooms' shared wall; where rooms end at the walls' faces
    // ("inner") the door stays where it is drawn, in the wall between them.
    const placed = inner ? null : doorOnBoundary(pair[0], pair[1], mid, Number(p.width) > 0 ? Number(p.width) : length > 0.2 ? length : 0.9, rooms);
    if (placed && !placed.touching) warnings.push(`Feature ${i}: its rooms do not touch; placed on the nearer one's wall.`);
    const at = placed?.at || mid;
    model.doors.push({
      id: String(p.id ?? `door-${i + 1}`),
      position: [at[0], at[1], Math.min(pair[0]?.ffl ?? Infinity, pair[1]?.ffl ?? Infinity)],
      direction: placed?.direction ?? (length > 0.2 ? [(last[0] - first[0]) / length, (last[1] - first[1]) / length] : null),
      level,
      kind: kind === "exit" ? "exit" : kind === "opening" || kind === "open" ? "opening" : "door",
      rooms: [pair[0]?.id ?? null, pair[1]?.id ?? null],
      width: Number(p.width) > 0 ? Number(p.width) : length > 0.2 ? length : null,
      assumed: Boolean(p.assumed),
    });
  }
  model.meta = { source: "geojson", name: props.name ?? null, north: props.north ?? null, boundary: inner ? "inner" : "shared", counts: countsOf(model), warnings };
  return model;
}

const round = (v) => Math.round(v * 1e6) / 1e6;

/**
 * The model as a GeoJSON plan (the schema of parseGeoJSONPlan): rooms as
 * Polygons, doors as Points, stairs and lifts as Points on their lower level.
 * Parsing the result gives the same model back.
 */
export function lod1ToGeoJSON(model) {
  const close = (r) => [...r.map(([x, y]) => [round(x), round(y)]), [round(r[0][0]), round(r[0][1])]];
  const byId = new Map(model.rooms.map((r) => [r.id, r]));
  const features = [];
  for (const r of model.rooms) {
    features.push({
      type: "Feature",
      geometry: { type: "Polygon", coordinates: [close(r.footprint), ...(r.holes || []).map(close)] },
      properties: { id: r.id, name: r.name, level: r.level, ffl: round(r.ffl), height: round(r.height), class: r.class, ...(r.walkable === isWalkableClass(r.class) ? {} : { walkable: r.walkable }) },
    });
  }
  for (const d of model.doors) {
    features.push({
      type: "Feature",
      geometry: { type: "Point", coordinates: [round(d.position[0]), round(d.position[1])] },
      properties: { id: d.id, level: d.level, kind: d.kind, width: d.width ?? null, rooms: d.rooms, ...(d.assumed ? { assumed: true } : {}) },
    });
  }
  for (const l of model.links) {
    const lower = byId.get(l.rooms[0]), upper = byId.get(l.rooms[1]);
    if (!lower || !upper) continue;
    const at = l.position || roomPoint(lower);
    features.push({
      type: "Feature",
      // A walking line as drawn, foot to head; else the link's point.
      geometry: l.line?.length >= 2 ? { type: "LineString", coordinates: l.line.map(([x, y]) => [round(x), round(y)]) } : { type: "Point", coordinates: [round(at[0]), round(at[1])] },
      properties: { id: l.id, level: lower.level, toLevel: upper.level, kind: l.kind, width: l.width ?? null, rooms: l.rooms },
    });
  }
  return {
    type: "FeatureCollection",
    properties: {
      name: model.meta?.name ?? null,
      north: model.meta?.north ?? null,
      units: "m",
      boundary: model.meta?.boundary === "inner" ? "inner" : "shared",
      levels: model.levels.map((l) => ({ id: l.id, name: l.name, ffl: round(l.ffl), height: round(l.height) })),
    },
    features,
  };
}

// ------------------------------------------------------- IfcSpace rooms

/** Plan bounding box [x0, y0, x1, y1] of a ring. */
function bboxOf(ring) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of ring) {
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  return [x0, y0, x1, y1];
}

/** Overlap area of two boxes over the smaller box's area (0 when apart). */
function boxOverlap(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || h <= 0) return 0;
  const area = (r) => Math.max((r[2] - r[0]) * (r[3] - r[1]), 1e-9);
  return (w * h) / Math.min(area(a), area(b));
}

/**
 * The model from an IFC's spaces, doors and storeys alone: the fast path for
 * a large file, which needs none of its walls, slabs or furniture.
 *
 *   Rooms are the IfcSpace volumes, one per item: the footprint is the
 *   outline of the volume's lowest horizontal faces (edges used by one floor
 *   triangle only, as rooms.js measures the perimeter), the floor level and
 *   height its lowest and highest point. A space whose footprint holds two or
 *   more other spaces of its floor is a gross-area space drawn over its rooms,
 *   and is left out. Names are the spaces' LongName or Name (semantics); the
 *   class follows from the name (classFromName).
 *   Levels are the floor levels clustered (a new level where they jump by
 *   more than a metre), named after the IfcBuildingStorey within a metre of
 *   each, after `storeyShift`.
 *   Doors stand where their meshes are (lattice.js doorWaypoints): the room
 *   on each side is the space under a probe 0.15 to 1.2 m across the opening,
 *   within 0.6 m of the sill. A door with the same space on both sides, or
 *   none, is left out. Width and exit flag from ifcSemantics.
 *   Links: stair rooms (and lift shafts) stacked on consecutive levels, over
 *   half the smaller one's box; and each stack of IfcStair meshes, which
 *   joins the space under it at each level it spans to the one under it at
 *   the next.
 *   Walls stand outside the spaces (meta.boundary "inner"): IfcSpace volumes
 *   end at the walls' inner faces. Where two spaces touch there is no wall.
 *
 * @param {object} input  spaces, doors, stairs: meshes {itemId, positions,
 *                        indices} in the frame of `upAxis` (the viewer's, as
 *                        extractEgressGeometry gives them); semantics:
 *                        ifcSemantics output; storeyShift: added to the
 *                        storeys' elevations to bring them into that frame
 *                        (the coordination matrix's vertical term)
 */
export function lod1FromIfcSpaces({ spaces = [], doors = [], stairs = [], semantics = null, storeyShift = 0, upAxis = "y" } = {}) {
  const warnings = [];
  // Mesh frame to IFC axes: the viewer's (x, y, z) is IFC (x, z, -y).
  const ifc = upAxis === "y" ? (p, o) => [p[o], -p[o + 2], p[o + 1]] : (p, o) => [p[o], p[o + 1], p[o + 2]];
  const names = new Map((semantics?.spaces || []).map((s) => [s.id, s.longName || s.name || null]));
  const byItem = new Map();
  spaces.forEach((mesh, i) => {
    if (!mesh?.positions?.length || !mesh?.indices?.length) return;
    const k = mesh.itemId ?? `mesh:${i}`;
    if (!byItem.has(k)) byItem.set(k, []);
    byItem.get(k).push(mesh);
  });
  const minUp = Math.cos((10 * Math.PI) / 180);
  let rooms = [];
  for (const [itemId, meshes] of byItem) {
    let lo = Infinity, hi = -Infinity;
    for (const { positions: p } of meshes) {
      for (let o = 0; o + 2 < p.length; o += 3) {
        const z = ifc(p, o)[2];
        if (z < lo) lo = z;
        if (z > hi) hi = z;
      }
    }
    const segments = [];
    let triArea = 0;
    for (const { positions: p, indices: ix } of meshes) {
      for (let t = 0; t + 2 < ix.length; t += 3) {
        const a = ifc(p, ix[t] * 3), b = ifc(p, ix[t + 1] * 3), c = ifc(p, ix[t + 2] * 3);
        if (Math.max(a[2], b[2], c[2]) > lo + 0.05) continue;
        const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        const nz = e1[0] * e2[1] - e1[1] * e2[0];
        const mag = Math.hypot(e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], nz);
        if (mag < 1e-9 || Math.abs(nz / mag) < minUp) continue;
        triArea += Math.abs(nz) / 2;
        segments.push([a[0], a[1], b[0], b[1]], [b[0], b[1], c[0], c[1]], [c[0], c[1], a[0], a[1]]);
      }
    }
    const outline = segments.length ? outlineFromSegments(segments) : null;
    const id = String(itemId);
    if (!outline || !(hi - lo > 0.5)) {
      warnings.push(`Space ${id}: no floor outline; left out.`);
      continue;
    }
    let area = roomArea(outline.footprint, outline.holes);
    if (Math.abs(area - triArea) > 0.1 * triArea + 0.05) {
      // Faces split along each other's edges leave stray loops: keep the outer one if it alone fits.
      const outer = Math.abs(ringArea(outline.footprint));
      if (Math.abs(outer - triArea) <= 0.1 * triArea + 0.05) {
        outline.holes = [];
        area = outer;
      } else warnings.push(`Space ${id}: outline ${area.toFixed(2)} m2, floor faces ${triArea.toFixed(2)} m2.`);
    }
    const name = names.get(itemId) ?? `Space ${id}`;
    const cls = classFromName(name);
    rooms.push({ id, sourceId: itemId, name, level: null, ffl: lo, height: hi - lo, class: cls, area, footprint: outline.footprint, holes: outline.holes, walkable: isWalkableClass(cls) });
  }

  // Gross-area spaces drawn over their rooms.
  const points = rooms.map(roomPoint);
  const gross = new Set();
  rooms.forEach((r, i) => {
    let inside = 0;
    rooms.forEach((s, j) => {
      if (i !== j && Math.abs(s.ffl - r.ffl) < 0.5 && s.area < r.area && pointInRoom(points[j][0], points[j][1], r)) inside += 1;
    });
    if (inside >= 2) gross.add(r.id);
  });
  if (gross.size) {
    warnings.push(`${gross.size} gross-area space(s) drawn over other spaces left out.`);
    rooms = rooms.filter((r) => !gross.has(r.id));
  }

  // Levels: floor levels clustered, named after the storeys.
  const ffls = [...new Set(rooms.map((r) => Math.round(r.ffl * 1000) / 1000))].sort((a, b) => a - b);
  const clusters = [];
  for (const f of ffls) {
    const last = clusters[clusters.length - 1];
    if (last && f - last[last.length - 1] <= 1.0) last.push(f);
    else clusters.push([f]);
  }
  const storeys = (semantics?.storeys || []).map((s) => ({ name: s.name, z: s.elevation + storeyShift }));
  const levels = clusters.map((c, k) => {
    const ffl = c[0];
    const named = storeys.filter((s) => Math.abs(s.z - ffl) <= 1.0).sort((p, q) => Math.abs(p.z - ffl) - Math.abs(q.z - ffl))[0];
    return { id: `L${k}`, name: named?.name || `Level ${k}`, ffl, height: 0 };
  });
  // (Levels start at floor levels rounded to the millimetre.)
  const levelOf = (z) => {
    let best = 0;
    for (let k = 0; k < levels.length; k += 1) if (levels[k].ffl <= z + 0.001) best = k;
    return best;
  };
  for (const r of rooms) {
    const l = levels[levelOf(r.ffl)];
    r.level = l.id;
    l.height = Math.max(l.height, r.height);
  }

  const model = { levels, rooms, doors: [], links: [], meta: {} };
  const roomAt = (x, y, z) => {
    let best = null;
    for (const r of rooms) if (Math.abs(r.ffl - z) <= 0.6 && pointInRoom(x, y, r) && (!best || r.area < best.area)) best = r;
    return best;
  };

  // Doors: the space on each side of the opening.
  const declared = new Map((semantics?.doors || []).map((d) => [d.id, d]));
  let inside = 0, unplaced = 0;
  for (const [h0, h1, sill, du, dv, measured, itemId] of doorWaypoints(doors, upAxis)) {
    const [x, y] = upAxis === "y" ? [h0, -h1] : [h0, h1];
    const [ux, uy] = upAxis === "y" ? [du, -dv] : [du, dv];
    const [nx, ny] = [-uy, ux];
    const sides = [1, -1].map((s) => {
      for (const k of [0.15, 0.3, 0.5, 0.8, 1.2]) {
        const r = roomAt(x + s * k * nx, y + s * k * ny, sill);
        if (r) return r;
      }
      return null;
    });
    if (!sides[0] && !sides[1]) {
      unplaced += 1;
      continue;
    }
    if (sides[0] === sides[1]) {
      inside += 1;
      continue;
    }
    const info = declared.get(itemId) || {};
    const room = sides[0] || sides[1];
    model.doors.push({
      id: String(itemId ?? `door-${model.doors.length + 1}`),
      sourceId: itemId ?? null,
      position: [x, y, Math.min(sides[0]?.ffl ?? Infinity, sides[1]?.ffl ?? Infinity)],
      direction: [ux, uy],
      level: room.level,
      kind: info.exit ? "exit" : "door",
      rooms: [sides[0]?.id ?? null, sides[1]?.id ?? null],
      width: info.widthM > 0 ? info.widthM : measured > 0.3 ? measured : null,
      assumed: false,
    });
  }
  if (inside) warnings.push(`${inside} door(s) with one space on both sides left out.`);
  if (unplaced) warnings.push(`${unplaced} door(s) with no space beside them left out.`);

  // Links: stacked stair rooms and lift shafts.
  const levelIndex = new Map(levels.map((l, k) => [l.id, k]));
  const boxes = new Map(rooms.map((r) => [r.id, bboxOf(r.footprint)]));
  const joined = new Set();
  const link = (kind, lower, upper, footprint = null, position = null, line = null, width = null) => {
    const key = `${lower.id}|${upper.id}`;
    if (joined.has(key)) return;
    joined.add(key);
    model.links.push({ id: `${kind}-${model.links.length + 1}`, kind, rooms: [lower.id, upper.id], position, width, footprint, ...(line ? { line } : {}) });
  };
  for (const kind of ["stair", "lift"]) {
    const cores = rooms.filter((r) => coreRoom(r, kind));
    for (const a of cores) {
      const k = levelIndex.get(a.level);
      for (const b of cores) {
        if (levelIndex.get(b.level) !== k + 1) continue;
        if (boxOverlap(boxes.get(a.id), boxes.get(b.id)) >= 0.5) link(kind, a, b);
      }
    }
  }
  // Stacks of IfcStair geometry: flights and landings of one stair, their
  // boxes overlapping in plan and meeting in height.
  const stairBoxes = [];
  const stairItems = new Map();
  const stairPoints = new Map();
  for (const mesh of stairs) {
    const k = mesh.itemId ?? `mesh:${stairItems.size}`;
    let b = stairItems.get(k);
    if (!b) {
      stairItems.set(k, (b = [Infinity, Infinity, -Infinity, -Infinity, Infinity, -Infinity]));
      stairPoints.set(k, []);
    }
    const p = mesh.positions, list = stairPoints.get(k);
    for (let o = 0; o + 2 < (p?.length || 0); o += 3) {
      const [x, y, z] = ifc(p, o);
      list.push(x, y, z);
      b[0] = Math.min(b[0], x);
      b[1] = Math.min(b[1], y);
      b[2] = Math.max(b[2], x);
      b[3] = Math.max(b[3], y);
      b[4] = Math.min(b[4], z);
      b[5] = Math.max(b[5], z);
    }
  }
  const boxPoints = [];
  for (const [k, b] of stairItems) {
    if (!Number.isFinite(b[0])) continue;
    stairBoxes.push(b);
    boxPoints.push(stairPoints.get(k));
  }
  const parent = stairBoxes.map((_, i) => i);
  const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  for (let i = 0; i < stairBoxes.length; i += 1) {
    for (let j = i + 1; j < stairBoxes.length; j += 1) {
      const a = stairBoxes[i], b = stairBoxes[j];
      const apart = Math.max(a[0], b[0]) - Math.min(a[2], b[2]) > 0.3 || Math.max(a[1], b[1]) - Math.min(a[3], b[3]) > 0.3;
      if (!apart && Math.max(a[4], b[4]) - Math.min(a[5], b[5]) <= 0.5) parent[find(i)] = find(j);
    }
  }
  const stacks = new Map();
  stairBoxes.forEach((b, i) => {
    const r = find(i);
    const s = stacks.get(r);
    if (!s) stacks.set(r, { box: b.slice(), points: [boxPoints[i]] });
    else {
      // [x0, y0, x1, y1, z0, z1]: the least of the lows, the most of the highs.
      for (const d of [0, 1, 4]) s.box[d] = Math.min(s.box[d], b[d]);
      for (const d of [2, 3, 5]) s.box[d] = Math.max(s.box[d], b[d]);
      s.points.push(boxPoints[i]);
    }
  });
  let unjoined = 0;
  // The room on a level that holds a point, else the one whose outline passes within 1.5 m of it.
  const nearestRoom = (x, y, k) => {
    let best = null, bestD = 1.5;
    for (const r of rooms) {
      if (r.level !== levels[k].id) continue;
      if (pointInRoom(x, y, r)) return r;
      for (const ring of [r.footprint, ...r.holes]) for (let i = 0; i < ring.length; i += 1) {
        const { d } = toSegment(x, y, ring[i], ring[(i + 1) % ring.length]);
        if (d < bestD) [best, bestD] = [r, d];
      }
    }
    return best;
  };
  for (const { box: [x0, y0, x1, y1, z0, z1], points } of stacks.values()) {
    const served = levels.map((l, k) => k).filter((k) => levels[k].ffl >= z0 - 0.6 && levels[k].ffl <= z1 + 0.6);
    // The foot and head: the middles of the stair's lowest and highest vertices.
    const end = (top) => {
      let sx = 0, sy = 0, n = 0;
      for (const list of points) for (let o = 0; o < list.length; o += 3) {
        if (top ? list[o + 2] < z1 - 0.25 : list[o + 2] > z0 + 0.25) continue;
        sx += list[o];
        sy += list[o + 1];
        n += 1;
      }
      return n ? [sx / n, sy / n] : [(x0 + x1) / 2, (y0 + y1) / 2];
    };
    const foot = end(false), head = end(true);
    // The space each level has most of the stair over (a 5 by 5 grid of probes).
    const under = (k) => {
      const votes = new Map();
      for (let i = 0; i < 5; i += 1) {
        for (let j = 0; j < 5; j += 1) {
          const r = roomAt(x0 + ((i + 0.5) / 5) * (x1 - x0), y0 + ((j + 0.5) / 5) * (y1 - y0), levels[k].ffl);
          if (r && r.level === levels[k].id) votes.set(r, (votes.get(r) || 0) + 1);
        }
      }
      return [...votes].sort((p, q) => q[1] - p[1])[0]?.[0] || null;
    };
    for (let s = 0; s + 1 < served.length; s += 1) {
      const single = served.length === 2;
      // A one-storey stair: the rooms at its foot and head (the head often
      // lands beside the space above, which leaves the stair's opening out).
      const lower = (single && nearestRoom(...foot, served[s])) || under(served[s]);
      const upper = (single && nearestRoom(...head, served[s + 1])) || under(served[s + 1]);
      if (!lower || !upper) {
        unjoined += 1;
        continue;
      }
      // Walked straight from foot to head, unless they are too close for its
      // rise (a dog-leg or spiral): then along the assumed dog-leg in its box.
      const straight = single && Math.hypot(head[0] - foot[0], head[1] - foot[1]) >= 0.8 * (upper.ffl - lower.ffl);
      link("stair", lower, upper, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], [(x0 + x1) / 2, (y0 + y1) / 2], straight ? [foot, head] : null, Math.min(x1 - x0, y1 - y0));
    }
  }
  if (unjoined) warnings.push(`${unjoined} stair flight(s) with no space at the foot or head left out.`);

  model.meta = { source: "ifc-spaces", name: null, north: null, boundary: "inner", counts: countsOf(model), warnings };
  return model;
}

// --------------------------------------------------- the egress geometry

/** Sizes the converter assumes where a model gives none (m). */
export const LOD1_DEFAULTS = Object.freeze({ wallThickness: 0.2, slabThickness: 0.2, doorHeight: 2.1, doorWidth: 0.9 });

/** Is a room the vertical core a link of this kind runs in: a stair or ramp room, or a lift shaft. */
function coreRoom(room, kind) {
  const byName = coreKind({ type: "IFCSPACE", name: room.name || "" });
  if (kind === "lift") return byName === CORE_LIFT || /^(lift|elevator)$/i.test(room.class);
  return byName === CORE_STAIR || /^(stair|ramp)/i.test(room.class);
}

/**
 * A room's footprint less other rooms' footprints, as rectangles: the plan
 * cut on every vertex coordinate (in the room's own frame) into a grid, and
 * the cells whose middles are on the room's floor and on none of the others',
 * merged along each row. Exact for rectilinear outlines; null when any edge
 * is not square to the room's longest edge (the caller keeps the whole room).
 */
function slabMinus(room, cutters) {
  const frame = footprintFrame(room.footprint);
  const [ux, uy] = frame.plan(1, 0);
  const toLocal = (ring) => ring.map(([x, y]) => [x * ux + y * uy, -x * uy + y * ux]);
  const rings = [room.footprint, ...(room.holes || []), ...cutters.flatMap((r) => [r.footprint, ...(r.holes || [])])].map(toLocal);
  for (const ring of rings) {
    for (let i = 0; i < ring.length; i += 1) {
      const a = ring[i], b = ring[(i + 1) % ring.length];
      if (Math.abs(a[0] - b[0]) > 1e-3 && Math.abs(a[1] - b[1]) > 1e-3) return null;
    }
  }
  const local = (r) => ({ footprint: toLocal(r.footprint), holes: (r.holes || []).map(toLocal) });
  const own = local(room), cut = cutters.map(local);
  const [x0, y0, x1, y1] = bboxOf(own.footprint);
  const coords = (d, lo, hi) => [...new Set(rings.flat().map((p) => Math.round(p[d] * 1000) / 1000).filter((v) => v >= lo - 1e-6 && v <= hi + 1e-6))].sort((a, b) => a - b);
  const us = coords(0, x0, x1), vs = coords(1, y0, y1);
  const out = [];
  for (let j = 0; j + 1 < vs.length; j += 1) {
    let open = null;
    for (let i = 0; i + 1 <= us.length; i += 1) {
      const keep = i + 1 < us.length && (() => {
        const cu = (us[i] + us[i + 1]) / 2, cv = (vs[j] + vs[j + 1]) / 2;
        return pointInRoom(cu, cv, own) && !cut.some((c) => pointInRoom(cu, cv, c));
      })();
      if (keep && open === null) open = us[i];
      if (!keep && open !== null) {
        const corners = [[open, vs[j]], [us[i], vs[j]], [us[i], vs[j + 1]], [open, vs[j + 1]]];
        out.push(corners.map(([u, v]) => frame.plan(u, v)));
        open = null;
      }
    }
  }
  return out;
}

/**
 * Where a link is walked or ridden, in plan: its own footprint; else the
 * stair room (or lift shaft) at its foot or head; else where its two rooms
 * overlap; else a 2.4 m square at the link.
 */
function linkFootprint(link, lower, upper) {
  if (link.footprint) return link.footprint;
  if (coreRoom(lower, link.kind)) return lower.footprint;
  if (coreRoom(upper, link.kind)) return upper.footprint;
  const [a0, b0, a1, b1] = bboxOf(lower.footprint), [c0, d0, c1, d1] = bboxOf(upper.footprint);
  const x0 = Math.max(a0, c0), y0 = Math.max(b0, d0), x1 = Math.min(a1, c1), y1 = Math.min(b1, d1);
  if (x1 > x0 && y1 > y0 && (x1 - x0) * (y1 - y0) >= 1) return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  const [x, y] = link.position || roomPoint(lower);
  return [[x - 1.2, y - 1.2], [x + 1.2, y - 1.2], [x + 1.2, y + 1.2], [x - 1.2, y + 1.2]];
}

/**
 * A footprint's own frame (u along its longest edge, v across) and, in it,
 * the largest box inside the footprint (a cores.js box {u0, u1, v0, v1}): the
 * footprint itself when it is a rectangle, else the best box whose sides lie
 * on the footprint's vertex coordinates, so a stair in an L-shaped room or
 * one with a curved corner stays inside its walls.
 */
function footprintFrame(footprint) {
  let best = -1, ux = 1, uy = 0;
  for (let i = 0; i < footprint.length; i += 1) {
    const a = footprint[i], b = footprint[(i + 1) % footprint.length];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (len > best) [best, ux, uy] = [len, (b[0] - a[0]) / len, (b[1] - a[1]) / len];
  }
  const local = footprint.map(([x, y]) => [x * ux + y * uy, -x * uy + y * ux]);
  const [u0, v0, u1, v1] = bboxOf(local);
  const plan = (u, v) => [u * ux - v * uy, u * uy + v * ux];
  const us = [...new Set(local.map((p) => Math.round(p[0] * 1000) / 1000))].sort((a, b) => a - b);
  const vs = [...new Set(local.map((p) => Math.round(p[1] * 1000) / 1000))].sort((a, b) => a - b);
  if (Math.abs(ringArea(local)) >= 0.98 * (u1 - u0) * (v1 - v0) || us.length > 24 || vs.length > 24) return { box: { u0, u1, v0, v1 }, plan };
  // A box fits when its middle, its corners and its sides' middles (1 cm in)
  // are inside, and no corner of the footprint is.
  const fits = (a0, a1, b0, b1) => {
    const e = 0.01;
    const probes = [[(a0 + a1) / 2, (b0 + b1) / 2]];
    for (const a of [a0 + e, (a0 + a1) / 2, a1 - e]) for (const b of [b0 + e, (b0 + b1) / 2, b1 - e]) probes.push([a, b]);
    if (!probes.every(([a, b]) => pointInRing(a, b, local))) return false;
    return !local.some(([a, b]) => a > a0 + e && a < a1 - e && b > b0 + e && b < b1 - e);
  };
  let box = null, area = 0;
  for (let i = 0; i < us.length; i += 1) {
    for (let j = i + 1; j < us.length; j += 1) {
      for (let k = 0; k < vs.length; k += 1) {
        for (let l = k + 1; l < vs.length; l += 1) {
          const a = (us[j] - us[i]) * (vs[l] - vs[k]);
          if (a > area && fits(us[i], us[j], vs[k], vs[l])) [box, area] = [{ u0: us[i], u1: us[j], v0: vs[k], v1: vs[l] }, a];
        }
      }
    }
  }
  return { box: box || { u0, u1, v0, v1 }, plan };
}

/** Plan distance from p to the nearest of `points` (0 when there are none). */
const nearestTo = (p, points) => (points.length ? Math.min(...points.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1]))) : 0);

/**
 * A stair between floor levels ha and hb as treads and landings, laid along
 * cores.js's assumed dog-leg walking line in the footprint (walkingLine:
 * flights along the longer side, two lanes a quarter of the width in, or one
 * on the centre line under 1.2 m, risers of about 175 mm on 280 mm goings),
 * in whichever of its reflections starts nearest a door of the room below and
 * ends nearest one of the room above. Each flight is a tread per riser up to
 * the last but one; its top step is a landing, or the floor above. The lattice
 * walks it as any modelled stair (stairs.js): a node per tread or landing,
 * linked up the line, its two ends joined to the floors. Landings are kept
 * under 0.6 m deep, so stairs.js takes them as treads: a landing deep enough
 * to be sampled as floor would merge with a floor less than 0.9 m above or
 * below it (lattice.js columnGap) and cut the line, as it did in a 1.7 m
 * spiral stair's box. Returns boxes [{ring, z0, z1}], 5 cm thick.
 */
function stairSolids(footprint, ha, hb, { foot = [], head = [], walk = null, walkWidth = null } = {}) {
  if (walk?.length >= 2) return straightFlight(walk, ha, hb, walkWidth);
  const frame = footprintFrame(footprint);
  const { u0, u1, v0, v1 } = frame.box;
  let line = null, score = Infinity;
  for (const mirrorAlong of [false, true]) {
    for (const mirrorAcross of [false, true]) {
      const candidate = walkingLine(frame.box, ha, hb, { step: Infinity, mirrorAlong, mirrorAcross });
      const first = frame.plan(candidate[0][0], candidate[0][1]);
      const last = frame.plan(candidate[candidate.length - 1][0], candidate[candidate.length - 1][1]);
      const s = nearestTo(first, foot) + nearestTo(last, head);
      if (s < score - 1e-9) [line, score] = [candidate, s];
    }
  }
  const alongU = u1 - u0 >= v1 - v0;
  const width = alongU ? v1 - v0 : u1 - u0;
  const lane = width >= 1.2 ? width / 2 : width;
  const rect = (P, d, e, s0, s1, c0, c1) => {
    const at = (s, c) => [P[0] + d[0] * s + e[0] * c, P[1] + d[1] * s + e[1] * c];
    return oriented([at(s0, c0), at(s1, c0), at(s1, c1), at(s0, c1)], true);
  };
  // Treads stop 7.5 cm short of each side of their lane, so two stairs side
  // by side (or a stair's two lanes) never touch and become one walking line.
  const half = Math.max(lane / 2 - 0.075, 0.2);
  const out = [];
  let previous = null;
  for (let k = 0; k + 1 < line.length; k += 1) {
    const [pu, pv, ph] = line[k], [qu, qv, qh] = line[k + 1];
    const P = frame.plan(pu, pv), Q = frame.plan(qu, qv);
    const run = Math.hypot(Q[0] - P[0], Q[1] - P[1]);
    if (qh - ph > 1e-6 && run > 1e-6) {
      const d = [(Q[0] - P[0]) / run, (Q[1] - P[1]) / run], e = [-d[1], d[0]];
      const n = Math.max(2, Math.round((qh - ph) / 0.175));
      const going = run / n;
      // A single lane turns on the spot: the next flight comes back over
      // this one's last going, from a turn the width of the lane.
      if (previous?.turn) out.push({ ring: rect(P, previous.d, previous.e, -previous.going, Math.min(0.25, previous.room), -half, half), z0: ph - 0.05, z1: ph });
      // The stair's first tread stands at least 0.22 m up, so the last tread
      // of the stair below this floor and the first of this one are more than
      // stairs.js's 0.36 m apart: each ends at the floor and joins it, rather
      // than the two running on past it as one flight.
      const first = k === 0 ? Math.max(((qh - ph) / n) * 1.3, 0.22) : (qh - ph) / n;
      for (let i = 1; i < n; i += 1) {
        const h = ph + first + ((qh - ph - first) * (i - 1)) / (n - 1);
        out.push({ ring: rect(P, d, e, going * (i - 1), going * i, -half, half), z0: h - 0.05, z1: h });
      }
      // Which way the flight ran along the footprint, and the room left beyond its top.
      const a = alongU ? qu - pu : qv - pv, end = alongU ? qu : qv;
      const next = line[k + 2];
      previous = { d, e, going, room: a > 0 ? (alongU ? u1 : v1) - end : end - (alongU ? u0 : v0), turn: Boolean(next && next[2] - qh > 1e-6) };
    } else if (previous && run > 1e-6) {
      // A landing from one lane to the next, over the arriving flight's last
      // going and 0.25 m beyond it where the footprint has room.
      const e = [(Q[0] - P[0]) / run, (Q[1] - P[1]) / run];
      out.push({ ring: rect(P, previous.d, e, -previous.going, Math.max(0, Math.min(0.25, previous.room)), -half, run + half), z0: ph - 0.05, z1: ph });
      previous = { ...previous, turn: false };
    }
  }
  return out;
}

/**
 * A stair walked along a given line in plan (a polyline, foot to head), as
 * treads of equal going on risers of about 175 mm, by the rules of
 * stairSolids (the floor above is the top step; the first tread at least
 * 0.22 m up). The treads are the stair's width (1 m when not given, at most
 * 1.2 m), less 7.5 cm each side.
 */
function straightFlight(line, ha, hb, width) {
  const legs = [];
  let total = 0;
  for (let i = 0; i + 1 < line.length; i += 1) {
    const a = line[i], b = line[i + 1];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (len < 1e-6) continue;
    legs.push({ a, d: [(b[0] - a[0]) / len, (b[1] - a[1]) / len], s0: total, len });
    total += len;
  }
  if (!legs.length) return [];
  const at = (s) => {
    const leg = legs.find((l) => s <= l.s0 + l.len + 1e-9) || legs[legs.length - 1];
    const t = s - leg.s0;
    return { p: [leg.a[0] + leg.d[0] * t, leg.a[1] + leg.d[1] * t], d: leg.d };
  };
  const half = Math.max(Math.min(width > 0 ? width : 1, 1.2) / 2 - 0.075, 0.2);
  const n = Math.max(2, Math.round((hb - ha) / 0.175));
  const first = Math.max(((hb - ha) / n) * 1.3, 0.22);
  const out = [];
  for (let i = 1; i < n; i += 1) {
    const s0 = (total * (i - 1)) / n, s1 = (total * i) / n;
    const { d } = at((s0 + s1) / 2);
    const P = at(s0).p, Q = at(s1).p;
    const e = [-d[1], d[0]];
    const ring = [[P[0] - e[0] * half, P[1] - e[1] * half], [Q[0] - e[0] * half, Q[1] - e[1] * half], [Q[0] + e[0] * half, Q[1] + e[1] * half], [P[0] + e[0] * half, P[1] + e[1] * half]];
    const h = ha + first + ((hb - ha - first) * (i - 1)) / (n - 1);
    out.push({ ring: oriented(ring, true), z0: h - 0.05, z1: h });
  }
  return out;
}

/**
 * A ramp between floor levels ha and hb as one sloping slab along the
 * footprint's longer side (0.3 m in from each end and side), its foot at the
 * end nearer a door of the room below, or null when that would be steeper
 * than 12 degrees (it is then walked as a stair). The lattice samples a slope
 * as floor (stairs.js). Returns the slab's top corners [x, y, z], anticlockwise.
 */
function rampSolid(footprint, ha, hb, { foot = [], head = [] } = {}) {
  const frame = footprintFrame(footprint);
  const { u0, u1, v0, v1 } = frame.box;
  const alongU = u1 - u0 >= v1 - v0;
  const [a0, a1] = alongU ? [u0 + 0.3, u1 - 0.3] : [v0 + 0.3, v1 - 0.3];
  const [c0, c1] = alongU ? [v0 + 0.3, v1 - 0.3] : [u0 + 0.3, u1 - 0.3];
  if (a1 - a0 < 1 || c1 - c0 < 0.6 || Math.atan2(hb - ha, a1 - a0) > (12 * Math.PI) / 180) return null;
  const at = (a, c) => (alongU ? frame.plan(a, c) : frame.plan(c, a));
  // The foot at the lower room's door: the floor under the low end of a ramp
  // is the ramp itself, so a foot away from the door could not be reached.
  // The head follows the upper room's doors only when that settles nothing.
  const mid = (c0 + c1) / 2;
  const footGain = nearestTo(at(a0, mid), foot) - nearestTo(at(a1, mid), foot);
  const flip = Math.abs(footGain) > 0.5 ? footGain > 0 : nearestTo(at(a0, mid), head) < nearestTo(at(a1, mid), head);
  const z = (a) => (flip ? hb + ((ha - hb) * (a - a0)) / (a1 - a0) : ha + ((hb - ha) * (a - a0)) / (a1 - a0));
  const corners = [[a0, c0], [a1, c0], [a1, c1], [a0, c1]].map(([a, c]) => [...at(a, c), z(a)]);
  return { top: ringArea(corners) > 0 ? corners : corners.reverse() };
}

/**
 * The walls of a model as boxes along its rooms' outlines, level by level
 * (rooms grouped by floor level). Each piece: plan ends a, b; the unit normal
 * n and the offsets o0 < o1 along it that give the box its thickness; z0, z1.
 *
 * "shared" outlines (rooms meet on one line): collinear edges of one floor
 * level are merged, so a wall two rooms share is one box, centred on the line
 * (two coincident boxes would read as open in the lattice's inside test,
 * which counts crossings). Where the rooms on either side differ in height
 * or floor level the box takes the taller span. "inner" outlines (IfcSpace volumes, which end at
 * the walls' inner faces): each edge stands on the outside of its room,
 * except where another room's edge runs along it within 3 cm: spaces that
 * touch have no wall between them.
 */
function wallPieces(model, { boundary, thickness, slab }) {
  const groups = new Map();
  for (const room of model.rooms) {
    // By level: rooms of one storey whose floors differ by millimetres (as
    // IfcSpace volumes often do) still share their walls, or touch.
    const key = room.level ?? Math.round(room.ffl * 100);
    if (!groups.has(key)) groups.set(key, []);
    const edges = groups.get(key);
    for (const ring of [room.footprint, ...(room.holes || [])]) {
      for (let i = 0; i < ring.length; i += 1) {
        const a = ring[i], b = ring[(i + 1) % ring.length];
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        if (len < 0.01) continue;
        let th = Math.atan2(b[1] - a[1], b[0] - a[0]);
        if (th < 0) th += Math.PI;
        if (th >= Math.PI - 1e-4) th -= Math.PI;
        const ux = Math.cos(th), uy = Math.sin(th);
        const ta = ux * a[0] + uy * a[1], tb = ux * b[0] + uy * b[1];
        edges.push({
          a, b, th, ux, uy,
          c: -uy * a[0] + ux * a[1],
          lo: Math.min(ta, tb),
          hi: Math.max(ta, tb),
          z0: room.ffl - slab,
          z1: room.ffl + room.height,
          ffl: room.ffl,
          room,
          // Outward: to the right of travel (outer rings run anticlockwise, holes clockwise).
          out: [(b[1] - a[1]) / len, -(b[0] - a[0]) / len],
        });
      }
    }
  }
  const pieces = [];
  const tolC = boundary === "inner" ? 0.03 : 0.01;
  for (const edges of groups.values()) {
    edges.sort((p, q) => p.th - q.th);
    const lines = [];
    for (let s = 0; s < edges.length; ) {
      let e = s + 1;
      while (e < edges.length && edges[e].th - edges[e - 1].th < 1e-3) e += 1;
      const run = edges.slice(s, e).sort((p, q) => p.c - q.c);
      for (let i = 0; i < run.length; ) {
        let j = i + 1;
        while (j < run.length && run[j].c - run[j - 1].c < tolC) j += 1;
        lines.push(run.slice(i, j));
        i = j;
      }
      s = e;
    }
    for (const line of lines) {
      const { ux, uy, c } = line[0];
      const at = (t, cc = c) => [t * ux - cc * uy, t * uy + cc * ux];
      if (boundary === "inner") {
        for (const m of line) {
          // This edge less the stretches other rooms' edges run along.
          let parts = [[m.lo, m.hi]];
          for (const o of line) {
            if (o.room === m.room || o.hi <= m.lo + 1e-3 || o.lo >= m.hi - 1e-3) continue;
            parts = parts.flatMap(([p, q]) => [[p, Math.min(q, o.lo)], [Math.max(p, o.hi), q]]).filter(([p, q]) => q - p > 0.05);
          }
          for (const [p, q] of parts) {
            pieces.push({ a: at(p, m.c), b: at(q, m.c), u: [ux, uy], n: m.out, o0: 0, o1: thickness, z0: m.z0, z1: m.z1, ffl: m.ffl });
          }
        }
        continue;
      }
      // Shared: the line's elementary stretches, each as tall as the rooms on it.
      const cuts = [...new Set(line.flatMap((m) => [m.lo, m.hi]).map((t) => Math.round(t * 1000) / 1000))].sort((p, q) => p - q);
      const stretches = [];
      for (let k = 0; k + 1 < cuts.length; k += 1) {
        const p = cuts[k], q = cuts[k + 1];
        let z0 = Infinity, z1 = -Infinity, ffl = Infinity;
        for (const m of line) {
          if (m.lo > p + 1e-3 || m.hi < q - 1e-3) continue;
          z0 = Math.min(z0, m.z0);
          z1 = Math.max(z1, m.z1);
          ffl = Math.min(ffl, m.ffl);
        }
        if (!Number.isFinite(z0)) continue;
        const last = stretches[stretches.length - 1];
        if (last && Math.abs(last.z0 - z0) < 1e-3 && Math.abs(last.z1 - z1) < 1e-3 && Math.abs(last.q - p) < 1e-3) last.q = q;
        else stretches.push({ p, q, z0, z1, ffl });
      }
      for (const { p, q, z0, z1, ffl } of stretches) {
        pieces.push({ a: at(p), b: at(q), u: [ux, uy], n: [-uy, ux], o0: -thickness / 2, o1: thickness / 2, z0, z1, ffl });
      }
    }
  }
  return pieces;
}

/**
 * Where a door stands and which way its wall runs: its position, and its
 * direction if the model gives one, else the room edge (of its rooms') that
 * passes nearest it.
 */
function doorFrame(door, roomsById) {
  const [x, y] = door.position;
  if (door.direction) return { x, y, ux: door.direction[0], uy: door.direction[1] };
  let best = null, bestD = Infinity;
  for (const id of door.rooms) {
    const room = roomsById.get(id);
    if (!room) continue;
    for (const ring of [room.footprint, ...(room.holes || [])]) {
      for (let i = 0; i < ring.length; i += 1) {
        const a = ring[i], b = ring[(i + 1) % ring.length];
        const { d } = toSegment(x, y, a, b);
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        if (d < bestD && len > 1e-6) [best, bestD] = [[(b[0] - a[0]) / len, (b[1] - a[1]) / len], d];
      }
    }
  }
  return { x, y, ux: best ? best[0] : 1, uy: best ? best[1] : 0 };
}

/**
 * The egress geometry of a model, in the shape ViewerManager's
 * extractEgressGeometry returns ({floors, stairs, doors, walls, spaces,
 * obstacles, roofs, roofMeshes, cores, coreMeshes, terminals}), so the lattice
 * (lattice.js), the two-zone model (rooms.js) and Flamescapes
 * (flamescapesRun.js buildScene) take it unchanged. Meshes are in the viewer
 * frame, IFC (x, y, z) to (x, z, -y) for upAxis "y" (as the viewer draws an
 * IFC), or as they are for "z", each with an itemId.
 *
 *   spaces     each walkable room's prism, its itemId the room's: the
 *              lattice samples its bottom as floor, and the two-zone model
 *              takes it as a room.
 *   floors     each walkable room's slab, slabThickness under its floor
 *              level: a closed solid, as the Flamescapes voxeliser needs. A
 *              room a stair or ramp comes up into has none, so the stairwell
 *              is open to smoke; the lattice still has its floor from spaces.
 *   roofMeshes each walkable room's ceiling slab where no room above covers
 *              it (the top storey, setbacks), for Flamescapes. On a floor
 *              level where a room has a door to the outside that is not an
 *              exit, these slabs are floors instead: a roof terrace, walked
 *              between the doors that open onto it.
 *   walls      boxes along the outlines (wallPieces), from the slab's
 *              underside to the room's top, with a gap at every door and
 *              opening (doorHeight high; an opening the full height), and
 *              where a stair lands outside its room, so the lattice's wall
 *              section at 0.75 m leaves the doorways open.
 *   doors      a box per door at its position, across its wall, width by
 *              doorHeight: the lattice makes a door node there and links the
 *              rooms either side (lattice.js doorWaypoints). Each mesh carries
 *              its model kind; "opening" meshes are not leaves, and stay out
 *              of the solids when Flamescapes closes the doors. Doors into
 *              rooms nobody walks in (lift doors into shafts) are left out.
 *   stairs     each stair link as treads (stairSolids: along its walking
 *              line, else cores.js's assumed dog-leg line in its footprint),
 *              each ramp as a sloping slab (rampSolid), so the lattice walks
 *              them as modelled stairs (stairs.js), each joining exactly its
 *              two rooms. The footprint: the link's own, else the stair room
 *              or ramp at either end, else where the two rooms overlap.
 *   cores      each lift link as a cores.js lift over its shaft, from the
 *   coreMeshes lower floor to the upper: a stop at each floor, ridden only
 *              with useLifts, as for an IFC.
 *   obstacles, terminals  none.
 *   semantics  what ifcSemantics gives for an IFC: doors (exit flags,
 *              widths), storeys, spaces, so exits and the two-zone model
 *              work as on an IFC.
 *   lod1       for the viewer: rooms, doors and links by itemId, and the
 *              prisms of the rooms nobody walks in (shafts, voids, tanks).
 *
 * @param {object} model  a LOD1 model (see the header)
 * @param {object} opts   upAxis; wallThickness, slabThickness, doorHeight,
 *                        doorWidth (where a door gives none), in metres
 */
export function lod1ToEgressGeometry(model, opts = {}) {
  const { upAxis = "y", wallThickness, slabThickness, doorHeight, doorWidth } = { ...LOD1_DEFAULTS, ...opts };
  const frame = upAxis === "y" ? (x, y, z) => [x, z, -y] : (x, y, z) => [x, y, z];
  const boundary = model.meta?.boundary === "inner" ? "inner" : "shared";
  const roomsById = new Map(model.rooms.map((r) => [r.id, r]));
  let nextId = 1;

  const builder = () => ({ pos: [], idx: [] });
  const finish = (m, itemId, extra = {}) => {
    const n = m.pos.length / 3;
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n; i += 1) {
      const p = frame(m.pos[i * 3], m.pos[i * 3 + 1], m.pos[i * 3 + 2]);
      positions[i * 3] = p[0];
      positions[i * 3 + 1] = p[1];
      positions[i * 3 + 2] = p[2];
    }
    return { itemId, positions, indices: Uint32Array.from(m.idx), ...extra };
  };
  // A closed prism: top faces up, bottom down, sides out (outer rings run
  // anticlockwise and holes clockwise, so outward is right of travel).
  const prism = (m, footprint, holes, z0, z1) => {
    const outer = oriented(footprint, true);
    const inner = (holes || []).map((h) => oriented(h, false));
    const { points, indices } = triangulate(outer, inner);
    const base = m.pos.length / 3, n = points.length;
    for (const z of [z0, z1]) for (const [x, y] of points) m.pos.push(x, y, z);
    for (let k = 0; k < indices.length; k += 3) {
      const a = indices[k], b = indices[k + 1], c = indices[k + 2];
      m.idx.push(base + n + a, base + n + b, base + n + c, base + a, base + c, base + b);
    }
    let start = 0;
    for (const ring of [outer, ...inner]) {
      for (let i = 0; i < ring.length; i += 1) {
        const a = base + start + i, b = base + start + ((i + 1) % ring.length);
        m.idx.push(a, b, b + n, a, b + n, a + n);
      }
      start += ring.length;
    }
  };
  // A tread or landing: a box whose top and bottom are fans round their
  // middle. stairs.js measures a patch's narrow side along the principal
  // axes of its triangles' corners; two triangles count one diagonal's
  // corners twice, which turns the axes (8 degrees on a 1.45 by 0.54 m
  // landing) and made such a landing measure 0.74 m, wider than a tread.
  const fanBox = (m, ring, z0, z1) => {
    const r = oriented(ring, true);
    const base = m.pos.length / 3, k = r.length;
    const cx = r.reduce((s, p) => s + p[0], 0) / k, cy = r.reduce((s, p) => s + p[1], 0) / k;
    for (const z of [z0, z1]) {
      for (const [x, y] of r) m.pos.push(x, y, z);
      m.pos.push(cx, cy, z);
    }
    const top = base + k + 1;
    for (let i = 0; i < k; i += 1) {
      const a = i, b = (i + 1) % k;
      m.idx.push(top + k, top + a, top + b, base + k, base + b, base + a);
      m.idx.push(base + a, base + b, top + b, base + a, top + b, top + a);
    }
  };
  const solid = (footprint, holes, z0, z1, itemId, extra) => {
    const m = builder();
    prism(m, footprint, holes, z0, z1);
    return finish(m, itemId, extra);
  };
  // A box centred on (cx, cy) along u, `half` either way, from d0 to d1 along n.
  const box = (cx, cy, [ux, uy], [nx, ny], half, d0, d1, z0, z1) => {
    const ring = [
      [cx - ux * half + nx * d0, cy - uy * half + ny * d0],
      [cx + ux * half + nx * d0, cy + uy * half + ny * d0],
      [cx + ux * half + nx * d1, cy + uy * half + ny * d1],
      [cx - ux * half + nx * d1, cy - uy * half + ny * d1],
    ];
    return [oriented(ring, true), z0, z1];
  };

  // Links: which rooms a stair or ramp comes up into, or goes up from.
  const upperEnd = new Set(), lowerEnd = new Set();
  for (const l of model.links) {
    if (l.kind === "lift") continue;
    const [p, q] = l.rooms.map((id) => roomsById.get(id));
    if (!p || !q) continue;
    const [lo, hi] = p.ffl <= q.ffl ? [p, q] : [q, p];
    lowerEnd.add(lo.id);
    upperEnd.add(hi.id);
  }

  // Rooms: prisms, floor slabs, ceilings.
  const spaces = [], floors = [], roofMeshes = [], shafts = [];
  const lod1Rooms = [];
  const itemOf = new Map();
  for (const room of model.rooms) {
    const itemId = nextId++;
    itemOf.set(room.id, itemId);
    lod1Rooms.push({ itemId, id: room.id, name: room.name, class: room.class, level: room.level, walkable: room.walkable !== false });
    const top = room.ffl + room.height;
    if (room.walkable === false) {
      shafts.push(solid(room.footprint, room.holes, room.ffl, top, itemId));
      continue;
    }
    spaces.push(solid(room.footprint, room.holes, room.ffl, top, itemId));
  }
  const slabbed = model.rooms.filter((r) => r.walkable !== false && !upperEnd.has(r.id));
  const byFloor = new Map();
  for (const r of slabbed) {
    const k = Math.round(r.ffl * 100);
    if (!byFloor.has(k)) byFloor.set(k, []);
    byFloor.get(k).push(r);
  }
  for (const room of slabbed) floors.push(solid(room.footprint, room.holes, room.ffl - slabThickness, room.ffl, nextId++));
  // Terraces: on a floor level where a room has a door to the outside (not a
  // final exit), the uncovered roofs of the rooms below are walked on (a roof
  // terrace between a stair and a flat, a roof service level).
  const terraces = new Set();
  for (const door of model.doors) {
    if (door.kind === "exit" || (door.rooms[0] !== null && door.rooms[1] !== null)) continue;
    const room = roomsById.get(door.rooms[0] ?? door.rooms[1]);
    if (room && room.walkable !== false) terraces.add(Math.round(room.ffl * 100));
  }
  let terraceSlabs = 0;
  for (const room of model.rooms) {
    if (room.walkable === false || lowerEnd.has(room.id)) continue;
    const top = room.ffl + room.height;
    const above = byFloor.get(Math.round(top * 100)) || [];
    // Covered when every probe (its middle, and each corner and edge middle
    // drawn 5 cm in) stands on a floor slab of a room above.
    const [cx, cy] = roomPoint(room);
    const probes = [[cx, cy]];
    const ring = room.footprint;
    for (let i = 0; i < ring.length; i += 1) {
      const a = ring[i], b = ring[(i + 1) % ring.length];
      for (const [px, py] of [a, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]]) {
        const d = Math.hypot(cx - px, cy - py) || 1;
        probes.push([px + ((cx - px) / d) * 0.05, py + ((cy - py) / d) * 0.05]);
      }
    }
    if (above.length && probes.every(([x, y]) => above.some((r) => pointInRoom(x, y, r)))) continue;
    const slab = solid(room.footprint, room.holes, top - slabThickness, top, nextId++);
    if (!terraces.has(Math.round(top * 100))) {
      roofMeshes.push(slab);
      continue;
    }
    // A terrace, less the tanks, shafts and voids standing on it (whose
    // insides would otherwise be walled-in pieces of the graph). The whole
    // slab stays a roof for Flamescapes.
    const standing = model.rooms.filter((r) => r.walkable === false && Math.abs(r.ffl - top) < 0.01 && boxOverlap(bboxOf(r.footprint), bboxOf(room.footprint)) > 0);
    const pieces = standing.length ? slabMinus(room, standing) : [room.footprint];
    if (!pieces) {
      roofMeshes.push(slab);
      continue;
    }
    if (standing.length) roofMeshes.push(slab);
    for (const ring of pieces) floors.push(solid(ring, standing.length ? [] : room.holes, top - slabThickness, top, nextId++));
    terraceSlabs += 1;
  }

  // Doors the graph walks through: rooms on both sides walked in (or the outside).
  const egressDoors = [];
  for (const door of model.doors) {
    const sides = door.rooms.map((id) => (id === null || id === undefined ? null : roomsById.get(id)));
    if (sides.some((r, s) => door.rooms[s] !== null && door.rooms[s] !== undefined && !r)) continue;
    if (sides.some((r) => r && r.walkable === false)) continue;
    if (!sides[0] && !sides[1]) continue;
    if (sides[0] && sides[0] === sides[1]) continue;
    const f = doorFrame(door, roomsById);
    const width = door.width > 0 ? door.width : doorWidth;
    const sill = Number.isFinite(door.position[2]) ? door.position[2] : Math.min(...sides.filter(Boolean).map((r) => r.ffl));
    egressDoors.push({ door, ...f, width, sill, opening: door.kind === "opening" });
  }
  // Links. A stair joins exactly its two rooms with treads along its walking
  // line or the assumed dog-leg line, a gentle ramp with a sloping slab, and a
  // lift is a cores.js lift over its shaft. (Stairs as cores would be stacked
  // by coreStacks with any core over or under them, a ramp under a stair's
  // foot included, and walked over the union of their footprints.) Where a
  // stair's foot or head is outside its room (an IfcStair whose head is in
  // the opening beside the space above), that room's nearest wall gets a gap
  // the stair's width, as a door has.
  const stairs = [], cores = [], coreMeshes = [], lod1Links = [];
  const doorsOf = (room) => model.doors.filter((d) => d.rooms.includes(room.id)).map((d) => d.position);
  const landing = (room, point, width) => {
    if (pointInRoom(point[0], point[1], room)) return;
    const placed = doorOnBoundary(room, null, point, width);
    if (!placed?.direction || Math.hypot(placed.at[0] - point[0], placed.at[1] - point[1]) > 1.5) return;
    egressDoors.push({ door: null, x: placed.at[0], y: placed.at[1], ux: placed.direction[0], uy: placed.direction[1], width, sill: room.ffl, opening: false });
  };
  const middle = (ring) => [ring.reduce((s, p) => s + p[0], 0) / ring.length, ring.reduce((s, p) => s + p[1], 0) / ring.length];
  for (const link of model.links) {
    const [p, q] = link.rooms.map((id) => roomsById.get(id));
    if (!p || !q) continue;
    const [lower, upper] = p.ffl <= q.ffl ? [p, q] : [q, p];
    if (upper.ffl - lower.ffl < 0.5) continue;
    const footprint = linkFootprint(link, lower, upper);
    const itemId = nextId++;
    lod1Links.push({ itemId, id: link.id, kind: link.kind, rooms: [lower.id, upper.id] });
    if (link.kind === "lift") {
      coreMeshes.push(solid(footprint, [], lower.ffl, upper.ffl, itemId));
      cores.push({ itemId, kind: CORE_LIFT, name: `lift ${lower.level} to ${upper.level}`, source: "LOD1 link", parts: [] });
      continue;
    }
    const ends = { foot: doorsOf(lower), head: doorsOf(upper) };
    const ramp = link.kind === "ramp" ? rampSolid(footprint, lower.ffl, upper.ffl, ends) : null;
    if (ramp) {
      const m = builder();
      const base = 0;
      for (const [x, y, z] of ramp.top) m.pos.push(x, y, z);
      for (const [x, y, z] of ramp.top) m.pos.push(x, y, z - slabThickness);
      // Top up, bottom down, four sides out (the corners run anticlockwise).
      m.idx.push(base, base + 1, base + 2, base, base + 2, base + 3, base + 4, base + 6, base + 5, base + 4, base + 7, base + 6);
      for (let i = 0; i < 4; i += 1) {
        const a = base + i, b = base + ((i + 1) % 4);
        m.idx.push(a + 4, b + 4, b, a + 4, b, a);
      }
      stairs.push(finish(m, itemId));
      // The slab's low and high edges.
      const [c0, c1, c2, c3] = ramp.top;
      const ends2 = [[c0, c3], [c1, c2]].map(([a, b]) => ({ at: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], z: a[2] }));
      ends2.sort((a, b) => a.z - b.z);
      landing(lower, ends2[0].at, 1.2);
      landing(upper, ends2[1].at, 1.2);
      continue;
    }
    const m = builder();
    const boxes = stairSolids(footprint, lower.ffl, upper.ffl, { ...ends, walk: link.line, walkWidth: link.width });
    for (const { ring, z0, z1 } of boxes) fanBox(m, ring, z0, z1);
    stairs.push(finish(m, itemId));
    if (boxes.length) {
      const width = Math.min(link.width > 0 ? link.width : 1, 1.2);
      landing(lower, middle(boxes[0].ring), width);
      landing(upper, middle(boxes[boxes.length - 1].ring), width);
    }
  }

  // Walls, with a gap at every door: cut where a door stands within reach of
  // the wall's line, its wall running the same way.
  const walls = [];
  const pieces = wallPieces(model, { boundary, thickness: wallThickness, slab: slabThickness });
  const reach = boundary === "inner" ? wallThickness + 0.35 : wallThickness / 2 + 0.25;
  for (const piece of pieces) {
    const [ax, ay] = piece.a, [ux, uy] = piece.u, [nx, ny] = piece.n;
    const length = (piece.b[0] - ax) * ux + (piece.b[1] - ay) * uy;
    const gaps = [];
    for (const d of egressDoors) {
      // Any wall the opening passes through: one of its storey, or a taller
      // room's (a stairwell's two storeys) where the door is on its upper one.
      if (d.sill < piece.z0 - 0.35 || d.sill + 1 > piece.z1) continue;
      if (Math.abs(d.ux * uy - d.uy * ux) > 0.15) continue;
      const along = (d.x - ax) * ux + (d.y - ay) * uy;
      const across = (d.x - ax) * nx + (d.y - ay) * ny;
      if (across < Math.min(piece.o0, piece.o1) - reach || across > Math.max(piece.o0, piece.o1) + reach) continue;
      const p = Math.max(0, along - d.width / 2), q = Math.min(length, along + d.width / 2);
      if (q - p < 0.05) continue;
      const sill = d.sill > piece.z0 + slabThickness + 0.3 ? d.sill : piece.z0;
      gaps.push([p, q, sill, d.opening ? piece.z1 : d.sill + doorHeight]);
    }
    gaps.sort((g, h) => g[0] - h[0]);
    // The wall between the gaps at full height; over each gap a lintel, and
    // under one on an upper storey the wall below its sill.
    const stretches = [];
    let t = 0;
    for (const [p, q, sill, head] of gaps) {
      if (p > t + 0.02) stretches.push([t, p, piece.z0, piece.z1]);
      const from = Math.max(p, t);
      if (q > from + 0.02) {
        if (piece.z1 - head > 0.05) stretches.push([from, q, head, piece.z1]);
        if (sill - piece.z0 > 0.05) stretches.push([from, q, piece.z0, sill]);
      }
      t = Math.max(t, q);
    }
    if (length > t + 0.02) stretches.push([t, length, piece.z0, piece.z1]);
    for (const [p, q, z0, z1] of stretches) {
      const half = (q - p) / 2;
      const cx = ax + ux * (p + half), cy = ay + uy * (p + half);
      const [ring, lo, hi] = box(cx, cy, piece.u, piece.n, half, piece.o0, piece.o1, z0, z1);
      walls.push(solid(ring, [], lo, hi, nextId++));
    }
  }

  // Door boxes, and the doors as ifcSemantics would give them.
  const doors = [], semanticDoors = [], lod1Doors = [];
  for (const d of egressDoors) {
    if (!d.door) continue;
    const itemId = nextId++;
    // Narrower than wide, so doorWaypoints takes its long axis along the wall.
    const depth = Math.min(wallThickness + 0.04, 0.45 * d.width);
    const [ring, lo, hi] = box(d.x, d.y, [d.ux, d.uy], [-d.uy, d.ux], d.width / 2, -depth / 2, depth / 2, d.sill, d.sill + doorHeight);
    doors.push(solid(ring, [], lo, hi, itemId, { kind: d.door.kind }));
    const exit = d.door.kind === "exit" ? "LOD1 exit" : null;
    semanticDoors.push({ id: itemId, name: d.door.id, operation: null, width: d.width, height: doorHeight, widthM: d.door.width > 0 ? d.door.width : null, heightM: null, exit });
    lod1Doors.push({ itemId, id: d.door.id, kind: d.door.kind, exit: Boolean(exit), rooms: d.door.rooms });
  }


  return {
    floors,
    stairs,
    doors,
    walls,
    spaces,
    obstacles: [],
    roofs: roofMeshes.length,
    roofMeshes,
    cores,
    coreMeshes,
    terminals: [],
    semantics: {
      schema: "LOD1",
      doors: semanticDoors,
      roofSlabs: [],
      storeys: model.levels.map((l, k) => ({ id: k + 1, name: l.name ?? l.id, elevation: l.ffl })),
      spaces: lod1Rooms.map((r) => ({ id: r.itemId, name: r.id, longName: r.name })),
      lengthScale: 1,
    },
    lod1: { upAxis, boundary, rooms: lod1Rooms, doors: lod1Doors, links: lod1Links, shafts, itemOf, terraceSlabs },
  };
}
