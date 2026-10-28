/**
 * The navigation graph as one lattice for the whole building.
 *
 * Replaces per-triangle sampling plus k-nearest linking, which produced the
 * "diagonals": IFC slabs are tessellated into arbitrary thin triangles, each
 * got its own sheared sample pattern, and linking every point to its 12
 * nearest neighbours turned that irregularity into edges at every angle (54-57%
 * of edges on the Duplex and the buildingSMART sample scene; see
 * wiki/roadmap/browser-core-2026-10-27.md).
 *
 * Here every walkable triangle is sampled at the same lattice positions, a
 * lattice aligned to the building's dominant wall direction, and nodes link to
 * lattice neighbours only: the 4 axis neighbours, plus the 4 diagonals unless
 * `rectilinear`, never cutting a corner. Pure functions on typed arrays, so the
 * worker and node's test runner can both import it.
 */

import { CORE_LIFT, CORE_NAMES, CORE_STAIR, coreStacks, levelsOf, walkingLine } from "./cores.js";
import { stairLinks, stairPatches } from "./stairs.js";

export const KIND_FLOOR = 0;
export const KIND_STAIR = 1;
export const KIND_DOOR = 2;
// A lift or escalator stop: a node on the core's centre line at each floor it
// serves (cores.js). Assumed stair lines are KIND_STAIR.
export const KIND_LIFT = 3;

const AXIS = { x: 0, y: 1, z: 2 };
const KEY_OFFSET = 1 << 20;
const key = (i, j) => (i + KEY_OFFSET) * 2 ** 21 + (j + KEY_OFFSET);

export function axisIndex(upAxis) {
  return AXIS[upAxis] ?? 2;
}

export function horizontalAxes(upAxis) {
  const up = axisIndex(upAxis);
  return [0, 1, 2].filter((i) => i !== up);
}

// ----------------------------------------------------------------- walls

/**
 * Each wall mesh as a plan centreline, thickness and vertical extent.
 *
 * The direction is the principal axis of the mesh's plan vertices, not its
 * bounding box: on a building turned off the world axes the box's long side is
 * not the wall, and the old box-based line was simply wrong.
 */
export function wallSegments(meshes, upAxis) {
  const up = axisIndex(upAxis);
  const [h0, h1] = horizontalAxes(upAxis);
  const out = [];
  for (const mesh of meshes) {
    const p = mesh.positions;
    if (!p || p.length < 9) continue;
    const n = Math.floor(p.length / 3);
    let mx = 0, my = 0, lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i += 1) {
      mx += p[i * 3 + h0];
      my += p[i * 3 + h1];
      const u = p[i * 3 + up];
      if (u < lo) lo = u;
      if (u > hi) hi = u;
    }
    mx /= n;
    my /= n;
    let sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < n; i += 1) {
      const dx = p[i * 3 + h0] - mx;
      const dy = p[i * 3 + h1] - my;
      sxx += dx * dx;
      sxy += dx * dy;
      syy += dy * dy;
    }
    const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const ux = Math.cos(angle), uy = Math.sin(angle);
    let a = Infinity, b = -Infinity, c = Infinity, d = -Infinity;
    for (let i = 0; i < n; i += 1) {
      const dx = p[i * 3 + h0] - mx;
      const dy = p[i * 3 + h1] - my;
      const along = dx * ux + dy * uy;
      const across = -dx * uy + dy * ux;
      if (along < a) a = along;
      if (along > b) b = along;
      if (across < c) c = across;
      if (across > d) d = across;
    }
    out.push({
      segment: [
        [mx + ux * a, my + uy * a],
        [mx + ux * b, my + uy * b],
      ],
      thickness: d - c,
      up_min: lo,
      up_max: hi,
    });
  }
  return out;
}

/**
 * The building's grid direction in radians, in [-pi/4, pi/4): the
 * length-weighted mean of wall directions modulo 90 degrees, taken on the
 * circle of 4 * angle so 0 and 90 degrees agree. 0 when there are no walls.
 */
export function dominantAngle(walls) {
  let s = 0, c = 0;
  for (const w of walls) {
    const [[x0, y0], [x1, y1]] = w.segment;
    const len = Math.hypot(x1 - x0, y1 - y0);
    if (len < 0.3) continue;
    const a = Math.atan2(y1 - y0, x1 - x0);
    s += len * Math.sin(4 * a);
    c += len * Math.cos(4 * a);
  }
  if (Math.hypot(s, c) < 1e-9) return 0;
  return Math.atan2(s, c) / 4;
}

// --------------------------------------------------------------- sampling

/**
 * Sample walkable triangles at lattice positions (i * spacing, j * spacing)
 * in the frame turned by `angle`. Returns parallel arrays of i, j, height.
 */
function sampleTriangles(meshes, opts, out, requireUpward) {
  // `scale` maps this group's lattice indices onto the shared half-step index:
  // floors sample every s and use 2i, stairs sample every s/2 and use i.
  const { up, h0, h1, spacing, cos, sin, minUp, excludeAbove, kind, scale = 1, bottomOnly = false } = opts;
  const eps = 1e-7;
  let added = 0;
  for (const mesh of meshes) {
    const { positions: p, indices: ix } = mesh;
    if (!p || !ix || ix.length < 3) continue;
    // Rooms (IfcSpace): their floor is the volume's bottom, whichever way the
    // exporter wound it, so take the horizontal faces at the mesh's lowest
    // height and ignore the normal's sign. Trusting "faces pointing down"
    // would pick each room's ceiling on a model wound inside out.
    let bottom = -Infinity;
    if (bottomOnly) {
      bottom = Infinity;
      for (let o = up; o < p.length; o += 3) if (p[o] < bottom) bottom = p[o];
    }
    for (let t = 0; t + 2 < ix.length; t += 3) {
      const a = ix[t] * 3, b = ix[t + 1] * 3, c = ix[t + 2] * 3;
      if (c + 2 >= p.length || a + 2 >= p.length || b + 2 >= p.length) continue;
      const ah = p[a + up], bh = p[b + up], ch = p[c + up];
      if (excludeAbove !== null && ah >= excludeAbove && bh >= excludeAbove && ch >= excludeAbove) {
        continue;
      }
      // Upward normal, in world axes.
      const e1 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
      const e2 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
      const nrm = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
      ];
      const mag = Math.hypot(nrm[0], nrm[1], nrm[2]);
      if (mag < 1e-9) continue;
      const nUp = nrm[up] / mag;
      if (bottomOnly) {
        if (Math.abs(nUp) < minUp || Math.max(ah, bh, ch) > bottom + 0.05) continue;
      } else if ((requireUpward ? nUp : Math.abs(nUp)) < minUp) {
        continue;
      }

      // Plan coordinates in the lattice frame.
      const pu = (o) => (p[o + h0] * cos + p[o + h1] * sin) / spacing;
      const pv = (o) => (-p[o + h0] * sin + p[o + h1] * cos) / spacing;
      const au = pu(a), av = pv(a), bu = pu(b), bv = pv(b), cu = pu(c), cv = pv(c);
      const det = (bv - cv) * (au - cu) + (cu - bu) * (av - cv);
      if (Math.abs(det) < 1e-12) continue;
      const i0 = Math.ceil(Math.min(au, bu, cu) - eps);
      const i1 = Math.floor(Math.max(au, bu, cu) + eps);
      const j0 = Math.ceil(Math.min(av, bv, cv) - eps);
      const j1 = Math.floor(Math.max(av, bv, cv) + eps);
      for (let i = i0; i <= i1; i += 1) {
        for (let j = j0; j <= j1; j += 1) {
          const w0 = ((bv - cv) * (i - cu) + (cu - bu) * (j - cv)) / det;
          const w1 = ((cv - av) * (i - cu) + (au - cu) * (j - cv)) / det;
          const w2 = 1 - w0 - w1;
          if (w0 < -eps || w1 < -eps || w2 < -eps) continue;
          out.i.push(i * scale);
          out.j.push(j * scale);
          out.h.push(w0 * ah + w1 * bh + w2 * ch);
          out.kind.push(kind);
          added += 1;
        }
      }
    }
  }
  return added;
}

/** Sample a group; if insisting on upward normals finds nothing, the model's
 *  winding cannot be trusted, so accept either orientation. */
function sampleGroup(meshes, opts, out) {
  const mark = out.i.length;
  if (sampleTriangles(meshes, opts, out, true) > 0 || meshes.length === 0) return false;
  out.i.length = out.j.length = out.h.length = out.kind.length = mark;
  return sampleTriangles(meshes, opts, out, false) > 0;
}

/**
 * Collapse each lattice column to its walking surfaces: sort by height, split
 * where the gap exceeds `gap`, keep the top of each cluster. A slab's top and
 * underside, or a ceiling under the floor above, merge into one surface; two
 * storeys stay two.
 */
function collapse(samples, gap) {
  const columns = new Map();
  for (let n = 0; n < samples.i.length; n += 1) {
    const k = key(samples.i[n], samples.j[n]);
    let list = columns.get(k);
    if (!list) columns.set(k, (list = []));
    list.push(n);
  }
  const nodes = [];
  for (const list of columns.values()) {
    list.sort((x, y) => samples.h[x] - samples.h[y]);
    let top = list[0];
    for (let m = 1; m < list.length; m += 1) {
      if (samples.h[list[m]] - samples.h[top] > gap) nodes.push(top);
      top = list[m];
    }
    nodes.push(top);
  }
  return nodes;
}

// ------------------------------------------------------------- obstacles

/**
 * Plan outline of a group of meshes cut by the horizontal plane at `height`:
 * every triangle that straddles the plane contributes one segment. A
 * watertight wall gives closed loops; a door void gives a gap, whether or not
 * the exporter put an IfcDoor in it.
 *
 * `ranges` (from heightRanges) lets a cut skip the triangles the plane misses
 * without reading their vertices: a model with many floor heights is cut many
 * times, and each cut used to visit every triangle (5 of 8 s on a 70 MB model).
 */
function sectionAt(meshes, height, up, h0, h1, ranges = null) {
  const segs = [];
  let k = 0;
  for (const { positions: p, indices: ix } of meshes) {
    if (!p || !ix) continue;
    for (let t = 0; t + 2 < ix.length; t += 3, k += 1) {
      if (ranges && (ranges.lo[k] > height || ranges.hi[k] <= height)) continue;
      const va = ix[t] * 3, vb = ix[t + 1] * 3, vc = ix[t + 2] * 3;
      const aa = p[va + up] > height, ab = p[vb + up] > height, ac = p[vc + up] > height;
      const n = aa + ab + ac;
      if (n === 0 || n === 3) continue;
      // The lone vertex on one side, and the two edges leaving it, in the
      // vertex order of the triangle.
      let a, b, c;
      if (n === 1 ? aa : !aa) [a, b, c] = [va, vb, vc];
      else if (n === 1 ? ab : !ab) [a, b, c] = [vb, vc, va];
      else [a, b, c] = [vc, va, vb];
      const s = (height - p[a + up]) / (p[b + up] - p[a + up]);
      const r = (height - p[a + up]) / (p[c + up] - p[a + up]);
      segs.push(
        p[a + h0] + s * (p[b + h0] - p[a + h0]),
        p[a + h1] + s * (p[b + h1] - p[a + h1]),
        p[a + h0] + r * (p[c + h0] - p[a + h0]),
        p[a + h1] + r * (p[c + h1] - p[a + h1]),
      );
    }
  }
  return Float64Array.from(segs);
}

/** Each triangle's lowest and highest vertex along `up`, in sectionAt's order. */
function heightRanges(meshes, up) {
  let count = 0;
  for (const { positions: p, indices: ix } of meshes) if (p && ix) count += Math.max(0, Math.floor(ix.length / 3));
  const lo = new Float64Array(count), hi = new Float64Array(count);
  let k = 0;
  for (const { positions: p, indices: ix } of meshes) {
    if (!p || !ix) continue;
    for (let t = 0; t + 2 < ix.length; t += 3, k += 1) {
      const a = p[ix[t] * 3 + up], b = p[ix[t + 1] * 3 + up], c = p[ix[t + 2] * 3 + up];
      lo[k] = Math.min(a, b, c);
      hi[k] = Math.max(a, b, c);
    }
  }
  return { lo, hi };
}

/** A uniform grid over plan segments, for edge and point queries. */
export class SegmentIndex {
  constructor(segs, cell = 1.0) {
    this.segs = segs;
    this.cell = cell;
    this.cells = new Map();
    this.stamp = new Int32Array(segs.length / 4);
    this.tick = 0;
    for (let s = 0; s < segs.length / 4; s += 1) {
      const [x0, y0, x1, y1] = segs.subarray(s * 4, s * 4 + 4);
      this.forCells(Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1), (k) => {
        let list = this.cells.get(k);
        if (!list) this.cells.set(k, (list = []));
        list.push(s);
      });
    }
    this.cx0 = Infinity;
    this.cx1 = -Infinity;
    for (const k of this.cells.keys()) {
      const cx = Math.floor(k / 2 ** 21) - KEY_OFFSET;
      if (cx < this.cx0) this.cx0 = cx;
      if (cx > this.cx1) this.cx1 = cx;
    }
  }

  forCells(x0, y0, x1, y1, fn) {
    const c = this.cell;
    for (let i = Math.floor(x0 / c); i <= Math.floor(x1 / c); i += 1) {
      for (let j = Math.floor(y0 / c); j <= Math.floor(y1 / c); j += 1) fn(key(i, j));
    }
  }

  /** Each segment near the box, once. */
  near(x0, y0, x1, y1, fn) {
    this.tick += 1;
    this.forCells(x0, y0, x1, y1, (k) => {
      for (const s of this.cells.get(k) || []) {
        if (this.stamp[s] === this.tick) continue;
        this.stamp[s] = this.tick;
        fn(s);
      }
    });
  }

  /** Whether the segment a-b properly crosses any outline segment. */
  crosses(ax, ay, bx, by) {
    let hit = false;
    const g = this.segs;
    this.near(Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by), (s) => {
      if (!hit && segmentsCross(ax, ay, bx, by, g[s * 4], g[s * 4 + 1], g[s * 4 + 2], g[s * 4 + 3])) hit = true;
    });
    return hit;
  }

  /** Fraction along a-b of the first outline crossing, or Infinity. */
  firstHit(ax, ay, bx, by) {
    let best = Infinity;
    const g = this.segs;
    const rx = bx - ax, ry = by - ay;
    this.near(Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by), (s) => {
      const cx = g[s * 4], cy = g[s * 4 + 1];
      const sx = g[s * 4 + 2] - cx, sy = g[s * 4 + 3] - cy;
      const den = rx * sy - ry * sx;
      if (Math.abs(den) < 1e-12) return;
      const t = ((cx - ax) * sy - (cy - ay) * sx) / den;
      const u = ((cx - ax) * ry - (cy - ay) * rx) / den;
      if (t > 1e-9 && t <= 1 && u >= -1e-9 && u <= 1 + 1e-9 && t < best) best = t;
    });
    return best;
  }

  /**
   * Whether (x, y) is inside a closed outline: parity of crossings on a ray
   * along +x. Only trusted within `reach` of an outline segment, so an open
   * (non-watertight) mesh cannot swallow a whole floor.
   */
  inside(x, y, reach = 0.6) {
    let near = false;
    const g = this.segs;
    this.near(x - reach, y - reach, x + reach, y + reach, (s) => {
      if (!near && distanceToSegment(x, y, [[g[s * 4], g[s * 4 + 1]], [g[s * 4 + 2], g[s * 4 + 3]]]) <= reach) near = true;
    });
    if (!near) return false;
    let parity = 0;
    const c = this.cell;
    const end = (this.cx1 + 1) * c;
    this.near(x, y, Math.max(x, end), y, (s) => {
      const x0 = g[s * 4], y0 = g[s * 4 + 1], x1 = g[s * 4 + 2], y1 = g[s * 4 + 3];
      if (y0 > y === y1 > y) return;
      const xi = x0 + ((y - y0) * (x1 - x0)) / (y1 - y0);
      if (xi > x) parity ^= 1;
    });
    return parity === 1;
  }
}

// ---------------------------------------------------------------- linking

/**
 * Whether segments ab and cd cross at a point strictly inside both. Exactly
 * that, in exact arithmetic (formal/fireegress Segments.lean, crossTest_iff):
 * touching at an endpoint, a T-junction and collinear overlap all return
 * false. Signs are compared rather than multiplied, so tiny or huge
 * determinants cannot underflow or overflow the product.
 */
function segmentsCross(ax, ay, bx, by, cx, cy, dx, dy) {
  const d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx);
  const d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);
  const d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  const d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
  const opposite = (u, v) => (u > 0 && v < 0) || (u < 0 && v > 0);
  return opposite(d1, d2) && opposite(d3, d4);
}

function distanceToSegment(px, py, [[x0, y0], [x1, y1]]) {
  const vx = x1 - x0, vy = y1 - y0;
  const len2 = vx * vx + vy * vy;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - x0) * vx + (py - y0) * vy) / len2)) : 0;
  return Math.hypot(px - (x0 + t * vx), py - (y0 + t * vy));
}

/**
 * Build the lattice navigation graph.
 *
 * @returns {{points: Float32Array, edges: Uint32Array, kinds: Uint8Array,
 *            blocked: Uint8Array, walls: object[], angle: number, spacing: number,
 *            stats: object}}
 */
export function buildLatticeGraph({
  floors = [],
  stairs = [],
  doors = [],
  walls = [],
  spaces = [],
  obstacles = [],
  // Vertical cores (cores.js): items {itemId, kind, name, source, parts},
  // with their meshes found among stairs, spaces and coreMeshes. Lifts and
  // escalators are linked only when useLifts; a fire evacuation does not use
  // them unless they are evacuation lifts.
  cores = [],
  coreMeshes = [],
  useLifts = false,
  sectionHeight = 0.75,
  hugWalls = true,
  // Wall-face nodes sit this far off the face, and are only added when the
  // grid line falls at least minHug short of the wall.
  wallInset = 0.02,
  minHug = 0.05,
  upAxis = "y",
  spacing = 0.5,
  agentHeight = 0.75,
  maxEdgeRise = 0.35,
  // The highest riser a staircase may have: treads link up to this plus a
  // margin, and a staircase's ends link to floors within it.
  stairRiser = 0.3,
  minFragment = 8,
  columnGap = 0.9,
  rectilinear = false,
  maxPoints = 40000,
}) {
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const up = axisIndex(upAxis);
  const [h0, h1] = horizontalAxes(upAxis);
  const wallList = wallSegments(walls, upAxis);
  const angle = dominantAngle(wallList);
  const cos = Math.cos(angle), sin = Math.sin(angle);

  // Obstacles are what is solid 0.75 m above the floor, cut from the actual
  // geometry (walls plus windows, railings, columns, curtain walls), as
  // TopologicPy's Navigation does. A wall reduced to one centreline closed
  // every opening without an IfcDoor and got L-shaped and curved walls wrong
  // (HouseZ: rooms that plainly connect stayed apart). Sections are cached per
  // 10 cm of floor height.
  const solid = [...walls, ...obstacles];
  const sections = new Map();
  let ranges = null;
  const sectionFor = (floorHeight) => {
    const k = Math.round((floorHeight + sectionHeight) * 10);
    let index = sections.get(k);
    if (!index) {
      ranges ||= heightRanges(solid, up);
      sections.set(k, (index = new SegmentIndex(sectionAt(solid, k / 10, up, h0, h1, ranges))));
    }
    return index;
  };

  // Roof slabs are walkable geometry but not floors. Exclude surfaces within a
  // metre of the top of the whole model (walls included), and only when there
  // is floor at least 2 m lower. The old rule measured from the highest floor
  // alone, so a single-storey model lost its only floor and came out empty.
  const extent = (groups) => {
    let lo = Infinity, hi = -Infinity;
    for (const group of groups) {
      for (const mesh of group) {
        const p = mesh.positions;
        for (let i = up; p && i < p.length; i += 3) {
          if (p[i] < lo) lo = p[i];
          if (p[i] > hi) hi = p[i];
        }
      }
    }
    return [lo, hi];
  };
  const [floorLow] = extent([floors, stairs, spaces]);
  const [, modelTop] = extent([floors, stairs, walls, doors]);
  const excludeAbove =
    Number.isFinite(modelTop) && modelTop - floorLow >= 2.0 ? modelTop - 1.0 : null;

  const stairParts = stairPatches(stairs, { up, h0, h1 });
  const stacks = cores.length ? coreStacks(cores, [...stairs, ...spaces, ...coreMeshes], { up, h0, h1, cos, sin }) : [];

  // Coarsen until the lattice fits the point budget.
  let s = Math.max(spacing, 0.05);
  let samples, kept, windingFallback;
  for (let attempt = 0; attempt < 6; attempt += 1) {
    samples = { i: [], j: [], h: [], kind: [] };
    const base = { up, h0, h1, spacing: s, cos, sin, excludeAbove };
    // Stair treads are not sampled here: each staircase becomes its walking
    // line below (stairs.js). A ramp's sloping surface is sampled as floor.
    // Room floors (IfcSpace bottoms) as well as slabs: some architectural
    // models have no floor slabs at all (Office A has 5 m2 of slab and floors
    // only as spaces), and a space's footprint is the floor inside the walls.
    // Where both exist the column collapse keeps the walking surface.
    sampleTriangles(spaces, { ...base, scale: 2, minUp: Math.cos((10 * Math.PI) / 180), kind: KIND_FLOOR, bottomOnly: true }, samples, false);
    windingFallback = sampleGroup(floors, { ...base, scale: 2, minUp: Math.cos((10 * Math.PI) / 180), kind: KIND_FLOOR }, samples);
    if (stairParts.ramp.indices.length) {
      sampleGroup([stairParts.ramp], { ...base, scale: 2, minUp: Math.cos((15 * Math.PI) / 180), kind: KIND_FLOOR }, samples);
    }
    kept = collapse(samples, columnGap);
    if (kept.length <= maxPoints) break;
    s *= Math.sqrt(kept.length / maxPoints) * 1.05;
  }

  // Drop nodes standing inside solid wall (the slab runs under it). Door voids
  // have no material at section height, so doorways keep their nodes.
  const doorPoints = doorWaypoints(doors, upAxis);
  // Indices count half steps (see sampleTriangles' `scale`).
  const hs = s / 2;
  const plan = (i, j) => [i * hs * cos - j * hs * sin, i * hs * sin + j * hs * cos];
  kept = kept.filter((n) => {
    const [x, y] = plan(samples.i[n], samples.j[n]);
    return !sectionFor(samples.h[n]).inside(x, y);
  });

  const count = kept.length;
  const patches = stairParts.patches;
  const maxRise = stairRiser + 0.06;
  const walk = stairLinks(patches, { maxRise });
  // Room for up to four wall-face nodes per lattice node, one per door and one
  // per stair tread or landing (see the steps below), and for each core a
  // walking line or a stop per storey over the model's whole height.
  const coreStep = Math.min(s, 0.5);
  const modelRise = Number.isFinite(modelTop - floorLow) ? Math.max(modelTop - floorLow, 0) : 0;
  const perCore = Math.ceil(((modelRise + 4) * 2.2) / coreStep) + 16 * Math.ceil(modelRise / 2 + 2);
  const capacity = count * 5 + doorPoints.length + patches.length + stacks.length * perCore;
  const points = new Float32Array(capacity * 3);
  const kinds = new Uint8Array(capacity);
  const heights = new Float64Array(capacity);
  const cells = new Map();
  kept.forEach((n, idx) => {
    const [x, y] = plan(samples.i[n], samples.j[n]);
    points[idx * 3 + h0] = x;
    points[idx * 3 + h1] = y;
    points[idx * 3 + up] = samples.h[n] + agentHeight;
    heights[idx] = samples.h[n];
    kinds[idx] = samples.kind[n];
    const k = key(samples.i[n], samples.j[n]);
    let list = cells.get(k);
    if (!list) cells.set(k, (list = []));
    list.push(idx);
  });
  const lattice = kept.map((n) => [samples.i[n], samples.j[n]]);

  // A floor link may span maxEdgeRise.
  const reach = (a, b) => Math.abs(heights[a] - heights[b]) <= maxEdgeRise + 1e-6;
  const best = (from, di, dj, want) => {
    const [i, j] = lattice[from];
    const list = cells.get(key(i + di, j + dj));
    if (!list) return -1;
    let pick = -1, gap = Infinity;
    for (const m of list) {
      if (want !== undefined && kinds[m] !== want) continue;
      const dh = Math.abs(heights[m] - heights[from]);
      if (dh < gap && reach(from, m)) {
        pick = m;
        gap = dh;
      }
    }
    return pick;
  };

  // Floor nodes sit on even half-step indices and link a full step apart. Each
  // unordered pair is visited once: every node looks the same four directions.
  const edges = [];
  // Each floor diagonal with its two axis detours (a -> c -> b), checked once
  // the walls are known (below).
  const diagonals = [];
  const fullSteps = rectilinear ? [[2, 0], [0, 2]] : [[2, 0], [0, 2], [2, 2], [2, -2]];
  for (let a = 0; a < count; a += 1) {
    for (const [di, dj] of fullSteps) {
      const b = best(a, di, dj, KIND_FLOOR);
      if (b < 0) continue;
      // No corner cutting: both axis neighbours must be walkable too.
      if (di !== 0 && dj !== 0) {
        const c1 = best(a, di, 0, KIND_FLOOR);
        const c2 = best(a, 0, dj, KIND_FLOOR);
        if (c1 < 0 || c2 < 0) continue;
        diagonals.push([edges.length / 2, a, c1, c2, b]);
      }
      edges.push(a, b);
    }
  }

  // Hug the walls. Where a grid line stops short of a wall - the next lattice
  // point is inside the wall, or the link to it crosses one - extend it to the
  // wall face with a node there, as TopologicPy's trimmed-cell recipe does
  // (Shell.Rectangle cut by the walls with Topology.Difference, so boundary
  // cells are trimmed and their lines reach the wall; ForNeha.ipynb). Linked
  // along the grid line only, never along the wall, matching that recipe's
  // use of interior edges.
  let total = count;
  let hugged = 0;
  if (hugWalls) {
    const axes = [[2, 0], [-2, 0], [0, 2], [0, -2]];
    for (let a = 0; a < count; a += 1) {
      if (kinds[a] !== KIND_FLOOR) continue;
      const [i, j] = lattice[a];
      const [ax, ay] = plan(i, j);
      const index = sectionFor(heights[a]);
      for (const [di, dj] of axes) {
        const b = best(a, di, dj, KIND_FLOOR);
        if (b >= 0) {
          const bx = points[b * 3 + h0], by = points[b * 3 + h1];
          if (!index.crosses(ax, ay, bx, by)) continue; // an open link: nothing to hug
        }
        const [tx, ty] = plan(i + di, j + dj);
        const t = index.firstHit(ax, ay, tx, ty);
        if (!Number.isFinite(t)) continue; // the floor ends here, not a wall
        const reach = t * s - wallInset;
        if (reach < minHug) continue;
        const ux = (tx - ax) / s, uy = (ty - ay) / s;
        const n = total++;
        points[n * 3 + h0] = ax + ux * reach;
        points[n * 3 + h1] = ay + uy * reach;
        points[n * 3 + up] = heights[a] + agentHeight;
        heights[n] = heights[a];
        kinds[n] = KIND_FLOOR;
        edges.push(a, n);
        hugged += 1;
      }
    }
  }

  // Staircases (stairs.js): a node per tread or landing at its centroid, which
  // on a flight is its centre line, linked up the flight. Each end of a
  // staircase links once to the floor: to the floor node nearest the point
  // one going beyond the end along the walking line, on the outward side,
  // within one riser of it and not through a wall. These links, like the
  // treads', are not blocked
  // afterwards: each was checked against the walls when it was made.
  const stairEdge = new Set();
  const stairBase = total;
  patches.forEach((patch, k) => {
    const n = stairBase + k;
    points[n * 3 + h0] = patch.x;
    points[n * 3 + h1] = patch.y;
    points[n * 3 + up] = patch.h + agentHeight;
    heights[n] = patch.h;
    kinds[n] = KIND_STAIR;
  });
  total += patches.length;
  for (const [a, b] of walk.links) {
    stairEdge.add(edges.length / 2);
    edges.push(stairBase + a, stairBase + b);
  }
  const endReach = 1.5;
  const floorAt = (x, y, h, side, dir, going) => {
    // Lattice cells near (x, y): the inverse of plan().
    const ci = (x * cos + y * sin) / hs, cj = (-x * sin + y * cos) / hs;
    const r = Math.ceil(endReach / hs) + 1;
    let pick = -1, score = Infinity;
    for (let i = Math.floor(ci) - r; i <= Math.ceil(ci) + r; i += 1) {
      for (let j = Math.floor(cj) - r; j <= Math.ceil(cj) + r; j += 1) {
        const list = cells.get(key(i, j));
        if (!list) continue;
        for (const m of list) {
          if (kinds[m] !== KIND_FLOOR) continue;
          // Within one riser. A single step (no direction) joins the floor
          // below at its foot and the floor above at its head; a flight's
          // end may land a little proud of or below its floor (Wellness: a
          // top tread 12 cm above the slab), and its direction picks the side.
          const dh = heights[m] - h;
          if (Math.abs(dh) > maxRise || (!dir && ((side < 0 && dh > 0.05) || (side > 0 && dh < -0.05)))) continue;
          const mx = points[m * 3 + h0], my = points[m * 3 + h1];
          const d = Math.hypot(mx - x, my - y);
          if (d > endReach) continue;
          let rank = d;
          if (dir) {
            if ((mx - x) * dir[0] + (my - y) * dir[1] <= 0) continue;
            rank = Math.hypot(mx - (x + dir[0] * going), my - (y + dir[1] * going));
          }
          if (rank >= score) continue;
          const index = sectionFor(heights[m]);
          if (index.crosses(x, y, mx, my) || index.inside((x + mx) / 2, (y + my) / 2)) continue;
          pick = m;
          score = rank;
        }
      }
    }
    return pick;
  };
  let stairEnds = 0, stairEndsLinked = 0;
  const linkEnd = (k, side, neighbours) => {
    const patch = patches[k];
    let dir = null, going = 0;
    if (neighbours.length) {
      let mx = 0, my = 0;
      for (const q of neighbours) {
        mx += patches[q].x / neighbours.length;
        my += patches[q].y / neighbours.length;
      }
      const dx = patch.x - mx, dy = patch.y - my, len = Math.hypot(dx, dy);
      if (len > 1e-6) {
        dir = [dx / len, dy / len];
        going = Math.max(0.25, Math.min(0.6, len));
      }
    }
    stairEnds += 1;
    // Failing a floor along the walking line (a wall or a void beyond the
    // end), the nearest floor at the end's level in any direction.
    let m = floorAt(patch.x, patch.y, patch.h, side, dir, going);
    if (m < 0 && dir) m = floorAt(patch.x, patch.y, patch.h, side, null, 0);
    if (m < 0) return;
    stairEndsLinked += 1;
    stairEdge.add(edges.length / 2);
    edges.push(stairBase + k, m);
  };
  for (let k = 0; k < patches.length; k += 1) {
    if (!walk.lower[k].length) linkEnd(k, -1, walk.higher[k]);
    if (!walk.higher[k].length) linkEnd(k, 1, walk.lower[k]);
  }

  // Doorways. Exporters often stop the slab at the wall face, so a doorway has
  // no floor under it and the lattice alone leaves the rooms either side apart
  // (on the Duplex, separate pieces at ground level). The old 2.25 m
  // nearest-neighbour links jumped such gaps - and party walls with them. Here
  // a lattice node standing in the opening becomes the door node; failing
  // that, a node is added in the opening and linked to the nearest floor nodes
  // on each side within a metre. Those links are never blocked by walls. They
  // are the only links off the lattice, and each one is a real doorway.
  const doorEdge = new Set();
  const doorReport = [];
  const pairKey = (a, b) => (a < b ? a * 4294967296 + b : b * 4294967296 + a);
  const linked = new Set();
  for (let e = 0; e < edges.length; e += 2) linked.add(pairKey(edges[e], edges[e + 1]));
  for (const [dx, dy, dlow, ux, uy, width, itemId] of doorPoints) {
    const report = { x: dx, y: dy, sill: dlow, width, itemId: itemId ?? null, node: -1, added: false, sides: [0, 0] };
    doorReport.push(report);
    const nx = -uy, ny = ux;
    const half = Math.max(width / 2, s / 2) + 1e-6;
    const local = [];
    for (let n = 0; n < stairBase + patches.length; n += 1) {
      if (n === count) n = stairBase; // skip the wall-face nodes in between
      if (n >= stairBase + patches.length) break;
      if (Math.abs(heights[n] - dlow) > maxEdgeRise + 0.05) continue;
      const rx = points[n * 3 + h0] - dx, ry = points[n * 3 + h1] - dy;
      const along = rx * ux + ry * uy, across = rx * nx + ry * ny;
      if (Math.abs(along) <= half && Math.abs(across) <= 1.0 + s) local.push([n, along, across]);
    }
    if (!local.length) continue;
    let door = -1, score = Infinity;
    for (const [n, along, across] of local) {
      if (Math.abs(across) <= s / 2 && Math.abs(along) + Math.abs(across) < score) {
        door = n;
        score = Math.abs(along) + Math.abs(across);
      }
    }
    // Even when a lattice node stands in the opening, still bridge both sides:
    // the nodes inside the wall either side of it were removed, so on the
    // Duplex a reused door node sat on one side with no link to the other.
    if (door < 0) {
      door = total;
      total += 1;
      report.added = true;
      points[door * 3 + h0] = dx;
      points[door * 3 + h1] = dy;
      points[door * 3 + up] = dlow + agentHeight;
      heights[door] = dlow;
    }
    kinds[door] = KIND_DOOR;
    report.node = door;
    for (const side of [1, -1]) {
      const near = local
        .filter(([n, , across]) => n !== door && side * across > 1e-6 && !linked.has(pairKey(door, n)))
        .sort((p, q) => Math.abs(p[2]) - Math.abs(q[2]) || Math.abs(p[1]) - Math.abs(q[1]))
        .slice(0, 3);
      report.sides[side > 0 ? 0 : 1] = near.length;
      for (const [n] of near) {
        doorEdge.add(edges.length / 2);
        linked.add(pairKey(door, n));
        edges.push(door, n);
      }
    }
  }

  // Vertical cores (cores.js). The floor levels beside a core are the heights
  // of floor nodes within a metre of its footprint (two for a lift, whose
  // landing lies outside the shaft), split where they jump by more than a
  // metre. A stair or escalator core serves the levels within its own height
  // range; a lift serves every level beside it, since exporters often draw
  // only the car. Between two consecutive levels 1 to 8 m apart:
  // - a stair core is skipped where a modelled tread or landing stands in its
  //   footprint between them (the treads already join them); otherwise it gets
  //   an assumed walking line (walkingLine), linked at its foot and head to
  //   the nearest floor node of each level in or beside the footprint, not
  //   through a wall, in the reflection of the line whose ends lie nearest
  //   those floor nodes;
  // - an escalator is walked the same way, when useLifts;
  // - a lift gets a stop at its footprint's centre on each level, linked to
  //   the nearest floor node there (through the shaft wall if it must, as a
  //   lift door is often not cut in it) and to the stops above and below,
  //   when useLifts. Lift links cost their length like any other, so a route
  //   using one counts the ride as a walk of the same rise: no waiting.
  // Core links, like stair links, are not blocked by walls afterwards.
  const coreReport = [];
  const lattice2uv = (n) => {
    const x = points[n * 3 + h0], y = points[n * 3 + h1];
    return [x * cos + y * sin, -x * sin + y * cos];
  };
  const fromUV = (u, v) => [u * cos - v * sin, u * sin + v * cos];
  const addNode = (u, v, h, kind) => {
    const n = total++;
    const [x, y] = fromUV(u, v);
    points[n * 3 + h0] = x;
    points[n * 3 + h1] = y;
    points[n * 3 + up] = h + agentHeight;
    heights[n] = h;
    kinds[n] = kind;
    return n;
  };
  const link = (a, b) => {
    stairEdge.add(edges.length / 2);
    edges.push(a, b);
  };
  for (const core of stacks) {
    const report = {
      kind: CORE_NAMES[core.kind],
      names: core.names.slice(0, 4),
      sources: core.sources,
      itemIds: core.itemIds.slice(0, 16),
      levels: 0,
      linked: 0,
      modelled: 0,
      unjoinedTreads: 0,
      reason: null,
    };
    coreReport.push(report);
    const lift = core.kind === CORE_LIFT;
    const ring = lift ? 2.0 : 1.0;
    const within = (u, v, pad) => u >= core.u0 - pad && u <= core.u1 + pad && v >= core.v0 - pad && v <= core.v1 + pad;
    const beside = [];
    for (let n = 0; n < stairBase; n += 1) {
      if (kinds[n] !== KIND_FLOOR) continue;
      const [u, v] = lattice2uv(n);
      if (within(u, v, ring)) beside.push(n);
    }
    let levels = levelsOf(beside.map((n) => heights[n]));
    if (!lift) levels = levels.filter((h) => h >= core.lo - 0.6 && h <= core.hi + 0.6);
    report.levels = levels.length;
    const footprint = (core.u1 - core.u0) * (core.v1 - core.v0);
    if (core.kind !== CORE_STAIR && !useLifts) {
      report.reason = "lifts and escalators are off";
      continue;
    }
    if (footprint > 400) {
      report.reason = "footprint over 400 m2";
      continue;
    }
    if (levels.length < 2) {
      report.reason = "fewer than two floor levels beside it";
      continue;
    }
    // The nearest floor node at height h within `radius` of (u, v), not
    // through a wall; failing that, through one if `throughWall`. Returns
    // [node, plan distance], node -1 if none.
    const anchor = (u, v, h, radius, throughWall = false) => {
      const [x, y] = fromUV(u, v);
      let pick = -1, score = Infinity, crossPick = -1, crossScore = Infinity;
      for (const m of beside) {
        if (Math.abs(heights[m] - h) > maxRise) continue;
        const mx = points[m * 3 + h0], my = points[m * 3 + h1];
        const d = Math.hypot(mx - x, my - y);
        if (d > radius || d >= score) continue;
        if (sectionFor(heights[m]).crosses(x, y, mx, my)) {
          if (d < crossScore) {
            crossPick = m;
            crossScore = d;
          }
          continue;
        }
        pick = m;
        score = d;
      }
      if (pick >= 0) return [pick, score];
      return throughWall && crossPick >= 0 ? [crossPick, crossScore] : [-1, Infinity];
    };
    // Anywhere in the footprint or the ring around it (the nodes `beside`).
    const reachIn = Math.hypot(core.u1 - core.u0, core.v1 - core.v0) + ring;
    if (lift) {
      const uc = (core.u0 + core.u1) / 2, vc = (core.v0 + core.v1) / 2;
      const radius = Math.hypot(core.u1 - core.u0, core.v1 - core.v0) / 2 + ring;
      let below = -1, belowH = -Infinity;
      for (const h of levels) {
        const [m] = anchor(uc, vc, h, radius, true);
        if (m < 0 || total + 1 > capacity) continue;
        const stop = addNode(uc, vc, heights[m], KIND_LIFT);
        link(stop, m);
        if (below >= 0 && heights[m] - belowH <= 12) {
          link(below, stop);
          report.linked += 1;
        }
        below = stop;
        belowH = heights[m];
      }
      if (!report.linked) report.reason = "no landing beside it on two levels";
      continue;
    }
    // Whether two levels are already joined here: floor nodes of both in one
    // part of the floor and stair nodes within a metre of the ring, by the
    // links made so far. Treads standing in the core are not enough; they
    // must reach both floors.
    const local = new Map();
    for (let n = 0; n < total; n += 1) {
      if (kinds[n] !== KIND_FLOOR && kinds[n] !== KIND_STAIR) continue;
      const [u, v] = lattice2uv(n);
      if (within(u, v, ring + 1.0)) local.set(n, local.size);
    }
    const root = Int32Array.from({ length: local.size }, (_, i) => i);
    const find = (x) => {
      while (root[x] !== x) x = root[x] = root[root[x]];
      return x;
    };
    for (let e = 0; e < edges.length; e += 2) {
      const a = local.get(edges[e]), b = local.get(edges[e + 1]);
      if (a !== undefined && b !== undefined) root[find(a)] = find(b);
    }
    const joined = (ha, hb) => {
      const parts = new Set();
      for (const [n, i] of local) if (kinds[n] === KIND_FLOOR && Math.abs(heights[n] - ha) <= maxRise) parts.add(find(i));
      for (const [n, i] of local) if (kinds[n] === KIND_FLOOR && Math.abs(heights[n] - hb) <= maxRise && parts.has(find(i))) return true;
      return false;
    };
    for (let k = 0; k + 1 < levels.length; k += 1) {
      const ha = levels[k], hb = levels[k + 1];
      if (hb - ha < 1.0 || hb - ha > 8.0) continue;
      if (joined(ha, hb)) {
        report.modelled += 1;
        continue;
      }
      // Treads that stand here but do not join the two levels (Smiley West,
      // HITOS) leave the floors apart, so the assumed line is added anyway.
      for (let n = stairBase; n < stairBase + patches.length; n += 1) {
        const [u, v] = lattice2uv(n);
        if (within(u, v, 0.5) && heights[n] > ha + 0.3 && heights[n] < hb - 0.3) {
          report.unjoinedTreads += 1;
          break;
        }
      }
      // The line in whichever of its four reflections starts and ends nearest
      // the floors: in a double-height stair hall the upper floor is a
      // landing along one side, not under the line's end.
      let line = null, a = -1, b = -1, best = Infinity;
      for (const mirrorAlong of [false, true]) {
        for (const mirrorAcross of [false, true]) {
          const candidate = walkingLine(core, ha, hb, { step: coreStep, mirrorAlong, mirrorAcross });
          const first = candidate[0], last = candidate[candidate.length - 1];
          const [foot, df] = anchor(first[0], first[1], ha, reachIn);
          const [head, dh] = anchor(last[0], last[1], hb, reachIn);
          if (foot < 0 || head < 0 || df + dh >= best) continue;
          [line, a, b, best] = [candidate, foot, head, df + dh];
        }
      }
      if (!line) {
        report.reason ||= "no floor at the foot or head of the assumed stair";
        continue;
      }
      if (total + line.length > capacity) {
        report.reason ||= "node budget reached";
        break;
      }
      let prev = a;
      for (const [u, v, h] of line) {
        const n = addNode(u, v, h, KIND_STAIR);
        link(prev, n);
        prev = n;
      }
      link(prev, b);
      report.linked += 1;
    }
    if (!report.linked && !report.reason) report.reason = report.modelled ? "modelled stair" : "no two levels 1 to 8 m apart";
  }

  const edgeCount = edges.length / 2;
  const blocked = new Uint8Array(edgeCount);
  for (let e = 0; e < edgeCount; e += 1) {
    if (doorEdge.has(e) || stairEdge.has(e)) continue;
    const a = edges[e * 2], b = edges[e * 2 + 1];
    const ax = points[a * 3 + h0], ay = points[a * 3 + h1];
    const bx = points[b * 3 + h0], by = points[b * 3 + h1];
    const index = sectionFor(Math.min(heights[a], heights[b]));
    // Blocked if it crosses an outline strictly, or if a sample at 1/4, 1/2 or
    // 3/4 lies in solid. The samples catch a link through an outline vertex
    // (where the strict test says no) only when one lands in the solid; a
    // short chord between two vertices can slip between them.
    if (
      index.crosses(ax, ay, bx, by) ||
      index.inside(ax + (bx - ax) * 0.25, ay + (by - ay) * 0.25) ||
      index.inside((ax + bx) / 2, (ay + by) / 2) ||
      index.inside(ax + (bx - ax) * 0.75, ay + (by - ay) * 0.75)
    ) {
      blocked[e] = 1;
    }
  }

  // A diagonal is kept only if one of its two axis detours is open, so it
  // never squeezes between two walls that block both. This makes the
  // no-corner-cutting rule hold with walls as well as missing floor
  // (formal/fireegress Lattice.lean, diagonal_detour); before, only the
  // axis neighbours' existence was checked.
  let cornerBlocked = 0;
  if (diagonals.length) {
    const edgeOf = new Map();
    for (let e = 0; e < edgeCount; e += 1) edgeOf.set(pairKey(edges[e * 2], edges[e * 2 + 1]), e);
    const open = (u, v) => {
      const e = edgeOf.get(pairKey(u, v));
      return e !== undefined && !blocked[e];
    };
    for (const [e, a, c1, c2, b] of diagonals) {
      if (blocked[e]) continue;
      if (!(open(a, c1) && open(c1, b)) && !(open(a, c2) && open(c2, b))) {
        blocked[e] = 1;
        cornerBlocked += 1;
      }
    }
  }

  // Fragments: pieces smaller than minFragment nodes with no door or stair in
  // them, typically slab edges poking out past the exterior walls (FZK-Haus:
  // strips of 11 and 6 nodes outside the facade). Removed and counted.
  let pruned = 0;
  let outEdges = edges;
  let outBlocked = blocked;
  if (minFragment > 1 && total > 0) {
    const parent = Int32Array.from({ length: total }, (_, i) => i);
    const find = (x) => {
      while (parent[x] !== x) x = parent[x] = parent[parent[x]];
      return x;
    };
    for (let e = 0; e < edgeCount; e += 1) {
      if (!blocked[e]) parent[find(edges[e * 2])] = find(edges[e * 2 + 1]);
    }
    const size = new Int32Array(total);
    const special = new Uint8Array(total);
    for (let n = 0; n < total; n += 1) {
      const r = find(n);
      size[r] += 1;
      if (kinds[n] !== KIND_FLOOR) special[r] = 1;
    }
    const remap = new Int32Array(total).fill(-1);
    let next = 0;
    for (let n = 0; n < total; n += 1) {
      const r = find(n);
      if (size[r] >= minFragment || special[r]) remap[n] = next++;
    }
    pruned = total - next;
    if (pruned) {
      for (let n = 0; n < total; n += 1) {
        const m = remap[n];
        if (m < 0) continue;
        points[m * 3] = points[n * 3];
        points[m * 3 + 1] = points[n * 3 + 1];
        points[m * 3 + 2] = points[n * 3 + 2];
        kinds[m] = kinds[n];
        heights[m] = heights[n];
      }
      outEdges = [];
      const keptBlocked = [];
      for (let e = 0; e < edgeCount; e += 1) {
        const a = remap[edges[e * 2]], b = remap[edges[e * 2 + 1]];
        if (a < 0 || b < 0) continue;
        outEdges.push(a, b);
        keptBlocked.push(blocked[e]);
      }
      outBlocked = Uint8Array.from(keptBlocked);
      for (const d of doorReport) d.node = d.node >= 0 ? remap[d.node] : -1;
      total = next;
    }
  }

  const t1 = typeof performance !== "undefined" ? performance.now() : Date.now();
  let stairNodes = 0, doorNodes = 0, liftNodes = 0, blockedCount = 0;
  for (let n = 0; n < total; n += 1) {
    if (kinds[n] === KIND_STAIR) stairNodes += 1;
    else if (kinds[n] === KIND_DOOR) doorNodes += 1;
    else if (kinds[n] === KIND_LIFT) liftNodes += 1;
  }
  for (const b of outBlocked) blockedCount += b;
  return {
    points: points.slice(0, total * 3),
    edges: Uint32Array.from(outEdges),
    kinds: kinds.slice(0, total),
    blocked: outBlocked,
    walls: wallList,
    // The wall sections the links were blocked with, keyed by 10 cm of
    // section height, so routes can be straightened against the same walls.
    sections: [...sections].map(([key, index]) => ({ key, segs: index.segs })),
    sectionHeight,
    doorPoints,
    doorReport,
    // One entry per vertical core: kind, names, sources, levels beside it,
    // links made (storeys joined), storeys skipped for a modelled stair, and
    // why it was not linked.
    cores: coreReport,
    angle,
    spacing: s,
    stats: {
      nodes: total,
      edges: outEdges.length / 2,
      prunedNodes: pruned,
      wallFaceNodes: hugged,
      stairNodes,
      stairEnds,
      stairEndsLinked,
      doorNodes,
      liftNodes,
      cores: coreReport.length,
      coreStairsAssumed: coreReport.filter((c) => c.kind === "stair").reduce((sum, c) => sum + c.linked, 0),
      liftLinks: coreReport.filter((c) => c.kind !== "stair").reduce((sum, c) => sum + c.linked, 0),
      blockedEdges: blockedCount,
      cornerBlocked,
      walls: wallList.length,
      angleDeg: (angle * 180) / Math.PI,
      spacing: s,
      coarsened: s > spacing + 1e-9,
      windingFallback: Boolean(windingFallback),
      samples: samples.i.length,
      ms: t1 - t0,
    },
  };
}

/**
 * Each door item as [x, y, bottom, ux, uy, width, itemId]: plan centre, sill
 * height, the unit direction across the opening, its width, and its IFC id
 * (null when the mesh carries none). Several meshes of one
 * door (frame, leaf, glazing) share an itemId and are pooled. The direction is
 * the principal axis of the door's plan vertices, as for walls.
 */
export function doorWaypoints(meshes, upAxis) {
  const up = axisIndex(upAxis);
  const [h0, h1] = horizontalAxes(upAxis);
  const byItem = new Map();
  for (const mesh of meshes) {
    const p = mesh.positions;
    if (!p || p.length < 9) continue;
    const k = mesh.itemId ?? `mesh:${byItem.size}`;
    let acc = byItem.get(k);
    if (!acc) byItem.set(k, (acc = { xs: [], ys: [], low: Infinity, itemId: mesh.itemId ?? null }));
    for (let i = 0; i + 2 < p.length; i += 3) {
      acc.xs.push(p[i + h0]);
      acc.ys.push(p[i + h1]);
      if (p[i + up] < acc.low) acc.low = p[i + up];
    }
  }
  const out = [];
  for (const { xs, ys, low, itemId } of byItem.values()) {
    const n = xs.length;
    if (!n) continue;
    let mx = 0, my = 0;
    for (let i = 0; i < n; i += 1) {
      mx += xs[i];
      my += ys[i];
    }
    mx /= n;
    my /= n;
    let sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < n; i += 1) {
      sxx += (xs[i] - mx) ** 2;
      sxy += (xs[i] - mx) * (ys[i] - my);
      syy += (ys[i] - my) ** 2;
    }
    const a = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const ux = Math.cos(a), uy = Math.sin(a);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i += 1) {
      const t = (xs[i] - mx) * ux + (ys[i] - my) * uy;
      if (t < lo) lo = t;
      if (t > hi) hi = t;
    }
    // Centre on the opening, not the vertex mean, which a swing leaf skews.
    const shift = (lo + hi) / 2;
    // The IFC id last, so exits can be matched to the door's property sets.
    out.push([mx + ux * shift, my + uy * shift, low, ux, uy, hi - lo, itemId]);
  }
  return out;
}

/**
 * Storey levels: heights of upward-facing floor surfaces. Surfaces are
 * grouped within `step` of each other and weighted by area; groups under
 * `minArea` square metres (landings, steps) are dropped before groups within
 * `gap` are merged, so a landing half way up cannot chain two storeys into
 * one. Ascending.
 */
export function floorLevels(meshes, upAxis = "y", { gap = 1.5, step = 0.3, minArea = 4 } = {}) {
  const up = axisIndex(upAxis);
  const samples = [];
  for (const mesh of meshes || []) {
    const p = mesh.positions, idx = mesh.indices;
    if (!p || !idx) continue;
    for (let t = 0; t + 2 < idx.length; t += 3) {
      const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
      const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
      const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
      const n = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
      const twice = Math.hypot(n[0], n[1], n[2]);
      if (!twice || n[up] / twice < 0.9) continue;
      samples.push([(p[a + up] + p[b + up] + p[c + up]) / 3, twice / 2]);
    }
  }
  samples.sort((x, y) => x[0] - y[0]);
  // Group by height, then keep the groups big enough to be floors.
  const group = (items, within) => {
    const out = [];
    for (const [h, w] of items) {
      const g = out[out.length - 1];
      if (g && h - g.top <= within) {
        g.sum += h * w;
        g.area += w;
        g.top = h;
      } else {
        out.push({ sum: h * w, area: w, top: h });
      }
    }
    return out.map((g) => [g.sum / g.area, g.area]);
  };
  const floors = group(samples, step).filter(([, area]) => area >= minArea);
  return group(floors, gap).map(([h]) => h);
}
