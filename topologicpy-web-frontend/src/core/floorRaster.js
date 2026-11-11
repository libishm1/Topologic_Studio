/**
 * A fine measuring grid per storey (wiki/roadmap/evacuation-map-and-accessibility-audit-plan-2026-11-11.md):
 * the walkable graph stays the route network (its proofs, the fire, the crowd
 * and the slime run on it); this grid measures. Each storey's floor is
 * rasterised at 5 to 10 cm from the IFC's floor faces, cut by the walls and
 * fixed obstacles at 0.75 m over the floor (lattice.js sectionAt, the cut the
 * graph uses), less the stairs, and kept where it joins the graph's floor.
 *
 * - The exact Euclidean distance transform (Felzenszwalb and Huttenlocher
 *   2012, Theory of Computing 8:415-428, doi:10.4086/toc.2012.v008a019): the
 *   clearance at every cell, in linear time.
 * - Openings by a disc (mathematical morphology: Matheron 1975; Serra 1982):
 *   the floor a disc of radius r sweeps is the cells within r of the centres
 *   at least r clear, a second distance transform; a passage at least w wide
 *   is where the disc of diameter w goes; the turning space is the 1.5 m
 *   circle's.
 * - The walk to the nearest exit on 16 neighbours (Dijkstra 1959 on a lattice
 *   whose moves include the knight's), at most 2.75 % longer than the straight
 *   walk in the open (1 / cos(atan(1/2) / 2), formal/fireegress
 *   Sixteen.lean), against 8.24 % on the graph's 8 neighbours; checked by Fast
 *   Marching (Sethian 1996, PNAS 93:1591-1595, doi:10.1073/pnas.93.4.1591),
 *   which solves the Eikonal equation with no lattice directions at all.
 * - Changes of level between cells (a step, a raised floor) from the floor's
 *   own heights, for the wheelchair.
 *
 * Coordinates: plan axes (a, b) are the two that are not up; the grid's cell
 * (i, j) has its centre at (a0 + (i + 0.5) cell, b0 + (j + 0.5) cell).
 */
import { sectionAt } from "./lattice.js";
import { VOID_NAME } from "./spaceFloors.js";

/**
 * Grid defaults: cell size (m), cell budget a storey, the floor band (m)
 * round the storey's level, the heights (m over the floor) the walls and
 * fixed obstacles are cut at (a walking body's band, 0.10 to 1.80 m, so a
 * low wall, a counter or a stair's underside all count), the step a walker
 * takes (m) and a walking body's radius (m; half a shoulder width: the walk
 * keeps its centre this far off obstacles, the configuration space of
 * Lozano-Perez 1983, IEEE Trans. Computers C-32:108-120).
 */
export const RASTER_DEFAULTS = Object.freeze({ cell: 0.05, maxCells: 2.5e6, band: 0.35, cuts: [0.1, 0.45, 0.75, 1.1, 1.45, 1.8], step: 0.35, bodyRadius: 0.25 });

/** The 16 moves: the 8 king moves and the 8 knight moves, with the cells each passes through. */
export const MOVES16 = (() => {
  const out = [];
  for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1], [1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]]) {
    const pass = [];
    if (Math.abs(di) === 1 && Math.abs(dj) === 1) pass.push([di, 0], [0, dj]);
    else if (Math.abs(di) === 1 && Math.abs(dj) === 2) pass.push([0, Math.sign(dj)], [di, Math.sign(dj)]);
    else if (Math.abs(di) === 2 && Math.abs(dj) === 1) pass.push([Math.sign(di), 0], [Math.sign(di), dj]);
    out.push({ di, dj, len: Math.hypot(di, dj), pass });
  }
  return out;
})();

/** The 16-neighbour lattice's worst ratio to the straight walk in the open: 1 / cos(atan(1/2) / 2). */
export const SIXTEEN_BOUND = 1 / Math.cos(Math.atan(0.5) / 2);

const AXES = { x: 0, y: 1, z: 2 };

/** Scan-convert a plan triangle into the grid, calling visit(cell, u, v) with its barycentric weights. */
function fillTriangle(R, ax, ay, bx, by, cx, cy, visit) {
  const { a0, b0, cell, w, h } = R;
  const minI = Math.max(0, Math.floor((Math.min(ax, bx, cx) - a0) / cell));
  const maxI = Math.min(w - 1, Math.floor((Math.max(ax, bx, cx) - a0) / cell));
  const minJ = Math.max(0, Math.floor((Math.min(ay, by, cy) - b0) / cell));
  const maxJ = Math.min(h - 1, Math.floor((Math.max(ay, by, cy) - b0) / cell));
  const d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
  if (Math.abs(d) < 1e-12) return;
  for (let j = minJ; j <= maxJ; j += 1) {
    const y = b0 + (j + 0.5) * cell;
    for (let i = minI; i <= maxI; i += 1) {
      const x = a0 + (i + 0.5) * cell;
      const u = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / d;
      const v = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / d;
      if (u < -1e-9 || v < -1e-9 || u + v > 1 + 1e-9) continue;
      visit(j * w + i, u, v);
    }
  }
}

/**
 * Mark the cells a plan segment passes through the inside of (sampled at a
 * quarter cell), so a panel thinner than a cell still blocks; a stretch lying
 * on a grid line marks neither side (the outline's fill decides it), or every
 * wall face on a line would narrow the floor by a cell.
 */
function drawSegment(R, mask, x0, y0, x1, y1) {
  const { a0, b0, cell, w, h } = R;
  const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / (cell / 4)));
  for (let s = 0; s <= steps; s += 1) {
    const t = s / steps;
    const u = (x0 + t * (x1 - x0) - a0) / cell, v = (y0 + t * (y1 - y0) - b0) / cell;
    // On a grid line within a thousandth of a cell: model coordinates come as 32-bit floats.
    if (Math.abs(u - Math.round(u)) < 1e-3 || Math.abs(v - Math.round(v)) < 1e-3) continue;
    const i = Math.floor(u), j = Math.floor(v);
    if (i >= 0 && j >= 0 && i < w && j < h) mask[j * w + i] = 1;
  }
}

/** Mark the cells whose centres lie inside a closed outline of plan segments (even-odd, row by row). */
function fillOutline(R, mask, segs) {
  const { a0, b0, cell, w, h } = R;
  let lo = Infinity, hi = -Infinity;
  for (let s = 0; s < segs.length; s += 4) {
    lo = Math.min(lo, segs[s + 1], segs[s + 3]);
    hi = Math.max(hi, segs[s + 1], segs[s + 3]);
  }
  const j0 = Math.max(0, Math.floor((lo - b0) / cell)), j1 = Math.min(h - 1, Math.floor((hi - b0) / cell));
  const xs = [];
  for (let j = j0; j <= j1; j += 1) {
    const yc = b0 + (j + 0.5) * cell;
    xs.length = 0;
    for (let s = 0; s < segs.length; s += 4) {
      const ya = segs[s + 1], yb = segs[s + 3];
      if (ya <= yc === yb <= yc) continue;
      xs.push(segs[s] + ((yc - ya) * (segs[s + 2] - segs[s])) / (yb - ya));
    }
    xs.sort((p, q) => p - q);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const i0 = Math.max(0, Math.ceil((xs[k] - a0) / cell - 0.5)), i1 = Math.min(w - 1, Math.floor((xs[k + 1] - a0) / cell - 0.5));
      for (let i = i0; i <= i1; i += 1) mask[j * w + i] = 1;
    }
  }
}

/**
 * The exact squared Euclidean distance transform (Felzenszwalb and
 * Huttenlocher 2012): for every cell, the squared distance in cells to the
 * nearest cell where `inside` is 0. One-dimensional lower envelopes of
 * parabolas, by columns then rows.
 */
export function edt(inside, w, h) {
  const INF = 1e20;
  const out = new Float64Array(w * h);
  for (let k = 0; k < w * h; k += 1) out[k] = inside[k] ? INF : 0;
  const n = Math.max(w, h);
  const f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
  const pass = (len) => {
    let k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    for (let q = 1; q < len; q += 1) {
      let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      while (s <= z[k]) {
        k -= 1;
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      }
      k += 1;
      v[k] = q;
      z[k] = s;
      z[k + 1] = INF;
    }
    k = 0;
    for (let q = 0; q < len; q += 1) {
      while (z[k + 1] < q) k += 1;
      d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
    }
  };
  for (let i = 0; i < w; i += 1) {
    for (let j = 0; j < h; j += 1) f[j] = out[j * w + i];
    pass(h);
    for (let j = 0; j < h; j += 1) out[j * w + i] = d[j];
  }
  for (let j = 0; j < h; j += 1) {
    for (let i = 0; i < w; i += 1) f[i] = out[j * w + i];
    pass(w);
    for (let i = 0; i < w; i += 1) out[j * w + i] = d[i];
  }
  return out;
}

/**
 * Clearance (m) at each walkable cell: the distance to the nearest cell that
 * is not walkable, less half a cell (the obstacle's face is at its cell's
 * edge, not its centre). A corridor N cells wide has its centre's clearance
 * N / 2 cells: twice the clearance is the width.
 */
export function clearanceOf(R, walk = R.walk) {
  const sq = edt(walk, R.w, R.h);
  const out = new Float32Array(R.w * R.h);
  for (let k = 0; k < out.length; k += 1) out[k] = walk[k] ? Math.sqrt(sq[k]) * R.cell - R.cell / 2 : 0;
  return out;
}

/**
 * The opening by a disc of radius r (m): the walkable cells a disc of radius
 * r reaches with all of it clear, which are those within r of a centre at
 * least r clear. `allowed` (optional) limits the centres, `clear` the
 * clearance (clearanceOf).
 */
export function opening(R, clear, r, allowed = null) {
  const { w, h, cell } = R;
  const notCentre = new Uint8Array(w * h);
  let any = false;
  for (let k = 0; k < w * h; k += 1) {
    const centre = clear[k] >= r - 1e-9 && (!allowed || allowed[k]);
    notCentre[k] = centre ? 0 : 1;
    if (centre) any = true;
  }
  const out = new Uint8Array(w * h);
  if (!any) return out;
  const sq = edt(notCentre, w, h);
  const reach = (r + cell / 2) / cell;
  const r2 = reach * reach;
  for (let k = 0; k < w * h; k += 1) out[k] = R.walk[k] && sq[k] <= r2 + 1e-9 ? 1 : 0;
  return out;
}

/** An indexed binary min-heap over cells, keyed by an outside array, with decrease-key. */
export class CellHeap {
  constructor(n, keys) {
    this.keys = keys;
    this.heap = new Int32Array(n);
    this.pos = new Int32Array(n).fill(-1);
    this.size = 0;
  }
  up(i) {
    const { heap, pos, keys } = this;
    const c = heap[i], key = keys[c];
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (keys[heap[p]] <= key) break;
      heap[i] = heap[p];
      pos[heap[i]] = i;
      i = p;
    }
    heap[i] = c;
    pos[c] = i;
  }
  push(c) {
    if (this.pos[c] >= 0) this.up(this.pos[c]);
    else {
      this.heap[this.size] = c;
      this.pos[c] = this.size;
      this.up(this.size++);
    }
  }
  pop() {
    const { heap, pos, keys } = this;
    const top = heap[0];
    pos[top] = -2;
    const n = --this.size;
    if (n > 0) {
      const c = heap[n], key = keys[c];
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        if (l >= n) break;
        const r = l + 1;
        const m = r < n && keys[heap[r]] < keys[heap[l]] ? r : l;
        if (keys[heap[m]] >= key) break;
        heap[i] = heap[m];
        pos[heap[i]] = i;
        i = m;
      }
      heap[i] = c;
      pos[c] = i;
    }
    return top;
  }
}

/**
 * The walk from the nearest source on 16 neighbours (m), the source each
 * cell reaches and the cell before it. A move needs its end and the cells it
 * passes through `passable`, and a change of floor height no more than
 * max(levelChange, slope x its length) (the wheelchair's: a 13 mm lip, a 1 in
 * 12 ramp) or `step` (walking).
 */
export function gridField(R, sources, { passable = R.walk, levelChange = Infinity, slope = 0, waive = null } = {}) {
  const { w, h, cell, height } = R;
  const N = w * h;
  const dist = new Float32Array(N).fill(Infinity);
  const source = new Int32Array(N).fill(-1);
  const parent = new Int32Array(N).fill(-1);
  const heap = new CellHeap(N, dist);
  // Each source cell starts with its own step still to walk to the exit (`offsets`, m), not at
  // nought: a disc of cells at nought credits everyone in it with a walk never made (Poux 2026).
  for (const { cells, id, offsets = null } of sources) {
    cells.forEach((c, k) => {
      const d0 = offsets ? offsets[k] : 0;
      if (!passable[c] || d0 >= dist[c]) return;
      dist[c] = d0;
      source[c] = id;
      heap.push(c);
    });
  }
  while (heap.size) {
    const c = heap.pop();
    const i = c % w, j = (c - i) / w;
    const dc = dist[c];
    for (const m of MOVES16) {
      const ni = i + m.di, nj = j + m.dj;
      if (ni < 0 || nj < 0 || ni >= w || nj >= h) continue;
      const nc = nj * w + ni;
      if (!passable[nc] || heap.pos[nc] === -2) continue;
      let ok = true;
      for (const [pi, pj] of m.pass) if (!passable[(j + pj) * w + (i + pi)]) ok = false;
      if (!ok) continue;
      // `waive`: cells where the level rule does not apply (a ramp proposed at a step).
      const rise = Math.abs(height[nc] - height[c]);
      if (rise > Math.max(levelChange, slope * m.len * cell) + 1e-9 && !(waive && waive[c] && waive[nc])) continue;
      const nd = dc + m.len * cell;
      if (nd < dist[nc]) {
        dist[nc] = nd;
        source[nc] = source[c];
        parent[nc] = c;
        heap.push(nc);
      }
    }
  }
  return { dist, source, parent };
}

/**
 * Fast Marching (Sethian 1996): the first-arrival time of a front at unit
 * speed from the sources, first-order upwind on the four axis neighbours;
 * isotropic, with no lattice directions. The check on gridField's walk.
 */
export function fastMarching(R, sources, { passable = R.walk } = {}) {
  const { w, h, cell } = R;
  const N = w * h;
  const T = new Float32Array(N).fill(Infinity);
  const heap = new CellHeap(N, T);
  // Sources start with their own step to the exit, as gridField's.
  for (const { cells, offsets = null } of sources) {
    cells.forEach((c, k) => {
      const t0 = offsets ? offsets[k] : 0;
      if (!passable[c] || t0 >= T[c]) return;
      T[c] = t0;
      heap.push(c);
    });
  }
  const solve = (c) => {
    const i = c % w, j = (c - i) / w;
    const left = i > 0 && heap.pos[c - 1] === -2 ? T[c - 1] : Infinity;
    const right = i < w - 1 && heap.pos[c + 1] === -2 ? T[c + 1] : Infinity;
    const down = j > 0 && heap.pos[c - w] === -2 ? T[c - w] : Infinity;
    const upv = j < h - 1 && heap.pos[c + w] === -2 ? T[c + w] : Infinity;
    const a = Math.min(left, right), b = Math.min(down, upv);
    if (!Number.isFinite(a) && !Number.isFinite(b)) return Infinity;
    if (!Number.isFinite(a) || !Number.isFinite(b) || Math.abs(a - b) >= cell) return Math.min(a, b) + cell;
    return (a + b + Math.sqrt(2 * cell * cell - (a - b) * (a - b))) / 2;
  };
  while (heap.size) {
    const c = heap.pop();
    const i = c % w, j = (c - i) / w;
    for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const ni = i + di, nj = j + dj;
      if (ni < 0 || nj < 0 || ni >= w || nj >= h) continue;
      const nc = nj * w + ni;
      if (!passable[nc] || heap.pos[nc] === -2) continue;
      const t = solve(nc);
      if (t < T[nc]) {
        T[nc] = t;
        heap.push(nc);
      }
    }
  }
  return T;
}

/** The cells within `radius` (m) of a plan point that pass `mask`, with their straight distance to it. */
export function cellsNear(R, x, y, radius, mask = R.walk) {
  const { a0, b0, cell, w, h } = R;
  const cells = [], offsets = [];
  const r = Math.ceil(radius / cell);
  const ci = Math.floor((x - a0) / cell), cj = Math.floor((y - b0) / cell);
  for (let j = cj - r; j <= cj + r; j += 1) {
    for (let i = ci - r; i <= ci + r; i += 1) {
      if (i < 0 || j < 0 || i >= w || j >= h) continue;
      const k = j * w + i;
      const d = Math.hypot(a0 + (i + 0.5) * cell - x, b0 + (j + 0.5) * cell - y);
      if (mask[k] && d <= radius) {
        cells.push(k);
        offsets.push(d);
      }
    }
  }
  return { cells, offsets };
}

/** The plan centre of a cell. */
export const cellCentre = (R, k) => [R.a0 + ((k % R.w) + 0.5) * R.cell, R.b0 + (Math.floor(k / R.w) + 0.5) * R.cell];

/**
 * One storey's measuring grid.
 *
 * @param {object} geometry  meshes in the viewer frame: floors, spaces (with itemId), walls, obstacles, stairs
 * @param {object} opts  level (the storey's floor height), upAxis, seeds: plan points of the graph's floor
 *                       nodes on the storey (the walkable floor joins them), rooms (roomsFromSpaces) for
 *                       each cell's room, spaceNames (itemId -> name) to leave voids out, cell, maxCells,
 *                       band, sectionHeight, step
 * @returns {object} {a0, b0, cell, w, h, level, height, walk, stair, room, clear}
 */
export function storeyRaster(geometry, opts = {}) {
  const { level = 0, upAxis = "y", seeds = [], rooms = null, spaceNames = null } = opts;
  const { cell: minCell, maxCells, band, cuts, step } = { ...RASTER_DEFAULTS, ...opts };
  const up = AXES[upAxis] ?? 1;
  const [ha, hb] = [0, 1, 2].filter((c) => c !== up);
  // Bounds: the storey's graph nodes, with a margin.
  let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
  for (const [x, y] of seeds) {
    a0 = Math.min(a0, x);
    a1 = Math.max(a1, x);
    b0 = Math.min(b0, y);
    b1 = Math.max(b1, y);
  }
  if (!(a1 >= a0)) return null;
  const margin = 1;
  a0 -= margin;
  b0 -= margin;
  a1 += margin;
  b1 += margin;
  const cell = Math.max(minCell, Math.sqrt(((a1 - a0) * (b1 - b0)) / maxCells));
  const w = Math.ceil((a1 - a0) / cell), h = Math.ceil((b1 - b0) / cell);
  const R = { a0, b0, cell, w, h, level, up, ha, hb };
  const N = w * h;
  const height = new Float32Array(N).fill(-Infinity);
  const minUp = Math.cos((12 * Math.PI) / 180);
  const isVoid = (m) => spaceNames && VOID_NAME.test(spaceNames.get?.(m.itemId) || "");
  const floorFaces = (meshes) => {
    for (const m of meshes || []) {
      if (!m?.positions || !m?.indices || isVoid(m)) continue;
      const p = m.positions, ix = m.indices;
      for (let t = 0; t + 2 < ix.length; t += 3) {
        const A = ix[t] * 3, B = ix[t + 1] * 3, C = ix[t + 2] * 3;
        const za = p[A + up], zb = p[B + up], zc = p[C + up];
        if (Math.max(za, zb, zc) > level + band || Math.min(za, zb, zc) < level - band) continue;
        const e1 = [p[B] - p[A], p[B + 1] - p[A + 1], p[B + 2] - p[A + 2]];
        const e2 = [p[C] - p[A], p[C + 1] - p[A + 1], p[C + 2] - p[A + 2]];
        const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const len = Math.hypot(n[0], n[1], n[2]);
        if (len < 1e-12 || Math.abs(n[up]) / len < minUp) continue;
        fillTriangle(R, p[A + ha], p[A + hb], p[B + ha], p[B + hb], p[C + ha], p[C + hb], (k, u, v) => {
          const z = u * za + v * zb + (1 - u - v) * zc;
          if (z > height[k]) height[k] = z;
        });
      }
    }
  };
  floorFaces(geometry.floors);
  floorFaces(geometry.spaces);
  // Walls and fixed obstacles cut across a walking body's band over this floor.
  const solid = new Uint8Array(N);
  const blockers = [...(geometry.walls || []), ...(geometry.obstacles || [])];
  for (const mesh of blockers) {
    for (const z of cuts) {
      // Each element's own outline, filled on its own: overlapping elements would cancel under even-odd.
      const cut = sectionAt([mesh], level + z, up, ha, hb);
      if (!cut.length) continue;
      fillOutline(R, solid, cut);
      for (let s = 0; s < cut.length; s += 4) drawSegment(R, solid, cut[s], cut[s + 1], cut[s + 2], cut[s + 3]);
    }
  }
  // Doors: some exporters leave the wall whole at a door, and a model whose
  // floors are its spaces has none under the doorway. Each door's plan
  // rectangle (its leaf's main axis, by principal axes), widened across the
  // wall, is cut from the obstacles and given floor, as a scan's doors are
  // opened at their leaf (Poux 2026); the graph links through the same doors.
  for (const m of geometry.doors || []) {
    if (!m?.positions) continue;
    const p = m.positions;
    let lo = Infinity, cx = 0, cy = 0, count = 0;
    for (let o = 0; o + 2 < p.length; o += 3) {
      lo = Math.min(lo, p[o + up]);
      cx += p[o + ha];
      cy += p[o + hb];
      count += 1;
    }
    if (!count || Math.abs(lo - level) > 0.5) continue;
    cx /= count;
    cy /= count;
    let sxx = 0, syy = 0, sxy = 0;
    for (let o = 0; o + 2 < p.length; o += 3) {
      const dx = p[o + ha] - cx, dy = p[o + hb] - cy;
      sxx += dx * dx;
      syy += dy * dy;
      sxy += dx * dy;
    }
    const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const ux = Math.cos(angle), uy = Math.sin(angle), vx = -uy, vy = ux;
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (let o = 0; o + 2 < p.length; o += 3) {
      const dx = p[o + ha] - cx, dy = p[o + hb] - cy;
      const u = dx * ux + dy * uy, v = dx * vx + dy * vy;
      u0 = Math.min(u0, u);
      u1 = Math.max(u1, u);
      v0 = Math.min(v0, v);
      v1 = Math.max(v1, v);
    }
    const across = Math.max(0.35, (v1 - v0) / 2 + 0.25);
    const vc = (v0 + v1) / 2;
    const reach = Math.hypot(u1 - u0, 2 * across) / 2 + cell;
    const i0 = Math.max(0, Math.floor((cx - reach - a0) / cell)), i1 = Math.min(w - 1, Math.floor((cx + reach - a0) / cell));
    const j0 = Math.max(0, Math.floor((cy - reach - b0) / cell)), j1 = Math.min(h - 1, Math.floor((cy + reach - b0) / cell));
    for (let j = j0; j <= j1; j += 1) {
      for (let i = i0; i <= i1; i += 1) {
        const dx = a0 + (i + 0.5) * cell - cx, dy = b0 + (j + 0.5) * cell - cy;
        const u = dx * ux + dy * uy, v = dx * vx + dy * vy - vc;
        if (u < u0 || u > u1 || Math.abs(v) > across) continue;
        const k = j * w + i;
        solid[k] = 0;
        if (!Number.isFinite(height[k])) height[k] = level;
      }
    }
  }
  // Stairs: their faces off this floor's level (treads above or below it).
  const stair = new Uint8Array(N);
  for (const m of geometry.stairs || []) {
    if (!m?.positions || !m?.indices) continue;
    const p = m.positions, ix = m.indices;
    for (let t = 0; t + 2 < ix.length; t += 3) {
      const A = ix[t] * 3, B = ix[t + 1] * 3, C = ix[t + 2] * 3;
      const zs = [p[A + up], p[B + up], p[C + up]];
      const lo = Math.min(...zs), hi = Math.max(...zs);
      if (hi < level - 3 || lo > level + 3 || (Math.abs(lo - level) < 0.05 && Math.abs(hi - level) < 0.05)) continue;
      if (hi < level - 0.05 && lo < level - 2.5) continue;
      fillTriangle(R, p[A + ha], p[A + hb], p[B + ha], p[B + hb], p[C + ha], p[C + hb], (k) => {
        stair[k] = 1;
      });
    }
  }
  // Walkable: floor, not solid, not a stair, joined to the graph's floor (8 neighbours, steps up to `step`).
  const free = new Uint8Array(N);
  for (let k = 0; k < N; k += 1) free[k] = Number.isFinite(height[k]) && !solid[k] && !stair[k] ? 1 : 0;
  const walk = new Uint8Array(N);
  const stack = [];
  for (const [x, y] of seeds) {
    const i = Math.floor((x - a0) / cell), j = Math.floor((y - b0) / cell);
    // The node may sit on a cut wall face or a stair's edge: start from the nearest free cell within 0.3 m.
    let best = -1, bestD = Infinity;
    const r = Math.ceil(0.3 / cell);
    for (let dj = -r; dj <= r; dj += 1) {
      for (let di = -r; di <= r; di += 1) {
        const ii = i + di, jj = j + dj;
        if (ii < 0 || jj < 0 || ii >= w || jj >= h) continue;
        const k = jj * w + ii;
        const d = di * di + dj * dj;
        if (free[k] && d < bestD) {
          bestD = d;
          best = k;
        }
      }
    }
    if (best >= 0 && !walk[best]) {
      walk[best] = 1;
      stack.push(best);
    }
  }
  while (stack.length) {
    const c = stack.pop();
    const i = c % w, j = (c - i) / w;
    for (let dj = -1; dj <= 1; dj += 1) {
      for (let di = -1; di <= 1; di += 1) {
        if (!di && !dj) continue;
        const ni = i + di, nj = j + dj;
        if (ni < 0 || nj < 0 || ni >= w || nj >= h) continue;
        const nc = nj * w + ni;
        if (walk[nc] || !free[nc] || Math.abs(height[nc] - height[c]) > step) continue;
        if (di && dj && (!free[j * w + ni] || !free[nj * w + i])) continue;
        walk[nc] = 1;
        stack.push(nc);
      }
    }
  }
  R.height = height;
  R.walk = walk;
  R.stair = stair;
  // Each cell's room: the smallest room whose footprint holds it, its floor within the band.
  const room = new Int32Array(N).fill(-1);
  if (rooms?.length) {
    const order = rooms.map((r, i) => i).filter((i) => rooms[i]?.footprint && Math.abs(rooms[i].floor - level) <= band).sort((x, y) => rooms[y].area - rooms[x].area);
    for (const i of order) {
      const T = rooms[i].footprint;
      for (let t = 0; t < T.length; t += 6) fillTriangle(R, T[t], T[t + 1], T[t + 2], T[t + 3], T[t + 4], T[t + 5], (k) => {
        room[k] = i;
      });
    }
  }
  R.room = room;
  R.clear = clearanceOf(R);
  // Where a walking body's centre can be: clear by its radius.
  const body = opts.bodyRadius ?? RASTER_DEFAULTS.bodyRadius;
  const centre = new Uint8Array(N);
  let cells = 0, centres = 0;
  for (let k = 0; k < N; k += 1) {
    cells += walk[k];
    if (walk[k] && R.clear[k] >= body - 1e-9) {
      centre[k] = 1;
      centres += 1;
    }
  }
  R.body = centre;
  R.walkArea = cells * cell * cell;
  R.bodyArea = centres * cell * cell;
  return R;
}

/** Clear-width classes (m) the grid reports: a cell is at least the widest class whose disc passes over it. */
export const WIDTH_CLASSES = [0.45, 0.6, 0.7, 0.8, 0.815, 0.9, 0.915, 1.0, 1.2, 1.5, 1.525, 2.0];

/** Each walkable cell's clear-width class (an index into `classes`, -1 narrower than the first). */
export function widthClasses(R, classes = WIDTH_CLASSES) {
  const out = new Int8Array(R.w * R.h).fill(-1);
  classes.forEach((wd, k) => {
    const open = opening(R, R.clear, wd / 2);
    for (let c = 0; c < out.length; c += 1) if (open[c]) out[c] = k;
  });
  return out;
}

/** Statistics of one field against another where both are finite: mean and largest relative difference. */
export function compareFields(a, b, mask) {
  let sum = 0, count = 0, max = 0;
  for (let k = 0; k < a.length; k += 1) {
    if (!mask[k] || !Number.isFinite(a[k]) || !Number.isFinite(b[k]) || b[k] < 1) continue;
    const r = (a[k] - b[k]) / b[k];
    sum += r;
    count += 1;
    if (Math.abs(r) > Math.abs(max)) max = r;
  }
  return { mean: count ? sum / count : 0, max, cells: count };
}
