/**
 * Geodesic ground truth for route lengths (tools/benchmark.mjs).
 *
 * The free plane at one floor is the plane at section height minus the solid
 * the lattice itself was blocked with: the wall and obstacle sections that
 * buildLatticeGraph returns (`sections`, keyed by 10 cm of floor height), with
 * "solid" decided by the lattice's own SegmentIndex.inside. The shortest path
 * between two points of it bends only at outline vertices, so it is found
 * exactly by a shortest-path search over the visibility graph of the outline
 * corners plus the two end points (Lozano-Perez and Wesley 1979), restricted
 * to bitangent links. Links are tested lazily, A* with the straight-line
 * heuristic, and only corners inside the ellipse |s c| + |c t| <= U are
 * considered, U an upper bound (the grid route's plan length).
 *
 * Visibility is decided permissively, so that an error can only make the
 * geodesic shorter, never longer, and so can never invent a violation of
 * "a route along free links is never shorter than the geodesic"
 * (formal/fireegress/FireEgress/Eikonal.lean, graph_front_never_early):
 *   - a segment is blocked if it crosses an outline segment at a point
 *     strictly inside both (the lattice's own test, Segments.lean);
 *   - or if, between two consecutive outline vertices it touches, its middle
 *     is inside solid by SegmentIndex.inside and more than 1 mm from every
 *     outline segment.
 * Where outlines overlap (two wall meshes through each other) the parity test
 * calls the overlap free, as the lattice does; that too only shortens the
 * geodesic. Floor edges are not obstacles here: a geodesic may cut across a
 * void or outside the slab, which also only shortens it.
 */
import { SegmentIndex } from "../topologicpy-web-frontend/src/core/lattice.js";

const CELL = 0.5;
const SNAP = 1e5; // vertices merged to 10 micrometres

function properCross(ax, ay, bx, by, cx, cy, dx, dy) {
  const d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx);
  const d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);
  const d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  const d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
  const opposite = (u, v) => (u > 0 && v < 0) || (u < 0 && v > 0);
  return opposite(d1, d2) && opposite(d3, d4);
}

function pointSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
  const x = ax + t * dx - px, y = ay + t * dy - py;
  return Math.sqrt(x * x + y * y);
}

/** A min-heap of (priority, id) on typed arrays, for the geodesic search only. */
class MinHeap {
  constructor() {
    this.p = [];
    this.v = [];
  }
  get size() {
    return this.p.length;
  }
  push(pr, id) {
    const p = this.p, v = this.v;
    let i = p.length;
    p.push(pr);
    v.push(id);
    while (i > 0) {
      const up = (i - 1) >> 1;
      if (p[up] <= pr) break;
      p[i] = p[up];
      v[i] = v[up];
      i = up;
    }
    p[i] = pr;
    v[i] = id;
  }
  pop() {
    const p = this.p, v = this.v;
    const top = v[0];
    const lp = p.pop(), lv = v.pop();
    const n = p.length;
    if (n) {
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = -1, best = lp;
        if (l < n && p[l] < best) {
          m = l;
          best = p[l];
        }
        if (r < n && p[r] < best) m = r;
        if (m < 0) break;
        p[i] = p[m];
        v[i] = v[m];
        i = m;
      }
      p[i] = lp;
      v[i] = lv;
    }
    return top;
  }
}

export class FreePlane {
  /** @param {Float64Array} segs  outline segments x0, y0, x1, y1 in plan */
  constructor(segs) {
    this.segs = segs;
    const ns = segs.length / 4;
    this.index = new SegmentIndex(segs);
    // A grid of its own for segment queries along a line.
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < segs.length; i += 2) {
      x0 = Math.min(x0, segs[i]);
      x1 = Math.max(x1, segs[i]);
      y0 = Math.min(y0, segs[i + 1]);
      y1 = Math.max(y1, segs[i + 1]);
    }
    if (!ns) x0 = y0 = x1 = y1 = 0;
    this.ox = x0 - CELL;
    this.oy = y0 - CELL;
    this.W = Math.max(1, Math.ceil((x1 - this.ox) / CELL) + 2);
    this.H = Math.max(1, Math.ceil((y1 - this.oy) / CELL) + 2);
    const count = new Int32Array(this.W * this.H + 1);
    const each = (s, fn) => {
      const i0 = this.ci(Math.min(segs[s * 4], segs[s * 4 + 2])), i1 = this.ci(Math.max(segs[s * 4], segs[s * 4 + 2]));
      const j0 = this.cj(Math.min(segs[s * 4 + 1], segs[s * 4 + 3])), j1 = this.cj(Math.max(segs[s * 4 + 1], segs[s * 4 + 3]));
      for (let i = i0; i <= i1; i += 1) for (let j = j0; j <= j1; j += 1) fn(j * this.W + i);
    };
    for (let s = 0; s < ns; s += 1) each(s, (c) => (count[c + 1] += 1));
    for (let c = 0; c < this.W * this.H; c += 1) count[c + 1] += count[c];
    this.start = count;
    this.list = new Int32Array(count[this.W * this.H]);
    const fill = count.slice(0, this.W * this.H);
    for (let s = 0; s < ns; s += 1) each(s, (c) => (this.list[fill[c]++] = s));
    this.stamp = new Int32Array(ns);
    this.tick = 0;
    this.near = new Int32Array(Math.max(16, ns));

    // Outline vertices, merged, with their outline neighbours.
    const ids = new Map();
    const vx = [], vy = [], nb = [];
    const vid = (x, y) => {
      const k = `${Math.round(x * SNAP)},${Math.round(y * SNAP)}`;
      let id = ids.get(k);
      if (id === undefined) {
        id = vx.length;
        ids.set(k, id);
        vx.push(x);
        vy.push(y);
        nb.push([]);
      }
      return id;
    };
    for (let s = 0; s < ns; s += 1) {
      const a = vid(segs[s * 4], segs[s * 4 + 1]), b = vid(segs[s * 4 + 2], segs[s * 4 + 3]);
      if (a === b) continue;
      if (!nb[a].includes(b)) nb[a].push(b);
      if (!nb[b].includes(a)) nb[b].push(a);
    }
    // Corners: every vertex except the middle of a straight run. A vertex on
    // exactly two outline segments is tested for tangency (bitangent links
    // only); any other keeps every link, so odd junctions cannot hide a path.
    const cx = [], cy = [], n1x = [], n1y = [], n2x = [], n2y = [], tangent = [];
    for (let v = 0; v < vx.length; v += 1) {
      const list = nb[v];
      if (list.length === 2) {
        const ax = vx[list[0]] - vx[v], ay = vy[list[0]] - vy[v];
        const bx = vx[list[1]] - vx[v], by = vy[list[1]] - vy[v];
        const la = Math.hypot(ax, ay), lb = Math.hypot(bx, by);
        if (Math.abs(ax * by - ay * bx) <= 1e-9 * la * lb && ax * bx + ay * by < 0) continue;
        n1x.push(ax);
        n1y.push(ay);
        n2x.push(bx);
        n2y.push(by);
        tangent.push(1);
      } else {
        n1x.push(0);
        n1y.push(0);
        n2x.push(0);
        n2y.push(0);
        tangent.push(0);
      }
      cx.push(vx[v]);
      cy.push(vy[v]);
    }
    this.cx = Float64Array.from(cx);
    this.cy = Float64Array.from(cy);
    this.n1x = Float64Array.from(n1x);
    this.n1y = Float64Array.from(n1y);
    this.n2x = Float64Array.from(n2x);
    this.n2y = Float64Array.from(n2y);
    this.tangent = Uint8Array.from(tangent);
    this.vertices = vx.length;
    this.checks = 0;
  }

  ci(x) {
    return Math.min(this.W - 1, Math.max(0, Math.floor((x - this.ox) / CELL)));
  }

  cj(y) {
    return Math.min(this.H - 1, Math.max(0, Math.floor((y - this.oy) / CELL)));
  }

  /** Outline segments near the segment a-b, each once; returns their count in this.near. */
  gather(ax, ay, bx, by) {
    this.tick += 1;
    let k = 0;
    const len = Math.hypot(bx - ax, by - ay);
    const pieces = Math.max(1, Math.ceil(len / CELL));
    const e = 1e-7;
    for (let q = 0; q < pieces; q += 1) {
      const t0 = q / pieces, t1 = (q + 1) / pieces;
      const px = ax + (bx - ax) * t0, py = ay + (by - ay) * t0;
      const qx = ax + (bx - ax) * t1, qy = ay + (by - ay) * t1;
      const i0 = this.ci(Math.min(px, qx) - e), i1 = this.ci(Math.max(px, qx) + e);
      const j0 = this.cj(Math.min(py, qy) - e), j1 = this.cj(Math.max(py, qy) + e);
      for (let i = i0; i <= i1; i += 1) {
        for (let j = j0; j <= j1; j += 1) {
          const c = j * this.W + i;
          for (let z = this.start[c]; z < this.start[c + 1]; z += 1) {
            const s = this.list[z];
            if (this.stamp[s] === this.tick) continue;
            this.stamp[s] = this.tick;
            this.near[k++] = s;
          }
        }
      }
    }
    return k;
  }

  /** Whether (x, y) is in solid by the lattice's rule and more than 1 mm inside. */
  solid(x, y) {
    if (!this.index.inside(x, y)) return false;
    const r = 1e-3;
    const g = this.segs;
    let close = false;
    this.index.near(x - r, y - r, x + r, y + r, (s) => {
      if (!close && pointSegment(x, y, g[s * 4], g[s * 4 + 1], g[s * 4 + 2], g[s * 4 + 3]) <= r) close = true;
    });
    return !close;
  }

  /** Whether the straight segment a-b lies in the free plane (permissive; see the header). */
  visible(ax, ay, bx, by) {
    this.checks += 1;
    const g = this.segs;
    const k = this.gather(ax, ay, bx, by);
    const rx = bx - ax, ry = by - ay;
    const len2 = rx * rx + ry * ry;
    if (len2 < 1e-18) return true;
    const len = Math.sqrt(len2);
    const touches = [0, 1];
    for (let z = 0; z < k; z += 1) {
      const s = this.near[z] * 4;
      if (properCross(ax, ay, bx, by, g[s], g[s + 1], g[s + 2], g[s + 3])) return false;
      for (let end = 0; end < 4; end += 2) {
        const ex = g[s + end] - ax, ey = g[s + end + 1] - ay;
        if (Math.abs(rx * ey - ry * ex) / len > 1e-7) continue;
        const t = (ex * rx + ey * ry) / len2;
        if (t > 1e-9 && t < 1 - 1e-9) touches.push(t);
      }
    }
    touches.sort((a, b) => a - b);
    for (let i = 1; i < touches.length; i += 1) {
      if ((touches[i] - touches[i - 1]) * len < 1e-6) continue;
      const t = (touches[i] + touches[i - 1]) / 2;
      if (this.solid(ax + rx * t, ay + ry * t)) return false;
    }
    return true;
  }

  /** Whether the line through corner c and (px, py) touches the outline at c without entering it. */
  tangentAt(c, px, py) {
    if (!this.tangent[c]) return true;
    const dx = this.cx[c] - px, dy = this.cy[c] - py;
    const s1 = dx * this.n1y[c] - dy * this.n1x[c];
    const s2 = dx * this.n2y[c] - dy * this.n2x[c];
    const tol = 1e-12 * Math.hypot(dx, dy);
    return !((s1 > tol && s2 < -tol) || (s1 < -tol && s2 > tol));
  }

  /**
   * Shortest free path from s to t no longer than `bound`.
   *
   * @returns {{length: number, corners: number, status: string, checks: number}}
   *          status "direct" (s sees t), "found", "beyond bound" (no free
   *          path within it), or "capped" (the search budget ran out).
   */
  geodesic(sx, sy, tx, ty, { bound = Infinity, maxChecks = 400000, maxCandidates = 20000 } = {}) {
    const before = this.checks;
    const direct = Math.hypot(tx - sx, ty - sy);
    if (this.visible(sx, sy, tx, ty)) return { length: direct, corners: 0, status: "direct", checks: this.checks - before };
    const U = bound * (1 + 1e-9) + 1e-9;
    const cand = [];
    for (let c = 0; c < this.cx.length; c += 1) {
      const a = Math.hypot(this.cx[c] - sx, this.cy[c] - sy) + Math.hypot(tx - this.cx[c], ty - this.cy[c]);
      if (a <= U) cand.push(c);
    }
    if (cand.length > maxCandidates) return { length: NaN, corners: cand.length, status: "capped", checks: this.checks - before };
    // Search nodes: 0 = s, 1..m = corners, m + 1 = t.
    const m = cand.length;
    const X = new Float64Array(m + 2), Y = new Float64Array(m + 2);
    X[0] = sx;
    Y[0] = sy;
    for (let i = 0; i < m; i += 1) {
      X[i + 1] = this.cx[cand[i]];
      Y[i + 1] = this.cy[cand[i]];
    }
    X[m + 1] = tx;
    Y[m + 1] = ty;
    const T = m + 1;
    const corner = (i) => (i >= 1 && i <= m ? cand[i - 1] : -1);
    const h = (i) => Math.hypot(tx - X[i], ty - Y[i]);
    const g = new Float64Array(m + 2).fill(Infinity);
    const closed = new Uint8Array(m + 2);
    g[0] = 0;
    const heap = new MinHeap();
    heap.push(h(0), 0);
    while (heap.size) {
      const u = heap.pop();
      if (closed[u]) continue;
      if (u === T) return { length: g[T], corners: m, status: "found", checks: this.checks - before };
      closed[u] = 1;
      const cu = corner(u);
      for (let v = 1; v <= T; v += 1) {
        if (closed[v]) continue;
        const d = Math.hypot(X[v] - X[u], Y[v] - Y[u]);
        const nd = g[u] + d;
        if (nd >= g[v] || nd + h(v) > U) continue;
        const cv = corner(v);
        if (cv >= 0 && !this.tangentAt(cv, X[u], Y[u])) continue;
        if (cu >= 0 && !this.tangentAt(cu, X[v], Y[v])) continue;
        if (this.checks - before > maxChecks) return { length: NaN, corners: m, status: "capped", checks: this.checks - before };
        if (!this.visible(X[u], Y[u], X[v], Y[v])) continue;
        g[v] = nd;
        heap.push(nd + h(v), v);
      }
    }
    return { length: Infinity, corners: m, status: "beyond bound", checks: this.checks - before };
  }
}
