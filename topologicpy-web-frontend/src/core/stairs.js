/**
 * Staircases as their walking line.
 *
 * The lattice used to sample a stair's treads as a surface on a half-step grid
 * and join it to any floor node within reach, so a floor joined a flight
 * anywhere along its edges and sides, up to 2 m off its centre on C20's wide
 * stairs; and where the grid cells of a tread and a floor did not line up the
 * flight joined only one of its floors (the second dwelling of the Duplex in
 * the browser, the main stair of Revit ARC).
 *
 * Here each upward-facing patch of a stair (a tread or a landing) becomes one
 * node at its centroid: on a straight flight the treads' centroids lie on the
 * flight's centre line, and a landing's node sits in its middle. Patches link
 * when they touch in plan and rise by at most one riser, which also joins a
 * flight to a landing modelled as a separate element, so L- and U-shaped
 * stairs follow their turns. Each end of a staircase then links to the floor
 * once, at the floor node nearest the point one going beyond the end along
 * the walking line, on the outward side and at the right level.
 *
 * Ramps are walked anywhere across their width, so their sloping surfaces
 * (walkable triangles that are not level) are returned as a mesh for the
 * lattice to sample like a floor: even a 1:8 ramp rises only 6 cm per 0.5 m.
 */

/**
 * Upward-facing patches of the stair meshes.
 *
 * @param {Array} meshes  stair meshes ({positions, indices, itemId})
 * @param {object} opts   up, h0, h1: axis indices; minUp: least cosine of a
 *                        patch's slope; heightTol: triangles this close in
 *                        height and sharing a vertex form one patch; minArea
 *                        and minWidth drop slivers (nosings, handrail tops).
 * @returns {{patches: Array<{x: number, y: number, h: number, area: number, pts: number[][], tris: number[][]}>,
 *            ramp: {positions: Float32Array, indices: Uint32Array}}}  patches:
 *          level surfaces (treads, landings); ramp: the sloping walkable
 *          triangles, to be sampled as floor.
 */
export function stairPatches(meshes, { up, h0, h1, minUp = Math.cos((15 * Math.PI) / 180), heightTol = 0.03, minArea = 0.03, minWidth = 0.12, maxTread = 0.6, level = 0.02 }) {
  const tris = [];
  const sloped = [];
  for (const { positions: p, indices: ix } of meshes || []) {
    if (!p || !ix) continue;
    for (let t = 0; t + 2 < ix.length; t += 3) {
      const a = ix[t] * 3, b = ix[t + 1] * 3, c = ix[t + 2] * 3;
      const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
      const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
      const n = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
      const len = Math.hypot(n[0], n[1], n[2]);
      if (!len || n[up] / len < minUp) continue;
      // Not level: part of a ramp, sampled with the floors.
      if (Math.max(p[a + up], p[b + up], p[c + up]) - Math.min(p[a + up], p[b + up], p[c + up]) > level) {
        for (const o of [a, b, c]) sloped.push(p[o], p[o + 1], p[o + 2]);
        continue;
      }
      tris.push({
        h: (p[a + up] + p[b + up] + p[c + up]) / 3,
        area: len / 2,
        v: [[p[a + h0], p[a + h1]], [p[b + h0], p[b + h1]], [p[c + h0], p[c + h1]]],
        // The 3-D vertices, for a landing resampled as floor.
        v3: [[p[a], p[a + 1], p[a + 2]], [p[b], p[b + 1], p[b + 2]], [p[c], p[c + 1], p[c + 2]]],
      });
    }
  }
  // Triangles sharing a vertex (to 1 cm in plan) at nearly the same height
  // are one patch.
  const parent = tris.map((_, i) => i);
  const find = (i) => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]];
    return i;
  };
  const byVertex = new Map();
  tris.forEach((t, i) => {
    for (const [x, y] of t.v) {
      const k = `${Math.round(x * 100)},${Math.round(y * 100)}`;
      let list = byVertex.get(k);
      if (!list) byVertex.set(k, (list = []));
      list.push(i);
    }
  });
  for (const list of byVertex.values()) {
    list.sort((i, j) => tris[i].h - tris[j].h);
    for (let k = 1; k < list.length; k += 1) {
      if (tris[list[k]].h - tris[list[k - 1]].h <= heightTol) parent[find(list[k])] = find(list[k - 1]);
    }
  }
  const groups = new Map();
  tris.forEach((t, i) => {
    const r = find(i);
    let g = groups.get(r);
    if (!g) groups.set(r, (g = []));
    g.push(t);
  });
  const patches = [];
  for (const g of groups.values()) {
    let area = 0, sx = 0, sy = 0, sh = 0;
    const pts = [];
    for (const t of g) {
      const cx = (t.v[0][0] + t.v[1][0] + t.v[2][0]) / 3, cy = (t.v[0][1] + t.v[1][1] + t.v[2][1]) / 3;
      area += t.area;
      sx += cx * t.area;
      sy += cy * t.area;
      sh += t.h * t.area;
      pts.push(...t.v);
    }
    if (area < minArea) continue;
    const x = sx / area, y = sy / area;
    // Narrowest extent across the patch's principal axes.
    let sxx = 0, syy = 0, sxy = 0;
    for (const [px, py] of pts) {
      sxx += (px - x) ** 2;
      syy += (py - y) ** 2;
      sxy += (px - x) * (py - y);
    }
    const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const ax = Math.cos(angle), ay = Math.sin(angle);
    let lo1 = Infinity, hi1 = -Infinity, lo2 = Infinity, hi2 = -Infinity;
    for (const [px, py] of pts) {
      const s = (px - x) * ax + (py - y) * ay, r = -(px - x) * ay + (py - y) * ax;
      lo1 = Math.min(lo1, s);
      hi1 = Math.max(hi1, s);
      lo2 = Math.min(lo2, r);
      hi2 = Math.max(hi2, r);
    }
    const narrow = Math.min(hi1 - lo1, hi2 - lo2);
    if (narrow < minWidth) continue;
    // A tread is narrow (a going of 0.2 to 0.5 m). Anything wider both ways
    // is a landing or a level part of a ramp, walked anywhere across: it is
    // sampled with the floors, and the flights' ends join it as a floor.
    if (narrow > maxTread) {
      for (const t of g) for (const v of t.v3) sloped.push(...v);
      continue;
    }
    patches.push({ x, y, h: sh / area, area, pts, tris: g.map((t) => t.v) });
  }
  const ramp = { positions: Float32Array.from(sloped), indices: Uint32Array.from(sloped.map((_, i) => i).filter((i) => i % 3 === 0), (i) => i / 3) };
  return { patches, ramp };
}

/** Plan distance from point (px, py) to triangle [[ax,ay],[bx,by],[cx,cy]]. */
function pointTriangle(px, py, [[ax, ay], [bx, by], [cx, cy]]) {
  const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
  const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
  const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
  const neg = d1 < 0 || d2 < 0 || d3 < 0, pos = d1 > 0 || d2 > 0 || d3 > 0;
  if (!(neg && pos)) return 0;
  const seg = (x1, y1, x2, y2) => {
    const dx = x2 - x1, dy = y2 - y1;
    const l2 = dx * dx + dy * dy;
    const t = l2 ? Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / l2)) : 0;
    return Math.hypot(px - x1 - t * dx, py - y1 - t * dy);
  };
  return Math.min(seg(ax, ay, bx, by), seg(bx, by, cx, cy), seg(cx, cy, ax, ay));
}

/** Whether two patches touch in plan, within `tol`. */
function touching(a, b, tol) {
  for (const [px, py] of a.pts) for (const t of b.tris) if (pointTriangle(px, py, t) <= tol) return true;
  for (const [px, py] of b.pts) for (const t of a.tris) if (pointTriangle(px, py, t) <= tol) return true;
  return false;
}

/**
 * The walking line: links between patches that touch and rise by at most
 * `maxRise`, and each patch's lower and higher neighbours.
 */
export function stairLinks(patches, { maxRise = 0.36, touchTol = 0.08 } = {}) {
  const order = patches.map((_, i) => i).sort((i, j) => patches[i].h - patches[j].h);
  const box = patches.map((p) => {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of p.pts) {
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
    }
    return [x0, x1, y0, y1];
  });
  const links = [];
  const lower = patches.map(() => []);
  const higher = patches.map(() => []);
  for (let s = 0; s < order.length; s += 1) {
    const a = order[s];
    for (let t = s + 1; t < order.length && patches[order[t]].h - patches[a].h <= maxRise; t += 1) {
      const b = order[t];
      const [ax0, ax1, ay0, ay1] = box[a], [bx0, bx1, by0, by1] = box[b];
      if (bx0 > ax1 + touchTol || ax0 > bx1 + touchTol || by0 > ay1 + touchTol || ay0 > by1 + touchTol) continue;
      if (!touching(patches[a], patches[b], touchTol)) continue;
      links.push([a, b]);
      higher[a].push(b);
      lower[b].push(a);
    }
  }
  return { links, lower, higher };
}
