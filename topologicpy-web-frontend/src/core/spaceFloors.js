/**
 * Floor slabs for the fire grid from IfcSpace bottoms, for a model whose
 * floors are not modelled as slabs: the NIBS Office has two small plates and
 * walks on its spaces' bottoms, as the walking graph reads them (lattice.js).
 * Without a floor the fire grid has nowhere to stand the room contents, and
 * no fire starts (tools/sprinkler-check.mjs found 0 MJ released in every run
 * on that model).
 */

/**
 * A space named as a void ("open to below"): its bottom is no floor, in the
 * air over the atrium or stair it opens onto. The walkable graph leaves it
 * out (outdoorSpaces.js), the fire grid takes no slab from it, and the room
 * under it is open to its top (sprinklers.js openAtria).
 */
export const VOID_NAME = /\b(open to below|void|luftraum)\b/i;

const BOTTOM_BAND = 0.05;
const HORIZONTAL = 0.9;

/** The horizontal faces at a space's bottom, one facing only (down, else up), as vertex offsets. */
function bottomFaces(p, ix, up) {
  let lo = Infinity;
  for (let o = up; o < p.length; o += 3) lo = Math.min(lo, p[o]);
  const down = [], upward = [];
  for (let t = 0; t + 2 < ix.length; t += 3) {
    const a = ix[t] * 3, b = ix[t + 1] * 3, c = ix[t + 2] * 3;
    if (Math.max(p[a + up], p[b + up], p[c + up]) > lo + BOTTOM_BAND) continue;
    const e1 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
    const e2 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const mag = Math.hypot(n[0], n[1], n[2]);
    if (mag < 1e-12 || Math.abs(n[up] / mag) < HORIZONTAL) continue;
    (n[up] < 0 ? down : upward).push([a, b, c, mag / 2]);
  }
  return down.length ? down : upward;
}

/** Plan area of meshes' horizontal faces (m²), halved: a slab has a top and a bottom. */
function slabArea(meshes, up) {
  let area = 0;
  for (const m of meshes || []) {
    const p = m?.positions, ix = m?.indices;
    if (!p || !ix) continue;
    for (let t = 0; t + 2 < ix.length; t += 3) {
      const a = ix[t] * 3, b = ix[t + 1] * 3, c = ix[t + 2] * 3;
      const e1 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
      const e2 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
      const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const mag = Math.hypot(n[0], n[1], n[2]);
      if (mag > 1e-12 && Math.abs(n[up] / mag) >= HORIZONTAL) area += mag / 2;
    }
  }
  return area / 2;
}

/**
 * Whether a model's slabs cover less than half the floor its spaces stand on:
 * then the fire grid takes the spaces' bottoms as its floor (floorsFromSpaces).
 */
export function slabsMissing(floors, spaces, { upAxis = "y" } = {}) {
  const up = "xyz".indexOf(upAxis);
  let spaceArea = 0;
  for (const m of spaces || []) if (m?.positions && m?.indices) for (const f of bottomFaces(m.positions, m.indices, up)) spaceArea += f[3];
  return spaceArea > 0 && slabArea(floors, up) < 0.5 * spaceArea;
}

/**
 * Each space's bottom faces extruded `thickness` m down, closed (top, bottom
 * and sides), one mesh per space, in the spaces' frame.
 */
export function floorsFromSpaces(spaces, { upAxis = "y", thickness = 0.3 } = {}) {
  const up = "xyz".indexOf(upAxis);
  const out = [];
  for (const mesh of spaces || []) {
    const p = mesh?.positions, ix = mesh?.indices;
    if (!p || !ix) continue;
    const positions = [], indices = [];
    for (const [a, b, c] of bottomFaces(p, ix, up)) {
      const base = positions.length / 3;
      for (const v of [a, b, c]) positions.push(p[v], p[v + 1], p[v + 2]);
      for (const v of [a, b, c]) {
        const q = [p[v], p[v + 1], p[v + 2]];
        q[up] -= thickness;
        positions.push(q[0], q[1], q[2]);
      }
      indices.push(base, base + 1, base + 2, base + 5, base + 4, base + 3);
      for (let e = 0; e < 3; e += 1) {
        const i = e, j = (e + 1) % 3;
        indices.push(base + i, base + 3 + i, base + 3 + j, base + i, base + 3 + j, base + j);
      }
    }
    if (indices.length) out.push({ itemId: mesh.itemId, positions: Float32Array.from(positions), indices: Uint32Array.from(indices) });
  }
  return out;
}
