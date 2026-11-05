/**
 * The IFC model as the Flamescapes solver's solid fuel grid (Wrede, Wagner
 * et al. 2026, section 4.2.1; transcription in
 * wiki/sources/papers/flamescapes-2026.md).
 *
 * The grid is a uniform Cartesian grid in the viewer's frame with the up axis
 * second: cell (i, j, k) spans [o + (i, j, k) h, o + (i + 1, j + 1, k + 1) h]
 * along (h0, up, h1), the two horizontal axes and the up axis of the graph
 * (lattice.js horizontalAxes). Index = i + nx (j + ny k).
 *
 * Volume fractions, paper Eq. 1. The paper casts N = 64 rays per voxel and
 * divides the length each ray spends inside material m by the total ray
 * length. Here the rays run along the up axis, S x S of them per column of
 * cells (S = 2 by default), and each mesh is closed (an IFC solid), so a ray
 * is inside between its 1st and 2nd crossing of the mesh, its 3rd and 4th, and
 * so on: phi_m = sum over rays of the length inside m within the cell, over
 * S^2 h. A ray with an odd number of crossings (an open mesh) drops its last
 * one, the paper's guard against back faces outside the voxel taken the
 * simple way. Walls, slabs and roofs are vertical extrusions or plates, so a
 * vertical ray meets them in two faces; their side faces project to nothing
 * and are skipped.
 *
 * Interior fuel, paper Eqs. 3 to 5. The paper does not model furniture: it
 * samples a fire load density q from a residential survey, takes the energy
 * E = q A over the floor area A, the mass M = E / Delta h_eff, and spreads M
 * over the building's interior cells as m_i ~ N(mu, sigma^2) with
 * mu = M / N. Here the interior cells are the gas cells resting on a floor
 * (the cell below is solid) with a solid within `ceiling` metres above (a
 * slab or roof): the room contents, `layers` cells deep. They are porous, as
 * the paper's vegetation: gas flows through them, slowed.
 */

import { axisIndex, horizontalAxes } from "./lattice.js";

/** Material classes, after the paper's Table 3 (supplement). */
export const MATERIAL = Object.freeze({ STONE: 0, GLASS: 1, METAL: 2, WOOD: 3, CONTENTS: 4 });
export const MATERIALS = 5;

/** The total volume fraction above which a cell is solid, and porous between. */
export const SOLID_FRACTION = 0.5;

/** A small seeded generator (mulberry32, as rl.js), so a seed reproduces a run. */
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A standard normal draw from two uniforms (Box-Muller). */
export function normal(random) {
  const u = Math.max(random(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

/**
 * The grid that holds every mesh with `margin` cells of air round it, at
 * cell size `cell`, or coarser when that would exceed `maxCells`.
 */
export function gridFor(meshes, { upAxis = "y", cell = 0.3, margin = 2, headroom = 2, maxCells = 60000, bounds = null } = {}) {
  const up = axisIndex(upAxis);
  const [a0, a1] = horizontalAxes(upAxis);
  const axes = [a0, up, a1];
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const mesh of meshes) {
    const p = mesh.positions;
    for (let v = 0; v < p.length; v += 3) {
      for (let d = 0; d < 3; d += 1) {
        const x = p[v + axes[d]];
        if (!Number.isFinite(x)) continue;
        if (x < lo[d]) lo[d] = x;
        if (x > hi[d]) hi[d] = x;
      }
    }
  }
  if (!(lo[0] <= hi[0])) throw new Error("The model has no geometry to build a fire grid from.");
  // Clipped to `bounds` (local frame): meshes that cross it keep their whole
  // triangles, but the grid ends at the box.
  if (bounds) {
    for (let d = 0; d < 3; d += 1) {
      lo[d] = Math.max(lo[d], bounds.lo[d]);
      hi[d] = Math.min(hi[d], bounds.hi[d]);
    }
  }
  const extent = (h, d) => hi[d] - lo[d] + 2 * margin * h + (d === 1 ? headroom * h : 0);
  let h = cell;
  // Coarsen until the grid fits: the cell count falls as 1 / h^3.
  for (let guard = 0; guard < 60; guard += 1) {
    const count = Math.ceil(extent(h, 0) / h) * Math.ceil(extent(h, 1) / h) * Math.ceil(extent(h, 2) / h);
    if (count <= maxCells) break;
    h *= 1.08;
  }
  const n = [0, 1, 2].map((d) => Math.max(3, Math.ceil(extent(h, d) / h)));
  const origin = [lo[0] - margin * h, lo[1] - margin * h, lo[2] - margin * h];
  return { nx: n[0], ny: n[1], nz: n[2], h, origin, axes, upAxis, cells: n[0] * n[1] * n[2] };
}

/** World point -> local (a, b, c) along the grid axes. */
export function toLocal(grid, x, y, z) {
  const p = [x, y, z];
  return [p[grid.axes[0]], p[grid.axes[1]], p[grid.axes[2]]];
}

/** Local (a, b, c) -> world [x, y, z]. */
export function toWorld(grid, a, b, c) {
  const out = [0, 0, 0];
  out[grid.axes[0]] = a;
  out[grid.axes[1]] = b;
  out[grid.axes[2]] = c;
  return out;
}

/**
 * Per-cell, per-material volume fractions (paper Eq. 1). `meshes` are
 * {positions, indices, material}; returns a Float32Array of cells x MATERIALS.
 */
export function volumeFractions(grid, meshes, { rays = 2 } = {}) {
  const { nx, ny, nz, h, origin, axes } = grid;
  const S = Math.max(1, Math.floor(rays));
  const fraction = new Float32Array(grid.cells * MATERIALS);
  const columns = nx * S;
  // Sub-ray offsets inside a cell, nudged off the cell's symmetric points so a
  // ray rarely runs exactly along a triangle edge of an axis-aligned model.
  const offset = (s) => (s + 0.5) / S + 0.0137 / S;
  const rayA = (I) => origin[0] + (Math.floor(I / S) + offset(I % S)) * h;
  const rayC = (K) => origin[2] + (Math.floor(K / S) + offset(K % S)) * h;
  const weight = 1 / (S * S * h);

  for (const mesh of meshes) {
    const P = mesh.positions;
    const ix = mesh.indices;
    const m = mesh.material;
    const hits = new Map();
    for (let t = 0; t < ix.length; t += 3) {
      const A = ix[t] * 3, B = ix[t + 1] * 3, C = ix[t + 2] * 3;
      const ax = P[A + axes[0]], ab = P[A + axes[1]], ac = P[A + axes[2]];
      const bx = P[B + axes[0]], bb = P[B + axes[1]], bc = P[B + axes[2]];
      const cx = P[C + axes[0]], cb = P[C + axes[1]], cc = P[C + axes[2]];
      const det = (bx - ax) * (cc - ac) - (cx - ax) * (bc - ac);
      if (!(Math.abs(det) > 1e-12)) continue; // vertical or degenerate in plan
      const I0 = Math.max(0, Math.ceil(((Math.min(ax, bx, cx) - origin[0]) / h) * S - 0.5 - 0.0137));
      const I1 = Math.min(columns - 1, Math.floor(((Math.max(ax, bx, cx) - origin[0]) / h) * S - 0.5 - 0.0137) + 1);
      const K0 = Math.max(0, Math.ceil(((Math.min(ac, bc, cc) - origin[2]) / h) * S - 0.5 - 0.0137));
      const K1 = Math.min(nz * S - 1, Math.floor(((Math.max(ac, bc, cc) - origin[2]) / h) * S - 0.5 - 0.0137) + 1);
      for (let I = I0; I <= I1; I += 1) {
        const pa = rayA(I);
        for (let K = K0; K <= K1; K += 1) {
          const pc = rayC(K);
          // Barycentric coordinates of the ray's foot in the triangle's plan.
          const u = ((pa - ax) * (cc - ac) - (cx - ax) * (pc - ac)) / det;
          const v = ((bx - ax) * (pc - ac) - (pa - ax) * (bc - ac)) / det;
          if (u < 0 || v < 0 || u + v > 1) continue;
          const height = ab + u * (bb - ab) + v * (cb - ab);
          const key = I + columns * K;
          const list = hits.get(key);
          if (list) list.push(height);
          else hits.set(key, [height]);
        }
      }
    }
    for (const [key, list] of hits) {
      list.sort((p, q) => p - q);
      // Two triangles sharing an edge the ray passes through give one crossing.
      const ys = [];
      for (const y of list) if (!ys.length || y - ys[ys.length - 1] > 1e-6) ys.push(y);
      const I = key % columns;
      const K = Math.floor(key / columns);
      const i = Math.floor(I / S);
      const k = Math.floor(K / S);
      for (let q = 0; q + 1 < ys.length; q += 2) {
        const y0 = (ys[q] - origin[1]) / h;
        const y1 = (ys[q + 1] - origin[1]) / h;
        const j0 = Math.max(0, Math.floor(y0));
        const j1 = Math.min(ny - 1, Math.floor(y1));
        for (let j = j0; j <= j1; j += 1) {
          const overlap = Math.min(y1, j + 1) - Math.max(y0, j);
          if (overlap <= 0) continue;
          fraction[(i + nx * (j + ny * k)) * MATERIALS + m] += overlap * h * weight;
        }
      }
    }
  }
  // Meshes that overlap (a slab running into a wall) cannot fill a cell twice.
  for (let c = 0; c < grid.cells; c += 1) {
    let total = 0;
    for (let m = 0; m < MATERIALS; m += 1) total += fraction[c * MATERIALS + m];
    if (total > 1) for (let m = 0; m < MATERIALS; m += 1) fraction[c * MATERIALS + m] /= total;
  }
  return fraction;
}

/** Sum of a cell's volume fractions. */
export function solidFraction(fraction, c) {
  let total = 0;
  for (let m = 0; m < MATERIALS; m += 1) total += fraction[c * MATERIALS + m];
  return total;
}

/** Below this fraction a cell holds too little of an element to be its wall. */
export const THIN_FRACTION = 0.08;

/**
 * Which cells block the flow. A cell at least half full is solid (the
 * paper's density threshold). On a grid coarser than the paper's, a wall or
 * slab thinner than a cell can straddle two cells and fill neither by half,
 * and the room would leak through it; so a cell is also solid where its
 * fraction is the largest along some axis among itself and its two
 * neighbours, and at least THIN_FRACTION: thin elements snap to the cell that
 * holds most of them, as obstructions snap to the grid in FDS.
 */
export function occupancy(grid, fraction) {
  const { nx, ny, nz } = grid;
  const total = new Float32Array(grid.cells);
  for (let c = 0; c < grid.cells; c += 1) total[c] = solidFraction(fraction, c);
  const solid = new Uint8Array(grid.cells);
  const f = (i, j, k) => (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz ? 0 : total[i + nx * (j + ny * k)]);
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const c = i + nx * (j + ny * k);
        const v = total[c];
        if (v >= SOLID_FRACTION) solid[c] = 1;
        else if (v >= THIN_FRACTION) {
          const peak =
            (v >= f(i - 1, j, k) && v >= f(i + 1, j, k)) ||
            (v >= f(i, j - 1, k) && v >= f(i, j + 1, k)) ||
            (v >= f(i, j, k - 1) && v >= f(i, j, k + 1));
          if (peak) solid[c] = 1;
        }
      }
    }
  }
  return solid;
}

/** Rays per cell side so that rays are at most `spacing` apart (thin walls are not missed). */
export function raysFor(grid, spacing = 0.05) {
  return Math.max(2, Math.ceil(grid.h / spacing));
}

/**
 * The room-contents cells (paper section 4.2.1, "Modeling of interior
 * fuel"): gas cells resting on a solid with a solid at most `ceiling` metres
 * above, and the `layers` - 1 gas cells over them.
 */
export function interiorCells(grid, fraction, { ceiling = 5, layers = 2, solidMask = occupancy(grid, fraction) } = {}) {
  const { nx, ny, nz, h } = grid;
  const solid = (i, j, k) => solidMask[i + nx * (j + ny * k)] === 1;
  const reach = Math.max(1, Math.round(ceiling / h));
  const cells = [];
  for (let k = 0; k < nz; k += 1) {
    for (let i = 0; i < nx; i += 1) {
      for (let j = 1; j < ny; j += 1) {
        if (solid(i, j, k) || !solid(i, j - 1, k)) continue;
        let covered = false;
        for (let up = j + 1; up < Math.min(ny, j + reach + 1); up += 1) {
          if (solid(i, up, k)) {
            covered = true;
            break;
          }
        }
        if (!covered) continue;
        for (let d = 0; d < layers && j + d < ny && !solid(i, j + d, k); d += 1) cells.push(i + nx * (j + d + ny * k));
      }
    }
  }
  return cells;
}

/**
 * Paper Eqs. 3 to 5: total contents mass M = q A / Delta h_eff over the floor
 * area A of the interior cells, spread as m_i ~ N(mu, sigma^2), mu = M / N,
 * cut at zero (a cell cannot hold negative fuel). Returns per-cell masses (kg)
 * and the totals.
 */
export function interiorFuel(grid, cells, { fireLoad = 780e6, heatOfCombustion = 17.5e6, spread = 0.3, layers = 2, seed = 1 } = {}) {
  const area = (cells.length / Math.max(1, layers)) * grid.h * grid.h;
  const energy = fireLoad * area;
  const mass = energy / heatOfCombustion;
  const mu = cells.length ? mass / cells.length : 0;
  const sigma = spread * mu;
  const random = rng(seed);
  const masses = new Float32Array(cells.length);
  let total = 0;
  for (let q = 0; q < cells.length; q += 1) {
    masses[q] = Math.max(0, mu + sigma * normal(random));
    total += masses[q];
  }
  // Cutting draws at zero raises the expected total above M; rescaled, the
  // contents hold exactly the paper's budget (Eq. 4).
  if (total > 0) {
    const scale = mass / total;
    for (let q = 0; q < cells.length; q += 1) masses[q] *= scale;
    total = mass;
  }
  return { masses, area, energy, mass, total, mu, sigma };
}
