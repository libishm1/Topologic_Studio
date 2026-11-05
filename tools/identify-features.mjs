/**
 * Geometric features of one IFC element, computed from its triangles in the
 * viewer's frame (Y up) in milliseconds, for identifying elements and
 * furniture sub-types without a name (the furniture study plan:
 * wiki/research/ifcnet-furniture-study-plan-2026-11-05.md).
 *
 * The methods are implemented here from their published descriptions; no
 * code was copied from any repository:
 *   - covariance eigenvalue features of a point neighbourhood: linearity,
 *     planarity, scattering, omnivariance, anisotropy, eigenentropy, change
 *     of curvature and the eigenvalue sum (West et al. 2004,
 *     doi:10.1117/12.542536; Demantke et al. 2011,
 *     doi:10.5194/isprsarchives-XXXVIII-5-W12-97-2011; Weinmann et al. 2015,
 *     doi:10.1016/j.isprsjprs.2015.01.016; Hackel et al. 2016,
 *     doi:10.5194/isprs-annals-III-3-177-2016), verticality (Demantke et
 *     al. 2012, doi:10.5194/isprsannals-I-3-99-2012), as Poux and Billen
 *     (2019, doi:10.3390/ijgi8050213) use them for indoor scenes and
 *     furniture (Poux et al. 2018, doi:10.3390/rs10091412);
 *   - planar segments by region growing (after Poux et al. 2022,
 *     doi:10.1016/j.autcon.2022.104250), done here on the mesh itself:
 *     edge-adjacent triangles join a segment when their normals differ by
 *     less than 5 degrees and each centroid lies within 1 cm of the other's
 *     plane;
 *   - voxel occupancy (Poux and Billen 2019, section 3.1), on a grid fitted
 *     to the element's own box.
 *
 *   node tools/identify-features.mjs extract --bench DIR --out DIR
 *       Reads the corpus as tools/identify-bench.mjs does (the benchmark
 *       manifest, ~/ifc-samples and its more2 folder, the shipped examples),
 *       and writes OUT/features.json (every record of BENCH/dataset.json plus
 *       every furniture and appliance element of the corpus, with its
 *       features, the rules' answer and a furniture sub-type from its
 *       PredefinedType or name) and OUT/extra-points.bin (IFCNet's 2,048
 *       points for the elements not already in BENCH/points.bin). BENCH and
 *       OUT must be outside the repository: they hold third-party names.
 */
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { mkdir, readdir, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { GENERIC_CLASSES, boxOf, identify, labelForClass, subtypeOf } from "../topologicpy-web-frontend/src/core/identify.js";
import { rng, samplePoints } from "../topologicpy-web-frontend/src/core/identifyGeometry.js";

// ---------------------------------------------------------------- feature names

const HREL_BINS = 8;
/** Heights above the floor (m) that separate seats, tables, worktops and shelves. */
export const UP_HEIGHT_EDGES = [0.15, 0.32, 0.55, 0.68, 0.82, 1.0, 1.6];

export const FEATURE_NAMES = [
  // Box and placement.
  "len", "wid", "hgt", "above", "top", "logVol", "widLen", "hgtLen", "logArea", "areaRatio", "logTris", "floorKnown",
  // Covariance of the whole element (area-weighted samples).
  "linearity", "planarity", "scattering", "omnivariance", "anisotropy", "eigenentropy", "curvChange", "logEigSum", "axisVert", "verticality",
  // Normals, area-weighted over the triangles.
  "nUp", "nDown", "nSide", "nSlant", "meanAbsNy", "orientEntropy",
  // Where the surface is, relative to the element's height.
  ...Array.from({ length: HREL_BINS }, (_, i) => `hRel${i}`),
  // Where the upward-facing surface is, above the floor.
  ...Array.from({ length: UP_HEIGHT_EDGES.length + 1 }, (_, i) => `upH${i}`),
  "upLevels", "upTopShare", "upMainHeight",
  // Planar segments (region growing on the mesh) and connected parts.
  "segN1", "segN5", "segTop1", "segTop3", "segCover", "partsN", "partsTop1",
  // Voxel occupancy.
  "occ8", "planFill", "layer0", "layer1", "layer2", "layer3",
  // Local covariance features (k nearest samples), aggregated.
  "locLinMean", "locPlaMean", "locScaMean", "locLinStd", "locPlaStd", "locScaStd", "share1d", "share2d", "share3d", "locVertMean", "locCurvMean",
];

// ---------------------------------------------------------------- small maths

/**
 * Eigen-decomposition of a symmetric 3x3 matrix by cyclic Jacobi rotations.
 * Returns the eigenvalues in descending order and the matching unit vectors.
 */
export function eigSym3(a00, a01, a02, a11, a12, a22) {
  const A = [[a00, a01, a02], [a01, a11, a12], [a02, a12, a22]];
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 12; sweep += 1) {
    const off = A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2];
    const diag = A[0][0] * A[0][0] + A[1][1] * A[1][1] + A[2][2] * A[2][2];
    if (off <= 1e-24 * (diag + 1e-300)) break;
    for (let p = 0; p < 2; p += 1) {
      for (let q = p + 1; q < 3; q += 1) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 3; k += 1) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k += 1) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k += 1) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => A[j][j] - A[i][i]);
  return {
    values: order.map((i) => Math.max(A[i][i], 0)),
    vectors: order.map((i) => [V[0][i], V[1][i], V[2][i]]),
  };
}

/** The eigenvalue features of one covariance (Weinmann et al. 2015, Table 1). */
function eigenFeatures(l1, l2, l3) {
  const sum = l1 + l2 + l3;
  if (!(sum > 0) || !(l1 > 0)) return { lin: 0, pla: 0, sca: 0, omni: 0, ani: 0, ent: 0, curv: 0, sum: 0 };
  const e1 = l1 / sum, e2 = l2 / sum, e3 = l3 / sum;
  const ent = -[e1, e2, e3].reduce((s, e) => (e > 1e-12 ? s + e * Math.log(e) : s), 0);
  return {
    lin: (l1 - l2) / l1,
    pla: (l2 - l3) / l1,
    sca: l3 / l1,
    omni: Math.cbrt(e1 * e2 * e3),
    ani: (l1 - l3) / l1,
    ent,
    curv: e3,
    sum,
  };
}

function covariance(xs, ys, zs, idx, count) {
  let mx = 0, my = 0, mz = 0;
  for (let i = 0; i < count; i += 1) {
    const j = idx ? idx[i] : i;
    mx += xs[j]; my += ys[j]; mz += zs[j];
  }
  mx /= count; my /= count; mz /= count;
  let cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
  for (let i = 0; i < count; i += 1) {
    const j = idx ? idx[i] : i;
    const dx = xs[j] - mx, dy = ys[j] - my, dz = zs[j] - mz;
    cxx += dx * dx; cxy += dx * dy; cxz += dx * dz; cyy += dy * dy; cyz += dy * dz; czz += dz * dz;
  }
  return [cxx / count, cxy / count, cxz / count, cyy / count, cyz / count, czz / count];
}

function find(parent, i) {
  while (parent[i] !== i) {
    parent[i] = parent[parent[i]];
    i = parent[i];
  }
  return i;
}
function union(parent, a, b) {
  const ra = find(parent, a), rb = find(parent, b);
  if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
}

/** Shares of a list of areas, largest first. */
function sharesOf(map, total) {
  return [...map.values()].sort((a, b) => b - a).map((a) => a / total);
}

// ---------------------------------------------------------------- one element

/**
 * The features of one element.
 *
 * @param {Array<{positions: Float32Array, indices: Uint32Array}>} meshes  viewer frame (Y up), metres
 * @param {object} opts  floor: the storey floor's height (Y), or null; samples; local (query points); k; seed
 * @returns {Float64Array|null} FEATURE_NAMES.length values, or null when the meshes have no area
 */
export function elementFeatures(meshes, { floor = null, samples = 1024, local = 384, k = 16, seed = 1 } = {}) {
  let T = 0;
  for (const m of meshes || []) T += Math.floor((m.indices?.length || 0) / 3);
  if (!T) return null;
  const box = boxOf(meshes);
  if (!box) return null;

  // Triangles: corners, unit normals, centroids and areas; vertices welded at 1 mm.
  const tri = new Float64Array(T * 9);
  const nrm = new Float64Array(T * 3);
  const cen = new Float64Array(T * 3);
  const area = new Float64Array(T);
  const tv = new Int32Array(T * 3);
  let size = 1;
  while (size < T * 6) size <<= 1;
  const mask = size - 1;
  const hx = new Int32Array(size), hy = new Int32Array(size), hz = new Int32Array(size), hid = new Int32Array(size).fill(-1);
  let nv = 0;
  const weld = (x, y, z) => {
    const qx = Math.round(x * 1000), qy = Math.round(y * 1000), qz = Math.round(z * 1000);
    let h = (Math.imul(qx, 73856093) ^ Math.imul(qy, 19349663) ^ Math.imul(qz, 83492791)) & mask;
    for (;;) {
      if (hid[h] < 0) {
        hx[h] = qx; hy[h] = qy; hz[h] = qz; hid[h] = nv;
        return nv++;
      }
      if (hx[h] === qx && hy[h] === qy && hz[h] === qz) return hid[h];
      h = (h + 1) & mask;
    }
  };
  let t = 0, A = 0;
  for (const m of meshes) {
    const p = m.positions, idx = m.indices;
    for (let i = 0; i + 2 < idx.length; i += 3, t += 1) {
      const a = idx[i] * 3, b = idx[i + 1] * 3, c = idx[i + 2] * 3;
      const o = t * 9;
      tri[o] = p[a]; tri[o + 1] = p[a + 1]; tri[o + 2] = p[a + 2];
      tri[o + 3] = p[b]; tri[o + 4] = p[b + 1]; tri[o + 5] = p[b + 2];
      tri[o + 6] = p[c]; tri[o + 7] = p[c + 1]; tri[o + 8] = p[c + 2];
      const ux = tri[o + 3] - tri[o], uy = tri[o + 4] - tri[o + 1], uz = tri[o + 5] - tri[o + 2];
      const vx = tri[o + 6] - tri[o], vy = tri[o + 7] - tri[o + 1], vz = tri[o + 8] - tri[o + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const len = Math.hypot(nx, ny, nz);
      area[t] = len / 2;
      A += area[t];
      if (len > 0) {
        nrm[t * 3] = nx / len; nrm[t * 3 + 1] = ny / len; nrm[t * 3 + 2] = nz / len;
      }
      cen[t * 3] = (tri[o] + tri[o + 3] + tri[o + 6]) / 3;
      cen[t * 3 + 1] = (tri[o + 1] + tri[o + 4] + tri[o + 7]) / 3;
      cen[t * 3 + 2] = (tri[o + 2] + tri[o + 5] + tri[o + 8]) / 3;
      tv[t * 3] = weld(tri[o], tri[o + 1], tri[o + 2]);
      tv[t * 3 + 1] = weld(tri[o + 3], tri[o + 4], tri[o + 5]);
      tv[t * 3 + 2] = weld(tri[o + 6], tri[o + 7], tri[o + 8]);
    }
  }
  if (!(A > 0)) return null;
  const [len, wid, hgt] = box.dims;
  const bottom = box.bottom;
  const floorKnown = Number.isFinite(floor) ? 1 : 0;
  const ground = floorKnown ? floor : bottom;
  const above = bottom - ground;

  // Area-weighted samples (seeded).
  const cum = new Float64Array(T);
  let acc = 0;
  for (let i = 0; i < T; i += 1) cum[i] = acc += area[i];
  const r = rng(seed);
  const sx = new Float64Array(samples), sy = new Float64Array(samples), sz = new Float64Array(samples);
  for (let s = 0; s < samples; s += 1) {
    const target = r() * A;
    let lo = 0, hi = T - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    let u = r(), v = r();
    if (u + v > 1) { u = 1 - u; v = 1 - v; }
    const o = lo * 9;
    sx[s] = tri[o] + u * (tri[o + 3] - tri[o]) + v * (tri[o + 6] - tri[o]);
    sy[s] = tri[o + 1] + u * (tri[o + 4] - tri[o + 1]) + v * (tri[o + 7] - tri[o + 1]);
    sz[s] = tri[o + 2] + u * (tri[o + 5] - tri[o + 2]) + v * (tri[o + 8] - tri[o + 2]);
  }

  // Whole-element covariance.
  const C = covariance(sx, sy, sz, null, samples);
  const E = eigSym3(...C);
  const g = eigenFeatures(...E.values);
  const axisVert = Math.abs(E.vectors[0][1]);
  const verticality = 1 - Math.abs(E.vectors[2][1]);

  // Principal horizontal axis (plan), for orientation bins and the plan grid.
  let pxx = 0, pxz = 0, pzz = 0, mx = 0, mz = 0;
  for (let s = 0; s < samples; s += 1) { mx += sx[s]; mz += sz[s]; }
  mx /= samples; mz /= samples;
  for (let s = 0; s < samples; s += 1) {
    const dx = sx[s] - mx, dz = sz[s] - mz;
    pxx += dx * dx; pxz += dx * dz; pzz += dz * dz;
  }
  const angle = 0.5 * Math.atan2(2 * pxz, pxx - pzz);
  const ca = Math.cos(angle), sa = Math.sin(angle);

  // Normals.
  let nUp = 0, nDown = 0, nSide = 0, nSlant = 0, absNy = 0;
  const orient = new Float64Array(12);
  for (let i = 0; i < T; i += 1) {
    const a = area[i];
    if (!a) continue;
    const nx = nrm[i * 3], ny = nrm[i * 3 + 1], nz = nrm[i * 3 + 2];
    absNy += a * Math.abs(ny);
    if (ny > 0.9) { nUp += a; orient[0] += a; }
    else if (ny < -0.9) { nDown += a; orient[1] += a; }
    else if (Math.abs(ny) < 0.1) {
      nSide += a;
      const az = Math.atan2(-nx * sa + nz * ca, nx * ca + nz * sa);
      orient[2 + (Math.floor(((az + Math.PI) / (2 * Math.PI)) * 8) & 7)] += a;
    } else {
      nSlant += a;
      orient[ny > 0 ? 10 : 11] += a;
    }
  }
  let orientEntropy = 0;
  for (const o of orient) if (o > 0) orientEntropy -= (o / A) * Math.log(o / A);
  orientEntropy /= Math.log(12);

  // Relative height profile.
  const hRel = new Float64Array(HREL_BINS);
  for (let s = 0; s < samples; s += 1) {
    const q = hgt > 1e-6 ? (sy[s] - bottom) / hgt : 0;
    hRel[Math.min(HREL_BINS - 1, Math.max(0, Math.floor(q * HREL_BINS)))] += 1 / samples;
  }

  // Upward-facing surface: heights above the floor, and its distinct levels (2 cm bins).
  const upH = new Float64Array(UP_HEIGHT_EDGES.length + 1);
  const levels = new Map();
  for (let i = 0; i < T; i += 1) {
    if (!(nrm[i * 3 + 1] > 0.9)) continue;
    const y = cen[i * 3 + 1] - ground;
    let b = 0;
    while (b < UP_HEIGHT_EDGES.length && y >= UP_HEIGHT_EDGES[b]) b += 1;
    upH[b] += area[i];
    const key = Math.round(y / 0.02);
    levels.set(key, (levels.get(key) || 0) + area[i]);
  }
  let upLevels = 0, upTopShare = 0, upMainHeight = 0;
  if (nUp > 0) {
    for (let b = 0; b < upH.length; b += 1) upH[b] /= nUp;
    // Merge adjacent 2 cm bins into levels; keep levels with at least 3 % of the upward area.
    const keys = [...levels.keys()].sort((a, b) => a - b);
    const merged = [];
    for (const key of keys) {
      const last = merged[merged.length - 1];
      if (last && key - last.hiKey <= 1) {
        last.area += levels.get(key);
        last.hiKey = key;
        if (levels.get(key) > last.peak) { last.peak = levels.get(key); last.y = key * 0.02; }
      } else merged.push({ hiKey: key, area: levels.get(key), peak: levels.get(key), y: key * 0.02 });
    }
    const kept = merged.filter((l) => l.area >= 0.03 * nUp);
    upLevels = kept.length;
    if (kept.length) {
      upTopShare = kept[kept.length - 1].area / nUp;
      upMainHeight = kept.reduce((best, l) => (l.area > best.area ? l : best)).y;
    }
  }

  // Planar segments and connected parts: one pass over the edges.
  const seg = new Int32Array(T), part = new Int32Array(T);
  for (let i = 0; i < T; i += 1) { seg[i] = i; part[i] = i; }
  let esize = 1;
  while (esize < T * 6) esize <<= 1;
  const emask = esize - 1;
  const ea = new Int32Array(esize).fill(-1), eb = new Int32Array(esize), et = new Int32Array(esize);
  const COS5 = Math.cos((5 * Math.PI) / 180);
  for (let i = 0; i < T; i += 1) {
    if (!area[i]) continue;
    for (let e = 0; e < 3; e += 1) {
      let a = tv[i * 3 + e], b = tv[i * 3 + ((e + 1) % 3)];
      if (a === b) continue;
      if (a > b) [a, b] = [b, a];
      let h = (Math.imul(a, 0x9e3779b1) ^ Math.imul(b, 0x85ebca77)) & emask;
      for (;;) {
        if (ea[h] < 0) {
          ea[h] = a; eb[h] = b; et[h] = i;
          break;
        }
        if (ea[h] === a && eb[h] === b) {
          const j = et[h];
          union(part, i, j);
          const dot = nrm[i * 3] * nrm[j * 3] + nrm[i * 3 + 1] * nrm[j * 3 + 1] + nrm[i * 3 + 2] * nrm[j * 3 + 2];
          if (dot > COS5) {
            const di = nrm[i * 3] * (cen[j * 3] - cen[i * 3]) + nrm[i * 3 + 1] * (cen[j * 3 + 1] - cen[i * 3 + 1]) + nrm[i * 3 + 2] * (cen[j * 3 + 2] - cen[i * 3 + 2]);
            const dj = nrm[j * 3] * (cen[i * 3] - cen[j * 3]) + nrm[j * 3 + 1] * (cen[i * 3 + 1] - cen[j * 3 + 1]) + nrm[j * 3 + 2] * (cen[i * 3 + 2] - cen[j * 3 + 2]);
            if (Math.abs(di) < 0.01 && Math.abs(dj) < 0.01) union(seg, i, j);
          }
          break;
        }
        h = (h + 1) & emask;
      }
    }
  }
  const segArea = new Map(), partArea = new Map();
  for (let i = 0; i < T; i += 1) {
    if (!area[i]) continue;
    const s = find(seg, i), p = find(part, i);
    segArea.set(s, (segArea.get(s) || 0) + area[i]);
    partArea.set(p, (partArea.get(p) || 0) + area[i]);
  }
  const segShares = sharesOf(segArea, A);
  const partShares = sharesOf(partArea, A);
  const segN1 = segShares.filter((x) => x >= 0.01).length;
  const segN5 = segShares.filter((x) => x >= 0.05).length;
  const segCover = segShares.filter((x) => x >= 0.01).reduce((s, x) => s + x, 0);
  const partsN = partShares.filter((x) => x >= 0.01).length;

  // Voxel occupancy: an 8 x 8 x 8 grid on the element's own box (plan axes, height),
  // a 16 x 16 plan grid, and four height layers of an 8 x 8 plan grid.
  let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
  const su = new Float64Array(samples), sv = new Float64Array(samples);
  for (let s = 0; s < samples; s += 1) {
    const dx = sx[s] - mx, dz = sz[s] - mz;
    su[s] = dx * ca + dz * sa;
    sv[s] = -dx * sa + dz * ca;
    u0 = Math.min(u0, su[s]); u1 = Math.max(u1, su[s]);
    v0 = Math.min(v0, sv[s]); v1 = Math.max(v1, sv[s]);
  }
  const cell = (x, x0, x1, n) => (x1 - x0 > 1e-9 ? Math.min(n - 1, Math.floor(((x - x0) / (x1 - x0)) * n)) : 0);
  const vox = new Uint8Array(512), plan = new Uint8Array(256), layers = new Uint8Array(4 * 64);
  for (let s = 0; s < samples; s += 1) {
    const iu8 = cell(su[s], u0, u1, 8), iv8 = cell(sv[s], v0, v1, 8), iy8 = cell(sy[s], bottom, bottom + hgt, 8);
    vox[(iu8 * 8 + iv8) * 8 + iy8] = 1;
    plan[cell(su[s], u0, u1, 16) * 16 + cell(sv[s], v0, v1, 16)] = 1;
    layers[cell(sy[s], bottom, bottom + hgt, 4) * 64 + iu8 * 8 + iv8] = 1;
  }
  const count = (arr, from, to) => { let c = 0; for (let i = from; i < to; i += 1) c += arr[i]; return c; };
  const occ8 = count(vox, 0, 512) / 512;
  const planFill = count(plan, 0, 256) / 256;
  const layer = [0, 1, 2, 3].map((l) => count(layers, l * 64, (l + 1) * 64) / 64);

  // Local covariance features: k nearest samples of the first `local` samples (brute force).
  const Q = Math.min(local, samples);
  const kk = Math.min(k, samples);
  const nbr = new Int32Array(kk), nd = new Float64Array(kk);
  let sLin = 0, sPla = 0, sSca = 0, qLin = 0, qPla = 0, qSca = 0, n1 = 0, n2 = 0, n3 = 0, sVert = 0, sCurv = 0;
  for (let q = 0; q < Q; q += 1) {
    let filled = 0;
    for (let j = 0; j < samples; j += 1) {
      const d = (sx[j] - sx[q]) ** 2 + (sy[j] - sy[q]) ** 2 + (sz[j] - sz[q]) ** 2;
      if (filled < kk) {
        let p = filled++;
        while (p > 0 && nd[p - 1] > d) { nd[p] = nd[p - 1]; nbr[p] = nbr[p - 1]; p -= 1; }
        nd[p] = d; nbr[p] = j;
      } else if (d < nd[kk - 1]) {
        let p = kk - 1;
        while (p > 0 && nd[p - 1] > d) { nd[p] = nd[p - 1]; nbr[p] = nbr[p - 1]; p -= 1; }
        nd[p] = d; nbr[p] = j;
      }
    }
    const lc = covariance(sx, sy, sz, nbr, kk);
    const le = eigSym3(...lc);
    const f = eigenFeatures(...le.values);
    sLin += f.lin; sPla += f.pla; sSca += f.sca;
    qLin += f.lin * f.lin; qPla += f.pla * f.pla; qSca += f.sca * f.sca;
    if (f.lin >= f.pla && f.lin >= f.sca) n1 += 1;
    else if (f.pla >= f.sca) n2 += 1;
    else n3 += 1;
    sVert += 1 - Math.abs(le.vectors[2][1]);
    sCurv += f.curv;
  }
  const sd = (s, q2) => Math.sqrt(Math.max(0, q2 / Q - (s / Q) ** 2));

  const boxSurface = 2 * (len * wid + len * hgt + wid * hgt);
  const out = [
    len, wid, hgt, above, above + hgt, Math.log10(len * wid * hgt + 1e-6), len > 1e-9 ? wid / len : 0, hgt / Math.max(len, 1e-3),
    Math.log10(A + 1e-6), boxSurface > 1e-9 ? A / boxSurface : 0, Math.log10(T), floorKnown,
    g.lin, g.pla, g.sca, g.omni, g.ani, g.ent, g.curv, Math.log10(g.sum + 1e-9), axisVert, verticality,
    nUp / A, nDown / A, nSide / A, nSlant / A, absNy / A, orientEntropy,
    ...hRel,
    ...upH,
    upLevels, upTopShare, upMainHeight,
    Math.log2(1 + segN1), segN5, segShares[0] || 0, (segShares[0] || 0) + (segShares[1] || 0) + (segShares[2] || 0), segCover,
    Math.log2(1 + partsN), partShares[0] || 0,
    occ8, planFill, ...layer,
    sLin / Q, sPla / Q, sSca / Q, sd(sLin, qLin), sd(sPla, qPla), sd(sSca, qSca), n1 / Q, n2 / Q, n3 / Q, sVert / Q, sCurv / Q,
  ];
  if (out.length !== FEATURE_NAMES.length) throw new Error(`feature count ${out.length} != ${FEATURE_NAMES.length}`);
  return Float64Array.from(out, (x) => (Number.isFinite(x) ? x : 0));
}

// ---------------------------------------------------------------- furniture sub-types

/**
 * Furniture and appliance sub-types for the study, ordered from the specific
 * to the generic. They follow the ignition catalogue's scenarios where one
 * applies (src/core/ignitionCatalog.js): upholstered seating (smoking
 * materials), beds and mattresses (smoking materials, candles, chargers),
 * desks and workstations (waste bins), cooking, refrigeration, laundry,
 * dishwashers and electronics; the rest are what a furniture class holds.
 */
export const SUBTYPES = [
  "seat-upholstered", "bed", "chair", "table", "desk", "storage", "counter",
  "cooking", "fridge", "laundry", "dishwasher", "electronics", "lamp", "furniture-other",
];

const FURNITURE_CLASSES = new Set(["IFCFURNITURE", "IFCFURNISHINGELEMENT", "IFCSYSTEMFURNITUREELEMENT"]);
const APPLIANCE_CLASSES = new Set(["IFCELECTRICAPPLIANCE", "IFCAUDIOVISUALAPPLIANCE", "IFCCOMMUNICATIONSAPPLIANCE"]);

const SUBTYPE_PREDEFINED = {
  SOFA: "seat-upholstered", BED: "bed", CHAIR: "chair", TABLE: "table", DESK: "desk",
  FILECABINET: "storage", SHELF: "storage", TECHNICALCABINET: "storage", WORKSURFACE: "desk",
  ELECTRICCOOKER: "cooking", MICROWAVE: "cooking", REFRIGERATOR: "fridge", FREEZER: "fridge", FRIDGE_FREEZER: "fridge",
  WASHINGMACHINE: "laundry", TUMBLEDRYER: "laundry", DISHWASHER: "dishwasher",
  TV: "electronics", COMPUTER: "electronics", DISPLAY: "electronics", PRINTER: "electronics", PHOTOCOPIER: "electronics", PROJECTOR: "electronics",
};

const W = "(?:^|[^a-z])";
const SUBTYPE_NAMES = [
  ["lamp", new RegExp(`${W}(lamps?|lights?|lighting|luminaires?|chandelier|sconce|lys)(?![a-z])|leuchte|lampe`)],
  ["storage", new RegExp(`${W}(night ?stands?|night ?tables?|bedside|tv ?(stand|unit|cabinet|bench))(?![a-z])|nachttisch|nattbord`)],
  ["cooking", new RegExp(`${W}(cook ?tops?|cookers?|hob|stoves?|ovens?|range(?! ?hood)|induction|microwaves?|komfyr|ovn)(?![a-z])|herd(?![a-z])|kochfeld|backofen|mikrowelle`)],
  ["fridge", new RegExp(`${W}(refrigerators?|fridges?|freezers?|fryser)(?![a-z])|k(ü|ue)hlschrank|gefrier|kj(ø|o)leskap`)],
  ["laundry", new RegExp(`${W}(washing ?machines?|washers?|tumble ?dryers?|dryers?|driers?|vaskemaskin)(?![a-z])|waschmaschine|trockner|t(ø|o)rketrommel`)],
  ["dishwasher", new RegExp(`${W}(dish ?washers?)(?![a-z])|geschirrsp|sp(ü|ue)lmaschine|oppvask`)],
  ["electronics", new RegExp(`${W}(tvs?|televisions?|monitors?|computers?|pc|printers?|copiers?|photocopiers?|screens?|displays?|projectors?)(?![a-z])|fernseh|bildschirm|drucker`)],
  ["seat-upholstered", new RegExp(`${W}(sofas?|couch(es)?|settees?|love ?seats?|arm ?chairs?|easy ?chairs?|lounge ?chairs?|club ?chairs?|recliners?|ottomans?|poufs?|futons?|fauteuils?|divano|poltrona|sill(ó|o)n|lenestol)(?![a-z])|sessel|polster|kanapee|sofa`)],
  ["bed", new RegExp(`${W}(beds?|mattress(es)?|bunks?|cots?|cribs?|seng)(?![a-z])|bett(?![a-z])|matratze`)],
  ["desk", new RegExp(`${W}(desks?|work ?stations?|cubicles?|escritorio|scrivania|skrivebord)(?![a-z])|schreibtisch|arbeitsplatz`)],
  ["counter", new RegExp(`${W}(counters?|counter ?tops?|worktops?|work ?tops?|kitchen ?(units?|islands?|run)|islands?)(?![a-z])|arbeitsplatte|k(ü|ue)chenzeile`)],
  ["chair", new RegExp(`${W}(chairs?|stools?|benches?|seats?|seating|chaise|sillas?|sedias?|stoel|kruk|stol|stoler|taburete)(?![a-z])|stuhl|st(ü|ue)hle|hocker`)],
  ["table", new RegExp(`${W}(tables?|bord|mesas?|tavol[oi]|tafel)(?![a-z])|tisch`)],
  ["storage", new RegExp(`${W}(cabinets?|cupboards?|wardrobes?|closets?|shel(f|ves|ving)|bookcases?|book ?shel(f|ves)|drawers?|chests?|dressers?|lockers?|casework|sideboards?|credenzas?|vanity|vanities|pantry|armoires?|armadio|kast|skap|hylle|reol|filing)(?![a-z])|schrank|regal|kommode|garderobe`)],
  ["furniture-other", new RegExp(`${W}(planters?|plants?|piano|rugs?|carpets?|curtains?|blinds?|mirrors?|bins?|coat ?racks?|hat ?stands?|easel|whiteboards?|boards?|partitions?|screens?)(?![a-z])`)],
];

/**
 * The furniture sub-type of an element, for the study's ground truth: from
 * its PredefinedType (own or its type's), else its name. Null when neither
 * says (such elements are not used for training or scoring sub-types).
 */
export function furnitureSubtype({ cls, predefined, truthPredefined, name, objectType }) {
  const p = String(truthPredefined || predefined || "").toUpperCase();
  if (SUBTYPE_PREDEFINED[p]) return { subtype: SUBTYPE_PREDEFINED[p], by: "predefined" };
  const text = [name, objectType].filter(Boolean).join(" ").toLowerCase();
  if (!text.trim()) return null;
  for (const [label, re] of SUBTYPE_NAMES) if (re.test(text)) return { subtype: label, by: "name" };
  return null;
}

// ---------------------------------------------------------------- corpus extraction

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const FRONTEND = path.join(ROOT, "topologicpy-web-frontend");
const NOT_ELEMENTS = new Set([
  "IFCSITE", "IFCBUILDING", "IFCBUILDINGSTOREY", "IFCOPENINGELEMENT", "IFCOPENINGSTANDARDCASE", "IFCVIRTUALELEMENT",
  "IFCANNOTATION", "IFCDISTRIBUTIONPORT", "IFCPORT", "IFCGRID", "IFCVOIDINGFEATURE", "IFCPROJECTIONELEMENT",
  "IFCSURFACEFEATURE", "IFCFACILITY", "IFCFACILITYPART", "IFCBRIDGE", "IFCROAD", "IFCRAILWAY", "IFCMARINEFACILITY",
]);
const value = (a) => (a && typeof a === "object" && "value" in a ? a.value : a);
const median = (xs) => {
  if (!xs.length) return null;
  const s = xs.slice().sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

/** The corpus as tools/identify-bench.mjs reads it (same models, same skips). */
async function corpus() {
  const manifest = JSON.parse(readFileSync(path.join(ROOT, "benchmark/manifest.json"), "utf8"));
  const samples = process.env.BENCHMARK_SAMPLES || path.join(os.homedir(), "ifc-samples");
  const skip = new Set(["Ifc4_Revit_ARC_FireRatingAdded", "bsi-ifc4x3-Building-Architecture"]);
  const out = [];
  for (const m of manifest.models) {
    if (m.stressOnly || skip.has(m.name)) continue;
    const file = m.root === "examples" ? path.join(FRONTEND, "public/examples", m.path) : path.join(samples, m.path);
    if (existsSync(file)) out.push({ name: m.name, file });
  }
  const more2 = path.join(samples, "more2");
  if (existsSync(more2)) {
    for (const dir of await readdir(more2)) {
      const full = path.join(more2, dir);
      if (!(await stat(full)).isDirectory()) continue;
      for (const f of await readdir(full)) {
        if (!f.toLowerCase().endsWith(".ifc")) continue;
        const file = path.join(full, f);
        if ((await stat(file)).size >= 50e3) out.push({ name: `${dir}/${path.basename(f, ".ifc")}`, file });
      }
    }
  }
  return out;
}

async function extractModel(api, WebIFC, entry, inDataset) {
  const model = api.OpenModel(new Uint8Array(readFileSync(entry.file)), { COORDINATE_TO_ORIGIN: true });
  try {
    const lines = new Map();
    for (const t of api.GetAllTypesOfModel(model)) {
      const name = String(t.typeName).toUpperCase();
      if (NOT_ELEMENTS.has(name) || !api.IsIfcElement(t.typeID)) continue;
      const ids = api.GetLineIDsWithType(model, t.typeID);
      for (let i = 0; i < ids.size(); i += 1) lines.set(ids.get(i), name);
    }
    const container = new Map(), parent = new Map(), typeOf = new Map();
    const rel = (typeName, fn) => {
      const ids = api.GetLineIDsWithType(model, WebIFC[typeName]);
      for (let i = 0; i < ids.size(); i += 1) fn(api.GetLine(model, ids.get(i)));
    };
    rel("IFCRELCONTAINEDINSPATIALSTRUCTURE", (r) => { for (const e of r.RelatedElements || []) container.set(value(e), value(r.RelatingStructure)); });
    rel("IFCRELAGGREGATES", (r) => { for (const e of r.RelatedObjects || []) parent.set(value(e), value(r.RelatingObject)); });
    rel("IFCRELDEFINESBYTYPE", (r) => { for (const e of r.RelatedObjects || []) typeOf.set(value(e), value(r.RelatingType)); });
    const storeyIds = new Set();
    {
      const ids = api.GetLineIDsWithType(model, WebIFC.IFCBUILDINGSTOREY);
      for (let i = 0; i < ids.size(); i += 1) storeyIds.add(ids.get(i));
    }
    const storeyOf = (id) => {
      let cur = id;
      for (let hop = 0; hop < 8 && cur != null; hop += 1) {
        if (storeyIds.has(cur)) return cur;
        cur = container.get(cur) ?? parent.get(cur);
      }
      return null;
    };
    const records = [];
    for (const [id, cls] of lines) {
      const line = api.GetLine(model, id);
      const own = value(line.PredefinedType) || "";
      let typePredefined = "", typeName = "";
      const t = typeOf.get(id);
      if (t != null) {
        try {
          const tl = api.GetLine(model, t);
          typePredefined = value(tl?.PredefinedType) || "";
          typeName = value(tl?.Name) || "";
        } catch {
          typePredefined = "";
        }
      }
      const truthPredefined = own && !/^(NOTDEFINED|USERDEFINED)$/.test(own) ? own : typePredefined || own;
      records.push({
        expressId: id, cls, truthPredefined, label: labelForClass(cls, truthPredefined),
        name: value(line.Name) || "", objectType: value(line.ObjectType) || "", longName: value(line.LongName) || "",
        predefined: own || value(line.OperationType) || "", typeName, storeyId: storeyOf(id),
      });
    }
    const id = (r) => `${entry.name}#${r.expressId}`;
    const chosen = records.filter((r) => inDataset.has(id(r)) || FURNITURE_CLASSES.has(r.cls) || APPLIANCE_CLASSES.has(r.cls));
    const context = records.filter((r) => ["door", "wall"].includes(r.label));
    const want = [...new Set([...chosen, ...context].map((r) => r.expressId))];
    const meshesById = new Map();
    api.StreamMeshes(model, want, (mesh) => {
      const list = [];
      const placed = mesh.geometries;
      for (let g = 0; g < placed.size(); g += 1) {
        const pg = placed.get(g);
        const geom = api.GetGeometry(model, pg.geometryExpressID);
        const verts = api.GetVertexArray(geom.GetVertexData(), geom.GetVertexDataSize());
        const index = api.GetIndexArray(geom.GetIndexData(), geom.GetIndexDataSize());
        const m = pg.flatTransformation;
        const n = verts.length / 6;
        const positions = new Float32Array(n * 3);
        for (let v = 0; v < n; v += 1) {
          const x = verts[v * 6], y = verts[v * 6 + 1], z = verts[v * 6 + 2];
          positions[v * 3] = m[0] * x + m[4] * y + m[8] * z + m[12];
          positions[v * 3 + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
          positions[v * 3 + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
        }
        list.push({ positions, indices: Uint32Array.from(index) });
        geom.delete?.();
      }
      if (list.length) meshesById.set(mesh.expressID, (meshesById.get(mesh.expressID) || []).concat(list));
    });
    const boxes = new Map();
    for (const [eid, meshes] of meshesById) {
      const b = boxOf(meshes);
      if (b && Number.isFinite(b.dims[0])) boxes.set(eid, b);
    }
    // Floor level per storey, as the benchmark computes it: the median bottom of its doors, else of its walls.
    const floor = new Map();
    for (const s of storeyIds) {
      const doors = records.filter((r) => r.storeyId === s && r.label === "door" && boxes.has(r.expressId)).map((r) => boxes.get(r.expressId).bottom);
      const walls = records.filter((r) => r.storeyId === s && r.label === "wall" && boxes.has(r.expressId)).map((r) => boxes.get(r.expressId).bottom);
      const level = median(doors) ?? median(walls);
      if (level != null) floor.set(s, level);
    }
    const out = [];
    const times = [];
    for (const r of chosen) {
      const meshes = meshesById.get(r.expressId);
      const box = boxes.get(r.expressId);
      if (!meshes || !box) continue;
      const level = r.storeyId != null ? floor.get(r.storeyId) : undefined;
      const t0 = performance.now();
      const f = elementFeatures(meshes, { floor: level ?? null, seed: r.expressId });
      const ms = performance.now() - t0;
      if (!f) continue;
      times.push(ms);
      const dims = box.dims.map((v) => Math.round(v * 1000) / 1000);
      const above = level != null ? Math.round((box.bottom - level) * 1000) / 1000 : null;
      const rules = identify({ name: r.name, objectType: r.objectType, longName: r.longName, predefined: r.predefined, dims, above });
      const sub = FURNITURE_CLASSES.has(r.cls) || APPLIANCE_CLASSES.has(r.cls) || GENERIC_CLASSES.has(r.cls)
        ? furnitureSubtype({ ...r, objectType: [r.objectType, r.typeName].filter(Boolean).join(" ") })
        : null;
      const rec = {
        id: id(r), model: entry.name, expressId: r.expressId, cls: r.cls, label: GENERIC_CLASSES.has(r.cls) ? null : r.label,
        name: r.name, objectType: r.objectType, typeName: r.typeName, predefined: r.predefined, truthPredefined: r.truthPredefined,
        dims, above, triangles: meshes.reduce((s, m) => s + m.indices.length / 3, 0),
        rules,
        // The same rules with Name, ObjectType and LongName withheld (PredefinedType and box only): an element named by a code.
        rulesNoName: identify({ predefined: r.predefined, dims, above }),
        ignition: subtypeOf([r.name, r.objectType].filter(Boolean).join(" ")),
        subtype: sub?.subtype ?? null, subtypeBy: sub?.by ?? null, ms: Math.round(ms * 1000) / 1000, f: Array.from(f, (x) => Math.round(x * 1e6) / 1e6),
      };
      if (!inDataset.has(rec.id)) rec.points = samplePoints(meshes, { points: 2048, seed: r.expressId, frame: "viewer" });
      out.push(rec);
    }
    return { records: out, times, elements: records.length };
  } finally {
    api.CloseModel(model);
  }
}

async function extractAll(args) {
  const flag = (name, fallback) => {
    const i = args.indexOf(name);
    return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
  };
  const BENCH = path.resolve(flag("--bench", path.join(os.homedir(), "ifcnet-src", "bench-data")));
  const OUT = path.resolve(flag("--out", path.join(os.homedir(), "ifcnet-src", "pilot")));
  for (const dir of [BENCH, OUT]) if (dir.startsWith(ROOT)) throw new Error("--bench and --out must be outside the repository (third-party names stay local).");
  const ds = JSON.parse(readFileSync(path.join(BENCH, "dataset.json"), "utf8"));
  const datasetIndex = new Map(ds.records.map((r) => [r.id, r.index]));
  const require = createRequire(import.meta.url);
  const WebIFC = require(path.join(FRONTEND, "node_modules/web-ifc/web-ifc-api-node.js"));
  const api = new WebIFC.IfcAPI();
  api.SetWasmPath(path.join(FRONTEND, "node_modules/web-ifc/"), true);
  await api.Init();
  await mkdir(OUT, { recursive: true });
  const all = [];
  const times = [];
  for (const entry of await corpus()) {
    const t0 = performance.now();
    try {
      const { records, times: ts, elements } = await extractModel(api, WebIFC, entry, datasetIndex);
      all.push(...records);
      times.push(...ts);
      console.log(`${entry.name}: ${elements} elements, ${records.length} with features (${Math.round(performance.now() - t0)} ms)`);
    } catch (error) {
      console.log(`${entry.name}: failed (${error.message})`);
    }
  }
  const extra = all.filter((r) => r.points);
  const pts = new Float32Array(extra.length * 2048 * 3);
  extra.forEach((r, i) => {
    pts.set(r.points, i * 2048 * 3);
    r.extraIndex = i;
    delete r.points;
  });
  for (const r of all) if (datasetIndex.has(r.id)) r.datasetIndex = datasetIndex.get(r.id);
  const sorted = times.slice().sort((a, b) => a - b);
  const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  const timing = { n: sorted.length, medianMs: q(0.5), p90Ms: q(0.9), p99Ms: q(0.99), maxMs: sorted[sorted.length - 1], meanMs: sorted.reduce((a, b) => a + b, 0) / sorted.length };
  await writeFile(path.join(OUT, "extra-points.bin"), Buffer.from(pts.buffer));
  await writeFile(path.join(OUT, "features.json"), JSON.stringify({ featureNames: FEATURE_NAMES, subtypes: SUBTYPES, timing, node: process.version, records: all }));
  const have = new Set(all.map((r) => r.id));
  const missing = ds.records.filter((r) => !have.has(r.id)).length;
  console.log(`\n${all.length} records (${extra.length} beyond the dataset; ${missing} dataset records without features). Feature time per element:`, timing);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  if (args[0] === "extract") await extractAll(args);
  else console.log("usage: node tools/identify-features.mjs extract [--bench DIR] [--out DIR]");
}
