/**
 * Running the Flamescapes solver (src/core/flamescapes.js) on an IFC model
 * and reading it at the walkable graph, for the Simulate step's optional
 * "Flamescapes" fire models.
 *
 * Scene. The IFC's walls, slabs, roofs, stairs and other obstacles
 * (windows, columns, railings) are inert solids; doors are left open; the
 * room contents are the paper's statistical interior fuel (Eqs. 3 to 5) on
 * every covered floor (voxelize.js). The first item to burn sits at the
 * fire origin with the fuel of the chosen fire class, lit by a burner.
 *
 * Domain. The grid holds the whole model when it fits `maxCells` at a cell
 * of at most `maxCell` metres; a larger model is clipped to a box round the
 * fire (`reach` metres across, the fire's storey and the next two up). Nodes
 * outside the box are read as clear air, and the result says so.
 *
 * Reading. Each graph node stands `agentHeight` above its floor; the solver
 * is read at head height above that floor (`headHeight`), in the cell there
 * or, if that cell is solid, the nearest open cell above or below it:
 *   gas temperature (degrees C);
 *   visibility, S = C / (sigma_m rho Y_soot) with C = 3 for light-reflecting
 *   signs and the mass extinction coefficient of flaming smoke sigma_m =
 *   8700 m^2/kg (Mulholland and Croarkin 2000), capped at 100 m;
 *   CO in ppm by volume.
 *
 * Forecast. The planners (forecast.js, deadline.js) need, per node and
 * frame, a dose rate that never falls and the time each node becomes
 * untenable. Here the dose rate is the convective-heat FED rate of the gas
 * temperature plus the CO FED rate of ISO 13571, ppm / 35000 per minute,
 * kept as a running maximum; a node is untenable from the first frame its
 * gas is at the threshold or, with the smoke criterion, its visibility is
 * below the limit (10 m by default; PD 7974-6 gives 5 m for small rooms and
 * 10 m for large spaces).
 */

import { DEFAULTS, NS, SPECIES, MOLAR, SOLID as SOLID_KIND, createSolver, step } from "./flamescapes.js";
import { MATERIAL, gridFor, interiorCells, interiorFuel, occupancy, raysFor, toLocal, volumeFractions } from "./voxelize.js";
// The CO dose and the forecast from frames live in forecast.js, so the
// forecast worker does not load the solver; re-exported here.
export { CO_LIMIT, coDoseRate, forecastFromFrames } from "./forecast.js";

export const KELVIN = 273.15;
export const SOOT_EXTINCTION = 8700; // m^2 / kg, flaming smoke (Mulholland and Croarkin 2000)
export const SIGN_CONSTANT = 3; // light-reflecting signs
export const MAX_VISIBILITY = 100; // m

/** Visibility (m) through smoke of soot density rhoSoot (kg/m^3). */
export function visibility(rhoSoot, C = SIGN_CONSTANT, sigma = SOOT_EXTINCTION) {
  const K = sigma * rhoSoot;
  return K > C / MAX_VISIBILITY ? C / K : MAX_VISIBILITY;
}

/**
 * The scene for the solver: meshes tagged with materials, the grid, volume
 * fractions, occupancy, the contents and the ignition cell.
 *
 * @param {object} geometry  {walls, floors, roofs, stairs, obstacles} meshes in the viewer frame
 * @param {object} opts      upAxis; fire: [x, y, z] in the viewer frame; maxCells; maxCell (m);
 *                           reach (m); fireLoad (J/m^2); contentsFuel, ignitionFuel; item: {mass, area};
 *                           burner: {power (W), duration (s)}; seed
 */
export function buildScene(geometry, {
  upAxis = "y",
  fire,
  maxCells = 30000,
  minCell = 0.2,
  maxCell = 0.35,
  reach = null,
  storeyHeight = 3,
  fireLoad = 780e6,
  contentsFuel,
  ignitionFuel = contentsFuel,
  item = { mass: 5, area: 1 },
  burner = { temperature: 900, duration: 30 },
  seed = 1,
} = {}) {
  const tag = (list, material) => (list || []).map((m) => ({ positions: m.positions, indices: m.indices, material }));
  let meshes = [
    ...tag(geometry.walls, MATERIAL.STONE),
    ...tag(geometry.floors, MATERIAL.STONE),
    ...tag(geometry.roofs, MATERIAL.STONE),
    ...tag(geometry.stairs, MATERIAL.STONE),
    ...tag(geometry.obstacles, MATERIAL.STONE),
  ];
  if (!meshes.length) throw new Error("The model has no walls or floors to build a fire grid from.");

  // Try the whole model; clip round the fire when the cell would be too coarse.
  let grid = gridFor(meshes, { upAxis, cell: minCell, margin: 2, headroom: 2, maxCells });
  let clipped = false;
  if (grid.h > maxCell && fire) {
    // A square round the fire, as wide as the budget allows at maxCell,
    // over the fire's storey and the next one up.
    const height = storeyHeight * 2.5;
    const side = reach ?? Math.max(8, Math.sqrt((maxCells * maxCell ** 3) / height) - 4 * maxCell);
    const f = toLocal({ axes: grid.axes }, fire[0], fire[1], fire[2]);
    const box = { lo: [f[0] - side / 2, f[1] - storeyHeight * 0.5, f[2] - side / 2], hi: [f[0] + side / 2, f[1] + storeyHeight * 2, f[2] + side / 2] };
    meshes = meshes.map((m) => clipMesh(m, grid.axes, box)).filter((m) => m.indices.length);
    grid = gridFor(meshes, { upAxis, cell: minCell, margin: 2, headroom: 2, maxCells, bounds: box });
    clipped = true;
  }
  const fraction = volumeFractions(grid, meshes, { rays: raysFor(grid) });
  const solid = occupancy(grid, fraction);
  const cells = interiorCells(grid, fraction, { ceiling: 5, layers: 2, solidMask: solid });
  const fuel = interiorFuel(grid, cells, { fireLoad, heatOfCombustion: contentsFuel.heatOfCombustion, spread: 0.3, layers: 2, seed });

  // The first item: the contents cell nearest the fire origin with open air
  // above it for 1.5 m (not one inside a wall's column of cells), else the
  // open cell at the origin.
  let ignitionCell = -1;
  if (fire) {
    const [a, b, c] = toLocal(grid, fire[0], fire[1], fire[2]);
    const fi = (a - grid.origin[0]) / grid.h, fj = (b - grid.origin[1]) / grid.h, fk = (c - grid.origin[2]) / grid.h;
    const above = Math.max(1, Math.round(1.5 / grid.h));
    let best = Infinity;
    for (const cell of cells) {
      const i = cell % grid.nx, j = Math.floor(cell / grid.nx) % grid.ny, k = Math.floor(cell / (grid.nx * grid.ny));
      let open = true;
      for (let up = 1; up <= above && open; up += 1) if (j + up >= grid.ny || solid[i + grid.nx * (j + up + grid.ny * k)]) open = false;
      if (!open) continue;
      const d = (i + 0.5 - fi) ** 2 + 4 * (j + 0.5 - fj) ** 2 + (k + 0.5 - fk) ** 2;
      if (d < best) {
        best = d;
        ignitionCell = cell;
      }
    }
    if (ignitionCell < 0 || best > 9) {
      // No contents near the origin (an uncovered floor, or the origin in a
      // wall's column of cells): the nearest open cell standing on a solid
      // floor within 3 m, with open air above it.
      ignitionCell = -1;
      const reach = Math.ceil(3 / grid.h);
      const i0 = Math.floor(fi), j0 = Math.floor(fj), k0 = Math.floor(fk);
      let near = Infinity;
      for (let k = Math.max(0, k0 - reach); k <= Math.min(grid.nz - 1, k0 + reach); k += 1) {
        for (let i = Math.max(0, i0 - reach); i <= Math.min(grid.nx - 1, i0 + reach); i += 1) {
          for (let j = Math.max(1, j0 - reach); j <= Math.min(grid.ny - 2, j0 + reach); j += 1) {
            const cell = i + grid.nx * (j + grid.ny * k);
            if (solid[cell] || !solid[cell - grid.nx] || solid[cell + grid.nx]) continue;
            const d = (i + 0.5 - fi) ** 2 + 4 * (j + 0.5 - fj) ** 2 + (k + 0.5 - fk) ** 2;
            if (d < near) {
              near = d;
              ignitionCell = cell;
            }
          }
        }
      }
    }
  }
  return {
    grid,
    fraction,
    solid,
    contents: { cells, masses: fuel.masses },
    contentsSummary: { cells: cells.length, area: fuel.area, mass: fuel.total, energy: fuel.energy },
    ignition: ignitionCell >= 0 ? { cell: ignitionCell, mass: item.mass, area: item.area } : null,
    pilot: ignitionCell >= 0 ? { cells: [ignitionCell], ...burner } : null,
    clipped,
    fuels: { contentsFuel, ignitionFuel },
  };
}

/** Keep the triangles of a mesh with a vertex inside the box (local frame). */
function clipMesh(mesh, axes, box) {
  const P = mesh.positions;
  const kept = [];
  const ix = mesh.indices;
  // A triangle is kept when its bounding box overlaps the clip box on every
  // axis. Keeping only triangles with a vertex inside dropped large wall and
  // slab faces that cross the box (review 2): a dropped face leaves the ray
  // parity odd and that column of wall or slab vanishes. volumeFractions
  // clamps rays and heights to the grid, so the parts outside are harmless.
  for (let t = 0; t < ix.length; t += 3) {
    let overlaps = true;
    for (let d = 0; d < 3 && overlaps; d += 1) {
      const a = P[ix[t] * 3 + axes[d]], b = P[ix[t + 1] * 3 + axes[d]], c = P[ix[t + 2] * 3 + axes[d]];
      if (Math.max(a, b, c) < box.lo[d] || Math.min(a, b, c) > box.hi[d]) overlaps = false;
    }
    if (overlaps) kept.push(ix[t], ix[t + 1], ix[t + 2]);
  }
  return { ...mesh, indices: Uint32Array.from(kept) };
}

/**
 * For each graph node, the cell read at head height: the cell there, or the
 * nearest open cell within two above or below; -1 outside the grid.
 */
export function headCells(sim, nodes, { agentHeight = 0.75, headHeight = 1.8 } = {}) {
  const g = sim.grid;
  const n = nodes.length / 3;
  const out = new Int32Array(n).fill(-1);
  const lift = headHeight - agentHeight;
  for (let v = 0; v < n; v += 1) {
    const [a, b, c] = toLocal(g, nodes[v * 3], nodes[v * 3 + 1], nodes[v * 3 + 2]);
    const i = Math.floor((a - g.origin[0]) / g.h);
    const j0 = Math.floor((b + lift - g.origin[1]) / g.h);
    const k = Math.floor((c - g.origin[2]) / g.h);
    if (i < 0 || k < 0 || i >= g.nx || k >= g.nz) continue;
    for (const dj of [0, -1, 1, -2, 2]) {
      const j = j0 + dj;
      if (j < 0 || j >= g.ny) continue;
      const cell = i + g.nx * (j + g.ny * k);
      if (sim.kind[cell] !== 2) {
        out[v] = cell;
        break;
      }
    }
  }
  return out;
}

/** Read gas temperature (C), visibility (m) and CO (ppm) at the nodes' head cells. */
export function readNodes(sim, cells, out = null) {
  const n = cells.length;
  const T = out?.T || new Float32Array(n);
  const vis = out?.vis || new Float32Array(n);
  const co = out?.co || new Float32Array(n);
  const ambient = sim.P.ambient - KELVIN;
  for (let v = 0; v < n; v += 1) {
    const c = cells[v];
    if (c < 0) {
      T[v] = ambient;
      vis[v] = MAX_VISIBILITY;
      co[v] = 0;
      continue;
    }
    T[v] = sim.T[c] - KELVIN;
    vis[v] = visibility(sim.rho[c] * sim.Y[c * NS + SPECIES.SOOT]);
    let moles = 0;
    let rest = 1;
    for (let q = 0; q < NS; q += 1) {
      moles += sim.Y[c * NS + q] / MOLAR[q];
      rest -= sim.Y[c * NS + q];
    }
    moles += Math.max(0, rest) / 0.028;
    co[v] = moles > 0 ? ((sim.Y[c * NS + SPECIES.CO] / MOLAR[SPECIES.CO]) / moles) * 1e6 : 0;
  }
  return { T, vis, co };
}

/** Up to `max` particles in the viewer frame: [x, y, z] triples, temperatures (K) and kinds. */
export function particleSnapshot(sim, max = 3000) {
  const pr = sim.parts;
  const g = sim.grid;
  const count = Math.min(pr.n, max);
  const stride = pr.n > max ? pr.n / max : 1;
  const positions = new Float32Array(count * 3);
  const temps = new Float32Array(count);
  const kinds = new Uint8Array(count);
  for (let q = 0; q < count; q += 1) {
    const p = Math.floor(q * stride);
    const a = g.origin[0] + pr.x[p * 3] * g.h;
    const b = g.origin[1] + pr.x[p * 3 + 1] * g.h;
    const c = g.origin[2] + pr.x[p * 3 + 2] * g.h;
    positions[q * 3 + g.axes[0]] = a;
    positions[q * 3 + g.axes[1]] = b;
    positions[q * 3 + g.axes[2]] = c;
    temps[q] = pr.T[p];
    kinds[q] = pr.kind[p];
  }
  return { positions, temps, kinds };
}

/**
 * The smoke as a cloud to draw: the soot density (rho Y_soot, the paper's
 * residual species) averaged over blocks of `block` cubed cells, one point at
 * the centre of each block's open cells (viewer frame), with the opacity of
 * light crossing the block, 1 - exp(-sigma_m rho_s block h) with Mulholland
 * and Croarkin's sigma_m (the visibility's own coefficient), as 0 to 255.
 * Blocks under `min` opacity are left out. The GPU passes its read-back
 * fields, so both backends draw the same cloud.
 *
 * @returns {{positions: Float32Array, opacity: Uint8Array, size: number}}
 */
export function smokeCloud(grid, rho, Y, kind, { block = 2, min = 0.03 } = {}) {
  const { nx, ny, nz, h } = grid;
  const bx = Math.ceil(nx / block), by = Math.ceil(ny / block), bz = Math.ceil(nz / block);
  const sum = new Float32Array(bx * by * bz);
  const open = new Uint16Array(bx * by * bz);
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const c = i + nx * (j + ny * k);
        if (kind[c] === SOLID_KIND) continue;
        const b = Math.floor(i / block) + bx * (Math.floor(j / block) + by * Math.floor(k / block));
        sum[b] += rho[c] * Y[c * NS + SPECIES.SOOT];
        open[b] += 1;
      }
    }
  }
  const keep = [];
  const opacity = [];
  const depth = SOOT_EXTINCTION * block * h;
  for (let b = 0; b < sum.length; b += 1) {
    if (!open[b]) continue;
    const alpha = 1 - Math.exp(-depth * (sum[b] / open[b]));
    if (alpha < min) continue;
    keep.push(b);
    opacity.push(Math.min(255, Math.round(alpha * 255)));
  }
  const positions = new Float32Array(keep.length * 3);
  keep.forEach((b, q) => {
    const i = b % bx, j = Math.floor(b / bx) % by, k = Math.floor(b / (bx * by));
    const centre = (n, at) => (at * block + Math.min(block, n - at * block) / 2) * h;
    positions[q * 3 + grid.axes[0]] = grid.origin[0] + centre(nx, i);
    positions[q * 3 + grid.axes[1]] = grid.origin[1] + centre(ny, j);
    positions[q * 3 + grid.axes[2]] = grid.origin[2] + centre(nz, k);
  });
  return { positions, opacity: Uint8Array.from(opacity), size: block * h };
}

/**
 * Run the solver to `horizon` seconds, calling onFrame every `frameSeconds`
 * with the node readings, the frame's mean heat release rate (kW), the
 * particles and, with `smoke`, the smoke cloud (smokeCloud). Returns the
 * frames stacked: T, vis, co as Float32Array(K n).
 */
export function runScene(scene, { nodes, agentHeight = 0.75, headHeight = 1.8, horizon = 300, frameSeconds = 5, embers = false, dt = DEFAULTS.dt, onFrame = null, shouldStop = null, particles = 3000, smoke = false, solverOptions = {}, profileInto = null, keep = true } = {}) {
  const sim = createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, embers, dt, seed: 1, ...solverOptions });
  // Per-stage timing for the benchmark (tools/fire-bench.mjs).
  if (profileInto) sim.profile = profileInto;
  const cells = headCells(sim, nodes, { agentHeight, headHeight });
  const n = cells.length;
  const K = Math.max(1, Math.round(horizon / frameSeconds));
  // With keep, every frame is stacked (K x n); without, one frame's buffers
  // are reused and onFrame must copy what it keeps (the worker does).
  const rows = keep ? K : 1;
  const T = new Float32Array(rows * n);
  const vis = new Float32Array(rows * n);
  const co = new Float32Array(rows * n);
  const hrr = new Float32Array(K);
  let released = 0;
  for (let k = 0; k < K; k += 1) {
    const end = (k + 1) * frameSeconds;
    while (sim.time < end - 1e-9) {
      step(sim, Math.min(dt, end - sim.time));
      if (shouldStop?.()) return null;
    }
    const row = keep ? k : 0;
    const frame = readNodes(sim, cells, { T: T.subarray(row * n, (row + 1) * n), vis: vis.subarray(row * n, (row + 1) * n), co: co.subarray(row * n, (row + 1) * n) });
    hrr[k] = (sim.stats.released - released) / frameSeconds / 1000;
    released = sim.stats.released;
    onFrame?.({
      k,
      time: end,
      T: frame.T,
      vis: frame.vis,
      co: frame.co,
      hrr: hrr[k],
      maxT: sim.stats.maxT - KELVIN,
      embers: sim.stats.embersSpawned,
      spots: sim.stats.spotIgnitions,
      burnt: sim.stats.burnt,
      particles: particles ? particleSnapshot(sim, particles) : null,
      smoke: smoke ? smokeCloud(sim.grid, sim.rho, sim.Y, sim.kind) : null,
    });
  }
  const inGrid = cells.reduce((a, c) => a + (c >= 0 ? 1 : 0), 0);
  const out = { n, K, frameSeconds, horizon: K * frameSeconds, hrr, stats: { ...sim.stats }, inGrid };
  return keep ? { ...out, T, vis, co } : out;
}
