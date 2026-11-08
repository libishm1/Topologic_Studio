/**
 * Running the Flamescapes solver (src/core/flamescapes.js) on an IFC model
 * and reading it at the walkable graph, for the Simulate step's optional
 * "Flamescapes" fire models.
 *
 * Scene. The IFC's walls, slabs, roofs, suspended ceilings, stairs and other
 * obstacles (windows, columns, railings) are inert solids; doors are left open; the
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
 *   CO in ppm by volume;
 *   and, for the extra measures (core/tenability.js), the soot mass
 *   concentration (mg/m^3), CO2 and O2 (% by volume). Outside the grid a
 *   node reads clear air: 20.9 % O2, no CO2 and no soot.
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

import { DEFAULTS, NS, SPECIES, MOLAR, SOLID as SOLID_KIND, burnerRate, cellSpeed, createSolver, flameHeight, step } from "./flamescapes.js";
import { EVANS_MIN_DENSITY, activationFrame, activationSummary, createActivation, createSuppression, evansTau, openHeads, sprinklerSetup } from "./suppression.js";
import { systemsDetect, systemsSetup, systemsSummary } from "./suppressionSystems.js";
import { MATERIAL, gridFor, interiorCells, interiorFuel, occupancy, raysFor, rng, toLocal, volumeFractions } from "./voxelize.js";
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
 * @param {object} geometry  {walls, floors, roofs, ceilings?, stairs, obstacles, doors?} meshes in the viewer
 *                           frame; doors only when they are closed, and open windows left out of obstacles
 * @param {object} opts      upAxis; fire: [x, y, z] in the viewer frame; maxCells; maxCell (m);
 *                           reach (m); fireLoad (J/m^2); contentsFuel, ignitionFuel; item: {mass, area};
 *                           burner: {power (W), duration (s)}; seed;
 *                           ventilation: {terminals: [[x, y, z]] in the viewer frame, flow (m^3/s a terminal)},
 *                           the running ventilation's air terminals (ductsFor)
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
  designFire = null,
  ventilation = null,
  seed = 1,
} = {}) {
  const tag = (list, material) => (list || []).map((m) => ({ positions: m.positions, indices: m.indices, material }));
  let meshes = [
    ...tag(geometry.walls, MATERIAL.STONE),
    ...tag(geometry.floors, MATERIAL.STONE),
    ...tag(geometry.roofs, MATERIAL.STONE),
    // Suspended ceilings (IfcCovering CEILING): the heads hang from them and
    // the plume meets them, not the slab above. About 5 cm thick, they are a
    // seventh of a 0.37 m cell and snap to it as thin elements (occupancy).
    ...tag(geometry.ceilings, MATERIAL.STONE),
    ...tag(geometry.stairs, MATERIAL.STONE),
    ...tag(geometry.obstacles, MATERIAL.STONE),
    // Closed doors, when the run closes them: inert leaves in their openings.
    ...tag(geometry.doors, MATERIAL.STONE),
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
  // The first item as a t-squared design fire (fireClasses.js designFireFor):
  // a burner over its area, with no pilot and none of its fuel in the grid.
  // It stands on its own floor area: the room's statistical contents there
  // are the item, so they leave the grid (the burner heated them directly,
  // as if another item stood inside it).
  const itemFire = designFire && ignitionCell >= 0 ? itemBurner(grid, solid, ignitionCell, { ...designFire, item, fuel: ignitionFuel }) : null;
  let contentsCells = cells, contentsMasses = fuel.masses;
  if (itemFire) {
    const under = new Set(itemFire.layers.flat());
    const keep = [];
    for (let q = 0; q < cells.length; q += 1) if (!under.has(cells[q])) keep.push(q);
    contentsCells = keep.map((q) => cells[q]);
    contentsMasses = keep.map((q) => fuel.masses[q]);
  }
  return {
    grid,
    fraction,
    solid,
    contents: { cells: contentsCells, masses: contentsMasses },
    contentsSummary: { cells: cells.length, area: fuel.area, mass: fuel.total, energy: fuel.energy },
    ignition: ignitionCell >= 0 ? { cell: ignitionCell, mass: item.mass, area: item.area, ...(itemFire ? { designFire: itemFire.design } : {}) } : null,
    pilot: ignitionCell >= 0 && !itemFire ? { cells: [ignitionCell], ...burner } : null,
    ...(itemFire ? { burner: itemFire } : {}),
    ducts: ventilation ? ductsFor(grid, solid, ventilation) : null,
    clipped,
    fuels: { contentsFuel, ignitionFuel },
  };
}

/**
 * The first item as a design fire: Q = alpha t^2 until its energy (mass
 * times heat of combustion) is spent, then nothing, released by a burner
 * (flamescapes.js createBurner) over the item's area: the open cells of its
 * row within its radius, a layer at a time up to the ceiling over it, the
 * flame filling as many layers as Heskestad's mean flame height reaches. Its
 * products are the item's fuel's, soot included; its radiation is the
 * solver's own from the hot, sooty flame cells (radiant fraction 0). The
 * table has a row each second (each 1/400 of the burn, if longer), so its
 * trapezoids hold the curve's energy to within alpha/6 J a second.
 */
export function itemBurner(grid, solid, cell, { alpha, growth = null, item, fuel }) {
  const { nx, ny, nz, h } = grid;
  const i0 = cell % nx, j0 = Math.floor(cell / nx) % ny, k0 = Math.floor(cell / (nx * ny));
  const radius = Math.sqrt(item.area / Math.PI);
  const reach = Math.ceil(radius / h);
  const columns = [];
  for (let k = k0 - reach; k <= k0 + reach; k += 1) {
    for (let i = i0 - reach; i <= i0 + reach; i += 1) {
      if (i < 0 || k < 0 || i >= nx || k >= nz) continue;
      if ((i === i0 && k === k0) || Math.hypot(i - i0, k - k0) * h <= radius) columns.push([i, k]);
    }
  }
  const layers = [];
  for (let j = j0; j < ny && !solid[i0 + nx * (j + ny * k0)]; j += 1) {
    const layer = [];
    for (const [i, k] of columns) if (!solid[i + nx * (j + ny * k)]) layer.push(i + nx * (j + ny * k));
    layers.push(layer);
  }
  const energy = item.mass * fuel.heatOfCombustion;
  const a = alpha * 1000;
  const tEnd = Math.cbrt((3 * energy) / a);
  const row = Math.max(1, tEnd / 400);
  const hrr = [];
  for (let t = 0; t < tEnd; t += row) hrr.push([t, a * t * t]);
  hrr.push([tEnd, a * tEnd * tEnd], [tEnd + 1e-3, 0]);
  return {
    layers,
    hrr,
    radiantFraction: 0,
    rays: 0,
    fuel,
    diameter: 2 * radius,
    design: { alpha, growth, mass: item.mass, area: item.area, energy, end: tEnd, peak: a * tEnd * tEnd },
  };
}

/**
 * The running ventilation's air terminals on the grid: each terminal at the
 * open cell nearest its centre within two cells (a ceiling grille sits in
 * the ceiling's solid cells), once per cell; terminals outside the grid are
 * left out. One system: the IFC import does not read which system a
 * terminal belongs to, so every terminal shares one duct volume, the case
 * that spreads smoke furthest.
 *
 * @returns {{cells: Int32Array, system: Uint8Array, systems: number, flow: number, outside: number}}
 */
export function ductsFor(grid, solid, { terminals = [], flow = 0.1 } = {}) {
  const cells = [];
  const seen = new Set();
  let outside = 0;
  for (const p of terminals) {
    const [a, b, c] = toLocal(grid, p[0], p[1], p[2]);
    const i0 = Math.floor((a - grid.origin[0]) / grid.h), j0 = Math.floor((b - grid.origin[1]) / grid.h), k0 = Math.floor((c - grid.origin[2]) / grid.h);
    let best = -1, bestD = Infinity;
    for (let dk = -2; dk <= 2; dk += 1) {
      for (let dj = -2; dj <= 2; dj += 1) {
        for (let di = -2; di <= 2; di += 1) {
          const i = i0 + di, j = j0 + dj, k = k0 + dk;
          if (i < 0 || j < 0 || k < 0 || i >= grid.nx || j >= grid.ny || k >= grid.nz) continue;
          const cell = i + grid.nx * (j + grid.ny * k);
          if (solid[cell]) continue;
          const d = di * di + dj * dj + dk * dk;
          if (d < bestD) [best, bestD] = [cell, d];
        }
      }
    }
    if (best < 0) outside += 1;
    else if (!seen.has(best)) {
      seen.add(best);
      cells.push(best);
    }
  }
  return { cells: Int32Array.from(cells), system: new Uint8Array(cells.length), systems: 1, flow, outside };
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

/**
 * Read, at the nodes' head cells: gas temperature (C), visibility (m), CO
 * (ppm), soot (mg/m^3), CO2 and O2 (% by volume).
 */
export function readNodes(sim, cells, out = null) {
  const n = cells.length;
  const T = out?.T || new Float32Array(n);
  const vis = out?.vis || new Float32Array(n);
  const co = out?.co || new Float32Array(n);
  const soot = out?.soot || new Float32Array(n);
  const co2 = out?.co2 || new Float32Array(n);
  const o2 = out?.o2 || new Float32Array(n);
  const ambient = sim.P.ambient - KELVIN;
  for (let v = 0; v < n; v += 1) {
    const c = cells[v];
    if (c < 0) {
      T[v] = ambient;
      vis[v] = MAX_VISIBILITY;
      co[v] = 0;
      soot[v] = 0;
      co2[v] = 0;
      o2[v] = AIR_O2_PERCENT;
      continue;
    }
    T[v] = sim.T[c] - KELVIN;
    const rhoSoot = sim.rho[c] * sim.Y[c * NS + SPECIES.SOOT];
    vis[v] = visibility(rhoSoot);
    soot[v] = rhoSoot * 1e6;
    let moles = 0;
    let rest = 1;
    for (let q = 0; q < NS; q += 1) {
      moles += sim.Y[c * NS + q] / MOLAR[q];
      rest -= sim.Y[c * NS + q];
    }
    moles += Math.max(0, rest) / 0.028;
    const share = (q) => (moles > 0 ? sim.Y[c * NS + q] / MOLAR[q] / moles : 0);
    co[v] = share(SPECIES.CO) * 1e6;
    co2[v] = share(SPECIES.CO2) * 100;
    o2[v] = share(SPECIES.O2) * 100;
  }
  return { T, vis, co, soot, co2, o2 };
}

/** O2 in air, % by volume, as the solver's air (Y_O2 0.232, the rest N2) reads. */
export const AIR_O2_PERCENT = (0.232 / 0.032 / (0.232 / 0.032 + 0.768 / 0.028)) * 100;

/** How far from the fire C/VM2 puts the first smoke detector, m (Table 3.2.2.2). */
export const DETECTOR_RADIUS = 7.0;

/**
 * Virtual smoke detectors under the ceiling round the fire, where NZ C/VM2
 * (Table 3.2.2.2) puts a design's first detector: 7.0 m from the fire in
 * plan, or nearer when the room is smaller (its note 1). Along `rays`
 * directions a ray walks out from the ignition cell through the cells 1.5 m
 * above the fire's floor, and stops at the radius, before a solid cell (a
 * wall) or where the ceiling steps by more than a cell (a door's lintel or a
 * downstand: the edge of the fire's ceiling space). The detector is the top
 * open cell of that column, its centre half a cell below the ceiling (C/VM2
 * asks at least 25 mm). Only the rays that get as far as the farthest one,
 * within a cell, keep a detector: a wall beside the fire makes the room no
 * smaller, so a ray it stops at once does not put a detector over the
 * plume, which would time detection too early.
 *
 * @returns {{cells: Int32Array, distance: Float32Array}} grid cells and their distances from the fire (m)
 */
export function ceilingDetectors(scene, { radius = DETECTOR_RADIUS, rays = 16 } = {}) {
  const none = { cells: new Int32Array(0), distance: new Float32Array(0) };
  if (!scene.ignition) return none;
  const { grid: g, solid } = scene;
  const at = (i, j, k) => i + g.nx * (j + g.ny * k);
  const c0 = scene.ignition.cell;
  const i0 = c0 % g.nx, j0 = Math.floor(c0 / g.nx) % g.ny, k0 = Math.floor(c0 / (g.nx * g.ny));
  const head = Math.min(g.ny - 1, j0 + Math.max(1, Math.round(1.5 / g.h)));
  // The top open cell's height in a column, climbing from head height; -1 where a wall stands at head height.
  const top = (i, k) => {
    if (solid[at(i, head, k)]) return -1;
    let j = head;
    while (j + 1 < g.ny && !solid[at(i, j + 1, k)]) j += 1;
    return j;
  };
  const ceiling = top(i0, k0);
  if (ceiling < 0) return none;
  const stride = 0.5 * g.h;
  const steps = Math.ceil(radius / stride);
  const ends = [];
  for (let r = 0; r < rays; r += 1) {
    const angle = (2 * Math.PI * r) / rays;
    const di = Math.cos(angle), dk = Math.sin(angle);
    let found = -1, far = 0;
    for (let s = 0; s <= steps; s += 1) {
      const d = Math.min(radius, s * stride);
      const i = Math.floor(i0 + 0.5 + (di * d) / g.h), k = Math.floor(k0 + 0.5 + (dk * d) / g.h);
      if (i < 0 || k < 0 || i >= g.nx || k >= g.nz) break;
      const j = top(i, k);
      if (j < 0 || Math.abs(j - ceiling) > 1) break;
      found = at(i, j, k);
      far = d;
    }
    if (found >= 0) ends.push([found, far]);
  }
  const farthest = ends.reduce((m, [, d]) => Math.max(m, d), 0);
  const seen = new Set();
  const cells = [], distance = [];
  for (const [cell, d] of ends) {
    if (d < farthest - g.h - 1e-9 || seen.has(cell)) continue;
    seen.add(cell);
    cells.push(cell);
    distance.push(d);
  }
  return { cells: Int32Array.from(cells), distance: Float32Array.from(distance) };
}

/** Node cells followed by detector cells, to be read in one go. */
export function withDetectors(cells, detectors) {
  if (!detectors.length) return cells;
  const all = new Int32Array(cells.length + detectors.length);
  all.set(cells);
  all.set(detectors, cells.length);
  return all;
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
  return withBurnerFlame(sim, { positions, temps, kinds });
}

/**
 * The design-fire item's flame, to draw. The solver's flame-front particles
 * come from cells where its own combustion took place (particleStep), while
 * the item's burner puts its heat and products into its cells directly, so
 * without these the item showed no flame until the room's contents caught.
 *
 * Shaped as a flame. Heskestad's mean flame height L (flameHeight, for the
 * last step's heat release over the item's diameter) is where the flame is
 * seen half the time, so a point at height z over the item's base is kept
 * with a probability falling from 1 at L / 2 to 0 at 3L / 2, one half at L;
 * and the flame narrows toward its tip, the item's footprint drawn in toward
 * its centre by 1 - z / (3L / 2), never below a fifth, and no higher than
 * the ceiling over it (its burner's layers). Coloured as the solver colours
 * its own flame: each point at its cell's gas temperature, as particleStep
 * spawns flame-front particles at their cell's, so the item burns in the
 * room's colours. The first version filled every heated cell with six points
 * at McCaffrey's centreline temperature: a straight column, too hot for the
 * colour scale, that the additive sprites summed to a pinkish white.
 *
 * About two points to a cell of the mean flame, placed afresh each frame
 * from a seed of the time (the same run draws the same flame) and never in a
 * solid cell. Drawn only: the solver never sees them, and they take nothing
 * from its random numbers. Appended to a particle snapshot in the viewer
 * frame, as flame-front particles (kind 0); the GPU's snapshot too, with
 * `temperatureOf(cell)` giving its burner cells' read-back temperatures (K).
 */
export function withBurnerFlame(sim, snapshot, temperatureOf = null) {
  const b = sim.burner;
  if (!(b?.design && b.flameCells)) return snapshot;
  const tempAt = temperatureOf || ((c) => sim.T[c]);
  const g = sim.grid;
  const { nx, ny, nz } = g;
  const base = b.cells.subarray(b.layerStart[0], b.layerStart[1]);
  // The base's centre and floor, in cells.
  let ci = 0, ck = 0;
  for (const c of base) {
    ci += (c % nx) + 0.5;
    ck += Math.floor(c / (nx * ny)) + 0.5;
  }
  ci /= base.length;
  ck /= base.length;
  const floor = Math.floor(base[0] / nx) % ny;
  const L = Math.max(flameHeight(b.flameRate, b.diameter), 0.5 * g.h) / g.h;
  const top = Math.min(1.5 * L, b.layerStart.length - 1);
  const want = Math.max(base.length, Math.round(6 * base.length * L));
  const random = rng(Math.round(sim.time * 1000) + 1);
  const count = snapshot.temps.length;
  const positions = new Float32Array((count + want) * 3);
  const temps = new Float32Array(count + want);
  const kinds = new Uint8Array(count + want);
  positions.set(snapshot.positions);
  temps.set(snapshot.temps);
  kinds.set(snapshot.kinds);
  const x = [0, 0, 0];
  let o = count;
  for (let tries = 0; o < count + want && tries < 8 * want; tries += 1) {
    const z = random() * top;
    const shrink = Math.max(0.2, 1 - z / (1.5 * L));
    // Kept as often as the flame is seen there, and as the narrowing footprint's area.
    if (random() > Math.min(1, (1.5 * L - z) / L) * shrink * shrink) continue;
    const c = base[Math.floor(random() * base.length)];
    x[0] = ci + ((c % nx) + random() - ci) * shrink;
    x[1] = floor + z;
    x[2] = ck + (Math.floor(c / (nx * ny)) + random() - ck) * shrink;
    const i = Math.floor(x[0]), j = Math.floor(x[1]), k = Math.floor(x[2]);
    if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) continue;
    const cell = i + nx * (j + ny * k);
    const T = sim.kind[cell] === SOLID_KIND ? undefined : tempAt(cell);
    if (!(T > 0)) continue;
    for (let a = 0; a < 3; a += 1) positions[o * 3 + g.axes[a]] = g.origin[a] + x[a] * g.h;
    temps[o] = T;
    o += 1;
  }
  return { positions: positions.subarray(0, o * 3), temps: temps.subarray(0, o), kinds: kinds.subarray(0, o) };
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
 * The sprinklers of a run (suppression.js), or null when it has none: the
 * heads on the grid, their links, and the suppression state the solver
 * reads (sim.suppression for the CPU; the GPU copies it). Nothing of this
 * exists, and no step does any of its work, when the run has no sprinklers.
 *
 * @param {object} sprinklers  { heads (sprinklers.js records), mode: "activation" | "evans" | "spray", openOn: "estimate" | "late" }
 */
export function sprinklerRun(scene, sim, sprinklers, upAxis = "y") {
  if (!sprinklers?.heads?.length) return null;
  // A scene that burns only a prescribed burner (a validation run) has no
  // first item: the plume the estimate reads rises from the burner's axis.
  const plumeScene = scene.ignition || !sim.burner ? scene : { ...scene, ignition: { cell: sim.burner.axis } };
  const setup = sprinklerSetup(plumeScene, sprinklers.heads, { upAxis, fuel: sim.fuelMass });
  // The heads over a design-fire item: those with a column of their footprint over its base.
  let itemHeads = null;
  if (sim.burner?.design) {
    const g = scene.grid;
    const key = (c) => (c % g.nx) + g.nx * Math.floor(c / (g.nx * g.ny));
    const base = new Set(Array.from(sim.burner.cells.subarray(0, sim.burner.layerStart[1]), key));
    itemHeads = setup.heads.flatMap((hd, h) => {
      for (let q = 0; q < hd.columns.length; q += 2) if (base.has(key(hd.columns[q]))) return [h];
      return [];
    });
  }
  return {
    setup,
    state: createActivation(setup, { ambient: sim.P.ambient }),
    sup: createSuppression(sim.N, { mode: sprinklers.mode || "spray" }),
    sensors: Int32Array.from(setup.heads.map((h) => h.sensor)),
    reading: sprinklers.openOn === "late" ? "late" : "estimate",
    itemHeads,
  };
}

/**
 * Water or clean agent on a design-fire item (scene.burner), at a frame's
 * end t, for both solvers. Once a head over the item has opened, in a mode
 * with water, the item's heat release decays from its value then as Evans
 * (1993) measured for the open heads' density over it (the item is
 * prescribed, so it cannot be wetted; Evans's decay stands in for the water
 * on it, in every water mode). In a foam room the foam-water does the same
 * from its discharge; in a clean agent room the item stops when the agent
 * reaches the class A extinguishing concentration. Like the solver's caps on
 * the fuel, the water takes effect from the end of the frame a head opened in.
 */
export function itemSuppression(sim, spr, sys, t) {
  const b = sim.burner;
  if (!b?.design) return;
  if (spr && spr.sup.mode !== "activation" && spr.itemHeads?.length) {
    const open = new Set(spr.sup.open);
    let w = 0;
    for (const h of spr.itemHeads) {
      if (!open.has(h)) continue;
      const hd = spr.setup.heads[h];
      w += (hd.flowLpm || 0) / 60 / Math.max(hd.coverage || 0, 1e-9);
    }
    if (w >= EVANS_MIN_DENSITY) {
      const tau = evansTau(w);
      if (!b.suppression) b.suppression = { start: t, tau, q: burnerRate(b, t), by: "water" };
      else b.suppression.tau = Math.min(b.suppression.tau, tau);
    }
  }
  if (sys) {
    b.room ??= sys.room[b.axis] || 0;
    const rec = b.room ? sys.rooms[b.room - 1] : null;
    if (rec && Number.isFinite(rec.t0)) {
      if (rec.system === "clean-agent") {
        if (rec.design >= rec.extA) b.stopAt = rec.t0 + (rec.discharge * rec.extA) / rec.design;
      } else if (!b.suppression && t >= rec.t0) {
        b.suppression = { start: t, tau: evansTau(rec.rate / 60), q: burnerRate(b, t), by: "foam" };
      }
    }
  }
}

/** What a design-fire item did in a run, for its card, or null without one. */
export function itemResult(sim) {
  const b = sim.burner;
  if (!b?.design) return null;
  return { ...b.design, released: b.released, suppressed: b.suppression?.by ?? null, suppressedAt: b.suppression?.start ?? null, stoppedAt: Number.isFinite(b.stopAt) ? b.stopAt : null };
}

/**
 * One frame of the sprinklers: their links step on the gas at their sensor
 * cells (K) and its speed (m/s), and the heads that open are opened in the
 * suppression state. Returns what the frame reports.
 */
export function sprinklerFrame(run, { t0, t1, T, u, hrr }) {
  const opened = activationFrame(run.setup, run.state, { t0, t1, T, u, hrr, reading: run.reading });
  const changed = opened.length ? openHeads(run.sup, run.setup, opened) : null;
  return { opened, changed, open: run.sup.open.slice(), water: run.sup.columns.reduce((a, c) => a + c.flux, 0) };
}

/**
 * The rooms of a run flooded with clean agent or blanketed with foam
 * (suppressionSystems.js), or null when it has none: each room's cells,
 * detectors and parameters, and the arrays the solver reads (sim.systems
 * for the CPU; the GPU copies them). Nothing of this exists, and no step
 * does any of its work, when the run has no such rooms.
 *
 * @param {object} systems  { rooms: [{id, name, system: "clean-agent" | "foam", footprint, floor, ceiling (or cells), params}] }
 */
export function systemsRun(scene, sim, systems, upAxis = "y") {
  if (!systems?.rooms?.length) return null;
  return systemsSetup(scene, systems, { upAxis, fuelMass: sim.fuelMass });
}

/** One frame of the systems' detection: the visibility at their detectors (m) at the frame's end. */
export function systemsFrame(sys, { t1, vis }) {
  const detected = systemsDetect(sys, { t1, vis });
  return { detected, discharged: sys.rooms.filter((r) => Number.isFinite(r.t0) && r.t0 <= t1).map((r) => r.id) };
}

/** What a run's systems did, for its card. */
export function systemsResult(sys, opts) {
  return { rooms: systemsSummary(sys, opts) };
}

/** What a run's sprinklers came to, for its card. */
export function sprinklerResult(run) {
  return {
    mode: run.sup.mode,
    reading: run.reading,
    heads: activationSummary(run.setup, run.state),
    open: run.sup.open.slice(),
    covered: run.sup.tau.reduce((a, t) => a + (t > 0 ? 1 : 0), 0),
    columns: run.sup.columns.length,
    stats: { ...run.sup.stats },
    plumeHeight: run.setup.plume?.H ?? null,
  };
}

/**
 * Run the solver to `horizon` seconds, calling onFrame every `frameSeconds`
 * with the node readings, the frame's mean heat release rate (kW), the
 * particles and, with `smoke`, the smoke cloud (smokeCloud), and the
 * visibility at the virtual ceiling detectors (ceilingDetectors) as
 * `detectors`. Returns the frames stacked: T, vis, co, soot, co2, o2 as
 * Float32Array(K n), the detectors' visibility as Float32Array(K D), and
 * `detectorDistance`. With `sensorGas` (validation runs), each frame's
 * `sprinklers` also carries the gas at the heads' sensor cells, `gas`
 * {T (K), u (m/s)}. With `systems` (systemsRun), each frame's `systems`
 * says which rooms were detected in it and which have discharged, and the
 * result's `systems.rooms` when each was detected, discharged and what it did.
 */
export function runScene(scene, { nodes, upAxis = "y", agentHeight = 0.75, headHeight = 1.8, horizon = 300, frameSeconds = 5, embers = false, dt = DEFAULTS.dt, onFrame = null, shouldStop = null, particles = 3000, smoke = false, solverOptions = {}, profileInto = null, keep = true, sprinklers = null, sensorGas = false, systems = null } = {}) {
  const sim = createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, embers, dt, seed: 1, ...solverOptions });
  // Per-stage timing for the benchmark (tools/fire-bench.mjs).
  if (profileInto) sim.profile = profileInto;
  const spr = sprinklerRun(scene, sim, sprinklers, upAxis);
  if (spr) sim.suppression = spr.sup;
  const sys = systemsRun(scene, sim, systems, upAxis);
  if (sys) sim.systems = sys;
  const cells = headCells(sim, nodes, { agentHeight, headHeight });
  const n = cells.length;
  const detectors = ceilingDetectors(scene);
  const D = detectors.cells.length;
  const K = Math.max(1, Math.round(horizon / frameSeconds));
  // With keep, every frame is stacked (K x n); without, one frame's buffers
  // are reused and onFrame must copy what it keeps (the worker does).
  const rows = keep ? K : 1;
  const T = new Float32Array(rows * n);
  const vis = new Float32Array(rows * n);
  const co = new Float32Array(rows * n);
  const soot = new Float32Array(rows * n);
  const co2 = new Float32Array(rows * n);
  const o2 = new Float32Array(rows * n);
  const detectorVis = new Float32Array(rows * D);
  const hrr = new Float32Array(K);
  let released = 0;
  for (let k = 0; k < K; k += 1) {
    const end = (k + 1) * frameSeconds;
    while (sim.time < end - 1e-9) {
      step(sim, Math.min(dt, end - sim.time));
      if (shouldStop?.()) return null;
    }
    const row = keep ? k : 0;
    const at = (a) => a.subarray(row * n, (row + 1) * n);
    const frame = readNodes(sim, cells, { T: at(T), vis: at(vis), co: at(co), soot: at(soot), co2: at(co2), o2: at(o2) });
    const atDetectors = detectorVis.subarray(row * D, (row + 1) * D);
    if (D) atDetectors.set(readNodes(sim, detectors.cells).vis);
    hrr[k] = (sim.stats.released - released) / frameSeconds / 1000;
    released = sim.stats.released;
    let sprinkled = null;
    if (spr) {
      const gas = readNodes(sim, spr.sensors).T;
      const Tk = Float64Array.from(gas, (t) => t + KELVIN);
      const u = Float64Array.from(spr.sensors, (c) => (c >= 0 ? cellSpeed(sim, c) : 0));
      sprinkled = sprinklerFrame(spr, { t0: end - frameSeconds, t1: end, T: Tk, u, hrr: hrr[k] });
      if (sensorGas) sprinkled.gas = { T: Tk, u };
    }
    const flooded = sys ? systemsFrame(sys, { t1: end, vis: readNodes(sim, sys.detectorCells).vis }) : null;
    itemSuppression(sim, spr, sys, end);
    onFrame?.({
      k,
      time: end,
      T: frame.T,
      vis: frame.vis,
      co: frame.co,
      soot: frame.soot,
      co2: frame.co2,
      o2: frame.o2,
      detectors: atDetectors,
      sprinklers: sprinkled && { opened: sprinkled.opened, open: sprinkled.open, water: sprinkled.water, ...(sprinkled.gas ? { gas: sprinkled.gas } : {}) },
      ...(flooded ? { systems: flooded } : {}),
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
  const out = { n, K, frameSeconds, horizon: K * frameSeconds, hrr, stats: { ...sim.stats }, inGrid, detectorDistance: detectors.distance, sprinklers: spr ? sprinklerResult(spr) : null };
  // Only when the run has systems, so a run without them keeps its result's shape.
  if (sys) out.systems = systemsResult(sys);
  // Likewise a design-fire first item.
  const item = itemResult(sim);
  if (item) out.designFire = item;
  // A prescribed burner's totals (J), only when the scene has one.
  if (sim.burner) out.burner = { released: sim.burner.released, convective: sim.burner.convective, radiant: sim.burner.radiant, cells: sim.burner.cells.length };
  return keep ? { ...out, T, vis, co, soot, co2, o2, detectorVis } : out;
}
