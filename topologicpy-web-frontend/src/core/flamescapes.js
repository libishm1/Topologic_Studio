/**
 * A JavaScript version of the Flamescapes fire solver (Wrede, Wagner,
 * Palubicki, Michels, Pirk 2026, ACM TOG 45(6) 246,
 * https://doi.org/10.1145/3842518; CC BY 4.0), following its Algorithm 1 on
 * a coarse grid built from the IFC model (src/core/voxelize.js). Equation
 * numbers below are the paper's (transcription:
 * wiki/sources/papers/flamescapes-2026.md); its gas chemistry is that of
 * Fire-X (Wrede et al. 2025, https://doi.org/10.1145/3763338,
 * wiki/sources/papers/firex-2025.md).
 *
 * What is the paper's, per frame (Algorithm 1):
 *   particles   embers spawned from burning solids and advected by the gas
 *               with first-order drag (section 4.8); flame-front particles
 *               spawned six per combusting cell, carrying its temperature at
 *               the flame-front speed (section 4.7)
 *   radiation   rays from hot solid surfaces (Eq. 16) and sooty gas (Eq. 17),
 *               absorbed by soot with probability 1 - exp(-kappa Y_r l) and
 *               deposited in the first solid they meet (supplement Alg. 1)
 *   particle -> grid  embers deposit their excess heat m c (T_e - T_s) in the
 *               solid they reach; the hottest flame-front particle in a cell
 *               supersedes its temperature in gas combustion (supplement Alg. 2)
 *   gas         semi-Lagrangian advection, buoyancy (1 - rho_0 / rho) g with
 *               rho from the equation of state (Eqs. 6, 7), vorticity
 *               confinement, species diffusion (Eq. 11)
 *   solid       evaporation of moisture, oxidation (Eq. 15), oxygen-limited
 *               pyrolysis (Eq. 14), each an Arrhenius rate; gas combustion
 *               (Eqs. 9, 10, 12); conduction in solids (Eq. 18) and exchange
 *               at the solid-gas interface (Eq. 19); pressure projection by
 *               red-black Gauss-Seidel
 *
 * What is changed, and why (Methods page, "Flamescapes in the browser"):
 *   - The grid is 0.25 to 0.5 m, not centimetres, and the step 0.05 to 0.2 s,
 *     not 4 to 8 ms: it has to run on a phone's CPU. Every reaction is
 *     therefore integrated exactly over the step for its rate frozen at the
 *     step's start (m <- m exp(-k dt), never below zero), and every two-body
 *     heat exchange is the exact relaxation of the pair (energy conserved, no
 *     overshoot). The paper's explicit forms are the first-order expansions
 *     of these (formal/fireegress/FireEgress/FlamescapesPort.lean).
 *   - Radiation is attenuated deterministically along each ray (it keeps the
 *     fraction exp(-kappa Y_r l) of its energy per cell, the paper's
 *     absorption probability as an expected value), with a fixed number of
 *     rays per emitting cell instead of a fixed energy per ray.
 *   - Products of combustion follow the fire class of the burning item
 *     (src/core/fireClasses.js): heat of combustion, oxygen consumed at
 *     417 kJ per mol O2 (Fire-X Eq. 4; Huggett 1980), and CO, soot and CO2 yields; water
 *     closes the mass balance. The paper's "residual" species is the soot.
 *   - Room contents are the paper's statistical interior fuel (Eqs. 3 to 5)
 *     on the floor of every covered space; walls, slabs and roofs are inert
 *     (the trimmed IFC import does not read materials). Doors are open and
 *     windows closed unless the run says otherwise: closed doors are inert
 *     leaves in their openings, open windows holes in their walls
 *     (buildScene in flamescapesRun.js).
 */

import { MATERIALS, occupancy, rng } from "./voxelize.js";
import { T_BOIL, TERMINAL_SPEED, VAPOUR_CP, WATER_CP, WATER_TEMPERATURE, evansFactor } from "./suppression.js";
import { STATE } from "./suppressionSystems.js";

export const R_GAS = 8.314462618; // J / (mol K)
export const SIGMA = 5.670374419e-8; // W / (m^2 K^4)
export const P_AMBIENT = 101325; // Pa
export const KELVIN = 273.15;
/**
 * Heat released per kg of oxygen consumed. Fire-X's stoichiometric heat
 * release, Delta_c H = -417 (c + h/4 - o/2) kJ per mol of fuel, is 417 kJ per
 * mol of O2 (its Eq. 4), 13.03 MJ per kg: Huggett's (1980) oxygen-consumption
 * constant, 13.1 MJ per kg, which holds within 5% for most fuels.
 */
export const HEAT_PER_O2 = 417e3 / 0.032; // J per kg O2

/** Gas species, mass fractions; nitrogen is the rest. The paper's "residual" is SOOT. */
export const SPECIES = Object.freeze({ FUEL: 0, O2: 1, CO2: 2, H2O: 3, SOOT: 4, CO: 5 });
export const NS = 6;
/** Molar masses, kg / mol: fuel vapour as a C1 pyrolysate, soot as carbon. */
export const MOLAR = Float64Array.from([0.03, 0.032, 0.044, 0.018, 0.012, 0.028]);
const MOLAR_N2 = 0.028;
/** Specific heats at constant pressure, J / (kg K), near ambient. */
const CP = Float64Array.from([2200, 918, 844, 1864, 710, 1040]);
const CP_N2 = 1040;
/** Air by mass. */
export const Y_O2_AIR = 0.232;

/** Cell kinds. */
export const GAS = 0;
export const POROUS = 1;
export const SOLID = 2;

/**
 * Inert materials (the supplement's Table 3 classes with the paper's Table 4
 * properties): density kg/m^3, specific heat J/(kg K), conductivity
 * W/(m K); emissivities are ours (the paper gives none per material).
 */
export const INERT = [
  { name: "stone", rho: 2700, c: 800, k: 2.5, eps: 0.9 },
  { name: "glass", rho: 2500, c: 840, k: 1.0, eps: 0.9 },
  { name: "metal", rho: 7850, c: 470, k: 52.5, eps: 0.7 },
];

/**
 * Solver parameters. Defaults follow the supplement's wind-driven structure
 * experiment (its Table 6) where that experiment sets them, and the
 * literature named beside each otherwise.
 */
export const DEFAULTS = Object.freeze({
  dt: 0.1, // s
  ambient: 293.15, // K
  gravity: 9.81, // m / s^2
  // Interface coefficient h of Eq. 19, W / (m^2 K): the top of the paper's
  // range (Table 5), raised "to accelerate the interface heat flux"
  // (section 4.9) for sub-grid turbulence and surfaces; with 25, a natural
  // convection value, the first item does not light its neighbours.
  convection: 200,
  // The same interface for inert solids (walls, slabs, ceilings, doors: a
  // solid cell never holds fuel), W / (m^2 K). The raised value above is for
  // the fuel; on walls and ceilings it took most of the ceiling jet's heat and
  // opened sprinklers late (validation R6.1). 10 is FDS's default convection
  // model in these rooms (4 to 14 on walls and ceilings) and Cooper's ceiling
  // correlation 2 m from a plume (7 to 13); with it the estimate opened every
  // measured head of NIST's tests within twice the measured time, the first
  // at medians 0.78 to 1.26 of it (FDS: 0.73 to 1.13)
  // (wiki/research/sprinkler-activation-fix-2026-11-08.md).
  wallConvection: 10,
  sootAbsorption: 500, // the paper's residual absorption coefficient kappa
  vorticity: 10, // confinement strength
  porousVelocity: 0.5, // velocity multiplier in porous cells
  projection: 30, // red-black Gauss-Seidel sweeps at most
  projectionTolerance: 1e-3, // m/s: stop sweeping when a sweep changes the flow less
  diffusivity: 2e-5, // m^2 / s, species at ambient (Eq. 11), scaled by (T/T0)^1.75
  conductivity: 0.026, // W / (m K), gas at ambient
  emitAbove: 500, // K: surfaces and gas cooler than this do not cast rays
  rays: 8, // per emitting cell and step
  rayReach: 40, // cells
  // Solid fuel: the paper's Table 3 (biomass) and Table 4 (wood).
  pyrolysisA: 4.78e4, // 1 / s
  pyrolysisE: 9.22e4, // J / mol
  pyrolysisHeat: 2.5e5, // J / kg, endothermic
  oxidationA: 1.8e8, // 1 / s
  oxidationE: 9.22e4, // J / mol
  pyrolysisFuel: 0.9, // pyrolysate that is fuel vapour; the rest is residual (soot)
  latentHeat: 4.07e4 / 0.01802, // J / kg, water (Table 3, 40.7 kJ / mol)
  moisture: 0.1, // kg water per kg contents
  fuelHeatCapacity: 2000, // J / (kg K), wood (Table 4); a fuel's own `heatCapacity` overrides it
  // Thermally thick fuel on a coarse grid: only a surface layer of this
  // areal density over the fuel's exposed area heats and reacts; it is
  // replenished from the cool bulk as it burns.
  skinDensity: 1.0, // kg / m^2
  contentsArea: 2.0, // exposed fuel area per unit floor area of a contents cell
  itemArea: 1.0, // exposed area of the ignition item, m^2
  // Gas combustion, Eq. 10 (Table 3): methane-like fuel vapour with
  // Westbrook and Dryer's exponents.
  gasA: 8.3e5, // 1 / s with the concentration exponents below
  gasE: 1.26e5, // J / mol
  gasFuelExponent: -0.3,
  gasOxygenExponent: 1.3,
  // Flame front (section 4.7) and embers (section 4.8, supplement Table 6).
  flameSpeed: 1.0, // m / s
  flameLifetime: 0.5, // s
  flameShare: 0.75, // flame-front particles fill at most this share of the buffer, so embers always find room
  // A solid counts as burning (spawns flame-front particles) once its
  // oxidation releases this much per m^2 of exposed fuel. The Arrhenius rate
  // never stops: at 20 C it releases about 0.1 W/m^2, and without a floor every
  // contents cell spawned particles from the first step. 1 kW/m^2 is reached
  // near 110 C with the paper's kinetics and lies far below what a solid
  // releases at incipient piloted ignition, 24 kW/m^2 at 1 g/(m^2 s) (Lyon &
  // Quintiere 2007, doi:10.1016/j.combustflame.2007.07.020), so no flame is lost.
  burningFlux: 1e3, // W / m^2
  embers: false,
  emberRate: 20, // embers per kg of charring solid burnt
  emberPatch: 1e-3, // m^2 of surface layer an ember heats where it lands
  emberMass: [2e-5, 2e-4], // kg
  emberLifetime: 10, // s
  emberMassLoss: 0.05, // per second, fraction of the ember's mass
  emberDrag: 0.5, // grid velocity factor (Table 5: 0.03 to 0.5)
  emberEmissivity: 0.95, // Table 5
  emberHeatCapacity: 2000, // J / (kg K), wood
  maxParticles: 20000,
  seed: 1,
});

/** First-order Arrhenius rate k(T) = A exp(-E / (R T)), 1 / s (Eq. 10's form). */
export function arrhenius(A, E, T) {
  return T > 0 ? A * Math.exp(-E / (R_GAS * T)) : 0;
}

/** Fraction of a first-order reactant left after dt at rate k: exp(-k dt). */
export function survives(k, dt) {
  return Math.exp(-Math.max(0, k) * dt);
}

/**
 * Exact exchange of heat between two bodies of heat capacities Ca, Cb
 * (J/K) through conductance G (W/K) over dt: the energy moved from a to b.
 * dTa/dt = G (Tb - Ta) / Ca and back, solved in closed form.
 */
export function exchange(Ta, Tb, Ca, Cb, G, dt) {
  if (!(Ca > 0) || !(Cb > 0) || !(G > 0)) return 0;
  const mu = (Ca * Cb) / (Ca + Cb);
  return (Ta - Tb) * mu * (1 - Math.exp((-G * dt) / mu));
}

/** Equation of state, Eq. 7: rho = p / (R T sum Y_k / M_k), nitrogen the rest. */
export function gasDensity(T, Y, offset = 0) {
  let inv = 0;
  let rest = 1;
  for (let s = 0; s < NS; s += 1) {
    const y = Y[offset + s];
    inv += y / MOLAR[s];
    rest -= y;
  }
  inv += Math.max(0, rest) / MOLAR_N2;
  return P_AMBIENT / (R_GAS * T * inv);
}

/** Specific heat of the mixture, J / (kg K). */
export function mixtureCp(Y, offset = 0) {
  let cp = 0;
  let rest = 1;
  for (let s = 0; s < NS; s += 1) {
    cp += Y[offset + s] * CP[s];
    rest -= Y[offset + s];
  }
  return cp + Math.max(0, rest) * CP_N2;
}

/**
 * Products of burning 1 kg of fuel of a fire class: oxygen consumed
 * (Huggett) and CO2, CO, soot and water produced, water closing the balance.
 */
export function stoichiometry(fuel) {
  // A metal's oxygen is its own (Mg: 0.658 kg per kg), not Huggett's.
  const o2 = fuel.o2 ?? fuel.heatOfCombustion / HEAT_PER_O2;
  const co2 = fuel.yields.co2;
  const co = fuel.yields.co;
  const soot = fuel.yields.soot;
  const ash = fuel.ash ?? 0; // solid oxide that stays out of the gas
  // Water closes the balance; yields that leave none to close it with would
  // create mass (FlamescapesPort: addProducts conserves mass only then).
  if (co2 + co + soot + ash > 1 + o2 + 1e-9) {
    throw new Error(`A fuel's yields (${(co2 + co + soot + ash).toFixed(3)} kg per kg) exceed the fuel and its oxygen (${(1 + o2).toFixed(3)}).`);
  }
  const h2o = 1 + o2 - co2 - co - soot - ash;
  return { o2, co2, co, soot, h2o, ash, heat: fuel.heatOfCombustion };
}

/**
 * Build the solver state from a voxelised scene.
 *
 * scene: { grid, fraction (cells x MATERIALS), contents: {cells, masses},
 *          ignition: {cell, mass, fuel} }
 * fuels: the contents' fuel and the ignition item's (fireClasses.js shape:
 *        heatOfCombustion, yields {co2, co, soot}, embers, ignitionTemperature)
 */
export function createSolver(scene, { contentsFuel, ignitionFuel = contentsFuel, ...options } = {}) {
  const P = { ...DEFAULTS, ...options };
  const { grid, fraction } = scene;
  const { nx, ny, nz, h } = grid;
  const N = nx * ny * nz;
  const V = h * h * h;
  const kind = new Uint8Array(N);
  const inertC = new Float64Array(N); // J / K of the inert solid in a cell
  const inertK = new Float32Array(N); // W / (m K)
  const eps = new Float32Array(N).fill(0.9);
  const solidMask = scene.solid || occupancy(grid, fraction);
  for (let c = 0; c < N; c += 1) {
    let C = 0, K = 0, E = 0, w = 0;
    for (let m = 0; m < INERT.length; m += 1) {
      const f = fraction[c * MATERIALS + m];
      if (f <= 0) continue;
      C += f * INERT[m].rho * INERT[m].c * V;
      K += f * INERT[m].k;
      E += f * INERT[m].eps;
      w += f;
    }
    // Only a solid cell holds inert mass. Part of a wall in an open cell is
    // below the grid's resolution, and lumping its stone with the fuel's
    // surface layer would make that layer a wall's thermal mass.
    inertC[c] = solidMask[c] ? C : 0;
    inertK[c] = w > 0 ? K / w : 0;
    if (w > 0) eps[c] = E / w;
    if (solidMask[c]) kind[c] = SOLID;
  }
  const fuelMass = new Float32Array(N);
  const fuelKind = new Uint8Array(N); // 0 contents, 1 ignition item
  const moisture = new Float32Array(N); // kg water in the surface layer
  const area = new Float32Array(N); // exposed fuel surface, m^2
  const contents = scene.contents || { cells: [], masses: [] };
  for (let q = 0; q < contents.cells.length; q += 1) {
    const c = contents.cells[q];
    if (kind[c] === SOLID) continue;
    kind[c] = POROUS;
    fuelMass[c] += contents.masses[q];
    area[c] += P.contentsArea * h * h;
  }
  const fuels = [contentsFuel, ignitionFuel];
  const stoich = fuels.map(stoichiometry);
  // A design-fire first item is the scene's burner, not fuel in the grid.
  if (scene.ignition && !scene.ignition.designFire) {
    const { cell, mass } = scene.ignition;
    if (kind[cell] !== SOLID) {
      kind[cell] = POROUS;
      fuelMass[cell] += mass;
      fuelKind[cell] = 1;
      area[cell] += scene.ignition.area ?? P.itemArea;
    }
  }
  // The surface layer starts with its share of the fuel's water (none for a
  // liquid, metal or pack first item: fireClasses.js waterResponse).
  const ownWater = fuels.map((f) => (f?.water?.dry ? 0 : P.moisture));
  for (let c = 0; c < N; c += 1) if (fuelMass[c] > 0) moisture[c] = Math.min(fuelMass[c], P.skinDensity * area[c]) * ownWater[fuelKind[c]];

  const T = new Float32Array(N).fill(P.ambient);
  const Ts = new Float32Array(N).fill(P.ambient);
  const Y = new Float32Array(N * NS);
  for (let c = 0; c < N; c += 1) Y[c * NS + SPECIES.O2] = Y_O2_AIR;
  const rho = new Float32Array(N);
  const rho0 = gasDensity(P.ambient, Y, 0);
  rho.fill(rho0);

  return {
    P,
    grid,
    N,
    V,
    kind,
    inertC,
    inertK,
    eps,
    fuelMass,
    fuelKind,
    moisture,
    area,
    initialFuel: fuelMass.reduce((a, b) => a + b, 0),
    fuels,
    stoich,
    T,
    Ts,
    Y,
    rho,
    rho0,
    u: new Float32Array((nx + 1) * ny * nz),
    v: new Float32Array(nx * (ny + 1) * nz),
    w: new Float32Array(nx * ny * (nz + 1)),
    p: new Float32Array(N),
    flame: new Float32Array(N), // hottest flame-front or ember temperature in a cell this step
    burning: new Uint8Array(N), // combustion took place in the cell this step
    ignitedAt: new Float32Array(N).fill(Infinity),
    // Particles: struct of arrays, kind 0 flame front, 1 ember.
    parts: {
      n: 0,
      kind: new Uint8Array(P.maxParticles),
      x: new Float32Array(P.maxParticles * 3),
      vel: new Float32Array(P.maxParticles * 3),
      T: new Float32Array(P.maxParticles),
      m: new Float32Array(P.maxParticles),
      age: new Float32Array(P.maxParticles),
      life: new Float32Array(P.maxParticles),
    },
    random: rng(P.seed),
    time: 0,
    kindVersion: 0,
    pilot: scene.pilot || null,
    // A prescribed heat release (createBurner); null in every app run.
    burner: scene.burner ? createBurner(scene.burner, kind, grid) : null,
    ducts: scene.ducts?.cells?.length ? scene.ducts : null,
    // Open sprinklers (suppression.js createSuppression), set by the run;
    // declared here so every solver has one shape for the engine.
    suppression: null,
    // Rooms flooded with clean agent or blanketed with foam
    // (suppressionSystems.js systemsSetup), set by the run; null when off.
    systems: null,
    stats: { hrr: 0, burnt: 0, embersSpawned: 0, spotIgnitions: 0, maxT: P.ambient, released: 0, radiated: 0 },
  };
}

// ---------------------------------------------------------------- indexing

const at = (g, i, j, k) => i + g.nx * (j + g.ny * k);
const uAt = (g, i, j, k) => i + (g.nx + 1) * (j + g.ny * k);
const vAt = (g, i, j, k) => i + g.nx * (j + (g.ny + 1) * k);
const wAt = (g, i, j, k) => i + g.nx * (j + g.ny * k);

/** Trilinear sample of a cell-centred gas field that gives solid cells no weight (their values are not gas). */
function sampleGas(sim, F, x, y, z) {
  const g = sim.grid;
  x = Math.min(Math.max(x - 0.5, 0), g.nx - 1.001);
  y = Math.min(Math.max(y - 0.5, 0), g.ny - 1.001);
  z = Math.min(Math.max(z - 0.5, 0), g.nz - 1.001);
  const i = Math.floor(x), j = Math.floor(y), k = Math.floor(z);
  const fx = x - i, fy = y - j, fz = z - k;
  let sum = 0, total = 0;
  for (let q = 0; q < 8; q += 1) {
    const a = q & 1, b = (q >> 1) & 1, d = (q >> 2) & 1;
    const ii = Math.min(i + a, g.nx - 1), jj = Math.min(j + b, g.ny - 1), kk = Math.min(k + d, g.nz - 1);
    const c = ii + g.nx * (jj + g.ny * kk);
    if (sim.kind[c] === SOLID) continue;
    const w = (a ? fx : 1 - fx) * (b ? fy : 1 - fy) * (d ? fz : 1 - fz);
    sum += w * F[c];
    total += w;
  }
  return total > 1e-9 ? sum / total : sim.P.ambient;
}

/** Trilinear sample of a face field with dims (dx, dy, dz) and the face's offset from cell centres. */
function sampleFace(F, dx, dy, dz, x, y, z) {
  x = Math.min(Math.max(x, 0), dx - 1.001);
  y = Math.min(Math.max(y, 0), dy - 1.001);
  z = Math.min(Math.max(z, 0), dz - 1.001);
  const i = Math.floor(x), j = Math.floor(y), k = Math.floor(z);
  const fx = x - i, fy = y - j, fz = z - k;
  const b = i + dx * (j + dy * k);
  const sj = dx, sk = dx * dy;
  const c00 = F[b] * (1 - fx) + F[b + 1] * fx;
  const c10 = F[b + sj] * (1 - fx) + F[b + sj + 1] * fx;
  const c01 = F[b + sk] * (1 - fx) + F[b + sk + 1] * fx;
  const c11 = F[b + sk + sj] * (1 - fx) + F[b + sk + sj + 1] * fx;
  return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz;
}

/**
 * Gas velocity (m/s) at grid coordinates (cell units, origin at the domain
 * corner), from `field` = {u, v, w} (the solver's own by default).
 */
export function velocityAt(sim, x, y, z, out = [0, 0, 0], field = sim) {
  const g = sim.grid;
  out[0] = sampleFace(field.u, g.nx + 1, g.ny, g.nz, x, y - 0.5, z - 0.5);
  out[1] = sampleFace(field.v, g.nx, g.ny + 1, g.nz, x - 0.5, y, z - 0.5);
  out[2] = sampleFace(field.w, g.nx, g.ny, g.nz + 1, x - 0.5, y - 0.5, z);
  return out;
}

// ------------------------------------------------------------- particles


function spawn(sim, kind, x, y, z, vx, vy, vz, T, m, life) {
  const pr = sim.parts;
  if (pr.n >= sim.P.maxParticles) return false;
  if (kind === 0 && pr.n >= sim.P.flameShare * sim.P.maxParticles) return false;
  const q = pr.n;
  pr.kind[q] = kind;
  pr.x[q * 3] = x;
  pr.x[q * 3 + 1] = y;
  pr.x[q * 3 + 2] = z;
  pr.vel[q * 3] = vx;
  pr.vel[q * 3 + 1] = vy;
  pr.vel[q * 3 + 2] = vz;
  pr.T[q] = T;
  pr.m[q] = m;
  pr.age[q] = 0;
  pr.life[q] = life;
  pr.n += 1;
  return true;
}

function removeParticle(pr, q) {
  const last = pr.n - 1;
  if (q !== last) {
    pr.kind[q] = pr.kind[last];
    for (let d = 0; d < 3; d += 1) {
      pr.x[q * 3 + d] = pr.x[last * 3 + d];
      pr.vel[q * 3 + d] = pr.vel[last * 3 + d];
    }
    pr.T[q] = pr.T[last];
    pr.m[q] = pr.m[last];
    pr.age[q] = pr.age[last];
    pr.life[q] = pr.life[last];
  }
  pr.n = last;
}

const DIRS = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
];

/** Algorithm 1 lines 8 to 14: spawn and move embers and flame-front particles. */
function particleStep(sim, dt) {
  const { P, grid: g, parts: pr } = sim;
  const h = g.h;
  const vel = [0, 0, 0];
  // Flame front (section 4.7): six particles from every cell that burned,
  // one per axis direction; a flame particle keeps its direction in `m`.
  for (let c = 0; c < sim.N; c += 1) {
    if (!sim.burning[c]) continue;
    const i = c % g.nx, j = Math.floor(c / g.nx) % g.ny, k = Math.floor(c / (g.nx * g.ny));
    const temp = Math.max(sim.T[c], sim.Ts[c]);
    for (let d = 0; d < 6; d += 1) {
      const [a, b, e] = DIRS[d];
      spawn(sim, 0, i + 0.5, j + 0.5, k + 0.5, a * P.flameSpeed, b * P.flameSpeed, e * P.flameSpeed, temp, d, P.flameLifetime);
    }
  }
  // Embers (section 4.8): from solids burning a fuel that makes them.
  if (P.embers) {
    for (let c = 0; c < sim.N; c += 1) {
      const loss = sim.solidLoss?.[c] || 0;
      if (!(loss > 0) || !sim.fuels[sim.fuelKind[c]].embers) continue;
      // Expected count this step; the remainder is drawn.
      const expected = P.emberRate * loss;
      let count = Math.floor(expected);
      if (sim.random() < expected - count) count += 1;
      if (!count) continue;
      const i = c % g.nx, j = Math.floor(c / g.nx) % g.ny, k = Math.floor(c / (g.nx * g.ny));
      for (let e = 0; e < count; e += 1) {
        const x = i + sim.random(), y = j + 1 + sim.random() * 0.5, z = k + sim.random();
        velocityAt(sim, x, y, z, vel);
        const m = P.emberMass[0] + (P.emberMass[1] - P.emberMass[0]) * sim.random();
        if (spawn(sim, 1, x, y, z, vel[0], vel[1], vel[2], Math.max(sim.Ts[c], 900), m, P.emberLifetime)) sim.stats.embersSpawned += 1;
      }
    }
  }
  // Move (lines 10 to 14). Embers relax toward the drag-scaled gas velocity
  // and cool by radiation and convection while they burn down.
  const relax = 1 - Math.exp(-dt / 0.5);
  for (let q = pr.n - 1; q >= 0; q -= 1) {
    const x = pr.x[q * 3], y = pr.x[q * 3 + 1], z = pr.x[q * 3 + 2];
    velocityAt(sim, x, y, z, vel);
    if (pr.kind[q] === 1) {
      for (let d = 0; d < 3; d += 1) pr.vel[q * 3 + d] += (P.emberDrag * vel[d] - pr.vel[q * 3 + d]) * relax;
      pr.vel[q * 3 + 1] -= 0.5 * dt; // settling of a heavy particle the drag only partly carries
      const m = pr.m[q];
      const r = Math.cbrt((3 * m) / (4 * Math.PI * 300));
      const A = 4 * Math.PI * r * r;
      const Tg = sampleGas(sim, sim.T, x, y, z);
      const loss = (P.emberEmissivity * SIGMA * (pr.T[q] ** 4 - P.ambient ** 4) + P.convection * (pr.T[q] - Tg)) * A * dt;
      pr.T[q] = Math.max(Tg, pr.T[q] - loss / (m * P.emberHeatCapacity));
      pr.m[q] = m * Math.exp(-P.emberMassLoss * dt);
    } else {
      // Flame-front particles: the fuel's flame speed along their own
      // direction, plus the gas velocity (section 4.7).
      const dirn = DIRS[pr.m[q]];
      for (let d = 0; d < 3; d += 1) pr.vel[q * 3 + d] = dirn[d] * P.flameSpeed + vel[d];
    }
    for (let d = 0; d < 3; d += 1) pr.x[q * 3 + d] += (pr.vel[q * 3 + d] * dt) / h;
    pr.age[q] += dt;
    const nxp = pr.x[q * 3], nyp = pr.x[q * 3 + 1], nzp = pr.x[q * 3 + 2];
    const out = nxp < 0 || nyp < 0 || nzp < 0 || nxp >= g.nx || nyp >= g.ny || nzp >= g.nz;
    if (out || pr.age[q] > pr.life[q]) {
      removeParticle(pr, q);
      continue;
    }
    // Boundary handling: flame-front particles stop at solids; embers
    // deposit there (particleToGrid) and are removed.
    const cell = at(g, Math.floor(nxp), Math.floor(nyp), Math.floor(nzp));
    if (sim.kind[cell] === SOLID && pr.kind[q] === 0) removeParticle(pr, q);
  }
}

/** Supplement Algorithm 2: embers heat the solids they reach; flame-front temperatures for ignition. */
function particleToGrid(sim) {
  const { P, grid: g, parts: pr } = sim;
  sim.flame.fill(0);
  for (let q = pr.n - 1; q >= 0; q -= 1) {
    const cell = at(g, Math.floor(pr.x[q * 3]), Math.floor(pr.x[q * 3 + 1]), Math.floor(pr.x[q * 3 + 2]));
    if (pr.T[q] > sim.flame[cell]) sim.flame[cell] = pr.T[q];
    if (pr.kind[q] !== 1) continue;
    const k = sim.kind[cell];
    if (k === GAS) continue;
    // Excess thermal energy m c (T_e - T_s), and the ember's own fuel burnt
    // in place (section 4.8: "a small pyrolysis and combustion").
    const C = solidCapacity(sim, cell);
    if (C > 0) {
      const excess = pr.m[q] * P.emberHeatCapacity * Math.max(0, pr.T[q] - sim.Ts[cell]);
      const own = pr.m[q] * sim.stoich[0].heat;
      sim.Ts[cell] += (excess + own) / C;
      // Sub-grid spot ignition. The paper's centimetre voxel is about the
      // size of the ember's contact patch; a coarse cell's surface layer is
      // a hundred times larger and would dilute the ember's heat. So the
      // ember is taken to heat a patch of `emberPatch` m^2 of the surface
      // layer with its excess thermal energy m c (T_e - T_s) (supplement
      // Algorithm 2, line 5): if that brings the patch to the fuel's ignition
      // temperature, the cell's surface ignites there (it is set to the
      // ignition temperature) and the solver burns it from then on. The
      // ember's own burning heats the whole layer over time (above), not the
      // patch at once; counting it in the patch lit any fuel an ember
      // touched (291 spot fires in a 30 m2 room), whereas firebrands mostly
      // ignite fuel that is already hot (manzello2020).
      const ign = ignitionK(sim, cell);
      if (sim.fuelMass[cell] > 0 && sim.Ts[cell] < ign) {
        const patch = P.skinDensity * P.emberPatch * fuelCp(sim, cell);
        if (sim.Ts[cell] + excess / patch >= ign) {
          sim.Ts[cell] = ign;
          sim.stats.spotIgnitions += 1;
        }
      }
    }
    removeParticle(pr, q);
  }
}

/**
 * The most a liquid pool of exposed area `area` gives off in `dt`:
 * m''_inf (1 - exp(-k beta D)) A dt, D the diameter of a round pool of that
 * area (Babrauskas 1983, via the SFPE Handbook table in NUREG-1805 Suppl. 1).
 * Without k beta the rate does not depend on the diameter; without m''_inf
 * there is no cap.
 */
export function poolCap(liquid, area, dt) {
  if (!(liquid?.mInf > 0)) return Infinity;
  const D = Math.sqrt((4 * area) / Math.PI);
  const size = liquid.kBeta > 0 ? 1 - Math.exp(-liquid.kBeta * D) : 1;
  return liquid.mInf * size * area * dt;
}

function ignitionK(sim, c) {
  return sim.fuels[sim.fuelKind[c]].ignitionTemperature + KELVIN;
}

/**
 * Specific heat of a cell's fuel, J/(kg K): the fuel's own (a metal's is
 * about half wood's), else wood's. With magnesium at wood's value, burning it
 * could lower the solid's heat capacity even with its ash kept
 * (FlamescapesPort.chemistry_ash_capacity_counterexample).
 */
function fuelCp(sim, c) {
  return sim.fuels[sim.fuelKind[c]].heatCapacity ?? sim.P.fuelHeatCapacity;
}

/** Fuel in a cell's reacting surface layer (kg). */
export function skinMass(sim, c) {
  return Math.min(sim.fuelMass[c], sim.P.skinDensity * sim.area[c]);
}

/** Heat capacity of a cell's solid: inert part, the fuel's surface layer and its water (J / K). */
function solidCapacity(sim, c) {
  return sim.inertC[c] + skinMass(sim, c) * fuelCp(sim, c) + sim.moisture[c] * 4186;
}

// -------------------------------------------------------------- radiation

/**
 * Supplement Algorithm 1 with Eqs. 16 and 17: hot solid surfaces and sooty
 * gas emit; each emitting cell casts `rays` rays carrying equal shares, which
 * lose the fraction 1 - exp(-kappa Y_soot l) in every gas cell they cross and
 * leave the rest in the first solid or porous cell they meet.
 */
function radiation(sim, dt) {
  const { P, grid: g } = sim;
  const h = g.h;
  const A = h * h;
  const Ta4 = P.ambient ** 4;
  const dir = [0, 0, 0];
  for (let c = 0; c < sim.N; c += 1) {
    const kc = sim.kind[c];
    let emitted = 0;
    let fromSolid = false;
    if (kc !== GAS && sim.Ts[c] > P.emitAbove) {
      // Eq. 16 over the cell's faces open to gas.
      const i = c % g.nx, j = Math.floor(c / g.nx) % g.ny, k = Math.floor(c / (g.nx * g.ny));
      let open = 0;
      for (const [a, b, d] of DIRS) {
        const ii = i + a, jj = j + b, kk = k + d;
        if (ii < 0 || jj < 0 || kk < 0 || ii >= g.nx || jj >= g.ny || kk >= g.nz || sim.kind[at(g, ii, jj, kk)] === GAS) open += 1;
      }
      if (!open) continue;
      const surface = kc === POROUS ? Math.max(sim.area[c], A) : A * open;
      emitted = sim.eps[c] * SIGMA * (sim.Ts[c] ** 4 - Ta4) * surface * dt;
      // Never emit more than half the heat above ambient in one step.
      emitted = Math.min(emitted, 0.5 * solidCapacity(sim, c) * (sim.Ts[c] - P.ambient));
      if (emitted <= 0) continue;
      sim.Ts[c] -= emitted / solidCapacity(sim, c);
      fromSolid = true;
    } else if (kc === GAS && sim.T[c] > P.emitAbove) {
      // Eq. 17: 4 kappa Y_soot sigma (T^4 - T_amb^4) per unit volume.
      const ys = sim.Y[c * NS + SPECIES.SOOT];
      if (!(ys > 0)) continue;
      const Cg = sim.rho[c] * mixtureCp(sim.Y, c * NS) * sim.V;
      emitted = 4 * P.sootAbsorption * ys * SIGMA * (sim.T[c] ** 4 - Ta4) * sim.V * dt;
      emitted = Math.min(emitted, 0.5 * Cg * (sim.T[c] - P.ambient));
      if (emitted <= 0) continue;
      sim.T[c] -= emitted / Cg;
    } else continue;
    sim.stats.radiated += emitted;
    const i = c % g.nx, j = Math.floor(c / g.nx) % g.ny, k = Math.floor(c / (g.nx * g.ny));
    const share = emitted / P.rays;
    const faces = fromSolid ? openFaces(sim, i, j, k) : null;
    for (let r = 0; r < P.rays; r += 1) {
      // A uniform direction on the sphere; from a surface, added to the
      // normal of one of its open faces, which gives Lambert's cosine law
      // over that face's hemisphere.
      const zc = 2 * sim.random() - 1;
      const phi = 2 * Math.PI * sim.random();
      const s = Math.sqrt(1 - zc * zc);
      dir[0] = s * Math.cos(phi);
      dir[1] = zc;
      dir[2] = s * Math.sin(phi);
      if (faces) {
        const n = DIRS[faces[Math.floor(sim.random() * faces.length)]];
        for (let d = 0; d < 3; d += 1) dir[d] += n[d];
        const len = Math.hypot(dir[0], dir[1], dir[2]) || 1;
        for (let d = 0; d < 3; d += 1) dir[d] /= len;
      }
      march(sim, i + 0.5, j + 0.5, k + 0.5, dir, share, c, fromSolid);
    }
  }
}

/** The directions (DIRS indices) in which a cell has an open neighbour or the open domain side. */
function openFaces(sim, i, j, k) {
  const g = sim.grid;
  const out = [];
  for (let d = 0; d < 6; d += 1) {
    const [a, b, e] = DIRS[d];
    const ii = i + a, jj = j + b, kk = k + e;
    if (ii < 0 || jj < 0 || kk < 0 || ii >= g.nx || jj >= g.ny || kk >= g.nz || sim.kind[at(g, ii, jj, kk)] === GAS) out.push(d);
  }
  return out;
}

/** March one ray cell by cell (Amanatides-Woo), depositing its energy. */
function march(sim, x, y, z, dir, energy, from, skipOwn) {
  const { P, grid: g } = sim;
  const h = g.h;
  let i = Math.floor(x), j = Math.floor(y), k = Math.floor(z);
  const step = dir.map((d) => (d > 0 ? 1 : -1));
  const tDelta = dir.map((d) => (d !== 0 ? Math.abs(1 / d) : Infinity));
  const pos = [x, y, z];
  const cell0 = [i, j, k];
  const tMax = dir.map((d, a) => (d !== 0 ? (d > 0 ? cell0[a] + 1 - pos[a] : pos[a] - cell0[a]) / Math.abs(d) : Infinity));
  let left = energy;
  let t = 0;
  for (let n = 0; n < P.rayReach && left > 1e-9 * energy; n += 1) {
    const a = tMax[0] < tMax[1] ? (tMax[0] < tMax[2] ? 0 : 2) : tMax[1] < tMax[2] ? 1 : 2;
    const length = (tMax[a] - t) * h;
    t = tMax[a];
    const c = at(g, i, j, k);
    if (sim.kind[c] === GAS && c !== from) {
      // Absorbed by soot along the segment (Section 4.6).
      const tau = P.sootAbsorption * sim.Y[c * NS + SPECIES.SOOT] * length;
      if (tau > 0) {
        const taken = left * (1 - Math.exp(-tau));
        sim.T[c] += taken / (sim.rho[c] * mixtureCp(sim.Y, c * NS) * sim.V);
        left -= taken;
      }
    }
    tMax[a] += tDelta[a];
    if (a === 0) i += step[0];
    else if (a === 1) j += step[1];
    else k += step[2];
    if (i < 0 || j < 0 || k < 0 || i >= g.nx || j >= g.ny || k >= g.nz) return; // leaves the domain
    const next = at(g, i, j, k);
    if (sim.kind[next] !== GAS && !(skipOwn && next === from)) {
      sim.Ts[next] += left / Math.max(solidCapacity(sim, next), 1e-6);
      return;
    }
  }
  // Out of reach (or below the cut-off): what the ray still carries is
  // absorbed in the gas cell it ended in, so no energy is lost.
  if (left > 0 && i >= 0 && j >= 0 && k >= 0 && i < g.nx && j < g.ny && k < g.nz) {
    const c = at(g, i, j, k);
    if (sim.kind[c] === GAS) sim.T[c] += left / (sim.rho[c] * mixtureCp(sim.Y, c * NS) * sim.V);
    else sim.Ts[c] += left / Math.max(solidCapacity(sim, c), 1e-6);
  }
}

// ------------------------------------------------------------- gas phase

/** Semi-Lagrangian advection (Algorithm 1 line 20) of velocity, temperature and species. */
function advect(sim, dt) {
  const { grid: g } = sim;
  const { nx, ny, nz, h } = g;
  const s = dt / h;
  const vel = [0, 0, 0];
  const pos = [0, 0, 0];
  // The back-trace reads the velocity of the step's start, not the faces
  // being overwritten.
  const u = copyOf(sim, "u0", sim.u), v = copyOf(sim, "v0", sim.v), w = copyOf(sim, "w0", sim.w);
  const old = { u, v, w };
  // Midpoint (RK2) back-trace in cell units, into `pos` (no allocation).
  const back = (x, y, z) => {
    velocityAt(sim, x, y, z, vel, old);
    const mx = x - 0.5 * s * vel[0], my = y - 0.5 * s * vel[1], mz = z - 0.5 * s * vel[2];
    velocityAt(sim, mx, my, mz, vel, old);
    pos[0] = x - s * vel[0];
    pos[1] = y - s * vel[1];
    pos[2] = z - s * vel[2];
  };
  const T = copyOf(sim, "T0", sim.T), Y = copyOf(sim, "Y0", sim.Y);
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i <= nx; i += 1) {
        back(i, j + 0.5, k + 0.5);
        sim.u[uAt(g, i, j, k)] = sampleFace(u, nx + 1, ny, nz, pos[0], pos[1] - 0.5, pos[2] - 0.5);
      }
    }
  }
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j <= ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        back(i + 0.5, j, k + 0.5);
        sim.v[vAt(g, i, j, k)] = sampleFace(v, nx, ny + 1, nz, pos[0] - 0.5, pos[1], pos[2] - 0.5);
      }
    }
  }
  for (let k = 0; k <= nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        back(i + 0.5, j + 0.5, k);
        sim.w[wAt(g, i, j, k)] = sampleFace(w, nx, ny, nz + 1, pos[0] - 0.5, pos[1] - 0.5, pos[2]);
      }
    }
  }
  // Temperature and the six species share one back-trace and one set of
  // trilinear weights per cell (as sampleCell computes them).
  const sx = 1, sy = nx, sz = nx * ny;
  const corner = new Int32Array(8);
  const weight = new Float64Array(8);
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const c = at(g, i, j, k);
        if (sim.kind[c] === SOLID) continue;
        back(i + 0.5, j + 0.5, k + 0.5);
        const x = Math.min(Math.max(pos[0] - 0.5, 0), nx - 1.001);
        const y = Math.min(Math.max(pos[1] - 0.5, 0), ny - 1.001);
        const z = Math.min(Math.max(pos[2] - 0.5, 0), nz - 1.001);
        const i0 = Math.floor(x), j0 = Math.floor(y), k0 = Math.floor(z);
        const fx = x - i0, fy = y - j0, fz = z - k0;
        const di = i0 + 1 < nx ? sx : 0, dj = j0 + 1 < ny ? sy : 0, dk = k0 + 1 < nz ? sz : 0;
        const b = i0 + nx * (j0 + ny * k0);
        // Corners inside solids hold no gas (their T and Y are never updated):
        // they get no weight, and the rest are renormalised; a cell whose
        // back-trace lands among solids only keeps its own state (review 1:
        // walls had been an endless source of cool fresh air).
        corner[0] = b; corner[1] = b + di; corner[2] = b + dj; corner[3] = b + dj + di;
        corner[4] = b + dk; corner[5] = b + dk + di; corner[6] = b + dk + dj; corner[7] = b + dk + dj + di;
        weight[0] = (1 - fx) * (1 - fy) * (1 - fz); weight[1] = fx * (1 - fy) * (1 - fz);
        weight[2] = (1 - fx) * fy * (1 - fz); weight[3] = fx * fy * (1 - fz);
        weight[4] = (1 - fx) * (1 - fy) * fz; weight[5] = fx * (1 - fy) * fz;
        weight[6] = (1 - fx) * fy * fz; weight[7] = fx * fy * fz;
        let total = 0;
        for (let q = 0; q < 8; q += 1) {
          if (sim.kind[corner[q]] === SOLID) weight[q] = 0;
          total += weight[q];
        }
        if (!(total > 1e-9)) continue;
        let t = 0;
        for (let q = 0; q < 8; q += 1) t += weight[q] * T[corner[q]];
        sim.T[c] = t / total;
        for (let q = 0; q < NS; q += 1) {
          let acc = 0;
          for (let r = 0; r < 8; r += 1) acc += weight[r] * Y[corner[r] * NS + q];
          sim.Y[c * NS + q] = acc / total;
        }
      }
    }
  }
}

/**
 * Buoyancy (Algorithm 1 line 21): Eq. 6's body force (1 - rho_0 / rho) g on
 * the vertical faces, g pointing down, so gas lighter than the ambient rises
 * with acceleration (rho_0 / rho - 1) |g|.
 */
function buoyancy(sim, dt) {
  const { grid: g, P } = sim;
  for (let k = 0; k < g.nz; k += 1) {
    for (let j = 1; j < g.ny; j += 1) {
      for (let i = 0; i < g.nx; i += 1) {
        const a = at(g, i, j - 1, k), b = at(g, i, j, k);
        if (sim.kind[a] === SOLID || sim.kind[b] === SOLID) continue;
        const r = 0.5 * (sim.rho[a] + sim.rho[b]);
        sim.v[vAt(g, i, j, k)] += dt * P.gravity * (sim.rho0 / r - 1);
      }
    }
  }
}

/** Vorticity confinement (Algorithm 1 line 22; Fedkiw, Stam and Jensen 2001). */
function vorticityConfinement(sim, dt) {
  const { grid: g, P } = sim;
  if (!(P.vorticity > 0)) return;
  const { nx, ny, nz, h } = g;
  const N = sim.N;
  const wx = workArray(sim, "wx", N), wy = workArray(sim, "wy", N), wz = workArray(sim, "wz", N), mag = workArray(sim, "mag", N);
  mag.fill(0);
  // Cell-centred velocity: the mean of each cell's two faces per axis.
  const cu = workArray(sim, "cu", N), cv = workArray(sim, "cv", N), cw = workArray(sim, "cw", N);
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const c = at(g, i, j, k);
        cu[c] = 0.5 * (sim.u[uAt(g, i, j, k)] + sim.u[uAt(g, i + 1, j, k)]);
        cv[c] = 0.5 * (sim.v[vAt(g, i, j, k)] + sim.v[vAt(g, i, j + 1, k)]);
        cw[c] = 0.5 * (sim.w[wAt(g, i, j, k)] + sim.w[wAt(g, i, j, k + 1)]);
      }
    }
  }
  const fields = [cu, cv, cw];
  const vel = (i, j, k, a) => fields[a][at(g, i, j, k)];
  for (let k = 1; k < nz - 1; k += 1) {
    for (let j = 1; j < ny - 1; j += 1) {
      for (let i = 1; i < nx - 1; i += 1) {
        const c = at(g, i, j, k);
        if (sim.kind[c] === SOLID) continue;
        wx[c] = (vel(i, j + 1, k, 2) - vel(i, j - 1, k, 2) - vel(i, j, k + 1, 1) + vel(i, j, k - 1, 1)) / (2 * h);
        wy[c] = (vel(i, j, k + 1, 0) - vel(i, j, k - 1, 0) - vel(i + 1, j, k, 2) + vel(i - 1, j, k, 2)) / (2 * h);
        wz[c] = (vel(i + 1, j, k, 1) - vel(i - 1, j, k, 1) - vel(i, j + 1, k, 0) + vel(i, j - 1, k, 0)) / (2 * h);
        mag[c] = Math.hypot(wx[c], wy[c], wz[c]);
      }
    }
  }
  for (let k = 2; k < nz - 2; k += 1) {
    for (let j = 2; j < ny - 2; j += 1) {
      for (let i = 2; i < nx - 2; i += 1) {
        const c = at(g, i, j, k);
        if (sim.kind[c] === SOLID || !(mag[c] > 0)) continue;
        let Nx = (mag[at(g, i + 1, j, k)] - mag[at(g, i - 1, j, k)]) / (2 * h);
        let Ny = (mag[at(g, i, j + 1, k)] - mag[at(g, i, j - 1, k)]) / (2 * h);
        let Nz = (mag[at(g, i, j, k + 1)] - mag[at(g, i, j, k - 1)]) / (2 * h);
        const len = Math.hypot(Nx, Ny, Nz);
        if (!(len > 1e-9)) continue;
        Nx /= len;
        Ny /= len;
        Nz /= len;
        const f = P.vorticity * h * dt;
        const fx = f * (Ny * wz[c] - Nz * wy[c]);
        const fy = f * (Nz * wx[c] - Nx * wz[c]);
        const fz = f * (Nx * wy[c] - Ny * wx[c]);
        sim.u[uAt(g, i, j, k)] += 0.5 * fx;
        sim.u[uAt(g, i + 1, j, k)] += 0.5 * fx;
        sim.v[vAt(g, i, j, k)] += 0.5 * fy;
        sim.v[vAt(g, i, j + 1, k)] += 0.5 * fy;
        sim.w[wAt(g, i, j, k)] += 0.5 * fz;
        sim.w[wAt(g, i, j, k + 1)] += 0.5 * fz;
      }
    }
  }
}

/**
 * Running ventilation (sim.ducts, from ductsFor): each system's air terminals
 * swap a share of their cell's gas with the system's ducts, taken as one
 * well-mixed volume. A terminal draws f = Q dt / V of its cell's gas, mass
 * m_t = f rho_t V, and gets the same mass of the system's mixture back, so
 * each species and the heat (as m T) are conserved over each system, and
 * smoke drawn in at one terminal comes out at all of them. Not the paper's:
 * Flamescapes and Fire-X model no ducts (DuctMixing.lean proves the sums).
 */
export function ducts(sim, dt) {
  const D = sim.ducts;
  if (!D) return;
  const f = Math.min(1, (D.flow * dt) / sim.V);
  if (!(f > 0)) return;
  const S = D.systems;
  const mass = new Float64Array(S), heat = new Float64Array(S), species = new Float64Array(S * NS);
  for (let q = 0; q < D.cells.length; q += 1) {
    const c = D.cells[q], s = D.system[q];
    const m = f * sim.rho[c] * sim.V;
    mass[s] += m;
    heat[s] += m * sim.T[c];
    for (let i = 0; i < NS; i += 1) species[s * NS + i] += m * sim.Y[c * NS + i];
  }
  for (let q = 0; q < D.cells.length; q += 1) {
    const c = D.cells[q], s = D.system[q];
    if (!(mass[s] > 0)) continue;
    sim.T[c] = (1 - f) * sim.T[c] + (f * heat[s]) / mass[s];
    for (let i = 0; i < NS; i += 1) sim.Y[c * NS + i] = (1 - f) * sim.Y[c * NS + i] + (f * species[s * NS + i]) / mass[s];
  }
}

/** Explicit diffusion of species and heat (Eq. 11 and Eq. 8's conduction) between open cells. */
function diffuse(sim, dt) {
  const { grid: g, P } = sim;
  const h = g.h;
  const Y = copyOf(sim, "Y0", sim.Y);
  const T = copyOf(sim, "T0", sim.T);
  const topo = topology(sim);
  const off = topo.offsets;
  // rho D per open cell; a face takes the mean of its two cells, so the
  // species mass flux rho D (Y_n - Y_c) / h is the same seen from both sides
  // (FlamescapesPort: diffusion conserves each species).
  const rhoD = sim.scratchRhoD && sim.scratchRhoD.length === sim.N ? sim.scratchRhoD : (sim.scratchRhoD = new Float64Array(sim.N));
  for (const list of topo.colours) {
    for (let q = 0; q < list.length; q += 1) {
      const c = list[q];
      rhoD[c] = sim.rho[c] * P.diffusivity * Math.pow(T[c] / P.ambient, 1.75);
    }
  }
  for (const list of topo.colours) {
    for (let q = 0; q < list.length; q += 1) {
      const c = list[q];
      const mask = topo.mask[c];
      if (!mask) continue;
      const alpha = P.conductivity / (sim.rho[c] * mixtureCp(Y, c * NS));
      const lamT = Math.min((alpha * dt) / (h * h), 1 / 7);
      for (let d = 0; d < 6; d += 1) {
        if (!(mask & (1 << d))) continue;
        const n = c + off[d];
        const lamD = Math.min((0.5 * (rhoD[c] + rhoD[n]) * dt) / (sim.rho[c] * h * h), 1 / 7);
        for (let s2 = 0; s2 < NS; s2 += 1) sim.Y[c * NS + s2] += lamD * (Y[n * NS + s2] - Y[c * NS + s2]);
        sim.T[c] += lamT * (T[n] - T[c]);
      }
    }
  }
}

// ---------------------------------------------------------------- chemistry

/**
 * A cell of a flooded room whose agent is at or above the extinguishing
 * concentration of its vapour's fuel (heptane's cup-burner value for a
 * liquid, class A's otherwise; suppressionSystems.js), counted in the room's
 * tally of flames stopped.
 */
function flooded(sim, sys, c) {
  const r = sys.room[c];
  const o = r * STATE.stride;
  const ext = sim.fuels[sim.fuelVapourKind?.[c] ?? 0].liquid ? sys.state[o + STATE.extB] : sys.state[o + STATE.extA];
  if (sys.state[o + STATE.agent] < ext) return false;
  sys.rooms[r - 1].stats.flameSteps += 1;
  return true;
}

/**
 * Lines 24 to 27: evaporation, oxidation of solids (Eq. 15), oxygen-limited
 * pyrolysis (Eq. 14) and gas combustion (Eqs. 9, 10, 12). A solid cell
 * trades species with its open neighbours, a porous cell with its own gas.
 */
function chemistry(sim, dt) {
  const { P, grid: g } = sim;
  const { nx, ny, nz } = g;
  // Open sprinklers cap the fuel under them (suppression.js; null when off).
  const sup = sim.suppression;
  // Rooms flooded with clean agent or blanketed with foam
  // (suppressionSystems.js; null when off): each room's state this step.
  const sys = sim.systems;
  if (sys) sys.update(sim.time);
  sim.burning.fill(0);
  if (!sim.solidLoss) sim.solidLoss = new Float32Array(sim.N);
  sim.solidLoss.fill(0);
  let released = 0;
  let burnt = 0;
  for (let c = 0; c < sim.N; c += 1) {
    const kc = sim.kind[c];
    // ---- solid fuel ----
    if (kc !== GAS && sim.fuelMass[c] > 0) {
      // The gas this solid exchanges with: its own (porous) or the first
      // open neighbour, upward first (where the hot gas goes).
      let gcell = c;
      if (kc === SOLID) {
        gcell = -1;
        const i = c % nx, j = Math.floor(c / nx) % ny, k = Math.floor(c / (nx * ny));
        for (const [a, b, d] of [DIRS[2], DIRS[0], DIRS[1], DIRS[4], DIRS[5], DIRS[3]]) {
          const ii = i + a, jj = j + b, kk = k + d;
          if (ii < 0 || jj < 0 || kk < 0 || ii >= nx || jj >= ny || kk >= nz) continue;
          const n = at(g, ii, jj, kk);
          if (sim.kind[n] !== SOLID) {
            gcell = n;
            break;
          }
        }
        if (gcell < 0) continue;
      }
      const fuel = sim.fuels[sim.fuelKind[c]];
      const st = sim.stoich[sim.fuelKind[c]];
      const C = solidCapacity(sim, c);
      const gasMass = sim.rho[gcell] * sim.V;
      const o2 = sim.Y[gcell * NS + SPECIES.O2] * gasMass;
      // Evaporation: water leaves with the heat above 100 C.
      if (sim.moisture[c] > 0 && sim.Ts[c] > 373.15) {
        const dw = Math.min(sim.moisture[c], (C * (sim.Ts[c] - 373.15)) / P.latentHeat);
        sim.moisture[c] -= dw;
        sim.Ts[c] -= (dw * P.latentHeat) / C;
        addGas(sim, gcell, SPECIES.H2O, dw);
      }
      // A liquid first item (fire classes B and K) evaporates instead: the
      // heat above its boiling point turns its surface into fuel vapour
      // (Fire-X's energy-limited evaporation), which burns in the gas. The
      // pool gives off at most its large-pool burning rate,
      // m'' = m''_inf (1 - exp(-k beta D)) (Babrauskas 1983): the coarse grid
      // cannot resolve the flame's heat feedback to the pool, and without the
      // cap the pilot holding the surface hot boiled the whole pool off in a
      // few steps into one cell, too rich to burn.
      if (fuel.liquid) {
        if (sim.Ts[c] > fuel.liquid.boiling) {
          const skin0 = skinMass(sim, c);
          let dv = Math.min(skin0, (C * (sim.Ts[c] - fuel.liquid.boiling)) / fuel.liquid.latent, poolCap(fuel.liquid, sim.area[c], dt));
          if (sup && sup.tau[c] > 0 && fuel.water?.evans !== false) dv *= evansFactor(sup, c, dv, sim.time, dt);
          // Foam over the pool: the share of its evaporation the blanket leaves.
          if (sys && sys.room[gcell]) {
            const left = sys.state[sys.room[gcell] * STATE.stride + STATE.blanket];
            if (left < 1) {
              sys.rooms[sys.room[gcell] - 1].stats.withheld += dv * (1 - left);
              dv *= left;
            }
          }
          if (dv > 0) {
            sim.fuelMass[c] -= dv;
            sim.Ts[c] -= (dv * fuel.liquid.latent) / C;
            addGas(sim, gcell, SPECIES.FUEL, dv);
            sim.fuelVapourKind ??= new Uint8Array(sim.N);
            sim.fuelVapourKind[gcell] = sim.fuelKind[c];
            sim.solidLoss[c] = dv;
            if (sim.fuelMass[c] <= 1e-7) {
              sim.fuelMass[c] = 0;
              if (kc === POROUS && sim.inertC[c] === 0) {
                sim.kind[c] = GAS;
                sim.kindVersion += 1;
              }
            }
          }
        }
      } else {
      // Oxidation, Eq. 15: an Arrhenius rate in the solid temperature, first
      // order in the surface layer's fuel and in the oxygen beside it.
      const skin = skinMass(sim, c);
      const ko = arrhenius(P.oxidationA, P.oxidationE, sim.Ts[c]) * (sim.Y[gcell * NS + SPECIES.O2] / Y_O2_AIR);
      let dOx = skin * (1 - survives(ko, dt));
      dOx = Math.min(dOx, o2 / st.o2);
      // Clean agent at or above the class A extinguishing concentration in
      // the gas beside the fuel: no oxidation (pyrolysis goes on below).
      if (sys && sys.room[gcell]) {
        const o = sys.room[gcell] * STATE.stride;
        if (sys.state[o + STATE.agent] >= sys.state[o + STATE.extA]) {
          sys.rooms[sys.room[gcell] - 1].stats.oxBlocked += dOx;
          dOx = 0;
        }
      }
      // Pyrolysis, Eq. 14: suppressed as oxygen becomes available.
      const starved = 1 - Math.min(1, sim.Y[gcell * NS + SPECIES.O2] / Y_O2_AIR);
      const kp = arrhenius(P.pyrolysisA, P.pyrolysisE, sim.Ts[c]) * (0.25 + 0.75 * starved);
      let dPy = Math.max(0, skin - dOx) * (1 - survives(kp, dt));
      // Evans's cap under an open sprinkler: both reactions scaled alike
      // (solids only: fireClasses.js waterResponse).
      if (sup && sup.tau[c] > 0 && fuel.water?.evans !== false) {
        const f = evansFactor(sup, c, dOx + dPy, sim.time, dt);
        dOx *= f;
        dPy *= f;
      }
      if (dOx > 0) {
        sim.fuelMass[c] -= dOx;
        // A metal's oxide stays as ash (MgO, about 900 J/(kg K)).
        if (st.ash > 0) sim.inertC[c] += st.ash * dOx * 900;
        const heat = dOx * st.heat;
        sim.Ts[c] += heat / Math.max(solidCapacity(sim, c), 1e-6);
        addProducts(sim, gcell, st, dOx);
        released += heat;
        burnt += dOx;
        if (heat >= P.burningFlux * sim.area[c] * dt) sim.burning[c] = 1;
      }
      if (dPy > 0) {
        sim.fuelMass[c] -= dPy;
        sim.Ts[c] -= (dPy * P.pyrolysisHeat) / Math.max(solidCapacity(sim, c), 1e-6);
        addGas(sim, gcell, SPECIES.FUEL, dPy * P.pyrolysisFuel);
        addGas(sim, gcell, SPECIES.SOOT, dPy * (1 - P.pyrolysisFuel));
        sim.fuelVapourKind ??= new Uint8Array(sim.N);
        sim.fuelVapourKind[gcell] = sim.fuelKind[c];
      }
      sim.solidLoss[c] = dOx + dPy;
      // Fresh fuel from the bulk replaces what reacted, at ambient
      // temperature and with its water: mixing it in costs the layer heat.
      const fresh = Math.min(dOx + dPy, Math.max(0, sim.fuelMass[c] - (skin - dOx - dPy)));
      if (fresh > 0) {
        const water = fuel.water?.dry ? 0 : fresh * P.moisture;
        const Cfresh = fresh * fuelCp(sim, c) + water * 4186;
        sim.moisture[c] += water;
        const Cnow = solidCapacity(sim, c);
        sim.Ts[c] =(sim.Ts[c] * Math.max(0, Cnow - Cfresh) + P.ambient * Cfresh) / Math.max(Cnow, 1e-9);
      }
      if (sim.burning[c] && sim.ignitedAt[c] === Infinity) sim.ignitedAt[c] = sim.time;
      if (sim.fuelMass[c] <= 1e-7) {
        sim.fuelMass[c] = 0;
        if (kc === POROUS && sim.inertC[c] === 0) {
          sim.kind[c] = GAS; // burnt out
          sim.kindVersion += 1;
        }
      }
      }
    }
    // ---- gas combustion, Eqs. 9, 10, 12 ----
    if (kc !== SOLID) {
      const yF = sim.Y[c * NS + SPECIES.FUEL];
      const yO = sim.Y[c * NS + SPECIES.O2];
      // Clean agent at or above the extinguishing concentration of the
      // vapour's fuel: no flame in the cell (tested only where a flame could be).
      if (yF > 1e-7 && yO > 1e-6 && !(sys && sys.room[c] && flooded(sim, sys, c))) {
        const mass = sim.rho[c] * sim.V;
        const kindF = sim.fuelVapourKind?.[c] ?? 0;
        const st = sim.stoich[kindF];
        // Flame-front particles and embers supersede the cell's temperature
        // when hotter (section 4.7, supplement Alg. 2 line 11).
        const Teff = Math.max(sim.T[c], sim.flame[c]);
        const cF = (sim.rho[c] * yF) / MOLAR[SPECIES.FUEL];
        const cO = (sim.rho[c] * yO) / MOLAR[SPECIES.O2];
        const omega = arrhenius(P.gasA, P.gasE, Teff) * cF ** (P.gasFuelExponent - 1) * cO ** P.gasOxygenExponent;
        let dF = yF * mass * (1 - survives(omega, dt));
        dF = Math.min(dF, (yO * mass) / st.o2);
        if (dF > 1e-12) {
          addProducts(sim, c, st, dF, true);
          const heat = dF * st.heat;
          sim.T[c] += heat / (mass * mixtureCp(sim.Y, c * NS));
          released += heat;
          burnt += dF;
          sim.burning[c] = 1;
          if (sim.ignitedAt[c] === Infinity) sim.ignitedAt[c] = sim.time;
        }
      }
    }
  }
  sim.stats.hrr = released / dt;
  sim.stats.released += released;
  sim.stats.burnt += burnt;
}

/**
 * The sprinklers' spray (suppression.js; sprinkler-physics-2026-11-07.md
 * 7.2): down each column under an open head, a share of the water left
 * evaporates in each cell, 1 - exp(-eta (T - T_boil) h), at most what cools
 * the cell to boiling. The steam joins the gas and the gas keeps its energy,
 * the vapour carrying its own heat (Suppression.lean evaporate_energy:
 * C T = (C + c_v dv) T' + dv (e - c_v T_boil)); the falling water drags the
 * gas down, (water / (v_t h^2 rho)) g; what is left reaches the floor.
 * Flame-front particles are left to the solver's own cull (where combustion
 * stops): culling them in every sprayed cell, as the plan first proposed,
 * put the flame out by fiat, halving a test fire's heat while the water
 * absorbed a few hundredths of it.
 */
function spray(sim, dt) {
  const sup = sim.suppression;
  if (!sup || !sup.columns.length) return;
  const g = sim.grid;
  const { nx, ny, h } = g;
  const e = WATER_CP * (T_BOIL - WATER_TEMPERATURE) + sim.P.latentHeat;
  const drag = Math.abs(sim.P.gravity) / (TERMINAL_SPEED * h * h);
  let emitted = 0, evaporated = 0, wetted = 0, floor = 0;
  const wet = sup.wet || 0;
  for (const col of sup.columns) {
    let m = col.flux * dt;
    emitted += m;
    let c = col.top;
    for (let n = 0; n < col.count && m > 0; n += 1, c -= nx) {
      if (sim.kind[c] === SOLID) break;
      const T = sim.T[c];
      if (T > T_BOIL) {
        const Cg = sim.rho[c] * mixtureCp(sim.Y, c * NS) * sim.V;
        const dv = Math.min((1 - Math.exp(-sup.eta * (T - T_BOIL) * h)) * m, (Cg * (T - T_BOIL)) / e);
        if (dv > 0) {
          sim.T[c] = (Cg * T - dv * (e - VAPOUR_CP * T_BOIL)) / (Cg + VAPOUR_CP * dv);
          addGas(sim, c, SPECIES.H2O, dv);
          m -= dv;
          evaporated += dv;
        }
      }
      const i = c % nx, j = Math.floor(c / nx) % ny, k = Math.floor(c / (nx * ny));
      sim.v[vAt(g, i, j, k)] -= (drag * m) / Math.max(sim.rho[c], 1e-6);
    }
    // Wetting: the column ends on the fuel's top surface (sprinklerSetup's
    // columns stop at it) or the floor. A share of the water left soaks into
    // that fuel cell's surface layer, mixed in at the water's temperature so
    // the layer keeps its energy; fuel stacked below it stays dry (shielded).
    if (wet > 0 && m > 0 && c >= 0 && sim.kind[c] !== SOLID && sim.fuelMass[c] > 0 && sim.fuels[sim.fuelKind[c]]?.water?.wets !== false) {
      const dw = wet * m;
      const Cs = solidCapacity(sim, c);
      sim.Ts[c] = (Cs * sim.Ts[c] + dw * WATER_CP * WATER_TEMPERATURE) / (Cs + dw * WATER_CP);
      sim.moisture[c] += dw;
      m -= dw;
      wetted += dw;
    }
    floor += m;
  }
  sup.stats.emitted += emitted;
  sup.stats.evaporated += evaporated;
  sup.stats.wetted = (sup.stats.wetted || 0) + wetted;
  sup.stats.floor += floor;
}

/**
 * Foam-water sprinklers' water on class A fuel (suppressionSystems.js):
 * while a room's foam is applied, the first fuel cell down each of its
 * columns takes a share of the solution landing on its column, mixed in at
 * the water's temperature, as the spray's wetting does. Liquids (blanketed
 * in chemistry), metals and packs take none (fireClasses.js waterResponse).
 */
function foamWet(sim, dt) {
  const sys = sim.systems;
  if (!sys || !sys.wet.length) return;
  for (let q = 0; q < sys.wet.length; q += 1) {
    const share = sys.state[sys.wetRoom[q] * STATE.stride + STATE.wet];
    if (!(share > 0)) continue;
    const c = sys.wet[q];
    if (sim.kind[c] === SOLID || !(sim.fuelMass[c] > 0) || sim.fuels[sim.fuelKind[c]]?.water?.wets === false) continue;
    const dw = share * sys.wetFlux[q] * dt;
    const Cs = solidCapacity(sim, c);
    sim.Ts[c] = (Cs * sim.Ts[c] + dw * WATER_CP * WATER_TEMPERATURE) / (Cs + dw * WATER_CP);
    sim.moisture[c] += dw;
    sys.rooms[sys.wetRoom[q] - 1].stats.wetted += dw;
  }
}

/** The gas speed at a cell's centre from its six faces (m/s), for a sprinkler's link. */
export function cellSpeed(sim, c) {
  const g = sim.grid;
  const i = c % g.nx, j = Math.floor(c / g.nx) % g.ny, k = Math.floor(c / (g.nx * g.ny));
  const u = 0.5 * (sim.u[uAt(g, i, j, k)] + sim.u[uAt(g, i + 1, j, k)]);
  const v = 0.5 * (sim.v[vAt(g, i, j, k)] + sim.v[vAt(g, i, j + 1, k)]);
  const w = 0.5 * (sim.w[wAt(g, i, j, k)] + sim.w[wAt(g, i, j, k + 1)]);
  return Math.hypot(u, v, w);
}

const scratch = new Float64Array(NS);

/**
 * Change a gas cell's species masses by `delta` (kg each), nitrogen kept,
 * and renormalise: Y_k = m_k / sum m, rho = sum m / V.
 */
export function changeGas(sim, c, delta) {
  const o = c * NS;
  const mass = sim.rho[c] * sim.V;
  let n2 = mass;
  let total = 0;
  for (let q = 0; q < NS; q += 1) {
    const m = sim.Y[o + q] * mass;
    n2 -= m;
    scratch[q] = Math.max(0, m + delta[q]);
    total += scratch[q];
  }
  total += Math.max(0, n2);
  if (!(total > 0)) return;
  for (let q = 0; q < NS; q += 1) sim.Y[o + q] = scratch[q] / total;
  sim.rho[c] = total / sim.V;
}

const delta = new Float64Array(NS);

/** Put dm kg of one species into a gas cell. */
function addGas(sim, c, s, dm) {
  if (!(dm > 0)) return;
  delta.fill(0);
  delta[s] = dm;
  changeGas(sim, c, delta);
}

/**
 * Burning dm kg of fuel (Eq. 12 with the fire class's yields): o2 dm of
 * oxygen consumed, (1 + o2) dm of CO2, CO, soot and water produced. From a
 * solid the gas gains dm in all; from fuel vapour (`vapour`) the vapour is
 * taken from the gas too, and the gas's mass is unchanged.
 */
function addProducts(sim, c, st, dm, vapour = false) {
  delta.fill(0);
  delta[SPECIES.O2] = -st.o2 * dm;
  delta[SPECIES.CO2] = st.co2 * dm;
  delta[SPECIES.CO] = st.co * dm;
  delta[SPECIES.SOOT] = st.soot * dm;
  delta[SPECIES.H2O] = st.h2o * dm;
  if (vapour) delta[SPECIES.FUEL] = -dm;
  changeGas(sim, c, delta);
}

// ---------------------------------------------------------- heat transfer

/**
 * Line 30: conduction in solids (Eq. 18) and the interface exchange of Eq. 19
 * between each solid or porous cell and the gas beside or inside it, each as
 * an exact two-body relaxation.
 */
function conduct(sim, dt) {
  const { P, grid: g } = sim;
  const { nx, ny, nz, h } = g;
  const A = h * h;
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const c = at(g, i, j, k);
        const kc = sim.kind[c];
        if (kc === GAS) continue;
        const Cs = solidCapacity(sim, c);
        if (!(Cs > 0)) continue;
        if (kc === POROUS) {
          // Gas flows through: exchange with the cell's own gas over the
          // fuel's exposed surface.
          const Cg = sim.rho[c] * mixtureCp(sim.Y, c * NS) * sim.V;
          const q = exchange(sim.Ts[c], sim.T[c], Cs, Cg, P.convection * Math.max(sim.area[c], A), dt);
          sim.Ts[c] -= q / Cs;
          sim.T[c] += q / Cg;
        }
        // A face to gas: the fuel's coefficient, or the wall's for an inert solid.
        const hc = kc === SOLID ? P.wallConvection : P.convection;
        // Forward neighbours only, so each pair is visited once.
        for (const [a, b, d] of [DIRS[0], DIRS[2], DIRS[4], DIRS[1], DIRS[3], DIRS[5]]) {
          const ii = i + a, jj = j + b, kk = k + d;
          if (ii < 0 || jj < 0 || kk < 0 || ii >= nx || jj >= ny || kk >= nz) continue;
          const n = at(g, ii, jj, kk);
          const kn = sim.kind[n];
          if (kn === GAS) {
            const Cg = sim.rho[n] * mixtureCp(sim.Y, n * NS) * sim.V;
            const q = exchange(sim.Ts[c], sim.T[n], Cs, Cg, hc * A, dt);
            sim.Ts[c] -= q / Cs;
            sim.T[n] += q / Cg;
          } else if (n > c) {
            const Cn = solidCapacity(sim, n);
            if (!(Cn > 0)) continue; // contents drawn with no mass: nothing to exchange (0 / 0)
            const kk2 = 0.5 * (Math.max(sim.inertK[c], 0.1) + Math.max(sim.inertK[n], 0.1));
            const q = exchange(sim.Ts[c], sim.Ts[n], Cs, Cn, (kk2 * A) / h, dt);
            sim.Ts[c] -= q / Cs;
            sim.Ts[n] += q / Cn;
          }
        }
      }
    }
  }
}

/** Line 31: the equation of state for every open cell. */
function updateDensity(sim) {
  for (let c = 0; c < sim.N; c += 1) {
    if (sim.kind[c] === SOLID) continue;
    sim.rho[c] = gasDensity(sim.T[c], sim.Y, c * NS);
  }
}

/** Line 32: the domain's sides are open to still ambient air. */
function boundaries(sim) {
  const { grid: g, P } = sim;
  const { nx, ny, nz } = g;
  const reset = (c) => {
    sim.T[c] = P.ambient;
    for (let q = 0; q < NS; q += 1) sim.Y[c * NS + q] = 0;
    sim.Y[c * NS + SPECIES.O2] = Y_O2_AIR;
    sim.rho[c] = sim.rho0;
  };
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        if (i === 0 || k === 0 || i === nx - 1 || k === nz - 1 || j === ny - 1) {
          const c = at(g, i, j, k);
          if (sim.kind[c] !== SOLID) reset(c);
        }
      }
    }
  }
}

/**
 * Per open cell: which of its six neighbours are open cells (bit d for
 * DIRS[d]), how many pressure links it has (open neighbours plus open domain
 * sides; the ground is closed), and the open cells split red and black.
 * Rebuilt only when a cell changes kind (fuel burnt out).
 */
function topology(sim) {
  if (sim.topo && sim.topo.version === sim.kindVersion) return sim.topo;
  const g = sim.grid;
  const { nx, ny, nz } = g;
  const mask = new Uint8Array(sim.N);
  const count = new Uint8Array(sim.N);
  const red = [];
  const black = [];
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const c = at(g, i, j, k);
        if (sim.kind[c] === SOLID) continue;
        let m = 0, n = 0;
        for (let d = 0; d < 6; d += 1) {
          const [a, b, e] = DIRS[d];
          const ii = i + a, jj = j + b, kk = k + e;
          if (jj < 0) continue; // the ground
          if (ii < 0 || kk < 0 || ii >= nx || jj >= ny || kk >= nz) {
            n += 1; // open side, pressure zero
            continue;
          }
          if (sim.kind[at(g, ii, jj, kk)] === SOLID) continue;
          m |= 1 << d;
          n += 1;
        }
        mask[c] = m;
        count[c] = Math.max(1, n);
        ((i + j + k) % 2 ? black : red).push(c);
      }
    }
  }
  const offsets = DIRS.map(([a, b, e]) => a + nx * (b + ny * e));
  sim.topo = { version: sim.kindVersion, mask, count, colours: [Int32Array.from(red), Int32Array.from(black)], offsets };
  return sim.topo;
}

/**
 * Line 33: make the flow divergence-free by red-black Gauss-Seidel on the
 * pressure Poisson equation. Faces against solids carry no flow; the domain's
 * sides are open (pressure zero outside). Faces in porous cells are slowed.
 */
function project(sim, dt) {
  const { grid: g, P } = sim;
  const { nx, ny, nz, h } = g;
  const solid = (i, j, k) => i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz && sim.kind[at(g, i, j, k)] === SOLID;
  // Zero the faces of solid cells.
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i <= nx; i += 1) if (solid(i - 1, j, k) || solid(i, j, k)) sim.u[uAt(g, i, j, k)] = 0;
    }
  }
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j <= ny; j += 1) {
      for (let i = 0; i < nx; i += 1) if (solid(i, j - 1, k) || solid(i, j, k) || j === 0) sim.v[vAt(g, i, j, k)] = 0;
    }
  }
  for (let k = 0; k <= nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) if (solid(i, j, k - 1) || solid(i, j, k)) sim.w[wAt(g, i, j, k)] = 0;
    }
  }
  const div = workArray(sim, "div", sim.N);
  div.fill(0);
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const c = at(g, i, j, k);
        if (sim.kind[c] === SOLID) continue;
        div[c] =
          (sim.u[uAt(g, i + 1, j, k)] - sim.u[uAt(g, i, j, k)] +
            sim.v[vAt(g, i, j + 1, k)] - sim.v[vAt(g, i, j, k)] +
            sim.w[wAt(g, i, j, k + 1)] - sim.w[wAt(g, i, j, k)]) / h;
      }
    }
  }
  const scale = (sim.rho0 * h * h) / dt;
  const p = sim.p;
  const topo = topology(sim);
  const off = topo.offsets;
  // Sweeps stop early once the largest pressure change in a sweep would move
  // the flow by less than `projectionTolerance` m/s (the pressure is kept
  // from the last step, so a calm flow needs few).
  const settle = (P.projectionTolerance * sim.rho0 * h) / dt;
  sim.stats.sweeps = 0;
  for (let sweep = 0; sweep < P.projection; sweep += 1) {
    let change = 0;
    for (const list of topo.colours) {
      for (let q = 0; q < list.length; q += 1) {
        const c = list[q];
        const mask = topo.mask[c];
        let sum = 0;
        if (mask & 1) sum += p[c + off[0]];
        if (mask & 2) sum += p[c + off[1]];
        if (mask & 4) sum += p[c + off[2]];
        if (mask & 8) sum += p[c + off[3]];
        if (mask & 16) sum += p[c + off[4]];
        if (mask & 32) sum += p[c + off[5]];
        const next = (sum - scale * div[c]) / topo.count[c];
        const d = Math.abs(next - p[c]);
        if (d > change) change = d;
        p[c] = next;
      }
    }
    sim.stats.sweeps = sweep + 1;
    if (sweep >= 3 && change < settle) break;
  }
  const k2 = dt / (sim.rho0 * h);
  const pr = (i, j, k) => (i < 0 || k < 0 || i >= nx || j >= ny || k >= nz || j < 0 ? 0 : p[at(g, i, j, k)]);
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i <= nx; i += 1) {
        if (solid(i - 1, j, k) || solid(i, j, k)) continue;
        sim.u[uAt(g, i, j, k)] -= k2 * (pr(i, j, k) - pr(i - 1, j, k));
      }
    }
  }
  for (let k = 0; k < nz; k += 1) {
    for (let j = 1; j <= ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        if (solid(i, j - 1, k) || solid(i, j, k)) continue;
        sim.v[vAt(g, i, j, k)] -= k2 * (pr(i, j, k) - pr(i, j - 1, k));
      }
    }
  }
  for (let k = 0; k <= nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        if (solid(i, j, k - 1) || solid(i, j, k)) continue;
        sim.w[wAt(g, i, j, k)] -= k2 * (pr(i, j, k) - pr(i, j, k - 1));
      }
    }
  }
  // Porous cells slow the flow through them (the paper's velocity bound).
  if (P.porousVelocity < 1) {
    for (let c = 0; c < sim.N; c += 1) {
      if (sim.kind[c] !== POROUS) continue;
      const i = c % nx, j = Math.floor(c / nx) % ny, k = Math.floor(c / (nx * ny));
      sim.u[uAt(g, i, j, k)] *= P.porousVelocity;
      sim.u[uAt(g, i + 1, j, k)] *= P.porousVelocity;
      sim.v[vAt(g, i, j, k)] *= P.porousVelocity;
      sim.v[vAt(g, i, j + 1, k)] *= P.porousVelocity;
      sim.w[wAt(g, i, j, k)] *= P.porousVelocity;
      sim.w[wAt(g, i, j, k + 1)] *= P.porousVelocity;
    }
  }
}

/**
 * Algorithm 1 line 35: a flame-front particle in a cell where no combustion
 * took place this step has left the flame front and is removed.
 */
function cullFlameFront(sim) {
  const { grid: g, parts: pr } = sim;
  for (let q = pr.n - 1; q >= 0; q -= 1) {
    if (pr.kind[q] !== 0) continue;
    const cell = at(g, Math.floor(pr.x[q * 3]), Math.floor(pr.x[q * 3 + 1]), Math.floor(pr.x[q * 3 + 2]));
    if (!sim.burning[cell]) removeParticle(pr, q);
  }
}

/**
 * The ignition source. With `temperature`, the first item is alight: its
 * surface layer is held at least at that temperature for `duration` seconds
 * (a design fire starts with its first item burning), after which the fire
 * must sustain itself. With `power` instead, a burner of that many watts
 * heats the item's surface for `duration` seconds. A prescribed burner
 * (createBurner), when the scene has one, releases its heat here too.
 */
function pilot(sim, dt) {
  if (sim.burner) burnerStep(sim, dt);
  const pl = sim.pilot;
  if (!pl || sim.time >= pl.duration || !pl.cells.length) return;
  if (pl.temperature) {
    for (const c of pl.cells) if (sim.Ts[c] < pl.temperature) sim.Ts[c] = pl.temperature;
    return;
  }
  const share = (pl.power * dt) / pl.cells.length;
  for (const c of pl.cells) {
    const C = solidCapacity(sim, c);
    if (C > 0) sim.Ts[c] += share / C;
    else sim.T[c] += share / (sim.rho[c] * mixtureCp(sim.Y, c * NS) * sim.V);
  }
}

// ---------------------------------------------------------- burner

/**
 * Methane, a prescribed burner's default fuel: 50.0 MJ/kg and 2.72 kg CO2
 * per kg, no CO or soot (Tewarson's table in the SFPE Handbook). Its oxygen
 * is Huggett's, as for every fuel here (stoichiometry).
 */
export const METHANE = Object.freeze({ heatOfCombustion: 50.0e6, yields: Object.freeze({ co2: 2.72, co: 0, soot: 0 }) });

/**
 * A prescribed heat release, for checking the model against measured tests
 * (tools/sprinkler-validation.mjs). Not the paper's, and no app run makes
 * one. scene.burner = { cells, hrr, radiantFraction = 0.2, fuel = METHANE, rays = 16 }:
 *   - hrr is a table [[t (s), Q (W)], ...], linear between its rows and
 *     held at its end values outside them (as an FDS ramp); each step
 *     releases the table's exact integral over the step;
 *   - the convective part, (1 - radiantFraction) Q dt, heats the gas of
 *     `cells` (the open cells over the burner's area: a flame volume),
 *     shared equally, and with it go the products of burning Q dt / H_c of
 *     `fuel` (oxygen at 417 kJ per mol O2, stoichiometry). The fuel enters
 *     at ambient temperature, so each cell's gas enthalpy above ambient,
 *     m c_p (T - T_amb), rises by exactly its share;
 *   - the radiant part leaves the burner's axis cell along `rays`
 *     directions (a Fibonacci sphere, turned each step), absorbed by soot on
 *     the way and by the first solid it meets (march), as the gas's own
 *     radiation is;
 *   - all of Q dt counts as released heat (sim.stats.released), so the
 *     frame heat release that the sprinklers' Alpert estimate reads
 *     (runScene) includes the burner.
 * The axis cell is the lowest-layer cell nearest the cells' centroid: the
 * plume's axis for the sprinklers when the scene has no first item. With
 * `rays` 0 the radiant part leaves the domain. (buildScene's `burner` option
 * is something else: the pilot that lights the first item.)
 *
 * The first item as a design fire (buildScene's `designFire`, 2026-11-08)
 * is a burner too: its table is the item's t-squared curve until its energy
 * is spent, its fuel the item's (with its soot), `radiantFraction` 0 (the
 * solver's own radiation from the hot, sooty flame cells takes that part),
 * and `layers` the open cells over the item's area a layer at a time from
 * its base: each step's heat goes to as many layers as Heskestad's mean
 * flame height reaches for that heat release over a fire of `diameter`
 * (burnerCells). The run caps it once water or clean agent reaches the item
 * (b.suppression, b.stopAt; burnerHeat). The GPU port runs this kind of
 * burner (its radiant part is none); the validation's, with rays, is CPU only.
 */
export function createBurner(spec, kind, grid) {
  const { nx, ny } = grid;
  const open = (c) => c >= 0 && c < kind.length && kind[c] !== SOLID;
  const layers = (spec.layers || [spec.cells || []]).map((layer) => layer.filter(open)).filter((layer) => layer.length);
  const cells = Int32Array.from(layers.flat());
  if (!cells.length) throw new Error("A burner needs at least one open cell.");
  // Where each layer starts in `cells`, and where the last ends.
  const layerStart = new Int32Array(layers.length + 1);
  for (let l = 0; l < layers.length; l += 1) layerStart[l + 1] = layerStart[l] + layers[l].length;
  const table = spec.hrr || [];
  if (!table.length) throw new Error("A burner needs a heat release table.");
  const times = Float64Array.from(table, (row) => row[0]);
  const values = Float64Array.from(table, (row) => row[1]);
  for (let q = 0; q < times.length; q += 1) {
    if (!Number.isFinite(times[q]) || !Number.isFinite(values[q]) || values[q] < 0 || (q && times[q] < times[q - 1])) {
      throw new Error("A burner's heat release table needs finite, non-negative values at non-decreasing times.");
    }
  }
  // The table's integral at each row, J.
  const cumulative = new Float64Array(times.length);
  for (let q = 1; q < times.length; q += 1) cumulative[q] = cumulative[q - 1] + 0.5 * (values[q - 1] + values[q]) * (times[q] - times[q - 1]);
  const radiantFraction = spec.radiantFraction ?? 0.2;
  if (!(radiantFraction >= 0 && radiantFraction <= 1)) throw new Error("A burner's radiant fraction must be between 0 and 1.");
  // The axis: the lowest layer's cell nearest the centroid of all the cells.
  const ijk = (c) => [c % nx, Math.floor(c / nx) % ny, Math.floor(c / (nx * ny))];
  let si = 0, sk = 0, low = Infinity;
  for (const c of cells) {
    const [i, j, k] = ijk(c);
    si += i;
    sk += k;
    low = Math.min(low, j);
  }
  si /= cells.length;
  sk /= cells.length;
  let axis = cells[0], best = Infinity;
  for (const c of cells) {
    const [i, j, k] = ijk(c);
    const d = (i - si) ** 2 + (k - sk) ** 2;
    if (j === low && d < best) [axis, best] = [c, d];
  }
  const [ai, aj, ak] = ijk(axis);
  const rays = Math.max(0, Math.round(spec.rays ?? 16));
  const dirs = new Float64Array(rays * 3);
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let r = 0; r < rays; r += 1) {
    const y = 1 - (2 * (r + 0.5)) / rays;
    const s = Math.sqrt(1 - y * y);
    dirs[r * 3] = s * Math.cos(golden * r);
    dirs[r * 3 + 1] = y;
    dirs[r * 3 + 2] = s * Math.sin(golden * r);
  }
  return {
    cells,
    layerStart,
    diameter: spec.diameter ?? 0,
    h: grid.h,
    axis,
    origin: [ai + 0.5, aj + 0.5, ak + 0.5],
    times,
    values,
    cumulative,
    radiantFraction,
    stoich: stoichiometry(spec.fuel || METHANE),
    dirs,
    dir: [0, 0, 0],
    turn: 0,
    // What it has given off so far, J.
    released: 0,
    convective: 0,
    radiant: 0,
    // A design-fire item's curve (buildScene), and, set by the run, water
    // on the item ({start, tau, q, by}) and the time clean agent puts its flames out.
    design: spec.design ?? null,
    suppression: null,
    stopAt: Infinity,
  };
}

/** A burner's heat release rate at t, W: its table, linear between rows, held at its ends. */
export function burnerRate(b, t) {
  const { times: T, values: Q } = b;
  const n = T.length;
  if (t <= T[0]) return Q[0];
  if (t >= T[n - 1]) return Q[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (T[mid] <= t) lo = mid;
    else hi = mid;
  }
  return Q[lo] + ((t - T[lo]) / (T[hi] - T[lo])) * (Q[hi] - Q[lo]);
}

/** Heskestad's mean flame height (m) for a heat release Q (W) over a fire of diameter D (m): 0.235 Q^(2/5) - 1.02 D, Q in kW. */
export function flameHeight(Q, D) {
  return 0.235 * Math.max(0, Q / 1000) ** 0.4 - 1.02 * D;
}

/** How many of a burner's cells take a step's heat at a heat release Q (W): all of them, or a design-fire item's layers its flame reaches (at least one). */
export function burnerCells(b, Q) {
  const L = b.layerStart.length - 1;
  if (L <= 1 || !(b.diameter > 0)) return b.layerStart[L];
  return b.layerStart[Math.min(L, Math.max(1, Math.ceil(flameHeight(Q, b.diameter) / b.h)))];
}

/**
 * The heat (J) a burner gives off over [t0, t1]: its table's integral; once
 * water reaches the item (b.suppression = {start, tau, q}), the rate decays
 * from q as Evans (1993) measured on wetted furnishings, q exp(-(t -
 * start) / tau), never above the table; once the room's clean agent reaches
 * the class A extinguishing concentration (b.stopAt), nothing.
 */
export function burnerHeat(b, t0, t1) {
  const end = Math.min(t1, b.stopAt ?? Infinity);
  if (!(end > t0)) return 0;
  const s = b.suppression;
  if (!s || !(end > s.start)) return burnerEnergy(b, end) - burnerEnergy(b, t0);
  const a = Math.max(t0, s.start);
  const before = a > t0 ? burnerEnergy(b, a) - burnerEnergy(b, t0) : 0;
  const decay = s.q * s.tau * (Math.exp(-(a - s.start) / s.tau) - Math.exp(-(end - s.start) / s.tau));
  return before + Math.max(0, Math.min(decay, burnerEnergy(b, end) - burnerEnergy(b, a)));
}

/** The integral of a burner's table from its first row to time t, J (burnerEnergy(b, t1) - burnerEnergy(b, t0) is released over [t0, t1]). */
export function burnerEnergy(b, t) {
  const { times: T, values: Q, cumulative: F } = b;
  const n = T.length;
  if (t <= T[0]) return (t - T[0]) * Q[0];
  if (t >= T[n - 1]) return F[n - 1] + (t - T[n - 1]) * Q[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (T[mid] <= t) lo = mid;
    else hi = mid;
  }
  const q = Q[lo] + ((t - T[lo]) / (T[hi] - T[lo])) * (Q[hi] - Q[lo]);
  return F[lo] + 0.5 * (Q[lo] + q) * (t - T[lo]);
}

/** One step of the prescribed burner (createBurner): its heat, products and radiation over [time, time + dt]. */
export function burnerStep(sim, dt) {
  const b = sim.burner;
  const E = burnerHeat(b, sim.time, sim.time + dt);
  if (!(E > 0)) return;
  const n = burnerCells(b, E / dt);
  const convective = (1 - b.radiantFraction) * E;
  const share = convective / n;
  const fuel = E / b.stoich.heat / n;
  const Ta = sim.P.ambient;
  for (let q = 0; q < n; q += 1) {
    const c = b.cells[q];
    const excess = sim.rho[c] * sim.V * mixtureCp(sim.Y, c * NS) * (sim.T[c] - Ta);
    addProducts(sim, c, b.stoich, fuel);
    sim.T[c] = Ta + (excess + share) / (sim.rho[c] * sim.V * mixtureCp(sim.Y, c * NS));
  }
  const rays = b.dirs.length / 3;
  const radiant = E - convective;
  if (rays && radiant > 0) {
    // The directions turn about the vertical by the golden angle each step,
    // so the radiant heat spreads over the surfaces without random numbers.
    b.turn += Math.PI * (3 - Math.sqrt(5));
    const cos = Math.cos(b.turn), sin = Math.sin(b.turn);
    for (let r = 0; r < rays; r += 1) {
      const x = b.dirs[r * 3], z = b.dirs[r * 3 + 2];
      b.dir[0] = cos * x - sin * z;
      b.dir[1] = b.dirs[r * 3 + 1];
      b.dir[2] = sin * x + cos * z;
      march(sim, b.origin[0], b.origin[1], b.origin[2], b.dir, radiant / rays, b.axis, false);
    }
  }
  b.released += E;
  b.convective += convective;
  b.radiant += radiant;
  sim.stats.released += E;
}

/**
 * A persistent working array on the solver, made once (review 14: each
 * step allocated about 25 arrays of N cells).
 */
function workArray(sim, name, length, Type = Float32Array) {
  sim.work ||= {};
  let a = sim.work[name];
  if (!a || a.length !== length || !(a instanceof Type)) a = sim.work[name] = new Type(length);
  return a;
}

/** A persistent copy of a field: its scratch array, filled from the field. */
function copyOf(sim, name, field) {
  const a = workArray(sim, name, field.length, field.constructor);
  a.set(field);
  return a;
}

/** Algorithm 1's stages in order, as [name, function(sim, dt)]. */
const STAGES = [
  ["pilot", pilot],
  ["particles", particleStep],
  ["radiation", radiation],
  ["particleToGrid", (sim) => particleToGrid(sim)],
  ["advect", advect],
  ["density", (sim) => updateDensity(sim)],
  ["buoyancy", buoyancy],
  ["vorticity", vorticityConfinement],
  ["diffuse", diffuse],
  ["ducts", ducts],
  ["chemistry", chemistry],
  ["spray", spray],
  ["foamWet", foamWet],
  ["cullFlameFront", (sim) => cullFlameFront(sim)],
  ["conduct", conduct],
  ["density2", (sim) => updateDensity(sim)],
  ["boundaries", (sim) => boundaries(sim)],
  ["project", project],
];

/**
 * One frame of Algorithm 1. With `sim.profile` set to an object, each
 * stage's wall time (ms) is added to sim.profile[stage].
 */
export function step(sim, dt = sim.P.dt) {
  const profile = sim.profile;
  if (profile) {
    for (const [name, fn] of STAGES) {
      const t0 = performance.now();
      fn(sim, dt);
      profile[name] = (profile[name] || 0) + performance.now() - t0;
    }
  } else {
    for (const [, fn] of STAGES) fn(sim, dt);
  }
  let maxT = sim.P.ambient;
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] !== SOLID && sim.T[c] > maxT) maxT = sim.T[c];
  sim.stats.maxT = maxT;
  sim.time += dt;
  return sim;
}
