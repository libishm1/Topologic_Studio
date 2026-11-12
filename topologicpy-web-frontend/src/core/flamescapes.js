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
 *   - A wall one cell thick whose material lists its layers (a validation
 *     scene's boards) conducts through them normal to its faces, in sub-cell
 *     nodes, as FDS's walls do (wallSetup, walls; calibration plan 13.4 item
 *     1); the app's materials list none and stay lumped into their cells.
 *   - Soot deposits on solid surfaces, thermophoretically onto cooler walls
 *     and by settling onto those below, as FDS deposits aerosols (deposition;
 *     plan 13.4 item 2): the gas's smoke thins near cold walls.
 *   - A sealed region (a room behind closed doors) is a pressure zone, as in
 *     FDS: it keeps its gas's mass, and its pressure follows that mass and its
 *     heat (pressureZones; plan 13.8 item 5). Open regions stay at ambient.
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
  // One-dimensional walls (calibration plan 13.4, item 1; wallSetup): a
  // solid cell one cell thick whose inert material lists its `layers`
  // conducts normal to its faces through `wallNodes` nodes across its real
  // thickness, finer by `wallStretch` towards each layer's faces, instead of
  // holding one temperature. 20 nodes at 1.2 keep a heated gypsum board within
  // 0.4 % of the series solution and a concrete face within 0.8 % of erfc
  // (flamescapes.wall.test.mjs), and the burner room's heat closes as with
  // lumped walls (tools/calibration/heat-closure.mjs, lumped and layered); so
  // it is on. 0 lumps every wall into its cells as before. The app's
  // materials list no layers, so its runs are unchanged either way.
  wallLayers: 1,
  wallNodes: 20,
  wallStretch: 1.2,
  // Soot deposition on solid surfaces (calibration plan 13.4, item 2;
  // deposition): thermophoretic onto walls cooler than their gas, and
  // gravitational onto upward-facing surfaces, as FDS models them (Floyd,
  // Overholt and Ezekoye 2014, FEMTC). FDS's soot: particles of 1800 kg/m^3
  // (theirs), 0.26 W/(m K) (FDS's CONDUCTIVITY_SOLID, Ben-Dor et al. 2002),
  // 0.03 um for thermophoresis (FDS's THERMOPHORETIC_DIAMETER, a primary
  // particle); 1 um for settling (FDS's examples' MEAN_DIAMETER; FDS sets
  // none). On a cold wall in 600 K gas it deposits at Talbot's velocity within
  // 0.05 %, and the soot released is the soot in the gas, on the surfaces and
  // gone (flamescapes.soot.test.mjs, heat-closure.mjs's soot budgets); so it
  // is on. 0 keeps all soot in the gas, as before.
  sootDeposition: 1,
  sootDensity: 1800, // kg/m^3
  sootDiameter: 1e-6, // m, settling
  sootThermophoreticDiameter: 0.03e-6, // m
  sootConductivity: 0.26, // W/(m K)
  sootAbsorption: 500, // the paper's residual absorption coefficient kappa
  // Confinement strength (Fedkiw, Stam and Jensen 2001: a force vorticity x h
  // x (N x omega)), the solver's one fitted parameter: 3, refitted 2026-11-12
  // on the heat-conserving transport (conservative, expansion) by the rule
  // fixed on 2026-11-11 (tools/calibration/fit-confinement.mjs: candidates 0
  // to 10 on McCaffrey's plume and Alpert's ceiling jet; scores 0.11 for 3,
  // 0.18 for 2; 10 lost the plume's axis). 6 had been fitted on the old
  // transport, which made heat (wiki/research/fire-calibration-claim-protocol-2026-11-12.md).
  vorticity: 3,
  // A frame's steps are cut so that the flow moves at most `cfl` cells a step
  // and the confinement force at most vorticity x step = `vorticityStep`, in
  // at most `maxSubsteps` (substepsFor; wiki/research/fire-calibration-plan-2026-11-11.md, B).
  cfl: 1,
  vorticityStep: 0.5,
  maxSubsteps: 8,
  // Heat-conserving transport (proposal F, wiki/research/fire-calibration-plan-2026-11-11.md,
  // F6): the gas's mass and species move through the faces, conserved
  // (transportGas), where back-tracing T made heat; and heat put into gas
  // expands it, a divergence ln(T1/T0)/dt in the projection (the low-Mach
  // form FDS uses), so heated gas leaves its cell instead of thinning in
  // place. 0 restores the paper's transport.
  conservative: 1,
  expansion: 1,
  // A pressure for each sealed region (calibration plan 13.8, item 5;
  // pressureZones), as FDS's pressure zones: a region no open side reaches
  // keeps its gas's mass, and its pressure follows that mass and its heat.
  // 0 holds every region at ambient pressure, as before: a sealed box cooled
  // by its walls then drew in 31 % more gas in 60 s.
  pressureZones: 1,
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
  // The inert materials' properties: INERT, or a test's own (options.inert,
  // the same order) where a validation run matches its walls (2026-11-12).
  // A material may list its `layers` from its exposed face ({thickness m,
  // rho, c, k}): with wallLayers its one-cell-thick walls conduct through
  // them (wallSetup).
  const inert = P.inert || INERT;
  for (let c = 0; c < N; c += 1) {
    let C = 0, K = 0, E = 0, w = 0;
    for (let m = 0; m < inert.length; m += 1) {
      const f = fraction[c * MATERIALS + m];
      if (f <= 0) continue;
      C += f * inert[m].rho * inert[m].c * V;
      K += f * inert[m].k;
      E += f * inert[m].eps;
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
  // One-dimensional walls, null when off or no material lists its layers.
  const wall = P.wallLayers > 0 && inert.some((m) => m.layers?.length) ? wallSetup(grid, fraction, solidMask, inert, inertC, P) : null;
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
    wall,
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
    // Soot each open cell has lost to the solid surfaces beside it (kg, deposition), when deposition is on.
    sootDeposit: P.sootDeposition > 0 ? new Float32Array(N) : null,
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
    // Steps taken: the GPU's hash streams number steps alike (flamescapesGpu.js stepBase).
    steps: 0,
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
    // Soot's budget (kg): released by burning and pyrolysis, deposited on surfaces, carried out of the domain by its shell.
    stats: { hrr: 0, burnt: 0, embersSpawned: 0, spotIgnitions: 0, maxT: P.ambient, released: 0, radiated: 0, pilotHeat: 0, shellHeat: 0, massFloor: 0, courantClamped: 0, sootReleased: 0, sootDeposited: 0, sootLeft: 0 },
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

/** The GPU's PCG hash (kernels.js pcg), on 32-bit unsigned integers. */
export function pcgHash(v) {
  const s = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const w = Math.imul((s >>> ((s >>> 28) + 4)) ^ s, 277803737) >>> 0;
  return ((w >>> 22) ^ w) >>> 0;
}

/** Cell c's random stream in a step (kernels.js seedOf): its state before the first draw. */
export function cellStream(c, step, salt, seed) {
  return pcgHash((c ^ pcgHash((Math.imul(step, 8) + salt + pcgHash(seed >>> 0)) >>> 0)) >>> 0);
}

/**
 * The GPU's first draw from cell c's stream in a step (kernels.js seedOf then
 * rnd), in [0, 1): the same number on both paths for the same cell, step,
 * salt and seed.
 */
export function hashUnit(c, step, salt, seed) {
  return (pcgHash(cellStream(c, step, salt, seed)) >>> 8) / 16777216;
}

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
export function particleStep(sim, dt) {
  const { P, grid: g, parts: pr } = sim;
  const h = g.h;
  const vel = [0, 0, 0];
  // Flame front (section 4.7): six particles from every cell that burned,
  // one per axis direction; a flame particle keeps its direction in `m`.
  // When they would overfill the flame particles' share of the buffer, each
  // burning cell spawns with the same chance, by a hash of the cell and the
  // step that the GPU draws alike (kernels.js spawnFlame). Refused in cell
  // order instead (before 2026-11-09), the cells numbered last never spawned
  // during a flare, and the CPU's fire fell back where the GPU's, refused by
  // whichever thread came last, burned on (tools/flamescapes-gpu-check.mjs
  // --phase spread).
  let burningCells = 0;
  for (let c = 0; c < sim.N; c += 1) if (sim.burning[c]) burningCells += 1;
  const room = Math.max(0, Math.min(P.maxParticles, Math.ceil(P.flameShare * P.maxParticles)) - pr.n);
  const share = 6 * burningCells > room ? room / (6 * burningCells) : 1;
  for (let c = 0; c < sim.N; c += 1) {
    if (!sim.burning[c]) continue;
    if (share < 1 && hashUnit(c, sim.steps, 5, P.seed) >= share) continue;
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
    // The move, in pieces of at most half a cell, stopping in the first solid
    // it enters: moved whole, a particle faster than a cell a step jumped
    // walls a cell thick and lit the contents of a sealed room (2026-11-12).
    const dx = (pr.vel[q * 3] * dt) / h, dy = (pr.vel[q * 3 + 1] * dt) / h, dz = (pr.vel[q * 3 + 2] * dt) / h;
    // At most 64 pieces (the GPU's cap), so a non-finite speed cannot hang the step.
    const pieces = Math.min(64, Math.max(1, Math.ceil(2 * Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)))) || 1);
    let nxp = pr.x[q * 3], nyp = pr.x[q * 3 + 1], nzp = pr.x[q * 3 + 2];
    let out = false, cell = -1;
    for (let piece = 0; piece < pieces; piece += 1) {
      nxp += dx / pieces;
      nyp += dy / pieces;
      nzp += dz / pieces;
      out = nxp < 0 || nyp < 0 || nzp < 0 || nxp >= g.nx || nyp >= g.ny || nzp >= g.nz;
      if (out) break;
      cell = at(g, Math.floor(nxp), Math.floor(nyp), Math.floor(nzp));
      if (sim.kind[cell] === SOLID) break;
    }
    pr.x[q * 3] = nxp;
    pr.x[q * 3 + 1] = nyp;
    pr.x[q * 3 + 2] = nzp;
    pr.age[q] += dt;
    if (out || pr.age[q] > pr.life[q]) {
      removeParticle(pr, q);
      continue;
    }
    // Boundary handling: flame-front particles stop at solids; embers
    // deposit there (particleToGrid) and are removed.
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
 *
 * Every cell emits from its temperature at the start of the stage; what the
 * rays leave is added once all have emitted, as on the GPU. Added as each ray
 * went (before 2026-11-09), a cell later in the loop re-emitted heat it had
 * taken in the same step, so the result hung on the cells' numbering: on the
 * CPU a 200 kW burner's room cooled after the contents flared and its fire
 * fell back, where the GPU's went on at 1 to 2 MW (tools/flamescapes-gpu-check.mjs
 * --phase spread). The rays' directions come from each cell's own stream of
 * the step (cellStream), as the GPU draws them, so both paths cast the same
 * rays and differ only by float32 round-off.
 */
function radiation(sim, dt) {
  const { P, grid: g } = sim;
  const h = g.h;
  const A = h * h;
  const Ta4 = P.ambient ** 4;
  const dir = [0, 0, 0];
  // One-dimensional walls emit from, and take rays on, the face concerned
  // (wallEmit, march): their ends' deposits wait in lo and hi with the rest.
  const W = sim.wall;
  const dep = {
    T: workArray(sim, "radGas", sim.N, Float64Array).fill(0),
    Ts: workArray(sim, "radSolid", sim.N, Float64Array).fill(0),
    lo: W ? workArray(sim, "radWallLo", W.cells.length, Float64Array).fill(0) : null,
    hi: W ? workArray(sim, "radWallHi", W.cells.length, Float64Array).fill(0) : null,
  };
  const faceE = [0, 0, 0, 0, 0, 0];
  let stream = 0;
  const draw = () => {
    stream = pcgHash(stream);
    return (stream >>> 8) / 16777216;
  };
  for (let c = 0; c < sim.N; c += 1) {
    const kc = sim.kind[c];
    let emitted = 0;
    let fromSolid = false;
    const w = W ? W.index[c] : -1;
    if (w >= 0) {
      emitted = wallEmit(sim, w, c, dt, faceE);
      if (!(emitted > 0)) continue;
      fromSolid = true;
    } else if (kc !== GAS && sim.Ts[c] > P.emitAbove) {
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
    // The cell's own stream of the step (salt 2), the GPU's: both paths cast the same rays.
    stream = cellStream(c, sim.steps, 2, P.seed);
    for (let r = 0; r < P.rays; r += 1) {
      // A uniform direction on the sphere; from a surface, added to the
      // normal of one of its open faces, which gives Lambert's cosine law
      // over that face's hemisphere.
      const zc = 2 * draw() - 1;
      const phi = 2 * Math.PI * draw();
      const s = Math.sqrt(1 - zc * zc);
      dir[0] = s * Math.cos(phi);
      dir[1] = zc;
      dir[2] = s * Math.sin(phi);
      if (faces) {
        let f = 0;
        if (w >= 0) {
          // A one-dimensional wall's face drawn by its share of the emission (its faces differ in temperature).
          let u = draw() * emitted;
          for (const d of faces) {
            if (!(faceE[d] > 0)) continue;
            f = d;
            u -= faceE[d];
            if (u < 0) break;
          }
        } else f = faces[Math.min(Math.floor(draw() * faces.length), faces.length - 1)];
        const n = DIRS[f];
        for (let d = 0; d < 3; d += 1) dir[d] += n[d];
        const len = Math.hypot(dir[0], dir[1], dir[2]) || 1;
        for (let d = 0; d < 3; d += 1) dir[d] /= len;
      }
      march(sim, i + 0.5, j + 0.5, k + 0.5, dir, share, c, fromSolid, dep);
    }
  }
  for (let c = 0; c < sim.N; c += 1) {
    if (dep.T[c]) sim.T[c] += dep.T[c];
    if (dep.Ts[c]) sim.Ts[c] += dep.Ts[c];
  }
  if (W) {
    for (let q = 0; q < W.cells.length; q += 1) {
      if (dep.lo[q]) wallEnd(sim, q, W.cells[q], 0, dep.lo[q]);
      if (dep.hi[q]) wallEnd(sim, q, W.cells[q], 1, dep.hi[q]);
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

/**
 * March one ray cell by cell (Amanatides-Woo), depositing its energy: into
 * `dep` ({T, Ts}, kelvin a cell) when given, to be added later, else at once.
 */
function march(sim, x, y, z, dir, energy, from, skipOwn, dep = null) {
  const { P, grid: g } = sim;
  const gasT = dep ? dep.T : sim.T, solidTs = dep ? dep.Ts : sim.Ts;
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
        gasT[c] += taken / (sim.rho[c] * mixtureCp(sim.Y, c * NS) * sim.V);
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
      const rise = left / Math.max(solidCapacity(sim, next), 1e-6);
      const w = sim.wall ? sim.wall.index[next] : -1;
      if (w >= 0 && a === (sim.wall.meta[w] & 3)) {
        // A one-dimensional wall reached along its normal takes the ray on the
        // face it crossed: its low end when the ray runs up the axis.
        const end = step[a] > 0 ? 0 : 1;
        if (dep) (end ? dep.hi : dep.lo)[w] += rise;
        else wallEnd(sim, w, next, end, rise);
        return;
      }
      solidTs[next] += rise;
      return;
    }
  }
  // Out of reach (or below the cut-off): what the ray still carries is
  // absorbed in the gas cell it ended in, so no energy is lost.
  if (left > 0 && i >= 0 && j >= 0 && k >= 0 && i < g.nx && j < g.ny && k < g.nz) {
    const c = at(g, i, j, k);
    if (sim.kind[c] === GAS) gasT[c] += left / (sim.rho[c] * mixtureCp(sim.Y, c * NS) * sim.V);
    else solidTs[c] += left / Math.max(solidCapacity(sim, c), 1e-6);
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
  // Midpoint (RK2) back-trace in cell units, into `pos` (no allocation), from
  // the velocity (a, b, c) at its start.
  const backFrom = (x, y, z, a, b, c) => {
    const mx = x - 0.5 * s * a, my = y - 0.5 * s * b, mz = z - 0.5 * s * c;
    velocityAt(sim, mx, my, mz, vel, old);
    pos[0] = x - s * vel[0];
    pos[1] = y - s * vel[1];
    pos[2] = z - s * vel[2];
  };
  const back = (x, y, z) => {
    velocityAt(sim, x, y, z, vel, old);
    backFrom(x, y, z, vel[0], vel[1], vel[2]);
  };
  // A back-trace starts at a face or a cell centre, where sampleFace's
  // trilinear weights are 0, 1/2 or 1. Away from the domain's sides (where it
  // clamps) the start's velocity is read here with sampleFace's own products
  // and sums, less the terms it weights 0: the same numbers (but for the sign
  // of an exact zero), three of the seven samples a face takes (2026-11-09).
  const u1 = nx + 1, vk = nx * (ny + 1), wk = nx * ny, uk = (nx + 1) * ny;
  const T = copyOf(sim, "T0", sim.T), Y = copyOf(sim, "Y0", sim.Y);
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i <= nx; i += 1) {
        if (i >= 1 && i <= nx - 1 && j <= ny - 2 && k <= nz - 2) {
          const bv = i - 1 + nx * j + vk * k, bw = i - 1 + nx * j + wk * k;
          const v0 = v[bv] * 0.5 + v[bv + 1] * 0.5, v1 = v[bv + nx] * 0.5 + v[bv + nx + 1] * 0.5;
          const w0 = w[bw] * 0.5 + w[bw + 1] * 0.5, w1 = w[bw + wk] * 0.5 + w[bw + wk + 1] * 0.5;
          backFrom(i, j + 0.5, k + 0.5, u[i + u1 * j + uk * k], v0 * 0.5 + v1 * 0.5, w0 * 0.5 + w1 * 0.5);
        } else back(i, j + 0.5, k + 0.5);
        sim.u[uAt(g, i, j, k)] = sampleFace(u, nx + 1, ny, nz, pos[0], pos[1] - 0.5, pos[2] - 0.5);
      }
    }
  }
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j <= ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        if (i <= nx - 2 && j >= 1 && j <= ny - 1 && k <= nz - 2) {
          const bu = i + u1 * (j - 1) + uk * k, bw = i + nx * (j - 1) + wk * k;
          const u0 = u[bu] * 0.5 + u[bu + 1] * 0.5, uu = u[bu + u1] * 0.5 + u[bu + u1 + 1] * 0.5;
          const wv = (w[bw] * 0.5 + w[bw + nx] * 0.5) * 0.5 + (w[bw + wk] * 0.5 + w[bw + wk + nx] * 0.5) * 0.5;
          backFrom(i + 0.5, j, k + 0.5, u0 * 0.5 + uu * 0.5, v[i + nx * j + vk * k], wv);
        } else back(i + 0.5, j, k + 0.5);
        sim.v[vAt(g, i, j, k)] = sampleFace(v, nx, ny + 1, nz, pos[0] - 0.5, pos[1], pos[2] - 0.5);
      }
    }
  }
  for (let k = 0; k <= nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        if (i <= nx - 2 && j <= ny - 2 && k >= 1 && k <= nz - 1) {
          const bu = i + u1 * j + uk * (k - 1), bv = i + nx * j + vk * (k - 1);
          const u0 = u[bu] * 0.5 + u[bu + 1] * 0.5, uu = u[bu + uk] * 0.5 + u[bu + uk + 1] * 0.5;
          const vv = (v[bv] * 0.5 + v[bv + nx] * 0.5) * 0.5 + (v[bv + vk] * 0.5 + v[bv + vk + nx] * 0.5) * 0.5;
          backFrom(i + 0.5, j + 0.5, k, u0 * 0.5 + uu * 0.5, vv, w[i + nx * j + wk * k]);
        } else back(i + 0.5, j + 0.5, k);
        sim.w[wAt(g, i, j, k)] = sampleFace(w, nx, ny, nz + 1, pos[0] - 0.5, pos[1] - 0.5, pos[2]);
      }
    }
  }
  // The gas itself: conserved (proposal F), or the paper's back-traced T and Y.
  if (sim.P.conservative > 0) {
    transportGas(sim, dt, u, v, w);
    return;
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
        if (i <= nx - 2 && j <= ny - 2 && k <= nz - 2) {
          const bu = i + u1 * j + uk * k, bv = i + nx * j + vk * k, bw = i + nx * j + wk * k;
          backFrom(i + 0.5, j + 0.5, k + 0.5, u[bu] * 0.5 + u[bu + 1] * 0.5, v[bv] * 0.5 + v[bv + nx] * 0.5, w[bw] * 0.5 + w[bw + wk] * 0.5);
        } else back(i + 0.5, j + 0.5, k + 0.5);
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
 * The gas's transport (proposal F): its density and its species' densities
 * move through the faces by the step's starting flow, one sweep along each
 * axis in turn. A face carries its upwind cell's density and mass fractions,
 * plus a van Leer limited slope where the Courant number leaves room (van
 * Leer 1974). Density, not T, is what mixes: gases mix by volume at constant
 * pressure, and averaging T made heat (F6). Each cell takes the flux
 * difference and gives back its own density times the faces' divergence,
 * the advective form d rho / dt + u . grad rho = 0 in fluxes: a uniform gas
 * stays uniform whatever divergence the projection leaves (with T following
 * the density, the fully conservative form turned that remainder into hot
 * and cold spots), the heat sources' own thinning of the gas they heat is
 * not taken twice, and where the flow is divergence-free no mass is made or
 * lost. Outside the domain is still ambient air; the ground and the faces of
 * solids carry nothing. T follows from the density by the gas law.
 */
function transportGas(sim, dt, u, v, w) {
  const { grid: g } = sim;
  const { nx, ny, nz, h } = g;
  const N = sim.N;
  const kind = sim.kind;
  const rho = workArray(sim, "gasRho", N, Float64Array);
  const Y = workArray(sim, "gasY", N * NS, Float64Array);
  const dm = workArray(sim, "gasDm", N, Float64Array);
  const dY = workArray(sim, "gasDY", N * NS, Float64Array);
  const dd = workArray(sim, "gasDiv", N, Float64Array);
  // The densities at the start: each sweep gives back these times its faces'
  // divergence, so that over the three sweeps the divergence-free part of the
  // flow gives back nothing (the current density would leave a splitting
  // error, 5 % of the heat in the sealed box's mixing).
  const rho0 = workArray(sim, "gasRho0", N, Float64Array);
  const ry0 = workArray(sim, "gasRY0", N * NS, Float64Array);
  for (let c = 0; c < N; c += 1) {
    if (kind[c] === SOLID) continue;
    rho[c] = rho0[c] = gasDensity(sim.T[c], sim.Y, c * NS) * zonePressure(sim, c);
    for (let q = 0; q < NS; q += 1) {
      Y[c * NS + q] = sim.Y[c * NS + q];
      ry0[c * NS + q] = rho[c] * Y[c * NS + q];
    }
  }
  const rhoA = sim.rho0;
  const yA = new Float64Array(NS);
  yA[SPECIES.O2] = Y_O2_AIR;
  const s = dt / h;
  const fy = new Float64Array(NS);
  // One sweep along an axis: n cells along it, stride `step` between them,
  // `lines` the first cells of the lines, `vel(line, a)` the flow through the
  // face before cell a of the line (a from 0 to n).
  const sweep = (n, stride, lines, vel, ground) => {
    dm.fill(0);
    dY.fill(0);
    dd.fill(0);
    for (let l = 0; l < lines.length; l += 1) {
      const base = lines[l];
      // A cell along the line: -1 outside the domain (ambient air), -2 a solid.
      const cellAt = (a) => (a < 0 || a >= n ? -1 : kind[base + a * stride] === SOLID ? -2 : base + a * stride);
      const r = (c) => (c === -1 ? rhoA : rho[c]);
      for (let a = 0; a <= n; a += 1) {
        if (a === 0 && ground) continue;
        const vf = vel(l, a);
        if (vf === 0) continue;
        const from = vf > 0 ? cellAt(a - 1) : cellAt(a);
        const to = vf > 0 ? cellAt(a) : cellAt(a - 1);
        if (from === -2 || to === -2) continue;
        const back = vf > 0 ? cellAt(a - 2) : cellAt(a + 1);
        const courant = Math.min(1, Math.abs(vf) * s);
        // Faces moving more than a cell a step (the cut steps' cap of 8 can leave some on fine grids): counted, so a budget can see them.
        if (Math.abs(vf) * s > 1) sim.stats.courantClamped += 1;
        // The face's flow out of the cell before it and into the cell after.
        const out = vf > 0 ? courant : -courant;
        const left = cellAt(a - 1), right = cellAt(a);
        if (left >= 0) dd[left] += out;
        if (right >= 0) dd[right] -= out;
        // The face's density: the donor's, with a limited slope towards the acceptor.
        const rd = r(from), ra = r(to);
        let rf = rd;
        if (back !== -2) {
          const dr = ra - rd;
          if (dr !== 0) {
            const q = (rd - r(back)) / dr;
            rf += 0.5 * ((q + Math.abs(q)) / (1 + Math.abs(q))) * (1 - courant) * dr;
          }
        }
        const m = courant * rf; // mass through the face, per unit cell volume
        for (let q = 0; q < NS; q += 1) {
          const yd = from === -1 ? yA[q] : Y[from * NS + q];
          const ya = to === -1 ? yA[q] : Y[to * NS + q];
          let yf = yd;
          if (back !== -2) {
            const dy = ya - yd;
            if (dy !== 0) {
              const yb = back === -1 ? yA[q] : Y[back * NS + q];
              const t = (yd - yb) / dy;
              yf += 0.5 * ((t + Math.abs(t)) / (1 + Math.abs(t))) * (1 - courant) * dy;
            }
          }
          fy[q] = m * yf;
        }
        if (from >= 0) {
          dm[from] -= m;
          for (let q = 0; q < NS; q += 1) dY[from * NS + q] -= fy[q];
        }
        if (to >= 0) {
          dm[to] += m;
          for (let q = 0; q < NS; q += 1) dY[to * NS + q] += fy[q];
        }
      }
    }
    for (let c = 0; c < N; c += 1) {
      if (kind[c] === SOLID) continue;
      const before = rho[c];
      let after = before + dm[c] + rho0[c] * dd[c];
      // A cell emptied faster than it fills (two faces out at Courant
      // numbers summing past 1): kept at a hundredth of ambient, the mass
      // added counted, so that a budget can see it.
      if (after < 0.01 * rhoA) {
        sim.stats.massFloor += (0.01 * rhoA - after) * sim.V;
        after = 0.01 * rhoA;
      }
      for (let q = 0; q < NS; q += 1) {
        const y = Y[c * NS + q];
        Y[c * NS + q] = Math.min(1, Math.max(0, (before * y + ry0[c * NS + q] * dd[c] + dY[c * NS + q]) / after));
      }
      rho[c] = after;
    }
  };
  const xLines = [], yLines = [], zLines = [];
  for (let k = 0; k < nz; k += 1) for (let j = 0; j < ny; j += 1) xLines.push(nx * (j + ny * k));
  for (let k = 0; k < nz; k += 1) for (let i = 0; i < nx; i += 1) yLines.push(i + nx * ny * k);
  for (let j = 0; j < ny; j += 1) for (let i = 0; i < nx; i += 1) zLines.push(i + nx * j);
  const sweepX = () => sweep(nx, 1, xLines, (l, a) => u[a + (nx + 1) * l], false);
  const sweepY = () =>
    sweep(ny, nx, yLines, (l, a) => {
      const i = l % nx, k = Math.floor(l / nx);
      return v[i + nx * (a + (ny + 1) * k)];
    }, true);
  const sweepZ = () => sweep(nz, nx * ny, zLines, (l, a) => w[l + nx * ny * a], false);
  // x, y, z on even steps and z, y, x on odd ones: a fixed order favoured the
  // first axis swept (15 % in a ceiling jet's rise, review 2026-11-12).
  if (sim.steps % 2 === 0) {
    sweepX();
    sweepY();
    sweepZ();
  } else {
    sweepZ();
    sweepY();
    sweepX();
  }
  for (let c = 0; c < N; c += 1) {
    if (kind[c] === SOLID) continue;
    for (let q = 0; q < NS; q += 1) sim.Y[c * NS + q] = Y[c * NS + q];
    sim.T[c] = (gasDensity(1, sim.Y, c * NS) * zonePressure(sim, c)) / rho[c];
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
        // In a pressure zone, against the zone's own ambient density (its gas at its pressure and the ambient temperature).
        sim.v[vAt(g, i, j, k)] += dt * P.gravity * ((sim.rho0 * zonePressure(sim, a)) / r - 1);
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
        if (dPy > 0) sim.stats.sootReleased += dPy * (1 - P.pyrolysisFuel);
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

/**
 * A run's health (wiki/research/fire-calibration-plan-2026-11-11.md, proposal
 * A): a guard, not physics. A run is stopped when its gas holds a non-finite
 * or non-positive value, is hotter than maxT (K, well above any hydrocarbon's
 * adiabatic flame temperature in air), moves faster than maxSpeed (m/s, far
 * above a building fire's gas), or holds more heat above ambient than
 * energyFactor times what the fire has released plus energySlack (J): the
 * calibration's diverging runs reached 6,400 K and 10^4 m/s within seconds.
 * The gas's heat is read at a fixed cp, the same on both paths.
 */
export const HEALTH = Object.freeze({ maxT: 3000, maxSpeed: 50, energyFactor: 1.5, energySlack: 1e6, cp: 1005 });

/**
 * One frame's health readings over the gas cells of a state {T, rho, u, v, w,
 * kind} (the solver itself, or fields read back from the GPU): the hottest
 * gas (K), the fastest cell-centred speed (m/s), the gas's heat above ambient
 * (J), whether any value is non-finite or non-positive, and the cell that
 * most fails (the first non-finite one, else the hottest or fastest).
 */
export function healthReadings(state, grid, P, V = grid.h ** 3) {
  const { nx, ny, nz } = grid;
  const { T, rho, u, v, w, kind } = state;
  const Ta = P.ambient;
  let maxT = Ta, maxSpeed = 0, gasHeat = 0, nonFinite = false, hotCell = -1, fastCell = -1, badCell = -1;
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const c = at(grid, i, j, k);
        if (kind[c] === SOLID) continue;
        const a = 0.5 * (u[uAt(grid, i, j, k)] + u[uAt(grid, i + 1, j, k)]);
        const b = 0.5 * (v[vAt(grid, i, j, k)] + v[vAt(grid, i, j + 1, k)]);
        const d = 0.5 * (w[wAt(grid, i, j, k)] + w[wAt(grid, i, j, k + 1)]);
        const s = Math.hypot(a, b, d);
        const t = T[c], r = rho[c];
        if (!(Number.isFinite(t) && Number.isFinite(r) && Number.isFinite(s)) || !(r > 0)) {
          if (!nonFinite) badCell = c;
          nonFinite = true;
          continue;
        }
        if (t > maxT) [maxT, hotCell] = [t, c];
        if (s > maxSpeed) [maxSpeed, fastCell] = [s, c];
        if (t > Ta) gasHeat += r * V * HEALTH.cp * (t - Ta);
      }
    }
  }
  return { maxT, maxSpeed, gasHeat, nonFinite, badCell, hotCell, fastCell };
}

/**
 * The verdict on a frame's readings, given the heat put into the run so far
 * (J): what the fire released and what a pilot added (stats.released +
 * stats.pilotHeat). {ok: true} or {ok: false, reason, value, cell}.
 */
export function healthVerdict(r, released) {
  if (r.nonFinite) return { ok: false, reason: "non-finite", value: null, cell: r.badCell };
  if (r.maxT > HEALTH.maxT) return { ok: false, reason: "temperature", value: r.maxT, cell: r.hotCell };
  if (r.maxSpeed > HEALTH.maxSpeed) return { ok: false, reason: "speed", value: r.maxSpeed, cell: r.fastCell };
  if (r.gasHeat > HEALTH.energyFactor * Math.max(0, released) + HEALTH.energySlack) return { ok: false, reason: "energy", value: r.gasHeat, cell: r.hotCell };
  return { ok: true };
}

/**
 * The steps a frame's time step dt is cut into (proposal B), from the fastest
 * gas of the frame before: the flow moves at most `cfl` cells a step, and the
 * explicit confinement force adds at most about the flow's own speed a step
 * (vorticity times the step at most `vorticityStep`: the force is vorticity
 * h |omega| per unit mass and |omega| is up to about twice the speed over h);
 * at most `maxSubsteps`, past which the health guard stops a runaway. The
 * same function on both paths, from the same frame reading, so the CPU and
 * the GPU cut their steps alike.
 */
export function substepsFor(P, h, maxSpeed, dt) {
  const byFlow = P.cfl > 0 && maxSpeed > 0 ? Math.ceil((maxSpeed * dt) / (P.cfl * h) - 1e-6) : 1;
  const byForce = P.vorticity > 0 && P.vorticityStep > 0 ? Math.ceil((P.vorticity * dt) / P.vorticityStep - 1e-6) : 1;
  return Math.max(1, Math.min(P.maxSubsteps || 1, Math.max(byFlow, byForce)));
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
  // A sealed region's gas gains or loses only what its cells' species do (pressureZones).
  const Z = sim.zones;
  if (Z && Z.cell[c] >= 0) Z.source[Z.cell[c]] += total - mass;
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
  sim.stats.sootReleased += st.soot * dm;
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
  // A one-dimensional wall exchanges through its own profile (walls), and
  // with no solid beside it, as FDS's walls conduct normal to their faces only.
  const wall = sim.wall ? sim.wall.index : null;
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const c = at(g, i, j, k);
        const kc = sim.kind[c];
        if (kc === GAS || (wall && wall[c] >= 0)) continue;
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
          if (wall && wall[n] >= 0) continue;
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

// ---------------------------------------------------- one-dimensional walls

/** Most nodes a wall's profile may have (the GPU's per-thread arrays hold this many). */
export const WALL_NODES_MAX = 24;

/**
 * A layered wall (calibration plan 13.4, item 1) as `nodes` nodes across its
 * thickness, from its first layer's exposed face to its last layer's back:
 * the vertex-centred finite volumes of FDS's solid phase (FDS Technical
 * Reference Guide, chapter 7). Every layer has at least one interval; the rest
 * go to the layers by their thermal thickness L / sqrt(alpha), the largest
 * remainders first. Within a layer the intervals grow by `stretch` from each of
 * its two faces towards its middle (FDS stretches from both faces too), so the
 * steep early profile under a heated face is resolved. Per unit area: node
 * capacities C (J/(m^2 K), half of each interval beside the node) and the
 * conductances G between neighbouring nodes (k / interval, W/(m^2 K)); heat,
 * their sum; x, the nodes' depths.
 */
export function wallStack(layers, nodes = DEFAULTS.wallNodes, stretch = DEFAULTS.wallStretch) {
  const n = Math.min(WALL_NODES_MAX, Math.max(layers.length + 1, nodes | 0));
  const spare = n - 1 - layers.length;
  const weight = layers.map((l) => l.thickness / Math.sqrt(l.k / (l.rho * l.c)));
  const total = weight.reduce((a, b) => a + b, 0);
  const want = weight.map((w) => (spare * w) / total);
  const intervals = want.map((w) => 1 + Math.floor(w));
  let left = n - 1 - intervals.reduce((a, b) => a + b, 0);
  for (const [, l] of want.map((w, l) => [w - Math.floor(w), l]).sort((a, b) => b[0] - a[0] || a[1] - b[1])) {
    if (left <= 0) break;
    intervals[l] += 1;
    left -= 1;
  }
  const C = new Float64Array(n), G = new Float64Array(n - 1), x = new Float64Array(n);
  let q = 0;
  layers.forEach((l, li) => {
    const m = intervals[li];
    let sum = 0;
    for (let i = 0; i < m; i += 1) sum += stretch ** Math.min(i, m - 1 - i);
    for (let i = 0; i < m; i += 1) {
      const d = (l.thickness * stretch ** Math.min(i, m - 1 - i)) / sum;
      G[q] = l.k / d;
      C[q] += (l.rho * l.c * d) / 2;
      C[q + 1] += (l.rho * l.c * d) / 2;
      x[q + 1] = x[q] + d;
      q += 1;
    }
  });
  return { n, C, G, x, heat: C.reduce((a, b) => a + b, 0), thickness: x[n - 1] };
}

const wallCp = new Float64Array(WALL_NODES_MAX);
const wallDp = new Float64Array(WALL_NODES_MAX);

/**
 * One step of a wall's profile: nodes T[o .. o + n) (in place), capacities
 * C[co ..] and conductances G[go ..] per unit area, by backward Euler, so that
 * a thin node next to a conductive one is stable at any step. Each end
 * exchanges with a gas at Tlo or Thi through hLo or hHi (W/(m^2 K); 0, an
 * insulated end), the gas held at its temperature for the step. The Thomas
 * algorithm solves the tridiagonal system for the nodes' changes (the delta
 * form): its right-hand side is made of temperature differences, so a wall
 * level with its gas stays exactly level in float32 too (solved for the
 * temperatures themselves, the GPU's nodes drifted by two ulps a step).
 * `out` gets the heat each end gave its gas over the step, J/m^2: the wall's
 * own heat changes by exactly minus their sum.
 */
export function wallStep(T, o, n, C, co, G, go, dt, hLo, Tlo, hHi, Thi, out) {
  for (let i = 0; i < n; i += 1) {
    const a = i > 0 ? -G[go + i - 1] : 0;
    const c = i < n - 1 ? -G[go + i] : 0;
    const t = T[o + i];
    let b = C[co + i] / dt - a - c;
    let d = (i > 0 ? a * (t - T[o + i - 1]) : 0) + (i < n - 1 ? c * (t - T[o + i + 1]) : 0);
    if (i === 0) {
      b += hLo;
      d += hLo * (Tlo - t);
    }
    if (i === n - 1) {
      b += hHi;
      d += hHi * (Thi - t);
    }
    const m = i > 0 ? b - a * wallCp[i - 1] : b;
    wallCp[i] = c / m;
    wallDp[i] = i > 0 ? (d - a * wallDp[i - 1]) / m : d / m;
  }
  let change = wallDp[n - 1];
  const hi = T[o + n - 1] + change;
  T[o + n - 1] = hi;
  let lo = hi;
  for (let i = n - 2; i >= 0; i -= 1) {
    change = wallDp[i] - wallCp[i] * change;
    lo = T[o + i] + change;
    T[o + i] = lo;
  }
  out[0] = hLo * (lo - Tlo) * dt;
  out[1] = hHi * (hi - Thi) * dt;
  return out;
}

/** A stack with its nodes in the opposite order (its last layer's back at node 0). */
function reversed(s) {
  return { ...s, C: s.C.slice().reverse(), G: s.G.slice().reverse(), x: s.x.map((v) => s.thickness - v).reverse() };
}

/**
 * The one-dimensional walls of a scene: the solid cells whose largest inert
 * material lists its layers and that are one cell thick along some axis (no
 * solid on either side along it; the domain's side counts as open), that
 * axis the wall's normal (the most gas neighbours, then the lowest axis).
 * The first layer faces the grid's interior along the normal (the room, in
 * a validation scene's compartment). Each holds its stack's real heat
 * capacity (inertC, J/K); its nodes start at ambient. Per wall: its cell, its
 * axis and stack (meta = axis + 4 stack), its nodes, the mean temperature its
 * nodes had when the walls stage last wrote it (last: what other stages add
 * to Ts since is theirs), and the heat each of its six faces gave its gas in
 * the last walls stage (Q, J). Per stack (two per material, one reversed):
 * C, G (float32, the GPU's) and its heat and thickness.
 */
function wallSetup(grid, fraction, solid, inert, inertC, P) {
  const { nx, ny, nz, h } = grid;
  const N = nx * ny * nz;
  // A material's layers that can be a profile: of positive thickness and
  // properties, no more than the nodes allow. Any other stays lumped.
  const usable = inert.map((m) => {
    const layers = (m.layers || []).filter((l) => l.thickness > 0);
    const ok = layers.length && layers.length < WALL_NODES_MAX && layers.every((l) => l.k > 0 && l.rho > 0 && l.c > 0);
    return ok ? layers : null;
  });
  // Every wall has the same number of nodes (the GPU's layout): enough for the most layered stack.
  const n = Math.min(WALL_NODES_MAX, Math.max(2, P.wallNodes | 0, ...usable.map((l) => (l ? l.length + 1 : 0))));
  const stacks = [];
  const stackOf = usable.map((layers) => {
    if (!layers) return -1;
    const s = wallStack(layers, n, P.wallStretch);
    stacks.push(s, reversed(s));
    return stacks.length - 2;
  });
  const dims = [nx, ny, nz], strides = [1, nx, nx * ny];
  const index = new Int32Array(N).fill(-1);
  const cells = [], meta = [];
  for (let c = 0; c < N; c += 1) {
    if (!solid[c]) continue;
    let best = -1;
    for (let m = 0; m < inert.length; m += 1) if (fraction[c * MATERIALS + m] > 0 && (best < 0 || fraction[c * MATERIALS + m] > fraction[c * MATERIALS + best])) best = m;
    if (best < 0 || stackOf[best] < 0) continue;
    const p = [c % nx, Math.floor(c / nx) % ny, Math.floor(c / (nx * ny))];
    let axis = -1, most = -1;
    for (let a = 0; a < 3; a += 1) {
      const lo = p[a] > 0, hi = p[a] < dims[a] - 1;
      if ((lo && solid[c - strides[a]]) || (hi && solid[c + strides[a]])) continue;
      const gas = (lo ? 1 : 0) + (hi ? 1 : 0);
      if (gas > most) {
        most = gas;
        axis = a;
      }
    }
    if (axis < 0) continue;
    // In the lower half along the normal the interior is up the axis: the first layer at the high end.
    const s = stackOf[best] + (2 * p[axis] + 1 < dims[axis] ? 1 : 0);
    index[c] = cells.length;
    cells.push(c);
    meta.push(axis + 4 * s);
  }
  const W = cells.length;
  // No wall one cell thick (the walls are thicker than the cells): the walls stay lumped, at no cost.
  if (!W) return null;
  const C = new Float32Array(stacks.length * n), G = new Float32Array(stacks.length * (n - 1));
  stacks.forEach((s, k) => {
    C.set(s.C, k * n);
    G.set(s.G, k * (n - 1));
  });
  // Each stack's heat from its float32 capacities, as both paths sum them.
  const heat = Float32Array.from(stacks, (s, k) => C.subarray(k * n, (k + 1) * n).reduce((a, b) => a + b, 0));
  for (let w = 0; w < W; w += 1) inertC[cells[w]] = heat[meta[w] >> 2] * h * h;
  return {
    n,
    cells: Int32Array.from(cells),
    index,
    meta: Uint32Array.from(meta),
    T: new Float32Array(W * n).fill(P.ambient),
    last: new Float32Array(W).fill(P.ambient),
    Q: new Float32Array(W * 6),
    C,
    G,
    heat,
    thickness: Float32Array.from(stacks, (s) => s.thickness),
  };
}

/** Wall w's area on a side face (not on its normal): its edge, its thickness by the cell, at most the face. */
function wallSide(sim, w) {
  const h = sim.grid.h;
  return Math.min(sim.wall.thickness[sim.wall.meta[w] >> 2], h) * h;
}

/**
 * Heat into one end of wall w (in cell c), as the rise k (K) it would give
 * the whole cell: its end node takes it, and the cell's mean temperature
 * (Ts) and `last` rise by k alike, so the walls stage does not spread it again.
 */
function wallEnd(sim, w, c, end, k) {
  const W = sim.wall, n = W.n, s = W.meta[w] >> 2;
  const q = end ? n - 1 : 0;
  W.T[w * n + q] += (k * W.heat[s]) / W.C[s * n + q];
  sim.Ts[c] += k;
  W.last[w] += k;
}

/**
 * Radiation leaving wall w (cell c) over its faces open to gas or the
 * domain's side (Eq. 16 per face): an end face at its end node's temperature,
 * a side face at the cell's mean over its edge's area (wallSide), each above
 * emitAbove, never more than half its node's (or the cell's) heat above
 * ambient. `faceE` (six, by DIRS) gets each face's energy; the energy leaves
 * the nodes it came from. Returns the total.
 */
function wallEmit(sim, w, c, dt, faceE) {
  const { P, grid: g } = sim;
  const W = sim.wall, n = W.n;
  const o = w * n;
  // Cool walls (most of them, most of the time) cast nothing.
  if (!(W.T[o] > P.emitAbove || W.T[o + n - 1] > P.emitAbove || sim.Ts[c] > P.emitAbove)) return 0;
  const A = g.h * g.h, side = wallSide(sim, w);
  const axis = W.meta[w] & 3, s = W.meta[w] >> 2;
  const total = W.heat[s] * A;
  const Ta = P.ambient, Ta4 = Ta ** 4;
  const i = c % g.nx, j = Math.floor(c / g.nx) % g.ny, k = Math.floor(c / (g.nx * g.ny));
  let all = 0, sides = 0;
  for (let d = 0; d < 6; d += 1) {
    faceE[d] = 0;
    const dir = DIRS[d];
    const ii = i + dir[0], jj = j + dir[1], kk = k + dir[2];
    if (!(ii < 0 || jj < 0 || kk < 0 || ii >= g.nx || jj >= g.ny || kk >= g.nz || sim.kind[at(g, ii, jj, kk)] === GAS)) continue;
    const end = d >> 1 === axis;
    // An odd direction is down the axis: the end at node 0.
    const q = d & 1 ? 0 : n - 1;
    const Tf = end ? W.T[o + q] : sim.Ts[c];
    if (!(Tf > P.emitAbove)) continue;
    const cap = end ? W.C[s * n + q] * A : total;
    let emitted = sim.eps[c] * SIGMA * (Tf ** 4 - Ta4) * (end ? A : side) * dt;
    emitted = Math.min(emitted, 0.5 * cap * (Tf - Ta));
    if (!(emitted > 0)) continue;
    faceE[d] = emitted;
    all += emitted;
    if (end) W.T[o + q] -= emitted / cap;
    else sides += emitted;
  }
  if (!(all > 0)) return 0;
  if (sides > 0) for (let q = 0; q < n; q += 1) W.T[o + q] -= sides / total;
  sim.Ts[c] -= all / total;
  W.last[w] -= all / total;
  return all;
}

/**
 * One-dimensional walls (calibration plan 13.4, item 1), after conduct. Each
 * wall first takes into its nodes, evenly, what other stages added to its
 * cell's mean since its last step (embers, rays through its side faces); then
 * its side faces exchange its mean with the gas beside them over their edge's
 * area, exactly as conduct exchanges a lumped wall; then its profile steps
 * (wallStep) with each end exchanging with the gas beyond it (insulated
 * against the domain's side), the gas held at its temperature at the stage's
 * start. Every wall reads the gas before any is changed, and each gas cell
 * then takes what its wall neighbours gave it, gathered in direction order (as
 * the GPU gathers it), so no wall depends on another's order and the heat each
 * wall loses is the heat its gas gains.
 */
export function walls(sim, dt) {
  const W = sim.wall;
  if (!W) return;
  const { P, grid: g } = sim;
  const { nx, ny, h } = g;
  const A = h * h, hc = P.wallConvection, n = W.n;
  const out = wallOut;
  for (let w = 0; w < W.cells.length; w += 1) {
    const c = W.cells[w];
    const axis = W.meta[w] & 3, s = W.meta[w] >> 2;
    const o = w * n, co = s * n, total = W.heat[s] * A;
    const added = sim.Ts[c] - W.last[w];
    if (added !== 0) for (let q = 0; q < n; q += 1) W.T[o + q] += added;
    let mean = 0;
    for (let q = 0; q < n; q += 1) mean += W.C[co + q] * W.T[o + q];
    mean = (mean * A) / total;
    const i = c % nx, j = Math.floor(c / nx) % ny, k = Math.floor(c / (nx * ny));
    let sides = 0;
    const side = Math.min(W.thickness[s], h) * h;
    for (let d = 0; d < 6; d += 1) {
      W.Q[w * 6 + d] = 0;
      if (d >> 1 === axis) continue;
      const nb = openNeighbour(sim, i, j, k, d);
      if (nb < 0) continue;
      const q = exchange(mean, sim.T[nb], total, sim.rho[nb] * mixtureCp(sim.Y, nb * NS) * sim.V, hc * side, dt);
      W.Q[w * 6 + d] = q;
      sides += q;
    }
    if (sides !== 0) for (let q = 0; q < n; q += 1) W.T[o + q] -= sides / total;
    // Its ends: down the axis (an odd direction) at node 0, up it at node n - 1.
    const lo = openNeighbour(sim, i, j, k, 2 * axis + 1), hi = openNeighbour(sim, i, j, k, 2 * axis);
    wallStep(W.T, o, n, W.C, co, W.G, s * (n - 1), dt, lo >= 0 ? hc : 0, lo >= 0 ? sim.T[lo] : 0, hi >= 0 ? hc : 0, hi >= 0 ? sim.T[hi] : 0, out);
    W.Q[w * 6 + 2 * axis + 1] = out[0] * A;
    W.Q[w * 6 + 2 * axis] = out[1] * A;
    mean = 0;
    for (let q = 0; q < n; q += 1) mean += W.C[co + q] * W.T[o + q];
    sim.Ts[c] = (mean * A) / total;
    W.last[w] = sim.Ts[c];
  }
  // Each open cell beside a wall takes what the wall's faces towards it gave
  // (summed in float64 first: a cell between two walls takes both at once).
  const heat = workArray(sim, "wallGas", sim.N, Float64Array);
  for (let pass = 0; pass < 2; pass += 1) {
    for (let w = 0; w < W.cells.length; w += 1) {
      const c = W.cells[w];
      for (let d = 0; d < 6; d += 1) {
        const q = W.Q[w * 6 + d];
        if (q === 0) continue;
        const nb = c + (d === 0 ? 1 : d === 1 ? -1 : d === 2 ? nx : d === 3 ? -nx : d === 4 ? nx * ny : -nx * ny);
        if (pass === 0) heat[nb] += q;
        else if (heat[nb] !== 0) {
          sim.T[nb] += heat[nb] / (sim.rho[nb] * mixtureCp(sim.Y, nb * NS) * sim.V);
          heat[nb] = 0;
        }
      }
    }
  }
}

const wallOut = [0, 0];

// ------------------------------------------------------ soot deposition

/** Air's dynamic viscosity, Pa s: Sutherland's law (1.716e-5 Pa s at 273.15 K, S = 110.4 K). */
export function viscosity(T) {
  return 1.716e-5 * (T / 273.15) ** 1.5 * ((273.15 + 110.4) / (T + 110.4));
}

/**
 * A particle's Cunningham slip factor at gas temperature T (Floyd, Overholt
 * and Ezekoye 2014, eqs. 2 and 3): Cn = 1 + 1.25 Kn + 0.41 Kn exp(-0.88 / Kn),
 * Kn = (lambda / r) (T / T_inf), the gas's mean free path lambda 0.065 um at 25 C.
 */
export function slipFactor(d, T) {
  const Kn = (0.065e-6 / (d / 2)) * (T / 298.15);
  return { Kn, Cn: 1 + 1.25 * Kn + 0.41 * Kn * Math.exp(-0.88 / Kn) };
}

/**
 * Soot's thermophoretic velocity towards a wall at Tw from gas at Tg (m/s;
 * 0 from gas no hotter than the wall): Talbot, Cheng, Schefer and Willis's
 * (1980) u = K_th nu (dT/dx) / Tg with K_th = 2 Cs (a + Ct Kn) Cn / ((1 + 3
 * Cm Kn)(1 + 2 a + 2 Ct Kn)), Cs 1.17, Ct 2.18, Cm 1.14, a = kg / kp; and
 * FDS's wall gradient from the wall's heat transfer coefficient, dT/dx =
 * h (Tg - Tw) / kg, the gradient a coarse grid cannot resolve (Floyd et al.
 * eqs. 4 to 6). The gas's conductivity from its viscosity, kg = mu cp / 0.71.
 */
export function thermophoreticVelocity(Tg, Tw, rhoGas, cp, h, P) {
  if (!(Tg > Tw) || !(h > 0)) return 0;
  const mu = viscosity(Tg), kg = (mu * cp) / 0.71;
  const { Kn, Cn } = slipFactor(P.sootThermophoreticDiameter, Tg);
  const a = kg / P.sootConductivity;
  const K = (2 * 1.17 * (a + 2.18 * Kn) * Cn) / ((1 + 3 * 1.14 * Kn) * (1 + 2 * a + 2 * 2.18 * Kn));
  return ((K * mu) / (rhoGas * Tg)) * ((h * (Tg - Tw)) / kg);
}

/** Soot's settling velocity in gas at T (m/s): Stokes's with the slip factor, rho_p d^2 g Cn / (18 mu) (Floyd et al. eq. 1, spheres). */
export function settlingVelocity(T, P) {
  const d = P.sootDiameter;
  return (P.sootDensity * d * d * Math.abs(P.gravity) * slipFactor(d, T).Cn) / (18 * viscosity(T));
}

/**
 * Soot deposition (calibration plan 13.4, item 2), after the walls: each open
 * cell loses soot to every solid face beside it at the deposition velocity
 * of that face, thermophoretic towards a cooler surface (its surface
 * temperature: a one-dimensional wall's end node facing the cell, else the
 * cell's Ts) plus gravitational settling onto the solid below; FDS's
 * velocities, summed per face as FDS sums them (Floyd et al.; turbulent
 * deposition, also FDS's, is not modelled). Over the step the cell's soot
 * falls as exp(-sum(u) dt / h), all its faces h^2 over its volume h^3 (the
 * exact solution of FDS's explicit removal, as every rate here is
 * integrated). What leaves the gas is the cell's sootDeposit and
 * stats.sootDeposited, so the soot released equals that in the gas, on the
 * surfaces and gone out of the domain. Gravitational settling within the gas
 * (FDS's drift in the species transport) is not modelled: 1 um soot settles
 * at about 0.1 mm/s.
 */
export function deposition(sim, dt) {
  const { P, grid: g } = sim;
  if (!(P.sootDeposition > 0) || !sim.sootDeposit) return;
  const { nx, ny, h } = g;
  const kind = sim.kind;
  const hc = P.wallConvection;
  // A cell with all six neighbours open (most of them) has no surface.
  const mask = topology(sim).mask;
  for (let c = 0; c < sim.N; c += 1) {
    if (kind[c] === SOLID || mask[c] === 63) continue;
    const ys = sim.Y[c * NS + SPECIES.SOOT];
    if (!(ys > 0)) continue;
    const i = c % nx, j = Math.floor(c / nx) % ny, k = Math.floor(c / (nx * ny));
    let rate = 0;
    for (let d = 0; d < 6; d += 1) {
      const n = solidNeighbour(sim, i, j, k, d);
      if (n < 0) continue;
      rate += thermophoreticVelocity(sim.T[c], surfaceTemperature(sim, n, d), sim.rho[c], mixtureCp(sim.Y, c * NS), hc, P);
      // Settling onto the solid below (DIRS[3] is down).
      if (d === 3) rate += settlingVelocity(sim.T[c], P);
    }
    if (!(rate > 0)) continue;
    const dm = sim.rho[c] * ys * sim.V * -Math.expm1((-rate * dt) / h);
    delta.fill(0);
    delta[SPECIES.SOOT] = -dm;
    changeGas(sim, c, delta);
    sim.sootDeposit[c] += dm;
    sim.stats.sootDeposited += dm;
  }
}

/** The solid neighbour of cell (i, j, k) in direction d, or -1. */
function solidNeighbour(sim, i, j, k, d) {
  const g = sim.grid;
  const dir = DIRS[d];
  const ii = i + dir[0], jj = j + dir[1], kk = k + dir[2];
  if (ii < 0 || jj < 0 || kk < 0 || ii >= g.nx || jj >= g.ny || kk >= g.nz) return -1;
  const n = at(g, ii, jj, kk);
  return sim.kind[n] === SOLID ? n : -1;
}

/**
 * The temperature of solid n's face towards the cell in direction d from it:
 * a one-dimensional wall's end node when the face is on its normal (the cell
 * up the axis faces its low end), else n's Ts.
 */
function surfaceTemperature(sim, n, d) {
  const W = sim.wall;
  const w = W ? W.index[n] : -1;
  if (w < 0 || d >> 1 !== (W.meta[w] & 3)) return sim.Ts[n];
  // From the cell, an even direction is up the axis: the wall is above it, the cell faces its low end (node 0).
  return W.T[w * W.n + (d & 1 ? W.n - 1 : 0)];
}

/** The open (not solid) neighbour of cell (i, j, k) in direction d, or -1 (a solid, or outside the domain). */
function openNeighbour(sim, i, j, k, d) {
  const g = sim.grid;
  const dir = DIRS[d];
  const ii = i + dir[0], jj = j + dir[1], kk = k + dir[2];
  if (ii < 0 || jj < 0 || kk < 0 || ii >= g.nx || jj >= g.ny || kk >= g.nz) return -1;
  const nb = at(g, ii, jj, kk);
  return sim.kind[nb] === SOLID ? -1 : nb;
}

/**
 * The gas's expansion (proposal F): at constant pressure a cell whose gas is
 * heated from T0 to T1 swells by ln(T1/T0) of its volume, and that volume
 * must leave it. Each step's heat sources are read as the change of T they
 * make, apart from the advection's: those before it (the pilot or burner,
 * particles, radiation) at the step's start, and those after it (diffusion,
 * ducts, chemistry, spray, conduction) since the advection. The projection
 * then asks the flow for that divergence. The domain's shell is held at
 * ambient and expands nothing.
 */
function expansionStart(sim) {
  if (sim.P.expansion > 0) copyOf(sim, "expandT0", sim.T);
}

function expansionBeforeAdvect(sim) {
  if (!(sim.P.expansion > 0)) return;
  const D = workArray(sim, "expand", sim.N, Float64Array);
  const T0 = sim.work.expandT0;
  for (let c = 0; c < sim.N; c += 1) D[c] = sim.kind[c] === SOLID ? 0 : Math.log(sim.T[c] / T0[c]);
}

function expansionAfterAdvect(sim) {
  if (sim.P.expansion > 0) copyOf(sim, "expandT1", sim.T);
}

function expansionEnd(sim, dt) {
  if (!(sim.P.expansion > 0)) return;
  const { nx, ny, nz } = sim.grid;
  const D = sim.work.expand, T1 = sim.work.expandT1;
  for (let c = 0; c < sim.N; c += 1) {
    if (sim.kind[c] === SOLID) continue;
    const i = c % nx, j = Math.floor(c / nx) % ny, k = Math.floor(c / (nx * ny));
    if (i === 0 || k === 0 || i === nx - 1 || k === nz - 1 || j === ny - 1) D[c] = 0;
    else D[c] = (D[c] + Math.log(sim.T[c] / T1[c])) / dt;
  }
}

/**
 * Line 31: the equation of state for every open cell, a pressure zone's at
 * its pressure: after the heat sources (resolve) each zone's pressure is
 * found again from its gas (pressureZones); after the advection, zoneTransport.
 */
function updateDensity(sim, resolve = true) {
  for (let c = 0; c < sim.N; c += 1) {
    if (sim.kind[c] === SOLID) continue;
    sim.rho[c] = gasDensity(sim.T[c], sim.Y, c * NS);
  }
  if (resolve) pressureZones(sim);
  else zoneTransport(sim);
}

/**
 * A pressure zone after the advection. Its T is the gas law's at the zone's
 * pressure already, but the transport does not keep a zone's mass exactly
 * (in the heat-closure mix box it lost 0.9 % in 10 s, its volume kept): the
 * zone's densities and pressure are scaled back to its mass at the same T,
 * so the transport's error is not taken as the zone's gas being compressed
 * (which heated the mix box's gas by 3 % of its heat).
 */
function zoneTransport(sim) {
  const Z = zoneState(sim);
  if (!Z) return;
  const S = Z.pi.length;
  if (zoneScratch.G.length < S) {
    zoneScratch.G = new Float64Array(S);
    zoneScratch.K = new Float64Array(S);
  }
  const M = zoneScratch.G;
  M.fill(0, 0, S);
  for (let q = 0; q < Z.list.length; q += 1) {
    const c = Z.list[q], z = Z.cell[c];
    sim.rho[c] *= Z.pi[z];
    M[z] += sim.rho[c] * sim.V;
  }
  for (let z = 0; z < S; z += 1) {
    Z.ratio[z] = Z.mass[z] / M[z];
    Z.pi[z] *= Z.ratio[z];
  }
  for (let q = 0; q < Z.list.length; q += 1) {
    const c = Z.list[q];
    sim.rho[c] *= Z.ratio[Z.cell[c]];
  }
}

/**
 * The pressure zones (calibration plan 13.8, item 5): the sealed regions of
 * topology(), each with its pressure as a share of ambient (pi) and its gas's
 * mass, which changes only by its cells' species sources (changeGas). At the
 * first step every zone is at ambient pressure with the gas it has then. Null
 * when off or no region is sealed.
 */
export function zoneState(sim) {
  if (!(sim.P.pressureZones > 0)) return null;
  const topo = topology(sim);
  let Z = sim.zones;
  if (Z && Z.version === topo.version) return Z;
  const index = new Int32Array(topo.sealed.length).fill(-1);
  let S = 0;
  for (let r = 0; r < topo.sealed.length; r += 1) if (topo.sealed[r]) index[r] = S++;
  if (!S) {
    sim.zones = null;
    return null;
  }
  const cell = new Int32Array(sim.N).fill(-1);
  const list = [];
  for (let c = 0; c < sim.N; c += 1) {
    if (topo.region[c] >= 0) cell[c] = index[topo.region[c]];
    if (cell[c] >= 0) list.push(c);
  }
  // The regions only change if a cell turns solid, which none does; should
  // their number change, every zone starts again at ambient pressure.
  const keep = Z && Z.pi.length === S;
  Z = sim.zones = {
    version: topo.version,
    cell,
    // The zones' cells in cell order.
    list: Int32Array.from(list),
    pi: keep ? Z.pi : new Float64Array(S).fill(1),
    mass: keep ? Z.mass : zoneMasses(sim, cell, S),
    source: keep ? Z.source : new Float64Array(S),
    ratio: new Float64Array(S),
    // The last update's rise of T, ratio^kappa.
    kappa: new Float64Array(S),
  };
  return Z;
}

/** Each zone's gas mass at ambient pressure (kg): its cells' gas-law densities at their T. */
function zoneMasses(sim, cell, S) {
  const m = new Float64Array(S);
  for (let c = 0; c < sim.N; c += 1) if (cell[c] >= 0) m[cell[c]] += gasDensity(sim.T[c], sim.Y, c * NS) * sim.V;
  return m;
}

/** Cell c's pressure as a share of ambient: its zone's, else 1. */
export function zonePressure(sim, c) {
  const Z = sim.zones;
  return Z && Z.cell[c] >= 0 ? Z.pi[Z.cell[c]] : 1;
}

/**
 * Each zone's pressure from its gas's mass and heat (FDS's pressure zones;
 * plan 13.8, item 5). The heat sources add heat as at constant pressure (the
 * solver's convention); a zone's gas cannot expand, so it is compressed
 * adiabatically back into its volume: its pressure goes from pi_0 to pi_1 and
 * every cell's T rises by (pi_1 / pi_0)^kappa, kappa = R / (M c_p) (the zone's
 * mass-weighted mean), which is heating at constant volume. pi_1 is the one at
 * which the zone's cells hold its mass exactly: with G the cells' mass at
 * ambient pressure and their present T, M = pi_1^(1 - kappa) pi_0^kappa G, so
 * pi_1 / pi_0 = (1 + y)^(1 / (1 - kappa)) with y = M / (pi_0 G) - 1, the
 * zone's small imbalance (taken through ln(1 + y), which float32 keeps where
 * a power of a number near 1 does not). Then each cell's density is the gas
 * law's at its new T times pi_1. Run before the advection and after the heat
 * sources.
 */
export function pressureZones(sim) {
  const Z = zoneState(sim);
  if (!Z) return;
  const S = Z.pi.length;
  if (zoneScratch.G.length < S) {
    zoneScratch.G = new Float64Array(S);
    zoneScratch.K = new Float64Array(S);
  }
  const G = zoneScratch.G, K = zoneScratch.K;
  G.fill(0, 0, S);
  K.fill(0, 0, S);
  for (let q = 0; q < Z.list.length; q += 1) {
    const c = Z.list[q], z = Z.cell[c], o = c * NS;
    // The gas law's moles per kg and the mixture's c_p in one pass (gasDensity, mixtureCp).
    let inv = 0, cp = 0, rest = 1;
    for (let s = 0; s < NS; s += 1) {
      const y = sim.Y[o + s];
      inv += y / MOLAR[s];
      cp += y * CP[s];
      rest -= y;
    }
    inv += Math.max(0, rest) / MOLAR_N2;
    cp += Math.max(0, rest) * CP_N2;
    const m = (P_AMBIENT / (R_GAS * sim.T[c] * inv)) * sim.V;
    G[z] += m;
    K[z] += (m * R_GAS * inv) / cp;
  }
  for (let z = 0; z < S; z += 1) {
    Z.mass[z] += Z.source[z];
    Z.source[z] = 0;
    const kappa = K[z] / G[z];
    const scaled = Z.pi[z] * G[z];
    const lr = Math.log1p((Z.mass[z] - scaled) / scaled) / (1 - kappa);
    Z.ratio[z] = Math.exp(lr);
    // Each cell's T rises by ratio^kappa.
    Z.kappa[z] = Math.exp(kappa * lr);
    Z.pi[z] *= Z.ratio[z];
  }
  for (let q = 0; q < Z.list.length; q += 1) {
    const c = Z.list[q], z = Z.cell[c];
    const T = sim.T[c] * Z.kappa[z];
    sim.T[c] = T;
    sim.rho[c] = gasDensity(T, sim.Y, c * NS) * Z.pi[z];
  }
}

const zoneScratch = { G: new Float64Array(0), K: new Float64Array(0) };

/** Line 32: the domain's sides are open to still ambient air. */
function boundaries(sim) {
  const { grid: g, P } = sim;
  const { nx, ny, nz } = g;
  const reset = (c) => {
    // The heat and the soot the shell's gas carried, leaving the domain (for budgets only).
    sim.stats.shellHeat += sim.rho[c] * sim.V * mixtureCp(sim.Y, c * NS) * (sim.T[c] - P.ambient);
    sim.stats.sootLeft += sim.rho[c] * sim.V * sim.Y[c * NS + SPECIES.SOOT];
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
export function topology(sim) {
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
  // The open cells' connected regions, and which reach an open side of the
  // domain. A sealed region (a room behind closed doors) cannot let its gas
  // expand out, so the projection takes the region's mean expansion off: its
  // heat stays in it as hotter, thinner gas, as before (proposal F).
  const region = new Int32Array(sim.N).fill(-1);
  const sealed = [];
  const queue = new Int32Array(sim.N);
  for (let s = 0; s < sim.N; s += 1) {
    if (sim.kind[s] === SOLID || region[s] >= 0) continue;
    const id = sealed.length;
    let head = 0, tail = 0, open = false;
    region[s] = id;
    queue[tail++] = s;
    while (head < tail) {
      const c = queue[head++];
      let links = 0;
      for (let d = 0; d < 6; d += 1) {
        if (!(mask[c] & (1 << d))) continue;
        links += 1;
        const n = c + offsets[d];
        if (region[n] < 0) {
          region[n] = id;
          queue[tail++] = n;
        }
      }
      if (count[c] > links) open = true;
    }
    sealed.push(open ? 0 : 1);
  }
  sim.topo = { version: sim.kindVersion, mask, count, colours: [Int32Array.from(red), Int32Array.from(black)], offsets, region, sealed: Uint8Array.from(sealed) };
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
  // Porous cells slow the flow through them (the paper's velocity bound), as a
  // weight on each face, the product of its two cells' factors, inside the
  // projection: the weighted flow still carries the gas's expansion. Scaled
  // after the projection (before 2026-11-12), div u was no longer the
  // expansion in and round them, and the transport made heat there.
  const pv = P.porousVelocity;
  let porous = false;
  if (pv < 1) for (let c = 0; c < sim.N && !porous; c += 1) porous = sim.kind[c] === POROUS;
  const pf = (i, j, k) => (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz ? 1 : sim.kind[at(g, i, j, k)] === POROUS ? pv : 1);
  const div = workArray(sim, "div", sim.N);
  div.fill(0);
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const c = at(g, i, j, k);
        if (sim.kind[c] === SOLID) continue;
        if (porous) {
          const me = pf(i, j, k);
          div[c] =
            (me * pf(i + 1, j, k) * sim.u[uAt(g, i + 1, j, k)] - me * pf(i - 1, j, k) * sim.u[uAt(g, i, j, k)] +
              me * pf(i, j + 1, k) * sim.v[vAt(g, i, j + 1, k)] - me * pf(i, j - 1, k) * sim.v[vAt(g, i, j, k)] +
              me * pf(i, j, k + 1) * sim.w[wAt(g, i, j, k + 1)] - me * pf(i, j, k - 1) * sim.w[wAt(g, i, j, k)]) / h;
          continue;
        }
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
  // The gas's expansion (expansionEnd): the flow is asked for it, less each
  // sealed region's mean, which has nowhere to go.
  if (P.expansion > 0 && sim.work?.expand) {
    const D = sim.work.expand;
    const { region, sealed } = topo;
    const sum = new Float64Array(sealed.length), cells = new Float64Array(sealed.length);
    for (let c = 0; c < sim.N; c += 1) {
      const r = region[c];
      if (r >= 0 && sealed[r]) {
        sum[r] += D[c];
        cells[r] += 1;
      }
    }
    const used = workArray(sim, "expandUsed", sim.N, Float64Array);
    for (let c = 0; c < sim.N; c += 1) {
      const r = region[c];
      used[c] = r < 0 ? 0 : sealed[r] ? D[c] - sum[r] / cells[r] : D[c];
      div[c] -= used[c];
    }
  }
  // Sweeps stop early once the largest pressure change in a sweep would move
  // the flow by less than `projectionTolerance` m/s (the pressure is kept
  // from the last step, so a calm flow needs few).
  const settle = (P.projectionTolerance * sim.rho0 * h) / dt;
  sim.stats.sweeps = 0;
  const o0 = off[0], o1 = off[1], o2 = off[2], o3 = off[3], o4 = off[4], o5 = off[5];
  // With porous cells: each link's weight (the face's), and their sum, open
  // sides counted at the cell's own factor; the ground and solids carry none.
  let beta = null, wsum = null;
  if (porous) {
    beta = workArray(sim, "porousBeta", sim.N * 6, Float64Array);
    wsum = workArray(sim, "porousWsum", sim.N, Float64Array);
    for (let c = 0; c < sim.N; c += 1) {
      if (sim.kind[c] === SOLID) continue;
      const i = c % nx, j = Math.floor(c / nx) % ny, k = Math.floor(c / (nx * ny));
      const me = pf(i, j, k);
      let ws = 0;
      for (let d = 0; d < 6; d += 1) {
        const [a, b, e] = DIRS[d];
        const ii = i + a, jj = j + b, kk = k + e;
        let w = 0;
        if (jj < 0) w = 0;
        else if (ii < 0 || kk < 0 || ii >= nx || jj >= ny || kk >= nz) w = me;
        else if (sim.kind[at(g, ii, jj, kk)] !== SOLID) w = me * pf(ii, jj, kk);
        beta[c * 6 + d] = w;
        ws += w;
      }
      wsum[c] = ws > 0 ? ws : 1;
    }
  }
  for (let sweep = 0; sweep < P.projection; sweep += 1) {
    let change = 0;
    for (const list of topo.colours) {
      for (let q = 0; q < list.length; q += 1) {
        const c = list[q];
        const mask = topo.mask[c];
        if (porous) {
          let sum = 0;
          for (let d = 0; d < 6; d += 1) if (mask & (1 << d)) sum += beta[c * 6 + d] * p[c + off[d]];
          const next = (sum - scale * div[c]) / wsum[c];
          const dd = Math.abs(next - p[c]);
          if (dd > change) change = dd;
          p[c] = next;
          continue;
        }
        let sum;
        // A cell with all six neighbours open (most of them): the same sum in the same order, unbranched.
        if (mask === 63) sum = p[c + o0] + p[c + o1] + p[c + o2] + p[c + o3] + p[c + o4] + p[c + o5];
        else {
          sum = 0;
          if (mask & 1) sum += p[c + o0];
          if (mask & 2) sum += p[c + o1];
          if (mask & 4) sum += p[c + o2];
          if (mask & 8) sum += p[c + o3];
          if (mask & 16) sum += p[c + o4];
          if (mask & 32) sum += p[c + o5];
        }
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
  // A face's weight: 1 without porous cells (the scheme as it was), else the
  // product of its two cells' factors (outside the domain counts 1).
  const fw = (i0, j0, k0, i1, j1, k1) => (porous ? pf(i0, j0, k0) * pf(i1, j1, k1) : 1);
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i <= nx; i += 1) {
        if (solid(i - 1, j, k) || solid(i, j, k)) continue;
        const f = uAt(g, i, j, k);
        if (porous) sim.u[f] = fw(i - 1, j, k, i, j, k) * (sim.u[f] - k2 * (pr(i, j, k) - pr(i - 1, j, k)));
        else sim.u[f] -= k2 * (pr(i, j, k) - pr(i - 1, j, k));
      }
    }
  }
  for (let k = 0; k < nz; k += 1) {
    for (let j = 1; j <= ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        if (solid(i, j - 1, k) || solid(i, j, k)) continue;
        const f = vAt(g, i, j, k);
        if (porous) sim.v[f] = fw(i, j - 1, k, i, j, k) * (sim.v[f] - k2 * (pr(i, j, k) - pr(i, j - 1, k)));
        else sim.v[f] -= k2 * (pr(i, j, k) - pr(i, j - 1, k));
      }
    }
  }
  for (let k = 0; k <= nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        if (solid(i, j, k - 1) || solid(i, j, k)) continue;
        const f = wAt(g, i, j, k);
        if (porous) sim.w[f] = fw(i, j, k - 1, i, j, k) * (sim.w[f] - k2 * (pr(i, j, k) - pr(i, j, k - 1)));
        else sim.w[f] -= k2 * (pr(i, j, k) - pr(i, j, k - 1));
      }
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
  // The heat the pilot adds is tallied (stats.pilotHeat): it is heat the
  // fire has not released, and the health guard's energy check counts it.
  if (pl.temperature) {
    for (const c of pl.cells) {
      if (sim.Ts[c] < pl.temperature) {
        sim.stats.pilotHeat += solidCapacity(sim, c) * (pl.temperature - sim.Ts[c]);
        sim.Ts[c] = pl.temperature;
      }
    }
    return;
  }
  const share = (pl.power * dt) / pl.cells.length;
  for (const c of pl.cells) {
    const C = solidCapacity(sim, c);
    if (C > 0) sim.Ts[c] += share / C;
    else sim.T[c] += share / (sim.rho[c] * mixtureCp(sim.Y, c * NS) * sim.V);
    sim.stats.pilotHeat += share;
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
  // Where the flame may reach for oxygen its own cells lack (burnerStep):
  // the open cells joined to its cells, in breadth-first order, with their
  // distance in cells, out to Heskestad's flame height at the table's
  // largest heat release (at most 4096 cells). Fixed for the run: walls do
  // not move, and contents that burn out stay open.
  const diameter = spec.diameter > 0 ? spec.diameter : Math.sqrt((4 * layers[0].length * grid.h * grid.h) / Math.PI);
  const farthest = Math.max(1, Math.ceil(flameHeight(Math.max(...values), diameter) / grid.h));
  const reachDist = new Map(Array.from(cells, (c) => [c, 0]));
  const reach = [];
  let front = Array.from(cells);
  for (let d = 1; d <= farthest && front.length && reach.length < 4096; d += 1) {
    const next = [];
    for (const c of front) {
      const [i, j, k] = ijk(c);
      for (const [a, b, e] of DIRS) {
        const ii = i + a, jj = j + b, kk = k + e;
        if (ii < 0 || jj < 0 || kk < 0 || ii >= nx || jj >= ny || kk >= grid.nz) continue;
        const n = ii + nx * (jj + ny * kk);
        if (reachDist.has(n) || !open(n)) continue;
        reachDist.set(n, d);
        next.push(n);
        if (reach.length + next.length >= 4096) break;
      }
      if (reach.length + next.length >= 4096) break;
    }
    reach.push(...next);
    front = next;
  }
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
    // The cells beyond its own the flame may reach, and their distance in cells (burnerStep).
    reach: Int32Array.from(reach),
    reachDist: Int32Array.from(reach, (c) => reachDist.get(c)),
    diameterUsed: diameter,
    // What it has given off so far, J, and the heat whose fuel has found no
    // oxygen yet (it burns in a later step, when oxygen comes).
    released: 0,
    convective: 0,
    radiant: 0,
    pending: 0,
    // A design-fire item's curve (buildScene), and, set by the run, water
    // on the item ({start, tau, q, by}) and the time clean agent puts its flames out.
    design: spec.design ?? null,
    suppression: null,
    stopAt: Infinity,
    // The last step's flame, for drawing it (flamescapesRun.js withBurnerFlame):
    // the cells it reached (burnerCells) and its heat release, W.
    flameCells: 0,
    flameRate: 0,
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

/** McCaffrey's (1979) Table 1: the flame's rise (K), and the coefficients of the intermittent flame's (K m kW^-2/5) and the plume's (K m^5/3 kW^-2/3). */
const MCCAFFREY = Object.freeze({ flame: 797, intermittent: 62.9, plume: 21.6 });

/**
 * McCaffrey's mean centreline temperature rise (K) at a height z (m) over a
 * fire of heat release Q (W), with z* = z / Q^(2/5) (Q in kW): 797 K in the
 * continuous flame, 62.9 / z* in the intermittent flame, 21.6 z*^(-5/3) in
 * the plume, the weighted averages of his Table 1 (McCaffrey 1979, NBSIR
 * 79-1910, doi:10.6028/NBS.IR.79-1910). The regions meet where these are
 * equal, z* = 0.079 and 0.201, as he found.
 */
export function plumeRise(z, Q) {
  const q = Math.max(0, Q / 1000);
  if (!(q > 0)) return 0;
  const zs = Math.max(0, z) / q ** 0.4;
  const { flame, intermittent, plume } = MCCAFFREY;
  if (zs <= intermittent / flame) return flame;
  if (zs <= (plume / intermittent) ** 1.5) return intermittent / zs;
  return plume * zs ** (-5 / 3);
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

/**
 * Burns up to `fuel` kg of a burner's fuel in cell c, as far as the cell's
 * oxygen allows: its products, and the convective part of its heat into the
 * gas (the fuel enters at ambient temperature). Returns the kg burnt.
 */
function burnInCell(sim, b, c, fuel) {
  const mass = sim.rho[c] * sim.V;
  const burnt = Math.min(fuel, Math.max(0, (sim.Y[c * NS + SPECIES.O2] * mass) / b.stoich.o2));
  if (!(burnt > 0)) return 0;
  const Ta = sim.P.ambient;
  const excess = mass * mixtureCp(sim.Y, c * NS) * (sim.T[c] - Ta);
  addProducts(sim, c, b.stoich, burnt);
  sim.T[c] = Ta + (excess + (1 - b.radiantFraction) * burnt * b.stoich.heat) / (sim.rho[c] * sim.V * mixtureCp(sim.Y, c * NS));
  return burnt;
}

/**
 * One step of the prescribed burner (createBurner): its heat, products and
 * radiation over [time, time + dt]. Its fuel burns only with oxygen
 * (2026-11-12): each flame cell takes an equal share, as far as its own
 * oxygen allows; what a cell cannot burn goes to the next, then to the open
 * cells beyond the flame's own in breadth-first order (the flame lengthening
 * and running under a ceiling to find air), no farther than Heskestad's
 * flame height for the heat to release; what still finds no oxygen waits
 * (b.pending) and is tried again in the next step. Before, the share heated
 * each cell whatever its oxygen, and a cell under a corridor's ceiling with
 * none left passed 3,000 K.
 */
export function burnerStep(sim, dt) {
  const b = sim.burner;
  const E = burnerHeat(b, sim.time, sim.time + dt) + b.pending;
  b.flameCells = 0;
  if (!(E > 0)) return;
  const n = burnerCells(b, E / dt);
  b.flameCells = n;
  const H = b.stoich.heat;
  let left = E / H;
  const each = left / n;
  for (let q = 0; q < n; q += 1) left -= burnInCell(sim, b, b.cells[q], each);
  // The burner's own cells beyond those its flame reached, then the cells round them.
  for (let q = n; q < b.cells.length && left > 1e-12; q += 1) left -= burnInCell(sim, b, b.cells[q], left);
  if (left > 1e-12) {
    const far = Math.max(1, Math.ceil(flameHeight(E / dt, b.diameterUsed) / b.h));
    for (let q = 0; q < b.reach.length && left > 1e-12 && b.reachDist[q] <= far; q += 1) left -= burnInCell(sim, b, b.reach[q], left);
  }
  left = Math.max(0, left);
  b.pending = left * H;
  const burnt = E - b.pending;
  b.flameRate = burnt / dt;
  if (!(burnt > 0)) return;
  const convective = (1 - b.radiantFraction) * burnt;
  const rays = b.dirs.length / 3;
  const radiant = burnt - convective;
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
  b.released += burnt;
  b.convective += convective;
  b.radiant += radiant;
  sim.stats.released += burnt;
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
  ["expansionStart", expansionStart],
  ["pilot", pilot],
  ["particles", particleStep],
  ["radiation", radiation],
  ["particleToGrid", (sim) => particleToGrid(sim)],
  ["expansionBeforeAdvect", expansionBeforeAdvect],
  // The heat added so far raises a pressure zone's pressure before the advection moves its gas.
  ["zones", (sim) => pressureZones(sim)],
  ["advect", advect],
  ["expansionAfterAdvect", expansionAfterAdvect],
  ["density", (sim) => updateDensity(sim, false)],
  ["buoyancy", buoyancy],
  ["vorticity", vorticityConfinement],
  ["diffuse", diffuse],
  ["ducts", ducts],
  ["chemistry", chemistry],
  ["spray", spray],
  ["foamWet", foamWet],
  ["cullFlameFront", (sim) => cullFlameFront(sim)],
  ["conduct", conduct],
  ["walls", walls],
  ["deposition", deposition],
  ["density2", (sim) => updateDensity(sim)],
  ["expansionEnd", expansionEnd],
  ["boundaries", (sim) => boundaries(sim)],
  ["project", project],
];

/**
 * One frame of Algorithm 1. With `sim.profile` set to an object, each
 * stage's wall time (ms) is added to sim.profile[stage].
 */
export function step(sim, dt = sim.P.dt) {
  // Pressure zones start at ambient pressure with the gas they have when the first step begins.
  if (sim.P.pressureZones > 0 && sim.zones === undefined) zoneState(sim);
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
  sim.steps += 1;
  return sim;
}
