/**
 * The WGSL compute kernels of the Flamescapes solver on the GPU
 * (src/core/flamescapesGpu.js), written from the CPU solver
 * (src/core/flamescapes.js) and the paper (Wrede, Wagner et al. 2026,
 * https://doi.org/10.1145/3842518, CC BY 4.0; transcription in
 * wiki/sources/papers/flamescapes-2026.md). One module, one entry point per
 * stage of Algorithm 1; each entry point is one compute pass.
 *
 * What is the paper's: Algorithm 1 in the CPU solver's order, its equations
 * and constants unchanged, and the paper's own choice of WGSL compute kernels
 * (section 5: Rust, WGPU and WGSL). What is the CPU solver's and kept as it
 * is: the exact integration of each reaction and two-body exchange; the
 * liquid first item's evaporation and a metal's ash; each fuel's own specific
 * heat; solid-aware trilinear sampling (advected T and Y, and the gas round
 * an ember, give solid corners no weight); species diffusion with each face's
 * mean rho D; rays that leave what they still carry in the cell they end in;
 * flame-front particles kept to a share of the buffer; spot ignition by an
 * ember's excess heat; and the pressure projection's red-black Gauss-Seidel
 * with its early stop (at least four sweeps, then stop once a sweep would
 * move the flow by less than projectionTolerance). The red and black passes
 * of a sweep are exact: each colour reads only the other.
 *
 * What is changed from the CPU solver, and why:
 *   - Arithmetic is float32 throughout, where the CPU keeps float64
 *     intermediates and stores float32. 1 - exp(-x) is evaluated by its
 *     series for small x (omexp), because in float32 it would round to zero
 *     for the slow ambient reactions the CPU still integrates.
 *   - Conduction (Eqs. 18, 19) visits each pair of cells once per step, as
 *     the CPU does, in ten race-free passes instead of the CPU's cell order,
 *     arranged so that every solid or porous cell meets its neighbours in
 *     the CPU's own order: its own gas (porous cells), then its +x, +y and +z
 *     faces (gas or solid), each axis split by the parity of the cell's
 *     coordinate so that no cell is in two pairs of one pass, then its -x,
 *     -y and -z faces where they are gas (one pass each: a gas cell is the
 *     backward neighbour of one cell only). The order matters: a nearly
 *     burnt-out item's surface layer holds a fraction of a J/K, each exact
 *     exchange settles it at the other body's temperature, and the last face
 *     wins. The exchange of each pair is the CPU's exact relaxation.
 *   - Vorticity confinement gathers each face's two half forces from its two
 *     cells instead of scattering them, so no two threads write one face.
 *   - Radiation: every emitting cell computes its emission from the
 *     temperatures at the start of the pass (the CPU's later cells see
 *     earlier cells' deposits), and rays deposit temperature rises as
 *     fixed-point integers through atomicAdd (DEP_SCALE steps per kelvin).
 *     Ray directions come from a per-thread PCG hash, not the CPU's seeded
 *     stream.
 *   - Particles live in a fixed-capacity buffer with an atomic counter; dead
 *     particles are flagged and the buffer is compacted once per step. The
 *     hottest particle per cell is an atomicMax on the temperature's bits
 *     (positive floats order like their bit patterns). Embers that land in a
 *     solid add their heat through a fixed-point atomicAdd; the spot-ignition
 *     test then uses the cell's temperature after all of this step's embers
 *     and the largest single ember, where the CPU tests embers one by one.
 *   - Reductions (heat released, fuel burnt, energy radiated, peak
 *     temperature, a sweep's largest pressure change) are per-workgroup trees
 *     and a final pass or one atomicMax per workgroup, in a different order
 *     from the CPU's running sums.
 *   - The projection's early stop cannot read the GPU's answer on the CPU
 *     without a round trip per sweep, so every sweep is queued and each
 *     sweep's threads decide alike from the previous sweep's largest change
 *     (gsSkip); a skipped sweep costs a workgroup launch and one read.
 *   - The conserved transport (proposal F, the CPU's transportGas) runs as one
 *     pass per sweep in which every cell computes both of its faces' fluxes
 *     (its neighbour computes the same numbers for the face they share),
 *     where the CPU walks each line's faces once; the sweeps ping-pong
 *     between a scratch state in `snap` and the live fields, x, y, z on even
 *     steps and z, y, x on odd ones as the CPU's. The CPU counts the mass its
 *     floor adds (stats.massFloor), the faces whose Courant number it clamps
 *     (stats.courantClamped) and the heat the shell's gas carries out
 *     (stats.shellHeat); the GPU keeps none of these tallies.
 *   - Porous cells weight their faces inside the projection (the CPU's), but
 *     only the cells with a porous cell among themselves and their
 *     neighbours take the weighted sums (topology's bit 6); elsewhere every
 *     weight is 1 and the plain sums are the same numbers.
 *   - A sealed region's mean expansion is a float32 tree per region (one
 *     workgroup each) over a cell list the CPU's topology made at upload, where
 *     the CPU sums in cell order each step; only porous cells change kind on
 *     the GPU, and they stay open, so the regions do not change.
 *   - One-dimensional walls (the CPU's walls stage, calibration plan 13.4
 *     item 1): one thread per wall steps its profile; the heat its faces give
 *     their gas waits in the wall buffer and each open cell gathers it in
 *     direction order (the CPU sums it in float64), so no two threads write
 *     one cell. Rays reaching a wall's end deposit in the fixed point of the
 *     other deposits.
 *   - A value every thread needs from an atomic (the particle count, the
 *     sweep's state) is read by one thread per workgroup and shared through
 *     workgroupUniformLoad; kernels that only read the atomic buffer bind it a
 *     second time as plain u32 (atomPlain). On D3D12 an atomicLoad is a
 *     read-modify-write, and thousands on one address serialise.
 */

import { WALL_NODES_MAX } from "../flamescapes.js";

/** Uniform parameters, all 4-byte scalars, then the two fuels' stoichiometry. */
export const PARAMS = [
  ["nx", "u32"], ["ny", "u32"], ["nz", "u32"], ["N", "u32"],
  ["nu", "u32"], ["nv", "u32"], ["nw", "u32"], ["stepNo", "u32"],
  ["slot", "u32"], ["rays", "u32"], ["rayReach", "u32"], ["cap", "u32"],
  ["nPart", "u32"], ["nNodes", "u32"], ["snapMax", "u32"], ["pilotN", "u32"],
  ["pilotMode", "u32"], ["ioNodes", "u32"], ["ioOut", "u32"], ["ioSnap", "u32"],
  ["redRad", "u32"], ["redSteps", "u32"], ["seed", "u32"], ["gsAt", "u32"],
  ["h", "f32"], ["dt", "f32"], ["V", "f32"], ["rho0", "f32"],
  ["settle", "f32"], ["flameCap", "u32"],
  ["ambient", "f32"], ["gravity", "f32"], ["convection", "f32"], ["kappa", "f32"],
  ["vorticity", "f32"], ["porousVel", "f32"], ["diffusivity", "f32"], ["conductivity", "f32"],
  ["emitAbove", "f32"], ["pyA", "f32"], ["pyE", "f32"], ["pyHeat", "f32"],
  ["oxA", "f32"], ["oxE", "f32"], ["pyFuel", "f32"], ["latent", "f32"],
  ["moistFrac", "f32"], ["fuelHC", "f32"], ["skinDensity", "f32"], ["gasA", "f32"],
  ["gasE", "f32"], ["gasFuelExp", "f32"], ["gasOxExp", "f32"], ["flameSpeed", "f32"],
  ["flameLife", "f32"], ["emberRate", "f32"], ["emberPatch", "f32"], ["emberMassLo", "f32"],
  ["emberMassHi", "f32"], ["emberLife", "f32"], ["emberMassLoss", "f32"], ["emberDrag", "f32"],
  ["emberEmis", "f32"], ["emberHC", "f32"], ["pilotTemp", "f32"], ["pilotPower", "f32"],
  ["burnFlux", "f32"], ["ductN", "u32"], ["ductSystems", "u32"], ["ductAt", "u32"],
  ["ductFlow", "f32"],
  // Sprinklers (suppression.js): the step's start time, Evans on, the spray's
  // columns (count and offset in `sup`), its evaporation coefficient, and the
  // gathered cells whose speed is read (sensor cells, from speedFrom on).
  ["time", "f32"], ["supOn", "u32"], ["sprayN", "u32"], ["supCol", "u32"],
  ["sprayEta", "f32"], ["speedFrom", "u32"], ["ioSpeed", "u32"],
  // "wetting" mode: the share of the water left that soaks into each fuel cell.
  ["sprayWet", "f32"],
  // Rooms flooded with clean agent or blanketed with foam (suppressionSystems.js):
  // on, where each room's state starts in `sys`, and the foam's wet list (offset, count).
  ["sysOn", "u32"], ["sysState", "u32"], ["sysWetAt", "u32"], ["sysWetN", "u32"],
  // The interface coefficient of inert solids (the CPU's wallConvection).
  ["wallConvection", "f32"],
  // A design-fire first item (flamescapes.js createBurner, burnerStep): its
  // cells (count, where io lists them), its layers' starts in that list
  // (layers + 1 of them, from burnerLayerAt), the cells its flame may reach
  // beyond them and their distances (reachN of each, from reachAt), its
  // diameter for the flame's height and the one its reach used, its radiant
  // fraction, the table's heat this step (J, the host's burnerHeat), and
  // where in `red` its frame's tallies and the heat still waiting for oxygen are.
  ["burnerN", "u32"], ["burnerAt", "u32"], ["burnerLayers", "u32"], ["burnerLayerAt", "u32"],
  ["reachN", "u32"], ["reachAt", "u32"], ["burnerD", "f32"], ["burnerDUsed", "f32"],
  ["burnerRf", "f32"], ["burnerHeat", "f32"], ["redBurner", "u32"],
  // Where in `red` the pilot's added heat (J) accumulates over a frame (the CPU's stats.pilotHeat).
  ["redPilot", "u32"],
  // Proposal F (flamescapes.js DEFAULTS.conservative, .expansion): on or off;
  // where in `snap` the transport's scratch state (gasAt) and the step's
  // starting temperatures (expandAt) are; the sealed regions (count, their
  // cell lists in `io` from sealedAt) and their mean expansion in `red` (redMean).
  ["conservative", "u32"], ["expansion", "u32"], ["gasAt", "u32"], ["expandAt", "u32"],
  ["sealedN", "u32"], ["sealedAt", "u32"], ["redMean", "u32"],
  // One-dimensional walls (flamescapes.js wallSetup, calibration plan 13.4
  // item 1): how many, their nodes each, their stacks (the `wall` buffer's
  // layout, WALL), and where in `atom` their ends' ray deposits are (two per wall).
  ["wallN", "u32"], ["wallNodes", "u32"], ["wallStacks", "u32"], ["wallDepAt", "u32"],
];
/**
 * Scalars padded to a 16-byte boundary; the fuel table follows, four vec4
 * per fuel (contents 0, first item 1): (o2, co2, co, soot), (h2o, heat,
 * ignition K, embers), (liquid, boiling K, latent J/kg, ash kg/kg),
 * (specific heat J/(kg K), pool m''_inf kg/(m^2 s) or -1, pool k beta /m or -1, 0).
 */
export const PARAM_SCALARS = Math.ceil(PARAMS.length / 4) * 4;
export const FUEL_VEC4 = 4;
export const PARAM_BYTES = PARAM_SCALARS * 4 + 2 * FUEL_VEC4 * 16;

/** Per-cell float fields, each N long, in the `cf` buffer (D: the gas's expansion, proposal F). */
export const CF = Object.freeze({ T: 0, Ts: 1, rho: 2, p: 3, fuel: 4, moist: 5, area: 6, inertC: 7, inertK: 8, eps: 9, loss: 10, div: 11, wx: 12, wy: 13, wz: 14, mag: 15, fx: 16, fy: 17, fz: 18, D: 19 });
export const CF_FIELDS = 20;
/**
 * Per-cell integer fields in the `ci` buffer (sealed: the cell's sealed
 * region, numbered from 0, or NO_CELL; the CPU's topology, uploaded once:
 * only porous cells change kind on the GPU, and they stay open; wall: the
 * cell's one-dimensional wall, or NO_CELL, read only with walls).
 */
export const CI = Object.freeze({ kind: 0, fuelKind: 1, burning: 2, vapour: 3, mask: 4, count: 5, sealed: 6, wall: 7 });
export const CI_FIELDS = 8;
/**
 * Per-cell atomic fields in the `atom` buffer, then (from U.gsAt = AT_FIELDS
 * N) one slot per pressure sweep: the largest pressure change of that sweep;
 * then (from U.wallDepAt) two per one-dimensional wall, its low and high
 * ends' ray deposits.
 */
export const AT = Object.freeze({ flame: 0, depGas: 1, depSolid: 2, ember: 3, emberMax: 4 });
export const AT_FIELDS = 5;
/**
 * Counters in the `ctr` buffer (burning: the cells that burned last step, for
 * spawnFlame's share). The frame's health (flamescapes.js healthReadings), set
 * by peakTemperature: speed, the fastest cell-centred gas speed (f32 bits);
 * heat, the gas's heat above ambient in units of HEAT_UNIT J; bad, 1 when a
 * gas cell holds a non-finite or non-positive value.
 */
export const CTR = Object.freeze({ count: 0, countB: 1, embers: 2, spots: 3, maxT: 4, before: 5, burning: 6, speed: 7, heat: 8, bad: 9 });
export const CTR_SIZE = 12;
/** Joules per unit of the gas-heat counter: 17 GJ before it wraps. */
export const HEAT_UNIT = 4;
/** Floats per particle (three vec4). */
export const PARTICLE_FLOATS = 12;
export const WG = 64;

/**
 * Bindings: 0 the parameters, 12 the pressure sweep's index (a uniform at a
 * dynamic offset, one per sweep), the rest storage buffers. 13 is the `atom`
 * buffer again as plain u32, for kernels that only read it: an atomicLoad is a
 * read-modify-write on some backends (D3D12), a plain load is not.
 */
export const B = Object.freeze({ U: 0, cf: 1, Y: 2, ci: 3, vel: 4, snap: 5, atom: 6, parts: 7, partsB: 8, ctr: 9, red: 10, io: 11, sweep: 12, atomPlain: 13, sup: 14, sys: 15, wall: 16 });
/**
 * The `wall` buffer (binding 16), only as large as a run with one-dimensional
 * walls needs (16 bytes otherwise, never read), W walls of n nodes and S
 * stacks: each wall's cell (u32 bits), from W its meta (axis + 4 stack, u32
 * bits), from 2W its `last` mean, from 3W its nodes (n each), from (3 + n) W
 * the heat its six faces gave their gas (J, six each); from (9 + n) W each
 * stack's node capacities (n each, J/(m^2 K)), then its conductances (n - 1
 * each), its heat per area and its thickness (one each). The CPU's
 * sim.wall, in that order.
 */
export const WALL = Object.freeze({ faces: 6, fixed: 9 });
/**
 * The `sys` buffer (binding 15), only as large as a run with flooded or
 * foamed rooms needs (16 bytes otherwise, never read): per cell its room
 * (u32 bits, 0 none), then from U.sysState eight floats per room
 * (suppressionSystems.js STATE: agent %, class A and B extinguishing %,
 * foam's share of evaporation left, foam's wetting share; room 0 is none),
 * written by the CPU before every step, then from U.sysWetAt four words per
 * foam wet cell: cell, room, solution kg/s on its column, unused.
 */
export const SYS = Object.freeze({ stride: 8, wet: 4 });
/**
 * The `sup` buffer (binding 14), only as large as a run with sprinklers
 * needs (16 bytes otherwise, never read): per cell Evans's decay time
 * (0: not covered), its head's opening time and its rate when it opened
 * (-1: armed), then from U.supCol four words per spray column: top cell,
 * open cells below it, water (kg/s), unused.
 */
export const SUP = Object.freeze({ tau: 0, tAct: 1, rate: 2, fields: 3, column: 4 });
/** Bytes of the sweep-index uniform binding. */
export const SWEEP_BYTES = 16;

/**
 * Entry points: the bindings each uses (its pipeline layout) and how many
 * threads it runs, as a function of the sizes d = {N, nu, nv, nw, nx, ny, nz,
 * cap, nNodes, snapMax}.
 */
const cells = (d) => d.N;
const faces = (d) => d.nu + d.nv + d.nw;
const particles = (d) => d.cap;
const half = (d) => Math.floor((d.nx + 1) / 2) * d.ny * d.nz;
const one = () => 1;
export const KERNELS = {
  pilot: { uses: [0, 1, 2, 3, 10, 11], threads: one, wg: 1 },
  burner: { uses: [0, 1, 2, 3, 10, 11], threads: one, wg: 1 },
  ducts: { uses: [0, 1, 2, 11], threads: one, wg: 1 },
  countBurning: { uses: [0, 3, 9], threads: cells },
  spawnFlame: { uses: [0, 1, 3, 7, 9], threads: cells },
  spawnFlameEnd: { uses: [0, 9], threads: one, wg: 1 },
  spawnEmbers: { uses: [0, 1, 3, 4, 7, 9], threads: cells },
  moveParticles: { uses: [0, 1, 3, 4, 7, 9], threads: particles },
  radEmit: { uses: [0, 1, 2, 3, 6, 10, 16], threads: cells },
  radApply: { uses: [0, 1, 3, 13, 16], threads: cells },
  particleToGrid: { uses: [0, 1, 3, 6, 7, 9], threads: particles },
  emberDeposit: { uses: [0, 1, 3, 9, 13], threads: cells },
  // With the conservative transport the cells' T and Y are the transport's, not the back-trace's.
  advect: { uses: [0, 1, 2, 3, 4, 5], threads: (d) => d.nu + d.nv + d.nw + (d.conservative ? 0 : d.N) },
  expandBefore: { uses: [0, 1, 3, 5], threads: cells },
  // The transport's sweeps by axis and source: x, y, z on even steps, z, y, x on odd ones (the CPU's transportGas).
  transport_x_start: { uses: [0, 1, 2, 3, 5], threads: cells },
  transport_z_start: { uses: [0, 1, 2, 3, 5], threads: cells },
  transport_y_scratch: { uses: [0, 1, 2, 3, 5], threads: cells },
  transport_z_live: { uses: [0, 1, 2, 3, 5], threads: cells },
  transport_x_live: { uses: [0, 1, 2, 3, 5], threads: cells },
  transportEnd: { uses: [0, 1, 2, 3, 5], threads: cells },
  expandEnd: { uses: [0, 1, 3, 5], threads: cells },
  // One workgroup per sealed region.
  sealedMean: { uses: [0, 1, 10, 11], threads: (d) => (d.sealedN || 0) * WG },
  density: { uses: [0, 1, 2, 3], threads: cells },
  buoyancy: { uses: [0, 1, 3, 4], threads: (d) => d.nv },
  vortCurl: { uses: [0, 1, 3, 4], threads: cells },
  vortForce: { uses: [0, 1, 3], threads: cells },
  vortApply: { uses: [0, 1, 4], threads: faces },
  diffuse: { uses: [0, 1, 2, 3, 5], threads: cells },
  chemistry: { uses: [0, 1, 2, 3, 10, 13, 14, 15], threads: cells },
  spray: { uses: [0, 1, 2, 3, 4, 14], threads: (d) => d.sprayCols || 0 },
  systemsWet: { uses: [0, 1, 3, 15], threads: (d) => d.sysWet || 0 },
  reduceFinish: { uses: [0, 10], threads: () => WG },
  cullFlameFront: { uses: [0, 3, 7, 9], threads: particles },
  conductPorous: { uses: [0, 1, 2, 3], threads: cells },
  conduct_x0: { uses: [0, 1, 2, 3], threads: cells },
  conduct_x1: { uses: [0, 1, 2, 3], threads: cells },
  conduct_y0: { uses: [0, 1, 2, 3], threads: cells },
  conduct_y1: { uses: [0, 1, 2, 3], threads: cells },
  conduct_z0: { uses: [0, 1, 2, 3], threads: cells },
  conduct_z1: { uses: [0, 1, 2, 3], threads: cells },
  conduct_bx: { uses: [0, 1, 2, 3], threads: cells },
  conduct_by: { uses: [0, 1, 2, 3], threads: cells },
  conduct_bz: { uses: [0, 1, 2, 3], threads: cells },
  // One-dimensional walls: one thread per wall, then each open cell gathers what its wall neighbours gave it.
  walls: { uses: [0, 1, 2, 3, 16], threads: (d) => d.wallN || 0 },
  wallsApply: { uses: [0, 1, 2, 3, 16], threads: cells },
  densityBoundaries: { uses: [0, 1, 2, 3], threads: cells },
  topology: { uses: [0, 3], threads: cells },
  projectZero: { uses: [0, 3, 4], threads: faces },
  divergence: { uses: [0, 1, 3, 4, 10], threads: cells },
  gsRed: { uses: [0, 1, 3, 6, 12], threads: half },
  gsBlack: { uses: [0, 1, 3, 6, 12], threads: half },
  projectUpdate: { uses: [0, 1, 3, 4], threads: faces },
  compact: { uses: [0, 7, 8, 9], threads: particles },
  compactFinish: { uses: [0, 9], threads: one, wg: 1 },
  gather: { uses: [0, 1, 2, 4, 11], threads: (d) => d.nNodes },
  peakTemperature: { uses: [0, 1, 3, 4, 9], threads: cells },
  snapshot: { uses: [0, 7, 9, 11], threads: (d) => d.snapMax },
};

/** Fixed-point steps per kelvin of a radiation deposit, and per joule of ember heat. */
export const DEP_SCALE = 16384;
export const EMBER_SCALE = 16;

const fieldConsts = (prefix, table) => Object.entries(table).map(([k, v]) => `const ${prefix}${k[0].toUpperCase()}${k.slice(1)}: u32 = ${v}u;`).join("\n");

/** Trilinear sample of a face field (the CPU's sampleFace) and the velocity there (velocityAt), from buffer `buf`. */
const faceSample = (suffix, buf) => `
fn face${suffix}(base: u32, dx: u32, dy: u32, dz: u32, p: vec3<f32>) -> f32 {
  let x = min(max(p.x, 0.0), f32(dx) - 1.001);
  let y = min(max(p.y, 0.0), f32(dy) - 1.001);
  let z = min(max(p.z, 0.0), f32(dz) - 1.001);
  let i = u32(floor(x)); let j = u32(floor(y)); let k = u32(floor(z));
  let fx = x - f32(i); let fy = y - f32(j); let fz = z - f32(k);
  let b = base + i + dx * (j + dy * k);
  let sj = dx; let sk = dx * dy;
  let c00 = ${buf}[b] * (1.0 - fx) + ${buf}[b + 1u] * fx;
  let c10 = ${buf}[b + sj] * (1.0 - fx) + ${buf}[b + sj + 1u] * fx;
  let c01 = ${buf}[b + sk] * (1.0 - fx) + ${buf}[b + sk + 1u] * fx;
  let c11 = ${buf}[b + sk + sj] * (1.0 - fx) + ${buf}[b + sk + sj + 1u] * fx;
  return (c00 * (1.0 - fy) + c10 * fy) * (1.0 - fz) + (c01 * (1.0 - fy) + c11 * fy) * fz;
}
fn velocity${suffix}(p: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    face${suffix}(0u, U.nx + 1u, U.ny, U.nz, p - vec3<f32>(0.0, 0.5, 0.5)),
    face${suffix}(U.nu, U.nx, U.ny + 1u, U.nz, p - vec3<f32>(0.5, 0.0, 0.5)),
    face${suffix}(U.nu + U.nv, U.nx, U.ny, U.nz + 1u, p - vec3<f32>(0.5, 0.5, 0.0)));
}`;

/** A per-cell entry point: `body` runs for cell c < N. */
const perCell = (name, body) => `
@compute @workgroup_size(64)
fn ${name}(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let c = lin(gid, nwg);
  if (c >= U.N) { return; }
${body}
}`;

/** An entry point over `count` threads (index t). */
const perThread = (name, count, body) => `
@compute @workgroup_size(64)
fn ${name}(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = lin(gid, nwg);
  if (t >= ${count}) { return; }
${body}
}`;

/**
 * An entry point over the live particles (index t < n): one thread per
 * workgroup reads the atomic count and the workgroup shares it.
 */
const perParticle = (name, body) => `
@compute @workgroup_size(64)
fn ${name}(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) { wshare = min(atomicLoad(&ctr[0]), U.cap); }
  let n = workgroupUniformLoad(&wshare);
  let t = lin(gid, nwg);
${name === "snapshot" ? "" : "  if (t >= n) { return; }"}
${body}
}`;

/** Conduction across the +axis faces of the solid cells whose coordinate has the given parity. */
const conductPass = (axis, parity) => {
  const coord = ["x", "y", "z"][axis];
  const dim = ["U.nx", "U.ny", "U.nz"][axis];
  const stride = ["1u", "U.nx", "U.nx * U.ny"][axis];
  return perCell(`conduct_${coord}${parity}`, `
  let p = coords(c);
  if ((p.${coord} & 1u) != ${parity}u || p.${coord} + 1u >= ${dim}) { return; }
  conductForward(c, c + ${stride});`);
};

/** Conduction across the -axis faces of the solid cells, where the neighbour is gas. */
const conductBack = (axis) => {
  const coord = ["x", "y", "z"][axis];
  const stride = ["1u", "U.nx", "U.nx * U.ny"][axis];
  return perCell(`conduct_b${coord}`, `
  let p = coords(c);
  if (p.${coord} == 0u) { return; }
  conductBackward(c, c - ${stride});`);
};

/**
 * One colour of a red-black Gauss-Seidel sweep, over the cells of that colour
 * only, unless the projection has settled (gsSkip); the workgroup's largest
 * pressure change goes to the sweep's slot (an atomicMax on the bits of a
 * non-negative float).
 */
const gsPass = (name, colour) => `
@compute @workgroup_size(64)
fn ${name}(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  // One thread reads the sweep's state; the workgroup shares it (and returns together).
  if (li == 0u) { wshare = select(0u, 1u, gsSkip()); }
  if (workgroupUniformLoad(&wshare) != 0u) { return; }
  wmax[li] = bitcast<u32>(gsCell(lin(gid, nwg), ${colour}u));
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) { wmax[li] = max(wmax[li], wmax[li + s]); }
    workgroupBarrier();
  }
  if (li == 0u && wmax[0] > 0u) { atomicMax(&atom[U.gsAt + SW.s], wmax[0]); }
}`;

/**
 * The gas states a transport sweep reads and writes (proposal F): "start", the
 * step's starting state in the snapshot (its density from T by the gas law);
 * "scratch", the transport's own state in `snap` from U.gasAt; "live", cf's
 * rho and the Y buffer.
 */
const GAS_STATE = {
  start: { rho: (c) => `rhoStart(${c})`, y: (c, q) => `yStart(${c}, ${q})` },
  scratch: { rho: (c) => `snap[U.gasAt + ${c}]`, y: (c, q) => `snap[U.gasAt + U.N + ${c} * NS + ${q}]` },
  live: { rho: (c) => `cf[ix(fRho, ${c})]`, y: (c, q) => `Y[${c} * NS + ${q}]` },
};

/**
 * One sweep of the gas's transport along an axis (the CPU's transportGas
 * sweep), from state `src` to state `dst`. The CPU walks each line's faces
 * and adds every face's flux to its two cells; here each cell computes its
 * own two faces (the one before it first, as the CPU's line order adds
 * them), and its neighbour computes the same numbers for the face they
 * share, so a face takes out of one cell what it puts into the next and no
 * two threads write one cell.
 */
const transportSweep = (axis, src, dst) => {
  const X = ["x", "y", "z"][axis];
  const n = ["U.nx", "U.ny", "U.nz"][axis];
  const stride = ["1", "i32(U.nx)", "i32(U.nx * U.ny)"][axis];
  const face = ["uAt(u32(a), p.y, p.z)", "vAt(p.x, u32(a), p.z)", "wAt(p.x, p.y, u32(a))"][axis];
  const S = GAS_STATE[src], D = GAS_STATE[dst];
  // Named by axis and source: as the order alternates, x and z each run first (from the start) or last (from the live fields).
  const K = `${X}_${src}`;
  return `
/** A cell along this sweep's line at position b: -1 outside the domain (ambient air), -2 a solid. */
fn line_${K}(c: u32, pos: i32, b: i32) -> i32 {
  if (b < 0 || b >= i32(${n})) { return -1; }
  let q = i32(c) + (b - pos) * ${stride};
  if (kindOf(u32(q)) == SOLID) { return -2; }
  return q;
}
fn rho_${K}(q: i32) -> f32 {
  if (q < 0) { return U.rho0; }
  return ${S.rho("u32(q)")};
}
fn y_${K}(q: i32, s: u32) -> f32 {
  if (q < 0) { return select(0.0, Y_O2_AIR, s == 1u); }
  return ${S.y("u32(q)", "s")};
}
${perCell(`transport_${K}`, `
  if (kindOf(c) == SOLID) { return; }
  let p = coords(c);
  let pos = i32(p.${X});
  let s = U.dt / U.h;
  var dm = 0.0;
  var dd = 0.0;
  var dY = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var f = 0; f < 2; f++) {
    let a = pos + f;
    ${axis === 1 ? "if (a == 0) { continue; } // the ground carries nothing" : ""}
    let vf = snap[${face}];
    if (vf == 0.0) { continue; }
    let left = line_${K}(c, pos, a - 1);
    let right = line_${K}(c, pos, a);
    let donor = select(right, left, vf > 0.0);
    let acceptor = select(left, right, vf > 0.0);
    if (donor == -2 || acceptor == -2) { continue; }
    let back = select(line_${K}(c, pos, a + 1), line_${K}(c, pos, a - 2), vf > 0.0);
    let courant = min(1.0, abs(vf) * s);
    // The face's flow out of the cell before it and into the cell after: this cell is after the first face, before the second.
    let outflow = select(-courant, courant, vf > 0.0);
    if (f == 0) { dd -= outflow; } else { dd += outflow; }
    // The face's density: the donor's, with a van Leer limited slope towards the acceptor.
    let rd = rho_${K}(donor);
    let ra = rho_${K}(acceptor);
    var rf = rd;
    if (back != -2) {
      let dr = ra - rd;
      if (dr != 0.0) {
        let ratio = (rd - rho_${K}(back)) / dr;
        rf += 0.5 * ((ratio + abs(ratio)) / (1.0 + abs(ratio))) * (1.0 - courant) * dr;
      }
    }
    let m = courant * rf;
    let sgn = select(-1.0, 1.0, acceptor == i32(c));
    dm += sgn * m;
    for (var q = 0u; q < NS; q++) {
      let yd = y_${K}(donor, q);
      var yf = yd;
      if (back != -2) {
        let dy = y_${K}(acceptor, q) - yd;
        if (dy != 0.0) {
          let t = (yd - y_${K}(back, q)) / dy;
          yf += 0.5 * ((t + abs(t)) / (1.0 + abs(t))) * (1.0 - courant) * dy;
        }
      }
      dY[q] += sgn * (m * yf);
    }
  }
  // The flux difference, and the cell's starting density times its faces'
  // divergence given back (the advective form: a uniform gas stays uniform).
  let before = rho_${K}(i32(c));
  let r0 = rhoStart(c);
  var after = before + dm + r0 * dd;
  // A cell emptied faster than it fills is kept at a hundredth of ambient (the CPU counts the mass added; the GPU does not).
  if (after < 0.01 * U.rho0) { after = 0.01 * U.rho0; }
  for (var q = 0u; q < NS; q++) {
    let y = y_${K}(i32(c), q);
    ${D.y("c", "q")} = clamp((before * y + r0 * yStart(c, q) * dd + dY[q]) / after, 0.0, 1.0);
  }
  ${D.rho("c")} = after;`)}`;
};

export function buildShader() {
  const scalars = PARAMS.map(([name, type]) => `  ${name}: ${type},`);
  for (let i = PARAMS.length; i < PARAM_SCALARS; i += 1) scalars.push(`  padEnd${i}: u32,`);
  return /* wgsl */ `
struct Params {
${scalars.join("\n")}
  fuel: array<vec4<f32>, ${2 * FUEL_VEC4}>,
}
struct Particle { p: vec4<f32>, v: vec4<f32>, q: vec4<f32> }
struct Fuel { o2: f32, co2: f32, co: f32, soot: f32, h2o: f32, heat: f32, ign: f32, embers: f32, liquid: f32, boiling: f32, latent: f32, ash: f32, cp: f32, mInf: f32, kBeta: f32, water: f32 }
struct Tri { b: u32, di: u32, dj: u32, dk: u32, f: vec3<f32> }

@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read_write> cf: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<storage, read_write> ci: array<u32>;
@group(0) @binding(4) var<storage, read_write> vel: array<f32>;
@group(0) @binding(5) var<storage, read_write> snap: array<f32>;
@group(0) @binding(6) var<storage, read_write> atom: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(8) var<storage, read_write> partsB: array<Particle>;
@group(0) @binding(9) var<storage, read_write> ctr: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> red: array<f32>;
@group(0) @binding(11) var<storage, read_write> io: array<u32>;
struct Sweep { s: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(12) var<uniform> SW: Sweep;
@group(0) @binding(13) var<storage, read_write> atomPlain: array<u32>;
@group(0) @binding(14) var<storage, read_write> sup: array<f32>;
@group(0) @binding(15) var<storage, read_write> sys: array<f32>;
@group(0) @binding(16) var<storage, read_write> wall: array<f32>;

var<workgroup> wsum: array<vec4<f32>, 64>;
var<workgroup> wmax: array<u32, 64>;
var<workgroup> wspeed: array<u32, 64>;
var<workgroup> wheat: array<f32, 64>;
var<workgroup> wbad: array<u32, 64>;
var<workgroup> wshare: u32;

${fieldConsts("f", CF)}
${fieldConsts("i", CI)}
${fieldConsts("a", AT)}
const GAS: u32 = 0u;
const POROUS: u32 = 1u;
const SOLID: u32 = 2u;
const NS: u32 = 6u;
const R_GAS: f32 = 8.314462618;
const SIGMA: f32 = 5.670374419e-8;
const P_AMBIENT: f32 = 101325.0;
const KELVIN: f32 = 273.15;
const Y_O2_AIR: f32 = 0.232;
const PI: f32 = 3.14159265358979;
const DEP_SCALE: f32 = ${DEP_SCALE}.0;
const EMBER_SCALE: f32 = ${EMBER_SCALE}.0;
const NO_CELL: u32 = 0xffffffffu;
const WALL_MAX: u32 = ${WALL_NODES_MAX}u;

// ------------------------------------------------------------------ indexing

fn lin(g: vec3<u32>, n: vec3<u32>) -> u32 { return g.x + g.y * n.x * 64u; }
fn ix(field: u32, c: u32) -> u32 { return field * U.N + c; }
fn coords(c: u32) -> vec3<u32> { return vec3<u32>(c % U.nx, (c / U.nx) % U.ny, c / (U.nx * U.ny)); }
fn at3(i: u32, j: u32, k: u32) -> u32 { return i + U.nx * (j + U.ny * k); }
fn inside(q: vec3<i32>) -> bool {
  return q.x >= 0 && q.y >= 0 && q.z >= 0 && q.x < i32(U.nx) && q.y < i32(U.ny) && q.z < i32(U.nz);
}
fn atI(q: vec3<i32>) -> u32 { return u32(q.x) + U.nx * (u32(q.y) + U.ny * u32(q.z)); }
fn kindOf(c: u32) -> u32 { return ci[ix(iKind, c)]; }
fn solidAt(q: vec3<i32>) -> bool { return inside(q) && kindOf(atI(q)) == SOLID; }
fn porousAt(q: vec3<i32>) -> bool { return inside(q) && kindOf(atI(q)) == POROUS; }
/** The CPU's DIRS: +x, -x, +y, -y, +z, -z. */
fn dirv(d: u32) -> vec3<i32> {
  var D = array<vec3<i32>, 6>(vec3<i32>(1, 0, 0), vec3<i32>(-1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(0, -1, 0), vec3<i32>(0, 0, 1), vec3<i32>(0, 0, -1));
  return D[d];
}
/** Index offset of the neighbour in direction d. */
fn noff(d: u32) -> i32 {
  let sx = 1; let sy = i32(U.nx); let sz = i32(U.nx * U.ny);
  var D = array<i32, 6>(sx, -sx, sy, -sy, sz, -sz);
  return D[d];
}
fn uAt(i: u32, j: u32, k: u32) -> u32 { return i + (U.nx + 1u) * (j + U.ny * k); }
fn vAt(i: u32, j: u32, k: u32) -> u32 { return U.nu + i + U.nx * (j + (U.ny + 1u) * k); }
fn wAt(i: u32, j: u32, k: u32) -> u32 { return U.nu + U.nv + i + U.nx * (j + U.ny * k); }

// ------------------------------------------------------------------ physics helpers

/** 1 - exp(-x), accurate in float32 for small x (the CPU's 1 - survives(k, dt)). */
fn omexp(x: f32) -> f32 {
  if (x < 0.5) {
    return x * (1.0 - x / 2.0 * (1.0 - x / 3.0 * (1.0 - x / 4.0 * (1.0 - x / 5.0 * (1.0 - x / 6.0 * (1.0 - x / 7.0 * (1.0 - x / 8.0)))))));
  }
  return 1.0 - exp(-x);
}
/** First-order Arrhenius rate (Eq. 10's form). */
fn arrhenius(A: f32, E: f32, T: f32) -> f32 {
  if (T > 0.0) { return A * exp(-E / (R_GAS * T)); }
  return 0.0;
}
/** Exact exchange between two bodies (the CPU's exchange). */
fn exchange(Ta: f32, Tb: f32, Ca: f32, Cb: f32, G: f32, dt: f32) -> f32 {
  if (!(Ca > 0.0) || !(Cb > 0.0) || !(G > 0.0)) { return 0.0; }
  let mu = (Ca * Cb) / (Ca + Cb);
  return (Ta - Tb) * mu * omexp(G * dt / mu);
}
fn molar(s: u32) -> f32 {
  var M = array<f32, 6>(0.03, 0.032, 0.044, 0.018, 0.012, 0.028);
  return M[s];
}
fn heatCap(s: u32) -> f32 {
  var C = array<f32, 6>(2200.0, 918.0, 844.0, 1864.0, 710.0, 1040.0);
  return C[s];
}
/** Equation of state, Eq. 7, nitrogen the rest. */
fn gasDensity(T: f32, c: u32) -> f32 {
  var inv = 0.0;
  var rest = 1.0;
  for (var s = 0u; s < NS; s++) {
    let y = Y[c * NS + s];
    inv += y / molar(s);
    rest -= y;
  }
  inv += max(0.0, rest) / 0.028;
  return P_AMBIENT / (R_GAS * T * inv);
}
fn mixtureCp(c: u32) -> f32 {
  var cp = 0.0;
  var rest = 1.0;
  for (var s = 0u; s < NS; s++) {
    cp += Y[c * NS + s] * heatCap(s);
    rest -= Y[c * NS + s];
  }
  return cp + max(0.0, rest) * 1040.0;
}
fn mixtureCpSnap(c: u32, base: u32) -> f32 {
  var cp = 0.0;
  var rest = 1.0;
  for (var s = 0u; s < NS; s++) {
    let y = snap[base + c * NS + s];
    cp += y * heatCap(s);
    rest -= y;
  }
  return cp + max(0.0, rest) * 1040.0;
}
fn gasCap(c: u32) -> f32 { return cf[ix(fRho, c)] * mixtureCp(c) * U.V; }
fn fuelOf(k: u32) -> Fuel {
  let a = U.fuel[${FUEL_VEC4}u * k];
  let b = U.fuel[${FUEL_VEC4}u * k + 1u];
  let l = U.fuel[${FUEL_VEC4}u * k + 2u];
  let e = U.fuel[${FUEL_VEC4}u * k + 3u];
  return Fuel(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, l.x, l.y, l.z, l.w, e.x, e.y, e.z, e.w);
}
/** A fuel's water rule (fireClasses.js waterResponse): 1 wets, 2 Evans's cap, 4 holds no water. */
fn waterBit(f: Fuel, bit: u32) -> bool { return (u32(f.water + 0.5) & bit) != 0u; }
fn skinOf(fuel: f32, area: f32) -> f32 { return min(fuel, U.skinDensity * area); }
fn capOf(inertC: f32, fuel: f32, area: f32, moist: f32, cp: f32) -> f32 {
  return inertC + skinOf(fuel, area) * cp + moist * 4186.0;
}
/** Specific heat of a cell's fuel: the fuel's own, else wood's (the CPU's fuelCp). */
fn fuelCp(c: u32) -> f32 { return fuelOf(ci[ix(iFuelKind, c)]).cp; }
/** Heat capacity of a cell's solid: inert part, fuel surface layer, water (J/K). */
fn solidCap(c: u32) -> f32 {
  return capOf(cf[ix(fInertC, c)], cf[ix(fFuel, c)], cf[ix(fArea, c)], cf[ix(fMoist, c)], fuelCp(c));
}

/** A cell's one-dimensional wall (its index in the wall buffer), or NO_CELL; the field is never read without walls. */
fn wallOf(c: u32) -> u32 {
  if (U.wallN == 0u) { return NO_CELL; }
  return ci[ix(iWall, c)];
}
/** The wall buffer's layout (WALL): meta, last, nodes, face heats, then the stacks. */
fn wMeta(w: u32) -> u32 { return bitcast<u32>(wall[U.wallN + w]); }
fn wLast(w: u32) -> u32 { return 2u * U.wallN + w; }
fn wNode(w: u32, q: u32) -> u32 { return 3u * U.wallN + w * U.wallNodes + q; }
fn wQ(w: u32, d: u32) -> u32 { return (3u + U.wallNodes) * U.wallN + 6u * w + d; }
fn wStacks() -> u32 { return (9u + U.wallNodes) * U.wallN; }
fn wC(s: u32, q: u32) -> u32 { return wStacks() + s * U.wallNodes + q; }
fn wG(s: u32, q: u32) -> u32 { return wStacks() + U.wallStacks * U.wallNodes + s * (U.wallNodes - 1u) + q; }
fn wHeat(s: u32) -> u32 { return wStacks() + U.wallStacks * (2u * U.wallNodes - 1u) + s; }
fn wThick(s: u32) -> u32 { return wHeat(s) + U.wallStacks; }
/** Wall w's area on a side face: its edge, its thickness by the cell, at most the face (the CPU's wallSide). */
fn wSide(w: u32) -> f32 { return min(wall[wThick(wMeta(w) >> 2u)], U.h) * U.h; }
/** Heat into one end of wall w, as the rise k it would give the whole cell (the CPU's wallEnd). */
fn wallEnd(w: u32, c: u32, end: u32, k: f32) {
  let n = U.wallNodes;
  let s = wMeta(w) >> 2u;
  let q = select(0u, n - 1u, end == 1u);
  wall[wNode(w, q)] += (k * wall[wHeat(s)]) / wall[wC(s, q)];
  cf[ix(fTs, c)] += k;
  wall[wLast(w)] += k;
}

/** Change a gas cell's species masses by d (kg each), nitrogen kept; renormalise (the CPU's changeGas). */
fn changeGas(c: u32, d0: array<f32, 6>) {
  var d = d0;
  var s: array<f32, 6>;
  let o = c * NS;
  let mass = cf[ix(fRho, c)] * U.V;
  var n2 = mass;
  var total = 0.0;
  for (var q = 0u; q < NS; q++) {
    let m = Y[o + q] * mass;
    n2 -= m;
    s[q] = max(0.0, m + d[q]);
    total += s[q];
  }
  total += max(0.0, n2);
  if (!(total > 0.0)) { return; }
  for (var q = 0u; q < NS; q++) { Y[o + q] = s[q] / total; }
  cf[ix(fRho, c)] = total / U.V;
}
fn addGas(c: u32, s: u32, dm: f32) {
  if (!(dm > 0.0)) { return; }
  var d = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  d[s] = dm;
  changeGas(c, d);
}
/** Burning dm kg of fuel (Eq. 12 with the fire class's yields). */
fn addProducts(c: u32, st: Fuel, dm: f32, vapour: bool) {
  var d = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  d[1] = -st.o2 * dm;
  d[2] = st.co2 * dm;
  d[5] = st.co * dm;
  d[4] = st.soot * dm;
  d[3] = st.h2o * dm;
  if (vapour) { d[0] = -dm; }
  changeGas(c, d);
}

// ------------------------------------------------------------------ sampling

fn triOf(p: vec3<f32>) -> Tri {
  let x = min(max(p.x - 0.5, 0.0), f32(U.nx) - 1.001);
  let y = min(max(p.y - 0.5, 0.0), f32(U.ny) - 1.001);
  let z = min(max(p.z - 0.5, 0.0), f32(U.nz) - 1.001);
  let i = u32(floor(x)); let j = u32(floor(y)); let k = u32(floor(z));
  var t: Tri;
  t.f = vec3<f32>(x - f32(i), y - f32(j), z - f32(k));
  t.di = select(0u, 1u, i + 1u < U.nx);
  t.dj = select(0u, U.nx, j + 1u < U.ny);
  t.dk = select(0u, U.nx * U.ny, k + 1u < U.nz);
  t.b = i + U.nx * (j + U.ny * k);
  return t;
}
/** Trilinear sample of the gas temperature that gives solid cells no weight (the CPU's sampleGas). */
fn sampleGasT(p: vec3<f32>) -> f32 {
  let x = min(max(p.x - 0.5, 0.0), f32(U.nx) - 1.001);
  let y = min(max(p.y - 0.5, 0.0), f32(U.ny) - 1.001);
  let z = min(max(p.z - 0.5, 0.0), f32(U.nz) - 1.001);
  let i = u32(floor(x)); let j = u32(floor(y)); let k = u32(floor(z));
  let fx = x - f32(i); let fy = y - f32(j); let fz = z - f32(k);
  var sum = 0.0;
  var total = 0.0;
  for (var q = 0u; q < 8u; q++) {
    let a = q & 1u; let b = (q >> 1u) & 1u; let d = (q >> 2u) & 1u;
    let c = at3(min(i + a, U.nx - 1u), min(j + b, U.ny - 1u), min(k + d, U.nz - 1u));
    if (kindOf(c) == SOLID) { continue; }
    let w = select(1.0 - fx, fx, a == 1u) * select(1.0 - fy, fy, b == 1u) * select(1.0 - fz, fz, d == 1u);
    sum += w * cf[ix(fT, c)];
    total += w;
  }
  return select(U.ambient, sum / total, total > 1e-9);
}
${faceSample("Snap", "snap")}
${faceSample("Vel", "vel")}

// ------------------------------------------------------------------ random numbers

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn seedOf(c: u32, salt: u32) -> u32 { return pcg(c ^ pcg(U.stepNo * 8u + salt + pcg(U.seed))); }
fn rnd(s: ptr<function, u32>) -> f32 {
  *s = pcg(*s);
  return f32(*s >> 8u) * (1.0 / 16777216.0);
}

// ------------------------------------------------------------------ ignition source

/** Heskestad's (1983) flame height (m) for Q W over a fire of diameter D m (flamescapes.js flameHeight). */
fn flameHeight(Q: f32, D: f32) -> f32 {
  // pow(0, y) is not defined in WGSL: no heat, no height.
  let kw = max(0.0, Q / 1000.0);
  return 0.235 * select(0.0, pow(kw, 0.4), kw > 0.0) - 1.02 * D;
}

/**
 * Burns up to fuel kg of the item's fuel in cell c, as far as the cell's
 * oxygen allows (flamescapes.js burnInCell): its products, and the
 * convective part of its heat into the gas. Returns the kg burnt.
 */
fn burnIn(c: u32, fuel: f32, st: Fuel) -> f32 {
  let mass = cf[ix(fRho, c)] * U.V;
  let burnt = min(fuel, max(0.0, Y[c * NS + 1u] * mass / st.o2));
  if (!(burnt > 0.0)) { return 0.0; }
  let excess = mass * mixtureCp(c) * (cf[ix(fT, c)] - U.ambient);
  addProducts(c, st, burnt, false);
  cf[ix(fT, c)] = U.ambient + (excess + (1.0 - U.burnerRf) * burnt * st.heat) / (cf[ix(fRho, c)] * U.V * mixtureCp(c));
  return burnt;
}

/**
 * A design-fire first item's step (flamescapes.js burnerStep; any radiant
 * part is not cast as rays here): the table's heat this step and what found
 * no oxygen before burn only with oxygen. Each of the flame's cells (as many
 * as Heskestad's height for that heat reaches, burnerCells) takes an equal
 * share as far as its oxygen allows; what is left goes to the item's other
 * cells, then to the cells beyond them in breadth-first order out to the
 * flame's height; what still finds none waits for the next step. One thread,
 * in the CPU's order, so the cells burn as the CPU's do.
 *
 * In red from U.redBurner: the frame's heat released (J, cleared by
 * readFrame), the last step's, the heat waiting for oxygen (J), and the last
 * step's flame cells.
 */
@compute @workgroup_size(1)
fn burner() {
  let st = fuelOf(1u);
  let E = U.burnerHeat + red[U.redBurner + 2u];
  red[U.redBurner + 1u] = 0.0;
  red[U.redBurner + 3u] = 0.0;
  if (!(E > 0.0)) { return; }
  // burnerCells: the layers Heskestad's height for this heat reaches.
  let L = U.burnerLayers;
  var layer = L;
  if (L > 1u && U.burnerD > 0.0) { layer = min(L, u32(max(1.0, ceil(flameHeight(E / U.dt, U.burnerD) / U.h)))); }
  let n = io[U.burnerLayerAt + layer];
  // What each cell could not burn, summed: exactly zero when all burnt (as the CPU's left).
  let each = E / st.heat / f32(n);
  var left = 0.0;
  for (var q = 0u; q < n; q++) { left += each - burnIn(io[U.burnerAt + q], each, st); }
  for (var q = n; q < U.burnerN && left > 1e-12; q++) { left -= burnIn(io[U.burnerAt + q], left, st); }
  if (left > 1e-12) {
    let far = u32(max(1.0, ceil(flameHeight(E / U.dt, U.burnerDUsed) / U.h)));
    for (var q = 0u; q < U.reachN && left > 1e-12 && io[U.reachAt + U.reachN + q] <= far; q++) {
      left -= burnIn(io[U.reachAt + q], left, st);
    }
  }
  let pending = max(0.0, left) * st.heat;
  let burnt = E - pending;
  red[U.redBurner + 2u] = pending;
  red[U.redBurner + 3u] = f32(n);
  if (!(burnt > 0.0)) { return; }
  red[U.redBurner] += burnt;
  red[U.redBurner + 1u] = burnt;
}

@compute @workgroup_size(1)
fn pilot() {
  // The heat added, tallied as the CPU's stats.pilotHeat (the health guard's energy check).
  var added = 0.0;
  for (var q = 0u; q < U.pilotN; q++) {
    let c = io[q];
    if (U.pilotMode == 0u) {
      if (cf[ix(fTs, c)] < U.pilotTemp) {
        added += solidCap(c) * (U.pilotTemp - cf[ix(fTs, c)]);
        cf[ix(fTs, c)] = U.pilotTemp;
      }
    } else {
      let share = U.pilotPower * U.dt / f32(U.pilotN);
      let C = solidCap(c);
      if (C > 0.0) { cf[ix(fTs, c)] += share / C; }
      else { cf[ix(fT, c)] += share / (cf[ix(fRho, c)] * mixtureCp(c) * U.V); }
      added += share;
    }
  }
  red[U.redPilot] += added;
}

// ------------------------------------------------------------------ ventilation (flamescapes.js ducts)

/**
 * Running ventilation: the CPU's ducts, in the same order. Each system's
 * terminals (io[ductAt + 2q] the cell, + 1 the system, at most 8 systems)
 * draw f = Q dt / V of their cell's gas into the system's ducts and get the
 * same mass of the mixture back. Few terminals, so one thread.
 */
@compute @workgroup_size(1)
fn ducts() {
  let f = min(1.0, U.ductFlow * U.dt / U.V);
  if (f <= 0.0) { return; }
  var mass: array<f32, 8>;
  var heat: array<f32, 8>;
  var spec: array<f32, 48>;
  for (var q = 0u; q < U.ductN; q++) {
    let c = io[U.ductAt + 2u * q];
    let s = min(io[U.ductAt + 2u * q + 1u], 7u);
    let m = f * cf[ix(fRho, c)] * U.V;
    mass[s] += m;
    heat[s] += m * cf[ix(fT, c)];
    for (var i = 0u; i < NS; i++) { spec[s * NS + i] += m * Y[c * NS + i]; }
  }
  for (var q = 0u; q < U.ductN; q++) {
    let c = io[U.ductAt + 2u * q];
    let s = min(io[U.ductAt + 2u * q + 1u], 7u);
    if (!(mass[s] > 0.0)) { continue; }
    cf[ix(fT, c)] = (1.0 - f) * cf[ix(fT, c)] + f * heat[s] / mass[s];
    for (var i = 0u; i < NS; i++) { Y[c * NS + i] = (1.0 - f) * Y[c * NS + i] + f * spec[s * NS + i] / mass[s]; }
  }
}

// ------------------------------------------------------------------ particles (Alg. 1 lines 8 to 14)

${perCell("countBurning", `
  if (ci[ix(iBurning, c)] != 0u) { atomicAdd(&ctr[6], 1u); }`)}

${perCell("spawnFlame", `
  if (ci[ix(iBurning, c)] == 0u) { return; }
  // More spawns than the flame particles' room: each burning cell spawns with
  // the same chance, by the hash of the cell and the step that the CPU's
  // particleStep draws alike (salt 5), not by which thread comes first.
  let burningCells = atomicLoad(&ctr[6]);
  let alive = atomicLoad(&ctr[5]);
  let room = select(0u, U.flameCap - alive, U.flameCap > alive);
  if (6u * burningCells > room) {
    var rs = seedOf(c, 5u);
    if (rnd(&rs) >= f32(room) / f32(6u * burningCells)) { return; }
  }
  let p = vec3<f32>(coords(c)) + 0.5;
  let temp = max(cf[ix(fT, c)], cf[ix(fTs, c)]);
  let base = atomicAdd(&ctr[0], 6u);
  for (var d = 0u; d < 6u; d++) {
    let q = base + d;
    if (q >= U.flameCap) { break; }
    let dv = vec3<f32>(dirv(d)) * U.flameSpeed;
    parts[q] = Particle(vec4<f32>(p, 0.0), vec4<f32>(dv, temp), vec4<f32>(f32(d), 0.0, U.flameLife, 0.0));
  }`)}

/**
 * Flame-front particles may fill at most flameCap slots (the CPU's
 * flameShare of the buffer), so embers always find room: the count drops
 * back to the last slot written (or to where it was, if none could be).
 */
@compute @workgroup_size(1)
fn spawnFlameEnd() {
  let limit = max(atomicLoad(&ctr[5]), U.flameCap);
  if (atomicLoad(&ctr[0]) > limit) { atomicStore(&ctr[0], limit); }
  atomicStore(&ctr[6], 0u);
}

${perCell("spawnEmbers", `
  let loss = cf[ix(fLoss, c)];
  if (!(loss > 0.0)) { return; }
  if (fuelOf(ci[ix(iFuelKind, c)]).embers == 0.0) { return; }
  var rs = seedOf(c, 1u);
  let expected = U.emberRate * loss;
  var count = u32(min(floor(expected), 64.0));
  if (rnd(&rs) < expected - floor(expected)) { count += 1u; }
  let g = vec3<f32>(coords(c));
  let Ts = cf[ix(fTs, c)];
  for (var e = 0u; e < count; e++) {
    let x = vec3<f32>(g.x + rnd(&rs), g.y + 1.0 + rnd(&rs) * 0.5, g.z + rnd(&rs));
    let v = velocityVel(x);
    let m = U.emberMassLo + (U.emberMassHi - U.emberMassLo) * rnd(&rs);
    let q = atomicAdd(&ctr[0], 1u);
    if (q < U.cap) {
      parts[q] = Particle(vec4<f32>(x, 1.0), vec4<f32>(v, max(Ts, 900.0)), vec4<f32>(m, 0.0, U.emberLife, 0.0));
      atomicAdd(&ctr[2], 1u);
    }
  }`)}

${perParticle("moveParticles", `
  var P = parts[t];
  if (P.p.w < 0.0) { return; }
  let x = P.p.xyz;
  let vg = velocityVel(x);
  if (P.p.w == 1.0) {
    // Embers relax toward the drag-scaled gas velocity, settle, cool and burn down.
    let relax = 1.0 - exp(-U.dt / 0.5);
    var v = P.v.xyz + (U.emberDrag * vg - P.v.xyz) * relax;
    v.y -= 0.5 * U.dt;
    let m = P.q.x;
    let r = pow(3.0 * m / (4.0 * PI * 300.0), 1.0 / 3.0);
    let A = 4.0 * PI * r * r;
    let Tg = sampleGasT(x);
    let Tp = P.v.w;
    let Ta = U.ambient;
    let lossE = (U.emberEmis * SIGMA * (Tp * Tp * Tp * Tp - Ta * Ta * Ta * Ta) + U.convection * (Tp - Tg)) * A * U.dt;
    P.v = vec4<f32>(v, max(Tg, Tp - lossE / (m * U.emberHC)));
    P.q.x = m * exp(-U.emberMassLoss * U.dt);
  } else {
    // Flame-front particles: the flame speed along their own direction plus the gas velocity.
    P.v = vec4<f32>(vec3<f32>(dirv(u32(P.q.x))) * U.flameSpeed + vg, P.v.w);
  }
  // The move, in pieces of at most half a cell, stopping in the first solid
  // it enters (the CPU's particleStep): moved whole, a particle faster than a
  // cell a step jumped walls a cell thick. At most 64 pieces on both paths:
  // an unbounded count (a non-finite speed) would hang the GPU.
  let d = P.v.xyz * U.dt / U.h;
  let pieces = clamp(u32(ceil(2.0 * max(abs(d.x), max(abs(d.y), abs(d.z))))), 1u, 64u);
  var np = x;
  var outside = false;
  for (var piece = 0u; piece < pieces; piece++) {
    np += d / f32(pieces);
    outside = np.x < 0.0 || np.y < 0.0 || np.z < 0.0 || np.x >= f32(U.nx) || np.y >= f32(U.ny) || np.z >= f32(U.nz);
    if (outside) { break; }
    if (kindOf(at3(u32(np.x), u32(np.y), u32(np.z))) == SOLID) { break; }
  }
  P.p = vec4<f32>(np, P.p.w);
  P.q.y += U.dt;
  if (outside || P.q.y > P.q.z) {
    P.p.w = -1.0;
  } else if (P.p.w == 0.0 && kindOf(at3(u32(np.x), u32(np.y), u32(np.z))) == SOLID) {
    P.p.w = -1.0;
  }
  parts[t] = P;`)}

/** Supplement Alg. 2: the hottest particle per cell; embers reaching a solid deposit their heat and go. */
${perParticle("particleToGrid", `
  let P = parts[t];
  if (P.p.w < 0.0) { return; }
  let cell = at3(u32(P.p.x), u32(P.p.y), u32(P.p.z));
  atomicMax(&atom[ix(aFlame, cell)], bitcast<u32>(max(P.v.w, 0.0)));
  if (P.p.w != 1.0 || kindOf(cell) == GAS) { return; }
  let m = P.q.x;
  let excess = m * U.emberHC * max(0.0, P.v.w - cf[ix(fTs, cell)]);
  let own = m * fuelOf(0u).heat;
  atomicAdd(&atom[ix(aEmber, cell)], u32(clamp((excess + own) * EMBER_SCALE, 0.0, 4.0e9)));
  atomicMax(&atom[ix(aEmberMax, cell)], u32(clamp(excess * EMBER_SCALE, 0.0, 4.0e9)));
  parts[t].p.w = -1.0;`)}

${perCell("emberDeposit", `
  let Eu = atomPlain[ix(aEmber, c)];
  if (Eu == 0u) { return; }
  let C = solidCap(c);
  if (!(C > 0.0)) { return; }
  var Ts = cf[ix(fTs, c)] + f32(Eu) / EMBER_SCALE / C;
  // Sub-grid spot ignition (the CPU's emberPatch test) by the largest single
  // ember's excess thermal energy m c (T_e - T_s).
  let fu = fuelOf(ci[ix(iFuelKind, c)]);
  let ign = fu.ign;
  if (cf[ix(fFuel, c)] > 0.0 && Ts < ign) {
    let patchC = U.skinDensity * U.emberPatch * fu.cp;
    let Emax = f32(atomPlain[ix(aEmberMax, c)]) / EMBER_SCALE;
    if (Ts + Emax / patchC >= ign) {
      Ts = ign;
      atomicAdd(&ctr[3], 1u);
    }
  }
  cf[ix(fTs, c)] = Ts;`)}

${perParticle("cullFlameFront", `
  let P = parts[t];
  if (P.p.w != 0.0) { return; }
  let cell = at3(u32(P.p.x), u32(P.p.y), u32(P.p.z));
  if (ci[ix(iBurning, cell)] == 0u) { parts[t].p.w = -1.0; }`)}

${perParticle("compact", `
  let P = parts[t];
  if (P.p.w < 0.0) { return; }
  let q = atomicAdd(&ctr[1], 1u);
  partsB[q] = P;`)}

@compute @workgroup_size(1)
fn compactFinish() {
  let n = atomicLoad(&ctr[1]);
  atomicStore(&ctr[0], n);
  atomicStore(&ctr[5], n);
  atomicStore(&ctr[1], 0u);
}

// ------------------------------------------------------------------ radiation (supplement Alg. 1, Eqs. 16, 17)

fn deposit(field: u32, c: u32, dT: f32) {
  let v = i32(round(clamp(dT, -20000.0, 20000.0) * DEP_SCALE));
  atomicAdd(&atom[ix(field, c)], bitcast<u32>(v));
}
/** A ray's rise into one end of wall w (the CPU's dep.lo and dep.hi), in the deposits' fixed point. */
fn wallDeposit(w: u32, end: u32, dT: f32) {
  let v = i32(round(clamp(dT, -20000.0, 20000.0) * DEP_SCALE));
  atomicAdd(&atom[U.wallDepAt + 2u * w + end], bitcast<u32>(v));
}

/**
 * Radiation leaving wall w (cell c at p) over its faces open to gas or the
 * domain's side (the CPU's wallEmit): an end face at its end node, a side
 * face at the cell's mean over its edge's area, each above emitAbove and at
 * most half its node's (or the cell's) heat above ambient; faceE gets each
 * face's energy, which leaves the nodes it came from. Returns the total.
 */
fn wallEmit(w: u32, c: u32, p: vec3<i32>, faceE: ptr<function, array<f32, 6>>) -> f32 {
  let n = U.wallNodes;
  // Cool walls cast nothing.
  if (!(wall[wNode(w, 0u)] > U.emitAbove || wall[wNode(w, n - 1u)] > U.emitAbove || cf[ix(fTs, c)] > U.emitAbove)) { return 0.0; }
  let A = U.h * U.h;
  let side = wSide(w);
  let wm = wMeta(w);
  let axis = wm & 3u;
  let s = wm >> 2u;
  let total = wall[wHeat(s)] * A;
  let Ta = U.ambient;
  let Ta4 = Ta * Ta * Ta * Ta;
  var all = 0.0;
  var sides = 0.0;
  for (var d = 0u; d < 6u; d++) {
    (*faceE)[d] = 0.0;
    let r = p + dirv(d);
    if (inside(r) && kindOf(atI(r)) != GAS) { continue; }
    let end = (d >> 1u) == axis;
    // An odd direction is down the axis: the end at node 0.
    let q = select(n - 1u, 0u, (d & 1u) == 1u);
    var Tf = cf[ix(fTs, c)];
    var cap = total;
    if (end) {
      Tf = wall[wNode(w, q)];
      cap = wall[wC(s, q)] * A;
    }
    if (!(Tf > U.emitAbove)) { continue; }
    var e = cf[ix(fEps, c)] * SIGMA * (Tf * Tf * Tf * Tf - Ta4) * select(side, A, end) * U.dt;
    e = min(e, 0.5 * cap * (Tf - Ta));
    if (!(e > 0.0)) { continue; }
    (*faceE)[d] = e;
    all += e;
    if (end) { wall[wNode(w, q)] -= e / cap; } else { sides += e; }
  }
  if (!(all > 0.0)) { return 0.0; }
  if (sides > 0.0) {
    for (var q = 0u; q < n; q++) { wall[wNode(w, q)] -= sides / total; }
  }
  cf[ix(fTs, c)] -= all / total;
  wall[wLast(w)] -= all / total;
  return all;
}

/** March one ray cell by cell (Amanatides-Woo), depositing its energy (the CPU's march). */
fn march(src: u32, p0: vec3<i32>, dir: vec3<f32>, energy: f32, skipOwn: bool) {
  var q = p0;
  let stepv = select(vec3<i32>(-1), vec3<i32>(1), dir > vec3<f32>(0.0));
  var tDelta = vec3<f32>(1e30);
  var tMax = vec3<f32>(1e30);
  for (var a = 0u; a < 3u; a++) {
    if (dir[a] != 0.0) {
      tDelta[a] = abs(1.0 / dir[a]);
      tMax[a] = 0.5 / abs(dir[a]);
    }
  }
  var left = energy;
  var t = 0.0;
  for (var n = 0u; n < U.rayReach && left > 1e-9 * energy; n++) {
    var a = 2u;
    if (tMax.x < tMax.y) { if (tMax.x < tMax.z) { a = 0u; } } else if (tMax.y < tMax.z) { a = 1u; }
    let seg = (tMax[a] - t) * U.h;
    t = tMax[a];
    let c = atI(q);
    if (kindOf(c) == GAS && c != src) {
      let tau = U.kappa * Y[c * NS + 4u] * seg;
      if (tau > 0.0) {
        let taken = left * omexp(tau);
        deposit(aDepGas, c, taken / (cf[ix(fRho, c)] * mixtureCp(c) * U.V));
        left -= taken;
      }
    }
    tMax[a] += tDelta[a];
    q[a] += stepv[a];
    if (!inside(q)) { return; }
    let next = atI(q);
    if (kindOf(next) != GAS && !(skipOwn && next == src)) {
      let rise = left / max(solidCap(next), 1e-6);
      let w = wallOf(next);
      if (w != NO_CELL && a == (wMeta(w) & 3u)) {
        // A one-dimensional wall reached along its normal takes the ray on the face it crossed.
        wallDeposit(w, select(1u, 0u, stepv[a] > 0), rise);
        return;
      }
      deposit(aDepSolid, next, rise);
      return;
    }
  }
  // Out of reach (or below the cut-off): the rest is absorbed in the cell the
  // ray ended in, so no energy is lost (the CPU's march).
  if (left > 0.0 && inside(q)) {
    let c = atI(q);
    if (kindOf(c) == GAS) { deposit(aDepGas, c, left / (cf[ix(fRho, c)] * mixtureCp(c) * U.V)); }
    else { deposit(aDepSolid, c, left / max(solidCap(c), 1e-6)); }
  }
}

fn radiateCell(c: u32) -> f32 {
  let kc = kindOf(c);
  let p = vec3<i32>(coords(c));
  let Ta = U.ambient;
  let Ta4 = Ta * Ta * Ta * Ta;
  var emitted = 0.0;
  var fromSolid = false;
  var faces: array<u32, 6>;
  var nf = 0u;
  let Ts = cf[ix(fTs, c)];
  let T = cf[ix(fT, c)];
  let w = wallOf(c);
  var faceE = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  if (w != NO_CELL) {
    // A one-dimensional wall: each open face from its own surface (wallEmit).
    emitted = wallEmit(w, c, p, &faceE);
    if (!(emitted > 0.0)) { return 0.0; }
    for (var d = 0u; d < 6u; d++) {
      let q = p + dirv(d);
      if (!inside(q) || kindOf(atI(q)) == GAS) { faces[nf] = d; nf++; }
    }
    fromSolid = true;
  } else if (kc != GAS && Ts > U.emitAbove) {
    // Eq. 16 over the cell's faces open to gas.
    for (var d = 0u; d < 6u; d++) {
      let q = p + dirv(d);
      if (!inside(q) || kindOf(atI(q)) == GAS) { faces[nf] = d; nf++; }
    }
    if (nf == 0u) { return 0.0; }
    let A = U.h * U.h;
    let surface = select(A * f32(nf), max(cf[ix(fArea, c)], A), kc == POROUS);
    let C = solidCap(c);
    emitted = cf[ix(fEps, c)] * SIGMA * (Ts * Ts * Ts * Ts - Ta4) * surface * U.dt;
    emitted = min(emitted, 0.5 * C * (Ts - Ta));
    if (!(emitted > 0.0)) { return 0.0; }
    cf[ix(fTs, c)] = Ts - emitted / C;
    fromSolid = true;
  } else if (kc == GAS && T > U.emitAbove) {
    // Eq. 17: 4 kappa Y_soot sigma (T^4 - T_amb^4) per unit volume.
    let ys = Y[c * NS + 4u];
    if (!(ys > 0.0)) { return 0.0; }
    let Cg = cf[ix(fRho, c)] * mixtureCp(c) * U.V;
    emitted = 4.0 * U.kappa * ys * SIGMA * (T * T * T * T - Ta4) * U.V * U.dt;
    emitted = min(emitted, 0.5 * Cg * (T - Ta));
    if (!(emitted > 0.0)) { return 0.0; }
    cf[ix(fT, c)] = T - emitted / Cg;
  } else {
    return 0.0;
  }
  let share = emitted / f32(U.rays);
  var rs = seedOf(c, 2u);
  for (var r = 0u; r < U.rays; r++) {
    // Uniform on the sphere; from a surface, plus the normal of one open face (Lambert).
    let zc = 2.0 * rnd(&rs) - 1.0;
    let phi = 2.0 * PI * rnd(&rs);
    let s = sqrt(max(0.0, 1.0 - zc * zc));
    var dir = vec3<f32>(s * cos(phi), zc, s * sin(phi));
    if (fromSolid) {
      var f = 0u;
      if (w != NO_CELL) {
        // A one-dimensional wall's face drawn by its share of the emission.
        var u = rnd(&rs) * emitted;
        for (var k = 0u; k < nf; k++) {
          let d = faces[k];
          if (!(faceE[d] > 0.0)) { continue; }
          f = d;
          u -= faceE[d];
          if (u < 0.0) { break; }
        }
      } else {
        f = faces[min(u32(rnd(&rs) * f32(nf)), nf - 1u)];
      }
      dir += vec3<f32>(dirv(f));
      let len = length(dir);
      dir = dir / select(1.0, len, len > 0.0);
    }
    march(c, p, dir, share, fromSolid);
  }
  return emitted;
}

@compute @workgroup_size(64)
fn radEmit(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
           @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let c = lin(gid, nwg);
  var e = 0.0;
  if (c < U.N) { e = radiateCell(c); }
  wsum[li] = vec4<f32>(e, 0.0, 0.0, 0.0);
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) { wsum[li] += wsum[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { red[U.redRad + wid.x + wid.y * nwg.x] = wsum[0].x; }
}

${perCell("radApply", `
  let g = bitcast<i32>(atomPlain[ix(aDepGas, c)]);
  if (g != 0) { cf[ix(fT, c)] += f32(g) / DEP_SCALE; }
  let s = bitcast<i32>(atomPlain[ix(aDepSolid, c)]);
  if (s != 0) { cf[ix(fTs, c)] += f32(s) / DEP_SCALE; }
  // A one-dimensional wall's ends take what reached them (the CPU's wallEnd).
  let w = wallOf(c);
  if (w != NO_CELL) {
    let lo = bitcast<i32>(atomPlain[U.wallDepAt + 2u * w]);
    if (lo != 0) { wallEnd(w, c, 0u, f32(lo) / DEP_SCALE); }
    let hi = bitcast<i32>(atomPlain[U.wallDepAt + 2u * w + 1u]);
    if (hi != 0) { wallEnd(w, c, 1u, f32(hi) / DEP_SCALE); }
  }`)}

// ------------------------------------------------------------------ gas phase (Alg. 1 lines 20 to 23)

/** Midpoint (RK2) back-trace in cell units through the velocity of the step's start. */
fn backTrace(p: vec3<f32>) -> vec3<f32> {
  let s = U.dt / U.h;
  let v1 = velocitySnap(p);
  let m = p - 0.5 * s * v1;
  let v2 = velocitySnap(m);
  return p - s * v2;
}

/** Semi-Lagrangian advection of u, v, w, T and Y; reads the snapshot, writes the live fields. */
${perThread("advect", "U.nu + U.nv + U.nw + U.N", `
  let nx = U.nx; let ny = U.ny;
  if (t < U.nu) {
    let i = t % (nx + 1u); let j = (t / (nx + 1u)) % ny; let k = t / ((nx + 1u) * ny);
    let x = backTrace(vec3<f32>(f32(i), f32(j) + 0.5, f32(k) + 0.5));
    vel[t] = faceSnap(0u, nx + 1u, ny, U.nz, x - vec3<f32>(0.0, 0.5, 0.5));
    return;
  }
  if (t < U.nu + U.nv) {
    let r = t - U.nu;
    let i = r % nx; let j = (r / nx) % (ny + 1u); let k = r / (nx * (ny + 1u));
    let x = backTrace(vec3<f32>(f32(i) + 0.5, f32(j), f32(k) + 0.5));
    vel[t] = faceSnap(U.nu, nx, ny + 1u, U.nz, x - vec3<f32>(0.5, 0.0, 0.5));
    return;
  }
  if (t < U.nu + U.nv + U.nw) {
    let r = t - U.nu - U.nv;
    let i = r % nx; let j = (r / nx) % ny; let k = r / (nx * ny);
    let x = backTrace(vec3<f32>(f32(i) + 0.5, f32(j) + 0.5, f32(k)));
    vel[t] = faceSnap(U.nu + U.nv, nx, ny, U.nz + 1u, x - vec3<f32>(0.5, 0.5, 0.0));
    return;
  }
  // Temperature and the six species share one back-trace and one set of
  // eight trilinear weights per cell (the CPU's advect). Corners inside
  // solids hold no gas: they get no weight and the rest are renormalised; a
  // cell whose back-trace lands among solids only keeps its own state. With
  // the conservative transport (proposal F) the transport moves them instead.
  let c = t - U.nu - U.nv - U.nw;
  if (U.conservative != 0u || kindOf(c) == SOLID) { return; }
  let x = backTrace(vec3<f32>(coords(c)) + 0.5);
  let r = triOf(x);
  let fx = r.f.x; let fy = r.f.y; let fz = r.f.z;
  var w = array<f32, 8>(
    (1.0 - fx) * (1.0 - fy) * (1.0 - fz), fx * (1.0 - fy) * (1.0 - fz),
    (1.0 - fx) * fy * (1.0 - fz), fx * fy * (1.0 - fz),
    (1.0 - fx) * (1.0 - fy) * fz, fx * (1.0 - fy) * fz,
    (1.0 - fx) * fy * fz, fx * fy * fz);
  var n = array<u32, 8>(r.b, r.b + r.di, r.b + r.dj, r.b + r.dj + r.di, r.b + r.dk, r.b + r.dk + r.di, r.b + r.dk + r.dj, r.b + r.dk + r.dj + r.di);
  var total = 0.0;
  for (var q = 0u; q < 8u; q++) {
    if (kindOf(n[q]) == SOLID) { w[q] = 0.0; }
    total += w[q];
  }
  if (!(total > 1e-9)) { return; }
  let sT = U.nu + U.nv + U.nw;
  let sY = sT + U.N;
  var T = 0.0;
  for (var q = 0u; q < 8u; q++) { T += w[q] * snap[sT + n[q]]; }
  cf[ix(fT, c)] = T / total;
  for (var s = 0u; s < NS; s++) {
    var y = 0.0;
    for (var q = 0u; q < 8u; q++) { y += w[q] * snap[sY + n[q] * NS + s]; }
    Y[c * NS + s] = y / total;
  }`)}

${perCell("density", `
  if (kindOf(c) == SOLID) { return; }
  cf[ix(fRho, c)] = gasDensity(cf[ix(fT, c)], c);`)}

// ------------------------------------------------------------------ proposal F: the conserved transport and the expansion

/** The step's starting gas (the snapshot): its mass fractions, and its density by the gas law. */
fn yStart(c: u32, s: u32) -> f32 { return snap[U.nu + U.nv + U.nw + U.N + c * NS + s]; }
fn rhoStart(c: u32) -> f32 {
  var inv = 0.0;
  var rest = 1.0;
  for (var s = 0u; s < NS; s++) {
    let y = yStart(c, s);
    inv += y / molar(s);
    rest -= y;
  }
  inv += max(0.0, rest) / 0.028;
  return P_AMBIENT / (R_GAS * snap[U.nu + U.nv + U.nw + c] * inv);
}

// The gas's transport (the CPU's transportGas): the first axis from the
// step's start into the scratch state, y from it into the live fields, the
// last axis back into the scratch, then the cells take their mass fractions
// and, by the gas law, their T. x first on even steps, z first on odd ones.
${transportSweep(0, "start", "scratch")}
${transportSweep(2, "start", "scratch")}
${transportSweep(1, "scratch", "live")}
${transportSweep(2, "live", "scratch")}
${transportSweep(0, "live", "scratch")}
${perCell("transportEnd", `
  if (kindOf(c) == SOLID) { return; }
  for (var q = 0u; q < NS; q++) { Y[c * NS + q] = snap[U.gasAt + U.N + c * NS + q]; }
  cf[ix(fT, c)] = gasDensity(1.0, c) / snap[U.gasAt + c];`)}

/**
 * The gas's expansion (the CPU's expansionBeforeAdvect and expansionEnd): the
 * change of T the heat sources made before the advection, from the step's
 * starting T (copied to snap at U.expandAt), and after it, from the T the
 * advection left (the snapshot diffusion reads); the domain's shell expands nothing.
 */
${perCell("expandBefore", `
  if (kindOf(c) == SOLID) { cf[ix(fD, c)] = 0.0; return; }
  cf[ix(fD, c)] = log(cf[ix(fT, c)] / snap[U.expandAt + c]);`)}
${perCell("expandEnd", `
  if (kindOf(c) == SOLID) { return; }
  let p = coords(c);
  if (p.x == 0u || p.z == 0u || p.x == U.nx - 1u || p.z == U.nz - 1u || p.y == U.ny - 1u) { cf[ix(fD, c)] = 0.0; return; }
  cf[ix(fD, c)] = (cf[ix(fD, c)] + log(cf[ix(fT, c)] / snap[U.nu + U.nv + U.nw + c])) / U.dt;`)}

/**
 * A sealed region's mean expansion, which its gas has nowhere to take
 * (the CPU's project): one workgroup per region, over its cells listed in io
 * (offsets from U.sealedAt, then the cells), into red at U.redMean. The sum
 * is a tree in float32, where the CPU sums in cell order in float64.
 */
@compute @workgroup_size(64)
fn sealedMean(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let s = wid.x + wid.y * nwg.x;
  if (s >= U.sealedN) { return; }
  let start = io[U.sealedAt + s];
  let end = io[U.sealedAt + s + 1u];
  let list = U.sealedAt + U.sealedN + 1u;
  var acc = 0.0;
  for (var q = start + li; q < end; q += 64u) { acc += cf[ix(fD, io[list + q])]; }
  wheat[li] = acc;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w >>= 1u) {
    if (li < w) { wheat[li] = wheat[li] + wheat[li + w]; }
    workgroupBarrier();
  }
  if (li == 0u) { red[U.redMean + s] = wheat[0] / f32(end - start); }
}

/** Buoyancy, Eq. 6's (1 - rho_0 / rho) g on the vertical faces. */
${perThread("buoyancy", "U.nv", `
  let i = t % U.nx; let j = (t / U.nx) % (U.ny + 1u); let k = t / (U.nx * (U.ny + 1u));
  if (j < 1u || j >= U.ny) { return; }
  let a = at3(i, j - 1u, k);
  let b = at3(i, j, k);
  if (kindOf(a) == SOLID || kindOf(b) == SOLID) { return; }
  let r = 0.5 * (cf[ix(fRho, a)] + cf[ix(fRho, b)]);
  vel[U.nu + t] += U.dt * U.gravity * (U.rho0 / r - 1.0);`)}

fn cu(i: u32, j: u32, k: u32) -> f32 { return 0.5 * (vel[uAt(i, j, k)] + vel[uAt(i + 1u, j, k)]); }
fn cv(i: u32, j: u32, k: u32) -> f32 { return 0.5 * (vel[vAt(i, j, k)] + vel[vAt(i, j + 1u, k)]); }
fn cw(i: u32, j: u32, k: u32) -> f32 { return 0.5 * (vel[wAt(i, j, k)] + vel[wAt(i, j, k + 1u)]); }

/** Vorticity confinement, 1: the curl of the cell-centred velocity. */
${perCell("vortCurl", `
  let p = coords(c);
  var wx = 0.0; var wy = 0.0; var wz = 0.0;
  let interior = p.x >= 1u && p.y >= 1u && p.z >= 1u && p.x + 1u < U.nx && p.y + 1u < U.ny && p.z + 1u < U.nz;
  if (interior && kindOf(c) != SOLID) {
    let i = p.x; let j = p.y; let k = p.z;
    let h2 = 2.0 * U.h;
    wx = (cw(i, j + 1u, k) - cw(i, j - 1u, k) - cv(i, j, k + 1u) + cv(i, j, k - 1u)) / h2;
    wy = (cu(i, j, k + 1u) - cu(i, j, k - 1u) - cw(i + 1u, j, k) + cw(i - 1u, j, k)) / h2;
    wz = (cv(i + 1u, j, k) - cv(i - 1u, j, k) - cu(i, j + 1u, k) + cu(i, j - 1u, k)) / h2;
  }
  cf[ix(fWx, c)] = wx; cf[ix(fWy, c)] = wy; cf[ix(fWz, c)] = wz;
  cf[ix(fMag, c)] = sqrt(wx * wx + wy * wy + wz * wz);`)}

/** Vorticity confinement, 2: the confinement force per cell. */
${perCell("vortForce", `
  let p = coords(c);
  var f = vec3<f32>(0.0);
  let inner = p.x >= 2u && p.y >= 2u && p.z >= 2u && p.x + 2u < U.nx && p.y + 2u < U.ny && p.z + 2u < U.nz;
  let mag = cf[ix(fMag, c)];
  if (inner && kindOf(c) != SOLID && mag > 0.0) {
    let sx = 1u; let sy = U.nx; let sz = U.nx * U.ny;
    let h2 = 2.0 * U.h;
    var Nv = vec3<f32>(
      (cf[ix(fMag, c + sx)] - cf[ix(fMag, c - sx)]) / h2,
      (cf[ix(fMag, c + sy)] - cf[ix(fMag, c - sy)]) / h2,
      (cf[ix(fMag, c + sz)] - cf[ix(fMag, c - sz)]) / h2);
    let len = sqrt(Nv.x * Nv.x + Nv.y * Nv.y + Nv.z * Nv.z);
    if (len > 1e-9) {
      Nv = Nv / len;
      let s = U.vorticity * U.h * U.dt;
      let wx = cf[ix(fWx, c)]; let wy = cf[ix(fWy, c)]; let wz = cf[ix(fWz, c)];
      f = vec3<f32>(s * (Nv.y * wz - Nv.z * wy), s * (Nv.z * wx - Nv.x * wz), s * (Nv.x * wy - Nv.y * wx));
    }
  }
  cf[ix(fFx, c)] = f.x; cf[ix(fFy, c)] = f.y; cf[ix(fFz, c)] = f.z;`)}

/** Vorticity confinement, 3: each face gathers half the force of each of its two cells. */
${perThread("vortApply", "U.nu + U.nv + U.nw", `
  let nx = U.nx; let ny = U.ny; let nz = U.nz;
  if (t < U.nu) {
    let i = t % (nx + 1u); let j = (t / (nx + 1u)) % ny; let k = t / ((nx + 1u) * ny);
    var u = vel[t];
    if (i >= 1u) { u += 0.5 * cf[ix(fFx, at3(i - 1u, j, k))]; }
    if (i < nx) { u += 0.5 * cf[ix(fFx, at3(i, j, k))]; }
    vel[t] = u;
  } else if (t < U.nu + U.nv) {
    let r = t - U.nu;
    let i = r % nx; let j = (r / nx) % (ny + 1u); let k = r / (nx * (ny + 1u));
    var v = vel[t];
    if (j >= 1u) { v += 0.5 * cf[ix(fFy, at3(i, j - 1u, k))]; }
    if (j < ny) { v += 0.5 * cf[ix(fFy, at3(i, j, k))]; }
    vel[t] = v;
  } else {
    let r = t - U.nu - U.nv;
    let i = r % nx; let j = (r / nx) % ny; let k = r / (nx * ny);
    var w = vel[t];
    if (k >= 1u) { w += 0.5 * cf[ix(fFz, at3(i, j, k - 1u))]; }
    if (k < nz) { w += 0.5 * cf[ix(fFz, at3(i, j, k))]; }
    vel[t] = w;
  }`)}

/** rho D of an open cell at its snapshot temperature (Eq. 11's diffusivity, scaled by (T/T0)^1.75). */
fn rhoD(c: u32, sT: u32) -> f32 {
  return cf[ix(fRho, c)] * U.diffusivity * pow(snap[sT + c] / U.ambient, 1.75);
}

/**
 * Explicit diffusion of species (Eq. 11) and heat between open cells, from
 * the snapshot (Jacobi, as the CPU). A face's rho D is the mean of its two
 * cells', so the species flux is the same seen from both sides.
 */
${perCell("diffuse", `
  let mask = ci[ix(iMask, c)];
  if ((mask & 63u) == 0u) { return; }
  let nuvw = U.nu + U.nv + U.nw;
  let sT = nuvw;
  let sY = nuvw + U.N;
  let Tc = snap[sT + c];
  let rho = cf[ix(fRho, c)];
  let rDc = rhoD(c, sT);
  let alpha = U.conductivity / (rho * mixtureCpSnap(c, sY));
  let h2 = U.h * U.h;
  let lamT = min(alpha * U.dt / h2, 1.0 / 7.0);
  var y: array<f32, 6>;
  for (var s = 0u; s < NS; s++) { y[s] = snap[sY + c * NS + s]; }
  var T = Tc;
  for (var d = 0u; d < 6u; d++) {
    if ((mask & (1u << d)) == 0u) { continue; }
    let n = u32(i32(c) + noff(d));
    let lamD = min((0.5 * (rDc + rhoD(n, sT)) * U.dt) / (rho * h2), 1.0 / 7.0);
    for (var s = 0u; s < NS; s++) { y[s] += lamD * (snap[sY + n * NS + s] - snap[sY + c * NS + s]); }
    T += lamT * (snap[sT + n] - Tc);
  }
  for (var s = 0u; s < NS; s++) { Y[c * NS + s] = y[s]; }
  cf[ix(fT, c)] = T;`)}

// ------------------------------------------------------------------ sprinklers (suppression.js)

/**
 * Evans's cap on cell c this step: the factor (at most 1) for its reactions
 * of total loss (kg) uncapped. Armed (rate -1): this step's rate is read
 * and nothing is capped; a cell not reacting then has rate 0 and never
 * ignites. The CPU's evansFactor.
 */
fn evans(c: u32, loss: f32) -> f32 {
  if (U.supOn == 0u) { return 1.0; }
  let tau = sup[c];
  if (!(tau > 0.0)) { return 1.0; }
  let r = sup[2u * U.N + c];
  if (r < 0.0) {
    sup[2u * U.N + c] = max(0.0, loss) / U.dt;
    return 1.0;
  }
  if (!(loss > 0.0)) { return 1.0; }
  let allowed = r * exp(-(U.time - sup[U.N + c]) / tau) * U.dt;
  return select(1.0, allowed / loss, loss > allowed);
}

/**
 * The spray, one thread per column (no two columns share a cell or a
 * face): down its open cells the water evaporates a share
 * 1 - exp(-eta (T - T_boil) h) of what is left, at most what cools the cell
 * to boiling; the gas keeps its energy with the vapour carrying its own
 * heat, the steam joins it, and the water drags the gas down. The CPU's
 * spray (flamescapes.js).
 */
${perThread("spray", "U.sprayN", `
  let base = U.supCol + 4u * t;
  var c = bitcast<u32>(sup[base]);
  let count = bitcast<u32>(sup[base + 1u]);
  var m = sup[base + 2u] * U.dt;
  let TB = 373.15;
  let e = 4186.0 * (TB - 293.15) + U.latent;
  let drag = abs(U.gravity) / (5.0 * U.h * U.h);
  for (var n = 0u; n < count; n++) {
    if (!(m > 0.0) || kindOf(c) == SOLID) { break; }
    let T = cf[ix(fT, c)];
    if (T > TB) {
      let Cg = gasCap(c);
      let dv = min(omexp(U.sprayEta * (T - TB) * U.h) * m, (Cg * (T - TB)) / e);
      if (dv > 0.0) {
        cf[ix(fT, c)] = (Cg * T - dv * (e - 1864.0 * TB)) / (Cg + 1864.0 * dv);
        addGas(c, 3u, dv);
        m -= dv;
      }
    }
    let i = c % U.nx;
    let j = (c / U.nx) % U.ny;
    let k = c / (U.nx * U.ny);
    vel[vAt(i, j, k)] -= (drag * m) / max(cf[ix(fRho, c)], 1e-6);
    c -= U.nx;
  }
  // Wetting ("wetting" mode): the column ends on the fuel's top surface; a
  // share of the water left soaks into that cell, mixed in at 20 C.
  if (U.sprayWet > 0.0 && m > 0.0 && c < U.nx * U.ny * U.nz && kindOf(c) == POROUS && cf[ix(fFuel, c)] > 0.0 && waterBit(fuelOf(ci[ix(iFuelKind, c)]), 1u)) {
    let st = fuelOf(ci[ix(iFuelKind, c)]);
    let moist = cf[ix(fMoist, c)];
    let Cs = capOf(cf[ix(fInertC, c)], cf[ix(fFuel, c)], cf[ix(fArea, c)], moist, st.cp);
    let dw = U.sprayWet * m;
    cf[ix(fTs, c)] = (Cs * cf[ix(fTs, c)] + dw * 4186.0 * 293.15) / (Cs + dw * 4186.0);
    cf[ix(fMoist, c)] = moist + dw;
  }`)}

// ------------------------------------------------------------------ systems (suppressionSystems.js)

/** Cell c's flooded or foamed room (0: none, and always 0 when the run has no systems). */
fn roomOf(c: u32) -> u32 {
  if (U.sysOn == 0u) { return 0u; }
  return bitcast<u32>(sys[c]);
}
/** Room r's state this step: 0 agent %, 1 and 2 class A and B extinguishing %, 3 foam's share of evaporation left, 4 foam's wetting share. */
fn roomState(r: u32, k: u32) -> f32 { return sys[U.sysState + 8u * r + k]; }
/** Cell c's agent at or above its vapour's extinguishing concentration (heptane's for a liquid, class A's otherwise). The CPU's flooded. */
fn flooded(c: u32) -> bool {
  let r = roomOf(c);
  if (r == 0u) { return false; }
  let ext = select(roomState(r, 1u), roomState(r, 2u), fuelOf(ci[ix(iVapour, c)]).liquid != 0.0);
  return roomState(r, 0u) >= ext;
}

/**
 * Foam-water sprinklers' water on class A fuel, one thread per wet cell (no
 * two share a cell): while the room's foam is applied, the cell takes the
 * wetting share of the solution on its column, mixed in at 20 C. The CPU's
 * foamWet (flamescapes.js).
 */
${perThread("systemsWet", "U.sysWetN", `
  let base = U.sysWetAt + 4u * t;
  let c = bitcast<u32>(sys[base]);
  let share = roomState(bitcast<u32>(sys[base + 1u]), 4u);
  if (!(share > 0.0) || kindOf(c) == SOLID || !(cf[ix(fFuel, c)] > 0.0)) { return; }
  let st = fuelOf(ci[ix(iFuelKind, c)]);
  if (!waterBit(st, 1u)) { return; }
  let moist = cf[ix(fMoist, c)];
  let Cs = capOf(cf[ix(fInertC, c)], cf[ix(fFuel, c)], cf[ix(fArea, c)], moist, st.cp);
  let dw = share * sys[base + 2u] * U.dt;
  cf[ix(fTs, c)] = (Cs * cf[ix(fTs, c)] + dw * 4186.0 * 293.15) / (Cs + dw * 4186.0);
  cf[ix(fMoist, c)] = moist + dw;`)}

// ------------------------------------------------------------------ chemistry (Alg. 1 lines 24 to 27)

/**
 * Evaporation, oxidation (Eq. 15), oxygen-limited pyrolysis (Eq. 14) and gas
 * combustion (Eqs. 9, 10, 12) of one cell, as the CPU's chemistry. A porous
 * cell trades species with its own gas; solid cells hold no fuel (the runner
 * checks this). Returns (heat released, fuel burnt).
 */
fn chemCell(c: u32) -> vec2<f32> {
  var released = 0.0;
  var burnt = 0.0;
  let kc = kindOf(c);
  ci[ix(iBurning, c)] = 0u;
  cf[ix(fLoss, c)] = 0.0;
  let dt = U.dt;
  if (kc == POROUS && cf[ix(fFuel, c)] > 0.0) {
    let fk = ci[ix(iFuelKind, c)];
    let st = fuelOf(fk);
    var inertC = cf[ix(fInertC, c)];
    let area = cf[ix(fArea, c)];
    var fuel = cf[ix(fFuel, c)];
    var moist = cf[ix(fMoist, c)];
    var Ts = cf[ix(fTs, c)];
    let C = capOf(inertC, fuel, area, moist, st.cp);
    let gasMass = cf[ix(fRho, c)] * U.V;
    let o2 = Y[c * NS + 1u] * gasMass;
    // Evaporation: water leaves with the heat above 100 C.
    if (moist > 0.0 && Ts > 373.15) {
      let dw = min(moist, (C * (Ts - 373.15)) / U.latent);
      moist -= dw;
      Ts -= (dw * U.latent) / C;
      addGas(c, 3u, dw);
    }
    if (st.liquid != 0.0) {
      // A liquid first item (fire classes B and K) evaporates instead: the
      // heat above its boiling point turns its surface into fuel vapour.
      if (Ts > st.boiling) {
        // At most the pool's large-pool burning rate (flamescapes.js poolCap).
        var cap = 3.0e38;
        if (st.mInf > 0.0) {
          var size = 1.0;
          if (st.kBeta > 0.0) { size = 1.0 - exp(-st.kBeta * sqrt(4.0 * area / 3.14159265)); }
          cap = st.mInf * size * area * dt;
        }
        var dv = min(min(skinOf(fuel, area), (C * (Ts - st.boiling)) / st.latent), cap);
        if (waterBit(st, 2u)) { dv *= evans(c, dv); }
        // Foam over the pool: the share of its evaporation the blanket leaves.
        let rl = roomOf(c);
        if (rl != 0u) {
          let left = roomState(rl, 3u);
          if (left < 1.0) { dv *= left; }
        }
        if (dv > 0.0) {
          fuel -= dv;
          Ts -= (dv * st.latent) / C;
          addGas(c, 0u, dv);
          ci[ix(iVapour, c)] = fk;
          cf[ix(fLoss, c)] = dv;
          if (fuel <= 1e-7) {
            fuel = 0.0;
            if (inertC == 0.0) { ci[ix(iKind, c)] = GAS; }
          }
        }
      }
    } else {
    // Oxidation, Eq. 15, then pyrolysis, Eq. 14, suppressed as oxygen becomes available.
    let skin = skinOf(fuel, area);
    let yO = Y[c * NS + 1u];
    let ko = arrhenius(U.oxA, U.oxE, Ts) * (yO / Y_O2_AIR);
    var dOx = skin * omexp(max(0.0, ko) * dt);
    dOx = min(dOx, o2 / st.o2);
    // Clean agent at or above the class A extinguishing concentration: no oxidation.
    let ro = roomOf(c);
    if (ro != 0u && roomState(ro, 0u) >= roomState(ro, 1u)) { dOx = 0.0; }
    let starved = 1.0 - min(1.0, yO / Y_O2_AIR);
    let kp = arrhenius(U.pyA, U.pyE, Ts) * (0.25 + 0.75 * starved);
    var dPy = max(0.0, skin - dOx) * omexp(max(0.0, kp) * dt);
    // Evans's cap under an open sprinkler: both reactions scaled alike
    // (solids only: fireClasses.js waterResponse).
    var f = 1.0;
    if (waterBit(st, 2u)) { f = evans(c, dOx + dPy); }
    dOx *= f;
    dPy *= f;
    if (dOx > 0.0) {
      fuel -= dOx;
      // A metal's oxide stays as ash (MgO, about 900 J/(kg K)).
      if (st.ash > 0.0) {
        inertC += st.ash * dOx * 900.0;
        cf[ix(fInertC, c)] = inertC;
      }
      let heat = dOx * st.heat;
      Ts += heat / max(capOf(inertC, fuel, area, moist, st.cp), 1e-6);
      addProducts(c, st, dOx, false);
      released += heat;
      burnt += dOx;
      // flamescapes.js's burning floor (DEFAULTS.burningFlux).
      if (heat >= U.burnFlux * area * dt) { ci[ix(iBurning, c)] = 1u; }
    }
    if (dPy > 0.0) {
      fuel -= dPy;
      Ts -= (dPy * U.pyHeat) / max(capOf(inertC, fuel, area, moist, st.cp), 1e-6);
      addGas(c, 0u, dPy * U.pyFuel);
      addGas(c, 4u, dPy * (1.0 - U.pyFuel));
      ci[ix(iVapour, c)] = fk;
    }
    cf[ix(fLoss, c)] = dOx + dPy;
    // Fresh fuel from the bulk replaces what reacted, at ambient temperature and with its water.
    let fresh = min(dOx + dPy, max(0.0, fuel - (skin - dOx - dPy)));
    if (fresh > 0.0) {
      let water = select(fresh * U.moistFrac, 0.0, waterBit(st, 4u));
      let Cfresh = fresh * st.cp + water * 4186.0;
      moist += water;
      let Cnow = capOf(inertC, fuel, area, moist, st.cp);
      Ts = (Ts * max(0.0, Cnow - Cfresh) + U.ambient * Cfresh) / max(Cnow, 1e-9);
    }
    if (fuel <= 1e-7) {
      fuel = 0.0;
      if (inertC == 0.0) { ci[ix(iKind, c)] = GAS; }
    }
    }
    cf[ix(fFuel, c)] = fuel;
    cf[ix(fMoist, c)] = moist;
    cf[ix(fTs, c)] = Ts;
  }
  if (kc != SOLID) {
    let yF = Y[c * NS];
    let yO = Y[c * NS + 1u];
    // Clean agent at or above the vapour's extinguishing concentration: no flame (tested only where a flame could be).
    if (yF > 1e-7 && yO > 1e-6 && !flooded(c)) {
      let rho = cf[ix(fRho, c)];
      let mass = rho * U.V;
      let st = fuelOf(ci[ix(iVapour, c)]);
      // Flame-front particles and embers supersede the cell's temperature when hotter.
      let Teff = max(cf[ix(fT, c)], bitcast<f32>(atomPlain[ix(aFlame, c)]));
      let cF = (rho * yF) / 0.03;
      let cO = (rho * yO) / 0.032;
      let omega = arrhenius(U.gasA, U.gasE, Teff) * pow(cF, U.gasFuelExp - 1.0) * pow(cO, U.gasOxExp);
      var dF = yF * mass * omexp(max(0.0, omega) * dt);
      dF = min(dF, (yO * mass) / st.o2);
      if (dF > 1e-12) {
        addProducts(c, st, dF, true);
        let heat = dF * st.heat;
        cf[ix(fT, c)] += heat / (mass * mixtureCp(c));
        released += heat;
        burnt += dF;
        ci[ix(iBurning, c)] = 1u;
      }
    }
  }
  return vec2<f32>(released, burnt);
}

@compute @workgroup_size(64)
fn chemistry(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let c = lin(gid, nwg);
  var r = vec2<f32>(0.0);
  if (c < U.N) { r = chemCell(c); }
  wsum[li] = vec4<f32>(r, 0.0, 0.0);
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) { wsum[li] += wsum[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    let w = wid.x + wid.y * nwg.x;
    red[2u * w] = wsum[0].x;
    red[2u * w + 1u] = wsum[0].y;
  }
}

/** Sum the per-workgroup partials into this step's slot: released, burnt, radiated. */
@compute @workgroup_size(64)
fn reduceFinish(@builtin(local_invocation_index) li: u32) {
  var s = vec4<f32>(0.0);
  for (var w = li; w < U.nPart; w += 64u) {
    s += vec4<f32>(red[2u * w], red[2u * w + 1u], red[U.redRad + w], 0.0);
  }
  wsum[li] = s;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) {
    if (li < k) { wsum[li] += wsum[li + k]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    let o = U.redSteps + 4u * U.slot;
    red[o] += wsum[0].x;
    red[o + 1u] += wsum[0].y;
    red[o + 2u] += wsum[0].z;
    red[o + 3u] += 1.0;
  }
}

// ------------------------------------------------------------------ heat transfer (Alg. 1 line 30)

${perCell("conductPorous", `
  if (kindOf(c) != POROUS) { return; }
  let Cs = solidCap(c);
  if (!(Cs > 0.0)) { return; }
  let Cg = gasCap(c);
  let A = U.h * U.h;
  let q = exchange(cf[ix(fTs, c)], cf[ix(fT, c)], Cs, Cg, U.convection * max(cf[ix(fArea, c)], A), U.dt);
  cf[ix(fTs, c)] -= q / Cs;
  cf[ix(fT, c)] += q / Cg;`)}

/** The solid-gas interface (Eq. 19) between solid s and gas g: the fuel's coefficient, or the wall's for an inert solid. */
fn conductGas(s: u32, g: u32, Cs: f32) {
  let Cg = gasCap(g);
  let hc = select(U.convection, U.wallConvection, kindOf(s) == SOLID);
  let q = exchange(cf[ix(fTs, s)], cf[ix(fT, g)], Cs, Cg, hc * U.h * U.h, U.dt);
  cf[ix(fTs, s)] -= q / Cs;
  cf[ix(fT, g)] += q / Cg;
}

/** Solid c's +axis face to n: solid-solid conduction (Eq. 18) or the interface with gas (Eq. 19). */
fn conductForward(c: u32, n: u32) {
  if (kindOf(c) == GAS) { return; }
  // A one-dimensional wall exchanges through its own profile (walls), with no solid beside it.
  if (wallOf(c) != NO_CELL || wallOf(n) != NO_CELL) { return; }
  let Cs = solidCap(c);
  if (!(Cs > 0.0)) { return; }
  if (kindOf(n) == GAS) {
    conductGas(c, n, Cs);
    return;
  }
  let Cn = solidCap(n);
  // A porous cell whose contents mass was drawn as zero holds no heat; the
  // CPU's q / Cn makes its Ts 0 / 0 = NaN there (it stays in that cell).
  if (!(Cn > 0.0)) { return; }
  let kk = 0.5 * (max(cf[ix(fInertK, c)], 0.1) + max(cf[ix(fInertK, n)], 0.1));
  let q = exchange(cf[ix(fTs, c)], cf[ix(fTs, n)], Cs, Cn, (kk * U.h * U.h) / U.h, U.dt);
  cf[ix(fTs, c)] -= q / Cs;
  cf[ix(fTs, n)] += q / Cn;
}

/** Solid c's -axis face to n, when n is gas (a solid there met c across its own +axis face). */
fn conductBackward(c: u32, n: u32) {
  if (kindOf(c) == GAS || kindOf(n) != GAS || wallOf(c) != NO_CELL) { return; }
  let Cs = solidCap(c);
  if (!(Cs > 0.0)) { return; }
  conductGas(c, n, Cs);
}
${conductPass(0, 0)}
${conductPass(0, 1)}
${conductPass(1, 0)}
${conductPass(1, 1)}
${conductPass(2, 0)}
${conductPass(2, 1)}
${conductBack(0)}
${conductBack(1)}
${conductBack(2)}

/**
 * One-dimensional walls (the CPU's walls, calibration plan 13.4 item 1), one
 * thread per wall: what other stages added to its cell's mean spread over
 * its nodes; its side faces' exact exchange of its mean with the gas beside
 * them over its edge's area; then its profile's backward-Euler step by the
 * Thomas algorithm (the CPU's wallStep), each end exchanging with the gas
 * beyond it, the gas held at its temperature at the stage's start. The heat
 * each face gave its gas waits in the wall's Q for wallsApply, so no two
 * threads write one gas cell.
 */
${perThread("walls", "U.wallN", `
  let w = t;
  let c = bitcast<u32>(wall[w]);
  let wm = wMeta(w);
  let axis = wm & 3u;
  let s = wm >> 2u;
  let n = U.wallNodes;
  let A = U.h * U.h;
  let total = wall[wHeat(s)] * A;
  let hc = U.wallConvection;
  var T: array<f32, WALL_MAX>;
  let added = cf[ix(fTs, c)] - wall[wLast(w)];
  for (var q = 0u; q < n; q++) {
    T[q] = wall[wNode(w, q)];
    if (added != 0.0) { T[q] += added; }
  }
  var mean = 0.0;
  for (var q = 0u; q < n; q++) { mean += wall[wC(s, q)] * T[q]; }
  mean = (mean * A) / total;
  let p = vec3<i32>(coords(c));
  let side = wSide(w);
  var sides = 0.0;
  for (var d = 0u; d < 6u; d++) {
    wall[wQ(w, d)] = 0.0;
    if ((d >> 1u) == axis) { continue; }
    let r = p + dirv(d);
    if (!inside(r) || kindOf(atI(r)) == SOLID) { continue; }
    let nb = atI(r);
    let q = exchange(mean, cf[ix(fT, nb)], total, gasCap(nb), hc * side, U.dt);
    wall[wQ(w, d)] = q;
    sides += q;
  }
  if (sides != 0.0) {
    for (var q = 0u; q < n; q++) { T[q] -= sides / total; }
  }
  // Its ends: down the axis (an odd direction) at node 0, up it at node n - 1; insulated against a solid or the domain's side.
  let rl = p + dirv(2u * axis + 1u);
  let rh = p + dirv(2u * axis);
  var hLo = 0.0; var tLo = 0.0; var hHi = 0.0; var tHi = 0.0;
  if (inside(rl) && kindOf(atI(rl)) != SOLID) { hLo = hc; tLo = cf[ix(fT, atI(rl))]; }
  if (inside(rh) && kindOf(atI(rh)) != SOLID) { hHi = hc; tHi = cf[ix(fT, atI(rh))]; }
  // The nodes' changes (the delta form of the CPU's wallStep): its right-hand side of differences keeps a level wall level in float32.
  var cp: array<f32, WALL_MAX>;
  var dp: array<f32, WALL_MAX>;
  for (var i = 0u; i < n; i++) {
    var a = 0.0;
    if (i > 0u) { a = -wall[wG(s, i - 1u)]; }
    var cc = 0.0;
    if (i < n - 1u) { cc = -wall[wG(s, i)]; }
    let tt = T[i];
    var b = wall[wC(s, i)] / U.dt - a - cc;
    var dd = 0.0;
    if (i > 0u) { dd = a * (tt - T[i - 1u]); }
    if (i < n - 1u) { dd += cc * (tt - T[i + 1u]); }
    if (i == 0u) { b += hLo; dd += hLo * (tLo - tt); }
    if (i == n - 1u) { b += hHi; dd += hHi * (tHi - tt); }
    var m = b;
    if (i > 0u) { m = b - a * cp[i - 1u]; }
    cp[i] = cc / m;
    if (i > 0u) { dp[i] = (dd - a * dp[i - 1u]) / m; } else { dp[i] = dd / m; }
  }
  var change = dp[n - 1u];
  let hi = T[n - 1u] + change;
  T[n - 1u] = hi;
  var lo = hi;
  for (var i = i32(n) - 2; i >= 0; i--) {
    change = dp[i] - cp[i] * change;
    lo = T[i] + change;
    T[i] = lo;
  }
  wall[wQ(w, 2u * axis + 1u)] = hLo * (lo - tLo) * U.dt * A;
  wall[wQ(w, 2u * axis)] = hHi * (hi - tHi) * U.dt * A;
  var after = 0.0;
  for (var q = 0u; q < n; q++) {
    wall[wNode(w, q)] = T[q];
    after += wall[wC(s, q)] * T[q];
  }
  let Ts = (after * A) / total;
  cf[ix(fTs, c)] = Ts;
  wall[wLast(w)] = Ts;`)}

/** Each open cell takes what the faces of its wall neighbours towards it gave (the walls stage's gather). */
${perCell("wallsApply", `
  if (kindOf(c) == SOLID) { return; }
  let p = vec3<i32>(coords(c));
  var heat = 0.0;
  for (var d = 0u; d < 6u; d++) {
    let r = p + dirv(d);
    if (!inside(r)) { continue; }
    let w = wallOf(atI(r));
    if (w == NO_CELL) { continue; }
    // The wall's face towards this cell is the opposite direction.
    heat += wall[wQ(w, d ^ 1u)];
  }
  if (heat != 0.0) { cf[ix(fT, c)] += heat / gasCap(c); }`)}

/** The equation of state (line 31) and the open sides of the domain (line 32). */
${perCell("densityBoundaries", `
  if (kindOf(c) == SOLID) { return; }
  let p = coords(c);
  if (p.x == 0u || p.z == 0u || p.x == U.nx - 1u || p.z == U.nz - 1u || p.y == U.ny - 1u) {
    cf[ix(fT, c)] = U.ambient;
    for (var q = 0u; q < NS; q++) { Y[c * NS + q] = 0.0; }
    Y[c * NS + 1u] = Y_O2_AIR;
    cf[ix(fRho, c)] = U.rho0;
  } else {
    cf[ix(fRho, c)] = gasDensity(cf[ix(fT, c)], c);
  }`)}

// ------------------------------------------------------------------ pressure projection (line 33)

/**
 * A cell's porous factor (the CPU's project): porousVelocity in a porous
 * cell, 1 elsewhere and outside the domain. A face's weight is the product of
 * its two cells' factors.
 */
fn porousOf(q: vec3<i32>) -> f32 {
  if (U.porousVel < 1.0 && porousAt(q)) { return U.porousVel; }
  return 1.0;
}

/**
 * Open neighbours (bit d for DIRS[d]) and pressure links per open cell; the
 * ground is closed. Bit 6: the cell or an open neighbour is porous, so its
 * links are weighted (where none is, every weight is 1 and the weighted
 * projection is the plain one, number for number, as the CPU's).
 */
${perCell("topology", `
  if (kindOf(c) == SOLID) { ci[ix(iMask, c)] = 0u; ci[ix(iCount, c)] = 0u; return; }
  let p = vec3<i32>(coords(c));
  var m = 0u;
  var n = 0u;
  var weighted = porousOf(p) != 1.0;
  for (var d = 0u; d < 6u; d++) {
    let q = p + dirv(d);
    if (q.y < 0) { continue; }
    if (!inside(q)) { n++; continue; }
    if (kindOf(atI(q)) == SOLID) { continue; }
    m |= 1u << d;
    n++;
    if (porousOf(q) != 1.0) { weighted = true; }
  }
  if (weighted) { m |= 64u; }
  ci[ix(iMask, c)] = m;
  ci[ix(iCount, c)] = max(1u, n);`)}

/** Zero the faces of solid cells (and the ground). */
${perThread("projectZero", "U.nu + U.nv + U.nw", `
  let nx = U.nx; let ny = U.ny;
  if (t < U.nu) {
    let q = vec3<i32>(vec3<u32>(t % (nx + 1u), (t / (nx + 1u)) % ny, t / ((nx + 1u) * ny)));
    if (solidAt(q - vec3<i32>(1, 0, 0)) || solidAt(q)) { vel[t] = 0.0; }
  } else if (t < U.nu + U.nv) {
    let r = t - U.nu;
    let q = vec3<i32>(vec3<u32>(r % nx, (r / nx) % (ny + 1u), r / (nx * (ny + 1u))));
    if (solidAt(q - vec3<i32>(0, 1, 0)) || solidAt(q) || q.y == 0) { vel[t] = 0.0; }
  } else {
    let r = t - U.nu - U.nv;
    let q = vec3<i32>(vec3<u32>(r % nx, (r / nx) % ny, r / (nx * ny)));
    if (solidAt(q - vec3<i32>(0, 0, 1)) || solidAt(q)) { vel[t] = 0.0; }
  }`)}

${perCell("divergence", `
  if (kindOf(c) == SOLID) { cf[ix(fDiv, c)] = 0.0; return; }
  let p = coords(c);
  let i = p.x; let j = p.y; let k = p.z;
  var div = 0.0;
  if ((ci[ix(iMask, c)] & 64u) != 0u) {
    // Porous links: the divergence of the weighted flow (the CPU's project).
    let q = vec3<i32>(p);
    let me = porousOf(q);
    div = (me * porousOf(q + vec3<i32>(1, 0, 0)) * vel[uAt(i + 1u, j, k)] - me * porousOf(q - vec3<i32>(1, 0, 0)) * vel[uAt(i, j, k)] +
      me * porousOf(q + vec3<i32>(0, 1, 0)) * vel[vAt(i, j + 1u, k)] - me * porousOf(q - vec3<i32>(0, 1, 0)) * vel[vAt(i, j, k)] +
      me * porousOf(q + vec3<i32>(0, 0, 1)) * vel[wAt(i, j, k + 1u)] - me * porousOf(q - vec3<i32>(0, 0, 1)) * vel[wAt(i, j, k)]) / U.h;
  } else {
    div = (vel[uAt(i + 1u, j, k)] - vel[uAt(i, j, k)] + vel[vAt(i, j + 1u, k)] - vel[vAt(i, j, k)] + vel[wAt(i, j, k + 1u)] - vel[wAt(i, j, k)]) / U.h;
  }
  // The gas's expansion is asked of the flow (proposal F), less a sealed region's mean, which has nowhere to go.
  var used = 0.0;
  if (U.expansion != 0u) {
    used = cf[ix(fD, c)];
    let s = ci[ix(iSealed, c)];
    if (s != NO_CELL) { used -= red[U.redMean + s]; }
  }
  cf[ix(fDiv, c)] = div - used;`)}

/** One cell of a sweep: half-index t of the given colour; returns |new p - old p|. */
fn gsCell(t: u32, colour: u32) -> f32 {
  let hx = (U.nx + 1u) / 2u;
  if (t >= hx * U.ny * U.nz) { return 0.0; }
  let j = (t / hx) % U.ny;
  let k = t / (hx * U.ny);
  let i = 2u * (t % hx) + ((colour + j + k) & 1u);
  if (i >= U.nx) { return 0.0; }
  let c = i + U.nx * (j + U.ny * k);
  if (ci[ix(iKind, c)] == SOLID) { return 0.0; }
  let mask = ci[ix(iMask, c)];
  var sum = 0.0;
  var links = f32(ci[ix(iCount, c)]);
  if ((mask & 64u) != 0u) {
    // Porous links (the CPU's project): each neighbour weighted by its face's
    // weight, over the weights' sum; an open side weighs the cell's own
    // factor, the ground and solids nothing.
    let q = vec3<i32>(vec3<u32>(i, j, k));
    let me = porousOf(q);
    links = 0.0;
    for (var d = 0u; d < 6u; d++) {
      let r = q + dirv(d);
      if (r.y < 0) { continue; }
      if (!inside(r)) { links += me; continue; }
      if ((mask & (1u << d)) == 0u) { continue; }
      let w = me * porousOf(r);
      sum += w * cf[ix(fP, u32(i32(c) + noff(d)))];
      links += w;
    }
    if (!(links > 0.0)) { links = 1.0; }
  } else {
    for (var d = 0u; d < 6u; d++) {
      if ((mask & (1u << d)) != 0u) { sum += cf[ix(fP, u32(i32(c) + noff(d)))]; }
    }
  }
  let scale = U.rho0 * U.h * U.h / U.dt;
  let next = (sum - scale * cf[ix(fDiv, c)]) / links;
  let change = abs(next - cf[ix(fP, c)]);
  cf[ix(fP, c)] = next;
  return change;
}
/**
 * The CPU's stopping rule: at least four sweeps, then stop once a sweep's
 * largest pressure change would move the flow by less than
 * projectionTolerance. Every thread of sweep s decides alike from sweep s - 1;
 * a skipped sweep leaves its slot at zero, so the sweeps after it skip too.
 */
fn gsSkip() -> bool {
  let s = SW.s;
  if (s < 4u) { return false; }
  return bitcast<f32>(atomicLoad(&atom[U.gsAt + s - 1u])) < U.settle;
}
${gsPass("gsRed", 0)}
${gsPass("gsBlack", 1)}

fn pressureAt(q: vec3<i32>) -> f32 {
  if (!inside(q)) { return 0.0; }
  return cf[ix(fP, atI(q))];
}
/**
 * Subtract the pressure gradient. A porous cell's faces carry their weight
 * times the result (the CPU's project): the slowed flow is the one the
 * projection solved for, so it still carries the gas's expansion. (Scaled
 * after the projection, before 2026-11-12, it did not, and the transport
 * made heat in burning contents.) Faces of solids and the ground keep what
 * projectZero left them.
 */
${perThread("projectUpdate", "U.nu + U.nv + U.nw", `
  let nx = U.nx; let ny = U.ny;
  let k2 = U.dt / (U.rho0 * U.h);
  var q: vec3<i32>;
  var e: vec3<i32>;
  if (t < U.nu) {
    q = vec3<i32>(vec3<u32>(t % (nx + 1u), (t / (nx + 1u)) % ny, t / ((nx + 1u) * ny)));
    e = vec3<i32>(1, 0, 0);
  } else if (t < U.nu + U.nv) {
    let r = t - U.nu;
    q = vec3<i32>(vec3<u32>(r % nx, (r / nx) % (ny + 1u), r / (nx * (ny + 1u))));
    e = vec3<i32>(0, 1, 0);
    if (q.y == 0) { return; }
  } else {
    let r = t - U.nu - U.nv;
    q = vec3<i32>(vec3<u32>(r % nx, (r / nx) % ny, r / (nx * ny)));
    e = vec3<i32>(0, 0, 1);
  }
  if (solidAt(q - e) || solidAt(q)) { return; }
  vel[t] = porousOf(q - e) * porousOf(q) * (vel[t] - k2 * (pressureAt(q) - pressureAt(q - e)));`)}

// ------------------------------------------------------------------ readings

/**
 * Gas temperature (C), visibility (m), CO (ppm), soot (mg/m^3), CO2 and O2
 * (% by volume) at the nodes' head cells (the CPU's readNodes); outside the
 * grid, clear air.
 */
${perThread("gather", "U.nNodes", `
  let c = io[U.ioNodes + t];
  var T = U.ambient - KELVIN;
  var vis = 100.0;
  var co = 0.0;
  var soot = 0.0;
  var co2 = 0.0;
  var o2 = (0.232 / 0.032) / (0.232 / 0.032 + 0.768 / 0.028) * 100.0;
  if (c != NO_CELL) {
    T = cf[ix(fT, c)] - KELVIN;
    let rhoSoot = cf[ix(fRho, c)] * Y[c * NS + 4u];
    let K = 8700.0 * rhoSoot;
    vis = select(100.0, 3.0 / K, K > 3.0 / 100.0);
    soot = rhoSoot * 1e6;
    var moles = 0.0;
    var rest = 1.0;
    for (var q = 0u; q < NS; q++) {
      moles += Y[c * NS + q] / molar(q);
      rest -= Y[c * NS + q];
    }
    moles += max(0.0, rest) / 0.028;
    o2 = 0.0;
    if (moles > 0.0) {
      co = ((Y[c * NS + 5u] / 0.028) / moles) * 1e6;
      co2 = ((Y[c * NS + 2u] / 0.044) / moles) * 100.0;
      o2 = ((Y[c * NS + 1u] / 0.032) / moles) * 100.0;
    }
  }
  io[U.ioOut + t] = bitcast<u32>(T);
  io[U.ioOut + U.nNodes + t] = bitcast<u32>(vis);
  io[U.ioOut + 2u * U.nNodes + t] = bitcast<u32>(co);
  io[U.ioOut + 3u * U.nNodes + t] = bitcast<u32>(soot);
  io[U.ioOut + 4u * U.nNodes + t] = bitcast<u32>(co2);
  io[U.ioOut + 5u * U.nNodes + t] = bitcast<u32>(o2);
  // The sprinklers' sensor cells, last in the list: the speed at the cell's centre.
  if (t >= U.speedFrom) {
    var speed = 0.0;
    if (c != NO_CELL) {
      let i = c % U.nx;
      let j = (c / U.nx) % U.ny;
      let k = c / (U.nx * U.ny);
      let a = 0.5 * (vel[uAt(i, j, k)] + vel[uAt(i + 1u, j, k)]);
      let b = 0.5 * (vel[vAt(i, j, k)] + vel[vAt(i, j + 1u, k)]);
      let d = 0.5 * (vel[wAt(i, j, k)] + vel[wAt(i, j, k + 1u)]);
      speed = sqrt(a * a + b * b + d * d);
    }
    io[U.ioSpeed + t - U.speedFrom] = bitcast<u32>(speed);
  }`)}

/** A non-finite f32 by its exponent bits (WGSL may assume no NaN or infinity in comparisons). */
fn nonFinite(x: f32) -> bool { return ((bitcast<u32>(x) >> 23u) & 0xffu) == 0xffu; }

/**
 * The frame's peak gas temperature and its health (flamescapes.js
 * healthReadings): the fastest cell-centred speed, the gas's heat above
 * ambient at cp 1005 (in units of ${HEAT_UNIT} J) and whether any gas cell holds a
 * non-finite or non-positive value.
 */
@compute @workgroup_size(64)
fn peakTemperature(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
                   @builtin(local_invocation_index) li: u32) {
  let c = lin(gid, nwg);
  var v = 0u;
  var fastest = 0u;
  var heat = 0.0;
  var bad = 0u;
  if (c < U.N) {
    if (kindOf(c) != SOLID) {
      let T = cf[ix(fT, c)];
      let rho = cf[ix(fRho, c)];
      let i = c % U.nx;
      let j = (c / U.nx) % U.ny;
      let k = c / (U.nx * U.ny);
      let a = 0.5 * (vel[uAt(i, j, k)] + vel[uAt(i + 1u, j, k)]);
      let b = 0.5 * (vel[vAt(i, j, k)] + vel[vAt(i, j + 1u, k)]);
      let d = 0.5 * (vel[wAt(i, j, k)] + vel[wAt(i, j, k + 1u)]);
      let speed = sqrt(a * a + b * b + d * d);
      if (nonFinite(T) || nonFinite(rho) || nonFinite(speed) || !(rho > 0.0)) {
        bad = 1u;
      } else {
        v = bitcast<u32>(max(T, 0.0));
        fastest = bitcast<u32>(speed);
        heat = rho * U.V * 1005.0 * max(T - U.ambient, 0.0);
      }
    }
  }
  wmax[li] = v;
  wspeed[li] = fastest;
  wheat[li] = heat;
  wbad[li] = bad;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) {
      wmax[li] = max(wmax[li], wmax[li + s]);
      wspeed[li] = max(wspeed[li], wspeed[li + s]);
      wheat[li] = wheat[li] + wheat[li + s];
      wbad[li] = max(wbad[li], wbad[li + s]);
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    atomicMax(&ctr[4], wmax[0]);
    atomicMax(&ctr[7], wspeed[0]);
    atomicAdd(&ctr[8], u32(round(min(wheat[0], 1.0e10) / ${HEAT_UNIT}.0)));
    atomicMax(&ctr[9], wbad[0]);
  }
}

/** Up to snapMax particles, evenly strided (the CPU's particleSnapshot), in grid units. */
${perParticle("snapshot", `
  let count = min(n, U.snapMax);
  if (t == 0u) { io[U.ioSnap] = count; }
  if (t >= count) { return; }
  let stride = select(1.0, f32(n) / f32(U.snapMax), n > U.snapMax);
  let P = parts[min(u32(floor(f32(t) * stride)), n - 1u)];
  let o = U.ioSnap + 1u + t * 5u;
  io[o] = bitcast<u32>(P.p.x);
  io[o + 1u] = bitcast<u32>(P.p.y);
  io[o + 2u] = bitcast<u32>(P.p.z);
  io[o + 3u] = bitcast<u32>(P.v.w);
  io[o + 4u] = bitcast<u32>(P.p.w);`)}
`;
}
