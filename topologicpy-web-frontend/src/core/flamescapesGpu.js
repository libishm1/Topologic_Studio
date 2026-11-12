/**
 * The Flamescapes fire solver on the visitor's GPU through WebGPU (Wrede,
 * Wagner, Palubicki, Michels, Pirk 2026, ACM TOG 45(6) 246,
 * https://doi.org/10.1145/3842518; CC BY 4.0). The paper's own
 * implementation runs WGSL compute kernels through WGPU (its section 5); this
 * is the same idea in the browser, so it runs from a static GitHub Pages site
 * with no server.
 *
 * What is the paper's: Algorithm 1, in the order the CPU solver
 * (src/core/flamescapes.js) runs it, every stage a WGSL compute pass
 * (src/core/flamescapesGpu/kernels.js) over storage buffers that mirror the
 * CPU's arrays; red-black Gauss-Seidel pressure projection with the CPU's
 * sweeps and its early stop (the same sweep count, step by step); particles
 * and radiation on the GPU with atomics. One command buffer per step, so
 * nothing waits for the GPU between steps.
 *
 * What is changed from the CPU solver, and why (details in kernels.js):
 *   - float32 arithmetic; conduction in ten race-free passes (porous own
 *     gas, the +x, +y, +z faces split by parity, then the -x, -y, -z gas
 *     faces) that give every solid cell the CPU's own order of exchanges; radiation's
 *     deposits in fixed point (both paths emit from the stage's starting
 *     temperatures and add the deposits after, since 2026-11-09); embers' spot-ignition test once per cell and step;
 *     flame particles beyond their share refused by the same hash of cell and
 *     step as the CPU's (since 2026-11-09; before, by whichever thread came last);
 *     random numbers from a per-thread PCG hash. With radiation and particles
 *     off, the GPU agrees with the CPU to float32 round-off and summation
 *     order: no more than the CPU differs from itself started one ulp apart
 *     (tools/flamescapes-gpu-check.mjs).
 *   - The conserved transport and the expansion (proposal F, behind the
 *     CPU's own flags): each transport sweep a pass in which every cell
 *     computes both its faces, a sealed region's mean expansion a float32
 *     tree over the regions the CPU's topology found at upload; the CPU's
 *     tallies of the mass floor and the shell's heat are not kept.
 *   - The initial state is the CPU's createSolver, uploaded, so both paths
 *     start from the same state; node cells are the CPU's headCells.
 *   - Per frame only the node readings, the frame's per-step heat release,
 *     fuel burnt and energy radiated, the counters and (when asked) a
 *     particle snapshot come back to the CPU. Whole fields never do.
 *   - A scene the kernels do not cover (fuel inside a solid cell, which
 *     createSolver never makes, or a grid beyond the device's buffer limits)
 *     runs on the CPU solver instead.
 */

import { DEFAULTS, NS, burnerHeat, createSolver, gasDensity, healthReadings, healthVerdict, substepsFor, topology } from "./flamescapes.js";
import {
  KELVIN, ceilingDetectors, failedHealth, headCells, itemResult, itemSuppression, smokeCloud, sprinklerFrame, sprinklerResult, sprinklerRun, systemsFrame, systemsResult, systemsRun, withBurnerFlame, withDetectors, runScene as cpuRunScene,
} from "./flamescapesRun.js";
import {
  AT, AT_FIELDS, B, CF, CF_FIELDS, CI, CI_FIELDS, CTR, CTR_SIZE, HEAT_UNIT, KERNELS, PARAMS, PARAM_BYTES, PARAM_SCALARS,
  PARTICLE_FLOATS, SWEEP_BYTES, WG, buildShader,
  SUP, SYS, WALL, ZONE_BLOCKS,
} from "./flamescapesGpu/kernels.js";

/** A fuel's water rule (fireClasses.js waterResponse) as the bits the WGSL reads: 1 wets, 2 Evans's cap, 4 holds no water. */
const waterFlags = (fuel) => (fuel?.water?.wets === false ? 0 : 1) + (fuel?.water?.evans === false ? 0 : 2) + (fuel?.water?.dry ? 4 : 0);

const MAX_WG_PER_DIM = 65535;
/** Conduction's passes, in the order that gives every solid cell the CPU's order of exchanges. */
const CONDUCTION = ["conductPorous", "conduct_x0", "conduct_x1", "conduct_y0", "conduct_y1", "conduct_z0", "conduct_z1", "conduct_bx", "conduct_by", "conduct_bz"];
/** The conserved transport's passes (proposal F): the CPU's three sweeps, then T from the gas law. */
const TRANSPORT = ["transport_x_start", "transport_y_scratch", "transport_z_live", "transportEnd"];
/** On odd steps the sweeps run z, y, x (the CPU's order alternates by sim.steps: a fixed one favoured the first axis). */
const TRANSPORT_BACK = ["transport_z_start", "transport_y_scratch", "transport_x_live", "transportEnd"];

/**
 * The sealed regions of the CPU's topology (open cells that reach no open
 * side of the domain), for the GPU's expansion: per cell its sealed region
 * (0xffffffff none), and per region its cells in cell order, as offsets into
 * one list. Only porous cells change kind on the GPU, and they stay open, so
 * the regions hold for the whole run.
 */
function sealedRegions(sim) {
  const topo = topology(sim);
  const index = new Int32Array(topo.sealed.length).fill(-1);
  let S = 0;
  for (let r = 0; r < topo.sealed.length; r += 1) if (topo.sealed[r]) index[r] = S++;
  const cellSealed = new Uint32Array(sim.N).fill(0xffffffff);
  const counts = new Uint32Array(S);
  for (let c = 0; c < sim.N; c += 1) {
    const r = topo.region[c];
    if (r >= 0 && index[r] >= 0) {
      cellSealed[c] = index[r];
      counts[index[r]] += 1;
    }
  }
  const offsets = new Uint32Array(S + 1);
  for (let s = 0; s < S; s += 1) offsets[s + 1] = offsets[s] + counts[s];
  const list = new Uint32Array(offsets[S]);
  const next = offsets.slice(0, S);
  for (let c = 0; c < sim.N; c += 1) if (cellSealed[c] !== 0xffffffff) list[next[cellSealed[c]]++] = c;
  return { S, cellSealed, offsets, list };
}
/** Workgroups for a 1D thread count, folded into 2D past the per-dimension limit (kernels use `lin`). */
function workgroups(threads, size) {
  const wg = Math.ceil(threads / size);
  return wg <= MAX_WG_PER_DIM ? [wg, 1] : [MAX_WG_PER_DIM, Math.ceil(wg / MAX_WG_PER_DIM)];
}

/**
 * A WebGPU runner for the worker: resolves to { runScene } or to null when
 * WebGPU, a device or a needed limit is missing, or the kernels fail to
 * compile. Never throws.
 */
export async function createGpuRunner(adapter) {
  let device = null;
  try {
    if (!adapter) adapter = await globalThis.navigator?.gpu?.requestAdapter?.({ powerPreference: "high-performance" });
    if (!adapter) return null;
    const L = adapter.limits;
    // Seven storage buffers in the widest kernel, chemistry (WebGPU guarantees eight); 64-thread workgroups.
    if (!(L.maxStorageBuffersPerShaderStage >= 8) || !(L.maxComputeInvocationsPerWorkgroup >= WG) || !(L.maxComputeWorkgroupSizeX >= WG)) return null;
    device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
        maxBufferSize: L.maxBufferSize,
      },
    });
    if (!device) return null;
    const ctx = { device, lost: false, error: null };
    device.lost.then(() => { ctx.lost = true; });
    device.addEventListener?.("uncapturederror", (event) => { ctx.error ??= event.error?.message || String(event.error); });
    ctx.kernels = await compileKernels(device);
    return {
      device,
      runScene: (scene, opts) => runSceneGpu(ctx, scene, opts),
      /** Low-level access for tests: a GPU twin of a CPU solver state. */
      createSim: (sim, opts) => GpuSim.create(ctx, sim, opts),
    };
  } catch {
    device?.destroy();
    return null;
  }
}

async function compileKernels(device) {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: buildShader(), label: "flamescapes" });
  const info = await module.getCompilationInfo?.();
  const errors = (info?.messages || []).filter((m) => m.type === "error");
  if (errors.length) {
    await device.popErrorScope();
    const text = errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n");
    throw new Error(`Flamescapes WGSL: ${text}`);
  }
  const kernels = {};
  await Promise.all(Object.entries(KERNELS).map(async ([name, spec]) => {
    const layout = device.createBindGroupLayout({
      label: name,
      entries: spec.uses.map((binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: binding === B.U ? { type: "uniform" } : binding === B.sweep ? { type: "uniform", hasDynamicOffset: true } : { type: "storage" },
      })),
    });
    const pipeline = await device.createComputePipelineAsync({
      label: name,
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: name },
    });
    kernels[name] = { ...spec, layout, pipeline, wg: spec.wg || WG };
  }));
  const error = await device.popErrorScope();
  if (error) throw new Error(`Flamescapes pipelines: ${error.message}`);
  return kernels;
}

/** The GPU state of one run: buffers, bind groups, stepping and readback. */
class GpuSim {
  static create(ctx, sim, opts) {
    const g = new GpuSim(ctx, sim, opts);
    return g;
  }

  /**
   * sim: a CPU solver state from createSolver (uploaded as the initial state).
   * opts: nodeCells (Int32Array, -1 outside), snapMax (particles per snapshot),
   *       slots (steps per frame to keep separately); for sprinklers,
   *       speedFrom (the first of nodeCells whose gas speed is read: the
   *       heads' sensor cells, last) and suppression ({columns, eta}: the
   *       `sup` buffer for Evans and the spray; null for none); systems
   *       (suppressionSystems.js systemsSetup: the `sys` buffer for flooded
   *       and foamed rooms; null for none); flameFrom (where a design-fire
   *       item's cells, sim.burner.cells, start in nodeCells: their
   *       temperatures colour its drawn flame; null for none).
   */
  constructor(ctx, sim, { nodeCells = new Int32Array(0), snapMax = 0, slots = 64, speedFrom = null, suppression = null, systems = null, flameFrom = null } = {}) {
    const { device, kernels } = ctx;
    this.ctx = ctx;
    this.device = device;
    this.kernels = kernels;
    this.sim = sim;
    const P = sim.P;
    const { nx, ny, nz, h } = sim.grid;
    const N = sim.N;
    const d = (this.dims = {
      nx, ny, nz, N,
      nu: (nx + 1) * ny * nz,
      nv: nx * (ny + 1) * nz,
      nw: nx * ny * (nz + 1),
      cap: Math.max(0, P.maxParticles | 0),
      nNodes: nodeCells.length,
      snapMax: Math.max(0, snapMax | 0),
    });
    const nuvw = d.nu + d.nv + d.nw;
    const [wx, wy] = workgroups(N, WG);
    this.nPart = wx * wy;
    this.slots = Math.max(1, slots | 0);
    // One slot per pressure sweep for its largest change, and its index uniform.
    this.gsSlots = Math.max(1, P.projection | 0);
    this.sweepStride = Math.max(SWEEP_BYTES, device.limits.minUniformBufferOffsetAlignment || 256);
    this.time = sim.time || 0;
    this.stepCount = 0;
    // The CPU's step count at upload: the random streams number steps from the run's start (sim.steps).
    this.stepBase = sim.steps || 0;
    this.slot = 0;
    this.lastDt = P.dt;
    this.particlesOn = d.cap > 0;
    this.radiationOn = Number.isFinite(P.emitAbove) && P.emitAbove < 1e8;
    this.embersOn = Boolean(P.embers) && this.particlesOn;
    this.vorticityOn = P.vorticity > 0;
    // Proposal F, as the CPU's flags: the conserved transport and the expansion.
    this.conservativeOn = P.conservative > 0;
    this.expansionOn = P.expansion > 0;
    // Pressure zones (flamescapes.js pressureZones): the sealed regions with a pressure each, when the CPU has them.
    this.zonesOn = P.pressureZones > 0 && sim.zones !== null && topology(sim).sealed.some((x) => x);
    this.sealed = this.expansionOn || this.zonesOn ? sealedRegions(sim) : null;
    d.conservative = this.conservativeOn;
    d.sealedN = this.sealed ? this.sealed.S : 0;
    d.zonesOn = this.zonesOn;
    // One-dimensional walls (flamescapes.js wallSetup), when the scene has any.
    this.wall = sim.wall && sim.wall.cells.length ? sim.wall : null;
    d.wallN = this.wall ? this.wall.cells.length : 0;
    const pilot = sim.pilot && sim.pilot.cells?.length ? sim.pilot : null;
    this.pilot = pilot;
    // A design-fire first item (flamescapes.js createBurner); its heat for each step is the host's (burnerHeat).
    this.burner = sim.burner?.design ? sim.burner : null;
    // Running ventilation (flamescapes.js ducts): at most 8 systems.
    const ducts = sim.ducts && sim.ducts.cells?.length && sim.ducts.systems <= 8 ? sim.ducts : null;
    this.ducts = ducts;

    // ---- layout of the shared buffers
    const io = { nodes: pilot ? pilot.cells.length : 0 };
    io.out = io.nodes + d.nNodes;
    // Six readings a node (gather): T, vis, co, soot, co2, o2.
    io.snap = io.out + 6 * d.nNodes;
    io.ducts = io.snap + 1 + 5 * d.snapMax;
    // The sensor cells' gas speeds (gather), when the run has sprinklers.
    this.speedFrom = speedFrom ?? d.nNodes;
    this.nSpeed = d.nNodes - this.speedFrom;
    this.flameFrom = flameFrom;
    io.speed = io.ducts + (ducts ? 2 * ducts.cells.length : 0);
    // The design-fire item's cells, a layer at a time from its base.
    io.burner = io.speed + this.nSpeed;
    // Then its layers' starts in that list, and the cells its flame may reach beyond its own and their distances (createBurner).
    const b = this.burner;
    io.burnerLayers = io.burner + (b ? b.cells.length : 0);
    io.reach = io.burnerLayers + (b ? b.layerStart.length : 0);
    // The sealed regions' offsets and cells (sealedRegions), when the gas expands.
    io.sealed = io.reach + (b ? 2 * b.reach.length : 0);
    io.size = io.sealed + (this.sealed ? this.sealed.S + 1 + this.sealed.list.length : 0);
    this.io = io;
    this.suppressionOn = Boolean(suppression);
    d.sprayCols = 0;
    // Flooded and foamed rooms: per cell its room, per room its state, the foam's wet cells.
    this.systems = systems && systems.R > 0 ? systems : null;
    d.sysWet = this.systems ? this.systems.wet.length : 0;
    const sysAt = { state: N, wet: N + SYS.stride * ((this.systems?.R ?? 0) + 1) };
    this.sysAt = sysAt;
    const red = { rad: 2 * this.nPart, steps: 3 * this.nPart };
    // After the step sums, four floats: the pilot's added heat over the frame, then padding.
    red.pilot = red.steps + 4 * this.slots;
    // Then each sealed region's mean expansion this step (sealedMean).
    // Then a design-fire item's four (kernels.js burner): the frame's heat
    // released, the last step's, the heat waiting for oxygen and the last
    // step's flame cells.
    red.burner = red.pilot + 4;
    red.mean = red.burner + 4;
    // Then each pressure zone's pi, mass, last ratio and kappa, and each cell's mass source since the last update.
    red.zone = red.mean + d.sealedN;
    red.zoneSrc = red.zone + (this.zonesOn ? 4 * d.sealedN : 0);
    // Then the reduction's partial sums, three a block (kernels.js zonePartial).
    red.zonePart = red.zoneSrc + (this.zonesOn ? N : 0);
    red.size = red.zonePart + (this.zonesOn ? 3 * ZONE_BLOCKS * d.sealedN : 0);
    this.red = red;
    // In `snap` after the step's start (velocities, T, Y): the transport's
    // scratch state (rho, then Y) and the step's starting T for the expansion.
    const snapAt = { gas: nuvw + 7 * N };
    snapAt.expand = snapAt.gas + (this.conservativeOn ? 7 * N : 0);
    // Then, with soot deposition, each open cell's deposited soot (the CPU's sootDeposit).
    this.depositOn = P.sootDeposition > 0 && Boolean(sim.sootDeposit);
    snapAt.dep = snapAt.expand + (this.expansionOn ? N : 0);
    snapAt.size = snapAt.dep + (this.depositOn ? N : 0);
    this.snapAt = snapAt;

    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (label, bytes) => device.createBuffer({ label, size: Math.max(16, Math.ceil(bytes / 4) * 4), usage });
    const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    // The wall buffer's words (kernels.js WALL): per wall its cell, meta, last, nodes and six face heats; per stack its C, G, heat and thickness.
    const Wn = this.wall ? this.wall.n : 0, Ws = this.wall ? this.wall.heat.length : 0;
    this.wallWords = this.wall ? (WALL.fixed + Wn) * d.wallN + Ws * (2 * Wn + 1) : 0;
    const sizes = {
      cf: CF_FIELDS * N * 4,
      Y: 6 * N * 4,
      ci: CI_FIELDS * N * 4,
      vel: nuvw * 4,
      snap: snapAt.size * 4,
      // Then two ray deposits per one-dimensional wall, after the sweeps' slots.
      atom: (AT_FIELDS * N + this.gsSlots + 2 * d.wallN) * 4,
      parts: Math.max(1, d.cap) * PARTICLE_FLOATS * 4,
      red: red.size * 4,
      io: io.size * 4,
      // Evans's three fields per cell and the spray's columns; nothing without sprinklers.
      sup: suppression ? (SUP.fields * N + SUP.column * Math.max(1, suppression.columns)) * 4 : 16,
      // Rooms per cell, their states and the wet cells; nothing without systems.
      sys: this.systems ? (sysAt.wet + SYS.wet * Math.max(1, d.sysWet)) * 4 : 16,
      // One-dimensional walls; nothing without them.
      wall: this.wall ? this.wallWords * 4 : 16,
    };
    for (const [name, bytes] of Object.entries(sizes)) if (bytes > limit) throw new RangeError(`Flamescapes GPU: ${name} needs ${bytes} bytes, the device binds ${limit}`);
    this.buf = {
      U: device.createBuffer({ label: "params", size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      cf: make("cf", sizes.cf),
      Y: make("Y", sizes.Y),
      ci: make("ci", sizes.ci),
      vel: make("vel", sizes.vel),
      snap: make("snap", sizes.snap),
      atom: make("atom", sizes.atom),
      parts: make("parts", sizes.parts),
      partsB: make("partsB", sizes.parts),
      ctr: make("ctr", CTR_SIZE * 4),
      red: make("red", sizes.red),
      io: make("io", sizes.io),
      sweep: device.createBuffer({ label: "sweep index", size: this.sweepStride * this.gsSlots, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      sup: make("sup", sizes.sup),
      sys: make("sys", sizes.sys),
      wall: make("wall", sizes.wall),
    };
    // Sweep s's index at offset s * sweepStride, bound at that dynamic offset.
    const index = new Uint32Array((this.sweepStride / 4) * this.gsSlots);
    for (let s = 0; s < this.gsSlots; s += 1) index[(s * this.sweepStride) / 4] = s;
    device.queue.writeBuffer(this.buf.sweep, 0, index);
    const byBinding = [this.buf.U, this.buf.cf, this.buf.Y, this.buf.ci, this.buf.vel, this.buf.snap, this.buf.atom, this.buf.parts, this.buf.partsB, this.buf.ctr, this.buf.red, this.buf.io, this.buf.sweep, this.buf.atom, this.buf.sup, this.buf.sys, this.buf.wall];
    this.groups = {};
    for (const [name, k] of Object.entries(kernels)) {
      this.groups[name] = device.createBindGroup({
        label: name,
        layout: k.layout,
        entries: k.uses.map((binding) => ({ binding, resource: binding === B.sweep ? { buffer: byBinding[binding], size: SWEEP_BYTES } : { buffer: byBinding[binding] } })),
      });
    }
    this.staging = device.createBuffer({ label: "frame readback", size: Math.max(16, (6 * d.nNodes + this.nSpeed + CTR_SIZE + this.gsSlots + 4 * this.slots + 4 + 4 + 1 + 5 * d.snapMax) * 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    // ---- parameters
    this.paramBytes = new ArrayBuffer(PARAM_BYTES);
    this.paramF = new Float32Array(this.paramBytes);
    this.paramU = new Uint32Array(this.paramBytes);
    this.paramIndex = Object.fromEntries(PARAMS.map(([name], i) => [name, i]));
    const fuelRow = (k) => {
      const st = sim.stoich[k];
      const fuel = sim.fuels[k];
      const liquid = fuel.liquid;
      return [st.o2, st.co2, st.co, st.soot, st.h2o, st.heat, fuel.ignitionTemperature + KELVIN, fuel.embers ? 1 : 0, liquid ? 1 : 0, liquid?.boiling ?? 0, liquid?.latent ?? 1, st.ash ?? 0, fuel.heatCapacity ?? P.fuelHeatCapacity, liquid?.mInf ?? -1, liquid?.kBeta ?? -1, waterFlags(fuel)];
    };
    this.paramF.set([...fuelRow(0), ...fuelRow(1)], PARAM_SCALARS);
    this.setParams({
      nx, ny, nz, N, nu: d.nu, nv: d.nv, nw: d.nw, stepNo: 0, slot: 0,
      rays: Math.max(0, P.rays | 0), rayReach: Math.max(0, P.rayReach | 0), cap: d.cap,
      flameCap: Math.min(d.cap, Math.ceil(P.flameShare * d.cap)),
      nPart: this.nPart, nNodes: d.nNodes, snapMax: d.snapMax, pilotN: io.nodes,
      pilotMode: pilot && pilot.temperature ? 0 : 1, ioNodes: io.nodes, ioOut: io.out, ioSnap: io.snap,
      redRad: red.rad, redSteps: red.steps, redPilot: red.pilot, seed: P.seed >>> 0, gsAt: AT_FIELDS * N,
      h, dt: P.dt, V: sim.V, rho0: sim.rho0, ambient: P.ambient, gravity: P.gravity, convection: P.convection, wallConvection: P.wallConvection,
      kappa: P.sootAbsorption, vorticity: P.vorticity, porousVel: P.porousVelocity, diffusivity: P.diffusivity,
      conductivity: P.conductivity, emitAbove: Math.min(P.emitAbove, 3e38), pyA: P.pyrolysisA, pyE: P.pyrolysisE,
      pyHeat: P.pyrolysisHeat, oxA: P.oxidationA, oxE: P.oxidationE, pyFuel: P.pyrolysisFuel, latent: P.latentHeat,
      moistFrac: P.moisture, fuelHC: P.fuelHeatCapacity, skinDensity: P.skinDensity, gasA: P.gasA, gasE: P.gasE,
      gasFuelExp: P.gasFuelExponent, gasOxExp: P.gasOxygenExponent, flameSpeed: P.flameSpeed, flameLife: P.flameLifetime,
      emberRate: P.emberRate, emberPatch: P.emberPatch, emberMassLo: P.emberMass[0], emberMassHi: P.emberMass[1],
      emberLife: P.emberLifetime, emberMassLoss: P.emberMassLoss, emberDrag: P.emberDrag, emberEmis: P.emberEmissivity,
      emberHC: P.emberHeatCapacity, pilotTemp: pilot?.temperature || 0, pilotPower: pilot?.power || 0,
      burnFlux: P.burningFlux,
      ductN: ducts ? ducts.cells.length : 0, ductSystems: ducts ? ducts.systems : 0, ductAt: io.ducts, ductFlow: ducts ? ducts.flow : 0,
      time: this.time, supOn: suppression ? 1 : 0, sprayN: 0, supCol: SUP.fields * N, sprayEta: suppression?.eta ?? 0, sprayWet: suppression?.wet ?? 0,
      speedFrom: this.speedFrom, ioSpeed: io.speed,
      sysOn: this.systems ? 1 : 0, sysState: sysAt.state, sysWetAt: sysAt.wet, sysWetN: d.sysWet,
      burnerN: b ? b.cells.length : 0, burnerAt: io.burner, burnerLayers: b ? b.layerStart.length - 1 : 0, burnerLayerAt: io.burnerLayers,
      reachN: b ? b.reach.length : 0, reachAt: io.reach, burnerD: b?.diameter ?? 0, burnerDUsed: b?.diameterUsed ?? 0,
      burnerRf: b?.radiantFraction ?? 0, burnerHeat: 0, redBurner: red.burner,
      conservative: this.conservativeOn ? 1 : 0, expansion: this.expansionOn ? 1 : 0, gasAt: snapAt.gas, expandAt: snapAt.expand,
      sealedN: d.sealedN, sealedAt: io.sealed, redMean: red.mean,
      wallN: d.wallN, wallNodes: Wn, wallStacks: Ws, wallDepAt: AT_FIELDS * N + this.gsSlots,
      sootDeposition: P.sootDeposition > 0 ? 1 : 0, sootDensity: P.sootDensity, sootDiameter: P.sootDiameter,
      sootThDiameter: P.sootThermophoreticDiameter, sootConductivity: P.sootConductivity, depAt: snapAt.dep,
      zonesOn: this.zonesOn ? 1 : 0, redZone: red.zone, zoneSrcAt: red.zoneSrc, zonePartAt: red.zonePart,
    });
    this.writeParams();
    this.upload(nodeCells);
  }

  setParams(values) {
    for (const [name, value] of Object.entries(values)) {
      const i = this.paramIndex[name];
      if (i === undefined) throw new Error(`unknown parameter ${name}`);
      if (PARAMS[i][1] === "u32") this.paramU[i] = value >>> 0;
      else this.paramF[i] = value;
    }
  }

  writeParams() {
    this.device.queue.writeBuffer(this.buf.U, 0, this.paramBytes);
  }

  /** The CPU state as the initial GPU state. */
  upload(nodeCells) {
    const { sim, device, dims: d } = this;
    const q = device.queue;
    const N = d.N;
    const cf = new Float32Array(CF_FIELDS * N);
    cf.set(sim.T, CF.T * N);
    cf.set(sim.Ts, CF.Ts * N);
    cf.set(sim.rho, CF.rho * N);
    cf.set(sim.p, CF.p * N);
    cf.set(sim.fuelMass, CF.fuel * N);
    cf.set(sim.moisture, CF.moist * N);
    cf.set(sim.area, CF.area * N);
    cf.set(sim.inertC, CF.inertC * N);
    cf.set(sim.inertK, CF.inertK * N);
    cf.set(sim.eps, CF.eps * N);
    if (sim.solidLoss) cf.set(sim.solidLoss, CF.loss * N);
    q.writeBuffer(this.buf.cf, 0, cf);
    q.writeBuffer(this.buf.Y, 0, sim.Y);
    const ci = new Uint32Array(CI_FIELDS * N);
    ci.set(sim.kind, CI.kind * N);
    ci.set(sim.fuelKind, CI.fuelKind * N);
    ci.set(sim.burning, CI.burning * N);
    if (sim.fuelVapourKind) ci.set(sim.fuelVapourKind, CI.vapour * N);
    if (this.sealed) ci.set(this.sealed.cellSealed, CI.sealed * N);
    else ci.fill(0xffffffff, CI.sealed * N, (CI.sealed + 1) * N);
    // Each cell's one-dimensional wall (-1, none, is NO_CELL's bits).
    if (this.wall) ci.set(new Uint32Array(this.wall.index.buffer, this.wall.index.byteOffset, N), CI.wall * N);
    else ci.fill(0xffffffff, CI.wall * N, (CI.wall + 1) * N);
    q.writeBuffer(this.buf.ci, 0, ci);
    if (this.depositOn) q.writeBuffer(this.buf.snap, this.snapAt.dep * 4, sim.sootDeposit);
    if (this.wall) {
      // The CPU's sim.wall in the wall buffer's order (kernels.js WALL).
      const W = this.wall, n = W.n, Wc = d.wallN;
      const words = new ArrayBuffer(this.wallWords * 4);
      const u = new Uint32Array(words), f = new Float32Array(words);
      u.set(new Uint32Array(W.cells.buffer, W.cells.byteOffset, Wc), 0);
      u.set(W.meta, Wc);
      f.set(W.last, 2 * Wc);
      f.set(W.T, 3 * Wc);
      f.set(W.Q, (3 + n) * Wc);
      const stacks = (WALL.fixed + n) * Wc, S = W.heat.length;
      f.set(W.C, stacks);
      f.set(W.G, stacks + S * n);
      f.set(W.heat, stacks + S * (2 * n - 1));
      f.set(W.thickness, stacks + S * (2 * n - 1) + S);
      q.writeBuffer(this.buf.wall, 0, words);
    }
    if (this.suppressionOn) {
      const sup = new Float32Array(SUP.fields * N);
      sup.fill(-1, SUP.rate * N);
      q.writeBuffer(this.buf.sup, 0, sup);
    }
    if (this.systems) {
      // Rooms per cell (u32), the states at the start, the wet cells (cell, room, kg/s).
      const s = this.systems;
      const at = this.sysAt;
      const words = new ArrayBuffer((at.wet + SYS.wet * d.sysWet) * 4);
      const u = new Uint32Array(words), f = new Float32Array(words);
      u.set(s.room, 0);
      s.update(this.time);
      f.set(s.state, at.state);
      for (let k = 0; k < d.sysWet; k += 1) {
        u[at.wet + SYS.wet * k] = s.wet[k];
        u[at.wet + SYS.wet * k + 1] = s.wetRoom[k];
        f[at.wet + SYS.wet * k + 2] = s.wetFlux[k];
      }
      q.writeBuffer(this.buf.sys, 0, words);
    }
    const vel = new Float32Array(d.nu + d.nv + d.nw);
    vel.set(sim.u, 0);
    vel.set(sim.v, d.nu);
    vel.set(sim.w, d.nu + d.nv);
    q.writeBuffer(this.buf.vel, 0, vel);
    if (sim.flame.some((x) => x > 0)) {
      const atom = new Float32Array(N);
      atom.set(sim.flame);
      q.writeBuffer(this.buf.atom, AT.flame * N * 4, atom);
    }
    // Particles already alive in the CPU state (none at the start of a run).
    const pr = sim.parts;
    const n = Math.min(pr.n, d.cap);
    if (n > 0) {
      const parts = new Float32Array(n * PARTICLE_FLOATS);
      for (let i = 0; i < n; i += 1) {
        const o = i * PARTICLE_FLOATS;
        parts.set([pr.x[i * 3], pr.x[i * 3 + 1], pr.x[i * 3 + 2], pr.kind[i], pr.vel[i * 3], pr.vel[i * 3 + 1], pr.vel[i * 3 + 2], pr.T[i], pr.m[i], pr.age[i], pr.life[i], 0], o);
      }
      q.writeBuffer(this.buf.parts, 0, parts);
    }
    const ctr = new Uint32Array(CTR_SIZE);
    ctr[CTR.count] = n;
    ctr[CTR.before] = n;
    ctr[CTR.maxT] = new Uint32Array(Float32Array.of(sim.P.ambient).buffer)[0];
    q.writeBuffer(this.buf.ctr, 0, ctr);
    const io = new Uint32Array(this.io.size);
    if (this.pilot) io.set(this.pilot.cells, 0);
    if (this.burner) {
      const b = this.burner;
      io.set(b.cells, this.io.burner);
      io.set(b.layerStart, this.io.burnerLayers);
      io.set(b.reach, this.io.reach);
      io.set(b.reachDist, this.io.reach + b.reach.length);
      // Heat already waiting for oxygen in the CPU state (none at a run's start).
      if (b.pending > 0) q.writeBuffer(this.buf.red, (this.red.burner + 2) * 4, Float32Array.of(b.pending));
    }
    if (this.ducts) {
      for (let q = 0; q < this.ducts.cells.length; q += 1) {
        io[this.io.ducts + 2 * q] = this.ducts.cells[q];
        io[this.io.ducts + 2 * q + 1] = this.ducts.system[q];
      }
    }
    for (let v = 0; v < nodeCells.length; v += 1) io[this.io.nodes + v] = nodeCells[v] < 0 ? 0xffffffff : nodeCells[v];
    if (this.sealed) {
      io.set(this.sealed.offsets, this.io.sealed);
      io.set(this.sealed.list, this.io.sealed + this.sealed.S + 1);
    }
    if (this.zonesOn) {
      // The CPU's zones (sealed regions in the same order), or, before its first step, each at ambient pressure with its gas then.
      const S = this.sealed.S, z = new Float32Array(4 * S);
      if (sim.zones) {
        z.set(sim.zones.pi, 0);
        for (let s = 0; s < S; s += 1) z[S + s] = sim.zones.mass[s] + sim.zones.source[s];
      } else {
        z.fill(1, 0, S);
        // Summed in float64, then stored: summed in float32 the mass drifted by 2e-5.
        const mass = new Float64Array(S);
        for (let c = 0; c < N; c += 1) {
          const s = this.sealed.cellSealed[c];
          if (s !== 0xffffffff) mass[s] += gasDensity(sim.T[c], sim.Y, c * NS) * sim.V;
        }
        z.set(mass, S);
      }
      q.writeBuffer(this.buf.red, this.red.zone * 4, z);
    }
    q.writeBuffer(this.buf.io, 0, io);
    // Open-neighbour masks for the first diffusion and projection.
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.dispatch(pass, "topology");
    pass.end();
    q.submit([enc.finish()]);
  }

  /** One compute pass of a kernel; `offsets` are its dynamic uniform offsets (the sweep index). */
  dispatch(pass, name, offsets) {
    const k = this.kernels[name];
    const threads = k.threads(this.dims);
    if (!(threads > 0)) return;
    pass.setPipeline(k.pipeline);
    if (offsets) pass.setBindGroup(0, this.groups[name], offsets);
    else pass.setBindGroup(0, this.groups[name]);
    const [x, y] = workgroups(threads, k.wg);
    pass.dispatchWorkgroups(x, y);
  }

  /** With pressure zones, each zone's pressure from its mass and heat, then its cells (the CPU's pressureZones). */
  zoneUpdate(pass) {
    if (!this.zonesOn) return;
    this.dispatch(pass, "zonePartial");
    this.dispatch(pass, "zoneCommit");
    this.dispatch(pass, "zoneApply");
  }

  /** Sweeps the last step ran, by the CPU's stopping rule on the sweeps' largest changes. */
  sweepsFrom(change) {
    const P = this.sim.P;
    const settle = ((P.projectionTolerance ?? 0) * this.sim.rho0 * this.sim.grid.h) / this.lastDt;
    const n = Math.max(0, P.projection | 0);
    for (let s = 4; s < n; s += 1) if (change[s - 1] < settle) return s;
    return n;
  }

  /** One frame of Algorithm 1 (the CPU's step), queued on the GPU. */
  step(dt = this.sim.P.dt) {
    const { device, buf, dims: d } = this;
    const N = d.N;
    const nuvw = d.nu + d.nv + d.nw;
    const slot = Math.min(this.slot, this.slots - 1);
    const P = this.sim.P;
    // The projection stops once a sweep would move the flow by less than this (the CPU's settle).
    const settle = ((P.projectionTolerance ?? 0) * this.sim.rho0 * this.sim.grid.h) / dt;
    // The design-fire item's table heat this step (the host's: water on the
    // item and clean agent change it); the GPU burns it, and what waited for
    // oxygen, where there is oxygen, and tallies what burnt (kernels.js burner).
    const burnerE = this.burner ? burnerHeat(this.burner, this.time, this.time + dt) : 0;
    this.setParams({ dt, stepNo: this.stepBase + this.stepCount, slot, settle, time: this.time, burnerHeat: burnerE });
    this.writeParams();
    // Each flooded or foamed room's state for this step (the CPU's chemistry computes the same at its start).
    if (this.systems) device.queue.writeBuffer(buf.sys, this.sysAt.state * 4, this.systems.update(this.time));
    const enc = device.createCommandEncoder();
    enc.clearBuffer(buf.atom, AT_FIELDS * N * 4, this.gsSlots * 4);
    // The step's starting T, from which the expansion reads what the heat sources did (the CPU's expansionStart).
    if (this.expansionOn) enc.copyBufferToBuffer(buf.cf, CF.T * N * 4, buf.snap, this.snapAt.expand * 4, N * 4);
    let pass = enc.beginComputePass();
    // Ignition source, then particles (lines 8 to 14).
    if (this.burner) this.dispatch(pass, "burner");
    if (this.pilot && this.time < this.pilot.duration) this.dispatch(pass, "pilot");
    if (this.particlesOn) {
      this.dispatch(pass, "countBurning");
      this.dispatch(pass, "spawnFlame");
      this.dispatch(pass, "spawnFlameEnd");
      if (this.embersOn) this.dispatch(pass, "spawnEmbers");
      this.dispatch(pass, "moveParticles");
    }
    pass.end();
    // Radiation (line 15) and particle-to-grid (lines 16 to 18).
    if (this.radiationOn) enc.clearBuffer(buf.atom, AT.depGas * N * 4, 2 * N * 4);
    if (this.radiationOn && d.wallN) enc.clearBuffer(buf.atom, (AT_FIELDS * N + this.gsSlots) * 4, 2 * d.wallN * 4);
    if (this.particlesOn) {
      enc.clearBuffer(buf.atom, AT.flame * N * 4, N * 4);
      if (this.embersOn) enc.clearBuffer(buf.atom, AT.ember * N * 4, 2 * N * 4);
    }
    if (this.radiationOn || this.particlesOn) {
      pass = enc.beginComputePass();
      if (this.radiationOn) {
        this.dispatch(pass, "radEmit");
        this.dispatch(pass, "radApply");
      }
      if (this.particlesOn) {
        this.dispatch(pass, "particleToGrid");
        if (this.embersOn) this.dispatch(pass, "emberDeposit");
      }
      pass.end();
    }
    // The expansion the heat sources made so far, and with pressure zones each zone's pressure from its heat (the CPU's zones stage), before the advection's snapshot.
    if (this.zonesOn) {
      pass = enc.beginComputePass();
      if (this.expansionOn) this.dispatch(pass, "expandBefore");
      this.zoneUpdate(pass);
      pass.end();
    }
    // Advection reads the step's start (line 20).
    enc.copyBufferToBuffer(buf.vel, 0, buf.snap, 0, nuvw * 4);
    enc.copyBufferToBuffer(buf.cf, CF.T * N * 4, buf.snap, nuvw * 4, N * 4);
    enc.copyBufferToBuffer(buf.Y, 0, buf.snap, (nuvw + N) * 4, 6 * N * 4);
    pass = enc.beginComputePass();
    if (this.expansionOn && !this.zonesOn) this.dispatch(pass, "expandBefore");
    this.dispatch(pass, "advect");
    if (this.conservativeOn) for (const name of (this.stepBase + this.stepCount) % 2 === 0 ? TRANSPORT : TRANSPORT_BACK) this.dispatch(pass, name);
    // The equation of state; a pressure zone's at its pressure, scaled back to its mass (the CPU's zoneTransport).
    this.dispatch(pass, "density");
    if (this.zonesOn) {
      this.dispatch(pass, "zoneMassPartial");
      this.dispatch(pass, "zoneMassCommit");
      this.dispatch(pass, "zoneScale");
    }
    this.dispatch(pass, "buoyancy");
    if (this.vorticityOn) {
      this.dispatch(pass, "vortCurl");
      this.dispatch(pass, "vortForce");
      this.dispatch(pass, "vortApply");
    }
    pass.end();
    // Diffusion reads the advected fields (line 23).
    enc.copyBufferToBuffer(buf.cf, CF.T * N * 4, buf.snap, nuvw * 4, N * 4);
    enc.copyBufferToBuffer(buf.Y, 0, buf.snap, (nuvw + N) * 4, 6 * N * 4);
    pass = enc.beginComputePass();
    this.dispatch(pass, "diffuse");
    if (this.ducts) this.dispatch(pass, "ducts");
    this.dispatch(pass, "chemistry");
    this.dispatch(pass, "reduceFinish");
    if (d.sprayCols) this.dispatch(pass, "spray");
    if (d.sysWet) this.dispatch(pass, "systemsWet");
    if (this.particlesOn) this.dispatch(pass, "cullFlameFront");
    for (const name of CONDUCTION) this.dispatch(pass, name);
    // One-dimensional walls after conduction (the CPU's walls stage), then their gas takes what they gave.
    if (d.wallN) {
      this.dispatch(pass, "walls");
      this.dispatch(pass, "wallsApply");
    }
    // Soot deposition on the solids, after the walls (the CPU's deposition stage).
    if (this.depositOn) this.dispatch(pass, "deposition");
    // The expansion the step's heat made, and each sealed region's mean (the CPU's expansionEnd and project).
    if (this.expansionOn) {
      this.dispatch(pass, "expandEnd");
      this.dispatch(pass, "sealedMean");
    }
    this.dispatch(pass, "densityBoundaries");
    this.zoneUpdate(pass);
    this.dispatch(pass, "topology");
    this.dispatch(pass, "projectZero");
    this.dispatch(pass, "divergence");
    for (let sweep = 0; sweep < P.projection; sweep += 1) {
      const offsets = [sweep * this.sweepStride];
      this.dispatch(pass, "gsRed", offsets);
      this.dispatch(pass, "gsBlack", offsets);
    }
    this.dispatch(pass, "projectUpdate");
    if (this.particlesOn) {
      this.dispatch(pass, "compact");
      this.dispatch(pass, "compactFinish");
    }
    pass.end();
    if (this.particlesOn) enc.copyBufferToBuffer(buf.partsB, 0, buf.parts, 0, d.cap * PARTICLE_FLOATS * 4);
    device.queue.submit([enc.finish()]);
    this.lastDt = dt;
    this.lastSlot = slot;
    this.time += dt;
    this.stepCount += 1;
    this.slot += 1;
  }

  /**
   * End a frame: gather the node readings, the peak gas temperature, the
   * frame's per-step sums and the counters (and a particle snapshot), read
   * them back, and start the next frame's sums.
   */
  async readFrame({ particles = true, smoke = false } = {}) {
    const { device, buf, dims: d } = this;
    if (this.ctx.lost) throw new Error("The GPU device was lost during the fire run.");
    device.queue.writeBuffer(buf.ctr, CTR.maxT * 4, Float32Array.of(this.sim.P.ambient));
    // The frame's health counters start afresh: speed, heat and bad (peakTemperature).
    device.queue.writeBuffer(buf.ctr, CTR.speed * 4, new Uint32Array(3));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.dispatch(pass, "gather");
    this.dispatch(pass, "peakTemperature");
    const snap = particles && d.snapMax > 0;
    if (snap) this.dispatch(pass, "snapshot");
    pass.end();
    const nodeBytes = 6 * d.nNodes * 4;
    const speedBytes = this.nSpeed * 4;
    const stepBytes = 4 * this.slots * 4;
    let o = 0;
    if (nodeBytes) enc.copyBufferToBuffer(buf.io, this.io.out * 4, this.staging, o, nodeBytes);
    o += nodeBytes;
    if (speedBytes) enc.copyBufferToBuffer(buf.io, this.io.speed * 4, this.staging, o, speedBytes);
    o += speedBytes;
    enc.copyBufferToBuffer(buf.ctr, 0, this.staging, o, CTR_SIZE * 4);
    o += CTR_SIZE * 4;
    enc.copyBufferToBuffer(buf.atom, AT_FIELDS * d.N * 4, this.staging, o, this.gsSlots * 4);
    o += this.gsSlots * 4;
    enc.copyBufferToBuffer(buf.red, this.red.steps * 4, this.staging, o, stepBytes);
    o += stepBytes;
    // The pilot's added heat over the frame (and its padding), straight after the step sums.
    enc.copyBufferToBuffer(buf.red, this.red.pilot * 4, this.staging, o, 16);
    o += 16;
    // A design-fire item's four (kernels.js burner), straight after.
    enc.copyBufferToBuffer(buf.red, this.red.burner * 4, this.staging, o, 16);
    o += 16;
    const snapBytes = (1 + 5 * d.snapMax) * 4;
    if (snap) enc.copyBufferToBuffer(buf.io, this.io.snap * 4, this.staging, o, snapBytes);
    // The frame's sums start afresh; the item's heat waiting for oxygen does not.
    enc.clearBuffer(buf.red, this.red.steps * 4, stepBytes + 16);
    enc.clearBuffer(buf.red, this.red.burner * 4, 4);
    // The smoke cloud needs the gas density and the species (soot among them),
    // read into their own buffer only when it is drawn.
    if (smoke) {
      this.smokeStaging ||= device.createBuffer({ label: "smoke readback", size: (NS + 1) * d.N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      enc.copyBufferToBuffer(buf.Y, 0, this.smokeStaging, 0, NS * d.N * 4);
      enc.copyBufferToBuffer(buf.cf, CF.rho * d.N * 4, this.smokeStaging, NS * d.N * 4, d.N * 4);
    }
    device.queue.submit([enc.finish()]);
    let cloud = null;
    if (smoke) {
      await this.smokeStaging.mapAsync(GPUMapMode.READ);
      const fields = new Float32Array(this.smokeStaging.getMappedRange().slice(0));
      this.smokeStaging.unmap();
      cloud = smokeCloud(this.sim.grid, fields.subarray(NS * d.N), fields.subarray(0, NS * d.N), this.sim.kind);
    }
    await this.staging.mapAsync(GPUMapMode.READ);
    const bytes = this.staging.getMappedRange().slice(0);
    this.staging.unmap();
    if (this.ctx.error) throw new Error(`WebGPU: ${this.ctx.error}`);
    const f = new Float32Array(bytes);
    const u = new Uint32Array(bytes);
    const n = d.nNodes;
    const T = f.slice(0, n);
    const vis = f.slice(n, 2 * n);
    const co = f.slice(2 * n, 3 * n);
    const soot = f.slice(3 * n, 4 * n);
    const co2 = f.slice(4 * n, 5 * n);
    const o2 = f.slice(5 * n, 6 * n);
    const speed = f.slice(6 * n, 6 * n + this.nSpeed);
    let at = 6 * n + this.nSpeed;
    const counters = u.slice(at, at + CTR_SIZE);
    at += CTR_SIZE;
    const sweeps = this.stepCount > 0 ? this.sweepsFrom(f.subarray(at, at + this.gsSlots)) : 0;
    at += this.gsSlots;
    let released = 0, burnt = 0, radiated = 0;
    const used = Math.min(this.slot, this.slots);
    for (let s = 0; s < used; s += 1) {
      released += f[at + 4 * s];
      burnt += f[at + 4 * s + 1];
      radiated += f[at + 4 * s + 2];
    }
    const lastReleased = this.slot > 0 ? f[at + 4 * this.lastSlot] : 0;
    at += 4 * this.slots;
    const pilotHeat = f[at];
    at += 4;
    // What the item burnt over the frame and in its last step, what waits for
    // oxygen, and its last flame cells: its record, as the CPU's burnerStep keeps it.
    if (this.burner) {
      const b = this.burner;
      const frameBurnt = f[at];
      b.released += frameBurnt;
      b.convective += (1 - b.radiantFraction) * frameBurnt;
      b.radiant += b.radiantFraction * frameBurnt;
      b.pending = f[at + 2];
      b.flameCells = f[at + 3];
      b.flameRate = f[at + 1] / this.lastDt;
    }
    at += 4;
    const maxT = new Float32Array(counters.buffer, CTR.maxT * 4, 1)[0];
    const maxSpeed = new Float32Array(counters.buffer, CTR.speed * 4, 1)[0];
    const gasHeat = counters[CTR.heat] * HEAT_UNIT;
    const nonFinite = counters[CTR.bad] > 0;
    let snapshot = null;
    if (snap) snapshot = this.snapshotFrom(f, u, at);
    else if (particles) snapshot = { positions: new Float32Array(0), temps: new Float32Array(0), kinds: new Uint8Array(0) };
    // A design-fire item's flame, drawn from the host's record of the step (the CPU's particleSnapshot does the same),
    // at its cells' temperatures read with the nodes.
    if (snapshot) {
      const from = this.flameFrom;
      const at = from === null ? null : new Map(Array.from(this.sim.burner.cells, (c, q) => [c, T[from + q] + KELVIN]));
      snapshot = withBurnerFlame(this.sim, snapshot, at && ((c) => at.get(c)));
    }
    this.slot = 0;
    return {
      T, vis, co, soot, co2, o2, speed, released, burnt, radiated, maxT, lastReleased, maxSpeed, gasHeat, nonFinite, pilotHeat,
      particleCount: counters[CTR.count] > d.cap ? d.cap : counters[CTR.count],
      embersSpawned: counters[CTR.embers],
      spotIgnitions: counters[CTR.spots],
      sweeps,
      particles: snapshot,
      smoke: cloud,
    };
  }

  /** Grid-unit particles to the viewer frame (the CPU's particleSnapshot). */
  snapshotFrom(f, u, at) {
    const g = this.sim.grid;
    const count = u[at];
    const positions = new Float32Array(count * 3);
    const temps = new Float32Array(count);
    const kinds = new Uint8Array(count);
    for (let q = 0; q < count; q += 1) {
      const o = at + 1 + q * 5;
      positions[q * 3 + g.axes[0]] = g.origin[0] + f[o] * g.h;
      positions[q * 3 + g.axes[1]] = g.origin[1] + f[o + 1] * g.h;
      positions[q * 3 + g.axes[2]] = g.origin[2] + f[o + 2] * g.h;
      temps[q] = f[o + 3];
      kinds[q] = f[o + 4];
    }
    return { positions, temps, kinds };
  }

  /** Whole fields, for tests only: T, Ts, rho, p, fuel, moisture, Y, u, v, w, kind, soot deposited per cell (sootDeposit), one-dimensional walls' nodes (wallT), and pressure zones' pi and mass (zones). */
  async readFields() {
    const { device, buf, dims: d } = this;
    const N = d.N;
    const nuvw = d.nu + d.nv + d.nw;
    const wallNodes = this.wall ? d.wallN * this.wall.n : 0;
    const zoneWords = this.zonesOn ? 2 * this.sealed.S : 0;
    const size = (CF_FIELDS * N + 6 * N + CI_FIELDS * N + nuvw + CTR_SIZE + this.gsSlots + wallNodes + (this.depositOn ? N : 0) + zoneWords) * 4;
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    let o = 0;
    const parts = [[buf.cf, 0, CF_FIELDS * N * 4], [buf.Y, 0, 6 * N * 4], [buf.ci, 0, CI_FIELDS * N * 4], [buf.vel, 0, nuvw * 4], [buf.ctr, 0, CTR_SIZE * 4], [buf.atom, AT_FIELDS * N * 4, this.gsSlots * 4]];
    if (wallNodes) parts.push([buf.wall, 3 * d.wallN * 4, wallNodes * 4]);
    if (this.depositOn) parts.push([buf.snap, this.snapAt.dep * 4, N * 4]);
    if (zoneWords) parts.push([buf.red, this.red.zone * 4, zoneWords * 4]);
    for (const [b, from, bytes] of parts) {
      enc.copyBufferToBuffer(b, from, staging, o, bytes);
      o += bytes;
    }
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const bytes = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    if (this.ctx.error) throw new Error(`WebGPU: ${this.ctx.error}`);
    const f = new Float32Array(bytes);
    const u = new Uint32Array(bytes);
    const field = (k) => f.slice(k * N, (k + 1) * N);
    let at = CF_FIELDS * N;
    const Y = f.slice(at, at + 6 * N);
    at += 6 * N;
    const kind = u.slice(at + CI.kind * N, at + (CI.kind + 1) * N);
    const burning = u.slice(at + CI.burning * N, at + (CI.burning + 1) * N);
    at += CI_FIELDS * N;
    const vel = f.slice(at, at + nuvw);
    at += nuvw;
    const wallT = wallNodes ? f.slice(at + CTR_SIZE + this.gsSlots, at + CTR_SIZE + this.gsSlots + wallNodes) : null;
    const depAt = at + CTR_SIZE + this.gsSlots + wallNodes;
    const sootDeposit = this.depositOn ? f.slice(depAt, depAt + N) : null;
    // Each pressure zone's pressure share and mass (the CPU's sim.zones.pi and .mass).
    const zoneAt = depAt + (this.depositOn ? N : 0);
    const zones = zoneWords ? { pi: f.slice(zoneAt, zoneAt + zoneWords / 2), mass: f.slice(zoneAt + zoneWords / 2, zoneAt + zoneWords) } : null;
    return {
      T: field(CF.T), Ts: field(CF.Ts), rho: field(CF.rho), p: field(CF.p), fuelMass: field(CF.fuel), moisture: field(CF.moist),
      sootDeposit, Y, kind, burning, u: vel.subarray(0, d.nu), v: vel.subarray(d.nu, d.nu + d.nv), w: vel.subarray(d.nu + d.nv),
      particles: Math.min(u[at + CTR.count], d.cap),
      sweeps: this.stepCount > 0 ? this.sweepsFrom(f.subarray(at + CTR_SIZE, at + CTR_SIZE + this.gsSlots)) : 0,
      wallT,
      zones,
    };
  }

  /** Resolves when the queued steps have run (for timing). */
  finish() {
    return this.device.queue.onSubmittedWorkDone();
  }

  /**
   * Heads opened (suppression.js openHeads): copy Evans's decay times and
   * opening times, and the spray's columns. The rates stay on the GPU: every
   * cell starts armed, and its first covered step reads its own.
   */
  applySuppression(sup, changed) {
    if (!this.suppressionOn) return;
    const q = this.device.queue;
    const N = this.dims.N;
    if (changed.touched.length) {
      q.writeBuffer(this.buf.sup, SUP.tau * N * 4, sup.tau);
      q.writeBuffer(this.buf.sup, SUP.tAct * N * 4, sup.tAct);
    }
    const cols = sup.columns;
    if (cols.length) {
      const records = new ArrayBuffer(cols.length * SUP.column * 4);
      const u = new Uint32Array(records), f = new Float32Array(records);
      cols.forEach((col, k) => {
        u[k * SUP.column] = col.top;
        u[k * SUP.column + 1] = col.count;
        f[k * SUP.column + 2] = col.flux;
      });
      q.writeBuffer(this.buf.sup, SUP.fields * N * 4, records);
      this.dims.sprayCols = cols.length;
      this.setParams({ sprayN: cols.length });
    }
  }

  destroy() {
    for (const b of Object.values(this.buf)) b.destroy();
    this.staging.destroy();
    this.smokeStaging?.destroy();
  }
}

/** Fuel inside a SOLID cell trades gas with a neighbour, which the per-cell chemistry kernel does not do. */
function supported(sim) {
  // A validation burner (radiating along rays) runs on the CPU; a design-fire item's runs here.
  if (sim.burner && !sim.burner.design) return false;
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] === 2 && sim.fuelMass[c] > 0) return false;
  return true;
}

/**
 * runScene on the GPU, with the CPU runScene's signature and result
 * (src/core/flamescapesRun.js). Async: resolves to the stacked frames, or
 * null when shouldStop() says so. `profileInto` (the CPU's per-stage wall
 * times) is accepted and left empty: GPU stages overlap and would need
 * timestamp queries to time.
 *
 * Where the GPU cannot run the scene (the device lost, its setup failed, a
 * scene the kernels do not cover) the CPU runs it, unless `cpuFallback` is
 * false: then it throws. The app's worker asks for that, because the scene
 * was sized for the GPU (up to 400,000 cells, far past the CPU's budget) and
 * its run would be reported, and its speed kept, as the GPU's; the worker
 * rebuilds the scene on the CPU's own budget instead.
 */
async function runSceneGpu(ctx, scene, opts = {}) {
  const { nodes, agentHeight = 0.75, headHeight = 1.8, horizon = 300, frameSeconds = 5, embers = false, dt = DEFAULTS.dt, onFrame = null, shouldStop = null, particles = 3000, smoke = false, solverOptions = {}, keep = true, cpuFallback = true } = opts;
  const fallback = (why) => {
    if (cpuFallback) return cpuRunScene(scene, opts);
    throw new Error(why);
  };
  if (ctx.lost) return fallback("The GPU device was lost.");
  const sim = createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, embers, dt, seed: 1, ...solverOptions });
  if (!supported(sim)) return fallback("The GPU solver does not cover this scene.");
  const cells = headCells(sim, nodes, { agentHeight, headHeight });
  const n = cells.length;
  // The ceiling detectors are read with the nodes, after them, then the
  // flooded and foamed rooms' detectors, then a design-fire item's cells
  // (its drawn flame's colours), then the sprinklers' sensor cells (with
  // their gas speed).
  const detectors = ceilingDetectors(scene);
  const D = detectors.cells.length;
  const sys = systemsRun(scene, sim, opts.systems, opts.upAxis);
  const Ds = sys ? sys.detectorCells.length : 0;
  const flame = particles && sim.burner?.design ? sim.burner.cells : new Int32Array(0);
  const B = flame.length;
  const spr = sprinklerRun(scene, sim, opts.sprinklers, opts.upAxis);
  const S = spr ? spr.sensors.length : 0;
  const suppression = spr && spr.sup.mode !== "activation" ? { columns: spr.setup.heads.reduce((a, h) => a + h.columns.length / 2, 0), eta: spr.sup.eta, wet: spr.sup.wet } : null;
  const K = Math.max(1, Math.round(horizon / frameSeconds));
  let gpu;
  const { device } = ctx;
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  try {
    gpu = new GpuSim(ctx, sim, {
      nodeCells: withDetectors(withDetectors(withDetectors(withDetectors(cells, detectors.cells), sys ? sys.detectorCells : new Int32Array(0)), flame), spr ? spr.sensors : new Int32Array(0)),
      snapMax: particles || 0,
      // A frame's steps may be cut into up to maxSubsteps each (substepsFor).
      slots: Math.ceil(frameSeconds / dt) * Math.max(1, sim.P.maxSubsteps || 1) + 2,
      speedFrom: n + D + Ds + B,
      suppression,
      systems: sys,
      flameFrom: B ? n + D + Ds : null,
    });
  } catch {
    gpu = null;
  }
  const setupErrors = [await device.popErrorScope(), await device.popErrorScope()].filter(Boolean);
  if (!gpu || setupErrors.length) {
    gpu?.destroy();
    return fallback(`The GPU could not set up the run${setupErrors.length ? `: ${setupErrors[0].message}` : ""}.`);
  }
  // With keep, every frame is stacked (K x n); without, one frame's buffers
  // are reused and onFrame must copy what it keeps (the CPU's runScene).
  const rows = keep ? K : 1;
  const T = new Float32Array(rows * n);
  const vis = new Float32Array(rows * n);
  const co = new Float32Array(rows * n);
  const soot = new Float32Array(rows * n);
  const co2 = new Float32Array(rows * n);
  const o2 = new Float32Array(rows * n);
  const detectorVis = new Float32Array(rows * D);
  const hrr = new Float32Array(K);
  const stats = { hrr: 0, burnt: 0, embersSpawned: 0, spotIgnitions: 0, maxT: sim.P.ambient, released: 0, radiated: 0, sweeps: 0, pilotHeat: 0 };
  // What a design-fire item has released by the last frame (the GPU's tallies leave it out).
  let itemBefore = 0;
  // Proposals A and B, as the CPU's runScene: each frame's steps cut by the
  // frame before's fastest gas, and each frame checked.
  let speed = 0, mostSubsteps = 1, health = null, frames = K;
  try {
    for (let k = 0; k < K; k += 1) {
      const end = (k + 1) * frameSeconds;
      const substeps = substepsFor(sim.P, sim.grid.h, speed, dt);
      mostSubsteps = Math.max(mostSubsteps, substeps);
      while (gpu.time < end - 1e-9) {
        gpu.step(Math.min(dt / substeps, end - gpu.time));
        if (shouldStop?.()) return null;
      }
      const frame = await gpu.readFrame({ particles: Boolean(particles), smoke });
      if (shouldStop?.()) return null;
      if (sim.burner) {
        frame.released += sim.burner.released - itemBefore;
        itemBefore = sim.burner.released;
      }
      speed = frame.maxSpeed;
      stats.pilotHeat += frame.pilotHeat;
      const reading = { maxT: frame.maxT, maxSpeed: frame.maxSpeed, gasHeat: frame.gasHeat, nonFinite: frame.nonFinite };
      const verdict = healthVerdict(reading, stats.released + frame.released + stats.pilotHeat);
      if (!verdict.ok) {
        // Where: the same reading on the fields, read back once.
        const fields = await gpu.readFields();
        const where = healthReadings(fields, sim.grid, sim.P, sim.V);
        const cell = verdict.reason === "non-finite" ? where.badCell : verdict.reason === "speed" ? where.fastCell : where.hotCell;
        health = failedHealth(sim.grid, { ...verdict, cell }, end, reading);
        frames = k;
        break;
      }
      const row = keep ? k : 0;
      const nodesOf = (a) => (D + Ds + B + S ? a.subarray(0, n) : a);
      T.set(nodesOf(frame.T), row * n);
      vis.set(nodesOf(frame.vis), row * n);
      co.set(nodesOf(frame.co), row * n);
      soot.set(nodesOf(frame.soot), row * n);
      co2.set(nodesOf(frame.co2), row * n);
      o2.set(nodesOf(frame.o2), row * n);
      if (D) detectorVis.set(frame.vis.subarray(n, n + D), row * D);
      hrr[k] = frame.released / frameSeconds / 1000;
      stats.released += frame.released;
      stats.burnt += frame.burnt;
      stats.radiated += frame.radiated;
      stats.hrr = frame.lastReleased / gpu.lastDt;
      stats.maxT = frame.maxT;
      stats.embersSpawned = frame.embersSpawned;
      stats.spotIgnitions = frame.spotIgnitions;
      stats.sweeps = frame.sweeps;
      // Sprinklers: the links step on the sensor cells' gas, heads that open
      // go to the GPU for the next frame. The water the open heads delivered
      // this frame is counted here; its evaporation stays on the GPU.
      let sprinkled = null;
      if (spr) {
        spr.sup.stats.emitted += spr.sup.columns.reduce((a, col) => a + col.flux, 0) * frameSeconds;
        const Tk = Float64Array.from(frame.T.subarray(n + D + Ds + B, n + D + Ds + B + S), (t) => t + KELVIN);
        sprinkled = sprinklerFrame(spr, { t0: end - frameSeconds, t1: end, T: Tk, u: frame.speed, hrr: hrr[k] });
        if (sprinkled.changed) gpu.applySuppression(spr.sup, sprinkled.changed);
      }
      // Flooded and foamed rooms: detection at their detectors; each step's
      // state follows from the discharge times (GpuSim.step).
      const flooded = sys ? systemsFrame(sys, { t1: end, vis: frame.vis.subarray(n + D, n + D + Ds) }) : null;
      itemSuppression(sim, spr, sys, end);
      onFrame?.({
        k,
        time: end,
        T: T.subarray(row * n, (row + 1) * n),
        vis: vis.subarray(row * n, (row + 1) * n),
        co: co.subarray(row * n, (row + 1) * n),
        soot: soot.subarray(row * n, (row + 1) * n),
        co2: co2.subarray(row * n, (row + 1) * n),
        o2: o2.subarray(row * n, (row + 1) * n),
        detectors: detectorVis.subarray(row * D, (row + 1) * D),
        sprinklers: sprinkled && { opened: sprinkled.opened, open: sprinkled.open, water: sprinkled.water },
        ...(flooded ? { systems: flooded } : {}),
        hrr: hrr[k],
        maxT: stats.maxT - KELVIN,
        embers: stats.embersSpawned,
        spots: stats.spotIgnitions,
        burnt: stats.burnt,
        particles: particles ? frame.particles : null,
        smoke: frame.smoke,
      });
    }
  } finally {
    gpu.destroy();
  }
  const sprinklers = spr ? sprinklerResult(spr) : null;
  // The GPU keeps no tally of the spray's evaporation.
  if (sprinklers) sprinklers.stats = { emitted: sprinklers.stats.emitted, evaporated: null, wetted: null, floor: null };
  // A failed run keeps the frames before the failure only (the CPU's runScene).
  const cut = (a, per) => (frames < K ? a.slice(0, frames * per) : a);
  const out = { n, K: frames, frameSeconds, horizon: frames * frameSeconds, hrr: cut(hrr, 1), stats, inGrid: cells.reduce((a, c) => a + (c >= 0 ? 1 : 0), 0), detectorDistance: detectors.distance, sprinklers };
  out.health = health || { ok: true, maxSpeed: +speed.toFixed(2), substeps: mostSubsteps };
  // The GPU keeps no per-room tallies of what the agent or the foam stopped.
  if (sys) out.systems = systemsResult(sys, { cpu: false });
  const item = itemResult(sim);
  if (item) out.designFire = item;
  return keep ? { ...out, T: cut(T, n), vis: cut(vis, n), co: cut(co, n), soot: cut(soot, n), co2: cut(co2, n), o2: cut(o2, n), detectorVis: cut(detectorVis, D) } : out;
}
