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
 *     faces) that give every solid cell the CPU's own order of exchanges; radiation emission from start-of-pass temperatures with
 *     fixed-point deposits; embers' spot-ignition test once per cell and step;
 *     random numbers from a per-thread PCG hash. With radiation and particles
 *     off, the GPU agrees with the CPU to float32 round-off and summation
 *     order: no more than the CPU differs from itself started one ulp apart
 *     (tools/flamescapes-gpu-check.mjs).
 *   - The initial state is the CPU's createSolver, uploaded, so both paths
 *     start from the same state; node cells are the CPU's headCells.
 *   - Per frame only the node readings, the frame's per-step heat release,
 *     fuel burnt and energy radiated, the counters and (when asked) a
 *     particle snapshot come back to the CPU. Whole fields never do.
 *   - A scene the kernels do not cover (fuel inside a solid cell, which
 *     createSolver never makes, or a grid beyond the device's buffer limits)
 *     runs on the CPU solver instead.
 */

import { DEFAULTS, NS, createSolver } from "./flamescapes.js";
import { KELVIN, ceilingDetectors, headCells, smokeCloud, withDetectors, runScene as cpuRunScene } from "./flamescapesRun.js";
import {
  AT, AT_FIELDS, B, CF, CF_FIELDS, CI, CI_FIELDS, CTR, CTR_SIZE, KERNELS, PARAMS, PARAM_BYTES, PARAM_SCALARS,
  PARTICLE_FLOATS, SWEEP_BYTES, WG, buildShader,
} from "./flamescapesGpu/kernels.js";

const MAX_WG_PER_DIM = 65535;
/** Conduction's passes, in the order that gives every solid cell the CPU's order of exchanges. */
const CONDUCTION = ["conductPorous", "conduct_x0", "conduct_x1", "conduct_y0", "conduct_y1", "conduct_z0", "conduct_z1", "conduct_bx", "conduct_by", "conduct_bz"];
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
    // Five storage buffers in the widest kernel (WebGPU guarantees eight); 64-thread workgroups.
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
   *       slots (steps per frame to keep separately).
   */
  constructor(ctx, sim, { nodeCells = new Int32Array(0), snapMax = 0, slots = 64 } = {}) {
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
    this.slot = 0;
    this.lastDt = P.dt;
    this.particlesOn = d.cap > 0;
    this.radiationOn = Number.isFinite(P.emitAbove) && P.emitAbove < 1e8;
    this.embersOn = Boolean(P.embers) && this.particlesOn;
    this.vorticityOn = P.vorticity > 0;
    const pilot = sim.pilot && sim.pilot.cells?.length ? sim.pilot : null;
    this.pilot = pilot;
    // Running ventilation (flamescapes.js ducts): at most 8 systems.
    const ducts = sim.ducts && sim.ducts.cells?.length && sim.ducts.systems <= 8 ? sim.ducts : null;
    this.ducts = ducts;

    // ---- layout of the shared buffers
    const io = { nodes: pilot ? pilot.cells.length : 0 };
    io.out = io.nodes + d.nNodes;
    // Six readings a node (gather): T, vis, co, soot, co2, o2.
    io.snap = io.out + 6 * d.nNodes;
    io.ducts = io.snap + 1 + 5 * d.snapMax;
    io.size = io.ducts + (ducts ? 2 * ducts.cells.length : 0);
    this.io = io;
    const red = { rad: 2 * this.nPart, steps: 3 * this.nPart };
    red.size = red.steps + 4 * this.slots;
    this.red = red;

    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (label, bytes) => device.createBuffer({ label, size: Math.max(16, Math.ceil(bytes / 4) * 4), usage });
    const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    const sizes = {
      cf: CF_FIELDS * N * 4,
      Y: 6 * N * 4,
      ci: CI_FIELDS * N * 4,
      vel: nuvw * 4,
      snap: (nuvw + 7 * N) * 4,
      atom: (AT_FIELDS * N + this.gsSlots) * 4,
      parts: Math.max(1, d.cap) * PARTICLE_FLOATS * 4,
      red: red.size * 4,
      io: io.size * 4,
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
    };
    // Sweep s's index at offset s * sweepStride, bound at that dynamic offset.
    const index = new Uint32Array((this.sweepStride / 4) * this.gsSlots);
    for (let s = 0; s < this.gsSlots; s += 1) index[(s * this.sweepStride) / 4] = s;
    device.queue.writeBuffer(this.buf.sweep, 0, index);
    const byBinding = [this.buf.U, this.buf.cf, this.buf.Y, this.buf.ci, this.buf.vel, this.buf.snap, this.buf.atom, this.buf.parts, this.buf.partsB, this.buf.ctr, this.buf.red, this.buf.io, this.buf.sweep, this.buf.atom];
    this.groups = {};
    for (const [name, k] of Object.entries(kernels)) {
      this.groups[name] = device.createBindGroup({
        label: name,
        layout: k.layout,
        entries: k.uses.map((binding) => ({ binding, resource: binding === B.sweep ? { buffer: byBinding[binding], size: SWEEP_BYTES } : { buffer: byBinding[binding] } })),
      });
    }
    this.staging = device.createBuffer({ label: "frame readback", size: Math.max(16, (6 * d.nNodes + CTR_SIZE + this.gsSlots + 4 * this.slots + 1 + 5 * d.snapMax) * 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    // ---- parameters
    this.paramBytes = new ArrayBuffer(PARAM_BYTES);
    this.paramF = new Float32Array(this.paramBytes);
    this.paramU = new Uint32Array(this.paramBytes);
    this.paramIndex = Object.fromEntries(PARAMS.map(([name], i) => [name, i]));
    const fuelRow = (k) => {
      const st = sim.stoich[k];
      const fuel = sim.fuels[k];
      const liquid = fuel.liquid;
      return [st.o2, st.co2, st.co, st.soot, st.h2o, st.heat, fuel.ignitionTemperature + KELVIN, fuel.embers ? 1 : 0, liquid ? 1 : 0, liquid?.boiling ?? 0, liquid?.latent ?? 1, st.ash ?? 0, fuel.heatCapacity ?? P.fuelHeatCapacity, liquid?.mInf ?? -1, liquid?.kBeta ?? -1, 0];
    };
    this.paramF.set([...fuelRow(0), ...fuelRow(1)], PARAM_SCALARS);
    this.setParams({
      nx, ny, nz, N, nu: d.nu, nv: d.nv, nw: d.nw, stepNo: 0, slot: 0,
      rays: Math.max(0, P.rays | 0), rayReach: Math.max(0, P.rayReach | 0), cap: d.cap,
      flameCap: Math.min(d.cap, Math.ceil(P.flameShare * d.cap)),
      nPart: this.nPart, nNodes: d.nNodes, snapMax: d.snapMax, pilotN: io.nodes,
      pilotMode: pilot && pilot.temperature ? 0 : 1, ioNodes: io.nodes, ioOut: io.out, ioSnap: io.snap,
      redRad: red.rad, redSteps: red.steps, seed: P.seed >>> 0, gsAt: AT_FIELDS * N,
      h, dt: P.dt, V: sim.V, rho0: sim.rho0, ambient: P.ambient, gravity: P.gravity, convection: P.convection,
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
    q.writeBuffer(this.buf.ci, 0, ci);
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
    if (this.ducts) {
      for (let q = 0; q < this.ducts.cells.length; q += 1) {
        io[this.io.ducts + 2 * q] = this.ducts.cells[q];
        io[this.io.ducts + 2 * q + 1] = this.ducts.system[q];
      }
    }
    for (let v = 0; v < nodeCells.length; v += 1) io[this.io.nodes + v] = nodeCells[v] < 0 ? 0xffffffff : nodeCells[v];
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
    this.setParams({ dt, stepNo: this.stepCount, slot, settle });
    this.writeParams();
    const enc = device.createCommandEncoder();
    enc.clearBuffer(buf.atom, AT_FIELDS * N * 4, this.gsSlots * 4);
    let pass = enc.beginComputePass();
    // Ignition source, then particles (lines 8 to 14).
    if (this.pilot && this.time < this.pilot.duration) this.dispatch(pass, "pilot");
    if (this.particlesOn) {
      this.dispatch(pass, "spawnFlame");
      this.dispatch(pass, "spawnFlameEnd");
      if (this.embersOn) this.dispatch(pass, "spawnEmbers");
      this.dispatch(pass, "moveParticles");
    }
    pass.end();
    // Radiation (line 15) and particle-to-grid (lines 16 to 18).
    if (this.radiationOn) enc.clearBuffer(buf.atom, AT.depGas * N * 4, 2 * N * 4);
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
    // Advection reads the step's start (line 20).
    enc.copyBufferToBuffer(buf.vel, 0, buf.snap, 0, nuvw * 4);
    enc.copyBufferToBuffer(buf.cf, CF.T * N * 4, buf.snap, nuvw * 4, N * 4);
    enc.copyBufferToBuffer(buf.Y, 0, buf.snap, (nuvw + N) * 4, 6 * N * 4);
    pass = enc.beginComputePass();
    this.dispatch(pass, "advect");
    this.dispatch(pass, "density");
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
    if (this.particlesOn) this.dispatch(pass, "cullFlameFront");
    for (const name of CONDUCTION) this.dispatch(pass, name);
    this.dispatch(pass, "densityBoundaries");
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
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.dispatch(pass, "gather");
    this.dispatch(pass, "peakTemperature");
    const snap = particles && d.snapMax > 0;
    if (snap) this.dispatch(pass, "snapshot");
    pass.end();
    const nodeBytes = 6 * d.nNodes * 4;
    const stepBytes = 4 * this.slots * 4;
    let o = 0;
    if (nodeBytes) enc.copyBufferToBuffer(buf.io, this.io.out * 4, this.staging, o, nodeBytes);
    o += nodeBytes;
    enc.copyBufferToBuffer(buf.ctr, 0, this.staging, o, CTR_SIZE * 4);
    o += CTR_SIZE * 4;
    enc.copyBufferToBuffer(buf.atom, AT_FIELDS * d.N * 4, this.staging, o, this.gsSlots * 4);
    o += this.gsSlots * 4;
    enc.copyBufferToBuffer(buf.red, this.red.steps * 4, this.staging, o, stepBytes);
    o += stepBytes;
    const snapBytes = (1 + 5 * d.snapMax) * 4;
    if (snap) enc.copyBufferToBuffer(buf.io, this.io.snap * 4, this.staging, o, snapBytes);
    enc.clearBuffer(buf.red, this.red.steps * 4, stepBytes);
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
    let at = 6 * n;
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
    const maxT = new Float32Array(counters.buffer, CTR.maxT * 4, 1)[0];
    let snapshot = null;
    if (snap) snapshot = this.snapshotFrom(f, u, at);
    else if (particles) snapshot = { positions: new Float32Array(0), temps: new Float32Array(0), kinds: new Uint8Array(0) };
    this.slot = 0;
    return {
      T, vis, co, soot, co2, o2, released, burnt, radiated, maxT, lastReleased,
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

  /** Whole fields, for tests only: T, Ts, rho, p, fuel, moisture, Y, u, v, w, kind. */
  async readFields() {
    const { device, buf, dims: d } = this;
    const N = d.N;
    const nuvw = d.nu + d.nv + d.nw;
    const size = (CF_FIELDS * N + 6 * N + CI_FIELDS * N + nuvw + CTR_SIZE + this.gsSlots) * 4;
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    let o = 0;
    for (const [b, from, bytes] of [[buf.cf, 0, CF_FIELDS * N * 4], [buf.Y, 0, 6 * N * 4], [buf.ci, 0, CI_FIELDS * N * 4], [buf.vel, 0, nuvw * 4], [buf.ctr, 0, CTR_SIZE * 4], [buf.atom, AT_FIELDS * N * 4, this.gsSlots * 4]]) {
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
    return {
      T: field(CF.T), Ts: field(CF.Ts), rho: field(CF.rho), p: field(CF.p), fuelMass: field(CF.fuel), moisture: field(CF.moist),
      Y, kind, burning, u: vel.subarray(0, d.nu), v: vel.subarray(d.nu, d.nu + d.nv), w: vel.subarray(d.nu + d.nv),
      particles: Math.min(u[at + CTR.count], d.cap),
      sweeps: this.stepCount > 0 ? this.sweepsFrom(f.subarray(at + CTR_SIZE, at + CTR_SIZE + this.gsSlots)) : 0,
    };
  }

  /** Resolves when the queued steps have run (for timing). */
  finish() {
    return this.device.queue.onSubmittedWorkDone();
  }

  destroy() {
    for (const b of Object.values(this.buf)) b.destroy();
    this.staging.destroy();
    this.smokeStaging?.destroy();
  }
}

/** Fuel inside a SOLID cell trades gas with a neighbour, which the per-cell chemistry kernel does not do. */
function supported(sim) {
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] === 2 && sim.fuelMass[c] > 0) return false;
  return true;
}

/**
 * runScene on the GPU, with the CPU runScene's signature and result
 * (src/core/flamescapesRun.js). Async: resolves to the stacked frames, or
 * null when shouldStop() says so. `profileInto` (the CPU's per-stage wall
 * times) is accepted and left empty: GPU stages overlap and would need
 * timestamp queries to time.
 */
async function runSceneGpu(ctx, scene, opts = {}) {
  const { nodes, agentHeight = 0.75, headHeight = 1.8, horizon = 300, frameSeconds = 5, embers = false, dt = DEFAULTS.dt, onFrame = null, shouldStop = null, particles = 3000, smoke = false, solverOptions = {}, keep = true } = opts;
  if (ctx.lost) return cpuRunScene(scene, opts);
  const sim = createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, embers, dt, seed: 1, ...solverOptions });
  if (!supported(sim)) return cpuRunScene(scene, opts);
  const cells = headCells(sim, nodes, { agentHeight, headHeight });
  const n = cells.length;
  // The ceiling detectors are read with the nodes, after them.
  const detectors = ceilingDetectors(scene);
  const D = detectors.cells.length;
  const K = Math.max(1, Math.round(horizon / frameSeconds));
  let gpu;
  const { device } = ctx;
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  try {
    gpu = new GpuSim(ctx, sim, { nodeCells: withDetectors(cells, detectors.cells), snapMax: particles || 0, slots: Math.ceil(frameSeconds / dt) + 2 });
  } catch {
    gpu = null;
  }
  const setupErrors = [await device.popErrorScope(), await device.popErrorScope()].filter(Boolean);
  if (!gpu || setupErrors.length) {
    gpu?.destroy();
    return cpuRunScene(scene, opts);
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
  const stats = { hrr: 0, burnt: 0, embersSpawned: 0, spotIgnitions: 0, maxT: sim.P.ambient, released: 0, radiated: 0, sweeps: 0 };
  try {
    for (let k = 0; k < K; k += 1) {
      const end = (k + 1) * frameSeconds;
      while (gpu.time < end - 1e-9) {
        gpu.step(Math.min(dt, end - gpu.time));
        if (shouldStop?.()) return null;
      }
      const frame = await gpu.readFrame({ particles: Boolean(particles), smoke });
      if (shouldStop?.()) return null;
      const row = keep ? k : 0;
      const nodesOf = (a) => (D ? a.subarray(0, n) : a);
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
  const out = { n, K, frameSeconds, horizon: K * frameSeconds, hrr, stats, inGrid: cells.reduce((a, c) => a + (c >= 0 ? 1 : 0), 0), detectorDistance: detectors.distance };
  return keep ? { ...out, T, vis, co, soot, co2, o2, detectorVis } : out;
}
