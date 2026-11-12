/**
 * The Flamescapes fire solver, off the main thread (src/core/flamescapes.js,
 * src/core/flamescapesRun.js). One job runs one fire: it builds the grid from
 * the IFC meshes, steps the solver to the horizon and streams a frame every
 * `frameSeconds` as progress ({ id, progress }), then answers with the run's
 * summary. Ending the worker is the only way to stop a run.
 *
 * Backend: WebGPU on the visitor's GPU where the browser offers it and the
 * GPU solver starts (src/core/flamescapesGpu.js), one CPU core otherwise
 * (src/lib/hardware.js); the grid is sized to the backend. A GPU lost
 * mid-run (a driver reset, a phone suspending the tab) restarts the run on
 * the CPU: a second "scene" message tells the page to drop the frames so far.
 *
 * The grid (payload.grid): "standard" by default, the same budget on every
 * device (lib/hardware.js STANDARD_CELLS), so the GPU builds the CPU's grid
 * and gives its fire; a phone's or small machine's CPU keeps its own smaller
 * budget. "speed" sizes the GPU's grid by its memory and its speed: the first
 * run measures how many cells a millisecond the GPU steps (between its first
 * two frames, with the page drawing beside it) and, if that is slower than
 * GPU_STEP_MS a step, starts again on the coarser grid level that keeps up;
 * the page keeps the measured speed per GPU (payload.gpuRate, gpuName), so
 * later runs on that GPU start on the same level. "finest" asks for the
 * finest grid the GPU's memory allows. "calibrated" builds cells of exactly
 * the size the held-out tests calibrated the smoke layer's depth on
 * (core/calibrationStatus.js CALIBRATED_CELL, 0.27 m), clipped round the fire
 * to the backend's own budget, so the run lies on a validated grid.
 *
 * Repeat runs (payload.repeats, a second job after a GPU run): the same fire
 * on the first run's grid (payload.maxCells) with the room k thousandths of
 * a degree warmer, k = 1, 2, ..., each answered as progress with its summary
 * (core/forecast.js runSummary), so the page can show the range. GPU only.
 */
import { DEFAULTS } from "../core/flamescapes.js";
import { buildScene, runScene } from "../core/flamescapesRun.js";
import { runSummary } from "../core/forecast.js";
import { chooseBackend, standardCells } from "../lib/hardware.js";
import { CALIBRATED_CELL } from "../core/calibrationStatus.js";

// Wall time a solver step (0.1 s) may take on the GPU: three times real time,
// about what one CPU core manages on its coarser grid.
const GPU_STEP_MS = 33;
// The grid sizes a speed-sized GPU grid snaps to, so timing noise does not
// change the grid, and the result, from one run to the next.
const GPU_LEVELS = [400000, 300000, 220000, 160000, 120000, 90000, 60000];
// A run projected to take less than this is not restarted on a coarser grid.
const RESTART_WORTH_MS = 15000;

/** The GPU grid's cell budget: its memory's, capped by the level its measured speed keeps up with. */
function gpuCells(backend, rate) {
  if (!(rate > 0)) return backend.cells;
  const want = rate * GPU_STEP_MS;
  return Math.min(backend.cells, GPU_LEVELS.find((n) => n <= want) ?? GPU_LEVELS.at(-1));
}

async function gpuRunner(backend) {
  if (backend.kind !== "gpu") return null;
  try {
    const module = await import("../core/flamescapesGpu.js");
    return module.createGpuRunner ? await module.createGpuRunner(backend.adapter) : null;
  } catch {
    return null;
  }
}

function sceneFor(payload, backend) {
  const scene = buildScene(payload.geometry, {
    upAxis: payload.upAxis,
    fire: payload.fire,
    maxCells: payload.maxCells || backend.cells,
    // The calibrated grid: cells of exactly the validated size, clipped to the budget.
    ...(payload.grid === "calibrated" ? { minCell: CALIBRATED_CELL, maxCell: CALIBRATED_CELL } : {}),
    fireLoad: payload.fireLoad,
    contentsFuel: payload.contentsFuel,
    ignitionFuel: payload.ignitionFuel,
    item: payload.item,
    burner: payload.burner,
    designFire: payload.designFire ?? null,
    ventilation: payload.ventilation || null,
    seed: payload.seed ?? 1,
  });
  if (!scene.ignition) throw new Error("The fire origin is not inside the fire grid. Pick it on a floor of the building.");
  // The solver asked for, for the run card (core/solverVersions.js).
  scene.solver = payload.solver || "validated";
  return scene;
}

/** The grid's budget for a backend under a policy: the standard grid's, or the backend's own for "speed" and "finest". */
function sized(backend, policy) {
  return policy === "standard" ? { ...backend, cells: standardCells(backend) } : backend;
}

function describe(scene, backend, t0, restarted, resized = false, openings = null, policy = "standard", budget = backend.cells) {
  const g = scene.grid;
  return {
    backend: { kind: backend.kind, name: backend.name },
    solver: scene.solver || "validated",
    // The grid asked for and the budget it was built to (repeat runs build theirs to it).
    policy,
    budget,
    grid: [g.nx, g.ny, g.nz],
    cell: g.h,
    cells: g.cells,
    clipped: scene.clipped,
    contents: scene.contentsSummary,
    // The first item's t-squared curve, when it grows as a design fire.
    designFire: scene.ignition?.designFire ?? null,
    setupMs: performance.now() - t0,
    restarted,
    resized,
    openings: openings && { ...openings, ductCells: scene.ducts?.cells.length ?? 0, ductsOutside: scene.ducts?.outside ?? 0 },
  };
}

/** The repeat runs of a GPU run: each one's summary as progress, then the list. */
async function repeatRuns(id, payload) {
  const backend = await chooseBackend(payload.backend || "auto");
  const gpu = await gpuRunner(backend);
  // On the CPU each would take as long as the run itself: none.
  if (!gpu) return { members: [], skipped: "no GPU" };
  const scene = sceneFor(payload, backend);
  const { count, tenability } = payload.repeats;
  const members = [];
  for (let k = 1; k <= count; k += 1) {
    let frames;
    try {
      frames = await gpu.runScene(scene, {
        nodes: payload.nodes,
        upAxis: payload.upAxis,
        agentHeight: payload.agentHeight,
        headHeight: payload.headHeight,
        horizon: payload.horizon,
        frameSeconds: payload.frameSeconds,
        embers: payload.embers,
        sprinklers: payload.sprinklers ?? null,
        systems: payload.systems ?? null,
        // Nothing to draw: particles and smoke snapshots only feed the view.
        particles: 0,
        smoke: false,
        solverOptions: { ...(payload.solverOptions || {}), ambient: DEFAULTS.ambient + k * 1e-3 },
        // Not on the CPU in its place: the scene is sized for the GPU.
        cpuFallback: false,
      });
    } catch {
      // The GPU stopped: the members run so far, and the rest not run.
      return { members, skipped: "the GPU stopped" };
    }
    // A member the health guard stopped is cut short: left out of the range, and said.
    if (frames?.health?.ok === false) {
      self.postMessage({ id, progress: { type: "member", stopped: { k, time: frames.health.time } } });
      continue;
    }
    // CO2 and O2 count in the doses only when the page's run kept them.
    const summary = { k, ...runSummary(payload.extras ? frames : { ...frames, co2: null, o2: null }, tenability) };
    members.push(summary);
    self.postMessage({ id, progress: { type: "member", summary } });
  }
  return { members };
}

self.onmessage = async (event) => {
  const { id, payload } = event.data || {};
  try {
    if (payload.repeats) {
      self.postMessage({ id, ok: true, result: await repeatRuns(id, payload) });
      return;
    }
    const t0 = performance.now();
    const policy = payload.grid || "standard";
    let backend = await chooseBackend(payload.backend || "auto");
    const gpu = await gpuRunner(backend);
    if (backend.kind === "gpu" && !gpu) backend = await chooseBackend("cpu");
    const bySpeed = policy === "speed";
    const knownRate = bySpeed && payload.gpuName === backend.name ? payload.gpuRate : null;
    backend = sized(backend, policy);
    if (gpu && knownRate) backend = { ...backend, cells: gpuCells(backend, knownRate) };
    let scene = sceneFor(payload, backend);
    let info = describe(scene, backend, t0, false, false, payload.openings, policy);
    self.postMessage({ id, progress: { type: "scene", info } });
    let frames = 0;
    const options = {
      nodes: payload.nodes,
      upAxis: payload.upAxis,
      agentHeight: payload.agentHeight,
      headHeight: payload.headHeight,
      horizon: payload.horizon,
      frameSeconds: payload.frameSeconds,
      embers: payload.embers,
      particles: payload.particles ?? 2500,
      smoke: payload.smoke ?? true,
      // The solver asked for (core/solverVersions.js): none for the validated one.
      solverOptions: payload.solverOptions || {},
      // Sprinklers (src/core/suppression.js): heads, mode and reading; none when off.
      sprinklers: payload.sprinklers ?? null,
      // Rooms flooded with clean agent or blanketed with foam (src/core/suppressionSystems.js); none when off.
      systems: payload.systems ?? null,
      // On the GPU: a run it cannot make throws (below, the scene is rebuilt
      // on the CPU's budget) instead of running the GPU-sized scene on the CPU.
      cpuFallback: false,
      keep: false,
      onFrame: (frame) => {
        frames += 1;
        // Copies: the solver reuses its frame buffers for the next frame.
        // Soot, CO2 and O2 only when asked for (payload.extras): the page
        // keeps every frame, and they double what each one holds.
        // The ceiling detectors' visibility (a few numbers) goes with every frame.
        const out = { ...frame, type: "frame", T: frame.T.slice(), vis: frame.vis.slice(), co: frame.co.slice(), detectors: frame.detectors?.slice() ?? null };
        if (payload.extras && frame.soot) {
          out.soot = frame.soot.slice();
          out.co2 = frame.co2.slice();
          out.o2 = frame.o2.slice();
        } else {
          delete out.soot;
          delete out.co2;
          delete out.o2;
        }
        const transfers = [out.T.buffer, out.vis.buffer, out.co.buffer, ...(out.soot ? [out.soot.buffer, out.co2.buffer, out.o2.buffer] : [])];
        if (out.detectors) transfers.push(out.detectors.buffer);
        if (out.particles) transfers.push(out.particles.positions.buffer, out.particles.temps.buffer, out.particles.kinds.buffer);
        if (out.smoke) transfers.push(out.smoke.positions.buffer, out.smoke.opacity.buffer);
        self.postMessage({ id, progress: out }, transfers);
      },
    };
    let result = null;
    let gpuRate = null;
    if (gpu) {
      try {
        // The GPU's speed between its first two frames; too slow, and with no
        // speed known yet, the run starts again on the level that keeps up.
        const stepsPerFrame = Math.max(1, Math.round(payload.frameSeconds / DEFAULTS.dt));
        const totalSteps = Math.round(payload.horizon / DEFAULTS.dt);
        let tFirst = 0, coarser = 0;
        const measuring = {
          ...options,
          onFrame: (frame) => {
            options.onFrame(frame);
            if (frame.k === 0) tFirst = performance.now();
            if (frame.k !== 1) return;
            const msPerStep = (performance.now() - tFirst) / stepsPerFrame;
            gpuRate = scene.grid.cells / msPerStep;
            const fits = gpuCells(backend, gpuRate);
            if (bySpeed && !knownRate && fits < 0.8 * scene.grid.cells && msPerStep * totalSteps > RESTART_WORTH_MS) coarser = fits;
          },
          shouldStop: () => coarser > 0,
        };
        result = await gpu.runScene(scene, measuring);
        if (!result && coarser) {
          backend = { ...backend, cells: coarser };
          scene = sceneFor(payload, backend);
          info = describe(scene, backend, t0, true, true, payload.openings, policy);
          frames = 0;
          self.postMessage({ id, progress: { type: "scene", info } });
          result = await gpu.runScene(scene, options);
        }
      } catch {
        // The GPU stopped mid-run, or could not start it (cpuFallback: false):
        // start again on the CPU, on a grid sized for it, and keep no GPU speed.
        backend = sized(await chooseBackend("cpu"), policy);
        scene = sceneFor(payload, backend);
        info = describe(scene, backend, t0, true, false, payload.openings, policy);
        frames = 0;
        self.postMessage({ id, progress: { type: "scene", info } });
      }
    }
    if (!result) result = await runScene(scene, options);
    // The frames went out one by one; the answer carries only the summary.
    // The speed is kept for "speed" grids only: measured on a smaller grid, it
    // would undersell the GPU and size its later "speed" grids too coarse.
    const done = { ...result, info: { ...info, frames, ms: performance.now() - t0, gpuRate: backend.kind === "gpu" && bySpeed ? gpuRate : null } };
    self.postMessage({ id, ok: true, result: done }, [done.hrr.buffer]);
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};
