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
 * The GPU's grid is sized by its memory and by its speed. The first run
 * measures how many cells a millisecond the GPU steps (between its first two
 * frames, with the page drawing beside it) and, if that is slower than
 * GPU_STEP_MS a step, starts again on the coarser grid level that keeps up;
 * the page keeps the measured speed per GPU (payload.gpuRate, gpuName), so
 * later runs on that GPU start on the same level. payload.sizeBySpeed false
 * asks for the finest grid the GPU's memory allows instead.
 */
import { DEFAULTS } from "../core/flamescapes.js";
import { buildScene, runScene } from "../core/flamescapesRun.js";
import { chooseBackend } from "../lib/hardware.js";

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
    fireLoad: payload.fireLoad,
    contentsFuel: payload.contentsFuel,
    ignitionFuel: payload.ignitionFuel,
    item: payload.item,
    burner: payload.burner,
    ventilation: payload.ventilation || null,
    seed: payload.seed ?? 1,
  });
  if (!scene.ignition) throw new Error("The fire origin is not inside the fire grid. Pick it on a floor of the building.");
  return scene;
}

function describe(scene, backend, t0, restarted, resized = false, openings = null) {
  const g = scene.grid;
  return {
    backend: { kind: backend.kind, name: backend.name },
    grid: [g.nx, g.ny, g.nz],
    cell: g.h,
    cells: g.cells,
    clipped: scene.clipped,
    contents: scene.contentsSummary,
    setupMs: performance.now() - t0,
    restarted,
    resized,
    openings: openings && { ...openings, ductCells: scene.ducts?.cells.length ?? 0, ductsOutside: scene.ducts?.outside ?? 0 },
  };
}

self.onmessage = async (event) => {
  const { id, payload } = event.data || {};
  try {
    const t0 = performance.now();
    let backend = await chooseBackend(payload.backend || "auto");
    const gpu = await gpuRunner(backend);
    if (backend.kind === "gpu" && !gpu) backend = await chooseBackend("cpu");
    const bySpeed = payload.sizeBySpeed !== false;
    const knownRate = bySpeed && payload.gpuName === backend.name ? payload.gpuRate : null;
    if (gpu && knownRate) backend = { ...backend, cells: gpuCells(backend, knownRate) };
    let scene = sceneFor(payload, backend);
    let info = describe(scene, backend, t0, false, false, payload.openings);
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
      keep: false,
      onFrame: (frame) => {
        frames += 1;
        // Copies: the solver reuses its frame buffers for the next frame.
        const out = { ...frame, type: "frame", T: frame.T.slice(), vis: frame.vis.slice(), co: frame.co.slice() };
        const transfers = [out.T.buffer, out.vis.buffer, out.co.buffer];
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
          info = describe(scene, backend, t0, true, true, payload.openings);
          frames = 0;
          self.postMessage({ id, progress: { type: "scene", info } });
          result = await gpu.runScene(scene, options);
        }
      } catch {
        // The GPU stopped mid-run: start again on the CPU, on a grid sized for it.
        backend = await chooseBackend("cpu");
        scene = sceneFor(payload, backend);
        info = describe(scene, backend, t0, true, false, payload.openings);
        frames = 0;
        self.postMessage({ id, progress: { type: "scene", info } });
      }
    }
    if (!result) result = await runScene(scene, options);
    // The frames went out one by one; the answer carries only the summary.
    const done = { ...result, info: { ...info, frames, ms: performance.now() - t0, gpuRate: backend.kind === "gpu" ? gpuRate : null } };
    self.postMessage({ id, ok: true, result: done }, [done.hrr.buffer]);
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};
