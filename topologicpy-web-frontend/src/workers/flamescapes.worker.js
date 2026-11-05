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
 */
import { buildScene, runScene } from "../core/flamescapesRun.js";
import { chooseBackend } from "../lib/hardware.js";

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
    seed: payload.seed ?? 1,
  });
  if (!scene.ignition) throw new Error("The fire origin is not inside the fire grid. Pick it on a floor of the building.");
  return scene;
}

function describe(scene, backend, t0, restarted) {
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
  };
}

self.onmessage = async (event) => {
  const { id, payload } = event.data || {};
  try {
    const t0 = performance.now();
    let backend = await chooseBackend(payload.backend || "auto");
    const gpu = await gpuRunner(backend);
    if (backend.kind === "gpu" && !gpu) backend = await chooseBackend("cpu");
    let scene = sceneFor(payload, backend);
    let info = describe(scene, backend, t0, false);
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
      keep: false,
      onFrame: (frame) => {
        frames += 1;
        // Copies: the solver reuses its frame buffers for the next frame.
        const out = { ...frame, type: "frame", T: frame.T.slice(), vis: frame.vis.slice(), co: frame.co.slice() };
        const transfers = [out.T.buffer, out.vis.buffer, out.co.buffer];
        if (out.particles) transfers.push(out.particles.positions.buffer, out.particles.temps.buffer, out.particles.kinds.buffer);
        self.postMessage({ id, progress: out }, transfers);
      },
    };
    let result = null;
    if (gpu) {
      try {
        result = await gpu.runScene(scene, options);
      } catch {
        // The GPU stopped mid-run: start again on the CPU, on a grid sized for it.
        backend = await chooseBackend("cpu");
        scene = sceneFor(payload, backend);
        info = describe(scene, backend, t0, true);
        frames = 0;
        self.postMessage({ id, progress: { type: "scene", info } });
      }
    }
    if (!result) result = await runScene(scene, options);
    // The frames went out one by one; the answer carries only the summary.
    const done = { ...result, info: { ...info, frames, ms: performance.now() - t0 } };
    self.postMessage({ id, ok: true, result: done }, [done.hrr.buffer]);
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};
