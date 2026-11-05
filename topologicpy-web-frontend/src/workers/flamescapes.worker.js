/**
 * The Flamescapes fire solver, off the main thread (src/core/flamescapes.js,
 * src/core/flamescapesRun.js). One job runs one fire: it builds the grid from
 * the IFC meshes, steps the solver to the horizon and streams a frame every
 * `frameSeconds` as progress ({ id, progress }), then answers with the frames
 * stacked for the planners. Ending the worker is the only way to stop a run.
 *
 * Backend: WebGPU on the visitor's GPU where the browser offers it and the
 * GPU solver is present (src/core/flamescapesGpu.js), one CPU core otherwise
 * (src/lib/hardware.js); the grid is sized to the backend.
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

self.onmessage = async (event) => {
  const { id, payload } = event.data || {};
  try {
    const t0 = performance.now();
    let backend = await chooseBackend(payload.backend || "auto");
    const gpu = await gpuRunner(backend);
    if (backend.kind === "gpu" && !gpu) backend = await chooseBackend("cpu");
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
    const g = scene.grid;
    const info = {
      backend: { kind: backend.kind, name: backend.name },
      grid: [g.nx, g.ny, g.nz],
      cell: g.h,
      cells: g.cells,
      clipped: scene.clipped,
      contents: scene.contentsSummary,
      setupMs: performance.now() - t0,
    };
    self.postMessage({ id, progress: { type: "scene", info } });
    const run = gpu ? gpu.runScene : runScene;
    let frames = 0;
    const result = await run(scene, {
      nodes: payload.nodes,
      upAxis: payload.upAxis,
      agentHeight: payload.agentHeight,
      headHeight: payload.headHeight,
      horizon: payload.horizon,
      frameSeconds: payload.frameSeconds,
      embers: payload.embers,
      particles: payload.particles ?? 2500,
      onFrame: (frame) => {
        frames += 1;
        // Copies: the stacked arrays stay here until the run ends.
        const out = { ...frame, type: "frame", T: frame.T.slice(), vis: frame.vis.slice(), co: frame.co.slice() };
        const transfers = [out.T.buffer, out.vis.buffer, out.co.buffer];
        if (out.particles) transfers.push(out.particles.positions.buffer, out.particles.temps.buffer, out.particles.kinds.buffer);
        self.postMessage({ id, progress: out }, transfers);
      },
    });
    const done = { ...result, info: { ...info, frames, ms: performance.now() - t0 } };
    self.postMessage({ id, ok: true, result: done }, [done.T.buffer, done.vis.buffer, done.co.buffer, done.hrr.buffer]);
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};
