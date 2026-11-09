/**
 * What the visitor's device can give the Flamescapes solver. GitHub Pages
 * only serves files, so the solver runs on the visitor's own hardware: on
 * the GPU through WebGPU where the browser offers it (as the paper's own
 * implementation runs WGSL compute kernels through WGPU), on one CPU core in
 * a worker otherwise. Works in a page or a worker.
 */

/**
 * The standard grid's cell budget: every device's default, the CPU's on a
 * desktop, so a GPU builds the grid a CPU does and gives the same fire
 * (wiki/research/backend-benchmark-2026-11-09.md). A finer grid on the GPU
 * is asked for by name.
 */
export const STANDARD_CELLS = 30000;

/** A backend's budget on the standard grid: STANDARD_CELLS, or less where the CPU's own budget is smaller (phones, small machines). */
export function standardCells(backend) {
  return Math.min(backend.cells, STANDARD_CELLS);
}

/** Cells the CPU path can step in about 20 to 40 ms on this device. */
export function cpuBudget(nav = globalThis.navigator) {
  const cores = nav?.hardwareConcurrency || 4;
  const memory = nav?.deviceMemory || 8; // GB, Chromium only
  const mobile = /Android|iPhone|iPad|Mobile/i.test(nav?.userAgent || "");
  if (mobile || memory <= 4 || cores <= 4) return 15000;
  return 30000;
}

/**
 * The WebGPU adapter, if any, with what the solver needs to size its grid.
 * Resolves to null without WebGPU or when no adapter is granted.
 */
export async function detectGpu(nav = globalThis.navigator) {
  if (!nav?.gpu?.requestAdapter) return null;
  try {
    const adapter = await nav.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return null;
    const info = adapter.info || (await adapter.requestAdapterInfo?.()) || {};
    const limits = adapter.limits || {};
    // About 196 bytes a cell across the GPU solver's buffers plus 96 per
    // particle slot; the largest single binding is 76 bytes a cell, so a
    // thirty-second of the binding limit leaves room.
    const byBinding = Math.floor((limits.maxStorageBufferBindingSize || 128 * 2 ** 20) / 32);
    const cells = Math.max(30000, Math.min(400000, byBinding));
    return {
      adapter,
      name: [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "WebGPU device",
      fallback: Boolean(adapter.isFallbackAdapter || info.isFallbackAdapter),
      cells,
    };
  } catch {
    return null;
  }
}

/** The backend a run will use: the GPU if offered and allowed, else the CPU. */
export async function chooseBackend(preference = "auto", nav = globalThis.navigator) {
  const cpu = { kind: "cpu", name: `CPU, one core of ${nav?.hardwareConcurrency || "?"}`, cells: cpuBudget(nav) };
  if (preference === "cpu") return cpu;
  const gpu = await detectGpu(nav);
  if (!gpu || gpu.fallback) return cpu;
  return { kind: "gpu", name: gpu.name, cells: gpu.cells, adapter: gpu.adapter };
}
