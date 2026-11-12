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

/**
 * Bytes a cell in the GPU solver's largest single binding, the per-cell
 * float fields `cf` (flamescapesGpu/kernels.js CF_FIELDS x 4; a unit test
 * holds the two together). All its buffers hold about 236 bytes a cell (cf
 * 80, the transport's snapshot and scratch state 72 (76 with soot
 * deposition), ci 32, Y 24, atom 20, faces 12; sprinklers add 12) plus 96
 * per particle slot; one-dimensional walls add their own buffer.
 */
export const GPU_BINDING_BYTES_PER_CELL = 80;

/** Whether the browser says it runs on a phone or tablet. */
export function isPhone(nav = globalThis.navigator) {
  return /Android|iPhone|iPad|Mobile/i.test(nav?.userAgent || "");
}

/** Cells the CPU path can step in about 20 to 40 ms on this device. */
export function cpuBudget(nav = globalThis.navigator) {
  const cores = nav?.hardwareConcurrency || 4;
  const memory = nav?.deviceMemory || 8; // GB, Chromium only
  const mobile = isPhone(nav);
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
    // The grid is as large as the largest binding can hold (and a buffer can
    // be), at most 400,000 cells.
    const limit = Math.min(limits.maxStorageBufferBindingSize || 128 * 2 ** 20, limits.maxBufferSize || 256 * 2 ** 20);
    const byBinding = Math.floor(limit / GPU_BINDING_BYTES_PER_CELL);
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

/**
 * Whether a WebGPU adapter's name is a laptop's integrated graphics. Chrome on
 * Windows ignores powerPreference (crbug.com/369219127), so a laptop with a
 * second, faster card (an NVIDIA Quadro or GeForce, an AMD Radeon) runs the
 * fire on the integrated one unless Windows is told to give the browser the
 * faster card; the run card says how when this is true.
 */
export function integratedGpu(name) {
  // Intel's discrete Arc cards report "gen-12hp" or "xe-2hpg"; the integrated ones "gen-12lp", "gen-11", "xe-lpg".
  const arch = (name || "").split(" ")[1] || "";
  return /^intel\b/i.test(name || "") && !/hp/i.test(arch);
}

/** What the card and the panel say on integrated graphics: Chrome's own setting and Windows'. */
export const FASTER_GPU_HINT =
  "On a laptop that also has an NVIDIA or AMD card, give the browser the faster one: in Chrome open chrome://flags/#force-high-performance-gpu, enable it and restart Chrome; or in Windows Settings, System, Display, Graphics, set the browser to High performance and restart it.";

/** The backend a run will use: the GPU if offered and allowed, else the CPU. */
export async function chooseBackend(preference = "auto", nav = globalThis.navigator) {
  const cpu = { kind: "cpu", name: `CPU, one core of ${nav?.hardwareConcurrency || "?"}`, cells: cpuBudget(nav) };
  if (preference === "cpu") return cpu;
  const gpu = await detectGpu(nav);
  if (!gpu || gpu.fallback) return cpu;
  return { kind: "gpu", name: gpu.name, cells: gpu.cells, adapter: gpu.adapter };
}
